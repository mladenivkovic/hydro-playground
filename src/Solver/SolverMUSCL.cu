#include "SolverMUSCL.h"
#include "Gas.h"
#include "Riemann.h"
#include "Limiter.h"

using CState = ConservedState;
using CFlux  = ConservedFlux;

/**

*/

namespace Kernels{
  __global__ void updateMids(Grid grid, Float dt, int direction);
  __global__ void computeFluxes(Grid grid, Float dt, int direction);

  // not a kernel, but rather a device function
  __device__ static void getBoundaryExtrapolatedValues(  Cell& c, const CState& UiP1, const CState& UiM1, const Float dt_half, int direction, float dx );
  __device__ static void computeIntercellFluxes( Cell& left, Cell& right, int direction );

} // namespace Kernels

template<>
__host__ void SolverMUSCL::computeFluxes<Device::gpu>(const Float dt_step) {
  int first = _grid.getFirstCellIndex() - 1;
  int last  = _grid.getLastCellIndex()  + 1;

  // use 1 extra warp
  const int numThreads = minNumberOfThreads( last );

  int numBlocks = last; //over subscribe for the hell of it

  Kernels::updateMids<<<numBlocks, numThreads, numThreads * sizeof(ConservedState)>>>( _grid, dt_step, _direction );
  cudaDeviceSynchronize();
  Kernels::computeFluxes<<<numBlocks, numThreads>>>( _grid, dt_step, _direction );
  cudaDeviceSynchronize();
}


__global__ void Kernels::updateMids(Grid grid, Float dt, int direction) {
  extern __shared__ CState buff[];

  const int bid = blockIdx.x;
  const int tid = threadIdx.x;

  const int first = grid.getFirstCellIndex() - 1;
  const int last  = grid.getLastCellIndex()  + 1;

  if (direction == 0) {
    /*
    In the original, j=first; j<last. And we load 1 to the left of first
    and the final one we load is at j==last
    */
    if ( 
      tid >= first - 1 and tid < last + 1 and
      bid >= first     and bid < last
    )
      buff[tid] = grid.getCell( tid, bid ).getCons();
    
    __syncthreads();

    if ( tid >= first and tid < last and bid >= first and bid < last ) {
      Cell& c = grid.getCell( tid, bid );

      Kernels::getBoundaryExtrapolatedValues( c, buff[tid+1], buff[tid-1], dt * 0.5, direction, grid.getDx() );
    }
  
    // now we have updated URMid and ULmid in all of the cells
    // __syncthreads();
  }

  else if (direction == 1) {
    if ( 
      tid >= first - 1 and tid < last + 1 and
      bid >= first     and bid < last
    )
      // load CStates into shared memory
      // note how the indices are the other way around here
      buff[tid] = grid.getCell( bid, tid ).getCons();
  
    __syncthreads();
  
    if ( tid >= first and tid < last and bid >= first and bid < last ) {
      Cell& c = grid.getCell( bid, tid );
      Kernels::getBoundaryExtrapolatedValues( c, buff[tid+1], buff[tid-1], dt * 0.5, direction, grid.getDx() );
    }
    
    // now we have updated URMid and ULmid in all of the cells
    // __syncthreads();
  }

}

__global__ void Kernels::computeFluxes(Grid grid, Float dt, int direction) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;

  const int first = grid.getFirstCellIndex() - 1;
  const int last  = grid.getLastCellIndex()  + 1;

  if (direction == 0) {
    if ( tid >= first and tid < last and bid >= first and bid < last ) {
      Cell& left  = grid.getCell( tid    , bid );
      Cell& right = grid.getCell( tid + 1, bid );
      Kernels::computeIntercellFluxes( left, right, direction );
    }
      
  }

  /*
    Note we have unrolled this loop
  */
  
  else if (direction == 1) {
    if ( tid >= first and tid < last and bid >= first and bid < last ) {
      // these two cells are miles apart in memory
      Cell& left  = grid.getCell( bid, tid );
      Cell& right = grid.getCell( bid, tid + 1 );
      Kernels::computeIntercellFluxes( left, right, direction );
    }
  }
}


__device__ static void Kernels::computeIntercellFluxes(Cell& left, Cell& right, int direction) {
  PrimitiveState WL;
  WL.fromCons(left.getURMid());

  PrimitiveState WR;
  WR.fromCons(right.getULMid());

  riemann::Riemann solver(WL, WR, direction);
  ConservedFlux    csol = solver.solveOnGpu();

  left.setCFlux(csol);
}


/**
  TODO: check that limiter::limiterGetLimitedSlope produces same result on cpu as gpu
  TODO: general correctness check
*/
__device__ static void Kernels::getBoundaryExtrapolatedValues( Cell& c, const CState& UiP1, const CState& UiM1, const Float dt_half, int direction, float dx ) {
  // First get the slope.
  CState        slope;
  const CState& Ui = c.getCons();

  limiter::limiterGetLimitedSlope( UiP1, Ui, UiM1, slope );


  Float rhoi   = Ui.getRho();
  Float rhovxi = Ui.getRhov(0);
  Float rhovyi = Ui.getRhov(1);
  Float Ei     = Ui.getE();

  // Get the left sloped state
  Float  rhoL   = rhoi - 0.5 * slope.getRho();
  Float  rhovxL = rhovxi - 0.5 * slope.getRhov(0);
  Float  rhovyL = rhovyi - 0.5 * slope.getRhov(1);
  Float  EL     = Ei - 0.5 * slope.getE();
  CState UL(rhoL, rhovxL, rhovyL, EL);

  // Get the left flux given the states.
  CFlux FL;
  FL.getCFluxFromCstate(UL, direction);

  // Get the right sloped state
  Float  rhoR   = rhoi + 0.5 * slope.getRho();
  Float  rhovxR = rhovxi + 0.5 * slope.getRhov(0);
  Float  rhovyR = rhovyi + 0.5 * slope.getRhov(1);
  Float  ER     = Ei + 0.5 * slope.getE();
  CState UR(rhoR, rhovxR, rhovyR, ER);

  // Get the right flux given the states.
  CFlux FR;
  FR.getCFluxFromCstate(UR, direction);

  Float dtdx_half = dt_half / dx;


  Float rhoLmid   = rhoi + dtdx_half * (FL.getRho() - FR.getRho()) - 0.5 * slope.getRho();
  Float rhovxLmid = rhovxi + dtdx_half * (FL.getRhov(0) - FR.getRhov(0)) - 0.5 * slope.getRhov(0);
  Float rhovyLmid = rhovyi + dtdx_half * (FL.getRhov(1) - FR.getRhov(1)) - 0.5 * slope.getRhov(1);
  Float ELmid     = Ei + dtdx_half * (FL.getE() - FR.getE()) - 0.5 * slope.getE();

  CState ULmid(rhoLmid, rhovxLmid, rhovyLmid, ELmid);
  c.setULMid(ULmid);


  Float rhoRmid   = rhoi + dtdx_half * (FL.getRho() - FR.getRho()) + 0.5 * slope.getRho();
  Float rhovxRmid = rhovxi + dtdx_half * (FL.getRhov(0) - FR.getRhov(0)) + 0.5 * slope.getRhov(0);
  Float rhovyRmid = rhovyi + dtdx_half * (FL.getRhov(1) - FR.getRhov(1)) + 0.5 * slope.getRhov(1);
  Float ERmid     = Ei + dtdx_half * (FL.getE() - FR.getE()) + 0.5 * slope.getE();

  CState URmid(rhoRmid, rhovxRmid, rhovyRmid, ERmid);
  c.setURMid(URmid);
}


