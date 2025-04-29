#include "SolverMUSCL.h"
#include "Gas.h"
#include "Limiter.h"

using CState = ConservedState;
using CFlux  = ConservedFlux;

/**

  TODO: add in 

*/

namespace Kernels{
  __global__ void computeFluxes(Grid grid, Float dt, int direction);

  // not a kernel, but rather a device function
  __device__ static void getBoundaryExtrapolatedValues(  Cell& c, const CState& UiP1, const CState& UiM1, const Float dt_half, int direction, float dx );
  __device__ static void computeIntercellFluxes( Cell& left, Cell& right );

} // namespace Kernels

template<>
__host__ void SolverMUSCL::computeFluxes<Device::gpu>(const Float dt_step) {

  // from j=first; j<last means we need blocks from 1 -> 258
  Kernels::computeFluxes<<<258, 256, sizeof(ConservedState)>>>( _grid, dt_step, _direction );
}


// just launch with 256 threads and use the first 4 to clean up
// we go from 0 -> 259 inclusive!
__global__ void Kernels::computeFluxes(Grid grid, Float dt, int direction) {

  // start by loading a lot of stuff into shared memory
  extern __shared__ CState buff[];

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  if ( direction == 0 ) {
    // load CStates into shared memory
    buff[tid] = grid.getCell( tid, bid + 1 ).getCons();

    if ( tid < 4 ) {
      buff[ tid + 256 ] = grid.getCell( tid + 256, bid + 1 ).getCons();
    }

    __syncthreads();

    // wrong indices!
    Cell& c = grid.getCell( tid + 1, bid );
    Kernels::getBoundaryExtrapolatedValues( c, buff[tid], buff[tid+2], dt * 0.5, direction, grid.getDx() );

    if ( tid < 4 ) {
      // clean up the other cells we missed
    }

    __syncthreads();

    // now we have updated URMid and ULmid in all of the cells
    // Kernels::computeIntercellFluxes();

  }

  else if ( direction == 1 ) {

  }
  
}

/**
  TODO: check that limiter::limiterGetLimitedSlope produces same result on cpu as gpu
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

