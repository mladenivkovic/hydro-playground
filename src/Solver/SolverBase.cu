#include "SolverBase.h"
#include "Gas.h"

namespace Kernels{
  __global__ void integrateHydro(Grid grid, const Float dt_step, int direction, float dx, size_t first, size_t last);

} // namespace Kernels


/**
  Including some device stuff for the solver base
*/
template <>
void SolverBase::integrateHydro<Device::gpu>(const Float dt_step) {
  // just need to launch the kernel here. No returns needed

  size_t first = _grid.getFirstCellIndex();
  size_t last  = _grid.getLastCellIndex();

  // Yes we are hardcoding the 256 width here
  // Need an extra one because we read from one to the left
  Kernels::integrateHydro<<<256, 256, (256+1)*sizeof(ConservedState)>>>(_grid, dt_step, _direction, _grid.getDx(), first, last);
  cudaDeviceSynchronize();
}

/*
  Todo: write a getter for direction

*/
__global__ void Kernels::integrateHydro(Grid grid, const Float dt_step, int direction, float dx, size_t first, size_t last) {
  extern __shared__ ConservedState flux_buff[];

  int bid = blockIdx.x;
  int tid = threadIdx.x;

  const Float dtdx = dt_step / dx;

  if (direction==0) {
    // load the whole column into shared memory
    flux_buff[tid + 1] = grid.getCell( first+tid, first+bid ).getCFlux();
    
    if (tid==0) {
      flux_buff[0] = grid.getCell( first-1, first+bid ).getCFlux();
    }
  }
  
  else if (direction == 1) {
    // load the whole row into shared memory
    flux_buff[tid + 1] = grid.getCell( first+bid, first+tid ).getCFlux();
  
    if (tid==0) {
      flux_buff[0] = grid.getCell( first+bid, first-1 ).getCFlux();
    }
  }

  __syncthreads();

  // take referebce to our cons state
  ConservedState& cr = grid.getCell( first+bid, first+tid ).getCons();

  // rho: remeber we're updating "right". And because of stupid indexing
  // the right hand one is tid+1
  Float rho = cr.getRho()  + dtdx * ( flux_buff[ tid ].getRho() - flux_buff[ tid+1 ].getRho() );
  
  // rhov 0
  Float vx = cr.getRhov(0) + dtdx * ( flux_buff[ tid ].getRhov(0) - flux_buff[ tid+1 ].getRhov(0) );
  
  // rhov 1
  Float vy = cr.getRhov(1) + dtdx * ( flux_buff[ tid ].getRhov(1) - flux_buff[ tid+1 ].getRhov(1) );
  
  // e
  Float e = cr.getE()      + dtdx * ( flux_buff[ tid ].getE() - flux_buff[ tid+1 ].getE() );
  
  cr.setRho(rho);
  cr.setRhov(0, vx);
  cr.setRhov(1, vy);
  cr.setE(e);
}

