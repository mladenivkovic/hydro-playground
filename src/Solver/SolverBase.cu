#include "SolverBase.h"
#include "Gas.h"

namespace Kernels{
  __global__ void integrateHydro(Grid grid, const Float dt_step, int direction, float dx, size_t first, size_t last);
  __global__ void computeDt(Grid grid, Float ccfl, Float* dt, size_t first, size_t last, bool suppress_dt);

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

  const int numThreads = minNumberOfThreads( 257 );
  const int numBlocks  = numThreads;
  // const int numThreads = 296;

  Kernels::integrateHydro<<<numBlocks, numThreads, numThreads*sizeof(ConservedState)>>>(_grid, dt_step, _direction, _grid.getDx(), first, last);
  cudaDeviceSynchronize();
}

/**
  TODO: correctness test

  The shared memory use here absolutely results in bank conflicts. But I think it should be faster
  than loads from main memory anyhow (most of the cache line is wasted when we load the fluxes from
  main memory, and by loading into shared memory we can pay this cost only once, I hope).

  Besides, on my local machine 256 cells doesn't fit into L1 cache, whereas 256 ConservedStates do.

*/
__global__ void Kernels::integrateHydro(Grid grid, const Float dt_step, int direction, float dx, size_t first, size_t last) {
  extern __shared__ ConservedState flux_buff[];
  Cell& chcs = grid.getCell(97,67);
  
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;

  const Float dtdx = dt_step / dx;

  if (
    direction==0     and
    tid >= first - 1 and tid < last and
    bid >= first     and bid < last
  ) {
    // load the whole column into shared memory
    flux_buff[tid] = grid.getCell( tid, bid ).getCFlux();
  }
  
  /*
  We still ask the "strong" condition on tid because these are the
  ones we want to be algined in memory
  */
  else if (
    direction == 1   and
    tid >= first - 1 and tid < last and
    bid >= first     and bid < last
  ) {
    // load the whole row into shared memory
    flux_buff[tid] = grid.getCell( bid,tid ).getCFlux();
  
  }

  __syncthreads();

  // take reference to our cons state
  if (
    tid >= first and tid < last and
    bid >= first and bid < last
  ) {
    ConservedState& cr = (direction==0)  ?
      grid.getCell( tid, bid ).getCons() :
      grid.getCell( bid, tid ).getCons() ;
    
    // rho: remember we're updating "right".
    Float rho = cr.getRho()  + dtdx * ( flux_buff[ tid - 1 ].getRho() - flux_buff[ tid ].getRho() );
    
    // rhov 0
    Float vx = cr.getRhov(0) + dtdx * ( flux_buff[ tid - 1 ].getRhov(0) - flux_buff[ tid ].getRhov(0) );
    
    // rhov 1
    Float vy = cr.getRhov(1) + dtdx * ( flux_buff[ tid - 1 ].getRhov(1) - flux_buff[ tid ].getRhov(1) );
    
    // e
    Float e = cr.getE()      + dtdx * ( flux_buff[ tid - 1 ].getE() - flux_buff[ tid ].getE() );
    
    cr.setRho(rho);
    cr.setRhov(0, vx);
    cr.setRhov(1, vy);
    cr.setE(e);
  
  }
}


/**
  TODO: correctness test and add logging
    - update - seems to give the same values as the cpu version

*/
template <>
void SolverBase::computeDt<Device::gpu>() {
  size_t first = _grid.getFirstCellIndex();
  size_t last  = _grid.getLastCellIndex();

  // alloc space for the dt
  Float* d_dt = nullptr;
  Float  h_dt;
  cudaErrorCheck(cudaMalloc( (void**)&d_dt, sizeof(Float) ));

  size_t num_threads = minNumberOfThreads( _grid.getLastCellIndex() );

  // easier to do with 1 block due to synchronisation
  Kernels::computeDt<<<1,num_threads, 2*num_threads*sizeof(Float)>>>(_grid, _params.getCcfl(), d_dt, first, last, _step_count <= 5); // ...

  cudaDeviceSynchronize();
  cudaErrorCheck(cudaMemcpy( (void*)&h_dt, (void*)d_dt, sizeof(Float), cudaMemcpyDeviceToHost ));

  _dt = h_dt; // could do the copy directly into the member variable

  cudaFree( d_dt );
}


/**
  TODO: correctness test

*/
__global__ void Kernels::computeDt( Grid grid, Float ccfl, Float* dt, size_t first, size_t last, bool suppress_dt ) {
  extern __shared__ Float buff[]; // should be two for each thread

  // int bid = blockIdx.x;
  int tid = threadIdx.x;

  Float vxmax = 0.;
  Float vymax = 0.;

  if ( tid >= first and tid < last ) {
    for (size_t j=first; j<last; j++) {
      Cell&           c  = grid.getCell( tid, j );
      PrimitiveState& p  = c.getPrim();
      Float           vx = abs(p.getV(0));
      Float           vy = abs(p.getV(1));
      Float           a  = p.getSoundSpeed();
      Float           Sx = a + vx;
      vxmax              = Sx > vxmax ? Sx : vxmax;
      Float Sy           = a + vy;
      vymax              = Sy > vymax ? Sy : vymax;

    // store in shared mem
    buff[2*tid+0] = vxmax;
    buff[2*tid+1] = vymax;
    }
  }
  __syncthreads();

  if ( tid == 0 ) {
    Float vxdx = -9E10;
    Float vydx = -9E10;
    // get the max of each over the shared mem
    for (size_t i=first; i<last; i++) {
      Float vx = buff[2*i + 0];
      Float vy = buff[2*i + 1];

      /**
        Note the stupid name here!!! vxdx was so called because 
        it was vx and dx together. We need to change that name
      */
      vxdx = max( vxdx, vx );
      vydx = max( vydx, vy );
    }

    *dt = ccfl * grid.getDx() / ( vxdx + vydx );

    if ( suppress_dt ) *dt *= 0.2;
  }
}

