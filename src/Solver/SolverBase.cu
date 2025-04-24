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
  Kernels::integrateHydro<<<256, 256, (256+1)*sizeof(ConservedState)>>>(_grid, dt_step, _direction, _grid.getDx(), first, last);
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


/**
  TODO: correctness test and add logging

*/
template <>
void SolverBase::computeDt<Device::gpu>() {
  size_t first = _grid.getFirstCellIndex();
  size_t last  = _grid.getLastCellIndex();

  // alloc space for the dt
  Float* d_dt = nullptr;
  Float  h_dt;
  cudaMalloc( (void**)&d_dt, sizeof(Float) );

  // easier to do with 1 block due to synchronisation
  Kernels::computeDt<<<1,256, 2*256*sizeof(Float)>>>(_grid, _params.getCcfl(), d_dt, first, last, _step_count <= 5); // ...

  cudaDeviceSynchronize();
  cudaMemcpy( (void*)&h_dt, (void*)d_dt, sizeof(Float), cudaMemcpyDeviceToHost );

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

  for (size_t j=first; j<last; j++) {
    Cell&           c  = grid.getCell( first + tid, j );
    PrimitiveState& p  = c.getPrim();
    Float           vx = abs(p.getV(0));
    Float           vy = abs(p.getV(1));
    Float           a  = p.getSoundSpeed();
    Float           Sx = a + vx;
    Float           Sy = a + vy;
    vxmax              = max( Sx, vxmax ); // vxmax = std::max( Sx, vxmax )
    vymax              = max( Sy, vymax ); // vymax = std::max( Sy, vymax )

    // store in shared mem
    buff[2*tid+0] = vxmax;
    buff[2*tid+1] = vymax;

  }

  __syncthreads();

  Float vxdx = -9E10;
  Float vydx = -9E10;
  if ( tid == 0 ) {
    // get the max of each over the shared mem
    for (size_t i=0; i<blockDim.x; i++) {
      vxdx = max( vxdx, buff[2*i + 0] );
      vydx = max( vydx, buff[2*i + 1] );
    }

    // vxdx *= 1. / grid.getDx();
    // vydx *= 1. / grid.getDx();

    *dt = ccfl * grid.getDx() / ( vxdx + vydx );

    if ( suppress_dt ) *dt *= 0.2;
  }
}



