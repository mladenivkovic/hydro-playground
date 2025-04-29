#include <iostream>

//! Not nice but I can't get the tests to link, so we move the cuda stuff in here
#include "Grid.h"
#include "assert.h"

/**

NOTE - when we do getCell(i,j) - the cell at (i+1,j) will be next in memory

*/

// Reuse names by putting them in this namespace
namespace Kernels{
  __global__ void collectTotalMassFromGpu(Grid, Float*, size_t, size_t);
  __global__ void convertPrimToCons(Grid, size_t, size_t);
  __global__ void resetFluxes(Grid, size_t, size_t);
  __global__ void applyBoundaryConditions(Grid);

} // namespace Kernels

namespace DeviceFunctions{
  static __device__ void realToGhost(Grid&, Cell**, Cell**, Cell**, Cell**, size_t);
} // namespace Device

__host__ void Grid::transferCellsToDevice() {
  size_t nxTot       = getNxTot();
  size_t total_cells = 0;

  if      (Dimensions==1)
    total_cells = nxTot;
  else if (Dimensions==2)
    total_cells = nxTot * nxTot;
  else
    error("Not implemented yet");

  // malloc on the device
  cudaErrorCheck(cudaMalloc( (void**)&_dev_cells, total_cells * sizeof(Cell) ));

  // copy over
  cudaErrorCheck(cudaMemcpy( (void*)_dev_cells, (void*)_host_cells, total_cells * sizeof(Cell), cudaMemcpyHostToDevice ));
}

__host__ void Grid::clean() {
  if (_host_cells == nullptr)
  error("Where did the cells array go??");
  delete[] _host_cells;
  cudaFree(_dev_cells);
}

/**
  - offset - "first" from the original function. We have enough threads to 

  Put in some trivial multithreading for my enjoyment...
*/
__global__ void Kernels::collectTotalMassFromGpu( Grid grid, Float* result, size_t first, size_t last ) {
  // shared memory for fun
  extern __shared__ Float buff[];

  int threadId   = (blockIdx.x * blockDim.x + threadIdx.x);
  buff[threadId] = 0.0;

  for (int i=first; i<last; i++) {
    // add on offset
    buff[threadId] += grid.getCell(i, threadId + first).getPrim().getRho();
  }

  __syncthreads();

  // clean up
  if ( threadId == 0 ) {
    *result = 0.;
    const Float dx2 = grid.getDx() * grid.getDx();

    // 1d kernel
    for (int i=0; i<blockDim.x; i++)
      *result += buff[i];

    *result *= dx2;
  }

}

__host__ Float Grid::collectTotalMassFromGpu() {
  Float  h_output;
  Float* d_output = nullptr;
  
  size_t first = getFirstCellIndex();
  size_t last  = getLastCellIndex();
  
  // malloc
  cudaErrorCheck( cudaMalloc((void**)&d_output, sizeof(Float)) );
  
  // launch kernel
  // I happen to know that the grid is 256 * 256
  Kernels::collectTotalMassFromGpu<<<1,256, 256 * sizeof(Float)>>>( *this, d_output, first, last );

  // block
  cudaDeviceSynchronize();

  // copy back
  cudaErrorCheck(cudaMemcpy( (void*)&h_output, (void*)d_output, sizeof(Float), cudaMemcpyDeviceToHost ));

  cudaFree( d_output );

  return h_output;
}

/**
  TODO: correctness test

*/
template<>
__host__
void Grid::convertPrim2Cons<Device::gpu>() {
  size_t first = getFirstCellIndex();
  size_t last  = getLastCellIndex();

  Kernels::convertPrimToCons<<<256,256>>>( *this, first, last );
  cudaDeviceSynchronize();
}


__global__ void Kernels::convertPrimToCons( Grid grid, size_t first, size_t last ) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  grid.getCell( first + tid, first + bid ).prim2cons();
}


template<>
__host__
void Grid::resetFluxes<Device::gpu>() {
  // launch
  Kernels::resetFluxes<<<256,256>>>(*this, getFirstCellIndex(), getLastCellIndex());
  cudaDeviceSynchronize();
}

/**
  TODO: correctness test

*/
__global__ void Kernels::resetFluxes( Grid grid, size_t first, size_t last ) {
  int bid = blockIdx.x;
  int tid = threadIdx.x;

  grid.getCell( first + tid, first + bid ).getCFlux().clear();
}


template<>
__host__ void Grid::applyBoundaryConditions<Device::gpu>() {
  // launch kernel single-threaded (but we take up a whole warp just in case)
  // This one is really crucial - so let's do it single threaded until everything
  // else works
  Kernels::applyBoundaryConditions<<<1,32>>>(*this);
  cudaDeviceSynchronize();
}

__global__ void Kernels::applyBoundaryConditions(Grid grid) {
  int tid = threadIdx.x;

  // single threaded for the sake of correctness
  if ( tid == 0 ) {
    const size_t nbc       = grid.getNBC();
    const size_t firstReal = grid.getFirstCellIndex();
    const size_t lastReal  = grid.getLastCellIndex();

    assert(Dimensions==2);

    Cell** real_left   = new Cell*[nbc];
    Cell** real_right  = new Cell*[nbc];
    Cell** ghost_left  = new Cell*[nbc];
    Cell** ghost_right = new Cell*[nbc];

    // left-right boundaries
    for (size_t j = firstReal; j < lastReal; j++) {
      for (size_t i = 0; i < firstReal; i++) {
        real_left[i]   = &(grid.getCell(firstReal + i, j));
        real_right[i]  = &(grid.getCell(lastReal - firstReal + i, j));
        ghost_left[i]  = &(grid.getCell(i, j));
        ghost_right[i] = &(grid.getCell(lastReal + i, j));
      }
      DeviceFunctions::realToGhost(grid, real_left, real_right, ghost_left, ghost_right, 0);
    }

    // upper-lower boundaries
    for (size_t i = firstReal; i < lastReal; i++) {
      for (size_t j = 0; j < firstReal; j++) {
        real_left[j]   = &(grid.getCell(i, firstReal + j));
        real_right[j]  = &(grid.getCell(i, lastReal - firstReal + j));
        ghost_left[j]  = &(grid.getCell(i, j));
        ghost_right[j] = &(grid.getCell(i, lastReal + j));
      }
      DeviceFunctions::realToGhost(grid, real_left, real_right, ghost_left, ghost_right, 1);
    }

    delete[] real_left;
    delete[] real_right;
    delete[] ghost_left;
    delete[] ghost_right;
  }
}

// all these arrays should have size nbc!
static __device__ void DeviceFunctions::realToGhost(Grid& grid, Cell** real_left, Cell** real_right, Cell** ghost_left, Cell** ghost_right, size_t dimension) {
  size_t nbc = grid.getNBC();

  switch (grid.getBoundaryType()) {
    case BC::BoundaryCondition::Periodic:
      for (size_t i = 0; i < nbc; i++) {
        ghost_left[i]->copyBoundaryData(real_right[i]);
        ghost_right[i]->copyBoundaryData(real_left[i]);
      }
      break;
  
    case BC::BoundaryCondition::Reflective:
      for (size_t i = 0; i < nbc; i++) {
        ghost_left[i]->copyBoundaryDataReflective(real_left[nbc - i - 1], dimension);
        ghost_right[i]->copyBoundaryDataReflective(real_right[nbc - i - 1], dimension);
      }
      break;
  
    case BC::BoundaryCondition::Transmissive:
      for (size_t i = 0; i < nbc; i++) {
        ghost_left[i]->copyBoundaryData(real_left[nbc - i - 1]);
        ghost_right[i]->copyBoundaryData(real_right[nbc - i - 1]);
      }
      break;
  
    default:
      assert(false);
    }
}
