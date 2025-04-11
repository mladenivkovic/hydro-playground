#include <iostream>

//! Not nice but I can't get the tests to link, so we move the cuda stuff in here
#include "Grid.h"

// Reuse names by putting them in this namespace
namespace Kernels{
  __global__ void collectTotalMassFromGpu(Grid, Float*, size_t, size_t);

} // namespace Kernels

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

  int offset = last - first;
  
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
  
  size_t numThreadsInEachDimension = last - first;
  size_t totalThreads              = numThreadsInEachDimension * numThreadsInEachDimension;

  
  // malloc
  cudaErrorCheck( cudaMalloc((void**)&d_output, sizeof(Float)) );
  
  // launch kernel
  // I happen to know that the grid is 256 * 256
  Kernels::collectTotalMassFromGpu<<<1,256, 256 * sizeof(Float)>>>( *this, d_output, first, last );

  // block
  cudaDeviceSynchronize();

  // copy back
  cudaErrorCheck(cudaMemcpy( (void*)&h_output, (void*)d_output, sizeof(Float), cudaMemcpyDeviceToHost ));

  return h_output;
}

