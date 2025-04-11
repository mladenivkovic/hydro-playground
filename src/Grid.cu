#include <iostream>

//! Not nice but I can't get the tests to link, so we move the cuda stuff in here
#include "Grid.h"


__host__ void Grid::transferCellsToDevice() {
  cudaError_t err;

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

