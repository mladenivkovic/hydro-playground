#pragma once

// Shouldn't be necessary if we compile with nvcc, but since this file is included
// in cpp files, we need to make sure that __global__ is defined
#include <cuda_runtime_api.h>

#include "Grid.h"

__global__ void dummyKernel();

void launchDummyKernel();

__global__ void testGridKernel( Grid* g );
void launchTestGridKernel(Grid* p);

