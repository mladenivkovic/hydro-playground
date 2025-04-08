#include "Dummy.cuh"

#include <iostream>

__global__ void dummyKernel() {
  printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}


void launchDummyKernel(){
  int numDevices = 0;
  cudaGetDeviceCount( &numDevices );

  printf("Inside launchDummyKernel(). Found %d devices\n", numDevices);
  
  if ( numDevices > 0 ) {
    printf("Attempting to launch!\n");
    dummyKernel<<<2,8>>>();
    cudaDeviceSynchronize();
    printf("Finished launch!\n");
  }
}


__global__ void testGridKernel( Grid* g ) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i==0) {
    printf( "Copied grid! g->nx=%d\n", g->getNx() );
  }
    
}


void launchTestGridKernel(Grid* p) {
  testGridKernel<<<1,1>>>(p);
}

