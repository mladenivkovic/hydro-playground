#include "RiemannHLLC.h"

__device__ ConservedFlux RiemannHLLC::solveOnGpu() {

  if (hasVacuum()) {
    PrimitiveState vac = solveVacuum<Device::gpu>();
    ConservedFlux  sol(vac, _dim);
    return sol;
  }

  computeWaveSpeedEstimates();

  return sampleHLLCSolution();
}
