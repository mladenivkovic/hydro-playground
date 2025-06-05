#include "RiemannHLLC.h"

#include <cmath>

#include "Constants.h"
#include "Gas.h"
#include "Timer.h"

/**
 * @brief solve the Riemann problem with the HLLC solver.
 *
 * @return the intercell flux of conserved variables corresponding to the
 * solution sampled at x=0.
 */
ConservedFlux RiemannHLLC::solve() {

  timer::Timer tick(timer::Category::Riemann);

  if (hasVacuum()) {
    PrimitiveState vac = solveVacuum<Device::cpu>();
    ConservedFlux  sol(vac, _dim);
    return sol;
  }

  computeWaveSpeedEstimates();

  return sampleHLLCSolution();
}

