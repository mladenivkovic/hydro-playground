#include "RiemannExact.h"

#include <cmath>

#include "Constants.h"
#include "Gas.h"
#include "Timer.h"

/**
 * @brief solve the Riemann problem with the Exact solver.
 *
 * @return the intercell flux of conserved variables corresponding to the
 * solution sampled at x=0.
 */
ConservedFlux RiemannExact::solve() {

  timer::Timer tick(timer::Category::Riemann);

  if (hasVacuum()) {
    PrimitiveState vac = solveVacuum();
    ConservedFlux  sol(vac, _dim);
    return sol;
  }

  computeStarStates();
  ConservedFlux sol = sampleSolution();
  return sol;
}


/**
 * Computes the star region pressure and velocity given the left and right
 * primitive states. This is the iterative part that determines the star
 * state pressure. See Section 3.3 in the theory document.
 */
inline void RiemannExact::computeStarStates() {

  Float rhoL = _left.getRho();
  Float rhoR = _right.getRho();

  Float pL = _left.getP();
  Float pR = _right.getP();

  Float vLdim = _left.getV(_dim);
  Float vRdim = _right.getV(_dim);


  Float AL = cst::TWOOVERGP1 / rhoL;
  Float AR = cst::TWOOVERGP1 / rhoR;
  Float BL = cst::GM1OGP1 * pL;
  Float BR = cst::GM1OGP1 * pR;
  Float aL = _left.getSoundSpeed();
  Float aR = _right.getSoundSpeed();

  Float delta_v = vRdim - vLdim;


  /* Find initial guess for star pressure */
  Float ppv    = 0.5 * (pL + pR) - 0.125 * delta_v * (rhoL + rhoR) * (aL + aR);
  Float pguess = ppv;

  if (pguess < cst::SMALLP) {
    pguess = cst::SMALLP;
  }

  // Newton-Raphson iteration
  int   niter = 0;
  Float pold  = pguess;

  do {
    niter++;
    pold         = pguess;
    Float fL     = fp(pguess, _left, AL, BL, aL);
    Float fR     = fp(pguess, _right, AR, BR, aR);
    Float dfpdpL = dfpdp(pguess, _left, AL, BL, aL);
    Float dfpdpR = dfpdp(pguess, _right, AR, BR, aR);
    pguess       = pold - (fL + fR + delta_v) / (dfpdpL + dfpdpR);
    if (pguess < cst::EPSILON_ITER) {
      pguess = cst::SMALLP;
    }
    if (niter > 100) {
      warning(
        "Iteration for central pressure needs more than " + std::to_string(niter)
        + " steps. Force-quitting iteration. Old-to-new ratio is "
        + std::to_string(std::abs(1. - pguess / pold))
      );
      break;
    }
  } while (2. * std::abs((pguess - pold) / (pguess + pold)) >= cst::EPSILON_ITER);

  if (pguess <= cst::SMALLP) {
    pguess = cst::SMALLP;
  }

  _vstar = vLdim - fp(pguess, _left, AL, BL, aL);
  _pstar = pguess;
}


/**
 * The left/right part of the pressure function.
 * Equation 60-63 in Theory document.
 *
 * @param pguess Star state pressure guess
 * @param state The left or right state for which to compute f_p
 * @param A   A_L or A_R (Eq. 62)
 * @param B   B_L or B_R (Eq. 63)
 * @param cs  soundspeed of state
 */
inline Float RiemannExact::fp(
  const Float                     pguess,
  const PrimitiveState& state,
  const Float                     A,
  const Float                     B,
  const Float                     cs
) {

  Float p = state.getP();

  if (pguess > p) {
    // we have a shock situation
    return (pguess - p) * std::sqrt(A / (pguess + B));
  }
  // we have a rarefaction situation
  return cst::TWOOVERGM1 * cs * (std::pow(pguess / p, cst::BETA) - 1.);
}


/**
 * The derivative of the left/right part of the pressure function.
 * Equation 64 in Theory document.
 *
 * @param pguess Star state pressure guess
 * @param state The left or right state for which to compute f_p
 * @param A   A_L or A_R (Eq. 65)
 * @param B   B_L or B_R (Eq. 66)
 * @param cs  soundspeed of state
 */
inline Float RiemannExact::dfpdp(
  const Float                     pguess,
  const PrimitiveState& state,
  const Float                     A,
  const Float                     B,
  const Float                     cs
) {

  Float p   = state.getP();
  Float rho = state.getRho();

  if (pguess > p) {
    // we have a shock situation
    return std::sqrt(A / (pguess + B)) * (1. - 0.5 * (pguess - p) / (pguess + B));
  }
  // we have a rarefaction situation
  return 1. / (rho * cs) * std::pow(pguess / p, -0.5 * cst::GP1 * cst::ONEOVERGAMMA);
}
