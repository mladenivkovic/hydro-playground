#include "RiemannExact.h"

#include <cmath>

#include "Constants.h"
#include "Gas.h"
#include "Timer.h"



/**
 * Computes the star region pressure and velocity given the left and right
 * primitive states. This is the iterative part that determines the star
 * state pressure. See Section 3.3 in the theory document.
 */
 template <>
 inline void RiemannExact::computeStarStates<Device::cpu>() {
 
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
 * @brief solve the Riemann problem with the Exact solver.
 *
 * @return the intercell flux of conserved variables corresponding to the
 * solution sampled at x=0.
 */
ConservedFlux RiemannExact::solve() {

  timer::Timer tick(timer::Category::Riemann);

  if (hasVacuum()) {
    PrimitiveState vac = solveVacuum<Device::cpu>();
    ConservedFlux  sol(vac, _dim);
    return sol;
  }

  computeStarStates<Device::cpu>();
  ConservedFlux sol = sampleSolution<Device::cpu>();
  return sol;
}
