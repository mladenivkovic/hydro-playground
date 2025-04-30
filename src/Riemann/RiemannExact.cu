#include "RiemannExact.h"


/**
 * Computes the star region pressure and velocity given the left and right
 * primitive states. This is the iterative part that determines the star
 * state pressure. See Section 3.3 in the theory document.


  TODO: correctness test
  TODO: figure out if there's a way around this highly divergent code - might be quicker to just do all iters!
 */
 template <>
 __device__ inline void RiemannExact::computeStarStates<Device::gpu>() {
 
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
      //  warning(
      //    "Iteration for central pressure needs more than " + std::to_string(niter)
      //    + " steps. Force-quitting iteration. Old-to-new ratio is "
      //    + std::to_string(std::abs(1. - pguess / pold))
      //  );
       break;
     }
   } while (2. * abs((pguess - pold) / (pguess + pold)) >= cst::EPSILON_ITER);
 
   if (pguess <= cst::SMALLP) {
     pguess = cst::SMALLP;
   }
 
   _vstar = vLdim - fp(pguess, _left, AL, BL, aL);
   _pstar = pguess;
 }

__device__ ConservedFlux RiemannExact::solveOnGpu() {
  if ( hasVacuum() ) {
    PrimitiveState vac = solveVacuum<Device::gpu>();
    ConservedFlux  sol(vac, _dim); // pretty sure we have access to these member variables
    return sol;
  }
  else {
    computeStarStates<Device::gpu>();
    ConservedFlux sol = sampleSolution<Device::gpu>();
    return sol;
  }
}

