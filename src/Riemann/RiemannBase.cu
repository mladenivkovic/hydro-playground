#include "RiemannBase.h"
#include "Constants.h"
#include <math.h>

/*
  APPARENTLY THE STD::POW THING IS A NON ISSUE.

*/

//! I'm just redoing it because of the std::pow thing. It won't work in device code.
//! TODO: again massively divergent
//! TODO: correctnesss
template <>
__device__ PrimitiveState RiemannBase::solveVacuum<Device::gpu>() {

  size_t otherdim = (_dim + 1) % 2;
  // x / t. We always center the problem at x=0, but to sample the solution in
  // general, we need to adapt this value.
  constexpr Float xovert = 0.;

  Float aL = _left.getSoundSpeed();
  Float aR = _right.getSoundSpeed();

  Float rhoL = _left.getRho();
  Float rhoR = _right.getRho();

  Float vLdim   = _left.getV(_dim);
  Float vLother = _left.getV(otherdim);
  Float vRdim   = _right.getV(_dim);
  Float vRother = _right.getV(otherdim);

  Float pL = _left.getP();
  Float pR = _right.getP();


  // Both vacuum states
  if (rhoL <= cst::SMALLRHO and rhoR <= cst::SMALLRHO) {
    return PrimitiveState(cst::SMALLRHO, cst::SMALLV, cst::SMALLV, cst::SMALLP);
  }

  Float rho_sol;
  Float vdim_sol;
  Float vother_sol;
  Float p_sol;
  if (rhoL <= cst::SMALLRHO) {
    // ------------------------
    // Left vacuum state
    // ------------------------
    Float SR  = vRdim - aR * cst::TWOOVERGM1; // vacuum front speed
    Float SHR = vRdim + aR;                   // speed of head of right rarefaction fan

    if (xovert <= SR) {
      // left vacuum
      rho_sol    = cst::SMALLRHO;
      vdim_sol   = cst::SMALLV;
      vother_sol = cst::SMALLV;
      p_sol      = cst::SMALLP;
    } else if (xovert < SHR) {
      // inside rarefaction
      Float precomp = powf(
        (cst::TWOOVERGP1 - cst::GM1OGP1 / aR * (vRdim - xovert)), cst::TWOOVERGM1
      );
      rho_sol    = rhoR * precomp;
      vdim_sol   = cst::TWOOVERGP1 * (cst::GM1HALF * vRdim - aR + xovert);
      vother_sol = vRother;
      p_sol      = pR * powf(precomp, cst::GAMMA);
    } else {
      // original right pstate
      rho_sol    = rhoR;
      vdim_sol   = vRdim;
      vother_sol = vRother;
      p_sol      = pR;
    }
  }

  else if (rhoR <= cst::SMALLRHO) {
    // ------------------------
    // Right vacuum state
    // ------------------------

    Float SL  = vLdim + aL * cst::TWOOVERGM1; // vacuum front speed
    Float SHL = vLdim - aL;                   // speed of head of left rarefaction fan

    if (xovert >= SL) {
      // right vacuum
      rho_sol    = cst::SMALLRHO;
      vdim_sol   = cst::SMALLV;
      vother_sol = cst::SMALLV;
      p_sol      = cst::SMALLP;
    } else if (xovert > SHL) {
      // inside rarefaction
      Float precomp = powf(
        (cst::TWOOVERGP1 + cst::GM1OGP1 / aL * (vLdim - xovert)), (cst::TWOOVERGM1)
      );
      rho_sol    = rhoL * precomp;
      vdim_sol   = cst::TWOOVERGP1 * (cst::GM1HALF * vLdim + aL + xovert);
      vother_sol = vLother;
      p_sol      = pL * powf(precomp, cst::GAMMA);
    } else {
      // original left pstate
      rho_sol    = rhoL;
      vdim_sol   = vLdim;
      vother_sol = vLother;
      p_sol      = pL;
    }
  } else {
    // ------------------------
    // Vacuum generating case
    // ------------------------

    Float SL  = vLdim + aL * cst::TWOOVERGM1; // vacuum front speed
    Float SR  = vRdim - aR * cst::TWOOVERGM1; // vacuum front speed
    Float SHL = vLdim - aL;                   // speed of head of left rarefaction fan
    Float SHR = vRdim + aR;                   // speed of head of right rarefaction fan

    if (xovert <= SHL) {
      // left original pstate
      rho_sol    = rhoL;
      vdim_sol   = vLdim;
      vother_sol = vLother;
      p_sol      = pL;
    } else if (xovert < SL) {
      // inside rarefaction fan from right to left
      Float precomp = powf(
        (cst::TWOOVERGP1 + cst::GM1OGP1 / aL * (vLdim - xovert)), cst::TWOOVERGM1
      );
      rho_sol    = rhoL * precomp;
      vdim_sol   = cst::TWOOVERGP1 * (cst::GM1HALF * vLdim + aL + xovert);
      vother_sol = vLother;
      p_sol      = pL * powf(precomp, cst::GAMMA);
    } else if (xovert < SR) {
      // vacuum region
      rho_sol    = cst::SMALLRHO;
      vdim_sol   = cst::SMALLV;
      vother_sol = cst::SMALLV;
      p_sol      = cst::SMALLP;
    } else if (xovert < SHR) {
      // inside rarefaction fan from left to right
      Float precomp = powf(
        (cst::TWOOVERGP1 - cst::GM1OGP1 / aR * (vRdim - xovert)), cst::TWOOVERGM1
      );
      rho_sol    = rhoR * precomp;
      vdim_sol   = cst::TWOOVERGP1 * (cst::GM1HALF * vRdim - aR + xovert);
      vother_sol = vRother;
      p_sol      = pR * powf(precomp, cst::GAMMA);
    } else {
      // right original pstate
      rho_sol    = rhoR;
      vdim_sol   = vRdim;
      vother_sol = vRother;
      p_sol      = pR;
    }
  }

  PrimitiveState sol(rho_sol, 0., 0., p_sol);
  sol.setV(_dim, vdim_sol);
  sol.setV(otherdim, vother_sol);
  return sol;

}

/**
  TODO: This function diverges enormously. It will be awful for perfornance. We might be better off launching sub-kernels
  TODO: correctness
*/
template <>
__device__ ConservedFlux RiemannBase::sampleSolution<Device::gpu>() {

  constexpr Float xovert   = 0.;
  size_t          otherdim = (_dim + 1) % 2;

  Float rhoL = _left.getRho();
  Float rhoR = _right.getRho();

  Float vLdim   = _left.getV(_dim);
  Float vRdim   = _right.getV(_dim);
  Float vLother = _left.getV(otherdim);
  Float vRother = _right.getV(otherdim);

  Float pL = _left.getP();
  Float pR = _right.getP();


  Float rho_sol;
  Float vdim_sol;
  Float vother_sol;
  Float p_sol;


  if (xovert <= _vstar) {

    // We're on the left side

    Float aL          = _left.getSoundSpeed();
    Float pstaroverpL = _pstar / pL;

    if (_pstar <= pL) {

      // left rarefaction

      Float SHL = vLdim - aL; // speed of head of left rarefaction fan
      if (xovert < SHL) {
        // we're outside the rarefaction fan
        rho_sol    = rhoL;
        vdim_sol   = vLdim;
        vother_sol = vLother;
        p_sol      = pL;
      } else {
        Float astarL = aL * powf(pstaroverpL, cst::BETA);
        Float STL    = _vstar - astarL; // speed of tail of left rarefaction fan
        if (xovert < STL) {
          // we're inside the fan
          Float precomp = powf(
            (cst::TWOOVERGP1 + cst::GM1OGP1 / aL * (vLdim - xovert)), cst::TWOOVERGM1
          );
          rho_sol    = rhoL * precomp;
          vdim_sol   = cst::TWOOVERGP1 * (cst::GM1HALF * vLdim + aL + xovert);
          vother_sol = vLother;
          p_sol      = pL * powf(precomp, cst::GAMMA);
        } else {
          // we're in the star region
          rho_sol    = rhoL * powf(pstaroverpL, cst::ONEOVERGAMMA);
          vdim_sol   = _vstar;
          vother_sol = vLother;
          p_sol      = _pstar;
        }
      }
    } else {

      // left shock

      // left shock speed
      Float tempsqrt = 0.5 * cst::GP1 * cst::ONEOVERGAMMA * pstaroverpL + cst::BETA;
      Float SL       = vLdim - aL * sqrtf(tempsqrt);

      if (xovert < SL) {
        // we're outside the shock
        rho_sol    = rhoL;
        vdim_sol   = vLdim;
        vother_sol = vLother;
        p_sol      = pL;
      } else {
        // we're in the star region
        rho_sol    = (pstaroverpL + cst::GM1OGP1) / (cst::GM1OGP1 * pstaroverpL + 1.) * rhoL;
        vdim_sol   = _vstar;
        vother_sol = vLother;
        p_sol      = _pstar;
      }
    }
  } else {
    // We're on the right side
    Float aR          = _right.getSoundSpeed();
    Float pstaroverpR = _pstar / pR;

    if (_pstar <= pR) {

      // right rarefaction

      Float SHR = vRdim + aR; // speed of head of right rarefaction fan
      if (xovert > SHR) {
        // we're outside the rarefaction fan
        rho_sol    = rhoR;
        vdim_sol   = vRdim;
        vother_sol = vRother;
        p_sol      = pR;
      } else {
        Float astarR = aR * powf(pstaroverpR, cst::BETA);
        Float STR    = _vstar + astarR; // speed of tail of right rarefaction fan
        if (xovert > STR) {
          // we're inside the fan
          Float precomp = powf(
            (cst::TWOOVERGP1 - cst::GM1OGP1 / aR * (vRdim - xovert)), cst::TWOOVERGM1
          );
          rho_sol    = rhoR * precomp;
          vdim_sol   = cst::TWOOVERGP1 * (cst::GM1HALF * vRdim - aR + xovert);
          vother_sol = vRother;
          p_sol      = pR * powf(precomp, cst::GAMMA);
        } else {
          // we're in the star region
          rho_sol    = rhoR * powf(pstaroverpR, cst::ONEOVERGAMMA);
          vdim_sol   = _vstar;
          vother_sol = vRother;
          p_sol      = _pstar;
        }
      }
    } else {

      // right shock

      // right shock speed
      Float tempsqrt = 0.5 * cst::GP1 * cst::ONEOVERGAMMA * pstaroverpR + cst::BETA;
      Float SR       = vRdim + aR * sqrtf(tempsqrt);

      if (xovert > SR) {
        // we're outside the shock
        rho_sol    = rhoR;
        vdim_sol   = vRdim;
        vother_sol = vRother;
        p_sol      = pR;
      } else {
        // we're in the star region
        rho_sol    = (pstaroverpR + cst::GM1OGP1) / (cst::GM1OGP1 * pstaroverpR + 1.) * rhoR;
        vdim_sol   = _vstar;
        vother_sol = vRother;
        p_sol      = _pstar;
      }
    }
  }


// #if DEBUG_LEVEL > 0
//   assert(not std::isnan(rho_sol));
//   assert(not std::isnan(vdim_sol));
//   assert(not std::isnan(vother_sol));
//   assert(not std::isnan(p_sol));
// #endif

  // std::array<Float, Dimensions> v_sol;
  // Float v_sol[Dimensions];
  // v_sol[_dim]     = vdim_sol;
  // v_sol[otherdim] = vother_sol;

  PrimitiveState sol(rho_sol, vdim_sol, vother_sol, p_sol);

  ConservedFlux Fsol(sol, _dim);
  return Fsol;
}
