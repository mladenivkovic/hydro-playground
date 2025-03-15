#include "RiemannBase.h"

#include <cmath>

#include "Constants.h"
#include "Gas.h"


/**
 * Do we have a vacuum or vacuum generating conditions?
 *
 * TODO: Refer to equations in theory document
 */
bool riemann::RiemannBase::hasVacuum() {

  if (_left.getRho() <= cst::SMALLRHO)
    return true;
  if (_right.getRho() <= cst::SMALLRHO)
    return true;

  Float delta_v = _right.getV(_dim) - _left.getV(_dim);
  Float v_crit = 2. * cst::ONEOVERGAMMAM1 * (_left.getSoundSpeed() + _right.getSoundSpeed());

  return delta_v >= v_crit;
}


/**
 * Compute the vacuum solution.
 *
 * TODO: Refer to equations in theory document
 * @return the primitive state corresponding to the solution sampled at x=0.
 */
idealGas::PrimitiveState riemann::RiemannBase::solveVacuum(){

  size_t otherdim = (_dim + 1) % 2;
  // x / t. We always center the problem at x=0, but to sample the solution in general,
  // we need to adapt this value.
  constexpr Float xovert = 0.;

  Float aL = _left.getSoundSpeed();
  Float aR = _right.getSoundSpeed();

  Float rhoL = _left.getRho();
  Float rhoR = _right.getRho();

  Float vLdim = _left.getV(_dim);
  Float vLother = _left.getV(otherdim);
  Float vRdim = _right.getV(_dim);
  Float vRother = _right.getV(otherdim);

  Float pL = _left.getP();
  Float pR = _right.getP();


  // Both vacuum states
  if (rhoL <= cst::SMALLRHO and rhoR <= cst::SMALLRHO) {
    return idealGas::PrimitiveState(cst::SMALLRHO, cst::SMALLV, cst::SMALLV, cst::SMALLP);
  }

  Float rho_sol = NAN;
  Float vdim_sol = NAN;
  Float vother_sol = NAN;
  Float p_sol = NAN;

  if (rhoL <= cst::SMALLRHO) {
    // ------------------------
    // Left vacuum state
    // ------------------------
    Float SR  = vRdim - aR * cst::TWOOVERGAMMAM1; // vacuum front speed
    Float SHR = vRdim + aR; // speed of head of right rarefaction fan

    if (xovert <= SR) {
      // left vacuum
      rho_sol = cst::SMALLRHO;
      vdim_sol = cst::SMALLV;
      vother_sol = cst::SMALLV;
      p_sol = cst::SMALLP;
    } else if (xovert < SHR) {
      // inside rarefaction
      Float precomp = std::pow(
        (cst::TWOOVERGAMMAP1 - cst::GM1OGP1 / aR * (vRdim - xovert)), cst::TWOOVERGAMMAM1
      );
      rho_sol    = rhoR * precomp;
      vdim_sol   = cst::TWOOVERGAMMAP1 / cst::GP1 * (cst::GM1HALF * vRdim - aR + xovert);
      vother_sol = vRother;
      p_sol      = pR * std::pow(precomp, cst::GAMMA);
    } else {
      // original right pstate
      rho_sol   = rhoR;
      vdim_sol  = vRdim;
      vother_sol = vRother;
      p_sol     = pR;
    }
  }

  else if (rhoR <= cst::SMALLRHO) {
    // ------------------------
    // Right vacuum state
    // ------------------------

    Float SL  = vLdim + aL * cst::TWOOVERGAMMAM1; // vacuum front speed
    Float SHL = vLdim - aL; // speed of head of left rarefaction fan

    if (xovert >= SL) {
      // right vacuum
      rho_sol = cst::SMALLRHO;
      vdim_sol = cst::SMALLV;
      vother_sol = cst::SMALLV;
      p_sol   = cst::SMALLP;
    } else if (xovert > SHL) {
      // inside rarefaction
      Float precomp = std::pow( (cst::TWOOVERGAMMAP1 + cst::GM1OGP1 / aL * (vLdim - xovert)), (cst::TWOOVERGAMMAP1));
      rho_sol       = rhoL * precomp;
      vdim_sol    = cst::TWOOVERGAMMAP1 * (cst::GM1HALF * vLdim + aL + xovert);
      vother_sol = vLother;
      p_sol      = pL * std::pow(precomp,cst::GAMMA);
    } else {
      // original left pstate
      rho_sol       = rhoL;
      vdim_sol      = vLdim;
      vother_sol    = vLother;
      p_sol         = pL;
    }
  } else {
    // ------------------------
    // Vacuum generating case
    // ------------------------

    Float SL  = vLdim + aL * cst::TWOOVERGAMMAM1;  // vacuum front speed
    Float SR  = vRdim - aR * cst::TWOOVERGAMMAP1; // vacuum front speed
    Float SHL = vLdim - aL;  // speed of head of left rarefaction fan
    Float SHR = vRdim + aR; // speed of head of right rarefaction fan

    if (xovert <= SHL) {
      // left original pstate
      rho_sol    = rhoL;
      vdim_sol   = vLdim;
      vother_sol = vLother;
      p_sol      = pL;
    }
    else if (xovert < SL) {
      // inside rarefaction fan from right to left
      Float precomp = std::pow(
        (cst::TWOOVERGAMMAP1 + cst::GM1OGP1 / aL * (vLdim - xovert)), (cst::TWOOVERGAMMAM1)
      );
      rho_sol    = rhoL * precomp;
      vdim_sol   = cst::TWOOVERGAMMAP1 * (cst::GM1HALF * vLdim + aL + xovert);
      vother_sol = vLother;
      p_sol      = pL * std::pow(precomp,cst::GAMMA);
    }
    else if (xovert < SR) {
      // vacuum region
      rho_sol = cst::SMALLRHO;
      vdim_sol = cst::SMALLV;
      vother_sol = cst::SMALLV;
      p_sol  = cst::SMALLP;
    }
    else if (xovert < SHR) {
      // inside rarefaction fan from left to right
      Float precomp = std::pow( (cst::TWOOVERGAMMAP1 - cst::GM1OGP1 / aR * (vRdim - xovert)), (cst::TWOOVERGAMMAM1));
      rho_sol    = rhoR * precomp;
      vdim_sol   = cst::TWOOVERGAMMAP1 * (cst::GM1HALF * vRdim - aR + xovert);
      vother_sol = vRother;
      p_sol      = pR * std::pow(precomp,cst::GAMMA);
    }
    else {
      // right original pstate
      rho_sol       = rhoR;
      vdim_sol    = vRdim;
      vother_sol = vRother;
      p_sol         = pR;
    }
  }


#if DEBUG_LEVEL > 0
  assert(not std::isnan(rho_sol));
  assert(not std::isnan(vdim_sol));
  assert(not std::isnan(vother_sol));
  assert(not std::isnan(p_sol));
#endif


  idealGas::PrimitiveState sol(rho_sol, 0., 0., p_sol);
  sol.setV(_dim, vdim_sol);
  sol.setV(otherdim, vother_sol);
  return sol;
}
