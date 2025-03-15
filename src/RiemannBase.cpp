#include "RiemannBase.h"

#include <cmath>

#include "Constants.h"
#include "Gas.h"


/**
 * Do we have a vacuum or vacuum generating conditions?
 *
 * TODO: Refer to equations in theory document
 */
using idealGas::PrimitiveState;

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
 */
idealGas::PrimitiveState riemann::RiemannBase::solveVacuum(){

  size_t otherdim = (_dim + 1) % 2;
  // x / t. We always center the problem at x=0, but to sample the solution in general,
  // we need to adapt this value.
  constexpr Float xovert = 0.;

  // Both vacuum states
  if (_left.getRho() <= cst::SMALLRHO and _right.getRho() <= cst::SMALLRHO) {
    return idealGas::PrimitiveState(cst::SMALLRHO, cst::SMALLV, cst::SMALLV, cst::SMALLP);
  }

  Float rho_sol = NAN;
  Float vdim_sol = NAN;
  Float vother_sol = NAN;
  Float p_sol = NAN;

  if (_left.getRho() <= cst::SMALLRHO) {
    // ------------------------
    // Left vacuum state
    // ------------------------
    Float aR  = _right.getSoundSpeed();
    Float SR  = _right.getV(_dim) - aR * cst::TWOOVERGAMMAM1; // vacuum front speed
    Float SHR = _right.getV(_dim) + aR; // speed of head of right rarefaction fan

    if (xovert <= SR) {
      // left vacuum
      rho_sol = cst::SMALLRHO;
      vdim_sol = cst::SMALLV;
      vother_sol = cst::SMALLV;
      p_sol = cst::SMALLP;
    } else if (xovert < SHR) {
      // inside rarefaction
      Float precomp = std::pow(
        (cst::TWOOVERGAMMAP1 - cst::GM1OGP1 / aR * (_right.getV(_dim) - xovert)), cst::TWOOVERGAMMAM1
      );
      rho_sol    = _right.getRho() * precomp;
      vdim_sol   = cst::TWOOVERGAMMAP1 / cst::GP1 * (cst::GM1HALF * _right.getV(_dim) - aR + xovert);
      vother_sol = _right.getV(otherdim);
      p_sol      = _right.getP() * std::pow(precomp, cst::GAMMA);
    } else {
      // original right pstate
      rho_sol   = _right.getRho();
      vdim_sol  = _right.getV(_dim);
      vother_sol = _right.getV(otherdim);
      p_sol     = _right.getP();
    }
  }

  else if (_right.getRho() <= cst::SMALLRHO) {
    // ------------------------
    // Right vacuum state
    // ------------------------

    Float aL  = _left.getSoundSpeed();
    Float SL  = _left.getV(_dim) + aL * cst::TWOOVERGAMMAM1; // vacuum front speed
    Float SHL = _left.getV(_dim) - aL; // speed of head of left rarefaction fan

    if (xovert >= SL) {
      // right vacuum
      rho_sol = cst::SMALLRHO;
      vdim_sol = cst::SMALLV;
      vother_sol = cst::SMALLV;
      p_sol   = cst::SMALLP;
    } else if (xovert > SHL) {
      // inside rarefaction
      Float precomp = std::pow( (cst::TWOOVERGAMMAP1 + cst::GM1OGP1 / aL * (_left.getV(_dim) - xovert)), (cst::TWOOVERGAMMAP1));
      rho_sol       = _left.getRho() * precomp;
      vdim_sol    = cst::TWOOVERGAMMAP1 * (cst::GM1HALF * _left.getV(_dim) + aL + xovert);
      vother_sol = _left.getV(otherdim);
      p_sol      = _left.getP() * std::pow(precomp,cst::GAMMA);
    } else {
      // original left pstate
      rho_sol       = _left.getRho();
      vdim_sol      = _left.getV(_dim);
      vother_sol    = _left.getV(otherdim);
      p_sol         = _left.getP();
    }
  } else {
    // ------------------------
    // Vacuum generating case
    // ------------------------

    Float aL  = _left.getSoundSpeed();
    Float aR  = _right.getSoundSpeed();
    Float SL  = _left.getV(_dim) + aL * cst::TWOOVERGAMMAM1;  // vacuum front speed
    Float SR  = _right.getV(_dim) - aR * cst::TWOOVERGAMMAP1; // vacuum front speed
    Float SHL = _left.getV(_dim) - aL;  // speed of head of left rarefaction fan
    Float SHR = _right.getV(_dim) + aR; // speed of head of right rarefaction fan

    if (xovert <= SHL) {
      // left original pstate
      rho_sol    = _left.getRho();
      vdim_sol   = _left.getV(_dim);
      vother_sol = _left.getV(otherdim);
      p_sol      = _left.getP();
    } else if (xovert < SL) {
      // inside rarefaction fan from right to left
      Float precomp = std::pow(
        (cst::TWOOVERGAMMAP1 + cst::GM1OGP1 / aL * (_left.getV(_dim) - xovert)), (cst::TWOOVERGAMMAM1)
      );
      rho_sol    = _left.getRho() * precomp;
      vdim_sol   = cst::TWOOVERGAMMAP1 * (cst::GM1HALF * _left.getV(_dim) + aL + xovert);
      vother_sol = _left.getV(otherdim);
      p_sol      = _left.getP() * std::pow(precomp,cst::GAMMA);
    } else if (xovert < SR) {
      // vacuum region
      rho_sol = cst::SMALLRHO;
      vdim_sol = cst::SMALLV;
      vother_sol = cst::SMALLV;
      p_sol  = cst::SMALLP;
    } else if (xovert < SHR) {
      // inside rarefaction fan from left to right
      Float precomp = std::pow(
        (cst::TWOOVERGAMMAP1 - cst::GM1OGP1 / aR * (_right.getV(_dim) - xovert)), (cst::TWOOVERGAMMAM1)
      );
      rho_sol    = _right.getRho() * precomp;
      vdim_sol   = cst::TWOOVERGAMMAP1 * (cst::GM1HALF * _right.getV(_dim) - aR + xovert);
      vother_sol = _right.getV(otherdim);
      p_sol      = _right.getP() * std::pow(precomp,cst::GAMMA);
    } else {
      // right original pstate
      rho_sol       = _right.getRho();
      vdim_sol    = _right.getV(_dim);
      vother_sol = _right.getV(otherdim);
      p_sol         = _right.getP();
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
