#include "RiemannBase.h"

#include <cmath>

#include "Constants.h"
#include "Gas.h"


/**
 * Do we have a vacuum or vacuum generating conditions?
 *
 * Section 3.5 in theory document, and eq. 86
 */
bool riemann::RiemannBase::hasVacuum() {

  if (_left.getRho() <= cst::SMALLRHO)
    return true;
  if (_right.getRho() <= cst::SMALLRHO)
    return true;

  Float delta_v = _right.getV(_dim) - _left.getV(_dim);
  Float v_crit  = cst::TWOOVERGM1 * (_left.getSoundSpeed() + _right.getSoundSpeed());

  return delta_v >= v_crit;
}


/**
 * Compute the vacuum solution. See Section 3.5 in theory document.
 *
 * @return the state in primitive variables corresponding to the solution
 * sampled at x=0.
 */
idealGas::PrimitiveState riemann::RiemannBase::solveVacuum() {

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
    return idealGas::PrimitiveState(cst::SMALLRHO, cst::SMALLV, cst::SMALLV, cst::SMALLP);
  }

  Float rho_sol    = NAN;
  Float vdim_sol   = NAN;
  Float vother_sol = NAN;
  Float p_sol      = NAN;

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
      Float precomp = std::pow(
        (cst::TWOOVERGP1 - cst::GM1OGP1 / aR * (vRdim - xovert)), cst::TWOOVERGM1
      );
      rho_sol    = rhoR * precomp;
      vdim_sol   = cst::TWOOVERGP1 * (cst::GM1HALF * vRdim - aR + xovert);
      vother_sol = vRother;
      p_sol      = pR * std::pow(precomp, cst::GAMMA);
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
      Float precomp = std::pow(
        (cst::TWOOVERGP1 + cst::GM1OGP1 / aL * (vLdim - xovert)), (cst::TWOOVERGP1)
      );
      rho_sol    = rhoL * precomp;
      vdim_sol   = cst::TWOOVERGP1 * (cst::GM1HALF * vLdim + aL + xovert);
      vother_sol = vLother;
      p_sol      = pL * std::pow(precomp, cst::GAMMA);
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
      Float precomp = std::pow(
        (cst::TWOOVERGP1 + cst::GM1OGP1 / aL * (vLdim - xovert)), cst::TWOOVERGM1
      );
      rho_sol    = rhoL * precomp;
      vdim_sol   = cst::TWOOVERGP1 * (cst::GM1HALF * vLdim + aL + xovert);
      vother_sol = vLother;
      p_sol      = pL * std::pow(precomp, cst::GAMMA);
    } else if (xovert < SR) {
      // vacuum region
      rho_sol    = cst::SMALLRHO;
      vdim_sol   = cst::SMALLV;
      vother_sol = cst::SMALLV;
      p_sol      = cst::SMALLP;
    } else if (xovert < SHR) {
      // inside rarefaction fan from left to right
      Float precomp = std::pow(
        (cst::TWOOVERGP1 - cst::GM1OGP1 / aR * (vRdim - xovert)), cst::TWOOVERGM1
      );
      rho_sol    = rhoR * precomp;
      vdim_sol   = cst::TWOOVERGP1 * (cst::GM1HALF * vRdim - aR + xovert);
      vother_sol = vRother;
      p_sol      = pR * std::pow(precomp, cst::GAMMA);
    } else {
      // right original pstate
      rho_sol    = rhoR;
      vdim_sol   = vRdim;
      vother_sol = vRother;
      p_sol      = pR;
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


/**
 * Compute the solution of the riemann problem at given time t and x,
 * specified as xovert = x/t. Here, we always set x/t = x = 0 at the
 * cell interface.
 * Section 3.6 in theory document.
 */
idealGas::ConservedFlux riemann::RiemannBase::sampleSolution() {

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


  Float rho_sol    = NAN;
  Float vdim_sol   = NAN;
  Float vother_sol = NAN;
  Float p_sol      = NAN;


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
        Float astarL = aL * std::pow(pstaroverpL, cst::BETA);
        Float STL    = _vstar - astarL; // speed of tail of left rarefaction fan
        if (xovert < STL) {
          // we're inside the fan
          Float precomp = std::pow(
            (cst::TWOOVERGP1 + cst::GM1OGP1 / aL * (vLdim - xovert)), cst::TWOOVERGM1
          );
          rho_sol    = rhoL * precomp;
          vdim_sol   = cst::TWOOVERGP1 * (cst::GM1HALF * vLdim + aL + xovert);
          vother_sol = vLother;
          p_sol      = pL * pow(precomp, cst::GAMMA);
        } else {
          // we're in the star region
          rho_sol    = rhoL * std::pow(pstaroverpL, cst::ONEOVERGAMMA);
          vdim_sol   = _vstar;
          vother_sol = vLother;
          p_sol      = _pstar;
        }
      }
    } else {

      // left shock

      // left shock speed
      Float SL = vLdim
                 - aL * std::sqrt(0.5 * cst::GP1 * cst::ONEOVERGAMMA * pstaroverpL + cst::BETA);
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
        Float astarR = aR * std::pow(pstaroverpR, cst::BETA);
        Float STR    = _vstar + astarR; // speed of tail of right rarefaction fan
        if (xovert > STR) {
          // we're inside the fan
          Float precomp = std::pow(
            (cst::TWOOVERGP1 - cst::GM1OGP1 / aR * (vRdim - xovert)), cst::TWOOVERGM1
          );
          rho_sol    = rhoR * precomp;
          vdim_sol   = cst::TWOOVERGP1 * (cst::GM1HALF * vRdim - aR + xovert);
          vother_sol = vRother;
          p_sol      = pR * std::pow(precomp, cst::GAMMA);
        } else {
          // we're in the star region
          rho_sol    = rhoR * pow(pstaroverpR, cst::ONEOVERGAMMA);
          vdim_sol   = _vstar;
          vother_sol = vRother;
          p_sol      = _pstar;
        }
      }
    } else {

      // right shock

      // right shock speed
      Float SR = vRdim
                 + aR * std::sqrt(0.5 * cst::GP1 * cst::ONEOVERGAMMA * pstaroverpR + cst::BETA);

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


#if DEBUG_LEVEL > 0
  assert(not std::isnan(rho_sol));
  assert(not std::isnan(vdim_sol));
  assert(not std::isnan(vother_sol));
  assert(not std::isnan(p_sol));
#endif

  std::array<Float, Dimensions> v_sol;
  v_sol[_dim]     = vdim_sol;
  v_sol[otherdim] = vother_sol;

  idealGas::PrimitiveState sol(rho_sol, v_sol, p_sol);

  idealGas::ConservedFlux Fsol(sol, _dim);
  return Fsol;
}
