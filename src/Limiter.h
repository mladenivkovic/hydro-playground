#pragma once

/**
 * @file Limiter.h
 * @brief Main header to include slope limiters.
 * See Section 5.2 in theory document.
 */

#include "Config.h"
#include "Gas.h"


#if LIMITER == LIMITER_NONE
#include "Limiter/LimiterNone.h"
#elif LIMITER == LIMITER_MINMOD
#include "Limiter/LimiterMinmod.h"
#elif LIMITER == LIMITER_VANLEER
#include "Limiter/LimiterVanLeer.h"
#else
#error Invalid limiter selected
#endif


namespace limiter {

  /**
   * Eq. 107 in theory document.
   *
   * in case of advection:
   * if v > 0:
   *    compute r = (u_{i} - u_{i-1}) / (u_{i+1} - u_{i})
   * else:
   *    compute r = (u_{i+1} - u_{i+2}) / (u_{i} - u_{i+1})
   *
   * In the tex documents, r for v < 0 is given as
   *    r = (u_{i+2} - u_{i+1}) / (u_{i+1} - u_{i})
   * which can be transformed into the above expression by multiplying
   * the numerator and the denominator by -1.
   * So then we can write
   *
   *          top_left - top_right
   * r = ---------------------------------
   *          bottom_left - top_left
   *
   * regardless of what sign the velocity v has. We only need to
   * switch what topleft, topright, and bottomleft are, which is
   * done in the function that is calling this one.
   */
  inline Float limiterR(const Float topleft, const Float topright, const Float bottomleft) {

    // avoid div by zero
    if (bottomleft == topleft)
      return ((topleft - topright) * 1.e6);
    return ((topleft - topright) / (bottomleft - topleft));
  }


  /**
   * Compute the flow parameter r for every component of the conserved states.
   * We always compute r = (u_{i} - u_{i-1}) / (u_{i+1} - u_{i}), and for hydro
   * purposes, we don't really need to care about upwinding.
   *
   * @param UiP1:  U_{i+1}
   * @param Ui:    U_{i}
   * @param UiM1:  U_{i-1}
   * @param r:     where flow parameter r for every conserved state will be stored
   */
  inline void limiterGetRCstate(
    const idealGas::ConservedState& UiP1,
    const idealGas::ConservedState& Ui,
    const idealGas::ConservedState& UiM1,
    idealGas::ConservedState&       r
  ) {

    Float rho   = limiterR(Ui.getRho(), UiM1.getRho(), UiP1.getRho());
    Float rhovx = limiterR(Ui.getRhov(0), UiM1.getRhov(0), UiP1.getRhov(0));
    Float rhovy = limiterR(Ui.getRhov(1), UiM1.getRhov(1), UiP1.getRhov(1));
    Float E     = limiterR(Ui.getE(), UiM1.getE(), UiP1.getE());

    r = idealGas::ConservedState(rho, rhovx, rhovy, E);
  }


  /**
   * Compute the slope of given cell c using slope limiters, as it is needed
   * for the MUSCL-Hancock scheme.
   * Remember:
   *  slope = 0.5(1 + omega)(U_{i} - U_{i-1}) + 0.5(1 - omega)(U_{i+1} - U_{i})
   * where omega is set in defines.h
   *
   * @param UiP1: State of cell U_{i+1}
   * @param Ui: State of cell U_{i}
   * @param UiM1: State of cell U_{i-1}
   */
  inline void limiterGetLimitedSlope(
    const idealGas::ConservedState& UiP1,
    const idealGas::ConservedState& Ui,
    const idealGas::ConservedState& UiM1,
    idealGas::ConservedState&       slope
  ) {

    idealGas::ConservedState r;
    limiterGetRCstate(UiP1, Ui, UiM1, r);

    Float xi_rho   = limiterXiOfR(r.getRho());
    Float xi_rhovx = limiterXiOfR(r.getRhov(0));
    Float xi_rhovy = limiterXiOfR(r.getRhov(1));
    Float xi_E     = limiterXiOfR(r.getE());

    constexpr Float OMEGA = cst::MUSCL_SLOPE_OMEGA;

    Float r1        = (1. + OMEGA) * (Ui.getRho() - UiM1.getRho());
    Float r2        = (1. - OMEGA) * (UiP1.getRho() - Ui.getRho());
    Float slope_rho = xi_rho * 0.5 * (r1 + r2);

    Float vx1         = (1. + OMEGA) * (Ui.getRhov(0) - UiM1.getRhov(0));
    Float vx2         = (1. - OMEGA) * (UiP1.getRhov(0) - Ui.getRhov(0));
    Float slope_rhovx = xi_rhovx * 0.5 * (vx1 + vx2);

    Float vy1         = (1. + OMEGA) * (Ui.getRhov(1) - UiM1.getRhov(1));
    Float vy2         = (1. - OMEGA) * (UiP1.getRhov(1) - Ui.getRhov(1));
    Float slope_rhovy = xi_rhovy * 0.5 * (vy1 + vy2);

    Float E1      = (1. + OMEGA) * (Ui.getE() - UiM1.getE());
    Float E2      = (1. - OMEGA) * (UiP1.getE() - Ui.getE());
    Float slope_E = xi_E * 0.5 * (E1 + E2);

    slope = idealGas::ConservedState(slope_rho, slope_rhovx, slope_rhovy, slope_E);
  }


} // namespace limiter
