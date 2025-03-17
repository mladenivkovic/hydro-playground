#pragma once

/**
 * @file LimiterMinmod.h
 * @brief The Minmod Limiter.
 */

#include "Constants.h"

namespace limiter {

  /**
   * Compute the actual slope limiter xi(r) for the minmod limiter
   */
  inline Float limiterXiOfR(const Float r){

    Float xi = r > 0. ? r : 0.;
    if (r > 1.) {
      Float d   = 1. - cst::MUSCL_SLOPE_OMEGA + (1. + cst::MUSCL_SLOPE_OMEGA) * r;
      Float xiR = 2. / d;

      // xi = min(1, xiR)
      xi = (xiR < xi) ? xiR : 1.;
    }
    return xi;
  }

}  // namespace limiter


