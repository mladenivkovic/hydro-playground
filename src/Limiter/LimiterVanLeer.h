#pragma once

/**
 * @file LimiterVanLeer.h
 * @brief The van Leer Limiter.
 */

#include "../Constants.h"

namespace limiter {

  /**
   * Compute the actual slope limiter xi(r) for the van Leer limiter
   */
  inline Float limiterXiOfR(const Float r) {

    if (r <= 0.)
      return 0.;

    Float xi  = (2. * r) / (1. + r);
    Float d   = 1. - cst::MUSCL_SLOPE_OMEGA + (1. + cst::MUSCL_SLOPE_OMEGA) * r;
    Float xiR = 2. / d;

    if (xiR < xi)
      xi = xiR;

    return xi;
  }
} // namespace limiter
