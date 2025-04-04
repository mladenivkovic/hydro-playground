#pragma once

/**
 * @file Constants.h
 * @brief Contains (physical) constants used across the project.
 */

#include <cstddef>

#include "Config.h"

namespace cst {

#pragma omp declare target

  static constexpr Float GAMMA = (5.0 / 3.0);

  static constexpr Float GM1          = (GAMMA - 1.);
  static constexpr Float GP1          = (GAMMA + 1.);
  static constexpr Float GP1OGM1      = ((GAMMA + 1.) / (GAMMA - 1.));
  static constexpr Float GM1OGP1      = ((GAMMA - 1.) / (GAMMA + 1.));
  static constexpr Float ONEOVERGAMMA = (1. / GAMMA);
  static constexpr Float ONEOVERGM1   = (1. / (GAMMA - 1.));
  static constexpr Float TWOOVERGM1   = (2. / (GAMMA - 1.));
  static constexpr Float TWOOVERGP1   = (2. / (GAMMA + 1.));
  static constexpr Float GM1HALF      = (0.5 * (GAMMA - 1.));
  static constexpr Float BETA         = (0.5 * (GAMMA - 1.) / GAMMA);
  static constexpr Float ONEOVERBETA  = (2. * GAMMA / (GAMMA - 1.));

  // "cheat" for stability in Godunov type finite volume schemes
  static constexpr Float SMALLRHO = 1e-8;
  static constexpr Float SMALLV   = 1e-8;
  static constexpr Float SMALLP   = 1e-8;

  static constexpr Float       DT_MIN                = 1e-10;
  static constexpr Float       EPSILON_ITER          = 1e-6;
  static constexpr std::size_t EXACT_SOLVER_MAX_ITER = 100;


  // define slope of each cell in MUSCL scheme as
  //  slope = 0.5* (1 + omega) (U_{i}-U_{i-1}) +  0.5 * (1 - omega) (U_{i+1} - U_{i})
  constexpr Float MUSCL_SLOPE_OMEGA = 0.;

#pragma omp end declare target
} // namespace cst
