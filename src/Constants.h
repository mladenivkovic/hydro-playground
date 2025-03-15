#pragma once

/**
 * @file Constants.h
 * @brief Contains (physical) constants used across the project.
 */

#include "Config.h"

namespace cst {

static constexpr Float GAMMA = (5.0 / 3.0);

static constexpr Float GM1          = GAMMA - 1.;
static constexpr Float GP1          = GAMMA + 1.;
static constexpr Float GP1OGM1      = (GAMMA + 1.) / (GAMMA - 1.);
static constexpr Float GM1OGP1      = (GAMMA - 1.) / (GAMMA + 1.);
static constexpr Float ONEOVERGAMMA = 1. / GAMMA;
static constexpr Float GM1HALF      = 0.5 * (GAMMA - 1.);
static constexpr Float BETA         = 0.5 * (GAMMA - 1.) / GAMMA;

// "cheat" for stability in Godunov type finite volume schemes
static constexpr Float SMALLRHO = 1e-6;
static constexpr Float SMALLU   = 1e-6;
static constexpr Float SMALLP   = 1e-6;

static constexpr Float DT_MIN = 1e-10;
static constexpr Float EPSILON_ITER = 1e-6;

} // namespace cst

