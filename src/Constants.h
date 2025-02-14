#pragma once

/**
 * @file Constants.h
 * @brief Contains (physical) constants used across the project.
 */

#include "Config.h"

static constexpr float_t GAMMA = (5.0 / 3.0);

static constexpr float_t GM1          = GAMMA - 1.;
static constexpr float_t GP1          = GAMMA + 1.;
static constexpr float_t GP1OGM1      = (GAMMA + 1.) / (GAMMA - 1.);
static constexpr float_t GM1OGP1      = (GAMMA - 1.) / (GAMMA + 1.);
static constexpr float_t ONEOVERGAMMA = 1. / GAMMA;
static constexpr float_t GM1HALF      = 0.5 * (GAMMA - 1.);
static constexpr float_t BETA         = 0.5 * (GAMMA - 1.) / GAMMA;

// "cheat" for stability in Godunov type finite volume schemes
static constexpr float_t SMALLRHO = 1e-6;
static constexpr float_t SMALLU   = 1e-6;
static constexpr float_t SMALLP   = 1e-6;
