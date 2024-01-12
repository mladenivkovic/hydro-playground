#pragma once

#include "Config.h"

static constexpr Precision GAMMA = (5.0 / 3.0);


/* JUST PUTTING THESE IN SO IT COMPILES */
#ifdef USE_AS_RIEMANN_SOLVER
/* set the "small" values to actually zero, so that only correct vacuum sates
 * are recognized as such */
static constexpr Precision SMALLRHO = 0;
static constexpr Precision SMALLU   = 0;
static constexpr Precision SMALLP   = 0;
#else
/* cheat for stability in Godunov type finite volume schemes*/
static constexpr Precision SMALLRHO = 1e-6;
static constexpr Precision SMALLU   = 1e-6;
static constexpr Precision SMALLP   = 1e-6;
#endif

static constexpr Precision GM1          = GAMMA - 1.;
static constexpr Precision GP1          = GAMMA + 1.;
static constexpr Precision GP1OGM1      = (GAMMA + 1.) / (GAMMA - 1.);
static constexpr Precision GM1OGP1      = (GAMMA - 1.) / (GAMMA + 1.);
static constexpr Precision ONEOVERGAMMA = 1. / GAMMA;
static constexpr Precision GM1HALF      = 0.5 * (GAMMA - 1.);
static constexpr Precision BETA         = 0.5 * (GAMMA - 1.) / GAMMA;
