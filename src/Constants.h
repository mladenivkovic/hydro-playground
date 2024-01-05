#pragma once

#define GAMMA (5. / 3.)



/* JUST PUTTING THESE IN SO IT COMPILES */
#ifdef USE_AS_RIEMANN_SOLVER
/* set the "small" values to actually zero, so that only correct vacuum sates
 * are recognized as such */
#define SMALLRHO 0.
#define SMALLU 0.
#define SMALLP 0.
#else
/* cheat for stability in Godunov type finite volume schemes*/
#define SMALLRHO 1e-6
#define SMALLU 1e-6
#define SMALLP 1e-6
#endif

static const float GM1 = GAMMA - 1.;
static const float GP1 = GAMMA + 1.;
static const float GP1OGM1 = (GAMMA + 1.) / (GAMMA - 1.);
static const float GM1OGP1 = (GAMMA - 1.) / (GAMMA + 1.);
static const float ONEOVERGAMMA = 1. / GAMMA;
static const float GM1HALF = 0.5 * (GAMMA - 1.);
static const float BETA = 0.5 * (GAMMA - 1.) / GAMMA;
