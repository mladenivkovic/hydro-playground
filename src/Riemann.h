/**
 * @file Riemann.h
 * @brief Main header to include Riemann solvers.
 */

#include "Config.h"
#include "RiemannExact.h"
#include "RiemannHLLC.h"


namespace riemann {

#if RIEMANN_SOLVER == RIEMANN_SOLVER_HLLC
  using Riemann = RiemannHLLC;
#elif RIEMANN_SOLVER == RIEMANN_SOLVER_EXACT
  using Riemann = RiemannExact;
#else
#error Invalid Riemann solver selected
#endif

} // namespace riemann
