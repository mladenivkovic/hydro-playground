/**
 * @file Solver.h
 * @brief Main header to include solvers.
 */

#include "Config.h"

#include "SolverMUSCL.h"
#include "SolverGodunov.h"



namespace solver {

#if SOLVER == SOLVER_MUSCL
  using Solver = SolverMUSCL;
#elif SOLVER == SOLVER_GODUNOV
  using Solver = SolverGodunov;
#else
#error Invalid Solver selected
#endif

} // namespace solver

