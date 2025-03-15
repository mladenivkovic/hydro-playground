/**
 * @file SolverGodunov.cpp
 * @brief The Godunov hydro solver.
 */


#include "SolverGodunov.h"

//! Run a single step.
void solver::SolverGodunov::step() {

  std::cout << "Called Godunov step; ";
  std::cout << "stepCount=" << stepCount;
  std::cout << ", t=" << t << "\n";

  stepCount += 1;
}


