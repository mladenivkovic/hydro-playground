/**
 * @file SolverGodunov.cpp
 * @brief The Godunov hydro solver.
 */


#include "SolverGodunov.h"

solver::SolverGodunov::SolverGodunov(parameters::Parameters& params_, grid::Grid& grid_) :
  SolverBase(params_, grid_){}


//! Run a single step.
void solver::SolverGodunov::step() {

  std::cout << "Called Godunov step; ";
  std::cout << "stepCount=" << stepCount;
  std::cout << ", t=" << t << "\n";

  stepCount += 1;
}


