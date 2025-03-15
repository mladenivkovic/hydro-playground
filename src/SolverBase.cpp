#include "SolverBase.h"
#include "Logging.h"



/**
 * Constructor
 */
solver::SolverBase::SolverBase():
  t(0.), dt(0.), stepCount(0){};



/**
 * @brief Main solver routine.
 * Should be the same for all solvers. What differs is the contents of
 * Solver::step();
 */
void solver::SolverBase::solve(parameters::Parameters& params, grid::Grid& grid){

  logging::setStage(logging::LogStage::Step);

  std::cout << "Called solve \n";

  for (int i = 0; i < 3; i++){
    step();
    t = t + 1.;
  }

}

