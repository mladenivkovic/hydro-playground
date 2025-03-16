#include "SolverMUSCL.h"


solver::SolverMUSCL::SolverMUSCL(parameters::Parameters& params_, grid::Grid& grid_):
  SolverBase(params_, grid_) {
}

void solver::SolverMUSCL::step() {

  std::cout << "Called MUSCL step; ";
  std::cout << "stepCount=" << step_count;
  std::cout << ", t=" << t << "\n";

  step_count += 1;
}
