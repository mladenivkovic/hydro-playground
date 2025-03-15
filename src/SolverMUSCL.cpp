#include "SolverMUSCL.h"


void solver::SolverMUSCL::step() {

  std::cout << "Called MUSCL step; ";
  std::cout << "stepCount=" << stepCount;
  std::cout << ", t=" << t << "\n";

  stepCount += 1;

}

