/**
 * @file SolverMUSCL.h
 * @brief The MUSCL-Hancock hydro solver.
 */
#pragma once

#include "SolverBase.h"



namespace solver {


  class SolverMUSCL : public SolverBase {

  protected:

  public:

    SolverMUSCL() = default;
    ~SolverMUSCL() = default;

    //! Run a single step.
    void step() override;

  };

} // namespace solver

