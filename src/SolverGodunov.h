/**
 * @file SolverGodunov.h
 * @brief The Godunov hydro solver.
 */

#pragma once

#include "SolverBase.h"



namespace solver {

  class SolverGodunov : public SolverBase {

  public:

    SolverGodunov() = default;
    ~SolverGodunov() = default;

    //! Run a single step.
    void step() override;
  };

} // namespace solver

