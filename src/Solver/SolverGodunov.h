/**
 * @file SolverGodunov.h
 * @brief The Godunov hydro solver.
 */

#pragma once

#include "SolverBase.h"


namespace solver {

  class SolverGodunov: public SolverBase {

    //! Compute the intercell fluxes needed for the update
    void computeFluxes();

    //! Compute the intercell fluxes between two cells along the given
    //! dimension.
    void computeIntercellFluxes(cell::Cell& left, cell::Cell& right);

  public:
    SolverGodunov(parameters::Parameters& params_, grid::Grid& grid_);
    ~SolverGodunov() = default;

    //! Run a single step.
    void step() override;
  };

} // namespace solver
