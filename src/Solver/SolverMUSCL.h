/**
 * @file SolverMUSCL.h
 * @brief The MUSCL-Hancock hydro solver.
 */
#pragma once

#include "SolverBase.h"


namespace solver {

  class SolverMUSCL: public SolverBase {

    //! Compute the intercell fluxes needed for the update
    void computeFluxes(const Float dt_step);

    //! Compute the boundary extrapolated values.
    void getBoundaryExtrapolatedValues(
      cell::Cell&                     c,
      const idealGas::ConservedState& UiP1,
      const idealGas::ConservedState& UiM1,
      const Float                     dt_half
    );

    //! Compute the intercell fluxes between two cells along the given
    //! dimension.
    void computeIntercellFluxes(cell::Cell& left, cell::Cell& right);

  public:
    SolverMUSCL(parameters::Parameters& params_, grid::Grid& grid_);
    ~SolverMUSCL() = default;

    //! Run a single step.
    void step() override;
  };

} // namespace solver
