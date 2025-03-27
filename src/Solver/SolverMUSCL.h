/**
 * @file SolverMUSCL.h
 * @brief The MUSCL-Hancock hydro solver.
 */
#pragma once

#include "SolverBase.h"


class SolverMUSCL: public SolverBase {

  //! Compute the intercell fluxes needed for the update
  void computeFluxes(const Float dt_step);

  //! Compute the boundary extrapolated values.
  void getBoundaryExtrapolatedValues(
    Cell& c, const ConservedState& UiP1, const ConservedState& UiM1, const Float dt_half
  );

  //! Compute the intercell fluxes between two cells along the given
  //! dimension.
  void computeIntercellFluxes(Cell& left, Cell& right);

public:
  SolverMUSCL(Parameters& params_, Grid& grid_);
  ~SolverMUSCL() = default;

  //! Run a single step.
  void step() override;
};
