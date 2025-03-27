/**
 * @file SolverGodunov.h
 * @brief The Godunov hydro solver.
 */

#pragma once

#include "SolverBase.h"


class SolverGodunov: public SolverBase {

  //! Compute the intercell fluxes needed for the update
  void computeFluxes();

  //! Compute the intercell fluxes between two cells along the given
  //! dimension.
  void computeIntercellFluxes(Cell& left, Cell& right);

public:
  SolverGodunov(Parameters& params_, Grid& grid_);
  ~SolverGodunov() = default;

  //! Run a single step.
  void step() override;
};

