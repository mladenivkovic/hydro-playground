/**
 * @file SolverGodunov.cpp
 * @brief The Godunov hydro solver.
 */


#include "SolverGodunov.h"

#include "Gas.h"
#include "Riemann.h"
#include "Timer.h"

solver::SolverGodunov::SolverGodunov(parameters::Parameters& params_, grid::Grid& grid_):
  SolverBase(params_, grid_) {
}


/**
 * Compute the intercell fluxes between two cells along the given dimension.
 * We store the result in the left cell.
 */
inline void solver::SolverGodunov::computeIntercellFluxes(
  cell::Cell& left, cell::Cell& right, const size_t dimension
) {

  riemann::Riemann         solver(left.getPrim(), right.getPrim(), dimension);
  idealGas::PrimitiveState sol = solver.solve();
  left.getCFlux().getCFluxFromPstate(sol, dimension);
}


/**
 * Compute all the intercell fluxes needed for a step update along the
 * direction of @param dimension.
 */
void solver::SolverGodunov::computeFluxes(const size_t dimension) {

  timer::Timer tick(timer::Category::HydroFluxes);

  // NOTE: we start earlier here!
  size_t first = grid.getFirstCellIndex() - 1;
  size_t last  = grid.getLastCellIndex();

  if (dimension == 0) {
    for (size_t j = first; j < last; j++) {
      for (size_t i = first; i < last; i++) {
        cell::Cell& left  = grid.getCell(i, j);
        cell::Cell& right = grid.getCell(i + 1, j);
        computeIntercellFluxes(left, right, dimension);
      }
    }
  } else if (dimension == 1) {
    for (size_t j = first; j < last; j++) {
      for (size_t i = first; i < last; i++) {
        cell::Cell& left  = grid.getCell(i, j);
        cell::Cell& right = grid.getCell(i, j + 1);
        computeIntercellFluxes(left, right, dimension);
      }
    }
  }

  // timing("Computing fluxes took " + tick.tock());
}


//! Run a single step.
void solver::SolverGodunov::step() {

  if (Dimensions != 2)
    error("Not implemented.");

  // zero out fluxes.
  grid.resetFluxes();
  // Send around value
  grid.applyBoundaryConditions();


  // First sweep
  size_t dimension = step_count % 2;
  computeFluxes(dimension);
  integrateHydro(dimension);

  dimension = (step_count + 1) % 2;
  computeFluxes(dimension);
  integrateHydro(dimension);


  // Get solution from previous step from conserved into primitive vars.
  grid.convertCons2Prim();

  // Compute next time step.
  computeDt();
}
