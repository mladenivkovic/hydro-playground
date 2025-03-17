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
 * Compute all the intercell fluxes needed for a step update along the
 * direction of @param dimension.
 */
void solver::SolverGodunov::computeFluxes() {

  timer::Timer tick(timer::Category::HydroFluxes);

  // NOTE: we start earlier here!
  size_t first = grid.getFirstCellIndex() - 1;
  size_t last  = grid.getLastCellIndex();

  if (dimension == 0) {
    for (size_t j = first; j < last; j++) {
      for (size_t i = first; i < last; i++) {
        cell::Cell& left  = grid.getCell(i, j);
        cell::Cell& right = grid.getCell(i + 1, j);
        computeIntercellFluxes(left, right);
      }
    }
  } else if (dimension == 1) {
    for (size_t j = first; j < last; j++) {
      for (size_t i = first; i < last; i++) {
        cell::Cell& left  = grid.getCell(i, j);
        cell::Cell& right = grid.getCell(i, j + 1);
        computeIntercellFluxes(left, right);
      }
    }
  }

  // timing("Computing fluxes took " + tick.tock());
}


/**
 * Compute the intercell fluxes between two cells along the given dimension.
 * We store the result in the left cell.
 */
void solver::SolverGodunov::computeIntercellFluxes(cell::Cell& left, cell::Cell& right) {

  riemann::Riemann        solver(left.getPrim(), right.getPrim(), dimension);
  idealGas::ConservedFlux csol = solver.solve();

  left.getCFlux() = csol;
}


/**
 * Run a simulation step.
 */
void solver::SolverGodunov::step() {

  if (Dimensions != 2)
    error("Not implemented.");

  // First sweep
  // -----------------

  dimension = step_count % 2;

  // zero out fluxes.
  grid.resetFluxes();
  // No need to convert conserved quantities to primitive ones - see below.
  // grid.convertCons2Prim();
  // Send around updated boundary values
  grid.applyBoundaryConditions();

  // Compute updated fluxes
  computeFluxes();

  // Apply fluxes and update current states
  integrateHydro();

  // Second sweep
  // -----------------

  // change dimension
  dimension = (step_count + 1) % 2;
  // zero out fluxes.
  grid.resetFluxes();
  // Transfer results from conserved states to primitive ones.
  grid.convertCons2Prim();
  // Send around updated boundary values
  grid.applyBoundaryConditions();

  // Compute updated fluxes
  computeFluxes();
  // Apply fluxes and update current states
  integrateHydro();


  // Get solution from previous step from conserved into primitive vars.
  // Do this here instead of at the start of this function so we can compute
  // dt. During startup, primitive values are correct already since that's
  // what we read from the ICs.
  grid.convertCons2Prim();

  // Compute next time step.
  computeDt();
}
