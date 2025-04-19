/**
 * @file SolverGodunov.cpp
 * @brief The Godunov hydro solver.
 */


#include "SolverGodunov.h"

#include "Gas.h"
#include "Riemann.h"
#include "Timer.h"

SolverGodunov::SolverGodunov(Parameters& params_, Grid& grid_):
  SolverBase(params_, grid_) {
}


/**
 * Compute all the intercell fluxes needed for a step update along the
 * direction of @param dimension.
 * This is where we solve the Riemann problems.
 */
void SolverGodunov::computeFluxes() {

  timer::Timer tick(timer::Category::HydroFluxes);

  // NOTE: we start earlier here!
  size_t first = _grid.getFirstCellIndex() - 1;
  size_t last  = _grid.getLastCellIndex();

  if (_direction == 0) {
    for (size_t j = first; j < last; j++) {
      for (size_t i = first; i < last; i++) {
        Cell& left  = _grid.getCell(i, j);
        Cell& right = _grid.getCell(i + 1, j);
        computeIntercellFluxes(left, right);
      }
    }
  } else if (_direction == 1) {
    for (size_t j = first; j < last; j++) {
      for (size_t i = first; i < last; i++) {
        Cell& left  = _grid.getCell(i, j);
        Cell& right = _grid.getCell(i, j + 1);
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
void SolverGodunov::computeIntercellFluxes(Cell& left, Cell& right) {

  riemann::Riemann solver(left.getPrim(), right.getPrim(), _direction);
  ConservedFlux    csol = solver.solve();

  left.setCFlux(csol);
}


/**
 * Run a simulation step.
 * We're using the first order accurate dimensional splitting approach here
 * (Section 7 in theory document).
 */
void SolverGodunov::step() {

  if (Dimensions != 2)
    error("Not implemented.");

  // First sweep
  // -----------------

  _direction = _step_count % 2;

  // zero out fluxes.
  _grid.resetFluxes();
  // No need to convert conserved quantities to primitive ones - see below.
  // grid.convertCons2Prim();
  // Send around updated boundary values
  _grid.applyBoundaryConditions();

  // Compute updated fluxes
  computeFluxes();

  // Apply fluxes and update current states
  integrateHydro<Device::cpu>(_dt);

  // Second sweep
  // -----------------

  // change dimension
  _direction = (_step_count + 1) % 2;
  // zero out fluxes.
  _grid.resetFluxes();
  // Transfer results from conserved states to primitive ones.
  _grid.convertCons2Prim();
  // Send around updated boundary values
  _grid.applyBoundaryConditions();

  // Compute updated fluxes
  computeFluxes();
  // Apply fluxes and update current states
  integrateHydro<Device::cpu>(_dt);


  // Get solution from previous step from conserved into primitive vars.
  // Do this here instead of at the start of this function so we can compute
  // dt. During startup, primitive values are correct already since that's
  // what we read from the ICs.
  _grid.convertCons2Prim();

  // Compute next time step.
  computeDt();
}
