/**
 * @file SolverMUSCL.cpp
 * @brief The MUSCL hydro solver.
 */

#include "SolverMUSCL.h"

#include "Gas.h"
#include "Limiter.h"
#include "Riemann.h"
#include "Timer.h"


SolverMUSCL::SolverMUSCL(Parameters& params_, Grid& grid_):
  SolverBase(params_, grid_) {
}


/**
 * Compute the flux F_{i+1/2} for a given cell w.r.t. a specific cell pair
 *
 * Here, we just solve the Riemann problem  with
 *
 *  U_L = U^R_{i,BEXT}, U_R = U^L_{i+1, BEXT},
 *
 * where
 *
 *  U^R_{i,BEXT} is the intermediate right extrapolated boundary value of cell i
 *
 *  U^L_{i+1, BEXT} is the intermediate left extrap. boundary value of cell i+1
 *
 * We then sample the solution at x = x/t = 0, because that is where we set the
 * initial boundary in the local coordinate system between the left and right
 * cell.
 *
 * @param left:  cell which stores the left state
 * @param right: cell which stores the right state
 */
void SolverMUSCL::computeIntercellFluxes(Cell& left, Cell& right) {

  PrimitiveState WL;
  WL.fromCons(left.getURMid());

  PrimitiveState WR;
  WR.fromCons(right.getULMid());

  riemann::Riemann solver(WL, WR, _direction);
  ConservedFlux    csol = solver.solve();

  left.setCFlux(csol);
}


/**
 * For the MUSCL-Hancock scheme, we need to first compute the slopes for each
 * conserved variable and each cell, and then compute the updated boundary
 * extrapolated values. Only then can we correctly compute the intercell
 * fluxes.
 * This function first computes the fluxes, and then computes the updated
 * intermediate state for each cell, and stores them in the cell.
 */
void SolverMUSCL::getBoundaryExtrapolatedValues(
  Cell& c, const ConservedState& UiP1, const ConservedState& UiM1, const Float dt_half
) {

  using CState = ConservedState;
  using CFlux  = ConservedFlux;

  // First get the slope.
  CState       slope;
  const CState Ui = c.getCons();

  limiter::limiterGetLimitedSlope(UiP1, Ui, UiM1, slope);

  Float rhoi   = Ui.getRho();
  Float rhovxi = Ui.getRhov(0);
  Float rhovyi = Ui.getRhov(1);
  Float Ei     = Ui.getE();


  // Get the left sloped state
  Float  rhoL   = rhoi - 0.5 * slope.getRho();
  Float  rhovxL = rhovxi - 0.5 * slope.getRhov(0);
  Float  rhovyL = rhovyi - 0.5 * slope.getRhov(1);
  Float  EL     = Ei - 0.5 * slope.getE();
  CState UL(rhoL, rhovxL, rhovyL, EL);

  // Get the left flux given the states.
  CFlux FL;
  FL.getCFluxFromCstate(UL, _direction);

  // Get the right sloped state
  Float  rhoR   = rhoi + 0.5 * slope.getRho();
  Float  rhovxR = rhovxi + 0.5 * slope.getRhov(0);
  Float  rhovyR = rhovyi + 0.5 * slope.getRhov(1);
  Float  ER     = Ei + 0.5 * slope.getE();
  CState UR(rhoR, rhovxR, rhovyR, ER);

  // Get the right flux given the states.
  CFlux FR;
  FR.getCFluxFromCstate(UR, _direction);

  Float dtdx_half = dt_half / _grid.getDx();


  Float rhoLmid   = rhoi + dtdx_half * (FL.getRho() - FR.getRho()) - 0.5 * slope.getRho();
  Float rhovxLmid = rhovxi + dtdx_half * (FL.getRhov(0) - FR.getRhov(0)) - 0.5 * slope.getRhov(0);
  Float rhovyLmid = rhovyi + dtdx_half * (FL.getRhov(1) - FR.getRhov(1)) - 0.5 * slope.getRhov(1);
  Float ELmid     = Ei + dtdx_half * (FL.getE() - FR.getE()) - 0.5 * slope.getE();

  CState ULmid(rhoLmid, rhovxLmid, rhovyLmid, ELmid);
  c.setULMid(ULmid);


  Float rhoRmid   = rhoi + dtdx_half * (FL.getRho() - FR.getRho()) + 0.5 * slope.getRho();
  Float rhovxRmid = rhovxi + dtdx_half * (FL.getRhov(0) - FR.getRhov(0)) + 0.5 * slope.getRhov(0);
  Float rhovyRmid = rhovyi + dtdx_half * (FL.getRhov(1) - FR.getRhov(1)) + 0.5 * slope.getRhov(1);
  Float ERmid     = Ei + dtdx_half * (FL.getE() - FR.getE()) + 0.5 * slope.getE();

  CState URmid(rhoRmid, rhovxRmid, rhovyRmid, ERmid);
  c.setURMid(URmid);
}


/**
 * Compute all the intercell fluxes needed for a step update along the
 * direction of @param dimension.
 */
void SolverMUSCL::computeFluxes(const Float dt_step) {

  timer::Timer tick(timer::Category::HydroFluxes);

  using CState = ConservedState;

  Float dt_half = 0.5 * dt_step;

  // NOTE: we start earlier here!
  size_t first = _grid.getFirstCellIndex() - 1;
  size_t last  = _grid.getLastCellIndex() + 1;

  if (_direction == 0) {

    // First, get the boundary extrapolated values.

    for (size_t j = first; j < last; j++) {
      for (size_t i = first; i < last; i++) {

        Cell&  cp1  = _grid.getCell(i + 1, j);
        CState UiP1 = cp1.getCons();

        Cell& c = _grid.getCell(i, j);

        Cell&  cm1  = _grid.getCell(i - 1, j);
        CState UiM1 = cm1.getCons();

        getBoundaryExtrapolatedValues(c, UiP1, UiM1, dt_half);
      }
    }

    // Then get the actual fluxes.

    for (size_t j = first; j < last; j++) {
      for (size_t i = first; i < last; i++) {
        Cell& left  = _grid.getCell(i, j);
        Cell& right = _grid.getCell(i + 1, j);
        computeIntercellFluxes(left, right);
      }
    }


  } else if (_direction == 1) {

    // First, get the boundary extrapolated values.

    for (size_t j = first; j < last; j++) {
      for (size_t i = first; i < last; i++) {

        Cell&  cp1  = _grid.getCell(i, j + 1);
        CState UiP1 = cp1.getCons();

        Cell& c = _grid.getCell(i, j);

        Cell&  cm1  = _grid.getCell(i, j - 1);
        CState UiM1 = cm1.getCons();

        getBoundaryExtrapolatedValues(c, UiP1, UiM1, dt_half);
      }
    }

    // Then get the actual fluxes.

    for (size_t j = first; j < last; j++) {
      for (size_t i = first; i < last; i++) {
        Cell& left  = _grid.getCell(i, j);
        Cell& right = _grid.getCell(i, j + 1);
        computeIntercellFluxes(left, right);
      }
    }
  }
}


/**
 * Run a simulation step.
 * We're using the second order accurate dimensional splitting approach here
 * (Section 7 in theory document).
 */
void SolverMUSCL::step() {

  if (Dimensions != 2)
    error("Not implemented.");


  // First sweep: One direction, half dt
  // -----------------------------------

  _direction = _step_count % 2;

  // zero out fluxes.
  _grid.resetFluxes();
  // No need to convert conserved quantities to primitive ones - see below.
  // grid.convertCons2Prim();
  // Send around updated boundary values
  _grid.applyBoundaryConditions();

  // Compute updated fluxes over half time step
  computeFluxes(0.5 * _dt);

  // Apply fluxes and update current states
  integrateHydro(0.5 * _dt);


  // Second sweep: Other direction, full dt
  // --------------------------------------

  // change dimension
  _direction = (_step_count + 1) % 2;
  // zero out fluxes.
  _grid.resetFluxes();
  // Transfer results from conserved states to primitive ones.
  // TODO: I'm pretty sure we can skip this unless we're writing output or computing dt.
  _grid.convertCons2Prim();
  // Send around updated boundary values
  _grid.applyBoundaryConditions();

  // Compute updated fluxes
  computeFluxes(_dt);
  // Apply fluxes and update current states
  integrateHydro(_dt);


  // Third sweep: First direction, full dt
  // --------------------------------------

  // change dimension
  _direction = _step_count % 2;
  // zero out fluxes.
  _grid.resetFluxes();
  // Transfer results from conserved states to primitive ones.
  // TODO: I'm pretty sure we can skip this unless we're writing output or computing dt.
  _grid.convertCons2Prim();
  // Send around updated boundary values
  _grid.applyBoundaryConditions();

  // Compute updated fluxes
  computeFluxes(0.5 * _dt);
  // Apply fluxes and update current states
  integrateHydro(0.5 * _dt);


  // Wrap-up
  // ------------

  // Get solution from previous step from conserved into primitive vars.
  // Do this here instead of at the start of this function so we can compute
  // dt. During startup, primitive values are correct already since that's
  // what we read from the ICs.
  _grid.convertCons2Prim();

  // Compute next time step.
  computeDt();
}
