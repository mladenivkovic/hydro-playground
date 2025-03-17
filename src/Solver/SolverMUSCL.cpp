/**
 * @file SolverMUSCL.cpp
 * @brief The MUSCL hydro solver.
 */


#include "SolverMUSCL.h"

#include "Gas.h"
#include "Riemann.h"
#include "Limiter.h"
#include "Timer.h"


solver::SolverMUSCL::SolverMUSCL(parameters::Parameters& params_, grid::Grid& grid_):
  SolverBase(params_, grid_) {
}

/**
 * Compute the flux F_{i+1/2} for a given cell w.r.t. a specific cell pair
 *
 * Here, we just solve the Riemann problem  with
 *  U_L = U^R_{i,BEXT}, U_R = U^L_{i+1, BEXT},
 * where
 *  U^R_{i,BEXT} is the intermediate right extrapolated boundary value of cell i
 *  U^L_{i+1, BEXT} is the intermediate left extrap. boundary value of cell i+1
 * and then sample the solution at x = x/t = 0, because that
 * is where we set the initial boundary in the local coordinate system between
 * the left and right cell.
 *
 * @param left:  cell which stores the left state
 * @param right: cell which stores the right state
 * @param dt:   current time step
 */

void solver::SolverMUSCL::computeIntercellFluxes(cell::Cell& left, cell::Cell& right){

  idealGas::PrimitiveState WL;
  WL.fromCons(left.getURMid());

  idealGas::PrimitiveState WR;
  WR.fromCons(left.getURMid());

  riemann::Riemann        solver(WL, WR, dimension);
  idealGas::ConservedFlux csol = solver.solve();

  left.getCFlux() = csol;
}



/**
 *
 */
void solver::SolverMUSCL::getBoundaryExtrapolatedValues(
    cell::Cell& c,
    const idealGas::ConservedState& UiP1,
    const idealGas::ConservedState& UiM1,
    const Float dt_half){

  // First get the slope.
  idealGas::ConservedState slope;
  idealGas::ConservedState Ui = c.getCons();

  limiter::limiterGetLimitedSlope(UiP1, Ui, UiM1, slope);

  Float rhoi = Ui.getRho() ;
  Float rhovxi = Ui.getRhov(0);
  Float rhovyi = Ui.getRhov(1);
  Float Ei = Ui.getE() ;


  // Get the left sloped state
  Float rhoL = - 0.5 * slope.getRho();
  Float rhovxL = rhovxi - 0.5 * slope.getRhov(0);
  Float rhovyL = rhovyi - 0.5 * slope.getRhov(1);
  Float EL = Ei - 0.5 * slope.getE();
  idealGas::ConservedState UL(rhoL, rhovxL, rhovyL, EL);

  // Get the left flux given the states.
  idealGas::ConservedFlux FL;
  FL.getCFluxFromCstate(UL, dimension);

  // Get the right sloped state
  Float rhoR = rhoi + 0.5 * slope.getRho();
  Float rhovxR = rhovxi + 0.5 * slope.getRhov(0);
  Float rhovyR = rhovyi + 0.5 * slope.getRhov(1);
  Float ER = Ei + 0.5 * slope.getE();
  idealGas::ConservedState UR(rhoR, rhovxR, rhovyR, ER);

  // Get the right flux given the states.
  idealGas::ConservedFlux FR;
  FR.getCFluxFromCstate(UR, dimension);

  Float dtdx_half = dt_half / grid.getDx();


  Float rhoLmid = rhoi + dtdx_half * (FL.getRho() - FR.getRho()) - 0.5 * slope.getRho();
  Float rhovxLmid = rhovxi + dtdx_half * (FL.getRhov(0) - FR.getRhov(0)) - 0.5 * slope.getRhov(0);
  Float rhovyLmid = rhovyi + dtdx_half * (FL.getRhov(1) - FR.getRhov(1)) - 0.5 * slope.getRhov(1);
  Float ELmid = Ei + dtdx_half * (FL.getE() - FR.getE()) - 0.5 * slope.getE();

  idealGas::ConservedState ULmid(rhoLmid, rhovxLmid, rhovyLmid, ELmid);
  c.getULMid() = ULmid;


  Float rhoRmid = rhoi + dtdx_half * (FL.getRho() - FR.getRho()) + 0.5 * slope.getRho();
  Float rhovxRmid = rhovxi + dtdx_half * (FL.getRhov(0) - FR.getRhov(0)) + 0.5 * slope.getRhov(0);
  Float rhovyRmid = rhovyi + dtdx_half * (FL.getRhov(1) - FR.getRhov(1)) + 0.5 * slope.getRhov(1);
  Float ERmid = Ei + dtdx_half * (FL.getE() - FR.getE()) + 0.5 * slope.getE();

  idealGas::ConservedState URmid(rhoRmid, rhovxRmid, rhovyRmid, ERmid);
  c.getURMid() = URmid;
}


/**
 * Compute all the intercell fluxes needed for a step update along the
 * direction of @param dimension.
 */
void solver::SolverMUSCL::computeFluxes(const Float dt_step) {

  timer::Timer tick(timer::Category::HydroFluxes);

  Float dt_half = 0.5 * dt_step;

  // NOTE: we start earlier here!
  size_t first = grid.getFirstCellIndex();
  size_t last  = grid.getLastCellIndex();

  if (dimension == 0) {

    // First, get the boundary extrapolated values.

    for (size_t j = first; j < last; j++) {
      for (size_t i = first; i < last; i++) {

        cell::Cell& cp1  = grid.getCell(i+1, j);
        idealGas::ConservedState UiP1 = cp1.getCons();

        cell::Cell& c  = grid.getCell(i, j);

        cell::Cell& cm1  = grid.getCell(i-1, j);
        idealGas::ConservedState UiM1 = cm1.getCons();

        getBoundaryExtrapolatedValues(c, UiP1, UiM1, dt_half);

      }
    }

    // Then get the actual fluxes.

    for (size_t j = first; j < last; j++) {
      for (size_t i = first; i < last; i++) {
        cell::Cell& left  = grid.getCell(i, j);
        cell::Cell& right = grid.getCell(i + 1, j);
        computeIntercellFluxes(left, right);
      }
    }


  } else if (dimension == 1) {

    // First, get the boundary extrapolated values.

    for (size_t j = first; j < last; j++) {
      for (size_t i = first; i < last; i++) {

        cell::Cell& cp1  = grid.getCell(i, j+1);
        idealGas::ConservedState UiP1 = cp1.getCons();

        cell::Cell& c  = grid.getCell(i, j);

        cell::Cell& cm1  = grid.getCell(i, j+1);
        idealGas::ConservedState UiM1 = cm1.getCons();

        getBoundaryExtrapolatedValues(c, UiP1, UiM1, dt_half);

      }
    }

    // Then get the actual fluxes.

    for (size_t j = first; j < last; j++) {
      for (size_t i = first; i < last; i++) {
        cell::Cell& left  = grid.getCell(i, j);
        cell::Cell& right = grid.getCell(i, j + 1);
        computeIntercellFluxes(left, right);
      }
    }
  }
}



/**
 * Run a simulation step.
 */
void solver::SolverMUSCL::step() {

  if (Dimensions != 2)
    error("Not implemented.");

  // First sweep: One direction, half dt
  // -----------------------------------

  dimension = step_count % 2;

  grid.resetFluxes();
  // zero out fluxes.

  // grid.convertCons2Prim();
  // Send around updated boundary values
  grid.applyBoundaryConditions();

  // Compute updated fluxes over half time step
  computeFluxes(0.5 * dt);

  // Apply fluxes and update current states
  integrateHydro();


  // Second sweep: Other direction, full dt
  // --------------------------------------

  // change dimension
  dimension = (step_count + 1) % 2;
  // zero out fluxes.
  grid.resetFluxes();
  // Transfer results from conserved states to primitive ones.
  grid.convertCons2Prim();
  // Send around updated boundary values
  grid.applyBoundaryConditions();

  // Compute updated fluxes
  computeFluxes(dt);
  // Apply fluxes and update current states
  integrateHydro();


  // Third sweep: First direction, full dt
  // --------------------------------------

  // change dimension
  dimension = step_count % 2;
  // zero out fluxes.
  grid.resetFluxes();
  // Transfer results from conserved states to primitive ones.
  grid.convertCons2Prim();
  // Send around updated boundary values
  grid.applyBoundaryConditions();

  // Compute updated fluxes
  computeFluxes(0.5 * dt);
  // Apply fluxes and update current states
  integrateHydro();

}

