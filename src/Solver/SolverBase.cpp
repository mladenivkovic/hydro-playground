#include "SolverBase.h"

#include <iomanip>
#include <ios>

#include "Constants.h"
#include "Gas.h"
#include "IO.h"
#include "Logging.h"
#include "Riemann.h"
#include "Timer.h"


/**
 * Constructor
 */
solver::SolverBase::SolverBase(parameters::Parameters& params_, grid::Grid& grid_):
  _t(0.),
  _dt(0.),
  _dt_old(0.),
  _direction(0),
  _step_count(0),
  _total_mass_init(0.),
  _total_mass_current(0.),
  _params(params_),
  _grid(grid_) {};


/**
 * @brief Get the maximally perimissible time step size.
 */
void solver::SolverBase::computeDt() {

  message("Computing next dt.", logging::LogLevel::Debug);
  timer::Timer tick(timer::Category::CollectDt);

  if (Dimensions != 2) {
    error("Not Implemented");
    return;
  }

  size_t first = _grid.getFirstCellIndex();
  size_t last  = _grid.getLastCellIndex();

  Float vxmax = 0.;
  Float vymax = 0.;

  for (size_t j = first; j < last; j++) {
    for (size_t i = first; i < last; i++) {
      cell::Cell&               c  = _grid.getCell(i, j);
      idealGas::PrimitiveState& p  = c.getPrim();
      Float                     vx = std::abs(p.getV(0));
      Float                     vy = std::abs(p.getV(1));
      Float                     a  = p.getSoundSpeed();
      Float                     Sx = a + vx;
      vxmax                        = Sx > vxmax ? Sx : vxmax;
      Float Sy                     = a + vy;
      vymax                        = Sy > vymax ? Sy : vymax;
    }
  }

  Float dx_inv = 1. / _grid.getDx();
  Float vxdx   = vxmax * dx_inv;
  Float vydx   = vymax * dx_inv;

  _dt = _params.getCcfl() / (vxdx + vydx);

  // sometimes there might be trouble with sharp discontinuities at the
  // beginning, so reduce the timestep for the first few steps.
  if (_step_count <= 5)
    _dt *= 0.2;

  if (_dt < cst::DT_MIN) {
    std::stringstream msg;
    msg << "Got weird time step? dt=" << _dt;
    error(msg.str());
  }

  // timing("Compute next dt took " + tick.tock());
}


/**
 * @brief Apply the actual time integration step.
 *
 * @param dt_step time interval to integrate over. In more sophisticated
 * schemes, like MUSCL-Hancock, we do several time integrations over
 * sub-intervals in a single step. So this is necessary.
 */
void solver::SolverBase::integrateHydro(const Float dt_step) {

  message("Integrating dim=" + std::to_string(_direction), logging::LogLevel::Debug);
  timer::Timer tick(timer::Category::HydroIntegrate);

  if (Dimensions != 2) {
    error("Not Implemented");
    return;
  }

  size_t first = _grid.getFirstCellIndex();
  size_t last  = _grid.getLastCellIndex();

  const Float dtdx = dt_step / _grid.getDx();

  if (_direction == 0) {
    for (size_t j = first; j < last; j++) {
      for (size_t i = first; i < last; i++) {
        cell::Cell& left  = _grid.getCell(i - 1, j);
        cell::Cell& right = _grid.getCell(i, j);
        applyTimeUpdate(left, right, dtdx);
      }
    }
  } else if (_direction == 1) {
    for (size_t j = first; j < last; j++) {
      for (size_t i = first; i < last; i++) {
        cell::Cell& left  = _grid.getCell(i, j - 1);
        cell::Cell& right = _grid.getCell(i, j);
        applyTimeUpdate(left, right, dtdx);
      }
    }
  } else {
    error("Unknown dimension " + std::to_string(_direction));
  }

  // timing("Hydro integration took " + tick.tock());
}


/**
 * Apply the time update for a pair of cells. This Updates the conserved state
 * using the fluxes in the cells. Eq. 87 and 91 in theory document.
 *
 * @param right is the cell with index i that we are trying to update;
 * @param left is the cell i-1, which stores the flux at i-1/2
 * @param dtdx: dt / dx
 */
void solver::SolverBase::applyTimeUpdate(cell::Cell& left, cell::Cell& right, const Float dtdx) {

  idealGas::ConservedState&       cr     = right.getCons();
  const idealGas::ConservedState& lcflux = left.getCFlux();
  const idealGas::ConservedState& rcflux = right.getCFlux();

  Float rho = cr.getRho() + dtdx * (lcflux.getRho() - rcflux.getRho());
  cr.setRho(rho);

  Float vx = cr.getRhov(0) + dtdx * (lcflux.getRhov(0) - rcflux.getRhov(0));
  cr.setRhov(0, vx);

  if (Dimensions > 1) {
    Float vy = cr.getRhov(1) + dtdx * (lcflux.getRhov(1) - rcflux.getRhov(1));
    cr.setRhov(1, vy);
  }

  Float e = cr.getE() + dtdx * (lcflux.getE() - rcflux.getE());
  cr.setE(e);
}


/**
 * @brief Main solver routine.
 * Should be the same for all solvers. What differs is the contents of
 * Solver::step();
 */
void solver::SolverBase::solve() {

  // Set the stage.
  logging::setStage(logging::LogStage::Step);
  timer::Timer tick(timer::Category::SolverTot);

  // Fill out conserved variables from read-in primitive ones.
  _grid.convertPrim2Cons();


#if DEBUG_LEVEL > 1
  // Collect the total mass to verify that we're actually conservative.
  _total_mass_init = _grid.collectTotalMass();
#endif

  auto writer = IO::OutputWriter(_params, _grid);

  // Dump step 0 data first
  writer.dump(_t, _step_count);

  // Get current time step size
  computeDt();

  // Show the output header.
  writeLogHeader();
  bool written_output = false;

  // Main loop.
  while (keepRunning()) {

    // Do this first, since it may modify dt.
    bool write_output = writer.dumpThisStep(_step_count, _t, _dt);
    written_output    = false;

    // Store this time step.
    _dt_old = _dt;

    timer::Timer tickStep(timer::Category::Step);

    // The actual solver step.
    step();

    std::string timingStep = tickStep.tock();

    // update time and step. dt is next time step size at this point.
    _t += _dt_old;
    _step_count++;

#if DEBUG_LEVEL > 1
    // Collect the total mass to verify that we're actually conservative.
    _total_mass_current = _grid.collectTotalMass();
#endif

    // Write output files
    if (write_output) {
      writer.dump(_t, _step_count);
      written_output = true;
    }

    // Talk to me
    writeLog(timingStep);
  }

  // if you haven't written the output in the final step, do it now
  if (not written_output)
    writer.dump(_t, _step_count);

  timing("Main solver took " + tick.tock());
}


/**
 * @brief Do we need to run another step?
 */
bool solver::SolverBase::keepRunning() {

  Float tmax = _params.getTmax();
  if (tmax > 0. and _t >= tmax)
    return false;

  size_t nsteps = _params.getNsteps();
  if (nsteps > 0 and _step_count == nsteps)
    return false;

  return true;
}


/**
 * @brief Write a log to screen, if requested.
 *
 * @param timingstr: String returned from a timer object
 * containing the time measurement fo the step to be logged
 */
void solver::SolverBase::writeLog(const std::string& timingstr) {

  size_t nstepsLog = _params.getNstepsLog();
  bool   write     = ((nstepsLog == 0) or (_step_count % nstepsLog == 0));
  if (not write)
    return;

  std::stringstream msg;
  constexpr size_t  w = 14;
  constexpr size_t  p = 6;

  msg << std::setw(w) << std::left;
  msg << _step_count;
  msg << " ";
  msg << std::setw(w) << std::setprecision(p) << std::scientific << std::left;
  msg << _t << " ";
  msg << _dt_old;
  msg << " ";
#if DEBUG_LEVEL > 1
  msg << _total_mass_current / _total_mass_init;
  msg << " ";
#endif
  msg << std::right << std::setw(w) << timingstr;
  message(msg.str());
}


/**
 * @brief Write a log to screen, if requested.
 */
void solver::SolverBase::writeLogHeader() {

  std::stringstream msg;
  constexpr size_t  w = 14;

  msg << std::setw(w) << std::left;
  msg << "step";
  msg << " ";
  msg << std::setw(w) << std::left;
  msg << "time";
  msg << " ";
  msg << std::setw(w) << std::left;
  msg << "dt";
  msg << " ";
#if DEBUG_LEVEL > 1
  msg << std::setw(w) << std::left;
  msg << "M_now/M_init";
  msg << " ";
#endif
  msg << "step duration";
  message(msg.str());
}
