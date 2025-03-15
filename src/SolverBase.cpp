#include "SolverBase.h"
#include <iomanip>
#include "Gas.h"
#include "IO.h"
#include "Logging.h"
#include "Timer.h"
#include "Constants.h"



/**
 * Constructor
 */
solver::SolverBase::SolverBase(parameters::Parameters& params_, grid::Grid& grid_):
  t(0.),
  dt(0.),
  step_count(0),
  total_mass_init(0.),
  total_mass_current(0.),
  params(params_),
  grid(grid_)
{ };


/**
 * @brief Do we need to run another step?
 */
bool solver::SolverBase::keepRunning(){

  Float tmax = params.getTmax();
  if (tmax > 0 and t >= tmax)
    return false;

  size_t nsteps = params.getNsteps();
  if (nsteps > 0 and step_count == nsteps)
    return false;

  return true;
}


/**
 * @brief Write a log to screen, if requested.
 *
 * @param timingstr: String returned from a timer object
 * containing the time measurement fo the step to be logged
 */
void solver::SolverBase::writeLog(const std::string& timingstr){

  size_t nstepsLog = params.getNstepsLog();
  bool write = ((nstepsLog == 0) or (step_count % nstepsLog == 0));
  if (not write)
    return;

  std::stringstream msg;
  constexpr size_t w = 14;
  constexpr size_t p = 6;

  msg << std::setw(w) << std::left;
  msg << step_count;
  msg << " ";
  msg << std::setw(w) << std::setprecision(p) << std::left;
  msg << t;
  msg << " ";
  msg << std::setw(w) << std::setprecision(p) << std::left;
  msg << dt;
  msg << " ";
#if DEBUG_LEVEL > 1
  msg << std::setw(w) << std::setprecision(p) << std::left;
  msg << total_mass_current / total_mass_init;
  msg << " ";
#endif
  msg << std::setw(w) << std::setprecision(p) << std::left;
  msg << timingstr;
  message(msg.str());
}



/**
 * @brief Write a log to screen, if requested.
 */
void solver::SolverBase::writeLogHeader(){

  size_t nsteps_log = params.getNstepsLog();
  bool write = ((nsteps_log == 0) or (step_count % nsteps_log == 0));
  if (not write)
    return;

  std::stringstream msg;
  constexpr size_t w = 14;

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
  msg << "Mnow/Minit";
  msg << " ";
#endif
  msg << std::setw(w) << std::left;
  msg << "step duration";
  message(msg.str());
}





/**
 * @brief Get the maximally perimissible time step size.
 */
void solver::SolverBase::computeDt(){

  message("Computing next dt.", logging::LogLevel::Debug);
  timer::Timer tick(timer::Category::CollectDt);

  if (Dimensions != 2)
    error("Not Implemented");

  size_t first = grid.getFirstCellIndex();
  size_t last = grid.getLastCellIndex();

  Float vxmax = 0.;
  Float vymax = 0.;

  for (size_t j = first; j < last; j++){
    for (size_t i = first; i < last; i++){
      cell::Cell& c = grid.getCell(i, j);
      idealGas::PrimitiveState& p = c.getPrim();
      Float vx = std::abs(p.getV(0));
      Float vy = std::abs(p.getV(1));
      Float a = p.getSoundSpeed();
      Float Sx = a + vx;
      vxmax = Sx > vxmax ? Sx : vxmax;
      Float Sy = a + vy;
      vymax = Sy > vymax ? Sy : vymax;
    }
  }

  Float dx_inv = 1. / grid.getDx();
  Float vxdx = vxmax * dx_inv;
  Float vydx = vymax * dx_inv;

  dt = params.getCcfl() / (vxdx + vydx);

  // sometimes there might be trouble with sharp discontinuities at the
  // beginning, so reduce the timestep for the first few steps.
  if (step_count <= 5)
    dt *= 0.2;

  if (dt < cst::DT_MIN){
    std::stringstream msg;
    msg << "Got weird time step? dt=" << dt;
    error(msg.str());
  }


  // timing("Compute next dt took " + tick.tock());
}




/**
 * @brief Apply the actual time integration step.
 *
 * @param dimension In which dimension to apply the updates.
 */
void solver::SolverBase::integrateHydro(const size_t dimension){

  message("Integrating dim="+std::to_string(dimension), logging::LogLevel::Debug);
  timer::Timer tick(timer::Category::HydroIntegrate);

  if (Dimensions != 2)
    error("Not Implemented");

  size_t first = grid.getFirstCellIndex();
  size_t last = grid.getLastCellIndex();

  const Float dtdx = dt / grid.getDx();

  if (dimension == 0){
    for (size_t j = first; j < last; j++){
      for (size_t i = first; i < last; i++){
        cell::Cell& left = grid.getCell(i-1, j);
        cell::Cell& right = grid.getCell(i, j);
        applyTimeUpdate(left, right, dtdx);
      }
    }
  }
  else if (dimension == 1){
    for (size_t j = first; j < last; j++){
      for (size_t i = first; i < last; i++){
        cell::Cell& left = grid.getCell(i, j-1);
        cell::Cell& right = grid.getCell(i, j);
        applyTimeUpdate(left, right, dtdx);
      }
    }
  }
  else {
    error("Unknown dimension " + std::to_string(dimension));
  }

  // timing("Hydro integration took " + tick.tock());
}



/**
 * Apply the time update for a pair of cells. This Updates the conserved state
 * using the fluxes in the cells.
 *
 * TODO: Write down equation from theory document
 *
 * @param right is the cell with index i that we are trying to update;
 * @param left is the cell i-1, which stores the flux at i-1/2
 * @param dtdx: dt / dx
 */
void solver::SolverBase::applyTimeUpdate(cell::Cell& left, cell::Cell& right, const Float dtdx){

  idealGas::ConservedState& r = right.getCons();
  const idealGas::ConservedState& lflux = left.getCFlux();
  const idealGas::ConservedState& rflux = right.getCFlux();

  Float rho = r.getRho() + dtdx * (lflux.getRho() - rflux.getRho());
  r.setRho(rho);

  Float vx = r.getRhov(0) + dtdx * (lflux.getRhov(0) - rflux.getRhov(0));
  r.setRhov(0, vx);

  if (Dimensions > 1){
    Float vy = r.getRhov(0) + dtdx * (lflux.getRhov(1) - rflux.getRhov(1));
    r.setRhov(1, vy);
  }

  Float e = r.getE() + dtdx * (lflux.getE() - rflux.getE());
  r.setE(e);
}


/**
 * @brief Main solver routine.
 * Should be the same for all solvers. What differs is the contents of
 * Solver::step();
 */
void solver::SolverBase::solve(){

  // Set the stage.
  logging::setStage(logging::LogStage::Step);
  timer::Timer tick(timer::Category::SolverTot);

  // Fill out conserved variables from read-in primitive ones.
  grid.convertCons2Prim();

#if DEBUG_LEVEL > 1
  // Collect the total mass to verify that we're actually conservative.
  total_mass_init = grid.collectTotalMass();
#endif

  auto writer = IO::OutputWriter(params, grid);

  // Dump step 0 data first
  writer.dump(t, step_count);

  // Get current time step size
  computeDt();

  // Show the output header.
  writeLogHeader();

  // Main loop.
  while (keepRunning()){

    // TODO(mivkov): test that this actually modifies dt
    bool write_output = writer.dumpThisStep(step_count, t, dt);

    timer::Timer tickStep(timer::Category::Step);

    step();

    std::string timingStep = tickStep.tock();

    // update time and step
    t += dt;
    step_count++;

    // Write output files
    if (write_output)
      writer.dump(t, step_count);

    // Talk to me
    writeLog(timingStep);
  }

  // if you haven't written the output in the final step, do it now
  if (not writer.dumpThisStep(step_count, t, dt))
    writer.dump(t, step_count);

  timing("Main solver took " + tick.tock());
}

