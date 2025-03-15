#include "SolverBase.h"
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
  stepCount(0),
  params(params_),
  grid(grid_)
{ };


void solver::SolverBase::computeDt(){

  message("Computing next dt.", logging::LogLevel::Debug);
  timer::Timer tick(timer::Category::Ignore);

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
  if (stepCount <= 5)
    dt *= 0.2;

  if (dt < cst::DT_MIN){
    std::stringstream msg;
    msg << "Got weird time step? dt=" << dt;
    error(msg.str());
  }


  timing("Compute next dt took " + tick.tock());
}


/**
 * @brief Main solver routine.
 * Should be the same for all solvers. What differs is the contents of
 * Solver::step();
 */
void solver::SolverBase::solve(){

  logging::setStage(logging::LogStage::Step);

  auto writer = IO::OutputWriter();



  std::cout << "Called solve \n";

  for (int i = 0; i < 3; i++){
    step();
    t = t + 1.;
  }

}

