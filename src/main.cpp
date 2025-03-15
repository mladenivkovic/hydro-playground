
#include "Grid.h"
#include "IO.h"
#include "Logging.h"
#include "Parameters.h"
#include "Solver.h"
#include "Timer.h"
#include "Utils.h"


int main(int argc, char* argv[]) {

  // Start timing!
  timer::Timer tickTotal(timer::Category::Total);

  // Set the logging stage. We're in the header phase.
  logging::setStage(logging::LogStage::Header);

  // Set default verbosity levels.
  // Note that this can be changed through cmdline flags.
  logging::setVerbosity(logging::LogLevel::Quiet);

  // Get a handle on global vars so they're always in scope
  auto params = parameters::Parameters();
  auto grid   = grid::Grid();

  // Useless things first :)
  utils::printHeader();

  // Were' in the initialisation phase now.
  logging::setStage(logging::LogStage::Init);
  timer::Timer tickInit(timer::Category::Init);

  // Fire up IO
  IO::InputParse input(argc, argv);

  // Read the parameters from the parameter file and initialise global paramters...
  input.readParamFile(params);
  params.initDerived();
  grid.initGrid(params);

  // When very verbose, print out used parameters
  message("Running with parameters:", logging::LogLevel::Debug);
  message(params.toString(), logging::LogLevel::Debug);

  // This is the end of the init phase.
  (void)tickInit.tock();

  // Read initial conditions
  input.readICFile(grid);

  // Set boundary conditions
  grid.setBoundary();

  // Launch the solver.
  solver::Solver solver(params, grid);
  solver.solve();

  // Wrapup
  logging::setStage(logging::LogStage::Shutdown);
  message("Done. Bye!");

  (void)tickTotal.tock();
  // Use message intead of timing here: Always print timing at the end.
  message(tickTotal.getTimings());

  return 0;
}
