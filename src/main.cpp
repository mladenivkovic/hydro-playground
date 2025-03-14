
#include "Grid.h"
#include "IO.h"
#include "Logging.h"
#include "Parameters.h"
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
  auto writer = IO::OutputWriter();

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

  std::ostringstream msg;
  msg << "Got params nx=" << params.getNx();
  message(msg.str());

  logging::setStage(logging::LogStage::Step);
  writer.dump(params, grid, 0., 1);
  writer.dump(params, grid, 1., 2);

  (void)tickTotal.tock();
  timing(tickTotal.getTimings());

  return 0;
}
