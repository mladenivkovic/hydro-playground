
#include "Grid.h"
#include "IO.h"
#include "Logging.h"
#include "Parameters.h"
#include "Utils.h"


int main(int argc, char* argv[]) {

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

  // Fire up IO
  IO::InputParse input(argc, argv);

  // Read the parameters from the parameter file and initialise global paramters...
  input.readParamFile(params);
  params.initDerived();
  grid.initGrid(params);

  // When very verbose, print out used parameters
  message("Running with parameters:", logging::LogLevel::Debug);
  message(params.toString(), logging::LogLevel::Debug);

  // Read initial conditions
  logging::setStage(logging::LogStage::IO);
  input.readICFile(grid);

  std::ostringstream msg;
  msg << "Got params nx=" << params.getNx();
  message(msg.str());

  return 0;
}
