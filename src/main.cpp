

#include <iostream> // todo: necessary?
#include <sstream>

#include "Grid.h"
// #include "Config.h" // todo: necessary?
#include "IO.h"
#include "Logging.h"
#include "Parameters.h"
#include "Utils.h"


int main(int argc, char* argv[]) {

  // Set the logging stage. Were' in the header phase.
  logging::Log::setStage(logging::LogStage::Header);

  // Set default verbosity levels.
  // Note that this can be changed through cmdline flags.
  logging::Log::setVerbosity(logging::LogLevel::Quiet);

  // Get a handle on global vars so they're always in scope
  auto params = parameters::Parameters();
  auto grid = grid::Grid();

  // Useless things first :)
  utils::printHeader();

  // Were' in the initialisation phase.
  logging::Log::setStage(logging::LogStage::Init);

  // Fire up IO
  IO::InputParse input(argc, argv);

  // Read the parameters from the config file and initialise global paramters...
  input.readConfigFile(params);
  params.initDerived();

  // When very verbose, print out used parameters
  message("Running with parameters:", logging::LogLevel::Debug);
  message(params.toString(), logging::LogLevel::Debug);

  // Read initial conditions
  input.readICFile(grid, params);

  // grid::Grid::Instance.setBoundary(params);

  std::ostringstream msg;
  msg << "Got params nx=" << params.getNx();
  message(msg.str());

  // initialise the grid of cells
  // grid.InitGrid();
  // grid.setBoundary();

  return 0;
}
