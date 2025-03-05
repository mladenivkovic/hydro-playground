

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
#if DEBUG_LEVEL == 0
  logging::Log::setVerbosity(logging::LogLevel::Quiet);
#else
  logging::Log::setVerbosity(logging::LogLevel::Debug);
#endif

  // Get a handle on singletons
  auto params = parameters::Parameters::Instance;

  // Useless things first :)
  utils::print_header();

  // Were' in the initialisation phase.
  logging::Log::setStage(logging::LogStage::Init);

  // Fire up IO
  IO::InputParse input(argc, argv);

  // Read the parameters from the config file and initialise global paramters...
  input.parseConfigFile();
  params.initDerived();

  // initialise the grid of cells
  grid::Grid::Instance.initGrid();

  // input.readICFile();

  grid::Grid::Instance.setBoundary();

  // Initialise global parameters.
  // auto grid   = cell::Grid::Instance;


  std::ostringstream msg;
  msg << "Got params dx=" << params.getDx();
  message(msg.str());

  // initialise the grid of cells
  // grid.InitGrid();
  // grid.setBoundary();

  return 0;
}
