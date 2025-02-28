

#include <iostream> // todo: necessary?
#include <sstream>

#include "Grid.h"
// #include "Config.h" // todo: necessary?
#include "IO.h"
#include "Logging.h"
#include "Parameters.h"
#include "Utils.h"


int main(int argc, char* argv[]) {

  // Were' in the header phase.
  logging::Log::setStage(logging::LogStage::Header);

  // Useless things first :)
  utils::print_header();

  // Were' in the initialisation phase.
  logging::Log::setStage(logging::LogStage::Init);

  // Fire up IO
  IO::InputParse input(argc, argv);

  // Initialise global paramters.
  parameters::Parameters::Instance.init();

  // initialise the grid of cells
  grid::Grid::Instance.initGrid();

  input.readICFile();

  grid::Grid::Instance.setBoundary();

  // Initialise global parameters.
  auto params = parameters::Parameters::Instance;
  // auto grid   = cell::Grid::Instance;


  std::ostringstream msg;
  msg << "Got params dx=" << params.getDx();
  message(msg.str());

  // initialise the grid of cells
  // grid.InitGrid();
  // grid.setBoundary();

  return 0;
}
