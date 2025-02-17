

#include <iostream> // todo: necessary?
#include <sstream>

#include "Grid.h"
// #include "Config.h" // todo: necessary?
#include "Logging.h"
#include "Parameters.h"
#include "Utils.h"
#include "IO.h"



int main(int argc, char* argv[]) {


  // Useless things first :)
  utils::print_header();

  // Fire up IO
  IO::InputParse input(argc, argv);
  if ( not input.inputIsValid() ) {
    return 1;
  }

  // all is good, let's keep going
  input.readCommandOptions();

  // Initialise global paramters.
  parameters::Parameters::Instance.init();

  // initialise the grid of cells
  grid::Grid::Instance.InitGrid();

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
