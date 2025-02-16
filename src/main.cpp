

#include <iostream> // todo: necessary?
#include <sstream>

#include "Cell.h"
// #include "Config.h" // todo: necessary?
#include "Logging.h"
#include "Parameters.h"
#include "Utils.h"


int main() {

  // Useless things first :)
  utils::print_header();

  // Initialise global parameters.
  auto params = parameters::Parameters::Instance;
  auto grid   = cell::Grid::Instance;


  std::ostringstream msg;
  msg << "Got params dx=" << params.getDx();
  message(msg.str());

  // initialise the grid of cells
  grid.InitGrid();
  grid.setBoundary();

  return 0;
}
