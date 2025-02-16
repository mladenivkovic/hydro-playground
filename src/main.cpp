

#include <iostream> // todo: necessary?
#include <sstream>

#include "Cell.h"
#include "Config.h" // todo: necessary?
<<<<<<< HEAD
#include "Logging.h"
=======
// #include "Gas.h"    // probably not necessary. wanna catch compile errors
// #include "Logging.h"
>>>>>>> main
#include "Parameters.h"
#include "Utils.h"


int main(void) {

  // Useless things first :)
  utils::print_header();

  // Initialise global paramters.

  auto               params = parameters::Parameters::Instance;
  std::ostringstream msg;
  msg << "Got params dx=" << params.getDx();
  message(msg.str());

  // initialise the grid of cells
  Grid::Instance.InitGrid();

  Grid::Instance.setBoundary();

  return 0;
}
