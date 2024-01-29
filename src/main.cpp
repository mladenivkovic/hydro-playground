

#include <iostream> // todo: necessary?

#include "Cell.h"
#include "Config.h" // todo: necessary?
#include "Logging.h"
#include "Parameters.h"
#include "Utils.h"


int main(void) {

  using namespace hydro_playground;

  // Useless things first :)
  utils::print_header();

  // Initialise global paramters.
  parameters::Parameters::Instance.init();

  // initialise the grid of cells
  Grid::Instance.InitGrid();

  Grid::Instance.setBoundary();

  return 0;
}
