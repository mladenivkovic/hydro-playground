

#include <iostream> // todo: necessary?

#include "Config.h" // todo: necessary?
#include "Gas.h"    // probably not necessary. wanna catch compile errors
#include "Cell.h"   // probably not necessary. wanna catch compile errors
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
  Grid::Instance.InitCells();

  message("Does this work?");

  return 0;
}
