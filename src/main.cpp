

#include <iostream> // todo: necessary?

#include "Config.h" // todo: necessary?
#include "Gas.h"    // probably not necessary. wanna catch compile errors
#include "Cell.h"   // probably not necessary. wanna catch compile errors
#include "Logging.h"
#include "Parameters.h"
#include "Utils.h"
#include "IO.h"



int main(int argc, char* argv[]) {

  using namespace hydro_playground;

  // Useless things first :)
  utils::print_header();

  // Fire up IO
  IO::InputParse input(argc, argv);
  input.readCommandOptions();

  // Initialise global paramters.
  parameters::Parameters::Instance.init();

  // initialise the grid of cells
  Grid::Instance.InitGrid();

  Grid::Instance.setBoundary();

  return 0;
}
