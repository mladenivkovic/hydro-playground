

#include <iostream> // todo: necessary?

#include "Config.h" // todo: necessary?
#include "Logging.h"
#include "Parameters.h"
#include "Utils.h"
#include "Gas.h" // probably not necessary. wanna catch compile errors


int main(void) {

  using namespace hydro_playground;

  // Useless things first :)
  utils::print_header();

  // Initialise global paramters.
  parameters::Parameters::init();

  return 0;
}
