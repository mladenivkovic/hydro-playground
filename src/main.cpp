

#include <iostream> // todo: necessary?

#include "Config.h" // todo: necessary?
#include "Gas.h"    // probably not necessary. wanna catch compile errors
#include "Logging.h"
#include "Parameters.h"
#include "Utils.h"


int main(void) {

  // Useless things first :)
  utils::print_header();

  // Initialise global paramters.
  parameters::Parameters::init();

  return 0;
}
