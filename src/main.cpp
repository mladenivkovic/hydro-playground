

#include <iostream> // todo: necessary?
#include <sstream>

#include "Config.h" // todo: necessary?
// #include "Gas.h"    // probably not necessary. wanna catch compile errors
// #include "Logging.h"
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
  return 0;
}
