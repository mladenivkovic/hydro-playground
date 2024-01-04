

#include <iostream> // todo: necessary?

#include "config.h" // todo: necessary?
#include "logging.h"
#include "utils.h"



int main(void) {

  using namespace hydro_playground;

  std::cout << "Hello world!" << std::endl;
  utils::print_header();

  for (int v = 0; v < 4; v++){
    std::cout << "verbose = " << v << "\n";

    logging::Log("", logging::LogLevelQuiet, logging::LogStageHeader, v);
    logging::Log("", logging::LogLevelVerbose, logging::LogStageHeader, v);
    logging::Log("", logging::LogLevelDebug, logging::LogStageHeader, v);
    logging::Log("", logging::LogLevelQuiet, logging::LogStageHeader, v);
    logging::Log("", logging::LogLevelQuiet, logging::LogStageHeader, v);
  }

  return 0;
}
