#include <iomanip>
#include <iostream>
#include <sstream>

#include "Logging.h"


void printFatLine(void) {
  std::cout << "====================================================";
  std::cout << "====================================================";
  std::cout << "====================================================" << std::endl;
}
void printThinLine(void) {
  std::cout << "----------------------------------------------------";
  std::cout << "----------------------------------------------------";
  std::cout << "----------------------------------------------------" << std::endl;
}


int main(void) {

  using namespace hydro_playground;

  // -----------------------------------------------------------------
  // First, let's test the different Log functions without the macros.
  // If this doesn't work, we're in big trouble trying to debug that
  // with macros involved.
  // -----------------------------------------------------------------

  const char*       char_msg = "Const Char message";
  std::string       str_msg  = std::string("String message");
  std::stringstream ss_msg;
  ss_msg << "String stream message";

  logging::Log::logMessage(
    __FILENAME__, __FUNCTION__, __LINE__, char_msg, logging::LogLevel::Quiet, logging::LogStage::Init
  );
  logging::Log::logMessage(
    __FILENAME__, __FUNCTION__, __LINE__, str_msg, logging::LogLevel::Quiet, logging::LogStage::Init
  );
  logging::Log::logMessage(
    __FILENAME__, __FUNCTION__, __LINE__, ss_msg, logging::LogLevel::Quiet, logging::LogStage::Init
  );
  logging::Log::logMessage(
    __FILENAME__,
    __FUNCTION__,
    __LINE__,
    "Directly writing in here",
    logging::LogLevel::Quiet,
    logging::LogStage::Init
  );

  logging::Log::logWarning(__FILENAME__, __FUNCTION__, __LINE__, char_msg);
  logging::Log::logWarning(__FILENAME__, __FUNCTION__, __LINE__, str_msg);
  logging::Log::logWarning(__FILENAME__, __FUNCTION__, __LINE__, ss_msg);
  logging::Log::logWarning(__FILENAME__, __FUNCTION__, __LINE__, "Directly writing in here");

  // Now try the message() macros
  message(char_msg);
  message(char_msg, logging::LogLevel::Quiet, logging::LogStage::Undefined);

  message(str_msg);
  message(str_msg, logging::LogLevel::Quiet, logging::LogStage::Undefined);

  message(ss_msg);
  message(ss_msg, logging::LogLevel::Quiet, logging::LogStage::Undefined);

  message("Directly writing in here");
  message("Directly writing in here", logging::LogLevel::Quiet, logging::LogStage::Undefined);

  // Now try the warning() macros
  warning(char_msg);
  warning(str_msg);
  warning(ss_msg);
  warning("Directly writing in here");


  printFatLine();


  // -----------------------------------------------------------------
  // Now, let's see whether the verbosity levels work as intended.
  // -----------------------------------------------------------------

  int levelMin = static_cast<int>(logging::LogLevel::Undefined);
  int levelMax = static_cast<int>(logging::LogLevel::Count);
  int stageMin = static_cast<int>(logging::LogStage::Undefined);
  int stageMax = static_cast<int>(logging::LogStage::Count);

  // Vary verbosity levels
  for (int verb = levelMin; verb <= levelMax; verb++) {

    logging::Log::setVerbosity(verb);

    // vary code stages
    for (int stage = stageMin; stage < stageMax; stage++) {

      logging::LogStage s = static_cast<logging::LogStage>(stage);

      // vary verbosity level of messages
      for (int level = levelMin; level < levelMax; level++) {

        logging::LogLevel l = static_cast<logging::LogLevel>(level);

        bool expect_print = (verb >= level);

        std::stringstream msg;

        msg << std::setw(12) << "Verbosity=" << std::setw(3) << verb;
        msg << std::setw(12) << " msg level=" << std::setw(3) << level;
        msg << std::setw(12) << " code stage=" << std::setw(3) << stage;
        msg << std::setw(12) << " should print?=" << expect_print;
        msg << " | ";

        std::cout << msg.str();
        if (not expect_print)
          std::cout << "\n";

        message("My Message", l, s);
      }
      printThinLine();
    }
    printFatLine();
  }


  return 0;
}
