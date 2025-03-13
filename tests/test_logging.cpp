#include <iomanip>
#include <iostream>
#include <sstream>

#include "Logging.h"


void printFatLine() {
  std::cout << "====================================================";
  std::cout << "====================================================";
  std::cout << "====================================================\n";
}
void printThinLine() {
  std::cout << "----------------------------------------------------";
  std::cout << "----------------------------------------------------";
  std::cout << "----------------------------------------------------\n";
}


int main() {

  // -----------------------------------------------------------------
  // First, let's test the different Log functions without the macros.
  // If this doesn't work, we're in big trouble trying to debug that
  // with macros involved.
  // -----------------------------------------------------------------

  const char* char_msg = "Const Char message";
  // added std::allocator<char> here, which is the default argument, to stop
  // the linter from screaming at me.
  const std::string str_msg = std::string("String message", std::allocator<char>());
  std::stringstream ss_msg;
  ss_msg << "String stream message";

  logging::Log& logger = logging::Log::getInstance();

  logger.logMessage(
    char_msg,
    logging::LogLevel::Quiet,
    logging::LogStage::Init,
    std::source_location::current().file_name(),
    std::source_location::current().function_name(),
    std::source_location::current().line()
  );
  logger.logMessage(
    str_msg,
    logging::LogLevel::Quiet,
    logging::LogStage::Init,
    std::source_location::current().file_name(),
    std::source_location::current().function_name(),
    std::source_location::current().line()
  );
  logger.logMessage(
    "Directly writing in here",
    logging::LogLevel::Quiet,
    logging::LogStage::Init,
    std::source_location::current().file_name(),
    std::source_location::current().function_name(),
    std::source_location::current().line()
  );

  logger.logWarning(
    char_msg,
    std::source_location::current().file_name(),
    std::source_location::current().function_name(),
    std::source_location::current().line()
  );
  logger.logWarning(
    str_msg,
    std::source_location::current().file_name(),
    std::source_location::current().function_name(),
    std::source_location::current().line()
  );
  logger.logWarning(
    "Directly writing in here",
    std::source_location::current().file_name(),
    std::source_location::current().function_name(),
    std::source_location::current().line()
  );

  // Now try the message() macros
  message(char_msg);
  message(char_msg, logging::LogLevel::Quiet, logging::LogStage::Undefined);

  message(str_msg);
  message(str_msg, logging::LogLevel::Quiet, logging::LogStage::Undefined);

  message("Directly writing in here");
  message("Directly writing in here", logging::LogLevel::Quiet, logging::LogStage::Undefined);

  // Now try the warning() macros
  warning(char_msg);
  warning(str_msg);
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

    logging::setVerbosity(verb);

    // vary code stages
    for (int stage = stageMin; stage < stageMax; stage++) {

      auto s = static_cast<logging::LogStage>(stage);

      // vary verbosity level of messages
      for (int level = levelMin; level < levelMax; level++) {

        auto l = static_cast<logging::LogLevel>(level);

        bool expect_print = (verb >= level);

        std::stringstream msg;
        constexpr int     w = 12;

        msg << std::setw(w) << "Verbosity=" << std::setw(3) << verb;
        msg << std::setw(w) << " msg level=" << std::setw(3) << level;
        msg << std::setw(w) << " code stage=" << std::setw(3) << stage;
        msg << std::setw(w) << " should print?=" << expect_print;
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
