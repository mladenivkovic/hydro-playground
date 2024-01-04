#include <iostream>
#include <iomanip>
#include <sstream>

#include "Logging.h"


void printFatLine(void){
  std::cout << "====================================================";
  std::cout << "====================================================";
  std::cout << "====================================================" << std::endl;
}
void printThinLine(void){
  std::cout << "----------------------------------------------------";
  std::cout << "----------------------------------------------------";
  std::cout << "----------------------------------------------------" << std::endl;
}


int main(void) {

  using namespace hydro_playground;


  int levelMin = static_cast<int>(logging::LogLevel::Undefined);
  int levelMax = static_cast<int>(logging::LogLevel::Count);
  int stageMin = static_cast<int>(logging::LogStage::Undefined);
  int stageMax = static_cast<int>(logging::LogStage::Count);

  // First, let's test verbosity level printouts and that stages
  // are named correctly.
  for (int verb = 0; verb < 4; verb++){

    for (int stage = stageMin; stage < stageMax; stage++){
      logging::LogStage s = static_cast<logging::LogStage>(stage);

      for (int level = levelMin; level < levelMax; level++){
        logging::LogLevel l = static_cast<logging::LogLevel>(level);

        std::stringstream msg;

        msg << std::setw(16) << "Verbose=" << std::setw(3) << verb;
        msg << std::setw(16) << " log verb level=" << std::setw(3) << level;
        msg << std::setw(16) << " log code stage=" << std::setw(3) << stage;
        msg << "      | ";

        std::cout << msg.str();

        logging::Log(msg, l, s, verb);
        std::cout << "\n";

      }
      printThinLine();
    }
    printFatLine();
  }


  printFatLine();

  // Now let's test different constructors.

  const char *char_msg = "Const Char message";
  std::string str_msg  = std::string("String message");
  std::stringstream ss_msg;
  ss_msg << "String stream message";


  logging::Log{char_msg};
  logging::Log(char_msg, logging::LogLevel::Quiet, logging::LogStage::Init);
  logging::Log(char_msg, logging::LogLevel::Quiet, logging::LogStage::Init, 1);

  logging::Log{str_msg};
  logging::Log(str_msg, logging::LogLevel::Quiet, logging::LogStage::Init);
  logging::Log(str_msg, logging::LogLevel::Quiet, logging::LogStage::Init, 1);

  logging::Log{ss_msg};
  logging::Log(ss_msg, logging::LogLevel::Quiet, logging::LogStage::Init);
  logging::Log(ss_msg, logging::LogLevel::Quiet, logging::LogStage::Init, 1);

  logging::Log{"Directly writing in here"};
  logging::Log("Directly writing in here");
  logging::Log("Directly writing in here", logging::LogLevel::Quiet, logging::LogStage::Init);
  logging::Log("Directly writing in here", logging::LogLevel::Quiet, logging::LogStage::Init, 1);


  return 0;

}
