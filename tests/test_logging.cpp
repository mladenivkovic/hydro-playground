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


  int levelMin = static_cast<int>(logging::LogLevelUndefined);
  int levelMax = static_cast<int>(logging::LogLevelCount);
  int stageMin = static_cast<int>(logging::LogStageUndefined);
  int stageMax = static_cast<int>(logging::LogStageCount);

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

        logging::Log(msg.str(), l, s, verb);
        std::cout << "\n";

      }
      printThinLine();
    }
    printFatLine();
  }

  return 0;

}
