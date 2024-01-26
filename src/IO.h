#pragma once
#include "Cell.h"
#include "Parameters.h"
#include "Logging.h"
#include <string>
#include <vector>
#include <algorithm>
#include <map>

/* Routines to read in IC file */

namespace hydro_playground{
  namespace IO{

    class InputParse{
      public:
        //! Deleted default constructor
        InputParse() = delete;

        //! Constructor with argc and argv
        InputParse(int argc, char* argv[]);

        std::string getCommandOption(const std::string& option);

        bool commandOptionExists(const std::string& option);

        //! Drive everything from this function. This is 
        //! what we expose to main.
        void readCommandOptions();

        void readConfigFile();

        void readICFile();

        std::string getHelpMessage() {return helpMessage;}

        //! Use this to return early from main if the input
        //! is invalid. 
        bool inputIsValid();

      private:
        //! Vector to hold incoming command line args
        std::vector<std::string> clArguments;

        //! Vector containing all the valid options we
        //! accept. Iterate over this to check if the 
        //! cmd options we expect to see are present
        //! This is defined in the cpp file.
        //! 
        static const std::vector< std::string > requiredArgs;
        
        static const std::vector< std::string > optionalArgs;

        //! Help message
        static const std::string helpMessage;
    };

  }
}