#pragma once
#include "Cell.h"
#include "Parameters.h"
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

        void readCommandOptions();

      private:
        //! Vector to hold incoming command line args
        std::vector<std::string> clArguments;

        //! Vector containing all the valid options we
        //! accept. Iterate over this to check if the 
        //! cmd options we expect to see are present
        //! This is defined in the cpp file.
        //! 
        //! We implement this as a vector of pairs so 
        //! that "--help" and "-h" are equivalent,
        //! for instance
        static const std::vector< std::string > validOptions;
        
        //! Help message
        static const std::string helpMessage;
    };

  }
}