#pragma once
#include <algorithm>
#include <map>
#include <string>
#include <vector>

#include "Cell.h"
#include "Logging.h"
#include "Parameters.h"

/* Routines to read in IC file */

namespace IO {

  class InputParse {
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

    static std::string getHelpMessage() {
      return helpMessage;
    }

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
    static const std::vector<std::string> requiredArgs;

    static const std::vector<std::string> optionalArgs;

    //! Help message
    static const std::string helpMessage;
  };

} // namespace IO
