#pragma once
/* #include <algorithm> */
/* #include <map> */
#include <string>
#include <vector>


/* Routines to read in IC file */

namespace IO {

  class InputParse {
  public:
    //! Deleted default constructor
    InputParse() = delete;

    //! Constructor with argc and argv
    InputParse(const int argc, char* argv[]);

    std::string getCommandOption(const std::string& option);

    //! Has a cmdline option been provided?
    bool commandOptionExists(const std::string& option);

    //! Read the config file and fill out the configuration parameters
    void readConfigFile();

    void readICFile();

    static std::string getHelpMessage() {
      return helpMessage;
    }

    /**
     * Check whether cmdline args are valid.
     */
    void checkCmdLineArgsAreValid();

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
