#pragma once
/* #include <algorithm> */
#include <map>
#include <string>
#include <vector>


/* Routines to read in IC file */

namespace IO {

  /**
   * A container for read in parameters.
   */
  struct configEntry {
    //! constructors
    configEntry() = delete;
    configEntry(std::string parameter);
    configEntry(std::string parameter, std::string value);
    // configEntry(std::string parameter, std::string value, std::string default_val);

    //! Parameter name
    std::string param;

    //! Parameter value
    std::string value;

    // bool optional;

    //! Whether parameter has been used
    bool used;
  };


  namespace internal {

     //! Is the line whitespace only?
     bool isWhitespace(std::string& line);

     //! Is this line a comment?
     bool isComment(std::string& line);
  } // namespace internal


  class InputParse {
  public:
    //! Deleted default constructor
    InputParse() = delete;

    //! Constructor with argc and argv
    InputParse(const int argc, char* argv[]);

    //! Read the config file and fill out the configuration parameters
    void readConfigFile();

    //! Read the initial conditions file.
    void readICFile();

    //! Check whether cmdline args are valid.
    void checkCmdLineArgsAreValid();

  private:

    //! Help message
    static const std::string _helpMessage;

    /**
     * Vector containing all the valid options we accept. Iterate over this to
     * check if the cmd options we expect to see are present This is defined in
     * the cpp file.
     */
    static const std::vector<std::string> _requiredArgs;
    // static const std::vector<std::string> optionalArgs;

    //! Vector to hold incoming command line args
    std::vector<std::string> _clArguments;

    //! Storage for all read-in configuration parameters
    std::map<std::string, configEntry> _config_params;


    //! Config file name. Verified that file exists.
    std::string _configfile;

    //! Initial Conditions file name. Verified that file exists.
    std::string _icfile;

    //! Has a cmdline option been provided?
    bool _commandOptionExists(const std::string& option);

    //! Retreive an option passed by cmdline args
    std::string _getCommandOption(const std::string& option);

    static std::string _getHelpMessage() {
      return _helpMessage;
    }
  }; // class InputParse
} // namespace IO
