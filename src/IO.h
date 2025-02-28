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
    explicit configEntry(std::string parameter);
    configEntry(std::string parameter, std::string value);

    //! Parameter name
    std::string param;

    //! Parameter value
    std::string value;

    //! Whether parameter has been used
    bool used;
  };


  namespace internal {

    //! Is the line whitespace only?
    bool isWhitespace(std::string& line);

    //! Is this line a comment?
    bool isComment(std::string& line);

    //! Remove leading and trailing whitespaces from a string.
    std::string removeWhitespace(std::string& str);

    //! Split a line at an = char. Raise warnings if warn=true and something is amiss.
    std::pair<std::string, std::string> splitEquals(std::string& str, bool warn=false);

    //! Remove trailing comment from a line
    std::string removeTrailingComment(std::string& line);

    //! Get a pair of name,value from a parameter line
    std::pair<std::string, std::string> extractParameter(std::string& line);

    //! Get a string representing something gone wrong in parsing/evaluation
    std::string somethingWrong();

  } // namespace internal


  class InputParse {
  public:
    //! Deleted default constructor
    InputParse() = delete;

    //! Constructor with argc and argv
    InputParse(const int argc, char* argv[]);

    //! Read the config file and fill out the configuration parameters
    void parseConfigFile();

    //! Read the initial conditions file.
    void readICFile();

    //! Check whether cmdline args are valid.
    void checkCmdLineArgsAreValid();

  private:
    //! Help message
    static std::string helpMessage();

    //! Map holding incoming command line args
    std::map<std::string, std::string> _clArguments;

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

  }; // class InputParse
} // namespace IO
