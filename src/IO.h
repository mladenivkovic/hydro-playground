/**
 * @file IO.h
 * Classes to deal with Reading/Writing files
 */

#pragma once

#include <map>
#include <sstream>
#include <string>

#include "Grid.h"
#include "Parameters.h"
#include "Utils.h"

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


  //! parameter file argument types
  enum class ArgType {
    Integer = 0,
    Size_t  = 1,
    Float   = 2,
    Bool    = 3,
    String  = 4
  };


  /**
   * @brief Class parsing input arguments and files.
   */
  class InputParse {
    // private vars
  private:
    //! Map holding incoming command line args
    std::map<std::string, std::string> _clArguments;

    //! Storage for all read-in configuration parameters
    std::map<std::string, configEntry> _config_params;

    //! Config file name. Verified that file exists.
    std::string _configfile;

    //! Initial Conditions file name. Verified that file exists.
    std::string _icfile;

  public:

    //! Constructor with argc and argv
    InputParse(const int argc, char* argv[]);

    //! Read the config file and fill out the configuration parameters
    void readConfigFile(parameters::Parameters& params);

    //! Read the initial conditions file.
    void readICFile(grid::Grid& grid);

    //! Get a pair of name, value from a parameter line
    static std::pair<std::string, std::string> _extractParameter(std::string& line);

  private:
    //! Help message
    static std::string _helpMessage();

    //! Check whether cmdline args are valid.
    void _checkCmdLineArgsAreValid();

    //! Has a cmdline option been provided?
    bool _commandOptionExists(const std::string& option);

    //! Retreive an option passed by cmdline args
    std::string _getCommandOption(const std::string& option);

    //! Check whether some of read in parameters are unused
    void _checkUnusedParameters();

    //! Do we have a two-state format for initial conditions?
    bool _icIsTwoState();

    //! Get a pair of name, value from a parameter line
    float_t _extractTwoStateVal(std::string& line, std::string expectedName);

    //! Read an IC file with the Two-State format
    void _readTwoStateIC(grid::Grid& grid);

    //! Read an IC file with the arbitrary format
    void _readArbitraryIC(grid::Grid& grid);

    //! convert a parameter from read-in strings to a native type
    template <typename T>
    T _convertParameterString(
      std::string param, ArgType type, bool optional = false, T defaultVal = 0
    );

  }; // class InputParse
} // namespace IO


// --------------------------------------------------------
// Definitions
// --------------------------------------------------------


/**
 * Convert a provided parameter from a string value into a native type
 *
 * @param param string containing the parameter name.
 * @param type type of the argument to convert into.
 * @param optional whether this argument is optional. If true, should the
 * parameter not be available in the internal "database", the defaultVal will
 * be returned.
 * @param defaultVal the default value to use if this parameter
 * has not been explicitly provided.
 */
template <typename T>
T IO::InputParse::_convertParameterString(
  std::string param, ArgType argtype, bool optional, T defaultVal
) {

#if DEBUG_LEVEL > 0
  if (argtype == ArgType::String)
    error("Got type string, should be using its own specialisation");
#endif

  // Grab parameter from storage
  auto search = _config_params.find(param);
  if (search == _config_params.end()) {
    // we didn't find it.
    std::stringstream msg;
    msg << "No parameter '" << param << "' provided";
    if (optional) {
      // just raise warning, not error
      msg << "; Using default=" << defaultVal;
      return defaultVal;
    }
    error(msg);
  }

  configEntry& entry = search->second;
  std::string  val   = entry.value;
  entry.used         = true;

  switch (argtype) {
  case ArgType::Integer:
    return static_cast<T>(utils::string2int(val));
    break;
  case ArgType::Size_t:
    return static_cast<T>(utils::string2size_t(val));
    break;
  case ArgType::Float:
    return static_cast<T>(utils::string2float(val));
    break;
  case ArgType::Bool:
    return static_cast<T>(utils::string2bool(val));
    break;
    // case ArgType::String:
    //   return static_cast<T>(val);
    //   break;
  default:
    std::stringstream msg;
    msg << "Unknown type " << static_cast<int>(argtype);
    error(msg);
    return defaultVal;
  }
}

/**
 * explicit specialization for T = std::string.
 * Specialisation needs to be either inlined or in the .cpp file, otherwise
 * you're violating the One Definition Rule.
 */
template <>
inline std::string IO::InputParse::_convertParameterString<std::string>(
  std::string param, ArgType argtype, bool optional, std::string defaultVal
) {

#if DEBUG_LEVEL > 0
  if (argtype != ArgType::String) {
    std::stringstream msg;
    msg << "Wrong type passed? type=";
    msg << static_cast<int>(argtype);
    error(msg);
  }
#endif

  // Grab parameter from storage
  auto search = _config_params.find(param);
  if (search == _config_params.end()) {
    // we didn't find it.
    std::stringstream msg;
    msg << "No parameter '" << param << "' provided";
    if (optional) {
      // just raise warning, not error
      msg << "; Using default=" << defaultVal;
      return defaultVal;
    }
    error(msg);
  }

  configEntry& entry = search->second;
  std::string  val   = entry.value;
  entry.used         = true;

  return val;
}
