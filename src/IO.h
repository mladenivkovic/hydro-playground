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
  struct ParamEntry {

    //! constructors
    ParamEntry() = delete;
    explicit ParamEntry(std::string parameter);
    ParamEntry(std::string parameter, std::string value);

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

  private:
    //! Map holding incoming command line args
    std::map<std::string, std::string> _cmdlargs;

    //! Storage for all read-in configuration parameters
    std::map<std::string, ParamEntry> _config_params;

    //! Param file name. Verified that file exists.
    std::string _paramfile;

    //! Initial Conditions file name. Verified that file exists.
    std::string _icfile;

  public:
    //! Constructor with argc and argv
    InputParse(const int argc, char* argv[]);

    //! Read the param file and fill out the configuration parameters
    void readParamFile(parameters::Parameters& params);

    //! Read the initial conditions file.
    void readICFile(grid::Grid& grid);

    //! Get a pair of name, value from a parameter line
    static std::pair<std::string, std::string> extractParameter(std::string& line);

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
    Float _extractTwoStateVal(
      std::string& line, const char* expected_name, const char* alternative_name = ""
    );

    //! Extract a primitive state from a line of an arbitary-type IC file
    idealGas::PrimitiveState _extractArbitraryICVal(std::string& line, size_t linenr);

    //! Read an IC file with the Two-State format
    void _readTwoStateIC(grid::Grid& grid);

    //! Read an IC file with the arbitrary format
    void _readArbitraryIC(grid::Grid& grid);

    //! convert a parameter from read-in strings to a native type
    template <typename T>
    T _convertParameterString(
      std::string param, ArgType type, bool optional = false, T default_val = 0
    );

  }; // class InputParse


  class OutputWriter {

  private:
    //! output index counter
    size_t _noutputs_written;

    //! time of last output
    Float _t_last_dump;

    //! time of next output
    Float _t_next_dump;

    //! Reference to runtime parameters.
    parameters::Parameters& _params;

    //! Reference to grid to dump.
    grid::Grid& _grid;

    //! Generate the output file name.
    std::string _getOutputFileName();


    /**
     * Write down the simulation time at which output was dumped. Also update
     * time of next dump.
     */
    void setTimeLastOutputWritten(const Float t){
#if DEBUG_LEVEL > 0
      if (t < _t_last_dump){
        error("Current time smaller than time of last dump: t=" +
            std::to_string(t) + ", t_last=" + std::to_string(_t_last_dump));
      }
#endif
      _t_last_dump = t;
      _t_next_dump = t + _params.getDtOut();
    }


    // Getters/setters. Keeping these for included debug checks.
    [[nodiscard]] size_t getNOutputsWritten() const {
      return _noutputs_written;
    }

    void setNOutputsWritten(const size_t n) {
      if (n > 9999) {
        error("Can't write more than 10k outputs; Change output file name format");
      }
      _noutputs_written = n;
    }

    void incNOutputsWritten() {
      setNOutputsWritten(_noutputs_written + 1);
    }


  public:
    explicit OutputWriter(parameters::Parameters& params, grid::Grid& grid):
      _noutputs_written(0),
      _t_last_dump(0.),
      _t_next_dump(0.),
      _params(params),
      _grid(grid)
    {};

    /**
     * Write the current state of the simulation into an output.
     *
     * @param params Simulation runtime parameters
     * @param grid the grid containing your simulation
     * @param t_current the current simulation time
     * @param step the current simulation step
     */
    void dump(Float t_current, size_t step);


    /**
     * Will we write output this step? Call this before actually doing the
     * computations. This function is allowed to modify the time step size,
     * dtCurrent, so that it writes output at the requested time.
     *
     * @param current_step the current simulation step index
     * @param t_current the current simulation time
     * @param dt_current the current simulation time step size
     */
     bool dumpThisStep(size_t current_step, Float t_current, Float& dt_current);
  };
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
  std::string param, ArgType argtype, bool optional, T default_val
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
      msg << "; Using default=" << default_val;
      return default_val;
    }
    error(msg.str());
  }

  ParamEntry& entry = search->second;
  std::string val   = entry.value;
  entry.used        = true;

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
    error(msg.str());
    return default_val;
  }
}

/**
 * explicit specialization for T = std::string.
 * Specialisation needs to be either inlined or in the .cpp file, otherwise
 * you're violating the One Definition Rule.
 */
template <>
inline std::string IO::InputParse::_convertParameterString<std::string>(
  std::string param, ArgType argtype, bool optional, std::string default_val
) {

#if DEBUG_LEVEL > 0
  if (argtype != ArgType::String) {
    std::stringstream msg;
    msg << "Wrong type passed? type=";
    msg << static_cast<int>(argtype);
    error(msg.str());
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
      msg << "; Using default=" << default_val;
      return default_val;
    }
    error(msg.str());
  }

  ParamEntry& entry = search->second;
  std::string val   = entry.value;
  entry.used        = true;

  return val;
}
