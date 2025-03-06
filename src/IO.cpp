#include "IO.h"

#include <cassert>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>

#include "BoundaryConditions.h"
#include "Logging.h"
#include "Utils.h"


/**
 * configEntry constructors
 */
IO::configEntry::configEntry(std::string parameter):
  param(std::move(parameter)),
  used(false) {};

IO::configEntry::configEntry(std::string parameter, std::string value):
  param(std::move(parameter)),
  value(std::move(value)),
  used(false) {};


/**
 * @brief Constructor. Takes over cmdline args from main(), checks them,
 * and stores them internally.
 */
IO::InputParse::InputParse(const int argc, char* argv[]) {

  constexpr int argc_max = 20;

#if DEBUG_LEVEL > 0
  if (argc > argc_max) {
    std::stringstream msg;
    msg << "Passed " << argc << " arguments, which is higher than the max: " << argc_max
        << ", ignoring everything past it.";
    warning(msg.str())
  }
#endif

  for (int i = 1; i < std::min(argc, argc_max); i++) {

    std::string arg = std::string(argv[i]);

    // Allow flags without values
    if (arg == "-h" or arg == "--help" or arg == "-v" or arg == "--verbose" or arg == "-vv"
        or arg == "--very-verbose") {
      _clArguments.insert(std::make_pair(arg, ""));
      continue;
    }

    // Do we have an arg=value situation?
    auto        split_pair = utils::splitEquals(arg);
    std::string name       = split_pair.first;
    std::string value      = split_pair.second;

    if (name != utils::somethingWrong()) {

      // We have an arg=value situation.
      _clArguments.insert(std::make_pair(name, value));

    } else {

      // value is next arg, if it exists. Otherwise, empty string.
      std::string val;
      if (i + 1 < argc)
        val = argv[i + 1];
      _clArguments.insert(std::make_pair(arg, val));
      i++;
    }
  }

  _checkCmdLineArgsAreValid();
}



/**
 * Read the configuration file and fill out the parameters singleton.
 */
void IO::InputParse::readConfigFile(parameters::Parameters& params) {

  message("Parsing config file.", logging::LogLevel::Verbose);

#if DEBUG_LEVEL > 0
  if (_configfile.size() == 0) {
    error("No config file specified?");
  }
#endif

  // First, we read the config file
  std::string   line;
  std::ifstream conf_ifs(_configfile);

  // Read in line by line
  while (std::getline(conf_ifs, line)) {
    auto        pair  = _extractParameter(line);
    std::string name  = pair.first;
    std::string value = pair.second;
    if (name == "")
      continue;
    if (name == utils::somethingWrong() or value == utils::somethingWrong()) {
      warning("Something wrong with config file line '" + line + "'; skipping it");
    }

    configEntry newEntry = configEntry(name, value);
    _config_params.insert(std::make_pair(name, newEntry));
  }


  // Now we parse each argument

  size_t nstepsLog = _convertParameterString(
    "nstep_log",
    parameters::ArgType::Size_t,
    /*optional=*/true,
    /*defaultVal=*/params.getNstepsLog()
  );
  params.setNstepsLog(nstepsLog);

  size_t nsteps = _convertParameterString(
    "nsteps",
    parameters::ArgType::Size_t,
    /*optional=*/true,
    /*defaultVal=*/params.getNsteps()
  );
  params.setNsteps(nsteps);

  size_t nx = _convertParameterString(
    "nx",
    parameters::ArgType::Size_t,
    /*optional=*/true,
    /*defaultVal=*/params.getNsteps()
  );
  params.setNx(nx);

  int boundary = _convertParameterString(
    "boundary",
    parameters::ArgType::Integer,
    /*optional=*/true,
    /*defaultVal=*/static_cast<int>(params.getBoundaryType())
  );
  params.setBoundaryType(static_cast<BC::BoundaryCondition>(boundary));

  float_t tmax = _convertParameterString(
    "tmax",
    parameters::ArgType::Float,
    /*optional=*/true,
    /*defaultVal=*/params.getTmax()
  );
  params.setTmax(tmax);

  float_t ccfl = _convertParameterString(
    "ccfl",
    parameters::ArgType::Float,
    /*optional=*/true,
    /*defaultVal=*/params.getCcfl()
  );
  params.setCcfl(ccfl);


  // Let me know if we missed something
  _checkUnusedParameters();

  // Mark that we've read in the param file
  params.setParamFileHasBeenRead();
}


/**
 * Read initial conditions.
 */
void IO::InputParse::readICFile(grid::Grid& grid, const parameters::Parameters& params) {

  std::string filename = _icfile;

  if (_icIsTwoState()){
    _readTwoStateIC(grid, params);
  } else {
    _readArbitraryIC(grid, params);
  }

  message("Finished reading ICs.", logging::LogLevel::Verbose);
}


/**
 * Extract parameters from a line following the format
 * ```
 *  name = value // possible comment
 * ```
 *
 * Returns a pair of strings, the name (before = sign) and value (after =
 * sign). If both these strings are empty, we're skipping this line
 * deliberately. This could be because it's a comment or empty.
 * If something goes wrong during the partising, name and value will be
 * utils::somethingWrong().
 */
std::pair<std::string, std::string> IO::InputParse::_extractParameter(std::string& line) {

  std::string name;
  std::string value;

  if (utils::isComment(line))
    return std::make_pair(name, value);
  if (utils::isWhitespace(line))
    return std::make_pair(name, value);

  std::string nocomment = utils::removeTrailingComment(line);
  auto        pair      = utils::splitEquals(nocomment, /*warn=*/true);
  name                  = pair.first;
  value                 = pair.second;

  return std::make_pair(name, value);
}


/**
 * Returns the help message.
 */
std::string IO::InputParse::_helpMessage() {

  std::stringstream msg;
  msg << "This is the hydro code help message.\n\nUsage: \n\n";
  msg << "Default run:\n  ./hydro --config-file <config-file> --ic-file <ic-file>\n";
  msg << "or:\n  ./hydro --config-file=<config-file> --ic-file=<ic-file>\n\n";
  msg << "      <config-file>: file containing your run parameter configuration.\n";
  msg << "                     See README for details.\n";
  msg << "      <ic-file>:     file containing your initial conditions.\n";
  msg << "                     See README for details.\n\n";
  msg << "Get this help message:\n  ./hydro -h\n  ./hydro --help\n\n";
  msg << "Optional flags:\n";
  msg << "  -v,--verbose:         Be talkative.\n";
  msg << "  -vv,--very-verbose:   Be very talkative (intended for debugging).\n";

  return msg.str();
}


/**
 * Verify that the provided command line arguments are valid.
 * May exit if help flag was raised.
 * Sets the verbosity level if it was provided.
 */
void IO::InputParse::_checkCmdLineArgsAreValid() {

  // If help is requested, print help and exit.
  if (_commandOptionExists("-h") or _commandOptionExists("--help")) {
    message(_helpMessage(), logging::LogLevel::Quiet);
    std::exit(0);
  }

  if (_commandOptionExists("-v") or _commandOptionExists("--verbose")) {
    logging::Log::setVerbosity(logging::LogLevel::Verbose);
  }

  if (_commandOptionExists("-vv") or _commandOptionExists("--very-verbose")) {
    logging::Log::setVerbosity(logging::LogLevel::Debug);
  }


  // Vector containing all the valid options we accept. Iterate over this to
  // check if the cmd options we expect to see are present This is defined in
  // the cpp file.
  const std::vector<std::string> _requiredArgs = {
    "--config-file",
    "--ic-file",
  };

  // check all the required options
  for (const auto& opt : _requiredArgs) {
    if (not _commandOptionExists(opt)) {
      std::stringstream msg;
      msg << "missing option: " << opt;
      error(msg);
    }
  }

  // Check whether the files we should have are fine
  std::string icfile = _getCommandOption("--ic-file");
  if (not(utils::fileExists(icfile))) {
    std::stringstream msg;
    msg << "Provided initial conditions file '" << icfile << "' doesn't exist.";
    error(msg.str());
  } else {
    // Store it.
    message("Found IC file " + icfile, logging::LogLevel::Debug);
    _icfile = icfile;
  }

  std::string configfile = _getCommandOption("--config-file");
  if (not(utils::fileExists(configfile))) {
    std::stringstream msg;
    msg << "Provided parameter file '" << configfile << "' doesn't exist.";
    error(msg.str());
  } else {
    // Store it.
    _configfile = configfile;
    message("Found config file " + configfile, logging::LogLevel::Debug);
  }
}


/**
 * Has a cmdline option been provided?
 */
bool IO::InputParse::_commandOptionExists(const std::string& option) {

  auto search = _clArguments.find(option);
  return (search != _clArguments.end());
}


/**
 * Get the value provided by the command option @param option.
 */
std::string IO::InputParse::_getCommandOption(const std::string& option) {

  auto search = _clArguments.find(option);
  if (search == _clArguments.end()) {
    warning("No option '" + option + "' available");
    const std::string emptyString;
    return emptyString;
  }

  return search->second;
}

/**
 * Find and list unused parameters
 */
void IO::InputParse::_checkUnusedParameters() {

  for (auto& _config_param : _config_params) {
    configEntry& entry = _config_param.second;
    if (not entry.used) {
      std::stringstream msg;
      msg << "Unused parameter: " << entry.param << "=" << entry.value;
      warning(msg);
    }
#if DEBUG_LEVEL > 1
    else {
      std::stringstream msg;
      msg << "USED parameter: " << entry.param << "=" << entry.value;
      warning(msg);
    }
#endif
  }
}


/**
 * Do we have a two-state format for initial conditions?
 */
bool IO::InputParse::_icIsTwoState(){

  std::string   line;
  std::ifstream conf_ifs(_icfile);
  while (std::getline(conf_ifs, line)) {

    if (utils::isComment(line))
      continue;

    std::string nocomment = utils::removeTrailingComment(line);
    auto        pair      = utils::splitEquals(nocomment, /*warn=*/true);
    std::string name      = pair.first;
    std::string value     = pair.second;

    if (name != "filetype"){
      std::stringstream msg;
      msg << "Invalid IC file type: first non-comment line must be ";
      msg << "`filetype = [two-state,arbitrary]`\n";
      msg << "current line is: `" << line << "`";
      error(msg);
    }

    bool out = false;
    if (value == "two-state") {
      out = true;
    } else if (value == "arbitrary"){
      out = false;
    } else {
      std::stringstream msg;
      msg << "Invalid IC file type specification:";
      msg << "`filetype` must be either `two-state` or `arbitrary`\n";
      msg << "I found: `" << value << "`";
      error(msg);
    }

    return out;
  }

  return false;
}


//! Read an IC file with the Two-State format
void IO::InputParse::_readTwoStateIC(grid::Grid& grid, const parameters::Parameters& params){}


//! Read an IC file with the arbitrary format
void IO::InputParse::_readArbitraryIC(grid::Grid& grid, const parameters::Parameters& params){}


