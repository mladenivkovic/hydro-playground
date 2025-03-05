#include "IO.h"

#include <algorithm> // std::find
#include <cassert>
#include <cctype>
#include <filesystem> // std::filesytem::exists
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#include "Cell.h"
#include "Grid.h"
#include "Logging.h"
#include "Parameters.h"

namespace IO {
  using std::string;


  namespace internal {

    /**
     * String signifying something's gone wrong.
     */
    std::string somethingWrong() {
      return std::string("__something_wrong__");
    }


    /**
     * Scan through the line buffer. If we see any character that isn't `\n`,
     * space, EOF or null then return false
     */
    bool isWhitespace(std::string& line) {
      for (const auto s : line) {
        if ((std::isspace(s) == 0) and (s != EOF) and static_cast<bool>(s)) {
          return false;
        }
      }
      return true;
    }


    /**
     * Scan past all the spaces, if the first non-space chars you see are // or
     * / *, then return true.
     */
    bool isComment(std::string& line) {

      for (auto s = line.cbegin(); s != line.cend(); s++) {
        if (std::isspace(*s) != 0) {
          // skip leading spaces
          continue;
        }
        if (*s == '/') {
          auto next = s + 1;
          return ((next != line.cend()) and ((*next == '/') or (*next == '*')));
        }
        return false;
      }
      return false;
    }


    /**
     * Remove leading and trailing whitespaces from a string.
     */
    std::string removeWhitespace(std::string& str) {

      if (str.size() == 0){
        return str;
      }
      if (str.size() == 1){
        if (str == " " or str == "\t" or str == "\n" or str == "\r" or str == "\f" or str == "\v"){
          return "";
        }
        return str;
      }

      std::string ltrim;
      std::string rtrim;

      size_t first = str.find_first_not_of(" \t\n\r\f\v");
      if (first < str.size()) {
        ltrim = str.erase(0, first);
      } else {
        ltrim = str;
      }

      size_t last = ltrim.find_last_not_of(" \t\n\r\f\v");
      if (last < str.size()) {
        // last + 1: start at index after last non-whitespace
        rtrim = ltrim.erase(last + 1, ltrim.size() - last);
      } else {
        rtrim = ltrim;
      }

      return rtrim;
    }


    /**
     * Split a string at an equality character.
     * Returns a pair of strings of the content (before, after)
     * the equality sign.
     * Returns internal::somethingWrong() if there is no or more than
     * one equality sign in the string.
     */
    std::pair<std::string, std::string> splitEquals(std::string& str, bool warn) {

      // Find where the equals sign is
      size_t equals_ind = 0;
      int    count      = 0;
      for (size_t i = 0; i < str.size(); i++) {
        if (str[i] == '=') {
          count++;
          if (equals_ind == 0)
            equals_ind = i;
        }
      }

      if (count > 1) {
        if (warn) warning("Got more than 1 equality sign in line '" + str + "'");
        return std::make_pair(internal::somethingWrong(), internal::somethingWrong());
      }

      if (equals_ind == 0) {
        if (warn) warning("No equality sign or no var name in line '" + str + "'");
        return std::make_pair(internal::somethingWrong(), internal::somethingWrong());
      }

      if (equals_ind == str.size()) {
        if (warn) warning("No var value in line '" + str + "'");
        return std::make_pair(internal::somethingWrong(), internal::somethingWrong());
      }

      std::string name = str.substr(0, equals_ind);
      std::string val  = str.substr(equals_ind + 1, str.size() - equals_ind - 1);

      return std::make_pair(removeWhitespace(name), removeWhitespace(val));
    }


    /**
     * Remove trailing comment from line.
     * Comment is / * or //.
     */
    std::string removeTrailingComment(std::string& line) {

      size_t comm1 = line.find("//");
      size_t comm2 = line.find("/*");
      // get the earlier of both
      size_t comm = std::min(comm1, comm2);
      // make sure we found at least one
      comm = std::min(comm, line.size());

      return line.substr(0, comm);
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
     * internal::somethingWrong().
     */
    std::pair<std::string, std::string> extractParameter(std::string& line) {

      std::string name;
      std::string value;

      if (isComment(line))
        return std::make_pair(name, value);
      if (isWhitespace(line))
        return std::make_pair(name, value);

      std::string nocomment = internal::removeTrailingComment(line);
      auto        pair      = internal::splitEquals(nocomment, /*warn=*/true);
      name                  = pair.first;
      value                 = pair.second;

      return std::make_pair(name, value);
    }


    /**
     * Does a file exist?
     */
    bool fileExists(const std::string& filename) {
      return std::filesystem::exists(filename);
    }


    /**
     * Convert value string to integer.
     * Do some additional sanity checks too.
     * @TODO: there probably is a better way. This does the trick for now.
     */
    size_t string2size_t(std::string& val){
      return static_cast<size_t>(string2int(val));
    }


    /**
     * Convert value string to integer.
     * Do some additional sanity checks too.
     */
    int string2int(std::string& val){
      std::string v = removeWhitespace(val);
      if (v.size() == 0){
        std::stringstream msg;
        msg << "Invalid string to convert to int: '" << val << "'";
        error(msg);
      }

      // todo: error catching
      int out = std::stoi(v);
      return out;
    }



    /**
     * Convert value string to float/double.
     * Do some additional sanity checks too.
     */
    float_t string2float(std::string& val){

      std::string v = removeWhitespace(val);
      if (v.size() == 0){
        std::stringstream msg;
        msg << "Invalid string to convert to int: '" << val << "'";
        error(msg);
      }

      // todo: error catching
      float_t out = (float_t) std::stof(v);
      return out;
    }



    /**
     * Convert value string to integer.
     * Do some additional sanity checks too.
     */
    bool string2bool(std::string& val){

      std::string v = removeWhitespace(val);

      if ((v == "true") or (v == "True") or (v == "TRUE") or (v == "1")) {
        return true;
      }
      if ((v == "false") or (v == "False") or (v == "FALSE") or (v == "0")) {
        return false;
      }

      std::stringstream msg;
      msg << "Invalid bool string '" << val << "'";
      error(msg);
      return false;
    }



    /**
     * "Convert" value string to string.
     * Basically just do some additional sanity checks.
     */
    std::string string2string(std::string val){

      std::string v = removeWhitespace(val);
      if (v.size() == 0){
        std::stringstream msg;
        msg << "Suspicious string: '" << val << "'";
        warning(msg);
      }

      return v;
    }


  } // namespace internal


  /**
   * configEntry constructors
   */
  configEntry::configEntry(std::string parameter):
    param(std::move(parameter)),
    used(false) {};
  configEntry::configEntry(std::string parameter, std::string value):
    param(std::move(parameter)),
    value(std::move(value)),
    used(false) {};


  /**
   * Returns the help message.
   */
  std::string InputParse::_helpMessage() {

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
   * @brief Constructor. Takes over cmdline args from main(), checks them,
   * and stores them internally.
   */
  InputParse::InputParse(const int argc, char* argv[]) {

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
      if (arg == "-h" or arg == "--help" or arg == "-v" or arg =="--verbose" or arg == "-vv" or arg =="--very-verbose"){
        _clArguments.insert(std::make_pair(arg, ""));
        continue;
      }

      // Do we have an arg=value situation?
      auto split_pair = internal::splitEquals(arg);
      std::string name = split_pair.first;
      std::string value = split_pair.second;

      if (name != internal::somethingWrong()){

        // We have an arg=value situation.
        _clArguments.insert(std::make_pair(name, value));

      } else {

        // value is next arg, if it exists. Otherwise, empty string.
        std::string val;
        if (i+1 < argc) val = argv[i+1];
        _clArguments.insert(std::make_pair(arg, val));
        i++;

      }
    }

    _checkCmdLineArgsAreValid();
  }


  /**
   * Get the value provided by the command option @param option.
   */
  std::string InputParse::_getCommandOption(const std::string& option) {

    auto search = _clArguments.find(option);
    if (search == _clArguments.end()){
      warning("No option '" + option + "' available");
      const std::string emptyString;
      return emptyString;
    }

    return search->second;
  }


  void InputParse::_checkUnusedParameters(){

    for (auto & _config_param : _config_params){
      configEntry& entry = _config_param.second;
      if (not entry.used){
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
   * Has a cmdline option been provided?
   */
  bool InputParse::_commandOptionExists(const std::string& option) {

    auto search = _clArguments.find(option);
    return (search != _clArguments.end());
  }


  /**
   * Verify that the provided command line arguments are valid.
   * May exit if help flag was raised.
   * Sets the verbosity level if it was provided.
   */
  void InputParse::_checkCmdLineArgsAreValid() {

    // If help is requested, print help and exit.
    if (_commandOptionExists("-h") or _commandOptionExists("--help")) {
      message(_helpMessage(), logging::LogLevel::Quiet);
      std::exit(0);
    }

    if (_commandOptionExists("-v") or _commandOptionExists("--verbose")) {
      logging::Log::setVerbosity(logging::LogLevel::Verbose);
    }

    if (_commandOptionExists("-vv") or _commandOptionExists("--very-verbose")){
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
      if (not _commandOptionExists(opt)){
        std::stringstream msg;
        msg << "missing option: " << opt;
        error(msg);
      }
    }

    // Check whether the files we should have are fine
    std::string icfile = _getCommandOption("--ic-file");
    if (not(internal::fileExists(icfile))) {
      std::stringstream msg;
      msg << "Provided initial conditions file '" << icfile << "' doesn't exist.";
      error(msg.str());
    } else {
      // Store it.
      message("Found IC file " + icfile, logging::LogLevel::Debug);
      _icfile = icfile;
    }

    std::string configfile = _getCommandOption("--config-file");
    if (not(internal::fileExists(configfile))) {
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
   * Read the configuration file and fill out the parameters singleton.
   */
  void InputParse::parseConfigFile() {

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
      auto        pair  = internal::extractParameter(line);
      std::string name  = pair.first;
      std::string value = pair.second;
      if (name == "")
        continue;
      if (name == internal::somethingWrong() or value == internal::somethingWrong()) {
        warning("Something wrong with config file line '" + line + "'; skipping it");
      }

      configEntry newEntry = configEntry(name, value);
      _config_params.insert(std::make_pair(name, newEntry));
    }


    // Now we parse each argument

    auto pars = parameters::Parameters::Instance;

    size_t nstepsLog = _convertParameterString(
        "nstep_log",
        parameters::ArgType::Size_t,
        /*optional=*/true,
        /*defaultVal=*/pars.getNstepsLog()
        );
    pars.setNstepsLog(nstepsLog);

    size_t nsteps = _convertParameterString(
        "nsteps",
        parameters::ArgType::Size_t,
        /*optional=*/true,
        /*defaultVal=*/pars.getNsteps()
        );
    pars.setNsteps(nsteps);

    size_t nx = _convertParameterString(
        "nx",
        parameters::ArgType::Size_t,
        /*optional=*/true,
        /*defaultVal=*/pars.getNsteps()
        );
    pars.setNx(nx);

    int boundary = _convertParameterString(
        "boundary",
        parameters::ArgType::Integer,
        /*optional=*/true,
        /*defaultVal=*/static_cast<int>(pars.getBoundaryType())
        );
    pars.setBoundaryType(static_cast<parameters::BoundaryCondition>(boundary));

    float_t tmax = _convertParameterString(
        "tmax",
        parameters::ArgType::Float,
        /*optional=*/true,
        /*defaultVal=*/pars.getTmax()
        );
    pars.setTmax(tmax);

    float_t ccfl = _convertParameterString(
        "ccfl",
        parameters::ArgType::Float,
        /*optional=*/true,
        /*defaultVal=*/pars.getCcfl()
        );
    pars.setCcfl(ccfl);


    // Let me know if we missed something
    _checkUnusedParameters();
  }


  /*
  This method is a bit of a mess
  */
  void InputParse::readICFile() {
    // std::string filename = _getCommandOption("--ic-file");
    // FILE*       icfile   = fopen(filename.c_str(), "rb");
    // if (icfile == nullptr)
    //   throw std::runtime_error("Invalid IC File!\n");
    //
    // // Let's find how many bytes we have in the file
    // fseek(icfile, 0, SEEK_END);
    // auto bytesToRead = ftell(icfile);
    // // seek back to the start...
    // fseek(icfile, 0, SEEK_SET);

    // Buffer to fill with data from the file
    // char lineBuffer[internal::lineLength] = {0};
    // Pointer to move across the buffer. We
    // use this to fill the buffer with data
    // char* lineptr(lineBuffer);

    // lambda to advance our file pointer. Would do this with
    // aux function but i wanna keep the pointers in the
    // stack frame
    // auto readUntil = [&](const char& ch) {
    // internal::resetBuffer(lineBuffer);
    // reset pointer to start of buffer
    // lineptr = lineBuffer;
    // while ((*lineptr = fgetc(icfile)) != EOF) {
    // Decrement the number of bytes we have to read...
    // bytesToRead--;
    // if (*lineptr == ch)
    // return true;
    // lineptr++;
    // }
    // return false;
    // };

    // // define another lambda to fetch a float
    // // value that falls between two pointers
    // char* ptr0;
    // char* ptr1;
    // auto  readVal = [&]() {
    //   // bring ptr0 up to speed with ptr0
    //   ptr0 = ptr1;
    //   // find the start of the number
    //   while (*ptr0 == ' ')
    //     ptr0++;
    //   ptr1 = ptr0;
    //   while (*ptr1 != ' ' and *ptr1 != '\n') {
    //     ptr1++;
    //   }
    //   if (std::distance(ptr0, ptr1) < 2)
    //     throw std::runtime_error("Invalid line!\n");
    //   return strtod(ptr0, &ptr1);
    // };
    //
    // // read filetype
    // readUntil('=');
    // readUntil('\n');
    // message(lineBuffer);
    //
    // // check ndim matches parameters dims
    // // read ndims
    // readUntil('=');
    // readUntil('\n');
    // int dims = strtol(lineBuffer, &lineptr, 10);
    // // parameters::Parameters::Instance.setDims(dims);
    // message(lineBuffer);
    //
    // // check nx matches params
    // readUntil('=');
    // readUntil('\n');
    // int nx = strtol(lineBuffer, &lineptr, 10);
    // if (Dimensions == 2)
    //   nx = nx * nx;
    // int valuesFetched = 0;
    //
    // message("setting nx to ");
    // message(lineBuffer);
    //
    // loop over remaining lines and store results
    // int valsToFetchPerLine = 2 + dims;

    /*
    Warning - make sure we don't place these
    outside of the boundary!

    start at bc etc!

    bytesToRead is decremented inside readUntil
    */
    // while (bytesToRead > 0) {
    //   // fill the line buffer with some data
    //   readUntil('\n');
    //   if (internal::lineIsInvalid(lineBuffer))
    //     continue;
    //
    //   std::vector<float> initialValuesToPassOver(valsToFetchPerLine, 0);
    //   // fetch values from the current line buffer
    //   for (int i = 0; i < valsToFetchPerLine; i++) {
    //     initialValuesToPassOver[i] = readVal();
    //   }
    //   // reset the pointers we use to the start of the buffer
    //   ptr0 = lineBuffer;
    //   ptr1 = lineBuffer;
    //
    //   // Send these off to the grid - handle indexing in the grid class
    //   grid::Grid::Instance.setInitialConditions(valuesFetched, initialValuesToPassOver);
    //
    //   valuesFetched++;
    // }

    // validation - does valuesFetched match nx?
    // assert(valuesFetched == nx);
    //
    //
    // fclose(icfile);
  }

} // namespace IO
