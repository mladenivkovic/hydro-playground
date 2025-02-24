#include <algorithm> // std::find
#include <cassert>
#include <cctype>
#include <filesystem> // std::filesytem::exists
#include <fstream>
#include <sstream>
#include <string>

// #include <sstream>

#include "Cell.h"
#include "Grid.h"
#include "IO.h"
#include "Logging.h"
#include "Parameters.h"


#include <iostream>
#include <utility>

namespace IO {

  namespace internal {

    // I don't see why we need more than this many args
    static constexpr int argc_max   = 20;
    static constexpr int lineLength = 256;

    // TODO: do we need this?
    template <typename T>
    void resetBuffer(T* buffer, const size_t len = lineLength) {
      for (size_t i = 0; i < len; i++) {
        buffer[i] = 0;
      }
    }


    /**
     * Scan through the line buffer. If we see any character that isn't `\n`,
     * space, EOF or null then return false
     */
     bool isWhitespace(std::string& line) {
      for (const auto s: line){
        if ((std::isspace(s) == 0) and (s != EOF) and static_cast<bool>(s) ){
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

      for (auto s = line.cbegin(); s != line.cend(); s++){
        if (std::isspace(*s) != 0) {
          // skip leading spaces
          continue;
        }
        if (*s == '/'){
          auto next = s+1;
          return ((next != line.cend()) and ((*next == '/') or (*next == '*')));
        }
        return false;
      }
      return true;
    }


    bool lineIsInvalid(std::string line) {
      bool output = false;
      output |= isWhitespace(line);
      output |= isComment(line);
      // room for other stuff...

      return output;
    }


    /**
     * Does a file exist?
     */
    bool fileExists(const std::string& filename){
      return std::filesystem::exists(filename);
    }

  } // namespace internal


  /**
   * configEntry constructors
   */
  configEntry::configEntry(std::string parameter) :
    param(std::move(parameter)),
    value(""),
    // optional(false),
    used(false) { };
  configEntry::configEntry(std::string parameter, std::string value) :
    param(std::move(parameter)),
    value(std::move(value)),
    // optional(false),
    used(false) { };



  const std::vector<std::string> InputParse::_requiredArgs = {
    "--config-file",
    "--ic-file",
  };


  const std::string InputParse::_helpMessage =
    std::string("This is the hydro code help message.\n\nUsage: \n\n") +
    "Default run:\n  ./hydro --config-file <config-file> --ic-file <ic-file>\n" +
    "    <config-file>: file containing your run parameter configuration. See README for details.\n"+
    "    <ic-file>: file containing your initial conditions. See README for details.\n\n"+
    "Get this help message:\n  ./hydro -h\n  ./hydro --help\n"
    ;


  /**
   * Constructor
   */
  InputParse::InputParse(const int argc, char* argv[]) {

#if DEBUG_LEVEL > 0
    if (argc > internal::argc_max) {
      std::stringstream msg;
      msg << "Passed " << argc << " arguments, which is higher than the max: " << internal::argc_max << ", ignoring everything past it.";
      warning(msg.str())
    }
#endif

    // push all the argv into the vector to hold them
    // start at 1; ignore the binary name
    for (int i = 1; i < std::min(argc, internal::argc_max); i++) {
      _clArguments.emplace_back(argv[i]);
    }
  }


  /**
   * Get the value provided by the command option @param option.
   */
  std::string InputParse::_getCommandOption(const std::string& option) {
    auto iter = std::find(_clArguments.begin(), _clArguments.end(), option);
    // make sure we aren't at the end, and that there's something to read...
    // mind the sneaky increment in the "if" clause... That's how we get
    // the actual value, and not the cmdline option itself.
    if (iter != _clArguments.end() and ++iter != _clArguments.end()) {
      return *iter;
    }

    // no luck. return the empty string
    static const std::string emptyString("");
    return emptyString;
  }


  /**
   * Has a cmdline option been provided?
   */
  bool InputParse::_commandOptionExists(const std::string& option) {
    auto iter = std::find(_clArguments.begin(), _clArguments.end(), option);
    return (iter != _clArguments.end());
  }


  /**
   * Verify that the provided command line arguments are valid.
   */
  void InputParse::checkCmdLineArgsAreValid() {

    // If help is requested, print help and exit.
    if (_commandOptionExists("-h") or _commandOptionExists("--help")) {
      message(_helpMessage, logging::LogStage::Init);
      std::exit(0);
    }

    // check all the required options
    for (const auto& opt : _requiredArgs) {
      if (std::find(_clArguments.begin(), _clArguments.end(), opt) == _clArguments.end()) {
        std::string msg = "missing option: " + opt;
        message(msg, logging::LogStage::Init);
      }
    }

    // Check whether the files we should have are fine
    std::string icfile = _getCommandOption("--ic-file");
    if (not (internal::fileExists(icfile))){
      std::stringstream msg;
      msg << "Provided initial conditions file '" << icfile << "' doesn't exist.";
      error(msg.str());
    } else {
      // Store it.
      _icfile = icfile;
    }

    std::string configfile = _getCommandOption("--config-file");
    if (not (internal::fileExists(configfile))){
      std::stringstream msg;
      msg << "Provided parameter file '" << configfile << "' doesn't exist.";
      error(msg.str());
    } else {
      _configfile = configfile;
    }
  }


  /**
   * Read the configuration file and fill out the parameters singleton.
   */
  void InputParse::readConfigFile(){

#if DEBUG_LEVEL > 0
    if (_configfile.size() == 0 ){
      error("No config file specified?")
    }
#endif

    std::string line;
    std::ifstream conf_ifs(_configfile);

    // Read in line by line
    while (std::getline(conf_ifs, line)) {
      if (internal::lineIsInvalid(line)) {
        continue;
      }
    }

  }



  /*
  This method is a bit of a mess
  */
  void InputParse::readICFile() {
    std::string filename = _getCommandOption("--ic-file");
    FILE*       icfile   = fopen(filename.c_str(), "rb");
    if (icfile == nullptr)
      throw std::runtime_error("Invalid IC File!\n");

    // Let's find how many bytes we have in the file
    fseek(icfile, 0, SEEK_END);
    auto bytesToRead = ftell(icfile);
    // seek back to the start...
    fseek(icfile, 0, SEEK_SET);

    // Buffer to fill with data from the file
    char lineBuffer[internal::lineLength] = {0};
    // Pointer to move across the buffer. We
    // use this to fill the buffer with data
    char* lineptr(lineBuffer);

    // lambda to advance our file pointer. Would do this with
    // aux function but i wanna keep the pointers in the
    // stack frame
    auto readUntil = [&](const char& ch) {
      internal::resetBuffer(lineBuffer);
      // reset pointer to start of buffer
      lineptr = lineBuffer;
      while ((*lineptr = fgetc(icfile)) != EOF) {
        // Decrement the number of bytes we have to read...
        bytesToRead--;
        if (*lineptr == ch)
          return true;
        lineptr++;
      }
      return false;
    };

    // define another lambda to fetch a float
    // value that falls between two pointers
    char* ptr0;
    char* ptr1;
    auto  readVal = [&]() {
      // bring ptr0 up to speed with ptr0
      ptr0 = ptr1;
      // find the start of the number
      while (*ptr0 == ' ')
        ptr0++;
      ptr1 = ptr0;
      while (*ptr1 != ' ' and *ptr1 != '\n') {
        ptr1++;
      }
      if (std::distance(ptr0, ptr1) < 2)
        throw std::runtime_error("Invalid line!\n");
      return strtod(ptr0, &ptr1);
    };

    // read filetype
    readUntil('=');
    readUntil('\n');
    message(lineBuffer);

    // check ndim matches parameters dims
    // read ndims
    readUntil('=');
    readUntil('\n');
    int dims = strtol(lineBuffer, &lineptr, 10);
    // parameters::Parameters::Instance.setDims(dims);
    message(lineBuffer);

    // check nx matches params
    readUntil('=');
    readUntil('\n');
    int nx = strtol(lineBuffer, &lineptr, 10);
    if (Dimensions == 2)
      nx = nx * nx;
    int valuesFetched = 0;

    message("setting nx to ");
    message(lineBuffer);

    // loop over remaining lines and store results
    int valsToFetchPerLine = 2 + dims;

    /*
    Warning - make sure we don't place these
    outside of the boundary!

    start at bc etc!

    bytesToRead is decremented inside readUntil
    */
    while (bytesToRead > 0) {
      // fill the line buffer with some data
      readUntil('\n');
      if (internal::lineIsInvalid(lineBuffer))
        continue;

      std::vector<float> initialValuesToPassOver(valsToFetchPerLine, 0);
      // fetch values from the current line buffer
      for (int i = 0; i < valsToFetchPerLine; i++) {
        initialValuesToPassOver[i] = readVal();
      }
      // reset the pointers we use to the start of the buffer
      ptr0 = lineBuffer;
      ptr1 = lineBuffer;

      // Send these off to the grid - handle indexing in the grid class
      grid::Grid::Instance.setInitialConditions(valuesFetched, initialValuesToPassOver);

      valuesFetched++;
    }

    // validation - does valuesFetched match nx?
    assert(valuesFetched == nx);


    fclose(icfile);
  }

} // namespace IO
