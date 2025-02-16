#include "Logging.h"

#include <iostream>
#include <sstream>
#include <string>

#include "Config.h"

namespace logging {

  // Initialise the verbosity as debug.
  LogLevel Log::_verbosity = LogLevel::Debug;


  void Log::logMessage(
    const char* file,
    const char* function,
    const int   line,
    std::string text,
    LogLevel    level,
    LogStage    stage
  ) {

    // Are we talkative enough?
    if (_verbosity < level)
      return;

    std::stringstream str;
    str << "[" << getStageName(stage) << "] ";
#if DEBUG_LEVEL > 0
    str << "{" << file << ":" << function << "():" << line << "} ";
#endif
    str << text << "\n";

    std::cout << str.str();

    // Do we want the message to be instantly flushed to screen?
    bool flush = level >= LogLevel::Debug;
    if (flush)
      std::cout << std::flush;
  }

  void Log::logMessage(
    const char*        file,
    const char*        function,
    const int          line,
    std::stringstream& text,
    LogLevel           level,
    LogStage           stage
  ) {
    logMessage(file, function, line, text.str(), level, stage);
  }

  void Log::logMessage(
    const char* file,
    const char* function,
    const int   line,
    const char* text,
    LogLevel    level,
    LogStage    stage
  ) {
    logMessage(file, function, line, std::string(text), level, stage);
  }


  void Log::logWarning(
    const char* file, const char* function, const int line, const std::string& text
  ) {

    std::stringstream str;
    str << "[WARNING] ";
    str << "{" << file << ":" << function << "():" << line << "} ";
    str << text << "\n";

    std::cerr << str.str();
  }

  void Log::logWarning(
    const char* file, const char* function, const int line, const std::stringstream& text
  ) {
    logWarning(file, function, line, text.str());
  }

  void Log::logWarning(const char* file, const char* function, const int line, const char* text) {
    logWarning(file, function, line, std::string(text));
  }


  void Log::logError(const char* file, const char* function, const int line, std::string text) {

    std::stringstream str;
    str << "[ERROR] ";
    str << "{" << file << ":" << function << "():" << line << "} ";
    str << text << "\n";

    std::cerr << str.str();

    std::cerr << std::flush;
    std::cout << std::flush;
    std::abort();
  }

  void Log::logError(
    const char* file, const char* function, const int line, std::stringstream& text
  ) {
    logError(file, function, line, text.str());
  }

  void Log::logError(const char* file, const char* function, const int line, const char* text) {
    logError(file, function, line, std::string(text));
  }


  void Log::setVerbosity(int verbosity) {
    LogLevel vlevel = static_cast<LogLevel>(verbosity);
    setVerbosity(vlevel);
  }

  void Log::setVerbosity(LogLevel verbosity) { Log::_verbosity = verbosity; }

  const char* Log::getStageName(LogStage stage) {

    switch (stage) {
    case LogStage::Undefined:
      return "Undefined";
    case LogStage::Header:
      return "Header";
    case LogStage::Init:
      return "Init";
    case LogStage::Step:
      return "Step";
    case LogStage::Count:
      return "Count";
    default:
      return "Unknown";
    }
  }


} // namespace logging
