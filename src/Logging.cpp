#include "Logging.h"

#include <iostream>
#include <sstream>
#include <string>

#include "Config.h"
#include "Version.h"

namespace hydro_playground {
  namespace logging {

    // Initialise the verbosity as debug.
    LogLevel Log::_verbosity = LogLevel::Debug;


    void Log::message(
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

    void Log::message(
      const char*        file,
      const char*        function,
      const int          line,
      std::stringstream& text,
      LogLevel           level,
      LogStage           stage
    ) {
      message(file, function, line, text.str(), level, stage);
    }

    void Log::message(
      const char* file,
      const char* function,
      const int   line,
      const char* text,
      LogLevel    level,
      LogStage    stage
    ) {
      message(file, function, line, std::string(text), level, stage);
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
} // namespace hydro_playground
