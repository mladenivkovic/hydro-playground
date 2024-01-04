#pragma once

#include <iostream>
#include <sstream>
#include <string>

// Utilities related to logging.


namespace hydro_playground {
namespace logging {

enum LogLevel {
  LogLevelUndefined = -1,
  LogLevelQuiet = 0,
  LogLevelVerbose = 1,
  LogLevelDebug = 2,
  LogLevelCount
};


enum LogStage {
  LogStageUndefined = -1,
  LogStageHeader = 0,
  LogStageInit = 1,
  LogStageStep = 2,
  LogStageCount,
};


class Log {
  public:
    /**
    * write a log message.
    *
    * @param message The message you want to print out.
    * @param level "verbosity level" of the log message.
    * @param stage stage of the code where this log is called from.
    * @param verbose how verbose the code is being run.
    */
    Log(std::string message, LogLevel level, LogStage stage, int verbose = LogLevelCount);
    Log(std::stringstream& messagestream, LogLevel level, LogStage stage, int verbose = LogLevelCount);
    Log(const char* message, LogLevel level, LogStage stage, int verbose = LogLevelCount);

    /**
     * shorthand to write a log message.
     * This will always be printed directly to screen, and is
     * intended for temporary development/debugging purposes.
     *
     * Note that instantiating the `Log` using parentheses, e.g.
     *
     *    const char *msg = "This is my message";
     *    Log(msg);
     *
     * Will lead to issues (see 'most vexing parse problem').
     * Instead, use braces:
     *
     *    Log{msg};
     *
     * However, passing a string literal directly to the constructor
     * works as intended.
     *
     *    Log("This works!");
     */
    Log(std::string message);
    Log(std::stringstream& messagestream);
    Log(const char* message);

  private:
    const char* getStageName(LogStage stage);
};


} // namespace logging
} // namespace hydro_playground
