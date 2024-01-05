#pragma once

#include <iostream>
#include <sstream>
#include <string>

// Utilities related to logging.


namespace hydro_playground {
  namespace logging {

    enum class LogLevel { Undefined = -1, Quiet = 0, Verbose = 1, Debug = 2, Count };


    enum class LogStage {
      Undefined = -1,
      Header    = 0,
      Init      = 1,
      Step      = 2,
      Count,
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
      Log(std::string message, LogLevel level, LogStage stage, int verbose = static_cast<int>(LogLevel::Count));
      Log(std::stringstream& messagestream, LogLevel level, LogStage stage, int verbose = static_cast<int>(LogLevel::Count));
      Log(const char* message, LogLevel level, LogStage stage, int verbose = static_cast<int>(LogLevel::Count));

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
