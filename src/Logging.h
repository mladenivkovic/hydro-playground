#pragma once

#include <iostream>
#include <sstream>
#include <string>

/**
 * @file Logging.h Utilities related to logging.
 */


namespace hydro_playground {
  namespace logging {

// This macro truncates the full path from the __FILE__ macro.
#ifdef SOURCE_PATH_SIZE
#define __FILENAME__ (__FILE__ + SOURCE_PATH_SIZE)
#else
#define __FILENAME__ __FILE__
#endif

#define log(msg, level, stage) \
  hydro_playground::logging::Log::message(__FILENAME__, __FUNCTION__, __LINE__, msg, level, stage);


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
      // Don't instantiate this class, ever.
      Log() = delete;

      /**
       * write a log message.
       *
       * @param file The current file. Intended to be the __FILE__ macro.
       * @param function The current function. Intended to be the __FUNCTION__ macro.
       * @param line The current line in the file. Intended to be the __LINE__ macro.
       * @param text The message you want to print out.
       * @param level "verbosity level" of the log message.
       * @param stage stage of the code where this log is called from.
       */
      static void message(
        const char* file,
        const char* function,
        const int   line,
        std::string text,
        LogLevel    level,
        LogStage    stage
      );
      static void message(
        const char*        file,
        const char*        function,
        const int          line,
        std::stringstream& text,
        LogLevel           level,
        LogStage           stage
      );
      static void message(
        const char* file,
        const char* function,
        const int   line,
        const char* text,
        LogLevel    level,
        LogStage    stage
      );

      /**
       * Set the global verbosity level.
       */
      static void setVerbosity(int verbosity);
      static void setVerbosity(LogLevel verbosity);

    private:
      static LogLevel _verbosity;

      /**
       * Get the name of the given stage.
       */
      static const char* getStageName(LogStage stage);
    };


  } // namespace logging
} // namespace hydro_playground
