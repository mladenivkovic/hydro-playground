#pragma once

#include <sstream>
#include <string>

/**
 * @file Logging.h
 * @brief Utilities related to logging.
 */

namespace logging {

  enum class LogLevel {
    Undefined = -1,
    Quiet     = 0,
    Verbose   = 1,
    Debug     = 2,
    Count
  };

  enum class LogStage {
    Undefined = -1,
    Header    = 0,
    Init      = 1,
    Step      = 2,
    Test      = 3,
    Count,
  };


  class Log {
  public:
    // Don't instantiate this class, ever.
    Log() = delete;

    /**
     * @brief write a log message to screen.
     *
     * @param file The current file. Intended to be the __FILE__ macro.
     * @param function The current function. Intended to be the __FUNCTION__ macro.
     * @param line The current line in the file. Intended to be the __LINE__ macro.
     * @param text The message you want to print out.
     * @param level "verbosity level" of the log message.
     * @param stage stage of the code where this log is called from.
     */
    static void logMessage(
      const char* file,
      const char* function,
      const int   line,
      std::string text,
      LogLevel    level,
      LogStage    stage
    );
    static void logMessage(
      const char*        file,
      const char*        function,
      const int          line,
      std::stringstream& text,
      LogLevel           level,
      LogStage           stage
    );
    static void logMessage(
      const char* file,
      const char* function,
      const int   line,
      const char* text,
      LogLevel    level,
      LogStage    stage
    );

    /**
     * @brief write a warning message.
     *
     * @param file The current file. Intended to be the __FILE__ macro.
     * @param function The current function. Intended to be the __FUNCTION__ macro.
     * @param line The current line in the file. Intended to be the __LINE__ macro.
     * @param text The message you want to print out.
     */
    static void logWarning(
      const char* file, const char* function, const int line, const std::string& text
    );
    static void logWarning(
      const char* file, const char* function, const int line, const std::stringstream& text
    );
    static void logWarning(
      const char* file, const char* function, const int line, const char* text
    );

    /**
     * @brief write an error message and abort the run.
     *
     * @param file The current file. Intended to be the __FILE__ macro.
     * @param function The current function. Intended to be the __FUNCTION__ macro.
     * @param line The current line in the file. Intended to be the __LINE__ macro.
     * @param text The message you want to print out.
     */
    static void logError(const char* file, const char* function, const int line, std::string text);
    static void logError(
      const char* file, const char* function, const int line, std::stringstream& text
    );
    static void logError(const char* file, const char* function, const int line, const char* text);


    /**
     * Set the global verbosity level.
     */
    static void setVerbosity(const int verbosity);
    static void setVerbosity(const LogLevel verbosity);

    /**
     * Set the global stage.
     */
    static void setStage(const LogStage stage);
    static void setStage(const int stage);

    //! Get the current stage.
    static LogStage getCurrentStage();

  private:
    static LogLevel _verbosity;
    static LogStage _currentStage;

    /**
     * Get the name of the given stage.
     */
    static const char* getStageName(LogStage stage);
  };


// This macro truncates the full path from the __FILE__ macro.
#ifdef SOURCE_PATH_SIZE
#define FILENAME_ (__FILE__ + SOURCE_PATH_SIZE)
#else
#define FILENAME_ __FILE__
#endif

#define MESSAGE_3_ARGS(msg, level, stage) \
  logging::Log::logMessage(FILENAME_, __FUNCTION__, __LINE__, msg, level, stage \
);

#define MESSAGE_2_ARGS(msg, level) \
  logging::Log::logMessage( \
    FILENAME_, __FUNCTION__, __LINE__, msg, level, logging::Log::getCurrentStage() \
  );

#define MESSAGE_1_ARG(msg) \
  logging::Log::logMessage( \
    FILENAME_, \
    __FUNCTION__, \
    __LINE__, \
    msg, \
    logging::LogLevel::Undefined, \
    logging::Log::getCurrentStage() \
  );

#define MESSAGE_GET_4TH_ARG(arg1, arg2, arg3, arg4, ...) arg4

// We return the 4th argument. The variable number of arguments passed
// to this macro (which are __VA_ARGS__) pushes the "correct" message
// macro we want to use to the 4th argument. So this way, we get the
// correct macro to use.
#define MESSAGE_STRING_MACRO_CHOOSER(...) \
  MESSAGE_GET_4TH_ARG(__VA_ARGS__, MESSAGE_3_ARGS, MESSAGE_2_ARGS, MESSAGE_1_ARG, )


// The main message() macro.
#define message(...) MESSAGE_STRING_MACRO_CHOOSER(__VA_ARGS__)(__VA_ARGS__)


#define error(msg) logging::Log::logError(FILENAME_, __FUNCTION__, __LINE__, msg);

#define warning(msg) logging::Log::logWarning(FILENAME_, __FUNCTION__, __LINE__, msg);


} // namespace logging
