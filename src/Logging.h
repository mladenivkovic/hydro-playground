#pragma once

#include <string>
#include <source_location>

#include "Config.h"

#include <iostream>

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


  //! Get/Set the current global code stage.
  void     setStage(const int stage);
  void     setStage(const LogStage stage);
  LogStage getCurrentStage();


  //! Get/Set the current global verbosity level.
  void     setVerbosity(const int level);
  void     setVerbosity(const LogLevel level);
  LogLevel getCurrentVerbosity();


  /**
   * Get the name of the given stage.
   */
  const char* getStageName(LogStage stage);


  constexpr std::string_view extractFileName(const char* path) {

    // First, get project root prefix
    // In a pinch, try extracting the root source dir from this file.
    // Better option: Let cmake tell me what it is.
    // std::source_location location = std::source_location::current();
    // auto thisfile = std::string_view(location.file_name());

    auto thisfile = std::string_view(CMAKE_SOURCE_DIR);

    size_t pos_this = thisfile.find_last_of("/\\");

    std::string_view prefix;
    if (pos_this != std::string_view::npos) {
      prefix = thisfile.substr(0, pos_this + 1);
    }

    // Get the path's prefix too.
    std::string_view prefix_path;
    if (pos_this != std::string_view::npos) {
      prefix_path = std::string_view(path).substr(0, pos_this + 1);
    }

    std::string_view trimmed(path);
    if (prefix_path == prefix){
      trimmed = trimmed.substr(pos_this+1);
    }

    return trimmed;
  }


  /**
   * @brief Logger class.
   *
   * Logs (messages to screen) are characterised by their "level" and "stage".
   *
   *  - "level" determines the verbosity threshold needed to actually print out
   *     the message. For example, a message marked at "Debug" or "Verbose"
   *     level won't be printed to screen if the global run setting is "Quiet."
   *     "Levels" are set by the corresponging enum logging::LogLevel.
   *
   *  - "stage" determines at which stage in the code we currently reside. This
   *    just helps orientation with the output logs."Levels" are set by the
   *    corresponging enum logging::LogStage. If you want to add a new LogStage,
   *    don't forget to add an entry in logging::getStageName().
   *
   * This class sets up a singleton in the background and is not intended to be
   * used directly. Instead, use the convenience macros defined below. To write
   * a log message to screen, use
   *
   *   message(<msg>);
   *
   * where <msg> is a std::string or string literal (char * like "Hello World").
   *
   * Note that std::stringstream doesn't work, as they aren't copyable, which is
   * a requirement for constexpr. We end up tyring to make a constexpr which is
   * never actually constant, and the compiler complains. Use
   * std::stringstream.str() instead.
   *
   * Note that any logging printout will add a newline character for you at the
   * end of your message.
   *
   * You can specify the LogLevel of the message too:
   *
   *   message(<msg>, <level>);
   *
   * where <level> is a logging::LogLevel enum. The default LogLevel is Quiet:
   * the message will always be printed unless you specify a higher verbosity
   * level requirement as in the example above.
   *
   * You can futhermore specify a stage of the message as well:
   *
   *   message(<msg>, <level>, <stage>);
   *
   * where <stage> is a logging::LogStage enum.
   * Additionally, you can raise warnings using the warning macro:
   *
   *   warning(<msg>);
   *
   * A warning will always be printed, regardless of your verbosity level.
   * Similarly, you can raise an error using
   *
   *   error(<msg>);
   *
   * Which prints yout your messages and exits with an errorcode.
   *
   * You can set global verbosity levels and code stage states using the
   * convenience functions
   *
   *  logging::setVerbosity(<level>);
   *  logging::setStage(<stage>);
   *
   */
  class Log {
  public:
    Log():
      _verbosity(LogLevel::Debug),
      _currentStage(LogStage::Undefined) {
    }

    static Log& getInstance() {
      static Log Instance;
      return Instance;
    }

    /**
     * @brief write a log message to screen.
     *
     * I can probably template this, but I don't think I'm using anything
     * besides strings or char arrays (string literals), so it's not worth
     * it at this point.
     * Also, constexpr doesn't work with constexpr, so we're only left with
     * 2 options anyway.
     *
     * @param text The message you want to print out.
     * @param level "verbosity level" of the log message.
     * @param stage stage of the code where this log is called from.
     * @param file The current file. Intended to be the (replacement of the)
     *   __FILE__ macro.
     * @param function The current function name. Intended to be the
     *   (replacement of the) __FUNCTION__ macro.
     * @param line The current line in the file. Intended to be the
     *   (replacement of the) __LINE__ macro.
     */
    void logMessage(
      const std::string& text,
      const LogLevel     level,
      const LogStage     stage,
      const std::string_view        file,
      const char*        function,
      const size_t       line
    );
    void logMessage(
      const char*    text,
      const LogLevel level,
      const LogStage stage,
      const std::string_view    file,
      const char*    function,
      const size_t   line
    );

    /**
     * @brief write a warning message.
     *
     * @param text The message you want to print out.
     * @param file The current file. Intended to be the (replacement of the)
     *   __FILE__ macro.
     * @param function The current function. Intended to be the (replacement of
     *   the) __FUNCTION__ macro.
     * @param line The current line in the file. Intended to be the
     *   (replacement of the) __LINE__ macro.
     */
    void logWarning(
      const std::string& text,
      const std::string_view file,
      const char* function,
      const size_t line
    );
    void logWarning(
        const char* text,
        const std::string_view file,
        const char* function,
        const size_t line
        );

    /**
     * @brief write an error message and abort the run.
     *
     * @param text The message you want to print out.
     * @param file The current file. Intended to be the (replacement of the)
     *   __FILE__ macro.
     * @param function The current function name. Intended to be the
     *   (replacement of the) __FUNCTION__ macro.
     * @param line The current line in the file. Intended to be the
     *   (replacement of the) __LINE__ macro.
     */
    void logError(
        const std::string& text,
        const std::string_view file,
        const char* function,
        const size_t line
        );
    void logError(
        const char* text,
        const std::string_view file,
        const char* function,
        const size_t line
        );


    /**
     * Set the global verbosity level.
     */
    void setVerbosity(const int verbosity);
    void setVerbosity(const LogLevel verbosity);

    //! Get the current verbosity level.
    LogLevel getCurrentVerbosity();

    /**
     * Set the global stage.
     */
    void setStage(const LogStage stage);
    void setStage(const int stage);

    //! Get the current stage.
    LogStage getCurrentStage();

  private:
    LogLevel _verbosity;
    LogStage _currentStage;
  };
} // namespace logging




template <typename T>
constexpr void message(
    const T msg,
    const std::source_location& location = std::source_location::current()
    ) {
  logging::Log::getInstance().logMessage(
    msg,
    logging::LogLevel::Quiet,
    logging::getCurrentStage(),
    location.file_name(),
    location.function_name(),
    location.line()
  );
}

template <typename T>
constexpr void message(
    const T msg,
    const logging::LogLevel level,
    const std::source_location& location = std::source_location::current()
    ) {
  logging::Log::getInstance().logMessage(
    msg,
    level,
    logging::getCurrentStage(),
    logging::extractFileName(location.file_name()),
    location.function_name(),
    location.line()
  );
}

template <typename T>
constexpr void message(
    const T msg,
    const logging::LogLevel level,
    const logging::LogStage stage,
    const std::source_location& location = std::source_location::current()
    ) {
  logging::Log::getInstance().logMessage(
    msg,
    level,
    stage,
    logging::extractFileName(location.file_name()),
    location.function_name(),
    location.line()
  );
}

template <typename T>
constexpr void message(
    const T msg,
    const logging::LogStage stage,
    const std::source_location& location = std::source_location::current()
    ) {
  logging::Log::getInstance().logMessage(
    msg,
    logging::LogLevel::Quiet,
    stage,
    logging::extractFileName(location.file_name()),
    location.function_name(),
    location.line()
  );
}


template <typename T>
constexpr void error(
    const T msg,
    const std::source_location& location = std::source_location::current()
    ) {
  logging::Log::getInstance().logError(
      msg,
      logging::extractFileName(location.file_name()),
      location.function_name(),
      location.line()
    );
}

template <typename T>
constexpr void warning(
    const T msg,
    const std::source_location& location = std::source_location::current()
    ) {
  logging::Log::getInstance().logWarning(
      msg,
      logging::extractFileName(location.file_name()),
      location.function_name(),
      location.line()
      );
}
