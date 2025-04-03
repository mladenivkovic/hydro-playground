#pragma once

#include <iostream>
#include <source_location>
#include <string>

#include "Config.h"
#include "Termcolors.h"

/**
 * @file Logging.h
 * @brief Utilities related to logging.
 */
template <typename T>
concept AllowedMessageType = std::same_as<T, std::string> || std::same_as<T, const char*>
                             || std::same_as<T, char*> || std::same_as<T, std::string_view>;


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
    IO        = 3,
    Shutdown  = 4,
    Test      = 5,
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


  /**
   * Get the name of the given stage, formatted for output (to screen).
   */
  std::string getStageNameForOutput(LogStage stage);


  /**
   * Get the colour for the given stage.
   */
  const char* getStageNameColour(LogStage stage);


  /**
   * Extract the file name from the path by trimming the project
   * root directory from the prefix.
   */
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
    if (prefix_path == prefix) {
      trimmed = trimmed.substr(pos_this + 1);
    }

    return trimmed;
  }


  /**
   * Extract the function name from func: remove return type and arguments
   */
  constexpr std::string_view extractFunctionName(const char* func) {

    auto funcv = std::string_view(func);

    // Trim return type: Everything before space
    size_t pos = funcv.find_first_of(' ');
    if (pos != std::string_view::npos) {
      funcv = funcv.substr(pos + 1, funcv.size() - pos);
    }

    size_t pos_arg = funcv.find_first_of('(');
    if (pos_arg != std::string_view::npos) {
      funcv = funcv.substr(0, pos_arg);
    }

    return funcv;
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
   * used directly. Instead, use the convenience functions defined below. To
   * write a log message to screen, use
   *
   *   message(<msg>);
   *
   * where <msg> is a std::string or string literal (char * like "Hello World")
   * or std::string_view.
   *
   * !!! Note that std::stringstream doesn't work !!!, as they aren't copyable,
   * which is a requirement for constexpr. We end up tyring to make a constexpr
   * which is never actually constant, and the compiler complains. Use
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
   * Additionally, you can raise warnings using the warning function:
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
   * Finally, there is also the
   *
   *   timing(<msg>)
   *
   * function as a special case to treat all timing outputs specially. In
   * particular, they will only be printed to screen in debug mode, i.e. if the
   * global verbosity was set to LogLevel::Debug or if the code was compiled
   * with DEBUG_LEVEL > 0. k
   *
   * You can set global verbosity levels and code stage states using the
   * convenience functions
   *
   *  logging::setVerbosity(<level>);
   *  logging::setStage(<stage>);
   *
   */
  class Log {
  private:
    LogLevel _verbosity;
    LogStage _currentStage;

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
     * @brief write a log message to screen. See the class documentation for an
     * explanation of the LogLevel and LogStage.
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
    template <AllowedMessageType T>
    void logMessage(
      const T        text,
      const LogLevel level,
      const LogStage stage,
      const char*    file,
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
    template <AllowedMessageType T>
    void logWarning(const T text, const char* file, const char* function, const size_t line);


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
#pragma omp declare target
    template <AllowedMessageType T>
    void logError(const T text, const char* file, const char* function, const size_t line);
#pragma omp end declare target


    /**
     * @brief write a timing message.
     *
     * @param text The message you want to print out.
     * @param file The current file. Intended to be the (replacement of the)
     *   __FILE__ macro.
     * @param function The current function. Intended to be the (replacement of
     *   the) __FUNCTION__ macro.
     * @param line The current line in the file. Intended to be the
     *   (replacement of the) __LINE__ macro.
     */
    template <AllowedMessageType T>
    void logTiming(const T text, const char* file, const char* function, const size_t line);


    /**
     * The function that actually constructs a message/log/warning/error
     */
    template <
      AllowedMessageType T1,
      AllowedMessageType T2,
      AllowedMessageType T3,
      AllowedMessageType T4>
    std::string constructMessage(
      const T1     prefix,
      const T2     text,
      const T3     file,
      const T4     function,
      const size_t line,
      const bool   debug = true
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
  };
} // namespace logging


template <AllowedMessageType T>
constexpr void message(
  const T msg, const std::source_location& location = std::source_location::current()
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


template <AllowedMessageType T>
constexpr void message(
  const T                     msg,
  const logging::LogLevel     level,
  const std::source_location& location = std::source_location::current()
) {
  logging::Log::getInstance().logMessage(
    msg,
    level,
    logging::getCurrentStage(),
    location.file_name(),
    location.function_name(),
    location.line()
  );
}

template <AllowedMessageType T>
constexpr void message(
  const T                     msg,
  const logging::LogLevel     level,
  const logging::LogStage     stage,
  const std::source_location& location = std::source_location::current()
) {
  logging::Log::getInstance().logMessage(
    msg, level, stage, location.file_name(), location.function_name(), location.line()
  );
}

template <AllowedMessageType T>
constexpr void message(
  const T                     msg,
  const logging::LogStage     stage,
  const std::source_location& location = std::source_location::current()
) {
  logging::Log::getInstance().logMessage(
    msg,
    logging::LogLevel::Quiet,
    stage,
    location.file_name(),
    location.function_name(),
    location.line()
  );
}


template <AllowedMessageType T>
constexpr void error(
  const T msg, const std::source_location& location = std::source_location::current()
) {
  logging::Log::getInstance().logError(
    msg, location.file_name(), location.function_name(), location.line()
  );
}


template <AllowedMessageType T>
constexpr void warning(
  const T msg, const std::source_location& location = std::source_location::current()
) {
  logging::Log::getInstance().logWarning(
    msg, location.file_name(), location.function_name(), location.line()
  );
}


template <AllowedMessageType T>
constexpr void timing(
  const T msg, const std::source_location& location = std::source_location::current()
) {
  logging::Log::getInstance().logTiming(
    msg, location.file_name(), location.function_name(), location.line()
  );
}


template <AllowedMessageType T>
void logging::Log::logMessage(
  const T        text,
  const LogLevel level,
  const LogStage stage,
  const char*    file,
  const char*    function,
  const size_t   line
) {

  // Are we talkative enough?
  if (_verbosity < level)
    return;

  std::string prefix;
  std::string stext;

  if (color_term) {
    prefix += getStageNameColour(stage);
    prefix += getStageNameForOutput(stage);
    prefix += tcols::reset;

    stext += getStageNameColour(stage);
    stext += text;
    stext += tcols::reset;
  } else {
    prefix = getStageNameForOutput(stage);
    stext  = text;
  }

  bool        debug = (DEBUG_LEVEL > 0) or (getCurrentVerbosity() >= LogLevel::Debug);
  std::string out   = constructMessage(prefix, stext, file, function, line, debug);
  std::cout << out;

  // Do we want the message to be instantly flushed to screen?
  bool flush = level >= LogLevel::Debug;
  if (flush)
    std::cout << std::flush;
}


template <AllowedMessageType T>
void logging::Log::logWarning(
  const T text, const char* file, const char* function, const size_t line
) {

  std::string prefix;
  std::string stext;

  if (color_term) {
    prefix += tcols::yellow;
    prefix += "[WARNING] ";
    prefix += tcols::reset;

    stext += tcols::yellow;
    stext += text;
    stext += tcols::reset;
  } else {
    prefix = "[WARNING] ";
    stext  = text;
  }

  std::string out = constructMessage(prefix, stext, file, function, line, true);
  std::cerr << out;
}


template <AllowedMessageType T>
#pragma omp declare target
void logging::Log::logError(
  const T text, const char* file, const char* function, const size_t line
) {

  std::string prefix;
  std::string stext;

  if (color_term) {
    prefix += tcols::red;
    prefix += "[ERROR]   ";
    prefix += tcols::reset;

    stext += tcols::red;
    stext += text;
    stext += tcols::reset;
  } else {
    prefix = "[ERROR]   ";
    stext  = text;
  }

  std::string out = constructMessage(prefix, stext, file, function, line, true);
  std::cerr << out;

  std::cerr << std::flush;
  std::cout << std::flush;
  std::abort();
}
#pragma omp end declare target


template <AllowedMessageType T>
void logging::Log::logTiming(
  const T text, const char* file, const char* function, const size_t line
) {

  std::string prefix;
  std::string stext;

  if (color_term) {
    prefix += tcols::yellow;
    prefix += "[Timing]  ";
    prefix += tcols::reset;

    stext += tcols::yellow;
    stext += text;
    stext += tcols::reset;
  } else {
    prefix = "[Timing]  ";
    stext  = text;
  }

  bool        debug = (DEBUG_LEVEL > 0) or (getCurrentVerbosity() >= LogLevel::Debug);
  std::string out   = constructMessage(prefix, stext, file, function, line, debug);
  std::cerr << out;
}


template <AllowedMessageType T1, AllowedMessageType T2, AllowedMessageType T3, AllowedMessageType T4>
std::string logging::Log::constructMessage(
  const T1     prefix,
  const T2     text,
  const T3     file,
  const T4     function,
  const size_t line,
  const bool   debug
) {

  std::string_view file_trimmed = extractFileName(file);
  std::string_view func_trimmed = extractFunctionName(function);

  std::string locs;


  if (debug) {
    if (color_term)
      locs += tcols::blue;
    locs += file_trimmed;
    locs += ":";
    locs += std::to_string(line);
    locs += " (";
    locs += func_trimmed;
    locs += "): ";
    if (color_term)
      locs += tcols::reset;
  }

  std::string out = prefix + locs;

  out += text;
  out += "\n";

  return out;
}
