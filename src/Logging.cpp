#include "Logging.h"

#include <iostream>
#include <sstream>
#include <string>

#include "Config.h"

  // Initialise the verbosity as debug.
logging::LogLevel logging::Log::_verbosity    = LogLevel::Debug;
logging::LogStage logging::Log::_currentStage = LogStage::Undefined;


void logging::Log::logMessage(
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
  str << "`" << file << ":" << function << "():" << line << "`: ";
#endif
  str << text << "\n";

  std::cout << str.str();

  // Do we want the message to be instantly flushed to screen?
  bool flush = level >= LogLevel::Debug;
  if (flush)
    std::cout << std::flush;
}

void logging::Log::logMessage(
  const char*        file,
  const char*        function,
  const int          line,
  std::stringstream& text,
  LogLevel           level,
  LogStage           stage
) {
  logMessage(file, function, line, text.str(), level, stage);
}

void logging::Log::logMessage(
  const char* file,
  const char* function,
  const int   line,
  const char* text,
  LogLevel    level,
  LogStage    stage
) {
  logMessage(file, function, line, std::string(text), level, stage);
}


void logging::Log::logWarning(
  const char* file, const char* function, const int line, const std::string& text
) {

  std::stringstream str;
  str << "[WARNING] ";
  str << "`" << file << ":" << function << "():" << line << "`: ";
  str << text << "\n";

  std::cerr << str.str();
}

void logging::Log::logWarning(
  const char* file, const char* function, const int line, const std::stringstream& text
) {
  logWarning(file, function, line, text.str());
}

void logging::Log::logWarning(const char* file, const char* function, const int line, const char* text) {
  logWarning(file, function, line, std::string(text));
}


void logging::Log::logError(const char* file, const char* function, const int line, std::string text) {

  std::stringstream str;
  str << "[ERROR] ";
  str << "`" << file << ":" << function << "():" << line << "`: ";
  str << text << "\n";

  std::cerr << str.str();

  std::cerr << std::flush;
  std::cout << std::flush;
  std::abort();
}

void logging::Log::logError(
  const char* file, const char* function, const int line, std::stringstream& text
) {
  logError(file, function, line, text.str());
}

void logging::Log::logError(const char* file, const char* function, const int line, const char* text) {
  logError(file, function, line, std::string(text));
}


void logging::Log::setVerbosity(int verbosity) {
  LogLevel vlevel = static_cast<LogLevel>(verbosity);
  setVerbosity(vlevel);
}

void logging::Log::setVerbosity(LogLevel verbosity) {
  Log::_verbosity = verbosity;
  std::stringstream msg;
  msg << "Setting verbosity to " << static_cast<int>(verbosity);
  // only print this if we're talky
  message(msg.str(), LogLevel::Verbose);
}


logging::LogLevel logging::Log::getCurrentVerbosity() {
  return Log::_verbosity;
}



void logging::Log::setStage(int stage) {
  auto stage_t = static_cast<LogStage>(stage);
  setStage(stage_t);
}

void logging::Log::setStage(LogStage stage) {
  std::stringstream msg;
  msg << "Setting stage " << Log::getStageName(stage);
  message(msg.str(), LogLevel::Verbose);
  Log::_currentStage = stage;
}


logging::LogStage logging::Log::getCurrentStage() {
  return Log::_currentStage;
}


const char* logging::Log::getStageName(LogStage stage) {

  switch (stage) {
  case LogStage::Undefined:
    return "Undefined";
  case LogStage::Header:
    return "Header";
  case LogStage::Init:
    return "Init";
  case LogStage::Step:
    return "Step";
  case LogStage::Test:
    return "Test";
  case LogStage::Count:
    return "Count";
  default:
    return "Unknown";
  }
}


