#include "Logging.h"

#include <iostream>
#include <sstream>
#include <string>

#include "Config.h"



void logging::setStage(const int stage){
  Log::getInstance().setStage(stage);
}

void logging::setStage(const logging::LogStage stage){
  Log::getInstance().setStage(stage);

}

logging::LogStage logging::getCurrentStage(){
  return Log::getInstance().getCurrentStage();
}

void logging::setVerbosity(const int level){
  Log::getInstance().setVerbosity(level);
}

void logging::setVerbosity(const logging::LogLevel level){
  Log::getInstance().setVerbosity(level);
}

logging::LogLevel logging::getCurrentVerbosity(){
  return Log::getInstance().getCurrentVerbosity();
}




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


void logging::Log::setVerbosity(const int verbosity) {
  auto vlevel = static_cast<LogLevel>(verbosity);
  getInstance().setVerbosity(vlevel);
}

void logging::Log::setVerbosity(const LogLevel verbosity) {
  getInstance()._verbosity = verbosity;
  std::stringstream msg;
  msg << "Setting verbosity to " << static_cast<int>(verbosity);
  // only print this if we're talky
  message(msg.str(), LogLevel::Verbose);
}


logging::LogLevel logging::Log::getCurrentVerbosity() {
  return getInstance().Log::_verbosity;
}



void logging::Log::setStage(int stage) {
  auto stage_t = static_cast<LogStage>(stage);
  setStage(stage_t);
}

void logging::Log::setStage(LogStage stage) {
  std::stringstream msg;
  msg << "Setting stage " << Log::getStageName(stage);
  message(msg.str(), LogLevel::Verbose);
  getInstance()._currentStage = stage;
}


logging::LogStage logging::Log::getCurrentStage() {
  return getInstance()._currentStage;
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


