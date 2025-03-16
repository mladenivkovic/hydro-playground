#include "Logging.h"

#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>


void logging::setStage(const int stage) {
  Log::getInstance().setStage(stage);
}

void logging::setStage(const logging::LogStage stage) {
  Log::getInstance().setStage(stage);
}


logging::LogStage logging::getCurrentStage() {
  return Log::getInstance().getCurrentStage();
}


void logging::setVerbosity(const int level) {
  Log::getInstance().setVerbosity(level);
}

void logging::setVerbosity(const logging::LogLevel level) {
  Log::getInstance().setVerbosity(level);
}


logging::LogLevel logging::getCurrentVerbosity() {
  return Log::getInstance().getCurrentVerbosity();
}


std::string logging::getStageNameForOutput(LogStage stage) {

  const char* name = getStageName(stage);

  std::string       s = "[" + std::string(name) + "]";
  std::stringstream ss;
  ss << std::left << std::setw(10) << s;
  return ss.str();
}


const char* logging::getStageName(LogStage stage) {

  switch (stage) {
  case LogStage::Undefined:
    return "Undefined";
  case LogStage::Header:
    return "Header";
  case LogStage::Init:
    return "Init";
  case LogStage::Step:
    return "Step";
  case LogStage::IO:
    return "IO";
  case LogStage::Shutdown:
    return "Shutdown";
  case LogStage::Test:
    return "Test";
  case LogStage::Count:
    return "Count";
  default:
    return "Unknown";
  }
}


const char* logging::getStageNameColour(LogStage stage) {

  switch (stage) {
  case LogStage::Undefined:
    return tcols::red;
  case LogStage::Header:
  case LogStage::Init:
  case LogStage::Shutdown:
    return tcols::green;
  case LogStage::Step:
    return tcols::cyan;
  case LogStage::IO:
  case LogStage::Test:
    return tcols::magenta;
  case LogStage::Count:
  default:
    return tcols::reset;
  }
}


void logging::Log::setVerbosity(const int verbosity) {
  auto vlevel = static_cast<LogLevel>(verbosity);
  setVerbosity(vlevel);
}


void logging::Log::setVerbosity(const LogLevel verbosity) {
  _verbosity = verbosity;

  // only notify me if we're talky
  std::stringstream msg;
  msg << "Setting verbosity to " << static_cast<int>(verbosity);
  message(msg.str(), LogLevel::Verbose);
}


logging::LogLevel logging::Log::getCurrentVerbosity() {
  return _verbosity;
}


void logging::Log::setStage(int stage) {
  auto stage_t = static_cast<LogStage>(stage);
  setStage(stage_t);
}

void logging::Log::setStage(LogStage stage) {

  _currentStage = stage;

  std::stringstream msg;
  msg << "Setting stage " << getStageName(stage);
  message(msg.str(), LogLevel::Verbose);
}


logging::LogStage logging::Log::getCurrentStage() {
  return _currentStage;
}
