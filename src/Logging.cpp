#include "Logging.h"

#include "Version.h"


#include <iostream>
#include <sstream>
#include <string>

namespace hydro_playground {
namespace logging {


Log::Log(std::string message, LogLevel level, LogStage stage, int verbose){

  // Are we talkative enough?
  if (verbose < static_cast<int>(level)) return;

  std::stringstream str;
  str << "[" << getStageName(stage) << "] ";
  str << message << "\n";

  std::cout << str.str() ;

  // Do we want the message to be instantly flushed to screen?
  bool flush = level >= LogLevel::Debug;
  if (flush) std::cout << std::flush;
}

Log::Log(std::stringstream& messagestream, LogLevel level, LogStage stage, int verbose){
  Log(messagestream.str(), level, stage, verbose);
}

Log::Log(const char* message, LogLevel level, LogStage stage, int verbose){
  Log(std::string(message), level, stage, verbose);
}

Log::Log(std::string message){
  // Always print these kinds of messages.
  Log(message, LogLevel::Undefined, LogStage::Undefined, -1);
}

Log::Log(std::stringstream& messagestream){
  Log(messagestream.str());
}

Log::Log(const char* message){
  Log(std::string(message));
}



/**
 * Get the name of the given stage.
 */
const char* Log::getStageName(LogStage stage){

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

