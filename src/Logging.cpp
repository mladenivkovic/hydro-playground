#include "Logging.h"

#include "Version.h"


#include <iostream>
#include <sstream>
#include <string>

namespace hydro_playground {
namespace logging {

/**
* write a log message.
*
* @param message The message you want to print out.
* @param level "verbosity level" of the log message.
* @param stage stage of the code where this log is called from.
* @param verbose how verbose the code is being run.
*/
Log::Log(std::string message, LogLevel level, LogStage stage, int verbose){

  // Are we talkative enough?
  if (verbose < level) return;

  std::stringstream str;
  str << "[" << getStageName(stage) << "] ";
  str << message << "\n";

  std::cout << str.str() ;

  // Do we want the message to be instantly flushed to screen?
  bool flush = level >= LogLevelDebug;
  if (flush) std::cout << std::flush;
}

Log::Log(std::stringstream messagestream, LogLevel level, LogStage stage, int verbose){
  Log(messagestream.str(), level, stage, verbose);
}


/**
 * Get the name of the given stage.
 */
const char* Log::getStageName(LogStage stage){

  switch (stage) {
    case LogStageUndefined:
      return "Undefined";
    case LogStageHeader:
      return "Header";
    case LogStageInit:
      return "Init";
    case LogStageStep:
      return "Step";
    case LogStageCount:
      return "Count";
    default:
      return "Unknown";
  }
}



} // namespace logging
} // namespace hydro_playground

