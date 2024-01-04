#pragma once

#include <iostream>
#include <sstream>
#include <string>

// Utilities related to logging.


namespace hydro_playground {
  namespace logging {

enum LogLevel {
  LogLevelUndefined = -1,
  LogLevelQuiet = 0,
  LogLevelVerbose = 1,
  LogLevelDebug = 2,
  LogLevelCount
};


enum LogStage {
  LogStageUndefined = -1,
  LogStageHeader = 0,
  LogStageInit = 1,
  LogStageStep = 2,
  LogStageCount,
};


class Log {
  public:
    Log(std::string message, LogLevel level, LogStage stage, int verbose = LogLevelCount);
    Log(std::stringstream messagestream, LogLevel level, LogStage stage, int verbose = LogLevelCount);

  private:
    const char* getStageName(LogStage stage);
};


// std::string get_stage_name(log_stage stage);


  } // namespace logging
} // namespace hydro_playground
