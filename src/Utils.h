/**
 * @file Utils.h
 * @brief Misc utils that don't fit anywhere else
 */

#pragma once

#include <sstream>
#include <string>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "Config.h"

namespace utils {

  //! Stores the actual string of the banner image
  std::stringstream banner();

  //! Prints out the header and banner
  void printHeader();

  //! Is the line whitespace only?
  bool isWhitespace(std::string& line);

  //! Is this line a comment?
  bool isComment(std::string& line);

  //! Remove leading and trailing whitespaces from a string.
  std::string removeWhitespace(std::string& str);

  //! Split a line at an = char. Raise warnings if warn=true and something is amiss.
  std::pair<std::string, std::string> splitEquals(std::string& str, bool warn = false);

  //! Remove trailing comment from a line
  std::string removeTrailingComment(std::string& line);

  //! Get a string representing something gone wrong in parsing/evaluation
  std::string somethingWrong();

  //! Does a file exist?
  bool fileExists(const std::string& filename);

  //! Convert value string to integer. Do some additional sanity checks too.
  int string2int(std::string& val);

  //! Convert value string to size_t. Do some additional sanity checks too.
  size_t string2size_t(std::string& val);

  //! Convert value string to float/double. Do some additional sanity checks too.
  Float string2float(std::string& val);

  //! Convert value string to integer. Do some additional sanity checks too.
  bool string2bool(std::string& val);

  //! "Convert" value string to string. Basically just do some additional sanity checks.
  std::string string2string(std::string val);

  //! Get solver name from macro.
  const char* getSolverName();

  //! Get Riemann solver name from macro.
  const char* getRiemannSolverName();

  //! Get limiter name from macro.
  const char* getLimiterName();
} // namespace utils


// hacky way of getting this definition in
#ifdef USE_CUDA

#define HOST        __host__
#define DEVICE      __device__
#define HOST_DEVICE __host__ __device__

#else

#define HOST       
#define DEVICE     
#define HOST_DEVICE

#endif
