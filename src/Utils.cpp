#include "Utils.h"

#include <filesystem> // std::filesytem::exists
#include <iomanip>
#include <iostream>
#include <sstream>

#include "Cell.h"
#include "Config.h"
#include "Logging.h"
#include "Termcolors.h"
#include "Version.h"


/**
 * @brief returns the banner for the header.
 */
std::stringstream utils::banner() {

  std::stringstream banner;

  if (color_term)
    banner << tcols::cyan;
  banner << "o  o o   o o-o   o--o   o-o      o--o  o      ";
  banner << "O  o   o  o-o  o--o   o-o  o   o o   o o-o   \n";
  if (color_term)
    banner << tcols::reset;

  if (color_term)
    banner << tcols::green;
  banner << "|  |  \\ /  |  \\  |   | o   o     |   | |     ";
  banner << "/ \\  \\ /  o     |   | o   o |   | |\\  | |  \\  \n";
  if (color_term)
    banner << tcols::reset;

  if (color_term)
    banner << tcols::yellow;
  banner << "O--O   O   |   O O-Oo  |   |     O--o  |    ";
  banner << "o---o  O   |  -o O-Oo  |   | |   | | \\ | |   O \n";
  if (color_term)
    banner << tcols::reset;

  if (color_term)
    banner << tcols::red;
  banner << "|  |   |   |  /  |  \\  o   o     |     |    ";
  banner << "|   |  |   o   | |  \\  o   o |   | |  \\| |  /  \n";
  if (color_term)
    banner << tcols::reset;

  if (color_term)
    banner << tcols::magenta;
  banner << "o  o   o   o-o   o   o  o-o      o     O---oo ";
  banner << "  o  o    o-o  o   o  o-o   o-o  o   o o-o   \n";
  if (color_term)
    banner << tcols::reset;

  banner << "\n";

  return banner;
}


/**
 * @brief Prints out the header at the start of the run.
 */
void utils::printHeader() {

  std::stringstream ban = banner();
  std::cout << ban.str();

  const int version_major = version::Version::MAJOR;
  const int version_minor = version::Version::MINOR;

  // Print this out even for the quiet runs.
  logging::LogLevel level = logging::LogLevel::Quiet;

  constexpr int w = 20;

  std::stringstream version_txt;
  version_txt << std::setw(w) << std::left;
  version_txt << "Version:" << version_major << "." << version_minor;
  message(version_txt.str(), level);

  std::stringstream git_branch_txt;
  git_branch_txt << std::setw(w) << std::left;
  git_branch_txt << "Git branch:" << version::Version::GIT_BRANCH;
  message(git_branch_txt.str(), level);

  std::stringstream git_comm_txt;
  git_comm_txt << std::setw(w) << std::left;
  git_comm_txt << "Git commit:" << version::Version::GIT_SHA1;
  message(git_comm_txt.str(), level);

  std::stringstream build_type;
  build_type << std::setw(w) << std::left;
  build_type << "Build type:" << CMAKE_BUILD_TYPE;
  message(build_type.str(), level);

  // Note this only displays the build date of Utils.cpp.o
  std::stringstream build_date;
  build_date << std::setw(w) << std::left;
  build_date << "Build date:" << __DATE__ << " - " << __TIME__;
  message(build_date.str(), level);

  std::stringstream solver;
  solver << std::setw(w) << std::left;
  solver << "Hydro solver: " << getSolverName();
  message(solver.str(), level);

  std::stringstream riemann;
  riemann << std::setw(w) << std::left;
  riemann << "Riemann solver: " << getRiemannSolverName();
  message(riemann.str(), level);

  std::stringstream limiter;
  limiter << std::setw(w) << std::left;
  limiter << "Limiter: " << getLimiterName();
  message(limiter.str(), level);

  std::stringstream debug;
  debug << std::setw(w) << std::left;
  debug << "Debug level: " << DEBUG_LEVEL;
  message(debug.str(), level);

  std::stringstream cellsize;
  cellsize << std::setw(w) << std::left;
  cellsize << "sizeof(cell::Cell): " << sizeof(Cell);
  message(cellsize.str());

#if DEBUG_LEVEL > 0
  warning("Code compiled with debugging enabled.");
#endif
}


/**
 * String signifying something's gone wrong.
 */
std::string utils::somethingWrong() {
  return std::string("__something_wrong__");
}


/**
 * Scan through the line buffer. If we see any character that isn't `\n`,
 * space, EOF or null then return false
 */
bool utils::isWhitespace(std::string& line) {
  for (const auto s : line) {
    if ((std::isspace(s) == 0) and (s != EOF) and static_cast<bool>(s)) {
      return false;
    }
  }
  return true;
}


/**
 * Scan past all the spaces, if the first non-space chars you see are // or
 * / *, then return true.
 */
bool utils::isComment(std::string& line) {

  for (auto s = line.cbegin(); s != line.cend(); s++) {
    if (std::isspace(*s) != 0) {
      // skip leading spaces
      continue;
    }
    if (*s == '/') {
      auto next = s + 1;
      return ((next != line.cend()) and ((*next == '/') or (*next == '*')));
    }
    return false;
  }
  return false;
}


/**
 * Remove leading and trailing whitespaces from a string.
 */
std::string utils::removeWhitespace(std::string& str) {

  if (str.size() == 0) {
    return str;
  }
  if (str.size() == 1) {
    if (str == " " or str == "\t" or str == "\n" or str == "\r" or str == "\f" or str == "\v") {
      return "";
    }
    return str;
  }

  std::string ltrim;
  std::string rtrim;

  size_t first = str.find_first_not_of(" \t\n\r\f\v");
  if (first < str.size()) {
    ltrim = str.erase(0, first);
  } else {
    ltrim = str;
  }

  size_t last = ltrim.find_last_not_of(" \t\n\r\f\v");
  if (last < str.size()) {
    // last + 1: start at index after last non-whitespace
    rtrim = ltrim.erase(last + 1, ltrim.size() - last);
  } else {
    rtrim = ltrim;
  }

  return rtrim;
}


/**
 * Split a string at an equality character.
 * Returns a pair of strings of the content (before, after)
 * the equality sign.
 * Returns internal::somethingWrong() if there is no or more than
 * one equality sign in the string.
 */
std::pair<std::string, std::string> utils::splitEquals(std::string& str, bool warn) {

  // Find where the equals sign is
  size_t equals_ind = 0;
  int    count      = 0;
  for (size_t i = 0; i < str.size(); i++) {
    if (str[i] == '=') {
      count++;
      if (equals_ind == 0)
        equals_ind = i;
    }
  }

  if (count > 1) {
    if (warn)
      warning("Got more than 1 equality sign in line '" + str + "'");
    return std::make_pair(utils::somethingWrong(), utils::somethingWrong());
  }

  if (equals_ind == 0) {
    if (warn)
      warning("No equality sign or no var name in line '" + str + "'");
    return std::make_pair(utils::somethingWrong(), utils::somethingWrong());
  }

  if (equals_ind == str.size()) {
    if (warn)
      warning("No var value in line '" + str + "'");
    return std::make_pair(utils::somethingWrong(), utils::somethingWrong());
  }

  std::string name = str.substr(0, equals_ind);
  std::string val  = str.substr(equals_ind + 1, str.size() - equals_ind - 1);

  return std::make_pair(removeWhitespace(name), removeWhitespace(val));
}


/**
 * Remove trailing comment from line.
 * Comment is / * or //.
 */
std::string utils::removeTrailingComment(std::string& line) {

  size_t comm1 = line.find("//");
  size_t comm2 = line.find("/*");
  // get the earlier of both
  size_t comm = std::min(comm1, comm2);
  // make sure we found at least one
  comm = std::min(comm, line.size());

  return line.substr(0, comm);
}


/**
 * Does a file exist?
 */
bool utils::fileExists(const std::string& filename) {
  return std::filesystem::exists(filename);
}


/**
 * Convert value string to size_t.
 * Do some additional sanity checks too.
 */
size_t utils::string2size_t(std::string& val) {
  std::string v = removeWhitespace(val);
  if (v.size() == 0) {
    std::stringstream msg;
    msg << "Invalid string to convert to size_t: '" << val << "'";
    error(msg.str());
  }

  size_t out = std::stoull(v);
  return out;
}


/**
 * Convert value string to integer.
 * Do some additional sanity checks too.
 */
int utils::string2int(std::string& val) {
  std::string v = removeWhitespace(val);
  if (v.size() == 0) {
    std::stringstream msg;
    msg << "Invalid string to convert to int: '" << val << "'";
    error(msg.str());
  }

  // todo: error catching
  int out = std::stoi(v);
  return out;
}


/**
 * Convert value string to float/double.
 * Do some additional sanity checks too.
 */
Float utils::string2float(std::string& val) {

  std::string v = removeWhitespace(val);
  if (v.size() == 0) {
    std::stringstream msg;
    msg << "Invalid string to convert to int: '" << val << "'";
    error(msg.str());
  }

  auto out = static_cast<Float>(std::stof(v));
  return out;
}


/**
 * Convert value string to integer.
 * Do some additional sanity checks too.
 */
bool utils::string2bool(std::string& val) {

  std::string v = removeWhitespace(val);

  if ((v == "true") or (v == "True") or (v == "TRUE") or (v == "1")) {
    return true;
  }
  if ((v == "false") or (v == "False") or (v == "FALSE") or (v == "0")) {
    return false;
  }

  std::stringstream msg;
  msg << "Invalid bool string '" << val << "'";
  error(msg.str());
  return false;
}


/**
 * "Convert" value string to string.
 * Basically just do some additional sanity checks.
 */
std::string utils::string2string(std::string val) {

  std::string v = removeWhitespace(val);
  if (v.size() == 0) {
    std::stringstream msg;
    msg << "Suspicious string: '" << val << "'";
    warning(msg.str());
  }

  return v;
}


/**
 * Get solver name from macro.
 */
const char* utils::getSolverName() {

#if SOLVER == SOLVER_GODUNOV
  return "Godunov";
#elif SOLVER == SOLVER_MUSCL
  return "MUSCL";
#else
#error Invalid Hydro Solver
  return "UndefinedSolver";
#endif
}


/**
 * Get Riemann solver name from macro.
 */
const char* utils::getRiemannSolverName() {

#if RIEMANN_SOLVER == RIEMANN_SOLVER_EXACT
  return "Exact";
#elif RIEMANN_SOLVER == RIEMANN_SOLVER_HLLC
  return "HLLC";
#else
#error Invalid Riemann Solver
  return "UndefinedRiemannSolver";
#endif
}


/**
 * Get limiter name from macro.
 */
const char* utils::getLimiterName() {

#if LIMITER == LIMITER_MINMOD
  return "Minmod";
#elif LIMITER == LIMITER_VANLEER
  return "VanLeer";
#else
#error Invalid Limiter
  return "UndefinedLimiter";
#endif
}
