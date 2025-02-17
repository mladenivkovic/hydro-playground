#include "Utils.h"

#include <iostream>
#include <sstream>

#include "Config.h"
#include "Logging.h"
#include "Version.h"


/**
 * @brief returns the banner for the header.
 */
std::stringstream utils::get_banner() {

  std::stringstream banner;
  banner
    << "o  o o   o o-o   o--o   o-o      o--o  o      O  o   o  o-o  o--o   o-o  o   o o   o o-o   \n";
  banner
    << "|  |  \\ /  |  \\  |   | o   o     |   | |     / \\  \\ /  o     |   | o   o |   | |\\  | |  \\  \n";
  banner
    << "O--O   O   |   O O-Oo  |   |     O--o  |    o---o  O   |  -o O-Oo  |   | |   | | \\ | |   O \n";
  banner
    << "|  |   |   |  /  |  \\  o   o     |     |    |   |  |   o   | |  \\  o   o |   | |  \\| |  /  \n";
  banner
    << "o  o   o   o-o   o   o  o-o      o     O---oo   o  o    o-o  o   o  o-o   o-o  o   o o-o   \n";
  banner << "\n";

  return banner;
}


/**
 * @brief Prints out the header at the start of the run.
 */
void utils::print_header() {

  std::stringstream banner = get_banner();
  std::cout << banner.str();

  const int version_major = version::Version::MAJOR;
  const int version_minor = version::Version::MINOR;

  logging::LogStage stage = logging::LogStage::Header;
  logging::LogLevel level = logging::LogLevel::Quiet;

  std::stringstream version_txt;
  version_txt << "Version:     " << version_major << "." << version_minor;
  message(version_txt, level, stage);

  std::stringstream git_branch_txt;
  git_branch_txt << "Git branch:  " << version::Version::GIT_BRANCH;
  message(git_branch_txt, level, stage);

  std::stringstream git_comm_txt;
  git_comm_txt << "Git commit:  " << version::Version::GIT_SHA1;
  message(git_comm_txt, level, stage);

  std::stringstream build_type;
  build_type << "Build type:  " << CMAKE_BUILD_TYPE;
  message(build_type, level, stage);

  message("Build date:  " __DATE__ " - " __TIME__, level, stage);

  std::stringstream debug;
  debug << "Debug level: " << DEBUG_LEVEL;
  message(debug, level, stage);

#if DEBUG_LEVEL > 0
  warning("Code compiled with debugging enabled.");
#endif
}
