// #include "defines.h"
#include "Utils.h"

#include <iostream>
#include <sstream>

#include "Logging.h"
#include "Version.h"
// #include <iostream>


namespace hydro_playground {

  /**
   * @brief returns the banner for the header.
   */
  std::stringstream utils::get_banner(void) {

    std::stringstream banner;
    banner << "o  o o   o o-o   o--o   o-o      o--o  o      O  o   o  o-o  o--o   o-o  o   o o   o o-o   \n";
    banner << "|  |  \\ /  |  \\  |   | o   o     |   | |     / \\  \\ /  o     |   | o   o |   | |\\  | |  \\  \n";
    banner << "O--O   O   |   O O-Oo  |   |     O--o  |    o---o  O   |  -o O-Oo  |   | |   | | \\ | |   O \n";
    banner << "|  |   |   |  /  |  \\  o   o     |     |    |   |  |   o   | |  \\  o   o |   | |  \\| |  /  \n";
    banner << "o  o   o   o-o   o   o  o-o      o     O---oo   o  o    o-o  o   o  o-o   o-o  o   o o-o   \n";
    banner << "\n";

    return banner;
  }


  /**
   * @brief Prints out the header at the start of the run.
   */
  void utils::print_header(void) {

    std::stringstream banner = get_banner();
    std::cout << banner.str();

    const int version_major = version::Version::MAJOR;
    const int version_minor = version::Version::MINOR;

    logging::LogStage stage = logging::LogStage::Header;
    logging::LogLevel level = logging::LogLevel::Quiet;

    std::stringstream version_txt;
    version_txt << "Version:     " << version_major << "." << version_minor;
    logging::Log(version_txt, level, stage);

    std::stringstream git_branch_txt;
    git_branch_txt << "Git branch:  " << version::Version::GIT_BRANCH;
    logging::Log(git_branch_txt, level, stage);

    std::stringstream git_comm_txt;
    git_comm_txt << "Git commit:  " << version::Version::GIT_SHA1;
    logging::Log(git_comm_txt, level, stage);
  }


} // namespace hydro_playground
