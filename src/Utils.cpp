#include "config.h"
// #include "defines.h"
#include "utils.h"

#include <iostream>
#include <sstream>
// #include <iostream>


namespace hydro_playground {

/**
 * @brief returns the banner for the header.
 */
std::stringstream utils::get_banner(void){

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
void utils::print_header(void){

  const int version_major = HYDRO_PLAYGROUND_VERSION_MAJOR;
  const int version_minor = HYDRO_PLAYGROUND_VERSION_MINOR;

  std::stringstream header;
  std::stringstream banner = get_banner();

  header << banner.str();

  header << "Version " << version_major << "." << version_minor;



  std::cout << header.str() << std::endl;

}

} // namespace hydro_playground

