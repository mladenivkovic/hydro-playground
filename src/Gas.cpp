#include "Gas.h"

#include <iomanip>

#include "Constants.h"
#include "Logging.h"


static constexpr int gas_print_width     = 5;
static constexpr int gas_print_precision = 2;


// Stuff for primitive state


/**
 * @brief construct a string with the contents.
 * Format: [rho, vx, vy, P]
 */
std::string PrimitiveState::toString() const {

  constexpr int w = gas_print_width;
  constexpr int p = gas_print_precision;

  std::stringstream out;
  out << "[";
  out << std::setprecision(p) << std::setw(w) << getRho() << ",";
  for (size_t i = 0; i < Dimensions; i++) {
    out << std::setprecision(p) << std::setw(w) << getV(i) << ",";
  }
  out << std::setprecision(p) << std::setw(w) << getP() << "]";

  return out.str();
}


// ------------------------------------
// Stuff for conserved state
// 
// MOVED A FEW OF THE CONSTRUCTORS INTO THE HEADER TO SIMPLIFY THE DEVICE CODE
// ------------------------------------


/**
 * @brief construct a string with the contents.
 * Format: [rho, rho * vx, rho * vy, E]
 */
std::string ConservedState::toString() const {

  constexpr int w = gas_print_width;
  constexpr int p = gas_print_precision;

  std::stringstream out;
  out << "[";
  out << std::setprecision(p) << std::setw(w) << getRho() << ",";
  for (size_t i = 0; i < Dimensions; i++) {
    out << std::setprecision(p) << std::setw(w) << getRhov(i) << ",";
  }
  out << std::setprecision(p) << std::setw(w) << getE() << "]";

  return out.str();
}
