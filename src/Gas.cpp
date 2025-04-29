#include "Gas.h"

#include <iomanip>

#include "Constants.h"
#include "Logging.h"


static constexpr int gas_print_width     = 5;
static constexpr int gas_print_precision = 2;


// Stuff for primitive state

/**
 * @brief Default constructor.
 */
PrimitiveState::PrimitiveState():
  _rho(0.),
  _p(0.) {
  for (size_t i = 0; i < Dimensions; i++) {
    _v[i] = 0.;
  }
}

/**
 * @brief Specialized constructor with initial values.
 * Using setters instead of initialiser lists so the debugging checks kick in.
 */
PrimitiveState::PrimitiveState(
  const Float rho, const std::array<Float, Dimensions> vel, const Float p
) {
  setRho(rho);
  for (size_t i = 0; i < Dimensions; i++) {
    setV(i, vel[i]);
  }
  setP(p);
}


/**
 * @brief Specialized constructor with initial values for 1D.
 * Using setters instead of initialiser lists so the debugging checks kick in.
 */
PrimitiveState::PrimitiveState(const Float rho, const Float vx, const Float p) {
#if DEBUG_LEVEL > 0
  if (Dimensions != 1) {
    error("This is a 1D function only!");
  }
#endif
  setRho(rho);
  setV(0, vx);
  setP(p);
}

/**
 * @brief Specialized constructor with initial values for 2D.
 * Using setters instead of initialiser lists so the debugging checks kick in.
 */
PrimitiveState::PrimitiveState(const Float rho, const Float vx, const Float vy, const Float p) {
#if DEBUG_LEVEL > 0
  if (Dimensions != 2) {
    error("This is a 2D function only!");
  }
#endif
  setRho(rho);
  setV(0, vx);
  setV(1, vy);
  setP(p);
}



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
 * Initialise a conserved flux along a dimension using primitive variables of
 * the state.
 */
ConservedState::ConservedState(const PrimitiveState& prim, const size_t dimension) {
  // next function undefined in device code. leave this one here
  getCFluxFromPState(prim, dimension);
}



/**
 * @brief Compute the flux of conserved variables of the Euler
 * equations given a primitive state vector
 *
 * The flux is not an entire tensor for 3D Euler equations, but
 * correpsonds to the dimensionally split vectors F, G as
 * described in the "Euler equations in 2D" section of the
 * documentation TeX files.
 * That's why you need to specify the dimension.
 *
 * The flux terms for each dimension are given as the second and
 * third term in Eq. 13.
 */
void ConservedState::getCFluxFromPState(const PrimitiveState& pstate, const size_t dimension) {

  size_t other  = (dimension + 1) % 2;
  Float  rho    = pstate.getRho();
  Float  vdim   = pstate.getV(dimension);
  Float  vother = pstate.getV(other);
  Float  p      = pstate.getP();

  // mass flux
  setRho(rho * vdim);
  // momentum flux along the requested dimension
  setRhov(dimension, rho * vdim * vdim + p);

  // momentum flux along the other dimension
  setRhov(other, rho * vdim * vother);

  // gas energy flux
  Float E = pstate.getE();
  setE((E + p) * vdim);
}


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
