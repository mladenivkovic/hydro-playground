#include "Gas.h"

#include <iomanip>

#include "Logging.h"


static constexpr int gas_print_width     = 5;
static constexpr int gas_print_precision = 2;


// Stuff for primitive state

/**
 * @brief Default constructor.
 */
idealGas::PrimitiveState::PrimitiveState():
  rho(0.),
  p(0.) {
  for (size_t i = 0; i < Dimensions; i++) {
    v[i] = 0.;
  }
}

/**
 * @brief Specialized constructor with initial values.
 * Using setters instead of initialiser lists so the debugging checks kick in.
 */
idealGas::PrimitiveState::PrimitiveState(
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
idealGas::PrimitiveState::PrimitiveState(const Float rho, const Float vx, const Float p) {
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
idealGas::PrimitiveState::PrimitiveState(
  const Float rho, const Float vx, const float vy, const Float p
) {
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
 * Convert a conserved state to a (this) primitive state.
 * Overwrites the contents of this primitive state.
 */
void idealGas::PrimitiveState::ConservedToPrimitive(const ConservedState& c) {
  if (c.getRho() <= SMALLRHO) {
    // execption handling for vacuum
    setRho(SMALLRHO);
    setV(0, SMALLU);
    setV(1, SMALLU);
    setP(SMALLP);
  } else {
    setRho(c.getRho());
    setV(0, c.getRhov(0) / c.getRho());
    setV(1, c.getRhov(1) / c.getRho());
    setP(GM1 * c.getE() - 0.5 * c.getRhoVSquared() / c.getRho());

    // handle negative pressure
    if (getP() <= SMALLP) {
      setP(SMALLP);
    }
  }
}


/**
 * @brief construct a string with the contents.
 * Format: [rho, vx, vy, P]
 */
std::string idealGas::PrimitiveState::toString() const {

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


// Stuff for conserved state

idealGas::ConservedState::ConservedState():
  rho(0.),
  E(0.) {
  for (size_t i = 0; i < Dimensions; i++) {
    rhov[i] = 0.;
  }
};


/**
 * Compute the conserved state vector of a given primitive state.
 */
void idealGas::ConservedState::PrimitiveToConserved(const PrimitiveState& p) {
  setRho(p.getRho());
  setRhov(0, p.getRho() * p.getV(0));
  setRhov(1, p.getRho() * p.getV(1));
  setE(0.5 * p.getRho() * p.getVSquared() + p.getP() / GM1);
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
 * TODO: make sure latex documentation has these equations
 */
void idealGas::ConservedState::GetCFluxFromPstate(const PrimitiveState& p, const size_t dimension) {

  Float rhoflux = p.getRho() * p.getV(dimension);
  setRho(rhoflux);

  // momentum flux along the requested dimension
  Float momentum_dim = p.getRho() * p.getV(dimension) * p.getV(dimension) + p.getP();
  setRhov(dimension, momentum_dim);

  // momentum flux along the other dimension
  Float momentum_other = p.getRho() * p.getV(0) * p.getV(1);
  setRhov((dimension + 1) % 2, momentum_other);

  // gas energy
  Float E     = 0.5 * p.getRho() * p.getVSquared() + p.getP() / GM1;
  Float Eflux = (E + p.getP()) * p.getV(dimension);
  setE(Eflux);
}


/**
 * Compute the flux of conserved variables of the Euler
 * equations given a conserved state vector
 *
 * The flux is not an entire tensor for 3D Euler equations, but
 * correpsonds to the dimensionally split vectors F, G as
 * described in the "Euler equations in 2D" section of the
 * documentation TeX files.
 * That's why you need to specify the dimension.
 *
 * Everything about this is copied from the original C code.
 * For now, just need to get the machinery in before making
 * it C++ style.
 *
 * TODO: make sure latex documentation has these equations
 */
void idealGas::ConservedState::GetCFluxFromCstate(const ConservedState& c, const size_t dimension) {

  // Mass flux
  setRho(c.getRhov(dimension));

  if (c.getRho() > 0.) {
    Float v = c.getRhov(dimension) / c.getRho();
    Float p = c.getE() - 0.5 * c.getRhoVSquared() / c.getRho();

    // momentum flux along the requested dimension
    Float momentum_dim = c.getRho() * v * v + p;
    setRhov(dimension, momentum_dim);

    // momentum flux along the other dimension
    size_t  other_index    = (dimension + 1) % 2;
    Float momentum_other = c.getRhov(other_index) * v;
    setRhov(other_index, momentum_other);

    Float E = (c.getE() + p) * v;
    setE(E);

  } else {

    setRhov(0, 0.);
    setRhov(1, 0.);
    setE(0.);
  }
}


/**
 * @brief construct a string with the contents.
 * Format: [rho, rho * vx, rho * vy, E]
 */
std::string idealGas::ConservedState::toString() const {

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
