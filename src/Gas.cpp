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
idealGas::PrimitiveState::PrimitiveState( const Float rho, const Float vx, const Float vy, const Float p) {
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
void idealGas::PrimitiveState::fromCons(const ConservedState& cons) {
  if (cons.getRho() <= cst::SMALLRHO) {
    // execption handling for vacuum
    setRho(cst::SMALLRHO);
    setV(0, cst::SMALLV);
    setV(1, cst::SMALLV);
    setP(cst::SMALLP);
  } else {
    setRho(cons.getRho());
    setV(0, cons.getRhov(0) / cons.getRho());
    setV(1, cons.getRhov(1) / cons.getRho());
    setP(cst::GM1 * cons.getE() - 0.5 * cons.getRhoVSquared() / cons.getRho());

    // handle negative pressure
    if (getP() <= cst::SMALLP) {
      setP(cst::SMALLP);
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


// ------------------------------------
// Stuff for conserved state
// ------------------------------------


idealGas::ConservedState::ConservedState():
  _rho(0.),
  _energy(0.) {
  for (size_t i = 0; i < Dimensions; i++) {
    _rhov[i] = 0.;
  }
}

idealGas::ConservedState::ConservedState(const Float rho, const Float rhovx, const Float rhovy, const Float E):
  _rho(rho),
  _energy(E)
{
  if (Dimensions != 2) error("This is for 2D only!");
  _rhov[0] = rhovx;
  _rhov[1] = rhovy;
}


/**
 * Compute the conserved state vector of a given primitive state.
 */
void idealGas::ConservedState::fromPrim(const PrimitiveState& p) {
  setRho(p.getRho());
  setRhov(0, p.getRho() * p.getV(0));
  setRhov(1, p.getRho() * p.getV(1));
  setE(0.5 * p.getRho() * p.getVSquared() + p.getP() / cst::GM1);
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
 *
 * TODO: make sure latex documentation has these equations
 */
void idealGas::ConservedState::getCFluxFromPstate(const PrimitiveState& prim, const size_t dimension) {

  Float rhoflux = prim.getRho() * prim.getV(dimension);
  setRho(rhoflux);

  // momentum flux along the requested dimension
  Float momentum_dim = prim.getRho() * prim.getV(dimension) * prim.getV(dimension) + prim.getP();
  setRhov(dimension, momentum_dim);

  // momentum flux along the other dimension
  Float momentum_other = prim.getRho() * prim.getV(0) * prim.getV(1);
  setRhov((dimension + 1) % 2, momentum_other);

  // gas energy
  Float E     = 0.5 * prim.getRho() * prim.getVSquared() + prim.getP() / cst::GM1;
  Float Eflux = (E + prim.getP()) * prim.getV(dimension);
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
void idealGas::ConservedState::getCFluxFromCstate(const ConservedState& cons, const size_t dimension) {

  // Mass flux
  setRho(cons.getRhov(dimension));

  if (cons.getRho() > 0.) {
    Float v = cons.getRhov(dimension) / cons.getRho();
    Float p = cons.getE() - 0.5 * cons.getRhoVSquared() / cons.getRho();

    // momentum flux along the requested dimension
    Float momentum_dim = cons.getRho() * v * v + p;
    setRhov(dimension, momentum_dim);

    // momentum flux along the other dimension
    size_t other_index    = (dimension + 1) % 2;
    Float  momentum_other = cons.getRhov(other_index) * v;
    setRhov(other_index, momentum_other);

    Float E = (cons.getE() + p) * v;
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
