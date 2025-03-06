#include "Gas.h"


// Stuff for primitive state

/**
 * Constructors
 */
idealGas::PrimitiveState::PrimitiveState():
  rho(0.),
  v({0., 0.}),
  p(0.) { }

idealGas::PrimitiveState::PrimitiveState(const float_t rho, const std::array<float_t,2> vel, const float_t p) :
  rho(rho), v({vel[0], vel[1]}), p(p){ }


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


// Stuff for conserved state

idealGas::ConservedState::ConservedState():
  // initialiser list
  rho(0.),
  rhov({0., 0.}),
  E(0.) {
    // empty body...
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

  float_t rhoflux = p.getRho() * p.getV(dimension);
  setRho(rhoflux);

  // momentum flux along the requested dimension
  float_t momentum_dim = p.getRho() * p.getV(dimension) * p.getV(dimension) + p.getP();
  setRhov(dimension, momentum_dim);

  // momentum flux along the other dimension
  float_t momentum_other = p.getRho() * p.getV(0) * p.getV(1);
  setRhov((dimension + 1) % 2, momentum_other);

  // gas energy
  float_t E     = 0.5 * p.getRho() * p.getVSquared() + p.getP() / GM1;
  float_t Eflux = (E + p.getP()) * p.getV(dimension);
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
    float_t v = c.getRhov(dimension) / c.getRho();
    float_t p = c.getE() - 0.5 * c.getRhoVSquared() / c.getRho();

    // momentum flux along the requested dimension
    float_t momentum_dim = c.getRho() * v * v + p;
    setRhov(dimension, momentum_dim);

    // momentum flux along the other dimension
    size_t  other_index    = (dimension + 1) % 2;
    float_t momentum_other = c.getRhov(other_index) * v;
    setRhov(other_index, momentum_other);

    float_t E = (c.getE() + p) * v;
    setE(E);

  } else {

    setRhov(0, 0.);
    setRhov(1, 0.);
    setE(0.);
  }
}
