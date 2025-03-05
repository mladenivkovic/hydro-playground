#include "Gas.h"


// Stuff for primitive state

IdealGas::PrimitiveState::PrimitiveState():
  rho(0.),
  u({0., 0.}),
  p(0.) {
    // empty body...
  };


/**
 * Convert a conserved state to a (this) primitive state.
 * Overwrites the contents of this primitive state.
 */
void IdealGas::PrimitiveState::ConservedToPrimitive(const ConservedState& c) {
  if (c.getRho() <= SMALLRHO) {
    // execption handling for vacuum
    setRho(SMALLRHO);
    setU(0, SMALLU);
    setU(1, SMALLU);
    setP(SMALLP);
  } else {
    setRho(c.getRho());
    setU(0, c.getRhou(0) / c.getRho());
    setU(1, c.getRhou(1) / c.getRho());
    setP(GM1 * c.getE() - 0.5 * c.getRhoUSquared() / c.getRho());

    // handle negative pressure
    if (getP() <= SMALLP) {
      setP(SMALLP);
    }
  }
}



// Stuff for conserved state

IdealGas::ConservedState::ConservedState():
  // initialiser list
  rho(0.),
  rhou({0., 0.}),
  E(0.) {
    // empty body...
  };


/**
 * Compute the conserved state vector of a given primitive state.
 */
void IdealGas::ConservedState::PrimitiveToConserved(const PrimitiveState& p) {
  setRho(p.getRho());
  setRhou(0, p.getRho() * p.getU(0));
  setRhou(1, p.getRho() * p.getU(1));
  setE(0.5 * p.getRho() * p.getUSquared() + p.getP() / GM1);
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
void IdealGas::ConservedState::GetCFluxFromPstate(const PrimitiveState& p, const size_t dimension) {

  float_t rhoflux = p.getRho() * p.getU(dimension);
  setRho(rhoflux);

  // momentum flux along the requested dimension
  float_t momentum_dim = p.getRho() * p.getU(dimension) * p.getU(dimension) + p.getP();
  setRhou(dimension, momentum_dim);

  // momentum flux along the other dimension
  float_t momentum_other = p.getRho() * p.getU(0) * p.getU(1);
  setRhou((dimension + 1) % 2, momentum_other);

  // gas energy
  float_t E     = 0.5 * p.getRho() * p.getUSquared() + p.getP() / GM1;
  float_t Eflux = (E + p.getP()) * p.getU(dimension);
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
void IdealGas::ConservedState::GetCFluxFromCstate(const ConservedState& c, const size_t dimension) {

  // Mass flux
  setRho(c.getRhou(dimension));

  if (c.getRho() > 0.) {
    float_t v = c.getRhou(dimension) / c.getRho();
    float_t p = c.getE() - 0.5 * c.getRhoUSquared() / c.getRho();

    // momentum flux along the requested dimension
    float_t momentum_dim = c.getRho() * v * v + p;
    setRhou(dimension, momentum_dim);

    // momentum flux along the other dimension
    size_t  other_index    = (dimension + 1) % 2;
    float_t momentum_other = c.getRhou(other_index) * v;
    setRhou(other_index, momentum_other);

    float_t E = (c.getE() + p) * v;
    setE(E);

  } else {

    setRhou(0, 0.);
    setRhou(1, 0.);
    setE(0.);
  }
}

