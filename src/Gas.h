#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <string>

#include "Config.h"
#include "Constants.h"
#include "Logging.h"
#include "Utils.h"


class ConservedState;
class PrimitiveState;

// Aliases for clarity. The states and fluxes will have the same components,
// so we can use the same data structure. But this aliasing should make things
// more clear.
using ConservedFlux = ConservedState;
using PrimitiveFlux = PrimitiveState;


/**
 * @brief Holds a primitive state (density, velocity, pressure)
 */
class PrimitiveState {
private:
  //! density
  Float _rho;

  //! velocity
  // std::array<Float, Dimensions> _v;

  // really sorry if you are reading this Mladen
  Float _v[Dimensions];

  //! pressure
  Float _p;


public:
  __host__ __device__ PrimitiveState();
  __host__            PrimitiveState(const Float rho, const std::array<Float, Dimensions> vel, const Float p);
  __host__ __device__ PrimitiveState(const Float rho, const Float vx, const Float p);
  __host__ __device__ PrimitiveState(const Float rho, const Float vx, const Float vy, const Float p);

  /**
   * Clear out contents.
   */
  __host__ __device__ void clear() {
    #if __CUDA_ARCH__
    _rho  = 0.;
    _p    = 0.;
    _v[0] = 0.;
    _v[1] = 0.;
    #else
    *this = PrimitiveState();
    #endif
  }

  /**
   * Set the current primitive state vector to equivalend of given conserved
   * state.
   */
  __host__ __device__ void fromCons(const ConservedState& cons);

  //! Get the local soundspeed given a primitive state
  __host__ __device__ [[nodiscard]] Float getSoundSpeed() const;

  //! Get the total gas energy from a primitive state
  __host__ __device__ [[nodiscard]] Float getE() const;

  //! Get a string of the state.
  [[nodiscard]] std::string toString() const;


  // Getters and setters!

  // Setter for Rho
  __host__ __device__ void                setRho(const Float val);
  __host__ __device__ [[nodiscard]] Float getRho() const;

  // same for u
  __host__ __device__ void                setV(const std::size_t index, const Float val);
  __host__ __device__ [[nodiscard]] Float getV(const std::size_t index) const;

  // used a lot, made a function for it
  __host__ __device__ [[nodiscard]] Float getVSquared() const;

  __host__ __device__ void                setP(const Float val);
  __host__ __device__ [[nodiscard]] Float getP() const;
};


/**
 * @brief Holds a conserved state (density, momentum, energy)
 */
class ConservedState {
private:
  //! Density
  Float _rho;

  //! Momentum: rho * v
  // std::array<Float, Dimensions> _rhov;
  Float _rhov[Dimensions];

  //! Energy
  Float _energy;

public:
  // Standard constructor, init variables to 0
  __host__ __device__ ConservedState();
  __host__ __device__ ConservedState(const Float rho, const Float rhovx, const Float rhovy, const Float E);
  __host__ __device__ explicit ConservedState(const PrimitiveState& prim, const size_t dimension);

  /**
   * Clear out contents.
   */
  __host__ __device__ void clear() {
    #if __CUDA_ARCH__
    _rho     = 0.;
    _energy  = 0.;
    _rhov[0] = 0.;
    _rhov[1] = 0.;
    #else
    *this = ConservedState();
    #endif
  }


  /**
   * Set the current conserved state vector to equivalent of given primitive
   * state.
   */
  __host__ __device__ void fromPrim(const PrimitiveState& prim);


  /**
   * Compute the flux of conserved variables of the Euler
   * equations given a primitive variable state vector
   */
  __host__ __device__ void getCFluxFromPState(const PrimitiveState& pstate, const std::size_t dimension);


  /**
   * Compute the flux of conserved variables of the Euler
   * equations given a conserved state vector
   */
  __host__ __device__ void getCFluxFromCstate(const ConservedState& cstate, const std::size_t dimension);


  //! Get a string of the state.
  [[nodiscard]] std::string toString() const;


  // Getters and setters!
  __host__ __device__               void  setRho(const Float val);
  __host__ __device__ [[nodiscard]] Float getRho() const;

  // same for u
  __host__ __device__ void                setRhov(const std::size_t index, const Float val);
  __host__ __device__ [[nodiscard]] Float getRhov(const std::size_t index) const;
  __host__ __device__ [[nodiscard]] Float getRhoVSquared() const;

  __host__ __device__ void                setE(const Float val);
  __host__ __device__ [[nodiscard]] Float getE() const;

  __host__ __device__ [[nodiscard]] Float getP() const;
};


// --------------------------------------------------------
// Definitions
// --------------------------------------------------------

// Primitive State Stuff
// --------------------------


/**
 * @brief Default constructor.
 */
__host__ __device__ inline PrimitiveState::PrimitiveState():
 _rho(0.),
 _p(0.) {
 for (size_t i = 0; i < Dimensions; i++) {
   _v[i] = 0.;
 }
}

/**
* @brief Specialized constructor with initial values.
* Using setters instead of initialiser lists so the debugging checks kick in.
* 
*/
__host__ inline PrimitiveState::PrimitiveState(
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
__host__ __device__ inline PrimitiveState::PrimitiveState(const Float rho, const Float vx, const Float p) {
#if DEBUG_LEVEL > 0 && !__CUDA_ARCH__
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
__host__ __device__ inline PrimitiveState::PrimitiveState(const Float rho, const Float vx, const Float vy, const Float p) {
#if DEBUG_LEVEL > 0 && !__CUDA_ARCH__
 if (Dimensions != 2) {
   error("This is a 2D function only!");
 }
#endif
 setRho(rho);
 setV(0, vx);
 setV(1, vy);
 setP(p);
}


__host__ __device__ inline void PrimitiveState::setRho(const Float val) {
  // These checks will fail because we (ab)use the PrimitiveState
  // as fluxes too, which can be negative
  // #if DEBUG_LEVEL > 0
  //   assert(val >= 0.);
  // #endif
  _rho = val;
}


__host__ __device__ inline Float PrimitiveState::getRho() const {
  // These checks will fail because we (ab)use the PrimitiveState
  // as fluxes too, which can be negative
  // #if DEBUG_LEVEL > 0
  //   assert(_rho >= 0.);
  // #endif
  return _rho;
}


__host__ __device__ inline void PrimitiveState::setV(const size_t index, const Float val) {
#if __CUDA_ARCH__
#if DEBUG_LEVEL > 0
  // assert(index >= 0); // always true for unsigned type
  assert(index < Dimensions);
#endif
#endif
  _v[index] = val;
}


__host__ __device__ inline Float PrimitiveState::getV(const size_t index) const {
#if __CUDA_ARCH__
#if DEBUG_LEVEL > 0
  // assert(index >= 0); // always true for unsigned type
  assert(index < Dimensions);
#endif
#endif
  return _v[index];
}


__host__ __device__ inline Float PrimitiveState::getVSquared() const {
  if (Dimensions == 1)
    return _v[0] * _v[0];

  if (Dimensions == 2)
    return _v[0] * _v[0] + _v[1] * _v[1];

  // error("Not implemented");
  // return 0.;
}


__host__ __device__ inline void PrimitiveState::setP(const Float val) {
  // These checks will fail because we (ab)use the PrimitiveState
  // as fluxes too, which can be negative
  // #if DEBUG_LEVEL > 0
  //   assert(val >= 0.);
  // #endif
  _p = val;
}


__host__ __device__ inline Float PrimitiveState::getP() const {
  // These checks will fail because we (ab)use the PrimitiveState
  // as fluxes too, which can be negative
  // #if DEBUG_LEVEL > 0
  //   assert(_p >= 0.);
  // #endif
  return _p;
}


/**
 * Compute the local sound speed given a primitive state.
 * Eq. 6
 */
__host__ __device__ inline Float PrimitiveState::getSoundSpeed() const {
  #if __CUDA_ARCH__
  return sqrtf( cst::GAMMA * getP() / getRho() );
  #else
  return std::sqrt(cst::GAMMA * getP() / getRho());
  #endif 
}


/**
 * Get the total gas energy from a primitive state.
 * Eq. 18
 */
__host__ __device__ inline Float PrimitiveState::getE() const {

  return 0.5 * getRho() * getVSquared() + getP() * cst::ONEOVERGM1;
}


// Conserved State Stuff
// --------------------------


__host__ __device__ inline ConservedState::ConservedState():
  _rho(0.),
  _energy(0.) {
  for (size_t i = 0; i < Dimensions; i++) {
    _rhov[i] = 0.;
  }
}

__host__ __device__ inline ConservedState::ConservedState(
  const Float rho, const Float rhovx, const Float rhovy, const Float E
):
  _rho(rho),
  _energy(E) {
#if DEBUG_LEVEL > 0 && !__CUDA_ARCH__
  if (Dimensions != 2)
    error("This is for 2D only!");
#endif
  _rhov[0] = rhovx;
  _rhov[1] = rhovy;
}


/**
 * Initialise a conserved flux along a dimension using primitive variables of
 * the state.
 */
__host__ __device__ inline ConservedState::ConservedState(const PrimitiveState& prim, const size_t dimension) {
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
__host__ __device__ inline void ConservedState::getCFluxFromPState(const PrimitiveState& pstate, const size_t dimension) {

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
 * Compute the flux of conserved variables of the Euler
 * equations given a conserved state vector
 *
 * The flux is not an entire tensor for 3D Euler equations, but
 * correpsonds to the dimensionally split vectors F, G as
 * described in the "Euler equations in 2D" section of the
 * documentation TeX files.
 * That's why you need to specify the dimension.
 *
 * The flux terms for each dimension are given as the second and
 * third term in Eq. 13.
 *
 * Moved from the cpp file to the header by Sean to make the cuda
 * files compare easilyer
 *
 */
__host__ __device__ inline void ConservedState::getCFluxFromCstate(const ConservedState& cons, const size_t dimension) {

  // Mass flux
  Float rho = cons.getRho();

  if (rho > 0.) {

    setRho(cons.getRhov(dimension));

    size_t other        = (dimension + 1) % 2;
    Float  one_over_rho = 1. / rho;
    Float  vdim         = cons.getRhov(dimension) * one_over_rho;
    Float  p            = cons.getP();

    // momentum flux along the requested dimension
    Float momentum_dim = rho * vdim * vdim + p;

    setRhov(dimension, momentum_dim);

    // momentum flux along the other dimension
    Float momentum_other = cons.getRhov(other) * vdim;
    setRhov(other, momentum_other);

    Float E = (cons.getE() + p) * vdim;
    setE(E);

  } else {

    setRhov(0, 0.);
    setRhov(1, 0.);
    setE(0.);
  }
}



__host__ __device__ inline void ConservedState::setRhov(const size_t index, const Float val) {
#if DEBUG_LEVEL > 0
  // assert(index >= 0); // always true for unsigned type
  assert(index < Dimensions);
#endif
  _rhov[index] = val;
}


__host__ __device__ inline Float ConservedState::getRhov(const size_t index) const {
#if DEBUG_LEVEL > 0
  // assert(index >= 0); // always true for unsigned type
  assert(index < Dimensions);
#endif
  return _rhov[index];
}


__host__ __device__ inline Float ConservedState::getRhoVSquared() const {
  return _rhov[0] * _rhov[0] + _rhov[1] * _rhov[1];
}


__host__ __device__ inline void ConservedState::setE(const Float val) {
  // These checks will fail because we (ab)use the ConservedState
  // as fluxes too, which can be negative
  // #if DEBUG_LEVEL > 0
  //   assert(val >= 0.);
  // #endif
  _energy = val;
}


__host__ __device__ inline Float ConservedState::getE() const {
  // These checks will fail because we (ab)use the ConservedState
  // as fluxes too, which can be negative
  // #if DEBUG_LEVEL > 0
  //   assert(_energy >= 0.);
  // #endif
  return _energy;
}


__host__ __device__ inline Float ConservedState::getRho() const {
  // These checks will fail because we (ab)use the ConservedState
  // as fluxes too, which can be negative
  // #if DEBUG_LEVEL > 0
  //   assert(_rho >= 0.);
  // #endif
  return _rho;
}


__host__ __device__ inline void ConservedState::setRho(const Float val) {
  // These checks will fail because we (ab)use the ConservedState
  // as fluxes too, which can be negative
  // #if DEBUG_LEVEL > 0
  //   assert(val >= 0.);
  // #endif
  _rho = val;
}

__host__ __device__ inline Float ConservedState::getP() const {
  // this makes prim->cons->prim conversion worse due to roundoff errors.
  // Float one_over_rho = 1. / rho;
  // Float rv2 = cons.getRhoVSquared() * one_over_rho;
  // this also makes it worse.
  // return (cst::GM1 * getE() - cst::GM1 * 0.5 * rv2);

  Float rho          = getRho();
  Float one_over_rho = 1. / rho;
  Float vx           = getRhov(0) * one_over_rho;
  Float vy           = getRhov(1) * one_over_rho;
  Float rv2          = rho * (vx * vx + vy * vy);
  return cst::GM1 * (getE() - 0.5 * rv2);
}



/**
 * Compute the conserved state vector of a given primitive state.
 *
 * See eqns. 16 - 18 in theory document.
 */
__host__ __device__ inline
void ConservedState::fromPrim(const PrimitiveState& p) {
  setRho(p.getRho());
  setRhov(0, p.getRho() * p.getV(0));
  setRhov(1, p.getRho() * p.getV(1));
  setE(p.getE());
}


/**
 * Convert a conserved state to a (this) primitive state.
 * Overwrites the contents of this primitive state.
 * See Eq. 19-21 in Theory document.
 */
 __host__ __device__ inline
void PrimitiveState::fromCons(const ConservedState& cons) {
  if (cons.getRho() <= cst::SMALLRHO) {
    // execption handling for vacuum
    setRho(cst::SMALLRHO);
    setV(0, cst::SMALLV);
    setV(1, cst::SMALLV);
    setP(cst::SMALLP);
  } else {
    setRho(cons.getRho());
    Float one_over_rho = 1. / cons.getRho();
    Float vx           = cons.getRhov(0) * one_over_rho;
    Float vy           = cons.getRhov(1) * one_over_rho;
    setV(0, vx);
    setV(1, vy);
    setP(cons.getP());

    // handle negative pressure
    if (getP() <= cst::SMALLP) {
      setP(cst::SMALLP);
    }
  }
}

