#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <string>

#include "Config.h"
#include "Constants.h"


namespace idealGas {
  // forward declaration, ConservedToPrimitive doesn't work without it
  class ConservedState;
  class PrimitiveState;

  /**
   * @brief Holds a primitive state (density, velocity, pressure)
   */
  class PrimitiveState {
  private:
    //! density
    Float _rho;

    //! velocity
    std::array<Float, Dimensions> _v;

    //! pressure
    Float _p;


  public:
    PrimitiveState();
    PrimitiveState(const Float rho, const std::array<Float, Dimensions> vel, const Float p);
    PrimitiveState(const Float rho, const Float vx, const Float p);
    PrimitiveState(const Float rho, const Float vx, const Float vy, const Float p);

    /**
     * Clear out contents.
     */
    void clear() {
      *this = PrimitiveState();
    }

    /**
     * Set the current primitive state vector to equivalend of given conserved
     * state.
     */
    void fromCons(const ConservedState& cons);

    //! Get the local soundspeed given a primitive state
    [[nodiscard]] Float getSoundSpeed() const;

    //! Get the total gas energy from a primitive state
    [[nodiscard]] Float getE() const;

    //! Get a string of the state.
    [[nodiscard]] std::string toString() const;


    // Getters and setters!

    // Setter for Rho
    void                setRho(const Float val);
    [[nodiscard]] Float getRho() const;

    // same for u
    void                setV(const std::size_t index, const Float val);
    [[nodiscard]] Float getV(const std::size_t index) const;

    // used a lot, made a function for it
    [[nodiscard]] Float getVSquared() const;

    void                setP(const Float val);
    [[nodiscard]] Float getP() const;
  };


  /**
   * @brief Holds a conserved state (density, momentum, energy)
   */
  class ConservedState {
  private:
    //! Density
    Float _rho;

    //! Momentum: rho * v
    std::array<Float, Dimensions> _rhov;

    //! Energy
    Float _energy;

  public:
    // Standard constructor, init variables to 0
    ConservedState();

    /**
     * Clear out contents.
     */
    void clear() {
      *this = ConservedState();
    }


    /**
     * Set the current primitive state vector to equivalend of given conserved
     * state.
     */
    void fromPrim(const PrimitiveState& prim);


    /**
     * Compute the flux of conserved variables of the Euler
     * equations given a primitive state vector
     */
    void getCFluxFromPstate(const PrimitiveState& pstate, const std::size_t dimension);


    /**
     * Compute the flux of conserved variables of the Euler
     * equations given a conserved state vector
     */
    void getCFluxFromCstate(const ConservedState& cstate, const std::size_t dimension);


    //! Get a string of the state.
    [[nodiscard]] std::string toString() const;


    // Getters and setters!
    void                setRho(const Float val);
    [[nodiscard]] Float getRho() const;

    // same for u
    void                setRhov(const std::size_t index, const Float val);
    [[nodiscard]] Float getRhov(const std::size_t index) const;
    [[nodiscard]] Float getRhoVSquared() const;

    void                setE(const Float val);
    [[nodiscard]] Float getE() const;
  };
} // namespace idealGas


// --------------------------------------------------------
// Definitions
// --------------------------------------------------------

// Primitive State Stuff
// --------------------------

inline void idealGas::PrimitiveState::setRho(const Float val) {
#if DEBUG_LEVEL > 0
  assert(val >= 0.);
#endif
  _rho = val;
}


inline Float idealGas::PrimitiveState::getRho() const {
#if DEBUG_LEVEL > 0
  assert(_rho >= 0.);
#endif
  return _rho;
}


inline void idealGas::PrimitiveState::setV(const size_t index, const Float val) {
#if DEBUG_LEVEL > 0
  // assert(index >= 0); // always true for unsigned type
  assert(index < Dimensions);
#endif
  _v[index] = val;
}


inline Float idealGas::PrimitiveState::getV(const size_t index) const {
#if DEBUG_LEVEL > 0
  // assert(index >= 0); // always true for unsigned type
  assert(index < Dimensions);
#endif
  return _v[index];
}


inline Float idealGas::PrimitiveState::getVSquared() const {
  return _v[0] * _v[0] + _v[1] * _v[1];
}


inline void idealGas::PrimitiveState::setP(const Float val) {
#if DEBUG_LEVEL > 0
  assert(val >= 0.);
#endif
  _p = val;
}


inline Float idealGas::PrimitiveState::getP() const {
#if DEBUG_LEVEL > 0
  assert(_p >= 0.);
#endif
  return _p;
}


/**
 * Compute the local sound speed given a primitive state
 */
inline Float idealGas::PrimitiveState::getSoundSpeed() const {
  return std::sqrt(cst::GAMMA * getP() / getRho());
}


/**
 * Get the total gas energy from a primitive state
 */
inline Float idealGas::PrimitiveState::getE() const {
  return 0.5 * getRho() * getVSquared() + getP() / cst::GM1;
}


// Conserved State Stuff
// --------------------------

inline void idealGas::ConservedState::setRhov(const size_t index, const Float val) {
#if DEBUG_LEVEL > 0
  // assert(index >= 0); // always true for unsigned type
  assert(index < Dimensions);
#endif
  _rhov[index] = val;
}


inline Float idealGas::ConservedState::getRhov(const size_t index) const {
#if DEBUG_LEVEL > 0
  // assert(index >= 0); // always true for unsigned type
  assert(index < Dimensions);
#endif
  return _rhov[index];
}


inline Float idealGas::ConservedState::getRhoVSquared() const {
  return _rhov[0] * _rhov[0] + _rhov[1] * _rhov[1];
}


inline void idealGas::ConservedState::setE(const Float val) {
#if DEBUG_LEVEL > 0
  assert(val >= 0.);
#endif
  _energy = val;
}


inline Float idealGas::ConservedState::getE() const {
#if DEBUG_LEVEL > 0
  assert(_energy >= 0.);
#endif
  return _energy;
}


inline Float idealGas::ConservedState::getRho() const {
#if DEBUG_LEVEL > 0
  assert(_rho >= 0.);
#endif
  return _rho;
}


inline void idealGas::ConservedState::setRho(const Float val) {
#if DEBUG_LEVEL > 0
  assert(val >= 0.);
#endif
  _rho = val;
}
