#pragma once

#include <array>
#include <cmath>

#include "Config.h"
#include "Constants.h"


namespace IdealGas {
  // forward declaration, ConservedToPrimitive doesn't work without it
  class ConservedState;
  class PrimitiveState;

  /**
   * @brief Holds a primitive state (density, velocity, pressure)
   */
  class PrimitiveState {
  private:
    //! density
    float_t rho;

    //! velocity
    std::array<float_t, 2> u;

    //! pressure
    float_t p;


  public:
    //! Standard constructor, init variables to 0
    PrimitiveState();

    // copy assignment
    // TODO(mivkov): This doesn't compile. Check with boundary conditions
    // PrimitiveState& operator=(const PrimitiveState& other) = default;

    // putting this in just in case it's needed
    void resetToInitialState() {
      *this = PrimitiveState();
    }

    /**
     * Convert a conserved state to a (this) primitive state.
     * Overwrites the contents of this primitive state.
     */
    void ConservedToPrimitive(const ConservedState& conservedState);

    //! Get the local soundspeed given a primitive state
    float_t getSoundSpeed();

    //! Get the total gas energy from a primitive state
    float_t getEnergy();


    // Getters and setters!

    // Setter for Rho
    void    setRho(const float_t val);
    float_t getRho() const;

    // same for u
    void    setU(const std::size_t index, const float_t val);
    float_t getU(const std::size_t index) const;

    // used a lot, made a function for it
    float_t getUSquared() const;

    void    setP(const float_t val);
    float_t getP() const;
  };


  /**
   * @brief Holds a conserved state (density, momentum, energy)
   */
  class ConservedState {
  private:
    //! Density
    float_t rho;

    //! Momentum: rho * u
    std::array<float_t, 2> rhou;

    //! Energy
    float_t E;

  public:
    // Standard constructor, init variables to 0
    ConservedState();

    // putting this in in case it's needed
    // TODO(mivkov): is this needed?
    void resetToInitialState() {
      *this = ConservedState();
    }

    //! Compute the conserved state vector of a given primitive state.
    void PrimitiveToConserved(const PrimitiveState& primState);

    /**
     * Compute the flux of conserved variables of the Euler
     * equations given a primitive state vector
     */
    void GetCFluxFromPstate(const PrimitiveState& pstate, const std::size_t dimension);

    /**
     * Compute the flux of conserved variables of the Euler
     * equations given a conserved state vector
     */
    void GetCFluxFromCstate(const ConservedState& cstate, const std::size_t dimension);

    // Getters and setters!
    void    setRho(const float_t val);
    float_t getRho() const;

    // same for u
    void    setRhou(const std::size_t index, const float_t val);
    float_t getRhou(const std::size_t index) const;
    float_t getRhoUSquared() const;

    void    setE(const float_t val);
    float_t getE() const;
  };
} // namespace IdealGas


// --------------------------------------------------------
// Definitions
// --------------------------------------------------------

// Primitive State Stuff
// --------------------------

inline void IdealGas::PrimitiveState::setRho(const float_t val) {
  rho = val;
}


inline float_t IdealGas::PrimitiveState::getRho() const {
  return rho;
}


inline void IdealGas::PrimitiveState::setU(const size_t index, const float_t val) {
  u[index] = val;
}


inline float_t IdealGas::PrimitiveState::getU(const size_t index) const {
  return u[index];
}


inline float_t IdealGas::PrimitiveState::getUSquared() const {
  return u[0] * u[0] + u[1] * u[1];
}


inline void IdealGas::PrimitiveState::setP(const float_t val) {
  p = val;
}


inline float_t IdealGas::PrimitiveState::getP() const {
  return p;
}


/**
 * Compute the local sound speed given a primitive state
 */
inline float_t IdealGas::PrimitiveState::getSoundSpeed() {
  return std::sqrt(GAMMA * getP() / getRho());
}


/**
 * Get the total gas energy from a primitive state
 */
inline float_t IdealGas::PrimitiveState::getEnergy() {
  return 0.5 * getRho() * getUSquared() + getP() / GM1;
}



// Conserved State Stuff
// --------------------------

inline void IdealGas::ConservedState::setRhou(const size_t index, const float_t val) {
  rhou[index] = val;
}


inline float_t IdealGas::ConservedState::getRhou(const size_t index) const {
  return rhou[index];
}


inline float_t IdealGas::ConservedState::getRhoUSquared() const {
  return rhou[0] * rhou[0] + rhou[1] * rhou[1];
}


inline void IdealGas::ConservedState::setE(const float_t val) {
  E = val;
}


inline float_t IdealGas::ConservedState::getE() const {
  return E;
}


inline float_t IdealGas::ConservedState::getRho() const {
  return rho;
}


inline void IdealGas::ConservedState::setRho(const float_t val) {
  rho = val;
}
