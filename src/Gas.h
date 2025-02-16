#pragma once

#include <array>

#include "Config.h"


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
    // Standard constructor, init variables to 0
    PrimitiveState();

    // copy assignment
    // TODO(mivkov): This doesn't compile
    // PrimitiveState& operator=(const PrimitiveState& other) = default;

    // putting this in just in case it's needed
    void resetToInitialState() {
      *this = PrimitiveState();
    }

    // Convert a conserved state to a (this) primitive state.
    // Overwrites the contents of this primitive state.
    // TODO(mivkov): implementation
    void ConservedToPrimitive(const ConservedState& conservedState);

    float_t getSoundSpeed();
    float_t getEnergy();

    // Getters and setters!
    //

    // Setter for Rho
    void    setRho(const float_t val);
    float_t getRho() const;

    // same for u
    void    setU(const int index, const float_t val);
    float_t getU(const int index) const;

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
    void resetToInitialState() {
      *this = ConservedState();
    }

    void PrimitiveToConserved(const PrimitiveState& primState);
    void GetCFluxFromPstate(const PrimitiveState& pstate, int dimension);
    void GetCFluxFromCstate(const ConservedState& cstate, int dimension);

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
