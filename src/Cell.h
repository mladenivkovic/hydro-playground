#pragma once

#include "Config.h"
#include "Gas.h"
#include "Parameters.h"


namespace cell {

  class Cell {

  public:
    //! Standard constructor
    Cell();

    //! copy assignment, for copying boundary data
    //! Return reference to this, for chaining calls
    // TODO: This doesn't compile
    // Cell& operator=(const Cell& other) = default;

    void CopyBoundaryData(const Cell* real);

    void CopyBoundaryDataReflective(const Cell* real, const std::size_t dimension);

    //! Calls conserved to primitive on the members
    void ConservedToPrimitive() {
      _prim.ConservedToPrimitive(_cons);
    };
    //! Calls primitive to conserved on the members
    void PrimitiveToConserved() {
      _cons.PrimitiveToConserved(_prim);
    };

  private:
    //! Cell ID
    int _id;

    //! x position of cell centre
    float_t _x;

    //! y position of cell centre
    float_t _y;

    //! Primitive gas state
    IdealGas::PrimitiveState _prim;

    //! Conserved gas state
    IdealGas::ConservedState _cons;

    //! Fluxes of primitive state
    IdealGas::PrimitiveState _pflux;

    //! Fluxes of conserved state
    IdealGas::ConservedState _cflux;

    //! Acceleration
    // std::array<float_t, Dimensions> _acc;

  public:
    // leaving these for now
    // std::string getIndexString();

    //! Set cell centre position X,Y
    void setX(float_t x);
    void setY(float_t y);

    void              setId(const int id);
    [[nodiscard]] int getID() const;

    //! Get cell index(es) in grid
    std::pair<std::size_t, std::size_t> getIJ(const std::size_t nxtot);

    //! return refs to the above
    IdealGas::PrimitiveState& getPrim();
    IdealGas::ConservedState& getCons();
    IdealGas::PrimitiveState& getPFlux();
    IdealGas::ConservedState& getCFlux();

    // const versions to shush the compiler
    [[nodiscard]] const IdealGas::PrimitiveState& getPrim() const;
    [[nodiscard]] const IdealGas::ConservedState& getCons() const;
  };

} // namespace cell


// --------------------------------------------------------
// Definitions
// --------------------------------------------------------


//! Set cell centre position X
inline void cell::Cell::setX(float_t x) {
  _x = x;
}

//! Set cell centre position Y
inline void cell::Cell::setY(float_t y) {
  _y = y;
}


inline void cell::Cell::setId(const int id) {
  _id = id;
}


inline int cell::Cell::getID() const {
  return _id;
}

//! return refs to the above
inline IdealGas::PrimitiveState& cell::Cell::getPrim() {
  return _prim;
}

inline IdealGas::ConservedState& cell::Cell::getCons() {
  return _cons;
}

inline IdealGas::PrimitiveState& cell::Cell::getPFlux() {
  return _pflux;
}

inline IdealGas::ConservedState& cell::Cell::getCFlux() {
  return _cflux;
}

inline const IdealGas::PrimitiveState& cell::Cell::getPrim() const {
  return _prim;
}

inline const IdealGas::ConservedState& cell::Cell::getCons() const {
  return _cons;
}
