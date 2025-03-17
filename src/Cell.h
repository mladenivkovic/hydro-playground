#pragma once

#include "Config.h"
#include "Gas.h"
#include "Logging.h"


namespace cell {

  class Cell {

  public:
    //! Standard constructor
    Cell();

    void copyBoundaryData(const Cell* other);

    void copyBoundaryDataReflective(const Cell* other, const std::size_t dimension);

    //! Update cell's primitive state to current conserved state
    void cons2prim();

    //! Update cell's conserved state to current primitive state
    void prim2cons();

  private:
    //! Cell ID
    // size_t _id;

    //! x position of cell centre
    Float _x;

    //! y position of cell centre
    Float _y;

    //! Primitive gas state
    idealGas::PrimitiveState _prim;

    //! Conserved gas state
    idealGas::ConservedState _cons;

    //! Fluxes of primitive state
    // idealGas::PrimitiveState _pflux;

    //! Fluxes of conserved state
    idealGas::ConservedState _cflux;

#if SOLVER == SOLVER_MUSCL
    //! Intermediate extrapolated states
    idealGas::ConservedState _U_left_mid;
    idealGas::ConservedState _U_right_mid;
#endif

  public:
    //! Set cell centre position X,Y
    void                setX(const Float x);
    [[nodiscard]] Float getX() const;

    void                setY(const Float y);
    [[nodiscard]] Float getY() const;

    // void                 setId(const size_t id);
    // [[nodiscard]] size_t getID() const;

    //! Retrieve a specific cell quantity. Intended for printouts.
    Float getQuantityForPrintout(const char* quantity) const;

    //! Get cell index(es) in grid
    // std::pair<std::size_t, std::size_t> getIJ(const std::size_t nxtot);

    //! Getters and setters
    idealGas::PrimitiveState& getPrim();
    idealGas::ConservedState& getCons();
    // idealGas::PrimitiveState& getPFlux();
    idealGas::ConservedState& getCFlux();
    idealGas::ConservedState& getULMid();
    idealGas::ConservedState& getURMid();

    // const versions to shush the compiler
    [[nodiscard]] const idealGas::PrimitiveState& getPrim() const;
    [[nodiscard]] const idealGas::ConservedState& getCons() const;

    void setPrim(const idealGas::PrimitiveState& prim);
    void setCons(const idealGas::ConservedState& cons);
  };

} // namespace cell


// --------------------------------------------------------
// Definitions
// --------------------------------------------------------

inline void cell::Cell::cons2prim() {
  _prim.fromCons(_cons);
};


inline void cell::Cell::prim2cons() {
  _cons.fromPrim(_prim);
};


//! Set cell centre position X
inline void cell::Cell::setX(const Float x) {
  _x = x;
}

inline Float cell::Cell::getX() const {
  return _x;
}


//! Set cell centre position Y
inline void cell::Cell::setY(const Float y) {
  _y = y;
}


inline Float cell::Cell::getY() const {
  return _y;
}


// inline void cell::Cell::setId(const size_t id) {
//   _id = id;
// }


// inline size_t cell::Cell::getID() const {
//   return _id;
// }


inline idealGas::PrimitiveState& cell::Cell::getPrim() {
  return _prim;
}


inline idealGas::ConservedState& cell::Cell::getCons() {
  return _cons;
}


// inline idealGas::PrimitiveState& cell::Cell::getPFlux() {
//   return _pflux;
// }


inline idealGas::ConservedState& cell::Cell::getCFlux() {
  return _cflux;
}


inline idealGas::ConservedState& cell::Cell::getULMid() {
#if SOLVER == SOLVER_MUSCL
  return _U_left_mid;
#else
  error("Shouldn't be used!");
#endif
  return _cflux;
}


inline idealGas::ConservedState& cell::Cell::getURMid() {
#if SOLVER == SOLVER_MUSCL
  return _U_right_mid;
#else
  error("Shouldn't be used!");
#endif
  return _cflux;
}


inline const idealGas::PrimitiveState& cell::Cell::getPrim() const {
  return _prim;
}


inline const idealGas::ConservedState& cell::Cell::getCons() const {
  return _cons;
}


inline void cell::Cell::setPrim(const idealGas::PrimitiveState& prim) {
  _prim = prim;
}


inline void cell::Cell::setCons(const idealGas::ConservedState& cons) {
  _cons = cons;
}
