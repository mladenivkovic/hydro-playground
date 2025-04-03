#pragma once

#include "Config.h"
#include "Gas.h"
#include "Logging.h"
#include "omp.h"


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
  PrimitiveState _prim;

  //! Conserved gas state
  ConservedState _cons;

  //! Fluxes of primitive state
  // PrimitiveState _pflux;

  //! Fluxes of conserved state
  ConservedState _cflux;

#if SOLVER == SOLVER_MUSCL
  //! Intermediate extrapolated states
  ConservedState U_left_mid;
  ConservedState U_right_mid;
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
  PrimitiveState& getPrim();
  ConservedState& getCons();
  // const versions to shush the compiler
  [[nodiscard]] const PrimitiveState& getPrim() const;
  [[nodiscard]] const ConservedState& getCons() const;
  void                                setPrim(const PrimitiveState& prim);
  void                                setCons(const ConservedState& cons);

  // PrimitiveState& getPFlux();

  ConservedState& getCFlux();
  void            setCFlux(ConservedFlux& flux);

  ConservedState& getULMid();
  ConservedState& getURMid();
  void            setULMid(const ConservedState& state);
  void            setURMid(const ConservedState& state);
};


// --------------------------------------------------------
// Definitions
// --------------------------------------------------------

inline void Cell::cons2prim() {
  _prim.fromCons(_cons);
};


inline void Cell::prim2cons() {
  _cons.fromPrim(_prim);
};


//! Set cell centre position X
inline void Cell::setX(const Float x) {
  _x = x;
}

inline Float Cell::getX() const {
  return _x;
}


//! Set cell centre position Y
inline void Cell::setY(const Float y) {
  _y = y;
}


inline Float Cell::getY() const {
  return _y;
}


// inline void Cell::setId(const size_t id) {
//   _id = id;
// }


// inline size_t Cell::getID() const {
//   return _id;
// }


inline PrimitiveState& Cell::getPrim() {
  return _prim;
}


inline ConservedState& Cell::getCons() {
  return _cons;
}


// inline PrimitiveState& Cell::getPFlux() {
//   return _pflux;
// }


inline ConservedState& Cell::getCFlux() {
  return _cflux;
}


inline void Cell::setCFlux(ConservedFlux& flux) {
  _cflux = flux;
}


inline ConservedState& Cell::getULMid() {
#if SOLVER == SOLVER_MUSCL
  return U_left_mid;
#else
  error("Shouldn't be used!");
  return _cflux;
#endif
}


inline ConservedState& Cell::getURMid() {
#if SOLVER == SOLVER_MUSCL
  return U_right_mid;
#else
  error("Shouldn't be used!");
  return _cflux;
#endif
}


inline void Cell::setULMid(const ConservedState& state) {
#if SOLVER == SOLVER_MUSCL
  U_left_mid = state;
#else
  error("Shouldn't be used!");
#endif
}


inline void Cell::setURMid(const ConservedState& state) {
#if SOLVER == SOLVER_MUSCL
  U_right_mid = state;
#else
  error("Shouldn't be used!");
#endif
}


inline const PrimitiveState& Cell::getPrim() const {
  return _prim;
}


inline const ConservedState& Cell::getCons() const {
  return _cons;
}


inline void Cell::setPrim(const PrimitiveState& prim) {
  _prim = prim;
}


inline void Cell::setCons(const ConservedState& cons) {
  _cons = cons;
}
