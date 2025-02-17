#include "Cell.h"
#include <cassert>

#include "Parameters.h"


/**
 * @brief Constructor for the cell. This has everything from the old
 * cell_init_cell() function
 */
cell::Cell::Cell():
  _id(0),
  _x(0),
  _y(0),
  // and the ideal gasses too
  _prim(),
  _cons(),
  _pflux(),
  _cflux(),
  _acc({0, 0}) { /* Empty body. */
}


void cell::Cell::CopyBoundaryData(const cell::Cell* real) {
  // This should be called from within the ghost

  // copy everything from the other!
  _prim = real->getPrim();
  _cons = real->getCons();
  // check this is taking a deep copy for real!
}


void cell::Cell::CopyBoundaryDataReflective(const cell::Cell* real, const int dimension) {
  /*
   * Copies the data we need. Dimension indiciates which dimension
   * We flip the velocities
   */

  // This should be called from within the ghost
  _prim = real->getPrim();
  _cons = real->getCons();

  // flip the velocities in specified dimension
  float_t u = getPrim().getU(dimension);
  getPrim().setU(dimension, -1. * u);

  float_t rhou = getCons().getRhou(dimension);
  getCons().setRhou(dimension, -1. * rhou);
}


std::pair<size_t, size_t> cell::Cell::getIJ() {
  std::pair<size_t, size_t> output;
  size_t                    nxtot = parameters::Parameters::Instance.getNxTot();
  if (Dimensions == 1) {
    output.first  = getID();
    output.second = 0;
  }
  if (Dimensions == 2) {
    size_t j      = getID() / (nxtot);
    size_t i      = getID() - j * nxtot;
    output.first  = i;
    output.second = j;
  }
  return output;
}


// Getters and setters for cell!
void cell::Cell::setX(float_t x) {
  _x = x;
}

void cell::Cell::setY(float_t y) {
  _y = y;
}


void cell::Cell::setId(const int id) {
  _id = id;
}


int cell::Cell::getID() const {
  return _id;
}
