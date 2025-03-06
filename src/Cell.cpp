#include "Cell.h"

#include <cassert>


/**
 * @brief Constructor for the cell. This has everything from the old
 * cell_init_cell() function
 */
cell::Cell::Cell():
  _id(0),
  _x(0.),
  _y(0.)
// and the ideal gasses too
// _acc({0, 0})
{
  /* Empty body. */
}


/**
 * @brief Copies the actual data needed for boundaries from a real
 * cell to a ghost cell.
 * Should be called from within the ghost cell.
 *
 * @param real the "real" cell, which we are copying data from
 */
void cell::Cell::CopyBoundaryData(const cell::Cell* real) {
  // This should be called from within the ghost

  // copy everything from the other
  _prim = real->getPrim();
  _cons = real->getCons();
  // TODO(mivkov): check this is taking a deep copy for real!
}


/**
 * @brief Copies the actual data needed for boundaries from a real
 * cell to a ghost cell. Here for a reflective boundary
 * condition, where we need to invert the velocities.
 * Should be called from within the ghost cell.
 *
 * @param cell* real: pointer to real cell from which we take data
 * @param dimension: in which dimension the reflection is supposed to be
 */
void cell::Cell::CopyBoundaryDataReflective(const cell::Cell* real, const size_t dimension) {

  // This should be called from within the ghost
  _prim = real->getPrim();
  _cons = real->getCons();

  // flip the velocities in specified dimension
  float_t u = getPrim().getU(dimension);
  getPrim().setU(dimension, -1. * u);

  float_t rhou = getCons().getRhou(dimension);
  getCons().setRhou(dimension, -1. * rhou);
}


/**
 * Compute the i and j indexes of a cell in the grid
 */
std::pair<size_t, size_t> cell::Cell::getIJ(const size_t nxtot) {

  std::pair<size_t, size_t> output;

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
