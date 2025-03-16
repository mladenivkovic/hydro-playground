#include "Cell.h"

#include <cassert>

#include "Logging.h"


/**
 * @brief Constructor for the cell.
 */
cell::Cell::Cell():
  _x(0.),
  _y(0.)
// _id(0),
{
  // Empty body.
}


/**
 * @brief Copies the gas data needed for boundaries from a real cell to this
 * cell. This means that in boundary exchanges, this should be called from
 * within the ghost cell.
 *
 * @param other the other cell, which we are copying data from
 */
void cell::Cell::copyBoundaryData(const cell::Cell* other) {
  // copy gas data from the other
  _prim = other->getPrim();
  _cons = other->getCons();
}


/**
 * @brief Copies the gas data needed for boundaries from a real cell to a ghost
 * cell. Here for a reflective boundary condition, where we need to invert the
 * velocities. Should be called from within the ghost cell.
 * TODO: Refer to equations in theory doc
 *
 * @param other: pointer to real cell from which we take data
 * @param dimension: in which dimension the reflection is supposed to be
 */
void cell::Cell::copyBoundaryDataReflective(const cell::Cell* other, const size_t dimension) {

  // This should be called from within the ghost
  _prim = other->getPrim();
  _cons = other->getCons();

  // flip the velocities in specified dimension
  Float u = getPrim().getV(dimension);
  getPrim().setV(dimension, -u);

  // Same for momentum.
  Float rhou = getCons().getRhov(dimension);
  getCons().setRhov(dimension, -rhou);
}


/**
 * Compute the i and j indexes of a cell in the grid
 */
// std::pair<size_t, size_t> cell::Cell::getIJ(const size_t nxtot) {
//
//   std::pair<size_t, size_t> output;
//
//   if (Dimensions == 1) {
//     output.first  = getID();
//     output.second = 0;
//   }
//   if (Dimensions == 2) {
//     size_t j      = getID() / (nxtot);
//     size_t i      = getID() - j * nxtot;
//     output.first  = i;
//     output.second = j;
//   }
//   return output;
// }


/**
 * Retrieve a specific cell quantity. Intended for printouts.
 */
Float cell::Cell::getQuantityForPrintout(const char* quantity) const {

  std::string q(quantity);

  if (q == "rho") {
    return getPrim().getRho();
  }
  if (q == "vx") {
    return getPrim().getV(0);
  }
  if (q == "vy") {
    return getPrim().getV(1);
  }
  if (q == "P") {
    return getPrim().getP();
  }
  if (q == "p") {
    return getPrim().getP();
  }
  if (q == "rhovx") {
    return getCons().getRhov(0);
  }
  if (q == "rhovy") {
    return getCons().getRhov(1);
  }
  if (q == "E") {
    return getCons().getE();
  }
  if (q == "e") {
    return getCons().getE();
  }

  error("Unknown quantity " + q);
  return 0.;
}
