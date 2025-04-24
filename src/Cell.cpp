#include "Cell.h"

#include <cassert>

#include "Logging.h"


/**
 * @brief Constructor for the cell.
 */
Cell::Cell():
  _x(0.),
  _y(0.)
// _id(0),
{
  // Empty body.
}



/**
 * Compute the i and j indexes of a cell in the grid
 */
// std::pair<size_t, size_t> Cell::getIJ(const size_t nxtot) {
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
Float Cell::getQuantityForPrintout(const char* quantity) const {

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
