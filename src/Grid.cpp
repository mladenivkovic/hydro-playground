#include "Grid.h"

#include <cassert>

#include "Cell.h"
#include "Parameters.h"

// define the static copy. Calls the default constructor but
// the user has to call InitCells()
// TODO: do the same as for parameters
grid::Grid grid::Grid::Instance;


/**
 * @brief Initialize the grid.
 *
 * _cell(0,0)             is the bottom left cell.
 * _cell(nxtot-1,0)       is the bottom right cell
 * _cell(0,nxtot-1)       is the top-left cell
 * _cell(nxtot-1,nxtot-1) is the top-right cell
 */
void grid::Grid::initGrid(
    const parameters::Parameters& pars
    ) {

#if DEBUG_LEVEL > 0
  // TODO(mivkov): assert sure parameters and IC file has been read.
#endif

  // TODO: write out what you're doing
  // log_extra("Initializing grid; ndim=%d, nx=%d", NDIM, pars.nx);

  size_t  nxTot = pars.getNxTot();
  size_t  nbc   = pars.getNBC();
  float_t dx    = pars.getDx();

  if (Dimensions == 1) {
    // make some room in the vector...
    _cells.resize(nxTot);

    for (size_t i = 0; i < nxTot; i++) {
      getCell(i).setX((i - nbc + 0.5) * dx);
      getCell(i).setId(i);
    }

  } else if (Dimensions == 2) {
    // make some room in the vector...
    _cells.resize(nxTot * nxTot);

    for (size_t i = 0; i < nxTot; i++) {
      for (size_t j = 0; j < nxTot; j++) {
        getCell(i, j, pars).setX((i - nbc + 0.5) * dx);
        getCell(i, j, pars).setY((j - nbc + 0.5) * dx);

        // this used to be i + j * pars.nxtot, but i have altered the
        // convention this time around
        getCell(i, j, pars).setId(i + j * nxTot);
      }
    }

  } else {
    error("Not implemented yet");
  }
}


/**
 * [density, velocity, pressure]
 */
void grid::Grid::setInitialConditions(
    size_t position, std::vector<float_t> vals,
    const parameters::Parameters& pars
    ) {

  assert((vals.size() == 4 and Dimensions == 2) or (vals.size() == 3 and Dimensions == 1));
  // Let's set i,j based on the position in the array we passed in
  size_t i;
  size_t j;
  if (Dimensions == 1) {
    i = position;
    j = 0;
  }
  if (Dimensions == 2) {
    i = position % pars.getNx();
    j = position / pars.getNx();
  }

  // alias the bc value
  size_t nbc = pars.getNBC();

  getCell(i + nbc, j + nbc, pars).getPrim().setRho(vals[0]);
  getCell(i + nbc, j + nbc, pars).getPrim().setU(0, vals[1]);
  if (Dimensions == 1) {
    getCell(i + nbc, j + nbc, pars).getPrim().setP(vals[2]);
  }
  if (Dimensions == 2) {
    getCell(i + nbc, j + nbc, pars).getPrim().setU(1, vals[2]);
    getCell(i + nbc, j + nbc, pars).getPrim().setP(vals[3]);
  }
}


/**
 * @brief get the total mass of the grid.
 */
float_t grid::Grid::getTotalMass(    const parameters::Parameters& pars) {

  float_t total = 0;
  size_t  bc    = pars.getNBC();
  size_t  nx    = pars.getNx();

  if (Dimensions == 1) {
    for (size_t i = bc; i < bc + nx; i++) {
      total += getCell(i).getPrim().getRho();
    }

    total *= pars.getDx();
  }

  else if (Dimensions == 2) {
    for (size_t i = bc; i < bc + nx; i++) {
      for (size_t j = bc; j < bc + nx; j++) {
        total += getCell(i, j, pars).getPrim().getRho();
      }
    }

    total *= pars.getDx() * pars.getDx();
  }
  return total;
}


/**
 * Reset all fluxes of the grid (both primitive and conservative) to zero.
 */
void grid::Grid::resetFluxes(const parameters::Parameters& pars) {

  constexpr auto dim2 = static_cast<size_t>(Dimensions == 2);
  size_t         nbc  = pars.getNBC();
  size_t         nx   = pars.getNx();

  for (size_t i = nbc; i < nbc + nx; i++) {
    for (size_t j = nbc * dim2; j < (nbc + nx) * dim2; j++) {
      // if we are in 1d, j will be fixed to zero
      getCell(i, j, pars).getPrim().resetToInitialState();
      getCell(i, j, pars).getCons().resetToInitialState();
    }
  }
}


/**
 * runs through interior cells and calls PrimitveToConserved()
 * on each.
 */
void grid::Grid::getCStatesFromPstates(const parameters::Parameters& pars) {

  constexpr auto dim2 = static_cast<size_t>(Dimensions == 2);

  size_t nbc  = pars.getNBC();
  size_t nx   = pars.getNx();

  for (size_t i = nbc; i < nbc + nx; i++) {
    for (size_t j = nbc * dim2; j < (nbc + nx) * dim2; j++) {
      // if we are in 1d, j will be fixed to zero
      getCell(i, j, pars).PrimitiveToConserved();
    }
  }
}


/**
 * runs through interior cells and alls ConservedToPrimitve()
 * on each.
 */
void grid::Grid::getPStatesFromCstates(const parameters::Parameters& pars) {

  constexpr auto dim2 = static_cast<size_t>(Dimensions == 2);

  size_t nbc  = pars.getNBC();
  size_t nx   = pars.getNx();

  for (size_t i = nbc; i < nbc + nx; i++) {
    for (size_t j = nbc * dim2; j < (nbc + nx) * dim2; j++) {
      // if we are in 1d, j will be fixed to zero
      getCell(i, j, pars).ConservedToPrimitive();
    }
  }
}

/**
 * enforce boundary conditions.
 * This function only picks out the pairs of real
 * and ghost cells in a row or column and then
 * calls the function that actually copies the data.
 */
void grid::Grid::setBoundary(const parameters::Parameters& pars) {

  const size_t nbc   = pars.getNBC();
  const size_t nx    = pars.getNx();
  const size_t bctot = pars.getNBCTot();

  // Make space to store pointers to real and ghost cells.
  std::vector<cell::Cell*> realLeft(nbc);
  std::vector<cell::Cell*> realRight(nbc);
  std::vector<cell::Cell*> ghostLeft(nbc);
  std::vector<cell::Cell*> ghostRight(nbc);

  // doesn't look like we will need this code often. so avoid hacky stuff
  if (Dimensions == 1) {
    for (size_t i = 0; i < nbc; i++) {
      realLeft[i]   = &(getCell(nbc + i));
      realRight[i]  = &(getCell(nx + i)); // = last index of a real cell = BC + (i + 1)
      ghostLeft[i]  = &(getCell(i));
      ghostRight[i] = &(getCell(nx + nbc + i));
    }
    realToGhost(realLeft, realRight, ghostLeft, ghostRight, pars);
  }

  else if (Dimensions == 2) {
    // left-right boundaries
    for (size_t j = 0; j < nx + bctot; j++) {
      for (size_t i = 0; i < nbc; i++) {
        realLeft[i]   = &(getCell(nbc + i, j, pars));
        realRight[i]  = &(getCell(nx + i, j, pars));
        ghostLeft[i]  = &(getCell(i, j, pars));
        ghostRight[i] = &(getCell(nx + nbc + i, j, pars));
      }
      realToGhost(realLeft, realRight, ghostLeft, ghostRight, pars, 0);
    }
  }

  // upper-lower boundaries
  // left -> lower, right -> upper
  for (size_t i = 0; i < nx + bctot; i++) {
    for (size_t j = 0; j < nbc; j++) {
      realLeft[j]   = &(getCell(i, nbc + j, pars));
      realRight[j]  = &(getCell(i, nx + j, pars));
      ghostLeft[j]  = &(getCell(i, j, pars));
      ghostRight[j] = &(getCell(i, nx + nbc + j, pars));
    }
    realToGhost(realLeft, realRight, ghostLeft, ghostRight, pars, 1);
  }
}


/**
 * apply the boundary conditions from real to ghost cells
 *
 * @param realL:     array of pointers to real cells with lowest index
 * @param realR:     array of pointers to real cells with highest index
 * @param ghostL:    array of pointers to ghost cells with lowest index
 * @param ghostR:    array of pointers to ghost cells with highest index
 * @param dimension: dimension integer. 0 for x, 1 for y. Needed for
 *                   reflective boundary conditions.
 *
 * all arguments are arrays of size params::_nbc (number of boundary cells)
 * lowest array index is also lowest index of cell in grid
 */
void grid::Grid::realToGhost(
  std::vector<cell::Cell*> realLeft,
  std::vector<cell::Cell*> realRight,
  std::vector<cell::Cell*> ghostLeft,
  std::vector<cell::Cell*> ghostRight,
  const parameters::Parameters& pars,
  const size_t             dimension
) // dimension defaults to 0
{
  // prevents crowding down there
  using BC    = parameters::BoundaryCondition;
  size_t nbc  = pars.getNBC();

  switch (pars.getBoundaryType()) {
  case BC::Periodic: {
    for (size_t i = 0; i < nbc; i++) {
      ghostLeft[i]->CopyBoundaryData(realLeft[i]);
      ghostRight[i]->CopyBoundaryData(realRight[i]);
    }
  } break;

  case BC::Reflective: {
    for (size_t i = 0; i < nbc; i++) {
      ghostLeft[i]->CopyBoundaryDataReflective(realLeft[i], dimension);
      ghostRight[i]->CopyBoundaryDataReflective(realRight[i], dimension);
    }
  } break;

  case BC::Transmissive: {
    for (size_t i = 0; i < nbc; i++) {
      ghostLeft[i]->CopyBoundaryData(realLeft[i]);

      // assumption that this vector has length "bc".
      ghostRight[i]->CopyBoundaryData(realRight[nbc - i - 1]);
    }
  } break;
  }
}
