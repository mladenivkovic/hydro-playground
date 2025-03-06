#include "Grid.h"

#include <cassert>
#include <iomanip>

#include "BoundaryConditions.h"
#include "Cell.h"
#include "Logging.h"
#include "Parameters.h"

/**
 * Constructor
 */
grid::Grid::Grid() : _cells(nullptr), _dx(1.), _initialised(false) {
  // Grab default values from default Parameters object.
  auto pars = parameters::Parameters();
  _nx = pars.getNx();
  _nx_norep = pars.getNx();
  _boundaryType = pars.getBoundaryType();
  _boxsize = pars.getBoxsize();
  _replicate = pars.getReplicate();
  _nbc = pars.getNBC();
}


/**
 * Destructor
 */
grid::Grid::~ Grid() {
  if (_cells == nullptr) error("Where did the cells array go??");
  delete [] _cells;
}


/**
 * @brief Initialize the grid.
 * This is mainly copying parameters from the parameters object
 * into the grid object. The actual grid is allocated later.
 *
 * @param pars A Parameters object holding global simulation parameters
 */
void grid::Grid::initGrid(const parameters::Parameters& pars) {

  message("Initialising grid parameters.", logging::LogLevel::Verbose);

#if DEBUG_LEVEL > 0
  if (not pars.getParamFileHasBeenRead())
    error("Parameter file is unread; Need that at this stage!");
#endif

  // Copy over relevant data.
  setNx(pars.getNx());
  setNxNorep(pars.getNx());
  setBoundaryType(pars.getBoundaryType());
  setNBC(pars.getNBC());
  setBoxsize(pars.getBoxsize());
  setReplicate(pars.getReplicate());


  // Mark that we did this
  _initialised = true;
}


/**
 * @brief Initialize the grid.
 * This is mainly copying parameters from the parameters object
 * into the grid object. The actual grid is allocated later.
 *
 * _cell(0,0)             is the bottom left cell.
 * _cell(nxtot-1,0)       is the bottom right cell
 * _cell(0,nxtot-1)       is the top-left cell
 * _cell(nxtot-1,nxtot-1) is the top-right cell
 */
void grid::Grid::initCells() {

#if DEBUG_LEVEL > 0
  if (not _initialised)
    error("Trying to alloc cells on uninitialised grid");
#endif

  size_t  nx = getNx();
  size_t  nx_norep = getNxNorep();
  size_t  nxTot = getNxTot();
  size_t  nbc   = getNBC();

  // TODO(mivkov): in arbitrary ICs, this needs to be set from reading the file first.
  if (nx == 0 or nxTot == 0)  {
    error("Got nx=0; The grid needs a size.");
  }


  // Compute derived quantities
  float_t dx = getBoxsize() / static_cast<float_t>(nx_norep);
  setDx(dx);


  size_t total_cells = 0;

  if (Dimensions == 1){

    // allocate space
    total_cells = nxTot;
    _cells = new cell::Cell[total_cells];

    // set cell positions and IDs
    for (size_t i = 0; i < nxTot; i++) {
      cell::Cell& c = getCell(i);
      float_t x = (static_cast<float_t>(i - nbc) + 0.5) * dx;
      c.setX(x);
      c.setId(i);
    }

  }
  else if (Dimensions == 2) {

    // allocate space
    total_cells = nxTot * nxTot;
    _cells = new cell::Cell[total_cells];

    // set cell positions and IDs
    for (size_t i = 0; i < nxTot; i++) {
      for (size_t j = 0; j < nxTot; j++) {
        cell::Cell& c = getCell(i, j);
        float_t x = (static_cast<float_t>(i - nbc) + 0.5) * dx;
        float_t y = (static_cast<float_t>(j - nbc) + 0.5) * dx;
        c.setX(x);
        c.setY(y);
        c.setId(i + j * nxTot);
      }
    }
  }
  else {
    error("Not implemented yet");
  }


  message("Initialised grid.", logging::LogLevel::Verbose);

  constexpr float_t KB = 1024.;
  constexpr float_t MB = 1024. * 1024.;
  constexpr float_t GB = 1024. * 1024. * 1024.;
  float_t gridsize = static_cast<float_t>(total_cells) * static_cast<float_t>(sizeof(cell::Cell));
  std::stringstream msg;
  msg << "Grid takes ";
  msg << std::setprecision(3) << std::setw(14) << gridsize / KB << " KB /";
  msg << std::setprecision(3) << std::setw(14) << gridsize / MB << " MB /";
  msg << std::setprecision(3) << std::setw(14) << gridsize / GB << " GB";

  message(msg);

}


/**
 * [density, velocity, pressure]
 */
void grid::Grid::setInitialConditions(
  size_t position, std::vector<float_t> vals, const parameters::Parameters& pars
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

  getCell(i + nbc, j + nbc).getPrim().setRho(vals[0]);
  getCell(i + nbc, j + nbc).getPrim().setV(0, vals[1]);
  if (Dimensions == 1) {
    getCell(i + nbc, j + nbc).getPrim().setP(vals[2]);
  }
  if (Dimensions == 2) {
    getCell(i + nbc, j + nbc).getPrim().setV(1, vals[2]);
    getCell(i + nbc, j + nbc).getPrim().setP(vals[3]);
  }
}


/**
 * @brief get the total mass of the grid.
 */
float_t grid::Grid::getTotalMass() {

  float_t total = 0;
  size_t  bc    = getNBC();
  size_t  nx    = getNx();

  if (Dimensions == 1) {
    for (size_t i = bc; i < bc + nx; i++) {
      total += getCell(i).getPrim().getRho();
    }

    total *= getDx();
  }

  else if (Dimensions == 2) {
    for (size_t i = bc; i < bc + nx; i++) {
      for (size_t j = bc; j < bc + nx; j++) {
        total += getCell(i, j).getPrim().getRho();
      }
    }

    total *= getDx() * getDx();
  }
  return total;
}


/**
 * Reset all fluxes of the grid (both primitive and conservative) to zero.
 */
void grid::Grid::resetFluxes() {

  constexpr auto dim2 = static_cast<size_t>(Dimensions == 2);
  size_t         nbc  = getNBC();
  size_t         nx   = getNx();

  for (size_t i = nbc; i < nbc + nx; i++) {
    for (size_t j = nbc * dim2; j < (nbc + nx) * dim2; j++) {
      // if we are in 1d, j will be fixed to zero
      getCell(i, j).getPrim().resetToInitialState();
      getCell(i, j).getCons().resetToInitialState();
    }
  }
}


/**
 * runs through interior cells and calls PrimitveToConserved()
 * on each.
 */
void grid::Grid::getCStatesFromPstates() {

  constexpr auto dim2 = static_cast<size_t>(Dimensions == 2);

  size_t nbc = getNBC();
  size_t nx  = getNx();

  for (size_t i = nbc; i < nbc + nx; i++) {
    for (size_t j = nbc * dim2; j < (nbc + nx) * dim2; j++) {
      // if we are in 1d, j will be fixed to zero
      getCell(i, j).PrimitiveToConserved();
    }
  }
}


/**
 * runs through interior cells and alls ConservedToPrimitve()
 * on each.
 */
void grid::Grid::getPStatesFromCstates() {

  constexpr auto dim2 = static_cast<size_t>(Dimensions == 2);

  size_t nbc = getNBC();
  size_t nx  = getNx();

  for (size_t i = nbc; i < nbc + nx; i++) {
    for (size_t j = nbc * dim2; j < (nbc + nx) * dim2; j++) {
      // if we are in 1d, j will be fixed to zero
      getCell(i, j).ConservedToPrimitive();
    }
  }
}

/**
 * enforce boundary conditions.
 * This function only picks out the pairs of real
 * and ghost cells in a row or column and then
 * calls the function that actually copies the data.
 */
void grid::Grid::setBoundary() {

  const size_t nbc   = getNBC();
  const size_t nx    = getNx();
  const size_t bctot = getNBCTot();

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
    realToGhost(realLeft, realRight, ghostLeft, ghostRight);
  }

  else if (Dimensions == 2) {
    // left-right boundaries
    for (size_t j = 0; j < nx + bctot; j++) {
      for (size_t i = 0; i < nbc; i++) {
        realLeft[i]   = &(getCell(nbc + i, j));
        realRight[i]  = &(getCell(nx + i, j));
        ghostLeft[i]  = &(getCell(i, j));
        ghostRight[i] = &(getCell(nx + nbc + i, j));
      }
      realToGhost(realLeft, realRight, ghostLeft, ghostRight, 0);
    }
  }

  // upper-lower boundaries
  // left -> lower, right -> upper
  for (size_t i = 0; i < nx + bctot; i++) {
    for (size_t j = 0; j < nbc; j++) {
      realLeft[j]   = &(getCell(i, nbc + j));
      realRight[j]  = &(getCell(i, nx + j));
      ghostLeft[j]  = &(getCell(i, j));
      ghostRight[j] = &(getCell(i, nx + nbc + j));
    }
    realToGhost(realLeft, realRight, ghostLeft, ghostRight, 1);
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
  std::vector<cell::Cell*>      realLeft,
  std::vector<cell::Cell*>      realRight,
  std::vector<cell::Cell*>      ghostLeft,
  std::vector<cell::Cell*>      ghostRight,
  const size_t                  dimension
) // dimension defaults to 0
{
  size_t nbc = getNBC();

  switch (getBoundaryType()) {
    case BC::BoundaryCondition::Periodic: {
    for (size_t i = 0; i < nbc; i++) {
      ghostLeft[i]->CopyBoundaryData(realLeft[i]);
      ghostRight[i]->CopyBoundaryData(realRight[i]);
    }
  } break;

  case BC::BoundaryCondition::Reflective: {
    for (size_t i = 0; i < nbc; i++) {
      ghostLeft[i]->CopyBoundaryDataReflective(realLeft[i], dimension);
      ghostRight[i]->CopyBoundaryDataReflective(realRight[i], dimension);
    }
  } break;

  case BC::BoundaryCondition::Transmissive: {
    for (size_t i = 0; i < nbc; i++) {
      ghostLeft[i]->CopyBoundaryData(realLeft[i]);

      // assumption that this vector has length "bc".
      ghostRight[i]->CopyBoundaryData(realRight[nbc - i - 1]);
    }
  } break;
  }
}
