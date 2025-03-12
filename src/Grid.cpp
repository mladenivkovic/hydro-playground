#include "Grid.h"

#include <cassert>
#include <iomanip>
#include <iostream>

#include "BoundaryConditions.h"
#include "Cell.h"
#include "Logging.h"
#include "Parameters.h"


constexpr size_t grid_print_width = 5;
constexpr size_t grid_print_precision = 3;

/**
 * Constructor
 */
grid::Grid::Grid():
  _cells(nullptr),
  _dx(1.),
  _initialised(false) {
  // Grab default values from default Parameters object.
  auto pars     = parameters::Parameters();
  _nx           = pars.getNx();
  _nx_norep     = pars.getNx();
  _boundaryType = pars.getBoundaryType();
  _boxsize      = pars.getBoxsize();
  _replicate    = pars.getReplicate();
  _nbc          = pars.getNBC();
}


/**
 * Destructor
 */
grid::Grid::~Grid() {
  if (_cells == nullptr)
    error("Where did the cells array go??");
  delete[] _cells;
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

  size_t nx       = getNx();
  size_t nxNorep = getNxNorep();
  size_t nxTot    = getNxTot();
  size_t first = getFirstCellIndex();

  if (nx == 0 or nxTot == 0 or nxNorep == 0) {
    std::stringstream msg;
    msg << "Got some nx=0; The grid needs a size. ";
    msg << "nx=" << nx << ", ";
    msg << "nxTot=" << nxTot << ", ";
    msg << "nxNorep=" << nxNorep;
    error(msg);
  }


  // Compute derived quantities
  float_t dx = getBoxsize() / static_cast<float_t>(nxNorep);
  setDx(dx);


  size_t total_cells = 0;

  if (Dimensions == 1) {

    // allocate space
    total_cells = nxTot;
    _cells      = new cell::Cell[total_cells];

    // set cell positions and IDs
    for (size_t i = 0; i < nxTot; i++) {
      cell::Cell& c = getCell(i);
      float_t     x = (static_cast<float_t>(i - first) + 0.5) * dx;
      c.setX(x);
      c.setId(i);
    }

  } else if (Dimensions == 2) {

    // allocate space
    total_cells = nxTot * nxTot;
    _cells      = new cell::Cell[total_cells];

    // set cell positions and IDs
    for (size_t i = 0; i < nxTot; i++) {
      for (size_t j = 0; j < nxTot; j++) {
        cell::Cell& c = getCell(i, j);
        float_t     x = (static_cast<float_t>(i - first) + 0.5) * dx;
        float_t     y = (static_cast<float_t>(j - first) + 0.5) * dx;
        c.setX(x);
        c.setY(y);
        c.setId(i + j * nxTot);
      }
    }
  } else {
    error("Not implemented yet");
  }


  message("Initialised grid.", logging::LogLevel::Verbose);

  constexpr float_t KB = 1024.;
  constexpr float_t MB = 1024. * 1024.;
  constexpr float_t GB = 1024. * 1024. * 1024.;
  constexpr size_t prec = 3;
  constexpr size_t wid = 10;

  float_t gridsize = static_cast<float_t>(total_cells) * static_cast<float_t>(sizeof(cell::Cell));
  std::stringstream msg;
  msg << "Grid memory takes [";
  msg << std::setprecision(prec) << std::setw(wid) << gridsize / KB << " KB /";
  msg << std::setprecision(prec) << std::setw(wid) << gridsize / MB << " MB /";
  msg << std::setprecision(prec) << std::setw(wid) << gridsize / GB << " GB  ]";
  msg << " for " << total_cells << " cells";

  message(msg);
}


/**
 * @brief replicate (copy-paste) the initial conditions over the entire grid.
 */
void grid::Grid::replicateICs() {

  if (Dimensions != 2) {
    error("Not Implemented");
  }

  if (getReplicate() <= 1) {
    warning("Called IC replication with replicate <=1? Skipping it.");
    return;
  }


  printGrid("rho");

  size_t nxTot = getNxTot();
  size_t nxNorep = getNxNorep();
  size_t first = getFirstCellIndex();
  size_t last = nxNorep + first;
  size_t lastInGrid = getLastCellIndex();

  for (size_t j = first; j < last; j++){

    // First, copy in x direction for const j
    for (size_t rep = 1; rep < getReplicate(); rep++){
      for (size_t i = first; i < last; i++){

        size_t target = rep * nxNorep + i;

#if DEBUG_LEVEL > 0
        if (target >= nxTot){
          std::stringstream msg;
          msg << "Index error: Out of bounds " << target << "/" << nxTot;
          error(msg);
        }
#endif

        getCell(target, j) = getCell(i, j);
        printGrid("rho");

      }
    }

    // Now replicate entire row along y axis
    for (size_t rep = 1; rep < getReplicate(); rep++){

      size_t target = rep * nxNorep + j;

#if DEBUG_LEVEL > 0
      if (target >= nxTot){
        std::stringstream msg;
        msg << "Index error: Out of bounds " << target << "/" << nxTot;
        error(msg);
      }
#endif

      for (size_t i = first; i < lastInGrid; i++){
        getCell(i, target) = getCell(i, j);
      }

    }

  }

  // TODO: Make a timer out of this.
  message("Finished replicating grid.");
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
  const size_t bctot = _getNBCTot();

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
  std::vector<cell::Cell*> realLeft,
  std::vector<cell::Cell*> realRight,
  std::vector<cell::Cell*> ghostLeft,
  std::vector<cell::Cell*> ghostRight,
  const size_t             dimension
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




/**
 * @brief print out the grid.
 *
 * @param boundaries if true, print out boundary cells too.
 * @param conserved if true, print out conserved state instead of primitive state.
 */
void grid::Grid::printGrid(bool boundaries, bool conserved){


  size_t first = getFirstCellIndex();
  size_t last = getLastCellIndex();
  size_t start = first;
  size_t end = last;

  if (boundaries){
    start = 0;
    end = getNxTot();
  }

  std::stringstream out;
  out << "Full grid output of ";
  if (conserved){
    out << "conserved states";
  } else {
    out << "primitive states";
  }

  if (boundaries){
    out << " (including boundaries)";
  }
  out << "\n";


  if (Dimensions == 1) {

    for (size_t i = start; i < end; i++){
      if (conserved){
        out << getCell(i).getCons().toString() << " ";
      } else {
        out << getCell(i).getPrim().toString() << " ";
      }

      bool at_boundary = (i == first - 1) or (i == last - 1);
      if (boundaries and at_boundary ) {
        out << " | ";
      } else {
        if (i != end - 1) {
          out << ", ";
        }
      }

    }

    out << "\n";

  } else if (Dimensions == 2) {

    // Put the top at the top, and (0, 0) at the bottom left.
    for (int j = end - 1; j >= 0; j--){

      for (size_t i = start; i < end; i++){
        if (conserved){
          out << getCell(i, j).getCons().toString() << " ";
        } else {
          out << getCell(i, j).getPrim().toString() << " ";
        }

        bool at_boundary = (i == first - 1) or (i == last - 1);
        if (boundaries and at_boundary ) {
          out << " | ";
        } else {
          if (i != end - 1) {
            out << ", ";
          }
        }
      }

      out << "\n";
      bool at_boundary = (j == static_cast<int>(first)) or (j == static_cast<int>(last));
      if (boundaries and at_boundary ) {
        out << "\n";
      }
    }

  } else {
    error("Not implemented");
  }

  std::cout << out.str() << std::endl;
}


/**
 * @brief print out a single quantity over the entire grid.
 *
 * @param boundaries if true, print out boundary cells too.
 */
void grid::Grid::printGrid(const char* quantity, bool boundaries){


  size_t first = getFirstCellIndex();
  size_t last = getLastCellIndex();
  size_t start = first;
  size_t end = last;

  if (boundaries){
    start = 0;
    end = getNxTot();
  }

  std::stringstream out;
  out << "Full grid output of " << quantity;
  if (boundaries){
    out << " (including boundaries)";
  }
  out << "\n";

  constexpr size_t w = grid_print_width;
  constexpr size_t p = grid_print_precision;


  if (Dimensions == 1) {

    for (size_t i = start; i < end; i++){
      out << std::setw(w) << std::setprecision(p) << getCell(i).getQuanityForPrintout(quantity);

      bool at_boundary = (i == first - 1) or (i == last - 1);
      if (boundaries and at_boundary ) {
        out << " | ";
      } else {
        if (i != end - 1) {
          out << ", ";
        }
      }
    }

    out << "\n";

  } else if (Dimensions == 2) {

    // Put the top at the top, and (0, 0) at the bottom left.
    for (int j = end - 1; j >= 0; j--){

      for (size_t i = start; i < end; i++){
        out << std::setw(w) << std::setprecision(p) << getCell(i,j).getQuanityForPrintout(quantity);

        bool at_boundary = (i == first - 1) or (i == last - 1);
        if (boundaries and at_boundary ) {
          out << " | ";
        } else {
          if (i != end - 1) {
            out << ", ";
          }
        }
      }

      out << "\n";
      bool at_boundary = (j == static_cast<int>(first)) or (j == static_cast<int>(last));
      if (boundaries and at_boundary ) {
        out << "\n";
      }
    }

  } else {
    error("Not implemented");
  }

  std::cout << out.str() << "\n";
}
