#include "Grid.h"

#include <cassert>
#include <iomanip>
#include <iostream>

#include "Cell.h"
#include "Logging.h"
#include "Parameters.h"
#include "Timer.h"


constexpr size_t grid_print_width     = 5;
constexpr size_t grid_print_precision = 3;


/**
 * This is mainly copying parameters from the parameters object
 * into the grid object. The actual grid is allocated later.
 *
 * @param pars A Parameters object holding global simulation parameters
 */
Grid::Grid(const Parameters& params):
  _cells(nullptr),
  _nx(params.getNx()),
  _nx_norep(params.getNx()),
  _dx(1.),
  _boxsize(params.getBoxsize()),
  _nbc(params.getNBC()),
  _replicate(params.getReplicate()),
  _boundary_type(params.getBoundaryType()) {

#if DEBUG_LEVEL > 0
  if (not params.getParamFileHasBeenRead())
    error("Parameter file is unread; Need that at this stage!");
#endif
}


/**
 * Destructor
 */
Grid::~Grid() {
  if (_cells == nullptr)
    error("Where did the cells array go??");
  delete[] _cells;
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
void Grid::initCells() {

  message("Initialising cells.", logging::LogLevel::Debug);
  timer::Timer tick(timer::Category::Reset);


  size_t nx      = getNx();
  size_t nxNorep = getNxNorep();
  size_t nxTot   = getNxTot();
  size_t first   = getFirstCellIndex();

  if (nx == 0 or nxTot == 0 or nxNorep == 0) {
    std::stringstream msg;
    msg << "Got some nx=0; The grid needs a size. ";
    msg << "nx=" << nx << ", ";
    msg << "nxTot=" << nxTot << ", ";
    msg << "nxNorep=" << nxNorep;
    error(msg.str());
  }


  // Compute derived quantities
  Float dx = getBoxsize() / static_cast<Float>(nxNorep);
  setDx(dx);


  size_t total_cells = 0;

  if (Dimensions == 1) {

    // allocate space
    total_cells = nxTot;
    _cells      = new Cell[total_cells];

    // set cell positions and IDs
    for (size_t i = 0; i < nxTot; i++) {
      Cell& c = getCell(i);
      Float x = (static_cast<Float>(i - first) + 0.5) * dx;
      c.setX(x);
      // c.setId(i);
    }

  } else if (Dimensions == 2) {

    // allocate space
    total_cells = nxTot * nxTot;
    _cells      = new Cell[total_cells];

    // set cell positions and IDs
    for (size_t i = 0; i < nxTot; i++) {
      for (size_t j = 0; j < nxTot; j++) {
        Cell& c = getCell(i, j);
        Float x = (static_cast<Float>(i - first) + 0.5) * dx;
        Float y = (static_cast<Float>(j - first) + 0.5) * dx;
        c.setX(x);
        c.setY(y);
        // c.setId(i + j * nxTot);
      }
    }
  } else {
    error("Not implemented yet");
  }


  message("Initialised grid.", logging::LogLevel::Verbose);

  constexpr Float  KB   = 1024.;
  constexpr Float  MB   = 1024. * 1024.;
  constexpr Float  GB   = 1024. * 1024. * 1024.;
  constexpr size_t prec = 3;
  constexpr size_t wid  = 10;

  auto              gridsize = static_cast<Float>(total_cells * sizeof(Cell));
  std::stringstream msg;
  msg << "Grid memory takes [";
  msg << std::setprecision(prec) << std::setw(wid) << gridsize / KB << " KB /";
  msg << std::setprecision(prec) << std::setw(wid) << gridsize / MB << " MB /";
  msg << std::setprecision(prec) << std::setw(wid) << gridsize / GB << " GB  ]";
  msg << " for " << total_cells << " cells";

  message(msg.str());
  // timing("Initialising grid took " + tick.tock());
}


/**
 * @brief replicate (copy-paste) the initial conditions over the entire grid.
 */
void Grid::replicateICs() {

  if (Dimensions != 2) {
    error("Not Implemented");
  }

  if (getReplicate() <= 1) {
    warning("Called IC replication with replicate <=1? Skipping it.");
    return;
  }

  // Ignore timing here:
  // Timer for global IC duration is already set up above in call stack
  timer::Timer tick(timer::Category::Ignore);

  size_t nxNorep    = getNxNorep();
  size_t first      = getFirstCellIndex();
  size_t last       = nxNorep + first;
  size_t lastInGrid = getLastCellIndex();

  for (size_t j = first; j < last; j++) {

    // First, copy in x direction for const j
    for (size_t rep = 1; rep < getReplicate(); rep++) {
      for (size_t i = first; i < last; i++) {

        size_t target = rep * nxNorep + i;

#if DEBUG_LEVEL > 0
        if (target >= getNxTot()) {
          std::stringstream msg;
          msg << "Index error: Out of bounds " << target << "/" << getNxTot();
          error(msg.str());
        }
#endif

        getCell(target, j) = getCell(i, j);
      }
    }

    // Now replicate entire row along y axis
    for (size_t rep = 1; rep < getReplicate(); rep++) {

      size_t target = rep * nxNorep + j;

#if DEBUG_LEVEL > 0
      if (target >= getNxTot()) {
        std::stringstream msg;
        msg << "Index error: Out of bounds " << target << "/" << getNxTot();
        error(msg.str());
      }
#endif

      for (size_t i = first; i < lastInGrid; i++) {
        getCell(i, target) = getCell(i, j);
      }
    }
  }


  // timing("Replicating grid took" + tick.tock());
}


/**
 * @brief get the total mass of the grid.
 */
Float Grid::collectTotalMass() {

  timer::Timer tick(timer::Category::CollectMass);
  message("Collecting total mass in grid.", logging::LogLevel::Debug);

  Float  total = 0.;
  size_t first = getFirstCellIndex();
  size_t last  = getLastCellIndex();

  if (Dimensions == 1) {
    for (size_t i = first; i < last; i++) {
      total += getCell(i).getPrim().getRho();
    }

    total *= getDx();
  }

  else if (Dimensions == 2) {
    for (size_t j = first; j < last; j++) {
      for (size_t i = first; i < last; i++) {
        total += getCell(i, j).getPrim().getRho();
      }
    }

    total *= getDx() * getDx();
  }


  // message("Collecting total mass in grid took" + tick.tock());

  return total;
}


/**
 * Reset all fluxes of the grid (both primitive and conservative) to zero.
 */
void Grid::resetFluxes() {

  timer::Timer tick(timer::Category::Reset);

  if (Dimensions != 2) {
    error("Not Implemented");
    return;
  }

  size_t first = getFirstCellIndex();
  size_t last  = getLastCellIndex();

#pragma omp target teams loop
  for (size_t j = first; j < last; j++) {
    for (size_t i = first; i < last; i++) {
      // getCell(i, j).getPFlux().clear();
      getCell(i, j).getCFlux().clear();
    }
  }

  // timing("Resetting fluxes took" + tick.tock());
}


/**
 * runs through interior cells and calls prim2cons()
 * on each.
 */
void Grid::convertPrim2Cons() {

  timer::Timer tick(timer::Category::Convert);

  if (Dimensions != 2) {
    error("Not Implemented");
    return;
  }

  size_t first = getFirstCellIndex();
  size_t last  = getLastCellIndex();

  for (size_t j = first; j < last; j++) {
    for (size_t i = first; i < last; i++) {
      getCell(i, j).prim2cons();
    }
  }

  // timing("Converting primitive to conserved vars took" + tick.tock());
}


/**
 * runs through interior cells and calls cons2prim()
 * on each.
 */
void Grid::convertCons2Prim() {

  timer::Timer tick(timer::Category::Convert);

  if (Dimensions != 2) {
    error("Not Implemented");
    return;
  }

  size_t first = getFirstCellIndex();
  size_t last  = getLastCellIndex();

  for (size_t j = first; j < last; j++) {
    for (size_t i = first; i < last; i++) {
      getCell(i, j).cons2prim();
    }
  }

  // timing("Converting conserved to primitive vars took" + tick.tock());
}


/**
 * @brief enforce boundary conditions.
 * This function only picks out the pairs of real
 * and ghost cells in a row or column and then
 * calls the function that actually copies the data.
 */
void Grid::applyBoundaryConditions() {

  timer::Timer tick(timer::Category::BoundaryConditions);
  message("Applying boundary conditions.", logging::LogLevel::Debug);

  const size_t nbc       = getNBC();
  const size_t firstReal = getFirstCellIndex();
  const size_t lastReal  = getLastCellIndex();

  // Select which BC to use.
  BC::BoundaryFunctionPtr real2ghost = selectBoundaryFunction();

  // Make some space.
  Boundary real_left(nbc);
  Boundary real_right(nbc);
  Boundary ghost_left(nbc);
  Boundary ghost_right(nbc);

  if (Dimensions == 1) {
    for (size_t i = 0; i < firstReal; i++) {
      real_left[i]   = &(getCell(firstReal + i));
      real_right[i]  = &(getCell(lastReal - firstReal + i));
      ghost_left[i]  = &(getCell(i));
      ghost_right[i] = &(getCell(lastReal + i));
    }
    real2ghost(real_left, real_right, ghost_left, ghost_right, nbc, 0);
  }

  else if (Dimensions == 2) {

    // left-right boundaries
#pragma omp target teams loop
    for (size_t j = firstReal; j < lastReal; j++) {
      for (size_t i = 0; i < firstReal; i++) {
        real_left[i]   = &(getCell(firstReal + i, j));
        real_right[i]  = &(getCell(lastReal - firstReal + i, j));
        ghost_left[i]  = &(getCell(i, j));
        ghost_right[i] = &(getCell(lastReal + i, j));
      }
// #ifdef DONT_DO_THIS
      real2ghost(real_left, real_right, ghost_left, ghost_right, nbc, 0);
// #endif
    }

#ifdef DONT_DO_THOS
    // upper-lower boundaries
    // left -> lower, right -> upper
#pragma omp target teams loop
    for (size_t i = firstReal; i < lastReal; i++) {
      for (size_t j = 0; j < firstReal; j++) {
        real_left[j]   = &(getCell(i, firstReal + j));
        real_right[j]  = &(getCell(i, lastReal - firstReal + j));
        ghost_left[j]  = &(getCell(i, j));
        ghost_right[j] = &(getCell(i, lastReal + j));
      }
      real2ghost(real_left, real_right, ghost_left, ghost_right, nbc, 1);
    }
#endif
  } else {
    error("Not implemented.");
  }

  // timing("Applying boundary conditions took " + tick.tock());
}


/**
 * Selects and returns the function that applied the correct boundary
 * conditions from ghost to real cells.
 *
 * The returned function takes 6 parameters, in this order:
 * @param realL:     array of pointers to real cells with lowest index
 * @param realR:     array of pointers to real cells with highest index
 * @param ghostL:    array of pointers to ghost cells with lowest index
 * @param ghostR:    array of pointers to ghost cells with highest index
 * @param nbc:       number of boundary cells.
 * @param dimension: dimension integer. 0 for x, 1 for y. Needed for
 *                   reflective boundary conditions.
 *
 * All arguments are arrays of size Grid::_nbc (number of boundary cells).
 * Lowest array index is also lowest index of cell in grid.
 */
#pragma omp declare target
//std::function<void(Boundary&, Boundary&, Boundary&, Boundary&, const size_t)> Grid::
BC::BoundaryFunctionPtr Grid::selectBoundaryFunction() {

  switch (getBoundaryType()) {
  case BC::BoundaryCondition::Periodic:
    return &BC::periodic;

  case BC::BoundaryCondition::Reflective:
    return &BC::reflective;

  case BC::BoundaryCondition::Transmissive:
    return &BC::transmissive;

  default:
    // std::stringstream msg;
    // msg << "Treatment for boundary conditions of type ";
    // msg << BC::getBoundaryConditionName(getBoundaryType());
    // msg << " not defined.";
    // error(msg.str());
    return &BC::periodic;
  }
}
#pragma omp end declare target


/**
 * @brief print out the grid.
 *
 * @param boundaries if true, print out boundary cells too.
 * @param conserved if true, print out conserved state instead of primitive state.
 */
void Grid::printGrid(bool boundaries, bool conserved) {


  size_t first = getFirstCellIndex();
  size_t last  = getLastCellIndex();
  size_t start = first;
  size_t end   = last;

  if (boundaries) {
    start = 0;
    end   = getNxTot();
  }

  std::stringstream out;
  out << "Full grid output of ";
  if (conserved) {
    out << "conserved states";
  } else {
    out << "primitive states";
  }

  if (boundaries) {
    out << " (including boundaries)";
  }
  out << "\n";


  if (Dimensions == 1) {

    for (size_t i = start; i < end; i++) {
      if (conserved) {
        out << getCell(i).getCons().toString() << " ";
      } else {
        out << getCell(i).getPrim().toString() << " ";
      }

      bool at_boundary = (i == first - 1) or (i == last - 1);
      if (boundaries and at_boundary) {
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
    for (int j = end - 1; j >= 0; j--) {

      for (size_t i = start; i < end; i++) {
        if (conserved) {
          out << getCell(i, j).getCons().toString() << " ";
        } else {
          out << getCell(i, j).getPrim().toString() << " ";
        }

        bool at_boundary = (i == first - 1) or (i == last - 1);
        if (boundaries and at_boundary) {
          out << " | ";
        } else {
          if (i != end - 1) {
            out << ", ";
          }
        }
      }

      out << "\n";
      bool at_boundary = (j == static_cast<int>(first)) or (j == static_cast<int>(last));
      if (boundaries and at_boundary) {
        out << "\n";
      }
    }

  } else {
    error("Not implemented");
  }

  std::cout << out.str();
}


/**
 * @brief print out a single quantity over the entire grid.
 *
 * @param boundaries if true, print out boundary cells too.
 */
void Grid::printGrid(const char* quantity, bool boundaries) {

  size_t first = getFirstCellIndex();
  size_t last  = getLastCellIndex();
  size_t start = first;
  size_t end   = last;

  if (boundaries) {
    start = 0;
    end   = getNxTot();
  }

  std::stringstream out;
  out << "Full grid output of " << quantity;
  if (boundaries) {
    out << " (including boundaries)";
  }
  out << "\n";

  constexpr size_t w = grid_print_width;
  constexpr size_t p = grid_print_precision;


  if (Dimensions == 1) {

    for (size_t i = start; i < end; i++) {
      out << std::setw(w) << std::setprecision(p);
      out << getCell(i).getQuantityForPrintout(quantity);

      bool at_boundary = (i == first - 1) or (i == last - 1);
      if (boundaries and at_boundary) {
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
    for (int j = end - 1; j >= 0; j--) {

      for (size_t i = start; i < end; i++) {
        out << std::setw(w) << std::setprecision(p);
        out << getCell(i, j).getQuantityForPrintout(quantity);

        bool at_boundary = (i == first - 1) or (i == last - 1);
        if (boundaries and at_boundary) {
          out << " | ";
        } else {
          if (i != end - 1) {
            out << ", ";
          }
        }
      }

      out << "\n";
      bool at_boundary = (j == static_cast<int>(first)) or (j == static_cast<int>(last));
      if (boundaries and at_boundary) {
        out << "\n";
      }
    }

  } else {
    error("Not implemented");
  }

  std::cout << out.str() << "\n";
}
