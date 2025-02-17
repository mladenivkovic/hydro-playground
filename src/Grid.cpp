#include "Cell.h"
#include "Grid.h"
#include <cassert>

#include "Parameters.h"

// define the static copy. Calls the default constructor but
// the user has to call InitCells()
grid::Grid grid::Grid::Instance;


void grid::Grid::InitGrid() {
  /**
   * _cell(0,0)             is the bottom left cell.
   * _cell(nxtot-1,0)       is the bottom right cell
   * _cell(0,nxtot-1)       is the top-left cell
   * _cell(nxtot-1,nxtot-1) is the top-right cell
   *
   */
  size_t  nxTot = parameters::Parameters::Instance.getNxTot();
  size_t  bc    = parameters::Parameters::Instance.getNBC();
  float_t dx    = parameters::Parameters::Instance.getDx();
  if (Dimensions == 1) {
    // make some room in the vector...
    _cells.resize(nxTot);

    for (size_t i = 0; i < nxTot; i++) {
      getCell(i).setX((i - bc + 0.5) * dx);
      getCell(i).setId(i);
    }

  } else if (Dimensions == 2) {
    // make some room in the vector...
    _cells.resize(nxTot * nxTot);

    for (size_t i = 0; i < nxTot; i++) {
      for (size_t j = 0; j < nxTot; j++) {
        getCell(i, j).setX((i - bc + 0.5) * dx);
        getCell(i, j).setY((j - bc + 0.5) * dx);

        // this used to be i + j * pars.nxtot, but i have altered the
        // convention this time around
        getCell(i, j).setId(i + j * nxTot);
      }
    }

  } else
    error("Not implemented yet");
}

/**
 * [density, velocity, pressure]
 */
void grid::Grid::SetInitialConditions(int position, std::vector<float_t> vals)
{
  assert(
    (vals.size() == 4 and Dimensions==2)
    or
    (vals.size() == 3 and Dimensions==1)
  );
  // Let's set i,j based on the position in the array we passed in
  int i,j;
  if ( Dimensions == 1 ) { i=position; j=0; }
  if ( Dimensions == 2 )
  {
    i = position % parameters::Parameters::Instance.getNx();
    j = position / parameters::Parameters::Instance.getNx();
  }

  // alias the bc value
  size_t bc = parameters::Parameters::Instance.getNBC();

  getCell(i+bc, j+bc).getPrim().setRho( vals[0] );
  getCell(i+bc, j+bc).getPrim().setU(0, vals[1]);
  if(Dimensions==1)
  {
    getCell(i+bc, j+bc).getPrim().setP(vals[2]);
  }
  if(Dimensions==2)
  {
    getCell(i+bc, j+bc).getPrim().setU(1,vals[2]);
    getCell(i+bc, j+bc).getPrim().setP(vals[3]);
  }
}


cell::Cell& grid::Grid::getCell(size_t i) {

#if DEBUG_LEVEL > 0
  if (Dimensions != 1) {
    error("This function is for 1D only!")
  }
#endif
  return _cells[i];
}


cell::Cell& grid::Grid::getCell(size_t i, size_t j) {
  static size_t nxTot = parameters::Parameters::Instance.getNxTot();

#if DEBUG_LEVEL > 0
  if (Dimensions != 2) {
    error("This function is for 2D only!")
  }
#endif
  return _cells[i + j * nxTot];
}

float_t grid::Grid::GetTotalMass() {
  float_t total = 0;
  size_t  bc    = parameters::Parameters::Instance.getNBC();
  size_t  nx    = parameters::Parameters::Instance.getNx();

  if (Dimensions == 1) {
    for (size_t i = bc; i < bc + nx; i++) {
      total += getCell(i).getPrim().getRho();
    }

    total *= parameters::Parameters::Instance.getDx();
  }

  else if (Dimensions == 2) {
    for (size_t i = bc; i < bc + nx; i++) {
      for (size_t j = bc; j < bc + nx; j++) {
        total += getCell(i, j).getPrim().getRho();
      }
    }

    total *= parameters::Parameters::Instance.getDx() * parameters::Parameters::Instance.getDx();
  }
  return total;
}

void grid::Grid::resetFluxes() {
  constexpr auto dim2 = static_cast<size_t>(Dimensions == 2);
  size_t         bc   = parameters::Parameters::Instance.getNBC();
  size_t         nx   = parameters::Parameters::Instance.getNx();

  for (size_t i = bc; i < bc + nx; i++) {
    for (size_t j = bc * dim2; j < (bc + nx) * dim2; j++) {
      // if we are in 1d, j will be fixed to zero
      getCell(i, j).getPrim().resetToInitialState();
      getCell(i, j).getCons().resetToInitialState();
    }
  }
}

void grid::Grid::getCStatesFromPstates() {
  /**
   * runs through interior cells. Calls PrimitveToConserved()
   * on each.
   */
  constexpr auto dim2 = static_cast<size_t>(Dimensions == 2);
  size_t         bc   = parameters::Parameters::Instance.getNBC();
  size_t         nx   = parameters::Parameters::Instance.getNx();

  for (size_t i = bc; i < bc + nx; i++) {
    for (size_t j = bc * dim2; j < (bc + nx) * dim2; j++) {
      // if we are in 1d, j will be fixed to zero
      getCell(i, j).PrimitiveToConserved();
    }
  }
}

void grid::Grid::getPStatesFromCstates() {
  /**
   * runs through interior cells. Calls ConservedToPrimitve()
   * on each.
   */
  constexpr auto dim2 = static_cast<size_t>(Dimensions == 2);
  size_t         bc   = parameters::Parameters::Instance.getNBC();
  size_t         nx   = parameters::Parameters::Instance.getNx();

  for (size_t i = bc; i < bc + nx; i++) {
    for (size_t j = bc * dim2; j < (bc + nx) * dim2; j++) {
      // if we are in 1d, j will be fixed to zero
      getCell(i, j).ConservedToPrimitive();
    }
  }
}

void grid::Grid::setBoundary() {
  std::vector<cell::Cell*> realLeft(parameters::Parameters::Instance.getNBC());
  std::vector<cell::Cell*> realRight(parameters::Parameters::Instance.getNBC());
  std::vector<cell::Cell*> ghostLeft(parameters::Parameters::Instance.getNBC());
  std::vector<cell::Cell*> ghostRight(parameters::Parameters::Instance.getNBC());

  size_t bc    = parameters::Parameters::Instance.getNBC();
  size_t nx    = parameters::Parameters::Instance.getNx();
  size_t bctot = parameters::Parameters::Instance.getNBCTot();

  // doesn't look like we will need this code often. so avoid hacky stuff
  if (Dimensions == 1) {
    for (size_t i = 0; i < bc; i++) {
      realLeft[i]   = &(getCell(bc + i));
      realRight[i]  = &(getCell(nx + i)); /* = last index of a real cell = BC + (i + 1) */
      ghostLeft[i]  = &(getCell(i));
      ghostRight[i] = &(getCell(nx + bc + i));
    }
    realToGhost(realLeft, realRight, ghostLeft, ghostRight);
  }

  else if (Dimensions == 2) {
    // left-right boundaries
    for (size_t j = 0; j < nx + bctot; j++) {
      for (size_t i = 0; i < bc; i++) {
        realLeft[i]   = &(getCell(bc + i, j));
        realRight[i]  = &(getCell(nx + i, j));
        ghostLeft[i]  = &(getCell(i, j));
        ghostRight[i] = &(getCell(nx + bc + i, j));
      }
      realToGhost(realLeft, realRight, ghostLeft, ghostRight, 0);
    }
  }

  // upper-lower boundaries
  for (size_t i = 0; i < nx + bctot; i++) {
    for (size_t j = 0; j < bc; j++) {
      realLeft[j]   = &(getCell(bc + i, j));
      realRight[j]  = &(getCell(nx + i, j));
      ghostLeft[j]  = &(getCell(i, j));
      ghostRight[j] = &(getCell(nx + bc + i, j));
    }
    realToGhost(realLeft, realRight, ghostLeft, ghostRight, 1);
  }
}

void grid::Grid::realToGhost(
  std::vector<cell::Cell*> realLeft,
  std::vector<cell::Cell*> realRight,
  std::vector<cell::Cell*> ghostLeft,
  std::vector<cell::Cell*> ghostRight,
  int                      dimension
) // dimension defaults to 0
{
  // prevents crowding down there
  size_t bc = parameters::Parameters::Instance.getNBC();

  switch (parameters::Parameters::Instance.getBoundaryType()) {
  case parameters::Parameters::BoundaryCondition::Periodic: {
    for (size_t i = 0; i < bc; i++) {
      ghostLeft[i]->CopyBoundaryData(realLeft[i]);
      ghostRight[i]->CopyBoundaryData(realRight[i]);
    }

  } break;

  case parameters::Parameters::BoundaryCondition::Reflective: {
    for (size_t i = 0; i < bc; i++) {
      ghostLeft[i]->CopyBoundaryDataReflective(realLeft[i], dimension);
      ghostRight[i]->CopyBoundaryDataReflective(realRight[i], dimension);
    }
  } break;

  case parameters::Parameters::BoundaryCondition::Transmissive: {
    for (size_t i = 0; i < bc; i++) {
      ghostLeft[i]->CopyBoundaryData(realLeft[i]);

      // assumption that this vector has length "bc".
      ghostRight[i]->CopyBoundaryData((realRight.back() - i)
      ); // need to dereference to obtain Cell* pointer

      // this line used to read:
      // cell_copy_boundary_data(realR[BC - 1 - i], ghostR[i]);
    }
  } break;
  }
}
