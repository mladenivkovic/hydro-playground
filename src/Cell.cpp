#include "Cell.h"

using namespace hydro_playground;

// define the static copy. Calls the default constructor but
// the user has to call InitCells()
Grid Grid::Instance;

Grid::Grid()
{
/* Empty body */
}

void Grid::InitGrid()
{ 
  /**
   * This time the convention is different.
   * _cell(0,0)             is the top left cell.
   * _cell(nxtot-1,0)       is the bottom left cell
   * _cell(0,nxtot-1)       is the top-right cell
   * _cell(nxtot-1,nxtot-1) is the bottom right cell
   * 
  */
  int       nxTot = parameters::Parameters::Instance.getNxTot();
  int       Bc    = parameters::Parameters::Instance.getBc();
  Precision Dx    = parameters::Parameters::Instance.getDx();
  if (Dimensions==1)
  {
    // make some room in the vector...
    _cells.reserve( nxTot );

    for (int i=0; i<nxTot; i++)
    {
      getCell(i).setX( (i-Bc+0.5) * Dx );
      getCell(i).setId(i);
    }

  }
  else if (Dimensions==2) 
  {
    // make some room in the vector...
    _cells.reserve( nxTot * nxTot );

    for (int i=0; i<nxTot; i++)
    {
      for (int j=0; j<nxTot; j++)
      {
        getCell(i,j).setX( (i-Bc+0.5) * Dx );
        getCell(i,j).setY( (j-Bc+0.5) * Dx );

        // this used to be i + j * pars.nxtot, but i have altered the 
        // convention this time around
        getCell(i,j).setId( i * nxTot + j );
      }
    }

  }
  else assert(false);
}

Cell& Grid::getCell(int i, int j)
{
  static int totalCells     = parameters::Parameters::Instance.getNxTot();
  static constexpr int dim2 = (Dimensions==2);
  /*
  This is hacky, but if we are in 1d, we wanna return cells[i], 
  but if we are in 2d we wanna return cells[ i*totalCells + j ]

  We trust that this function is called with j=0 in 1d
  */
  return _cells[ 
    i                  * dim2 + // in 2d this line is zero
    (i*totalCells + j) * dim2   // in 1d this line is zero
   ];
}

Precision Grid::GetTotalMass()
{
  /*
  Get total mass on the grid.

  Writing it naively here. There are a few ways to speed it up.
  Does this code get called often?
  */
  Precision total = 0;
  // would get a bit crowded down there if we don't define another variable for this
  int       bc    = parameters::Parameters::Instance.getBc();
  int       nx    = parameters::Parameters::Instance.getNx();

  // there is a way to do this with a branch or 
  // a macro. do it like this for now
  if ( Dimensions==1 )
  {
    for (int i=bc; i < bc+nx; i++)
      // this is a mouthful...
      total += getCell(i).getPrim().getRho();
    
    total *= parameters::Parameters::Instance.getDx();
  }

  else if (Dimensions==2)
  {
    for (int i=bc; i < bc+nx; i++)
    for (int j=bc; j < bc+nx; j++)
    {
      total += getCell(i,j).getPrim().getRho();
    }
    
    total *= parameters::Parameters::Instance.getDx() * parameters::Parameters::Instance.getDx();
  }
  return total;
}

void Grid::resetFluxes()
{
  constexpr int dim2 = (Dimensions == 2);
  int           bc   = parameters::Parameters::Instance.getBc();
  int           nx   = parameters::Parameters::Instance.getNx();

  for (int i=bc; i<bc + nx; i++)
  for (int j=bc*dim2; j<(bc+nx)*dim2; j++)
  {
    // if we are in 1d, j will be fixed to zero
    getCell(i,j).getPrim().resetToInitialState();
    getCell(i,j).getCons().resetToInitialState();
  }
}

void Grid::getCStatesFromPstates()
{
  /**
   * runs through interior cells. Calls PrimitveToConserved()
   * on each.
  */
  constexpr int dim2 = (Dimensions == 2);
  int           bc   = parameters::Parameters::Instance.getBc();
  int           nx   = parameters::Parameters::Instance.getNx();

  for (int i=bc; i<bc + nx; i++)
  for (int j=bc*dim2; j<(bc+nx)*dim2; j++)
  {
    // if we are in 1d, j will be fixed to zero
    getCell(i,j).PrimitiveToConserved();
  }
}

void Grid::getPStatesFromCstates()
{
  /**
   * runs through interior cells. Calls ConservedToPrimitve()
   * on each.
  */
  constexpr int dim2 = (Dimensions == 2);
  int           bc   = parameters::Parameters::Instance.getBc();
  int           nx   = parameters::Parameters::Instance.getNx();

  for (int i=bc; i<bc + nx; i++)
  for (int j=bc*dim2; j<(bc+nx)*dim2; j++)
  {
    // if we are in 1d, j will be fixed to zero
    getCell(i,j).ConservedToPrimitive();
  }
}

void Grid::setBoundary()
{
  std::vector<Cell*> realLeft  ( parameters::Parameters::Instance.getBc() );
  std::vector<Cell*> realRight ( parameters::Parameters::Instance.getBc() );
  std::vector<Cell*> ghostLeft ( parameters::Parameters::Instance.getBc() );
  std::vector<Cell*> ghostRight( parameters::Parameters::Instance.getBc() );

  int bc    = parameters::Parameters::Instance.getBc();
  int nx    = parameters::Parameters::Instance.getNx();
  int bctot = parameters::Parameters::Instance.getBcTot();

  // doesn't look like we will need this code often. so avoid hacky stuff
  if (Dimensions==1)
  {
    for (int i=0; i<bc; i++)
    {
      realLeft[i]   = &(getCell( bc+i ));
      realRight[i]  = &(getCell( nx+i )); /* = last index of a real cell = BC + (i + 1) */
      ghostLeft[i]  = &(getCell(i));
      ghostRight[i] = &(getCell( nx+bc+i ));
    }
    // call cell - real - to ghost!
    realToGhost( realLeft, realRight, ghostLeft, ghostRight );

  }

  else if (Dimensions==2)
  {
    // left-right boundaries

    // run over all the rows
    for (int i=0; i<nx+bctot; i++)
    {
      for (int j=0; j<bc; j++)
      {
        realLeft[j]   = &(getCell( i, bc+j ));
        realRight[j]  = &(getCell( i, nx+j ));
        ghostLeft[j]  = &(getCell( i,j ));
        ghostRight[j] = &(getCell( i, nx+bc+j ));
      }
    // here is the first major difference from switching convention 
    // (could always switch it back) - this code was used identically before
    // but used dimension 0
    realToGhost( realLeft, realRight, ghostLeft, ghostRight, 1 );
    }
  }

  // upper-lower boundaries
  for (int j=0; j<nx+bctot; j++)
  {
    for (int i=0; i<bc; i++)
    {
      realLeft[i]   = &(getCell( bc+i, j ));
      realRight[i]  = &(getCell( nx+i, j ));
      ghostLeft[i]  = &(getCell( i,j ));
      ghostRight[i] = &(getCell( nx+bc+i, j ));
    }
    realToGhost( realLeft, realRight, ghostLeft, ghostRight, 0 );
  }
}

void Grid::realToGhost(
  std::vector<Cell*> realLeft, 
  std::vector<Cell*> realRight, 
  std::vector<Cell*> ghostLeft, 
  std::vector<Cell*> ghostRight,
  int dimension) // dimension defaults to 0
{
  // prevents crowding down there
  int bc = parameters::Parameters::Instance.getBc();

  switch ( parameters::Parameters::Instance.getBoundary() )
  {
    case parameters::Parameters::BoundaryCondition::Periodic:
    {
      for (int i=0; i<bc; i++)
      {
        ghostLeft[i] ->CopyBoundaryData( realLeft[i] );
        ghostRight[i]->CopyBoundaryData( realRight[i] );
      }

    } break;

    case parameters::Parameters::BoundaryCondition::Reflective:
    {
      for (int i=0; i<bc; i++)
      {
        ghostLeft[i] ->CopyBoundaryDataReflective( realLeft[i] , dimension);
        ghostRight[i]->CopyBoundaryDataReflective( realRight[i], dimension);
      }
    } break;

    case parameters::Parameters::BoundaryCondition::Transmissive:
    {
      for (int i=0; i<bc; i++)
      {
        ghostLeft[i] ->CopyBoundaryData(realLeft[i]);

        // assumption that this vector has length "bc".
        ghostRight[i]->CopyBoundaryData( ( realRight.back() - i ) ); // need to dereference to obtain Cell* pointer

        //this line used to read:
        //cell_copy_boundary_data(realR[BC - 1 - i], ghostR[i]);

      }
    } break;
  }

}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

/*
Constructor for the cell. This has everything from the old
cell_init_cell() function
*/
Cell::Cell():
  _id(0), _x(0), _y(0),
  // and the ideal gasses too
  _prim(),  _cons(),
  _pflux(), _cflux(),
  _acc({0,0})
{/* Empty body. */}



void Cell::CopyBoundaryData(const Cell* real)
{
  // This should be called from within the ghost

  // copy everything from the other!
  _prim = real->getPrim();
  _cons = real->getCons();
  // check this is taking a deep copy for real!
}

void Cell::CopyBoundaryDataReflective(const Cell* real, int dimension)
{
  /*
  * Copies the data we need. Dimension indiciates which dimension
  * We flip the velocities
  */

  // This should be called from within the ghost
  _prim = real->getPrim();
  _cons = real->getCons();

  // flip the velocities in specified dimension
  Precision u = getPrim().getU(dimension);
  getPrim().setU(dimension, -1. * u);

  Precision rhou = getCons().getRhou(dimension);
  getCons().setRhou(dimension, -1. * rhou);
}

/*
Getters and setters for cell!
*/
void Cell::setX(Precision x) {_x = x;}
void Cell::setY(Precision y) {_y = y;}

void Cell::setId(int id) {_id = id;}
