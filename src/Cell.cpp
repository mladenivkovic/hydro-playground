#include "Cell.h"

using namespace hydro_playground;

// define the static copy. Calls the default constructor but
// the user has to call InitCells()
Grid Grid::Instance;

Grid::Grid()
{
/* Empty body */
}

void Grid::InitCells()
{ 
  int       totalCells = parameters::Parameters::Instance.getNxTot();
  int       Bc         = parameters::Parameters::Instance.getBc();
  Precision Dx         = parameters::Parameters::Instance.getDx();
  if (Dimensions==1)
  {
    // floods the vector with default constructed cells
    _cells.resize( totalCells );

    for (int i=0; i<totalCells; i++)
    {
      getCell(i).setX( (i-Bc+0.5) * Dx );
      getCell(i).setId(i);
    }

  }
  else if (Dimensions==2) 
  {
    _cells.resize( totalCells * totalCells );
    for (int j=0; j<totalCells; j++)
    {
      for (int i=0; i<totalCells; i++)
      {
        getCell(i,j).setX( (i-Bc+0.5) * Dx  );
        getCell(i,j).setY( (j-Bc+0.5) * Dx  );

        // this used to be i + j * pars.nxtot, but i disagree...
        getCell(i,j).setId( i * totalCells + j );
      }
    }

  }
  else assert(false);
}

Cell& Grid::getCell(int i, int j)
{
  static int totalCells = parameters::Parameters::Instance.getNxTot();
  /*
  This is hacky, but if we are in 1d, we wanna return cells[i], 
  but if we are in 2d we wanna return cells[ i*totalCells + j ]

  We trust that this function is called with j=0 in 1d
  */
  return _cells[ 
    i * (Dimensions==1) +
    i * totalCells * (Dimensions==2) +
    j
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

/*
Constructor when x, y, id are already known
*/
// Cell::Cell(int id, Precision x, Precision y):
//   _id(id), _x(x), _y(y), _acc({0,0})
// {/*Empty body*/}


/*
Getters and setters for cell!
*/
void Cell::setX(Precision x) {_x = x;}
void Cell::setY(Precision y) {_y = y;}

void Cell::setId(int id) {_id = id;}

