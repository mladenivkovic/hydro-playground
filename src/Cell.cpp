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
      _cells[i].setX( (i-Bc+0.5) * Dx );
      _cells[i].setId(i);
    }

  }
  if (Dimensions==2) 
  {
    totalCells = parameters::Parameters::Instance.getNxTot() * parameters::Parameters::Instance.getNxTot();
    _cells.reserve( totalCells * totalCells );
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

  // for (int i=0; i<totalCells; i++) cells.emplace_back();
}

// Cell& Grid::operator()(int i, int j=0)
// {
// }

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

/*
Constructor for the cell. This has everything from the old
cell_init_cell() function
*/
Cell::Cell():
  _id(0), _x(0), _y(0),
  _acc({0,0})
  {
    /* Empty body. We trust that the default constructors
    for the ideal gasses will be called
   */
  }

/*
Constructor when x, y, id are already known
*/
Cell::Cell(int id, Precision x, Precision y):
  _id(id), _x(x), _y(y), _acc({0,0})
{/*Empty body*/}


/*
Getters and setters for cell!
*/
void Cell::setX(Precision x) {_x = x;}
void Cell::setY(Precision y) {_y = y;}

void Cell::setId(int id) {_id = id;}

