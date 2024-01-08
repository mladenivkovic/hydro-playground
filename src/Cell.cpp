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
  std::cout << "here\n";
  int totalCells;
  if (Dimensions==1) totalCells = parameters::Parameters::Instance.getNxTot();
  if (Dimensions==2) totalCells = parameters::Parameters::Instance.getNxTot() * parameters::Parameters::Instance.getNxTot();
  else assert(false);

  cells.reserve( totalCells );
  for (int i=0; i<totalCells; i++) cells.emplace_back();
}

Cell& Grid::operator()(int i, int j=0)
{
  static int totalCells = parameters::Parameters::Instance.getNxTot();
  /*
  This is hacky, but if we are in 1d, we wanna return cells[i], 
  but if we are in 2d we wanna return cells[ i*totalCells + j ]

  We trust that this function is called with j=0 in 1d
  */
  return cells[ 
    i * (Dimensions==1) +
    i * totalCells * (Dimensions==2) +
    j
   ];
}
