#include "Cell.h"

using namespace hydro_playground;

Grid::Grid()
{
  cells.reserve(1);
}

Cell& Grid::getCell(int i, int j=0)
{
  int nxTot = parameters::Parameters::Instance.getNx();

  return cells[i*nxTot + j];
}
