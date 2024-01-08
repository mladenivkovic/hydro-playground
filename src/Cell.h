#include "Config.h"
#include "Gas.h"
#include "Parameters.h"

#include <vector>



namespace hydro_playground{
  class Cell;

  // template <int Dimensions>
  class Grid{
    private:
    // might have to be a vector, size to be determined later
      std::vector<Cell> cells;

    public:
      Grid();
      Cell& getCell(int i, int j);
  };
  
  class Cell{
    public:
      // variable for the compiler to shush
      static const int x = 1;
  };

} // hydro_playground
