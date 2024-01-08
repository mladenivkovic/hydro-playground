#include "Config.h"
#include "Gas.h"
#include "Parameters.h"
#include <cassert>

#include "Logging.h"

#include <vector>



namespace hydro_playground{
  class Cell;

  // template <int Dimensions>
  class Grid{
    private:
      std::vector<Cell> cells;

    public:
      Grid();
      // Cell&       getCell(int i, int j);
      // const Cell& getCell(int i, int j) const;

      void InitCells();

      
      // access cells with overloaded operator
      Cell& operator() (int i, int j);
      // const version of above, compiler should choose appropriate one
      const Cell& operator() (int i, int j) const;

      // static copy for global access
      static Grid  Instance;
      static Grid& getInstance() {return Instance;}
  };
  
  class Cell{
    public:
      // variable for the compiler to shush
      static const int x = 1;

      Cell() = default;
  };

} // hydro_playground
