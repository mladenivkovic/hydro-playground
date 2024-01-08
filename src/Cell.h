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
      std::vector<Cell> _cells;

    public:
      Grid();
      Cell&       getCell(int i, int j);
      // const Cell& getCell(int i, int j) const;

      void InitCells();

      
      // access cells with overloaded operator
      // Cell& operator() (int i, int j);
      // const version of above, compiler should choose appropriate one
      const Cell& operator() (int i, int j) const;

      // static copy for global access
      static Grid  Instance;
      static Grid& getInstance() {return Instance;}
  };
  
  class Cell{
    public:

      Cell();
      Cell(int id, Precision x, Precision y);
    private:
      int _id;

      /*
      Positions of centres
      */
      Precision _x;
      Precision _y;

      IdealGas::PrimitiveState prim;
      IdealGas::ConservedState cons;

      IdealGas::PrimitiveState pflux;
      IdealGas::ConservedState cflux;

      std::array<Precision, 2> _acc;

    public:
      /* getters and setters */
      void setX(Precision x);
      void setY(Precision x);

      void setId(int id);
  };

} // hydro_playground
