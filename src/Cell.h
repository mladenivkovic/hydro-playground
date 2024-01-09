#pragma once

#include "Config.h"
#include "Gas.h"
#include "Parameters.h"
#include "Logging.h"

#include <cassert>
#include <vector>



namespace hydro_playground{
  class Cell;

  // template <int Dimensions>
  class Grid{
    private:
      std::vector<Cell> _cells;

    public:
      Grid();
      Cell&       getCell(int i, int j=0);
      // const Cell& getCell(int i, int j) const;

      void      InitCells();
      Precision GetTotalMass();

      
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
      // Cell(int id, Precision x, Precision y);
    private:
      int _id;

      /*
      Positions of centres
      */
      Precision _x;
      Precision _y;

      IdealGas::PrimitiveState _prim;
      IdealGas::ConservedState _cons;

      IdealGas::PrimitiveState _pflux;
      IdealGas::ConservedState _cflux;

      std::array<Precision, 2> _acc;

    public:
      /* getters and setters */
      void setX(Precision x);
      void setY(Precision x);

      void setId(int id);

      // return const ref to the above
      const IdealGas::PrimitiveState& getPrim()  const { return _prim; }
      const IdealGas::ConservedState& getCons()  const { return _cons; }
      const IdealGas::PrimitiveState& getPFlux() const { return _pflux; }
      const IdealGas::ConservedState& getCFlux() const { return _cflux; }
  };

} // hydro_playground
