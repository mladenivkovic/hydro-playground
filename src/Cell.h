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

      void      InitGrid();
      Precision GetTotalMass();

      void getCStatesFromPstates();
      void getPStatesFromCstates();
      void resetFluxes();


      
      // static copy for global access
      static Grid  Instance;
      static Grid& getInstance() {return Instance;}
  };
  
  class Cell{
    public:
      //! Standard constructor
      Cell();
      //! copy assignment, for copying boundary data
      //! Return reference to this, for chaining calls
      Cell& operator= (const Cell& other) = default;

      //! This function generates a new temp cell to be stolen from
      void CopyBoundaryDataReflective(const Cell& real);

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
      // void 

      /* leaving these for now */
      std::string getIndexString();
    public:
      /* getters and setters */
      void setX(Precision x);
      void setY(Precision x);

      void               setId(int id);
      std::pair<int,int> getIJ();


      // return refs to the above
      IdealGas::PrimitiveState& getPrim()  { return _prim; }
      IdealGas::ConservedState& getCons()  { return _cons; }
      IdealGas::PrimitiveState& getPFlux() { return _pflux; }
      IdealGas::ConservedState& getCFlux() { return _cflux; }
  };

} // hydro_playground
