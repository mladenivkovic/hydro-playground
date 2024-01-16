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

      void setBoundary();
      void realToGhost(std::vector<Cell*>, std::vector<Cell*>, std::vector<Cell*>, std::vector<Cell*>, int dimension = 0);
      
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

      //! Should be called from within the ghost
      void CopyBoundaryData(const Cell* real);
      //! Should be called from within the ghost
      void CopyBoundaryDataReflective(const Cell* real, int dimension);

      //! Calls conserved to primitive on the members
      void ConservedToPrimitive() { _prim.ConservedToPrimitive(_cons); };
      //! Calls primitive to conserved on the members
      void PrimitiveToConserved() { _cons.PrimitiveToConserved(_prim); };

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
      
      // const versions to shush the compiler
      const IdealGas::PrimitiveState& getPrim() const  { return _prim; }
      const IdealGas::ConservedState& getCons() const  { return _cons; }
  };

} // hydro_playground
