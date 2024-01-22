#pragma once
#include <string>

/*

Turning this class into singleton pattern. In any file where you 
include Parameters.h, you can call 
hydro_playground::parameters::Parameters::Instance._nstepsLog (for example)
and it will be this single global copy. 

Since the member variables are nonstatic now i've un-deleted the default 
constructor. This will be called by default on the static member anyhow

We could remove the namespaceing
here as it is a bit of a mouthful to type... 

It's up to us whether we make the instance itself private and use the 
getter or just make it public. It doesn't make a difference, since we
need to return a reference anyway...

*/

namespace hydro_playground {
  namespace parameters {

    class Parameters {
    
    public:
      enum class BoundaryCondition {
        Periodic,
        Reflective,
        Transmissive
      };

    private:
      // Talking related parameters
      // --------------------------

      //! how verbose are we?
      // int _verbose;

      //! interval between steps to write current state to screen
      int _nstepsLog;


      // simulation related parameters
      // -----------------------------

      //! How many steps to do
      int _nsteps;

      //! at what time to end simulation
      float _tmax;

      //! number of cells to use (in each dimension)
      int _nx;

      //! CFL coefficient
      float _ccfl;

      //! time step sized used when enforcing a fixed time step size
      // float _force_dt;

      //! boundary condition
      BoundaryCondition _boundary;

      //! cell size
      float _dx;

      //! Number of Ghost cells at each edge
      int _bc;

      //! number of mesh points, including boundary cells
      int _nxTot;

      // Output related parameters
      // -------------------------

      //! after how many steps to write output
      // int _foutput;

      //! time interval between outputs
      // float _dt_out;

      //! Output file name basename
      std::string _outputfilename;

      //! file name containing output times
      // char _toutfilename[MAX_FNAME_SIZE];

      //! whether we're using the t_out_file
      // int _use_toutfile;

      //! how many outputs we will be writing. Only used if(use_toutfile)
      // int _noutput_tot;

      //! at which output we are. Only used if(use_toutfile)
      // int _noutput;

      //! array of output times given in the output file
      // float *_outputtimes;


      // IC related parameters
      // ---------------------

      //! dimension of IC file
      // int _ndim_ic;

      //! IC data filename
      std::string _icdatafilename;

      //! parameter filename
      // char _paramfilename[MAX_FNAME_SIZE];


      // Sources related parameters
      // --------------------------

      //! constant acceleration in x direction for constant source terms
      // float _src_const_acc_x;

      //! constant acceleration in y direction for constant source terms
      // float _src_const_acc_y;

      //! constant acceleration in radial direction for radial source terms
      // float _src_const_acc_r;

      //! whether the sources will be constant
      // int _constant_acceleration;

      //! whether the constant acceleration has been computed
      // int _constant_acceleration_computed;

      //! whether sources have been read in
      // int _sources_are_read;


    public:
      Parameters();

      void init();

      void cleanup();

      int  getNstepsLog() const;
      void setNstepsLog(const int nsteps_log);

      int  getNsteps() const;
      void setNsteps(const int nsteps);

      float getTmax() const;
      void  setTmax(const float tmax);

      int  getNx() const;
      void setNx(const int nx);

      float getCcfl() const;
      void  setCcfl(float ccfl);

      BoundaryCondition  getBoundary() const;
      void setBoundary(BoundaryCondition boundary);

      int  getNxTot() const;
      void setNxTot(const int nxTot);

      float getDx() const;
      void  setDx(const float dx);

      int  getBc() const;
      void setBc(const int bc);

      int getBcTot() const;

      void setOutputFilename(std::string);
      std::string getOutputFilename() const;

      void setIcDataFilename(std::string);
      std::string getIcDataFilename() const;

    
    public: 

      // single copy of the global variables
      static Parameters Instance;

      // getter for the single global copy
      Parameters& getInstance() {return Instance;}
    };


  } // namespace parameters
} // namespace hydro_playground
