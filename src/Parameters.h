#pragma once

#include "Config.h"
#include "Logging.h"

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

namespace parameters {

    public:
      enum class BoundaryCondition { Periodic, Reflective, Transmissive };

    private:
      // Talking related parameters
      // --------------------------

    //! how verbose are we?
    logging::LogLevel _verbose;

    //! interval between steps to write current state to screen
    int _nstepsLog;


    // simulation related parameters
    // -----------------------------


    //! How many steps to do
    int _nsteps;

    //! at what time to end simulation
    float_t _tmax;

    //! number of cells to use (in each dimension)
    int _nx;

    //! CFL coefficient
    float_t _ccfl;

    //! boundary condition
    // TODO(mivkov): Make enum
    BoundaryCondition _boundary;

    //! number of mesh points, including boundary cells
    int _nxTot;

    //! cell size
    float_t _dx;

    //! Number of Ghost cells at each edge
    int _bc;

    //! number of mesh points, including boundary cells
    int _nxTot;

    // Output related parameters
    // -------------------------

    //! after how many steps to write output
    // int _foutput;

    //! time interval between outputs
    // double _dt_out;

    //! Output file name basename
    // char _outputfilename[MAX_FNAME_SIZE];

    //! file name containing output times
    // char _toutfilename[MAX_FNAME_SIZE];

    //! whether we're using the t_out_file
    // bool _use_toutfile;

    //! how many outputs we will be writing. Only used if(use_toutfile)
    // int _noutput_tot;

    //! at which output we are. Only used if(use_toutfile)
    // int _noutput;

    //! array of output times given in the output file
    // float_t _*outputtimes;


    // IC related parameters
    // ---------------------

    //! dimension of IC file
    // int _ndim_ic;

    //! IC data filename
    // char _datafilename[MAX_FNAME_SIZE];

    //! parameter filename
    // char _paramfilename[MAX_FNAME_SIZE];


    // Sources related parameters
    // --------------------------

    //! constant acceleration in x direction for constant source terms
    // float_t _src_const_acc_x;

    //! constant acceleration in y direction for constant source terms
    // float_t _src_const_acc_y;

    //! constant acceleration in radial direction for radial source terms
    // float_t _src_const_acc_r;

    //! whether the sources will be constant
    // bool _constant_acceleration;

    //! whether the constant acceleration has been computed
    // bool _constant_acceleration_computed;

    //! whether sources have been read in
    // bool _sources_are_read;


  public:
    Parameters();

    // ToDo: Move in destructor
    void cleanup();

    logging::LogLevel getVerbose() const;
    void              setVerbose(const logging::LogLevel logLevel);

    int  getNstepsLog() const;
    void setNstepsLog(const int nstepsLog);

    int  getNsteps() const;
    void setNsteps(const int nsteps);

    float_t getTmax() const;
    void    setTmax(const float_t tmax);

    int  getNx() const;
    void setNx(const int nx);

    float_t getCcfl() const;
    void    setCcfl(float_t ccfl);

    BoundaryCondition getBoundary() const;
    void              setBoundary(BoundaryCondition boundary);

    int  getNxTot() const;
    void setNxTot(const int nxTot);

    float_t getDx() const;
    void    setDx(const float_t dx);

    // single copy of the global variables
    static Parameters Instance;

    // getter for the single global copy
    static Parameters& getInstance() {
      return Instance;
  };
} // namespace parameters
