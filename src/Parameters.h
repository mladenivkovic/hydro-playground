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

  class Parameters {

  private:
    // Talking related parameters
    // --------------------------

    //! how verbose are we?
    logging::LogLevel verbose;

    //! interval between steps to write current state to screen
    int nstepsLog;


    // simulation related parameters
    // -----------------------------

    //! How many steps to do
    int nsteps;

    //! at what time to end simulation
    double tmax;

    //! number of cells to use (in each dimension)
    int nx;

    //! CFL coefficient
    float_t ccfl;

    //! boundary condition
    int _boundary;

    //! number of mesh points, including boundary cells
    int _nxTot;

    //! cell size
    float _dx;


    // Output related parameters
    // -------------------------

    //! after how many steps to write output
    // int _foutput;

    //! time interval between outputs
    // float _dt_out;

    //! Output file name basename
    // char _outputfilename[MAX_FNAME_SIZE];

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
    // char _datafilename[MAX_FNAME_SIZE];

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

    int  getNstepsLog();
    void setNstepsLog(const int nsteps_log);

    int  getNsteps();
    void setNsteps(const int nsteps);

    float getTmax();
    void  setTmax(const float tmax);

    int  getNx();
    void setNx(const int nx);

    float getCcfl();
    void  setCcfl(float ccfl);

    int  getBoundary();
    void setBoundary(const int boundary);

    int  getNxTot();
    void setNxTot(const int nxTot);

    float getDx();
    void  setDx(const float dx);

  public:
    // single copy of the global variables
    static Parameters Instance;

    // getter for the single global copy
    static Parameters& getInstance() { return Instance; }
  };


} // namespace parameters
