#pragma once


namespace parameters {

  class Parameters {

  private:
    // Talking related parameters
    // --------------------------

    //! how verbose are we?
    // int _verbose;

    //! interval between steps to write current state to screen
    static int _nstepsLog;


    // simulation related parameters
    // -----------------------------

    //! How many steps to do
    static int _nsteps;

    //! at what time to end simulation
    static float _tmax;

    //! number of cells to use (in each dimension)
    static int _nx;

    //! CFL coefficient
    static float _ccfl;

    //! time step sized used when enforcing a fixed time step size
    // float _force_dt;

    //! boundary condition
    static int _boundary;

    //! number of mesh points, including boundary cells
    static int _nxTot;

    //! cell size
    static float _dx;


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
    Parameters() = delete;

    static void init();

    static void cleanup();

    static int  getNstepsLog();
    static void setNstepsLog(const int nsteps_log);

    static int  getNsteps();
    static void setNsteps(const int nsteps);

    static float getTmax();
    static void  setTmax(const float tmax);

    static int  getNx();
    static void setNx(const int nx);

    static float getCcfl();
    static void  setCcfl(float ccfl);

    static int  getBoundary();
    static void setBoundary(const int boundary);

    static int  getNxTot();
    static void setNxTot(const int nxTot);

    static float getDx();
    static void  setDx(const float dx);
  };


} // namespace parameters
