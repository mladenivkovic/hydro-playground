#pragma once
#include <string>

#include "Config.h"
#include "Logging.h" // for verbosity level

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

  public:
    enum class BoundaryCondition {
      Periodic = 0,
      Reflective = 1,
      Transmissive = 2
    };

  private:
    // Talking related parameters
    // --------------------------

    //! how verbose are we?
    logging::LogLevel _verbose;

    //! interval between steps to write current state to screen
    size_t _nstepsLog;


    // simulation related parameters
    // -----------------------------

    //! How many steps to do
    size_t _nsteps;

    //! at what time to end simulation
    float_t _tmax;

    //! number of cells to use (in each dimension)
    size_t _nx;

    //! CFL coefficient
    float_t _ccfl;

    //! boundary condition
    BoundaryCondition _boundaryType;

    //! number of mesh points, including boundary cells
    size_t _nxTot;

    //! cell size
    float_t _dx;

    //! Number of Ghost cells at each edge
    size_t _nbc;


    // Output related parameters
    // -------------------------

    //! after how many steps to write output
    // size_t _foutput;

    //! time interval between outputs
    // double _dt_out;

    //! Output file name basename
    std::string _outputfilename;

    //! file name containing output times
    // char _toutfilename[MAX_FNAME_SIZE];

    //! whether we're using the t_out_file
    // bool _use_toutfile;

    //! how many outputs we will be writing. Only used if(use_toutfile)
    // size_t _noutput_tot;

    //! at which output we are. Only used if(use_toutfile)
    // size_t _noutput;

    //! array of output times given in the output file
    // float_t _*outputtimes;


    // IC related parameters
    // ---------------------

    //! dimension of IC file
    // size_t _ndim_ic;

    //! IC data filename
    std::string _icdatafilename;

    //! parameter filename
    std::string _paramfilename;


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

    // Lock params after initial setup and throw errors if somebody
    // tries to modify them.
    bool _locked;

  public:
    Parameters();

    /**
     * @brief Sets up parameters.
     */
    void init(
      logging::LogLevel verbose      = logging::LogLevel::Quiet,
      size_t            nstepsLog    = 1,
      size_t            nsteps       = 1,
      float_t           tmax         = 1.,
      size_t            nx           = 100,
      float_t           Ccfl         = 0.9,
      BoundaryCondition boundaryType = BoundaryCondition::Periodic,
      size_t            nbc          = 2
    );

    // ToDo: Move to destructor
    void cleanup();

    /**
     * @brief Get the logging level
     */
    logging::LogLevel getVerbose() const;
    void              setVerbose(const logging::LogLevel logLevel);

    /**
     * @brief Get number of steps between writing log to screen
     */
    size_t getNstepsLog() const;
    void   setNstepsLog(const size_t nstepsLog);

    /**
     * @brief Get max nr of simulation steps to run
     */
    size_t getNsteps() const;
    void   setNsteps(const size_t nsteps);

    /**
     * @brief get simulation end time
     */
    float_t getTmax() const;
    void    setTmax(const float_t tmax);

    /**
     * @brief Get the number of cells with actual content per dimension
     */
    size_t getNx() const;
    void   setNx(const size_t nx);

    /**
     * @brief Get the CFL constant
     */
    float_t getCcfl() const;
    void    setCcfl(float_t ccfl);

    /**
     * @brief Get the type of boundary condition used
     */
    BoundaryCondition getBoundaryType() const;
    void              setBoundaryType(BoundaryCondition boundary);

    /**
     * @brief Get the number of boundary cells on each side of the box
     */
    size_t getNBC() const;
    void   setNBC(size_t nbc);

    /**
     * @brief get the total number of boundary cells per dimension.
     */
    size_t getNBCTot() const;

    /**
     * @brief get the total number of cells per dimension. This includes
     * boundary cells.
     * @TODO: what to do with replication
     */
    size_t getNxTot() const;

    /**
     * @brief Get the cell size
     */
    float_t getDx() const;
    void    setDx(const float_t dx);

    void setOutputFilename(std::string);
    std::string getOutputFilename() const;

    void setIcDataFilename(std::string);
    std::string getIcDataFilename() const;



    //! single copy of the global variables
    static Parameters Instance;

    /**
     * @brief getter for the single global copy
     */
    static Parameters& getInstance() {
      return Instance;
    }
  };
} // namespace parameters
