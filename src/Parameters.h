#pragma once
#include <string>

#include "Config.h"
#include "Logging.h" // for verbosity level

namespace parameters {

  //! Boundary condition types
  enum class BoundaryCondition {
    Periodic     = 0,
    Reflective   = 1,
    Transmissive = 2
  };

  //! parameter file argument types
  enum class ArgType {
    Integer     = 0,
    Size_t      = 1,
    Float       = 2,
    Bool        = 3,
    String      = 4
  };

  /**
   * @brief Holds global simulation parameters. There should be only one
   * instance of this globally, so this class is set up as a singleton. To use
   * it and its contents, get hold of an instance:
   *
   * ```
   *    parameters::Parameters::Instance
   * ```
   *
   * It is accessible for all files which include `Parameters.h`.
   *
   * Note: Adding new parameters:
   *   - Add private variable in class
   *   - Add getter and setters in class
   *   - Add default value in constructor
   *   - Add read in IO::InputParse::parseConfigFile()
   */
  class Parameters {

  private:

    // Talking related parameters
    // --------------------------

    //! interval between steps to write current state to screen
    size_t _nstepsLog;


    // simulation related parameters
    // -----------------------------

    //! How many steps to do
    size_t _nsteps;

    //! at what time to end simulation
    float_t _tmax;

    //! number of cells to use (in each dimension)
    // TODO(mivkov): Do we still need this if we're not doing twostate ICs?
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


    // Others
    // --------------------------

    // Lock params after initial setup and throw errors if somebody
    // tries to modify them.
    bool _locked;

  public:
    Parameters();

    /**
     * @brief Sets up derived parameters: Things that need computation etc after
     * default values have been set (in the constructor) and parameter file has
     * been read.
     * In debug mode, this also "locks" the parameters and watches for future
     * modifications.
     */
    void initDerived();


    // ToDo: Move to destructor
    void cleanup();


    /**
     * @brief Get number of steps between writing log to screen
     */
    [[nodiscard]] size_t getNstepsLog() const;
    void   setNstepsLog(const size_t nstepsLog);


    /**
     * @brief Get max nr of simulation steps to run
     */
    [[nodiscard]] size_t getNsteps() const;
    void   setNsteps(const size_t nsteps);


    /**
     * @brief get simulation end time
     */
    [[nodiscard]] float_t getTmax() const;
    void    setTmax(const float_t tmax);


    /**
     * @brief Get the number of cells with actual content per dimension
     */
    [[nodiscard]] size_t getNx() const;
    void   setNx(const size_t nx);


    /**
     * @brief Get the CFL constant
     */
    [[nodiscard]] float_t getCcfl() const;
    void    setCcfl(float_t ccfl);


    /**
     * @brief Get the type of boundary condition used
     */
    [[nodiscard]] BoundaryCondition getBoundaryType() const;
    void              setBoundaryType(BoundaryCondition boundary);


    /**
     * @brief Get the number of boundary cells on each side of the box
     */
    [[nodiscard]] size_t getNBC() const;
    void   setNBC(size_t nbc);


    /**
     * @brief get the total number of boundary cells per dimension.
     */
    [[nodiscard]] size_t getNBCTot() const;


    /**
     * @brief get the total number of cells per dimension. This includes
     * boundary cells.
     * @TODO: what to do with replication
     */
    [[nodiscard]] size_t getNxTot() const;


    /**
     * @brief Get the cell size
     */
    [[nodiscard]] float_t getDx() const;
    void    setDx(const float_t dx);


    /**
     * @brief Get the output file name base
     */
    [[nodiscard]] std::string getOutputFileBase() const;
    void        setOutputFileBase(std::string& ofname);


    /**
     * Get the IC file name.
     */
    [[nodiscard]] std::string getIcDataFilename() const;
    void        setIcDataFilename(std::string& icfname);


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
