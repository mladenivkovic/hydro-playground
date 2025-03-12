#pragma once
#include <string>

#include "BoundaryConditions.h"
#include "Config.h"
#include "Logging.h" // for verbosity level

namespace parameters {

  /**
   * @brief Holds global simulation parameters.
   *
   * Note: Adding new parameters:
   *   - Add private variable in class
   *   - Add getter and setters in class
   *   - Add default value in constructor
   *   - Does it need some computation? If so, add to initDerived
   *   - Add read in IO::InputParse::parseConfigFile()
   *   - Add entry in toString()
   */
  class Parameters {

  private:
    // Talking related parameters
    // --------------------------

    //! interval between steps to write current state to screen
    size_t _nstepsLog;

    //! Verbosity of the run.
    int _verbose;


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
    BC::BoundaryCondition _boundaryType;

    //! box size
    float_t _boxsize;

    //! Number of Ghost cells at each edge
    size_t _nbc;

    //! Do we replicate the box?
    size_t _replicate;


    // Output related parameters
    // -------------------------

    //! after how many steps to write output
    // size_t _foutput;

    //! time interval between outputs
    // double _dt_out;

    //! Output file name basename
    std::string _outputfilebase;

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

    //! Has the parameter file been read?
    bool _read;

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


    //! Get a sring of all parameters for printouts.
    [[nodiscard]] std::string toString() const;


    // ToDo: Move to destructor
    void cleanup();


    /**
     * @brief Get number of steps between writing log to screen
     */
    [[nodiscard]] size_t getNstepsLog() const;
    void                 setNstepsLog(const size_t nstepsLog);


    /**
     * @brief Get number of steps between writing log to screen
     */
    [[nodiscard]] int getVerbose() const;
    void              setVerbose(const int verbose);


    /**
     * @brief Get max nr of simulation steps to run
     */
    [[nodiscard]] size_t getNsteps() const;
    void                 setNsteps(const size_t nsteps);


    /**
     * @brief get simulation end time
     */
    [[nodiscard]] float_t getTmax() const;
    void                  setTmax(const float_t tmax);


    /**
     * @brief Get the number of cells with actual content per dimension
     */
    [[nodiscard]] size_t getNx() const;
    void                 setNx(const size_t nx);


    /**
     * @brief Get the number of cells with actual content per dimension
     */
    [[nodiscard]] float_t getBoxsize() const;
    void                  setBoxsize(const float_t boxsize);


    /**
     * @brief Get the CFL constant
     */
    [[nodiscard]] float_t getCcfl() const;
    void                  setCcfl(float_t ccfl);


    /**
     * @brief Get the type of boundary condition used
     */
    [[nodiscard]] BC::BoundaryCondition getBoundaryType() const;
    void                                setBoundaryType(BC::BoundaryCondition boundary);


    /**
     * @brief Get the number of boundary cells on each side of the box
     */
    [[nodiscard]] size_t getNBC() const;
    void                 setNBC(size_t nbc);


    /**
     * @brief Get the number of boundary cells on each side of the box
     */
    [[nodiscard]] size_t getReplicate() const;
    void                 setReplicate(size_t rep);


    /**
     * @brief Get the output file name base
     */
    [[nodiscard]] std::string getOutputFileBase() const;
    void                      setOutputFileBase(std::string& ofname);


    /**
     * @brief Get the output file name base
     */
    [[nodiscard]] bool getParamFileHasBeenRead() const;
    void               setParamFileHasBeenRead();
  };
} // namespace parameters


// --------------------------------------------------------
// Definitions
// --------------------------------------------------------


//! Print out argument and its value with Debug verbosity
#define paramSetLog(arg) \
  { \
    std::stringstream msg; \
    msg << "Parameters: Setting '" << #arg << "' = " << (arg) << "'"; \
    message(msg, logging::LogLevel::Debug); \
  }


inline size_t parameters::Parameters::getNstepsLog() const {
  return _nstepsLog;
}


inline void parameters::Parameters::setNstepsLog(const size_t nstepsLog) {

  _nstepsLog = nstepsLog;
  paramSetLog(nstepsLog);

#if DEBUG_LEVEL > 0
  if (_locked)
    error("Trying to overwrite locked parameters!");
#endif
}


inline int parameters::Parameters::getVerbose() const {
  return _verbose;
}


inline void parameters::Parameters::setVerbose(const int verbose) {

  _verbose = verbose;
  paramSetLog(verbose);

#if DEBUG_LEVEL > 0
  if (_locked)
    error("Trying to overwrite locked parameters!");
#endif
}


inline size_t parameters::Parameters::getNsteps() const {
  return _nsteps;
}


inline void parameters::Parameters::setNsteps(const size_t nsteps) {

  _nsteps = nsteps;
  paramSetLog(nsteps);

#if DEBUG_LEVEL > 0
  if (_locked)
    error("Trying to overwrite locked parameters!");
#endif
}


inline float_t parameters::Parameters::getTmax() const {
  return _tmax;
}


inline void parameters::Parameters::setTmax(const float_t tmax) {

  _tmax = tmax;
  paramSetLog(tmax);

#if DEBUG_LEVEL > 0
  if (_locked)
    error("Trying to overwrite locked parameters!");
#endif
}


inline size_t parameters::Parameters::getNx() const {
  return _nx;
}


inline void parameters::Parameters::setNx(const size_t nx) {

  _nx = nx;
  paramSetLog(nx);

#if DEBUG_LEVEL > 0
  if (_locked)
    error("Trying to overwrite locked parameters!");
#endif
}


inline float_t parameters::Parameters::getCcfl() const {
  return _ccfl;
}


inline void parameters::Parameters::setCcfl(const float_t ccfl) {

  _ccfl = ccfl;
  paramSetLog(ccfl);

#if DEBUG_LEVEL > 0
  if (_locked)
    error("Trying to overwrite locked parameters!");
#endif
}


inline BC::BoundaryCondition parameters::Parameters::getBoundaryType() const {
  return _boundaryType;
}


inline void parameters::Parameters::setBoundaryType(BC::BoundaryCondition boundaryType) {

  _boundaryType = boundaryType;
  paramSetLog((int)boundaryType);

#if DEBUG_LEVEL > 0
  if (_locked)
    error("Trying to overwrite locked parameters!");
#endif
}


inline size_t parameters::Parameters::getNBC() const {
  return _nbc;
}


inline void parameters::Parameters::setNBC(const size_t nbc) {

  _nbc = nbc;
  paramSetLog(nbc);

#if DEBUG_LEVEL > 0
  if (_locked)
    error("Trying to overwrite locked parameters!");
#endif
}


inline size_t parameters::Parameters::getReplicate() const {
  return _replicate;
}


inline void parameters::Parameters::setReplicate(const size_t replicate) {

  _replicate = replicate;
  paramSetLog(replicate);

#if DEBUG_LEVEL > 0
  if (_locked)
    error("Trying to overwrite locked parameters!");
#endif
}

inline std::string parameters::Parameters::getOutputFileBase() const {
  return _outputfilebase;
}


inline void parameters::Parameters::setOutputFileBase(std::string& ofname) {

  _outputfilebase = ofname;
  paramSetLog(ofname);

#if DEBUG_LEVEL > 0
  if (_locked)
    error("Trying to overwrite locked parameters!");
#endif
}


inline bool parameters::Parameters::getParamFileHasBeenRead() const {
  return _read;
}


inline void parameters::Parameters::setParamFileHasBeenRead() {
  _read = true;
}


inline float_t parameters::Parameters::getBoxsize() const {
  return _boxsize;
}


inline void parameters::Parameters::setBoxsize(const float_t boxsize) {
  _boxsize = boxsize;
}
