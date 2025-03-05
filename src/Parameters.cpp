#include "Parameters.h"

#include "Logging.h"
#include <iostream>


// TODO: These definitions are temporary and need to go.
// #define BC 2
#define BOXLEN 1.
// #define BCTOT 2 * BC


//! Print out argument and its value with Debug verbosity
#define paramSetLog(arg) \
  {                                                                 \
    std::stringstream msg;                                          \
    msg << "Parameters: Setting '" << #arg << "' = " << arg <<"'";      \
    message(msg, logging::LogLevel::Debug);                         \
  }

namespace parameters {

  // need to define it as well...
  Parameters Parameters::Instance;

  Parameters::Parameters():
    _nstepsLog(0),
    _nsteps(0),
    _tmax(0),
    _nx(1),
    _ccfl(0.),
    _boundaryType(BoundaryCondition::Periodic),
    _nxTot(0),
    _dx(1.0),
    _nbc(0),
    _locked(false)

    // nxtot used to be 100 + BCTOT = 100 + 2*BC. Fixing BC to be 2 and BCTOT to be
  // 2*BC

  { /* empty body */
  }


  // output related parameters
  // -------------------------

  // _foutput = 0;
  // _dt_out = 0;
  // strcpy(_outputfilename, "");

  // strcpy(_toutfilename, "");
  // _use_toutfile = 0;
  // _noutput_tot = 0;
  // _noutput = 0;
  // _outputtimes = NULL;

  // IC related parameters
  // ---------------------
  // _twostate_ic = 0;
  // _ndim_ic = -1;
  // strcpy(_datafilename, "");

  // strcpy(_paramfilename, "");


  // Sources related parameters
  // --------------------------
  // _src_const_acc_x = 0.;
  // _src_const_acc_y = 0.;
  // _src_const_acc_r = 0.;
  // _constant_acceleration = 0;
  // _constant_acceleration_computed = 0;
  // _sources_are_read = 0;


  void Parameters::initDerived() {
    size_t nx = getNx();
    float_t dx = static_cast<float_t>(BOXLEN) / static_cast<float_t>(nx);
    std::cout << "--------------------------" << dx << std::endl;
    std::cout << "--------------------------" << BOXLEN << std::endl;
    std::cout << "--------------------------" << nx << std::endl;
    setDx(dx);

#if DEBUG_LEVEL > 0
    _locked = true;
#endif
  }


  void Parameters::setOutputFileBase(std::string& ofname) {
    _outputfilename = ofname;
    paramSetLog(ofname);
#if DEBUG_LEVEL > 0
    if (_locked) error("Trying to overwrite locked parameters!");
#endif
  }


  std::string Parameters::getOutputFileBase() const {
    return _outputfilename;
  }


  void Parameters::setIcDataFilename(std::string& icfname) {
    _icdatafilename = icfname;
    paramSetLog(icfname);
#if DEBUG_LEVEL > 0
    if (_locked) error("Trying to overwrite locked parameters!");
#endif
  }


  std::string Parameters::getIcDataFilename() const {
    return _icdatafilename;
  }


  size_t Parameters::getNstepsLog() const {
    return _nstepsLog;
  }


  void Parameters::setNstepsLog(const size_t nstepsLog) {
    _nstepsLog = nstepsLog;
    paramSetLog(nstepsLog);
#if DEBUG_LEVEL > 0
    if (_locked) error("Trying to overwrite locked parameters!");
#endif
  }


  size_t Parameters::getNsteps() const {
    return _nsteps;
  }

  void Parameters::setNsteps(const size_t nsteps) {
    _nsteps = nsteps;
    paramSetLog(nsteps);
#if DEBUG_LEVEL > 0
    if (_locked) error("Trying to overwrite locked parameters!");
#endif
  }


  float_t Parameters::getTmax() const {
    return _tmax;
  }


  void Parameters::setTmax(const float tmax) {
    _tmax = tmax;
    paramSetLog(tmax);
#if DEBUG_LEVEL > 0
    if (_locked) error("Trying to overwrite locked parameters!");
#endif
  }


  size_t Parameters::getNx() const {
    return _nx;
  }


  void Parameters::setNx(const size_t nx) {
    _nx = nx;
    paramSetLog(nx);
#if DEBUG_LEVEL > 0
    if (_locked) error("Trying to overwrite locked parameters!");
#endif
  }


  float_t Parameters::getCcfl() const {
    return _ccfl;
  }


  void Parameters::setCcfl(const float ccfl) {
    _ccfl = ccfl;
    paramSetLog(ccfl);
#if DEBUG_LEVEL > 0
    if (_locked) error("Trying to overwrite locked parameters!");
#endif
  }


  size_t Parameters::getNxTot() const {
    return getNx() + 2 * getNBC();
  }


  float_t Parameters::getDx() const {
    return _dx;
  }


  void Parameters::setDx(const float_t dx) {
    _dx = dx;
    paramSetLog(dx);
#if DEBUG_LEVEL > 0
    if (_locked) error("Trying to overwrite locked parameters!");
#endif
  }


  BoundaryCondition Parameters::getBoundaryType() const {
    return _boundaryType;
  }


  void Parameters::setBoundaryType(BoundaryCondition boundaryType) {
    _boundaryType = boundaryType;
    paramSetLog((int) boundaryType);
#if DEBUG_LEVEL > 0
    if (_locked) error("Trying to overwrite locked parameters!");
#endif
  }

  size_t Parameters::getNBC() const {
    return _nbc;
  }

  void Parameters::setNBC(const size_t bc) {
    _nbc = bc;
    paramSetLog(bc);
#if DEBUG_LEVEL > 0
    if (_locked) error("Trying to overwrite locked parameters!");
#endif
  }

  size_t Parameters::getNBCTot() const {
    return 2 * getNBC();
  }


} // namespace parameters
