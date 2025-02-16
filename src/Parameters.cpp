#include "Parameters.h"

#include "Logging.h"


// TODO: These definitions are temporary and need to go.
// #define BC 2
#define BOXLEN 1.
// #define BCTOT 2 * BC

namespace parameters {

    // need to define it as well...
    Parameters Parameters::Instance;

    Parameters::Parameters():
      _verbose(logging::LogLevel::Quiet),
      _nstepsLog(0),
      _nsteps(0),
      _tmax(0),
      _nx(1),
      _ccfl(0.),
      _boundaryType(Parameters::BoundaryCondition::Periodic),
      _nxTot(0),
      _dx(1.0),
      _nbc(0),
      _locked(false)

    // nxtot used to be 100 + BCTOT = 100 + 2*BC. Fixing BC to be 2 and BCTOT to be
    // 2*BC

    { /* empty body */ }


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


  void Parameters::init(
      logging::LogLevel verbose,
      size_t nstepsLog,
      size_t nsteps,
      float_t tmax,
      size_t nx,
      float_t Ccfl,
      BoundaryCondition boundaryType,
      size_t nbc
      ) {

    setVerbose(verbose);
    setNstepsLog(nstepsLog);
    setTmax(tmax);
    setNx(nx);
    setDx(BOXLEN / static_cast<float_t>(nx));
    setCcfl(Ccfl);
    setBoundaryType(boundaryType);
    setNBC(nbc);

#if DEBUG_LEVEL > 0
    _locked = true;
#endif

  }

  logging::LogLevel Parameters::getVerbose() const {
    return _verbose;
  }

  void Parameters::setVerbose(const logging::LogLevel logLevel) {
    _verbose = logLevel;

#if DEBUG_LEVEL > 0
    if(_locked) {
      error("Trying to overwrite parameter values after init!");
    }
#endif
  }


  size_t Parameters::getNstepsLog() const {
    return _nstepsLog;
  }

  void Parameters::setNstepsLog(const size_t nstepsLog) {
    _nstepsLog = nstepsLog;

#if DEBUG_LEVEL > 0
    if(_locked) {
      error("Trying to overwrite parameter values after init!");
    }
#endif
  }

  size_t Parameters::getNsteps() const {
    return _nsteps;
  }

  void Parameters::setNsteps(const size_t nsteps) {
    _nsteps = nsteps;

#if DEBUG_LEVEL > 0
    if(_locked) {
      error("Trying to overwrite parameter values after init!");
    }
#endif
  }

  float_t Parameters::getTmax() const {
    return _tmax;
  }

  void Parameters::setTmax(const float tmax) {
    _tmax = tmax;

#if DEBUG_LEVEL > 0
    if(_locked) {
      error("Trying to overwrite parameter values after init!");
    }
#endif
  }

  size_t Parameters::getNx() const {
    return _nx;
  }

  void Parameters::setNx(const size_t nx) {
    _nx = nx;

#if DEBUG_LEVEL > 0
    if(_locked) {
      error("Trying to overwrite parameter values after init!");
    }
#endif
  }

  float_t Parameters::getCcfl() const {
    return _ccfl;
  }

  void Parameters::setCcfl(const float ccfl) {
    _ccfl = ccfl;

#if DEBUG_LEVEL > 0
    if(_locked) {
      error("Trying to overwrite parameter values after init!");
    }
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


#if DEBUG_LEVEL > 0
    if(_locked) {
      error("Trying to overwrite parameter values after init!");
    }
#endif
  }

  Parameters::BoundaryCondition Parameters::getBoundaryType() const {
    return _boundaryType; }

  void Parameters::setBoundaryType(Parameters::BoundaryCondition boundaryType) {
    _boundaryType = boundaryType;

#if DEBUG_LEVEL > 0
    if(_locked) {
      error("Trying to overwrite parameter values after init!");
    }
#endif
  }

  size_t Parameters::getNBC() const {
    return _nbc;
  }

  void Parameters::setNBC(const size_t bc) {
    _nbc = bc;

#if DEBUG_LEVEL > 0
    if(_locked) {
      error("Trying to overwrite parameter values after init!");
    }
#endif
  }

  size_t Parameters::getNBCTot() const {
    return 2 * getNBC();
  }


} // namespace parameters
