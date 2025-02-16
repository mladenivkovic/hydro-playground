#include "Parameters.h"
#include "Logging.h"


// TODO: These definitions are temporary and need to go.
#define BOXLEN 1.
#define BCTOT 2

namespace parameters {

  // need to define it as well...
  Parameters Parameters::Instance;

  Parameters::Parameters() :
    _verbose(logging::LogLevel::Quiet),
    _nstepsLog(0),
    _nsteps(0),
    _tmax(0.),
    _nx(0),
    _ccfl(0.),
    _boundary(0),
    _nxTot(0),
    _dx(0.) { /* empty body */ }


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


  logging::LogLevel Parameters::getVerbose() const {
    return _verbose;
  }

  void Parameters::setVerbose(const logging::LogLevel logLevel){
    _verbose = logLevel;
  }


  int Parameters::getNstepsLog() const {
    return _nstepsLog;
  }

  void Parameters::setNstepsLog(const int nstepsLog) {
    _nstepsLog = nstepsLog;
  }

  int Parameters::getNsteps() const {
    return _nsteps;
  }

  void Parameters::setNsteps(const int nsteps) {
    _nsteps = nsteps;
  }

  float_t Parameters::getTmax() const {
    return _tmax;
  }

  void Parameters::setTmax(const float tmax) {
    _tmax = tmax;
  }

  int Parameters::getNx() const {
    return _nx;
  }

  void Parameters::setNx(const int nx) {
    _nx = nx;
  }

  float_t Parameters::getCcfl() const {
    return _ccfl;
  }

  void Parameters::setCcfl(const float ccfl) {
    _ccfl = ccfl;
  }

  int Parameters::getBoundary() const {
    return _boundary;
  }

  void Parameters::setBoundary(const int boundary) {
    _boundary = boundary;
  }

  int Parameters::getNxTot() const {
    return _nxTot;
  }

  void Parameters::setNxTot(const int nxTot) { _nxTot = nxTot; }

  float_t Parameters::getDx() const {
    return _dx;
  }

  void Parameters::setDx(const float_t dx) {
    _dx = dx;
  }


} // namespace parameters
