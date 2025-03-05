#include "Parameters.h"

#include <iomanip>
#include <iostream>

#include "Logging.h"


// TODO: These definitions are temporary and need to go.
// #define BC 2
#define BOXLEN 1.
// #define BCTOT 2 * BC


//! Print out argument and its value with Debug verbosity
#define paramSetLog(arg) \
  { \
    std::stringstream msg; \
    msg << "Parameters: Setting '" << #arg << "' = " << arg << "'"; \
    message(msg, logging::LogLevel::Debug); \
  }

namespace parameters {

  // need to define it as well...
  Parameters Parameters::Instance;

  Parameters::Parameters() {

    // Set up default values here.
    // "Proper" cpp prefers these members initialised in an initialiser
    // list, but here, I find it more practical this way. So to avoid the
    // linter screaming at me, not linting this bit.

    // NOLINTBEGIN

    _nstepsLog = 0;

    _nsteps       = 0;
    _tmax         = 0.;
    _nx           = 1;
    _ccfl         = 0.9;
    _boundaryType = BoundaryCondition::Periodic;
    _nxTot        = 0;
    _dx           = 1.0;
    _nbc          = 2;

    _outputfilebase = "";

    _locked = false;

    // NOLINTEND

    // nxtot used to be 100 + BCTOT = 100 + 2*BC. Fixing BC to be 2 and BCTOT to be
    // 2*BC


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


    // Sources related parameters
    // --------------------------
    // _src_const_acc_x = 0.;
    // _src_const_acc_y = 0.;
    // _src_const_acc_r = 0.;
    // _constant_acceleration = 0;
    // _constant_acceleration_computed = 0;
    // _sources_are_read = 0;
  }


  void Parameters::initDerived() {
    size_t  nx = getNx();
    float_t dx = static_cast<float_t>(BOXLEN) / static_cast<float_t>(nx);
    setDx(dx);

#if DEBUG_LEVEL > 0
    _locked = true;
#endif
  }


  /**
   * Get a sring of all parameters for printouts.
   */
  std::string Parameters::toString() {

    std::stringstream out;
    out << "\nParameter List\n";
    out << std::setw(20) << "nstepsLog:" << std::setw(20) << getNstepsLog() << "\n";
    out << std::setw(20) << "nsteps:";
    out << std::setw(20) << getNsteps() << "\n";
    out << std::setw(20) << "tmax:";
    out << std::setw(20) << getTmax() << "\n";
    out << std::setw(20) << "nx:";
    out << std::setw(20) << getNx() << "\n";
    out << std::setw(20) << "Ccfl:";
    out << std::setw(20) << getCcfl() << "\n";
    out << std::setw(20) << "boundaryType:";
    out << std::setw(20) << static_cast<int>(getBoundaryType()) << "\n";
    out << std::setw(20) << "nxTot:";
    out << std::setw(20) << getNxTot() << "\n";
    out << std::setw(20) << "dx:";
    out << std::setw(20) << getDx() << "\n";
    out << std::setw(20) << "nbc:";
    out << std::setw(20) << getNBC() << "\n";

    return out.str();
  }


  void Parameters::setOutputFileBase(std::string& ofname) {
    Instance._outputfilebase = ofname;
    paramSetLog(ofname);
#if DEBUG_LEVEL > 0
    if (_locked)
      error("Trying to overwrite locked parameters!");
#endif
  }


  std::string Parameters::getOutputFileBase() const {
    return Instance._outputfilebase;
  }


  size_t Parameters::getNstepsLog() const {
    return Instance._nstepsLog;
  }


  void Parameters::setNstepsLog(const size_t nstepsLog) {
    Instance._nstepsLog = nstepsLog;
    paramSetLog(nstepsLog);
#if DEBUG_LEVEL > 0
    if (_locked)
      error("Trying to overwrite locked parameters!");
#endif
  }


  size_t Parameters::getNsteps() const {
    return Instance._nsteps;
  }


  void Parameters::setNsteps(const size_t nsteps) {
    Instance._nsteps = nsteps;
    paramSetLog(nsteps);
#if DEBUG_LEVEL > 0
    if (_locked)
      error("Trying to overwrite locked parameters!");
#endif
  }


  float_t Parameters::getTmax() const {
    return Instance._tmax;
  }


  void Parameters::setTmax(const float tmax) {
    Instance._tmax = tmax;
    paramSetLog(tmax);
#if DEBUG_LEVEL > 0
    if (_locked)
      error("Trying to overwrite locked parameters!");
#endif
  }


  size_t Parameters::getNx() const {
    return Instance._nx;
  }


  void Parameters::setNx(const size_t nx) {
    Instance._nx = nx;
    paramSetLog(nx);
#if DEBUG_LEVEL > 0
    if (_locked)
      error("Trying to overwrite locked parameters!");
#endif
  }


  float_t Parameters::getCcfl() const {
    return Instance._ccfl;
  }


  void Parameters::setCcfl(const float ccfl) {
    Instance._ccfl = ccfl;
    paramSetLog(ccfl);
#if DEBUG_LEVEL > 0
    if (_locked)
      error("Trying to overwrite locked parameters!");
#endif
  }


  size_t Parameters::getNxTot() const {
    return getNx() + 2 * getNBC();
  }


  float_t Parameters::getDx() const {
    return Instance._dx;
  }


  void Parameters::setDx(const float_t dx) {
    Instance._dx = dx;
    paramSetLog(dx);
#if DEBUG_LEVEL > 0
    if (_locked)
      error("Trying to overwrite locked parameters!");
#endif
  }


  BoundaryCondition Parameters::getBoundaryType() const {
    return Instance._boundaryType;
  }


  void Parameters::setBoundaryType(BoundaryCondition boundaryType) {
    Instance._boundaryType = boundaryType;
    paramSetLog((int)boundaryType);
#if DEBUG_LEVEL > 0
    if (_locked)
      error("Trying to overwrite locked parameters!");
#endif
  }

  size_t Parameters::getNBC() const {
    return Instance._nbc;
  }

  void Parameters::setNBC(const size_t bc) {
    Instance._nbc = bc;
    paramSetLog(bc);
#if DEBUG_LEVEL > 0
    if (_locked)
      error("Trying to overwrite locked parameters!");
#endif
  }

  size_t Parameters::getNBCTot() const {
    return 2 * getNBC();
  }


} // namespace parameters
