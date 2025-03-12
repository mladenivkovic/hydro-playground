#include "Parameters.h"

#include <iomanip>
#include <iostream>

#include "Logging.h"


// TODO: These definitions are temporary and need to go.
// #define BC 2
#define BOXLEN 1.
// #define BCTOT 2 * BC

namespace parameters {


  Parameters::Parameters() {

    // Set up default values here.
    // "Proper" cpp prefers these members initialised in an initialiser
    // list, but here, I find it more practical this way. So to avoid the
    // linter screaming at me, not linting this bit.

    // NOLINTBEGIN

    _nstepsLog = 0;
    _verbose   = 0;

    _nsteps       = 0;
    _tmax         = 0.;
    _nx           = 1;
    _ccfl         = 0.9;
    _boundaryType = BC::BoundaryCondition::Periodic;
    _boxsize      = 1.;
    _nbc          = 2;
    _replicate    = 1;

    _outputfilebase = "";

    _locked = false;
    _read   = false;

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


  /**
   * Initialise derived global quantities/parameters.
   * Lock the parameters struct.
   */
  void Parameters::initDerived() {

    int currentLevel = static_cast<int>(logging::getCurrentVerbosity());
    int paramVer     = getVerbose();
    if (currentLevel < paramVer) {
      logging::setVerbosity(paramVer);
      std::stringstream msg;
      msg << "Set run verbosity level to valueread from parameter file=" << paramVer;
      message(msg, logging::LogLevel::Verbose);
    }

#if DEBUG_LEVEL > 0
    message("Locking parameters storage.", logging::LogLevel::Debug);
    _locked = true;
#endif
  }


  /**
   * Get a sring of all parameters for printouts.
   */
  std::string Parameters::toString() const {

    constexpr int width = 20;

    std::stringstream out;
    out << "\nParameter List\n";
    out << std::setw(width) << "nstepsLog:";
    out << std::setw(width) << getNstepsLog() << "\n";
    out << std::setw(width) << "verbose:";
    out << std::setw(width) << getVerbose() << "\n";
    out << std::setw(width) << "nsteps:";
    out << std::setw(width) << getNsteps() << "\n";
    out << std::setw(width) << "tmax:";
    out << std::setw(width) << getTmax() << "\n";
    out << std::setw(width) << "nx:";
    out << std::setw(width) << getNx() << "\n";
    out << std::setw(width) << "Ccfl:";
    out << std::setw(width) << getCcfl() << "\n";
    out << std::setw(width) << "boundaryType:";
    out << std::setw(width) << static_cast<int>(getBoundaryType()) << "\n";
    out << std::setw(width) << "boxsize:";
    out << std::setw(width) << getBoxsize() << "\n";
    out << std::setw(width) << "nbc:";
    out << std::setw(width) << getNBC() << "\n";
    out << std::setw(width) << "replicate:";
    out << std::setw(width) << getReplicate() << "\n";

    return out.str();
  }


} // namespace parameters
