#include "Parameters.h"

#include <iomanip>
#include <iostream>

#include "Logging.h"


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
    _nx           = 0;
    _ccfl         = 0.9;
    _boundaryType = BC::BoundaryCondition::Periodic;
    _boxsize      = 1.;
    _nbc          = 2;
    _replicate    = 0;

    _write_replications = false;
    _outputfilebase     = "";
    _foutput            = 0;
    _dt_out             = 0;

    _locked = false;
    _read   = false;

    // NOLINTEND


    // strcpy(_toutfilename, "");
    // _use_toutfile = 0;
    // _noutput_tot = 0;
    // _noutput = 0;
    // _outputtimes = NULL;

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

    // Update run verbosity level, if necessary
    int currentLevel = static_cast<int>(logging::getCurrentVerbosity());
    int paramVer     = getVerbose();
    if (currentLevel < paramVer) {
      logging::setVerbosity(paramVer);
      std::stringstream msg;
      msg << "Set run verbosity level to value read from parameter file=" << paramVer;
      message(msg.str(), logging::LogLevel::Verbose);
    }

    // Do we need to resize the box?
    if (getReplicate() > 1) {

      setBoxsize(getBoxsize() * static_cast<Float>(getReplicate()));
      std::stringstream msg;
      msg << "Resizing box to" << getBoxsize() << " to accommodate replications";
      message(msg.str(), logging::LogLevel::Verbose);
    }

    // Set to "no value provided"...
    if (getOutputFileBase() == "None") {
      std::string empty("");
      setOutputFileBase(empty);
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

    out << std::setw(width) << "writeReplications:";
    out << std::setw(width) << getWriteReplications() << "\n";
    out << std::setw(width) << "output file basename:";
    out << std::setw(width) << getOutputFileBase() << "\n";
    out << std::setw(width) << "output frequency:";
    out << std::setw(width) << getFoutput() << "\n";
    out << std::setw(width) << "output time intervals:";
    out << std::setw(width) << getDtOut() << "\n";

    return out.str();
  }


} // namespace parameters
