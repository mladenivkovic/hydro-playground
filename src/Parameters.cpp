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

    _nsteps       = 0;
    _tmax         = 0.;
    _nx           = 1;
    _ccfl         = 0.9;
    _boundaryType = BC::BoundaryCondition::Periodic;
    _nxTot        = 0;
    _nbc          = 2;

    _outputfilebase = "";

    _locked = false;
    _read = false;

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
   */
  void Parameters::initDerived() {

    // size_t  nx = getNx();
    // float_t dx = static_cast<float_t>(BOXLEN) / static_cast<float_t>(nx);
    // setDx(dx);

#if DEBUG_LEVEL > 0
    _locked = true;
#endif
  }


  /**
   * Get a sring of all parameters for printouts.
   */
  std::string Parameters::toString() const {

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
    // out << std::setw(20) << "nxTot:";
    // out << std::setw(20) << getNxTot() << "\n";
    // out << std::setw(20) << "dx:";
    // out << std::setw(20) << getDx() << "\n";
    out << std::setw(20) << "nbc:";
    out << std::setw(20) << getNBC() << "\n";

    return out.str();
  }


} // namespace parameters
