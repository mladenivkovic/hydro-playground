#include "Parameters.h"


// TODO: These definitions are temporary and need to go.
// #define BC 2
// #define BOXLEN 1.
// #define BCTOT 2 * BC

namespace hydro_playground {
  namespace parameters {

    // need to define it as well...
    Parameters Parameters::Instance;

    Parameters::Parameters() :
    _nstepsLog(0), _nsteps(0),
    _tmax(0), _nx(100),
    _ccfl(0.9), _boundary(Parameters::BoundaryCondition::Periodic),
    _dx(1.0 / _nx), _bc(2), 
    _nxTot(100 + 2*_bc)

    // nxtot used to be 100 + BCTOT = 100 + 2*BC. Fixing BC to be 2 and BCTOT to be 
    // 2*BC

    {/* empty body */}


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

    /**
     * Initialize parameters to default values
     */
    void Parameters::init() {
      // TODO
    }

    int Parameters::getNstepsLog() const { return _nstepsLog; }

    void Parameters::setNstepsLog(const int nstepsLog) { _nstepsLog = nstepsLog; }

    int Parameters::getNsteps() const { return _nsteps; }

    void Parameters::setNsteps(const int nsteps) { _nsteps = nsteps; }

    float Parameters::getTmax() const { return _tmax; }

    void Parameters::setTmax(const float tmax) { _tmax = tmax; }

    int Parameters::getNx() const { return _nx; }

    void Parameters::setNx(const int nx) { _nx = nx; }

    float Parameters::getCcfl() const { return _ccfl; }

    void Parameters::setCcfl(const float ccfl) { _ccfl = ccfl; }

    Parameters::BoundaryCondition Parameters::getBoundary() const { return _boundary; }

    void Parameters::setBoundary(Parameters::BoundaryCondition boundary) { _boundary = boundary; }

    int Parameters::getNxTot() const { return _nxTot; }

    void Parameters::setNxTot(const int nxTot) { _nxTot = nxTot; }

    float Parameters::getDx() const { return _dx; }

    void Parameters::setDx(const float dx) { _dx = dx; }

    int Parameters::getBc() const { return _bc; }

    void Parameters::setBc(const int bc) {_bc = bc;}

    int Parameters::getBcTot() const {return 2*getBc();}


  } // namespace parameters
} // namespace hydro_playground
