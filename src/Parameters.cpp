#include "Parameters.h"


// TODO: These definitions are temporary and need to go.
#define BOXLEN 1.
#define BCTOT 2

namespace hydro_playground {
  namespace parameters {

    // First initialisation of static params

    // Talking related parameters
    // --------------------------
    int Parameters::_nstepsLog = 0;

    // simulation related parameters
    // -----------------------------
    int   Parameters::_nsteps = 0;
    float Parameters::_tmax   = 0;

    int   Parameters::_nx   = 100;
    float Parameters::_ccfl = 0.9;
    // float Parameters::_forceDt = 0;
    int Parameters::_boundary = 0;

    int   Parameters::_nxTot = 100 + BCTOT;
    float Parameters::_dx    = BOXLEN / _nx;


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

    int Parameters::getNstepsLog() { return _nstepsLog; }

    void Parameters::setNstepsLog(const int nstepsLog) { _nstepsLog = nstepsLog; }

    int Parameters::getNsteps() { return _nsteps; }

    void Parameters::setNsteps(const int nsteps) { _nsteps = nsteps; }

    float Parameters::getTmax() { return _tmax; }

    void Parameters::setTmax(const float tmax) { _tmax = tmax; }

    int Parameters::getNx() { return _nx; }

    void Parameters::setNx(const int nx) { _nx = nx; }

    float Parameters::getCcfl() { return _ccfl; }

    void Parameters::setCcfl(const float ccfl) { _ccfl = ccfl; }

    int Parameters::getBoundary() { return _boundary; }

    void Parameters::setBoundary(const int boundary) { _boundary = boundary; }

    int Parameters::getNxTot() { return _nxTot; }

    void Parameters::setNxTot(const int nxTot) { _nxTot = nxTot; }

    float Parameters::getDx() { return _dx; }

    void Parameters::setDx(const float dx) { _dx = dx; }


  } // namespace parameters
} // namespace hydro_playground
