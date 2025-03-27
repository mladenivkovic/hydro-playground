#include "IO.h"

#include <cassert>
#include <cctype>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>

#include "BoundaryConditions.h"
#include "Gas.h"
#include "Logging.h"
#include "Timer.h"
#include "Utils.h"


// -----------------------------------------
// paramEntry stuff
// -----------------------------------------

/**
 * paramEntry constructors
 */
ParamEntry::ParamEntry(std::string parameter):
  param(std::move(parameter)),
  used(false) {};

ParamEntry::ParamEntry(std::string parameter, std::string value):
  param(std::move(parameter)),
  value(std::move(value)),
  used(false) {};


// -----------------------------------------
// InputParse stuff
// -----------------------------------------


/**
 * @brief Constructor. Takes over cmdline args from main(), checks them,
 * and stores them internally.
 */
InputParse::InputParse(const int argc, char* argv[]) {

  constexpr int argc_max = 20;

#if DEBUG_LEVEL > 0
  if (argc > argc_max) {
    std::stringstream msg;
    msg << "Passed " << argc << " arguments, which is higher than the max: " << argc_max
        << ", ignoring everything past it.";
    warning(msg.str());
  }
#endif

  for (int i = 1; i < std::min(argc, argc_max); i++) {

    std::string arg = std::string(argv[i]);

    // Allow flags without values
    if (arg == "-h" or arg == "--help" or arg == "-v" or arg == "--verbose" or arg == "-vv"
        or arg == "--very-verbose") {
      _cmdlargs.insert(std::make_pair(arg, ""));
      continue;
    }

    // Do we have an arg=value situation?
    auto        split_pair = utils::splitEquals(arg);
    std::string name       = split_pair.first;
    std::string value      = split_pair.second;

    if (name != utils::somethingWrong()) {

      // We have an arg=value situation.
      _cmdlargs.insert(std::make_pair(name, value));

    } else {

      // value is next arg, if it exists. Otherwise, empty string.
      std::string val;
      if (i + 1 < argc)
        val = argv[i + 1];
      _cmdlargs.insert(std::make_pair(arg, val));
      i++;
    }
  }

  _checkCmdLineArgsAreValid();
}


/**
 * Read the parameter file and fill out the parameters object.
 */
void InputParse::readParamFile(Parameters& params) {

  message("Parsing param file.", logging::LogLevel::Verbose);

#if DEBUG_LEVEL > 0
  if (_paramfile.size() == 0) {
    error("No param file specified?");
  }
#endif

  // First, we read the param file
  std::string   line;
  std::ifstream conf_ifs(_paramfile);

  // Read in line by line
  while (std::getline(conf_ifs, line)) {
    auto        pair  = extractParameter(line);
    std::string name  = pair.first;
    std::string value = pair.second;
    if (name == "")
      continue;
    if (name == utils::somethingWrong() or value == utils::somethingWrong()) {
      warning("Something wrong with param file line '" + line + "'; skipping it");
    }

    ParamEntry newEntry(name, value);
    _config_params.insert(std::make_pair(name, newEntry));
  }


  // Now we parse each argument

  size_t nsteps_log = _convertParameterString(
    "nstep_log",
    ArgType::Size_t,
    /*optional=*/true,
    /*default_val=*/params.getNstepsLog()
  );
  params.setNstepsLog(nsteps_log);

  int verbose = _convertParameterString(
    "verbose",
    ArgType::Integer,
    /*optional=*/true,
    /*default_val=*/params.getVerbose()
  );
  params.setVerbose(verbose);

  size_t nsteps = _convertParameterString(
    "nsteps",
    ArgType::Size_t,
    /*optional=*/true,
    /*default_val=*/params.getNsteps()
  );
  params.setNsteps(nsteps);

  size_t nx = _convertParameterString(
    "nx",
    ArgType::Size_t,
    /*optional=*/true,
    /*default_val=*/params.getNsteps()
  );
  params.setNx(nx);

  size_t rep = _convertParameterString(
    "replicate",
    ArgType::Size_t,
    /*optional=*/true,
    /*default_val=*/params.getReplicate()
  );
  params.setReplicate(rep);

  int boundary = _convertParameterString(
    "boundary",
    ArgType::Integer,
    /*optional=*/true,
    /*default_val=*/static_cast<int>(params.getBoundaryType())
  );
  params.setBoundaryType(static_cast<BC::BoundaryCondition>(boundary));

  Float tmax = _convertParameterString(
    "tmax",
    ArgType::Float,
    /*optional=*/true,
    /*default_val=*/params.getTmax()
  );
  params.setTmax(tmax);

  Float boxsize = _convertParameterString(
    "boxsize",
    ArgType::Float,
    /*optional=*/true,
    /*default_val=*/params.getBoxsize()
  );
  params.setBoxsize(boxsize);

  Float ccfl = _convertParameterString(
    "ccfl",
    ArgType::Float,
    /*optional=*/true,
    /*default_val=*/params.getCcfl()
  );
  params.setCcfl(ccfl);

  std::string basename = _convertParameterString(
    "basename",
    ArgType::String,
    /*optional=*/true,
    /*default_val=*/params.getOutputFileBase()
  );
  params.setOutputFileBase(basename);

  bool write_replications = _convertParameterString(
    "write_replications",
    ArgType::Bool,
    /*optional=*/true,
    /*default_val=*/params.getWriteReplications()
  );
  params.setWriteReplications(write_replications);

  Float dt_out = _convertParameterString(
    "dt_out",
    ArgType::Float,
    /*optional=*/true,
    /*default_val=*/params.getDtOut()
  );
  params.setDtOut(dt_out);

  size_t foutput = _convertParameterString(
    "foutput",
    ArgType::Size_t,
    /*optional=*/true,
    /*default_val=*/params.getFoutput()
  );
  params.setFoutput(foutput);


  // Let me know if we missed something
  _checkUnusedParameters();

  // Mark that we've read in the param file
  params.setParamFileHasBeenRead();
}


/**
 * Read initial conditions.
 */
void InputParse::readICFile(Grid& grid) {

  logging::setStage(logging::LogStage::IO);
  timer::Timer tick(timer::Category::IC);

  if (_icIsTwoState()) {
    message("Found two-state IC file.", logging::LogLevel::Verbose);
    _readTwoStateIC(grid);
  } else {
    message("Found arbitrary IC file.", logging::LogLevel::Verbose);
    _readArbitraryIC(grid);
  }

  timing("Reading ICs took " + tick.tock());
}


/**
 * Extract parameters from a line following the format
 * ```
 *  name = value // possible comment
 * ```
 *
 * Returns a pair of strings, the name (before = sign) and value (after =
 * sign). If both these strings are empty, we're skipping this line
 * deliberately. This could be because it's a comment or empty.
 * If something goes wrong during the partising, name and value will be
 * utils::somethingWrong().
 */
std::pair<std::string, std::string> InputParse::extractParameter(std::string& line) {

  std::string name;
  std::string value;

  if (utils::isComment(line))
    return std::make_pair(name, value);
  if (utils::isWhitespace(line))
    return std::make_pair(name, value);

  std::string nocomment = utils::removeTrailingComment(line);
  auto        pair      = utils::splitEquals(nocomment, /*warn=*/true);
  name                  = pair.first;
  value                 = pair.second;

  return std::make_pair(name, value);
}


/**
 * Returns the help message.
 */
std::string InputParse::_helpMessage() {

  std::stringstream msg;
  msg << "This is the hydro code help message.\n\nUsage: \n\n";
  msg << "Default run:\n  ./hydro --param-file <param-file> --ic-file <ic-file>\n";
  msg << "or:\n  ./hydro --param-file=<param-file> --ic-file=<ic-file>\n\n";
  msg << "      <param-file>: file containing your run parameter configuration.\n";
  msg << "                     See README for details.\n";
  msg << "      <ic-file>:     file containing your initial conditions.\n";
  msg << "                     See README for details.\n\n";
  msg << "Get this help message:\n  ./hydro -h\n  ./hydro --help\n\n";
  msg << "Optional flags:\n";
  msg << "  -v,--verbose:         Be talkative.\n";
  msg << "  -vv,--very-verbose:   Be very talkative (intended for debugging).\n";

  return msg.str();
}


/**
 * Verify that the provided command line arguments are valid.
 * May exit if help flag was raised.
 * Sets the verbosity level if it was provided.
 */
void InputParse::_checkCmdLineArgsAreValid() {

  // If help is requested, print help and exit.
  if (_commandOptionExists("-h") or _commandOptionExists("--help")) {
    message(_helpMessage(), logging::LogLevel::Quiet);
    std::exit(0);
  }

  if (_commandOptionExists("-v") or _commandOptionExists("--verbose")) {
    logging::setVerbosity(logging::LogLevel::Verbose);
  }

  if (_commandOptionExists("-vv") or _commandOptionExists("--very-verbose")) {
    logging::setVerbosity(logging::LogLevel::Debug);
  }


  // Vector containing all the valid options we accept. Iterate over this to
  // check if the cmd options we expect to see are present This is defined in
  // the cpp file.
  const std::vector<std::string> _requiredArgs = {
    "--param-file",
    "--ic-file",
  };

  // check all the required options
  for (const auto& opt : _requiredArgs) {
    if (not _commandOptionExists(opt)) {
      std::stringstream msg;
      msg << "missing option: " << opt;
      error(msg.str());
    }
  }

  // Check whether the files we should have are fine
  std::string icfile = _getCommandOption("--ic-file");
  if (not(utils::fileExists(icfile))) {
    std::stringstream msg;
    msg << "Provided initial conditions file '" << icfile << "' doesn't exist.";
    error(msg.str());
  } else {
    // Store it.
    message("Found IC file " + icfile, logging::LogLevel::Debug);
    _icfile = icfile;
  }

  std::string paramfile = _getCommandOption("--param-file");
  if (not(utils::fileExists(paramfile))) {
    std::stringstream msg;
    msg << "Provided parameter file '" << paramfile << "' doesn't exist.";
    error(msg.str());
  } else {
    // Store it.
    _paramfile = paramfile;
    message("Found param file " + paramfile, logging::LogLevel::Debug);
  }
}


/**
 * Has a cmdline option been provided?
 */
bool InputParse::_commandOptionExists(const std::string& option) {

  auto search = _cmdlargs.find(option);
  return (search != _cmdlargs.end());
}


/**
 * Get the value provided by the command option @param option.
 */
std::string InputParse::_getCommandOption(const std::string& option) {

  auto search = _cmdlargs.find(option);
  if (search == _cmdlargs.end()) {
    warning("No option '" + option + "' available");
    const std::string emptyString;
    return emptyString;
  }

  return search->second;
}


/**
 * Find and list unused parameters. Prints unused params to screen with a
 * warning.
 */
void InputParse::_checkUnusedParameters() {

  for (auto& _config_param : _config_params) {
    ParamEntry& entry = _config_param.second;
    if (not entry.used) {
      std::stringstream msg;
      msg << "Unused parameter: " << entry.param << "=" << entry.value;
      warning(msg.str());
    }
#if DEBUG_LEVEL > 1
    else {
      std::stringstream msg;
      msg << "Used parameter: " << entry.param << "=" << entry.value;
      message(msg.str(), logging::LogLevel::Debug);
    }
#endif
  }
}


/**
 * Do we have a two-state format for initial conditions?
 */
bool InputParse::_icIsTwoState() {

  std::string   line;
  std::ifstream conf_ifs(_icfile);
  while (std::getline(conf_ifs, line)) {

    if (utils::isComment(line))
      continue;

    std::string nocomment = utils::removeTrailingComment(line);
    auto        pair      = utils::splitEquals(nocomment, /*warn=*/true);
    std::string name      = pair.first;
    std::string value     = pair.second;

    if (name != "filetype") {
      std::stringstream msg;
      msg << "Invalid IC file type: first non-comment line must be ";
      msg << "`filetype = [two-state,arbitrary]`\n";
      msg << "current line is: `" << line << "`";
      error(msg.str());
    }

    bool out = false;
    if (value == "two-state") {
      out = true;
    } else if (value == "arbitrary") {
      out = false;
    } else {
      std::stringstream msg;
      msg << "Invalid IC file type specification:";
      msg << "`filetype` must be either `two-state` or `arbitrary`\n";
      msg << "I found: `" << value << "`";
      error(msg.str());
    }

    return out;
  }

  return false;
}


/**
 * Extract the value from a single line of the two-state IC file.
 */
Float InputParse::_extractTwoStateVal(
  std::string& line, const char* expected_name, const char* alternative_name
) {

  std::string nocomment = utils::removeTrailingComment(line);
  auto        pair      = utils::splitEquals(nocomment);
  std::string name      = pair.first;
  std::string value     = pair.second;

  if (name != expected_name and name != alternative_name) {
    std::stringstream msg;
    msg << "Something wrong when parsing two-state IC file.\n";
    msg << "Expecting: `" << expected_name << "`\n";
    msg << "Line:`" << line << "`";
    error(msg.str());
  }

  Float out = utils::string2float(value);
  return out;
}


/**
 * Extract a line from an arbitrary-time IC file into a PrimitiveState.
 *
 * @param line the line to work on
 * @param linenr the current line number in the IC. Used in case of errors.
 */
PrimitiveState InputParse::_extractArbitraryICVal(std::string& line, size_t linenr) {

  Float rho = 0;
  Float vx  = 0;
  Float vy  = 0;
  Float p   = 0;

  std::string nocomment = utils::removeTrailingComment(line);
  std::string trimmed   = utils::removeWhitespace(nocomment);

  std::vector<std::string> split;
  std::string              chunk;
  constexpr char           delim = ' ';

  while (trimmed.size() > 0) {

    trimmed  = utils::removeWhitespace(trimmed);
    size_t i = trimmed.find(delim);

    if (i == std::string::npos) {
      // we're done.
      split.push_back(trimmed);
      break;
    }

    // extract
    chunk = trimmed.substr(0, i);
    // store this chunk
    split.push_back(chunk);
    // shorten what we need to process
    trimmed = trimmed.substr(i, trimmed.size() - i);
  }

  if (Dimensions == 1) {

    if (split.size() != 3) {
      std::stringstream msg;
      msg << "Error parsing IC line " << linenr << ": ";
      msg << "Found " << split.size() << " entries instead of 3";
      error(msg.str());
    }

    rho = utils::string2float(split[0]);
    vx  = utils::string2float(split[1]);
    p   = utils::string2float(split[2]);

    return PrimitiveState(rho, vx, p);
  }
  if (Dimensions == 2) {

    if (split.size() != 4) {
      std::stringstream msg;
      msg << "Error parsing IC line " << linenr << ": ";
      msg << "Found " << split.size() << " entries instead of 4";
      error(msg.str());
    }

    rho = utils::string2float(split[0]);
    vx  = utils::string2float(split[1]);
    vy  = utils::string2float(split[2]);
    p   = utils::string2float(split[3]);

    return PrimitiveState(rho, vx, vy, p);
  }

  // Dim != 1, != 2
  error("Not Implemented.");
  return {};
}


//! Read an IC file with the Two-State format
void InputParse::_readTwoStateIC(Grid& grid) {

  // first, read the file.
  std::string   line;
  std::ifstream icts_ifs(_icfile);

  // skip comments first.
  while (std::getline(icts_ifs, line)) {
    if (utils::isComment(line))
      continue;
    break;
  }

  // once we're out of comments, read in one-by-one.
  std::string nocomment = utils::removeTrailingComment(line);
  auto        pair      = utils::splitEquals(nocomment);
  std::string name      = pair.first;
  std::string value     = pair.second;

  if (name != "filetype" or value != "two-state") {
    std::stringstream msg;
    msg << "Something wrong when parsing two-state IC file. Line:`" << line << "`";
    error(msg.str());
  }

  std::getline(icts_ifs, line);
  Float rho_L = _extractTwoStateVal(line, "rho_L");

  std::getline(icts_ifs, line);
  Float v_L = _extractTwoStateVal(line, "u_L", "v_L");

  std::getline(icts_ifs, line);
  Float p_L = _extractTwoStateVal(line, "p_L");

  std::getline(icts_ifs, line);
  Float rho_R = _extractTwoStateVal(line, "rho_R");

  std::getline(icts_ifs, line);
  Float v_R = _extractTwoStateVal(line, "u_R", "v_R");

  std::getline(icts_ifs, line);
  Float p_R = _extractTwoStateVal(line, "p_R");

  PrimitiveState left;
  PrimitiveState right;

  if (Dimensions == 1) {
    left  = PrimitiveState(rho_L, v_L, p_L);
    right = PrimitiveState(rho_R, v_R, p_R);
  } else if (Dimensions == 2) {
    left  = PrimitiveState(rho_L, v_L, 0., p_L);
    right = PrimitiveState(rho_R, v_R, 0., p_R);
  } else {
    error("Not Implemented.");
  }


  // Now allocate and fill up the grid.
  grid.initCells();
  size_t nxtot  = grid.getNxTot();
  size_t nxhalf = nxtot / 2;
  size_t first  = grid.getFirstCellIndex();
  size_t last   = grid.getLastCellIndex();

  if (Dimensions == 1) {

    for (size_t i = first; i < nxhalf; i++) {
      Cell& c = grid.getCell(i);
      c.setPrim(left);
    }
    for (size_t i = nxhalf; i < last; i++) {
      Cell& c = grid.getCell(i);
      c.setPrim(right);
    }

  } else if (Dimensions == 2) {

    for (size_t j = first; j < last; j++) {
      for (size_t i = first; i < nxhalf; i++) {
        Cell& c = grid.getCell(i, j);
        c.setPrim(left);
      }
      for (size_t i = nxhalf; i < last; i++) {
        Cell& c = grid.getCell(i, j);
        c.setPrim(right);
      }
    }

  } else {
    error("Not implemented");
  }
}


//! Read an IC file with the arbitrary format
void InputParse::_readArbitraryIC(Grid& grid) {

  // first, read the file.
  std::string                         nocomment;
  std::pair<std::string, std::string> pair;
  std::string                         name;
  std::string                         value;

  std::string   line;
  std::ifstream ic_ifs(_icfile);
  size_t        linenr = 0;

  // skip comments first.
  while (std::getline(ic_ifs, line)) {
    linenr++;
    if (utils::isComment(line))
      continue;
    break;
  }

  // Make sure the first three required lines are correct.
  nocomment = utils::removeTrailingComment(line);
  pair      = utils::splitEquals(nocomment);
  name      = pair.first;
  value     = pair.second;

  if (name != "filetype" or value != "arbitrary") {
    std::stringstream msg;
    msg
      << "Something wrong when parsing arbitrary-type IC file. Line:" << linenr << "`" << line
      << "`" << "\n"
      << "First line should be `filetype = arbitrary`";
    error(msg.str());
  }

  std::getline(ic_ifs, line);
  linenr++;
  nocomment = utils::removeTrailingComment(line);
  pair      = utils::splitEquals(nocomment);
  name      = pair.first;
  value     = pair.second;

  if (name != "ndim") {
    std::stringstream msg;
    msg << "Something wrong when parsing arbitrary-type IC file. Line:" << linenr << "`" << line
        << "`" << "Second line should be `ndim = <ndim>`";
    error(msg.str());
  }

  int ndim = utils::string2int(value);

  if (ndim != Dimensions) {
    std::stringstream msg;
    msg << "Error: Code compiled for ndim=" << Dimensions << " but IC is for ndim=" << ndim;
    error(msg.str());
  }


  std::getline(ic_ifs, line);
  linenr++;
  nocomment = utils::removeTrailingComment(line);
  pair      = utils::splitEquals(nocomment);
  name      = pair.first;
  value     = pair.second;

  if (name != "nx") {
    std::stringstream msg;
    msg << "Something wrong when parsing arbitrary-type IC file. Line:" << linenr << "`" << line
        << "`" << "Third line should be `nx = <nx>`";
    error(msg.str());
  }

  int nx = utils::string2int(value);

  // Tell the grid how big it is.
  grid.setNx(nx);
  grid.setNxNorep(nx);

  // Now allocate and fill up the grid.
  if (grid.getReplicate() > 1) {
    message("Resizing grid for replications", logging::LogLevel::Verbose);
    grid.setNx(grid.getReplicate() * grid.getNx());
  }
  grid.initCells();


  // Read in the rest of the ICs
  const size_t first = grid.getFirstCellIndex();
  const size_t last  = nx + first;
  size_t       i     = first;
  size_t       j     = first;

  if (Dimensions == 1) {
    error("Not Implemented.");
  } else if (Dimensions == 2) {
    while (std::getline(ic_ifs, line)) {
      linenr++;
      if (utils::isComment(line) or utils::isWhitespace(line))
        continue;

      PrimitiveState pstate = _extractArbitraryICVal(line, linenr);
      grid.getCell(i, j).setPrim(pstate);
      i++;

      if (i == last) {
        i = first;
        j++;
      }
    }
  }

  if (grid.getReplicate() > 1) {
    grid.replicateICs();
  }
}


// -----------------------------------------
// OutputWriter stuff
// -----------------------------------------


/**
 * Generate the output file name to write into.
 */
std::string OutputWriter::_getOutputFileName() {

  std::stringstream fname;
  if (_params.getOutputFileBase() == "") {
    fname << "output";
  } else {
    fname << _params.getOutputFileBase();
  }
  fname << "_";
  fname << std::setfill('0') << std::setw(4) << getNOutputsWritten();
  fname << ".out";

  return fname.str();
}


/**
 * Write the output.
 */
void OutputWriter::dump(Float t_current, size_t step) {

  if (Dimensions != 2) {
    error("Not Implemented Yet");
  }

  // Change the stage we're in
  logging::LogStage prevStage = logging::getCurrentStage();
  logging::setStage(logging::LogStage::IO);
  timer::Timer tick(timer::Category::IO);

  std::stringstream msg;
  msg << "Writing output for step " << step << ", t=" << t_current;
  message(msg.str(), logging::LogLevel::Debug);


  size_t nx = _grid.getNxNorep();
  if (_params.getWriteReplications()) {
    nx = _grid.getNx();
  }
  size_t first = _grid.getFirstCellIndex();
  size_t last  = first + nx;

  constexpr int fwidth = 12;
  constexpr int fprec  = 6;

  std::string   fname = _getOutputFileName();
  std::ofstream out(fname);

  out << "# ndim = " << Dimensions << "\n";
  out << "# nx = " << nx << "\n";
  out << "# t = " << std::setw(fwidth) << std::setprecision(fprec) << t_current << "\n";
  out << "# nsteps = " << step << "\n";
  out << "#            x            y          rho          v_x          v_y            p\n";

  for (size_t j = first; j < last; j++) {
    for (size_t i = first; i < last; i++) {
      Cell& c = _grid.getCell(i, j);
      out << std::setw(fwidth) << std::scientific << std::setprecision(fprec) << c.getX();
      out << " ";
      out << c.getY();
      out << " ";

      PrimitiveState& p = c.getPrim();
      out << p.getRho();
      out << " ";
      out << p.getV(0);
      out << " ";
      out << p.getV(1);
      out << " ";
      out << p.getP();
      out << "\n";
    }
  }

  out.close();


  // We're done, take note of that
  message("Written output to " + fname, logging::LogLevel::Verbose);
  incNOutputsWritten();
  // Write down current time and update when next output is being written
  setTimeLastOutputWritten(t_current);

  // timing("Writing output took " + tick.tock());
  // change it back to where we were
  logging::setStage(prevStage);
}


/**
 * Will we write output this step? Call this before actually doing the
 * computations. This function is allowed to modify the time step size,
 * dtCurrent, so that it writes output at the requested time.
 *
 * @param currentStep the current simulation step index
 * @param tCurrent the current simulation time
 * @param dtCurrent the current simulation time step size
 */
bool OutputWriter::dumpThisStep(size_t current_step, Float t_current, Float& dt_current) {

  if (_params.getDtOut() > 0.) {
    // We're writing outputs based on time intervals.
    // Trim down time step size to fit, if necessary.
    if (t_current + dt_current >= _t_next_dump) {

      dt_current = _t_next_dump - t_current;
      message(
        "Trimmed dt to " + std::to_string(dt_current) + " for output.", logging::LogLevel::Debug
      );

      return true;
    }
    return false;
  }
  if (_params.getFoutput() > 0) {
    // Writing at step 0 is handled outside the main step loop. So check that
    // current_step > 0 too lest you write snapshot 0 twice.
    return ((current_step % _params.getFoutput() == 0) and (current_step > 0));
  }
  if (_params.getDtOut() == 0 and _params.getFoutput() == 0) {
    // Not writing outputs except first and last one.
    return false;
  }

  error("Invalid configuration: Neither dump after time interval nor after nsteps configured.");
  return false;
}
