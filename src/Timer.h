/**
 * @file Timer.h
 * @brief A small class and tools to help with timing.
 */
#pragma once

#include <chrono>
#include <string>
#include <typeinfo>

namespace timer {

  // Alias the namespace for convenience.
  namespace chr = std::chrono;


  // Rename time "units" for convenience.
  namespace unit {
    using ms = chr::milliseconds;
    using ns = chr::nanoseconds;
    using mus = chr::microseconds;
    using s = chr::seconds;
  } // namespace unit


  // More convenience aliasing
  using default_time_units = unit::mus;
  using ticks = int64_t;


  /**
   * Timer categories: Will be used to accumulate time spent
   * in different tasks of the code.
   */
  enum Category {
    Init = 0,
    IC,
    Step,
    IO,
    Total,
    Ignore,
    Count
  };


  /**
   * Get a name for a timer category
   */
  const char* getTimerName(Category t);


  /**
   * @brief a small class to time code execution.
   * Usage: Create a Timer object. Time count starts at instantiation.
   * The timing is printed out when the object destructs, or do it manually
   * using timer::Timer.end()
   */
  template <typename time_units = default_time_units>
    class Timer {

    // Variables

    //! Starting point of timing.
    chr::time_point<chr::high_resolution_clock> _start;


    //! Storage for global timings.
    static std::array<ticks, Category::Count> timings;


    //! This instance's category
    Category _category;


    //! Has this timing already ended?
    bool _ended;


    // Methods


    //! Start timing.
    void _start_timing();


    //! End the timing. This adds the current timing to the global timings
    ticks _end_timing();


    //! get duration since object was created.
    ticks _get_duration();


    /**
     * Get the used units as a string.
     */
    static const char* _units_str();


  public:

    //! Constructors
    Timer() : _category(Category::Ignore), _ended(false) {
      _start_timing();
    }

    explicit Timer(Category cat) : _category(cat), _ended(false) {
      _start_timing();
    }

    ~Timer() {
      // If user hasn't ended the measurement manually, do it now.
      if (not _ended){
        (void)_end_timing();
      }
    }

    //! End the timing and returns a string containing the measurement.
    std::string tock();

    //! Get a string for printouts of global timers
    static std::string getTimings();
  };
} // namespace timer


// -------------------------------------
// Definitions
// -------------------------------------


template <typename time_units>
std::array<timer::ticks, timer::Category::Count> timer::Timer<time_units>::timings = {static_cast<ticks>(0)};



template <typename time_units>
const char*
timer::Timer<time_units>::_units_str() {
  if (typeid(time_units) == typeid(chr::nanoseconds))
    return "[ns]";
  if (typeid(time_units) == typeid(chr::microseconds))
    return "[mus]";
  if (typeid(time_units) == typeid(chr::milliseconds))
    return "[ms]";
  if (typeid(time_units) == typeid(chr::seconds))
    return "[s]";
  if (typeid(time_units) == typeid(chr::minutes))
    return "[min]";
  if (typeid(time_units) == typeid(chr::hours))
    return "[h]";
  return "[unknown units]";
}


/**
 * Mark the starting point of the timing.
 */
template <typename time_units>
void timer::Timer<time_units>::_start_timing(){
  _start = chr::high_resolution_clock::now();
}


/**
 * Get a name for a timer category
 */
inline const char* timer::getTimerName(Category t){
  switch(t) {
    case Category::Init:
      return "Init";
    case Category::IC:
      return "Initial Conditions";
    case Category::Step:
      return "Step";
    case Category::IO:
      return "I/O";
    case Category::Total:
      return "Total";
    case Category::Ignore:
      return "ignored";
    case Category::Count:
      return "count";
    default:
      return "unknown";
  }
}




/**
 * Get duration since object was created.
 */
template <typename time_units>
timer::ticks timer::Timer<time_units>::_get_duration(){

  auto _stop = chr::high_resolution_clock::now();
  auto duration = duration_cast<time_units>(_stop - _start);
  return duration.count();
}


/**
 * End the timing and add the duration to the global counter
 */
template <typename time_units>
timer::ticks timer::Timer<time_units>::_end_timing(){

  ticks duration = _get_duration();
  timings[_category] += duration;
  _ended = true;

  return duration;
}


/**
 * End the timing and return a string with the measurement
 */
template <typename time_units>
std::string timer::Timer<time_units>::tock(){

  ticks duration = _end_timing();

  std::string out = std::to_string(duration);
  out += " ";
  out += _units_str();

  return out;
}


/**
 * Returns a string for printouts containing all global timings.
 */
template <typename time_units>
std::string timer::Timer<time_units>::getTimings(){

  std::stringstream out;
  out << "\nTiming units: ";
  out << _units_str();
  out << "\n";

  for (int i = 0; i < Category::Count - 1; i++){
    out << std::setw(20) << getTimerName(static_cast<Category>(i)) << ": ";
    out << std::setw(20) << timings[i] << "\n";
  }

  return out.str();
}
