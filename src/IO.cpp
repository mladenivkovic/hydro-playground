#include "IO.h"

namespace hydro_playground{
  namespace IO
  {
    
    const std::vector< std::string > InputParse::validOptions = {
      // {"--help",        "-h"},
      "--input-file",
      "--output-file",
    };

    const std::string InputParse::helpMessage = "\
    This is the help message!\n \
    ";

    //! Consutrctor
    InputParse::InputParse(int argc, char* argv[]){
      // push all the argv into the vector to hold them
      // start at 1; don't care about the binary name
      for (int i=1; i<argc; i++)
        clArguments.push_back( std::string( argv[i] ) );
    }

    std::string InputParse::getCommandOption( const std::string& option )
    {
      auto iter = std::find( clArguments.begin(), clArguments.end(), option );
      // make sure we aren't at the end, and that there's something to read...
      if ( iter != clArguments.end() and ++iter != clArguments.end() )
      {
        return *iter;
      }

      // no luck. return the empty string
      static const std::string emptyString("");
      return emptyString;

    }

    bool InputParse::commandOptionExists( const std::string& option )
    {
      auto iter = std::find( clArguments.begin(), clArguments.end(), option );
      return ( iter != clArguments.end() );
    }

    void InputParse::readCommandOptions()
    {
      // check if they're asking for help first
      if ( commandOptionExists("-h") or commandOptionExists("--help") )
      {
        std::cout << helpMessage; std::abort();
      }

      // check for input filename
      if ( commandOptionExists("--input-file") )
      {
        std::cout << "found input file: " << getCommandOption("--input-file") << "\n";
      }
      else {
        std::cout << "no input file provided...\n"; std::abort();
      }
    }
    
  } // namespace IO
  
} // namespace hydro_playground
