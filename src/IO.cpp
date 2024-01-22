#include "IO.h"

// i don't see why we need more than this many args
static constexpr int argc_max = 20;

namespace hydro_playground{
  namespace IO
  {
    
    const std::vector< std::string > InputParse::requiredArgs = {
      // {"--help",        "-h"},
      "--input-file",
      "--output-file",
    };

    const std::string InputParse::helpMessage = "This is the help message!\n";

    //! Consutrctor
    InputParse::InputParse(int argc, char* argv[]){
      // push all the argv into the vector to hold them
      // start at 1; don't care about the binary name
      for (int i=1; i<std::min(argc, argc_max); i++)
        clArguments.push_back( std::string( argv[i] ) );
    }

    std::string InputParse::getCommandOption( const std::string& option )
    {
      auto iter = std::find( clArguments.begin(), clArguments.end(), option );
      // make sure we aren't at the end, and that there's something to read...
      // mind the sneaky increment in the "if" clause...
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

    bool InputParse::inputIsValid()
    {
      bool output = true;

      if (commandOptionExists("-h") or commandOptionExists("--help"))
      {
        message(helpMessage);
        output = false;
      }

      // check all the required options
      for (const auto& opt:requiredArgs){
        if ( std::find( clArguments.begin(), clArguments.end(), opt ) == clArguments.end() )
        {
          std::string msg = "missing option: " + opt;
          message(msg);
          output = false;
        }
      }
      
      // check the ic file is valid!
      FILE* icfile = fopen( getCommandOption("--input-file").c_str(), "rb" );
      if (icfile == nullptr)
      {
        message("Invalid icfile!");
        output = false;
      }
      fclose(icfile);
      // 

      return output;
    }
    
    void InputParse::readCommandOptions()
    {


    }
    
  } // namespace IO
  
} // namespace hydro_playground
