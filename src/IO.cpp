#include "IO.h"

// i don't see why we need more than this many args
static constexpr int argc_max = 20;


/**
 * Helper functions which live here and nowhere else...
*/

void readUntil(char ch, const char* buffer, char* ptr, FILE* f){
  // set ptr to beginning of the buffer
  ptr = (char*)buffer;
  while ((*ptr = fgetc(f))!=EOF)
  {
    if ( *ptr == ch ) return;
    ptr++;
  }
}

//! I think remove leading whitespace and newline char
std::string readLine( FILE* f )
{
  // line buffer 
  char  lineBuffer[256] = {0};
  char* ptr = lineBuffer;

  // This line reads a character from the
  // file, places it in the buffer, but 
  // only if we haven't reached EOF yet.
  while (( *ptr = fgetc(f) )!=EOF)
  {
    if ( *ptr == '\n' ) break;
    ptr++;
  }

  static const std::string emptyString = "";
  if (
    lineBuffer[0] == '\n'
    or
    lineBuffer[0] == EOF
    ) return emptyString;
  return std::string(lineBuffer);
}

std::string removeWhiteSpace( const std::string& line )
{
  if (line.empty()) return line;
  // find the first non-space char
  int offset = line.find_first_not_of( ' ' );
  return line.substr(offset);
}

bool isComment(std::string line)
{
  return 
    removeWhiteSpace(line).substr(0,2) == "//"
    or
    removeWhiteSpace(line).substr(0,2) == "/*";
}

namespace hydro_playground{
  namespace IO
  {
    
    const std::vector< std::string > InputParse::requiredArgs = {
      // {"--help",        "-h"},
      "--config-file",
      "--ic-file",
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
      
      return output;
    }
    
    void InputParse::readCommandOptions()
    {
      // check it first...
      // readICFile( getCommandOption("--ic-file") );

    }

    void readICFile(std::string filename)
    {
      FILE* icfile = fopen(filename.c_str(), "rb");
      if (icfile==nullptr) throw std::runtime_error("Invalid IC File!\n");
  
      


      fclose(icfile);
    }
    
  } // namespace IO
  
} // namespace hydro_playground
