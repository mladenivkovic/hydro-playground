#include <cassert>

#include "Grid.h"
#include "IO.h"


// I don't see why we need more than this many args
static constexpr int argc_max    = 20;
static constexpr int lineLength  = 256;


template <typename T>
void resetBuffer(T* buffer, int len = lineLength)
{
  for (int i=0; i<len; i++) {
    buffer[i] = 0;
  }
}


bool isWhitespace( char* const line, int len=lineLength )
{
  // Scan through the line buffer. if we see
  // any character that isn't \n, space, EOF or null
  // then return false

  bool output = true;
  char* ptr   = line;
  for (int i=0; i<len; i++)
  {
    if ( ( *(ptr+i)!=' ' ) and ( *(ptr+i)!='\n' ) and ( *(ptr+i)!=EOF) and (bool)(*(ptr+i)) )
    {
      output = false;
      break;
    }
  }
  return output;
}

bool isComment( char* const line, int len=lineLength)
{
  // Scan past all the spaces, if the first non-space
  // chars you see are // or /*, then return true.
  // beware of dereferencing over the end of the
  // array.
  char* ptr = line;
  while( *ptr == ' ' )
  {
    if (std::distance(line, ptr)>len-2) return false;
    ptr++;
  }

  return (
    ( *ptr == '/' and *(ptr+1) == '/')
    or
    ( *ptr == '/' and *(ptr+1) == '*')
  );
}

bool lineIsInvalid(char* const line, int len=lineLength)
{
  bool output = false;
  output |= isWhitespace(line);
  output |= isComment(line);
  // room for other stuff...

  return output;
}

namespace IO {

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

  /*
  This method is a bit of a mess
  */
  void InputParse::readICFile()
  {
    std::string filename = getCommandOption("--ic-file");
    FILE* icfile = fopen(filename.c_str(), "rb");
    if (icfile==nullptr) throw std::runtime_error("Invalid IC File!\n");

    // Let's find how many bytes we have in the file
    fseek(icfile, 0, SEEK_END);
    auto bytesToRead = ftell(icfile);
    // seek back to the start...
    fseek(icfile, 0, SEEK_SET);

    // Buffer to fill with data from the file
    char  lineBuffer[lineLength] = {0};
    // Pointer to move across the buffer. We
    // use this to fill the buffer with data
    char* lineptr(lineBuffer);

    // lambda to advance our file pointer. Would do this with
    // aux function but i wanna keep the pointers in the
    // stack frame
    auto readUntil = [&]( const char& ch ){
      resetBuffer(lineBuffer);
      // reset pointer to start of buffer
      lineptr = lineBuffer;
      while ( ( *lineptr = fgetc(icfile) ) != EOF )
      {
        // Decrement the number of bytes we have to read...
        bytesToRead--;
        if (*lineptr == ch) return true;
        lineptr++;
      }
      return false;
    };

    // define another lambda to fetch a float
    // value that falls between two pointers
    char* ptr0; char* ptr1;
    auto readVal = [&](){
      // bring ptr0 up to speed with ptr0
      ptr0 = ptr1;
      // find the start of the number
      while (*ptr0==' ') ptr0++;
      ptr1 = ptr0;
      while (*ptr1!=' ' and *ptr1!='\n')
      {
        ptr1++;
      }
      if (std::distance(ptr0,ptr1)<2) throw std::runtime_error("Invalid line!\n");
      return strtod(ptr0, &ptr1);
    };

    // read filetype
    readUntil('=');
    readUntil('\n');
    message(lineBuffer);

    // check ndim matches parameters dims
    // read ndims
    readUntil('=');
    readUntil('\n');
    int dims = strtol(lineBuffer, &lineptr, 10);
    // parameters::Parameters::Instance.setDims(dims);
    message(lineBuffer);

    // check nx matches params
    readUntil('=');
    readUntil('\n');
    int nx = strtol(lineBuffer, &lineptr, 10);
    if ( Dimensions==2 ) nx = nx * nx;
    int valuesFetched = 0;

    message("setting nx to "); message(lineBuffer);

    // loop over remaining lines and store results
    int valsToFetchPerLine = 2 + dims;

    /*
    Warning - make sure we don't place these
    outside of the boundary!

    start at bc etc!

    bytesToRead is decremented inside readUntil
    */
    while ( bytesToRead > 0 )
    {
      // fill the line buffer with some data
      readUntil('\n');
      if ( lineIsInvalid(lineBuffer) ) continue;

      std::vector<float> initialValuesToPassOver( valsToFetchPerLine, 0 );
      // fetch values from the current line buffer
      for (int i=0; i<valsToFetchPerLine; i++)
      {
        initialValuesToPassOver[i] = readVal();
      }
      // reset the pointers we use to the start of the buffer
      ptr0 = lineBuffer; ptr1 = lineBuffer;

      // Send these off to the grid - handle indexing in the grid class
      grid::Grid::Instance.SetInitialConditions(valuesFetched, initialValuesToPassOver);

      valuesFetched++;
    }

    // validation - does valuesFetched match nx?
    assert(valuesFetched==nx);


    fclose(icfile);
  }

} // namespace IO

