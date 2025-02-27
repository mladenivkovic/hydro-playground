#include <cassert>
#include <string>

#include "IO.h"
#include "Logging.h"


/**
 * test the isWhitespace function
 */
void testIsWhitespace() {

  message("-- Running testIsWhitespace()");

  std::string line_empty("");
  std::string line_eof(1, static_cast<char>(EOF));
  std::string line_something("something");
  std::string line_start_whitespace(" line starts with whitespace ");
  std::string line_whitespace(" ");
  std::string line_whitespaces(" \t\n\r");

  if (not IO::internal::isWhitespace(line_empty))
    error("Wrong.");

  if (not IO::internal::isWhitespace(line_eof))
    error("Wrong.");

  if (IO::internal::isWhitespace(line_something))
    error("Wrong.");

  if (IO::internal::isWhitespace(line_start_whitespace))
    error("Wrong.");

  if (not IO::internal::isWhitespace(line_whitespace))
    error("Wrong.");

  if (not IO::internal::isWhitespace(line_whitespaces))
    error("Wrong.");

  message("-- Finished.");
}

/**
 * Test the isComment() function
 */
void testIsComment() {

  message("-- Running testIsComment()");

  std::string line_empty("");
  std::string line_eof(1, static_cast<char>(EOF));
  std::string line_something("something");
  std::string line_comment1("// something");
  std::string line_comment2("/* something */");
  std::string line_comment3("  // something");
  std::string line_comment4("  /* something */");
  std::string line_comment5("\t // something");
  std::string line_comment6("\t /* something */");


  if (IO::internal::isComment(line_empty))
    error("Wrong.");
  if (IO::internal::isComment(line_eof))
    error("Wrong.");
  if (IO::internal::isComment(line_something))
    error("Wrong.");
  if (not IO::internal::isComment(line_comment1))
    error("Wrong.");
  if (not IO::internal::isComment(line_comment2))
    error("Wrong.");
  if (not IO::internal::isComment(line_comment3))
    error("Wrong.");
  if (not IO::internal::isComment(line_comment4))
    error("Wrong.");
  if (not IO::internal::isComment(line_comment5))
    error("Wrong.");
  if (not IO::internal::isComment(line_comment6))
    error("Wrong.");

  message("-- finished.");

}



/**
 * test removeWhitespace()
 */
void testRemoveWhitespace(){

  message("-- Running testRemoveWhitespace()");

  std::string in;
  std::string out;

  in = "";
  out = IO::internal::removeWhitespace(in);
  if (out != "") error("Wrong:'" + out + "'");

  in = "word";
  out = IO::internal::removeWhitespace(in);
  if (out != "word") error("Wrong:'" + out + "'");

  in = " heading";
  out = IO::internal::removeWhitespace(in);
  if (out != "heading") error("Wrong:'" + out + "'");

  in = "       heading2";
  out = IO::internal::removeWhitespace(in);
  if (out != "heading2") error("Wrong:'" + out + "'");

  in = "trailing ";
  out = IO::internal::removeWhitespace(in);
  if (out != "trailing") error("Wrong:'" + out + "'");
  message("-- finished.");

  in = "trailing2       \n";
  out = IO::internal::removeWhitespace(in);
  if (out != "trailing2") error("Wrong:'" + out + "'");

  in = "  two words       \n";
  out = IO::internal::removeWhitespace(in);
  if (out != "two words") error("Wrong:'" + out + "'");

  in = "  three words or more            \n";
  out = IO::internal::removeWhitespace(in);
  if (out != "three words or more") error("Wrong:'" + out + "'");

  in = "\n\n heading newlines  \n";
  out = IO::internal::removeWhitespace(in);
  if (out != "heading newlines") error("Wrong:'" + out + "'");

  message("-- finished.");

}


/**
 * test splitEquals()
 */
void testSplitEquals(){


  message("-- Running testSplitEquals()");

  std::string in;
  std::string name;
  std::string val;
  std::string no = IO::internal::somethingWrong();

  in = "";
  auto out = IO::internal::splitEquals(in);
  name = out.first;
  val = out.second;
  if (name != no) error("Wrong:'" + name + "'");
  if (val != no) error("Wrong:'" + val + "'");

  in = "a = n";
  out = IO::internal::splitEquals(in);
  name = out.first;
  val = out.second;
  if (name != "a") error("Wrong:'" + name + "'");
  if (val != "n") error("Wrong:'" + val + "'");

  in = "a = b = c";
  out = IO::internal::splitEquals(in);
  name = out.first;
  val = out.second;
  if (name != no) error("Wrong:'" + name + "'");
  if (val != no) error("Wrong:'" + val + "'");





  message("-- finished.");

}



/**
 * test removeTrailingComments()
 */
void testRemoveTrailingComments(){

}




/**
 * Test the extractParamLine function
 */
void testExtractParamLine() {

  std::string line_empty("");
  std::string line_eof(1, static_cast<char>(EOF));
  std::string line_something("something");
  std::string line_comment1("// something");
  std::string line_comment2("/* something */");
  std::string line_comment3("  // something");
  std::string line_comment4("  /* something */");
  std::string line_comment5("\t // something");
  std::string line_comment6("\t /* something */");

  std::string line_valid1("myname = myvalue");
  std::string line_valid2("mynameNospace=myvalueNospace");
  std::string line_valid3("   mynameStartWithSpace   = myvalueNospace");
  std::string line_valid4("   mynameStartWithSpace   = myvalueNospace // comment");
  std::string line_valid5("   mynameStartWithSpace   = myvalueNospace /* comment");


  IO::internal::extractParameter(line_valid1);
  IO::internal::extractParameter(line_valid2);
  IO::internal::extractParameter(line_valid3);
}


/**
 * Runs unit tests on internals
 */
void unit_tests() {
  message("Running unit tests.")

  // testIsWhitespace();"
  // testIsComment();
  testRemoveWhitespace();
  testSplitEquals();
  // testExtractParamLine();

  message("Finished unit tests.")
}


int main() {

  logging::Log::setStage(logging::LogStage::Test);

  unit_tests();


  return 0;
}
