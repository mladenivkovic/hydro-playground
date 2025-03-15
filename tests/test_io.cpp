#include <cassert>
#include <string>

#include "IO.h"
#include "Logging.h"
#include "Utils.h"


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

  if (not utils::isWhitespace(line_empty))
    error("Wrong.");

  if (not utils::isWhitespace(line_eof))
    error("Wrong.");

  if (utils::isWhitespace(line_something))
    error("Wrong.");

  if (utils::isWhitespace(line_start_whitespace))
    error("Wrong.");

  if (not utils::isWhitespace(line_whitespace))
    error("Wrong.");

  if (not utils::isWhitespace(line_whitespaces))
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


  if (utils::isComment(line_empty))
    error("Wrong.");
  if (utils::isComment(line_eof))
    error("Wrong.");
  if (utils::isComment(line_something))
    error("Wrong.");
  if (not utils::isComment(line_comment1))
    error("Wrong.");
  if (not utils::isComment(line_comment2))
    error("Wrong.");
  if (not utils::isComment(line_comment3))
    error("Wrong.");
  if (not utils::isComment(line_comment4))
    error("Wrong.");
  if (not utils::isComment(line_comment5))
    error("Wrong.");
  if (not utils::isComment(line_comment6))
    error("Wrong.");

  message("-- finished.");
}


/**
 * test removeWhitespace()
 */
void testRemoveWhitespace() {

  message("-- Running testRemoveWhitespace()");

  std::string in;
  std::string out;

  in  = "";
  out = utils::removeWhitespace(in);
  if (out != "")
    error("Wrong:'" + out + "'");

  in  = " ";
  out = utils::removeWhitespace(in);
  if (out != "")
    error("Wrong:'" + out + "'");

  in  = "\t";
  out = utils::removeWhitespace(in);
  if (out != "")
    error("Wrong:'" + out + "'");

  in  = "word";
  out = utils::removeWhitespace(in);
  if (out != "word")
    error("Wrong:'" + out + "'");

  in  = " heading";
  out = utils::removeWhitespace(in);
  if (out != "heading")
    error("Wrong:'" + out + "'");

  in  = "       heading2";
  out = utils::removeWhitespace(in);
  if (out != "heading2")
    error("Wrong:'" + out + "'");

  in  = "trailing ";
  out = utils::removeWhitespace(in);
  if (out != "trailing")
    error("Wrong:'" + out + "'");
  message("-- finished.");

  in  = "trailing2       \n";
  out = utils::removeWhitespace(in);
  if (out != "trailing2")
    error("Wrong:'" + out + "'");

  in  = "  two words       \n";
  out = utils::removeWhitespace(in);
  if (out != "two words")
    error("Wrong:'" + out + "'");

  in  = "  three words or more            \n";
  out = utils::removeWhitespace(in);
  if (out != "three words or more")
    error("Wrong:'" + out + "'");

  in  = "\n\n heading newlines  \n";
  out = utils::removeWhitespace(in);
  if (out != "heading newlines")
    error("Wrong:'" + out + "'");

  message("-- finished.");
}


/**
 * test splitEquals()
 */
void testSplitEquals() {

  message("-- Running testSplitEquals()");

  std::string in;
  std::string name;
  std::string val;
  std::string no = utils::somethingWrong();

  in       = "";
  auto out = utils::splitEquals(in);
  name     = out.first;
  val      = out.second;
  if (name != no)
    error("Wrong:'" + name + "'");
  if (val != no)
    error("Wrong:'" + val + "'");

  in   = "a = n";
  out  = utils::splitEquals(in);
  name = out.first;
  val  = out.second;
  if (name != "a")
    error("Wrong:'" + name + "'");
  if (val != "n")
    error("Wrong:'" + val + "'");

  in   = " myname =    myValue    \n";
  out  = utils::splitEquals(in);
  name = out.first;
  val  = out.second;
  if (name != "myname")
    error("Wrong:'" + name + "'");
  if (val != "myValue")
    error("Wrong:'" + val + "'");

  in   = "a = b = c";
  out  = utils::splitEquals(in);
  name = out.first;
  val  = out.second;
  if (name != no)
    error("Wrong:'" + name + "'");
  if (val != no)
    error("Wrong:'" + val + "'");


  message("-- finished.");
}


/**
 * test removeTrailingComments()
 */
void testRemoveTrailingComments() {

  message("-- Running testRemoveTrailingComments()");

  std::string in;
  std::string out;

  in  = "";
  out = utils::removeTrailingComment(in);
  if (out != "")
    error("Wrong:'" + out + "'");

  in  = "word";
  out = utils::removeTrailingComment(in);
  if (out != "word")
    error("Wrong:'" + out + "'");

  in  = "// word";
  out = utils::removeTrailingComment(in);
  if (out != "")
    error("Wrong:'" + out + "'");

  in  = "/* word";
  out = utils::removeTrailingComment(in);
  if (out != "")
    error("Wrong:'" + out + "'");

  in  = " some text  // comment";
  out = utils::removeTrailingComment(in);
  if (out != " some text  ")
    error("Wrong:'" + out + "'");

  in  = " some text  /* comment";
  out = utils::removeTrailingComment(in);
  if (out != " some text  ")
    error("Wrong:'" + out + "'");

  in  = " some text  // /* comment";
  out = utils::removeTrailingComment(in);
  if (out != " some text  ")
    error("Wrong:'" + out + "'");

  in  = " some text  /* // /* comment";
  out = utils::removeTrailingComment(in);
  if (out != " some text  ")
    error("Wrong:'" + out + "'");

  message("-- finished.");
}


/**
 * Test the extractParamLine function
 */
void testExtractParamLine() {

  message("-- Running testExtractParamLine()");

  std::string line_valid1("myname = myvalue");
  std::string line_valid2("mynameNospace=myvalueNospace");
  std::string line_valid3("   mynameStartWithSpace   = myvalueNospace");
  std::string line_valid4("   mynameStartWithSpace   = myvalueNospace // comment");
  std::string line_valid5("   mynameStartWithSpace   = myvalueNospace /* comment");
  std::string line_1("//  mynameStartWithSpace   = myvalueNospace");
  std::string line_2("/*  mynameStartWithSpace   = myvalueNospace");
  std::string line_3(" mynameStartWithSpace   = myvalueNospace = secondValue");
  std::string line_4("");

  std::pair<std::string, std::string> out;
  std::string                         name;
  std::string                         value;
  std::string                         no = utils::somethingWrong();

  out   = IO::InputParse::extractParameter(line_valid1);
  name  = out.first;
  value = out.second;
  if (name != "myname" or value != "myvalue")
    error("Wrong: '" + name + "', '" + value + "'");

  out   = IO::InputParse::extractParameter(line_valid2);
  name  = out.first;
  value = out.second;
  if (name != "mynameNospace" or value != "myvalueNospace")
    error("Wrong: '" + name + "', '" + value + "'");

  out   = IO::InputParse::extractParameter(line_valid3);
  name  = out.first;
  value = out.second;
  if (name != "mynameStartWithSpace" or value != "myvalueNospace")
    error("Wrong: '" + name + "', '" + value + "'");

  out   = IO::InputParse::extractParameter(line_valid4);
  name  = out.first;
  value = out.second;
  if (name != "mynameStartWithSpace" or value != "myvalueNospace")
    error("Wrong: '" + name + "', '" + value + "'");

  out   = IO::InputParse::extractParameter(line_valid5);
  name  = out.first;
  value = out.second;
  if (name != "mynameStartWithSpace" or value != "myvalueNospace")
    error("Wrong: '" + name + "', '" + value + "'");

  out   = IO::InputParse::extractParameter(line_1);
  name  = out.first;
  value = out.second;
  if (name != "" or value != "")
    error("Wrong: '" + name + "', '" + value + "'");

  out   = IO::InputParse::extractParameter(line_2);
  name  = out.first;
  value = out.second;
  if (name != "" or value != "")
    error("Wrong: '" + name + "', '" + value + "'");

  out   = IO::InputParse::extractParameter(line_3);
  name  = out.first;
  value = out.second;
  if (name != no or value != no)
    error("Wrong: '" + name + "', '" + value + "'");

  out   = IO::InputParse::extractParameter(line_4);
  name  = out.first;
  value = out.second;
  if (name != "" or value != "")
    error("Wrong: '" + name + "', '" + value + "'");

  message("-- finished.");
}


/**
 * Test the extractParamLine function
 */
void testStringConversions() {

  message("-- Running testStringConversions()");

  std::string val;

  // int
  val = "2";
  if (utils::string2int(val) != 2)
    error("Wrong.");

  val = "-17";
  if (utils::string2int(val) != -17)
    error("Wrong.");

  // val = " ";
  // utils::string2int(val);

  val = "1.0";
  if (utils::string2float(val) != 1.)
    error("Wrong.");

  val = "-2.0";
  if (utils::string2float(val) != -2.)
    error("Wrong.");

  val = "true";
  if (not utils::string2bool(val))
    error("Wrong.");

  val = "0";
  if (utils::string2bool(val))
    error("Wrong.");

  message("-- finished.");
}


/**
 * Runs unit tests on internals
 */
void unit_tests() {
  message("Running unit tests.");

  testIsWhitespace();
  testIsComment();
  testRemoveWhitespace();
  testRemoveTrailingComments();
  testSplitEquals();
  testStringConversions();
  testExtractParamLine();

  message("Finished unit tests.");
}


int main() {

  logging::setStage(logging::LogStage::Test);

  unit_tests();


  return 0;
}
