#include <string>

#include "IO.h"
#include "Logging.h"

void testIsWhitespace() {

  message("-- Running testIsComment()");

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
}


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


  // if (IO::internal::isComment(line_empty))
  //   error("Wrong.")
  if (IO::internal::isComment(line_eof))
    error("Wrong.") if (IO::internal::isComment(line_something)) error("Wrong."
    ) if (not IO::internal::isComment(line_comment1)) error("Wrong."
    ) if (not IO::internal::isComment(line_comment2)) error("Wrong."
    ) if (not IO::internal::isComment(line_comment3)) error("Wrong."
    ) if (not IO::internal::isComment(line_comment4)) error("Wrong."
    ) if (not IO::internal::isComment(line_comment5)) error("Wrong."
    ) if (not IO::internal::isComment(line_comment6)) error("Wrong.")

      message("-- finished.")
}


/**
 * Runs unit tests on internals
 */
void unit_tests() {
  message("Running unit tests.")

    testIsWhitespace();
  testIsComment();

  message("Finished unit tests.")
}


int main() {

  logging::Log::setStage(logging::LogStage::Test);

  unit_tests();

  return 0;
}
