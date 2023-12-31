#pragma once

#include <string>

// https://stackoverflow.com/questions/1435953/how-can-i-pass-git-sha1-to-compiler-as-definition-using-cmake

namespace version {
  /**
   * Contains compile time information on code version, current
   * git state, etc.
   */
  struct Version {
    static const int MINOR;
    static const int MAJOR;

    static const std::string GIT_SHA1;
    static const std::string GIT_DATE;
    static const std::string GIT_BRANCH;
    static const std::string GIT_COMMIT_SUBJECT;
  };
} // namespace version
