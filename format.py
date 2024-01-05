#!/usr/bin/env python3

import os
import sys
import argparse
import subprocess

# Some repository setup.
repo_rootdir = "."
python_venv_dir = ".formatting_python_env"

# These can be overwritten by optional cmdline arg
default_clang_format_file_name = ".clang-format"
target_branch = "p4"

# Formatting tools versions
black_version = "23.3.0"
clang_version = "16.0.5"

# Global variables for easy access
verbose = True  # Will be overwritten by arg parser
black_checked = False
clang_checked = False


def get_repo_root():
    """
    Get the root dir of the repository.

    returns:
    repo_root: str
        root directory of repository
    """

    cmd = "git rev-parse --show-toplevel"

    #  if verbose:
    #      print("Searching for root dir via git.")
    #      print("Running '{}'".format(cmd))

    gitrevparse = subprocess.run(
        cmd,
        shell=True,
        check=True,
        capture_output=True,
    )
    gitstdout = gitrevparse.stdout
    if isinstance(gitstdout, bytes):
        gitstdout = gitstdout.decode("utf8")
    rootdir = gitstdout.strip()
    return rootdir


def get_default_clang_format(repo_root):
    """
    Try and find the default clang format style file
    """

    clang_format_style_file = os.path.join(repo_root, default_clang_format_file_name)

    if not os.path.exists(clang_format_style_file):
        raise FileNotFoundError(
            "Couldn't find default clang format style file " + clang_format_style_file
        )

    return clang_format_style_file


def get_venv_path(repo_root):
    return os.path.join(repo_root, python_venv_dir)


def git_is_available():
    """
    Check whether git is available. Returns True is that is the case.
    """
    # Does it run?
    cmd = "git --version"

    #  if verbose:
    #      print("Checking if git is available")
    #      print("Running '{}'".format(cmd))

    run = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    if run.returncode == 0:
        return True
    else:
        if verbose:
            print("Didn't find git. Continuing without.")
        return False


def check_black(black_cmd):
    """
    Check that the black executable runs, and is the correct version.
    The subprocess call crashes if there is an error with black.
    """
    global black_checked

    if not black_checked:
        # Does it run?

        cmd = black_cmd + " --version"

        if verbose:
            print("Checking black formatter runs.")
            print("Running '{}'".format(cmd))

        run = subprocess.run(
            cmd,
            shell=True,
            stdout=None,
            check=True,
            capture_output=True,
        )

        # Now check the version
        stdout = run.stdout.decode("UTF-8")
        stdout = stdout.strip()
        stdout_split = stdout.split()

        if stdout_split[0] != "black,":
            raise ValueError("black --version output was:", stdout, " ???")

        installed_version = stdout_split[1]
        if installed_version != black_version:
            raise ValueError(
                (
                    "black formatter executable has version {0} but I require {1}.\n"
                    + "Please remove all contents from the directory '{2}' and re-run this script."
                ).format(installed_version, black_version, python_venv_dir)
            )

        black_checked = True

    return


def check_clang(clang_cmd):
    """
    Check that the clang-format executable runs, and is the correct version.
    The subprocess call crashes if there is an error with clang-format.
    """
    global clang_checked
    if not clang_checked:
        # Does it run?
        run = subprocess.run(
            clang_cmd + " --version",
            shell=True,
            stdout=None,
            check=True,
            capture_output=True,
        )

        # Now check the version
        stdout = run.stdout.decode("UTF-8")
        stdout = stdout.strip()
        stdout_split = stdout.split()

        if stdout_split[0] != "clang-format":
            raise ValueError("clang-format --version output was:", stdout, " ???")

        installed_version = stdout_split[2]
        if installed_version != clang_version:
            raise ValueError(
                (
                    "clang-format executable has version {0} but I require {1}.\n"
                    + "Please remove all contents from the directory '{2}' and re-run this script."
                ).format(installed_version, clang_version, python_venv_dir)
            )

        clang_checked = True

    return


def install_black_formatting_tool():
    """
    Installs a specific version of `black`
    """
    venv_path = get_venv_path(repo_root)

    # Basically hydro-playground/.formatting_python_venv/bin/python3
    bfepython = os.path.join(venv_path, "bin", "python3")

    if not os.path.exists(bfepython):
        raise FileNotFoundError(
            "Something went horribly wrong when setting up the venv."
        )

    # The actual install
    cmd = (bfepython + " -m pip install click==8.1.3 black==" + black_version,)
    if verbose:
        print("Installing black formatter.")
        print("Running '{}'".format(cmd))

    run = subprocess.run(
        cmd,
        shell=True,
        check=True,
    )

    # Check we succeeded
    black_path = os.path.join(venv_path, "bin", "black")
    check_black(black_path)

    return


def install_clang_formatting_tool():
    """
    Installs a specific version of `clang-format`
    """
    venv_path = get_venv_path(repo_root)

    # Basically hydro-playground/.formatting_python_venv/bin/python3
    clangpython = os.path.join(venv_path, "bin", "python3")

    if not os.path.exists(clangpython):
        raise FileNotFoundError(
            "Something went horribly wrong when setting up the venv."
        )

    # The actual install
    cmd = clangpython + " -m pip install clang-format==" + clang_version
    if verbose:
        print("Installing clang-format.")
        print("Running '{}'".format(cmd))

    run = subprocess.run(
        cmd,
        shell=True,
        check=True,
    )

    # Check we succeeded
    clang_path = os.path.join(venv_path, "bin", "clang-format")
    check_clang(clang_path)

    return


def setup_venv(repo_root):
    """
    Set up a virtual env for the formatting tools, if it doesn't exist yet.
    """

    venv_path = get_venv_path(repo_root)

    if not os.path.exists(venv_path):
        print("Python formatting env not found, installing it.")
        os.mkdir(venv_path)

        cmd = "python3 -m venv " + venv_path

        if verbose:
            print("Running '{}'".format(cmd))
        run = subprocess.run(cmd, shell=True, check=True)

        if not os.path.exists(venv_path):
            raise FileNotFoundError("Something went horribly wrong when creating venv.")

        # If the venv didn't exist, neither do the formatting tools. Install them.
        install_black_formatting_tool()
        install_clang_formatting_tool()


def get_black_command(repo_root):
    """
    Get the full command used to invoke the python formatter 'black'.
    First get the path to the 'black' executable. It should be in the
    $REPOSITORY_ROOT/$PYTHON_VENV_DIR/bin/ directory.

    If that directory doesn't exist, we first need to set up a venv
    there and install a specific version of 'black'.
    """

    # Check if venv exists, and create it if it doesn't.
    setup_venv(repo_root)

    venv_path = get_venv_path(repo_root)
    black_path = os.path.join(venv_path, "bin", "black")

    if not os.path.exists(black_path):
        install_black_formatting_tool()

    check_black(black_path)

    black_cmd = black_path
    black_cmd += " --quiet "

    return black_cmd


def get_clang_command(clang_format_file):
    """
    Get the full command used to invoke the clang-format tool.
    First get the path to the 'clang-format' executable. It should be in the
    $REPOSITORY_ROOT/$PYTHON_VENV_DIR/bin/ directory.

    If that directory doesn't exist, or if the executable doesn't exist, we
    first need to set up a venv there and install a specific version of
    'clang-format'.
    """

    # Check if venv exists, and create it if it doesn't.
    setup_venv(repo_root)

    venv_path = get_venv_path(repo_root)
    clang_path = os.path.join(venv_path, "bin", "clang-format")

    if not os.path.exists(clang_path):
        install_clang_formatting_tool()

    check_clang(clang_path)

    clang_cmd = clang_path
    # Format files in-place
    clang_cmd += " -i"
    # Change warnings to errors so I can work with the error code as
    # return values
    clang_cmd += " --Werror"
    # No maximum error limit
    clang_cmd += " --ferror-limit=0"
    # Skip formatting if formatting file isn't found. Better safe than sorry.
    clang_cmd += " --fallback-style=none"
    # Set the file to use
    clang_cmd += " --style=file:" + clang_format_file

    return clang_cmd


def setup_parser(repo_root, clangfile):
    """
    Set up the argument parser.
    """

    parser = argparse.ArgumentParser(
        prog="format.py",
        description="Run 'clang-format' on C/C++ files, run 'black' on Python files.",
        allow_abbrev=True,
    )

    parser.add_argument(
        "-d",
        "--dry-run",
        action="store_true",
        required=False,
        default=False,
        help="Dry run where it only reports the files that have inappropriate formatting",
    )
    parser.add_argument(
        "-s",
        "--style",
        required=False,
        help="Path to a non-default clang-format style file (default is "
        + clangfile
        + ")",
    )
    parser.add_argument(
        "-f",
        "--files",
        required=False,
        nargs="+",
        help="Path to specific file(s) to run the formatter on",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Be verbose about what's going on",
    )

    return parser


def is_pythonfile(filename):
    """
    Does this look like a Python file?
    """

    return filename.endswith(".py")


def is_cppfile(filename):
    """
    Does this look like a C/C++ file?
    """

    suffixes = [".h", ".hpp", ".cpph", ".c", ".cpp", ".cu", ".cuh"]

    for suf in suffixes:
        if filename.endswith(suf):
            # Don't format jinja template files. clang-format can't handle those.
            if filename.endswith(".template" + suf):
                if verbose:
                    print(
                        "File '{}' looks like a jinja template file. Skipping.".format(
                            filename
                        )
                    )
                return False
            else:
                return True

    return False


def run_python_formatter(filename, dry_run, repo_root):
    """
    Run the black formatter on the provided file.
    If dry_run is True, don't modify the file, just report
    whether it would modify the file.

    Returns 1 if file was modified or if it could be modified
    and the dry_run flag was provided, or 0 otherwise.
    """

    black_cmd = get_black_command(repo_root)

    # First run in dry-run mode to accurately count whether we
    # need to/do format through the return code.
    cmd = black_cmd + " --check "

    if verbose:
        print("Checking '{}' using '{}'".format(filename, cmd))
    run = subprocess.run(cmd + filename, stdout=subprocess.DEVNULL, shell=True)

    returncode = run.returncode
    if returncode != 0:
        returncode = 1

    if dry_run:
        return returncode

    # If not dry run: Now actually format the file
    if verbose:
        print("Running '{} {}'".format(black_cmd, filename))
    run = subprocess.run(
        black_cmd + " " + filename, stdout=subprocess.DEVNULL, shell=True
    )

    return returncode


def run_clang_formatter(filename, dry_run, clang_format_file):
    """
    Run the clang formatter on the provided file.
    If dry_run is True, don't modify the file, just report
    whether it would modify the file.

    Returns 1 if file was modified or if it could be modified
    and the dry_run flag was provided, or 0 otherwise
    """

    clang_cmd = get_clang_command(clang_format_file)

    # First run in dry-run mode to accurately count whether we
    # need to/do format through the return code.
    cmd = clang_cmd + " --dry-run "
    if verbose:
        print("Checking '{}' using '{}'".format(filename, cmd))
    # Pipe sterr to /dev/null too here. In dry run mode, every line that needs
    # to be changed will otherwise be printed to stderr.
    run = subprocess.run(
        cmd + filename,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=True,
    )

    returncode = run.returncode
    if returncode != 0:
        returncode = 1

    if dry_run:
        return returncode

    if verbose:
        print("Running '{} {}'".format(clang_cmd, filename))
    # Pipe sterr to /dev/null too here. In dry run mode, every line that needs
    # to be changed will otherwise be printed to stderr.
    run = subprocess.run(
        clang_cmd + " " + filename,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        shell=True,
    )

    return returncode


if __name__ == "__main__":
    if git_is_available():
        repo_root = get_repo_root()
    else:
        repo_root = repo_rootdir
    clang_format_file = get_default_clang_format(repo_root)

    parser = setup_parser(repo_root, clang_format_file)
    args = parser.parse_args()

    # Are we using default clang file?
    if args.style is not None:
        clang_format_file = args.style
        if not os.path.exists(clang_format_file):
            raise FileNotFoundError(
                "Provided clang-format file '{}' doesn't exist.".format(
                    clang_format_file
                )
            )

    verbose = args.verbose

    if verbose:
        print("Using clang-format file '{}'.".format(clang_format_file))

    # Is this a dry run?
    dry_run = args.dry_run
    if dry_run and verbose:
        print("Starting a dry run")

    # Which files are we formatting?
    filelist = []

    # Check first if file list was given by user
    if args.files is not None:
        if os.path.isdir(args.files[0]):
            if verbose:
                print("Formatting all files in this directory.")
            for root, dirs, files in os.walk(repo_root):
                # Check if 'python_venv_dir' is in the list of subdirectories
                if python_venv_dir in dirs:
                    # If it's present, remove it from the list of subdirectories to skip it in the iteration.
                    dirs.remove(python_venv_dir)
                if "submodules" in dirs:
                    dirs.remove("submodules")
                for filename in files:
                    f = os.path.join(root, filename)
                    if not (is_pythonfile(f) or is_cppfile(f)):
                        if verbose:
                            print(
                                f"File '{f}' doesn't look like a Python nor C/C++ file to me, skipping it."
                            )
                        continue
                    filelist.append(f)
        else:
            if verbose:
                print("File list provided. Formatting only provided list.")

            for f in args.files:
                if not os.path.exists(f):
                    if verbose:
                        print(f"Didn't find file '{f}', ignoring it")
                    continue
                if not (is_pythonfile(f) or is_cppfile(f)):
                    if verbose:
                        print(
                            f"File '{f}' doesn't look like a Python nor C/C++ file to me, skipping it."
                        )
                    continue
                filelist.append(f)

    else:
        # get all files traced by git
        cmd = f"git ls-files {repo_root}"

        gitls = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
        )
        gitstdout = gitls.stdout
        if isinstance(gitstdout, bytes):
            gitstdout = gitstdout.decode("utf8")
        allfiles = gitstdout.split("\n")
        for f in allfiles:
            if is_pythonfile(f) or is_cppfile(f):
                filelist.append(f)

    if len(filelist) == 0:
        print("Found no files to be formatted. Exiting.")
        sys.exit(0)

    # Keep track whether any file was or could be modified.
    nmodified = 0
    formatted_files = []
    for f in filelist:
        if is_pythonfile(f):
            return_code = run_python_formatter(f, dry_run, repo_root)
            nmodified += return_code
            if return_code != 0:
                formatted_files.append(f)
        elif is_cppfile(f):
            return_code = run_clang_formatter(f, dry_run, clang_format_file)
            nmodified += return_code
            if return_code != 0:
                formatted_files.append(f)
        else:
            raise ValueError("How did we get here???")

    # Report what you did
    if dry_run:
        print(
            "Dry run:",
            nmodified,
            "files can be formatted (run with --verbose to see file list)",
        )
        prefix = "Can be formatted:"
    else:
        print("Formatted", nmodified, "files.")
        prefix = "Has been formatted:"

    if verbose:
        formatted_files.sort()
        for f in formatted_files:
            print(prefix, f)

    # "fail" if we can/did format any files, so CI can pick this
    # up as an "error".
    sys.exit(nmodified)
