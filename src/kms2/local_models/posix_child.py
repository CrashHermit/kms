"""Launch a POSIX local model child."""

import os
import sys


def command_from_arguments() -> list[str]:
    """Return the command following this bootstrap's separator."""
    try:
        separator = sys.argv.index('--')
    except ValueError as error:
        raise SystemExit('missing child command separator') from error
    command = sys.argv[separator + 1 :]
    if not command:
        raise SystemExit('missing child command')
    return command


def main() -> None:
    """Replace this bootstrap with the requested model-server command."""
    command = command_from_arguments()
    os.execvp(command[0], command)


if __name__ == '__main__':
    main()
