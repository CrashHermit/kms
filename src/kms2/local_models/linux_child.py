"""Launch a Linux local model child with parent-death cleanup."""

import ctypes
import os
import signal

from kms2.local_models.posix_child import command_from_arguments


def _set_parent_death_signal() -> None:
    """Terminate this child when its KMS2 parent dies."""
    expected_parent_pid = int(
        os.environ.pop('KMS2_PARENT_PID', str(os.getppid()))
    )
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if os.getppid() != expected_parent_pid:
        os.kill(os.getpid(), signal.SIGTERM)


def main() -> None:
    """Set Linux parent-death cleanup, then execute the requested command."""
    command = command_from_arguments()
    _set_parent_death_signal()
    os.execvp(command[0], command)


if __name__ == '__main__':
    main()
