"""Implementation guide for the future Windows process owner.

This module is intentionally documentation-only until Windows is a supported
KMS2 deployment target. The implementation must satisfy the ``ProcessOwner``
and ``OwnedProcess`` protocols from ``ownership.py`` without changing
``ManagedLlamaServer`` or ``LocalModelRuntime``.

Implementation checklist:

1. Add ``WindowsProcessOwner`` with:

   ``async def spawn(command: list[str]) -> OwnedProcess``

   Create a Windows Job Object before launching the model command. Configure
   the job with ``JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`` so every descendant is
   terminated when the owner closes the job. Launch the command without a
   shell, place the process in the job, and return a ``WindowsOwnedProcess``.
   Preserve the existing private environment and command-vector contract only
   if the Windows bootstrap needs it; do not pass POSIX signal assumptions
   through this module.

2. Add ``WindowsOwnedProcess`` implementing:

   - ``returncode``: the direct process exit code, or ``None`` while running.
   - ``async wait() -> int``: await the direct process handle without blocking
     the event loop; use an executor or an overlapped wait mechanism rather
     than polling indefinitely.
   - ``async stop(terminate_timeout: float) -> None``: request graceful
     termination of the owned process, wait for the direct process for the
     configured timeout, then terminate the Job Object if it remains alive.
     Always wait for the direct process to exit before clearing the stored
     process/job handles. Repeated calls must be no-ops after reaping.

3. Preserve the ownership invariants:

   - No existing endpoint is adopted.
   - The owner controls the complete process tree, not only the direct child.
   - ``stop()`` returns only after descendants are no longer running.
   - A startup/readiness failure can safely call ``stop()``.
   - Cancellation during shutdown must not clear handles before the process is
     reaped. If a wait is interrupted, continue cleanup or preserve the handle
     so a later close can finish it.
   - Translate Windows wait/termination failures at this boundary; do not add
     Windows conditionals to ``ManagedLlamaServer`` or the runtime state
     machine.

4. Replace the unsupported branch in ``ownership.create_process_owner()``:

   - Select ``WindowsProcessOwner`` when ``os.name == 'nt'``.
   - Keep the current exact unsupported-platform error for other platforms
     until they have an explicit owner implementation.
   - Do not add an external/shared-server fallback.

5. Add Windows-only contract tests alongside the existing POSIX ownership
   tests:

   - A direct child that ignores graceful termination is force-terminated.
   - A spawned grandchild is also terminated when the owner stops.
   - Repeated stop calls are safe and leave no live process tree.
   - Parent-side cancellation closes the Job Object before the application
     database closes.
   - The factory selects ``WindowsProcessOwner`` only on Windows.

Do not use ``os.killpg``, ``start_new_session``, POSIX signals, ``prctl``,
``/proc``, or ``os.name == 'posix'`` in this backend. Keep all platform-specific
logic here; the server readiness and runtime residency layers remain shared.
"""

# The first implementation should be added below this guide only when Windows
# process ownership is an actual supported deployment target. Until then,
# ownership.create_process_owner() intentionally fails on Windows so an
# unsupported backend cannot silently start an unmanaged model process.
