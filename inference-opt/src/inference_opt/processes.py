"""Bounded POSIX adapter execution; timeout kills only its newly created group.

Stopping this local group does not cancel a detached or remotely submitted job.
The caller must retain submission_unknown and reconcile before restoring state.
"""

import math
import os
import signal
import subprocess


class ProcessGroupTimeout(subprocess.TimeoutExpired):
    def __init__(self, command, timeout, cleanup, *, output=None, stderr=None):
        super().__init__(command, timeout, output=output, stderr=stderr)
        self.cleanup = cleanup


def run_adapter(command, *, input, timeout, cleanup_timeout=2):
    for duration in (timeout, cleanup_timeout):
        if type(duration) not in (int, float) or not math.isfinite(duration) or duration <= 0:
            raise ValueError("adapter and cleanup timeouts must be finite positive numbers")
    if os.name != "posix":
        raise NotImplementedError("adapter execution requires POSIX process-group supervision")
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(input=input, timeout=timeout)
    except BaseException as exc:
        cleanup = {
            "process_group": process.pid,
            "signal": "SIGKILL",
            "signal_sent": False,
            "leader_reaped": False,
            "requires_reconciliation": True,
        }
        try:
            os.killpg(process.pid, signal.SIGKILL)
            cleanup["signal_sent"] = True
        except ProcessLookupError:
            cleanup["group_already_absent"] = True
        except OSError as error:
            cleanup["signal_error"] = str(error)
        # A detached child can retain a pipe even after the group is killed.
        # Bound the drain/reap rather than waiting indefinitely on that pipe.
        try:
            stdout, stderr = process.communicate(timeout=cleanup_timeout)
        except subprocess.TimeoutExpired:
            stdout, stderr = getattr(exc, "output", None), getattr(exc, "stderr", None)
            cleanup["pipe_drain_timed_out"] = True
            for pipe in (process.stdin, process.stdout, process.stderr):
                if pipe is not None:
                    pipe.close()
        try:
            process.wait(timeout=cleanup_timeout)
            cleanup["leader_reaped"] = True
        except subprocess.TimeoutExpired:
            cleanup["leader_reap_timed_out"] = True
        if isinstance(exc, subprocess.TimeoutExpired):
            raise ProcessGroupTimeout(command, timeout, cleanup, output=stdout, stderr=stderr) from exc
        exc.cleanup = cleanup
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
