"""Optional stack traces for slow native action calls; no execution timeout."""

from __future__ import annotations

import faulthandler
from functools import wraps
import os
from pathlib import Path
import time


def trace_slow_action(function):
    @wraps(function)
    def traced(task_env, *args, **kwargs):
        interval = float(os.environ.get("RMBENCH_SLOW_ACTION_TRACE_SEC", "0"))
        output = os.environ.get("RMBENCH_OUTPUT_ROOT", "")
        if interval <= 0 or not output:
            return function(task_env, *args, **kwargs)

        root = Path(output)
        root.mkdir(parents=True, exist_ok=True)
        with (root / f"action_stacks_{os.getpid()}.log").open("a") as stream:
            start = time.monotonic()
            stream.write(
                f"action_start step={task_env.take_action_cnt} "
                f"action_type={kwargs.get('action_type', 'qpos')} "
                f"active_arm={kwargs.get('active_arm')}\n"
            )
            stream.flush()
            # The C watchdog also dumps the Python caller of a blocked native
            # call. It never interrupts a motion or changes its result.
            faulthandler.dump_traceback_later(interval, repeat=True, file=stream)
            try:
                return function(task_env, *args, **kwargs)
            finally:
                faulthandler.cancel_dump_traceback_later()
                stream.write(
                    f"action_return step={task_env.take_action_cnt} "
                    f"elapsed_sec={time.monotonic() - start:.3f}\n"
                )

    return traced
