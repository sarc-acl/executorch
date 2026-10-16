"""HOLD and device-lock boundaries for resumable local sessions."""

import fcntl
from pathlib import Path
import time


def hold_point(lock, artifacts, hold_name="HOLD", poll_seconds=60):
    marker = Path(artifacts) / hold_name
    held = Path(artifacts) / ("HELD" + hold_name[4:])
    acknowledged = False
    while True:
        while marker.exists():
            fcntl.flock(lock, fcntl.LOCK_UN)
            held.write_text("HELD before the next resumable run\n")
            acknowledged = True
            time.sleep(poll_seconds)
        if acknowledged:
            held.unlink(missing_ok=True)
        deadline = time.monotonic() + 900
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if marker.exists():
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("inherited device lock busy for 900 seconds")
                time.sleep(1)
        if not marker.exists():
            return
