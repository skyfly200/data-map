"""Prefix every printed line with a wall-clock timestamp.

A long fetch or enrichment can sit silently on one blocking call, and without a
clock on each line there's no way to tell "still working" from "frozen". Calling
``enable_timestamps()`` once at a stage's entry point wraps stdout/stderr so every
line comes out as ``[HH:MM:SS] …`` — the gap between two timestamps is exactly how
long that step took. Set ``LOG_TIMESTAMPS=0`` to turn it off.
"""

import os
import sys
import threading
import time

_ENABLED = False


class _TimestampWriter:
    """A stdout/stderr wrapper that stamps the start of every line.

    Writes are locked so lines from different threads (the parallel fetch and
    enrichment workers) never interleave in the middle of a line."""

    def __init__(self, stream):
        self._stream = stream
        self._lock = threading.RLock()
        self._at_line_start = True

    def write(self, text):
        if not text:
            return 0
        with self._lock:
            for line in text.splitlines(keepends=True):
                if self._at_line_start:
                    self._stream.write(time.strftime('[%H:%M:%S] '))
                self._stream.write(line)
                self._at_line_start = line.endswith('\n') or line.endswith('\r')
        return len(text)

    def flush(self):
        self._stream.flush()

    def __getattr__(self, name):
        # Delegate isatty(), fileno(), encoding, … to the wrapped stream.
        return getattr(self._stream, name)


def enable_timestamps(force=False):
    """Wrap stdout/stderr so every line is timestamped. Idempotent; a no-op when
    LOG_TIMESTAMPS is 0/false/off unless ``force`` is set."""
    global _ENABLED
    if _ENABLED:
        return
    value = os.getenv('LOG_TIMESTAMPS', '1').strip().lower()
    if not force and value in {'0', 'false', 'no', 'off'}:
        return
    sys.stdout = _TimestampWriter(sys.stdout)
    sys.stderr = _TimestampWriter(sys.stderr)
    _ENABLED = True
