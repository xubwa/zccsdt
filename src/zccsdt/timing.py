'''Accumulating wall-clock timers for per-term profiling.'''

import time
from collections import defaultdict
from contextlib import contextmanager

_totals = defaultdict(float)
_counts = defaultdict(int)


@contextmanager
def timer(name):
    t0 = time.perf_counter()
    try:
        yield
    finally:
        _totals[name] += time.perf_counter() - t0
        _counts[name] += 1


def reset():
    _totals.clear()
    _counts.clear()


def report(per=1):
    '''Return [(name, seconds / per, calls / per)] sorted by time.'''
    return sorted(((k, v / per, _counts[k] / per) for k, v in _totals.items()),
                  key=lambda x: -x[1])
