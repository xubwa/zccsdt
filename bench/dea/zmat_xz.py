'''Replace the section %xzccsdt of the ZMAT of a run directory by the given
lines (none: the section goes).

usage: python bench/dea/zmat_xz.py <rundir> ['option value' ...]'''
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from native_check import section  # noqa: E402

if __name__ == '__main__':
    section(os.path.join(sys.argv[1], 'ZMAT'), sys.argv[2:])
    print(open(os.path.join(sys.argv[1], 'ZMAT')).read().split('%xzdea')[1])
