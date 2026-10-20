'''Set joda's flags (IFLAGS of JOBARC) in a run directory of the native
program, e.g. ESTATE_MAXCYC (97) for a timing run of a few iterations, or the
memory (36, words), without running xjoda again (native_rundir.py has put
SCFENEG and NUMDROPA into JOBARC after xjoda).

usage: python bench/dea/jobarc_flags.py <dir> [k=value ...]     (k: the number of the flag from 1)
'''
import sys

import numpy as np


def main():
    from zccsdt import cfour
    arc = cfour.JobArc(sys.argv[1])
    fl = np.array(arc.get('IFLAGS'))
    for a in sys.argv[2:]:
        k, v = a.split('=')
        fl[int(k) - 1] = int(v)
    if len(sys.argv) > 2:
        arc.put('IFLAGS', fl.astype('<i8'))
    fl = arc.get('IFLAGS')
    print('IFLAGS 2 (CALC) %d, 4 (CC_CONV) %d, 36 (memory, words) %d, 97 (ESTATE_MAXCYC) %d, 98 (ESTATE_CONV) %d'
          % (fl[1], fl[3], fl[35], fl[96], fl[97]))


if __name__ == '__main__':
    main()
