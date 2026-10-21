'''Sizes, operation counts and the memory plan of EOM-DEA with complete 5p3h
on the sector (zccsdt.cc.dea5_sector) from the labels of a Hamiltonian file
alone (nothing is computed).

usage: python dea5_sizes.py <file.npz> [target ...]'''
import sys

import numpy as np


def main():
    from zccsdt.cc import dea_sector as D, dea5_sector as D5, t4engine_sym as S
    z = np.load(sys.argv[1])
    targets = [int(x) for x in sys.argv[2:]] or [0, 2]
    no = int(z['ncore'])
    lab = np.asarray(z['omega2'], dtype=np.int64)
    lo, lv = np.sort(lab[:no]), np.sort(lab[no:])
    lmin = min(lo.min(), lv.min())
    u = np.unique(lab)
    step = int(np.gcd.reduce(np.diff(u))) if len(u) > 1 else 1
    ko, kv = (lo - lmin) // step, (lv - lmin) // step
    lay = S.Layout(ko, kv, int(max(ko.max(), kv.max())) + 1)
    print('%s: %d + %d spinors, %d classes' % (sys.argv[1], no, len(lv), lay.ncls))
    for tgt in targets:
        d = tgt - 2 * lmin
        sh = d // step
        sec = D.Sector(lay, sh)
        s5 = D5.Sector5(sec)
        nQ, nK = lay.nQ, lay.nK
        pp = rows = ring = 0
        for M in s5.blocks5:
            nr = len(s5.rows5[M])
            pp += nr * sum(int(nQ[q] * nQ[q] * nK[M - q]) for q in range(lay.n2) if s5.off23[M, q] >= 0)
        fixed, per = D5.memory5(s5, 8, 1 << 30)
        print('2 Omega = %d: 2p %d, 3p1h %d, 4p2h %d, 5p3h %d (%.2f GB); pp ladder %.2e multiply-adds; '
              'memory plan %.1f GB fixed + %.1f GB per sigma vector'
              % (tgt, sec.n2, sec.size3, sec.size4, s5.size5, s5.size5 * 8e-9, pp, fixed / 1e9, per / 1e9),
              flush=True)


if __name__ == '__main__':
    main()
