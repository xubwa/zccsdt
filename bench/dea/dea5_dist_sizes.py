'''The sizes that decide how EOM-DEA with complete 5p3h is distributed over
MPI ranks (cfour/xzccsdt, docs/native.md section 21), from the labels of a
Hamiltonian file alone (.npz or a block file .h5); nothing is computed and
no quintuple list is made (only counted).

usage: python dea5_dist_sizes.py <file.npz|file.h5> [target ...]'''
import ctypes
import sys
from math import comb

import numpy as np


def labels(path):
    if path.endswith('.h5'):
        import h5py
        with h5py.File(path, 'r') as f:
            return int(f.attrs['ncore']), np.asarray(f['omega2'], dtype=np.int64)
    z = np.load(path)
    return int(z['ncore']), np.asarray(z['omega2'], dtype=np.int64)


def main():
    from zccsdt.cc import dea_sector as D, t4engine_sym as S
    lib = S._lib
    lib.zc_t5s_quints.restype = ctypes.c_int64
    lib.zc_t4s_quads.restype = ctypes.c_int64
    no, lab = labels(sys.argv[1])
    targets = [int(x) for x in sys.argv[2:]] or [0, 2]
    lo, lv = np.sort(lab[:no]), np.sort(lab[no:])
    lmin = min(lo.min(), lv.min())
    u = np.unique(lab)
    step = int(np.gcd.reduce(np.diff(u))) if len(u) > 1 else 1
    ko, kv = (lo - lmin) // step, (lv - lmin) // step
    ncls = int(max(ko.max(), kv.max())) + 1
    lay = S.Layout(ko, kv, ncls)
    nv = len(lv)
    kv32 = np.ascontiguousarray(kv, dtype=np.int32)
    n5c = 5 * ncls - 4
    nA5 = np.array([lib.zc_t5s_quints(ctypes.c_int(nv), kv32.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(M), None)
                    for M in range(n5c)], dtype=np.int64)
    nA4 = np.array([lib.zc_t4s_quads(ctypes.c_int(nv), kv32.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(A), None)
                    for A in range(4 * ncls - 3)], dtype=np.int64)
    nO, nV, nQ, nK = lay.nO, lay.nV, lay.nQ, lay.nK
    GB = 8e-9
    print('%s: %d + %d spinors, %d classes; C(nv, 5) = %.3e, C(nv, 4) = %.3e' % (sys.argv[1], no, nv, ncls,
                                                                              comb(nv, 5), comb(nv, 4)))
    print('R13 (Xv with the rows of T3; Xv2 the same size) %.2f GB' % (lay.r13size * GB))
    for tgt in targets:
        sh = (tgt - 2 * lmin) // step
        sec = D.Sector(lay, sh)
        n3c = lay.n3
        tri = lay.o3
        rows5 = {}
        for M in range(n5c):
            c = M - sh
            if 0 <= c < n3c and nA5[M]:
                rows5[M] = np.asarray(lay.O3cls[c], dtype=np.int64)
        size5 = sum(len(r) * nA5[M] for M, r in rows5.items())
        nq5 = sum(nA5[M] for M, r in rows5.items() if len(r))
        rowsz = np.zeros(len(tri), dtype=np.int64)
        for M, r in rows5.items():
            rowsz[r] = nA5[M]
        x4 = int(sum(sec.n13[sec.blk4[P]] for P in range(len(lay.o2)) if sec.blk4[P] >= 0))
        xv5 = np.zeros(no)
        for k in range(no):
            for ph in range(ncls):
                A = sh + ko[k] + ph
                if nV[ph] and 0 <= A < len(nA4):
                    xv5[k] += nV[ph] * nA4[A]
        keys = set()
        for k in range(no):
            for l in range(no):
                if l == k:
                    continue
                M4 = sec.blk4[sec.pair[l, k]]
                if M4 < 0:
                    continue
                for ph in range(ncls):
                    if sec.off13[M4, ph] < 0:
                        continue
                    for mu in range(ncls):
                        fe = mu + ko[l] - ph
                        if nO[mu] and 0 <= fe < ncls and nV[fe]:
                            keys.add((k, mu, fe, M4 - ph))
        z4 = sum(int(nO[mu] * nV[fe] * nK[K]) for _, mu, fe, K in keys)
        n14 = n23 = 0
        for M, r in rows5.items():
            a = sum(int(nV[p] * nA4[M - p]) for p in range(ncls) if nV[p] and 0 <= M - p < len(nA4))
            b = sum(int(nQ[q] * nK[M - q]) for q in range(lay.n2) if nQ[q] and 0 <= M - q < n3c)
            n14, n23 = max(n14, a), max(n23, b)
        ring = []
        for P, (uu, ww) in enumerate(lay.o2):
            tot = 0
            for m in range(no):
                if m in (uu, ww):
                    continue
                t = tuple(sorted((m, uu, ww)))
                I = int(np.flatnonzero((tri == t).all(axis=1))[0])
                tot += rowsz[I]
            ring.append(tot)
        zrows = []
        for z in range(no):
            zrows.append(sum(rowsz[I] for I in range(len(tri)) if z in tri[I]))
        xr = 0
        for P in range(len(lay.o2)):
            for m in range(no):
                K = sh + lay.kO2[P] - ko[m]
                if 0 <= K < n3c:
                    xr += int(nK[K])
        pp = sum(len(r) * sum(int(nQ[q] * nQ[q] * nK[M - q]) for q in range(lay.n2)
                              if nQ[q] and 0 <= M - q < n3c and nK[M - q]) for M, r in rows5.items())
        print('2 Omega = %d (shift %d):' % (tgt, sh))
        print('  r4 %d = %.2f GB; r4 on (h | cde) %.2f GB; r5 %.4e = %.1f GB (parity g about %.1f GB)'
              % (sec.size4, sec.size4 * GB, x4 * GB, size5, size5 * GB, size5 * GB / 2))
        print('  a row of r5: largest %.2f GB, mean %.2f GB over %d triples; blocks of r5 %d'
              % (rowsz.max() * GB, rowsz[rowsz > 0].mean() * GB, (rowsz > 0).sum(), len(rows5)))
        print('  quintuple lists of the blocks %.1f GB (int16 x 5)' % (nq5 * 10e-9))
        print('  Xv5 %.1f GB in all, per k %.2f .. %.2f GB; Z4 %.2f GB; XoR5 %.2f GB'
              % (xv5.sum() * GB, xv5.min() * GB, xv5.max() * GB, z4 * GB, xr * GB))
        print('  a row on (a | bcde) at most %.2f GB, on (ab | cde) %.2f GB' % (n14 * GB, n23 * GB))
        print('  the rows (m, u, w) of a spectator pair: %.2f .. %.2f GB; the rows (l, n, z) of an index z: %.1f .. %.1f GB'
              % (min(ring) * GB, max(ring) * GB, min(zrows) * GB, max(zrows) * GB))
        print('  pp ladder %.3e multiply-adds' % pp, flush=True)


if __name__ == '__main__':
    main()
