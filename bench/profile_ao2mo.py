'''Spinor AO-to-MO transformations side by side, on the two kinds of
integrals a spinor CCSD needs: (i p|q r) with i occupied and p, q, r all
MOs, and (a b|c d) with four virtuals.

  stock  PySCF's nrr_outcore (no AO permutational symmetry, HDF5 scratch):
         the occupied block only
  stockmem  the C kernels of nrr_outcore driven in memory (still no AO
         symmetry, nothing written): what symmetry alone is worth against
         fast; skipped when its half transform would take more than 16 GB
  fast   socutils' nrr_fast (lib/ao2mo: both passes in C, s4 AO symmetry,
         the cheaper index order); the second pass in batches of rows
  slab   zccsdt.zccsd.ao2mo (its own bra pass - real matrix products over
         blocks of AO pairs -, the ket pass of nrr_fast; the first index in
         batches and the AO integrals in ranges of shell pairs, planned
         within ZC_MAXMEM MB, default 24000; 'fill': the part of 'bra'
         that the AO integrals take)

usage: python bench/profile_ao2mo.py MOLECULE BASIS [METHODS] [BLOCKS]
  METHODS: comma separated from stock,fast,slab (default: all)
  BLOCKS:  comma separated from occ,vir (default: both)
X2C-1e spinors from PySCF (x2c.UHF), all electrons.  nrr_fast needs the
socutils of ~/apps/socutils-git on PYTHONPATH (~/apps/socutils-git-env).
The integrals are thrown away batch by batch; the rows of the first orbital
of each block are kept and compared between the methods.
'''
import os
import sys
import time

import numpy as np
from pyscf import gto, lib, x2c
from pyscf.ao2mo import nrr_outcore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_iter import GEOM, SPIN, MemWatch                          # noqa: E402

from zccsdt import timing                                              # noqa: E402
from zccsdt.zccsd import ao2mo as slab                                 # noqa: E402

BATCH_BYTES = 4 << 30


def run(name, f):
    watch = MemWatch()
    watch.start()
    t0 = time.perf_counter()
    first, stages = f()
    sec = time.perf_counter() - t0
    watch.stop()
    print('  %-22s %8.1f s   peak anonymous %5.2f GB   %s'
          % (name, sec, watch.anon / 1e6,
             '  '.join('%s %.4g' % kv for kv in stages)), flush=True)
    return first


def main():
    mol_name, basis = sys.argv[1:3]
    methods = sys.argv[3].split(',') if len(sys.argv) > 3 else ['stock', 'fast', 'slab']
    blocks = sys.argv[4].split(',') if len(sys.argv) > 4 else ['occ', 'vir']
    mol = gto.M(atom=GEOM[mol_name], basis=basis, spin=SPIN.get(mol_name, 0), verbose=0)
    mf = x2c.UHF(mol).run(conv_tol=1e-10)
    mo = mf.mo_coeff
    nocc, nmo, nao = mol.nelectron, mo.shape[1], mol.nao_nr()
    nvir = nmo - nocc
    Co, Cv = mo[:, :nocc], mo[:, nocc:]
    print('%s/%s  nocc=%d nvir=%d nao=%d  threads %d' % (mol_name, basis, nocc, nvir, nao,
                                                       lib.num_threads()), flush=True)
    if 'fast' in methods:
        from socutils.lib.ao2mo import nrr_fast

    # (bra pair, ket range, rows of one first-index orbital)
    cases = {'occ': ((Co, mo), mo, nmo), 'vir': ((Cv, Cv), Cv, nvir)}
    for blk in blocks:
        (A, B), K, nq = cases[blk]
        nfirst = A.shape[1]
        print('%s block: (%d x %d | %d x %d), %.1f GB as complex numbers'
              % (blk, nfirst, nq, nq, nq, 16e-9 * nfirst * nq ** 3), flush=True)
        firsts = {}

        def stock():
            t0 = time.perf_counter()
            g = nrr_outcore.general_iofree(mol, (A, B, K, K), intor='int2e_sph',
                                           motype='j-spinor', verbose=0)
            return np.array(g[:nq]), [('all', time.perf_counter() - t0)]

        def stockmem():
            from pyscf.ao2mo import _ao2mo
            ca, cb = mol.sph2spinor_coeff()
            nA, nB = A.shape[1], B.shape[1]
            moa = np.asarray(np.hstack((ca.dot(A), ca.dot(B))), order='F')
            mob = np.asarray(np.hstack((cb.dot(A), cb.dot(B))), order='F')
            tao = np.asarray(mol.tmap(), dtype=np.int32)
            t0 = time.perf_counter()
            half = nrr_outcore.r_e1('int2e_sph', moa, mob, (0, nA, nA, nA + nB),
                                    (0, mol.nbas ** 2, nao * nao), mol._atm, mol._bas, mol._env,
                                    tao, 's1')[0]             # (nao^2, nij)
            vin = lib.transpose(half)
            del half
            t1 = time.perf_counter()
            ka = np.asarray(ca.dot(K), dtype=np.complex128, order='F')
            kb = np.asarray(cb.dot(K), dtype=np.complex128, order='F')
            step = max(nq, int(BATCH_BYTES // (32 * nq * nq)) // nq * nq)
            first = None
            for r0 in range(0, len(vin), step):
                g = _ao2mo.r_e2(vin[r0:r0 + step], ka, (0, nq, 0, nq), tao, mol.ao_loc, 's1')
                g += _ao2mo.r_e2(vin[r0:r0 + step], kb, (0, nq, 0, nq), tao, mol.ao_loc, 's1')
                if first is None:
                    first = np.array(g[:nq])
                del g
            return first, [('bra', t1 - t0), ('ket', time.perf_counter() - t1)]

        def fast():
            t0 = time.perf_counter()
            vin = nrr_fast.bra_half(mol, A, B)                 # (rows, AO pairs)
            t1 = time.perf_counter()
            step = max(nq, int(BATCH_BYTES // (16 * nq * nq)) // nq * nq)
            first = None
            for r0 in range(0, len(vin), step):
                g = nrr_fast.ket_transform(mol, vin[r0:r0 + step], K, K)
                if first is None:
                    first = np.array(g[:nq])
                del g
            return first, [('bra', t1 - t0), ('ket', time.perf_counter() - t1)]

        def slabs():
            from zccsdt import memory
            timing.reset()
            Ca, Cb = slab.spin_components(mol, mo)
            f0 = (0, nocc) if blk == 'occ' else (nocc, nmo)
            rest = None if blk == 'occ' else (nocc, nmo)
            plan = slab.plan_slabs(mol, nfirst, nq,
                                   memory.budget(float(os.environ.get('ZC_MAXMEM', 24000))))
            first = None
            for i0, i1, g in slab.slabs(mol, Ca, Cb, f0, rest, plan):
                if first is None:
                    first = np.array(g[0]).reshape(nq, nq * nq)
            return first, ([(k[6:], t) for k, t, _ in timing.report() if k.startswith('ao2mo.')]
                           + [('orbitals per pass', plan.nbra), ('per slab', plan.nket),
                              ('AO pairs per range', plan.nkl)])

        for m, f in (('stock', stock), ('stockmem', stockmem), ('fast', fast), ('slab', slabs)):
            if m not in methods or (m == 'stock' and blk == 'vir'):
                continue
            if m == 'stockmem' and 48 * nao * nao * nfirst * nq > 16e9:
                continue
            label = {'stock': 'nrr_outcore (PySCF)', 'stockmem': 'its kernels in memory',
                     'fast': 'nrr_fast (socutils)', 'slab': 'slabs (zccsdt)'}[m]
            firsts[m] = run(label, f)
        ref = firsts.get('fast', firsts.get('slab'))
        for m, g in firsts.items():
            if g is not ref:
                print('  max |%s - %s| on the first orbital: %.1e'
                      % (m, 'fast' if 'fast' in firsts else 'slab', abs(g - ref).max()),
                      flush=True)


if __name__ == '__main__':
    main()
