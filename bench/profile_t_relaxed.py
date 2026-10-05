'''The relaxed one-particle density of the spinor CCSD(T) of zccsdt.zccsd
.DirectZCCSD, phase by phase: (T) with the sources of Lambda and the
densities of the orbital gradient (zccsd.triples.kernel, terms='gradient')
next to the (T) energy alone, the Lambda equations, the orbital gradient
(zccsd.orbgrad, zccsd.orbgrad_t) and the Z-vector equations (zccsd.zvector):
wall times with their timers and peak memory, and the dipole moment.  For the
comparison with CFOUR's xx2cccsd (CALC=CCSD(T), PROPS=FIRST_ORDER), xx2chbar,
xx2clcc, xx2cccdenocbp, xx2czvec, xsox2cprop (bench/cfour/slurm_x2ccc.sh).

usage: python bench/profile_t_relaxed.py MOLECULE BASIS
  MOLECULE: h2o | n2 | c6h6 | ... (the geometries of bench/profile_iter.py)
X2C-1e spinors from PySCF (x2c.UHF), all electrons.
environment: ZC_MAXMEM: max_memory in MB (default 24000); ZC_KRAMERS: 0 or 1:
             DirectZCCSD.kramers; ZC_CONV: the convergence of the CCSD energy
             (default 1e-9; the amplitudes and lambda 100 times that, the
             Z-vector equations 10 times); ZC_SCF_CONV, ZC_SCF_GRAD: conv_tol
             and conv_tol_grad of the SCF (default 1e-12 and 1e-8: PySCF's
             default for the orbital gradient, sqrt(conv_tol), leaves 1e-6,
             and a relaxed property has that error in first order);
             ZC_CCSD_CYCLES, ZC_LAMBDA_CYCLES: at most so many cycles (the
             times of the other phases do not depend on the convergence;
             the dipole then means nothing); ZC_T_ENERGY=0: without the run
             of the (T) energy alone.
'''
import os
import resource
import sys
import time

import numpy as np
from pyscf import gto, x2c

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_iter import GEOM, SPIN, MemWatch                           # noqa: E402

from zccsdt import blas, timing                                         # noqa: E402
from zccsdt.zccsd import DirectZCCSD, rdm, triples, zvector             # noqa: E402
from zccsdt.zccsd import kernels as K                                   # noqa: E402


def timers(prefix, wall, least=0.0):
    for name, w, calls in timing.report():
        if name.startswith(prefix) and w >= least * wall:
            print('    %-14s %9.1f s  %5.1f %%  (%d calls)'
                  % (name[len(prefix):], w, 100 * w / wall, calls))


def main():
    mol_name, basis = sys.argv[1:3]
    maxmem = float(os.environ.get('ZC_MAXMEM', 24000))
    conv = float(os.environ.get('ZC_CONV', 1e-9))
    mol = gto.M(atom=GEOM[mol_name], basis=basis, spin=SPIN.get(mol_name, 0), verbose=0,
                max_memory=maxmem)
    t0 = time.perf_counter()
    mf = x2c.UHF(mol)
    mf.conv_tol = float(os.environ.get('ZC_SCF_CONV', 1e-12))
    mf.conv_tol_grad = float(os.environ.get('ZC_SCF_GRAD', 1e-8))
    mf.max_cycle = 200
    mf.kernel()
    t_scf = time.perf_counter() - t0
    mycc = DirectZCCSD(mf)
    mycc.verbose, mycc.max_cycle = 0, int(os.environ.get('ZC_CCSD_CYCLES', 100))
    mycc.conv_tol, mycc.conv_tol_normt = conv, 100 * conv
    if 'ZC_KRAMERS' in os.environ:
        mycc.kramers = bool(int(os.environ['ZC_KRAMERS']))
    cycles = []
    mycc.callback = lambda envs: cycles.append(time.perf_counter())
    t0 = time.perf_counter()
    mycc.ao2mo()
    t1 = time.perf_counter()
    mycc.kernel()
    t2 = time.perf_counter()
    o, v = mycc.t1.shape
    eris = mycc.eris
    print('%s/%s DirectZCCSD  nocc=%d nvir=%d  Kramers pairs: %s  max_memory %.0f MB  threads %s'
          % (mol_name, basis, o, v, bool(eris.kramers), maxmem,
             os.environ.get('OMP_NUM_THREADS', '?')))
    print('matrix products: ' + blas.backend()[1])
    fock = mf.get_fock()
    C = mf.mo_coeff
    occ = mf.mo_occ > 0
    print('SCF %.1f s (converged %s, largest occupied-virtual element of its Fock matrix %.1e); '
          'integrals %.1f s; CCSD %.1f s: %d cycles (%.2f s each), converged %s, E_corr %.10f'
          % (t_scf, mf.converged, abs(C[:, ~occ].conj().T @ fock @ C[:, occ]).max(), t1 - t0,
             t2 - t1, len(cycles), (t2 - t1) / max(len(cycles), 1), mycc.converged, mycc.e_corr))
    mycc.verbose = 4

    wall = {}
    if os.environ.get('ZC_T_ENERGY', '1') != '0':
        # as DirectZCCSD.ccsd_t: on the packed t2, the layouts of the CCSD step released
        t2p = K.pack_t2(np.ascontiguousarray(mycc.t2))
        if getattr(eris, 'residual', None) is not None:
            eris.residual.close()
        if getattr(eris, 'aoladder', None) is not None:
            eris.aoladder.close()
        watch = MemWatch()
        watch.start()
        timing.reset()
        ta = time.perf_counter()
        e_t = triples.kernel(mycc, eris, mycc.t1, t2p, three_m=mycc.three_m)
        wall['energy'] = time.perf_counter() - ta
        watch.stop()
        del t2p
        print('(T) energy: %.1f s, E(T) %.10f, peak anonymous memory %.2f GB'
              % (wall['energy'], e_t, watch.anon / 1e6))
        timers('t.', wall['energy'])

    # the Lambda equations of CCSD(T): one pass over the triples for the sources
    # and the densities, then the iterations
    mycc.max_cycle = int(os.environ.get('ZC_LAMBDA_CYCLES', 100))
    inner, done = mycc.t_terms, {}

    def t_terms(*args, **kwargs):
        timing.reset()
        ta = time.perf_counter()
        out = inner(*args, **kwargs)
        done['wall'] = time.perf_counter() - ta
        done['timers'] = [x for x in timing.report() if x[0].startswith('t.')]
        return out
    mycc.t_terms = t_terms
    watch = MemWatch()
    watch.start()
    ta = time.perf_counter()
    mycc.solve_lambda(with_t='gradient')
    wall['lambda'] = time.perf_counter() - ta - done['wall']
    watch.stop()
    mycc.t_terms = inner
    wall['terms'] = done['wall']
    print('(T) with the sources of Lambda and the densities of the orbital gradient: %.1f s%s, '
          'E(T) %.10f' % (wall['terms'], ' (%.2f times the (T) energy)'
                          % (wall['terms'] / wall['energy']) if 'energy' in wall else '', mycc.e_t))
    for name, w, calls in done['timers']:
        print('    %-14s %9.1f s  %5.1f %%  (%d calls)' % (name[2:], w, 100 * w / wall['terms'],
                                                           calls))
    den = mycc._t_gradient[2]
    print('    the density of <bc||ek>: %.2f GB, %s' % (
        den.g1.nbytes / 1e9, 'in a scratch file' if getattr(den.g1, '_mmap', None) is not None
        else 'in memory'))
    print('Lambda equations of CCSD(T): %.1f s, converged %s; peak anonymous memory of both '
          '%.2f GB' % (wall['lambda'], mycc.converged_lambda, watch.anon / 1e6))

    watch = MemWatch()
    watch.start()
    timing.reset()
    ta = time.perf_counter()
    relaxed, unrelaxed, z = zvector.kernel(mycc, tol=10 * conv, with_t=True)
    wall['relaxed'] = time.perf_counter() - ta
    watch.stop()
    print('relaxed density of CCSD(T) (the density of t and lambda, the orbital gradient, the '
          'Z-vector equations): %.1f s, peak anonymous memory %.2f GB'
          % (wall['relaxed'], watch.anon / 1e6))
    timers('zv.', wall['relaxed'])
    for prefix, key in (('og.', 'zv.orbgrad'), ('ogt.', 'zv.orbgrad_t')):
        w = dict((n, x) for n, x, _ in timing.report()).get(key, 0.0)
        print('  %s: %.1f s' % (key[3:], w))
        timers(prefix, max(w, 1e-9), 0.02)
    print('  largest elements: z %.3e, occupied-occupied %.3e, virtual-virtual %.3e'
          % (abs(z).max(), abs((relaxed - unrelaxed)[:o, :o]).max(),
             abs((relaxed - unrelaxed)[o:, o:]).max()))
    # the rotations among occupied and among virtual spinors, y = G / (e_q - e_p):
    # how close the levels are that they connect
    mo_e = zvector.full_space(mycc)[1]
    for name, blk in (('occupied', slice(0, o)), ('virtual', slice(o, None))):
        y = (relaxed - unrelaxed)[blk, blk]
        de = abs(mo_e[blk][None, :] - mo_e[blk][:, None])
        np.fill_diagonal(de, np.inf)
        near = de < 1e-4
        skipped = de < zvector.DEGENERATE
        p, q = np.unravel_index(np.argmax(abs(y)), y.shape)
        print('  %s pairs: largest |y| %.3e at an energy difference of %.3e; %d pairs closer '
              'than 1e-4 (the closest that is rotated: %.1e), the largest |y| among them '
              '%.3e; %d degenerate pairs, not rotated'
              % (name, abs(y[p, q]), de[p, q], near.sum() // 2,
                 de[near & ~skipped].min() if (near & ~skipped).any() else 0.0,
                 abs(y[near]).max() if near.any() else 0.0, skipped.sum() // 2))
    total = wall['terms'] + wall['lambda'] + wall['relaxed']
    print('after the CCSD: %.1f s in all - %.1f CCSD iterations%s'
          % (total, total * len(cycles) / (t2 - t1),
             ', %.2f times the (T) energy' % (total / wall['energy']) if 'energy' in wall else ''))

    C = zvector.full_space(mycc)[0]
    nuclear = np.einsum('i,ix->x', mol.atom_charges(), mol.atom_coords())
    r = rdm.dipole_integrals(mycc, True)
    print('dipole moment, z component (a.u., picture-change transformed operator): SCF %.9f, '
          'CCSD(T) at fixed spinors %.9f, CCSD(T) relaxed %.9f' % tuple(
              (nuclear - np.einsum('xpq,qp->x', r, d).real)[2] for d in (
                  mf.make_rdm1(), C @ unrelaxed @ C.conj().T, C @ relaxed @ C.conj().T)))
    print('E(SCF) %.10f  E_corr(CCSD) %.10f  E(T) %.10f' % (mf.e_tot, mycc.e_corr, mycc.e_t))
    print('peak RSS %.2f GB' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))


if __name__ == '__main__':
    main()
