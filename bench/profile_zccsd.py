'''Per-term timing of the spinor CCSD and (T) copied from socutils
(zccsdt.zccsd), for the comparison with CFOUR's xx2cccsd
(bench/cfour/slurm_x2ccc.sh).

usage: python bench/profile_zccsd.py MOLECULE BASIS VARIANT [NITER] [T_ALGS]
  MOLECULE: h2o | n2 | c6h6 | ... (the geometries of bench/profile_iter.py)
  VARIANT:  direct (DirectZCCSD: AO ladder, the amplitude equations and the
            iterations of zccsdt.zccsd.amps) | mo (ZCCSD: packed <ab||ef>),
            both with the
            integrals of zccsdt.zccsd.ao2mo; the routes of socutils:
            direct-soc (its update_amps, ovvv stored), direct-ao (ovvv terms
            from the AO integrals) and mo-nrr (all MO integrals through
            nrr_outcore and HDF5)
  NITER:    CCSD iterations (default 4); the first is left out of the means;
            0: the integrals only
  T_ALGS:   (T) codes to time, comma separated: batch (zccsdt.zccsd.triples,
            the default for direct; batch-z: with one ZGEMM per product
            instead of three real GEMMs), occ_loop and vir_loop (the loops
            copied from socutils, the default for the other variants:
            occ_loop), none
X2C-1e spinors from PySCF (x2c.UHF), all electrons.  The copied code is not
changed: its functions are wrapped by zccsdt.timing timers here.
environment: ZC_MAXMEM: max_memory of the CC object in MB (default 24000): the
             integrals are planned within it (what does not fit is mapped from
             a scratch file, the passes are sliced);
             ZC_THREE_M: 0 or 1: DirectZCCSD.three_m (default: by memory);
             ZC_KRAMERS: 0 or 1: DirectZCCSD.kramers (default: Kramers pairs if
             the reference is time-reversal symmetric); ZC_T_FULL=1: with
             Kramers pairs, (T) on all triples as well;
             ZC_T_BATCH_GB: triples.BATCH_BYTES in GB;
             ZC_T_BUDGET: seconds for each (T) loop; when they are used up the
             loop is stopped and its time extrapolated from the triples done;
             PYSCF_TMPDIR: scratch files of ZCCSD (put it on a local disk).
'''
import os
import sys
import time
import resource
from functools import wraps
from math import comb

import numpy as np
from pyscf import gto, lib, x2c

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_iter import GEOM, SPIN, MemWatch                          # noqa: E402

from zccsdt import timing                                              # noqa: E402
from zccsdt.cc import gintermediates as imd                            # noqa: E402
from zccsdt.zccsd import gccsd_t, zccsd, zccsd_direct as zd            # noqa: E402


def wrap(mod, name, label=None):
    f = getattr(mod, name)

    @wraps(f)
    def timed(*args, **kwargs):
        with timing.timer(label or name):
            return f(*args, **kwargs)
    setattr(mod, name, timed)


class _OutOfTime(Exception):
    pass


def timed_t(mycc, eris, alg, budget):
    '''gccsd_t.kernel with an optional time budget (s).  Every triple of
    either loop ends with one energy contraction ('ijk,ijk' or 'abc,abc'),
    counted through the module's einsum; when the budget is used up the loop
    is stopped and (None, seconds, (done, all, extrapolated seconds)) is
    returned.'''
    einsum, done, t0 = gccsd_t.einsum, [0], time.perf_counter()

    def counting(subscripts, *args, **kwargs):
        if subscripts in ('ijk,ijk', 'abc,abc'):
            done[0] += 1
            if budget and time.perf_counter() - t0 > budget:
                raise _OutOfTime
        return einsum(subscripts, *args, **kwargs)
    gccsd_t.einsum = counting
    try:
        e = gccsd_t.kernel(mycc, eris, mycc.t1, mycc.t2, 0, alg=alg)
        return e, time.perf_counter() - t0, None
    except _OutOfTime:
        sec = time.perf_counter() - t0
        nocc, nvir = mycc.t1.shape
        n = comb(nocc, 3) if alg == 'occ_loop' else comb(nvir, 3)
        return None, sec, (done[0], n, sec * n / max(done[0], 1))
    finally:
        gccsd_t.einsum = einsum


def tree(times, parent, depth, total):
    for name, w, c, n in timing.children(times, parent):
        print('  %s%-*s %9.3f s %7.1f %%  (%g calls)'
              % ('  ' * depth, 30 - 2 * depth, name, w, 100 * w / total, n))
        tree(times, parent + (name,), depth + 1, total)


def main():
    mol_name, basis, variant = sys.argv[1:4]
    niter = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    algs = (sys.argv[5].split(',') if len(sys.argv) > 5
            else ['batch' if variant == 'direct' else 'occ_loop'])
    algs = [a for a in algs if a != 'none']
    maxmem = float(os.environ.get('ZC_MAXMEM', 24000))

    mol = gto.M(atom=GEOM[mol_name], basis=basis, spin=SPIN.get(mol_name, 0), verbose=0,
                max_memory=maxmem)
    t0 = time.perf_counter()
    mf = x2c.UHF(mol).run(conv_tol=1e-10)
    t_scf = time.perf_counter() - t0

    # the terms of update_amps (both variants call gintermediates as imd)
    for name in ('make_tau', 'cc_Foo', 'cc_Fov', 'cc_Woooo'):
        wrap(imd, name)
    base, _, route = variant.partition('-')
    if base == 'direct':
        for name in ('_cc_Fvv', '_cc_Wovvo', 'update_t2_vvvv_direct', '_ladder_lc',
                     '_ovvv_ladder', '_ovvv_t1', '_ovvv_t2', '_ovvv_fvv', '_ovvv_wovvo'):
            wrap(zd, name)
        mycc = zd.DirectZCCSD(mf)
        mycc.ovvv_direct = route == 'ao'
        mycc.socutils_amps = route == 'soc'
        if 'ZC_THREE_M' in os.environ:          # 0: one ZGEMM per o^3 v^3 product
            mycc.three_m = bool(int(os.environ['ZC_THREE_M']))
        if 'ZC_KRAMERS' in os.environ:          # 0: the spinors of the SCF as they are
            mycc.kramers = bool(int(os.environ['ZC_KRAMERS']))
    else:
        for name in ('cc_Fvv', 'cc_Wovvo', 'update_t2_vvvv_sym', '_make_vvvv_tril'):
            wrap(imd, name)
        mycc = zccsd.ZCCSD(mf)
        mycc.ao2mo_outcore = route == 'nrr'
        if route == 'nrr':              # the stages of the nrr_outcore route
            from pyscf.ao2mo import _ao2mo, nrr_outcore
            for mod, name in ((nrr_outcore, 'general'), (nrr_outcore, 'half_e1'),
                              (nrr_outcore, 'r_e1'), (_ao2mo, 'r_e2')):
                wrap(mod, name, 'nrr.' + name)
    mycc.max_memory = maxmem
    mycc.verbose = 0
    nocc = mycc.nocc
    nvir = mycc.nmo - nocc
    print('%s/%s %s  nocc=%d nvir=%d nao=%d  SCF %.1f s  E(SCF) %.10f  threads %s'
          % (mol_name, basis, type(mycc).__name__, nocc, nvir, mol.nao_nr(), t_scf, mf.e_tot,
             lib.num_threads()))
    from zccsdt import blas
    print('matrix products of zccsdt.zccsd.amps and triples: ' + blas.backend()[1])

    watch = MemWatch()
    watch.start()
    timing.reset()
    t0 = time.perf_counter()
    mycc.verbose = 4                # the memory plan of the integrals (log.info)
    eris = mycc.ao2mo()
    mycc.verbose = 0
    t_ints = time.perf_counter() - t0
    watch.stop()
    print('integrals %.1f s (peak anonymous %.2f GB, max_memory %.0f MB)'
          % (t_ints, watch.anon / 1e6, maxmem))
    stages = [(k, t) for k, t, _ in timing.report() if k.startswith('ao2mo.')]
    if stages:
        print('  ' + '  '.join('%s %.2f s' % (k[6:], t) for k, t in stages)
              + '  other %.2f s' % (t_ints - sum(t for _, t in stages)))
    nrr = {k[4:]: t for k, t, _ in timing.report() if k.startswith('nrr.')}
    if nrr:
        # pass 1: AO integrals and first half in C (r_e1), then the swap file;
        # pass 2: the swap file read back, second half in C (r_e2), all MO
        # integrals written; then _make_eris_outcore reads them back to
        # antisymmetrize into the blocks (HDF5)
        print('  nrr pass 1: integrals + first half %.1f s, swap file %.1f s;  pass 2: second '
              'half %.1f s, read swap + write MO integrals %.1f s;  blocks from the MO file '
              '(and Fock matrix) %.1f s'
              % (nrr['r_e1'], nrr['half_e1'] - nrr['r_e1'], nrr['r_e2'],
                 nrr['general'] - nrr['half_e1'] - nrr['r_e2'], t_ints - nrr['general']))

    # one pass of the CC loop = update_amps + DIIS + energy: timed start to
    # start; the timers are read at the start of every pass
    starts, snaps = [], []

    def marked(update):
        @wraps(update)
        def update_amps(*args, **kwargs):
            starts.append(time.perf_counter())
            snaps.append(timing.snapshot())
            with timing.timer('update_amps'):
                return update(*args, **kwargs)
        return update_amps
    if base == 'direct' and mycc._fixed_layouts():      # the iterations of zccsdt.zccsd.amps
        from zccsdt.zccsd import amps
        amps.Residual.update_packed = marked(amps.Residual.update_packed)
    else:
        mycc.update_amps = marked(mycc.update_amps)
    if niter == 0:
        return
    mycc.max_cycle, mycc.conv_tol, mycc.conv_tol_normt = niter, 1e-14, 1e-14
    watch = MemWatch()
    watch.start()
    timing.reset()
    mycc.kernel(eris=eris)
    t_end = time.perf_counter()
    snaps.append(timing.snapshot())
    watch.stop()
    walls = np.diff(starts + [t_end])
    print('CCSD iterations: ' + ' '.join('%.2f' % w for w in walls) + ' s; E_corr %.12f'
          % mycc.e_corr)
    if len(walls) > 1:
        times = timing.since(snaps[1], snaps[-1])          # iterations 2.. only
        per = len(walls) - 1
        times = {p: (w / per, c / per, n / per) for p, (w, c, n) in times.items()}
        mean = float(np.mean(walls[1:]))
        print('mean iteration %.3f s (iterations 2-%d); update_amps %.3f s; the rest '
              '(DIIS, energy, vectors) %.3f s' % (mean, len(walls), times[('update_amps',)][0],
                                                 mean - times[('update_amps',)][0]))
        tree(times, ('update_amps',), 0, mean)
        inline = times[('update_amps',)][0] - sum(
            w for _, w, _, _ in timing.children(times, ('update_amps',)))
        print('  %-30s %9.3f s %7.1f %%' % ('(not under a timer)', inline, 100 * inline / mean))
        for name, w, c, n in timing.children(times, ()):
            if name != 'update_amps':                   # DIIS and energy of amps.kernel
                print('  %-30s %9.3f s %7.1f %%  (%g calls)' % (name, w, 100 * w / mean, n))
                tree(times, (name,), 1, mean)
    print('peak anonymous memory in the iterations %.2f GB' % (watch.anon / 1e6))
    if getattr(eris, 'residual', None) is not None:
        print('memory plan of the step: ' + eris.residual.describe())

    for alg in algs:
        watch = MemWatch()
        watch.start()
        t0 = time.perf_counter()
        if alg in ('batch', 'batch-z'):
            from zccsdt.zccsd import triples
            if 'ZC_T_BATCH_GB' in os.environ:       # the arrays of a batch of (T), at most
                triples.BATCH_BYTES = int(float(os.environ['ZC_T_BATCH_GB']) * 2 ** 30)
            timing.reset()
            mycc.verbose = 4                        # where <bc||ei> is, the batches
            if getattr(eris, 'residual', None) is not None:     # as DirectZCCSD.ccsd_t does
                eris.residual.close()
                eris.residual = None
            e_t = triples.kernel(mycc, eris, mycc.t1, mycc.t2,
                                 three_m=None if alg == 'batch' else False)
            mycc.verbose = 0
            sec = time.perf_counter() - t0
            watch.stop()
            print('(T) %s: %.1f s (E(T) %.12f on the amplitudes of iteration %d; peak '
                  'anonymous %.2f GB)' % (alg, sec, e_t, niter, watch.anon / 1e6))
            print('  ' + '  '.join('%s %.1f s' % (k[2:], t) for k, t, _ in timing.report()
                                   if k.startswith('t.')))
            if eris.kramers and os.environ.get('ZC_T_FULL'):    # all triples as well
                t0 = time.perf_counter()
                e_all = triples.kernel(mycc, eris, mycc.t1, mycc.t2, 0, three_m=alg == 'batch',
                                       kramers=False)
                print('(T) %s on all triples: %.1f s, E(T) %.12f' % (alg, time.perf_counter() - t0,
                                                                    e_all))
            continue
        if eris.ovvv is None:
            eris.ovvv = zd.get_ovvv(eris)
            print('(T): ovvv rebuilt from the AO integrals in %.1f s'
                  % (time.perf_counter() - t0))
        e_t, sec, cut = timed_t(mycc, eris, alg, float(os.environ.get('ZC_T_BUDGET', 0)))
        watch.stop()
        if cut is None:
            print('(T) %s: %.1f s (E(T) %.12f on the amplitudes of iteration %d; peak '
                  'anonymous %.2f GB)' % (alg, sec, np.real(e_t), niter, watch.anon / 1e6))
        else:
            print('(T) %s: stopped after %.1f s at %d of %d triples: about %.0f s in all '
                  '(peak anonymous %.2f GB)' % (alg, sec, cut[0], cut[1], cut[2],
                                                watch.anon / 1e6))
    print('peak RSS %.2f GB' % (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))


if __name__ == '__main__':
    main()
