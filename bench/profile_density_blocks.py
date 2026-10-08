'''The relaxed density of CCSDT on the packed block and in the blocks of the
symmetry labels (zccsdt.cc.density_opt, density_blk), next to an amplitude
and a Lambda iteration in both layouts: one process, the same integrals,
amplitudes and Lambda.

usage: python bench/profile_density_blocks.py stages MOLECULE BASIS [NITER] [FROZEN] [NDROP]
           The amplitudes are converged in the blocks first (the front end).
           Then, the packed block before the blocks each time: NITER
           amplitude iterations after a warm-up iteration, the Lambda
           equations to convergence, and everything of relaxed_rdm1 -
           densities, orbital gradient, Z-vector equations - ZC_RUNS times
           (default 2) with the Lambda of the blocks.  For every stage: wall
           seconds, CPU / wall (the cores it keeps busy), the peak of the
           anonymous resident memory above what was resident before it, and
           the plan.
       python bench/profile_density_blocks.py front MOLECULE BASIS [FROZEN] [NDROP]
           The front end from the spinors of one SCF: CCSDT, Lambda and the
           relaxed density with ``symmetry`` False and then True; wall time
           of each stage.
  MOLECULE: hbr | n2 | hf | hcl (linear, on the z axis: X2C spinors adapted
  to m_j, the labels 2 m_j of the front end, real integrals)
  FROZEN: frozen occupied spinors (the lowest); NDROP: virtual spinors
  dropped (the highest)
environment: ZC_MEM_MB the batch buffer budget of the iterations (default
             2000); ZC_MAXMEM max_memory in MB (default: 80% of the
             available memory); ZC_TIGHT=1 (stages): the density in blocks
             once more under a budget a quarter of the way from the floor of
             its plan to the whole plan
'''
import os
import sys
import time

import numpy as np
from pyscf import gto, x2c

import zccsdt
from zccsdt import memory, timing
from zccsdt.cc import ccsdt_opt, density_blk, density_opt, lambda_opt, solver

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_lambda_blocks import Phase, iterations     # noqa: E402
from profile_lambda_ccsdt import tree                   # noqa: E402

GEOM = {'hbr': 'Br 0 0 0; H 0 0 1.4145', 'n2': 'N 0 0 0; N 0 0 1.0977', 'hf': 'F 0 0 0; H 0 0 0.9168',
        'hcl': 'Cl 0 0 0; H 0 0 1.2746'}


def reference(mol_name, basis):
    mol = gto.M(atom=GEOM[mol_name], basis=basis, verbose=0)
    mf = x2c.UHF(mol)
    mf.conv_tol, mf.conv_tol_grad = 1e-12, 1e-8
    t0 = time.perf_counter()
    mf.kernel()
    print('%s / %s, X2C: %d spinors, SCF %.1f s, E = %.10f  threads %s'
          % (mol_name, basis, mf.mo_coeff.shape[1], time.perf_counter() - t0, mf.e_tot,
             os.environ.get('OMP_NUM_THREADS')), flush=True)
    return mf


def frozen_list(mf, nfrozen, ndrop):
    '''``frozen`` of the front end: the lowest occupied and the highest
    virtual spinors.'''
    if not ndrop:
        return nfrozen
    n = len(mf.mo_occ)
    return list(range(nfrozen)) + list(range(n - ndrop, n))


def density_args(cc):
    '''(active spinors, keywords) of relaxed_rdm1 as GCCSDT.make_rdm1 passes them.'''
    fo, o, v, fv = cc._active()
    if getattr(cc._eris, 'labels', None) is not None and getattr(cc, '_sorted', None) is not None:
        o, v = cc._sorted
    axial = cc._axial()
    Cs = np.asarray(cc.mo_coeff) if axial is None else axial
    return Cs[:, np.concatenate((o, v))], dict(frozen_occ=Cs[:, fo] if len(fo) else None,
                                              dropped_vir=Cs[:, fv] if len(fv) else None)


def checksum(rel):
    w = np.linalg.eigvalsh(rel.relaxed)
    return ('|D|^2 %.12f  |D - D(unrelaxed)|^2 %.6e  occupation numbers %.12f .. %.12f%s'
            % (np.linalg.norm(rel.relaxed) ** 2, np.linalg.norm(rel.relaxed - rel.unrelaxed) ** 2, w[0], w[-1],
               '' if rel.gradient is None else '  |G|^2 %.12e' % np.linalg.norm(rel.gradient) ** 2))


def stages(mol_name, basis, niter, nfrozen, ndrop):
    mf = reference(mol_name, basis)
    mem = {'max_memory': float(os.environ['ZC_MAXMEM'])} if os.environ.get('ZC_MAXMEM') else {}
    cc = zccsdt.GCCSDT(mf, frozen=frozen_list(mf, nfrozen, ndrop))
    cc.verbose = 3
    cc.conv_tol, cc.conv_tol_normt = 1e-10, 1e-9
    if mem:
        cc.max_memory = mem['max_memory']
    t0 = time.perf_counter()
    cc.kernel()
    eris = cc._eris
    nocc, nvir = eris.nocc, eris.nmo - eris.nocc
    labels = np.asarray(eris.labels)
    print('nocc=%d nvir=%d (%d frozen, %d dropped) dtype=%s  labels of the occupied %s, of the virtual %s  '
          'integrals and CCSDT %.1fs (converged %s)'
          % (nocc, nvir, nfrozen, ndrop, eris.dtype, np.unique(labels[:nocc]), np.unique(labels[nocc:]),
             time.perf_counter() - t0, cc.converged), flush=True)
    t1, t2, t3b = cc.t1, cc.t2, np.array(cc.t3)
    dense, t3d = cc._dense_view()
    t3d = np.ascontiguousarray(t3d)
    print('T3: %d elements in the blocks, %d on the packed block (%.1f%%)'
          % (t3b.size, t3d.size, 100.0 * t3b.size / t3d.size), flush=True)
    kw = dict(verbose=False, mem_mb=float(os.environ.get('ZC_MEM_MB', 2000)), **mem)
    out = {}

    # ---- amplitude iterations
    for name, e, t3 in (('packed block', dense, t3d), ('blocks', eris, t3b)):
        ccsdt_opt.release_cached(eris)
        ccsdt_opt.release_cached(dense)

        def run(n, r0, cb, e=e, t3=t3):
            o = solver.Options(max_cycle=n or niter, conv_tol=1e-30, conv_tol_normt=1e-30,
                               plan=r0 and r0.plan, callback=cb, **kw)
            return solver.kernel(e, 'ccsdt', o, t1=t1, t2=t2, t3=t3)
        r, ph, w, cores = iterations('amplitude iterations, ' + name, run)
        out['amp', name] = (w, cores, ph.peak - ph.base)
        del r
        sys.stdout.flush()
    ccsdt_opt.release_cached(eris)
    ccsdt_opt.release_cached(dense)

    # ---- Lambda equations
    lam = {}
    for name, e, t3, blocks in (('packed block', dense, t3d, False), ('blocks', eris, t3b, True)):
        stat = []
        with Phase() as ph:
            r = lambda_opt.kernel(e, t1, t2, t3, solver.Options(
                conv_tol_normt=1e-9, callback=lambda info: stat.append((info['wall'], info['cpu'])),
                **kw), blocks=blocks)
        w, c = np.array(stat[1:]).T             # (without the first iteration)
        print('Lambda, %s: converged %s in %d cycles; intermediates %.3f s; mean %.3f s per iteration, '
              '%.1f cores busy;  peak anonymous %.2f GB above the %.2f GB before it'
              % (name, r.converged, r.ncycle, r.setup, w.mean(), c.sum() / w.sum(),
                 (ph.peak - ph.base) / 1e9, ph.base / 1e9))
        out['lam', name] = (float(w.mean()), float(c.sum() / w.sum()), ph.peak - ph.base)
        lam[name] = (r.l1, r.l2, r.l3)
        del r
        ccsdt_opt.release_cached(e)
        sys.stdout.flush()
    l1, l2, l3b = lam['blocks']
    l3d = np.ascontiguousarray(lambda_opt.packed_l3(eris, l3b))
    print('Lambda, blocks against the packed block: max |dl1| %.2e  |dl2| %.2e  |dl3| %.2e'
          % (abs(lam['packed block'][0] - l1).max(), abs(lam['packed block'][1] - l2).max(),
             abs(lam['packed block'][2].reshape(l3d.shape) - l3d).max()), flush=True)
    del lam

    # ---- the relaxed density, with the Lambda of the blocks in both layouts
    Ca, fz = density_args(cc)
    rels = {}
    for name, e, t3, l3, blocks in (('packed block', dense, t3d, l3d, False), ('blocks', eris, t3b, l3b, True)):
        for attempt in range(int(os.environ.get('ZC_RUNS', 2))):
            with Phase() as ph:
                rel = density_opt.relaxed_rdm1(mf, e, t1, t2, t3, l1, l2, l3, Ca, tol=1e-9, blocks=blocks,
                                               **fz, **mem)
            plan = rel.result.plan
            print('relaxed density, %s, run %d: %.3f s, %.1f cores busy;  peak anonymous %.2f GB above the '
                  '%.2f GB before it;  plan: need %.2f GB, %d pairs per batch, slab %s, route %s, %s frozen '
                  'spinors at a time%s'
                  % (name, attempt + 1, ph.w, ph.c / ph.w, (ph.peak - ph.base) / 1e9, ph.base / 1e9,
                     plan.need / 1e9, plan.nb, plan.slab_bytes, plan.route, plan.nk,
                     ', Kramers pairs (half of them)' if plan.kramers else ''))
            tree(ph.times, 1)
            sys.stdout.flush()
        out['dens', name] = (ph.w, ph.c / ph.w, ph.peak - ph.base, plan.need)
        rels[name] = rel
        print('  checksum: ' + checksum(rel))
        del rel
    a, b = rels['packed block'], rels['blocks']
    print('blocks against the packed block: relaxed density %.2e  unrelaxed %.2e  Gnh[a,i] %.2e%s  z %.2e'
          % (abs(a.relaxed - b.relaxed).max(), abs(a.unrelaxed - b.unrelaxed).max(),
             abs(a.result.gnh - b.result.gnh).max(),
             '' if a.gradient is None else '  gradient over all spinors %.2e' % abs(a.gradient - b.gradient).max(),
             abs(a.z - b.z).max()))
    print('\nsummary (s, cores busy, peak anonymous GB above the phase start)')
    for kind, what in (('amp', 'amplitude iteration'), ('lam', 'Lambda iteration'), ('dens', 'relaxed density')):
        d, k = out[kind, 'packed block'], out[kind, 'blocks']
        print('  %-20s packed block %8.3f s %4.1f cores %6.2f GB   blocks %8.3f s %4.1f cores %6.2f GB   '
              'ratio %.2f' % (what, d[0], d[1], d[2] / 1e9, k[0], k[1], k[2] / 1e9, d[0] / k[0]))
    print('  relaxed density in iterations of its layout: packed block %.2f, blocks %.2f;  plans %.2f and '
          '%.2f GB' % (out['dens', 'packed block'][0] / out['amp', 'packed block'][0],
                       out['dens', 'blocks'][0] / out['amp', 'blocks'][0],
                       out['dens', 'packed block'][3] / 1e9, out['dens', 'blocks'][3] / 1e9))

    # ---- the blocks under a budget a quarter of the way from the floor of the plan to all of it
    if os.environ.get('ZC_TIGHT') == '1':
        del rels, a, b
        full = density_opt.relaxed_rdm1(mf, eris, t1, t2, t3b, l1, l2, l3b, Ca, blocks=True, plan_only=True,
                                        **fz)
        eng = density_blk.engine(eris, t3b.dtype)
        lo, hi = 0, full.need
        while hi - lo > 1_000_000:          # the floor of the plan, to 1 MB
            mid = (lo + hi) // 2
            try:
                density_opt.relaxed_rdm1(mf, eris, t1, t2, t3b, l1, l2, l3b, Ca, blocks=True, plan_only=True,
                                         max_memory=(memory.resident() + mid) / 1e6, **fz)
                hi = mid
            except MemoryError:
                lo = mid
        budget = hi + (full.need - hi) // 4
        del eng
        with Phase() as ph:
            rel = density_opt.relaxed_rdm1(mf, eris, t1, t2, t3b, l1, l2, l3b, Ca, tol=1e-9, blocks=True,
                                           max_memory=(memory.resident() + budget) / 1e6, **fz)
        plan = rel.result.plan
        print('\nrelaxed density in blocks under a budget: the plan needs %.3f GB at least and %.3f GB with '
              'everything; budget %.3f GB -> need %.3f GB, %d pairs and %s triples per batch, slab %s, %s '
              'frozen spinors at a time'
              % (hi / 1e9, full.need / 1e9, budget / 1e9, plan.need / 1e9, plan.nb, plan.ng, plan.slab_bytes,
                 plan.nk))
        print('  %.3f s, %.1f cores busy;  peak anonymous %.3f GB above the %.2f GB before it'
              % (ph.w, ph.c / ph.w, (ph.peak - ph.base) / 1e9, ph.base / 1e9))
        print('  checksum: ' + checksum(rel))


def front(mol_name, basis, nfrozen, ndrop):
    mf = reference(mol_name, basis)
    mem = {'max_memory': float(os.environ['ZC_MAXMEM'])} if os.environ.get('ZC_MAXMEM') else {}
    res = {}
    for sym in (False, True):
        cc = zccsdt.GCCSDT(mf, frozen=frozen_list(mf, nfrozen, ndrop))
        cc.verbose = 0
        cc.symmetry = sym
        cc.conv_tol, cc.conv_tol_normt = 1e-10, 1e-9
        if mem:
            cc.max_memory = mem['max_memory']
        base = memory.resident()
        w = [time.perf_counter()]
        cc.kernel()
        w.append(time.perf_counter())
        cc.solve_lambda()
        w.append(time.perf_counter())
        D = cc.make_rdm1()
        w.append(time.perf_counter())
        eris = cc._eris
        s = np.diff(w)
        print('symmetry %-5s  %d + %d spinors  T3 %d elements  CCSDT %8.2f s (integrals and %s cycles, '
              'converged %s)  Lambda %8.2f s (in blocks: %s, converged %s)  relaxed density %8.2f s (in '
              'blocks: %s)  all %8.2f s  resident after it %.2f GB (%.2f before)'
              % (sym, eris.nocc, eris.nmo - eris.nocc, np.size(cc.t3), s[0], getattr(cc, 'cycles', '?'),
                 cc.converged, s[1], cc.lambda_blocks, cc.converged_lambda, s[2], cc.rdm1_blocks, s.sum(),
                 memory.resident() / 1e9, base / 1e9), flush=True)
        ev = np.linalg.eigvalsh(D)
        res[sym] = (cc.e_tot, cc.rdm1_basis @ D @ cc.rdm1_basis.conj().T, ev, s)
        ccsdt_opt.release_cached(eris)
        del cc, eris, D
    a, b = res[False], res[True]
    print('with and without symmetry: E(CCSDT) %.12f, difference %.1e;  relaxed density in the AO basis, '
          'max difference %.2e;  occupation numbers %.2e'
          % (b[0], abs(a[0] - b[0]), abs(a[1] - b[1]).max(), abs(a[2] - b[2]).max()))
    print('ratio without / with symmetry: CCSDT %.2f  Lambda %.2f  relaxed density %.2f  all %.2f'
          % tuple(list(a[3] / b[3]) + [a[3].sum() / b[3].sum()]))


def main():
    mode, mol_name, basis = sys.argv[1:4]
    rest = [int(x) for x in sys.argv[4:]]
    if mode == 'stages':
        niter, nfrozen, ndrop = (rest + [3, 0, 0][len(rest):])[:3]
        stages(mol_name, basis, niter, nfrozen, ndrop)
    else:
        nfrozen, ndrop = (rest + [0, 0][len(rest):])[:2]
        front(mol_name, basis, nfrozen, ndrop)


if __name__ == '__main__':
    main()
