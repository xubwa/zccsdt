'''Time and memory of the first derivatives of CCSDT(Q) next to (Q) itself and
to the properties of CCSDT, through the front end (zccsdt.CCSDT):

  (Q)                     ccsdt_q(): in the sectors of the symmetry labels where
                          there are any, and the dense code (cc.quadruples)
  (Q) gradient pass       the sources of Lambda and the densities of (Q): in
                          the sectors of the symmetry labels where there are
                          any (cc.quadruples_grad_sym), and the pass on the
                          packed block (cc.quadruples_grad)
  Lambda                  solve_lambda() and solve_lambda(with_q=True): time per
                          iteration and number of iterations
  relaxed density         make_rdm1() and make_rdm1(with_q=True), Lambda and the
                          pass over the quadruples not counted

usage: python bench/profile_q_gradient.py MOLECULE BASIS KIND [FROZEN]
  KIND: ghf   RHF orbitals as spin orbitals (real)
        x2c   spinors (ZC_HAM, ZC_SCF, ZC_GEOMETRY ... as in bench/spinors.py)
  FROZEN: frozen occupied spin orbitals or spinors
  ZC_MAXMEM: max_memory in MB (default: 80% of the available memory)
The electronic dipole derivative along z, trace(z D), and with the nuclei the
dipole moment are printed for CCSDT and CCSDT(Q).
'''
import os
import sys
import time
from math import comb

import numpy as np
from pyscf import scf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spinors                                                          # noqa: E402
from profile_sa import MemWatch                                         # noqa: E402

import zccsdt                                                           # noqa: E402
from zccsdt import memory, timing                                       # noqa: E402
from zccsdt.cc import ccsdt_opt, lambda_opt, quadruples as QD, quadruples_grad as QG      # noqa: E402
from zccsdt.cc import quadruples_grad_sym as GS                         # noqa: E402


def anon():
    with open('/proc/self/status') as f:
        return {l.split(':')[0]: int(l.split()[1]) for l in f if l.startswith('Rss')}['RssAnon'] / 1e6


def timed(tag, fn, prefixes=(), depth=2):
    '''fn() with its wall time, the cores busy and the peak of the anonymous memory.'''
    snap = timing.snapshot()
    before = anon()
    watch = MemWatch()
    watch.start()
    t0, c0 = time.perf_counter(), time.process_time()
    out = fn()
    dt, cpu = time.perf_counter() - t0, time.process_time() - c0
    watch.stop()
    print('%-34s %9.2f s  %5.2f cores busy  peak anonymous %.2f GB (%.2f GB before)'
          % (tag, dt, cpu / dt, watch.anon / 1e6, before), flush=True)
    for path, (w, c, k) in sorted(timing.since(snap).items(), key=lambda x: -x[1][0]):
        if path[-1].split('.')[0] in prefixes and len(path) <= depth and w > 0.01 * dt:
            print('    %-22s %9.2f s  %5.1f cores  %7d calls' % (path[-1], w, c / w if w else 0, k))
    return out, dt


def main():
    name, basis, kind = sys.argv[1:4]
    frozen = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    mol = spinors.molecule(name, basis)
    t0 = time.perf_counter()
    if kind == 'x2c':
        mf = spinors.scf(mol, 1e-12, 1e-9)
        with mol.with_common_orig((0, 0, 0)):
            z = mol.intor('int1e_r_spinor')[2]
    else:
        rhf = scf.RHF(mol)
        rhf.conv_tol, rhf.conv_tol_grad = 1e-12, 1e-9
        mf = scf.addons.convert_to_ghf(rhf.run())
        with mol.with_common_orig((0, 0, 0)):
            z = np.kron(np.eye(2), mol.intor('int1e_r')[2])
    nuc = sum(mol.atom_charge(i) * mol.atom_coord(i)[2] for i in range(mol.natm))
    print('%s/%s %s  E(SCF) = %.12f  %.1f s' % (name, basis, kind, mf.e_tot, time.perf_counter() - t0), flush=True)
    cc = zccsdt.CCSDT(mf, frozen=frozen or None)
    cc.verbose = 3
    cc.conv_tol, cc.conv_tol_normt = 1e-11, 1e-9
    max_memory = float(os.environ['ZC_MAXMEM']) if os.environ.get('ZC_MAXMEM') else None
    cc.max_memory = max_memory
    cc.run()
    eris = cc._eris
    no, nv = eris.nocc, eris.nmo - eris.nocc
    t3 = ccsdt_opt.packed_t3(eris, cc.t3)
    isz = np.dtype(np.result_type(cc.t2, t3, eris.oovv)).itemsize
    budget = memory.budget(max_memory)
    print('%d occupied, %d virtual, itemsize %d; T3 %.2f GB (%d elements; in the layout of the iterations %d); '
          'CCSDT %d iterations (converged %s), %.2f s per iteration; budget %s; anonymous memory now %.2f GB'
          % (no, nv, isz, t3.nbytes / 1e9, t3.size, np.size(cc.t3), cc.cycles or 0, cc.converged,
             float(np.mean(cc.timings[1:])) if cc.timings is not None and len(cc.timings) > 1 else float('nan'),
             'none' if budget is None else '%.1f GB' % (budget / 1e9), anon()), flush=True)

    # ---- (Q)
    e_q, t_q = timed('(Q), front end', cc.ccsdt_q, ('q', 'qs'))
    pl = QD.plan(no, nv, isz, memory.budget(max_memory))
    (e5, e_qd), t_qd = timed('(Q), dense code', lambda: QD.kernel(eris, cc.t2, t3, plan_=pl), ('q',))
    print('    plan %.2f GB; E(Q) %.12e (front end %.12e)' % (pl.need / 1e9, e_qd, e_q))
    del t3

    # ---- CCSDT: Lambda, relaxed density
    results = []
    kernel = lambda_opt.kernel

    def spy(*args, **kwargs):
        results.append(kernel(*args, **kwargs))
        return results[-1]
    lambda_opt.kernel = spy
    _, t_l0 = timed('Lambda of CCSDT', cc.solve_lambda)
    r0 = results[-1]
    print('    %d iterations (converged %s, in blocks: %s), %.2f s per iteration, setup %.2f s'
          % (r0.ncycle, r0.converged, r0.blocks, float(np.mean(r0.timings[1:])), r0.setup))
    D0, t_d0 = timed('relaxed density of CCSDT', lambda: cc.make_rdm1(ao_repr=True), ('dens',))
    v0 = np.einsum('pq,qp', z, D0).real
    print('    in blocks: %s; trace(z D) = %.12f, dipole moment %.10f' % (cc.rdm1_blocks, v0, nuc - v0), flush=True)

    # ---- CCSDT(Q): the pass, Lambda with the sources, relaxed density
    def dense_pass():
        t3 = ccsdt_opt.packed_t3(eris, cc.t3)
        pl = QG.plan(no, nv, isz, memory.budget(max_memory))
        return QG.kernel(eris, cc.t2, t3, plan_=pl)

    def describe(gp):
        if isinstance(gp, GS.Plan):
            return ('plan %.2f GB besides T3 and the integrals (%.2f GB of it the derivatives; the density of '
                    '<bc||ef> in blocks: %.3f GB of %.3f)'
                    % (gp.need / 1e9, gp.kept / 1e9, GS.pair_block_elems(gp.sectors) * isz / 1e9,
                       comb(nv, 2) ** 2 * isz / 1e9))
        return ('plan %.2f GB besides T3 and the integrals (%.2f GB of it the derivatives in memory, in a scratch '
                'file: %s; batches of %d / %d and %d / %d columns, Xv for %d occupied indices)'
                % (gp.need / 1e9, gp.kept / 1e9, ', '.join(gp.mapped) or 'nothing', gp.q.cols22, comb(nv, 2),
                   gp.q.cols13, comb(nv, 3), gp.q.xv))
    t_gd = None
    if os.environ.get('ZC_DENSE_PASS', '1') != '0' and GS.available(eris) and cc.symmetry:
        qd, t_gd = timed('(Q) gradient pass, dense code', dense_pass, ('q', 'qg'))
        print('    %s; E(Q) %.12e' % (describe(qd.plan), qd.e_q))
        del qd
    q, t_g = timed('(Q) gradient pass', cc._q_terms, ('q', 'qg', 'qgs'))
    print('    in sectors: %s; %s; E(Q) %.12e' % (cc.q_gradient_sectors, describe(q.plan), q.e_q))
    _, t_l1 = timed('Lambda of CCSDT(Q)', lambda: cc.solve_lambda(with_q=True))
    r1 = results[-1]
    print('    %d iterations (converged %s, in blocks: %s), %.2f s per iteration, setup %.2f s'
          % (r1.ncycle, r1.converged, r1.blocks, float(np.mean(r1.timings[1:])), r1.setup))
    D1, t_d1 = timed('relaxed density of CCSDT(Q)', lambda: cc.make_rdm1(ao_repr=True, with_q=True), ('dens',), 3)
    v1 = np.einsum('pq,qp', z, D1).real
    print('    in blocks: %s; trace(z D) = %.12f, dipole moment %.10f' % (cc.rdm1_blocks, v1, nuc - v1), flush=True)
    lambda_opt.kernel = kernel

    it0, it1 = float(np.mean(r0.timings[1:])), float(np.mean(r1.timings[1:]))
    print('\nsummary %s/%s %s, %d + %d, %d frozen' % (name, basis, kind, no, nv, frozen))
    print('  (Q) energy: front end %.2f s, dense code %.2f s' % (t_q, t_qd))
    print('  (Q) gradient pass %.2f s = %.2f x (Q) dense = %.2f x (Q) front end' % (t_g, t_g / t_qd, t_g / t_q))
    if t_gd is not None:
        print('  (Q) gradient pass, dense code %.2f s = %.2f x (Q) dense' % (t_gd, t_gd / t_qd))
    print('  Lambda iteration: %.2f s without, %.2f s with the sources (%.3f); %d and %d iterations'
          % (it0, it1, it1 / it0, r0.ncycle, r1.ncycle))
    print('  relaxed density (after Lambda): %.2f s CCSDT, %.2f s CCSDT(Q) (%.2f)' % (t_d0, t_d1, t_d1 / t_d0))
    print('  all of the property after the CCSDT energy: CCSDT %.2f s; CCSDT(Q) %.2f s = %.2f x (Q) dense'
          ' = %.2f x (Q) front end' % (t_l0 + t_d0, t_g + t_l1 + t_d1, (t_g + t_l1 + t_d1) / t_qd,
                                       (t_g + t_l1 + t_d1) / t_q))
    print('  dipole moment: CCSDT %.10f, CCSDT(Q) %.10f' % (nuc - v0, nuc - v1))


if __name__ == '__main__':
    main()
