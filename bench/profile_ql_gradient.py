'''Time and memory of the first derivatives of CCSDT(Q)_Lambda next to those of
CCSDT and CCSDT(Q), through the front end (zccsdt.CCSDT):

  CCSDT          Lambda, the relaxed density
  CCSDT(Q)       the pass over the quadruples (sources and densities of (Q)),
                 Lambda with the sources, the relaxed density
  CCSDT(Q)_L     Lambda of CCSDT, E(Q)_L, the pass over the quadruples (the
                 sources of the two equations, the densities of the
                 correction), zeta - iterations, amplitude steps, seconds
                 per iteration and per amplitude step -, the sources of
                 lambda', lambda', the relaxed density

and, with ZC_STENCILS=1, the equation of zeta once more with the central
differences and with the exact formula in every iteration.

usage: python bench/profile_ql_gradient.py MOLECULE BASIS KIND [FROZEN]
  KIND: ghf   RHF orbitals as spin orbitals (real)
        x2c   spinors (ZC_HAM, ZC_SCF, ZC_GEOMETRY ... as in bench/spinors.py)
  FROZEN: frozen occupied spin orbitals or spinors
  ZC_MAXMEM: max_memory in MB (default: 80% of the available memory)
The electronic dipole derivative along z, trace(z D), and with the nuclei the
dipole moment are printed for the three methods.
'''
import os
import sys
import time

import numpy as np
from pyscf import scf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spinors                                                          # noqa: E402
from profile_q_gradient import anon, timed                              # noqa: E402

import zccsdt                                                           # noqa: E402
from zccsdt import memory                                               # noqa: E402
from zccsdt.cc import ccsdt_opt, lambda_opt, solver, zeta_opt           # noqa: E402


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
    isz = np.dtype(np.result_type(cc.t2, cc.t3, eris.oovv)).itemsize
    budget = memory.budget(max_memory)
    it_cc = float(np.mean(cc.timings[1:])) if cc.timings is not None and len(cc.timings) > 1 else float('nan')
    t3_gb = np.size(cc.t3) * isz / 1e9
    print('%d occupied, %d virtual, itemsize %d; T3 in the layout of the iterations %.3f GB (%d elements); '
          'CCSDT %d iterations (converged %s), %.2f s per iteration; budget %s; anonymous memory now %.2f GB'
          % (no, nv, isz, t3_gb, np.size(cc.t3), cc.cycles or 0, cc.converged, it_cc,
             'none' if budget is None else '%.1f GB' % (budget / 1e9), anon()), flush=True)

    results = []
    kernel = lambda_opt.kernel

    def spy(*args, **kwargs):
        results.append(kernel(*args, **kwargs))
        return results[-1]
    lambda_opt.kernel = spy

    def lam_line(r):
        print('    %d iterations (converged %s, in blocks: %s), %.2f s per iteration, setup %.2f s; plan %.2f GB'
              % (r.ncycle, r.converged, r.blocks, float(np.mean(r.timings[1:])), r.setup, r.plan.need / 1e9), flush=True)
        return float(np.mean(r.timings[1:]))

    def dipole(tag, D):
        v = np.einsum('pq,qp', z, D).real
        print('    in blocks: %s; trace(z D) = %.12f, dipole moment %.10f' % (cc.rdm1_blocks, v, nuc - v), flush=True)
        return nuc - v

    # ---- CCSDT
    e_q, t_q = timed('(Q), front end', cc.ccsdt_q, ('q', 'qs'))
    _, t_l0 = timed('Lambda of CCSDT', cc.solve_lambda)
    it_l0 = lam_line(results[-1])
    n_l0 = results[-1].ncycle
    D0, t_d0 = timed('relaxed density of CCSDT', lambda: cc.make_rdm1(ao_repr=True), ('dens',))
    mu0 = dipole('CCSDT', D0)
    e_ql, t_ql = timed('(Q)_L energy, front end', cc.ccsdt_q_lambda, ('q', 'qs', 'ql', 'qls'))
    print('    E(Q) %.12e  E(Q)_L %.12e' % (e_q, e_ql), flush=True)

    # ---- CCSDT(Q)
    q, t_g = timed('(Q) gradient pass', cc._q_terms, ('q', 'qg', 'qgs'))
    print('    in sectors: %s; plan %.2f GB' % (cc.q_gradient_sectors, q.plan.need / 1e9))
    del q
    _, t_l1 = timed('Lambda of CCSDT(Q)', lambda: cc.solve_lambda(with_q=True))
    it_l1 = lam_line(results[-1])
    D1, t_d1 = timed('relaxed density of CCSDT(Q)', lambda: cc.make_rdm1(ao_repr=True, with_q=True), ('dens',), 3)
    mu1 = dipole('CCSDT(Q)', D1)
    cc._q_gradient = None

    # ---- CCSDT(Q)_Lambda: lambda of CCSDT is solved again (the object holds that of CCSDT(Q) now)
    _, t_l0b = timed('Lambda of CCSDT, again', cc.solve_lambda)
    lam_line(results[-1])
    ql, t_gl = timed('(Q)_L gradient pass', cc._ql_terms, ('q', 'ql', 'qg', 'qgs'))
    pl = ql.plan
    print('    in sectors: %s; plan %.2f GB besides T3, Lambda_3 and the integrals (%.2f GB of it the derivatives '
          'in memory; in a scratch file: %s); E(Q)_L %.12e, Im e %.1e'
          % (cc.ql_gradient_sectors, pl.need / 1e9, pl.kept / 1e9, ', '.join(getattr(pl, 'mapped', ())) or 'nothing',
             ql.e_q, ql.e_imag), flush=True)
    _, t_s = timed("zeta, sources, lambda'", lambda: cc.solve_lambda(with_q='lambda'), ('zeta', 'lam'))
    info = cc.ql_info
    zi, li, si = info['zeta'], info['lambda'], info['sources']
    it_z = zi['wall'] / zi['cycles']
    print('    zeta: %d iterations, %d of them with forward differences, %d amplitude steps, %.2f s; %.2f s per '
          'iteration = %.2f amplitude iterations of CCSDT (%.2f s each), %.2f s per amplitude step; the central '
          'step against the exact one at the last iterate: %.1e; vectors in a scratch file: %s; plan %.2f GB'
          % (zi['cycles'], zi['forward'], zi['steps'], zi['wall'], it_z, it_z / it_cc, it_cc,
             zi['wall'] / zi['steps'], zi['stencil_error'], zi['mapped'], zi['need'] / 1e9))
    print('    zeta, seconds per iteration: %s' % ' '.join('%.2f' % x for x in zi['timings']))
    print("    sources of lambda' (%s stencil, in blocks: %s): %.2f s = %.2f Lambda iterations"
          % (cc.response_stencil, si['blocks'], si['wall'], si['wall'] / it_l0))
    it_l2 = float(np.mean(li['timings'][1:]))
    print("    lambda': %d iterations (lambda: %d), %.2f s per iteration, setup %.2f s, %.2f s in all; in blocks: "
          '%s; plan %.2f GB' % (li['cycles'], n_l0, it_l2, li['setup'], li['wall'], li['blocks'], li['need'] / 1e9),
          flush=True)
    D2, t_d2 = timed('relaxed density of CCSDT(Q)_L', lambda: cc.make_rdm1(ao_repr=True, with_q='lambda'), ('dens',), 3)
    mu2 = dipole('CCSDT(Q)_L', D2)
    lambda_opt.kernel = kernel

    # ---- the other stencils of zeta
    if os.environ.get('ZC_STENCILS'):
        opts = solver.Options(conv_tol_normt=cc._conv_tol_normt(), max_cycle=cc.max_cycle, diis_space=cc.diis_space,
                              incore_complete=cc.incore_complete, max_memory=cc.max_memory)
        for stencil in ('central', 'exact'):
            ccsdt_opt.release_cached(eris)
            rz, t = timed('zeta, %s' % stencil, lambda: zeta_opt.kernel(eris, cc.t1, cc.t2, cc.t3, ql.q2, ql.q3, opts,
                                                                       stencil=stencil))
            d = np.linalg.norm(rz.z2 - cc.zeta2) + 6 * np.linalg.norm(np.asarray(rz.z3) - np.asarray(cc.zeta3))
            print('    %d iterations, %d amplitude steps, %.2f s per iteration = %.2f amplitude iterations; '
                  'stencil error %s; difference to the zeta of the front end %.1e (full norm; threshold %.0e)'
                  % (rz.ncycle, rz.steps, t / rz.ncycle, t / rz.ncycle / it_cc, rz.stencil_error, d,
                     opts.conv_tol_normt), flush=True)
            del rz
        ccsdt_opt.release_cached(eris)

    total0 = t_l0 + t_d0
    total1 = t_g + t_l1 + t_d1
    total2 = t_l0 + t_gl + t_s + t_d2
    print('\nsummary %s/%s %s, %d + %d, %d frozen, max_memory %s' % (name, basis, kind, no, nv, frozen, max_memory))
    print('  CCSDT iteration %.2f s; Lambda iteration %.2f s (CCSDT), %.2f s (with the sources of (Q)), %.2f s '
          "(lambda')" % (it_cc, it_l0, it_l1, it_l2))
    print('  energies: (Q) %.2f s, (Q)_L %.2f s (after Lambda: %.2f s)' % (t_q, t_ql, t_l0))
    print('  gradient passes: (Q) %.2f s = %.2f x (Q); (Q)_L %.2f s = %.2f x (Q)_L energy = %.2f x (Q)'
          % (t_g, t_g / t_q, t_gl, t_gl / t_ql, t_gl / t_q))
    print("  zeta %.2f s (%d iterations, %d amplitude steps; %.2f amplitude iterations per iteration); sources "
          "%.2f s; lambda' %.2f s (%d iterations)" % (zi['wall'], zi['cycles'], zi['steps'], it_z / it_cc, si['wall'],
                                                      li['wall'], li['cycles']))
    print('  relaxed density after the equations: CCSDT %.2f s, CCSDT(Q) %.2f s, CCSDT(Q)_L %.2f s'
          % (t_d0, t_d1, t_d2))
    print('  all of the property after the CCSDT energy: CCSDT %.2f s; CCSDT(Q) %.2f s; CCSDT(Q)_L %.2f s '
          '(of which Lambda of CCSDT, which E(Q)_L needs anyway: %.2f s) = %.1f CCSDT iterations'
          % (total0, total1, total2, t_l0, total2 / it_cc))
    print('  ... CCSDT(Q)_L over CCSDT %.2f, over CCSDT(Q) %.2f; what the property adds to the (Q)_L energy '
          '(Lambda and the energy pass not counted): %.2f s' % (total2 / total0, total2 / total1, total2 - t_l0))
    print('  dipole moment: CCSDT %.10f, CCSDT(Q) %.10f, CCSDT(Q)_L %.10f' % (mu0, mu1, mu2))
    print('  (Lambda of CCSDT solved a second time, not counted: %.2f s)' % t_l0b)


if __name__ == '__main__':
    main()
