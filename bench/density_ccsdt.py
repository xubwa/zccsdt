'''The relaxed one-particle density of CCSDT from packed amplitudes
(zccsdt.cc.density_opt): validation against a finite field, and timing.

usage: python bench/density_ccsdt.py validate MOLECULE BASIS KIND [NDROP]
           converged CCSDT (the packed solver), Lambda from the dense
           reference (zccsdt.cc.lambda_ref: small cases only), the relaxed
           density, and the dipole derivative of E(SCF) + E_corr(CCSDT)
           with the SCF repeated in the field (four-point differences).
           NDROP: the number of frozen occupied spin orbitals (the lowest)
           and, after a comma, of dropped virtual ones (the highest): 2 or 2,4.
       python bench/density_ccsdt.py time MOLECULE BASIS KIND [NITER] [NDROP]
           the time of the densities, the orbital gradient and the Z-vector
           equations next to that of a CCSDT iteration.  Lambda is the
           conjugate of t here (the cost does not depend on the numbers; the
           densities are not those of the molecule).
  MOLECULE: h2o | hf | n2 | hbr      KIND: ghf (RHF orbitals as spin orbitals)
  | ghfc (the same, every pair mixed by its own SU(2) matrix: complex) | x2c
environment: ZC_MAXMEM: max_memory in MB (default: 80% of what is available);
             ZC_DENS_PLAN: 'NB,MB' - the plan of the densities given (time):
             NB occupied pairs per batch, slabs of MB megabytes of dressed <ab||de>
'''
import os
import sys
import threading
import time

import numpy as np
from pyscf import gto, scf, x2c

from zccsdt import eris as E, memory, timing
from zccsdt.cc import ccsdtq_ref as R, ccsdtq_terms as Q, density_opt as D, lambda_ref as LR, solver

GEOM = {
    'h2o': 'O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587',
    'hf': 'H 0 0 0; F 0 0 0.9168',
    'n2': 'N 0 0 0; N 0 0 1.0977',
    'hbr': 'H 0 0 0; Br 0 0 1.4145',
}


class MemWatch(threading.Thread):
    '''Peak anonymous resident memory of this process, sampled.'''

    def __init__(self, dt=0.05):
        super().__init__(daemon=True)
        self.dt, self.anon, self._done = dt, 0, threading.Event()

    def run(self):
        while not self._done.wait(self.dt):
            with open('/proc/self/status') as f:
                kb = {l.split(':')[0]: int(l.split()[1]) for l in f if l.startswith('Rss')}
            self.anon = max(self.anon, kb.get('RssAnon', 0))

    def stop(self):
        self._done.set()
        self.join()
        return self.anon * 1024


def spin_mixed(ghf, seed=4):
    '''The orbitals with the two spin orbitals of every spatial orbital
    mixed by a random SU(2) matrix (tests/test_ccsdtq_ref.py).'''
    rng = np.random.default_rng(seed)
    n = ghf.mo_coeff.shape[1]
    u = np.zeros((n, n), dtype=complex)
    order = np.argsort(ghf.mo_energy, kind='stable')
    for k in range(0, n, 2):
        p, q = order[k], order[k + 1]
        a = rng.standard_normal(4)
        a /= np.linalg.norm(a)
        u[np.ix_([p, q], [p, q])] = [[a[0] + 1j * a[1], a[2] + 1j * a[3]],
                                     [-a[2] + 1j * a[3], a[0] - 1j * a[1]]]
    return ghf.mo_coeff @ u


def make_scf(mol, kind, eps=0.0):
    '''(SCF object, orbitals, the operator z in its AO basis) in the field
    eps z, converged tightly.'''
    with mol.with_common_orig((0, 0, 0)):
        z = mol.intor('int1e_r_spinor' if kind == 'x2c' else 'int1e_r')[2]
    if kind == 'x2c':
        mf = x2c.UHF(mol)
        h0 = mf.get_hcore()
        mf.get_hcore = lambda *args: h0 + eps * z
        mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-10, 300
        mf.kernel()
        assert mf.converged
        return mf, mf.mo_coeff, z
    rhf = scf.RHF(mol)
    h0 = rhf.get_hcore()
    rhf.get_hcore = lambda *args: h0 + eps * z
    rhf.conv_tol, rhf.conv_tol_grad, rhf.max_cycle = 1e-13, 1e-10, 300
    rhf.kernel()
    assert rhf.converged
    mf = scf.addons.convert_to_ghf(rhf)
    h2 = np.kron(np.eye(2), h0 + eps * z)
    mf.get_hcore = lambda *args: h2
    return mf, (spin_mixed(mf) if kind == 'ghfc' else mf.mo_coeff), np.kron(np.eye(2), z)


def spaces(mf, C, nfrozen, ndrop):
    '''The columns of C by energy: (frozen occupied, active occupied, active
    virtual, dropped virtual).'''
    occ, vir = np.where(mf.mo_occ > 0)[0], np.where(mf.mo_occ == 0)[0]
    return C[:, occ[:nfrozen]], C[:, occ[nfrozen:]], C[:, vir[:len(vir) - ndrop]], C[:, vir[len(vir) - ndrop:]]


def integrals(mf, C, nfrozen, ndrop, **kw):
    Cf, Co, Cv, Cd = spaces(mf, C, nfrozen, ndrop)
    cols = np.hstack((Cf, Co, Cv))
    occ = np.concatenate((np.ones(Cf.shape[1] + Co.shape[1]), np.zeros(Cv.shape[1])))
    return E.from_scf(mf, mo_coeff=cols, mo_occ=occ, frozen=nfrozen, **kw), (Cf, Co, Cv, Cd)


def validate(mol, kind, nfrozen, ndrop):
    mem = {'max_memory': float(os.environ['ZC_MAXMEM'])} if os.environ.get('ZC_MAXMEM') else {}
    opts = solver.Options(conv_tol=1e-13, conv_tol_normt=1e-10, incore_complete=True)

    def energy(eps):
        mf, C, z = make_scf(mol, kind, eps)
        eris, parts = integrals(mf, C, nfrozen, ndrop)
        r = solver.kernel(eris, 'ccsdt', opts)
        assert r.converged
        return mf.e_tot + float(np.real(r.e_corr)), (mf, C, z, eris, parts, r)

    t0 = time.perf_counter()
    e0, (mf, C, z, eris, (Cf, Co, Cv, Cd), r) = energy(0.0)
    nocc, nvir = r.t1.shape
    print('%d occupied and %d virtual spin orbitals (%d frozen, %d dropped), %s; E(SCF) %.12f  E(CCSDT) %.12f  (%.1f s)'
          % (nocc, nvir, nfrozen, ndrop, eris.dtype, mf.e_tot, e0, time.perf_counter() - t0))
    t0 = time.perf_counter()
    fock, g, mo_e, _ = R.integrals(eris)
    t3 = Q.expand(np.asarray(r.t3).reshape(len(Q.tuples(nocc, 3)), -1), nocc, nvir, 3)
    rl = LR.kernel(fock, g, mo_e, nocc, [r.t1, r.t2, t3], conv_tol_normt=1e-11)
    assert rl.converged
    print('Lambda (dense reference): %d iterations, %.1f s' % (rl.ncycle, time.perf_counter() - t0))
    t0 = time.perf_counter()
    gnh = LR.orbital_gradient(fock, g, nocc, [r.t1, r.t2, t3], rl.lams, dressed=True)
    d1 = LR.densities([r.t1, r.t2, t3], rl.lams, nocc)[0]
    t_dense = time.perf_counter() - t0
    del fock, g, t3
    l1, l2, l3 = rl.lams[0], rl.lams[1], Q.unique(rl.lams[2], 3)
    t0 = time.perf_counter()
    rel = D.relaxed_rdm1(mf, eris, r.t1, r.t2, r.t3, l1, l2, l3, np.hstack((Co, Cv)),
                         frozen_occ=Cf if nfrozen else None, dropped_vir=Cd if ndrop else None,
                         tol=1e-11, **mem)
    print('densities, gradient and Z-vector: %.1f s (the dense tables: %.1f s); |Gnh| %.2e, off the '
          'dense table %.1e; d1 off %.1e' % (time.perf_counter() - t0, t_dense, abs(gnh).max(),
                                             abs(rel.result.gnh - gnh).max(), abs(rel.result.d1 - d1).max()))
    Ca = rel.mo_coeff
    Vmo = Ca.conj().T @ z @ Ca
    relaxed = np.einsum('pq,qp', Vmo, rel.relaxed).real
    unrelaxed = np.einsum('pq,qp', Vmo, rel.unrelaxed).real
    print('trace %.12f  Hermitian to %.1e' % (np.trace(rel.relaxed).real, abs(rel.relaxed - rel.relaxed.conj().T).max()))
    for eps in (2e-3, 1e-3):
        e = {k: energy(k * eps)[0] for k in (-2, -1, 1, 2)}
        fd = (8 * (e[1] - e[-1]) - (e[2] - e[-2])) / (12 * eps)
        print('step %.0e: unrelaxed %.12f  relaxed %.12f  finite field %.12f  difference %.1e'
              % (eps, unrelaxed, relaxed, fd, relaxed - fd))


def tree(times, parent=(), indent=1):
    for name, wall, cpu, calls in timing.children(times, parent):
        print('%s%-28s %8.2f s   %4.1f cores   %d calls' % ('  ' * indent, name, wall, cpu / max(wall, 1e-9), calls))
        tree(times, tuple(parent) + (name,), indent + 1)


def profile(mol, kind, niter, nfrozen=0, ndrop=0):
    mem = {'max_memory': float(os.environ['ZC_MAXMEM'])} if os.environ.get('ZC_MAXMEM') else {}
    mf, C, z = make_scf(mol, kind)
    t0 = time.perf_counter()
    eris, (Cf, Co, Cv, Cd) = integrals(mf, C, nfrozen, ndrop, packed_vvvv=kind != 'ghfc', **mem)
    nocc, nvir = eris.nocc, eris.nmo - eris.nocc
    print('%d occupied and %d virtual spin orbitals (%d frozen, %d dropped), %s; integrals %.1f s'
          % (nocc, nvir, nfrozen, ndrop, eris.dtype, time.perf_counter() - t0))
    r0 = solver.kernel(eris, 'ccsdt', solver.Options(max_cycle=1, verbose=True, **mem))
    timing.reset()
    r = solver.kernel(eris, 'ccsdt', solver.Options(max_cycle=niter, verbose=True, plan=r0.plan, **mem),
                      t1=r0.t1, t2=r0.t2, t3=r0.t3)
    t_iter = float(np.mean(r.timings))
    print('CCSDT iteration: %.2f s (mean of %d after a first one of %.2f s); plan %s' % (t_iter, len(r.timings), r0.timings[0], r.plan))
    for name, sec, calls in timing.report(len(r.timings))[:8]:
        print('    %-16s %8.2f s' % (name, sec))
    from zccsdt.cc import ccsdt_opt
    ccsdt_opt.release_cached(eris)
    t1, t2, t3 = r.t1, r.t2, np.asarray(r.t3)
    l1, l2, l3 = t1.conj(), t2.conj(), t3.conj()
    print('T3 and Lambda3: %.2f GB each' % (t3.nbytes / 1e9))
    dmem = dict(mem)
    if os.environ.get('ZC_DENS_PLAN'):      # 'pairs per batch, MB of dressed rows per slab'
        nb, slab = os.environ['ZC_DENS_PLAN'].split(',')
        isz, nvp = t3.dtype.itemsize, nvir * (nvir - 1) // 2
        slab = max(1, int(float(slab) * 1e6) // (isz * nvp)) * isz * nvp
        dmem = {'plan': D.Plan(nb=int(nb), slab_bytes=slab, need=D.fixed_bytes(nocc, nvir, isz, nfrozen + ndrop) + max(
            int(nb) * D.pair_bytes(nocc, nvir, isz, nfrozen + ndrop), slab))}
    for attempt in range(2):
        timing.reset()
        base = memory.resident()
        watch = MemWatch()
        watch.start()
        t0, c0 = time.perf_counter(), time.process_time()
        rel = D.relaxed_rdm1(mf, eris, t1, t2, t3, l1, l2, l3, np.hstack((Co, Cv)),
                             frozen_occ=Cf if nfrozen else None, dropped_vir=Cd if ndrop else None,
                             tol=1e-9, **dmem)
        wall, cpu = time.perf_counter() - t0, time.process_time() - c0
        peak = watch.stop()
        print('run %d: densities + orbital gradient + Z-vector %.2f s, %.1f cores busy = %.2f CCSDT iterations; '
              'peak anonymous memory %.2f GB (%.2f GB before the call); plan: %d pairs per batch, slab %s, need %.2f GB'
              % (attempt + 1, wall, cpu / wall, wall / t_iter, peak / 1e9, base / 1e9, rel.result.plan.nb,
                 rel.result.plan.slab_bytes, rel.result.plan.need / 1e9))
        tree(timing.snapshot())
    print('Z-vector converged: %s; trace of the density %.10f' % (rel.converged, np.trace(rel.relaxed).real))


def main():
    mode, name, basis, kind = sys.argv[1:5]
    mol = gto.M(atom=GEOM[name], basis=basis, verbose=0)
    if mode == 'validate':
        fr = (sys.argv[5] if len(sys.argv) > 5 else '0').split(',')
        validate(mol, kind, int(fr[0]), int(fr[1]) if len(fr) > 1 else 0)
    else:
        fr = (sys.argv[6] if len(sys.argv) > 6 else '0').split(',')
        profile(mol, kind, int(sys.argv[5]) if len(sys.argv) > 5 else 2, int(fr[0]), int(fr[1]) if len(fr) > 1 else 0)


if __name__ == '__main__':
    main()
