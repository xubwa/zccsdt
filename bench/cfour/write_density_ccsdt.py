'''The relaxed one-particle density of CCSDT for CFOUR's property program:
in the directory of a CFOUR run (xjoda, xsox2catm, xsox2chf; ZMAT, GENBAS),
CCSDT on CFOUR's own spinors (zccsdt.cfour.reference: no SCF here, the
one-electron Hamiltonian from CFOUR's Fock matrix), Lambda, the density and
the orbital gradient from the packed amplitudes (zccsdt.cc.density_opt), the
Z-vector equations, and the relaxed density in the basis of CFOUR's spinors
written as the file ``denfinal`` that xsox2cprop evaluates all its
operators with (the full density, the SCF part included, D[p,q] = <q+ p>:
bench/cfour/write_density.py has that pinned against CFOUR's own file for
CCSD).

Lambda is that of the dense reference (zccsdt.cc.lambda_ref) until the
solver on packed amplitudes is there: small cases only.

usage: python bench/cfour/write_density_ccsdt.py RUNDIR [--ccsd] [--run]
  --ccsd: the CCSD limit of the same code path - the amplitudes and lambda
          of zccsdt.zccsd.DirectZCCSD (in its own Kramers-paired spinors),
          t3 = lambda3 = None; the density is compared with that of
          zccsd.zvector.kernel and, if CFOUR's own denfinal of a CCSD run is
          there (kept as denfinal.cfour), with it element by element
  --run:  xsox2cprop afterwards (from $CFOUR_BIN or ~/apps/cfour-x2cso/bin),
          its output as xsox2cprop.zccsdt.out, and every property it prints
          next to the trace of CFOUR's operator (socutils.prop.cfour) with
          the density made here
Frozen and dropped spinors are those of the CFOUR run (DROPMO, %vircut).
socutils has to be on the path.
'''
import os
import shutil
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import props                                                            # noqa: E402

from zccsdt import cfour, eris as E                                     # noqa: E402
from zccsdt.cc import ccsdtq_ref as R, ccsdtq_terms as Q, density_opt, lambda_ref, solver   # noqa: E402


def amplitudes_ccsdt(mf, C, nfrozen, ndropped, log):
    '''(eris, t1, t2, t3, l1, l2, l3, the spinors [frozen, active occupied,
    active virtual, dropped]) of CCSDT on the spinors C of ``mf``.'''
    n = C.shape[1]
    nocc = int((mf.mo_occ > 0).sum())
    cols = np.arange(n - ndropped)
    eris = E.from_scf(mf, mo_coeff=C[:, cols], mo_occ=mf.mo_occ[cols], frozen=nfrozen, packed_vvvv=True)
    t0 = time.perf_counter()
    r = solver.kernel(eris, 'ccsdt', solver.Options(conv_tol=1e-12, conv_tol_normt=1e-9))
    if not r.converged:
        raise RuntimeError('CCSDT did not converge')
    log('E(CCSDT) = %.12f  correlation %.12f  (%d iterations, %.1f s)'
        % (mf.e_tot + float(np.real(r.e_corr)), float(np.real(r.e_corr)), r.ncycle, time.perf_counter() - t0))
    # Lambda: the dense reference (full arrays)
    t0 = time.perf_counter()
    o, v = r.t1.shape
    fock, g, mo_e, _ = R.integrals(eris)
    t3 = Q.expand(np.asarray(r.t3).reshape(len(Q.tuples(o, 3)), -1), o, v, 3)
    rl = lambda_ref.kernel(fock, g, mo_e, o, [r.t1, r.t2, t3], conv_tol_normt=1e-10)
    if not rl.converged:
        raise RuntimeError('the Lambda equations of CCSDT did not converge')
    log('Lambda (dense reference): %d iterations, %.1f s' % (rl.ncycle, time.perf_counter() - t0))
    parts = (C[:, :nfrozen], C[:, nfrozen:nocc], C[:, nocc:n - ndropped], C[:, n - ndropped:])
    return eris, r.t1, r.t2, r.t3, rl.lams[0], rl.lams[1], Q.unique(rl.lams[2], 3), parts


def amplitudes_ccsd(mf, nfrozen, ndropped, log):
    '''The same of the spinor CCSD (DirectZCCSD, its own spinors), t3 and
    lambda3 None; and its relaxed density (zccsd.zvector.kernel) with its
    spinors.'''
    from zccsdt.zccsd import DirectZCCSD, zvector
    n = len(mf.mo_energy)
    nocc = int((mf.mo_occ > 0).sum())
    frozen = list(range(nfrozen)) + list(range(n - ndropped, n)) if ndropped else nfrozen
    mycc = DirectZCCSD(mf, frozen=frozen)
    mycc.verbose, mycc.conv_tol, mycc.conv_tol_normt, mycc.max_cycle = 0, 1e-12, 1e-10, 200
    mycc.kernel()
    l1, l2 = mycc.solve_lambda()
    log('E(CCSD) = %.12f  correlation %.12f' % (mf.e_tot + mycc.e_corr, mycc.e_corr))
    ref = zvector.kernel(mycc, l1=l1, l2=l2, mf=mf, tol=1e-11)
    C = zvector.full_space(mycc)[0]
    cols = np.arange(n - ndropped)
    eris = E.from_scf(mf, mo_coeff=C[:, cols], mo_occ=mf.mo_occ[cols], frozen=nfrozen, packed_vvvv=True)
    parts = (C[:, :nfrozen], C[:, nfrozen:nocc], C[:, nocc:n - ndropped], C[:, n - ndropped:])
    return (eris, mycc.t1, mycc.t2, None, l1, l2, None, parts), (ref[0], ref[1], C)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    d = args[0]
    log = lambda s: print(s, flush=True)                                # noqa: E731
    mf, nfrozen, ndropped = cfour.reference(d)
    mol, C0 = mf.mol, np.asarray(mf.mo_coeff)
    n = C0.shape[1]
    S = mol.intor('int1e_ovlp_spinor')
    log('%s: %d spinors, %d occupied; %d frozen, %d dropped; E(SCF) of CFOUR %.12f'
        % (d, n, int((mf.mo_occ > 0).sum()), nfrozen, ndropped, mf.e_tot))
    other = None
    if '--ccsd' in sys.argv:
        amps, other = amplitudes_ccsd(mf, nfrozen, ndropped, log)
    else:
        amps = amplitudes_ccsdt(mf, C0, nfrozen, ndropped, log)
    eris, t1, t2, t3, l1, l2, l3, (Cf, Co, Cv, Cd) = amps
    t0 = time.perf_counter()
    rel = density_opt.relaxed_rdm1(mf, eris, t1, t2, t3, l1, l2, l3, np.hstack((Co, Cv)),
                                   frozen_occ=Cf if nfrozen else None, dropped_vir=Cd if ndropped else None,
                                   tol=1e-11)
    log('densities, orbital gradient and Z-vector equations: %.1f s (converged: %s)'
        % (time.perf_counter() - t0, rel.converged))
    # through the AO basis into CFOUR's spinors
    Ca = rel.mo_coeff
    ao = {'scf': C0[:, mf.mo_occ > 0] @ C0[:, mf.mo_occ > 0].conj().T,
          'relaxed': Ca @ rel.relaxed @ Ca.conj().T, 'unrelaxed': Ca @ rel.unrelaxed @ Ca.conj().T}
    own = C0.conj().T @ S @ ao['relaxed'] @ S @ C0
    for name in ('unrelaxed', 'relaxed'):
        dip = [-np.einsum('pq,qp->', mol.intor('int1e_r_spinor')[k], ao[name]).real for k in range(3)]
        log('%-9s trace %.10f  electronic dipole (the operator of the spinor basis, no picture change) %s'
            % (name, np.trace(S @ ao[name]).real, np.array2string(np.array(dip), precision=10)))
    if other is not None:
        ref = other[2] @ other[0] @ other[2].conj().T
        log('CCSD limit: relaxed density against zccsd.zvector.kernel %.1e, unrelaxed %.1e (AO basis)'
            % (abs(ao['relaxed'] - ref).max(), abs(ao['unrelaxed'] - other[2] @ other[1] @ other[2].conj().T).max()))
    path = os.path.join(d, 'denfinal')
    if os.path.exists(path) and not os.path.exists(path + '.cfour'):
        shutil.copy(path, path + '.cfour')
    if other is not None and os.path.exists(path + '.cfour'):
        theirs = cfour._complex(path + '.cfour', (n, n))
        log("CFOUR's own denfinal: trace %.10f; the density made here differs by at most %.2e"
            % (np.trace(theirs).real, abs(own - theirs).max()))
    cfour._write_complex(os.path.join(d, 'denfinal.zccsdt'), own)
    np.savez(os.path.join(d, 'densities_zccsdt.npz'), **ao)
    log('denfinal.zccsdt written (Hermitian to %.1e); the densities in the spinor AO basis: densities_zccsdt.npz'
        % abs(own - own.conj().T).max())
    if '--run' not in sys.argv:
        return
    exe = os.path.join(os.environ.get('CFOUR_BIN', os.path.expanduser('~/apps/cfour-x2cso/bin')), 'xsox2cprop')
    shutil.copy(os.path.join(d, 'denfinal.zccsdt'), path)
    out = os.path.join(d, 'xsox2cprop.zccsdt.out')
    with open(out, 'w') as f:
        subprocess.run([exe], cwd=d, stdout=f, stderr=subprocess.STDOUT, check=False)
    if os.path.exists(path + '.cfour'):
        shutil.copy(path + '.cfour', path)
    log('xsox2cprop with the density made here: %s' % out)
    # CFOUR's operators traced here with the same densities
    from socutils.prop import cfour as pc
    store = pc.CfourPropStore(d, mol, atom_map=list(range(mol.natm)))
    ops = {}
    for name in store.available():
        if any(name.startswith(k) for k in props.TABLE):
            for key, atom, comp in props.items(store, name):
                ops[key] = store.op(name, atom=atom, comp=comp)
    np.savez(os.path.join(d, 'operators_zccsdt.npz'), **ops)
    props.compare(os.path.join(d, 'operators_zccsdt.npz'), os.path.join(d, 'densities_zccsdt.npz'), out)


if __name__ == '__main__':
    main()
