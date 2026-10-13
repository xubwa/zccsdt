'''O2(2+), closed shell: Kramers-restricted Hartree-Fock with the X2CMP
Hamiltonian (Gaunt and Breit in the mean field; Coulomb two-electron
integrals), Gaussian nuclei, r = 1.20752 Angstrom on the z axis, basis
DYVTZ-SOG (the file of the other sessions).  The canonical spinors adapted
to m_j (real coefficients), O 1s frozen (4 spinors), and the Hamiltonian of
truncated correlated spaces written as .npz for the DEA calculations and
for exact references on the same integrals (bench/dea/common.py: format).

usage: python o2_scf.py <outdir> [n ...]      n: correlated spinors (default 28 30 32 116)

The spinors of a file: the 10 occupied spinors of O2(2+) by energy, then
the virtual spinors by energy - the lowest four are the pi* spinors -, cut
between two levels (a level: a Kramers pair +-m_j).'''
import os
import runpy
import sys
import time

import numpy as np
from pyscf import gto, lib

BASIS = os.path.expanduser('~/scr4/zmr_runs/basis_opt/basis/pyscf/dyvtz_sog.py')
DYVDZ = os.path.expanduser('~/scr4/zmr_runs/o2_cbs/dz_basis/dyvdz_sog.py')
# On the command line, in the place of DYVTZ-SOG (files with a tag):
#   --dz             cc-pVDZ-DK (PySCF's; 56 spinors)                                     _dz
#   --basis dyvdz    DYVDZ-SOG (the j-adapted Dyall double zeta of the other sessions)    _dyvdz
#   --basis sogcv    DYVTZ-SOG plus the core-valence functions of cc-pCVTZ (the shells of
#                    cc-pCVTZ that cc-pVTZ does not have: tight 2s2p1d, uncontracted)     _sogcv
#   --ae             nothing frozen (O 1s correlated): tag _ae, 14 occupied spinors
DZ = '--dz' in sys.argv
if DZ:
    sys.argv.remove('--dz')
VARIANT = 'dz' if DZ else ''
if '--basis' in sys.argv:
    k = sys.argv.index('--basis')
    VARIANT = sys.argv[k + 1]
    del sys.argv[k:k + 2]
AE = '--ae' in sys.argv
if AE:
    sys.argv.remove('--ae')
NFROZEN = 0 if AE else 4
TAG = ('_' + VARIANT if VARIANT else '') + ('_ae' if AE else '')


def core_valence_shells():
    '''The shells of cc-pCVTZ of O that cc-pVTZ does not have, uncontracted.'''
    have = {(sh[0], round(float(pr[0]), 8)) for sh in gto.uncontract(gto.basis.load('cc-pvtz', 'O')) for pr in sh[1:]}
    out = []
    for sh in gto.uncontract(gto.basis.load('cc-pcvtz', 'O')):
        for pr in sh[1:]:
            if (sh[0], round(float(pr[0]), 8)) not in have:
                out.append([sh[0], 0, [float(pr[0]), 1.0]])
    return out


def add_primitives(shells, extra):
    '''``shells`` with the uncontracted primitives ``extra`` ([l, 0, [exponent, 1]]) in the form
    that the X2C step of a basis contracted per kappa takes: every shell of an l lists the same
    exponents (the new ones with zero coefficients), and the shell with kappa = 0 gets one more
    contraction per new exponent.  (New shells with other exponents next to the contracted ones
    made the X2CMP SCF diverge.)'''
    out = []
    for sh in shells:
        l, kappa, rows = sh[0], sh[1], [list(r) for r in sh[2:]]
        new = sorted((x[2][0] for x in extra if x[0] == l), reverse=True)
        ncol = len(rows[0]) - 1
        add_cols = len(new) if kappa == 0 else 0
        rows = [r + [0.0] * add_cols for r in rows]
        for k, e in enumerate(new):
            row = [e] + [0.0] * (ncol + add_cols)
            if kappa == 0:
                row[1 + ncol + k] = 1.0
            rows.append(row)
        rows.sort(key=lambda r: -r[0])
        out.append([l, kappa] + rows)
    assert all(any(sh[0] == x[0] and sh[1] == 0 for sh in shells) for x in extra)
    return out


def basis_of():
    if VARIANT == 'dz':
        return 'cc-pvdz-dk'
    if VARIANT == 'dyvdz':
        return runpy.run_path(DYVDZ)['O']
    sog = runpy.run_path(BASIS)['O']
    if VARIANT == 'sogcv':
        return add_primitives(sog, core_valence_shells())
    if VARIANT:
        raise ValueError(VARIANT)
    return sog


def run_scf(chk=None, verbose=4):
    from socutils.scf import spinor_hf
    from socutils.somf import x2cmp
    sog = basis_of()
    mol = gto.M(atom='O 0 0 0; O 0 0 1.20752', basis={'O': sog}, verbose=verbose, nucmod='G', charge=2, spin=0)
    mf = spinor_hf.KRHF(mol)
    mf.with_x2c = x2cmp.X2CMP(mol, x2cmp='x2cmp', with_gaunt=True, with_breit=True)
    # (with the tight functions of sogcv the gradient stays at 4e-9 while the energy is stationary to 1e-13)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-8 if VARIANT == 'sogcv' else 1e-9, 200
    if chk and os.path.exists(chk) and not os.path.exists(chk + '.done'):
        os.remove(chk)              # (left by a run that did not converge)
    if chk and os.path.exists(chk):
        # the SCF of an earlier run: the same spinors, phases included
        from pyscf.lib import chkfile
        d = chkfile.load(chk, 'scf')
        mf.mo_coeff, mf.mo_energy, mf.mo_occ, mf.e_tot = d['mo_coeff'], d['mo_energy'], d['mo_occ'], d['e_tot']
        mf.converged = True
        return mf
    if chk:
        mf.chkfile = chk
    mf.kernel()
    if not mf.converged:
        raise RuntimeError('SCF not converged')
    if chk:
        open(chk + '.done', 'w').close()
    return mf


def spinors(mf):
    '''(C, 2 m_j, parity, occupied by energy, virtual by energy, energies) of
    the SCF: the spinors adapted to the axis, real coefficients.'''
    from zccsdt import eris as E
    ax = E.axial_spinors(mf, with_mj=True)
    if ax is None:
        raise RuntimeError('the spinors are not adapted to the axis')
    C, level, mj = ax
    occ = np.asarray(mf.mo_occ) > 0
    e = np.asarray(mf.mo_energy)
    par = parities(mf.mol, C, mf.get_ovlp())
    assert abs(abs(par) - 1).max() < 1e-6, abs(abs(par) - 1).max()
    io = np.where(occ)[0][np.argsort(e[occ], kind='stable')]
    iv = np.where(~occ)[0][np.argsort(e[~occ], kind='stable')]
    return C, np.round(2 * mj).astype(np.int64), (par < 0).astype(np.int64), io, iv, e


def parities(mol, C, S):
    '''<p| inversion |p> of the spinors C (the inversion through the
    midpoint of the two equal atoms: the function of atom A goes to the
    same function of atom B with (-1)^l).'''
    n = mol.nao_2c()
    half = n // 2
    sign = []
    for b in range(mol.nbas):
        l, kappa = mol.bas_angular(b), mol.bas_kappa(b)
        dim = (4 * l + 2) if kappa == 0 else (2 * l if kappa > 0 else 2 * l + 2)
        sign += [(-1) ** l] * (dim * mol.bas_nctr(b))
    sign = np.array(sign, dtype=float)
    assert len(sign) == n
    perm = np.concatenate((np.arange(half, n), np.arange(half)))
    PC = (sign[:, None] * C)[perm]              # (P C)[mu at B] = (-1)^l C[mu at A]
    return np.einsum('mp,mn,np->p', C.conj(), S, PC).real


def main():
    outdir = sys.argv[1]
    sizes = [int(x) for x in sys.argv[2:]] or ([0] if TAG else [28, 30, 32, 116])
    os.makedirs(outdir, exist_ok=True)
    t0 = time.time()
    mf = run_scf(os.path.join(outdir, 'o2_2plus_krhf_x2cmp%s.chk' % TAG))
    mol = mf.mol
    print('SCF: E = %.12f  (%.0f s); nuclear repulsion %.12f' % (mf.e_tot, time.time() - t0, mol.energy_nuc()))
    from zccsdt import eris as E
    ax = E.axial_spinors(mf, with_mj=True)
    if ax is None:
        raise RuntimeError('the spinors are not adapted to the axis')
    C, level, mj = ax
    S = mf.get_ovlp()
    occ = np.asarray(mf.mo_occ) > 0
    e = np.asarray(mf.mo_energy)
    par = parities(mol, C, S)
    assert abs(abs(par) - 1).max() < 1e-6, abs(abs(par) - 1).max()
    par = (par < 0).astype(np.int64)
    om2 = np.round(2 * mj).astype(np.int64)
    io = np.where(occ)[0][np.argsort(e[occ], kind='stable')]
    iv = np.where(~occ)[0][np.argsort(e[~occ], kind='stable')]
    print('occupied spinors (energy, 2 m_j, parity):')
    for p in io:
        print('  %14.8f %3d %s' % (e[p], om2[p], 'gu'[par[p]]))
    print('virtual spinors:')
    for k, p in enumerate(iv):
        print('  %4d %14.8f %3d %s  level %d' % (k + 1, e[p], om2[p], 'gu'[par[p]], level[p]))
    # the Fock matrix of the spinors with the real coefficients, the frozen core
    dm = mf.make_rdm1(C, mf.mo_occ)
    fock_ao = mf.get_fock(dm=dm)
    e_scf = float(np.real(mf.energy_tot(dm=dm)))
    hcore = mf.get_hcore()
    cf = C[:, io[:NFROZEN]]
    dmc = cf @ cf.conj().T
    vc = mf.get_veff(mol, dmc)
    ecore = float(mol.energy_nuc() + np.einsum('ij,ji->', dmc, hcore).real + 0.5 * np.einsum('ij,ji->', dmc, vc).real)
    print('E(SCF) with the adapted spinors %.12f; frozen core: ecore = %.12f' % (e_scf, ecore))
    nv_all = len(iv)
    if VARIANT == 'sogcv':
        print('core-valence shells added (l, exponent): %s' % [(sh[0], sh[2][0]) for sh in core_valence_shells()])
    for n in sizes:
        n = n or len(io) - NFROZEN + nv_all         # (0: all of them)
        nv = n - (len(io) - NFROZEN)
        if nv < nv_all and abs(e[iv[nv]] - e[iv[nv - 1]]) < 1e-7:
            raise RuntimeError('%d spinors cut a degenerate level' % n)
        cols = np.concatenate((io[NFROZEN:], iv[:nv]))
        Ca = C[:, cols]
        h1 = Ca.conj().T @ (hcore + vc) @ Ca
        eri = E._mo_chemist(mol, Ca, 'spinor')
        f = Ca.conj().T @ fock_ao @ Ca
        no = len(io) - NFROZEN
        e_ref = ecore + np.einsum('ii->', h1[:no, :no]).real + 0.5 * (
            np.einsum('iijj->', eri[:no, :no, :no, :no]) - np.einsum('ijji->', eri[:no, :no, :no, :no])).real
        name = os.path.join(outdir, 'o2_hf2plus%s_n%d.npz' % (TAG, n))
        np.savez(name, h1=h1.astype(complex), eri=eri.astype(complex), ecore=ecore, nelec=no + 2, ncore=no, nact=4,
                 omega2=om2[cols], parity=par[cols], eps=f.diagonal().real, mo_coeff=Ca, e_scf=e_scf)
        print('n = %3d: %s  E(reference determinant) from the file %.12f (SCF %.12f, difference %.1e); '
              'max |imag| %.1e; pi*: 2 m_j %s parity %s; highest virtual kept %.6f, next %s  (%.0f s)'
              % (n, name, e_ref, e_scf, e_ref - e_scf, max(abs(h1.imag).max(), abs(eri.imag).max()),
                 om2[cols][no:no + 4], par[cols][no:no + 4], e[iv[nv - 1]],
                 '%.6f' % e[iv[nv]] if nv < nv_all else '-', time.time() - t0), flush=True)
        del eri


if __name__ == '__main__':
    lib.num_threads(int(os.environ.get('OMP_NUM_THREADS', 1)))
    main()
