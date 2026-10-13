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
NFROZEN = 4
# --dz on the command line: cc-pVDZ-DK (PySCF's; 56 spinors, 52 correlated) in the place of DYVTZ-SOG,
# files with the tag _dz
DZ = '--dz' in sys.argv
if DZ:
    sys.argv.remove('--dz')
TAG = '_dz' if DZ else ''


def run_scf(chk=None, verbose=4):
    from socutils.scf import spinor_hf
    from socutils.somf import x2cmp
    sog = 'cc-pvdz-dk' if DZ else runpy.run_path(BASIS)['O']
    mol = gto.M(atom='O 0 0 0; O 0 0 1.20752', basis={'O': sog}, verbose=verbose, nucmod='G', charge=2, spin=0)
    mf = spinor_hf.KRHF(mol)
    mf.with_x2c = x2cmp.X2CMP(mol, x2cmp='x2cmp', with_gaunt=True, with_breit=True)
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = 1e-13, 1e-9, 200
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
    sizes = [int(x) for x in sys.argv[2:]] or ([52] if DZ else [28, 30, 32, 116])
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
    for n in sizes:
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
        np.savez(name, h1=h1.astype(complex), eri=eri.astype(complex), ecore=ecore, nelec=12, ncore=no, nact=4,
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
