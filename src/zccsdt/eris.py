'''Spin-orbital MO integral container and generators.

The container uses the PySCF GCCSD field names so that the CC code accepts
either our own integrals or PySCF's ``gccsd._PhysicistsERIs``:
antisymmetrized physicist integrals <pq||rs> = <pq|rs> - <pq|sr> in the blocks
oooo, ooov, oovv, ovov, ovvo, ovvv, vvvv, plus the MO Fock matrix.

Generators accept any PySCF-style SCF object whose orbitals are either
  * GHF-type: mo_coeff has 2*nao rows (alpha block on top of beta block),
    two-electron integrals from the real AO 'int2e'; or
  * spinor-type: mo_coeff has mol.nao_2c() rows (j-adapted spinors, e.g.
    pyscf.x2c or socutils X2CAMF), integrals from 'int2e_spinor'.
Real and complex orbitals are handled by the same code.
'''

from dataclasses import dataclass
import numpy as np
import scipy.linalg

BLOCKS = ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv', 'vvvv')


@dataclass
class ERIs:
    nocc: int
    fock: np.ndarray
    mo_energy: np.ndarray
    e_hf: float
    oooo: np.ndarray
    ooov: np.ndarray
    oovv: np.ndarray
    ovov: np.ndarray
    ovvo: np.ndarray
    ovvv: np.ndarray
    vvvv: np.ndarray
    e_nuc: float = 0.0
    source: str = ''

    @property
    def nmo(self):
        return self.fock.shape[0]

    @property
    def dtype(self):
        return np.result_type(self.fock, self.oovv)


def _mo_chemist(mol, C, kind):
    '''Full MO chemist-notation integrals (pq|rs) for the columns of C.'''
    if kind == 'spinor':
        eri = mol.intor('int2e_spinor')
        blocks = [C]
    else:
        nao = mol.nao_nr()
        eri = mol.intor('int2e', aosym='s1')
        blocks = [C[:nao], C[nao:]]
    n = C.shape[1]
    out = np.zeros((n, n, n, n), dtype=np.result_type(C, eri))
    for ca in blocks:
        # (pq|ls): bra pair transformed within one spin block
        half = np.einsum('mp,mnls,nq->pqls', ca.conj(), eri, ca, optimize=True)
        for cb in blocks:
            out += np.einsum('pqls,lr,st->pqrt', half, cb.conj(), cb, optimize=True)
    return out


def _orbital_kind(mf):
    ''''ghf' for PySCF GHF objects (spin-orbital AO basis), 'spinor' otherwise
    (j-adapted spinor AO basis).  The mo_coeff shape cannot tell the two apart:
    for spherical bases nao_2c == 2 * nao_nr.'''
    from pyscf.scf import ghf
    return 'ghf' if isinstance(mf, ghf.GHF) else 'spinor'


def from_scf(mf, mo_coeff=None, mo_occ=None, frozen=0, kind=None):
    '''Build spin-orbital ERIs from a converged SCF object.

    ``mo_coeff`` may be any orbitals spanning the same occupied space as the
    SCF (e.g. after an occ-occ / vir-vir rotation); the Fock matrix is then
    non-diagonal and is used as such.  Occupied orbitals are the columns with
    mo_occ > 0; the first ``frozen`` occupied orbitals are dropped.
    '''
    if mo_coeff is None:
        mo_coeff = mf.mo_coeff
    if mo_occ is None:
        mo_occ = mf.mo_occ
    mo_coeff = np.asarray(mo_coeff)
    kind = kind or _orbital_kind(mf)
    occ = np.where(mo_occ > 0)[0]
    vir = np.where(mo_occ == 0)[0]
    dm = mf.make_rdm1(mo_coeff, mo_occ)
    fock_ao = mf.get_fock(dm=dm)
    e_hf = float(np.real(mf.energy_tot(dm=dm)))

    C = np.hstack((mo_coeff[:, occ[frozen:]], mo_coeff[:, vir]))
    nocc = len(occ) - frozen
    fock = C.conj().T @ fock_ao @ C
    eri = _mo_chemist(mf.mol, C, kind)
    g = eri.transpose(0, 2, 1, 3)              # <pq|rs> = (pr|qs)
    g = g - g.transpose(0, 1, 3, 2)            # <pq||rs>
    # enforce the bra antisymmetry exactly as well (the MO transformation
    # leaves ~1e-14 noise in <pp||rs>, which packed storage would drop)
    g = 0.5 * (g - g.transpose(1, 0, 2, 3))
    o, v = slice(0, nocc), slice(nocc, None)
    blocks = {name: np.ascontiguousarray(
        g[tuple(o if c == 'o' else v for c in name)]) for name in BLOCKS}
    if not np.iscomplexobj(mo_coeff) and kind == 'ghf':
        fock = fock.real
        blocks = {k: b.real for k, b in blocks.items()}
    return ERIs(nocc=nocc, fock=fock, mo_energy=fock.diagonal().real.copy(),
                e_hf=e_hf, e_nuc=float(mf.energy_nuc()),
                source='%s:%s' % (type(mf).__name__, kind), **blocks)


def reference_energy(eris):
    '''E_ref = E_nuc + sum_i f_ii - 1/2 sum_ij <ij||ij> (active space only).'''
    n = eris.nocc
    f = np.trace(eris.fock[:n, :n])
    g = np.einsum('ijij->', eris.oooo)
    return float(np.real(eris.e_nuc + f - 0.5 * g))


def random_unitary(n, rng):
    '''Haar-like random complex unitary of size n.'''
    z = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    q, r = scipy.linalg.qr(z)
    return q * (np.diag(r) / abs(np.diag(r)))


def rotate_occ_vir(mo_coeff, mo_occ, seed=0, frozen=0):
    '''Apply independent random complex unitaries to the active occupied and to
    the virtual orbitals (the first ``frozen`` occupied orbitals are left
    untouched, so the frozen space is unchanged).  CC energies are invariant
    under this transformation, while the integrals become genuinely complex
    (alpha/beta mixed for GHF) and the Fock matrix non-diagonal.'''
    rng = np.random.default_rng(seed)
    C = np.array(mo_coeff, dtype=complex)
    occ = np.where(mo_occ > 0)[0][frozen:]
    vir = np.where(mo_occ == 0)[0]
    for idx in (occ, vir):
        C[:, idx] = C[:, idx] @ random_unitary(len(idx), rng)
    return C
