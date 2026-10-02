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
    vvvv: np.ndarray            # None if only the packed form _vvvv_pp is kept
    e_nuc: float = 0.0
    source: str = ''
    spin_occ: np.ndarray = None    # 0/1 = alpha/beta per spin orbital (alpha first),
    spin_vir: np.ndarray = None    # only for pure-spin orbitals and spin_order=True

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


def _chem_block(mol, kind, C1, C2, C3, C4, max_memory=None):
    '''Chemist-notation MO block (12|34) via pyscf.ao2mo (no full n^4 tensor).
    ``max_memory`` (MB): work memory of the transformation.'''
    from pyscf import ao2mo
    from zccsdt import memory
    shape = (C1.shape[1], C2.shape[1], C3.shape[1], C4.shape[1])
    kw = {'max_memory': memory.ao2mo_memory() if max_memory is None else max_memory}
    if kind == 'spinor':
        out = ao2mo.general(mol, (C1, C2, C3, C4), intor='int2e_spinor', **kw)
        return np.asarray(out).reshape(shape)
    nao = mol.nao_nr()
    out = 0
    for s1 in (slice(0, nao), slice(nao, None)):
        for s2 in (slice(0, nao), slice(nao, None)):
            out = out + ao2mo.general(mol, (C1[s1], C2[s1], C3[s2], C4[s2]),
                                      compact=False, **kw).reshape(shape)
    return out


def _blocks_by_ao2mo(mol, kind, Co, Cv, vir_batch, max_memory=None):
    '''Antisymmetrized blocks from chemist blocks; vvvv returned packed over
    a<b, c<d, computed in batches of a.  With a memory budget (``max_memory``
    in MB for the whole process; default: 80% of what is available) the
    batches shrink to fit, and the packed vvvv is a mapped array of a scratch
    file when it does not fit twice (bare and dressed); the scratch object is
    returned as well (None if in memory).'''
    from zccsdt import memory
    from zccsdt.scratch import Scratch
    # (PySCF takes up to its max_memory for buffers: bound every transformation)
    ch = lambda a, b, c, d: _chem_block(mol, kind, a, b, c, d,       # noqa: E731
                                        memory.ao2mo_memory(max_memory))
    oooo = ch(Co, Co, Co, Co)                         # (ij|kl)
    ooov = ch(Co, Co, Co, Cv)                         # (ij|ka)
    ovov = ch(Co, Cv, Co, Cv)                         # (ia|jb)
    oovv = ch(Co, Co, Cv, Cv)                         # (ij|ab)
    ovvo = ch(Co, Cv, Cv, Co)                         # (ia|bj)
    ovvv = ch(Co, Cv, Cv, Cv)                         # (ia|bc)
    g = {}
    g['oooo'] = oooo.transpose(0, 2, 1, 3) - oooo.transpose(0, 2, 3, 1)
    # <ij||ka> = (ik|ja) - (ia|jk);  (ia|jk) = (jk|ia)
    g['ooov'] = np.einsum('ikja->ijka', ooov) - np.einsum('jkia->ijka', ooov)
    g['oovv'] = ovov.transpose(0, 2, 1, 3) - ovov.transpose(0, 2, 3, 1)
    # <ia||jb> = (ij|ab) - (ib|aj)
    g['ovov'] = oovv.transpose(0, 2, 1, 3) - ovvo.transpose(0, 2, 3, 1)
    # <ia||bj> = (ib|aj) - (ij|ab)
    g['ovvo'] = ovvo.transpose(0, 2, 1, 3) - oovv.transpose(0, 2, 3, 1)
    # <ia||bc> = (ib|ac) - (ic|ab)
    g['ovvv'] = ovvv.transpose(0, 2, 1, 3) - ovvv.transpose(0, 2, 3, 1)
    g = {k: np.ascontiguousarray(0.5 * (b - b.transpose(1, 0, 2, 3)))
         if k[0] == k[1] else np.ascontiguousarray(b) for k, b in g.items()}
    del oooo, ooov, ovov, oovv, ovvo, ovvv
    nv = Cv.shape[1]
    pa, pb = np.triu_indices(nv, 1)
    dtype = np.result_type(g['oooo'], Cv)
    isz = np.dtype(dtype).itemsize
    pp = len(pa) * len(pa) * isz
    # one a of (ac|bd): the chemist block, its antisymmetrized form and the
    # buffers of the transformation
    per_a = 4 * nv ** 3 * isz
    budget = memory.budget(max_memory)
    scr = None
    if budget is not None:
        if 2 * pp + min(vir_batch, nv) * per_a > budget:
            scr = Scratch(prefix='zccsdt_vvvv_', what='the packed <ab||cd> integrals',
                          hint=' or give the calculation more memory (max_memory)')
        else:
            budget -= pp
        vir_batch = int(max(1, min(vir_batch, budget // 2 // max(per_a, 1))))
    vpp = np.zeros((len(pa), len(pa)), dtype=dtype) if scr is None else \
        scr.create('vvvv', (len(pa), len(pa)), dtype)
    from zccsdt import scratch
    hints = scratch.hints(scr is not None)          # memory is short: say what is done
    try:
        row = 0
        for a0 in range(0, nv, vir_batch):
            a1 = min(a0 + vir_batch, nv)
            row0 = row
            X = ch(Cv[:, a0:a1], Cv, Cv, Cv)                      # (ac|bd) as [a, c, b, d]
            for a in range(a0, a1):
                Y = X[a - a0]
                G = Y.transpose(1, 0, 2) - Y.transpose(1, 2, 0)      # <ab||cd> as [b, c, d]
                nb = nv - a - 1
                vpp[row:row + nb] = G[a + 1:][:, pa, pb]
                row += nb
            del X
            scratch.written(vpp, row0 * len(pa), row * len(pa))     # to the file now
            scratch.unmap(vpp, row0 * len(pa), row * len(pa))
    finally:
        scratch.hints(hints)
    return g, vpp, scr


def _orbital_kind(mf):
    ''''ghf' for PySCF GHF objects (spin-orbital AO basis), 'spinor' otherwise
    (j-adapted spinor AO basis).  The mo_coeff shape cannot tell the two apart:
    for spherical bases nao_2c == 2 * nao_nr.'''
    from pyscf.scf import ghf
    return 'ghf' if isinstance(mf, ghf.GHF) else 'spinor'


def from_scf(mf, mo_coeff=None, mo_occ=None, frozen=0, kind=None, packed_vvvv=False,
             vir_batch=16, spin_order=False, max_memory=None):
    '''Build spin-orbital ERIs from a converged SCF object.

    ``mo_coeff`` may be any orbitals spanning the same occupied space as the
    SCF (e.g. after an occ-occ / vir-vir rotation); the Fock matrix is then
    non-diagonal and is used as such.  Occupied orbitals are the columns with
    mo_occ > 0; the first ``frozen`` occupied orbitals are dropped.

    ``packed_vvvv``: transform block by block and keep <ab||cd> only packed
    over a<b, c<d (needed beyond small bases), ``vir_batch`` values of a at a
    time.  ``max_memory`` (MB for the whole process, as in PySCF; default: 80%
    of what the node and the memory limit of the job leave) then bounds the
    transformation, and the packed vvvv becomes a mapped array of a scratch
    file if it does not fit in memory twice.
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
    spins = None
    if spin_order:
        if kind != 'ghf':
            raise ValueError('spin_order needs GHF-form orbitals of a nonrelativistic reference')
        nao = mf.mol.nao_nr()
        wa = np.einsum('mp,mp->p', C[:nao].conj(), C[:nao]).real
        wb = np.einsum('mp,mp->p', C[nao:].conj(), C[nao:]).real
        if np.max(np.minimum(wa, wb) / (wa + wb)) > 1e-10:
            raise ValueError('spin_order needs orbitals of pure alpha or beta spin')
        spins = (wb > wa).astype(np.int64)
        perm = np.concatenate((np.argsort(spins[:nocc], kind='stable'),
                               nocc + np.argsort(spins[nocc:], kind='stable')))
        C, spins = C[:, perm], spins[perm]
    fock = C.conj().T @ fock_ao @ C
    if packed_vvvv and not (kind == 'ghf' and np.iscomplexobj(C)):
        g, vpp, scr = _blocks_by_ao2mo(mf.mol, kind, C[:, :nocc], C[:, nocc:], vir_batch,
                                       max_memory)
        if kind == 'ghf':
            fock = fock.real
            g = {k: b.real for k, b in g.items()}
            vpp = vpp.real
        out = ERIs(nocc=nocc, fock=fock, mo_energy=fock.diagonal().real.copy(),
                   e_hf=e_hf, e_nuc=float(mf.energy_nuc()),
                   source='%s:%s' % (type(mf).__name__, kind), vvvv=None, **g)
        # (a mapped array stays a mapped array: no ascontiguousarray)
        out._vvvv_pp = vpp if scr is not None else np.ascontiguousarray(vpp)
        out._scratch = scr                      # keeps the scratch file open
        if spins is not None:
            out.spin_occ, out.spin_vir = spins[:nocc], spins[nocc:]
        return out
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
    out = ERIs(nocc=nocc, fock=fock, mo_energy=fock.diagonal().real.copy(),
               e_hf=e_hf, e_nuc=float(mf.energy_nuc()),
               source='%s:%s' % (type(mf).__name__, kind), **blocks)
    if spins is not None:
        out.spin_occ, out.spin_vir = spins[:nocc], spins[nocc:]
    return out


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
