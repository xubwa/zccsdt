'''UHF / RHF ("U" and "R" form) integrals as spin-blocked tensors.

The antisymmetrized spin-orbital blocks <pq||rs> (oooo, ooov, oovv, ovov,
ovvo, ovvv) are SpinTensors that store only Ms-conserving spin blocks, built
from spatial chemist integrals (pr|qs) of the alpha and beta orbitals:
    <p q || r s> = (pr|qs) d(s_p, s_r) d(s_q, s_s) - (ps|qr) d(s_p, s_s) d(s_q, s_r).
The packed vvvv is stored as its three diagonal pair blocks (alpha-alpha,
alpha-beta, beta-beta); the alpha-beta block is the plain spatial (ac|bd).
For an RHF reference the alpha and beta orbitals coincide: the spatial
integrals are computed once and spin-flipped blocks share their arrays.  The
CC code then also shares the beta-beta blocks of the dressed vvvv and of the
ladder operator with the alpha-alpha ones, which assumes closed-shell
(spin-flip symmetric) amplitudes, as produced by CC on an RHF reference.

Orbital order: active occupied alpha, active occupied beta, virtual alpha,
virtual beta, each in orbital-energy order (the order of from_scf(...,
spin_order=True) on the GHF form of the same reference).
'''

from dataclasses import dataclass, field

import numpy as np
from pyscf import ao2mo

from zccsdt.cc import t3pack
from zccsdt.cc.spintensor import SpinTensor, orbital_space, pair_space
from zccsdt.eris import ERIs, BLOCKS

SIG = (1, 1, -1, -1)


@dataclass
class SpinERIs:
    nocc: int
    fock: np.ndarray            # dense spin-orbital Fock (block diagonal in spin)
    mo_energy: np.ndarray
    e_hf: float
    e_nuc: float
    spin_occ: np.ndarray
    spin_vir: np.ndarray
    g: dict                     # 'oooo', ..., 'ovvv' -> SpinTensor
    vvvv_blocks: list           # packed diagonal pair blocks of <ab||cd>
    restricted: bool = False
    source: str = ''
    spaces: dict = field(default_factory=dict)

    @property
    def nmo(self):
        return self.fock.shape[0]

    def __getattr__(self, name):
        # dense views of the small blocks for code that expects ERIs-like
        # attributes (ovvv and vvvv are intentionally not provided)
        if name in ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo'):
            d = self.__dict__.setdefault('_dense', {})
            if name not in d:
                d[name] = self.g[name].to_dense()
            return d[name]
        raise AttributeError(name)

    @property
    def dtype(self):
        return self.fock.dtype

    def to_dense_eris(self):
        '''Dense spin-orbital ERIs (same orbital order), for testing and for
        the dense code path.'''
        blocks = {k: self.g[k].to_dense() for k in BLOCKS[:-1]}
        out = ERIs(nocc=self.nocc, fock=self.fock, mo_energy=self.mo_energy,
                   e_hf=self.e_hf, e_nuc=self.e_nuc, source=self.source,
                   vvvv=None, **blocks)
        vm = t3pack.triple_map(len(self.spin_vir), self.spin_vir)
        npv = vm.npair
        full = np.zeros((npv, npv), dtype=self.dtype)
        for (lo, hi), b in zip(vm.blocks, self.vvvv_blocks):
            full[lo:hi, lo:hi] = b
        # stored packed form is in triu (lexicographic) pair order
        n = vm.n
        md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
        tri = md * n - md * (md + 1) // 2 + (ne - md - 1)
        lex = np.zeros_like(full)
        lex[tri[:, None], tri] = full
        out._vvvv_pp = lex
        out.spin_occ, out.spin_vir = self.spin_occ, self.spin_vir
        return out


def from_scf(mf, frozen=0, vir_batch=16, max_memory=None):
    '''U/R-form integrals from a PySCF RHF, ROHF or UHF object.  ``frozen`` is
    the number of frozen spin orbitals (frozen // 2 alpha and beta each).
    ``max_memory`` (MB for the whole process; default: 80% of what is
    available) bounds the buffers of the integral transformations.'''
    from zccsdt import memory
    ao_mem = memory.ao2mo_memory(max_memory)
    mol = mf.mol
    restricted = getattr(mf, 'mo_coeff', None) is not None and np.asarray(mf.mo_coeff).ndim == 2
    if restricted:
        C = np.asarray(mf.mo_coeff)
        occ_a = np.asarray(mf.mo_occ) > 0           # RHF: 2, ROHF: 2 or 1
        occ_b = np.asarray(mf.mo_occ) > 1
        Ca, Cb = C, C
        rhf = bool(np.all(occ_a == occ_b))
        dm = mf.make_rdm1()
        if dm.ndim == 2:
            dm = np.array((dm * 0.5, dm * 0.5)) if rhf else None
        if dm is None:
            dm = np.array((np.einsum('pi,qi->pq', C[:, occ_a], C[:, occ_a]),
                           np.einsum('pi,qi->pq', C[:, occ_b], C[:, occ_b])))
    else:
        Ca, Cb = (np.asarray(c) for c in mf.mo_coeff)
        occ_a, occ_b = (np.asarray(o) > 0 for o in mf.mo_occ)
        rhf = False
        dm = np.asarray(mf.make_rdm1())
    from pyscf import scf as _scf
    vhf = _scf.uhf.get_veff(mol, dm) if not rhf else None
    hcore = mf.get_hcore()
    if rhf:
        fa_ao = mf.get_fock(dm=dm[0] + dm[1])
        fb_ao = fa_ao
    else:
        fa_ao, fb_ao = hcore + vhf[0], hcore + vhf[1]
    e_hf = float(mf.energy_tot(dm=dm[0] + dm[1]) if rhf else
                 _scf.uhf.energy_elec(mf, dm, hcore, vhf)[0] + mf.energy_nuc())

    fa = frozen // 2
    fb = frozen - fa
    oa = np.where(occ_a)[0][fa:]
    ob = np.where(occ_b)[0][fb:]
    va = np.where(~occ_a)[0]
    vb = np.where(~occ_b)[0]
    Co = {0: Ca[:, oa], 1: Cb[:, ob]}
    Cv = {0: Ca[:, va], 1: Cb[:, vb]}
    noa, nob, nva, nvb = len(oa), len(ob), len(va), len(vb)
    spin_occ = np.array([0] * noa + [1] * nob)
    spin_vir = np.array([0] * nva + [1] * nvb)
    nocc, nvir = noa + nob, nva + nvb

    # dense spin-orbital Fock (block diagonal), occupied first
    Call = {0: np.hstack((Co[0], Cv[0])), 1: np.hstack((Co[1], Cv[1]))}
    fmo = {0: Call[0].T @ fa_ao @ Call[0], 1: Call[1].T @ fb_ao @ Call[1]}
    n = nocc + nvir
    fock = np.zeros((n, n))
    idx_a = list(range(noa)) + list(range(nocc, nocc + nva))
    idx_b = list(range(noa, nocc)) + list(range(nocc + nva, n))
    fock[np.ix_(idx_a, idx_a)] = fmo[0]
    fock[np.ix_(idx_b, idx_b)] = fmo[1]

    o_sp = orbital_space('o', spin_occ)
    v_sp = orbital_space('v', spin_vir)
    sp = {'o': o_sp, 'v': v_sp}
    Cs = {'o': Co, 'v': Cv}

    cache = {}

    def chem(X, sx, Y, Z, sz, W):
        '''Spatial chemist block (XY|ZW) with X, Y of spin sx and Z, W of spin sz.'''
        key = (X, sx, Y, Z, sz, W)
        if rhf:
            key = (X, 0, Y, Z, 0, W)
        if key not in cache:
            a, b, c, d = Cs[X][sx], Cs[Y][sx], Cs[Z][sz], Cs[W][sz]
            if min(a.shape[1], b.shape[1], c.shape[1], d.shape[1]) == 0:
                cache[key] = np.zeros((a.shape[1], b.shape[1], c.shape[1], d.shape[1]))
            else:
                cache[key] = ao2mo.general(mol, (a, b, c, d), compact=False,
                                           max_memory=ao_mem).reshape(
                    a.shape[1], b.shape[1], c.shape[1], d.shape[1])
        return cache[key]

    g = {}
    for name in BLOCKS[:-1]:
        P, Q, R, S = name
        blocks = {}
        for s1 in (0, 1):
            for s2 in (0, 1):
                for s3 in (0, 1):
                    s4 = s1 + s2 - s3
                    if s4 not in (0, 1):
                        continue
                    shape = (sp[P].size(s1), sp[Q].size(s2), sp[R].size(s3), sp[S].size(s4))
                    if 0 in shape:
                        continue
                    flipped = (1 - s1, 1 - s2, 1 - s3, 1 - s4)
                    if rhf and flipped in blocks:
                        # closed shell: the spin-flipped block has the same values
                        blocks[(s1, s2, s3, s4)] = blocks[flipped]
                        continue
                    b = np.zeros(shape)
                    if s1 == s3 and s2 == s4:            # (pr|qs)
                        b += chem(P, s1, R, Q, s2, S).transpose(0, 2, 1, 3)
                    if s1 == s4 and s2 == s3:            # -(ps|qr)
                        b -= chem(P, s1, S, Q, s2, R).transpose(0, 2, 3, 1)
                    blocks[(s1, s2, s3, s4)] = b
        g[name] = SpinTensor((sp[P], sp[Q], sp[R], sp[S]), blocks)

    # packed vvvv diagonal pair blocks
    def vv_same(s):
        nv_s = Cv[s].shape[1]
        iu, ju = np.triu_indices(nv_s, 1)
        out = np.zeros((len(iu), len(iu)))
        row = 0
        for a0 in range(0, nv_s, vir_batch):
            a1 = min(a0 + vir_batch, nv_s)
            X = ao2mo.general(mol, (Cv[s][:, a0:a1], Cv[s], Cv[s], Cv[s]), compact=False,
                              max_memory=ao_mem).reshape(a1 - a0, nv_s, nv_s, nv_s)
            for a in range(a0, a1):
                Y = X[a - a0]                                 # (ac|bd) as [c, b, d]
                G = Y.transpose(1, 0, 2) - Y.transpose(1, 2, 0)   # [b, c, d]
                nb = nv_s - a - 1
                out[row:row + nb] = G[a + 1:][:, iu, ju]
                row += nb
        return out

    vaa = vv_same(0) if nva else np.zeros((0, 0))
    vbb = vaa if rhf else (vv_same(1) if nvb else np.zeros((0, 0)))
    if nva and nvb:
        X = ao2mo.general(mol, (Cv[0], Cv[0], Cv[1], Cv[1]), compact=False,
                          max_memory=ao_mem).reshape(nva, nva, nvb, nvb)   # (ac|bd): [a, c, b, d]
        vab = np.ascontiguousarray(X.transpose(0, 2, 1, 3)).reshape(nva * nvb, nva * nvb)
    else:
        vab = np.zeros((nva * nvb, nva * nvb))
    return SpinERIs(nocc=nocc, fock=fock, mo_energy=fock.diagonal().copy(), e_hf=e_hf,
                    e_nuc=float(mf.energy_nuc()), spin_occ=spin_occ, spin_vir=spin_vir,
                    g=g, vvvv_blocks=[vaa, vab, vbb], restricted=rhf,
                    source='%s:spin-blocked' % type(mf).__name__,
                    spaces={'o': o_sp, 'v': v_sp})
