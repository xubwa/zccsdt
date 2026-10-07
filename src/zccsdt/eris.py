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


def drop_virtuals(eris, nvir):
    '''The container with the ``nvir`` lowest virtual spin orbitals or
    spinors of ``eris`` only: a truncated virtual space (canonical orbitals;
    a Kramers pair should not be cut in two).  <vv||vv> full or on pairs, as
    ``eris`` has it.'''
    no = eris.nocc
    e = np.asarray(eris.mo_energy)
    nv = len(e) - no
    if nvir >= nv:
        return eris
    keep = np.sort(np.argsort(e[no:], kind='stable')[:nvir])
    idx = np.concatenate((np.arange(no), no + keep))
    fock = np.asarray(eris.fock)
    if abs(fock[np.ix_(idx, np.setdiff1d(np.arange(no + nv), idx))]).max() > 1e-7:
        raise ValueError('the virtual orbitals that are dropped are coupled to the others by the Fock matrix: '
                         'not canonical orbitals')

    def cut(key):
        x = getattr(eris, key)
        if x is None:
            return None
        x = np.asarray(x)
        for axis, c in enumerate(key):
            if c == 'v':
                x = np.take(x, keep, axis=axis)
        return np.ascontiguousarray(x)
    out = ERIs(nocc=no, fock=np.ascontiguousarray(fock[np.ix_(idx, idx)]), mo_energy=e[idx].copy(),
               e_hf=eris.e_hf, e_nuc=eris.e_nuc, source=eris.source,
               spin_occ=eris.spin_occ, spin_vir=None if eris.spin_vir is None else np.asarray(eris.spin_vir)[keep],
               **{k: cut(k) for k in ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv', 'vvvv')})
    pp = getattr(eris, '_vvvv_pp', None)
    if pp is not None:
        a, b = np.triu_indices(nvir, 1)
        rank = np.zeros((nv, nv), dtype=np.int64)
        rank[np.triu_indices(nv, 1)] = np.arange(nv * (nv - 1) // 2)
        sel = rank[keep[a], keep[b]]
        out._vvvv_pp = np.ascontiguousarray(np.asarray(pp)[np.ix_(sel, sel)])
    return out


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


def _ioblk(max_memory):
    '''PySCF's ioblk_size (MB) for a transformation with ``max_memory`` MB:
    the second half of the transformation works on blocks of this size and
    their images (2.5 times as much for spinors), whatever max_memory is.'''
    return float(min(256.0, max(16.0, max_memory / 4)))


def _spinor_block(mol, mo_coeffs, max_memory):
    '''(12|34) of spinor MOs by PySCF's out-of-core transformation, with its
    buffers bounded by ``max_memory`` (MB).  ao2mo.general does not pass
    max_memory on for spinor integrals (r_outcore.general_iofree drops it)
    and then takes its default of 4000 MB, so the file-based routine is
    called directly.'''
    import os
    import tempfile
    import h5py
    from pyscf.ao2mo import r_outcore
    from zccsdt.scratch import scratch_dir
    fd, name = tempfile.mkstemp(prefix='zccsdt_ao2mo_', suffix='.h5', dir=scratch_dir())
    os.close(fd)
    try:
        r_outcore.general(mol, mo_coeffs, name, dataname='eri_mo', intor='int2e_spinor',
                          max_memory=max_memory, ioblk_size=_ioblk(max_memory), verbose=0)
        with h5py.File(name, 'r') as f:
            return np.asarray(f['eri_mo'])
    finally:
        os.remove(name)


def _spinor_vvvv_rows(mol, Cv, max_memory):
    '''Yields (a, Y) for the virtual spinors a = 0 .. v - 1 with
    Y[c, b, d] = (ac|bd), Y being a buffer that the next step overwrites.

    PySCF's out-of-core transformation writes the rows of a range of a to a
    scratch file, a few ranges in turn: as many passes over the AO integrals
    (two, or more for files above 32 GB), whatever the memory, and v^3
    elements in memory.  Transforming a batch of a at a time in memory takes
    one pass over the AO integrals per batch instead.'''
    import os
    import tempfile
    import h5py
    from pyscf.ao2mo import r_outcore
    from zccsdt.scratch import scratch_dir
    nv = Cv.shape[1]
    nchunk = min(max(2, -(-16 * nv ** 4 // (32 << 30))), max(nv, 1))
    bounds = [nv * k // nchunk for k in range(nchunk + 1)]
    Y = np.empty((nv, nv * nv), dtype=np.complex128)
    for h0, h1 in zip(bounds[:-1], bounds[1:]):
        if h1 == h0:
            continue
        fd, name = tempfile.mkstemp(prefix='zccsdt_ao2mo_', suffix='.h5', dir=scratch_dir())
        os.close(fd)
        try:
            r_outcore.general(mol, (Cv[:, h0:h1], Cv, Cv, Cv), name, dataname='eri_mo',
                              intor='int2e_spinor', max_memory=max_memory,
                              ioblk_size=_ioblk(max_memory), verbose=0)
            with h5py.File(name, 'r') as f:
                ds = f['eri_mo']
                for a in range(h0, h1):
                    ds.read_direct(Y, np.s_[(a - h0) * nv:(a - h0 + 1) * nv])
                    yield a, Y.reshape(nv, nv, nv)
        finally:
            os.remove(name)


def _chem_block(mol, kind, C1, C2, C3, C4, max_memory=None):
    '''Chemist-notation MO block (12|34) via pyscf.ao2mo (no full n^4 tensor).
    ``max_memory`` (MB): work memory of the transformation.'''
    from pyscf import ao2mo
    from zccsdt import memory
    shape = (C1.shape[1], C2.shape[1], C3.shape[1], C4.shape[1])
    kw = {'max_memory': memory.ao2mo_memory() if max_memory is None else max_memory}
    if kind == 'spinor':
        return _spinor_block(mol, (C1, C2, C3, C4), kw['max_memory']).reshape(shape)
    nao = mol.nao_nr()
    out = 0
    for s1 in (slice(0, nao), slice(nao, None)):
        for s2 in (slice(0, nao), slice(nao, None)):
            out = out + ao2mo.general(mol, (C1[s1], C2[s1], C3[s2], C4[s2]),
                                      compact=False, **kw).reshape(shape)
    return out


def _blocks_by_ao2mo(mol, kind, Co, Cv, vir_batch, max_memory=None, real=False):
    '''Antisymmetrized blocks from chemist blocks; vvvv returned packed over
    a<b, c<d, built one a at a time: from the rows that PySCF's out-of-core
    transformation writes to a scratch file (spinors, _spinor_vvvv_rows), or
    from batches of ``vir_batch`` values of a transformed in memory.  With a
    memory budget (``max_memory`` in MB for the whole process; default: 80%
    of what is available) the batches shrink to fit, and the packed vvvv is
    a mapped array of a scratch file when it does not fit twice (bare and
    dressed); the scratch object is returned as well (None if in memory).'''
    from zccsdt import memory
    from zccsdt.scratch import Scratch
    # (PySCF takes up to its max_memory for buffers: bound every transformation)
    def ch(a, b, c, d):
        x = _chem_block(mol, kind, a, b, c, d, memory.ao2mo_memory(max_memory))
        return _real_part(x) if real else x
    # the ovvv block and its chemist form are what this step needs at least
    no, nv = Co.shape[1], Cv.shape[1]
    isz = 8 if real else (16 if kind == 'spinor' or np.iscomplexobj(Cv) else 8)
    floor = 2 * no * nv ** 3 * isz + 400_000_000
    limit = memory.budget(max_memory) if max_memory is not None else memory.available()
    if limit is not None and floor > limit:
        raise MemoryError('the integrals for %d occupied and %d virtual spin orbitals need '
                          'about %.1f GB of memory (the o v^3 block and the buffers of its '
                          'transformation); %.1f GB are available (max_memory, or what the '
                          'node and the memory limit of the job leave)'
                          % (no, nv, floor / 1e9, limit / 1e9))
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
    # <ia||bc> = (ib|ac) - (ic|ab), written into its own contiguous array (two
    # ovvv-sized arrays at a time, not three)
    g['ovvv'] = np.subtract(ovvv.transpose(0, 2, 1, 3), ovvv.transpose(0, 2, 3, 1),
                            out=np.empty((no, nv, nv, nv), dtype=ovvv.dtype))
    del ovvv
    g = {k: np.ascontiguousarray(0.5 * (b - b.transpose(1, 0, 2, 3)))
         if k[0] == k[1] else np.ascontiguousarray(b) for k, b in g.items()}
    del oooo, ooov, ovov, oovv, ovvo
    pa, pb = np.triu_indices(nv, 1)
    dtype = np.dtype(np.float64) if real else np.result_type(g['oooo'], Cv)
    isz = np.dtype(dtype).itemsize
    pp = len(pa) * len(pa) * isz
    # one a of (ac|bd): the chemist block, its antisymmetrized form and the
    # buffers of the transformation
    per_a = 4 * nv ** 3 * isz
    spinor = kind == 'spinor'
    budget = memory.budget(max_memory)
    scr = None
    if budget is not None:
        if 2 * pp + (1 if spinor else min(vir_batch, nv)) * per_a > budget:
            scr = Scratch(prefix='zccsdt_vvvv_', what='the packed <ab||cd> integrals',
                          hint=' or give the calculation more memory (max_memory)')
        else:
            budget -= pp
        vir_batch = int(max(1, min(vir_batch, budget // 2 // max(per_a, 1))))
    vpp = np.zeros((len(pa), len(pa)), dtype=dtype) if scr is None else \
        scr.create('vvvv', (len(pa), len(pa)), dtype)

    def batches():              # (a, Y[c, b, d] = (ac|bd)) from batches of a in memory
        for a0 in range(0, nv, vir_batch):
            X = ch(Cv[:, a0:min(a0 + vir_batch, nv)], Cv, Cv, Cv)       # [a, c, b, d]
            for a in range(a0, a0 + len(X)):
                yield a, X[a - a0]

    from zccsdt import scratch
    hints = scratch.hints(scr is not None)          # memory is short: say what is done
    try:
        row = 0
        for a, Y in (_spinor_vvvv_rows(mol, Cv, memory.ao2mo_memory(max_memory)) if spinor
                     else batches()):
            if real and np.iscomplexobj(Y):
                Y = _real_part(Y, copy=False)
            G = Y.transpose(1, 0, 2) - Y.transpose(1, 2, 0)          # <ab||cd> as [b, c, d]
            nb = nv - a - 1
            vpp[row:row + nb] = G[a + 1:][:, pa, pb]
            scratch.written(vpp, row * len(pa), (row + nb) * len(pa))   # to the file now
            scratch.unmap(vpp, row * len(pa), (row + nb) * len(pa))
            row += nb
    finally:
        scratch.hints(hints)
    return g, vpp, scr


def _real_part(x, copy=True, tol=1e-8):
    '''The real part of integrals that are real (spinors of a linear molecule
    adapted to j_z): an array of its own, or a view.'''
    if not np.iscomplexobj(x):
        return x
    if x.size and abs(x.imag).max() > tol:
        raise ValueError('integrals taken for real have imaginary parts up to %.1e' % abs(x.imag).max())
    return np.ascontiguousarray(x.real) if copy else x.real


def spinor_jz(mol):
    '''m_j of the basis functions of PySCF's spinor basis.'''
    out = []
    for b in range(mol.nbas):
        l, kappa = mol.bas_angular(b), mol.bas_kappa(b)
        js = (l - 0.5, l + 0.5) if kappa == 0 else ((l - 0.5,) if kappa > 0 else (l + 0.5,))
        for _ in range(mol.bas_nctr(b)):
            for j in js:
                if j > 0:
                    out += list(np.arange(-j, j + 1))
    return np.array(out)


def axial_spinors(mf, mo_coeff=None, mo_energy=None, mo_occ=None, tol=1e-7):
    '''The spinors of a linear molecule on the z axis as eigenfunctions of
    j_z with real coefficients: (C, level).  C[:, p] is in the place of
    mo_coeff[:, p] - the spinors of a degenerate level of one occupation
    rotated among themselves (a Kramers pair into m_j and -m_j), every
    spinor with a real phase; ``level`` numbers the levels (what must not be
    cut in two by freezing).  All MO integrals of such spinors are real:
    products of the phases exp(i m phi) that survive the integration are
    one.  What the SCF left of other m_j in a spinor is projected out.

    None if the spinors are not of that kind: another molecule or another
    orientation, an SCF solution without the symmetry of the axis (an open
    shell whose spin is not on the axis), GHF spin orbitals.'''
    if _orbital_kind(mf) != 'spinor':
        return None
    C = np.array(mf.mo_coeff if mo_coeff is None else mo_coeff)
    e = np.asarray(mf.mo_energy if mo_energy is None else mo_energy)
    occ = np.asarray(mf.mo_occ if mo_occ is None else mo_occ) > 0
    if C.ndim != 2 or e.shape != (C.shape[1],) or occ.shape != e.shape:
        return None
    m_ao = spinor_jz(mf.mol)
    S = mf.get_ovlp()
    if len(m_ao) != C.shape[0] or abs(S * (m_ao[:, None] != m_ao[None, :])).max() > tol:
        return None
    n = len(e)
    order = np.lexsort((e, ~occ))               # occupied first, each set by energy
    level = np.zeros(n, dtype=np.int64)
    mj = np.zeros(n)
    k, nlev = 0, 0
    while k < n:
        m = k + 1
        while m < n and occ[order[m]] == occ[order[k]] and e[order[m]] - e[order[m - 1]] < tol:
            m += 1
        idx = order[k:m]
        V = C[:, idx]
        w, U = np.linalg.eigh(V.conj().T @ (S * m_ao[None, :]) @ V)
        if abs(w - np.round(w - 0.5) - 0.5).max() > 1e-5:
            return None
        V = V @ U
        big = V[np.argmax(abs(V), axis=0), np.arange(m - k)]
        C[:, idx] = V * (abs(big) / big)
        level[idx], mj[idx] = nlev, np.round(w - 0.5) + 0.5
        k, nlev = m, nlev + 1
    if abs(C.imag).max() > 1e-7:
        return None
    C = np.ascontiguousarray(C.real) * (m_ao[:, None] == mj[None, :])
    # orthonormal again, those of one m_j among themselves: the occupied ones, then the
    # virtual ones in the complement of the occupied
    for m in np.unique(mj):
        o, v = np.where((mj == m) & occ)[0], np.where((mj == m) & ~occ)[0]
        for sel in (o, v):
            if not len(sel):
                continue
            if sel is v and len(o):
                C[:, v] -= C[:, o] @ (C[:, o].T @ S.real @ C[:, v])
            w, U = np.linalg.eigh(C[:, sel].T @ S.real @ C[:, sel])
            C[:, sel] = C[:, sel] @ (U / np.sqrt(w)) @ U.T
    return C, level


def _orbital_kind(mf):
    ''''ghf' for PySCF GHF objects (spin-orbital AO basis), 'spinor' otherwise
    (j-adapted spinor AO basis).  The mo_coeff shape cannot tell the two apart:
    for spherical bases nao_2c == 2 * nao_nr.'''
    from pyscf.scf import ghf
    return 'ghf' if isinstance(mf, ghf.GHF) else 'spinor'


def from_scf(mf, mo_coeff=None, mo_occ=None, frozen=0, kind=None, packed_vvvv=False,
             vir_batch=16, spin_order=False, max_memory=None, real=False):
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

    ``real``: the integrals of these orbitals are real although the AO
    integrals are not (the spinors of axial_spinors()): real arrays are kept
    (ValueError if an imaginary part is not zero).
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
    if real:
        fock = _real_part(fock)
    if packed_vvvv and not (kind == 'ghf' and np.iscomplexobj(C)):
        g, vpp, scr = _blocks_by_ao2mo(mf.mol, kind, C[:, :nocc], C[:, nocc:], vir_batch,
                                       max_memory, real)
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
    if real:
        eri = _real_part(eri)
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
