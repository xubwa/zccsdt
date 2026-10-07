'''MO integrals of DIRAC as an integral container.

DIRAC's integral transformation (``**MOLTRA``, with ``.4INDEX`` under
``**DIRAC``) leaves two unformatted files that its correlation modules and
other programs read:

MRCONEE   the number of active spinors and of electrons, the core energy
          (nuclear repulsion and frozen spinors), for every spinor its irrep
          in the parent group, its irrep in the Abelian subgroup and its
          energy, the multiplication table of the Abelian irreps, and the
          effective one-electron matrix (the Fock matrix of the frozen
          spinors) as (re, im) pairs.
MDCINT    a table Kramers pair (+i, -i) -> spinor, then records
          ``I, J, n, (K, L) x n, values``: the integrals (I J | K L) in
          Mulliken order with signed Kramers indices, closed by ``0 0 0``.
          Real numbers for the real groups (D2h, D2, C2v), complex ones
          otherwise.

Nothing depends on the order of the spinors: symmetry is a label per spinor,
the pairing a table.  Two things are implicit and done here:

- only one of two integrals related by time reversal is in the file,
  (-I -J | -K -L) = sign(I J K L) (I J | K L)^*;
- of the integrals related by a permutation, (pq|rs) = (rs|pq) = (qp|sr)^* =
  (sr|qp)^*, DIRAC 23 writes one.

The formats are those of ``src/moltra/traone.F`` and ``traout.F`` of DIRAC 23
(default integers of 8 bytes, the usual build) and of its
``utils/dirac_mointegral_export.F90``.

from_dirac() gives the container that the CC code takes (zccsdt.eris.ERIs):
occupied spinors first, real arrays for a real group.  The integrals pass
through the full array (pq|rs), n^4 numbers for n active spinors - what
high-order CC can treat fits.

write_mdcint() writes the complete list in the same format: the interface
of MRCC to DIRAC (``iface=dirac``; the release of 2018) reads MDCINT without
completing it by permutations.

The other direction - integrals of this program for a program that reads
DIRAC's files (MRCC):

    x = dirac.integrals(mf)             # canonical spinors of a PySCF SCF in Kramers pairs
    dirac.write('run', x)               # run/MRCONEE, run/MDCINT
    eris = dirac.container(x)           # the same integrals for the CC code here

integrals() pairs the spinors by time reversal, (psi, T psi), which is what
the files assume.  MRCC takes real integrals only, and write() real ones
only (DIRAC's real groups without inversion, C2v and D2: the unbarred
spinors in the irrep 1E of C2, their partners in 2E).  That covers real spin
orbitals (an RHF reference as GHF: alpha unbarred, beta barred) and the
spinors of a linear molecule on the z axis (the m_j-adapted spinors are real
in PySCF's spinor basis; unbarred m_j = 1/2, -3/2, 5/2, ..).
'''

from dataclasses import dataclass

import numpy as np
from scipy.io import FortranFile

from zccsdt.eris import ERIs

_I8, _F8 = np.dtype('<i8'), np.dtype('<f8')


def _records(path):
    f = FortranFile(path, 'r')
    out = []
    try:
        while True:
            out.append(f.read_record(np.uint8).tobytes())
    except Exception:           # (scipy raises its own end-of-file error)
        pass
    f.close()
    return out


@dataclass
class Integrals:
    '''The contents of MRCONEE and MDCINT, in the order of the spinors there.'''
    h: np.ndarray               # effective one-electron matrix [p, q]
    g: np.ndarray               # (pq|rs), complete
    e_core: float               # nuclear repulsion + frozen spinors
    energy: np.ndarray          # spinor energies of the SCF
    irrep: np.ndarray           # irrep in the parent group (1, 2, ..)
    abelian: np.ndarray         # irrep in the Abelian subgroup
    nelec: np.ndarray           # active electrons per irrep of the parent group
    group_type: int             # 1 real, 2 complex, 4 quaternionic
    kramers: np.ndarray         # [pair, (unbarred, barred)] -> spinor
    mdcint_header: bytes = None     # the first record of MDCINT (None: written anew)
    e_scf: float = None         # total SCF energy

    @property
    def occupied(self):
        '''The occupied spinors: of every irrep the first ones (DIRAC's convention).'''
        out = []
        for k, ne in enumerate(self.nelec):
            out += list(np.where(self.irrep == k + 1)[0][:ne])
        return np.array(sorted(out), dtype=np.int64)


def complete(g):
    '''(pq|rs) where it is not given, from (rs|pq), (qp|sr)^* and (sr|qp)^*.'''
    out = g.copy()
    for perm, conj in (((2, 3, 0, 1), False), ((1, 0, 3, 2), True), ((3, 2, 1, 0), True)):
        gt = g.transpose(perm)
        out = np.where(out != 0, out, gt.conj() if conj else gt)
    return out


def read(mrconee='MRCONEE', mdcint='MDCINT', fill=True):
    '''Integrals of the two files.  ``fill``: complete (pq|rs) by its
    permutations (without it: what the file has and its Kramers partners).'''
    r = _records(mrconee)
    n = int(np.frombuffer(r[0], _I8, 1, 0)[0])
    e_core = float(np.frombuffer(r[0], _F8, 1, 16)[0])
    group_type = int(np.frombuffer(r[0], _I8, 1, 32)[0])
    nsym = int(np.frombuffer(r[1], _I8, 1, 0)[0])
    nelec = np.frombuffer(r[1], _I8, nsym, 8 + 14 * nsym).copy()
    sp = np.frombuffer(r[4], np.dtype([('irp', _I8), ('irpa', _I8), ('e', _F8)]), n)
    fm = np.frombuffer(r[5], _F8, 2 * n * n).reshape(n, n, 2)          # [q, p, (re, im)]
    h = (fm[:, :, 0] + 1j * fm[:, :, 1]).T
    cplx = group_type != 1
    if not cplx:
        h = np.ascontiguousarray(h.real)
    m = _records(mdcint)
    nkr = int(np.frombuffer(m[0], _I8, 1, 18)[0])
    if 2 * nkr != n:
        raise ValueError('%s and %s do not belong together: %d spinors, %d Kramers pairs' % (mrconee, mdcint, n, nkr))
    kramers = np.frombuffer(m[0], _I8, 2 * nkr, 26).reshape(nkr, 2) - 1
    spinor = np.zeros(2 * nkr + 1, dtype=np.int64)          # signed Kramers index -> spinor
    spinor[1:nkr + 1], spinor[-nkr:] = kramers[:, 0], kramers[::-1, 1]
    g = np.zeros((n, n, n, n), dtype=np.complex128 if cplx else np.float64)
    for rec in m[1:]:
        i, j, nz = (int(x) for x in np.frombuffer(rec, _I8, 3, 0))
        if i == 0:
            if j != 0:
                raise NotImplementedError('%s continues in another file (more than 60 million integrals)' % mdcint)
            break
        if nz == 0:
            continue
        rcw = (len(rec) - 24 - 16 * nz) // (8 * nz)
        if len(rec) != 24 + nz * (16 + 8 * rcw) or rcw != (2 if cplx else 1):
            raise ValueError('%s: a record of %d bytes for %d integrals (integers of 8 bytes expected)'
                             % (mdcint, len(rec), nz))
        kl = np.frombuffer(rec, _I8, 2 * nz, 24).reshape(nz, 2)
        v = np.frombuffer(rec, _F8, rcw * nz, 24 + 16 * nz).reshape(nz, rcw)
        v = v[:, 0] + 1j * v[:, 1] if cplx else v[:, 0]
        k, l = kl[:, 0], kl[:, 1]
        g[spinor[i], spinor[j], spinor[k], spinor[l]] = v
        # the integrals of the Kramers partners
        g[spinor[-i], spinor[-j], spinor[-k], spinor[-l]] = np.sign(i * j * k * l) * np.conj(v)
    return Integrals(h, complete(g) if fill else g, e_core, sp['e'].copy(), sp['irp'].copy(), sp['irpa'].copy(), nelec,
                     group_type, kramers.copy(), m[0], float(np.frombuffer(r[0], _F8, 1, 56)[0]))


def from_dirac(mrconee='MRCONEE', mdcint='MDCINT', packed_vvvv=True):
    '''The integral container of the active spinors of a DIRAC calculation
    (closed shell): container() of the two files.'''
    x = read(mrconee, mdcint)
    out = container(x, packed_vvvv, 'MRCONEE, MDCINT')
    del x.g
    return out


def container(x, packed_vvvv=True, source='spinors in Kramers pairs'):
    '''The integral container (zccsdt.eris.ERIs) of the Integrals ``x``
    (closed shell), occupied spinors first, each set by energy; its
    ``spinors`` are their numbers in ``x``.  ``packed_vvvv``: keep <ab||cd>
    for a<b, c<d only (as the CC code wants it); otherwise the full block
    (the dense reference codes).'''
    n = len(x.energy)
    occ = x.occupied
    vir = np.setdiff1d(np.arange(n), occ)
    occ, vir = occ[np.argsort(x.energy[occ], kind='stable')], vir[np.argsort(x.energy[vir], kind='stable')]
    idx = np.concatenate((occ, vir))
    no = len(occ)
    h, g = x.h[np.ix_(idx, idx)], x.g[np.ix_(idx, idx, idx, idx)]
    o, v = slice(0, no), slice(no, n)

    def block(key):
        '''<pq||rs> = (pr|qs) - (ps|qr)'''
        p, q, r, s = (o if c == 'o' else v for c in key)
        return np.ascontiguousarray(g[p, r, q, s].transpose(0, 2, 1, 3) - g[p, s, q, r].transpose(0, 2, 3, 1))
    # f = h + sum_i <pi||qi>
    fock = h + np.einsum('pqii->pq', g[:, :, o, o]) - np.einsum('piiq->pq', g[:, o, o, :])
    oooo = block('oooo')
    e_hf = x.e_core + np.trace(h[o, o]).real + 0.5 * np.einsum('ijij->', oooo).real
    if abs(fock[o, v]).max() > 1e-6:
        raise ValueError('the Fock matrix of the occupied spinors of the files has an occupied-virtual block '
                         '(%.1e): not the closed-shell determinant of the SCF' % abs(fock[o, v]).max())
    blocks = {k: block(k) for k in ('ooov', 'oovv', 'ovov', 'ovvo', 'ovvv')}
    out = ERIs(nocc=no, fock=fock, mo_energy=fock.diagonal().real.copy(), e_hf=float(e_hf), e_nuc=x.e_core,
               source=source, oooo=oooo, vvvv=None if packed_vvvv else block('vvvv'), **blocks)
    if packed_vvvv:
        a, b = np.triu_indices(n - no, 1)
        a, b = a + no, b + no
        out._vvvv_pp = np.ascontiguousarray(g[a[:, None], a[None, :], b[:, None], b[None, :]]
                                            - g[a[:, None], b[None, :], b[:, None], a[None, :]])
    out.spinors, out.dirac = idx, x
    return out


def _header(nkr, kramers):
    '''The first record of MDCINT: date, time, the number of Kramers pairs
    and the spinors of each pair.'''
    from datetime import datetime
    now = datetime.now()
    return (now.strftime(' %Y%m%d ').encode() + now.strftime('%H:%M:%S').encode()
            + np.array([nkr], dtype=_I8).tobytes() + (np.asarray(kramers, dtype=_I8) + 1).tobytes())


def write_mdcint(path, x, threshold=1e-15):
    '''MDCINT with all integrals (I J | K L), I unbarred, of the Integrals
    ``x`` - the complete list, in DIRAC's records.  Returns their number.'''
    nkr = len(x.kramers)
    if x.mdcint_header is None:
        x.mdcint_header = _header(nkr, x.kramers)
    signed = np.array([s * k for k in range(1, nkr + 1) for s in (1, -1)], dtype=np.int64)
    cols = np.array([x.kramers[abs(s) - 1, 0 if s > 0 else 1] for s in signed])
    cplx = np.iscomplexobj(x.g)
    f = FortranFile(path, 'w')
    f.write_record(np.frombuffer(x.mdcint_header, np.uint8))
    count = 0
    for i in range(1, nkr + 1):
        for j, cj in zip(signed, cols):
            blk = x.g[x.kramers[i - 1, 0], cj][np.ix_(cols, cols)]
            ks, ls = np.nonzero(abs(blk) > threshold)
            if not len(ks):
                continue
            kl = np.stack((signed[ks], signed[ls]), axis=1).astype(_I8)
            v = blk[ks, ls]
            v = np.stack((v.real, v.imag), axis=1).ravel() if cplx else v
            f.write_record(np.array([i, j, len(ks)], dtype=_I8), kl.ravel(), np.asarray(v, dtype=_F8))
            count += len(ks)
    f.write_record(np.array([0, 0, 0], dtype=_I8))
    f.close()
    return count


def write_mrconee(path, x):
    '''MRCONEE of the Integrals ``x`` in DIRAC's records, for a real group
    without inversion (C2v, D2; group type 1): two irreps of the parent
    group and of its Abelian subgroup C2 (1E, 2E: unbarred and barred
    spinors), nothing frozen or deleted on the file's account (what is
    frozen is in the core energy and in the one-electron matrix).'''
    n = len(x.energy)
    if x.group_type != 1 or len(x.nelec) != 2:
        raise NotImplementedError('MRCONEE is written for real groups without inversion (two fermion irreps)')
    if np.iscomplexobj(x.h) and abs(x.h.imag).max() > 0:
        raise ValueError('a real group: the one-electron matrix has an imaginary part')
    i8 = lambda *v: np.array(v, dtype=_I8).ravel()
    f8 = lambda *v: np.array(v, dtype=_F8).ravel()
    chars = lambda names: np.frombuffer(''.join(names).encode(), np.uint8)
    e_scf = x.e_core if x.e_scf is None else x.e_scf
    table = np.array([[4, 3, 1, 2], [3, 4, 2, 1], [1, 2, 3, 4], [2, 1, 4, 3]])
    sp = np.zeros(n, dtype=[('irp', _I8), ('irpa', _I8), ('e', _F8)])
    sp['irp'], sp['irpa'], sp['e'] = x.irrep, x.abelian, x.energy
    fm = np.zeros((n, n, 2))
    fm[:, :, 0] = np.asarray(x.h).real.T                    # [q, p, (re, im)]
    f = FortranFile(path, 'w')
    # spinors, Breit, core energy, fermion irreps, group type, spin-free, orbitals, SCF energy, frozen, open
    f.write_record(i8(n, 0), f8(x.e_core), i8(1, 1, 0, n // 2), f8(e_scf), i8(0, 0, 0, 0, 0, 0, 0))
    # irreps of the parent group, active electrons in each, spinors, frozen (three kinds), deleted
    f.write_record(i8(2), chars(['   E  1E      ', '   E  2E      ']), i8(x.nelec), i8(n // 2, 0, 0, 0, 0))
    f.write_record(i8(2), chars(['  1E', '  2E', '   a', '   b']))
    f.write_record(i8(table.T))
    f.write_record(sp, i8(np.zeros(n)), i8(n // 2, 4))      # per spinor; supersymmetry labels; orbitals, boson irreps
    f.write_record(fm)
    f.close()


def write(directory, x, threshold=1e-15):
    '''MRCONEE and MDCINT of the Integrals ``x`` in ``directory``; returns
    the number of two-electron integrals written.'''
    import os
    os.makedirs(directory, exist_ok=True)
    write_mrconee(os.path.join(directory, 'MRCONEE'), x)
    return write_mdcint(os.path.join(directory, 'MDCINT'), x, threshold)


def time_reversed(mf, C):
    '''T psi for the columns psi of C: in PySCF's spinor basis by the map
    of the basis functions (T chi_i = +- chi_j, coefficients conjugated), for
    GHF spin orbitals T (a, b) = (-b^*, a^*).'''
    from zccsdt.eris import _orbital_kind
    C = np.asarray(C)
    out = np.empty_like(C)
    if _orbital_kind(mf) == 'ghf':
        nao = C.shape[0] // 2
        out[:nao], out[nao:] = -C[nao:].conj(), C[:nao].conj()
    else:
        tao = np.asarray(mf.mol.time_reversal_map())
        out[abs(tao) - 1] = np.sign(tao)[:, None] * C.conj()
    return out


def _jz(mf):
    '''m_j of the basis functions of PySCF's spinor basis.'''
    mol, out = mf.mol, []
    for b in range(mol.nbas):
        l, kappa = mol.bas_angular(b), mol.bas_kappa(b)
        js = (l - 0.5, l + 0.5) if kappa == 0 else ((l - 0.5,) if kappa > 0 else (l + 0.5,))
        for _ in range(mol.bas_nctr(b)):
            for j in js:
                if j > 0:
                    out += list(np.arange(-j, j + 1))
    return np.array(out)


def kramers_pairs(mf, mo_coeff=None, mo_energy=None, tol=1e-7):
    '''The spinors of a closed-shell SCF of PySCF in Kramers pairs: unbarred
    spinors C[:, k] and their time-reversed partners (time_reversed()), with
    the energies of the pairs.  They span the canonical spinors - each
    degenerate level by itself - so the Fock matrix stays diagonal.

    GHF spin orbitals of a spin-restricted reference: alpha unbarred.
    Spinors: those of a linear molecule on the z axis, adapted to m_j here
    if they are not (unbarred m_j = 1/2, -3/2, 5/2, ..: the irrep 1E of C2)
    and real then.  What the SCF left of other m_j in a spinor (its noise)
    is projected out and the unbarred spinors of one m_j are orthonormalized
    again, so that the symmetry of the integrals is exact.'''
    from zccsdt.eris import _orbital_kind
    C = np.asarray(mf.mo_coeff if mo_coeff is None else mo_coeff)
    e = np.asarray(mf.mo_energy if mo_energy is None else mo_energy)
    n = len(e)
    order = np.argsort(e, kind='stable')
    C, e = C[:, order], e[order]
    S = mf.get_ovlp()
    ghf = _orbital_kind(mf) == 'ghf'
    what = 's_z: not spin orbitals of one spin' if ghf else 'j_z: not a linear molecule on the z axis'
    m_ao = np.repeat([0.5, -0.5], C.shape[0] // 2) if ghf else _jz(mf)
    if len(m_ao) != C.shape[0]:
        raise ValueError('%d basis functions, %d rows of the coefficients' % (len(m_ao), C.shape[0]))
    if abs(S * (m_ao[:, None] != m_ao[None, :])).max() > tol:
        raise ValueError('the overlap couples basis functions of different %s' % what)
    unbarred, mj, energies, k = [], [], [], 0
    while k < n:
        m = k + 1
        while m < n and e[m] - e[m - 1] < tol:
            m += 1
        # the level in functions of definite m_j (m_s for spin orbitals)
        V = C[:, k:m]
        w, U = np.linalg.eigh(V.conj().T @ (S * m_ao[None, :]) @ V)
        if abs(w - np.round(w - 0.5) - 0.5).max() > 1e-5:
            raise ValueError('the level at %.8f is not made of eigenfunctions of %s' % (e[k], what))
        V = V @ U
        first = [q for q in range(m - k) if round(w[q] - 0.5) % 2 == 0]             # 1/2, -3/2, 5/2, ..
        if 2 * len(first) != m - k:
            raise ValueError('the level of %d spinors at %.8f is not made of Kramers pairs' % (m - k, e[k]))
        for q in first:
            c = V[:, q]
            big = c[np.argmax(abs(c))]
            unbarred.append(c * (abs(big) / big))
            mj.append(round(w[q] - 0.5) + 0.5)
            energies.append(e[k:m].mean())
        k = m
    Cu, mj = np.array(unbarred).T, np.array(mj)
    if np.iscomplexobj(Cu) and abs(Cu.imag).max() < 1e-8:
        Cu = np.ascontiguousarray(Cu.real)
    Cu = Cu * (m_ao[:, None] == mj[None, :])
    for m in np.unique(mj):
        sel = np.where(mj == m)[0]
        ovlp = Cu[:, sel].conj().T @ S @ Cu[:, sel]
        w, U = np.linalg.eigh(ovlp if np.iscomplexobj(Cu) else ovlp.real)
        Cu[:, sel] = Cu[:, sel] @ (U / np.sqrt(w)) @ U.conj().T
    return Cu, time_reversed(mf, Cu), np.array(energies)


def integrals(mf, frozen=0, nvir=None, mo_coeff=None, mo_energy=None, tol=1e-9):
    '''The Integrals of the canonical spinors of a closed-shell SCF object of
    PySCF, in Kramers pairs (kramers_pairs()): all unbarred spinors by
    energy, then their partners - the layout of DIRAC for a real group.
    ``frozen``: the lowest spinors (an even number) go into the core energy
    and the one-electron matrix; ``nvir``: the number of virtual spinors
    kept (the lowest; default all).  The two-electron integrals are one
    array (pq|rs) over the active spinors - for the sizes of high-order CC -
    made by PySCF's integral transformation.'''
    from zccsdt.eris import _chem_block, _mo_chemist, _orbital_kind
    if frozen % 2 or (nvir or 0) % 2:
        raise ValueError('spinors are frozen and dropped in pairs: %d, %s' % (frozen, nvir))
    Cu, Cb, e = kramers_pairs(mf, mo_coeff, mo_energy)
    nkr_all = Cu.shape[1]
    occ = np.asarray(mf.mo_occ) > 0
    e_all = np.asarray(mf.mo_energy if mo_energy is None else mo_energy)
    nocc = int(occ.sum()) // 2
    if occ.sum() % 2 or (occ.any() and (~occ).any() and e_all[occ].max() > e_all[~occ].min()):
        raise ValueError('the occupied spinors are not the lowest Kramers pairs')
    last = nkr_all if nvir is None else min(nkr_all, nocc + nvir // 2)
    core, act = slice(0, frozen // 2), slice(frozen // 2, last)
    C = np.hstack((Cu[:, act], Cb[:, act]))
    nkr = last - frozen // 2
    hcore = mf.get_hcore()
    e_core = float(mf.energy_nuc())
    heff = hcore
    if frozen:
        Cc = np.hstack((Cu[:, core], Cb[:, core]))
        dm = Cc @ Cc.conj().T
        vhf = mf.get_veff(mf.mol, dm)
        heff = hcore + vhf
        e_core += float(np.einsum('ij,ji->', hcore + 0.5 * vhf, dm).real)
    h = C.conj().T @ heff @ C
    kind = _orbital_kind(mf)
    if kind == 'ghf' and np.iscomplexobj(C):
        g = _mo_chemist(mf.mol, C, kind)
    else:
        g = _chem_block(mf.mol, kind, C, C, C, C)
    im = max(abs(np.imag(h)).max(), abs(np.imag(g)).max())
    if im < tol:
        h, g = np.ascontiguousarray(np.real(h)), np.ascontiguousarray(np.real(g))
    n = 2 * nkr
    kr = np.stack((np.arange(nkr), np.arange(nkr) + nkr), axis=1)
    half = np.repeat([1, 2], nkr)
    x = Integrals(h, g, e_core, np.concatenate((e[act], e[act])), half, half.copy(),
                  np.array([nocc - frozen // 2] * 2), 1 if im < tol else 2, kr, None, float(mf.e_tot))
    # the pairing that the files assume: (-I -J | -K -L) = sign(I J K L) (I J | K L)^*
    s = np.repeat([1.0, -1.0], nkr)
    bar = np.concatenate((np.arange(nkr) + nkr, np.arange(nkr)))
    gt = g[np.ix_(bar, bar, bar, bar)].conj() * np.einsum('p,q,r,s->pqrs', s, s, s, s)
    if abs(gt - g).max() > 1e-8:
        raise ValueError('the spinors are not in Kramers pairs: the integrals of the partners differ by %.1e'
                         % abs(gt - g).max())
    return x
