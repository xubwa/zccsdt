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
    mdcint_header: bytes        # the first record of MDCINT

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
                     group_type, kramers.copy(), m[0])


def from_dirac(mrconee='MRCONEE', mdcint='MDCINT', packed_vvvv=True):
    '''The integral container of the active spinors of a DIRAC calculation
    (closed shell), occupied spinors first, each set by energy; its
    ``spinors`` are their numbers in the files.  ``packed_vvvv``: keep
    <ab||cd> for a<b, c<d only (as the CC code wants it); otherwise the full
    block (the dense reference codes).'''
    x = read(mrconee, mdcint)
    n = len(x.energy)
    occ = x.occupied
    vir = np.setdiff1d(np.arange(n), occ)
    occ, vir = occ[np.argsort(x.energy[occ], kind='stable')], vir[np.argsort(x.energy[vir], kind='stable')]
    idx = np.concatenate((occ, vir))
    no = len(occ)
    h, g = x.h[np.ix_(idx, idx)], x.g[np.ix_(idx, idx, idx, idx)]
    del x.g
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
               source='DIRAC', oooo=oooo, vvvv=None if packed_vvvv else block('vvvv'), **blocks)
    if packed_vvvv:
        a, b = np.triu_indices(n - no, 1)
        a, b = a + no, b + no
        out._vvvv_pp = np.ascontiguousarray(g[a[:, None], a[None, :], b[:, None], b[None, :]]
                                            - g[a[:, None], b[None, :], b[:, None], a[None, :]])
    out.spinors, out.dirac = idx, x
    return out


def write_mdcint(path, x, threshold=1e-15):
    '''MDCINT with all integrals (I J | K L), I unbarred, of the Integrals
    ``x`` - the complete list, in DIRAC's records.  Returns their number.'''
    nkr = len(x.kramers)
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
