'''MO integrals of CFOUR's two-component (X2C, spin-orbit) programs as an
integral container.

The chain ``xjoda; xsox2catm; xsox2chf; xx2cvtran`` of cfour-x2cso leaves in
its directory (formats: ``x2cvtran/x2c_vtrandrv.f90``, ``vsort.f90`` and
``vsortp.f90``, ``sox2chf/dmpsohf.f90``, ``sox2chf/dgdropmo2c.f90``; -i8
build):

JAINDX, JOBARC   joda's archive: the SCF energy (SCFENEG), the number of
                 frozen Kramers pairs (NUMDROPA) and of dropped virtual
                 spinors (NUMVIRTA, with ``%vircut``).
nmototal         the number of spinors of the SCF.
scf2ceig         their energies, in the order of the transformation: the
                 frozen ones, the active occupied, the active virtual, the
                 dropped virtual ones.
dgcosym          the irrep of the double group of each spinor (same order).
moinhhhh         <ij||kl>   on pairs i<=j, k<=l
moinhhhp         <ij||ka>   on pairs i<=j, full k, a
moinhhpp         <ij||ab>   all elements, stored [a, i, b, j]
moinpphh         <ab||ij>   on pairs a<=b, i<=j
moinphhp         <ia||bj>   all elements, stored [b, i, a, j]
PPPH0001, ..     <ab||ci>   on pairs a<=b, full c: one file per occupied i
moinpppp         <ab||cd>   on pairs a<=b, c<=d

Every file is one record of an unformatted sequential file, complex numbers
as (re, im), the first index running fastest (pairs: the smaller index
fastest); the pairs include the diagonal, which is zero.  The integrals are
antisymmetrized already, in the convention of this program:
<pq||rs> = (pr|qs) - (ps|qr), the spinors p, q conjugated.  Each block is
there once: nothing of <ia||bc> or <ka||ij> (they are the conjugates of
<ab||ci> and <ij||ka>), and of the blocks on pairs the other three
quarters follow from the antisymmetry.  from_cfour() completes them.

``moinpppp`` is the one file that is not always there.  xx2cvtran writes it
when ABCDTYPE is not AOBASIS (ABCDTYPE=STANDARD: then the whole (pq|rs) is
transformed, n^4 complex numbers in core); with ABCDTYPE=AOBASIS, which the
CCSD of CFOUR is usually run with, <ab||cd> is never made.  That switch
needs an xx2cvtran built from sources that have ``vsort.f90`` and
``vtranso2.f90`` (in the tree since 2026-08).

There is no one-electron matrix in the files and none is needed: the
spinors are the canonical ones of the SCF, the Fock matrix of the active
spinors is the diagonal matrix of their energies (frozen spinors included
in it), and the SCF energy is in JOBARC.
'''

import os
from dataclasses import dataclass

import numpy as np

from zccsdt.eris import ERIs

_I8, _F8 = np.dtype('<i8'), np.dtype('<f8')
_NKEYS, _WORD = 1000, 8          # entries of JAINDX; bytes of an integer word (-i8)


def _record(path, dtype=_F8):
    '''The one record of a file written by CFOUR's wfile.'''
    with open(path, 'rb') as f:
        buf = f.read()
    n = int(np.frombuffer(buf, '<i4', 1, 0)[0])
    if n < 0 or n + 8 != len(buf) or int(np.frombuffer(buf, '<i4', 1, 4 + n)[0]) != n:
        raise ValueError('%s: not one record of a sequential unformatted file' % path)
    return np.frombuffer(buf, dtype, n // np.dtype(dtype).itemsize, 4)


def _complex(path, shape):
    '''A complex array of a file, in Fortran's order of the indices.'''
    a = _record(path)
    if a.size != 2 * int(np.prod(shape)):
        raise ValueError('%s holds %d numbers, not the %d of %s (complex)' % (path, a.size, 2 * int(np.prod(shape)), shape))
    return (a[0::2] + 1j * a[1::2]).reshape(shape, order='F')


class JobArc:
    '''The records of joda's archive (JOBARC, with its directory JAINDX) of
    a CFOUR build with integers of 8 bytes.'''

    def __init__(self, directory='.'):
        with open(os.path.join(directory, 'JAINDX'), 'rb') as f:
            buf = f.read()
        if len(buf) < 4 + 3 * 8 * _NKEYS:
            raise ValueError('JAINDX of %s: %d bytes (integers of 8 bytes expected)' % (directory, len(buf)))
        self.names = [buf[4 + 8 * i:12 + 8 * i].decode('ascii', 'replace') for i in range(_NKEYS)]
        self.loc = np.frombuffer(buf, _I8, _NKEYS, 4 + 8 * _NKEYS)
        self.size = np.frombuffer(buf, _I8, _NKEYS, 4 + 16 * _NKEYS)
        self.path = os.path.join(directory, 'JOBARC')

    def __contains__(self, key):
        return key.ljust(8) in self.names

    def get(self, key, dtype=_I8, default=None):
        '''The record ``key`` (integer words; dtype '<f8' for reals).'''
        if key not in self:
            if default is None:
                raise KeyError('JOBARC has no record %s' % key)
            return default
        k = self.names.index(key.ljust(8))
        with open(self.path, 'rb') as f:
            f.seek(_WORD * (int(self.loc[k]) - 1))
            return np.frombuffer(f.read(_WORD * int(self.size[k])), dtype)


@dataclass
class Integrals:
    '''What the files of a directory hold, as they are (first index fastest).'''
    nocc: int                   # active occupied spinors
    nvir: int                   # active virtual spinors
    nfrozen: int                # frozen occupied spinors
    ndropped: int               # dropped virtual spinors
    energy: np.ndarray          # energies of the active spinors, occupied first
    irrep: np.ndarray           # their irreps of the double group (dgcosym)
    e_scf: float                # total SCF energy
    hhhh: np.ndarray            # [i<=j, k<=l]   <ij||kl>
    hhhp: np.ndarray            # [i<=j, k, a]   <ij||ka>
    hhpp: np.ndarray            # [a, i, b, j]   <ij||ab>
    pphh: np.ndarray            # [a<=b, i<=j]   <ab||ij>
    phhp: np.ndarray            # [b, i, a, j]   <ia||bj>
    ppph: np.ndarray            # [a<=b, c, i]   <ab||ci>
    pppp: np.ndarray = None     # [a<=b, c<=d]   <ab||cd>; None: the file is not there
    directory: str = ''


def _pairs(n):
    '''(smaller, larger) index of the pairs p<=q in CFOUR's order: the
    smaller index runs fastest, pair p + q (q + 1) / 2.'''
    q, p = np.tril_indices(n)
    return p, q


def _sizes(directory):
    '''(nocc, nvir) from the sizes of moinhhhh and moinhhpp.'''
    def numbers(name):
        return (os.path.getsize(os.path.join(directory, name)) - 8) // 16
    no = int(round((np.sqrt(8 * np.sqrt(numbers('moinhhhh')) + 1) - 1) / 2))
    if no < 1 or (no * (no + 1) // 2) ** 2 != numbers('moinhhhh'):
        raise ValueError('%s: moinhhhh is not a block on pairs of occupied spinors' % directory)
    nv = int(round(np.sqrt(numbers('moinhhpp')) / no))
    if (no * nv) ** 2 != numbers('moinhhpp'):
        raise ValueError('%s: moinhhhh and moinhhpp do not belong together' % directory)
    return no, nv


def read(directory='.', pppp=True):
    '''The Integrals of the files of ``directory``.  ``pppp``: read moinpppp
    if it is there.'''
    d = lambda name: os.path.join(directory, name)
    for name in ('JAINDX', 'JOBARC', 'nmototal', 'scf2ceig', 'moinhhhh', 'moinhhhp', 'moinhhpp', 'moinpphh',
                 'moinphhp', 'PPPH0001'):
        if not os.path.isfile(d(name)):
            raise FileNotFoundError('%s: no file %s (kept after xjoda, xsox2catm, xsox2chf, xx2cvtran?)'
                                    % (directory, name))
    no, nv = _sizes(directory)
    ntot = int(_record(d('nmototal'), _I8)[0])
    arc = JobArc(directory)
    nfrozen = 2 * int(arc.get('NUMDROPA')[0])
    # (xx2cvtran takes NUMVIRTA from the archive; without %vircut the record is not written)
    ndropped = ntot - nfrozen - no - nv
    if ndropped < 0 or (ndropped and ndropped != int(arc.get('NUMVIRTA', default=[-1])[0])):
        raise ValueError('%s: %d spinors, %d frozen, %d occupied and %d virtual ones in the integral files: '
                         'the files are not those of one calculation' % (directory, ntot, nfrozen, no, nv))
    e = _record(d('scf2ceig'))
    if e.size != 2 * ntot:
        raise ValueError('%s: scf2ceig has %d numbers for %d spinors' % (directory, e.size, ntot))
    act = slice(nfrozen, nfrozen + no + nv)
    irrep = (_record(d('dgcosym'), _I8)[act].copy() if os.path.isfile(d('dgcosym'))
             else np.ones(no + nv, dtype=np.int64))
    po, pv = no * (no + 1) // 2, nv * (nv + 1) // 2
    ppph = np.empty((pv, nv, no), dtype=np.complex128, order='F')
    for i in range(no):
        ppph[:, :, i] = _complex(d('PPPH%04d' % (i + 1)), (pv, nv))
    return Integrals(
        nocc=no, nvir=nv, nfrozen=nfrozen, ndropped=ndropped, energy=e[0::2][act].copy(), irrep=irrep,
        e_scf=float(arc.get('SCFENEG', _F8)[0]),
        hhhh=_complex(d('moinhhhh'), (po, po)), hhhp=_complex(d('moinhhhp'), (po, no, nv)),
        hhpp=_complex(d('moinhhpp'), (nv, no, nv, no)), pphh=_complex(d('moinpphh'), (pv, po)),
        phhp=_complex(d('moinphhp'), (nv, no, nv, no)), ppph=ppph,
        pppp=_complex(d('moinpppp'), (pv, pv)) if pppp and os.path.isfile(d('moinpppp')) else None,
        directory=directory)


def _unpack(a, n, axis):
    '''The antisymmetric pair index (p<=q, CFOUR's order) at ``axis`` of
    ``a`` as two indices [p, q].'''
    p, q = _pairs(n)
    a = np.moveaxis(a, axis, 0)
    if a.shape[0] != len(p):
        raise ValueError('%d pairs of %d spinors expected, %d found' % (len(p), n, a.shape[0]))
    if abs(a[p == q]).max(initial=0) > 1e-12:
        raise ValueError('a block on pairs has a diagonal that is not zero: not antisymmetrized integrals')
    out = np.zeros((n, n) + a.shape[1:], dtype=a.dtype)
    out[q, p] = -a
    out[p, q] = a
    out[np.arange(n), np.arange(n)] = 0
    return np.moveaxis(out, (0, 1), (axis, axis + 1))


def container(x, packed_vvvv=True, tol=1e-9):
    '''The integral container (zccsdt.eris.ERIs) of the Integrals ``x``:
    the active spinors in CFOUR's order, occupied first.  ``packed_vvvv``:
    keep <ab||cd> for a<b, c<d only (as the CC code wants it); otherwise
    the full block (the dense reference codes).  Without moinpppp the
    container has no <ab||cd> at all - enough for what does not need it
    (the second-order energy, (T) for given amplitudes), not for an
    iteration.

    <ij||ab> is in the files twice (moinhhpp, and moinpphh conjugated): the
    two are compared (``tol``), which tells files of two calculations apart.'''
    no, nv = x.nocc, x.nvir
    oooo = _unpack(_unpack(x.hhhh, no, 0), no, 2)                               # [i, j, k, l]
    ooov = _unpack(x.hhhp, no, 0)                                               # [i, j, k, a]
    oovv = np.ascontiguousarray(x.hhpp.transpose(1, 3, 0, 2))                   # [a, i, b, j] -> [i, j, a, b]
    vvoo = _unpack(_unpack(x.pphh, nv, 0), no, 2)                               # [a, b, i, j]
    diff = abs(vvoo.conj().transpose(2, 3, 0, 1) - oovv).max()
    if diff > tol:
        raise ValueError('%s: moinhhpp and moinpphh differ by %.1e: not the integrals of one calculation'
                         % (x.directory, diff))
    ovvo = np.ascontiguousarray(x.phhp.transpose(1, 2, 0, 3))                   # [b, i, a, j] -> [i, a, b, j]
    ovov = np.ascontiguousarray(-ovvo.transpose(0, 1, 3, 2))
    # <ia||bc> = <bc||ia>^* = -<bc||ai>^*
    ovvv = np.ascontiguousarray(-_unpack(x.ppph, nv, 0).conj().transpose(3, 2, 0, 1))
    fock = np.diag(x.energy).astype(np.complex128)
    out = ERIs(nocc=no, fock=fock, mo_energy=x.energy.copy(), e_hf=x.e_scf, source='CFOUR, %s' % x.directory,
               oooo=np.ascontiguousarray(oooo), ooov=np.ascontiguousarray(ooov), oovv=oovv, ovov=ovov, ovvo=ovvo,
               ovvv=ovvv, vvvv=None)
    if x.pppp is not None:
        if packed_vvvv:
            # the pairs a<b in the order of the CC code (the first index slowest)
            p, q = _pairs(nv)
            where = np.full((nv, nv), -1)
            where[p, q] = np.arange(len(p))
            a, b = np.triu_indices(nv, 1)
            sel = where[a, b]
            out._vvvv_pp = np.ascontiguousarray(x.pppp[np.ix_(sel, sel)])
        else:
            out.vvvv = np.ascontiguousarray(_unpack(_unpack(x.pppp, nv, 0), nv, 2))
    out.spinors = x.nfrozen + np.arange(no + nv)
    out.irrep = x.irrep
    return out


def from_cfour(directory='.', packed_vvvv=True):
    '''The integral container of the active spinors of a calculation of
    CFOUR's two-component programs (closed shell) from the files that
    xx2cvtran left in ``directory``: container() of read().  The CC classes
    need <ab||cd>, the file moinpppp (ABCDTYPE=STANDARD).'''
    x = read(directory)
    if x.pppp is None:
        raise FileNotFoundError(
            '%s: no file moinpppp.  xx2cvtran writes <ab||cd> only when ABCDTYPE is not AOBASIS '
            '(ABCDTYPE=STANDARD), and only if it was built from sources that have vsort.f90'
            % directory)
    return container(x, packed_vvvv)


def mp2(eris):
    '''The second-order energy of a container - what xx2cvtran prints as
    "mp2 energy".'''
    no = eris.nocc
    e = eris.mo_energy
    d = e[:no, None, None, None] + e[None, :no, None, None] - e[None, None, no:, None] - e[None, None, None, no:]
    return float(0.25 * (abs(np.asarray(eris.oovv)) ** 2 / d).sum())
