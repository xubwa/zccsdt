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

The front end (docs/usage.md, "CFOUR's two-component programs"):

    python -m zccsdt.cfour RUNDIR               the CC that the ZMAT asks for
    python -m zccsdt.cfour --chain RUNDIR       CFOUR's programs first

job() reads the method, the thresholds and the memory from the ZMAT and
from joda's flags in JOBARC, run_programs() calls CFOUR's executables and
checks each of them, run() does the calculation; write() is the other
direction, a container in CFOUR's files.
'''

import os
import shutil
from dataclasses import dataclass

import numpy as np

from zccsdt.eris import ERIs

_I8, _F8 = np.dtype('<i8'), np.dtype('<f8')
_NKEYS, _WORD = 1000, 8          # entries of JAINDX; bytes of an integer word (-i8)
_RECORD = 128                    # words of a record of JOBARC


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

    def __init__(self, directory='.', create=False):
        self.index = os.path.join(directory, 'JAINDX')
        self.path = os.path.join(directory, 'JOBARC')
        if create:
            self.names = ['OPENSLOT'] * _NKEYS
            self.loc, self.size, self.tail = np.zeros(_NKEYS, dtype=_I8), np.zeros(_NKEYS, dtype=_I8), b''
            open(self.path, 'wb').close()
            self._write_index()
            return
        with open(self.index, 'rb') as f:
            buf = f.read()
        if len(buf) < 4 + 3 * 8 * _NKEYS + 8:
            raise ValueError('JAINDX of %s: %d bytes (integers of 8 bytes expected)' % (directory, len(buf)))
        self.names = [buf[4 + 8 * i:12 + 8 * i].decode('ascii', 'replace') for i in range(_NKEYS)]
        self.loc = np.frombuffer(buf, _I8, _NKEYS, 4 + 8 * _NKEYS).copy()
        self.size = np.frombuffer(buf, _I8, _NKEYS, 4 + 16 * _NKEYS).copy()
        # after the directory: the number of records of JOBARC, and whatever else the build writes
        self.tail = buf[4 + 24 * _NKEYS + 8:-4]

    def _write_index(self):
        nrec = -(-os.path.getsize(self.path) // (_WORD * _RECORD))
        body = (''.join(self.names).encode('ascii') + self.loc.astype(_I8).tobytes() + self.size.astype(_I8).tobytes()
                + np.array([nrec], dtype=_I8).tobytes() + self.tail)
        mark = np.array([len(body)], dtype='<i4').tobytes()
        with open(self.index, 'wb') as f:
            f.write(mark + body + mark)

    def put(self, key, values):
        '''Write the record ``key`` (an array of integers or reals of 8
        bytes) as CFOUR's PUTREC does: over the old one if there is one (not
        longer than it), else after the last record, with a new entry in
        JAINDX.'''
        values = np.ascontiguousarray(values)
        if values.dtype.itemsize != _WORD:
            raise ValueError('records are made of words of %d bytes' % _WORD)
        name = key.ljust(8)
        if name in self.names:
            k = self.names.index(name)
            if values.size > self.size[k]:
                raise ValueError('JOBARC: %d words for the record %s of %d' % (values.size, key, self.size[k]))
        else:
            k = self.names.index('OPENSLOT')
            self.names[k], self.size[k] = name, values.size
            self.loc[k] = self.loc[k - 1] + self.size[k - 1] if k else 1
        with open(self.path, 'r+b') as f:
            f.seek(_WORD * (int(self.loc[k]) - 1))
            f.write(values.tobytes())
            # (direct access: whole records)
            end = f.seek(0, 2)
            f.write(b'\0' * (-end % (_WORD * _RECORD)))
        self._write_index()

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
    out.irrep, out.cfour = x.irrep, x
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
    out = container(x, packed_vvvv)
    for name in ('hhhh', 'hhhp', 'hhpp', 'pphh', 'phhp', 'ppph', 'pppp'):         # (the copies of the files)
        setattr(x, name, None)
    return out


def mp2(eris):
    '''The second-order energy of a container - what xx2cvtran prints as
    "mp2 energy".'''
    no = eris.nocc
    e = eris.mo_energy
    d = e[:no, None, None, None] + e[None, :no, None, None] - e[None, None, no:, None] - e[None, None, None, no:]
    return float(0.25 * (abs(np.asarray(eris.oovv)) ** 2 / d).sum())


# ---------------------------------------------------------------------------
# the other direction: a container in CFOUR's files (tests, other programs)

def _write_record(path, a):
    a = np.ascontiguousarray(a)
    mark = np.array([a.nbytes], dtype='<i4').tobytes()
    with open(path, 'wb') as f:
        f.write(mark + a.tobytes() + mark)


def _write_complex(path, a):
    a = np.asarray(a, dtype=np.complex128).ravel(order='F')
    _write_record(path, np.stack((a.real, a.imag), axis=1).ravel())


def write(directory, eris, nfrozen=0, flags=None):
    '''The files that read() takes, for the container ``eris`` (full or
    packed <ab||cd>; canonical spinors: the Fock matrix diagonal), with a
    fresh JOBARC that holds the SCF energy, the frozen pairs and joda's
    flags (``flags``: {number from 1: value} of IFLAGS, e.g. 2: the level
    of the calculation, 4: CC_CONV, 7: CC_MAXCYC, 36: the memory in words).
    The energies of the ``nfrozen`` frozen spinors are written as zeros.'''
    os.makedirs(directory, exist_ok=True)
    d = lambda name: os.path.join(directory, name)
    no, nv = eris.nocc, eris.nmo - eris.nocc
    if abs(eris.fock - np.diag(np.diag(eris.fock))).max() > 1e-8:
        raise ValueError('not canonical spinors: the files have no Fock matrix, only its diagonal')
    po, pv = _pairs(no), _pairs(nv)
    oooo, ooov, oovv, ovvo, ovvv = (np.asarray(getattr(eris, k)) for k in ('oooo', 'ooov', 'oovv', 'ovvo', 'ovvv'))
    _write_complex(d('moinhhhh'), oooo[po[0], po[1]][:, po[0], po[1]])
    _write_complex(d('moinhhhp'), ooov[po[0], po[1]])
    _write_complex(d('moinhhpp'), oovv.transpose(2, 0, 3, 1))
    _write_complex(d('moinpphh'), oovv.conj().transpose(2, 3, 0, 1)[pv[0], pv[1]][:, po[0], po[1]])
    _write_complex(d('moinphhp'), ovvo.transpose(2, 0, 1, 3))
    ppph = -ovvv.conj().transpose(2, 3, 1, 0)[pv[0], pv[1]]            # [a<=b, c, i]
    for i in range(no):
        _write_complex(d('PPPH%04d' % (i + 1)), ppph[:, :, i])
    if eris.vvvv is not None:
        _write_complex(d('moinpppp'), np.asarray(eris.vvvv)[pv[0], pv[1]][:, pv[0], pv[1]])
    elif getattr(eris, '_vvvv_pp', None) is not None:
        a, b = np.triu_indices(nv, 1)
        where = np.full((nv, nv), -1)
        where[pv[0], pv[1]] = np.arange(len(pv[0]))
        sel = where[a, b]
        full = np.zeros((len(pv[0]),) * 2, dtype=np.complex128)
        full[np.ix_(sel, sel)] = np.asarray(eris._vvvv_pp)
        _write_complex(d('moinpppp'), full)
    e = np.concatenate((np.zeros(nfrozen), np.asarray(eris.mo_energy, dtype=float)))
    _write_record(d('scf2ceig'), np.stack((e, np.zeros_like(e)), axis=1).ravel())
    _write_record(d('nmototal'), np.array([len(e)], dtype=_I8))
    _write_record(d('dgcosym'), np.ones(len(e), dtype=_I8))
    arc = JobArc(directory, create=True)
    iflags = np.zeros(100, dtype=_I8)
    for k, v in (flags or {}).items():
        iflags[k - 1] = v
    arc.put('IFLAGS', iflags)
    arc.put('NUMDROPA', np.array([nfrozen // 2], dtype=_I8))
    arc.put('SCFENEG', np.array([eris.e_hf], dtype=_F8))


# ---------------------------------------------------------------------------
# the input of CFOUR as the input of the CC here

# CALC of joda (IFLAGS(2)) -> method
LEVELS = {0: 'scf', 10: 'ccsd', 22: 'ccsd(t)', 18: 'ccsdt', 30: 'ccsdt[q]', 54: 'ccsdt(q)', 46: 'ccsdtq'}
METHODS = ('ccsd', 'ccsd(t)', 'ccsdt', 'ccsdt[q]', 'ccsdt(q)', 'ccsdt(q)_l', 'ccsdtq', 'ccsdtq(p)')
# CALC=CC(n) and CC(n-1)(n) of CFOUR with EXCITATION=n (IFLAGS2(38))
BY_EXCITATION = {'cc(n)': {2: 'ccsd', 3: 'ccsdt', 4: 'ccsdtq'}, 'cc(n-1)(n)': {3: 'ccsd(t)', 4: 'ccsdt(q)', 5: 'ccsdtq(p)'}}
LEVELS_N = {49: 'cc(n)', 53: 'cc(n-1)(n)'}
SECTION = 'zccsdt'              # the section of the ZMAT with what CFOUR does not know
PROGRAMS = ('xjoda', 'xsox2catm', 'xsox2chf', 'xx2cvtran')
BIN = '/home/xwang405/apps/cfour-x2cso/bin'
# the line a program ends with when it is not the one of CFOUR's Fortran programs (@CHECKOUT-I)
ENDS = {'xx2camf': 'xx2camf FINISHED!'}
# what a program must leave behind
LEAVES = {'xjoda': ('JOBARC', 'JAINDX', 'MOL'), 'xsox2chf': ('scf2ceig', 'nmototal'),
          'xx2cvtran': ('moinhhhh', 'moinhhhp', 'moinhhpp', 'moinpphh', 'moinphhp', 'PPPH0001')}


def parse_zmat(path='ZMAT'):
    '''(keywords, sections) of a ZMAT: the keywords of the ``*CFOUR(...)``
    block as {NAME: value} (names in capitals, as written - joda also takes
    abbreviations, which are not resolved here), and the ``%name`` sections
    as {name: [lines]} (names in lower case, without a trailing ``*``; a
    section ends at an empty line).'''
    with open(path) as f:
        lines = f.read().splitlines()
    keywords, sections = {}, {}
    k = next((i for i, s in enumerate(lines) if s.strip().upper().startswith(('*CFOUR(', '*ACES2(', '*CRAPS('))), None)
    if k is not None:
        text, depth = '', 0
        for s in lines[k:]:
            s = s.strip()
            text += s + '\n'
            depth += s.count('(') - s.count(')')
            if depth <= 0:
                break
        text = text[text.index('(') + 1:text.rindex(')')]
        for item in text.replace('\n', ',').split(','):
            if item.strip():
                name, _, value = item.partition('=')
                keywords[name.strip().upper()] = value.strip()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        i += 1
        if s.startswith('%'):
            name = s[1:].split('=')[0].strip().rstrip('*').lower()
            body = []
            while i < len(lines) and lines[i].strip():
                body.append(lines[i].strip())
                i += 1
            sections[name] = body
    return keywords, sections


@dataclass
class Job:
    '''What the CC step is asked to do.'''
    method: str
    conv_tol: float
    conv_tol_normt: float
    max_cycle: int
    max_memory: float           # MB; None: the default of the CC classes
    source: dict = None         # where each setting came from
    max_disk: float = None      # MB for the scratch files (``max_disk=`` of %zccsdt); None: the default


def job(directory='.', method=None, max_memory=None):
    '''The Job of a run directory.  The method: the argument, else
    ``method=`` of the section %zccsdt of the ZMAT, else CALC of the ZMAT.
    Thresholds: ``conv_tol``, ``conv_tol_normt``, ``max_cycle`` of %zccsdt,
    else CC_CONV=N and CC_MAXCYC as joda understood them (IFLAGS of JOBARC):
    the change of the amplitudes below 10^-N, the energy below 10^-(N+2).
    Memory (MB): the argument, ``max_memory=`` of %zccsdt, else MEMORY_SIZE
    of CFOUR (in JOBARC in words of 8 bytes).  Disk (MB): ``max_disk=`` of
    %zccsdt (the CCSDT and CCSDTQ classes; default: theirs).'''
    zmat = os.path.join(directory, 'ZMAT')
    keywords, sections = parse_zmat(zmat) if os.path.isfile(zmat) else ({}, {})
    own = {}
    for line in sections.get(SECTION, []):
        name, _, value = line.replace('=', ' ', 1).partition(' ')
        own[name.strip().lower()] = value.strip()
    unknown = set(own) - {'method', 'max_memory', 'max_disk', 'conv_tol', 'conv_tol_normt', 'max_cycle'}
    if unknown:
        raise ValueError('%%%s of the ZMAT: unknown %s' % (SECTION, ', '.join(sorted(unknown))))
    flags = JobArc(directory).get('IFLAGS')
    src = {}
    calc = next((keywords[k] for k in ('CALC', 'CALCLEVEL', 'CALC_LEVEL') if k in keywords), None)
    for value, where in ((method, 'argument'), (own.get('method'), '%' + SECTION), (calc, 'CALC of the ZMAT'),
                         (LEVELS.get(int(flags[1]), LEVELS_N.get(int(flags[1]))), 'JOBARC')):
        if value:
            method, src['method'] = value.lower().replace(' ', ''), where
            break
    if method in BY_EXCITATION:
        n = int(JobArc(directory).get('IFLAGS2')[37])
        if n not in BY_EXCITATION[method]:
            raise ValueError('CALC=%s with EXCITATION=%d: here EXCITATION is one of %s'
                             % (method.upper(), n, ', '.join(str(k) for k in BY_EXCITATION[method])))
        method, src['method'] = BY_EXCITATION[method][n], src['method'] + ', EXCITATION=%d' % n
    if method not in METHODS:
        raise ValueError('method %s (%s): not one of %s' % (method, src.get('method'), ', '.join(METHODS)))
    n = int(flags[3])
    conv_normt, conv_e, cycles = 10.0 ** -n, 10.0 ** -(n + 2), int(flags[6])
    src.update(conv_tol='CC_CONV', conv_tol_normt='CC_CONV', max_cycle='CC_MAXCYC')
    for name in ('conv_tol', 'conv_tol_normt', 'max_cycle'):
        if name in own:
            src[name] = '%' + SECTION
    conv_e = float(own.get('conv_tol', conv_e))
    conv_normt = float(own.get('conv_tol_normt', conv_normt))
    cycles = int(own.get('max_cycle', cycles))
    for value, where in ((max_memory, 'argument'), (own.get('max_memory'), '%' + SECTION),
                         (8e-6 * int(flags[35]) or None, 'MEMORY_SIZE')):
        if value:
            max_memory, src['max_memory'] = float(value), where
            break
    return Job(method, conv_e, conv_normt, cycles, max_memory, src,
               float(own['max_disk']) if own.get('max_disk') else None)


def programs(directory='.'):
    '''The programs of CFOUR that lead from a ZMAT to the integral files:
    xjoda, xsox2catm, xsox2chf, xx2cvtran - with xx2camf after xjoda when
    the ZMAT has a section %amfiMethod (X2CAMF).'''
    _, sections = parse_zmat(os.path.join(directory, 'ZMAT'))
    out = list(PROGRAMS)
    if 'amfimethod' in sections:
        out.insert(1, 'xx2camf')
    return out


class ProgramFailed(RuntimeError):
    pass


def _preload(exe):
    '''LD_PRELOAD for a program whose libstdc++ is newer than the one of
    the environment (xx2camf): the library of CFOUR_STDCXX, else the one
    next to the Python that runs this.'''
    import subprocess
    import sys
    try:
        out = subprocess.run(['ldd', exe], capture_output=True, text=True).stdout
    except OSError:
        return None
    if 'GLIBCXX' not in out or 'not found' not in out:
        return None
    lib = os.environ.get('CFOUR_STDCXX') or os.path.join(sys.prefix, 'lib', 'libstdc++.so.6')
    return lib if os.path.isfile(lib) else None


def run_programs(directory='.', progs=None, bindir=None, before=None, log=print):
    '''Run programs of CFOUR one after the other in ``directory`` (which has
    ZMAT and GENBAS), each with its output in <program>.out and a line
    with its times in <program>.time.  After each one: the exit status,
    the line that CFOUR's programs end with (@CHECKOUT-I; ENDS), no "Something
    very wrong", and the files it must leave - otherwise ProgramFailed
    names the program and shows the end of its output.  ``bindir``: the
    directory of the executables (default: CFOUR_BIN, else BIN);
    ``before``: {program: shell command run in the directory before it}.'''
    import subprocess
    import time
    bindir = bindir or os.environ.get('CFOUR_BIN') or BIN
    progs = list(progs or programs(directory))
    for p in progs:
        exe = os.path.join(bindir, p)
        if not os.access(exe, os.X_OK):
            raise ProgramFailed('%s: no executable %s (CFOUR_BIN, --bin)' % (p, exe))
    for p in progs:
        if before and before.get(p):
            subprocess.run(before[p], shell=True, cwd=directory, check=False)
        exe = os.path.join(bindir, p)
        env = dict(os.environ)
        lib = _preload(exe)
        if lib:
            env['LD_PRELOAD'] = lib
        out = os.path.join(directory, p + '.out')
        t0 = time.perf_counter()
        with open(out, 'w') as f:
            proc = subprocess.Popen([exe], cwd=directory, stdout=f, stderr=subprocess.PIPE, env=env)
            err = proc.stderr.read().decode(errors='replace')
            _, status, usage = os.wait4(proc.pid, 0)
        code = os.waitstatus_to_exitcode(status)
        proc.returncode = code
        line = '%s wall %.2f s  user %.2f s  max RSS %d KB' % (p, time.perf_counter() - t0, usage.ru_utime, usage.ru_maxrss)
        with open(os.path.join(directory, p + '.time'), 'w') as f:
            f.write(err + line + '\n')
        log(line)
        with open(out, errors='replace') as f:
            text = f.read()
        end = ENDS.get(p, '@CHECKOUT-I')
        missing = [name for name in LEAVES.get(p, ()) if not os.path.isfile(os.path.join(directory, name))]
        why = ('exit status %d' % code if code
               else 'its output has "Something very wrong"' if 'Something very wrong' in text
               else 'its output does not end with %s' % end if end not in text
               else 'it did not write %s' % ', '.join(missing) if missing else None)
        if why:
            tail = '\n'.join(text.splitlines()[-20:])
            raise ProgramFailed('%s FAILED: %s.  The end of %s:\n%s%s'
                                % (p, why, out, tail, ('\n' + err) if err.strip() else ''))
    return progs


def read_mol(path='MOL'):
    '''[(symbol, (x, y, z))] of CFOUR's MOL file: its frame, bohr, its order.'''
    import re
    atoms = []
    with open(path) as f:
        for line in f:
            m = re.match(r'^([A-Za-z]{1,2})\s*#\s*\d+\s+(\S+)\s+(\S+)\s+(\S+)\s*$', line)
            if m:
                atoms.append((m.group(1).capitalize(), tuple(float(x) for x in m.groups()[1:])))
    if not atoms:
        raise ValueError('no atoms found in %s' % path)
    return atoms


def molecule(directory='.'):
    '''The PySCF molecule of a run directory: CFOUR's geometry in its own
    frame (MOL, bohr) with the basis of the ZMAT read from the GENBAS file
    there - BASIS=SPECIAL with one line ATOM:NAME per atom after the
    keywords, or BASIS=NAME for all atoms.'''
    import re
    from pyscf import gto
    from zccsdt.genbas import read_genbas
    atoms = read_mol(os.path.join(directory, 'MOL'))
    keywords, _ = parse_zmat(os.path.join(directory, 'ZMAT'))
    name = keywords.get('BASIS', '')
    if name.upper() == 'SPECIAL':
        with open(os.path.join(directory, 'ZMAT')) as f:
            keys = [s.strip() for s in f.read().splitlines() if re.match(r'^\s*[A-Za-z]{1,2}:\S+\s*$', s)]
        if len(keys) != len(atoms):
            raise ValueError('%s: %d basis lines in the ZMAT for %d atoms' % (directory, len(keys), len(atoms)))
    else:
        keys = ['%s:%s' % (sym.upper(), name) for sym, _ in atoms]
    genbas = os.path.join(directory, 'GENBAS')
    basis, geom = {}, []
    for k, ((sym, xyz), key) in enumerate(zip(atoms, keys)):
        if key.split(':')[0].upper() != sym.upper():
            raise ValueError('%s: basis line %s for the atom %s (the order of MOL is not that of the ZMAT)'
                             % (directory, key, sym))
        label = '%s%d' % (sym, k + 1)
        basis[label] = read_genbas(genbas, key)
        geom.append((label, xyz))
    return gto.M(atom=geom, unit='Bohr', basis=basis, verbose=0)


def reference(directory='.'):
    '''CFOUR's SCF as a PySCF object without an SCF calculation here, and
    with nothing of the Hamiltonian made here: the molecule (molecule()),
    CFOUR's spinors in PySCF's spinor basis (scf2ccmo; socutils maps the
    basis), their energies (scf2ceig), occupations and the SCF energy.  The
    one-electron Hamiltonian of the object is the one that gives CFOUR's
    Fock matrix with these spinors, h = S C e C+ S - G(D): whatever CFOUR's
    was (X2C-1e, atomic mean field, ...), as long as its two-electron
    operator is the bare Coulomb operator.  Returns (mf, frozen occupied
    spinors, dropped virtual spinors).'''
    from pyscf import x2c
    try:
        from socutils.tools.label import Labels
    except ImportError as err:
        raise ImportError('socutils is needed to map CFOUR\'s spinor basis to PySCF\'s (socutils.tools.label)') from err
    mol = molecule(directory)
    n = mol.nao_2c()
    ntot = int(_record(os.path.join(directory, 'nmototal'), _I8)[0])
    if ntot != n:
        raise ValueError('%s: %d spinors in CFOUR, %d in the basis read from GENBAS' % (directory, ntot, n))
    arc = JobArc(directory)
    C = Labels.spinor_cfour2pyscf(mol).conj().T @ _complex(os.path.join(directory, 'scf2ccmo'), (n, n))
    S = mol.intor('int1e_ovlp_spinor')
    err = abs(C.conj().T @ S @ C - np.eye(n)).max()
    if err > 1e-8:
        raise ValueError('%s: CFOUR\'s spinors are not orthonormal in the basis read from GENBAS (%.1e)'
                         % (directory, err))
    mf = x2c.UHF(mol)
    mf.mo_coeff = C
    mf.mo_energy = _record(os.path.join(directory, 'scf2ceig'))[0::2].copy()
    nfrozen = 2 * int(arc.get('NUMDROPA')[0])
    ndropped = int(arc.get('NUMVIRTA', default=[0])[0])
    mf.mo_occ = (np.arange(n) < mol.nelectron).astype(float)
    mf.e_tot, mf.converged = float(arc.get('SCFENEG', _F8)[0]), True
    SC = S @ C
    hcore = (SC * mf.mo_energy) @ SC.conj().T - mf.get_veff(mol, mf.make_rdm1(C, mf.mo_occ))
    mf.get_hcore = lambda *args, **kwargs: hcore
    return mf, nfrozen, ndropped


def write_density(directory, mf, dm_ao):
    '''A one-particle density in the spinor AO basis of ``mf``
    (reference()) as the file ``denfinal`` of the directory: in the basis of
    CFOUR's spinors, the SCF part included, D[p,q] = <q+ p> - what xx2czvec
    leaves for xsox2cprop, which evaluates all its operators with it.  A
    denfinal that is there already is kept once as denfinal.cfour.'''
    C0 = np.asarray(mf.mo_coeff)
    SC = mf.mol.intor('int1e_ovlp_spinor') @ C0
    D = SC.conj().T @ dm_ao @ SC
    path = os.path.join(directory, 'denfinal')
    if os.path.isfile(path) and not os.path.isfile(path + '.cfour'):
        shutil.copy(path, path + '.cfour')
    _write_complex(path, D)
    return D


def wants_density(directory='.'):
    '''Whether the ZMAT asks for first-order properties (PROPS other than
    OFF), i.e. for the relaxed density.'''
    zmat = os.path.join(directory, 'ZMAT')
    keywords = parse_zmat(zmat)[0] if os.path.isfile(zmat) else {}
    value = next((keywords[k] for k in ('PROPS', 'PROP', 'PROPERTIES') if k in keywords), 'OFF')
    return value.upper() not in ('OFF', '0', '')


def run_direct(directory='.', j=None, verbose=4, stdout=None, kramers=True, density=False):
    '''CCSD and CCSD(T) of zccsdt.zccsd.DirectZCCSD - AO integrals made
    here, the particle-particle ladder in the AO basis, Kramers pairs -, or
    CCSDT of zccsdt.CCSDT, on CFOUR's spinors (reference()): after xjoda,
    xsox2catm and xsox2chf, no integral file of xx2cvtran is read.
    ``density``: also Lambda, the orbital gradient and the Z-vector
    equations, and the relaxed one-particle density written for xsox2cprop
    (write_density).  Returns {name: total energy}.'''
    import sys
    from zccsdt.zccsd import DirectZCCSD
    stdout = stdout or sys.stdout
    j = j or job(directory)
    mf, nfrozen, ndropped = reference(directory)
    n = len(mf.mo_energy)
    frozen = list(range(nfrozen)) + list(range(n - ndropped, n)) if ndropped else nfrozen
    if j.method in ('ccsdt', 'ccsdt(q)'):
        import zccsdt
        with_q = j.method == 'ccsdt(q)'
        cc = zccsdt.CCSDT(mf, frozen=frozen)
        cc.verbose, cc.stdout = verbose, stdout
        cc.conv_tol, cc.conv_tol_normt, cc.max_cycle = j.conv_tol, j.conv_tol_normt, j.max_cycle
        if j.max_memory:
            cc.max_memory = j.max_memory
        cc.max_disk = j.max_disk
        cc.kernel()
        if not cc.converged:
            raise RuntimeError('CCSDT did not converge in %d cycles' % j.max_cycle)
        e = {'scf': float(mf.e_tot), 'ccsdt': float(cc.e_tot)}
        if with_q:
            e_q = cc.ccsdt_q()
            e['ccsdt[q]'], e['ccsdt(q)'] = float(cc.e_tot + cc.e_q5), float(cc.e_tot + e_q)
        if density:
            cc.solve_lambda(with_q=with_q)
            if not cc.converged_lambda:
                raise RuntimeError('the Lambda equations did not converge in %d cycles' % j.max_cycle)
            write_density(directory, mf, cc.make_rdm1(with_q=with_q, ao_repr=True))
        return e
    cc = DirectZCCSD(mf, frozen=frozen)
    cc.verbose, cc.stdout, cc.kramers = verbose, stdout, kramers
    cc.conv_tol, cc.conv_tol_normt, cc.max_cycle = j.conv_tol, j.conv_tol_normt, j.max_cycle
    if j.max_memory:
        cc.max_memory = j.max_memory
    cc.kernel()
    if not cc.converged:
        raise RuntimeError('CCSD did not converge in %d cycles' % j.max_cycle)
    e = {'scf': float(mf.e_tot), 'ccsd': float(mf.e_tot + cc.e_corr)}
    with_t = j.method == 'ccsd(t)'
    if with_t:
        e['ccsd(t)'] = e['ccsd'] + float(cc.ccsd_t())
    if density:
        from zccsdt.zccsd import zvector
        cc.solve_lambda(with_t='gradient') if with_t else cc.solve_lambda()
        relaxed = zvector.kernel(cc, tol=10 * j.conv_tol_normt, with_t=with_t)[0]
        C = zvector.full_space(cc)[0]
        write_density(directory, mf, C @ relaxed @ C.conj().T)
    return e


def run(directory='.', method=None, max_memory=None, verbose=4, stdout=None, jobarc=False, mo_integrals=False,
        density=None):
    '''The CC calculation that the ZMAT of ``directory`` asks for (job()) on
    the integral files there.  Returns {name: total energy}, 'scf' and the
    methods done on the way.  ``jobarc``: write the final energy into JOBARC
    as TOTENERG, the record CFOUR's own CC programs leave for joda.'''
    import sys
    import zccsdt
    from zccsdt.cc import solver
    stdout = stdout or sys.stdout
    j = job(directory, method, max_memory)
    say = lambda s: print(s, file=stdout, flush=True)
    scf_files = all(os.path.isfile(os.path.join(directory, f)) for f in ('scf2ccmo', 'MOL', 'GENBAS', 'ZMAT'))
    if j.method == 'ccsd(t)' and not scf_files:
        raise FileNotFoundError('%s: CCSD(T) is done on CFOUR\'s spinors with AO integrals made here and needs '
                                'the files scf2ccmo, MOL, GENBAS and ZMAT of the run' % directory)
    density = wants_density(directory) if density is None else density
    if density and (j.method not in ('ccsd', 'ccsd(t)', 'ccsdt', 'ccsdt(q)') or not scf_files or mo_integrals):
        raise ValueError('the relaxed density (PROPS of the ZMAT) is there for CCSD, CCSD(T), CCSDT and CCSDT(Q) on '
                         'CFOUR\'s spinors (the files scf2ccmo, MOL, GENBAS), not for %s%s'
                         % (j.method.upper(), ' on the integral files' if mo_integrals else ''))
    if (j.method in ('ccsd', 'ccsd(t)') and scf_files and not (mo_integrals and j.method == 'ccsd')) or \
            (j.method in ('ccsdt', 'ccsdt(q)') and density):
        # the AO-driven code on CFOUR's spinors: no integral files are read
        say('CFOUR run directory %s: %s (%s), AO-driven on CFOUR\'s spinors%s'
            % (os.path.abspath(directory), j.method.upper(), j.source['method'],
               '; relaxed density for xsox2cprop' if density else ''))
        e = run_direct(directory, j, verbose, stdout, density=density)
        if density:
            say('relaxed density written: %s' % os.path.join(os.path.abspath(directory), 'denfinal'))
        for name, value in e.items():
            say('E(%s)%s = %.12f%s' % (name.upper(), ' ' * (10 - len(name)), value,
                                        '' if name == 'scf' else '   correlation %.12f' % (value - e['scf'])))
        if jobarc:
            JobArc(directory).put('TOTENERG', np.array([e[j.method]], dtype=_F8))
            say('TOTENERG of JOBARC: %.12f' % e[j.method])
        return e
    eris = from_cfour(directory)
    x = eris.cfour
    say('CFOUR run directory %s: %s (%s)' % (os.path.abspath(directory), j.method.upper(), j.source['method']))
    say('  %d occupied, %d virtual spinors; %d frozen, %d virtual ones dropped; E(SCF) = %.12f'
        % (x.nocc, x.nvir, x.nfrozen, x.ndropped, x.e_scf))
    say('  conv_tol %.0e (%s)  conv_tol_normt %.0e (%s)  max_cycle %d (%s)  max_memory %s'
        % (j.conv_tol, j.source['conv_tol'], j.conv_tol_normt, j.source['conv_tol_normt'], j.max_cycle,
           j.source['max_cycle'], '%.0f MB (%s)' % (j.max_memory, j.source['max_memory']) if j.max_memory else 'default'))
    e = {'scf': x.e_scf}
    if j.method == 'ccsd':
        r = solver.kernel(eris, 'ccsd', solver.Options(conv_tol=j.conv_tol, conv_tol_normt=j.conv_tol_normt,
                                                       max_cycle=j.max_cycle, max_memory=j.max_memory))
        if not r.converged:
            raise RuntimeError('CCSD did not converge in %d cycles' % j.max_cycle)
        e['ccsd'] = float(r.e_tot)
    else:
        full = j.method.startswith('ccsdtq')
        cc = (zccsdt.GCCSDTQ if full else zccsdt.GCCSDT).from_integrals(eris, verbose=verbose, stdout=stdout)
        cc.conv_tol, cc.conv_tol_normt, cc.max_cycle = j.conv_tol, j.conv_tol_normt, j.max_cycle
        if j.max_memory:
            cc.max_memory = j.max_memory
        cc.max_disk = j.max_disk
        cc.kernel()
        run3 = cc.ccsdt_run if full else cc
        if not (cc.converged and run3.converged):
            raise RuntimeError('%s did not converge in %d cycles' % ('CCSDTQ' if full else 'CCSDT', j.max_cycle))
        e['ccsdt'] = float(run3.e_tot)
        if j.method in ('ccsdt[q]', 'ccsdt(q)'):
            e_q = cc.ccsdt_q()
            e['ccsdt[q]'], e['ccsdt(q)'] = float(run3.e_tot + run3.e_q5), float(run3.e_tot + e_q)
        if j.method == 'ccsdt(q)_l':
            e_q = cc.ccsdt_q_lambda()
            if not cc.converged_lambda:
                raise RuntimeError('the Lambda equations of CCSDT did not converge in %d cycles' % j.max_cycle)
            e['ccsdt[q]_l'], e['ccsdt(q)_l'] = float(run3.e_tot + cc.e_ql5), float(run3.e_tot + e_q)
        if full:
            e['ccsdtq'] = float(cc.e_tot)
        if j.method == 'ccsdtq(p)':
            cc.ccsdtq_p()
            e['ccsdtq(p)'] = float(cc.e_tot_p)
    for name, value in e.items():
        say('E(%s)%s = %.12f%s' % (name.upper(), ' ' * (10 - len(name)), value,
                                    '' if name == 'scf' else '   correlation %.12f' % (value - x.e_scf)))
    if jobarc:
        JobArc(directory).put('TOTENERG', np.array([e[j.method]], dtype=_F8))
        say('TOTENERG of JOBARC: %.12f' % e[j.method])
    return e


def main(argv=None):
    '''python -m zccsdt.cfour [options] RUNDIR'''
    import argparse
    import shutil
    import sys
    ap = argparse.ArgumentParser(
        prog='python -m zccsdt.cfour',
        description="High-order coupled cluster on the spinor integrals of CFOUR's two-component programs.  "
                    'Without --chain: RUNDIR is a directory in which xjoda, xsox2catm, xsox2chf and xx2cvtran '
                    '(ABCDTYPE=STANDARD) have run; its ZMAT says what to do (CALC, CC_CONV, CC_MAXCYC, '
                    'MEMORY_SIZE, and the section %%zccsdt).  With --chain: RUNDIR has ZMAT and GENBAS, and '
                    "CFOUR's programs are run first.")
    ap.add_argument('rundir', nargs='?', default='.')
    ap.add_argument('--chain', action='store_true', help="run CFOUR's programs first (xjoda ... xx2cvtran)")
    ap.add_argument('--programs', help='the programs of the chain, in order (default: from the ZMAT)')
    ap.add_argument('--bin', help="directory of CFOUR's executables (default: $CFOUR_BIN, else %s)" % BIN)
    ap.add_argument('--scratch', help='with --chain: run in a new directory under this one (node-local disk); '
                                      'the outputs come back to RUNDIR, the directory is removed at the end')
    ap.add_argument('--keep', action='store_true', help='with --scratch: do not remove the scratch directory')
    ap.add_argument('--before', action='append', default=[], metavar='PROGRAM=COMMAND',
                    help='a shell command run in the directory before a program of the chain')
    ap.add_argument('--no-cc', action='store_true', help="only CFOUR's programs")
    ap.add_argument('--method', help='instead of the method of the ZMAT: ' + ', '.join(METHODS))
    ap.add_argument('--max-memory', type=float, help='MB, instead of the memory of the ZMAT')
    ap.add_argument('--mo-integrals', action='store_true',
                    help="CCSD on the integral files of xx2cvtran (needs moinpppp) instead of the AO-driven code")
    ap.add_argument('--density', action='store_true', default=None,
                    help='the relaxed density as the file denfinal (default: if the ZMAT has PROPS)')
    ap.add_argument('--properties', action='store_true', help='xsox2cprop afterwards, on the density written here')
    ap.add_argument('--jobarc', action='store_true', help='write the final energy into JOBARC (TOTENERG)')
    ap.add_argument('--verbose', type=int, default=4)
    a = ap.parse_args(argv)
    start = work = os.path.abspath(a.rundir)
    try:
        if a.chain:
            if a.scratch:
                import tempfile
                os.makedirs(a.scratch, exist_ok=True)
                work = tempfile.mkdtemp(prefix='zccsdt_cfour_', dir=a.scratch)
                for name in os.listdir(start):          # the input: every plain file that is not an output
                    if os.path.isfile(os.path.join(start, name)) and not name.endswith(('.out', '.time')):
                        shutil.copy(os.path.join(start, name), work)
                print('CFOUR runs in %s' % work, flush=True)
            for name in ('ZMAT', 'GENBAS'):
                if not os.path.isfile(os.path.join(work, name)):
                    raise ProgramFailed('%s: no file %s' % (start, name))
            before = dict(item.split('=', 1) for item in a.before)
            try:
                run_programs(work, a.programs.split() if a.programs else None, a.bin, before,
                             log=lambda s: print(s, flush=True))
            finally:
                if work != start:
                    for name in os.listdir(work):
                        if name.endswith(('.out', '.time')):
                            shutil.copy(os.path.join(work, name), start)
        if not a.no_cc:
            run(work, a.method, a.max_memory, a.verbose, jobarc=a.jobarc, mo_integrals=a.mo_integrals,
                density=True if a.properties else a.density)
            if a.properties:
                run_programs(work, ['xsox2cprop'], a.bin, log=lambda s: print(s, flush=True))
                if work != start:
                    shutil.copy(os.path.join(work, 'xsox2cprop.out'), start)
    except (ProgramFailed, FileNotFoundError, ValueError, RuntimeError) as err:
        print('zccsdt.cfour: %s' % err, file=sys.stderr)
        return 1
    finally:
        if work != start and not a.keep:
            shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
