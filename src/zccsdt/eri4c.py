'''No-pair four-component two-electron integrals (Dirac-Coulomb, Gaunt,
Breit) of two-component X2C spinors, in the blocks of their labels, without
an n^4 array anywhere.

Origin: socutils.tools.fcidump_rel (the user's code), x2c_4c_spinors and
x2c_active_eri_4c.  The spinors C of an X2C calculation are taken back to
the four-component picture - large component R C', small X R C' / 2c, with
C' = contr C in the uncontracted basis of the X2C step - and

    (pq|rs) = (LL|LL) + (LL|SS) + (SS|LL) + (SS|SS)            Dirac-Coulomb
            + f [(LS|SL) + (LS|LS) + (SL|SL) + (SL|LS)]         Gaunt (f = -1, int2e_*)
                                                               or Breit (f = +1, int2e_breit_*)

with libcint's spinor integrals int2e_spinor, int2e_spsp2_spinor,
int2e_spsp1_spinor, int2e_spsp1spsp2_spinor and int2e_[breit_]ssp1sps2,
ssp1ssp2, sps1sps2, sps1ssp2_spinor, exactly the classes and signs of
x2c_active_eri_4c (with_full_coulomb, with_gaunt, with_breit).  Five of the
eight classes are computed here; the other three follow from them:

    (SS|LL)_pqrs = (LL|SS)_rspq,
    (SL|SL)_pqrs = conj (LS|LS)_qpsr,    (SL|LS)_pqrs = conj (LS|SL)_qpsr

(the AO integrals obey these to the last digits, 6e-13 of the largest
Breit integral of O2 / DYVDZ), so that with Y = (LL|LL)/2 + (SS|SS)/2 +
(LL|SS) + G/2, G the Gaunt or Breit part, (pq|rs) = Y_pqrs + Y_rspq.
In the classes that are symmetric under (ij|kl) -> (kl|ij) themselves
((LL|LL), (SS|SS), (LS|LS)) the half that this mirror gives is not made
either: Y takes their AO shells I <= K of the first and the third index
(I = K with weight 1/2).  By the times of libcint, 40 % of the AO integrals
of the eight classes are computed.  ``direct=True`` computes all eight classes instead (a check).

Labels.  On a linear molecule along z the spinor AO functions carry m_j, and
spinors adapted to the axis carry 2 Omega = 2 m_j ("labels"): the
coefficients of a spinor vanish on the AO functions of other m_j (to 1e-12;
they are set to zero here and the largest one is reported), and

    (pq|rs) = 0   unless   -l_p + l_q - l_r + l_s = 0

(and the parities, if given, add to an even number).  Only the allowed
blocks are made and stored: 5.6 % of n^4 for O2 / cc-pVQZ-X2C-TP.  The
imaginary parts vanish there as well (real spinors on the axis): the
integrals are made in real arithmetic, a complex chunk of AO integrals
raises; ``real=False`` keeps everything complex.

The transformation of one class (AO functions i, j, k, l; MOs p, q, r, s):

  pass 1, bra: chunks of AO integrals (ij|kl) for all i, j, the k of one
     shell and the l of a range of shells (libcint, OpenMP over the shell
     pairs ij), then two dense GEMMs
         W1[l,k,j,p] = sum_i conj c1[i,p] (ij|kl)     (on the real view of the
                                                       complex chunk: no copy)
         W2[l,k,p,q] = sum_j W1[l,k,j,p] c2[j,q]
     and W2 scattered into H_(l1,l2,rho)[k, l, p, q] for the k of label rho,
     the l of label sigma = l1 - l2 + rho, p, q of labels l1, l2: the
     half-transformed integrals that the labels allow (12 % of n^2 N^2);
  pass 2, ket: for every (l1, l2, rho)
         (pq|rs) += sum_k conj c3[k,r] sum_l c4[l,s] H[k,l,p,q]
     into the block (l1, l2, rho, sigma) - two GEMMs.

The integrals are accumulated in label blocks (l1, l2, l3, l4) stored as
[r, s, p, q] (LabelBuffer) and written by blocks of the labels and the
parities, as [p, q, r, s], into a block file (zccsdt.hblocks).

Memory (plan()): the label buffer, H of one class and the chunk of AO
integrals with its two products; H goes to a scratch file when it does not
fit, and when the label buffer does not fit either, the blocks are made in
slabs (sets of blocks closed under the two mirror relations above), one
pass over the AO integrals per slab.  Nothing else is n^4.

Work can be split over processes (independent jobs on several nodes):
``kshells`` restricts the k shells of the AO integrals; the label buffers of
the parts add up (LabelBuffer.save / load) before they are written.
'''

import time
from itertools import product

import numpy as np
from pyscf import lib
from pyscf.gto import moleintor

from zccsdt import memory
from zccsdt.timing import timer

COULOMB = ('int2e_spinor', 'int2e_spsp2_spinor', 'int2e_spsp1_spinor', 'int2e_spsp1spsp2_spinor')
GAUNT = ('ssp1sps2', 'ssp1ssp2', 'sps1sps2', 'sps1ssp2')
CHUNK_BYTES = 6 << 30       # a chunk of AO integrals with its products, at most
REAL_TOL = 1e-12            # largest imaginary part of an AO integral taken as zero (relative)


def _gb(x):
    return '%.2f GB' % (x / 1e9)


def x2c_4c_spinors(mf, mo_coeff):
    '''(xmol, large, small) of the two-component spinors ``mo_coeff`` of an
    X2C calculation ``mf``: R C' and X R C' in the spinor basis of the
    uncontracted molecule xmol of the X2C step, C' = contr C (the small
    component without the factor 1/2c).  Copied from
    socutils.tools.fcidump_rel.x2c_4c_spinors.'''
    from pyscf.x2c.x2c import _decontract_spinor
    x2cobj = mf.with_x2c
    x, r = x2cobj.get_xr()
    xmol, contr = _decontract_spinor(mf.mol, x2cobj.xuncontract)
    n2c = xmol.nao_2c()
    if x.shape != (n2c, n2c) or contr.shape != (n2c, mf.mol.nao_2c()):
        raise RuntimeError('X is %s, the uncontracted basis has %d spinors and the contraction is %s'
                           % (x.shape, n2c, contr.shape))
    s = contr.conj().T @ xmol.intor('int1e_ovlp_spinor') @ contr
    err = abs(s - mf.mol.intor('int1e_ovlp_spinor')).max()
    if err > 1e-8:
        raise RuntimeError('the contraction does not reproduce the overlap of the contracted basis (%.1e)' % err)
    mo_l = r @ (contr @ mo_coeff)
    return xmol, mo_l, x @ mo_l


def ao_m2(mol):
    '''2 m_j of the functions of PySCF's spinor basis of ``mol``.'''
    out = []
    for b in range(mol.nbas):
        l, kappa = mol.bas_angular(b), mol.bas_kappa(b)
        js = (2 * l - 1, 2 * l + 1) if kappa == 0 else ((2 * l - 1,) if kappa > 0 else (2 * l + 1,))
        for _ in range(mol.bas_nctr(b)):
            for j2 in js:
                if j2 > 0:
                    out += list(range(-j2, j2 + 1, 2))
    out = np.array(out, dtype=np.int64)
    assert len(out) == mol.nao_2c()
    return out


class Layout:
    '''The spinors in blocks of their labels.

    ``order``: the MOs "by blocks" - by label, then parity, then their index -
    as indices of the given MOs; ``mo[l]``: the slice of label l in that
    order, ``gslice[(l, par)]`` the slice of a group (label, parity) - groups
    sorted by label, then parity; ``ao[l]``: the AO functions of 2 m_j = l;
    ``quads``: the label blocks (l1, l2, l3, l4), l4 = l1 - l2 + l3.'''

    def __init__(self, labels, parity, m2):
        lab = np.asarray(labels, dtype=np.int64)
        n = len(lab)
        par = np.zeros(n, dtype=np.int64) if parity is None else np.asarray(parity, dtype=np.int64)
        self.n, self.lab, self.par, self.with_parity = n, lab, par, parity is not None
        self.order = np.lexsort((np.arange(n), par, lab))
        self.labels = sorted(set(lab.tolist()))
        lo = lab[self.order]
        po = par[self.order]
        self.mo = {}
        for x in self.labels:
            idx = np.flatnonzero(lo == x)
            self.mo[x] = slice(int(idx[0]), int(idx[-1]) + 1)
        self.groups = sorted(set(zip(lab.tolist(), par.tolist())))
        self.gslice = {}
        for g in self.groups:
            idx = np.flatnonzero((lo == g[0]) & (po == g[1]))
            self.gslice[g] = slice(int(idx[0]), int(idx[-1]) + 1)
            assert len(idx) == idx[-1] - idx[0] + 1
        m2 = np.asarray(m2, dtype=np.int64)
        self.m2 = m2
        self.ao = {x: np.flatnonzero(m2 == x) for x in self.labels}
        labset = set(self.labels)
        self.quads = [(a, b, c, a - b + c) for a, b, c in product(self.labels, repeat=3) if a - b + c in labset]

    def nmo(self, x):
        s = self.mo[x]
        return s.stop - s.start

    def gquads(self):
        '''Group blocks (g1, g2, g3, g4) that the labels and parities allow.'''
        gs = set(self.groups)
        out = []
        for g1, g2, g3 in product(self.groups, repeat=3):
            g4 = (g1[0] - g2[0] + g3[0], (g1[1] + g2[1] + g3[1]) % 2)
            if g4 in gs:
                out.append((g1, g2, g3, g4))
        return out

    def coefficients(self, c):
        '''(dense coefficients in the order by blocks with the elements off
        the labels set to zero, the largest of those).'''
        cb = np.asarray(c)[:, self.order]
        out = np.zeros_like(cb)
        for x in self.labels:
            a, s = self.ao[x], self.mo[x]
            out[a, s] = cb[a, s]
        return out, float(abs(cb - out).max(initial=0.0))


def orbits(quads):
    '''The label blocks in orbits of the two mirrors (l1 l2 l3 l4) ->
    (l3 l4 l1 l2) and (l2 l1 l4 l3).'''
    seen, out = set(), []
    for q in quads:
        if q in seen:
            continue
        a, b, c, d = q
        orb = sorted({q, (c, d, a, b), (b, a, d, c), (d, c, b, a)})
        seen.update(orb)
        out.append(orb)
    return out


class LabelBuffer:
    '''Label blocks (l1, l2, l3, l4) of (pq|rs), each a dense [r, s, p, q]
    array in one flat buffer (np.zeros: pages are taken when written).'''

    def __init__(self, layout, quads, dtype=np.float64, buf=None):
        self.layout, self.dtype = layout, np.dtype(dtype)
        self.quads = list(quads)
        self.offset, self.shape = {}, {}
        off = 0
        for q in self.quads:
            sh = (layout.nmo(q[2]), layout.nmo(q[3]), layout.nmo(q[0]), layout.nmo(q[1]))
            self.offset[q], self.shape[q] = off, sh
            off += int(np.prod(sh))
        self.size = off
        self.buf = np.zeros(off, dtype=self.dtype) if buf is None else buf
        assert self.buf.size == off

    @staticmethod
    def nbytes(layout, quads, itemsize=8):
        return itemsize * sum(layout.nmo(a) * layout.nmo(b) * layout.nmo(c) * layout.nmo(d) for a, b, c, d in quads)

    def __contains__(self, q):
        return q in self.offset

    def block(self, q):
        o = self.offset[q]
        return self.buf[o:o + int(np.prod(self.shape[q]))].reshape(self.shape[q])

    def mirror_add(self, kind):
        '''B += mirror(B): 'klij' (pq|rs) -> (rs|pq), 'jilk' (pq|rs) -> conj (qp|sr).'''
        done = set()
        for q in self.quads:
            if q in done:
                continue
            a, b, c, d = q
            p = (c, d, a, b) if kind == 'klij' else (b, a, d, c)
            perm = (2, 3, 0, 1) if kind == 'klij' else (1, 0, 3, 2)
            conj = kind == 'jilk' and self.dtype.kind == 'c'
            x = self.block(q)
            if p == q:
                y = x.transpose(perm).copy()
                x += y.conj() if conj else y
            else:
                y = self.block(p)
                xt = x.transpose(perm).copy()
                x += y.transpose(perm).conj() if conj else y.transpose(perm)
                y += xt.conj() if conj else xt
                done.add(p)
            done.add(q)

    def save(self, path):
        np.save(path, self.buf)

    def add_file(self, path):
        x = np.load(path, mmap_mode='r')
        assert x.shape == self.buf.shape
        step = 1 << 27
        for i in range(0, self.size, step):
            self.buf[i:i + step] += x[i:i + step]


class _Half:
    '''H_(l1,l2,rho)[k, l, p, q] of one class: the half-transformed
    integrals the labels allow, for the label triples of the blocks made.'''

    def __init__(self, layout, triples, dtype, scratch_file=None):
        self.offset, self.shape = {}, {}
        off = 0
        for t in triples:
            l1, l2, rho = t
            sig = l1 - l2 + rho
            sh = (len(layout.ao[rho]), len(layout.ao[sig]), layout.nmo(l1), layout.nmo(l2))
            if 0 in sh:
                continue
            self.offset[t], self.shape[t] = off, sh
            off += int(np.prod(sh))
        self.size = off
        self.scr = None
        if scratch_file is None:
            self.buf = np.zeros(off, dtype=dtype)
        else:
            from zccsdt.scratch import Scratch
            self.scr = Scratch(prefix='zccsdt_eri4c_', what='the half-transformed four-component integrals')
            self.buf = self.scr.create('h', off, dtype)

    @staticmethod
    def nbytes(layout, triples, itemsize=8):
        out = 0
        for l1, l2, rho in triples:
            out += len(layout.ao[rho]) * len(layout.ao[l1 - l2 + rho]) * layout.nmo(l1) * layout.nmo(l2)
        return itemsize * out

    def block(self, t):
        o = self.offset[t]
        return self.buf[o:o + int(np.prod(self.shape[t]))].reshape(self.shape[t])

    def close(self):
        self.buf = None
        if self.scr is not None:
            self.scr.close()


def classes(gaunt=True, breit=True, direct=False):
    '''[(intor, coefficient pattern, factor, half)] of the operator; with
    ``direct=False`` the five classes and factors of Y (module docstring).
    ``half``: the class is symmetric under (ij|kl) -> (kl|ij) with the same
    coefficients in the two pairs (LLLL, SSSS, LSLS), so that Y needs only
    the shells I <= K of the first and the third index (I = K with weight
    1/2): the mirror (pq|rs) -> (rs|pq) at the end gives the rest.'''
    p = 'int2e_breit_' if breit else 'int2e_'
    f = 1.0 if breit else -1.0
    if direct:
        out = [('int2e_spinor', 'LLLL', 1.0, False), ('int2e_spsp2_spinor', 'LLSS', 1.0, False),
               ('int2e_spsp1_spinor', 'SSLL', 1.0, False), ('int2e_spsp1spsp2_spinor', 'SSSS', 1.0, False)]
        if gaunt:
            out += [(p + 'ssp1sps2_spinor', 'LSSL', f, False), (p + 'ssp1ssp2_spinor', 'LSLS', f, False),
                    (p + 'sps1sps2_spinor', 'SLSL', f, False), (p + 'sps1ssp2_spinor', 'SLLS', f, False)]
        return out
    out = []
    if gaunt:
        out += [(p + 'ssp1ssp2_spinor', 'LSLS', f, True), (p + 'ssp1sps2_spinor', 'LSSL', 0.5 * f, False)]
    return out + [('int2e_spinor', 'LLLL', 1.0, True), ('int2e_spsp1spsp2_spinor', 'SSSS', 1.0, True),
                  ('int2e_spsp2_spinor', 'LLSS', 1.0, False)]


class Plan:
    '''Sizes: ``slabs`` (lists of label blocks), the l shells per chunk,
    whether H is in a scratch file, and the bytes of each part.'''

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def text(self):
        return ('label buffer %s in %d slab(s); H of one class %s (%s); chunks of AO integrals: one k shell and '
                'up to %d l functions, %s; planned peak %s of %s'
                % (_gb(self.buf_bytes), len(self.slabs), _gb(self.h_bytes),
                   'scratch file' if self.h_disk else 'memory', self.nl_max, _gb(self.chunk_bytes), _gb(self.peak),
                   'no limit' if self.budget is None else _gb(self.budget)))


def plan(layout, xmol, itemsize=8, budget=None, chunk_bytes=CHUNK_BYTES):
    '''Memory plan for ``budget`` bytes (None: no limit known).'''
    N = xmol.nao_2c()
    n = layout.n
    loc = xmol.ao_loc_2c()
    nk_max = int((loc[1:] - loc[:-1]).max())
    per_pair = 16 * N * N + itemsize * (N * n + 2 * n * n)      # chunk, W1, W2 and a gather of W2
    # pass 2: the largest products of one label triple
    p2 = 0
    for l1, l2, rho in {(a, b, c) for a, b, c, d in layout.quads}:
        sig = l1 - l2 + rho
        p2 = max(p2, itemsize * layout.nmo(l1) * layout.nmo(l2) * (len(layout.ao[rho]) * layout.nmo(sig)
                                                                   + layout.nmo(rho) * layout.nmo(sig)))
    small = itemsize * 8 * N * n + p2 + (64 << 20)
    allq = layout.quads
    full_buf = LabelBuffer.nbytes(layout, allq, itemsize)
    full_h = _Half.nbytes(layout, {q[:3] for q in allq}, itemsize)
    lim = budget if budget is not None else full_buf + full_h + chunk_bytes + small

    def chunk_for(rest):
        c = min(chunk_bytes, rest)
        nl = int(c // (per_pair * nk_max))
        return nl

    min_chunk = per_pair * nk_max * int((loc[1:] - loc[:-1]).max())
    # 1. everything in memory
    rest = lim - full_buf - full_h - small
    if rest >= min_chunk:
        slabs, h_disk, buf_b, h_b = [allq], False, full_buf, full_h
    else:
        rest = lim - full_buf - small
        if rest >= min_chunk:
            slabs, h_disk, buf_b, h_b = [allq], True, full_buf, full_h
        else:
            # 2. slabs of orbits, H in a scratch file
            orb = orbits(allq)
            room = lim - small - min_chunk
            if room <= 0:
                raise MemoryError('the four-component integrals need %s for one chunk of AO integrals, %s are left'
                                  % (_gb(min_chunk + small), _gb(lim)))
            slabs, cur, cb = [], [], 0
            for o in sorted(orb, key=lambda o: -LabelBuffer.nbytes(layout, o, itemsize)):
                b = LabelBuffer.nbytes(layout, o, itemsize)
                if b > room:
                    raise MemoryError('one orbit of label blocks takes %s, %s are left for them' % (_gb(b), _gb(room)))
                if cb + b > room and cur:
                    slabs.append(cur)
                    cur, cb = [], 0
                cur += o
                cb += b
            if cur:
                slabs.append(cur)
            h_disk = True
            buf_b = max(LabelBuffer.nbytes(layout, s, itemsize) for s in slabs)
            h_b = max(_Half.nbytes(layout, {q[:3] for q in s}, itemsize) for s in slabs)
            rest = lim - buf_b - small
    nl = max(int(nk_max), min(N, chunk_for(rest)))
    chunk_b = per_pair * nk_max * nl
    peak = buf_b + (0 if h_disk else h_b) + chunk_b + small
    return Plan(slabs=slabs, h_disk=h_disk, nl_max=nl, buf_bytes=buf_b, h_bytes=h_b, chunk_bytes=chunk_b,
                small=small, peak=peak, budget=budget, per_pair=per_pair, min_chunk=min_chunk)


def _lranges(loc, nl_max):
    '''Contiguous ranges of shells with at most nl_max functions (one shell at least).'''
    out, start = [], 0
    nbas = len(loc) - 1
    while start < nbas:
        stop = start + 1
        while stop < nbas and loc[stop + 1] - loc[start] <= nl_max:
            stop += 1
        out.append((start, stop))
        start = stop
    return out


def _by_label(layout, glob):
    '''{label: (local positions, positions in layout.ao[label])} of the AO functions ``glob``.'''
    m = layout.m2[glob]
    out = {}
    for x in layout.labels:
        loc = np.flatnonzero(m == x)
        if len(loc):
            out[x] = (loc, np.searchsorted(layout.ao[x], glob[loc]))
    return out


def transform_class(xmol, intor, coef, layout, out, factor, nl_max, kshells=None, real=True, h_disk=False,
                    verbose=False, stats=None, half_bra=False):
    '''out += factor * (pq|rs) of one class: ``coef`` the four dense
    coefficient matrices (N, n) in the order by blocks, labels respected.
    ``half_bra``: only the shells I <= K of the first index for the k shell
    K, those of I = K with weight 1/2 (classes()).'''
    N = xmol.nao_2c()
    n = layout.n
    loc = np.asarray(xmol.ao_loc_2c(), dtype=np.int32)
    nbas = xmol.nbas
    dtype = np.float64 if real else np.complex128
    c1, c2, c3, c4 = coef
    triples = sorted({q[:3] for q in out.quads})
    half = _Half(layout, triples, dtype, scratch_file=True if h_disk else None)
    pairs = sorted({t[:2] for t in half.offset})
    rhos = {}
    for t in half.offset:
        rhos.setdefault(t[:2], []).append(t[2])
    if real:
        c1i = np.zeros((2 * N, n))
        c1i[0::2] = c1.real
    else:
        c1h = np.ascontiguousarray(c1.conj())
    cintopt = moleintor.make_cintopt(xmol._atm, xmol._bas, xmol._env, intor)
    nk_max = int((loc[1:] - loc[:-1]).max())
    lr = _lranges(loc, nl_max)
    nl_top = max(loc[b] - loc[a] for a, b in lr)
    xbuf = np.empty(N * N * nk_max * nl_top, dtype=np.complex128)
    w1buf = np.empty(nk_max * nl_top * N * n, dtype=dtype)
    w2buf = np.empty(nk_max * nl_top * n * n, dtype=dtype)
    ks = range(nbas) if kshells is None else kshells
    t_int = t_q = t_q1 = t_sc = 0.0
    imag = 0.0
    for K in ks:
        kglob = np.arange(loc[K], loc[K + 1])
        kby = _by_label(layout, kglob)
        nk = len(kglob)
        if half_bra:
            ish1, ni = K + 1, int(loc[K + 1])
            # (the shell I = K with weight 1/2)
            if real:
                ck = c1i[:2 * ni].copy()
                ck[2 * loc[K]:] *= 0.5
            else:
                ck = c1h[:ni].copy()
                ck[loc[K]:] *= 0.5
        else:
            ish1, ni = nbas, N
            ck = c1i if real else c1h
        for L0, L1 in lr:
            lglob = np.arange(loc[L0], loc[L1])
            nl = len(lglob)
            t0 = time.perf_counter()
            x = xbuf[:ni * N * nk * nl]
            moleintor.getints4c(intor, xmol._atm, xmol._bas, xmol._env, (0, ish1, 0, nbas, K, K + 1, L0, L1), 1,
                                's1', loc, cintopt, out=x)
            # (filled in Fortran order (i, j, k, l): C order [l, k, j, i])
            t1 = time.perf_counter()
            t_int += t1 - t0
            w1 = w1buf[:nl * nk * N * n].reshape(nl * nk * N, n)
            if real:
                # the imaginary parts of the AO integrals of the first and the last (k, l) of
                # the chunk, all (i, j): every shell pair ij is seen in every chunk (a pass over
                # the whole chunk, single-threaded, took as long as its GEMMs)
                im = big = 0.0
                for part in (x[:ni * N], x[-ni * N:]):
                    im = max(im, float(abs(part.imag).max()))
                    big = max(big, float(abs(part.real).max()))
                imag = max(imag, im)
                if im > REAL_TOL * max(1.0, big):
                    raise ValueError('%s: imaginary AO integrals (%.1e): use real=False' % (intor, im))
                np.dot(x.view(np.float64).reshape(nl * nk * N, 2 * ni), ck, out=w1)
            else:
                np.dot(x.reshape(nl * nk * N, ni), ck, out=w1)
            t15 = time.perf_counter()
            t_q1 += t15 - t1
            w2 = w2buf[:nl * nk * n * n].reshape(nl * nk, n, n)
            np.matmul(w1.reshape(nl * nk, N, n).transpose(0, 2, 1), c2, out=w2)
            w2 = w2.reshape(nl, nk, n, n)
            t2 = time.perf_counter()
            t_q += t2 - t1
            lby = _by_label(layout, lglob)
            for l1, l2 in pairs:
                s1, s2 = layout.mo[l1], layout.mo[l2]
                for rho in rhos[(l1, l2)]:
                    if rho not in kby:
                        continue
                    sig = l1 - l2 + rho
                    if sig not in lby:
                        continue
                    kl, kp = kby[rho]
                    ll, lp = lby[sig]
                    sub = w2[ll[:, None], kl[None, :], s1, s2]          # (l, k, p, q)
                    half.block((l1, l2, rho))[kp[:, None], lp[None, :]] = sub.transpose(1, 0, 2, 3)
            t_sc += time.perf_counter() - t2
    del xbuf, w1buf, w2buf
    t3 = time.perf_counter()
    # pass 2: the ket
    for (l1, l2, rho), sh in half.shape.items():
        sig = l1 - l2 + rho
        q = (l1, l2, rho, sig)
        if q not in out:
            continue
        hb = half.block((l1, l2, rho)).reshape(sh[0], sh[1], sh[2] * sh[3])
        a3, a4 = layout.ao[rho], layout.ao[sig]
        c3b = c3[a3, layout.mo[rho]]
        c4b = c4[a4, layout.mo[sig]]
        y = np.matmul(c4b.T, hb)                                     # (k, s, pq)
        z = np.dot(c3b.conj().T, y.reshape(sh[0], -1))               # (r, s pq)
        blk = out.block(q)
        blk += factor * z.reshape(blk.shape)
    t_p2 = time.perf_counter() - t3
    hb_bytes = half.size * np.dtype(dtype).itemsize
    half.close()
    if stats is not None:
        stats.update(integrals=t_int, bra=t_q, bra1=t_q1, scatter=t_sc, ket=t_p2, imag=imag, half_bytes=hb_bytes)
    return out


def make(mf, mo_coeff, labels, parity=None, gaunt=True, breit=True, direct=False, real=True, max_memory=None,
         kshells=None, verbose=True, budget=None):
    '''The integrals of the spinors ``mo_coeff`` of the X2C calculation
    ``mf`` (their labels 2 Omega and parities): yields (LabelBuffer, layout)
    for each slab of label blocks (a single one when the plan allows).

    ``max_memory``: MB for the whole process (None: 80 % of what is free);
    ``budget``: bytes for the integrals instead (tests).'''
    xmol, ml, ms = x2c_4c_spinors(mf, mo_coeff)
    return make4c(xmol, ml, ms, labels, parity, gaunt, breit, direct, real, max_memory, kshells, verbose, budget)


def make4c(xmol, ml, ms, labels, parity=None, gaunt=True, breit=True, direct=False, real=True, max_memory=None,
           kshells=None, verbose=True, budget=None):
    '''make() for four-component spinors given by their large and small
    components ``ml``, ``ms`` in the spinor basis of ``xmol`` (the small
    one without the factor 1/2c).'''
    for name, c in (('large', ml), ('small', ms)):
        if real and abs(np.asarray(c).imag).max(initial=0.0) > 1e-12:
            raise ValueError('complex %s components (%.1e): use real=False' % (name, abs(np.asarray(c).imag).max()))
    if real:
        ml, ms = np.asarray(ml).real, np.asarray(ms).real
    ms = ms * (0.5 / lib.param.LIGHT_SPEED)
    layout = Layout(labels, parity, ao_m2(xmol))
    cl, offl = layout.coefficients(ml)
    cs, offs = layout.coefficients(ms)
    cof = {'L': cl, 'S': cs}
    itemsize = 8 if real else 16
    pl = plan(layout, xmol, itemsize, memory.budget(max_memory) if budget is None else budget)
    if verbose:
        print('eri4c: %d spinors in %d label groups, %d AO functions (uncontracted); coefficients off the labels '
              'set to zero (largest: large %.1e, small %.1e); %d label blocks; %s'
              % (layout.n, len(layout.labels), xmol.nao_2c(), offl, offs, len(layout.quads), pl.text()), flush=True)
    for islab, slab in enumerate(pl.slabs):
        t0 = time.perf_counter()
        buf = LabelBuffer(layout, slab, np.float64 if real else np.complex128)
        cls = classes(gaunt, breit, direct)
        mirrored = False
        for intor, pat, f, hb in cls:
            if not direct and gaunt and pat == 'LLLL' and not mirrored:
                buf.mirror_add('jilk')
                mirrored = True
            st = {}
            t1 = time.perf_counter()
            with timer('eri4c.' + pat):
                transform_class(xmol, intor, tuple(cof[x] for x in pat), layout, buf, f, pl.nl_max, kshells, real,
                                pl.h_disk, verbose, st, half_bra=hb)
            if verbose:
                print('  %-28s %s  %7.1f s: AO integrals %.1f, bra GEMMs %.1f (first quarter %.1f), scatter %.1f, '
                      'ket %.1f s; H %s; RssAnon %s; max |Im| %.1e'
                      % (intor, pat, time.perf_counter() - t1, st['integrals'], st['bra'], st['bra1'], st['scatter'],
                         st['ket'], _gb(st['half_bytes']), _gb(memory.resident()), st['imag']), flush=True)
        if not direct:
            buf.mirror_add('klij')
        if verbose:
            print('eri4c: slab %d of %d (%d label blocks) in %.1f s' % (islab + 1, len(pl.slabs), len(slab),
                                                                     time.perf_counter() - t0), flush=True)
        yield buf, layout
        del buf
