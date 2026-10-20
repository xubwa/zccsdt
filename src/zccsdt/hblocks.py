'''A Hamiltonian of spinors in the blocks of their labels, on disk: the block
file of zccsdt.eri4c, the Hartree-Fock in the space of its spinors, the
rotation to new spinors, block by block.

The spinors (global indices 0 .. n-1) fall into groups g = (2 Omega,
parity) (parity 0 for all if none is given); the members of a group are
listed in their order inside the group.  (pq|rs) is stored for the blocks
(g1 g2 | g3 g4) that the labels and parities allow,

    -l1 + l2 - l3 + l4 = 0,   p1 + p2 + p3 + p4 even,

each a dense [p, q, r, s] array (C order) at its offset in one contiguous
dataset.  Everything else is zero and not stored.

File (HDF5; the dataset 'eri' is contiguous, so that it is read through
numpy.memmap at ds.id.get_offset(), and a reader in another language can do
the same):

    attrs   format = FORMAT, n, ncore (spinors of the reference determinant:
            global 0 .. ncore-1), ecore, dtype ('float64' / 'complex128'),
            and scalars of the producer (e_hf, nelec, nact)
    group_label (G), group_parity (G), group_start (G+1), members (n):
            the members of group g are members[group_start[g]:group_start[g+1]]
    blocks (B, 4): the groups of each block; block_offset (B+1): element offsets
    eri:    the blocks
    h1 (n, n): one-electron operator; further arrays of the producer (omega2,
            parity, eps, mo_coeff, mo_frozen)

Everything here works block by block: the Fock matrices of a density that
is block diagonal in the groups need the blocks (g g | h h) and (g h | h g)
only; a rotation that keeps the groups (the Hartree-Fock spinors) maps every
block onto itself.'''

import time
from itertools import product

import h5py
import numpy as np

FORMAT = 'zccsdt-hblocks-1'


def _open_h5(path, mode):
    try:
        return h5py.File(path, mode, locking=False)
    except TypeError:                       # h5py < 3.5
        return h5py.File(path, mode)


def allowed_blocks(groups):
    '''Group blocks (indices into ``groups`` [(label, parity)]) that the labels and parities allow.'''
    index = {g: i for i, g in enumerate(groups)}
    out = []
    for g1, g2, g3 in product(groups, repeat=3):
        g4 = (g1[0] - g2[0] + g3[0], (g1[1] + g2[1] + g3[1]) % 2)
        if g4 in index:
            out.append((index[g1], index[g2], index[g3], index[g4]))
    return out


class BlockFile:
    '''A block file, open for reading ('r') or, when made by create(), for
    writing its blocks.'''

    def __init__(self, path, mode='r'):
        self.path = path
        self.h5 = _open_h5(path, mode)
        a = self.h5.attrs
        if a.get('format') != FORMAT:
            raise ValueError('%s is not a block file of zccsdt (%s)' % (path, a.get('format')))
        self.n, self.ncore, self.ecore = int(a['n']), int(a['ncore']), float(a['ecore'])
        self.dtype = np.dtype(str(a['dtype']))
        lab, par = self.h5['group_label'][()], self.h5['group_parity'][()]
        self.groups = [(int(x), int(y)) for x, y in zip(lab, par)]
        st, mem = self.h5['group_start'][()], self.h5['members'][()]
        self.members = [mem[st[g]:st[g + 1]] for g in range(len(self.groups))]
        self.size = [len(m) for m in self.members]
        blocks, off = self.h5['blocks'][()], self.h5['block_offset'][()]
        self.blocks = [tuple(int(x) for x in b) for b in blocks]
        self.offset = {b: int(o) for b, o in zip(self.blocks, off[:-1])}
        self.nelem = int(off[-1])
        ds = self.h5['eri']
        o = ds.id.get_offset()
        self.eri = np.memmap(path, dtype=self.dtype, mode='r+' if mode != 'r' else 'r', offset=o,
                             shape=(self.nelem,)) if self.nelem else np.zeros(0, self.dtype)
        self.group_of = np.empty(self.n, dtype=np.int64)
        self.local = np.empty(self.n, dtype=np.int64)
        for g, m in enumerate(self.members):
            self.group_of[m] = g
            self.local[m] = np.arange(len(m))

    @classmethod
    def create(cls, path, groups, members, ncore, ecore, h1, dtype=np.float64, attrs=None, arrays=None):
        '''A new block file with zeros in its blocks, open for writing.'''
        dtype = np.dtype(dtype)
        groups = [(int(a), int(b)) for a, b in groups]
        size = [len(m) for m in members]
        n = int(sum(size))
        blocks = allowed_blocks(groups)
        off = [0]
        for b in blocks:
            off.append(off[-1] + size[b[0]] * size[b[1]] * size[b[2]] * size[b[3]])
        with _open_h5(path, 'w') as f:
            f.attrs.update(format=FORMAT, n=n, ncore=int(ncore), ecore=float(ecore), dtype=dtype.name)
            for k, v in (attrs or {}).items():
                f.attrs[k] = v
            f['group_label'] = np.array([g[0] for g in groups], dtype=np.int64)
            f['group_parity'] = np.array([g[1] for g in groups], dtype=np.int64)
            f['group_start'] = np.concatenate(([0], np.cumsum(size))).astype(np.int64)
            f['members'] = np.concatenate([np.asarray(m, dtype=np.int64) for m in members])
            f['blocks'] = np.array(blocks, dtype=np.int64).reshape(-1, 4)
            f['block_offset'] = np.array(off, dtype=np.int64)
            f['h1'] = np.asarray(h1)
            for k, v in (arrays or {}).items():
                f[k] = v
            dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
            dcpl.set_layout(h5py.h5d.CONTIGUOUS)
            dcpl.set_alloc_time(h5py.h5d.ALLOC_TIME_EARLY)
            dcpl.set_fill_time(h5py.h5d.FILL_TIME_ALLOC)
            h5py.h5d.create(f.id, b'eri', h5py.h5t.py_create(dtype), h5py.h5s.create_simple((max(1, off[-1]),)), dcpl)
        return cls(path, 'r+')

    def close(self):
        if self.h5 is not None:
            if isinstance(self.eri, np.memmap) and self.eri.mode != 'r':
                self.eri.flush()
            self.eri = None
            self.h5.close()
            self.h5 = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __getitem__(self, key):
        return self.h5[key][()]

    def __contains__(self, key):
        return key in self.h5

    def attr(self, key, default=None):
        return self.h5.attrs.get(key, default)

    @property
    def h1(self):
        return self.h5['h1'][()]

    def nbytes(self):
        return self.nelem * self.dtype.itemsize

    def block(self, b):
        '''The block b = (g1, g2, g3, g4) as a [p, q, r, s] array (a view of the file).'''
        o = self.offset[b]
        sh = tuple(self.size[g] for g in b)
        return self.eri[o:o + int(np.prod(sh))].reshape(sh)

    def labels(self):
        '''(2 Omega, parity) of the spinors in their global order.'''
        lab = np.empty(self.n, dtype=np.int64)
        par = np.empty(self.n, dtype=np.int64)
        for g, m in enumerate(self.members):
            lab[m], par[m] = self.groups[g]
        return lab, par

    def jk(self, dens):
        '''{g: J_g - K_g} for a density {h: D_h} that is block diagonal in
        the groups (D_h in the order of the members of h):
        J_pq = sum_rs (pq|rs) D_sr, K_pq = sum_rs (ps|rq) D_sr.'''
        out = {}
        for g in range(len(self.groups)):
            v = np.zeros((self.size[g], self.size[g]), dtype=np.result_type(self.dtype, *dens.values()))
            for h, d in dens.items():
                b = (g, g, h, h)
                if b in self.offset:
                    v += np.einsum('pqrs,sr->pq', self.block(b), d)
                b = (g, h, h, g)
                if b in self.offset:
                    v -= np.einsum('psrq,sr->pq', self.block(b), d)
            out[g] = v
        return out

    def occupied_density(self, occ):
        '''{h: D_h} of the spinors with global index in ``occ`` (one per spinor).'''
        occ = set(int(i) for i in occ)
        out = {}
        for h, m in enumerate(self.members):
            d = np.array([1.0 if int(i) in occ else 0.0 for i in m])
            if d.any():
                out[h] = np.diag(d).astype(self.dtype)
        return out

    def fock(self, h1=None, occ=None):
        '''Dense Fock matrix h1 + sum_i <p i || q i> of the occupied spinors
        ``occ`` (default: the first ncore), zero between groups.'''
        h1 = self.h1 if h1 is None else h1
        occ = range(self.ncore) if occ is None else occ
        v = self.jk(self.occupied_density(occ))
        f = np.array(h1, copy=True)
        for g, m in enumerate(self.members):
            f[np.ix_(m, m)] += v[g]
        return f

    def reference_energy(self, h1=None, occ=None):
        '''ecore + sum_i h_ii + 1/2 sum_ij <ij||ij> of the occupied spinors.'''
        h1 = self.h1 if h1 is None else h1
        occ = np.arange(self.ncore) if occ is None else np.asarray(occ)
        dens = self.occupied_density(occ)
        v = self.jk(dens)
        e = self.ecore + float(np.einsum('ii->', h1[np.ix_(occ, occ)]).real)
        for g, d in dens.items():
            e += 0.5 * float(np.einsum('pq,qp->', v[g], d).real)
        return e

    def to_dense(self, sel=None):
        '''(pq|rs) as a dense array (for small cases and checks), of the
        spinors ``sel`` (global indices, default all) in that order.'''
        sel = np.arange(self.n) if sel is None else np.asarray(sel)
        pos = np.full(self.n, -1)
        pos[sel] = np.arange(len(sel))
        m = len(sel)
        out = np.zeros((m, m, m, m), dtype=self.dtype)
        loc = [pos[mm] for mm in self.members]
        keep = [np.flatnonzero(x >= 0) for x in loc]
        for b in self.blocks:
            if any(len(keep[g]) == 0 for g in b):
                continue
            x = self.block(b)[np.ix_(*(keep[g] for g in b))]
            out[np.ix_(*(loc[g][keep[g]] for g in b))] = x
        return out


def write_label_buffer(bf, buf, layout):
    '''Write the label blocks of an eri4c.LabelBuffer into the block file
    ``bf`` (open for writing), whose groups are those of the layout
    (layout_members).  Returns the largest element that the parities forbid.'''
    gidx = {g: i for i, g in enumerate(bf.groups)}
    forbidden = 0.0
    for q in buf.quads:
        x = buf.block(q)                                   # [r, s, p, q]
        subs = [[(g, layout.gslice[g]) for g in layout.groups if g[0] == lab] for lab in q]
        for (g1, s1), (g2, s2), (g3, s3), (g4, s4) in product(*subs):
            # (the slices are relative to the start of each label)
            r1 = slice(s1.start - layout.mo[q[0]].start, s1.stop - layout.mo[q[0]].start)
            r2 = slice(s2.start - layout.mo[q[1]].start, s2.stop - layout.mo[q[1]].start)
            r3 = slice(s3.start - layout.mo[q[2]].start, s3.stop - layout.mo[q[2]].start)
            r4 = slice(s4.start - layout.mo[q[3]].start, s4.stop - layout.mo[q[3]].start)
            sub = x[r3, r4, r1, r2]
            b = (gidx[g1], gidx[g2], gidx[g3], gidx[g4])
            if b in bf.offset:
                bf.block(b)[...] = sub.transpose(2, 3, 0, 1)
            else:
                forbidden = max(forbidden, float(abs(sub).max(initial=0.0)))
    return forbidden


def layout_members(layout, glob=None):
    '''(groups, members) of a block file for the spinors of an eri4c.Layout:
    ``glob[i]`` the global index of MO i (default i).'''
    glob = np.arange(layout.n) if glob is None else np.asarray(glob)
    return list(layout.groups), [glob[layout.order[layout.gslice[g]]] for g in layout.groups]


def hartree_fock(bf, h1, no, tol=1e-12, max_cycle=200, verbose=True):
    '''Closed-shell Hartree-Fock of the ``no`` electrons in the spinors of
    the block file (the first ``no`` occupied to start), group by group -
    the procedure of bench/dea/o2_x2ccorr.py hartree_fock on the blocks.

    Returns (U {g: unitary of the group, columns ordered as the new
    spinors of the group}, new members {g: new global indices},
    energies (new global order), E, E of the start): the new spinors are
    occupied by energy, then virtual by energy; within a group the columns
    of U follow the new global order; each column has its largest element
    real and positive.'''
    n, G = bf.n, len(bf.groups)
    ecore = bf.ecore
    dens = {}
    nocc = []
    for g, m in enumerate(bf.members):
        k = int((m < no).sum())
        nocc.append(k)
        # (the occupied members first in the start: the first no global indices)
        d = np.diag((m < no).astype(float)).astype(bf.dtype)
        dens[g] = d

    def fock(dens):
        v = bf.jk(dens)
        return {g: h1[np.ix_(bf.members[g], bf.members[g])] + v[g] for g in range(G)}

    def energy(dens, F):
        return float(ecore + 0.5 * sum(np.einsum('pq,qp->', h1[np.ix_(bf.members[g], bf.members[g])] + F[g],
                                                 dens[g]).real for g in range(G)))

    F = fock(dens)
    e_start = energy(dens, F)
    fov = 0.0
    for g, m in enumerate(bf.members):
        o, v = m < no, m >= no
        if o.any() and v.any():
            fov = max(fov, float(abs(F[g][np.ix_(o, v)]).max()))
    if verbose:
        print('HF in the correlated space: E(start) = %.12f; Fock matrix of the start: max |f_ov| %.2e '
              '(nothing between groups is stored)' % (e_start, fov), flush=True)
    for it in range(max_cycle):
        U, eps = {}, {}
        change = 0.0
        new = {}
        for g in range(G):
            w, v = np.linalg.eigh(F[g])
            U[g], eps[g] = v, w
            k = nocc[g]
            dn = v[:, :k] @ v[:, :k].conj().T
            change = max(change, float(abs(dn - dens[g]).max()))
            new[g] = dn
        dens = new
        F = fock(dens)
        e = energy(dens, F)
        if verbose:
            print('  cycle %2d  E = %.13f  |dD| = %.2e' % (it + 1, e, change), flush=True)
        if change < tol:
            break
    else:
        raise RuntimeError('HF not converged')
    # the new order: the columns of U by their old global positions, as in the dense version
    eps_all = np.zeros(n)
    occ = np.zeros(n, bool)
    for g, m in enumerate(bf.members):
        eps_all[m] = eps[g]
        occ[m[:nocc[g]]] = True
    # (the columns of U_g sit at the old global positions of the members of g, in that order;
    # energies equal to 1e-10 - the two spinors of a Kramers pair - in the order of the labels,
    # not of the last digits as in the dense version)
    lab_all, _ = bf.labels()

    def by_energy(idx):
        return idx[np.lexsort((lab_all[idx], np.round(eps_all[idx], 10)))]
    order = np.concatenate((by_energy(np.flatnonzero(occ)), by_energy(np.flatnonzero(~occ))))
    newpos = np.empty(n, dtype=np.int64)
    newpos[order] = np.arange(n)
    Uo, members = {}, {}
    for g, m in enumerate(bf.members):
        p = newpos[m]                   # new positions of the columns of U[g]
        srt = np.argsort(p, kind='stable')
        u = U[g][:, srt]
        big = u[np.argmax(abs(u), axis=0), np.arange(u.shape[1])]
        Uo[g] = u * (abs(big) / big)
        members[g] = p[srt]
    return Uo, members, eps_all[order], e, e_start


def rotate(bf, U, members, path, h1, attrs=None, arrays=None, verbose=False):
    '''A new block file at ``path`` of the spinors psi'_g = psi_g U_g (the
    groups kept; ``members`` {g: their new global indices}, ``h1`` the new
    one-electron operator): every block on its own,
    (pq|rs)' = sum conj U1[a,p] U2[b,q] conj U3[c,r] U4[d,s] (ab|cd).'''
    t0 = time.perf_counter()
    dtype = np.result_type(bf.dtype, *U.values())
    out = BlockFile.create(path, bf.groups, [members[g] for g in range(len(bf.groups))], bf.ncore, bf.ecore, h1,
                           dtype, attrs, arrays)
    assert out.blocks == bf.blocks
    for b in bf.blocks:
        x = np.asarray(bf.block(b))
        u1, u2, u3, u4 = (U[g] for g in b)
        x = np.tensordot(u1.conj(), x, axes=([0], [0]))          # p b c d
        x = np.tensordot(x, u2, axes=([1], [0]))                 # p c d q
        x = np.tensordot(x, u3.conj(), axes=([1], [0]))          # p d q r
        x = np.tensordot(x, u4, axes=([1], [0]))                 # p q r s
        out.block(b)[...] = x
    if verbose:
        print('rotation of %d blocks (%.2f GB) in %.1f s' % (len(bf.blocks), bf.nbytes() / 1e9,
                                                            time.perf_counter() - t0), flush=True)
    return out


def select(bf, keep, path, h1=None, ecore=None, ncore=None, arrays=None, attrs=None):
    '''A new block file of the spinors ``keep`` (global indices, ascending;
    their new global indices are their positions in ``keep``).'''
    keep = np.asarray(keep, dtype=np.int64)
    pos = np.full(bf.n, -1, dtype=np.int64)
    pos[keep] = np.arange(len(keep))
    loc = [np.flatnonzero(pos[mm] >= 0) for mm in bf.members]
    gs = [g for g in range(len(bf.groups)) if len(loc[g])]
    gnew = {g: i for i, g in enumerate(gs)}
    h1 = bf.h1[np.ix_(keep, keep)] if h1 is None else h1
    out = BlockFile.create(path, [bf.groups[g] for g in gs], [pos[bf.members[g][loc[g]]] for g in gs],
                           bf.ncore if ncore is None else ncore, bf.ecore if ecore is None else ecore, h1, bf.dtype,
                           attrs, arrays)
    for b in bf.blocks:
        if all(g in gnew for g in b):
            out.block(tuple(gnew[g] for g in b))[...] = np.asarray(bf.block(b))[np.ix_(*(loc[g] for g in b))]
    return out


def subset(bf, m, path, h1=None, arrays=None, attrs=None):
    '''A new block file of the spinors with global index < m.'''
    return select(bf, np.arange(m), path, h1, arrays=arrays, attrs=attrs)


def fold(bf, nfrozen, path, attrs=None, arrays=None):
    '''A new block file without the first ``nfrozen`` spinors (occupied):
    their mean field added to h1, h1' = h1 + sum_c <p c || q c>, and their
    energy to ecore, ecore' = ecore + sum_c h_cc + 1/2 sum_cd <cd||cd>.'''
    c = np.arange(nfrozen)
    h1 = bf.h1
    dens = bf.occupied_density(c)
    v = bf.jk(dens)
    vf = np.zeros_like(h1, dtype=np.result_type(h1, *v.values()))
    for g, m in enumerate(bf.members):
        vf[np.ix_(m, m)] = v[g]
    ecore = float((bf.ecore + np.trace(h1[:nfrozen, :nfrozen]) + 0.5 * np.trace(vf[:nfrozen, :nfrozen])).real)
    keep = np.arange(nfrozen, bf.n)
    h1n = (h1 + vf)[np.ix_(keep, keep)]
    return select(bf, keep, path, h1n, ecore, bf.ncore - nfrozen, arrays, attrs)
