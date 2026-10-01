'''Spin-blocked tensors for the nonrelativistic (UHF / RHF) code path.

Every index runs over a *space* whose members are grouped by a spin label:
orbital spaces have groups alpha (0) and beta (1); pair spaces (p<q) have
groups with 0, 1, 2 beta indices (alpha-alpha, alpha-beta, beta-beta), in the
enumeration of t3pack.TripleMap with spins.  A tensor stores only the dense
blocks of label combinations that conserve Ms; absent blocks are zero.

    T = SpinTensor.from_dense(arr, (occ, occ, vir, vir), sig=(-1, -1, 1, 1))
    C = einsum('ijab,abcd->ijcd', T, W)

``sig`` gives +1 for upper (creation-like) and -1 for lower indices; a block
is allowed when sum(sig * nbeta) == 0, nbeta being the label value.
'''

import itertools

import numpy as np
from pyscf import lib


class Space:
    '''An index space split into contiguous spin groups.'''

    def __init__(self, name, groups, labels=None):
        self.name = name
        self.groups = [(int(lo), int(hi)) for lo, hi in groups]
        self.labels = list(range(len(groups))) if labels is None else list(labels)
        self.n = self.groups[-1][1] if self.groups else 0

    def size(self, g):
        lo, hi = self.groups[g]
        return hi - lo

    def __repr__(self):
        return 'Space(%s, %s)' % (self.name, self.groups)


def orbital_space(name, spins):
    '''Space of orbitals with 0/1 = alpha/beta labels (alpha first).'''
    spins = np.asarray(spins)
    na = int(np.sum(spins == 0))
    assert np.all(spins[:na] == 0) and np.all(spins[na:] == 1)
    return Space(name, [(0, na), (na, len(spins))])


def pair_space(name, tmap):
    '''Space of the pairs p<q of a spin-grouped TripleMap.'''
    return Space(name, tmap.blocks)


class SpinTensor:
    def __init__(self, spaces, blocks=None, dtype=np.float64):
        self.spaces = tuple(spaces)
        self.blocks = {} if blocks is None else dict(blocks)
        self.dtype = np.dtype(dtype)
        for b in self.blocks.values():
            self.dtype = np.result_type(self.dtype, b.dtype)

    @property
    def ndim(self):
        return len(self.spaces)

    @property
    def shape(self):
        return tuple(s.n for s in self.spaces)

    def block_shape(self, key):
        return tuple(sp.size(g) for sp, g in zip(self.spaces, key))

    # ------------------------------------------------------------ dense
    @classmethod
    def from_dense(cls, arr, spaces, sig=None, keep_zero=False):
        arr = np.asarray(arr)
        blocks = {}
        for key in itertools.product(*[range(len(s.groups)) for s in spaces]):
            if sig is not None:
                nb = sum(sg * sp.labels[g] for sg, sp, g in zip(sig, spaces, key))
                if nb != 0:
                    continue
            sl = tuple(slice(*sp.groups[g]) for sp, g in zip(spaces, key))
            b = arr[sl]
            if b.size == 0 or (sig is None and not keep_zero and not np.any(b)):
                continue
            blocks[key] = np.ascontiguousarray(b)
        return cls(spaces, blocks, arr.dtype)

    def to_dense(self):
        out = np.zeros(self.shape, dtype=self.dtype)
        for key, b in self.blocks.items():
            sl = tuple(slice(*sp.groups[g]) for sp, g in zip(self.spaces, key))
            out[sl] = b
        return out

    # ------------------------------------------------------- elementwise
    def copy(self):
        return SpinTensor(self.spaces, {k: b.copy() for k, b in self.blocks.items()},
                          self.dtype)

    def conj(self):
        return SpinTensor(self.spaces, {k: b.conj() for k, b in self.blocks.items()},
                          self.dtype)

    def transpose(self, *axes):
        if len(axes) == 1 and not isinstance(axes[0], int):
            axes = tuple(axes[0])
        sp = tuple(self.spaces[a] for a in axes)
        bl = {tuple(k[a] for a in axes): b.transpose(axes) for k, b in self.blocks.items()}
        return SpinTensor(sp, bl, self.dtype)

    def _combine(self, other, fa, fb):
        assert self.shape == other.shape
        out = {}
        for k in set(self.blocks) | set(other.blocks):
            a = self.blocks.get(k)
            b = other.blocks.get(k)
            if a is None:
                out[k] = fb(b)
            elif b is None:
                out[k] = fa(a)
            else:
                out[k] = fa(a) + fb(b)
        return SpinTensor(self.spaces, out, np.result_type(self.dtype, other.dtype))

    def __add__(self, other):
        return self._combine(other, lambda a: a, lambda b: b)

    def __sub__(self, other):
        return self._combine(other, lambda a: a, lambda b: -b)

    def __mul__(self, x):
        return SpinTensor(self.spaces, {k: b * x for k, b in self.blocks.items()},
                          np.result_type(self.dtype, x))

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def iadd(self, other, scale=1.0):
        '''In-place self += scale * other (new blocks are created as needed).'''
        for k, b in other.blocks.items():
            if k in self.blocks:
                self.blocks[k] = self.blocks[k] + scale * b
            else:
                self.blocks[k] = scale * b
        self.dtype = np.result_type(self.dtype, other.dtype)
        return self


def einsum(subscripts, *ops):
    '''Block-sparse einsum: sum over all combinations of stored blocks with
    consistent spin labels for repeated letters.'''
    lhs, rhs = subscripts.replace(' ', '').split('->')
    terms = lhs.split(',')
    assert len(terms) == len(ops)
    spaces = {}
    for t, op in zip(terms, ops):
        for c, sp in zip(t, op.spaces):
            if c in spaces:
                assert spaces[c].groups == sp.groups, (c, spaces[c], sp)
            spaces[c] = sp
    out_spaces = tuple(spaces[c] for c in rhs)
    out = {}
    dtype = np.result_type(*[op.dtype for op in ops])

    def rec(i, labels, arrays):
        if i == len(ops):
            key = tuple(labels[c] for c in rhs)
            r = lib.einsum(subscripts, *arrays) if len(arrays) > 1 else \
                np.einsum(subscripts, *arrays)
            if key in out:
                out[key] += r
            else:
                out[key] = np.array(r, dtype=dtype, copy=True)
            return
        t = terms[i]
        for key, b in ops[i].blocks.items():
            ok = True
            new = dict(labels)
            for c, g in zip(t, key):
                if c in new and new[c] != g:
                    ok = False
                    break
                new[c] = g
            if ok:
                rec(i + 1, new, arrays + [b])

    rec(0, {}, [])
    return SpinTensor(out_spaces, out, dtype)


def pack_pairs(T, ax, pspace):
    '''Replace the axes ax, ax+1 (same orbital space, T antisymmetric in
    them) by one pair axis of ``pspace`` (pairs p<q grouped as aa, ab, bb).'''
    osp = T.spaces[ax]
    assert T.spaces[ax + 1] is osp or T.spaces[ax + 1].groups == osp.groups
    spaces = T.spaces[:ax] + (pspace,) + T.spaces[ax + 2:]
    out = {}
    for key, b in T.blocks.items():
        s1, s2 = key[ax], key[ax + 1]
        if s1 > s2:
            continue                      # redundant (minus the transposed block)
        g = osp.labels[s1] + osp.labels[s2]
        if s1 == s2:
            n = b.shape[ax]
            iu, ju = np.triu_indices(n, 1)
            pb = np.moveaxis(np.moveaxis(b, (ax, ax + 1), (0, 1))[iu, ju], 0, ax)
        else:
            sh = b.shape
            pb = b.reshape(sh[:ax] + (sh[ax] * sh[ax + 1],) + sh[ax + 2:])
        nk = key[:ax] + (g,) + key[ax + 2:]
        out[nk] = np.ascontiguousarray(pb)
    return SpinTensor(spaces, out, T.dtype)


def unpack_pairs(T, ax, ospace):
    '''Inverse of pack_pairs: expand pair axis ``ax`` into two antisymmetric
    axes of the orbital space ``ospace``.'''
    spaces = T.spaces[:ax] + (ospace, ospace) + T.spaces[ax + 1:]
    out = {}
    na = ospace.size(0)
    nb = ospace.size(1)
    for key, b in T.blocks.items():
        g = T.spaces[ax].labels[key[ax]]
        bm = np.moveaxis(b, ax, 0)
        rest = bm.shape[1:]
        if g in (0, 2):
            s = 0 if g == 0 else 1
            n = na if g == 0 else nb
            full = np.zeros((n, n) + rest, dtype=b.dtype)
            iu, ju = np.triu_indices(n, 1)
            full[iu, ju] = bm
            full[ju, iu] = -bm
            nk = key[:ax] + (s, s) + key[ax + 1:]
            out[nk] = np.ascontiguousarray(np.moveaxis(full, (0, 1), (ax, ax + 1)))
        else:
            ab = bm.reshape((na, nb) + rest)
            out[key[:ax] + (0, 1) + key[ax + 1:]] = np.ascontiguousarray(
                np.moveaxis(ab, (0, 1), (ax, ax + 1)))
            out[key[:ax] + (1, 0) + key[ax + 1:]] = np.ascontiguousarray(
                np.moveaxis(-ab.swapaxes(0, 1), (0, 1), (ax, ax + 1)))
    return SpinTensor(spaces, out, T.dtype)
