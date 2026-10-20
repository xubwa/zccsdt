'''A block file of o2_x2ccorr.py run-blocks against the .npz of the dense
route: h1, (pq|rs) element by element (the elements the labels allow, and
the largest of the others in the .npz), ecore, e_hf, the energies, labels
and coefficients of the spinors.  The (pq|rs) of the .npz are read slab by
slab of the first index from the file (an uncompressed .npz member is
mapped, nothing n^4 is loaded).

usage: python x2ccorr_compare.py <dense.npz> <blocks.h5>'''
import sys
import zipfile

import numpy as np

from zccsdt import hblocks


def npz_member(path, key):
    '''A member of an uncompressed .npz as a read-only numpy.memmap.'''
    zf = zipfile.ZipFile(path)
    info = zf.getinfo(key + '.npy')
    assert info.compress_type == zipfile.ZIP_STORED, 'compressed member'
    with open(path, 'rb') as f:
        f.seek(info.header_offset)
        h = f.read(30)
        nlen, xlen = int.from_bytes(h[26:28], 'little'), int.from_bytes(h[28:30], 'little')
        f.seek(info.header_offset + 30 + nlen + xlen)
        version = np.lib.format.read_magic(f)
        read = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = read(f)
        off = f.tell()
    return np.memmap(path, dtype=dtype, mode='r', offset=off, shape=shape, order='F' if fortran else 'C')


def match(z, bf):
    '''pn[p] = the spinor of the block file that is spinor p of the .npz: the
    same label and parity, the closest energy (the two spinors of a Kramers
    pair have the same energy and may come in either order).'''
    lab, par = bf['omega2'], bf['parity']
    eps = bf['eps']
    pn = np.empty(len(lab), dtype=np.int64)
    used = set()
    for p, (lo, po, eo) in enumerate(zip(z['omega2'], z['parity'], z['eps'])):
        cand = np.flatnonzero((lab == lo) & (par == po))
        cand = [c for c in cand[np.argsort(abs(eps[cand] - eo))] if c not in used]
        pn[p] = cand[0]
        used.add(cand[0])
    return pn


def main():
    old, new = sys.argv[1], sys.argv[2]
    z = np.load(old)
    bf = hblocks.BlockFile(new)
    n = bf.n
    print('%s: %d spinors; %s: %d spinors, %d blocks, %s' % (old, z['h1'].shape[0], new, n, len(bf.blocks),
                                                          bf.dtype.name))
    for k in ('ecore', 'e_hf'):
        a, b = float(z[k]), float(bf.attr(k) if k != 'ecore' else bf.ecore)
        print('  %-6s %.12f  %.12f  difference %.1e' % (k, a, b, b - a))
    pn = match(z, bf)
    print('  spinors in another order (Kramers pairs swapped): %d of %d' % (int((pn != np.arange(n)).sum()), n))
    print('  eps        max difference %.1e' % abs(z['eps'] - bf['eps'][pn]).max())
    print('  mo_coeff   max difference %.1e' % abs(z['mo_coeff'] - bf['mo_coeff'][:, pn]).max())
    h1 = bf.h1[np.ix_(pn, pn)]
    print('  h1         max difference %.1e (largest |h1| %.1e)' % (abs(z['h1'] - h1).max(), abs(z['h1']).max()))
    eri = npz_member(old, 'eri')
    lab, par = np.asarray(z['omega2']), np.asarray(z['parity'])
    worst, forb, big = 0.0, 0.0, 0.0
    where = None
    for p in range(n):
        x = np.asarray(eri[p])                    # (q, r, s) of the .npz
        y = np.zeros((n, n, n), dtype=bf.dtype)   # ... of the block file, in its own order
        g1, i1 = bf.group_of[pn[p]], bf.local[pn[p]]
        for b in bf.blocks:
            if b[0] != g1:
                continue
            y[np.ix_(bf.members[b[1]], bf.members[b[2]], bf.members[b[3]])] = bf.block(b)[i1]
        y = y[np.ix_(pn, pn, pn)]
        ok = lab[p] - lab[:, None, None] + lab[None, :, None] - lab[None, None, :] == 0
        ok &= (par[p] + par[:, None, None] + par[None, :, None] + par[None, None, :]) % 2 == 0
        diff = abs(x - y)
        e = float(diff[ok].max(initial=0.0))
        if e > worst:
            worst, where = e, (p,) + np.unravel_index(np.argmax(np.where(ok, diff, 0)), diff.shape)
        forb = max(forb, float(abs(x[~ok]).max(initial=0.0)))
        big = max(big, float(abs(x).max()))
    print('  (pq|rs)    max difference on the allowed elements %.2e at %s (largest |(pq|rs)| %.2e); largest element '
          'of the .npz that the labels forbid %.1e' % (worst, where, big, forb))


if __name__ == '__main__':
    main()
