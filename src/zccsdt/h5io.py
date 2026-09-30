'''HDF5 exchange format for spin-orbital MO integrals (see docs/data_format.md).

Antisymmetric like-space index pairs (oo or vv, first two or last two slots
of a block) are stored packed over p<q; the reader restores the full,
antisymmetric blocks.
'''

import h5py
import numpy as np

from zccsdt.eris import ERIs, BLOCKS

FORMAT_VERSION = 1
_SCALARS = ('nocc', 'e_hf', 'e_nuc')


def _pairs(n):
    return np.triu_indices(n, k=1)


def _pack_block(name, b):
    if name[0] == name[1]:
        p, q = _pairs(b.shape[0])
        b = b[p, q]
    else:
        b = b.reshape((b.shape[0] * b.shape[1],) + b.shape[2:])
    if name[2] == name[3]:
        p, q = _pairs(b.shape[-1])
        b = b[:, p, q]
    else:
        b = b.reshape(b.shape[0], -1)
    return np.ascontiguousarray(b)


def _unpack_pair(x, n, axis):
    '''Expand a packed p<q axis of x into two antisymmetric axes of length n.'''
    x = np.moveaxis(x, axis, -1)
    out = np.zeros(x.shape[:-1] + (n, n), dtype=x.dtype)
    p, q = _pairs(n)
    out[..., p, q] = x
    out[..., q, p] = -x
    return np.moveaxis(out, (-2, -1), (axis, axis + 1))


def _unpack_block(name, b, nocc, nvir):
    n = {'o': nocc, 'v': nvir}
    s = [n[c] for c in name]
    if name[2] == name[3]:
        b = _unpack_pair(b, s[2], 1)
    else:
        b = b.reshape(b.shape[0], s[2], s[3])
    if name[0] == name[1]:
        b = _unpack_pair(b, s[0], 0)
    else:
        b = b.reshape(s[0], s[1], s[2], s[3])
    return b


def save(path, eris):
    with h5py.File(path, 'w') as f:
        f.attrs['format_version'] = FORMAT_VERSION
        f.attrs['dtype'] = np.dtype(eris.dtype).name
        f.attrs['source'] = eris.source
        for k in _SCALARS:
            f.attrs[k] = getattr(eris, k)
        f['fock'] = np.asarray(eris.fock)
        f['mo_energy'] = np.asarray(eris.mo_energy)
        for name in BLOCKS:
            f['eri/' + name] = _pack_block(name, np.asarray(getattr(eris, name)))


def load(path):
    with h5py.File(path, 'r') as f:
        version = int(f.attrs['format_version'])
        if version != FORMAT_VERSION:
            raise ValueError('unsupported format_version %d' % version)
        nocc = int(f.attrs['nocc'])
        fock = f['fock'][()]
        nvir = fock.shape[0] - nocc
        blocks = {name: _unpack_block(name, f['eri/' + name][()], nocc, nvir)
                  for name in BLOCKS}
        return ERIs(nocc=nocc, fock=fock, mo_energy=f['mo_energy'][()],
                    e_hf=float(f.attrs['e_hf']), e_nuc=float(f.attrs['e_nuc']),
                    source=str(f.attrs['source']), **blocks)
