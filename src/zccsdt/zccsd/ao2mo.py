'''Spinor MO integrals from the scalar AO integrals, slab by slab.

For two-component spinors and the bare Coulomb operator (X2C-1e, X2CAMF,
GHF) the integrals are, with the alpha and beta components C^s of the MOs
in the scalar AO basis,

    (pq|rs) = sum_{s,s'} sum C^s*[mu,p] C^s[nu,q] (mu nu|la si) C^s'*[la,r] C^s'[si,s]:

the AO integrals are real, have 8-fold symmetry and carry no spin.  They are
computed once, packed over mu >= nu and la >= si (2 nao^4 bytes), and the
transformation runs over slabs of the first MO index:

- quarter 1 and 2 give Y[p, q, (la si)], contracting the unpacked index of
  the AO integrals with real GEMMs (real and imaginary part of both spin
  components at once);
- the second half unpacks Y to symmetric matrices and transforms them with
  PySCF's kernel ``_ao2mo.r_e2`` (one pair of ZGEMMs per row, OpenMP over the
  rows), once per spin component.

Only the integrals that are asked for are made: (ip|qr) with i occupied for
the blocks of a CCSD with an AO-driven ladder (occupied_blocks), (ae|bf) for
the packed <ab||ef> of a ladder in the MO basis (packed_vvvv).  Nothing goes
through a scratch file; a slab is antisymmetrized into its place as soon as
it is made.  Timers (zccsdt.timing): 'ao2mo.ao', 'ao2mo.half1',
'ao2mo.half2', 'ao2mo.sort', 'ao2mo.ladder'.
'''

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from pyscf import ao2mo, lib
from pyscf.ao2mo import _ao2mo

from zccsdt.timing import timer

SLAB_BYTES = 4 << 30        # default memory of one slab and its buffers
BLOCKS = ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv')


def spin_components(mol, mo_coeff, motype='j-spinor'):
    '''Alpha and beta components (nao, nmo) of spinor MOs in the scalar AO
    basis: ``mo_coeff`` in the j-adapted spinor basis of PySCF ('j-spinor')
    or already as the alpha block stacked on the beta block ('ghf').'''
    nao = mol.nao_nr()
    if motype == 'j-spinor':
        ca, cb = mol.sph2spinor_coeff()
        return np.dot(ca, mo_coeff), np.dot(cb, mo_coeff)
    assert motype == 'ghf' and mo_coeff.shape[0] == 2 * nao
    return np.asarray(mo_coeff[:nao]), np.asarray(mo_coeff[nao:])


def ao_eri_s4(mol):
    '''(mu nu|la si) packed over mu >= nu and la >= si: (npair, npair), real.'''
    with timer('ao2mo.ao'):
        return ao2mo.restore(4, mol.intor('int2e', aosym='s8'), mol.nao_nr())


def _pair_index(nao):
    idx = np.empty((nao, nao), dtype=np.int64)
    a, b = np.tril_indices(nao)
    idx[a, b] = idx[b, a] = np.arange(a.size)
    return idx


def _threads():
    return max(1, lib.num_threads())


def _first_half(eri4, idx, Ca, Cb, i0, i1, q0, q1):
    '''Y[i, q, K] = sum_s sum_{mu nu} conj(C^s[mu,i]) C^s[nu,q] (mu nu|K) for
    i0 <= i < i1 and q0 <= q < q1, K = (la >= si) packed.'''
    nao = Ca.shape[0]
    npair = eri4.shape[1]
    bi = i1 - i0
    # quarter 1: X^s[i, nu, K] = sum_mu conj(C^s[mu,i]) (mu nu|K); the rows
    # (mu nu) of one nu are gathered from the packed integrals
    A = np.vstack([Ca[:, i0:i1].real.T, -Ca[:, i0:i1].imag.T,
                   Cb[:, i0:i1].real.T, -Cb[:, i0:i1].imag.T])
    A = np.ascontiguousarray(A)                                  # (4 bi, nao)
    X = np.empty((2, bi, nao, npair), dtype=np.complex128)
    step = max(1, int(2.5e8 // (nao * npair * 8)))
    for n0 in range(0, nao, step):
        n1 = min(n0 + step, nao)
        R = np.matmul(A, eri4[idx[n0:n1]])                       # (nnu, 4 bi, npair)
        R = R.reshape(n1 - n0, 2, 2, bi, npair)                  # nu, spin, re/im, i, K
        for s in (0, 1):
            X[s, :, n0:n1].real = R[:, s, 0].transpose(1, 0, 2)
            X[s, :, n0:n1].imag = R[:, s, 1].transpose(1, 0, 2)
    # quarter 2: Y[i, q, K] = sum_s sum_nu C^s[nu,q] X^s[i, nu, K]
    Y = np.matmul(np.ascontiguousarray(Ca[:, q0:q1].T), X[0])
    Y += np.matmul(np.ascontiguousarray(Cb[:, q0:q1].T), X[1])
    return Y


def _add(a, b, pool, nthreads):
    '''a += b for large arrays, split over the threads of ``pool``.'''
    n = len(a)
    k = max(1, min(n, nthreads))
    edges = [n * j // k for j in range(k + 1)]

    def part(j):
        np.add(a[edges[j]:edges[j + 1]], b[edges[j]:edges[j + 1]],
               out=a[edges[j]:edges[j + 1]])
    list(pool.map(part, range(k)))


def _second_half(Y, CaF, CbF, q0, q1, out, pool, nthreads):
    '''out[x, r, s] = sum_s' C^s'[:, r]^+ unpack(Y[x]) C^s'[:, s] for the rows
    x of Y (packed symmetric matrices), r and s in q0:q1.'''
    nrow, nao = len(Y), CaF.shape[0]
    nq = q1 - q0
    tao = np.zeros(nao, dtype=np.int32)                  # not used without symmetry
    step = max(1, int(min(2.5e8 // (nao * nao * 16), 5e8 // (nq * nq * 16))))
    buf = np.empty((min(step, nrow), nq * nq), dtype=np.complex128)
    for r0 in range(0, nrow, step):
        r1 = min(r0 + step, nrow)
        M = lib.unpack_tril(Y[r0:r1], lib.SYMMETRIC).reshape(r1 - r0, nao * nao)
        _ao2mo.r_e2(M, CaF, (q0, q1, q0, q1), tao, None, 's1', out=out[r0:r1])
        _ao2mo.r_e2(M, CbF, (q0, q1, q0, q1), tao, None, 's1', out=buf[:r1 - r0])
        _add(out[r0:r1], buf[:r1 - r0], pool, nthreads)


def slabs(eri4, Ca, Cb, first, rest=None, slab_bytes=SLAB_BYTES):
    '''Generator of (i0, i1, g): g[i - i0, q, r, s] = (iq|rs) in chemists'
    notation for the first index i in first = (start, stop) and the other
    three in rest = (start, stop) (default: all MOs).  ``g`` is a buffer that
    the next slab overwrites.'''
    nao, nmo = Ca.shape
    rest = (0, nmo) if rest is None else rest
    q0, q1 = rest
    nq = q1 - q0
    npair = eri4.shape[1]
    idx = _pair_index(nao)
    CaF = np.asarray(Ca, dtype=np.complex128, order='F')
    CbF = np.asarray(Cb, dtype=np.complex128, order='F')
    per_i = 16 * (2 * nao * npair + nq * npair + nq ** 3)
    batch = int(max(1, min(first[1] - first[0], slab_bytes // per_i)))
    g = np.empty((batch * nq, nq * nq), dtype=np.complex128)
    nthreads = _threads()
    with ThreadPoolExecutor(nthreads) as pool:
        for i0 in range(first[0], first[1], batch):
            i1 = min(i0 + batch, first[1])
            with timer('ao2mo.half1'):
                Y = _first_half(eri4, idx, Ca, Cb, i0, i1, q0, q1)
            with timer('ao2mo.half2'):
                n = (i1 - i0) * nq
                _second_half(Y.reshape(n, npair), CaF, CbF, q0, q1, g[:n], pool, nthreads)
            del Y
            yield i0, i1, g[:n].reshape(i1 - i0, nq, nq, nq)


def occupied_blocks(eri4, Ca, Cb, nocc, with_ovvv=True, ovvv=None, slab_bytes=SLAB_BYTES):
    '''The antisymmetrized blocks <pq||rs> with a first occupied index, as a
    dict: oooo, ooov, oovv, ovov, ovvo and (``with_ovvv``) ovvv, written to
    the array ``ovvv`` if one is given (e.g. a mapped scratch array).'''
    nmo = Ca.shape[1]
    nvir = nmo - nocc
    dim = {'o': nocc, 'v': nvir}
    out = {k: np.empty([dim[c] for c in k], dtype=np.complex128) for k in BLOCKS[:5]}
    if with_ovvv:
        out['ovvv'] = np.empty((nocc, nvir, nvir, nvir), np.complex128) if ovvv is None else ovvv
    o, v = slice(0, nocc), slice(nocc, nmo)

    def sort_ovvv(i, gi):
        # <ia||bc> = (ib|ac) - (ic|ab): one a at a time, in the cache
        gv = gi[v, v, v]
        row = out['ovvv'][i]
        for a in range(nvir):
            P = np.ascontiguousarray(gv[:, a, :])
            np.subtract(P, P.T, out=row[a])

    with ThreadPoolExecutor(_threads()) as pool:
        for i0, i1, g in slabs(eri4, Ca, Cb, (0, nocc), None, slab_bytes):
            with timer('ao2mo.sort'):
                s = slice(i0, i1)
                # <pq||rs> = (pr|qs) - (ps|qr), the first index in the slab
                out['oooo'][s] = g[:, o, o, o].transpose(0, 2, 1, 3) - g[:, o, o, o].transpose(0, 2, 3, 1)
                out['ooov'][s] = g[:, o, o, v].transpose(0, 2, 1, 3) - g[:, v, o, o].transpose(0, 2, 3, 1)
                out['oovv'][s] = g[:, v, o, v].transpose(0, 2, 1, 3) - g[:, v, o, v].transpose(0, 2, 3, 1)
                out['ovov'][s] = g[:, o, v, v].transpose(0, 2, 1, 3) - g[:, v, v, o].transpose(0, 2, 3, 1)
                out['ovvo'][s] = g[:, v, v, o].transpose(0, 2, 1, 3) - g[:, o, v, v].transpose(0, 2, 3, 1)
                if with_ovvv:
                    list(pool.map(lambda i: sort_ovvv(i, g[i - i0]), range(i0, i1)))
    return out


def ladder_operators(eri4, nao):
    '''The two AO operators of the AO-driven pp ladder (DirectZCCSD), from the
    packed integrals: (mu nu|la si) as a matrix [(nu si), (mu la)], and the
    antisymmetrized same-spin integral (mu si|nu rho) - (mu rho|nu si) packed
    over mu > nu and si > rho.  Returns (ao_eri, W_same, (pi, pj)).'''
    idx = _pair_index(nao)
    ao_eri = np.empty((nao, nao, nao, nao))
    pi, pj = np.tril_indices(nao, -1)
    W = np.empty((pi.size, pi.size))
    rows = [slice(pi.size * j // nao, pi.size * (j + 1) // nao) for j in range(nao)]

    def fill(nu):
        # [nu, si, mu, la] = (mu nu|la si)
        ao_eri[nu] = eri4[idx[nu][None, :, None], idx[:, None, :]]
        r = rows[nu]
        mu, n2 = pi[r, None], pj[r, None]
        W[r] = eri4[idx[mu, pi[None, :]], idx[n2, pj[None, :]]]
        W[r] -= eri4[idx[mu, pj[None, :]], idx[n2, pi[None, :]]]

    with timer('ao2mo.ladder'), ThreadPoolExecutor(_threads()) as pool:
        list(pool.map(fill, range(nao)))
    return ao_eri.reshape(nao * nao, nao * nao), W, (pi, pj)


def fits_in_memory(nbytes, share=0.5):
    '''Whether an array of ``nbytes`` takes less than ``share`` of the
    memory that is available (the node and the limit of the job).'''
    from zccsdt import memory
    avail = memory.available()
    return avail is None or nbytes < share * avail


def packed_vvvv(eri4, Ca, Cb, nocc, out=None, slab_bytes=SLAB_BYTES):
    '''<ab||ef> packed over a > b and e > f, rows and columns in the order of
    numpy.tril_indices(nvir, -1), written to ``out`` ((npair, npair); a new
    array by default).  Returns (out, (ti, tj)).'''
    nmo = Ca.shape[1]
    nvir = nmo - nocc
    ti, tj = np.tril_indices(nvir, -1)
    npair = ti.size
    if out is None:
        out = np.empty((npair, npair), dtype=np.complex128)

    def sort_a(a, ga):
        # ga[e, b, f] = (ae|bf); <ab||ef> = (ae|bf) - (af|be) for b < a
        for b in range(a):
            P = np.ascontiguousarray(ga[:, b, :])
            P -= P.T.copy()
            out[a * (a - 1) // 2 + b] = P[ti, tj]

    with ThreadPoolExecutor(_threads()) as pool:
        for a0, a1, g in slabs(eri4, Ca, Cb, (nocc, nmo), (nocc, nmo), slab_bytes):
            with timer('ao2mo.sort'):
                list(pool.map(lambda a: sort_a(a - nocc, g[a - a0]), range(a0, a1)))
    return out, (ti, tj)
