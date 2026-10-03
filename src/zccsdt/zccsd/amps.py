'''The amplitude equations of the AO-driven spinor CCSD on packed amplitudes.

The equations are those of zccsd_direct.update_amps (PySCF's GCCSD with the
intermediates of Stanton, Gauss et al., bare integrals); only the way each
term is evaluated differs:

- T2, tau and the residual are packed, [(i>j), (a>b)] in
  numpy.tril_indices(n, -1) order; nothing of o^2 v^2 elements is made by
  fancy indexing: the layouts of the terms come from the OpenMP kernels of
  zccsdt.zccsd.kernels, and every matrix product goes through zccsdt.blas
  (the BLAS of the kernel library: with MKL on GNU OpenMP, the kernels and
  the products share one pool of threads and numpy's BLAS stays idle);
- the two o^3 v^3 terms, W(mbej) += <mn||ef> t2 and the ring t2 W, are
  ZGEMMs in the "ring layout" [(i a), (j b)], in which t2 and <mn||ef> are
  symmetric matrices.  The first pass runs over slabs of the rows (m e) and
  completes W(mbej) there; the integrals <mn||ef>, <mb||je> and <ma||ef>
  are each read once, by rows, in this pass.  The second pass runs over
  blocks of the columns (j b): the product with t2, antisymmetrized into the
  packed residual block by block.  The rows and columns of t2 are generated
  from the packed amplitudes; the two-index intermediates that contract
  <mn||ef> with t2 are partial traces of the first product;
- the five terms with <ma||ef> use the array packed over e > f (half of
  ovvv): one GEMM with tau for a slab, and the t1 terms of one occupied
  index at a time, unpacked in the cache;
- W(mnij) and the occupied-occupied Fock intermediate act on the packed
  pairs as one matrix; the terms that multiply t2 from the virtual side run
  over blocks of pairs;
- the pp ladder runs in the scalar AO basis over blocks of pairs i > j.
  The alpha-beta block of the back-transformed tau is split into its
  symmetric and antisymmetric parts in the two AO indices: the antisymmetric
  part joins alpha-alpha and beta-beta on the packed antisymmetrized
  integrals (mu si|nu rho) - (mu rho|nu si), the symmetric part takes the
  packed symmetrized integrals.  Two operators of 2 nao^4 bytes replace the
  nao^2 x nao^2 matrix, and the real GEMMs do o^2 nao^4 multiply-adds.

Memory.  Nothing but three packed T2 (the amplitudes, tau, the residual) and
arrays smaller than those is assumed to fit: plan_step sizes the slabs and
blocks from what ``max_memory`` leaves, keeps W(mbej) (o^2 v^2 elements) in
memory only if the smallest blocks still fit next to it (else it is a mapped
array of a scratch file, written by rows and read by columns), and takes the
integrals that zccsdt.zccsd.ao2mo has put into scratch files by rows.
Smaller blocks mean that rows of t2 are generated more often; MemoryError
below the working set of one occupied index.

kernel() is the CCSD iteration on these amplitudes: the scheme of PySCF's
driver (Jacobi step, damping, DIIS on the vector [t1, packed t2], the same
convergence test and log lines) with T2 packed from the first guess to the
converged amplitudes.

Timers (zccsdt.timing): 'amps.imds', 'amps.ovvv', 'amps.ring', 'amps.t2',
'amps.ladder', 'amps.t1' with sub-timers; 'amps.diis', 'amps.energy'.
'''

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from pyscf.lib import logger

from zccsdt import blas, memory, scratch
from zccsdt.cc.diis import DIIS
from zccsdt.timing import timer
from zccsdt.zccsd import kernels as K

PAIR_WORK = 2 << 30         # bytes of the arrays of a block of pairs, at most


def _blocks(n, size):
    size = max(1, min(n, int(size)))
    return [(p, min(p + size, n)) for p in range(0, n, size)]


def _size(nbytes):
    return '%.2f GB' % (nbytes / 1e9) if nbytes >= 1e9 else '%.1f MB' % (nbytes / 1e6)


@dataclass
class Plan:
    '''How one step is sliced (plan_step); blocks are lists of (start, stop).'''
    mblocks: list       # slabs of the first occupied index: rows of the ring layout, <ma||ef>
    jblocks: list       # blocks of the occupied index of the columns of the ring layout
    pblocks: list       # blocks of pairs of the terms on the virtual indices
    lblocks: list       # blocks of pairs of the ladder
    krows: int          # rows of an AO operator of the ladder at a time (0: all)
    w_mapped: bool      # W(mbej) is a mapped array of a scratch file
    diis_incore: bool   # the DIIS vectors are in memory
    work: int           # bytes of the largest working set of this plan
    need: int           # bytes of the smallest one
    budget: object      # bytes there were (None: unknown, everything in memory)


def plan_step(nocc, nvir, nao, budget=None, mapped=(), diis_space=6, diis_incore=None):
    '''The Plan of one step within ``budget`` bytes (None: unlimited).
    ``mapped``: the integrals that are mapped arrays, of 'V' (<mn||ef>), 'E'
    (<mb||je>), 'ovvv_p', 'W_same', 'W_plus'; their rows are read through the
    mapping and count as memory while they are used.  ``diis_incore``: keep
    the DIIS vectors in memory (None: if they take no more than a third of
    what is left).  MemoryError if the budget is below Plan.need.'''
    o, v, nao = int(nocc), int(nvir), int(nao)
    n1, npo, npv, n2 = o * v, K.npair(o), K.npair(v), 2 * nao
    budget = None if budget is None else int(budget)
    P = 16 * npo * npv                              # one packed T2
    col = 16 * n1 * v                               # the rows (m e) of one m, the columns (j b) of one j
    # t2, tau, the residual; W(mnij) and its Fock part, Nq, the t1 <mb||je> array, o^4
    fixed = 3 * P + 16 * (3 * npo * npo + 2 * o * v * npo + 2 * o * o * n1 + 4 * o ** 4)
    row = col * (('V' in mapped) + ('E' in mapped)) + 16 * v * npv * ('ovvv_p' in mapped)
    pair = 32 * v * v + 16 * o * v                  # virtual-side terms of one pair
    lad = max(16 * n2 * (n2 + v), 24 * n2 * n2, 16 * v * (n2 + v))      # the ladder of one pair
    kop = 8 * nao * (nao + 1) // 2 * (('W_same' in mapped) or ('W_plus' in mapped))

    def rows_work(nm, nj, w_mapped):
        '''Pass 1: the columns of t2, the mapped rows and the product block.'''
        return nj * col + nm * (row + col * w_mapped + (nj < o) * nj * 16 * v * v)

    floor = max(3 * col, rows_work(1, 1, True), pair, lad + kop)
    need = fixed + floor
    if budget is None:
        return Plan([(0, o)], [(0, o)], _blocks(npo, PAIR_WORK // pair),
                    _blocks(npo, PAIR_WORK // lad), 0, False,
                    True if diis_incore is None else diis_incore,
                    fixed + 3 * o * col + (2 * diis_space * P if diis_incore is not False else 0),
                    need, None)
    if budget < need:
        raise MemoryError(
            'the CCSD equations need %s (three packed T2 and the blocks of one occupied '
            'orbital: %d occupied and %d virtual spinors), %s are left of max_memory'
            % (_size(need), o, v, _size(budget)))
    rem = budget - fixed
    # W(mbej) in memory if the smallest blocks still fit next to it
    w_mapped = bool(o * col + max(2 * col, rows_work(1, 1, False), pair, lad + kop) > rem)
    if not w_mapped:
        rem -= o * col
    if diis_incore is None:
        diis_incore = bool(2 * diis_space * P <= rem // 3)
    if diis_incore:
        rem = max(rem - 2 * diis_space * P, floor)
    # pass 2 holds a block of columns of the product and the same number of rows
    # of t2 (and, through the mapping, the columns of W)
    nj = max(1, min(o, rem // ((3 if w_mapped else 2) * col)))
    while nj > 1 and rows_work(1, nj, w_mapped) > rem:
        nj -= 1
    per_m = rows_work(1, nj, w_mapped) - nj * col
    nm = o if per_m == 0 else max(1, min(o, (rem - nj * col) // per_m))
    lrem = min(rem, PAIR_WORK)
    krows = 0 if not kop else max(1, lrem // 2 // kop)
    npb = max(1, min(npo, lrem // pair))
    nlb = max(1, min(npo, (lrem - krows * kop) // lad))
    work = (budget - rem) + max(rows_work(nm, nj, w_mapped), (3 if w_mapped else 2) * nj * col,
                                npb * pair, nlb * lad + krows * kop)
    return Plan(_blocks(o, nm), _blocks(o, nj), _blocks(npo, npb), _blocks(npo, nlb), krows,
                w_mapped, diis_incore, work, need, budget)


class Residual:
    '''The CCSD equations for the integrals ``eris`` (the container of
    DirectZCCSD with the stored <ma||ef>).

    update_packed(t1, t2p) is one Jacobi step on the packed amplitudes;
    update(t1, t2) does the same on t2[i,j,a,b], as zccsd_direct.update_amps.
    ``max_memory`` (MB for the process, as in PySCF; None: 80% of what the
    node and the job leave) and the DIIS settings go into the Plan (``plan``:
    one given explicitly).'''

    def __init__(self, eris, max_memory=None, diis_space=6, diis_incore=None, plan=None):
        o = self.nocc = eris.nocc
        fock = np.asarray(eris.fock)
        v = self.nvir = fock.shape[0] - o
        n1 = o * v
        self.foo = np.ascontiguousarray(fock[:o, :o])
        self.fov = np.ascontiguousarray(fock[:o, o:])
        self.fvv = np.ascontiguousarray(fock[o:, o:])
        self.mo_e = np.asarray(eris.mo_energy, dtype=np.float64)
        io, jo = K.tril(o)
        npo = K.npair(o)

        with timer('amps.setup'):
            # ring layouts: V[(m e), (n f)] = <mn||ef>, E[(m e), (j b)] = -<mb||je>,
            # and <mn||ef> on packed pairs; made by zccsdt.zccsd.ao2mo, else from
            # the blocks oovv and ovov
            V, E = getattr(eris, 'oovv_r', None), getattr(eris, 'ovov_r', None)
            oovv_p = getattr(eris, 'oovv_p', None)
            if V is None:
                V = np.ascontiguousarray(np.asarray(eris.oovv).transpose(0, 2, 1, 3))
            if E is None:
                E = -np.ascontiguousarray(np.asarray(eris.ovov).transpose(0, 3, 2, 1))
            if oovv_p is None:
                oovv_p = K.pack_t2(np.ascontiguousarray(np.asarray(eris.oovv)))
            self.V, self.E = V.reshape(n1, n1), E.reshape(n1, n1)
            self.oovv_p = np.ascontiguousarray(oovv_p)                  # in memory
            self.oooo_p = np.ascontiguousarray(np.asarray(eris.oooo)[io, jo][:, io, jo])
            ooov = self.ooov = np.ascontiguousarray(np.asarray(eris.ooov))  # <mn||ie>: [m,n,i,e]
            self.ooov_r = np.ascontiguousarray(ooov.transpose(0, 3, 2, 1))  # [m, e, j, n]
            # [i, e, (m>n)]
            self.ooov_en = np.ascontiguousarray(ooov[io, jo].transpose(1, 2, 0))
            # <ma||ef> packed over e > f: [m, a, (e>f)]
            ovvv_p = getattr(eris, 'ovvv_p', None)
            if ovvv_p is None:
                av, bv = K.tril(v)
                ovvv_p = np.ascontiguousarray(np.asarray(eris.ovvv)[:, :, av, bv])
            self.ovvv_p = ovvv_p

        # AO side of the ladder
        self.nao = eris.nao_sph
        self.C2 = np.ascontiguousarray(eris.mo_ao_spin[:, o:])          # (2 nao, nvir)
        self.C2c = self.C2.conj()
        self.W_same, self.W_plus = eris.W_same, eris.W_plus

        # the pairs (m, k), k = 0 .. o-1, that contain the occupied m: packed
        # row and the sign of t2[m, k] against it (k = m: no row)
        k = np.arange(o)
        self.pair_rows = []
        for m in range(o):
            hi, lo = np.maximum(m, k), np.minimum(m, k)
            rows = hi * (hi - 1) // 2 + lo
            rows[m] = -1
            self.pair_rows.append((rows, np.where(k < m, 1.0, -1.0) * (k != m)))

        # -P(ij) sum_m F[m,j] t2[i,m] as a matrix G[(k>l), (i>j)] on the pairs,
        #   G = d(l,i) F[k,j] - d(l,j) F[k,i] - d(k,i) F[l,j] + d(k,j) F[l,i]:
        # (row, column, the two indices of F, sign) of its nonzeros
        x, kk, ll = k[None, :], io[:, None], jo[:, None]
        rows = np.broadcast_to(np.arange(npo)[:, None], (npo, o))

        def pair(a, b):
            hi, lo = np.maximum(a, b), np.minimum(a, b)
            return hi * (hi - 1) // 2 + lo
        self._g = []
        for mask, cols, f0, sign in ((x < ll, pair(ll, x), kk, 1.0), (x > ll, pair(x, ll), kk, -1.0),
                                     (x < kk, pair(kk, x), ll, -1.0), (x > kk, pair(x, kk), ll, 1.0)):
            self._g.append((rows[mask], cols[mask], np.broadcast_to(f0, (npo, o))[mask],
                            np.broadcast_to(x, (npo, o))[mask], sign))

        if plan is None:
            names = ('V', 'E', 'ovvv_p', 'W_same', 'W_plus')
            mapped = [n for n in names if scratch.is_mapped(getattr(self, n))]
            plan = plan_step(o, v, self.nao, memory.budget(max_memory), mapped, diis_space,
                             diis_incore)
        self.plan = plan
        self._scratch = None        # the file of W(mbej) when it is mapped
        self._w = None

    def describe(self):
        '''One line on the memory plan of a step.'''
        p, o = self.plan, self.nocc
        nb = lambda b: max(b1 - b0 for b0, b1 in b)                     # noqa: E731
        text = ('packed T2 %s; working set %s%s; W(mbej) %s; passes over %d and %d of %d '
                'occupied spinors; %d and %d pairs per block'
                % (_size(16 * K.npair(o) * K.npair(self.nvir)), _size(p.work),
                   '' if p.budget is None else ' of %s' % _size(p.budget),
                   'in a scratch file' if p.w_mapped else 'in memory',
                   nb(p.mblocks), nb(p.jblocks), o, nb(p.pblocks), nb(p.lblocks)))
        if p.krows:
            text += '; %d rows of an AO operator at a time' % p.krows
        return text + '; DIIS vectors in %s' % ('memory' if p.diis_incore else 'a scratch file')

    def close(self):
        '''Release W(mbej) and its scratch file.'''
        self._w = None
        if self._scratch is not None:
            self._scratch.close()
            self._scratch = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _w_array(self):
        '''W(mbej) in the ring layout: a new array, or the mapped one.'''
        n1 = self.nocc * self.nvir
        if not self.plan.w_mapped:
            return np.empty((n1, n1), dtype=np.complex128)
        if self._w is None:
            self._scratch = scratch.Scratch(
                prefix='zccsd_w_', what='W(mbej)',
                hint=' or give the calculation more memory (max_memory)')
            self._w = self._scratch.create('W', (n1, n1), np.complex128)
        return self._w

    # -- amplitudes and energy on packed pairs -------------------------------
    def init_amps(self):
        '''(E(MP2), t1, t2p): the guess of PySCF's GCCSD.'''
        o = self.nocc
        eo, ev = self.mo_e[:o], self.mo_e[o:]
        t1 = self.fov / (eo[:, None] - ev)
        t2p = K.div_packed(K.copy(self.oovv_p, conj=True), eo, ev)
        return float(K.dotu(self.oovv_p, t2p).real), t1, t2p

    def energy(self, t1, t2p):
        '''E_corr = f_ia t_ia + sum_{i>j, a>b} <ij||ab> tau_ij^ab.'''
        t1 = np.ascontiguousarray(t1, dtype=np.complex128)
        taup = K.outer_pp(t1, t1, 0.5, K.copy(np.ascontiguousarray(t2p, dtype=np.complex128)))
        return np.sum(self.fov * t1) + K.dotu(self.oovv_p, taup)

    # -- the pp ladder in the AO basis --------------------------------------
    def _ao_operator(self, A, W):
        '''A @ W for a real AO operator W: by blocks of its rows if the plan
        says so (the operator is a mapped array).'''
        krows = self.plan.krows
        if not krows:
            return blas.gemm(A, W)
        out = np.zeros((A.shape[0], W.shape[1]))
        for k0 in range(0, W.shape[0], krows):
            Wk = W[k0:k0 + krows]
            blas.gemm(A[:, k0:k0 + krows], Wk, C=out, beta=1.0)
            scratch.unmap(Wk)
        return out

    def ladder(self, taup, R):
        '''R[(i>j), (a>b)] += sum_ef (ae|bf) tau_ij^ef = 1/2 sum_ef <ab||ef> tau_ij^ef.'''
        v, nao = self.nvir, self.nao
        n2, C2, C2c = 2 * nao, self.C2, self.C2c
        for p0, p1 in self.plan.lblocks:
            n = p1 - p0
            with timer('ladder.ao'):
                # X[x, p, y] = sum_ef C2[x, e] tau[p, e, f] C2[y, f], x and y over (spin, AO)
                tf = K.unpack_tril(taup[p0:p1], v, 'apb')               # [e, p, f]
                H = blas.gemm(C2, tf.reshape(v, n * v))
                del tf
                X = blas.gemm(H.reshape(n2 * n, v), C2.T)               # [x, p, y]
                del H
            with timer('ladder.pack'):
                AR, SR = K.ladder_pack(X, n, nao)
                del X
            with timer('ladder.gemm'):
                # the real AO operators on real and imaginary parts at once
                GA = self._ao_operator(AR, self.W_same)
                GS = self._ao_operator(SR, self.W_plus)
                del AR, SR
            with timer('ladder.pack'):
                G = K.ladder_unpack(GA, GS, n, nao)                     # [x, p, y]
                del GA, GS
            with timer('ladder.ao'):
                # L[p] = C2^+ G[p] C2^*
                H = blas.gemm(C2c.T, G.reshape(n2, n * n2))
                del G
                L = blas.gemm(H.reshape(v * n, n2), C2c)                # [a, p, b]
                del H
            with timer('ladder.pack'):
                K.pack_tril_add(L.reshape(v, n, v), 1.0, R[p0:p1], 'apb')
                del L
        return R

    # -- the two passes over the ring layout -----------------------------------
    def _pass_rows(self, s, W):
        '''Pass 1, over slabs of the first occupied index m: the rows (m e)
        of W(mbej) - the bare integral, its t1 terms, the <ma||ef> t1 term
        and the product <mn||ef> t2 - and everything else that reads
        <mn||ef>, <mb||je> and <ma||ef>.'''
        o, v = self.nocc, self.nvir
        n1, npo, npv = o * v, K.npair(o), K.npair(v)
        plan, V, E = self.plan, self.V, self.E
        t1, t1T, t1c, t2p, taup, R = s.t1, s.t1T, s.t1.conj(), s.t2p, s.taup, s.R
        single = len(plan.jblocks) == 1
        s.g = np.empty((o, v), dtype=np.complex128)         # sum_nf <mn||ef> t1[n,f]
        # Kt[m, i, (j b)] = sum_e t1[i,e] E[(m e), (j b)]
        s.Kt = np.empty((o, o, n1), dtype=np.complex128)
        t1row = t1.reshape(1, n1)
        # Nq[m, a, (i>j)] = <ij||ma>* + sum_{e>f} <ma||ef> tau[(ij), (ef)]
        s.Nq = np.conj(self.ooov_en)
        s.X1 = np.zeros((o, v), dtype=np.complex128)
        s.r1 = np.zeros((o, v), dtype=np.complex128)
        s.foo_t = np.zeros((o, o), dtype=np.complex128)     # partial traces of 1/2 V T
        s.fvv_t = np.zeros((v, v), dtype=np.complex128)
        s.fvv_a = np.zeros((v, v), dtype=np.complex128)     # sum_mf <ma||fe> t1[m,f]
        fv = np.empty((v, v), dtype=np.complex128)
        x1 = np.empty((o, v), dtype=np.complex128)
        T = None
        for m0, m1 in plan.mblocks:
            nm = m1 - m0
            nr, rows = nm * v, slice(m0 * v, m1 * v)
            Vb, Eb, Wb = V[rows], E[rows], W[rows]
            with timer('amps.ring'), timer('ring.small'):
                # -<mb||je> - sum_n t1[n,b] (<mn||je> - sum_f <mn||ef> t1[j,f])
                s.g[m0:m1] = blas.gemm(Vb, t1row.T).reshape(nm, v)
                Z = blas.gemm(Vb.reshape(nr * o, v), t1T).reshape(nr, o, o)     # [(m e), n, j]
                N = self.ooov_r[m0:m1].reshape(nr, o, o) - Z.transpose(0, 2, 1)
                del Z
                K.copy(Eb, Wb)
                blas.gemm(N.reshape(nr * o, o), t1, C=Wb.reshape(nr * o, v), beta=1.0)
                del N
                # for - t1[m,a] t1[i,e] <mb||ej> and the T1 equation
                for m in range(m0, m1):
                    blas.gemm(t1, Eb[(m - m0) * v:(m - m0 + 1) * v], C=s.Kt[m])
                blas.gemm(t1row[:, rows], Eb, C=s.r1.reshape(1, n1), beta=1.0)

            with timer('amps.ovvv'):
                # the five terms with <ma||ef>
                O = self.ovvv_p[m0:m1]                                  # [m, a, (e>f)]
                with timer('ovvv.gemm'):
                    blas.gemm(O.reshape(nm * v, npv), taup.T,
                              C=s.Nq[m0:m1].reshape(nm * v, npo), beta=1.0)
                with timer('ovvv.t1'):
                    for m in range(m0, m1):
                        Om = O[m - m0]
                        prow, sgn = self.pair_rows[m]
                        # sum_e t1[i,e] <ab||em> as conj(sum_e t1*[i,e] <me||ab>)
                        K.add_rows(R, prow, sgn, blas.gemm(t1c, Om), conj=True)
                        # sum_f <mb||ef> t1[j,f] into W(mbej), Fvv and the T1 equation
                        K.ovvv_t1(Om, t1, m, Wb[(m - m0) * v:(m - m0 + 1) * v], fv, x1)
                        s.fvv_a += fv
                        # sum_{e>f} <ma||ef> t2[m,i,(ef)]: the tau term less its t1 t1 part
                        c = np.maximum(prow, 0)
                        s.X1 += ((s.Nq[m][:, c] - np.conj(self.ooov_en[m][:, c])) * sgn).T - x1
                scratch.unmap(O)
                del O

            with timer('amps.ring'):
                # + 1/2 sum_nf <mn||ef> t2[n,j,f,b]: the o^3 v^3 term of W; its
                # partial traces are the <mn||ef> t2 terms of Foo and Fvv
                for j0, j1 in plan.jblocks:
                    nj = j1 - j0
                    with timer('ring.layout'):
                        if T is None or not single:
                            T = K.ring_cols(t2p, o, v, j0, j1)          # [(n f), (j b)]
                    W4 = Wb.reshape(nm, v, o, v)[:, :, j0:j1]           # [m, e, j, b]
                    both = range(max(m0, j0), min(m1, j1))              # m as a row and a column
                    if single:
                        with timer('ring.layout'):
                            tro = np.einsum('meje->mj', W4)
                            trv = sum(W4[m - m0, :, m - j0] for m in both)
                        with timer('ring.gemm'):
                            blas.gemm(Vb, T, C=Wb, alpha=0.5, beta=1.0)
                        with timer('ring.layout'):
                            s.foo_t[m0:m1, j0:j1] += np.einsum('meje->mj', W4) - tro
                            s.fvv_t += sum(W4[m - m0, :, m - j0] for m in both) - trv
                    else:
                        with timer('ring.gemm'):
                            X4 = blas.gemm(Vb, T, alpha=0.5).reshape(nm, v, nj, v)
                        with timer('ring.layout'):
                            s.foo_t[m0:m1, j0:j1] += np.einsum('meje->mj', X4)
                            for m in both:
                                s.fvv_t += X4[m - m0, :, m - j0]
                            W4 += X4
                            del X4
            scratch.written(Wb)
            for a in (Vb, Eb, Wb):
                scratch.unmap(a)
        s.T = T if single else None                         # all of t2: kept for pass 2

    def _pass_columns(self, s, W):
        '''Pass 2, over blocks of the columns (j b): the ring term
        sum_me t2[i,m,a,e] W[(m e), (j b)] and the t1 t1 <mb||ej> term,
        antisymmetrized into the packed residual.'''
        o, v = self.nocc, self.nvir
        n1, jblocks = o * v, self.plan.jblocks
        for j0, j1 in jblocks:
            nj = j1 - j0
            cols = slice(j0 * v, j1 * v)
            WJ = W if nj == o else W[:, cols]
            with timer('ring.gemm'):
                if s.T is not None:
                    r4 = blas.gemm(s.T, WJ)                             # [(i a), (j b)]
                    s.T = None
                else:
                    r4 = np.empty((n1, nj * v), dtype=np.complex128)
                    for i0, i1 in jblocks:                              # rows of t2: it is symmetric
                        with timer('ring.layout'):
                            Ti = K.ring_cols(s.t2p, o, v, i0, i1)
                        blas.gemm(Ti.T, WJ, C=r4[i0 * v:i1 * v])
                        del Ti
            with timer('ring.small'):
                # - t1[m,a] t1[i,e] <mb||ej> (E is <mb||ej> in the ring layout)
                r43 = r4.reshape(o, v, nj * v)
                for i in range(o):
                    blas.gemm(s.t1T, s.Kt[:, i, cols], C=r43[i], alpha=-1.0, beta=1.0)
            with timer('ring.pack'):
                K.asym_pack_cols(r4, o, v, j0, j1, 1.0, s.R)
            del r4, WJ
            scratch.unmap(W)

    # -- one step -------------------------------------------------------------
    def update_packed(self, t1, t2p, level_shift=0.0):
        '''(t1new, t2pnew) from t1[i,a] and t2p[(i>j),(a>b)].'''
        o, v = self.nocc, self.nvir
        npo = K.npair(o)
        io, jo = K.tril(o)
        fov = self.fov
        eo, ev = self.mo_e[:o], self.mo_e[o:] + level_shift
        t1 = np.ascontiguousarray(t1, dtype=np.complex128)
        t2p = np.ascontiguousarray(t2p, dtype=np.complex128)
        t1T = np.ascontiguousarray(t1.T)
        s = SimpleNamespace(t1=t1, t1T=t1T, t2p=t2p)

        with timer('amps.imds'):
            taup = s.taup = K.outer_pp(t1, t1, 0.5, K.copy(t2p))
            # tmp[m, n, i, j] = sum_e <mn||ie> t1[j, e]
            tmp = blas.gemm(self.ooov.reshape(o ** 3, v), t1T).reshape(o, o, o, o)
            foo_x = np.einsum('mnin->mi', tmp)
            # W(mnij) on packed pairs, with the tau <mn||ef> term of the ladder
            tp = tmp[io, jo]
            Wp = self.oooo_p + tp[:, io, jo] - tp[:, jo, io]
            blas.gemm(self.oovv_p, taup.T, C=Wp, beta=1.0)
            del tmp, tp
            R = s.R = K.copy(self.oovv_p, conj=True)

        W = self._w_array()
        self._pass_rows(s, W)
        with timer('amps.ring'):
            self._pass_columns(s, W)
        del W
        r1, Nq = s.r1, s.Nq
        del s.Kt

        with timer('amps.t2'):
            Fov = fov + s.g
            Ft, tF = blas.gemm(Fov, t1.T), blas.gemm(t1T, Fov)          # o x o, v x v
            Foo = self.foo + 0.5 * Ft + foo_x + s.foo_t
            Fvv = self.fvv - 0.5 * tF - s.fvv_t.T - s.fvv_a
            Foo[np.diag_indices(o)] -= eo
            Fvv[np.diag_indices(v)] -= ev
            F2o = Foo + 0.5 * Ft
            F2v = Fvv - 0.5 * tF
            # occupied side: W(mnij) and -P(ij) F2o[m,j] t2[i,m] as one matrix on
            # the pairs, applied to tau; the t1 t1 part of tau is taken back
            for rows, cols, f0, f1, sign in self._g:
                Wp[rows, cols] += sign * F2o[f0, f1]
            blas.gemm(Wp.T, taup, C=R, beta=1.0)
            del Wp
            K.outer_pp(t1, blas.gemm(F2o.T, t1), 1.0, R)
            # virtual side, in blocks of pairs
            F2vT = np.ascontiguousarray(F2v.T)
            for p0, p1 in self.plan.pblocks:
                n = p1 - p0
                t2u = K.unpack_tril(t2p[p0:p1], v, 'apb').reshape(v * n, v)   # [(a p), e]
                Y = blas.gemm(t2u, F2vT)                                # sum_e t2[p,a,e] F2v[b,e]
                Nb = Nq[:, :, p0:p1].reshape(o, v * n)
                blas.gemm(Nb.T, t1, C=Y, beta=1.0)                      # + sum_m Nq[m,a,p] t1[m,b]
                K.asym_pack2(Y.reshape(v, n, v), 1.0, R[p0:p1], 'apb')
                # - 1/2 sum_mne <mn||ie> t2[m,n,a,e] of the T1 equation
                r1 += blas.gemm(self.ooov_en[:, :, p0:p1].reshape(o, v * n), t2u)
                del t2u, Y, Nb
            del Nq, s.Nq

        with timer('amps.ladder'):
            self.ladder(taup, R)

        with timer('amps.t1'):
            r1 += blas.gemm(t1, Fvv.T) - blas.gemm(Foo.T, t1) + s.X1 + fov.conj()
            K.t2p_matvec(t2p, Fov, r1)
            K.div_packed(R, eo, ev)
            r1 /= eo[:, None] - ev
        return r1, R

    def update(self, t1, t2, level_shift=0.0):
        '''(t1new, t2new) on t2[i,j,a,b]: update_packed between a pack and an
        unpack.'''
        o, v = self.nocc, self.nvir
        with timer('amps.unpack'):
            t2p = K.pack_t2(np.ascontiguousarray(t2, dtype=np.complex128))
        t1new, t2pnew = self.update_packed(t1, t2p, level_shift)
        with timer('amps.unpack'):
            t2new = K.unpack_t2(t2pnew, o, v)
        return t1new, t2new


def kernel(mycc, eris, t1=None, t2=None, max_cycle=50, tol=1e-8, tolnormt=1e-6, verbose=None,
           callback=None):
    '''The CCSD iterations of pyscf.cc.ccsd.kernel on packed amplitudes:
    (converged, E_corr, t1, t2) with t2[i,j,a,b] unpacked at the end.

    The DIIS vector is [t1, t2 packed over i > j and a > b] as in PySCF's
    GCCSD, so the iterates are the same; it is kept in memory if the memory
    plan of the step says so or mycc.incore_complete is set, else in a
    mapped scratch file (mycc.diis_file; zccsdt.cc.diis).
    ``callback(locals())`` is called after every step as in PySCF; the new
    amplitudes are 't1new' and 't2pnew' (packed).'''
    log = logger.new_logger(mycc, verbose)
    name = mycc.__class__.__name__
    if getattr(eris, 'residual', None) is None:
        eris.residual = Residual(eris, mycc.max_memory, mycc.diis_space,
                                 True if mycc.incore_complete else None)
    res = eris.residual
    log.info('CCSD equations: %s', res.describe())
    o, v = res.nocc, res.nvir
    if t1 is None and t2 is None:
        mycc.emp2, t1, t2p = res.init_amps()
        log.info('Init t2, MP2 energy = %.15g', mycc.emp2)
    else:
        guess = res.init_amps() if t1 is None or t2 is None else None
        t1 = guess[1] if t1 is None else np.ascontiguousarray(t1, dtype=np.complex128)
        t2p = (guess[2] if t2 is None
               else K.pack_t2(np.ascontiguousarray(t2, dtype=np.complex128)))

    def energy(t1, t2p):
        with timer('amps.energy'):
            e = res.energy(t1, t2p)
        if abs(e.imag) > 1e-4:
            log.warn('Non-zero imaginary part found in %s energy %s', name, e)
        return e.real

    cput1 = cput0 = (logger.process_clock(), logger.perf_counter())
    eold = 0
    eccsd = energy(t1, t2p)
    log.info('Init E_corr(%s) = %.15g', name, eccsd)
    adiis = None
    if mycc.diis:
        adiis = DIIS(space=mycc.diis_space, filename=mycc.diis_file,
                     incore=res.plan.diis_incore)
    conv = False
    try:
        for istep in range(max_cycle):
            t1new, t2pnew = res.update_packed(t1, t2p, mycc.level_shift)
            if callback is not None:
                callback(locals())
            with timer('amps.diis'):
                normt = np.sqrt(K.diff_norm2(t1new, t1) + K.diff_norm2(t2pnew, t2p))
                if mycc.iterative_damping < 1.0:
                    alpha = mycc.iterative_damping
                    t1new = (1 - alpha) * t1 + alpha * t1new
                    t2pnew *= alpha
                    t2pnew += (1 - alpha) * t2p
                t1, t2p = t1new, t2pnew
                t1new = t2pnew = None
                if (adiis is not None and istep >= mycc.diis_start_cycle
                        and abs(eccsd - eold) < mycc.diis_start_energy_diff):
                    x1, x2 = adiis.update([t1, t2p])
                    t1, t2p = x1.reshape(o, v), x2.reshape(t2p.shape)
                    log.debug1('DIIS for step %d', istep)
            eold, eccsd = eccsd, energy(t1, t2p)
            log.info('cycle = %d  E_corr(%s) = %.15g  dE = %.9g  norm(t1,t2) = %.6g',
                     istep + 1, name, eccsd, eccsd - eold, normt)
            cput1 = log.timer('%s iter' % name, *cput1)
            if abs(eccsd - eold) < tol and normt < tolnormt:
                conv = True
                break
    finally:
        if adiis is not None:
            adiis.close()
    log.timer(name, *cput0)
    return conv, eccsd, t1, K.unpack_t2(np.ascontiguousarray(t2p), o, v)

