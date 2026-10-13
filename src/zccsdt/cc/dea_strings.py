'''EOM-DEA of any rank on determinants: the reference route (docs/eom.md,
section 11).

The states with two electrons more than the reference are expanded in the
determinants |K, B> = v+(B) o+(K) |vac> - K the occupied spinors that are
still filled (no - h of them: h holes), B the h + 2 virtual spinors that
are filled, creators in ascending order - with h <= hmax:

    hmax = 1:  2p + 3p1h          hmax = 3:  .. + 5p3h
    hmax = 2:  .. + 4p2h          hmax = 4:  .. + 6p4h,   hmax = no: all (full CI).

The matrix is P exp(-T) H exp(T) P on them, for any T = T1 + .. + Tn given,
evaluated as it stands: exp(T) on the vector up to h = hmax + 2 (H lowers h
by two at most), H, exp(-T) (which only raises h) - products of operators
on vectors of determinants, no diagrams.  With T through T_m, m >= hmax,
this is the EOM matrix <mu| [Hbar, R] |0> + E_CC of the route through the
ghost spinors (zccsdt.cc.dea), which it reproduces; its eigenvalues are
total energies less ``Problem.const``.

Every operator is sum coef[X, Yc, Ya] v+(Yc) v(Ya) O(X), O(X) = o+(Xc) o(Xa):
for a determinant K' of the result, the K and X that reach it; the virtual
strings B = B'' + Ya gathered, one matrix product with the coefficients, the
result scattered to B' = B'' + Yc - label by label of B'' (labels: 2 Omega
and the parity of every spinor; a vector has one total label).

Cost: the intermediate determinants with hmax + 2 holes.  That is a
reference for small spaces - 22 virtual spinors and hmax = 3: 4.3e7 of
them before the labels; 106 virtual spinors: 3.4e12 - not a production
route: the products are not connected.
'''

from itertools import combinations
import time

import numpy as np

from zccsdt.cc import eom


def _tuples(n, k):
    if k == 0:
        return np.zeros((1, 0), dtype=np.int64)
    return np.array(list(combinations(range(n), k)), dtype=np.int64).reshape(-1, k)


class _Strings:
    '''The k-subsets of n orbitals as bit masks in ascending order, with the
    sums of their labels (w: adds; p: adds modulo 2), grouped by label.'''
    _memo = None

    def __init__(self, w, p):
        self.w, self.p, self.n = np.asarray(w, dtype=np.int64), np.asarray(p, dtype=np.int64), len(w)
        self._memo, self._groups, self._below = {}, {}, {}

    def raw(self, n, k):
        '''(masks, w sums, p sums) of the k-subsets of the first n orbitals.'''
        key = (n, k)
        if key not in self._memo:
            if k == 0:
                out = (np.zeros(1, dtype=np.int64),) * 3
            elif k > n:
                out = (np.zeros(0, dtype=np.int64),) * 3
            else:
                ms, ws, ps = [], [], []
                for j in range(k - 1, n):       # the highest orbital of the subset
                    m, w, p = self.raw(j, k - 1)
                    ms.append(m | (1 << j))
                    ws.append(w + self.w[j])
                    ps.append((p + self.p[j]) % 2)
                out = tuple(np.concatenate(x) for x in (ms, ws, ps))
            self._memo[key] = out
        return self._memo[key]

    def groups(self, k):
        '''{(w, p): masks (ascending)} of the k-subsets.'''
        if k not in self._groups:
            m, w, p = self.raw(self.n, k)
            code = w * 2 + p
            order = np.argsort(code, kind='stable')
            cs = code[order]
            cut = np.flatnonzero(np.diff(cs)) + 1
            out = {}
            for idx in np.split(order, cut):
                if len(idx):
                    out[(int(w[idx[0]]), int(p[idx[0]]))] = m[idx]
            self._groups[k] = out
        return self._groups[k]

    def tuples(self, k):
        '''The index tuples of an operator on k orbitals in lexicographic
        order (the order of the packed amplitudes): {(w, p): (positions,
        masks, tuples)}.'''
        key = ('t', k)
        if key not in self._groups:
            t = _tuples(self.n, k)
            m = (1 << t).sum(axis=1) if k else np.zeros(1, dtype=np.int64)
            w, p = self.w[t].sum(axis=1), self.p[t].sum(axis=1) % 2
            out = {}
            for kk in sorted({(int(a), int(b)) for a, b in zip(w, p)}):
                idx = np.flatnonzero((w == kk[0]) & (p == kk[1]))
                out[kk] = (idx, m[idx], t[idx])
            self._groups[key] = out
        return self._groups[key]


def _parity_below(masks, n):
    '''[string, p]: the parity of the number of orbitals below p in it.'''
    bits = ((masks[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)
    out = np.zeros_like(bits)
    np.cumsum(bits[:, :-1], axis=1, out=out[:, 1:])
    return out & 1


def _add(a, b, sign=1):
    return (a[0] + sign * b[0], (a[1] + b[1]) % 2)


class Problem:
    '''The matrix of one truncation (module docstring).

    fock, g        the Fock matrix of the reference and <pq||rs>, dense, occupied first
    amps           [t1, t2, ..]: t1[i,a], the others packed [C(no,k), C(nv,k)]
    hmax           the highest number of holes of the operator
    labels, parity one integer per spinor (2 Omega; 0 / 1), ``target`` and
                   ``target_parity`` the sums over the two electrons; None: no labels'''

    def __init__(self, fock, g, nocc, amps, hmax, labels=None, parity=None, target=0, target_parity=0, e_ref=0.0):
        self.no, self.n = no, n = nocc, fock.shape[0]
        self.nv = nv = n - no
        self.hmax = hmax = min(hmax, no)
        self.hcap = min(hmax + 2, no, nv - 2)
        lab = np.zeros(n, dtype=np.int64) if labels is None else np.asarray(labels, dtype=np.int64)
        par = np.zeros(n, dtype=np.int64) if parity is None else np.asarray(parity, dtype=np.int64)
        self.os, self.vs = _Strings(lab[:no], par[:no]), _Strings(lab[no:], par[no:])
        full = (int(lab[:no].sum()), int(par[:no].sum()) % 2)
        self.L = _add(full, (int(target), int(target_parity) % 2))
        self.dtype = np.dtype(np.result_type(fock, g, *amps))
        o, v = slice(0, no), slice(no, n)
        h1 = fock - np.einsum('piqi->pq', g[:, o, :, o])
        self.const = float(e_ref - np.einsum('ii->', h1[o, o]).real - 0.5 * np.einsum('ijij->', g[o, o, o, o]).real)
        self.eo, self.ev = fock.diagonal().real[o].copy(), fock.diagonal().real[v].copy()
        self.e0 = float(e_ref - self.const)
        # ---- the occupied strings of every number of holes, and the layout of a vector
        self.K, self.Kkey = [], []
        for h in range(self.hcap + 1):
            m, w, p = self.os.raw(no, no - h)
            self.K.append(m)
            self.Kkey.append([(int(a), int(b)) for a, b in zip(w, p)])
        self.sizes, self.rows = [], []
        for h in range(hmax + 1):
            grp = self.vs.groups(h + 2)
            ln = [len(grp.get(_add(self.L, kk, -1), ())) for kk in self.Kkey[h]]
            self.rows.append(np.concatenate(([0], np.cumsum(ln))).astype(np.int64))
            self.sizes.append(int(self.rows[-1][-1]))
        self._sub, self._otab = {}, {}
        # ---- the operators
        self.H = self._hamiltonian(h1, g)
        self.T = []
        for k, t in enumerate(amps):
            m = k + 1
            t = np.asarray(t).reshape(len(_tuples(no, m)), len(_tuples(nv, m)))
            # t[I, A] a+(A) i_m .. i_1: the annihilators written in ascending order with the sign
            self.T.append(dict(nco=0, nao=m, ncv=m, nav=0, M=(-1.0) ** (m * (m - 1) // 2) * t[:, :, None], blocks={}))
        self.nsigma, self.seconds = 0, 0.0

    # ------------------------------------------------------------------ operators
    def _hamiltonian(self, h1, g):
        '''The classes of H = sum h1 p+ q + 1/4 sum <pq||rs> p+ q+ s r by the
        numbers of creators and annihilators among the occupied spinors:
        coef[X, Yc, Ya] of v+(Yc) v(Ya) o+(Xc) o(Xa), all tuples ascending
        and written in that order.'''
        no, nv = self.no, self.nv
        out = []
        for nc, na, tot in ((0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 1)) + tuple(
                (a, b, 2) for a in range(3) for b in range(3)):
            ncv, nav = tot - nc, tot - na
            if ncv < 0 or nav < 0 or ncv > nv or nav > nv:
                continue
            Xc, Xa, Yc, Ya = _tuples(no, nc), _tuples(no, na), _tuples(nv, ncv) + no, _tuples(nv, nav) + no
            # the string v+(Yc) o+(Xc) v(Ya) o(Xa) = p+ q+ s r with (p, q) = (Yc, Xc), (s, r) = (Ya, Xa);
            # o+(Xc) past v(Ya): (-1)^(nc nav)
            M = np.zeros((len(Xc) * len(Xa), len(Yc), len(Ya)), dtype=self.dtype)
            for ic, xc in enumerate(Xc):
                for ia, xa in enumerate(Xa):
                    cr = [Yc[:, j][:, None] for j in range(ncv)] + [int(x) for x in xc]
                    an = [Ya[:, j][None, :] for j in range(nav)] + [int(x) for x in xa]
                    if tot == 1:
                        M[ic * len(Xa) + ia] = h1[cr[0], an[0]]
                    else:
                        M[ic * len(Xa) + ia] = g[cr[0], cr[1], an[1], an[0]]
            out.append(dict(nco=nc, nao=na, ncv=ncv, nav=nav, M=(-1.0) ** (nc * nav) * M, blocks={}))
        return out

    def _otable(self, nco, nao, h):
        '''For every occupied string K' of h + nao - nco holes: [(X, K, sign)]
        with o+(Xc) o(Xa) |K> = sign |K'>, K of h holes.'''
        key = (nco, nao, h)
        if key not in self._otab:
            no = self.no
            hp = h + nao - nco
            Xc, Xa = _tuples(no, nco), _tuples(no, nao)
            mc = [int(sum(1 << int(x) for x in t)) for t in Xc]
            ma = [int(sum(1 << int(x) for x in t)) for t in Xa]
            pos = {int(m): j for j, m in enumerate(self.K[hp])} if 0 <= hp <= self.hcap else {}
            out = [[] for _ in range(len(pos))]
            for k, K in enumerate(self.K[h]):
                K = int(K)
                for ia, a in enumerate(ma):
                    if a & ~K:
                        continue
                    s = sum(bin(K & ((1 << int(x)) - 1)).count('1') for x in Xa[ia])
                    K2 = K & ~a
                    for ic, c in enumerate(mc):
                        if c & K2:
                            continue
                        j = pos.get(K2 | c)
                        if j is None:
                            continue
                        s2 = s + sum(bin(K2 & ((1 << int(x)) - 1)).count('1') for x in Xc[ic])
                        out[j].append((ic * len(Xa) + ia, k, -1.0 if s2 % 2 else 1.0))
            self._otab[key] = out
        return self._otab[key]

    def _subtable(self, k, m, lam, tkey):
        '''v+(Y ascending) on the strings B'' of k particles and label
        ``lam`` for the tuples Y of m orbitals and label ``tkey``:
        (positions of B'' + Y in its group [nB'', nY], signs; 0 where Y is
        in B'').'''
        key = (k, m, lam, tkey)
        if key not in self._sub:
            rows = self.vs.groups(k)[lam]
            _, tm, tt = self.vs.tuples(m)[tkey]
            tgt = self.vs.groups(k + m).get(_add(lam, tkey))
            if tgt is None:
                self._sub[key] = None
            else:
                ok = (rows[:, None] & tm[None, :]) == 0
                pos = np.searchsorted(tgt, rows[:, None] | tm[None, :]).astype(np.int32)
                pos[~ok] = 0
                pb = _parity_below(rows, self.nv)
                s = np.zeros(pos.shape, dtype=np.int8)
                for j in range(m):
                    s ^= pb[:, tt[:, j]]
                sgn = np.where(ok, 1.0 - 2.0 * s, 0.0)
                self._sub[key] = (pos, sgn)
        return self._sub[key]

    def _mblock(self, term, kc, ka):
        key = (kc, ka)
        if key not in term['blocks']:
            ic = self.vs.tuples(term['ncv'])[kc][0]
            ia = self.vs.tuples(term['nav'])[ka][0]
            term['blocks'][key] = np.ascontiguousarray(term['M'][:, ic][:, :, ia])
        return term['blocks'][key]

    # ------------------------------------------------------------------ vectors
    def new_block(self, h):
        grp = self.vs.groups(h + 2)
        out = {}
        for k, kk in enumerate(self.Kkey[h]):
            g = grp.get(_add(self.L, kk, -1))
            if g is not None:
                out[k] = np.zeros(len(g), dtype=self.dtype)
        return out

    def unpack(self, x):
        out = {}
        for h in range(self.hmax + 1):
            r = self.rows[h]
            out[h] = {k: np.array(x[h][r[k]:r[k + 1]]) for k in range(len(r) - 1) if r[k + 1] > r[k]}
        return out

    def pack(self, blocks, y):
        for h in range(self.hmax + 1):
            r = self.rows[h]
            y[h][:] = 0
            for k, a in blocks.get(h, {}).items():
                y[h][r[k]:r[k + 1]] = a
        return y

    def apply(self, term, vin, h, out, scale=1.0, hlim=None):
        '''out[h'] += scale * (term on the block ``vin`` of h holes).'''
        nco, nao, ncv, nav = term['nco'], term['nao'], term['ncv'], term['nav']
        hp = h + nao - nco
        hlim = self.hcap if hlim is None else hlim
        if hp < 0 or hp > hlim or not vin:
            return
        p_in = h + 2
        k2 = p_in - nav
        if k2 < 0:
            return
        vs = self.vs
        gin, g2, gout = vs.groups(p_in), vs.groups(k2), vs.groups(hp + 2)
        tA, tC = vs.tuples(nav), vs.tuples(ncv)
        phase = scale * (-1.0) ** (nav * (nav - 1) // 2 + p_in * (nco + nao))
        tab = self._otable(nco, nao, h)
        res = out.setdefault(hp, {})
        for kp, pairs in enumerate(tab):
            if not pairs:
                continue
            key_out = _add(self.L, self.Kkey[hp][kp], -1)
            tg = gout.get(key_out)
            if tg is None:
                continue
            # the pairs by the label of K (of the strings B of the operand)
            by = {}
            for X, k, s in pairs:
                if k in vin:
                    by.setdefault(self.Kkey[h][k], []).append((X, k, s))
            acc = None
            for kkey, lst in by.items():
                key_in = _add(self.L, kkey, -1)
                Xs = np.array([X for X, _, _ in lst])
                xs = np.stack([vin[k] * s for _, k, s in lst])              # [pairs, nB]
                for lam in g2:
                    ka, kc = _add(key_in, lam, -1), _add(key_out, lam, -1)
                    if ka not in tA or kc not in tC:
                        continue
                    sa = self._subtable(k2, nav, lam, ka)
                    sc = self._subtable(k2, ncv, lam, kc)
                    if sa is None or sc is None:
                        continue
                    Mb = self._mblock(term, kc, ka)[Xs]                     # [pairs, nYc, nYa]
                    G = xs[:, sa[0]] * sa[1][None]                          # [pairs, nB'', nYa]
                    npair, nb, na = G.shape
                    W = G.transpose(1, 0, 2).reshape(nb, npair * na) @ Mb.transpose(0, 2, 1).reshape(npair * na, -1)
                    W *= sc[1]
                    add = np.bincount(sc[0].ravel(), weights=W.ravel(), minlength=len(tg))
                    acc = add if acc is None else acc + add
            if acc is not None:
                if kp in res:
                    res[kp] += phase * acc
                else:
                    res[kp] = phase * acc

    def _exp(self, blocks, sign, hlim):
        '''exp(sign T) on the blocks, up to ``hlim`` holes.'''
        total = {h: dict(b) for h, b in blocks.items()}
        cur, n = blocks, 0
        while cur:
            n += 1
            new = {}
            for h, b in cur.items():
                for t in self.T:
                    self.apply(t, b, h, new, sign / n, hlim)
            new = {h: b for h, b in new.items() if b}
            for h, b in new.items():
                tb = total.setdefault(h, {})
                for k, a in b.items():
                    tb[k] = tb[k] + a if k in tb else a
            cur = new
        return total

    def sigma(self, x, y):
        t0 = time.perf_counter()
        psi = self._exp(self.unpack(x), 1.0, self.hcap)
        hpsi = {}
        for h, b in psi.items():
            for term in self.H:
                self.apply(term, b, h, hpsi, 1.0, self.hmax)
        del psi
        self.pack(self._exp(hpsi, -1.0, self.hmax), y)
        self.nsigma += 1
        self.seconds += time.perf_counter() - t0
        return y

    # ------------------------------------------------------------------ solver
    def new(self):
        return [np.zeros(s, dtype=self.dtype) for s in self.sizes]

    def diagonal(self):
        '''E_ref - const + e_a + .. - e_i - .. of every determinant.'''
        out = []
        for h in range(self.hmax + 1):
            grp = self.vs.groups(h + 2)
            d = np.zeros(self.sizes[h])
            r = self.rows[h]
            full = (1 << self.no) - 1
            for k, kk in enumerate(self.Kkey[h]):
                g = grp.get(_add(self.L, kk, -1))
                if g is None:
                    continue
                holes = full & ~int(self.K[h][k])
                eh = sum(self.eo[i] for i in range(self.no) if holes >> i & 1)
                bits = ((g[:, None] >> np.arange(self.nv)[None, :]) & 1).astype(float)
                d[r[k]:r[k + 1]] = self.e0 + bits @ self.ev - eh
            out.append(d)
        return out

    def matrix(self):
        '''Dense (tests).'''
        n = sum(self.sizes)
        M = np.zeros((n, n), dtype=self.dtype)
        off = np.concatenate(([0], np.cumsum(self.sizes)))
        for c in range(n):
            x, y = self.new(), self.new()
            h = int(np.searchsorted(off, c, side='right') - 1)
            x[h][c - off[h]] = 1.0
            self.sigma(x, y)
            M[:, c] = np.concatenate(y)
        return M


def kernel(pb, nroots=1, nguess=None, conv_tol=1e-8, max_space=None, max_cycle=100, verbose=False, stdout=None):
    '''The ``nroots`` lowest roots of a Problem: (total energies less
    pb.const + pb.const = total energies, Solved of zccsdt.cc.eom.davidson).'''
    import sys
    out = stdout or sys.stdout
    diag = pb.diagonal()
    store = eom.Store(pb.sizes, pb.dtype)

    def precond(x, omega, floor=1e-3):
        for p, d in zip(x, diag):
            den = omega - d
            den[abs(den) < floor] = floor
            p /= den
        return x
    gs = []
    for k in np.argsort(diag[0], kind='stable')[:nguess or nroots + 2]:
        v = store.new()
        v[0][k] = 1.0
        gs.append(v)
    nroots = min(nroots, len(gs))
    if verbose:
        print('EOM-DEA on determinants: up to %d holes, vectors of %s elements, intermediate determinants up to %d holes'
              % (pb.hmax, ' + '.join(str(s) for s in pb.sizes), pb.hcap), file=out, flush=True)

    def log(it, n, w, res, conv, sec):
        if verbose:
            print('iter %3d  %3d vectors  %s  %.2f s' % (it, n, '  '.join(
                '%.11f %.1e%s' % (x.real + pb.const, r, '*' if c else ' ') for x, r, c in zip(w, res, conv)), sec),
                file=out, flush=True)
    res = eom.davidson(pb.sigma, precond, gs, store, nroots, conv_tol, max_space or max(8 + 4 * nroots, len(gs) + nroots),
                       max_cycle, log=log)
    order = np.argsort(res.e.real, kind='stable')
    return res.e.real[order] + pb.const, res
