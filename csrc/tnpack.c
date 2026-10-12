/*
 * tnpack.c -- packed antisymmetric tuples of any rank (up to 6): expansion
 * into two groups of indices and the antisymmetrizing reduction of an array
 * given on two groups (zccsdt/cc/t4pack.py).
 *
 * A sorted k-tuple a_0 < a_1 < .. of range(n) has the position
 *
 *     rank = C(n, k) - 1 - sum_t C(n - 1 - a_t, k - t)
 *
 * among all of them in lexicographic order.  A split (g1 | g2) of a tuple
 * into k1 and k2 = k - k1 of its indices has the sign of the permutation
 * that brings (g1, g2) to sorted order, and
 *
 *     t[g1, g2] = sgn * t[sorted tuple].
 *
 * No tables of the size of the tuples are used: the kernels walk through
 * the tuples in order and compute the ranks of the groups of every split
 * from the binomial coefficients - once per run of the last index.
 *
 * Addresses are given by strides (in elements), so that the two groups and
 * an outer index o may sit in any order in the array on the two groups:
 *
 *     X[(r1 - b0) * s1 + (r2 - c0) * s2 + o * so],    R[o * sR + A]
 *
 * with r1, r2 the ranks of the groups, restricted to b0 <= r1 < b1 and
 * c0 <= r2 < c1 (the block of X that is there), and A the rank of the tuple.
 *
 * Every kernel exists for double (suffix _d) and double complex (suffix _z).
 */

#include <complex.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define KMAX 6
#define NSPLIT_MAX 20           /* C(6, 3) */
#define CHUNK 256

typedef struct {
    int n, k, k1, k2, nsplit;
    int p1[NSPLIT_MAX][KMAX], p2[NSPLIT_MAX][KMAX];
    double sgn[NSPLIT_MAX];
    int64_t *cb;                /* cb[j * (n + 1) + y] = C(y, j), j <= KMAX */
    int64_t ntuple;
} tn_split;

#define CB(sp, j, y) ((sp)->cb[(int64_t)(j) * ((sp)->n + 1) + (y)])

static int tn_setup(tn_split *sp, int n, int k1, int k2)
{
    int k = k1 + k2;
    if (k < 1 || k > KMAX || k1 < 0 || k2 < 0 || n < 0)
        return 1;
    sp->n = n; sp->k = k; sp->k1 = k1; sp->k2 = k2;
    sp->cb = (int64_t *)malloc(sizeof(int64_t) * (KMAX + 1) * (size_t)(n + 1));
    if (!sp->cb)
        return 1;
    for (int y = 0; y <= n; y++) {
        CB(sp, 0, y) = 1;
        for (int j = 1; j <= KMAX; j++)
            CB(sp, j, y) = (y == 0) ? 0 : CB(sp, j - 1, y - 1) + CB(sp, j, y - 1);
    }
    sp->ntuple = (n >= k) ? CB(sp, k, n) : 0;
    sp->nsplit = 0;
    for (int mask = 0; mask < (1 << k); mask++) {
        if (__builtin_popcount(mask) != k1)
            continue;
        int s = sp->nsplit++, i1 = 0, i2 = 0, inv = 0;
        for (int x = 0; x < k; x++) {
            if (mask >> x & 1) {
                sp->p1[s][i1++] = x;
                inv += i2;          /* members of the rest below x */
            } else {
                sp->p2[s][i2++] = x;
            }
        }
        sp->sgn[s] = (inv & 1) ? -1.0 : 1.0;
    }
    return 0;
}

static void tn_unrank(const tn_split *sp, int64_t r, int *a)
{
    int x = 0;
    for (int t = 0; t < sp->k; t++) {
        int left = sp->k - 1 - t;
        while (CB(sp, left, sp->n - 1 - x) <= r) {
            r -= CB(sp, left, sp->n - 1 - x);
            x++;
        }
        a[t] = x++;
    }
}

static void tn_next(const tn_split *sp, int *a)
{
    int k = sp->k, t = k - 1;
    while (t >= 0 && a[t] == sp->n - k + t)
        t--;
    if (t < 0)
        return;
    a[t]++;
    for (int u = t + 1; u < k; u++)
        a[u] = a[u - 1] + 1;
}

/* The tuples are walked through by their first k - 1 indices (a "prefix"):
 * with these fixed, the last index d runs over a_{k-2} < d < n and the
 * tuples have consecutive ranks.  Of every split one group has d, and its
 * rank grows by one with d; the other group does not change.  So within the
 * block of X (b0 <= r1 < b1, c0 <= r2 < c1) a split of a prefix is a run:
 * steps tlo <= t < thi of d, X from ``base`` in strides of s1 (d in the
 * first group) or s2.  The work of a kernel is then that of the elements
 * of the block, whatever part of all tuples the block is. */
typedef struct {
    int64_t tlo, thi, base, stride;
    double sgn;
} tn_run;

#define PCHUNK 32               /* prefixes per piece of work */

/* the number of prefixes: (k-1)-tuples of range(n) */
static int64_t tn_nprefix(const tn_split *sp)
{
    return (sp->k > 1) ? CB(sp, sp->k - 1, sp->n) : 1;
}

static void tn_prefix_unrank(const tn_split *sp, int64_t r, int *a)
{
    const int m = sp->k - 1;
    int x = 0;
    for (int t = 0; t < m; t++) {
        int left = m - 1 - t;
        while (CB(sp, left, sp->n - 1 - x) <= r) {
            r -= CB(sp, left, sp->n - 1 - x);
            x++;
        }
        a[t] = x++;
    }
}

static void tn_prefix_next(const tn_split *sp, int *a)
{
    const int m = sp->k - 1;
    int t = m - 1;
    while (t >= 0 && a[t] == sp->n - m + t)
        t--;
    if (t < 0)
        return;
    a[t]++;
    for (int u = t + 1; u < m; u++)
        a[u] = a[u - 1] + 1;
}

/* the runs of the splits for the prefix a; returns the length of the run of
 * tuples (0: none) and the rank of its first tuple */
static int64_t tn_runs(const tn_split *sp, const int *a, int64_t b0, int64_t b1,
                       int64_t c0, int64_t c1, int64_t s1, int64_t s2,
                       tn_run *run, int64_t *A0)
{
    const int n = sp->n, k = sp->k, k1 = sp->k1, k2 = sp->k2;
    int full[KMAX];
    for (int u = 0; u < k - 1; u++)
        full[u] = a[u];
    const int d0 = (k > 1) ? a[k - 2] + 1 : 0;
    const int64_t L = n - d0;
    if (L <= 0)
        return 0;
    full[k - 1] = d0;
    int64_t A = CB(sp, k, n) - 1;
    for (int u = 0; u < k; u++)
        A -= CB(sp, k - u, n - 1 - full[u]);
    *A0 = A;
    const int64_t top1 = CB(sp, k1, n) - 1, top2 = CB(sp, k2, n) - 1;
    for (int s = 0; s < sp->nsplit; s++) {
        int64_t r1 = top1, r2 = top2, lo, hi;
        for (int u = 0; u < k1; u++)
            r1 -= CB(sp, k1 - u, n - 1 - full[sp->p1[s][u]]);
        for (int u = 0; u < k2; u++)
            r2 -= CB(sp, k2 - u, n - 1 - full[sp->p2[s][u]]);
        run[s].sgn = sp->sgn[s];
        run[s].tlo = run[s].thi = 0;
        if (k1 > 0 && sp->p1[s][k1 - 1] == k - 1) {     /* d in the first group */
            if (r2 < c0 || r2 >= c1)
                continue;
            lo = (b0 > r1) ? b0 - r1 : 0;
            hi = (b1 - r1 < L) ? b1 - r1 : L;
            run[s].stride = s1;
        } else {
            if (r1 < b0 || r1 >= b1)
                continue;
            lo = (c0 > r2) ? c0 - r2 : 0;
            hi = (c1 - r2 < L) ? c1 - r2 : L;
            run[s].stride = s2;
        }
        if (hi <= lo)
            continue;
        run[s].tlo = lo;
        run[s].thi = hi;
        run[s].base = (r1 - b0) * s1 + (r2 - c0) * s2;
    }
    return L;
}

/* the runs of PCHUNK prefixes from the prefix of rank p0 on: run[q * ns + s],
 * len[q], first[q]; returns the number of prefixes */
static int tn_chunk(const tn_split *sp, int64_t p0, int64_t b0, int64_t b1,
                    int64_t c0, int64_t c1, int64_t s1, int64_t s2,
                    tn_run *run, int64_t *len, int64_t *first)
{
    int a[KMAX];
    const int64_t np = tn_nprefix(sp);
    int cnt = (np - p0 < PCHUNK) ? (int)(np - p0) : PCHUNK;
    tn_prefix_unrank(sp, p0, a);
    for (int q = 0; q < cnt; q++) {
        len[q] = tn_runs(sp, a, b0, b1, c0, c1, s1, s2, run + q * sp->nsplit, first + q);
        tn_prefix_next(sp, a);
    }
    return cnt;
}

/* out[A] = sum_t e[a_t]: the sums of orbital energies of the sorted k-tuples */
void zc_tn_tuple_sums(const double *e, int n, int k, double *out)
{
    tn_split sp;
    if (tn_setup(&sp, n, k, 0))
        return;
    const int64_t nA = sp.ntuple, nchunk = (nA + CHUNK - 1) / CHUNK;
#pragma omp parallel for schedule(static)
    for (int64_t ch = 0; ch < nchunk; ch++) {
        int a[KMAX];
        int64_t A0 = ch * CHUNK, cnt = (nA - A0 < CHUNK) ? nA - A0 : CHUNK;
        tn_unrank(&sp, A0, a);
        for (int64_t t = 0; t < cnt; t++) {
            double v = 0;
            for (int u = 0; u < k; u++)
                v += e[a[u]];
            out[A0 + t] = v;
            tn_next(&sp, a);
        }
    }
    free(sp.cb);
}

/* The splits (a | K) of the sorted (1 + k2)-tuples with the single index
 * first, for the k2-tuples K of the ranks c0 <= K < c1:
 *   idx[a * nc + (K - c0)] = rank of the sorted tuple K + a  (-1: a in K),
 *   sg [a * nc + (K - c0)] = sign of (a, K) = (-1)^(members of K below a).
 * For a fixed a the ranks grow with K: a kernel that walks through K reads
 * or writes the packed row in one direction. */
static void tn_single_table(int n, int k2, int64_t c0, int64_t c1,
                            int64_t *idx, int8_t *sg)
{
    tn_split sp, sk;
    if (tn_setup(&sp, n, 1, k2))
        return;
    if (tn_setup(&sk, n, k2, 0)) {
        free(sp.cb);
        return;
    }
    const int k = k2 + 1;
    const int64_t nc = c1 - c0, top = CB(&sp, k, n) - 1;
    const int64_t nchunk = (nc + CHUNK - 1) / CHUNK;
#pragma omp parallel for schedule(static)
    for (int64_t ch = 0; ch < nchunk; ch++) {
        int K[KMAX];
        int64_t pre[KMAX + 1], suf[KMAX + 1];
        int64_t j0 = ch * CHUNK, cnt = (nc - j0 < CHUNK) ? nc - j0 : CHUNK;
        tn_unrank(&sk, c0 + j0, K);
        for (int64_t j = j0; j < j0 + cnt; j++) {
            /* the tuple with a at position p: K[0..p-1], a, K[p..] */
            pre[0] = 0;
            for (int t = 0; t < k2; t++)
                pre[t + 1] = pre[t] + CB(&sp, k - t, n - 1 - K[t]);
            suf[k2] = 0;
            for (int t = k2 - 1; t >= 0; t--)
                suf[t] = suf[t + 1] + CB(&sp, k - t - 1, n - 1 - K[t]);
            int p = 0;
            for (int a = 0; a < n; a++) {
                if (p < k2 && K[p] == a) {
                    idx[a * nc + j] = -1;
                    sg[a * nc + j] = 0;
                    p++;
                    continue;
                }
                idx[a * nc + j] = top - pre[p] - CB(&sp, k - p, n - 1 - a) - suf[p];
                sg[a * nc + j] = (p & 1) ? -1 : 1;
            }
            tn_next(&sk, K);
        }
    }
    free(sp.cb);
    free(sk.cb);
}

void zc_tn_zero(void *p, int64_t nbytes)
{
    const int64_t blk = 1 << 20, nblk = (nbytes + blk - 1) / blk;
#pragma omp parallel for schedule(static)
    for (int64_t b = 0; b < nblk; b++) {
        int64_t lo = b * blk, len = (nbytes - lo < blk) ? nbytes - lo : blk;
        memset((char *)p + lo, 0, (size_t)len);
    }
}

#define ZC_RE_d(x) (x)
#define ZC_RE_z(x) creal(x)
#define ZC_CONJ_d(x) (x)
#define ZC_CONJ_z(x) conj(x)

#define DEFINE_KERNELS(T, SFX)                                                 \
                                                                               \
/* R[o * sR + A] += scale * sum_splits sgn * X[..] */                          \
void zc_tn_reduce_##SFX(T *R, int64_t nout, int64_t sR, int n, int k1, int k2, \
                        int64_t b0, int64_t b1, int64_t c0, int64_t c1,        \
                        const T *X, int64_t s1, int64_t s2, int64_t so,        \
                        double scale)                                          \
{                                                                              \
    tn_split sp;                                                               \
    if (tn_setup(&sp, n, k1, k2))                                              \
        return;                                                                \
    const int ns = sp.nsplit;                                                  \
    const int64_t np = tn_nprefix(&sp), nchunk = (np + PCHUNK - 1) / PCHUNK;   \
    _Pragma("omp parallel")                                                    \
    {                                                                          \
        tn_run run[PCHUNK * NSPLIT_MAX];                                       \
        int64_t len[PCHUNK], first[PCHUNK];                                    \
        _Pragma("omp for schedule(dynamic, 2)")                                \
        for (int64_t ch = 0; ch < nchunk; ch++) {                              \
            int cnt = tn_chunk(&sp, ch * PCHUNK, b0, b1, c0, c1, s1, s2, run, len, first); \
            for (int64_t o = 0; o < nout; o++) {                               \
                const T *x = X + o * so;                                       \
                for (int q = 0; q < cnt; q++) {                                \
                    if (len[q] <= 0) continue;                                 \
                    T *r = R + o * sR + first[q];                              \
                    for (int s = 0; s < ns; s++) {                             \
                        const tn_run *u = run + q * ns + s;                    \
                        if (u->thi <= u->tlo) continue;                        \
                        const double f = scale * u->sgn;                       \
                        const T *xb = x + u->base;                             \
                        const int64_t st = u->stride;                          \
                        if (st == 1) for (int64_t t = u->tlo; t < u->thi; t++) r[t] += f * xb[t]; \
                        else         for (int64_t t = u->tlo; t < u->thi; t++) r[t] += f * xb[t * st]; \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    free(sp.cb);                                                               \
}                                                                              \
                                                                               \
/* X[..] = scale * sgn * R[o * sR + A] for every split in the block of X      \
 * (elements with a repeated index are not written) */                         \
void zc_tn_expand_##SFX(const T *R, int64_t nout, int64_t sR, int n, int k1,   \
                        int k2, int64_t b0, int64_t b1, int64_t c0,            \
                        int64_t c1, T *X, int64_t s1, int64_t s2, int64_t so,  \
                        double scale)                                          \
{                                                                              \
    tn_split sp;                                                               \
    if (tn_setup(&sp, n, k1, k2))                                              \
        return;                                                                \
    const int ns = sp.nsplit;                                                  \
    const int64_t np = tn_nprefix(&sp), nchunk = (np + PCHUNK - 1) / PCHUNK;   \
    _Pragma("omp parallel")                                                    \
    {                                                                          \
        tn_run run[PCHUNK * NSPLIT_MAX];                                       \
        int64_t len[PCHUNK], first[PCHUNK];                                    \
        _Pragma("omp for schedule(dynamic, 2)")                                \
        for (int64_t ch = 0; ch < nchunk; ch++) {                              \
            int cnt = tn_chunk(&sp, ch * PCHUNK, b0, b1, c0, c1, s1, s2, run, len, first); \
            for (int64_t o = 0; o < nout; o++) {                               \
                T *x = X + o * so;                                             \
                for (int q = 0; q < cnt; q++) {                                \
                    if (len[q] <= 0) continue;                                 \
                    const T *r = R + o * sR + first[q];                        \
                    for (int s = 0; s < ns; s++) {                             \
                        const tn_run *u = run + q * ns + s;                    \
                        if (u->thi <= u->tlo) continue;                        \
                        const double f = scale * u->sgn;                       \
                        T *xb = x + u->base;                                   \
                        const int64_t st = u->stride;                          \
                        if (st == 1) for (int64_t t = u->tlo; t < u->thi; t++) xb[t] = f * r[t]; \
                        else         for (int64_t t = u->tlo; t < u->thi; t++) xb[t * st] = f * r[t]; \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    free(sp.cb);                                                               \
}                                                                              \
                                                                               \
/* The same with a list of entries o in the place of the outer index: each    \
 * has a row of the packed array, an offset in X and a sign,                   \
 *                                                                             \
 *   R[row[o] * ldR + A] += scale * osgn[o] * sum_splits sgn                   \
 *                          * X[xoff[o] + (r1 - b0) * s1 + (r2 - c0) * s2]     \
 *                                                                             \
 * - the reduction of a product on two groups of occupied and two groups of    \
 * virtual indices: an entry per split of an occupied tuple (its row), the     \
 * kernel does the virtual side.  Entries with the same row must follow each   \
 * other: the parallel loop is over (rows) x (pieces of the tuples), so that   \
 * no two threads write the same element. */                                   \
void zc_tn_reduce2_##SFX(T *R, int64_t ldR, int64_t nout, const int64_t *row,  \
                         const int64_t *xoff, const double *osgn, int n,       \
                         int k1, int k2, int64_t b0, int64_t b1, int64_t c0,   \
                         int64_t c1, const T *X, int64_t s1, int64_t s2,       \
                         double scale)                                         \
{                                                                              \
    tn_split sp;                                                               \
    if (nout <= 0 || tn_setup(&sp, n, k1, k2))                                 \
        return;                                                                \
    const int ns = sp.nsplit;                                                  \
    const int64_t np = tn_nprefix(&sp), nchunk = (np + PCHUNK - 1) / PCHUNK;   \
    int64_t *gs = (int64_t *)malloc(sizeof(int64_t) * (size_t)(nout + 1));     \
    int64_t ng = 0;                                                            \
    for (int64_t o = 0; o < nout; o++)                                         \
        if (o == 0 || row[o] != row[o - 1])                                    \
            gs[ng++] = o;                                                      \
    gs[ng] = nout;                                                             \
    _Pragma("omp parallel")                                                    \
    {                                                                          \
        tn_run run[PCHUNK * NSPLIT_MAX];                                       \
        int64_t len[PCHUNK], first[PCHUNK];                                    \
        _Pragma("omp for collapse(2) schedule(dynamic, 16)")                   \
        for (int64_t g = 0; g < ng; g++)                                       \
        for (int64_t ch = 0; ch < nchunk; ch++) {                              \
            int cnt = tn_chunk(&sp, ch * PCHUNK, b0, b1, c0, c1, s1, s2, run, len, first); \
            T *rr = R + row[gs[g]] * ldR;                                      \
            for (int64_t o = gs[g]; o < gs[g + 1]; o++) {                      \
                if (osgn[o] == 0) continue;                                    \
                const T *x = X + xoff[o];                                      \
                for (int q = 0; q < cnt; q++) {                                \
                    if (len[q] <= 0) continue;                                 \
                    T *r = rr + first[q];                                      \
                    for (int s = 0; s < ns; s++) {                             \
                        const tn_run *u = run + q * ns + s;                    \
                        if (u->thi <= u->tlo) continue;                        \
                        const double f = scale * osgn[o] * u->sgn;             \
                        const T *xb = x + u->base;                             \
                        const int64_t st = u->stride;                          \
                        if (st == 1) for (int64_t t = u->tlo; t < u->thi; t++) r[t] += f * xb[t]; \
                        else         for (int64_t t = u->tlo; t < u->thi; t++) r[t] += f * xb[t * st]; \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    free(gs);                                                                  \
    free(sp.cb);                                                               \
}                                                                              \
                                                                               \
/* X[xoff[o] + ..] = scale * osgn[o] * sgn * M[row[o] * ldM + A]: packed rows  \
 * on two groups, each at its place in X (not written: elements with a         \
 * repeated index, entries with osgn 0) */                                     \
void zc_tn_expand2_##SFX(const T *M, int64_t ldM, int64_t nout,                \
                         const int64_t *row, const int64_t *xoff,              \
                         const double *osgn, int n, int k1, int k2,            \
                         int64_t b0, int64_t b1, int64_t c0, int64_t c1,       \
                         T *X, int64_t s1, int64_t s2, double scale)           \
{                                                                              \
    tn_split sp;                                                               \
    if (nout <= 0 || tn_setup(&sp, n, k1, k2))                                 \
        return;                                                                \
    const int ns = sp.nsplit;                                                  \
    const int64_t np = tn_nprefix(&sp), nchunk = (np + PCHUNK - 1) / PCHUNK;   \
    _Pragma("omp parallel")                                                    \
    {                                                                          \
        tn_run run[PCHUNK * NSPLIT_MAX];                                       \
        int64_t len[PCHUNK], first[PCHUNK];                                    \
        _Pragma("omp for collapse(2) schedule(dynamic, 16)")                   \
        for (int64_t o = 0; o < nout; o++)                                     \
        for (int64_t ch = 0; ch < nchunk; ch++) {                              \
            if (osgn[o] == 0) continue;                                        \
            int cnt = tn_chunk(&sp, ch * PCHUNK, b0, b1, c0, c1, s1, s2, run, len, first); \
            const T *rr = M + row[o] * ldM;                                    \
            T *x = X + xoff[o];                                                \
            for (int q = 0; q < cnt; q++) {                                    \
                if (len[q] <= 0) continue;                                     \
                const T *r = rr + first[q];                                    \
                for (int s = 0; s < ns; s++) {                                 \
                    const tn_run *u = run + q * ns + s;                        \
                    if (u->thi <= u->tlo) continue;                            \
                    const double f = scale * osgn[o] * u->sgn;                 \
                    T *xb = x + u->base;                                       \
                    const int64_t st = u->stride;                              \
                    if (st == 1) for (int64_t t = u->tlo; t < u->thi; t++) xb[t] = f * r[t]; \
                    else         for (int64_t t = u->tlo; t < u->thi; t++) xb[t * st] = f * r[t]; \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    free(sp.cb);                                                               \
}                                                                              \
                                                                               \
/* The splits (a | K) alone, for the k2-tuples c0 <= K < c1 (a block of       \
 * columns of a product): walks through the block, not through all tuples.     \
 *   X[xoff[o] + a * s1 + (K - c0)] = scale * osgn[o] * sgn * M[row[o] * ldM + A] \
 * for every entry o, A the sorted tuple K + a. */                             \
void zc_tn_expand1_##SFX(const T *M, int64_t ldM, int64_t nout,                \
                         const int64_t *row, const int64_t *xoff,              \
                         const double *osgn, int n, int k2, int64_t c0,        \
                         int64_t c1, T *X, int64_t s1, double scale)           \
{                                                                              \
    const int64_t nc = c1 - c0;                                                \
    if (nout <= 0 || nc <= 0)                                                  \
        return;                                                                \
    int64_t *idx = (int64_t *)malloc(sizeof(int64_t) * (size_t)(n * nc));      \
    int8_t *sg = (int8_t *)malloc((size_t)(n * nc));                           \
    tn_single_table(n, k2, c0, c1, idx, sg);                                   \
    _Pragma("omp parallel for collapse(2) schedule(dynamic, 8)")               \
    for (int64_t o = 0; o < nout; o++)                                         \
    for (int a = 0; a < n; a++) {                                              \
        if (osgn[o] == 0)                                                      \
            continue;                                                          \
        const T *r = M + row[o] * ldM;                                         \
        T *x = X + xoff[o] + a * s1;                                           \
        const int64_t *ia = idx + a * nc;                                      \
        const int8_t *sa = sg + a * nc;                                        \
        const double f = scale * osgn[o];                                      \
        for (int64_t j = 0; j < nc; j++)                                       \
            if (ia[j] >= 0)                                                    \
                x[j] = (f * sa[j]) * r[ia[j]];                                 \
    }                                                                          \
    free(idx);                                                                 \
    free(sg);                                                                  \
}                                                                              \
                                                                               \
/* R[row * ldR + A] += scale * sgn * sum over the entries o of the row of      \
 * osgn[o] * X[xoff[o] + a * s1 + (K - c0)]: the slabs of the splits of an     \
 * occupied tuple are summed first (in order), then one reduction of the sum.  \
 * Entries with the same row must follow each other; a thread has a row. */    \
void zc_tn_reduce1_##SFX(T *R, int64_t ldR, int64_t nout, const int64_t *row,  \
                         const int64_t *xoff, const double *osgn, int n,       \
                         int k2, int64_t c0, int64_t c1, const T *X,           \
                         int64_t s1, double scale)                             \
{                                                                              \
    const int64_t nc = c1 - c0;                                                \
    if (nout <= 0 || nc <= 0)                                                  \
        return;                                                                \
    int64_t *idx = (int64_t *)malloc(sizeof(int64_t) * (size_t)(n * nc));      \
    int8_t *sg = (int8_t *)malloc((size_t)(n * nc));                           \
    tn_single_table(n, k2, c0, c1, idx, sg);                                   \
    int64_t *gs = (int64_t *)malloc(sizeof(int64_t) * (size_t)(nout + 1));     \
    int64_t ng = 0;                                                            \
    for (int64_t o = 0; o < nout; o++)                                         \
        if (o == 0 || row[o] != row[o - 1])                                    \
            gs[ng++] = o;                                                      \
    gs[ng] = nout;                                                             \
    _Pragma("omp parallel")                                                    \
    {                                                                          \
        T *S = (T *)malloc(sizeof(T) * (size_t)nc);                            \
        _Pragma("omp for schedule(dynamic, 2)")                                \
        for (int64_t g = 0; g < ng; g++) {                                     \
            T *r = R + row[gs[g]] * ldR;                                       \
            for (int a = 0; a < n; a++) {                                      \
                int have = 0;                                                  \
                for (int64_t o = gs[g]; o < gs[g + 1]; o++) {                  \
                    const double f = osgn[o];                                  \
                    if (f == 0)                                                \
                        continue;                                              \
                    const T *x = X + xoff[o] + a * s1;                         \
                    if (!have) for (int64_t j = 0; j < nc; j++) S[j] = f * x[j]; \
                    else       for (int64_t j = 0; j < nc; j++) S[j] += f * x[j]; \
                    have = 1;                                                  \
                }                                                              \
                if (!have)                                                     \
                    continue;                                                  \
                const int64_t *ia = idx + a * nc;                              \
                const int8_t *sa = sg + a * nc;                                \
                for (int64_t j = 0; j < nc; j++)                               \
                    if (ia[j] >= 0)                                            \
                        r[ia[j]] += (scale * sa[j]) * S[j];                    \
            }                                                                  \
        }                                                                      \
        free(S);                                                               \
    }                                                                          \
    free(gs);                                                                  \
    free(idx);                                                                 \
    free(sg);                                                                  \
}                                                                              \
                                                                               \
/* R[tgt[e] * ldR + c] += sgn[e] * X[src[e] + c], c < ncol: rows of X added    \
 * to rows of R - the reduction over the splits of the occupied tuples alone,  \
 * one entry per split (threads take blocks of columns) */                     \
void zc_tn_axpy_rows_##SFX(T *R, int64_t ldR, int64_t nent,                    \
                           const int64_t *tgt, const int64_t *src,             \
                           const double *sgn, const T *X, int64_t ncol)        \
{                                                                              \
    const int64_t blk = 2048, nblk = (ncol + blk - 1) / blk;                   \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t b = 0; b < nblk; b++) {                                       \
        int64_t lo = b * blk, hi = (lo + blk < ncol) ? lo + blk : ncol;        \
        for (int64_t e = 0; e < nent; e++) {                                   \
            const double sg = sgn[e];                                          \
            if (sg == 0)                                                       \
                continue;                                                      \
            T *r = R + tgt[e] * ldR;                                           \
            const T *x = X + src[e];                                           \
            for (int64_t c = lo; c < hi; c++)                                  \
                r[c] += sg * x[c];                                             \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
/* out[r, :] = sgn[r] * M[rows[r], c0:c1]  (rows of a packed block with their  \
 * signs; sgn 0: zeros) */                                                     \
void zc_tn_rows_##SFX(const T *M, int64_t ld, const int64_t *rows,             \
                      const int8_t *sgn, int64_t nrows, int64_t c0,            \
                      int64_t c1, T *out, int64_t ldo)                         \
{                                                                              \
    const int64_t blk = 8192, nc = c1 - c0, nblk = (nc + blk - 1) / blk;       \
    _Pragma("omp parallel for collapse(2) schedule(static)")                   \
    for (int64_t r = 0; r < nrows; r++)                                        \
    for (int64_t b = 0; b < nblk; b++) {                                       \
        int64_t lo = b * blk, hi = (lo + blk < nc) ? lo + blk : nc;            \
        T *o = out + r * ldo;                                                  \
        if (sgn[r] == 0) {                                                     \
            memset(o + lo, 0, sizeof(T) * (size_t)(hi - lo));                  \
            continue;                                                          \
        }                                                                      \
        const T *m = M + rows[r] * ld + c0;                                    \
        if (sgn[r] > 0) for (int64_t c = lo; c < hi; c++) o[c] = m[c];         \
        else            for (int64_t c = lo; c < hi; c++) o[c] = -m[c];        \
    }                                                                          \
}                                                                              \
                                                                               \
/* e[0] += sum Re(conj(z) (r1 + r2)) / (eo - ev),                              \
 * e[1] += sum Re(conj(r1) (r1 + r2)) / (eo - ev):  the two energies of a      \
 * perturbative correction whose left-hand side is z or z + r1 */              \
void zc_tn_energy2_##SFX(const T *z, const T *r1, const T *r2,                 \
                         const double *ev, double eo, int64_t n, double *e)    \
{                                                                              \
    double e0 = 0, e1 = 0;                                                     \
    _Pragma("omp parallel for schedule(static) reduction(+:e0, e1)")           \
    for (int64_t A = 0; A < n; A++) {                                          \
        T r = (r1[A] + r2[A]) / (eo - ev[A]);                                  \
        e0 += ZC_RE_##SFX(ZC_CONJ_##SFX(z[A]) * r);                            \
        e1 += ZC_RE_##SFX(ZC_CONJ_##SFX(r1[A]) * r);                           \
    }                                                                          \
    e[0] += e0;                                                                \
    e[1] += e1;                                                                \
}

DEFINE_KERNELS(double, d)
DEFINE_KERNELS(double complex, z)

/*
 * (Q) in symmetry sectors (zccsdt/cc/quadruples_sym.py).  Every virtual
 * spinor has an integer label kv that adds (modulo `modulus`, or as it is
 * for 0); an occupied quadruple of total label M has r and z only on the
 * virtual quadruples a<b<c<d of the same total.  The products are given in
 * blocks: those on two pairs, X[(ab), (cd)], one block per label q of (ab)
 * at off22[q - qmin] with nc22 columns, the pairs at posQ inside their
 * sector; those on one index and a triple, X[a, (bcd)], one block per label
 * of a at off13[lid] with nc13 columns, a at posV and the triple at pos3.
 * The kernel walks through the quadruples of the sector, sums the splits of
 * each with their signs and adds the two energies of zc_tn_energy2:
 *
 *     r1 = b1 (22) + c1 (13),  r2 = b2 (22) + c2 (13),  z = b3 (22).
 */
#define DEFINE_QSEC(T, SFX)                                                    \
void zc_q_sector_energy_##SFX(int n, const int *kv, int modulus, int M,        \
        const double *ev, double eo, int kmin, int nk, const int *lid_of,      \
        const int *lstart, const int *lmem, int qmin, int nq,                  \
        const int64_t *off22, const int64_t *nc22, const int64_t *off13,       \
        const int64_t *nc13, const int32_t *posV, const int32_t *posQ,         \
        const int32_t *pos3, const T *b1, const T *b2, const T *b3,            \
        const T *c1, const T *c2, double *e)                                   \
{                                                                              \
    const int64_t nQ = (int64_t)n * (n - 1) / 2;                               \
    const int64_t nA3 = nQ * (n - 2) / 3;                                      \
    double e0 = 0, e1 = 0;                                                     \
    _Pragma("omp parallel for schedule(dynamic, 1) reduction(+:e0, e1)")       \
    for (int a = 0; a < n - 3; a++)                                            \
    for (int b = a + 1; b < n - 2; b++)                                        \
    for (int c = b + 1; c < n - 1; c++) {                                      \
        int kd = M - kv[a] - kv[b] - kv[c];                                    \
        if (modulus) kd = ((kd % modulus) + modulus) % modulus;                \
        if (kd < kmin || kd >= kmin + nk || lid_of[kd - kmin] < 0) continue;   \
        const int l = lid_of[kd - kmin];                                       \
        int lo = lstart[l], hi = lstart[l + 1];                                \
        while (lo < hi) {               /* the first member above c */         \
            int mid = (lo + hi) / 2;                                           \
            if (lmem[mid] > c) hi = mid; else lo = mid + 1;                    \
        }                                                                      \
        for (int t = lo; t < lstart[l + 1]; t++) {                             \
            const int v[4] = {a, b, c, lmem[t]};                               \
            static const int g22[6][4] = {{0,1,2,3}, {0,2,1,3}, {0,3,1,2},     \
                                          {1,2,0,3}, {1,3,0,2}, {2,3,0,1}};    \
            static const double s22[6] = {1, -1, 1, 1, -1, 1};                 \
            T r1 = 0, r2 = 0, z = 0;                                           \
            for (int s = 0; s < 6; s++) {                                      \
                const int p = v[g22[s][0]], q = v[g22[s][1]];                  \
                const int r = v[g22[s][2]], u = v[g22[s][3]];                  \
                int kq = kv[p] + kv[q];                                        \
                if (modulus) kq = ((kq % modulus) + modulus) % modulus;        \
                if (kq < qmin || kq >= qmin + nq || off22[kq - qmin] < 0)      \
                    continue;                                                  \
                const int64_t rk1 = nQ - 1 - (int64_t)(n - 1 - p) * (n - 2 - p) / 2 - (n - 1 - q); \
                const int64_t rk2 = nQ - 1 - (int64_t)(n - 1 - r) * (n - 2 - r) / 2 - (n - 1 - u); \
                const int64_t x = off22[kq - qmin] + posQ[rk1] * nc22[kq - qmin] + posQ[rk2];      \
                r1 += s22[s] * b1[x];                                          \
                r2 += s22[s] * b2[x];                                          \
                z += s22[s] * b3[x];                                           \
            }                                                                  \
            for (int s = 0; s < 4; s++) {                                      \
                const int p = v[s];                                            \
                const int q = v[s == 0], r = v[s <= 1 ? 2 : 1], u = v[s == 3 ? 2 : 3];             \
                const int lp = lid_of[kv[p] - kmin];                           \
                if (off13[lp] < 0) continue;                                   \
                const int64_t y1 = n - 1 - q, y2 = n - 1 - r;                  \
                const int64_t rk = nA3 - 1 - y1 * (y1 - 1) * (y1 - 2) / 6 - y2 * (y2 - 1) / 2 - (n - 1 - u); \
                const int64_t x = off13[lp] + posV[p] * nc13[lp] + pos3[rk];   \
                const double sg = (s & 1) ? -1.0 : 1.0;                        \
                r1 += sg * c1[x];                                              \
                r2 += sg * c2[x];                                              \
            }                                                                  \
            T rr = (r1 + r2) / (eo - ev[v[0]] - ev[v[1]] - ev[v[2]] - ev[v[3]]);                   \
            e0 += ZC_RE_##SFX(ZC_CONJ_##SFX(z) * rr);                          \
            e1 += ZC_RE_##SFX(ZC_CONJ_##SFX(r1) * rr);                         \
        }                                                                      \
    }                                                                          \
    e[0] += e0;                                                                \
    e[1] += e1;                                                                \
}

DEFINE_QSEC(double, d)
DEFINE_QSEC(double complex, z)

/*
 * Two more kernels of (Q) in symmetry sectors.
 *
 * zc_rows_gemm:  X[a, c] = sum_r L[r, a] sg[r] M[rows[r], cols[c]] - a
 * product with rows and columns picked from a packed block, which is not
 * copied (L: [nr, na], X: [na, nc]).
 *
 * zc_xv_sector:  out[e, t] = 1/2 sum over the splits (pq | r) of the sorted
 * triples t = (p<q<r) of one label of sgn * m3[(pq), e, r] - the
 * intermediate of T2 with <vv||vv> for the e of one label, from the blocks
 * m3 of the labels of (pq): block at off[q - qmin], [pairs at posQ, e, nd
 * virtual spinors of the label that is left, at posV].
 */
#define DEFINE_QSEC2(T, SFX)                                                   \
void zc_rows_gemm_##SFX(const T *M, int64_t ld, const int64_t *rows,           \
        const double *sg, int64_t nr, const int64_t *cols, int64_t nc,         \
        const T *L, int64_t na, T *X)                                          \
{                                                                              \
    _Pragma("omp parallel")                                                    \
    {                                                                          \
        T *tmp = malloc(sizeof(T) * (size_t)(nr > 0 ? nr : 1));                \
        _Pragma("omp for schedule(static)")                                    \
        for (int64_t c = 0; c < nc; c++) {                                     \
            for (int64_t r = 0; r < nr; r++)                                   \
                tmp[r] = sg[r] * M[rows[r] * ld + cols[c]];                    \
            for (int64_t a = 0; a < na; a++) {                                 \
                T x = 0;                                                       \
                for (int64_t r = 0; r < nr; r++) x += L[r * na + a] * tmp[r];  \
                X[a * nc + c] = x;                                             \
            }                                                                  \
        }                                                                      \
        free(tmp);                                                             \
    }                                                                          \
}                                                                              \
                                                                               \
void zc_xv_sector_##SFX(int n, const int *kv, int modulus, const int32_t *tri, \
        int64_t nt, int ne, int qmin, int nq, const int64_t *off,              \
        const int64_t *nd, const int32_t *posV, const int32_t *posQ,           \
        const T *m3, T *out)                                                   \
{                                                                              \
    const int64_t nQ = (int64_t)n * (n - 1) / 2;                               \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t t = 0; t < nt; t++) {                                         \
        const int v[3] = {tri[3 * t], tri[3 * t + 1], tri[3 * t + 2]};         \
        static const int g[3][3] = {{0, 1, 2}, {0, 2, 1}, {1, 2, 0}};          \
        static const double sg[3] = {0.5, -0.5, 0.5};                          \
        for (int e = 0; e < ne; e++) out[e * nt + t] = 0;                      \
        for (int s = 0; s < 3; s++) {                                          \
            const int p = v[g[s][0]], q = v[g[s][1]], r = v[g[s][2]];          \
            int kq = kv[p] + kv[q];                                            \
            if (modulus) kq = ((kq % modulus) + modulus) % modulus;            \
            if (kq < qmin || kq >= qmin + nq || off[kq - qmin] < 0) continue;  \
            const int64_t rk = nQ - 1 - (int64_t)(n - 1 - p) * (n - 2 - p) / 2 - (n - 1 - q);      \
            const int64_t d = nd[kq - qmin];                                   \
            const T *m = m3 + off[kq - qmin] + posQ[rk] * ne * d + posV[r];    \
            for (int e = 0; e < ne; e++) out[e * nt + t] += sg[s] * m[e * d];  \
        }                                                                      \
    }                                                                          \
}

DEFINE_QSEC2(double, d)
DEFINE_QSEC2(double complex, z)

/*
 * The first derivatives of (Q) in symmetry sectors
 * (zccsdt/cc/quadruples_grad_sym.py).
 *
 * zc_qg_sector_vectors:  zc_q_sector_energy backwards.  For every virtual
 * quadruple of the sector, with D = eo - ev,
 *
 *     y = (r1 + r2) / D,   w = (z + r1) / D,   c = Re(conj(w) y),
 *
 * and the left-hand quadruples are written to the places of the blocks that
 * the splits of the quadruple were read from, with their signs (every place
 * belongs to one split of one quadruple; the others - groups that share an
 * index - are not touched: the caller sets the blocks to zero):
 *
 *     e22y = conj(y),  e22w = conj(w)                       on two pairs
 *     e13w = conj(w),  e13y = y,  e13s = conj(y) + conj(w)  on an index and a triple
 *
 * e[0] += sum c D (E(Q) of the sector), e[1] += sum c.
 *
 * zc_tn_scatter_cols:  R[r, idx[j]] += sg[j] X[r, j] for nrow rows (threads
 * take rows; idx may repeat).
 * zc_rows_cols_get:  out[r, c] = sg[r] M[rows[r], cols[c]].
 * zc_rows_cols_add:  R[rows[r], cols[c]] += sg[r] X[r, c] (rows may repeat:
 * threads take columns).
 * zc_xv_sector_adj:  zc_xv_sector backwards: m3[(pq), e, r] = sgn / 2 *
 * out[e, t] for the splits of the triples t (the other places untouched).
 */
#define DEFINE_QGSEC(T, SFX)                                                   \
void zc_qg_sector_vectors_##SFX(int n, const int *kv, int modulus, int M,      \
        const double *ev, double eo, int kmin, int nk, const int *lid_of,      \
        const int *lstart, const int *lmem, int qmin, int nq,                  \
        const int64_t *off22, const int64_t *nc22, const int64_t *off13,       \
        const int64_t *nc13, const int32_t *posV, const int32_t *posQ,         \
        const int32_t *pos3, const T *b1, const T *b2, const T *b3,            \
        const T *c1, const T *c2, T *e22y, T *e22w, T *e13w, T *e13y,          \
        T *e13s, double *e)                                                    \
{                                                                              \
    const int64_t nQ = (int64_t)n * (n - 1) / 2;                               \
    const int64_t nA3 = nQ * (n - 2) / 3;                                      \
    double e0 = 0, e1 = 0;                                                     \
    _Pragma("omp parallel for schedule(dynamic, 1) reduction(+:e0, e1)")       \
    for (int a = 0; a < n - 3; a++)                                            \
    for (int b = a + 1; b < n - 2; b++)                                        \
    for (int c = b + 1; c < n - 1; c++) {                                      \
        int kd = M - kv[a] - kv[b] - kv[c];                                    \
        if (modulus) kd = ((kd % modulus) + modulus) % modulus;                \
        if (kd < kmin || kd >= kmin + nk || lid_of[kd - kmin] < 0) continue;   \
        const int l = lid_of[kd - kmin];                                       \
        int lo = lstart[l], hi = lstart[l + 1];                                \
        while (lo < hi) {               /* the first member above c */         \
            int mid = (lo + hi) / 2;                                           \
            if (lmem[mid] > c) hi = mid; else lo = mid + 1;                    \
        }                                                                      \
        for (int t = lo; t < lstart[l + 1]; t++) {                             \
            const int v[4] = {a, b, c, lmem[t]};                               \
            static const int g22[6][4] = {{0,1,2,3}, {0,2,1,3}, {0,3,1,2},     \
                                          {1,2,0,3}, {1,3,0,2}, {2,3,0,1}};    \
            static const double s22[6] = {1, -1, 1, 1, -1, 1};                 \
            int64_t x22[6], x13[4];                                            \
            T r1 = 0, r2 = 0, z = 0;                                           \
            for (int s = 0; s < 6; s++) {                                      \
                const int p = v[g22[s][0]], q = v[g22[s][1]];                  \
                const int r = v[g22[s][2]], u = v[g22[s][3]];                  \
                int kq = kv[p] + kv[q];                                        \
                x22[s] = -1;                                                   \
                if (modulus) kq = ((kq % modulus) + modulus) % modulus;        \
                if (kq < qmin || kq >= qmin + nq || off22[kq - qmin] < 0)      \
                    continue;                                                  \
                const int64_t rk1 = nQ - 1 - (int64_t)(n - 1 - p) * (n - 2 - p) / 2 - (n - 1 - q); \
                const int64_t rk2 = nQ - 1 - (int64_t)(n - 1 - r) * (n - 2 - r) / 2 - (n - 1 - u); \
                const int64_t x = off22[kq - qmin] + posQ[rk1] * nc22[kq - qmin] + posQ[rk2];      \
                x22[s] = x;                                                    \
                r1 += s22[s] * b1[x];                                          \
                r2 += s22[s] * b2[x];                                          \
                z += s22[s] * b3[x];                                           \
            }                                                                  \
            for (int s = 0; s < 4; s++) {                                      \
                const int p = v[s];                                            \
                const int q = v[s == 0], r = v[s <= 1 ? 2 : 1], u = v[s == 3 ? 2 : 3];             \
                const int lp = lid_of[kv[p] - kmin];                           \
                x13[s] = -1;                                                   \
                if (off13[lp] < 0) continue;                                   \
                const int64_t y1 = n - 1 - q, y2 = n - 1 - r;                  \
                const int64_t rk = nA3 - 1 - y1 * (y1 - 1) * (y1 - 2) / 6 - y2 * (y2 - 1) / 2 - (n - 1 - u); \
                const int64_t x = off13[lp] + posV[p] * nc13[lp] + pos3[rk];   \
                const double sg = (s & 1) ? -1.0 : 1.0;                        \
                x13[s] = x;                                                    \
                r1 += sg * c1[x];                                              \
                r2 += sg * c2[x];                                              \
            }                                                                  \
            const double D = eo - ev[v[0]] - ev[v[1]] - ev[v[2]] - ev[v[3]];   \
            const T y = (r1 + r2) / D, w = (z + r1) / D;                       \
            const T cy = ZC_CONJ_##SFX(y), cw = ZC_CONJ_##SFX(w);              \
            const double cc = ZC_RE_##SFX(cw * y);                             \
            e0 += cc * D;                                                      \
            e1 += cc;                                                          \
            for (int s = 0; s < 6; s++) {                                      \
                if (x22[s] < 0) continue;                                      \
                e22y[x22[s]] = s22[s] * cy;                                    \
                e22w[x22[s]] = s22[s] * cw;                                    \
            }                                                                  \
            for (int s = 0; s < 4; s++) {                                      \
                if (x13[s] < 0) continue;                                      \
                const double sg = (s & 1) ? -1.0 : 1.0;                        \
                e13w[x13[s]] = sg * cw;                                        \
                e13y[x13[s]] = sg * y;                                         \
                e13s[x13[s]] = sg * (cy + cw);                                 \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    e[0] += e0;                                                                \
    e[1] += e1;                                                                \
}                                                                              \
                                                                               \
void zc_tn_scatter_cols_##SFX(T *R, int64_t ldR, int64_t nrow,                 \
        const int64_t *idx, const double *sg, int64_t n, const T *X)           \
{                                                                              \
    _Pragma("omp parallel for schedule(static) if(nrow > 1)")                  \
    for (int64_t r = 0; r < nrow; r++) {                                       \
        T *o = R + r * ldR;                                                    \
        const T *x = X + r * n;                                                \
        for (int64_t j = 0; j < n; j++)                                        \
            if (sg[j] != 0) o[idx[j]] += sg[j] * x[j];                         \
    }                                                                          \
}                                                                              \
                                                                               \
void zc_rows_cols_get_##SFX(const T *M, int64_t ld, const int64_t *rows,       \
        const double *sg, int64_t nr, const int64_t *cols, int64_t nc, T *out) \
{                                                                              \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t r = 0; r < nr; r++) {                                         \
        const T *m = M + rows[r] * ld;                                         \
        T *o = out + r * nc;                                                   \
        for (int64_t c = 0; c < nc; c++) o[c] = sg[r] * m[cols[c]];            \
    }                                                                          \
}                                                                              \
                                                                               \
void zc_rows_cols_add_##SFX(T *R, int64_t ld, const int64_t *rows,             \
        const double *sg, int64_t nr, const int64_t *cols, int64_t nc,         \
        const T *X)                                                            \
{                                                                              \
    const int64_t blk = 1024, nblk = (nc + blk - 1) / blk;                     \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t b = 0; b < nblk; b++) {                                       \
        int64_t lo = b * blk, hi = (lo + blk < nc) ? lo + blk : nc;            \
        for (int64_t r = 0; r < nr; r++) {                                     \
            T *o = R + rows[r] * ld;                                           \
            const T *x = X + r * nc;                                           \
            for (int64_t c = lo; c < hi; c++) o[cols[c]] += sg[r] * x[c];      \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
void zc_xv_sector_adj_##SFX(int n, const int *kv, int modulus,                 \
        const int32_t *tri, int64_t nt, int ne, int qmin, int nq,              \
        const int64_t *off, const int64_t *nd, const int32_t *posV,            \
        const int32_t *posQ, T *m3, const T *out)                              \
{                                                                              \
    const int64_t nQ = (int64_t)n * (n - 1) / 2;                               \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t t = 0; t < nt; t++) {                                         \
        const int v[3] = {tri[3 * t], tri[3 * t + 1], tri[3 * t + 2]};         \
        static const int g[3][3] = {{0, 1, 2}, {0, 2, 1}, {1, 2, 0}};          \
        static const double sg[3] = {0.5, -0.5, 0.5};                          \
        for (int s = 0; s < 3; s++) {                                          \
            const int p = v[g[s][0]], q = v[g[s][1]], r = v[g[s][2]];          \
            int kq = kv[p] + kv[q];                                            \
            if (modulus) kq = ((kq % modulus) + modulus) % modulus;            \
            if (kq < qmin || kq >= qmin + nq || off[kq - qmin] < 0) continue;  \
            const int64_t rk = nQ - 1 - (int64_t)(n - 1 - p) * (n - 2 - p) / 2 - (n - 1 - q);      \
            const int64_t d = nd[kq - qmin];                                   \
            T *m = m3 + off[kq - qmin] + posQ[rk] * ne * d + posV[r];          \
            for (int e = 0; e < ne; e++) m[e * d] = sg[s] * out[e * nt + t];   \
        }                                                                      \
    }                                                                          \
}

DEFINE_QGSEC(double, d)
DEFINE_QGSEC(double complex, z)

/*
 * CC4 without stored T4 in symmetry sectors (zccsdt/cc/cc4fly.py).
 *
 * zc_cc4_sector_row:  the T4 row of one occupied quadruple in the places of
 * the blocks - zc_qg_sector_vectors for one right-hand side and nothing
 * else.  For every virtual quadruple of the sector
 *
 *     t = (the splits of b22 + the splits of c13) / (eo - ev),
 *
 * with the signs of the splits as in zc_q_sector_energy, written with these
 * signs to the places that the splits were read from: e22 on two pairs, e13
 * on an index and a triple (the other places - groups that share an index -
 * are not touched: the caller sets the blocks to zero).
 */
#define DEFINE_CC4SEC(T, SFX)                                                  \
void zc_cc4_sector_row_##SFX(int n, const int *kv, int modulus, int M,         \
        const double *ev, double eo, int kmin, int nk, const int *lid_of,      \
        const int *lstart, const int *lmem, int qmin, int nq,                  \
        const int64_t *off22, const int64_t *nc22, const int64_t *off13,       \
        const int64_t *nc13, const int32_t *posV, const int32_t *posQ,         \
        const int32_t *pos3, const T *b22, const T *c13, T *e22, T *e13)       \
{                                                                              \
    const int64_t nQ = (int64_t)n * (n - 1) / 2;                               \
    const int64_t nA3 = nQ * (n - 2) / 3;                                      \
    _Pragma("omp parallel for schedule(dynamic, 1)")                           \
    for (int a = 0; a < n - 3; a++)                                            \
    for (int b = a + 1; b < n - 2; b++)                                        \
    for (int c = b + 1; c < n - 1; c++) {                                      \
        int kd = M - kv[a] - kv[b] - kv[c];                                    \
        if (modulus) kd = ((kd % modulus) + modulus) % modulus;                \
        if (kd < kmin || kd >= kmin + nk || lid_of[kd - kmin] < 0) continue;   \
        const int l = lid_of[kd - kmin];                                       \
        int lo = lstart[l], hi = lstart[l + 1];                                \
        while (lo < hi) {               /* the first member above c */         \
            int mid = (lo + hi) / 2;                                           \
            if (lmem[mid] > c) hi = mid; else lo = mid + 1;                    \
        }                                                                      \
        for (int t = lo; t < lstart[l + 1]; t++) {                             \
            const int v[4] = {a, b, c, lmem[t]};                               \
            static const int g22[6][4] = {{0,1,2,3}, {0,2,1,3}, {0,3,1,2},     \
                                          {1,2,0,3}, {1,3,0,2}, {2,3,0,1}};    \
            static const double s22[6] = {1, -1, 1, 1, -1, 1};                 \
            int64_t x22[6], x13[4];                                            \
            T r = 0;                                                           \
            for (int s = 0; s < 6; s++) {                                      \
                const int p = v[g22[s][0]], q = v[g22[s][1]];                  \
                const int w = v[g22[s][2]], u = v[g22[s][3]];                  \
                int kq = kv[p] + kv[q];                                        \
                x22[s] = -1;                                                   \
                if (modulus) kq = ((kq % modulus) + modulus) % modulus;        \
                if (kq < qmin || kq >= qmin + nq || off22[kq - qmin] < 0)      \
                    continue;                                                  \
                const int64_t rk1 = nQ - 1 - (int64_t)(n - 1 - p) * (n - 2 - p) / 2 - (n - 1 - q); \
                const int64_t rk2 = nQ - 1 - (int64_t)(n - 1 - w) * (n - 2 - w) / 2 - (n - 1 - u); \
                const int64_t x = off22[kq - qmin] + posQ[rk1] * nc22[kq - qmin] + posQ[rk2];      \
                x22[s] = x;                                                    \
                r += s22[s] * b22[x];                                          \
            }                                                                  \
            for (int s = 0; s < 4; s++) {                                      \
                const int p = v[s];                                            \
                const int q = v[s == 0], w = v[s <= 1 ? 2 : 1], u = v[s == 3 ? 2 : 3];             \
                const int lp = lid_of[kv[p] - kmin];                           \
                x13[s] = -1;                                                   \
                if (off13[lp] < 0) continue;                                   \
                const int64_t y1 = n - 1 - q, y2 = n - 1 - w;                  \
                const int64_t rk = nA3 - 1 - y1 * (y1 - 1) * (y1 - 2) / 6 - y2 * (y2 - 1) / 2 - (n - 1 - u); \
                const int64_t x = off13[lp] + posV[p] * nc13[lp] + pos3[rk];   \
                x13[s] = x;                                                    \
                r += ((s & 1) ? -1.0 : 1.0) * c13[x];                          \
            }                                                                  \
            const T y = r / (eo - ev[v[0]] - ev[v[1]] - ev[v[2]] - ev[v[3]]);  \
            for (int s = 0; s < 6; s++)                                        \
                if (x22[s] >= 0) e22[x22[s]] = s22[s] * y;                     \
            for (int s = 0; s < 4; s++)                                        \
                if (x13[s] >= 0) e13[x13[s]] = ((s & 1) ? -1.0 : 1.0) * y;     \
        }                                                                      \
    }                                                                          \
}

DEFINE_CC4SEC(double, d)
DEFINE_CC4SEC(double complex, z)
