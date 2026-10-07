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
 * the tuples in order, in chunks, and compute the ranks of the groups of
 * every split from the binomial coefficients.
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

/* the offsets in X of the splits of ``cnt`` tuples from rank A0 on (-1:
 * outside the block of X).  From one tuple to the next mostly the last
 * index grows by one, and with it the rank of the group that has it. */
static void tn_offsets(const tn_split *sp, int64_t A0, int64_t cnt,
                       int64_t b0, int64_t b1, int64_t c0, int64_t c1,
                       int64_t s1, int64_t s2, int64_t *off)
{
    int a[KMAX], last1[NSPLIT_MAX];
    int64_t r1[NSPLIT_MAX], r2[NSPLIT_MAX];
    const int n = sp->n, k = sp->k, k1 = sp->k1, k2 = sp->k2, ns = sp->nsplit;
    const int64_t top1 = CB(sp, k1, n) - 1, top2 = CB(sp, k2, n) - 1;
    for (int s = 0; s < ns; s++)
        last1[s] = k1 > 0 && sp->p1[s][k1 - 1] == k - 1;
    tn_unrank(sp, A0, a);
    int fresh = 1;
    for (int64_t t = 0; t < cnt; t++) {
        if (fresh) {
            for (int s = 0; s < ns; s++) {
                int64_t x = top1, y = top2;
                for (int u = 0; u < k1; u++)
                    x -= CB(sp, k1 - u, n - 1 - a[sp->p1[s][u]]);
                for (int u = 0; u < k2; u++)
                    y -= CB(sp, k2 - u, n - 1 - a[sp->p2[s][u]]);
                r1[s] = x; r2[s] = y;
            }
        }
        int64_t *f = off + t * ns;
        for (int s = 0; s < ns; s++)
            f[s] = (r1[s] >= b0 && r1[s] < b1 && r2[s] >= c0 && r2[s] < c1)
                ? (r1[s] - b0) * s1 + (r2[s] - c0) * s2 : -1;
        if (a[k - 1] < n - 1) {
            a[k - 1]++;
            for (int s = 0; s < ns; s++) {
                r1[s] += last1[s];
                r2[s] += !last1[s];
            }
            fresh = 0;
        } else {
            tn_next(sp, a);
            fresh = 1;
        }
    }
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
    const int64_t nA = sp.ntuple, nchunk = (nA + CHUNK - 1) / CHUNK;           \
    const int ns = sp.nsplit;                                                  \
    _Pragma("omp parallel")                                                    \
    {                                                                          \
        int64_t *off = (int64_t *)malloc(sizeof(int64_t) * CHUNK * ns);        \
        _Pragma("omp for schedule(dynamic, 4)")                                \
        for (int64_t ch = 0; ch < nchunk; ch++) {                              \
            int64_t A0 = ch * CHUNK, cnt = (nA - A0 < CHUNK) ? nA - A0 : CHUNK; \
            tn_offsets(&sp, A0, cnt, b0, b1, c0, c1, s1, s2, off);             \
            for (int64_t o = 0; o < nout; o++) {                               \
                T *r = R + o * sR + A0;                                        \
                const T *x = X + o * so;                                       \
                for (int64_t t = 0; t < cnt; t++) {                            \
                    const int64_t *f = off + t * ns;                           \
                    T v = 0;                                                   \
                    for (int s = 0; s < ns; s++)                               \
                        if (f[s] >= 0)                                         \
                            v += sp.sgn[s] * x[f[s]];                          \
                    r[t] += scale * v;                                         \
                }                                                              \
            }                                                                  \
        }                                                                      \
        free(off);                                                             \
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
    const int64_t nA = sp.ntuple, nchunk = (nA + CHUNK - 1) / CHUNK;           \
    const int ns = sp.nsplit;                                                  \
    _Pragma("omp parallel")                                                    \
    {                                                                          \
        int64_t *off = (int64_t *)malloc(sizeof(int64_t) * CHUNK * ns);        \
        _Pragma("omp for schedule(dynamic, 4)")                                \
        for (int64_t ch = 0; ch < nchunk; ch++) {                              \
            int64_t A0 = ch * CHUNK, cnt = (nA - A0 < CHUNK) ? nA - A0 : CHUNK; \
            tn_offsets(&sp, A0, cnt, b0, b1, c0, c1, s1, s2, off);             \
            for (int64_t o = 0; o < nout; o++) {                               \
                const T *r = R + o * sR + A0;                                  \
                T *x = X + o * so;                                             \
                for (int64_t t = 0; t < cnt; t++) {                            \
                    const int64_t *f = off + t * ns;                           \
                    T v = scale * r[t];                                        \
                    for (int s = 0; s < ns; s++)                               \
                        if (f[s] >= 0)                                         \
                            x[f[s]] = sp.sgn[s] * v;                           \
                }                                                              \
            }                                                                  \
        }                                                                      \
        free(off);                                                             \
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
 * other: the parallel loop is over (rows) x (chunks of tuples), so that no    \
 * two threads write the same element, and a thread stays within one slab of   \
 * X for a while. */                                                           \
void zc_tn_reduce2_##SFX(T *R, int64_t ldR, int64_t nout, const int64_t *row,  \
                         const int64_t *xoff, const double *osgn, int n,       \
                         int k1, int k2, int64_t b0, int64_t b1, int64_t c0,   \
                         int64_t c1, const T *X, int64_t s1, int64_t s2,       \
                         double scale)                                         \
{                                                                              \
    tn_split sp;                                                               \
    if (nout <= 0 || tn_setup(&sp, n, k1, k2))                                 \
        return;                                                                \
    const int64_t nA = sp.ntuple, nchunk = (nA + CHUNK - 1) / CHUNK;           \
    const int ns = sp.nsplit;                                                  \
    int64_t *gs = (int64_t *)malloc(sizeof(int64_t) * (size_t)(nout + 1));     \
    int64_t ng = 0;                                                            \
    for (int64_t o = 0; o < nout; o++)                                         \
        if (o == 0 || row[o] != row[o - 1])                                    \
            gs[ng++] = o;                                                      \
    gs[ng] = nout;                                                             \
    _Pragma("omp parallel")                                                    \
    {                                                                          \
        int64_t *off = (int64_t *)malloc(sizeof(int64_t) * CHUNK * ns);        \
        _Pragma("omp for collapse(2) schedule(dynamic, 16)")                   \
        for (int64_t g = 0; g < ng; g++)                                       \
        for (int64_t ch = 0; ch < nchunk; ch++) {                              \
            int64_t A0 = ch * CHUNK, cnt = (nA - A0 < CHUNK) ? nA - A0 : CHUNK; \
            tn_offsets(&sp, A0, cnt, b0, b1, c0, c1, s1, s2, off);             \
            T *r = R + row[gs[g]] * ldR + A0;                                  \
            for (int64_t o = gs[g]; o < gs[g + 1]; o++) {                      \
                if (osgn[o] == 0)                                              \
                    continue;                                                  \
                const T *x = X + xoff[o];                                      \
                const double sg = scale * osgn[o];                             \
                for (int64_t t = 0; t < cnt; t++) {                            \
                    const int64_t *f = off + t * ns;                           \
                    T v = 0;                                                   \
                    for (int s = 0; s < ns; s++)                               \
                        if (f[s] >= 0)                                         \
                            v += sp.sgn[s] * x[f[s]];                          \
                    r[t] += sg * v;                                            \
                }                                                              \
            }                                                                  \
        }                                                                      \
        free(off);                                                             \
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
    const int64_t nA = sp.ntuple, nchunk = (nA + CHUNK - 1) / CHUNK;           \
    const int ns = sp.nsplit;                                                  \
    _Pragma("omp parallel")                                                    \
    {                                                                          \
        int64_t *off = (int64_t *)malloc(sizeof(int64_t) * CHUNK * ns);        \
        _Pragma("omp for collapse(2) schedule(dynamic, 16)")                   \
        for (int64_t o = 0; o < nout; o++)                                     \
        for (int64_t ch = 0; ch < nchunk; ch++) {                              \
            if (osgn[o] == 0)                                                  \
                continue;                                                      \
            int64_t A0 = ch * CHUNK, cnt = (nA - A0 < CHUNK) ? nA - A0 : CHUNK; \
            tn_offsets(&sp, A0, cnt, b0, b1, c0, c1, s1, s2, off);             \
            const T *r = M + row[o] * ldM + A0;                                \
            T *x = X + xoff[o];                                                \
            const double sg = scale * osgn[o];                                 \
            for (int64_t t = 0; t < cnt; t++) {                                \
                const int64_t *f = off + t * ns;                               \
                T v = sg * r[t];                                               \
                for (int s = 0; s < ns; s++)                                   \
                    if (f[s] >= 0)                                             \
                        x[f[s]] = sp.sgn[s] * v;                               \
            }                                                                  \
        }                                                                      \
        free(off);                                                             \
    }                                                                          \
    free(sp.cb);                                                               \
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
