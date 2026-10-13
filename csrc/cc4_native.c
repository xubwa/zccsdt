/*
 * cc4_native.c -- kernels of CC4 without stored quadruples for the native
 * program (cfour/xzccsdt/xzcc4.inc; docs/native.md, section 19).
 *
 * The quadruples of one occupied quadruple are made from the products of
 * the (Q) correction with the T1-dressed integrals, divided by the
 * denominator and contracted into the doubles and triples equations.  The
 * products and the contractions are GEMMs of the scheduler; here is what
 * lies between them:
 *
 * zc_cc4_divide         the row over the denominator (no symmetry labels)
 * zc_cc4_sector_t4      in the sectors of the labels: zc_q_sector_energy of
 *                       csrc/tnpack.c with the amplitudes written back to
 *                       the places of the blocks in the place of the energy
 * zc_cc4_red12          the reduction (c | ab) of blocks [c, (ab)] onto the
 *                       virtual triples of one group
 * zc_cc4_addrows        rows of a product added to rows of a packed block
 * zc_cc4_sector_t4i     zc_cc4_sector_t4 in place: the amplitudes take the
 *                       places of the right-hand side, the places that
 *                       belong to no quadruple are set to zero
 * zc_cc4_rowsmm         a product whose right operand is rows that stay
 *                       where they are (addresses), in blocks of columns
 * zc_cc4_addmm          a product added to rows that stay where they are
 *
 * Every kernel exists for double (_d) and double complex (_z).
 */
#include <complex.h>
#include <stdint.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

/* row-major GEMM of csrc/ccsdt_t3.c: one thread inside a parallel region */
void zc_gemm_d(int ta, int tb, int64_t m, int64_t n, int64_t k,
               const double *alpha, const double *a, int64_t lda,
               const double *b, int64_t ldb, const double *beta, double *c,
               int64_t ldc);
void zc_gemm_z(int ta, int tb, int64_t m, int64_t n, int64_t k,
               const double complex *alpha, const double complex *a, int64_t lda,
               const double complex *b, int64_t ldb, const double complex *beta,
               double complex *c, int64_t ldc);

#define DEFINE_CC4(T, SFX)                                                     \
                                                                               \
/* a buffer of n elements of the thread (kept: the kernels below are called    \
 * some ten times per occupied quadruple) */                                   \
static __thread T *cc4_buf_##SFX = NULL;                                       \
static __thread size_t cc4_cap_##SFX = 0;                                      \
static T *cc4_tmp_##SFX(size_t n)                                              \
{                                                                              \
    if (n > cc4_cap_##SFX) {                                                   \
        free(cc4_buf_##SFX);                                                   \
        cc4_buf_##SFX = malloc(sizeof(T) * n);                                 \
        cc4_cap_##SFX = n;                                                     \
    }                                                                          \
    return cc4_buf_##SFX;                                                      \
}                                                                              \
                                                                               \
/* r[A] /= eo - ev[A] */                                                       \
void zc_cc4_divide_##SFX(T *r, const double *ev, double eo, int64_t n)         \
{                                                                              \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t a = 0; a < n; a++) r[a] /= eo - ev[a];                        \
}                                                                              \
                                                                               \
/* For every virtual quadruple a<b<c<d of total label M:                      \
 *   t = (sum of its six splits in b + sum of its four splits in c) / (eo - ev)\
 * with the blocks and the tables of zc_q_sector_energy (csrc/tnpack.c), and   \
 * t written with the sign of the split to the place of every split in e22     \
 * (two pairs) and e13 (an index and a triple).  The places that belong to no  \
 * quadruple (groups with a common index) are not touched: the caller sets     \
 * e22 and e13 to zero. */                                                     \
void zc_cc4_sector_t4_##SFX(int n, const int *kv, int modulus, int M,          \
        const double *ev, double eo, int kmin, int nk, const int *lid_of,      \
        const int *lstart, const int *lmem, int qmin, int nq,                  \
        const int64_t *off22, const int64_t *nc22, const int64_t *off13,       \
        const int64_t *nc13, const int32_t *posV, const int32_t *posQ,         \
        const int32_t *pos3, const T *b, const T *c, T *e22, T *e13)           \
{                                                                              \
    const int64_t nQ = (int64_t)n * (n - 1) / 2;                               \
    const int64_t nA3 = nQ * (n - 2) / 3;                                      \
    _Pragma("omp parallel for schedule(dynamic, 1)")                           \
    for (int a = 0; a < n - 3; a++)                                            \
    for (int bb = a + 1; bb < n - 2; bb++)                                     \
    for (int cc = bb + 1; cc < n - 1; cc++) {                                  \
        int kd = M - kv[a] - kv[bb] - kv[cc];                                  \
        if (modulus) kd = ((kd % modulus) + modulus) % modulus;                \
        if (kd < kmin || kd >= kmin + nk || lid_of[kd - kmin] < 0) continue;   \
        const int l = lid_of[kd - kmin];                                       \
        int lo = lstart[l], hi = lstart[l + 1];                                \
        while (lo < hi) {               /* the first member above cc */        \
            int mid = (lo + hi) / 2;                                           \
            if (lmem[mid] > cc) hi = mid; else lo = mid + 1;                   \
        }                                                                      \
        for (int t = lo; t < lstart[l + 1]; t++) {                             \
            const int v[4] = {a, bb, cc, lmem[t]};                             \
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
                r += s22[s] * b[x];                                            \
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
                r += ((s & 1) ? -1.0 : 1.0) * c[x];                            \
            }                                                                  \
            r /= eo - ev[v[0]] - ev[v[1]] - ev[v[2]] - ev[v[3]];               \
            for (int s = 0; s < 6; s++)                                        \
                if (x22[s] >= 0) e22[x22[s]] = s22[s] * r;                     \
            for (int s = 0; s < 4; s++)                                        \
                if (x13[s] >= 0) e13[x13[s]] = ((s & 1) ? -1.0 : 1.0) * r;     \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
/* R[t] += sum over the splits (x | yz) of the sorted triple t = (p<q<r) of    \
 * sgn * Y[off[k(yz)] + posV[x] * nq[k(yz)] + posQ[(yz)]]: the blocks          \
 * [x, (yz)] of Y by the label of the pair (off < 0: no block), signs +, -, +  \
 * for x = p, q, r.  tri: the nt triples of the row R (3 indices each). */      \
void zc_cc4_red12_##SFX(T *R, const int32_t *tri, int64_t nt, int n,           \
        const int *kv, int modulus, int qmin, int nqg, const int64_t *off,     \
        const int64_t *nq, const int32_t *posV, const int32_t *posQ,           \
        const T *Y)                                                            \
{                                                                              \
    const int64_t nQ = (int64_t)n * (n - 1) / 2;                               \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t t = 0; t < nt; t++) {                                         \
        const int v[3] = {tri[3 * t], tri[3 * t + 1], tri[3 * t + 2]};         \
        static const int g[3][3] = {{0, 1, 2}, {1, 0, 2}, {2, 0, 1}};          \
        static const double sg[3] = {1, -1, 1};                                \
        T r = 0;                                                               \
        for (int s = 0; s < 3; s++) {                                          \
            const int x = v[g[s][0]], y = v[g[s][1]], z = v[g[s][2]];          \
            int kq = kv[y] + kv[z];                                            \
            if (modulus) kq = ((kq % modulus) + modulus) % modulus;            \
            if (kq < qmin || kq >= qmin + nqg || off[kq - qmin] < 0) continue; \
            const int64_t rk = nQ - 1 - (int64_t)(n - 1 - y) * (n - 2 - y) / 2 - (n - 1 - z);      \
            r += sg[s] * Y[off[kq - qmin] + posV[x] * nq[kq - qmin] + posQ[rk]];                   \
        }                                                                      \
        R[t] += r;                                                             \
    }                                                                          \
}                                                                              \
                                                                               \
/* X[a, c] += alpha sum_r L[r, a] sg[r] M[rows[r] * ld + c]: zc_rows_gemm of     \
 * csrc/tnpack.c on all columns of the rows, added to X (L: [nr, na],          \
 * X: [na, nc]) */                                                             \
void zc_cc4_rows_gemm_##SFX(const T *M, int64_t ld, const int64_t *rows,       \
        const double *sg, int64_t nr, int64_t nc, const T *L, int64_t na,      \
        double alpha, T *X)                                                    \
{                                                                              \
    _Pragma("omp parallel")                                                    \
    {                                                                          \
        T *tmp = malloc(sizeof(T) * (size_t)(nr > 0 ? nr : 1));                \
        _Pragma("omp for schedule(static)")                                    \
        for (int64_t c = 0; c < nc; c++) {                                     \
            for (int64_t r = 0; r < nr; r++)                                   \
                tmp[r] = (alpha * sg[r]) * M[rows[r] * ld + c];                \
            for (int64_t a = 0; a < na; a++) {                                 \
                T x = 0;                                                       \
                for (int64_t r = 0; r < nr; r++) x += L[r * na + a] * tmp[r];  \
                X[a * nc + c] += x;                                            \
            }                                                                  \
        }                                                                      \
        free(tmp);                                                             \
    }                                                                          \
}                                                                              \
                                                                               \
/* R[tgt[r] * ld + c] += X[r * nc + c], c < nc (rows may repeat: the threads   \
 * take blocks of columns) */                                                  \
void zc_cc4_addrows_##SFX(T *R, int64_t ld, const int64_t *tgt, int64_t nr,    \
        int64_t nc, const T *X)                                                \
{                                                                              \
    const int64_t blk = 1024, nblk = (nc + blk - 1) / blk;                     \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t bk = 0; bk < nblk; bk++) {                                    \
        int64_t lo = bk * blk, hi = (lo + blk < nc) ? lo + blk : nc;           \
        for (int64_t r = 0; r < nr; r++) {                                     \
            T *o = R + tgt[r] * ld;                                            \
            const T *x = X + r * nc;                                           \
            for (int64_t c = lo; c < hi; c++) o[c] += x[c];                    \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
/* zc_cc4_sector_t4 in place: for every virtual quadruple of total label M     \
 * t = (its six splits in b + its four splits in c) / (eo - ev) is written,    \
 * with the sign of the split, over the splits it was summed from (every       \
 * place of b and c with four different indices is one split of one            \
 * quadruple).  The places with a common index - [(pq)][(pu)], [p][(pyz)] -    \
 * belong to no quadruple and are set to zero.                                 \
 *                                                                             \
 * The ten places of a quadruple are found from the tables for the first d of  \
 * a run of consecutive d of one label; for the next d each moves by a fixed   \
 * step - one element where d is in the columns, one row where it is in the    \
 * rows (pairs and triples that differ in their last index alone follow each   \
 * other in their groups). */                                                  \
void zc_cc4_sector_t4i_##SFX(int n, const int *kv, int modulus, int M,         \
        const double *ev, double eo, int kmin, int nk, const int *lid_of,      \
        const int *lstart, const int *lmem, int qmin, int nq,                  \
        const int64_t *off22, const int64_t *nc22, const int64_t *off13,       \
        const int64_t *nc13, const int32_t *posV, const int32_t *posQ,         \
        const int32_t *pos3, T *b, T *c)                                       \
{                                                                              \
    const int64_t nQ = (int64_t)n * (n - 1) / 2;                               \
    const int64_t nA3 = nQ * (n - 2) / 3;                                      \
    _Pragma("omp parallel for schedule(dynamic, 1)")                           \
    for (int a = 0; a < n - 3; a++)                                            \
    for (int bb = a + 1; bb < n - 2; bb++)                                     \
    for (int cc = bb + 1; cc < n - 1; cc++) {                                  \
        int kd = M - kv[a] - kv[bb] - kv[cc];                                  \
        if (modulus) kd = ((kd % modulus) + modulus) % modulus;                \
        if (kd < kmin || kd >= kmin + nk || lid_of[kd - kmin] < 0) continue;   \
        const int l = lid_of[kd - kmin];                                       \
        int lo = lstart[l], hi = lstart[l + 1];                                \
        while (lo < hi) {               /* the first member above cc */        \
            int mid = (lo + hi) / 2;                                           \
            if (lmem[mid] > cc) hi = mid; else lo = mid + 1;                   \
        }                                                                      \
        static const int g22[6][4] = {{0,1,2,3}, {0,2,1,3}, {0,3,1,2},         \
                                      {1,2,0,3}, {1,3,0,2}, {2,3,0,1}};        \
        static const double s22[6] = {1, -1, 1, 1, -1, 1};                     \
        int64_t x22[6], x13[4], d22[6], d13[4];                                \
        const double eabc = eo - ev[a] - ev[bb] - ev[cc];                      \
        for (int t = lo; t < lstart[l + 1]; t++) {                             \
            const int v[4] = {a, bb, cc, lmem[t]};                             \
            if (t > lo && lmem[t] == lmem[t - 1] + 1) {                        \
                for (int s = 0; s < 6; s++) if (x22[s] >= 0) x22[s] += d22[s]; \
                for (int s = 0; s < 4; s++) if (x13[s] >= 0) x13[s] += d13[s]; \
            } else {                                                           \
                for (int s = 0; s < 6; s++) {                                  \
                    const int p = v[g22[s][0]], q = v[g22[s][1]];              \
                    const int w = v[g22[s][2]], u = v[g22[s][3]];              \
                    int kq = kv[p] + kv[q];                                    \
                    x22[s] = -1;                                               \
                    if (modulus) kq = ((kq % modulus) + modulus) % modulus;    \
                    if (kq < qmin || kq >= qmin + nq || off22[kq - qmin] < 0)  \
                        continue;                                              \
                    const int64_t rk1 = nQ - 1 - (int64_t)(n - 1 - p) * (n - 2 - p) / 2 - (n - 1 - q); \
                    const int64_t rk2 = nQ - 1 - (int64_t)(n - 1 - w) * (n - 2 - w) / 2 - (n - 1 - u); \
                    x22[s] = off22[kq - qmin] + posQ[rk1] * nc22[kq - qmin] + posQ[rk2];               \
                    /* d is the last index of the pair of the columns (splits 0, 1, 3) or of the rows */ \
                    d22[s] = (s == 0 || s == 1 || s == 3) ? 1 : nc22[kq - qmin];                       \
                }                                                              \
                for (int s = 0; s < 4; s++) {                                  \
                    const int p = v[s];                                        \
                    const int q = v[s == 0], w = v[s <= 1 ? 2 : 1], u = v[s == 3 ? 2 : 3];             \
                    const int lp = lid_of[kv[p] - kmin];                       \
                    x13[s] = -1;                                               \
                    if (off13[lp] < 0) continue;                               \
                    const int64_t y1 = n - 1 - q, y2 = n - 1 - w;              \
                    const int64_t rk = nA3 - 1 - y1 * (y1 - 1) * (y1 - 2) / 6 - y2 * (y2 - 1) / 2 - (n - 1 - u); \
                    x13[s] = off13[lp] + posV[p] * nc13[lp] + pos3[rk];        \
                    d13[s] = s == 3 ? nc13[lp] : 1;                            \
                }                                                              \
            }                                                                  \
            T r = 0;                                                           \
            for (int s = 0; s < 6; s++) if (x22[s] >= 0) r += s22[s] * b[x22[s]];                      \
            for (int s = 0; s < 4; s++) if (x13[s] >= 0) r += ((s & 1) ? -1.0 : 1.0) * c[x13[s]];      \
            r /= eabc - ev[v[3]];                                              \
            for (int s = 0; s < 6; s++)                                        \
                if (x22[s] >= 0) b[x22[s]] = s22[s] * r;                       \
            for (int s = 0; s < 4; s++)                                        \
                if (x13[s] >= 0) c[x13[s]] = ((s & 1) ? -1.0 : 1.0) * r;       \
        }                                                                      \
    }                                                                          \
    /* [(pq)][(wu)] with a common index */                                     \
    _Pragma("omp parallel for schedule(dynamic, 1)")                           \
    for (int p = 0; p < n - 1; p++)                                            \
    for (int q = p + 1; q < n; q++) {                                          \
        int kq = kv[p] + kv[q];                                                \
        if (modulus) kq = ((kq % modulus) + modulus) % modulus;                \
        if (kq < qmin || kq >= qmin + nq || off22[kq - qmin] < 0) continue;    \
        int kc = M - kq;                /* the label of the pairs (wu) */      \
        if (modulus) kc = ((kc % modulus) + modulus) % modulus;                \
        const int64_t rk1 = nQ - 1 - (int64_t)(n - 1 - p) * (n - 2 - p) / 2 - (n - 1 - q);         \
        T *row = b + off22[kq - qmin] + posQ[rk1] * nc22[kq - qmin];           \
        if (kc == kq) row[posQ[rk1]] = 0;                                      \
        for (int u = 0; u < n; u++) {                                          \
            if (u == p || u == q) continue;                                    \
            for (int h = 0; h < 2; h++) {                                      \
                const int f = h ? q : p;                                       \
                int k2 = kv[f] + kv[u];                                        \
                if (modulus) k2 = ((k2 % modulus) + modulus) % modulus;        \
                if (k2 != kc) continue;                                        \
                const int lo = f < u ? f : u, hi = f < u ? u : f;              \
                row[posQ[nQ - 1 - (int64_t)(n - 1 - lo) * (n - 2 - lo) / 2 - (n - 1 - hi)]] = 0;   \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    /* [p][(bcd)] with p in the triple */                                      \
    _Pragma("omp parallel for schedule(dynamic, 1)")                           \
    for (int p = 0; p < n; p++) {                                              \
        const int lp = lid_of[kv[p] - kmin];                                   \
        if (lp < 0 || off13[lp] < 0) continue;                                 \
        T *row = c + off13[lp] + posV[p] * nc13[lp];                           \
        for (int y = 0; y < n - 1; y++) {                                      \
            if (y == p) continue;                                              \
            for (int z = y + 1; z < n; z++) {                                  \
                if (z == p) continue;                                          \
                int k3 = 2 * kv[p] + kv[y] + kv[z] - M;                        \
                if (modulus) k3 = ((k3 % modulus) + modulus) % modulus;        \
                if (k3 != 0) continue;                                         \
                const int q = p < y ? p : y;                                   \
                const int u = p > z ? p : z;                                   \
                const int w = p + y + z - q - u;                               \
                const int64_t y1 = n - 1 - q, y2 = n - 1 - w;                  \
                row[pos3[nA3 - 1 - y1 * (y1 - 1) * (y1 - 2) / 6 - y2 * (y2 - 1) / 2 - (n - 1 - u)]] = 0; \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
/* X[a, c] = [X[a, c] +] sum_r L[r, a] row[r][c] (init: without the bracket):  \
 * a product whose right operand is nr rows of nc elements that stay where     \
 * they are (row: their addresses); L: [nr, na], X: [na, nc].  The threads     \
 * take blocks of columns: the rows of a block are gathered into the cache     \
 * and multiplied there. */                                                    \
void zc_cc4_rowsmm_##SFX(const T *const *row, const T *L, int64_t nr,          \
        int64_t na, int64_t nc, int64_t init, T *X)                            \
{                                                                              \
    const int64_t blk = nr <= 128 ? 512 : 256, nblk = (nc + blk - 1) / blk;    \
    const T one = 1, beta = init ? 0 : 1;                                      \
    _Pragma("omp parallel")                                                    \
    {                                                                          \
        T *tmp = cc4_tmp_##SFX((size_t)(nr * blk));                            \
        _Pragma("omp for schedule(static)")                                    \
        for (int64_t bk = 0; bk < nblk; bk++) {                                \
            const int64_t lo = bk * blk;                                       \
            const int64_t len = (lo + blk < nc ? lo + blk : nc) - lo;          \
            for (int64_t r = 0; r < nr; r++)                                   \
                memcpy(tmp + r * len, row[r] + lo, sizeof(T) * (size_t)len);   \
            zc_gemm_##SFX(1, 0, na, len, nr, &one, L, na, tmp, len, &beta,     \
                          X + lo, nc);                                         \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
/* row[r][c] += sum_a L[r, a] E[a, c]: a product added to nr rows of nc        \
 * elements that stay where they are (row: their addresses; a row may be       \
 * there more than once); L: [nr, na], E: [na, nc].  The threads take blocks   \
 * of columns. */                                                              \
void zc_cc4_addmm_##SFX(T *const *row, const T *L, int64_t nr, int64_t na,     \
        int64_t nc, const T *E)                                                \
{                                                                              \
    const int64_t blk = nr <= 128 ? 512 : 256, nblk = (nc + blk - 1) / blk;    \
    const T one = 1, zero = 0;                                                 \
    _Pragma("omp parallel")                                                    \
    {                                                                          \
        T *tmp = cc4_tmp_##SFX((size_t)(nr * blk));                            \
        _Pragma("omp for schedule(static)")                                    \
        for (int64_t bk = 0; bk < nblk; bk++) {                                \
            const int64_t lo = bk * blk;                                       \
            const int64_t len = (lo + blk < nc ? lo + blk : nc) - lo;          \
            zc_gemm_##SFX(0, 0, nr, len, na, &one, L, na, E + lo, nc, &zero,   \
                          tmp, len);                                           \
            for (int64_t r = 0; r < nr; r++) {                                 \
                T *x = row[r] + lo;                                            \
                const T *t = tmp + r * len;                                    \
                for (int64_t c = 0; c < len; c++) x[c] += t[c];                \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

DEFINE_CC4(double, d)
DEFINE_CC4(double complex, z)
