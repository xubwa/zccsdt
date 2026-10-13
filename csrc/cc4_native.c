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
 *
 * Every kernel exists for double (_d) and double complex (_z).
 */
#include <complex.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define DEFINE_CC4(T, SFX)                                                     \
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
}

DEFINE_CC4(double, d)
DEFINE_CC4(double complex, z)
