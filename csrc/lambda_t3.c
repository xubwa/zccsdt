/*
 * lambda_t3.c -- kernels of the Lambda equations of CCSDT on packed
 * Lambda_3 (zccsdt/cc/lambda_opt.py) that are not those of the amplitude
 * equations: an intermediate packed over a virtual pair a<b, Y[y, x, Q],
 * expanded to both orders of the pair, and contracted with the columns of
 * the packed pp operator.  The passes over the packed arrays themselves use
 * the gathers and reductions of t3pack.c and the BLAS.
 *
 * Pairs a<b of n indices in lexicographic (numpy.triu_indices) order:
 * Q(a, b) = a n - a (a + 1) / 2 + b - a - 1, the pairs of one a in a row.
 * Every function exists for double (_d) and double complex (_z).
 */

#include <complex.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define QBLK 64

#define DEFINE_LAM(T, SFX)                                                     \
                                                                               \
/* out[y, e, x, a] = Y[y, x, Q(a,e)] for a < e, -Y[y, x, Q(e,a)] for a > e,   \
 * 0 for a = e.   Y: (ny, nx, n (n-1) / 2), out: (ny, n, nx, n). */           \
void zc_lam_pair_expand_##SFX(const T *Y, T *out, int64_t ny, int64_t nx,      \
                              int64_t n)                                       \
{                                                                              \
    const int64_t nq = n * (n - 1) / 2;                                        \
    _Pragma("omp parallel for collapse(2) schedule(static)")                   \
    for (int64_t y = 0; y < ny; y++)                                           \
    for (int64_t e = 0; e < n; e++) {                                          \
        T *o = out + (y * n + e) * nx * n;                                     \
        /* pairs (e, a), a > e: a row of pairs */                              \
        const int64_t qe = e * n - e * (e + 1) / 2 - e - 1;                    \
        for (int64_t x = 0; x < nx; x++) {                                     \
            const T *r = Y + (y * nx + x) * nq;                                \
            T *ox = o + x * n;                                                 \
            for (int64_t a = 0; a < e; a++)                                    \
                ox[a] = r[a * n - a * (a + 1) / 2 + e - a - 1];                \
            ox[e] = 0;                                                         \
            for (int64_t a = e + 1; a < n; a++) ox[a] = -r[qe + a];            \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
/* acc[x, a] += sum_Q sum_h Y[h, x, Q] w[Q, (a,h)],  w[Q, (a,h)] = W[Q, Q(a,h)] \
 * for a < h and -W[Q, Q(h,a)] for a > h: the packed columns of W expanded     \
 * and contracted with the first index of Y.                                   \
 * Y: (n, nx, nq), W: (nq, nq), acc: (nx, n).  Per x (a thread each) the      \
 * rows of Y are taken QBLK pairs at a time into a buffer [Q, h]. */           \
void zc_lam_pair_matvec_##SFX(const T *Y, const T *W, T *acc, int64_t nx,      \
                              int64_t n)                                       \
{                                                                              \
    const int64_t nq = n * (n - 1) / 2;                                        \
    _Pragma("omp parallel")                                                    \
    {                                                                          \
        T *yb = malloc(sizeof(T) * QBLK * n);                                  \
        T *loc = malloc(sizeof(T) * n);                                        \
        _Pragma("omp for schedule(dynamic, 1)")                                \
        for (int64_t x = 0; x < nx; x++) {                                     \
            memset(loc, 0, sizeof(T) * n);                                     \
            for (int64_t q0 = 0; q0 < nq; q0 += QBLK) {                        \
                const int64_t nb = (q0 + QBLK <= nq) ? QBLK : nq - q0;         \
                for (int64_t h = 0; h < n; h++) {                              \
                    const T *r = Y + (h * nx + x) * nq + q0;                   \
                    for (int64_t q = 0; q < nb; q++) yb[q * n + h] = r[q];     \
                }                                                              \
                for (int64_t q = 0; q < nb; q++) {                             \
                    const T *w = W + (q0 + q) * nq;                            \
                    const T *y = yb + q * n;                                   \
                    for (int64_t a = 0; a < n - 1; a++) {                      \
                        /* the pairs (a, h), h > a, at w - (a + 1) */          \
                        const int64_t m = n - a - 1;                           \
                        const T ya = y[a];                                     \
                        T s = 0;                                               \
                        for (int64_t k = 0; k < m; k++) {                      \
                            s += w[k] * y[a + 1 + k];                          \
                            loc[a + 1 + k] -= w[k] * ya;                       \
                        }                                                      \
                        loc[a] += s;                                           \
                        w += m;                                                \
                    }                                                          \
                }                                                              \
            }                                                                  \
            for (int64_t a = 0; a < n; a++) acc[x * n + a] += loc[a];          \
        }                                                                      \
        free(yb); free(loc);                                                   \
    }                                                                          \
}

DEFINE_LAM(double, d)
DEFINE_LAM(double complex, z)
