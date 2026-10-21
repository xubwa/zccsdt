/*
 * Kernels of the T4 equation in blocks of a symmetry label
 * (zccsdt/cc/t4engine_sym.py).
 *
 * Every spinor has a label 0 .. ncls - 1 that adds, and the spinors are in
 * the order of their labels.  A block of T4 holds the occupied and the
 * virtual quadruples of one total label M, the virtual ones as a list
 * (zc_t4s_quads, lexicographic).  The kernels walk through such a list and
 * move elements between the block and the operand or the result of a matrix
 * product, which is in blocks itself: one per label of the first group of a
 * split of the quadruple,
 *
 *     (a | bcd)   a at posV[a] inside its class, the triple at pos3[rank]
 *     (ab | cd)   the pairs at posQ[a n + b] inside the classes of their labels
 *
 * and for every entry e - a row of the block with a sign: a split of an
 * occupied quadruple, or the quadruple itself - the block of label l starts
 * at xoff[e, l] (negative: there is none) and has the strides s1[e, l] and
 * s2[e, l] of the two groups:
 *
 *     X[xoff[e, l] + pos(group 1) s1[e, l] + pos(group 2) s2[e, l]]
 *
 * expand (mode 0):  X[..] = sgn[e] * sign of the split * T[row[e], A]
 * reduce (mode 1):  R[row[e], A] += sgn[e] * sum of the splits of sign * X[..]
 *
 * The same for triples and the splits (a | bc) (zc_t4s_x12).  The lists are
 * walked in chunks on all threads; a chunk belongs to one thread, so the
 * entries may share rows.
 *
 * zc_t4s_tasks runs a table of small matrix products C += alpha op(A) B on
 * blocks of arrays, the groups of products with one target on separate
 * threads and the BLAS sequential inside (the GEMM of the kernel library
 * is passed in: one BLAS for all).
 */
#include <complex.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define CHUNK 256

/* the quadruples a<b<c<d of n indices with kv[a] + kv[b] + kv[c] + kv[d] = M
 * (kv ascending): their number; written to out[.., 4] unless it is NULL */
int64_t zc_t4s_quads(int n, const int32_t *kv, int M, int16_t *out)
{
    int64_t cnt = 0;
    for (int a = 0; a < n - 3; a++)
    for (int b = a + 1; b < n - 2; b++)
    for (int c = b + 1; c < n - 1; c++) {
        const int kd = M - kv[a] - kv[b] - kv[c];
        if (kd < kv[c + 1] || kd > kv[n - 1]) continue;
        int lo = c + 1, hi = n;             /* the first d above c with kv[d] >= kd */
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (kv[mid] >= kd) hi = mid; else lo = mid + 1;
        }
        for (int d = lo; d < n && kv[d] == kd; d++) {
            if (out) {
                int16_t *o = out + 4 * cnt;
                o[0] = (int16_t)a; o[1] = (int16_t)b; o[2] = (int16_t)c; o[3] = (int16_t)d;
            }
            cnt++;
        }
    }
    return cnt;
}

typedef void (*zc_gemm_fn)(int ta, int tb, int64_t m, int64_t n, int64_t k,
                           const void *alpha, const void *a, int64_t lda,
                           const void *b, int64_t ldb, const void *beta, void *c,
                           int64_t ldc);

static const int G22[6][4] = {{0,1,2,3}, {0,2,1,3}, {0,3,1,2}, {1,2,0,3}, {1,3,0,2}, {2,3,0,1}};
static const double S22[6] = {1, -1, 1, 1, -1, 1};

#define DEFINE_T4S(T, SFX)                                                     \
                                                                               \
void zc_t4s_x13_##SFX(int mode, T *R, int64_t ldR, const int16_t *quads,       \
        int64_t nA, int64_t nent, const int64_t *row, const double *sgn,       \
        const int64_t *xoff, const int64_t *s1, const int64_t *s2, int ncls,   \
        int n, const int32_t *kv, const int32_t *posV, const int32_t *pos3,    \
        T *X)                                                                  \
{                                                                              \
    const int64_t nA3 = (int64_t)n * (n - 1) * (n - 2) / 6;                    \
    const int64_t nchunk = (nA + CHUNK - 1) / CHUNK;                           \
    _Pragma("omp parallel for schedule(dynamic, 4)")                           \
    for (int64_t ch = 0; ch < nchunk; ch++) {                                  \
        const int64_t A0 = ch * CHUNK, A1 = A0 + CHUNK < nA ? A0 + CHUNK : nA; \
        int lab[CHUNK][4];                                                     \
        int64_t p1[CHUNK][4], p2[CHUNK][4];                                    \
        for (int64_t A = A0; A < A1; A++) {                                    \
            const int16_t *v = quads + 4 * A;                                  \
            for (int s = 0; s < 4; s++) {                                      \
                const int q = v[s == 0], r = v[s <= 1 ? 2 : 1], u = v[s == 3 ? 2 : 3]; \
                const int64_t y1 = n - 1 - q, y2 = n - 1 - r;                  \
                lab[A - A0][s] = kv[v[s]];                                     \
                p1[A - A0][s] = posV[v[s]];                                    \
                p2[A - A0][s] = pos3[nA3 - 1 - y1 * (y1 - 1) * (y1 - 2) / 6    \
                                     - y2 * (y2 - 1) / 2 - (n - 1 - u)];       \
            }                                                                  \
        }                                                                      \
        for (int64_t e = 0; e < nent; e++) {                                   \
            T *Rr = R + row[e] * ldR;                                          \
            const int64_t *xo = xoff + e * ncls, *t1 = s1 + e * ncls, *t2 = s2 + e * ncls; \
            const double sg = sgn[e];                                          \
            for (int64_t A = A0; A < A1; A++) {                                \
                const int64_t c = A - A0;                                      \
                if (mode) {                                                    \
                    T acc = 0;                                                 \
                    for (int s = 0; s < 4; s++) {                              \
                        const int l = lab[c][s];                               \
                        if (xo[l] < 0) continue;                               \
                        const T x = X[xo[l] + p1[c][s] * t1[l] + p2[c][s] * t2[l]]; \
                        if (s & 1) acc -= x; else acc += x;                    \
                    }                                                          \
                    Rr[A] += sg * acc;                                         \
                } else {                                                       \
                    const T t = sg * Rr[A];                                    \
                    for (int s = 0; s < 4; s++) {                              \
                        const int l = lab[c][s];                               \
                        if (xo[l] < 0) continue;                               \
                        X[xo[l] + p1[c][s] * t1[l] + p2[c][s] * t2[l]] = (s & 1) ? -t : t; \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
void zc_t4s_x22_##SFX(int mode, T *R, int64_t ldR, const int16_t *quads,       \
        int64_t nA, int64_t nent, const int64_t *row, const double *sgn,       \
        const int64_t *xoff, const int64_t *s1, const int64_t *s2, int nq,     \
        int n, const int32_t *kv, const int32_t *posQ, T *X)                   \
{                                                                              \
    const int64_t nchunk = (nA + CHUNK - 1) / CHUNK;                           \
    _Pragma("omp parallel for schedule(dynamic, 4)")                           \
    for (int64_t ch = 0; ch < nchunk; ch++) {                                  \
        const int64_t A0 = ch * CHUNK, A1 = A0 + CHUNK < nA ? A0 + CHUNK : nA; \
        int lab[CHUNK][6];                                                     \
        int64_t p1[CHUNK][6], p2[CHUNK][6];                                    \
        for (int64_t A = A0; A < A1; A++) {                                    \
            const int16_t *v = quads + 4 * A;                                  \
            for (int s = 0; s < 6; s++) {                                      \
                const int p = v[G22[s][0]], q = v[G22[s][1]];                  \
                const int r = v[G22[s][2]], u = v[G22[s][3]];                  \
                lab[A - A0][s] = kv[p] + kv[q];                                \
                p1[A - A0][s] = posQ[(int64_t)p * n + q];                      \
                p2[A - A0][s] = posQ[(int64_t)r * n + u];                      \
            }                                                                  \
        }                                                                      \
        for (int64_t e = 0; e < nent; e++) {                                   \
            T *Rr = R + row[e] * ldR;                                          \
            const int64_t *xo = xoff + e * nq, *t1 = s1 + e * nq, *t2 = s2 + e * nq; \
            const double sg = sgn[e];                                          \
            for (int64_t A = A0; A < A1; A++) {                                \
                const int64_t c = A - A0;                                      \
                if (mode) {                                                    \
                    T acc = 0;                                                 \
                    for (int s = 0; s < 6; s++) {                              \
                        const int l = lab[c][s];                               \
                        if (xo[l] < 0) continue;                               \
                        acc += S22[s] * X[xo[l] + p1[c][s] * t1[l] + p2[c][s] * t2[l]]; \
                    }                                                          \
                    Rr[A] += sg * acc;                                         \
                } else {                                                       \
                    const T t = sg * Rr[A];                                    \
                    for (int s = 0; s < 6; s++) {                              \
                        const int l = lab[c][s];                               \
                        if (xo[l] < 0) continue;                               \
                        X[xo[l] + p1[c][s] * t1[l] + p2[c][s] * t2[l]] = S22[s] * t; \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
/* triples tri[nK, 3], the splits (a | bc): a at posV, the pair at posQ */    \
void zc_t4s_x12_##SFX(int mode, T *R, int64_t ldR, const int16_t *tri,         \
        int64_t nK, int64_t nent, const int64_t *row, const double *sgn,       \
        const int64_t *xoff, const int64_t *s1, const int64_t *s2, int ncls,   \
        int n, const int32_t *kv, const int32_t *posV, const int32_t *posQ,    \
        T *X)                                                                  \
{                                                                              \
    const int64_t nchunk = (nK + CHUNK - 1) / CHUNK;                           \
    _Pragma("omp parallel for schedule(dynamic, 4)")                           \
    for (int64_t ch = 0; ch < nchunk; ch++) {                                  \
        const int64_t A0 = ch * CHUNK, A1 = A0 + CHUNK < nK ? A0 + CHUNK : nK; \
        int lab[CHUNK][3];                                                     \
        int64_t p1[CHUNK][3], p2[CHUNK][3];                                    \
        for (int64_t A = A0; A < A1; A++) {                                    \
            const int16_t *v = tri + 3 * A;                                    \
            for (int s = 0; s < 3; s++) {                                      \
                const int q = v[s == 0], r = v[s == 2 ? 1 : 2];                \
                lab[A - A0][s] = kv[v[s]];                                     \
                p1[A - A0][s] = posV[v[s]];                                    \
                p2[A - A0][s] = posQ[(int64_t)q * n + r];                      \
            }                                                                  \
        }                                                                      \
        for (int64_t e = 0; e < nent; e++) {                                   \
            T *Rr = R + row[e] * ldR;                                          \
            const int64_t *xo = xoff + e * ncls, *t1 = s1 + e * ncls, *t2 = s2 + e * ncls; \
            const double sg = sgn[e];                                          \
            for (int64_t A = A0; A < A1; A++) {                                \
                const int64_t c = A - A0;                                      \
                if (mode) {                                                    \
                    T acc = 0;                                                 \
                    for (int s = 0; s < 3; s++) {                              \
                        const int l = lab[c][s];                               \
                        if (xo[l] < 0) continue;                               \
                        const T x = X[xo[l] + p1[c][s] * t1[l] + p2[c][s] * t2[l]]; \
                        if (s & 1) acc -= x; else acc += x;                    \
                    }                                                          \
                    Rr[A] += sg * acc;                                         \
                } else {                                                       \
                    const T t = sg * Rr[A];                                    \
                    for (int s = 0; s < 3; s++) {                              \
                        const int l = lab[c][s];                               \
                        if (xo[l] < 0) continue;                               \
                        X[xo[l] + p1[c][s] * t1[l] + p2[c][s] * t2[l]] = (s & 1) ? -t : t; \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
/* R[I, A] <- T[I, A] + R[I, A] / (eo[I] - sum of ev over the quadruple A) */ \
void zc_t4s_jacobi_##SFX(T *R, const T *T4, int64_t nI, const int16_t *quads,  \
        int64_t nA, const double *eo, const double *ev)                        \
{                                                                              \
    const int64_t nchunk = (nA + CHUNK - 1) / CHUNK;                           \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t ch = 0; ch < nchunk; ch++) {                                  \
        const int64_t A0 = ch * CHUNK, A1 = A0 + CHUNK < nA ? A0 + CHUNK : nA; \
        double d[CHUNK];                                                       \
        for (int64_t A = A0; A < A1; A++) {                                    \
            const int16_t *v = quads + 4 * A;                                  \
            d[A - A0] = ev[v[0]] + ev[v[1]] + ev[v[2]] + ev[v[3]];             \
        }                                                                      \
        for (int64_t i = 0; i < nI; i++) {                                     \
            T *r = R + i * nA;                                                 \
            const T *t = T4 + i * nA;                                          \
            for (int64_t A = A0; A < A1; A++)                                  \
                r[A] = t[A] + r[A] / (eo[i] - d[A - A0]);                      \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
/* the products tab[t] = (buffer of A, offset, lda, transposed, buffer of B,  \
 * offset, ldb, buffer of C, offset, ldc, m, n, k) of the groups              \
 * gstart[g] .. gstart[g + 1]: C += alpha[t] op(A) B, row-major; cshift is    \
 * added to the offsets of C */                                               \
void zc_t4s_tasks_##SFX(zc_gemm_fn fn, int64_t ngroup, const int64_t *gstart,  \
        const int64_t *tab, const double *alpha, void **bufs, int64_t cshift)  \
{                                                                              \
    _Pragma("omp parallel for schedule(dynamic, 1)")                           \
    for (int64_t g = 0; g < ngroup; g++)                                       \
        for (int64_t t = gstart[g]; t < gstart[g + 1]; t++) {                  \
            const int64_t *r = tab + 13 * t;                                   \
            const T al = alpha[t], one = 1;                                    \
            fn((int)r[3], 0, r[10], r[11], r[12], &al, (const T *)bufs[r[0]] + r[1], r[2], \
               (const T *)bufs[r[4]] + r[5], r[6], &one,                       \
               (T *)bufs[r[7]] + r[8] + cshift, r[9]);                         \
        }                                                                      \
}

DEFINE_T4S(double, d)
DEFINE_T4S(double complex, z)

/* ------------------------------------------------------------------------
 * Rank 5 (the 5p3h operators of EOM-DEA, zccsdt/cc/dea5_sector.py): the
 * quintuples a<b<c<d<e of one total label M as a list, and the moves
 * between a block of them and the operand or the result of a matrix product
 * on the splits
 *
 *     (ab | cde)  10 splits: the pair at posQ[a n + b], the triple at
 *                 pos3[rank of the triple] (zc_t5s_x23)
 *     (a | bcde)   5 splits: a at posV[a], the quadruple at
 *                 pos4[rank of the quadruple] (zc_t5s_x14)
 *
 * with the entries, offsets and strides of the kernels above (the label
 * of the first group selects the block of X).
 * ------------------------------------------------------------------------ */

int64_t zc_t5s_quints(int n, const int32_t *kv, int M, int16_t *out)
{
    int64_t cnt = 0;
    for (int a = 0; a < n - 4; a++)
    for (int b = a + 1; b < n - 3; b++)
    for (int c = b + 1; c < n - 2; c++)
    for (int d = c + 1; d < n - 1; d++) {
        const int ke = M - kv[a] - kv[b] - kv[c] - kv[d];
        if (ke < kv[d + 1] || ke > kv[n - 1]) continue;
        int lo = d + 1, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (kv[mid] >= ke) hi = mid; else lo = mid + 1;
        }
        for (int e = lo; e < n && kv[e] == ke; e++) {
            if (out) {
                int16_t *o = out + 5 * cnt;
                o[0] = (int16_t)a; o[1] = (int16_t)b; o[2] = (int16_t)c;
                o[3] = (int16_t)d; o[4] = (int16_t)e;
            }
            cnt++;
        }
    }
    return cnt;
}

/* the 10 splits (pair | triple) of 5 positions and their signs */
static const int G23[10][5] = {{0,1,2,3,4}, {0,2,1,3,4}, {0,3,1,2,4}, {0,4,1,2,3},
                               {1,2,0,3,4}, {1,3,0,2,4}, {1,4,0,2,3}, {2,3,0,1,4},
                               {2,4,0,1,3}, {3,4,0,1,2}};
static const double S23[10] = {1, -1, 1, -1, 1, -1, 1, 1, -1, 1};

#define DEFINE_T5S(T, SFX)                                                     \
                                                                               \
void zc_t5s_x23_##SFX(int mode, T *R, int64_t ldR, const int16_t *quints,      \
        int64_t nA, int64_t nent, const int64_t *row, const double *sgn,       \
        const int64_t *xoff, const int64_t *s1, const int64_t *s2, int nq,     \
        int n, const int32_t *kv, const int32_t *posQ, const int32_t *pos3,    \
        T *X)                                                                  \
{                                                                              \
    const int64_t nA3 = (int64_t)n * (n - 1) * (n - 2) / 6;                    \
    const int64_t nchunk = (nA + CHUNK - 1) / CHUNK;                           \
    _Pragma("omp parallel for schedule(dynamic, 4)")                           \
    for (int64_t ch = 0; ch < nchunk; ch++) {                                  \
        const int64_t A0 = ch * CHUNK, A1 = A0 + CHUNK < nA ? A0 + CHUNK : nA; \
        int lab[CHUNK][10];                                                    \
        int64_t p1[CHUNK][10], p2[CHUNK][10];                                  \
        for (int64_t A = A0; A < A1; A++) {                                    \
            const int16_t *v = quints + 5 * A;                                 \
            for (int s = 0; s < 10; s++) {                                     \
                const int p = v[G23[s][0]], q = v[G23[s][1]];                  \
                const int r = v[G23[s][2]], t = v[G23[s][3]], u = v[G23[s][4]]; \
                const int64_t y1 = n - 1 - r, y2 = n - 1 - t;                  \
                lab[A - A0][s] = kv[p] + kv[q];                                \
                p1[A - A0][s] = posQ[(int64_t)p * n + q];                      \
                p2[A - A0][s] = pos3[nA3 - 1 - y1 * (y1 - 1) * (y1 - 2) / 6    \
                                     - y2 * (y2 - 1) / 2 - (n - 1 - u)];       \
            }                                                                  \
        }                                                                      \
        for (int64_t e = 0; e < nent; e++) {                                   \
            T *Rr = R + row[e] * ldR;                                          \
            const int64_t *xo = xoff + e * nq, *t1 = s1 + e * nq, *t2 = s2 + e * nq; \
            const double sg = sgn[e];                                          \
            for (int64_t A = A0; A < A1; A++) {                                \
                const int64_t c = A - A0;                                      \
                if (mode) {                                                    \
                    T acc = 0;                                                 \
                    for (int s = 0; s < 10; s++) {                             \
                        const int l = lab[c][s];                               \
                        if (xo[l] < 0) continue;                               \
                        acc += S23[s] * X[xo[l] + p1[c][s] * t1[l] + p2[c][s] * t2[l]]; \
                    }                                                          \
                    Rr[A] += sg * acc;                                         \
                } else {                                                       \
                    const T t = sg * Rr[A];                                    \
                    for (int s = 0; s < 10; s++) {                             \
                        const int l = lab[c][s];                               \
                        if (xo[l] < 0) continue;                               \
                        X[xo[l] + p1[c][s] * t1[l] + p2[c][s] * t2[l]] = S23[s] * t; \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
void zc_t5s_x14_##SFX(int mode, T *R, int64_t ldR, const int16_t *quints,      \
        int64_t nA, int64_t nent, const int64_t *row, const double *sgn,       \
        const int64_t *xoff, const int64_t *s1, const int64_t *s2, int ncls,   \
        int n, const int32_t *kv, const int32_t *posV, const int32_t *pos4,    \
        T *X)                                                                  \
{                                                                              \
    const int64_t nA4 = (int64_t)n * (n - 1) * (n - 2) * (n - 3) / 24;         \
    const int64_t nchunk = (nA + CHUNK - 1) / CHUNK;                           \
    _Pragma("omp parallel for schedule(dynamic, 4)")                           \
    for (int64_t ch = 0; ch < nchunk; ch++) {                                  \
        const int64_t A0 = ch * CHUNK, A1 = A0 + CHUNK < nA ? A0 + CHUNK : nA; \
        int lab[CHUNK][5];                                                     \
        int64_t p1[CHUNK][5], p2[CHUNK][5];                                    \
        for (int64_t A = A0; A < A1; A++) {                                    \
            const int16_t *v = quints + 5 * A;                                 \
            for (int s = 0; s < 5; s++) {                                      \
                int w[4], k = 0;                                               \
                for (int x = 0; x < 5; x++) if (x != s) w[k++] = v[x];         \
                const int64_t y0 = n - 1 - w[0], y1 = n - 1 - w[1], y2 = n - 1 - w[2]; \
                lab[A - A0][s] = kv[v[s]];                                     \
                p1[A - A0][s] = posV[v[s]];                                    \
                p2[A - A0][s] = pos4[nA4 - 1 - y0 * (y0 - 1) * (y0 - 2) * (y0 - 3) / 24 \
                                     - y1 * (y1 - 1) * (y1 - 2) / 6 - y2 * (y2 - 1) / 2 \
                                     - (n - 1 - w[3])];                        \
            }                                                                  \
        }                                                                      \
        for (int64_t e = 0; e < nent; e++) {                                   \
            T *Rr = R + row[e] * ldR;                                          \
            const int64_t *xo = xoff + e * ncls, *t1 = s1 + e * ncls, *t2 = s2 + e * ncls; \
            const double sg = sgn[e];                                          \
            for (int64_t A = A0; A < A1; A++) {                                \
                const int64_t c = A - A0;                                      \
                if (mode) {                                                    \
                    T acc = 0;                                                 \
                    for (int s = 0; s < 5; s++) {                              \
                        const int l = lab[c][s];                               \
                        if (xo[l] < 0) continue;                               \
                        const T x = X[xo[l] + p1[c][s] * t1[l] + p2[c][s] * t2[l]]; \
                        if (s & 1) acc -= x; else acc += x;                    \
                    }                                                          \
                    Rr[A] += sg * acc;                                         \
                } else {                                                       \
                    const T t = sg * Rr[A];                                    \
                    for (int s = 0; s < 5; s++) {                              \
                        const int l = lab[c][s];                               \
                        if (xo[l] < 0) continue;                               \
                        X[xo[l] + p1[c][s] * t1[l] + p2[c][s] * t2[l]] = (s & 1) ? -t : t; \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}

DEFINE_T5S(double, d)
DEFINE_T5S(double complex, z)

/* zc_t5s_quints on all threads (the native program makes the lists of the
 * blocks of r5 when it uses them, cfour/xzccsdt/xzdea5.inc): the quintuples
 * of each first index a counted, then written at their places; the same
 * list as zc_t5s_quints.  cnt: work, n + 1 integers. */
static int64_t t5s_from(int n, const int32_t *kv, int M, int a, int16_t *out)
{
    int64_t c = 0;
    for (int b = a + 1; b < n - 3; b++)
    for (int cc = b + 1; cc < n - 2; cc++)
    for (int d = cc + 1; d < n - 1; d++) {
        const int ke = M - kv[a] - kv[b] - kv[cc] - kv[d];
        if (ke < kv[d + 1] || ke > kv[n - 1]) continue;
        int lo = d + 1, hi = n;
        while (lo < hi) {
            int mid = (lo + hi) / 2;
            if (kv[mid] >= ke) hi = mid; else lo = mid + 1;
        }
        for (int e = lo; e < n && kv[e] == ke; e++) {
            if (out) {
                int16_t *o = out + 5 * c;
                o[0] = (int16_t)a; o[1] = (int16_t)b; o[2] = (int16_t)cc;
                o[3] = (int16_t)d; o[4] = (int16_t)e;
            }
            c++;
        }
    }
    return c;
}

int64_t zc_t5s_quints_par(int n, const int32_t *kv, int M, int16_t *out, int64_t *cnt)
{
    if (n < 5) return 0;
    cnt[0] = 0;
    _Pragma("omp parallel for schedule(dynamic)")
    for (int a = 0; a < n - 4; a++) cnt[a + 1] = t5s_from(n, kv, M, a, NULL);
    for (int a = 0; a < n - 4; a++) cnt[a + 1] += cnt[a];
    if (out) {
        _Pragma("omp parallel for schedule(dynamic)")
        for (int a = 0; a < n - 4; a++) t5s_from(n, kv, M, a, out + 5 * cnt[a]);
    }
    return cnt[n - 4];
}
