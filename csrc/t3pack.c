/*
 * t3pack.c -- OpenMP gather / antisymmetrizing-reduction kernels for packed T3.
 *
 * Packed T3: T3p[I][A], I over occupied triples i<j<k, A over virtual
 * triples a<b<c (lexicographic).  A triple is written as (free index, pair)
 * in three ways r = 0, 1, 2 with signs +, -, + (see zccsdt/cc/t3pack.py).
 *
 * Every kernel exists for double (suffix _d) and double complex (suffix _z);
 * both are generated from one template below.  Index arrays are int64,
 * signs int8.  All arrays are C-contiguous.
 */

#include <complex.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define DEFINE_KERNELS(T, SFX)                                                 \
                                                                               \
/* out[P, d, m, Q] = so[P, m] * sv[d, Q] * T3p[orow[P, m], vtri[d, Q]]         \
 * orow/so: (nb, nocc); vtri/sv: (nvir, nvpair). */                            \
void zc_gather_half_##SFX(const T *T3p, int64_t nvtri,                         \
                          const int64_t *orow, const int8_t *so,               \
                          const int64_t *vtri, const int8_t *sv,               \
                          int64_t nb, int64_t nocc, int64_t nvir,              \
                          int64_t nvp, T *out)                                 \
{                                                                              \
    _Pragma("omp parallel for collapse(2) schedule(static)")                   \
    for (int64_t P = 0; P < nb; P++)                                           \
    for (int64_t d = 0; d < nvir; d++) {                                       \
        const int64_t *vt = vtri + d * nvp;                                    \
        const int8_t *vs = sv + d * nvp;                                       \
        for (int64_t m = 0; m < nocc; m++) {                                   \
            T *o = out + ((P * nvir + d) * nocc + m) * nvp;                    \
            int8_t s = so[P * nocc + m];                                       \
            if (s == 0) { memset(o, 0, sizeof(T) * nvp); continue; }           \
            const T *row = T3p + orow[P * nocc + m] * nvtri;                   \
            if (s > 0) for (int64_t Q = 0; Q < nvp; Q++) o[Q] = vs[Q] * row[vt[Q]]; \
            else       for (int64_t Q = 0; Q < nvp; Q++) o[Q] = -vs[Q] * row[vt[Q]]; \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
/* R[tgt[t], A] += sum_e esgn[t,e] * sum_s rs_s Z[eP[t,e], a_s(A), ex[t,e], Q_s(A)] \
 * for targets t (unique), up to 3 entries e each (eP < 0: unused).            \
 * Z: (nb, nvir, nocc, nvp).  vfree/vpair: (3, nvtri). */                      \
void zc_reduce_scatter_##SFX(T *R, int64_t nvtri, const T *Z,                  \
                             int64_t nocc, int64_t nvir, int64_t nvp,          \
                             int64_t ntgt, const int64_t *tgt,                 \
                             const int64_t *eP, const int64_t *ex,             \
                             const double *esgn,                               \
                             const int64_t *vfree, const int64_t *vpair)       \
{                                                                              \
    const int64_t sP = nvir * nocc * nvp, sa = nocc * nvp;                     \
    _Pragma("omp parallel for schedule(dynamic, 1)")                           \
    for (int64_t t = 0; t < ntgt; t++) {                                       \
        T *r = R + tgt[t] * nvtri;                                             \
        for (int e = 0; e < 3; e++) {                                          \
            int64_t P = eP[t * 3 + e];                                         \
            if (P < 0) continue;                                               \
            const T *z = Z + P * sP + ex[t * 3 + e] * nvp;                     \
            double sg = esgn[t * 3 + e];                                       \
            for (int64_t A = 0; A < nvtri; A++) {                              \
                T v = z[vfree[A] * sa + vpair[A]]                              \
                    - z[vfree[nvtri + A] * sa + vpair[nvtri + A]]              \
                    + z[vfree[2 * nvtri + A] * sa + vpair[2 * nvtri + A]];     \
                r[A] += sg * v;                                                \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
/* out[Q, I, c] = sv[c, Q] * T3p[rows[I], vtri[c, Q]]  (pair-major, so the  \
 * ladder GEMM W(Q', Q) @ out(Q, I c) needs no transpose) */                   \
void zc_gather_vir_half_##SFX(const T *T3p, int64_t nvtri,                     \
                              const int64_t *rows, int64_t nb,                 \
                              const int64_t *vtri, const int8_t *sv,           \
                              int64_t nvir, int64_t nvp, T *out)               \
{                                                                              \
    _Pragma("omp parallel for collapse(2) schedule(static)")                   \
    for (int64_t Q = 0; Q < nvp; Q++)                                          \
    for (int64_t ib = 0; ib < nb; ib++) {                                      \
        const T *row = T3p + rows[ib] * nvtri;                                 \
        T *o = out + (Q * nb + ib) * nvir;                                     \
        for (int64_t c = 0; c < nvir; c++)                                     \
            o[c] = sv[c * nvp + Q] * row[vtri[c * nvp + Q]];                   \
    }                                                                          \
}                                                                              \
                                                                               \
/* R[rows[I], A] += sum_s rs_s X[Q_s(A), I, c_s(A)],  X: (nvp, nb, nvir) */    \
void zc_reduce_pair_first_##SFX(T *R, int64_t nvtri, const T *X,               \
                                const int64_t *rows, int64_t nb,               \
                                int64_t nvir, int64_t nvp,                     \
                                const int64_t *vfree, const int64_t *vpair)    \
{                                                                              \
    const int64_t sQ = nb * nvir;                                              \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t ib = 0; ib < nb; ib++) {                                      \
        const T *x = X + ib * nvir;                                            \
        T *r = R + rows[ib] * nvtri;                                           \
        for (int64_t A = 0; A < nvtri; A++)                                    \
            r[A] += x[vpair[A] * sQ + vfree[A]]                                \
                  - x[vpair[nvtri + A] * sQ + vfree[nvtri + A]]                \
                  + x[vpair[2 * nvtri + A] * sQ + vfree[2 * nvtri + A]];       \
    }                                                                          \
}

DEFINE_KERNELS(double, d)
DEFINE_KERNELS(double complex, z)

/* ---- large-vector helpers for DIIS (OpenMP, double and complex) ---- */

#define DEFINE_VECOPS(T, SFX, CONJ)                                            \
/* out = a - b */                                                              \
void zc_vsub_##SFX(int64_t n, const T *a, const T *b, T *out)                  \
{                                                                              \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t i = 0; i < n; i++) out[i] = a[i] - b[i];                      \
}                                                                              \
/* *res = sum conj(a) b, per-thread partial sums (deterministic order) */      \
void zc_vdot_##SFX(int64_t n, const T *a, const T *b, T *res)                  \
{                                                                              \
    T part[256] = {0};                                                         \
    int nt = 1;                                                                \
    _Pragma("omp parallel")                                                    \
    {                                                                          \
        int id = omp_get_thread_num();                                         \
        _Pragma("omp single")                                                  \
        nt = omp_get_num_threads();                                            \
        T acc = 0;                                                             \
        _Pragma("omp for schedule(static)")                                    \
        for (int64_t i = 0; i < n; i++) acc += CONJ(a[i]) * b[i];              \
        if (id < 256) part[id] = acc;                                          \
    }                                                                          \
    T s = 0;                                                                   \
    for (int t = 0; t < nt && t < 256; t++) s += part[t];                      \
    *res = s;                                                                  \
}                                                                              \
/* out = sum_k c[k] * x[k] for k < nk (x: array of pointers) */                \
void zc_vlincomb_##SFX(int64_t n, int64_t nk, const T **x, const T *c, T *out) \
{                                                                              \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t i = 0; i < n; i++) {                                          \
        T s = 0;                                                               \
        for (int64_t k = 0; k < nk; k++) s += c[k] * x[k][i];                  \
        out[i] = s;                                                            \
    }                                                                          \
}

#include <omp.h>
#define NOCONJ(x) (x)
DEFINE_VECOPS(double, d, NOCONJ)
DEFINE_VECOPS(double complex, z, conj)

/* ---- antisymmetric pair packing of a 4-index intermediate ----
 * out[Q, x, y] += alpha * (X[a sa + b sb + x sx + y sy] - X[b sa + a sb + x sx + y sy])
 * for the pairs Q = (pa[Q], pb[Q]) (any enumeration), x < nx, y < ny.
 * Strides are in elements; out is C-contiguous (npair, nx, ny). */
#define DEFINE_PAIRPACK(T, SFX)                                                \
void zc_pair_antisym_add_##SFX(T *out, const T *X, int64_t npair,              \
                               const int64_t *pa, const int64_t *pb,           \
                               int64_t nx, int64_t ny, int64_t sa, int64_t sb, \
                               int64_t sx, int64_t sy, const T *alpha)         \
{                                                                              \
    const T al = *alpha;                                                       \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t Q = 0; Q < npair; Q++) {                                      \
        const int64_t a = pa[Q], b = pb[Q];                                    \
        T *o = out + Q * nx * ny;                                              \
        const T *p = X + a * sa + b * sb, *q = X + b * sa + a * sb;            \
        for (int64_t x = 0; x < nx; x++)                                       \
            for (int64_t y = 0; y < ny; y++)                                   \
                o[x * ny + y] += al * (p[x * sx + y * sy] - q[x * sx + y * sy]); \
    }                                                                          \
}
DEFINE_PAIRPACK(double, d)
DEFINE_PAIRPACK(double complex, z)

/* ---- packed-pair helpers for the T1 dressing of packed vvvv / vvvo ---- */

#define DEFINE_PAIRHELP(T, SFX)                                                \
/* out[Q, :] += alpha * (Y[a - x0, b, :] [a in batch]                          \
 *                     - Y[b - x0, a, :] [b in batch]),  (a, b) = pair Q.      \
 * Y: (x1 - x0, n, ncol) holds the rows of the first index x0 <= x < x1. */    \
void zc_pair_antisym_batch_add_##SFX(T *out, const T *Y, int64_t n,            \
                                     int64_t npair, const int64_t *pa,         \
                                     const int64_t *pb, int64_t ncol,          \
                                     int64_t x0, int64_t x1, const T *alpha)   \
{                                                                              \
    const T al = *alpha;                                                       \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t Q = 0; Q < npair; Q++) {                                      \
        const int64_t a = pa[Q], b = pb[Q];                                    \
        int ina = (a >= x0 && a < x1), inb = (b >= x0 && b < x1);              \
        if (!ina && !inb) continue;                                            \
        T *o = out + Q * ncol;                                                 \
        if (ina) {                                                             \
            const T *y = Y + ((a - x0) * n + b) * ncol;                        \
            for (int64_t k = 0; k < ncol; k++) o[k] += al * y[k];              \
        }                                                                      \
        if (inb) {                                                             \
            const T *y = Y + ((b - x0) * n + a) * ncol;                        \
            for (int64_t k = 0; k < ncol; k++) o[k] -= al * y[k];              \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
/* out[Q, e, i] += sum_{f != e} s(e,f) V[Q, pidx(min(e,f), max(e,f))] t[i, f]  \
 * (s = +1 for e < f, -1 for e > f), i.e. sum_f V[Q,e,f] t[i,f] for V          \
 * antisymmetric in (e, f) and packed with the pair map pidx (n x n).          \
 * V: (nq, npair), t: (no, n), out: (nq, n, no). */                            \
void zc_pair_t1_contract_##SFX(T *out, const T *V, const T *t, int64_t nq,     \
                               int64_t n, int64_t npair, const int64_t *pidx,  \
                               int64_t no)                                     \
{                                                                              \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t Q = 0; Q < nq; Q++) {                                         \
        const T *v = V + Q * npair;                                            \
        T *o = out + Q * n * no;                                               \
        for (int64_t e = 0; e < n; e++) {                                      \
            T *oe = o + e * no;                                                \
            for (int64_t f = 0; f < n; f++) {                                  \
                if (f == e) continue;                                          \
                const int64_t ix = (f > e) ? pidx[e * n + f] : pidx[f * n + e];\
                if (ix < 0) continue;              /* pair outside the block */\
                T x = (f > e) ? v[ix] : -v[ix];                                \
                for (int64_t i = 0; i < no; i++) oe[i] += x * t[i * n + f];    \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}
DEFINE_PAIRHELP(double, d)
DEFINE_PAIRHELP(double complex, z)

/* ---- vvvv x t1 on spin-blocked packed pairs ----
 * out[Q, e, i] += sum_f V[Q, (e,f)] t[i, f] for V antisymmetric in (e, f);
 * t conserves spin, so each pair block couples one virtual and one occupied
 * spin.  Output elements are out[Q * ldq + e * lde + i].
 *
 * Same-spin block: V (nq, n(n-1)/2), pairs e<f in triu order, t (no, n).
 * The row segment V[Q, (e, e+1..n-1)] is contiguous: the e-term is a dot
 * product, the f-term an axpy into a transposed scratch S (no, n). */
#define DEFINE_VT1(T, SFX)                                                     \
void zc_vt1_same_##SFX(T *out, const T *V, const T *t, int64_t nq, int64_t n,  \
                       int64_t no, int64_t ldq, int64_t lde)                   \
{                                                                              \
    const int64_t np = n * (n - 1) / 2;                                        \
    _Pragma("omp parallel")                                                    \
    {                                                                          \
        T *S = malloc(sizeof(T) * (no * n + 1));                               \
        _Pragma("omp for schedule(static)")                                    \
        for (int64_t Q = 0; Q < nq; Q++) {                                     \
            const T *v = V + Q * np;                                           \
            T *o = out + Q * ldq;                                              \
            memset(S, 0, sizeof(T) * no * n);                                  \
            for (int64_t e = 0; e + 1 < n; e++) {                              \
                const T *ve = v + e * n - e * (e + 1) / 2 - (e + 1);           \
                for (int64_t i = 0; i < no; i++) {                             \
                    const T *ti = t + i * n;                                   \
                    const T te = ti[e];                                        \
                    T *Si = S + i * n;                                         \
                    T acc = 0;                                                 \
                    _Pragma("omp simd reduction(+:acc)")                       \
                    for (int64_t f = e + 1; f < n; f++) {                      \
                        acc += ve[f] * ti[f];                                  \
                        Si[f] -= te * ve[f];                                   \
                    }                                                          \
                    o[e * lde + i] += acc;                                     \
                }                                                              \
            }                                                                  \
            for (int64_t f = 0; f < n; f++)                                    \
                for (int64_t i = 0; i < no; i++) o[f * lde + i] += S[i * n + f]; \
        }                                                                      \
        free(S);                                                               \
    }                                                                          \
}                                                                              \
                                                                               \
/* Alpha-beta block: V (nq, na, nb) = V[Q, (e alpha, f beta)], ta (noa, na),   \
 * tb (nob, nb):                                                               \
 *   oa[Q * ldq + e * lde + j] += sum_f V[Q, e, f] tb[j, f]  (e alpha, j beta) \
 *   ob[Q * ldq + f * lde + i] -= sum_e V[Q, e, f] ta[i, e]  (f beta, i alpha) \
 */                                                                            \
void zc_vt1_ab_##SFX(T *oa, T *ob, const T *V, const T *ta, const T *tb,       \
                     int64_t nq, int64_t na, int64_t nb, int64_t noa,          \
                     int64_t nob, int64_t ldq, int64_t lde)                    \
{                                                                              \
    _Pragma("omp parallel")                                                    \
    {                                                                          \
        T *S = malloc(sizeof(T) * (noa * nb + 1));                             \
        _Pragma("omp for schedule(static)")                                    \
        for (int64_t Q = 0; Q < nq; Q++) {                                     \
            const T *v = V + Q * na * nb;                                      \
            T *pa = oa + Q * ldq, *pb = ob + Q * ldq;                          \
            memset(S, 0, sizeof(T) * noa * nb);                                \
            for (int64_t e = 0; e < na; e++) {                                 \
                const T *ve = v + e * nb;                                      \
                for (int64_t j = 0; j < nob; j++) {                            \
                    const T *tj = tb + j * nb;                                 \
                    T acc = 0;                                                 \
                    _Pragma("omp simd reduction(+:acc)")                       \
                    for (int64_t f = 0; f < nb; f++) acc += ve[f] * tj[f];     \
                    pa[e * lde + j] += acc;                                    \
                }                                                              \
                for (int64_t i = 0; i < noa; i++) {                            \
                    const T x = ta[i * na + e];                                \
                    T *Si = S + i * nb;                                        \
                    _Pragma("omp simd")                                        \
                    for (int64_t f = 0; f < nb; f++) Si[f] -= x * ve[f];       \
                }                                                              \
            }                                                                  \
            for (int64_t f = 0; f < nb; f++)                                   \
                for (int64_t i = 0; i < noa; i++) pb[f * lde + i] += S[i * nb + f]; \
        }                                                                      \
        free(S);                                                               \
    }                                                                          \
}
DEFINE_VT1(double, d)
DEFINE_VT1(double complex, z)

/* out[r * ldo + c] += alpha * in[c * ldi + r] for r < nr, c < nc: a blocked
 * transpose-add (both operands row-major). */
#define DEFINE_ADDT(T, SFX)                                                    \
void zc_add_transposed_##SFX(T *out, const T *in, int64_t nr, int64_t nc,      \
                             int64_t ldo, int64_t ldi, const T *alpha)         \
{                                                                              \
    const T al = *alpha;                                                       \
    const int64_t B = 32;                                                      \
    _Pragma("omp parallel for collapse(2) schedule(static)")                   \
    for (int64_t r0 = 0; r0 < nr; r0 += B)                                     \
    for (int64_t c0 = 0; c0 < nc; c0 += B) {                                   \
        const int64_t r1 = r0 + B < nr ? r0 + B : nr;                          \
        const int64_t c1 = c0 + B < nc ? c0 + B : nc;                          \
        for (int64_t r = r0; r < r1; r++) {                                    \
            T *o = out + r * ldo;                                              \
            for (int64_t c = c0; c < c1; c++) o[c] += al * in[c * ldi + r];    \
        }                                                                      \
    }                                                                          \
}
DEFINE_ADDT(double, d)
DEFINE_ADDT(double complex, z)

/* out = beta * out + alpha * in over a 4-index box n[0..3], both operands
 * addressed through element strides os / is (any layout, e.g. transposed
 * or sliced numpy views); beta == 0 overwrites out without reading it.
 * Parallel over the first two indices; the last index is the inner loop. */
#define DEFINE_AXPBY4(T, SFX)                                                  \
void zc_axpby4_##SFX(T *out, const T *in, const int64_t *n, const int64_t *os, \
                     const int64_t *is, const T *alpha, const T *beta)         \
{                                                                              \
    const T al = *alpha, be = *beta;                                           \
    const int64_t n0 = n[0], n1 = n[1], n2 = n[2], n3 = n[3];                  \
    const int64_t o3 = os[3], i3 = is[3];                                      \
    _Pragma("omp parallel for collapse(2) schedule(static)")                   \
    for (int64_t a = 0; a < n0; a++)                                           \
    for (int64_t b = 0; b < n1; b++) {                                         \
        T *o = out + a * os[0] + b * os[1];                                    \
        const T *p = in + a * is[0] + b * is[1];                               \
        for (int64_t c = 0; c < n2; c++) {                                     \
            T *oc = o + c * os[2];                                             \
            const T *pc = p + c * is[2];                                       \
            if (be == 0)                                                       \
                for (int64_t d = 0; d < n3; d++) oc[d * o3] = al * pc[d * i3]; \
            else                                                               \
                for (int64_t d = 0; d < n3; d++)                               \
                    oc[d * o3] = be * oc[d * o3] + al * pc[d * i3];            \
        }                                                                      \
    }                                                                          \
}
DEFINE_AXPBY4(double, d)
DEFINE_AXPBY4(double complex, z)

/* T1/T2-dressed packed vvvv pair block in one pass over the bare block:
 *   out[q, c] = bare[q, c] + sum_m tA[m, pa[q]] GA[m, pb[q], c]
 *                          + sum_m tB[m, pb[q]] GB[m, pa[q], c]
 *                          + sum_P CT[q, P] H[P, c]
 * tA (nmA, ldA), GA (nmA, nxA, ncol), tB (nmB, ldB), GB (nmB, nxB, ncol),
 * CT (nrow, nP), H (nP, ncol); pa / pb are local orbital indices.  Column
 * tiles keep the G / H tiles in cache while the rows stream; tasks are
 * (column tile, row chunk) pairs.  out may alias bare. */
#define DRESS_TB 256
#define DRESS_RB 512
#define DEFINE_VVVV_DRESS(T, SFX)                                              \
void zc_vvvv_dress_##SFX(T *out, const T *bare, int64_t nrow, int64_t ncol,    \
                         const int64_t *pa, const int64_t *pb,                 \
                         const T *tA, int64_t ldA, const T *GA, int64_t nmA,   \
                         int64_t nxA,                                          \
                         const T *tB, int64_t ldB, const T *GB, int64_t nmB,   \
                         int64_t nxB,                                          \
                         const T *CT, const T *H, int64_t nP)                  \
{                                                                              \
    const int64_t nct = (ncol + DRESS_TB - 1) / DRESS_TB;                      \
    const int64_t nrc = (nrow + DRESS_RB - 1) / DRESS_RB;                      \
    _Pragma("omp parallel for collapse(2) schedule(dynamic, 1)")               \
    for (int64_t ct = 0; ct < nct; ct++)                                       \
    for (int64_t rc = 0; rc < nrc; rc++) {                                     \
        const int64_t c0 = ct * DRESS_TB;                                      \
        const int64_t nc = (c0 + DRESS_TB < ncol ? c0 + DRESS_TB : ncol) - c0; \
        const int64_t q0 = rc * DRESS_RB;                                      \
        const int64_t q1 = q0 + DRESS_RB < nrow ? q0 + DRESS_RB : nrow;        \
        T acc[DRESS_TB];                                                       \
        for (int64_t q = q0; q < q1; q++) {                                    \
            const T *b = bare + q * ncol + c0;                                 \
            for (int64_t c = 0; c < nc; c++) acc[c] = b[c];                    \
            const int64_t a = pa[q], bb = pb[q];                               \
            for (int64_t m = 0; m < nmA; m++) {                                \
                const T x = tA[m * ldA + a];                                   \
                const T *g = GA + (m * nxA + bb) * ncol + c0;                  \
                _Pragma("omp simd")                                            \
                for (int64_t c = 0; c < nc; c++) acc[c] += x * g[c];           \
            }                                                                  \
            for (int64_t m = 0; m < nmB; m++) {                                \
                const T x = tB[m * ldB + bb];                                  \
                const T *g = GB + (m * nxB + a) * ncol + c0;                   \
                _Pragma("omp simd")                                            \
                for (int64_t c = 0; c < nc; c++) acc[c] += x * g[c];           \
            }                                                                  \
            const T *cq = CT + q * nP;                                         \
            for (int64_t P = 0; P < nP; P++) {                                 \
                const T x = cq[P];                                             \
                const T *h = H + P * ncol + c0;                                \
                _Pragma("omp simd")                                            \
                for (int64_t c = 0; c < nc; c++) acc[c] += x * h[c];           \
            }                                                                  \
            T *o = out + q * ncol + c0;                                        \
            for (int64_t c = 0; c < nc; c++) o[c] = acc[c];                    \
        }                                                                      \
    }                                                                          \
}
DEFINE_VVVV_DRESS(double, d)
DEFINE_VVVV_DRESS(double complex, z)

/* One-body operator on both indices of packed pairs:
 *   W[(a<b), (d<e)] += scale * (F_ad d_be - F_bd d_ae - F_ae d_bd + F_be d_ad)
 * for the row pairs (pa[q], pb[q]) of one pair block; pidx (n, n) is the
 * block-local pair index (-1 outside the block), F (n, ldf). */
#define DEFINE_ONEBODY(T, SFX)                                                 \
void zc_pair_one_body_add_##SFX(T *W, int64_t nq, int64_t ncol,                \
                                const int64_t *pa, const int64_t *pb,          \
                                const int64_t *pidx, int64_t n, const T *F,    \
                                int64_t ldf, const T *scale)                   \
{                                                                              \
    const T s = *scale;                                                        \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t q = 0; q < nq; q++) {                                         \
        const int64_t a = pa[q], b = pb[q];                                    \
        T *w = W + q * ncol;                                                   \
        for (int64_t y = 0; y < n; y++) {     /* acting on a: pair {y, b} */   \
            if (y == b) continue;                                              \
            const int64_t P = y < b ? pidx[y * n + b] : pidx[b * n + y];       \
            if (P < 0) continue;                                               \
            const T x = s * F[a * ldf + y];                                    \
            w[P] += y < b ? x : -x;                                            \
        }                                                                      \
        for (int64_t y = 0; y < n; y++) {     /* acting on b: pair {y, a} */   \
            if (y == a) continue;                                              \
            const int64_t P = y < a ? pidx[y * n + a] : pidx[a * n + y];       \
            if (P < 0) continue;                                               \
            const T x = s * F[b * ldf + y];                                    \
            w[P] += y < a ? -x : x;                                            \
        }                                                                      \
    }                                                                          \
}
DEFINE_ONEBODY(double, d)
DEFINE_ONEBODY(double complex, z)

/* Spin-flip symmetrization X <- X + flip(X) of a strided 3-index array,
 *   flip(X)[i, j, k] = s0[i] s1[j] s2[k] X[f0[i], f1[j], f2[k]],
 * in place.  The f's are involutions with flip-invariant signs; rep lists one
 * index of axis 0 per orbit of f0 (f0 maps rep outside rep), so every pair of
 * elements {x, flip(x)} is visited once:  x <- x + s y,  y <- s x_new. */
#define DEFINE_FLIPSYM(T, SFX)                                                 \
void zc_flip_sym3_##SFX(T *X, const int64_t *n, const int64_t *st,             \
                        const int64_t *f0, const int64_t *f1, const int64_t *f2, \
                        const double *s0, const double *s1, const double *s2,  \
                        const int64_t *rep, int64_t nrep)                      \
{                                                                              \
    const int64_t n1 = n[1], n2 = n[2], st2 = st[2];                           \
    _Pragma("omp parallel for collapse(2) schedule(static)")                   \
    for (int64_t ir = 0; ir < nrep; ir++)                                      \
    for (int64_t j = 0; j < n1; j++) {                                         \
        const int64_t i = rep[ir];                                             \
        const double sij = s0[i] * s1[j];                                      \
        T *x = X + i * st[0] + j * st[1];                                      \
        T *y = X + f0[i] * st[0] + f1[j] * st[1];                              \
        for (int64_t k = 0; k < n2; k++) {                                     \
            const double sg = sij * s2[k];                                     \
            T *xk = x + k * st2, *yk = y + f2[k] * st2;                        \
            const T v = *xk + sg * *yk;                                        \
            *xk = v;                                                           \
            *yk = sg * v;                                                      \
        }                                                                      \
    }                                                                          \
}
DEFINE_FLIPSYM(double, d)
DEFINE_FLIPSYM(double complex, z)

/* Gather of half-expanded spin-blocked T3 with precombined virtual tables:
 *   out[p * ldP + (id * nm + im) * nQ + q] = t3[m, P_0, P_1, d, Q_0, Q_1]
 *     = so * osig[I] * vcs[d, Q] * T3b[obase[I] + vci[d, Q]],
 * I = otri[m, P], Q = Qlo + q; vci = vcol[vtri] (int32) and vcs = vsgn *
 * vsig[vtri] (int8) are precombined per (d, Q) so the inner loop reads them
 * sequentially.  Tasks are (p, im): one T3 row per task, streamed for all d. */
#define DEFINE_BGATHER2(T, SFX)                                                \
void zc_bgather2_##SFX(const T *T3b, const int64_t *obase, const double *osig, \
                       const int64_t *otri, const int8_t *osgn, int64_t npo,   \
                       const int32_t *vci, const int8_t *vcs, int64_t nvp,     \
                       int64_t nP, const int64_t *Pidx, int64_t nd,            \
                       const int64_t *didx, int64_t nm, const int64_t *midx,   \
                       int64_t Qlo, int64_t nQ, int64_t ldP, T *out)           \
{                                                                              \
    _Pragma("omp parallel for collapse(2) schedule(dynamic, 1)")               \
    for (int64_t p = 0; p < nP; p++)                                           \
    for (int64_t im = 0; im < nm; im++) {                                      \
        const int64_t P = Pidx[p], m = midx[im];                               \
        const int8_t so = osgn[m * npo + P];                                   \
        const int64_t ti = so ? otri[m * npo + P] : 0;                         \
        const T *row = T3b + (so ? obase[ti] : 0);                             \
        const double f = so ? so * osig[ti] : 0.0;                             \
        for (int64_t id = 0; id < nd; id++) {                                  \
            T *o = out + p * ldP + (id * nm + im) * nQ;                        \
            if (so == 0) { memset(o, 0, sizeof(T) * nQ); continue; }           \
            const int64_t d = didx[id];                                        \
            const int32_t *ci = vci + d * nvp + Qlo;                           \
            const int8_t *cs = vcs + d * nvp + Qlo;                            \
            for (int64_t q = 0; q < nQ; q++) o[q] = (f * cs[q]) * row[ci[q]];  \
        }                                                                      \
    }                                                                          \
}
DEFINE_BGATHER2(double, d)
DEFINE_BGATHER2(double complex, z)

/* Pass-2 reduction, loops inverted: the targets (it, r) of one pair batch
 * whose pair P = pair_r(it) is in the batch and whose occupied triple it
 * belongs to virtual group g are listed in (tbase = Rbase[it], tx = free_r(it),
 * tP = P - p0, tsg = rs_r).  For every A in [A0, A1) (threads over chunks of
 * A) the virtual decompositions are loaded once and applied to all targets:
 *   R[tbase + Rcol[A]] += tsg sum_s rs_s Z[zoff[tP*3 + gQ] + rowpos[a, x] nQg[gQ]
 *                                          + (Q - Qoff[gQ])],
 * a = free_s(A), Q = pair_s(A), gQ = vpgrp[Q]. */
#define RED_CH 128
#define DEFINE_BREDUCE2(T, SFX)                                                \
void zc_breduce2_##SFX(T *R, const int64_t *Rcol, int64_t ntg,                 \
                       const int64_t *tbase, const int64_t *tx,                \
                       const int64_t *tP, const double *tsg,                   \
                       int64_t A0, int64_t A1, int64_t nvt,                    \
                       const int64_t *vfree, const int64_t *vpair,             \
                       const int64_t *vpgrp, const int64_t *zoff,              \
                       const int64_t *nQg, const int64_t *Qoff,                \
                       const int64_t *rowpos, int64_t no, const T *Z)          \
{                                                                              \
    const int64_t nch = (A1 - A0 + RED_CH - 1) / RED_CH;                       \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t ch = 0; ch < nch; ch++) {                                     \
        const int64_t a_lo = A0 + ch * RED_CH;                                 \
        const int64_t a_hi = a_lo + RED_CH < A1 ? a_lo + RED_CH : A1;          \
        int64_t ra[3][RED_CH], zq[3][RED_CH], gg[3][RED_CH];                   \
        for (int64_t A = a_lo; A < a_hi; A++) {                                \
            for (int s = 0; s < 3; s++) {                                      \
                const int64_t Q = vpair[s * nvt + A], g = vpgrp[Q];            \
                ra[s][A - a_lo] = vfree[s * nvt + A] * no;                     \
                gg[s][A - a_lo] = g;                                           \
                zq[s][A - a_lo] = Q - Qoff[g];                                 \
            }                                                                  \
        }                                                                      \
        for (int64_t t = 0; t < ntg; t++) {                                    \
            const int64_t x = tx[t];                                           \
            const int64_t *zo = zoff + tP[t] * 3;                              \
            const double sg = tsg[t];                                          \
            T *r = R + tbase[t];                                               \
            for (int64_t k = 0; k < a_hi - a_lo; k++) {                        \
                const int64_t g0 = gg[0][k], g1 = gg[1][k], g2 = gg[2][k];     \
                const T v = Z[zo[g0] + rowpos[ra[0][k] + x] * nQg[g0] + zq[0][k]] \
                          - Z[zo[g1] + rowpos[ra[1][k] + x] * nQg[g1] + zq[1][k]] \
                          + Z[zo[g2] + rowpos[ra[2][k] + x] * nQg[g2] + zq[2][k]]; \
                r[Rcol[a_lo + k]] += sg * v;                                   \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}
DEFINE_BREDUCE2(double, d)
DEFINE_BREDUCE2(double complex, z)

/* ---- spin-adapted (closed-shell) T3 cubes, zccsdt/sa ----
 * Remove the redundant component of each v^3 cube in place:
 * r[x] -= (1/6) sum_sigma r[sigma(x)] over the six permutations of (a,b,c). */
#define DEFINE_SA_NULL(T, SFX)                                                 \
void zc_sa_project_null_##SFX(T *r, int64_t nb, int64_t v)                     \
{                                                                              \
    const int64_t v2 = v * v, v3 = v2 * v;                                     \
    _Pragma("omp parallel for collapse(2) schedule(dynamic, 4)")               \
    for (int64_t b = 0; b < nb; b++)                                           \
    for (int64_t a = 0; a < v; a++) {                                          \
        T *c = r + b * v3;                                                     \
        for (int64_t bb = a; bb < v; bb++)                                     \
        for (int64_t cc = bb; cc < v; cc++) {                                  \
            int64_t p[6] = {a * v2 + bb * v + cc, a * v2 + cc * v + bb,        \
                            bb * v2 + a * v + cc, bb * v2 + cc * v + a,        \
                            cc * v2 + a * v + bb, cc * v2 + bb * v + a};       \
            T s = 0;                                                           \
            for (int k = 0; k < 6; k++) s += c[p[k]];                          \
            s *= (1.0 / 6.0);                                                  \
            /* subtract once from every distinct position */                  \
            for (int k = 0; k < 6; k++) {                                      \
                int dup = 0;                                                   \
                for (int l = 0; l < k; l++) if (p[l] == p[k]) dup = 1;         \
                if (!dup) c[p[k]] -= s;                                        \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
/* Pair packing of views of cubes for the pp ladder.  View w is             \
 * X_w[d, e, x] = src[off[w] + d*sd[w] + e*se[w] + x*sx[w]] (x < nx); its      \
 * columns col0 = w*nx .. in the outputs (leading dimension ld):              \
 *   S[(d<=e), col] = X[d,e,x] + X[e,d,x]  (X[d,d,x] on the diagonal)          \
 *   A[(d<e), col]  = X[d,e,x] - X[e,d,x]                                      \
 * pairs in numpy.triu_indices order (k=0 for S, k=1 for A). */               \
void zc_sa_pair_pack_##SFX(const T *src, int64_t nview, const int64_t *off,    \
                           const int64_t *sd, const int64_t *se,               \
                           const int64_t *sx, int64_t v, int64_t nx,           \
                           T *S, int64_t ld, T *A, int64_t lda)                \
{                                                                              \
    _Pragma("omp parallel for collapse(2) schedule(static)")                   \
    for (int64_t w = 0; w < nview; w++)                                        \
    for (int64_t d = 0; d < v; d++) {                                          \
        const T *X = src + off[w];                                             \
        const int64_t col0 = w * nx;                                           \
        const int64_t ps = d * v - d * (d - 1) / 2;    /* first (d,d) in S */  \
        const int64_t pa = d * v - d * (d + 1) / 2;    /* first (d,d+1) in A */\
        for (int64_t e = d; e < v; e++) {                                      \
            const T *xde = X + d * sd[w] + e * se[w];                          \
            const T *xed = X + e * sd[w] + d * se[w];                          \
            T *srow = S + (ps + e - d) * ld + col0;                            \
            if (e == d) {                                                      \
                for (int64_t x = 0; x < nx; x++) srow[x] = xde[x * sx[w]];      \
            } else if (A == NULL) {                                            \
                for (int64_t x = 0; x < nx; x++)                               \
                    srow[x] = xde[x * sx[w]] + xed[x * sx[w]];                 \
            } else {                                                           \
                T *arow = A + (pa + e - d - 1) * lda + col0;                   \
                for (int64_t x = 0; x < nx; x++) {                             \
                    const T p = xde[x * sx[w]], q = xed[x * sx[w]];            \
                    srow[x] = p + q;                                           \
                    arow[x] = p - q;                                           \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
/* Inverse of zc_sa_pair_pack for the results: Y_w[a,b,x] += S + A (a<b),     \
 * S - A at (b,a), S at (a,a); Y_w addressed like X_w in zc_sa_pair_pack.     \
 * Views must not overlap each other (each element written by one view). */  \
void zc_sa_pair_unpack_add_##SFX(T *dst, int64_t nview, const int64_t *off,    \
                                 const int64_t *sd, const int64_t *se,         \
                                 const int64_t *sx, int64_t v, int64_t nx,     \
                                 const T *S, int64_t ld, const T *A,           \
                                 int64_t lda, const int64_t *cols)             \
{                                                                              \
    for (int64_t w = 0; w < nview; w++) {                                      \
        T *Y = dst + off[w];                                                   \
        const int64_t col0 = cols ? cols[w] * nx : w * nx;                     \
        _Pragma("omp parallel for schedule(static)")                           \
        for (int64_t a = 0; a < v; a++) {                                      \
            const int64_t ps = a * v - a * (a - 1) / 2;                        \
            const int64_t pa = a * v - a * (a + 1) / 2;                        \
            for (int64_t b = a; b < v; b++) {                                  \
                const T *srow = S + (ps + b - a) * ld + col0;                  \
                T *yab = Y + a * sd[w] + b * se[w];                            \
                if (b == a) {                                                  \
                    for (int64_t x = 0; x < nx; x++) yab[x * sx[w]] += srow[x]; \
                    continue;                                                  \
                }                                                              \
                T *yba = Y + b * sd[w] + a * se[w];                            \
                if (A == NULL) {                                               \
                    for (int64_t x = 0; x < nx; x++) {                         \
                        yab[x * sx[w]] += srow[x];                             \
                        yba[x * sx[w]] += srow[x];                             \
                    }                                                          \
                    continue;                                                  \
                }                                                              \
                const T *arow = A + (pa + b - a - 1) * lda + col0;             \
                for (int64_t x = 0; x < nx; x++) {                             \
                    yab[x * sx[w]] += srow[x] + arow[x];                       \
                    yba[x * sx[w]] += srow[x] - arow[x];                       \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}
DEFINE_SA_NULL(double, d)
DEFINE_SA_NULL(double complex, z)

/* out[a, b, e, i] += sum_f W_ab^ef t[i, f] for W given by its (e,f)
 * symmetric part Ws[(a<=b), (e<=f)] and antisymmetric part Wa[(a<b), (e<f)]
 * (pairs in numpy.triu_indices order): W_ab^ef = S_(ab)(ef) + A_(ab)(ef) and
 * W_ba^ef = S - A, S symmetric and A antisymmetric in (e,f).  out (v,v,v,no)
 * contiguous, t (no, v).  One pass over the rows of Ws / Wa. */
#define DEFINE_SA_VT1(T, SFX)                                                  \
void zc_sa_vvvv_t1_##SFX(T *out, const T *Ws, const T *Wa, const T *t,         \
                         int64_t v, int64_t no)                                \
{                                                                              \
    const int64_t ps = v * (v + 1) / 2, pa = v * (v - 1) / 2;                  \
    _Pragma("omp parallel")                                                    \
    {                                                                          \
        T *Ys = malloc(sizeof(T) * (2 * v * no + 1));                          \
        T *Ya = Ys + v * no;                                                   \
        T *St = malloc(sizeof(T) * (2 * no * v + 1));    /* transposed [i,f] */\
        T *At = St + no * v;                                                   \
        _Pragma("omp for schedule(dynamic, 16)")                               \
        for (int64_t r = 0; r < ps; r++) {                                     \
            /* row pair (a, b), a <= b, of the symmetric part */               \
            int64_t a = 0;                                                     \
            while ((a + 1) * v - (a + 1) * a / 2 <= r) a++;                    \
            const int64_t b = a + r - (a * v - a * (a - 1) / 2);               \
            const T *ws = Ws + r * ps;                                         \
            const T *wa = (a < b) ? Wa + (a * v - a * (a + 1) / 2 + b - a - 1) * pa : NULL; \
            memset(Ys, 0, sizeof(T) * 2 * v * no);                             \
            memset(St, 0, sizeof(T) * 2 * no * v);                             \
            for (int64_t e = 0; e < v; e++) {                                  \
                const T *se = ws + e * v - e * (e - 1) / 2 - e;   /* se[f] = S(e,f), f>=e */ \
                for (int64_t i = 0; i < no; i++) {                             \
                    const T *ti = t + i * v;                                   \
                    T acc = 0;                                                 \
                    _Pragma("omp simd reduction(+:acc)")                       \
                    for (int64_t f = e; f < v; f++) acc += se[f] * ti[f];      \
                    Ys[e * no + i] += acc;                                     \
                    const T te = ti[e];                                        \
                    T *sti = St + i * v;                                       \
                    _Pragma("omp simd")                                        \
                    for (int64_t f = e + 1; f < v; f++) sti[f] += te * se[f];  \
                }                                                              \
                if (wa != NULL && e + 1 < v) {                                 \
                    const T *ae = wa + e * v - e * (e + 1) / 2 - (e + 1);  /* ae[f] = A(e,f), f>e */ \
                    for (int64_t i = 0; i < no; i++) {                         \
                        const T *ti = t + i * v;                               \
                        T acc = 0;                                             \
                        _Pragma("omp simd reduction(+:acc)")                   \
                        for (int64_t f = e + 1; f < v; f++) acc += ae[f] * ti[f]; \
                        Ya[e * no + i] += acc;                                 \
                        const T te = ti[e];                                    \
                        T *ati = At + i * v;                                   \
                        _Pragma("omp simd")                                    \
                        for (int64_t f = e + 1; f < v; f++) ati[f] -= te * ae[f]; \
                    }                                                          \
                }                                                              \
            }                                                                  \
            T *oab = out + (a * v + b) * v * no;                               \
            T *oba = out + (b * v + a) * v * no;                               \
            for (int64_t e = 0; e < v; e++)                                    \
                for (int64_t i = 0; i < no; i++) {                             \
                    const T s_ = Ys[e * no + i] + St[i * v + e];               \
                    const T a_ = Ya[e * no + i] + At[i * v + e];               \
                    oab[e * no + i] += s_ + a_;                                \
                    if (a != b) oba[e * no + i] += s_ - a_;                    \
                }                                                              \
        }                                                                      \
        free(Ys);                                                              \
        free(St);                                                              \
    }                                                                          \
}
DEFINE_SA_VT1(double, d)
DEFINE_SA_VT1(double complex, z)

/* Gather of the cubes t_mpq (m = 0..n-1) of one occupied pair with all the
 * layouts the ring and T3 -> T1/T2/W need, in one pass over the source:
 *   T[m][x,y,z]   = s(x,y,z)                 s(x,y,z) = t_mpq^{xyz}
 *   T2[m][x,y,z]  = s(y,x,z)
 *   T3[m][x,y,z]  = s(y,z,x)
 *   Y[m][x,y,z]   = 2 s(x,y,z) - s(y,x,z) - s(z,y,x)
 *   Z[m][x,y,z]   = 2 s(x,z,y) - s(y,z,x) - s(z,x,y)
 * Source cube m: src[off[m] + x*st[3m] + y*st[3m+1] + z*st[3m+2]] (off < 0:
 * zero cube).  Every orbit {permutations of (x,y,z)} is read once. */
static const int L_ORBIT[6][6] = {{0, 1, 2, 3, 4, 5}, {1, 0, 5, 4, 3, 2}, {2, 4, 0, 5, 1, 3}, {3, 5, 4, 0, 2, 1}, {4, 2, 3, 1, 5, 0}, {5, 3, 1, 2, 0, 4}};
#define DEFINE_SA_GATHER(T, SFX)                                               \
void zc_sa_pair_gather_##SFX(const T *src, int64_t n, int64_t v,               \
                             const int64_t *off, const int64_t *st,            \
                             T *To, T *T2, T *T3, T *Y, T *Z,                  \
                             const int64_t *ost)                               \
{                                                                              \
    /* output k (To, T2, T3, Y, Z) element [m][x][y][z] at                    \
     * m*ost[2k] + x*ost[2k+1] + y*v + z */                                    \
    const int64_t v2 = v * v;                                                  \
    _Pragma("omp parallel for collapse(2) schedule(dynamic, 4)")               \
    for (int64_t m = 0; m < n; m++)                                            \
    for (int64_t x = 0; x < v; x++) {                                          \
                                                                            \
        const int64_t s0 = st[3 * m], s1 = st[3 * m + 1], s2 = st[3 * m + 2];  \
        const T *c = off[m] >= 0 ? src + off[m] : NULL;                        \
        for (int64_t y = x; y < v; y++)                                        \
        for (int64_t z = y; z < v; z++) {                                      \
            /* the six positions p of the orbit and their source values */    \
            const int64_t P[6][3] = {{x, y, z}, {y, x, z}, {z, y, x},          \
                                     {x, z, y}, {y, z, x}, {z, x, y}};         \
            T sv[6];                                                           \
            for (int k = 0; k < 6; k++)                                        \
                sv[k] = c ? c[P[k][0] * s0 + P[k][1] * s1 + P[k][2] * s2] : 0; \
            /* output at position P[k]: the values at its permutations are \
             * sv[L_ORBIT[k][.]] (L: composition table of the orbit) */         \
            for (int k = 0; k < 6; k++) {                                      \
                const int64_t yz = P[k][1] * v + P[k][2];                      \
                const int64_t x0 = P[k][0];                                    \
                const T sabc = sv[L_ORBIT[k][0]], sbac = sv[L_ORBIT[k][1]], scba = sv[L_ORBIT[k][2]]; \
                const T sacb = sv[L_ORBIT[k][3]], sbca = sv[L_ORBIT[k][4]], scab = sv[L_ORBIT[k][5]]; \
                To[m * ost[0] + x0 * ost[1] + yz] = sabc;                      \
                T2[m * ost[2] + x0 * ost[3] + yz] = sbac;                      \
                T3[m * ost[4] + x0 * ost[5] + yz] = sbca;                      \
                Y[m * ost[6] + x0 * ost[7] + yz] = 2 * sabc - sbac - scba;     \
                Z[m * ost[8] + x0 * ost[9] + yz] = 2 * sacb - sbca - scab;     \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
/* dst[x,y,z] += a0 s0[x,y,z] + a1 s1[x,y,z] + a2 s2[x,y,z] over a v^3 box,    \
 * each source addressed through element strides (st[3k..3k+2]); dst          \
 * through dstr[3].  Sources with null pointers are skipped. */              \
void zc_add3_strided_##SFX(T *dst, const int64_t *dstr, int64_t v,             \
                           const T *s0, const T *s1, const T *s2,              \
                           const int64_t *st, const T *al)                     \
{                                                                              \
    const T *src[3] = {s0, s1, s2};                                            \
    _Pragma("omp parallel for collapse(2) schedule(static)")                   \
    for (int64_t x = 0; x < v; x++)                                            \
    for (int64_t y = 0; y < v; y++) {                                          \
        T *d = dst + x * dstr[0] + y * dstr[1];                                \
        for (int k = 0; k < 3; k++) {                                          \
            if (src[k] == NULL) continue;                                      \
            const T *p = src[k] + x * st[3 * k] + y * st[3 * k + 1];            \
            const int64_t sz = st[3 * k + 2], dz = dstr[2];                    \
            const T a = al[k];                                                 \
            for (int64_t z = 0; z < v; z++) d[z * dz] += a * p[z * sz];        \
        }                                                                      \
    }                                                                          \
}
DEFINE_SA_GATHER(double, d)
DEFINE_SA_GATHER(double complex, z)

/* Tiled gather of the cubes t_mpq (m < n) of one occupied pair with all the
 * layouts the ring and T3 -> T1/T2/W need:
 *   out0 (Y)  [x][m][y][z] = 2 s(x,y,z) - s(y,x,z) - s(z,y,x)
 *   out1 (T)              = s(x,y,z)
 *   out2 (T2)             = s(y,x,z)
 *   out3 (T3)             = s(y,z,x)
 *   out4 (Z)              = 2 s(x,z,y) - s(y,z,x) - s(z,x,y)
 * s(x,y,z) = t_mpq^{xyz} = src[off[m] + x st[3m] + y st[3m+1] + z st[3m+2]]
 * (off < 0: zero cube); output element [x][m][y][z] at x*ox + m*om + y*v + z.
 * Work unit: one cube and one orbit of GT^3 tiles {permutations of the tile
 * triple X <= Y <= Z}; its source tiles are read once into a local buffer
 * and every distinct output tile of the orbit is written from there. */
#define GT 16
static const int G_PERM[6][3] = {{0, 1, 2}, {1, 0, 2}, {2, 1, 0}, {0, 2, 1}, {1, 2, 0}, {2, 0, 1}};
static const int G_PIDX[3][3] = {{-1, 0, 3}, {1, -1, 4}, {5, 2, -1}};
#define DEFINE_SA_GATHER_T(T, SFX)                                             \
void zc_sa_pair_gather_t_##SFX(const T *src, int64_t n, int64_t v,             \
                               const int64_t *off, const int64_t *st,          \
                               T *o0, T *o1, T *o2, T *o3, T *o4,              \
                               int64_t ox, int64_t om)                         \
{                                                                              \
    const int64_t nt = (v + GT - 1) / GT;                                      \
    const int64_t norb = nt * (nt + 1) * (nt + 2) / 6;                         \
    int64_t *orb = malloc(3 * norb * sizeof(int64_t));                         \
    int64_t q = 0;                                                             \
    for (int64_t X = 0; X < nt; X++)                                           \
    for (int64_t Y = X; Y < nt; Y++)                                           \
    for (int64_t Z = Y; Z < nt; Z++) {                                         \
        orb[3 * q] = X; orb[3 * q + 1] = Y; orb[3 * q + 2] = Z; q++;           \
    }                                                                          \
    _Pragma("omp parallel")                                                    \
    {                                                                          \
        T *L = malloc(6 * GT * GT * GT * sizeof(T));                           \
        _Pragma("omp for collapse(2) schedule(dynamic)")                       \
        for (int64_t m = 0; m < n; m++)                                        \
        for (int64_t w = 0; w < norb; w++) {                                   \
            const int64_t tl[3] = {orb[3 * w], orb[3 * w + 1], orb[3 * w + 2]}; \
            const T *c = off[m] >= 0 ? src + off[m] : NULL;                    \
            const int64_t s0 = st[3 * m], s1 = st[3 * m + 1], s2 = st[3 * m + 2]; \
            int64_t pos[6][3];                                                 \
            int lid[6];                                                        \
            for (int k = 0; k < 6; k++) {                                      \
                for (int a = 0; a < 3; a++) pos[k][a] = tl[G_PERM[k][a]];      \
                lid[k] = k;                                                    \
                for (int l = 0; l < k; l++)                                    \
                    if (pos[l][0] == pos[k][0] && pos[l][1] == pos[k][1]       \
                        && pos[l][2] == pos[k][2]) { lid[k] = l; break; }      \
                if (lid[k] != k) continue;                                     \
                /* load tile pos[k] as [u][w][t] */                            \
                T *Lk = L + k * GT * GT * GT;                                  \
                const int64_t U0 = pos[k][0] * GT, W0 = pos[k][1] * GT, V0 = pos[k][2] * GT; \
                const int64_t nu = (v - U0 < GT ? v - U0 : GT);                \
                const int64_t nw = (v - W0 < GT ? v - W0 : GT);                \
                const int64_t nv = (v - V0 < GT ? v - V0 : GT);                \
                for (int64_t u = 0; u < nu; u++)                               \
                for (int64_t ww = 0; ww < nw; ww++) {                          \
                    T *dst = Lk + (u * GT + ww) * GT;                          \
                    if (c == NULL) {                                           \
                        for (int64_t t = 0; t < nv; t++) dst[t] = 0;           \
                    } else {                                                   \
                        const T *sp = c + (U0 + u) * s0 + (W0 + ww) * s1 + V0 * s2; \
                        for (int64_t t = 0; t < nv; t++) dst[t] = sp[t * s2];  \
                    }                                                          \
                }                                                              \
            }                                                                  \
            for (int k = 0; k < 6; k++) {                                      \
                if (lid[k] != k) continue;                                     \
                const int *P = G_PERM[k];                                      \
                /* buffers of s(y,x,z), s(z,y,x), s(x,z,y), s(y,z,x), s(z,x,y) */ \
                const T *L0 = L + lid[k] * GT * GT * GT;                       \
                const T *L1 = L + lid[G_PIDX[P[1]][P[0]]] * GT * GT * GT;      \
                const T *L2 = L + lid[G_PIDX[P[2]][P[1]]] * GT * GT * GT;      \
                const T *L3 = L + lid[G_PIDX[P[0]][P[2]]] * GT * GT * GT;      \
                const T *L4 = L + lid[G_PIDX[P[1]][P[2]]] * GT * GT * GT;      \
                const T *L5 = L + lid[G_PIDX[P[2]][P[0]]] * GT * GT * GT;      \
                const int64_t A0 = pos[k][0] * GT, B0 = pos[k][1] * GT, C0 = pos[k][2] * GT; \
                const int64_t na = (v - A0 < GT ? v - A0 : GT);                \
                const int64_t nb = (v - B0 < GT ? v - B0 : GT);                \
                const int64_t nc = (v - C0 < GT ? v - C0 : GT);                \
                for (int64_t x = 0; x < na; x++)                               \
                for (int64_t y = 0; y < nb; y++) {                             \
                    const int64_t o = (A0 + x) * ox + m * om + (B0 + y) * v + C0; \
                    const T *l0 = L0 + (x * GT + y) * GT;                      \
                    const T *l1 = L1 + (y * GT + x) * GT;                      \
                    const T *l2 = L2 + y * GT + x;                             \
                    const T *l3 = L3 + x * GT * GT + y;                        \
                    const T *l4 = L4 + y * GT * GT + x;                        \
                    const T *l5 = L5 + x * GT + y;                             \
                    for (int64_t z = 0; z < nc; z++) {                         \
                        const T a0 = l0[z], a1 = l1[z], a2 = l2[z * GT * GT];   \
                        const T a3 = l3[z * GT], a4 = l4[z * GT], a5 = l5[z * GT * GT]; \
                        o0[o + z] = 2 * a0 - a1 - a2;                          \
                        o1[o + z] = a0;                                        \
                        o2[o + z] = a1;                                        \
                        o3[o + z] = a4;                                        \
                        o4[o + z] = 2 * a3 - a4 - a5;                          \
                    }                                                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
        free(L);                                                               \
    }                                                                          \
    free(orb);                                                                 \
}
DEFINE_SA_GATHER_T(double, d)
DEFINE_SA_GATHER_T(double complex, z)

/* Jacobi step of the spin-adapted T3 blocks with the DIIS vectors:
 *   q = r / (eo[b] - ev[x] - ev[y] - ev[z]),  xo = t + q,  eo_ = w[b] q
 * (w: DIIS metric weight of block b); returns sum r^2.  Real amplitudes. */
double zc_sa_jacobi_d(int64_t nb, int64_t v, const double *eo, const double *ev,
                      const double *w, const double *t, const double *r,
                      double *xo, double *er)
{
    const int64_t v2 = v * v, v3 = v2 * v;
    double s = 0;
    #pragma omp parallel for collapse(2) schedule(static) reduction(+:s)
    for (int64_t b = 0; b < nb; b++)
    for (int64_t x = 0; x < v; x++) {
        const double ex = eo[b] - ev[x], wb = w[b];
        const int64_t o = b * v3 + x * v2;
        for (int64_t y = 0; y < v; y++) {
            const double exy = ex - ev[y];
            for (int64_t z = 0; z < v; z++) {
                const int64_t i = o + y * v + z;
                const double rv = r[i], q = rv / (exy - ev[z]);
                xo[i] = t[i] + q;
                er[i] = wb * q;
                s += rv * rv;
            }
        }
    }
    return s;
}

/* x[0:n] = 0 (OpenMP; first touch by the threads that use the pages) */
void zc_vzero_d(int64_t n, double *x)
{
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++) x[i] = 0;
}

/* ---- spin-blocked T3 (Ms-conserving blocks, see zccsdt/cc/t3engine.py) ----
 * Element T3[I, A] = osig[I] vsig[A] T3b[obase[I] + vcol[A]] for occupied
 * triple I and virtual triple A of equal spin group. */

#define DEFINE_BLOCKED(T, SFX)                                                 \
/* out[p * ldP + (id * nm + im) * nQ + q] =                                    \
 *     t3[m, P_0, P_1, d, Q_0, Q_1],  P = Pidx[p], d = didx[id], m = midx[im], \
 *     Q = Qlo + q.  otri/osgn: (no, npo); vtri/vsgn: (nv, nvp). */            \
void zc_bgather_##SFX(const T *T3b, const int64_t *obase, const int64_t *vcol, \
                      const double *osig, const double *vsig,                  \
                      const int64_t *otri, const int8_t *osgn, int64_t npo,    \
                      const int64_t *vtri, const int8_t *vsgn, int64_t nvp,    \
                      int64_t nP, const int64_t *Pidx, int64_t nd,             \
                      const int64_t *didx, int64_t nm, const int64_t *midx,    \
                      int64_t Qlo, int64_t nQ, int64_t ldP, T *out)            \
{                                                                              \
    _Pragma("omp parallel for collapse(2) schedule(static)")                   \
    for (int64_t p = 0; p < nP; p++)                                           \
    for (int64_t id = 0; id < nd; id++) {                                      \
        const int64_t P = Pidx[p], d = didx[id];                               \
        const int64_t *vt = vtri + d * nvp + Qlo;                              \
        const int8_t *vs = vsgn + d * nvp + Qlo;                               \
        for (int64_t im = 0; im < nm; im++) {                                  \
            const int64_t m = midx[im];                                        \
            T *o = out + p * ldP + (id * nm + im) * nQ;                        \
            const int8_t so = osgn[m * npo + P];                               \
            if (so == 0) { memset(o, 0, sizeof(T) * nQ); continue; }           \
            const int64_t ti = otri[m * npo + P];                               \
            const T *row = T3b + obase[ti];                                     \
            const double f = so * osig[ti];                                     \
            for (int64_t q = 0; q < nQ; q++) {                                 \
                const int64_t A = vt[q];                                       \
                o[q] = (f * vs[q] * vsig[A]) * row[vcol[A]];                   \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
/* Pass-2 reduction into blocked R: for every computed target I (Rbase >= 0)  \
 * and r with P = pair_r(I) in [p0, p1), x = free_r(I):                        \
 *   R[I, A] += rs_r sum_s rs_s Z[zoff[(P-p0)*3 + gQ] + rowpos[a, x] nQg[gQ]    \
 *                                + (Q - Qoff[gQ])],                           \
 * a = free_s(A), Q = pair_s(A), gQ = vpgrp[Q], for A in the vir group of I. */ \
void zc_breduce_pass2_##SFX(T *R, const int64_t *Rbase, const int64_t *Rcol,   \
                            int64_t nto, const int64_t *ogrp,                  \
                            const int64_t *ofree, const int64_t *opair,        \
                            int64_t nvt, const int64_t *voff,                  \
                            const int64_t *vfree, const int64_t *vpair,        \
                            const int64_t *vpgrp, int64_t p0, int64_t p1,      \
                            const int64_t *zoff, const int64_t *nQg,           \
                            const int64_t *Qoff, const int64_t *rowpos,        \
                            int64_t no, const T *Z)                            \
{                                                                              \
    _Pragma("omp parallel for schedule(dynamic, 2)")                           \
    for (int64_t it = 0; it < nto; it++) {                                     \
        if (Rbase[it] < 0) continue;                                           \
        const int64_t g = ogrp[it];                                            \
        T *r = R + Rbase[it];                                                  \
        for (int rr = 0; rr < 3; rr++) {                                       \
            const int64_t P = opair[rr * nto + it];                            \
            if (P < p0 || P >= p1) continue;                                   \
            const int64_t x = ofree[rr * nto + it];                            \
            const int64_t *zo = zoff + (P - p0) * 3;                           \
            const double sg = (rr == 1) ? -1.0 : 1.0;                          \
            for (int64_t A = voff[g]; A < voff[g + 1]; A++) {                  \
                T acc = 0;                                                     \
                for (int s = 0; s < 3; s++) {                                  \
                    const int64_t a = vfree[s * nvt + A], Q = vpair[s * nvt + A]; \
                    const int64_t gq = vpgrp[Q];                               \
                    const T v = Z[zo[gq] + rowpos[a * no + x] * nQg[gq] + (Q - Qoff[gq])]; \
                    acc += (s == 1) ? -v : v;                                  \
                }                                                              \
                r[Rcol[A]] += sg * acc;                                        \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
/* out[(q * nrows + ib) * nc + cl] = t3[rows[ib], Q_0, Q_1, c],                \
 * Q = Qlo + q, c = c0 + cl. */                                                \
void zc_bgather_vir_##SFX(const T *T3b, const int64_t *obase,                  \
                          const int64_t *vcol, const double *osig,             \
                          const double *vsig, const int64_t *vtri,             \
                          const int8_t *vsgn, int64_t nvp, int64_t nrows,      \
                          const int64_t *rows, int64_t Qlo, int64_t nQ,        \
                          int64_t c0, int64_t nc, T *out)                      \
{                                                                              \
    _Pragma("omp parallel for collapse(2) schedule(static)")                   \
    for (int64_t q = 0; q < nQ; q++)                                           \
    for (int64_t ib = 0; ib < nrows; ib++) {                                   \
        const int64_t ti = rows[ib], Q = Qlo + q;                               \
        const T *row = T3b + obase[ti];                                         \
        const double f = osig[ti];                                              \
        T *o = out + (q * nrows + ib) * nc;                                    \
        for (int64_t cl = 0; cl < nc; cl++) {                                  \
            const int64_t c = c0 + cl;                                         \
            const int64_t A = vtri[c * nvp + Q];                               \
            o[cl] = (f * vsgn[c * nvp + Q] * vsig[A]) * row[vcol[A]];          \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
/* pp-ladder reduction into blocked R for the triples rows[] of group g:      \
 * R[I, A] += sum_s rs_s X_sc[(Q - hoff[sc]) nrows nc[sc] + ib nc[sc]          \
 *                            + (c - coff[sc])],  c = free_s(A), Q = pair_s(A), \
 * sc = vspin[c];  X0 / X1 are the GEMM outputs for c alpha / beta. */         \
void zc_breduce_ppl_##SFX(T *R, const int64_t *Rbase, const int64_t *Rcol,     \
                          int64_t nrows, const int64_t *rows, int64_t nvt,     \
                          int64_t A0, int64_t A1, const int64_t *vfree,        \
                          const int64_t *vpair, const int64_t *vspin,          \
                          const int64_t *hoff, const int64_t *ncs,             \
                          const int64_t *coff, const T *X0, const T *X1)       \
{                                                                              \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t ib = 0; ib < nrows; ib++) {                                   \
        T *r = R + Rbase[rows[ib]];                                            \
        for (int64_t A = A0; A < A1; A++) {                                    \
            T acc = 0;                                                         \
            for (int s = 0; s < 3; s++) {                                      \
                const int64_t c = vfree[s * nvt + A], Q = vpair[s * nvt + A];  \
                const int64_t sc = vspin[c];                                   \
                const T *X = sc ? X1 : X0;                                     \
                const T v = X[((Q - hoff[sc]) * nrows + ib) * ncs[sc] + (c - coff[sc])]; \
                acc += (s == 1) ? -v : v;                                      \
            }                                                                  \
            r[Rcol[A]] += acc;                                                 \
        }                                                                      \
    }                                                                          \
}
DEFINE_BLOCKED(double, d)
DEFINE_BLOCKED(double complex, z)
