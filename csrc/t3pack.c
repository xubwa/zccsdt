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
}                                                                              \
                                                                               \
/* The same for the rows [q0, q1) of the pair index only (a slab of the        \
 * sliced pp ladder):                                                          \
 *   R[rows[I], A] += sum_{s: q0 <= Q_s(A) < q1} rs_s X[Q_s(A) - q0, I, c_s(A)], \
 * X: (q1 - q0, nb, nvir). */                                                  \
void zc_reduce_pair_first_rows_##SFX(T *R, int64_t nvtri, const T *X,          \
                                     const int64_t *rows, int64_t nb,          \
                                     int64_t nvir, int64_t q0, int64_t q1,     \
                                     const int64_t *vfree, const int64_t *vpair) \
{                                                                              \
    const int64_t sQ = nb * nvir, nq = q1 - q0;                                \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t ib = 0; ib < nb; ib++) {                                      \
        const T *x = X + ib * nvir;                                            \
        T *r = R + rows[ib] * nvtri;                                           \
        for (int64_t A = 0; A < nvtri; A++) {                                  \
            const uint64_t Q0 = (uint64_t)(vpair[A] - q0);                     \
            const uint64_t Q1 = (uint64_t)(vpair[nvtri + A] - q0);             \
            const uint64_t Q2 = (uint64_t)(vpair[2 * nvtri + A] - q0);         \
            if (Q0 >= (uint64_t)nq && Q1 >= (uint64_t)nq && Q2 >= (uint64_t)nq) \
                continue;                                                      \
            T acc = 0;                                                         \
            if (Q0 < (uint64_t)nq) acc += x[Q0 * sQ + vfree[A]];               \
            if (Q1 < (uint64_t)nq) acc -= x[Q1 * sQ + vfree[nvtri + A]];       \
            if (Q2 < (uint64_t)nq) acc += x[Q2 * sQ + vfree[2 * nvtri + A]];   \
            r[A] += acc;                                                       \
        }                                                                      \
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
}                                                                              \
/* res[k] = sum conj(a) x[k] for k < nk <= 64 in one pass over a (blocks of    \
 * VB elements of a stay in cache while the x[k] stream by); per-thread        \
 * partial sums, combined in thread order. */                                  \
void zc_vmdot_##SFX(int64_t n, int64_t nk, const T *a, const T **x, T *res)    \
{                                                                              \
    enum { VB = 2048, MK = 64, MT = 256 };                                     \
    T (*part)[MK] = calloc(MT, sizeof(*part));                                 \
    int nt = 1;                                                                \
    _Pragma("omp parallel")                                                    \
    {                                                                          \
        const int id = omp_get_thread_num();                                   \
        _Pragma("omp single")                                                  \
        nt = omp_get_num_threads();                                            \
        T acc[MK] = {0};                                                       \
        _Pragma("omp for schedule(static)")                                    \
        for (int64_t b0 = 0; b0 < n; b0 += VB) {                               \
            const int64_t b1 = b0 + VB < n ? b0 + VB : n;                      \
            for (int64_t k = 0; k < nk; k++) {                                 \
                const T *xk = x[k];                                            \
                T sk = 0;                                                      \
                for (int64_t i = b0; i < b1; i++) sk += CONJ(a[i]) * xk[i];    \
                acc[k] += sk;                                                  \
            }                                                                  \
        }                                                                      \
        if (id < MT)                                                           \
            for (int64_t k = 0; k < nk; k++) part[id][k] = acc[k];             \
    }                                                                          \
    for (int64_t k = 0; k < nk; k++) {                                         \
        T sk = 0;                                                              \
        for (int t = 0; t < nt && t < MT; t++) sk += part[t][k];               \
        res[k] = sk;                                                           \
    }                                                                          \
    free(part);                                                                \
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
                       int64_t npg, const int64_t *nQg, const int64_t *Qoff,   \
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
            const int64_t *zo = zoff + tP[t] * npg;                            \
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
 * Views must not overlap each other (each element written by one view).     \
 * _rows: S and A hold only the rows of the pairs (a, b) with a0 <= a < a1    \
 * (first row: the pair (a0, a0), resp. (a0, a0 + 1)). */                    \
void zc_sa_pair_unpack_add_rows_##SFX(T *dst, int64_t nview,                   \
                                 const int64_t *off,                           \
                                 const int64_t *sd, const int64_t *se,         \
                                 const int64_t *sx, int64_t v, int64_t nx,     \
                                 const T *S, int64_t ld, const T *A,           \
                                 int64_t lda, const int64_t *cols,             \
                                 int64_t a0, int64_t a1)                       \
{                                                                              \
    const int64_t ps0 = a0 * v - a0 * (a0 - 1) / 2;                            \
    const int64_t pa0 = a0 * v - a0 * (a0 + 1) / 2;                            \
    for (int64_t w = 0; w < nview; w++) {                                      \
        T *Y = dst + off[w];                                                   \
        const int64_t col0 = cols ? cols[w] * nx : w * nx;                     \
        _Pragma("omp parallel for schedule(static)")                           \
        for (int64_t a = a0; a < a1; a++) {                                    \
            const int64_t ps = a * v - a * (a - 1) / 2 - ps0;                  \
            const int64_t pa = a * v - a * (a + 1) / 2 - pa0;                  \
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
}                                                                              \
                                                                               \
void zc_sa_pair_unpack_add_##SFX(T *dst, int64_t nview, const int64_t *off,    \
                                 const int64_t *sd, const int64_t *se,         \
                                 const int64_t *sx, int64_t v, int64_t nx,     \
                                 const T *S, int64_t ld, const T *A,           \
                                 int64_t lda, const int64_t *cols)             \
{                                                                              \
    zc_sa_pair_unpack_add_rows_##SFX(dst, nview, off, sd, se, sx, v, nx, S, ld, \
                                     A, lda, cols, 0, v);                      \
}
DEFINE_SA_NULL(double, d)
DEFINE_SA_NULL(double complex, z)

/* out[a, b, e, i] += sum_f W_ab^ef t[i, f] for W given by its (e,f)
 * symmetric part Ws[(a<=b), (e<=f)] and antisymmetric part Wa[(a<b), (e<f)]
 * (pairs in numpy.triu_indices order): W_ab^ef = S_(ab)(ef) + A_(ab)(ef) and
 * W_ba^ef = S - A, S symmetric and A antisymmetric in (e,f).  out (v,v,v,no)
 * contiguous, t (no, v).  One pass over the rows of Ws / Wa.
 * _rows: Ws and Wa hold only the rows of the pairs (a, b) with a0 <= a < a1. */
#define DEFINE_SA_VT1(T, SFX)                                                  \
void zc_sa_vvvv_t1_rows_##SFX(T *out, const T *Ws, const T *Wa, const T *t,    \
                              int64_t v, int64_t no, int64_t a0, int64_t a1)   \
{                                                                              \
    const int64_t ps = v * (v + 1) / 2, pa = v * (v - 1) / 2;                  \
    const int64_t rs0 = a0 * v - a0 * (a0 - 1) / 2;                            \
    const int64_t rs1 = a1 * v - a1 * (a1 - 1) / 2;                            \
    const int64_t ra0 = a0 * v - a0 * (a0 + 1) / 2;                            \
    _Pragma("omp parallel")                                                    \
    {                                                                          \
        T *Ys = malloc(sizeof(T) * (2 * v * no + 1));                          \
        T *Ya = Ys + v * no;                                                   \
        T *St = malloc(sizeof(T) * (2 * no * v + 1));    /* transposed [i,f] */\
        T *At = St + no * v;                                                   \
        _Pragma("omp for schedule(dynamic, 16)")                               \
        for (int64_t r = rs0; r < rs1; r++) {                                  \
            /* row pair (a, b), a <= b, of the symmetric part */               \
            int64_t a = a0;                                                    \
            while ((a + 1) * v - (a + 1) * a / 2 <= r) a++;                    \
            const int64_t b = a + r - (a * v - a * (a - 1) / 2);               \
            const T *ws = Ws + (r - rs0) * ps;                                 \
            const T *wa = (a < b) ? Wa + (a * v - a * (a + 1) / 2 + b - a - 1 - ra0) * pa : NULL; \
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
}                                                                              \
                                                                               \
void zc_sa_vvvv_t1_##SFX(T *out, const T *Ws, const T *Wa, const T *t,         \
                         int64_t v, int64_t no)                                \
{                                                                              \
    zc_sa_vvvv_t1_rows_##SFX(out, Ws, Wa, t, v, no, 0, v);                     \
}
DEFINE_SA_VT1(double, d)
DEFINE_SA_VT1(double complex, z)


/* Gather of the cubes t_mpq (m < n) of one occupied pair with all the
 * layouts the ring and T3 -> T1/T2/W need:
 *   out0 (Y)  [x][m][y][z] = 2 s(x,y,z) - s(y,x,z) - s(z,y,x)
 *   out1 (T)              = s(x,y,z)
 *   out2 (T2)             = s(y,x,z)
 *   out3 (T3)             = s(y,z,x)
 *   out4 (Z)              = 2 s(x,z,y) - s(y,z,x) - s(z,x,y)
 * s(x,y,z) = t_mpq^{xyz} = src[off[m] + x st[3m] + y st[3m+1] + z st[3m+2]]
 * (off < 0: zero cube); output element [x][m][y][z] at x*ox + m*om + y*v + z.
 * Tasks are (cube, 8 x 8 block of (x, y)) and write whole z rows; the
 * strided source reads of a block share cache lines along the unit-stride
 * axis of the cube. */
#define GT 16
#define GS 8
#define DEFINE_SA_GATHER_S(T, SFX)                                             \
void zc_sa_pair_gather_s_##SFX(const T *src, int64_t n, int64_t v,             \
                               const int64_t *off, const int64_t *st,          \
                               T *o0, T *o1, T *o2, T *o3, T *o4,              \
                               int64_t ox, int64_t om)                         \
{                                                                              \
    const int64_t nb = (v + GS - 1) / GS;                                      \
    _Pragma("omp parallel for collapse(3) schedule(dynamic)")                  \
    for (int64_t m = 0; m < n; m++)                                            \
    for (int64_t X = 0; X < nb; X++)                                           \
    for (int64_t Y = 0; Y < nb; Y++) {                                         \
        const T *c = off[m] >= 0 ? src + off[m] : NULL;                        \
        const int64_t s0 = st[3 * m], s1 = st[3 * m + 1], s2 = st[3 * m + 2];  \
        const int64_t x1 = (X + 1) * GS < v ? (X + 1) * GS : v;                \
        const int64_t y1 = (Y + 1) * GS < v ? (Y + 1) * GS : v;                \
        for (int64_t x = X * GS; x < x1; x++)                                  \
        for (int64_t y = Y * GS; y < y1; y++) {                                \
            const int64_t o = x * ox + m * om + y * v;                         \
            if (c == NULL) {                                                   \
                for (int64_t z = 0; z < v; z++)                                \
                    o0[o + z] = o1[o + z] = o2[o + z] = o3[o + z] = o4[o + z] = 0; \
                continue;                                                      \
            }                                                                  \
            const T *pxy = c + x * s0 + y * s1, *pyx = c + y * s0 + x * s1;    \
            const T *pzyx = c + y * s1 + x * s2, *pyzx = c + y * s0 + x * s2;  \
            const T *pxzy = c + x * s0 + y * s2, *pzxy = c + x * s1 + y * s2;  \
            for (int64_t z = 0; z < v; z++) {                                  \
                const T a0 = pxy[z * s2], a1 = pyx[z * s2], a2 = pzyx[z * s0]; \
                const T a3 = pxzy[z * s1], a4 = pyzx[z * s1], a5 = pzxy[z * s0]; \
                o0[o + z] = 2 * a0 - a1 - a2;                                  \
                o1[o + z] = a0;                                                \
                o2[o + z] = a1;                                                \
                o3[o + z] = a4;                                                \
                o4[o + z] = 2 * a3 - a4 - a5;                                  \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}
DEFINE_SA_GATHER_S(double, d)
DEFINE_SA_GATHER_S(double complex, z)

/* dst[c] (contiguous n0 x n1 x n2) = box c of the source, element [x,y,z]
 * at src[off[c] + x st[3c] + y st[3c+1] + z st[3c+2]] (off < 0: zeros), for
 * c < nc, in GT^3 tiles (strided reads stay within a cached tile). */
#define DEFINE_CUBES_TILED(T, SFX)                                             \
void zc_cubes_tiled_##SFX(T *dst, const T *src, int64_t nc, const int64_t *off, \
                          const int64_t *st, int64_t n0, int64_t n1, int64_t n2) \
{                                                                              \
    const int64_t t0 = (n0 + GT - 1) / GT, t1 = (n1 + GT - 1) / GT;            \
    const int64_t t2 = (n2 + GT - 1) / GT, nb = n0 * n1 * n2;                  \
    _Pragma("omp parallel for collapse(4) schedule(dynamic, 4)")               \
    for (int64_t c = 0; c < nc; c++)                                           \
    for (int64_t X = 0; X < t0; X++)                                           \
    for (int64_t Y = 0; Y < t1; Y++)                                           \
    for (int64_t Z = 0; Z < t2; Z++) {                                         \
        const int64_t x0 = X * GT, y0 = Y * GT, z0 = Z * GT;                   \
        const int64_t x1 = x0 + GT < n0 ? x0 + GT : n0;                        \
        const int64_t y1 = y0 + GT < n1 ? y0 + GT : n1;                        \
        const int64_t z1 = z0 + GT < n2 ? z0 + GT : n2;                        \
        T *d = dst + c * nb;                                                   \
        if (off[c] < 0) {                                                      \
            for (int64_t x = x0; x < x1; x++)                                  \
            for (int64_t y = y0; y < y1; y++)                                  \
            for (int64_t z = z0; z < z1; z++) d[(x * n1 + y) * n2 + z] = 0;     \
            continue;                                                          \
        }                                                                      \
        const T *sp = src + off[c];                                            \
        const int64_t s0 = st[3 * c], s1 = st[3 * c + 1], s2 = st[3 * c + 2];  \
        for (int64_t x = x0; x < x1; x++)                                      \
        for (int64_t y = y0; y < y1; y++) {                                    \
            const T *q = sp + x * s0 + y * s1;                                 \
            T *o = d + (x * n1 + y) * n2;                                      \
            for (int64_t z = z0; z < z1; z++) o[z] = q[z * s2];                \
        }                                                                      \
    }                                                                          \
}
DEFINE_CUBES_TILED(double, d)
DEFINE_CUBES_TILED(double complex, z)

/* Batched strided adds into T3 blocks.  Placement k adds into the box
 * [lo, lo+n) (box[6k..6k+2] = lo, box[6k+3..6k+5] = n) of its destination
 * block at dptr[k] (strides dstr[3k..]):
 *   dst[x,y,z] += sum_s al[s] src_{k,s}[x-lo0, y-lo1, z-lo2],
 * source s at sptr[k*ns+s] with strides sstr[3(k*ns+s)..] (in destination
 * orientation).  Placements are grouped by destination block (group g:
 * gptr[g] .. gptr[g+1]-1, applied in order) and the tasks are (group, GT^3
 * tile of the N0 x N1 x N2 destination), so no element is written twice
 * at the same time. */
#define DEFINE_SCATTER_TILED(T, SFX)                                           \
void zc_scatter_tiled_##SFX(int64_t ngroup, const int64_t *gptr, T **dptr,     \
                            const int64_t *dstr, const int64_t *box,           \
                            int64_t ns, const T **sptr, const int64_t *sstr,   \
                            const T *al, int64_t N0, int64_t N1, int64_t N2)   \
{                                                                              \
    const int64_t t0 = (N0 + GT - 1) / GT, t1 = (N1 + GT - 1) / GT;            \
    const int64_t t2 = (N2 + GT - 1) / GT;                                     \
    _Pragma("omp parallel for collapse(4) schedule(dynamic, 2)")               \
    for (int64_t g = 0; g < ngroup; g++)                                       \
    for (int64_t X = 0; X < t0; X++)                                           \
    for (int64_t Y = 0; Y < t1; Y++)                                           \
    for (int64_t Z = 0; Z < t2; Z++) {                                         \
        for (int64_t k = gptr[g]; k < gptr[g + 1]; k++) {                      \
            const int64_t *b = box + 6 * k;                                    \
            int64_t lo[3], hi[3], skip = 0;                                    \
            const int64_t tl[3] = {X * GT, Y * GT, Z * GT};                    \
            for (int a = 0; a < 3; a++) {                                      \
                lo[a] = tl[a] > b[a] ? tl[a] : b[a];                           \
                hi[a] = tl[a] + GT < b[a] + b[3 + a] ? tl[a] + GT : b[a] + b[3 + a]; \
                if (lo[a] >= hi[a]) skip = 1;                                  \
            }                                                                  \
            if (skip) continue;                                                \
            T *d = dptr[k];                                                    \
            const int64_t *ds = dstr + 3 * k;                                  \
            for (int64_t q = 0; q < ns; q++) {                                 \
                const int64_t *ss = sstr + 3 * (k * ns + q);                   \
                const T *sp = sptr[k * ns + q] - b[0] * ss[0] - b[1] * ss[1] - b[2] * ss[2]; \
                const T a = al[q];                                             \
                for (int64_t x = lo[0]; x < hi[0]; x++)                        \
                for (int64_t y = lo[1]; y < hi[1]; y++) {                      \
                    const T *src = sp + x * ss[0] + y * ss[1];                 \
                    T *o = d + x * ds[0] + y * ds[1];                          \
                    for (int64_t z = lo[2]; z < hi[2]; z++) o[z * ds[2]] += a * src[z * ss[2]]; \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}
DEFINE_SCATTER_TILED(double, d)
DEFINE_SCATTER_TILED(double complex, z)

/* Packed bare <ab|de> from the 4-fold packed (ad|be) of PySCF (pairs i >= j
 * at i(i+1)/2 + j):  Vs[(a<=b), (d<=e)] = (x + y)/2,  Va[(a<b), (d<e)] =
 * (x - y)/2 with x = (ad|be), y = (ae|bd); rows and columns in
 * numpy.triu_indices order. */
static inline int64_t tri_idx(int64_t i, int64_t j)
{
    return i >= j ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i;
}
void zc_sa_vvvv_pack_d(const double *E, int64_t v, double *Vs, double *Va)
{
    const int64_t np_ = v * (v + 1) / 2, pa = v * (v - 1) / 2;
    #pragma omp parallel for collapse(2) schedule(dynamic, 8)
    for (int64_t a = 0; a < v; a++)
    for (int64_t b = 0; b < v; b++) {
        if (b < a) continue;
        double *vs = Vs + (a * v - a * (a - 1) / 2 + (b - a)) * np_;
        double *va = a < b ? Va + (a * v - a * (a + 1) / 2 + (b - a - 1)) * pa : NULL;
        int64_t cs = 0, ca = 0;
        for (int64_t d = 0; d < v; d++) {
            const double *Ead = E + tri_idx(a, d) * np_;
            const int64_t pbd = tri_idx(b, d);
            for (int64_t e = d; e < v; e++) {
                const double x = Ead[tri_idx(b, e)];
                const double y = E[tri_idx(a, e) * np_ + pbd];
                vs[cs++] = 0.5 * (x + y);
                if (va != NULL && e > d) va[ca++] = 0.5 * (x - y);
            }
        }
    }
}

/* The same for the rows (a, b >= a) of one a, from R[d][tri(b,e)] = (ad|be)
 * (v rows of v(v+1)/2): vs (v - a rows), va (v - a - 1 rows). */
void zc_sa_vvvv_pack_a_d(const double *R, int64_t v, int64_t a, double *vs, double *va)
{
    const int64_t np_ = v * (v + 1) / 2, pa = v * (v - 1) / 2;
    #pragma omp parallel for schedule(dynamic, 4)
    for (int64_t b = a; b < v; b++) {
        double *s = vs + (b - a) * np_;
        double *t = b > a ? va + (b - a - 1) * pa : NULL;
        int64_t cs = 0, ca = 0;
        for (int64_t d = 0; d < v; d++) {
            const double *Rd = R + d * np_;
            const int64_t pbd = tri_idx(b, d);
            for (int64_t e = d; e < v; e++) {
                const double x = Rd[tri_idx(b, e)];
                const double y = R[e * np_ + pbd];
                s[cs++] = 0.5 * (x + y);
                if (t != NULL && e > d) t[ca++] = 0.5 * (x - y);
            }
        }
    }
}

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

/* ---- CCSDT(Q), closed shell (zccsdt/sa/rccsdt_q.py) ----
 * Energy of one occupied quadruple from its accumulators:
 *   out[0] = sum_x c[x] (M r)[x] / D[x],   out[1] = sum_x a[x] (M r)[x] / D[x],
 *   (M r)[x] = sum_tau w(tau) r[tau(x)],  w = sgn 2^cycles,  r = a + b,
 *   D[x] = eocc - ev[x0] - ev[x1] - ev[x2] - ev[x3],  x = (x0, x1, x2, x3).
 * a and b are sums of three arrays in the cyclic layouts
 *   a[x] = S[0][x0,x1,x2,x3] + S[1][x1,x2,x0,x3] + S[2][x2,x0,x1,x3]   (b: S[3..5]),
 * and c[x] = sum_k tc[k][x_p,x_q] gc[k][x_r,x_s] over the six pair sets
 * (p,q) = (0,1),(0,2),(0,3),(1,2),(1,3),(2,3), (r,s) the complement (v x v
 * matrices).
 *
 * The work unit is a sorted tile quadruple T0 <= T1 <= T2 <= T3 (tiles of
 * QT indices) with its 24 permuted tile positions.  The tile of position pi
 * is loaded into a local buffer indexed by the *base* element u (the element
 * of the sorted tile quadruple whose permutation it is): buf[pi][u] =
 * f[pi(x(u))].  The 24 permutations of one element then sit at the same
 * offset of 24 buffers, and M (in its factorized form
 *   M = 2 (2 - (01)) (2 - (02) - (12)) (2 - (03) - (13) - (23)),
 * Jucys-Murphy elements) and the energy are unit-stride operations on rows
 * of QT elements.  Where tiles coincide, only sorted base elements are
 * taken, with weight 1 / |stabilizer|.
 *
 * mirror: a and b are the sums above plus their images under x0 <-> x1
 * (quadruples whose first two occupied indices coincide: only one term of
 * each mirror pair was accumulated). */
#ifndef QT
#define QT 6
#endif
#define Q2 (QT * QT)
#define Q3 (QT * QT * QT)
#define Q4 (QT * QT * QT * QT)
static const int Q_PERM[24][4] = {{0, 1, 2, 3}, {0, 1, 3, 2}, {0, 2, 1, 3}, {0, 2, 3, 1}, {0, 3, 1, 2}, {0, 3, 2, 1}, {1, 0, 2, 3}, {1, 0, 3, 2}, {1, 2, 0, 3}, {1, 2, 3, 0}, {1, 3, 0, 2}, {1, 3, 2, 0}, {2, 0, 1, 3}, {2, 0, 3, 1}, {2, 1, 0, 3}, {2, 1, 3, 0}, {2, 3, 0, 1}, {2, 3, 1, 0}, {3, 0, 1, 2}, {3, 0, 2, 1}, {3, 1, 0, 2}, {3, 1, 2, 0}, {3, 2, 0, 1}, {3, 2, 1, 0}};
/* transpositions of the factors of M: (03), (13), (23); (02), (12); (01) */
static const int Q_TR[6][2] = {{0, 3}, {1, 3}, {2, 3}, {0, 2}, {1, 2}, {0, 1}};
/* part[pi][t]: the permutation pi with the two entries of transposition t swapped */
static void q_parts(int part[24][6])
{
    for (int pi = 0; pi < 24; pi++)
        for (int t = 0; t < 6; t++) {
            int q[4] = {Q_PERM[pi][0], Q_PERM[pi][1], Q_PERM[pi][2], Q_PERM[pi][3]};
            q[Q_TR[t][0]] = Q_PERM[pi][Q_TR[t][1]];
            q[Q_TR[t][1]] = Q_PERM[pi][Q_TR[t][0]];
            for (int p2 = 0; p2 < 24; p2++)
                if (Q_PERM[p2][0] == q[0] && Q_PERM[p2][1] == q[1] && Q_PERM[p2][2] == q[2]) part[pi][t] = p2;
        }
}

/* M r and the energy of one sorted tile quadruple from its 24 base-ordered
 * buffers (al, cl, rl: a, c and r at the 24 permuted positions of every base
 * element), row by row of the base tile; adds to e5 (c M r / D) and e6
 * (a M r / D).  bb / nn: first index and size of the four tiles. */
static void q_tile_energy(const double *al, const double *cl, const double *rl,
                          int part[24][6], const int64_t *bb, const int64_t *nn, int rep,
                          double eocc, const double *ev, double *e5, double *e6)
{
    const int64_t m3 = nn[3];
    for (int64_t u0 = 0; u0 < nn[0]; u0++)
    for (int64_t u1 = 0; u1 < nn[1]; u1++)
    for (int64_t u2 = 0; u2 < nn[2]; u2++) {
        const int64_t x0 = bb[0] + u0, x1 = bb[1] + u1, x2 = bb[2] + u2;
        if (rep && (x0 > x1 || x1 > x2)) continue;
        const int64_t row = (u0 * QT + u1) * Q2 + u2 * QT;
        double t1[24][QT], t2[24][QT], s5[QT] = {0}, s6[QT] = {0};
        for (int pi = 0; pi < 24; pi++) {
            const double *r0 = rl + pi * Q4 + row, *ra = rl + part[pi][0] * Q4 + row;
            const double *rb = rl + part[pi][1] * Q4 + row, *rc = rl + part[pi][2] * Q4 + row;
            for (int64_t u3 = 0; u3 < m3; u3++)
                t1[pi][u3] = 2 * r0[u3] - ra[u3] - rb[u3] - rc[u3];
        }
        for (int pi = 0; pi < 24; pi++) {
            const double *q0 = t1[pi], *qa = t1[part[pi][3]], *qb = t1[part[pi][4]];
            for (int64_t u3 = 0; u3 < m3; u3++) t2[pi][u3] = 2 * q0[u3] - qa[u3] - qb[u3];
        }
        for (int pi = 0; pi < 24; pi++) {
            const double *q0 = t2[pi], *qa = t2[part[pi][5]];
            const double *ar = al + pi * Q4 + row, *cr = cl + pi * Q4 + row;
            for (int64_t u3 = 0; u3 < m3; u3++) {
                const double m = 2 * q0[u3] - qa[u3];
                s5[u3] += cr[u3] * m;
                s6[u3] += ar[u3] * m;
            }
        }
        const double d0 = eocc - ev[x0] - ev[x1] - ev[x2];
        if (!rep) {
            for (int64_t u3 = 0; u3 < m3; u3++) {
                const double f = 2 / (d0 - ev[bb[3] + u3]);
                *e5 += s5[u3] * f;
                *e6 += s6[u3] * f;
            }
        } else {
            for (int64_t u3 = 0; u3 < m3; u3++) {
                const int64_t x3 = bb[3] + u3;
                if (x3 < x2) continue;
                /* 1 / |stabilizer| of the sorted element */
                const int e01 = x0 == x1, e12 = x1 == x2, e23 = x2 == x3;
                double st = 1;
                if (e01 && e12 && e23) st = 24;
                else if ((e01 && e12) || (e12 && e23)) st = 6;
                else if (e01 && e23) st = 4;
                else if (e01 || e12 || e23) st = 2;
                const double f = 2 / (st * (d0 - ev[x3]));
                *e5 += s5[u3] * f;
                *e6 += s6[u3] * f;
            }
        }
    }
}

/* copy the buffers of an earlier permutation whose tile position coincides
 * with that of pi (coinciding tiles); returns 1 if one was found */
static int q_copy_tile(double *al, double *cl, double *rl, int pi, const int64_t *T,
                       const int64_t *nn)
{
    static const int64_t QS[4] = {Q3, Q2, QT, 1};
    const int *P = Q_PERM[pi];
    int p0 = -1;
    for (int q = 0; q < pi && p0 < 0; q++)
        if (T[Q_PERM[q][0]] == T[P[0]] && T[Q_PERM[q][1]] == T[P[1]]
            && T[Q_PERM[q][2]] == T[P[2]] && T[Q_PERM[q][3]] == T[P[3]]) p0 = q;
    if (p0 < 0) return 0;
    const int *P0 = Q_PERM[p0];
    const int64_t n0 = nn[P[0]], n1 = nn[P[1]], n2 = nn[P[2]], n3 = nn[P[3]];
    const int64_t l0 = QS[P[0]], l1 = QS[P[1]], l2 = QS[P[2]], l3 = QS[P[3]];
    const int64_t m0 = QS[P0[0]], m1 = QS[P0[1]], m2 = QS[P0[2]], m3 = QS[P0[3]];
    double *ap = al + pi * Q4, *cp = cl + pi * Q4, *rp = rl + pi * Q4;
    const double *a0 = al + p0 * Q4, *c0 = cl + p0 * Q4, *r0 = rl + p0 * Q4;
    for (int64_t u0 = 0; u0 < n0; u0++)
    for (int64_t u1 = 0; u1 < n1; u1++)
    for (int64_t u2 = 0; u2 < n2; u2++) {
        const int64_t lo = u0 * l0 + u1 * l1 + u2 * l2;
        const int64_t mo = u0 * m0 + u1 * m1 + u2 * m2;
        for (int64_t u3 = 0; u3 < n3; u3++) {
            ap[lo + u3 * l3] = a0[mo + u3 * m3];
            rp[lo + u3 * l3] = r0[mo + u3 * m3];
            cp[lo + u3 * l3] = c0[mo + u3 * m3];
        }
    }
    return 1;
}

void zc_sa_q_energy_d(int64_t v, const double **S, const double *tc, const double *gc,
                      double eocc, const double *ev, int mirror, double *out)
{
    const int64_t nt = (v + QT - 1) / QT, v2 = v * v, v3 = v2 * v;
    const int64_t norb = nt * (nt + 1) * (nt + 2) * (nt + 3) / 24;
    int64_t *orb = malloc(4 * norb * sizeof(int64_t));
    int64_t n = 0;
    for (int64_t a = 0; a < nt; a++)
    for (int64_t b = a; b < nt; b++)
    for (int64_t c = b; c < nt; c++)
    for (int64_t d = c; d < nt; d++) {
        orb[4 * n] = a; orb[4 * n + 1] = b; orb[4 * n + 2] = c; orb[4 * n + 3] = d; n++;
    }
    static const int PQ[6][4] = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2},
                                 {1, 2, 0, 3}, {1, 3, 0, 2}, {2, 3, 0, 1}};
    static const int64_t QS[4] = {Q3, Q2, QT, 1};
    /* buffer of the permutation pi with two entries swapped */
    int part[24][6];
    q_parts(part);
    double e5 = 0, e6 = 0;
    #pragma omp parallel reduction(+:e5, e6)
    {
        double *al = malloc(3 * 24 * Q4 * sizeof(double));
        double *cl = al + 24 * Q4, *rl = al + 48 * Q4;
        #pragma omp for schedule(dynamic)
        for (int64_t w = 0; w < norb; w++) {
            const int64_t *T = orb + 4 * w;
            int64_t bb[4], nn[4];
            for (int k = 0; k < 4; k++) {
                bb[k] = T[k] * QT;
                nn[k] = v - bb[k] < QT ? v - bb[k] : QT;
            }
            /* ---- load the 24 tiles: position k of pi is the base index pi[k] */
            const int rep = T[0] == T[1] || T[1] == T[2] || T[2] == T[3];
            for (int pi = 0; pi < 24; pi++) {
                const int *P = Q_PERM[pi];
                const int64_t b0 = bb[P[0]], b1 = bb[P[1]], b2 = bb[P[2]], b3 = bb[P[3]];
                const int64_t n0 = nn[P[0]], n1 = nn[P[1]], n2 = nn[P[2]], n3 = nn[P[3]];
                const int64_t l0 = QS[P[0]], l1 = QS[P[1]], l2 = QS[P[2]], l3 = QS[P[3]];
                double *ap = al + pi * Q4, *cp = cl + pi * Q4, *rp = rl + pi * Q4;
                /* coinciding tiles: the same tile position may have been
                 * loaded for an earlier permutation; copy it in cache */
                if (rep && q_copy_tile(al, cl, rl, pi, T, nn)) continue;
                /* the rows of a tile are scattered over the accumulators:
                 * request them all before use, so that the misses overlap */
                for (int64_t u0 = 0; u0 < n0; u0++)
                for (int64_t u1 = 0; u1 < n1; u1++)
                for (int64_t u2 = 0; u2 < n2; u2++) {
                    const int64_t x0 = b0 + u0, x1 = b1 + u1, x2 = b2 + u2;
                    const int64_t i0 = x0 * v3 + x1 * v2 + x2 * v + b3;
                    const int64_t i1 = x1 * v3 + x2 * v2 + x0 * v + b3;
                    const int64_t i2 = x2 * v3 + x0 * v2 + x1 * v + b3;
                    for (int k = 0; k < 2; k++) {
                        __builtin_prefetch(S[3 * k] + i0, 0, 2);
                        __builtin_prefetch(S[3 * k] + i0 + n3 - 1, 0, 2);
                        __builtin_prefetch(S[3 * k + 1] + i1, 0, 2);
                        __builtin_prefetch(S[3 * k + 1] + i1 + n3 - 1, 0, 2);
                        __builtin_prefetch(S[3 * k + 2] + i2, 0, 2);
                        __builtin_prefetch(S[3 * k + 2] + i2 + n3 - 1, 0, 2);
                    }
                }
                /* c[x] = t01 g23 + t02 g13 + t12 g03 + g12 t03 + g02 t13 + g01 t23
                 * (pair sets in the order of tc / gc), rows along x3 */
                const double *t01 = tc, *t02 = tc + v2, *t03 = tc + 2 * v2;
                const double *t12 = tc + 3 * v2, *t13 = tc + 4 * v2, *t23 = tc + 5 * v2;
                const double *g23 = gc, *g13 = gc + v2, *g12 = gc + 2 * v2;
                const double *g03 = gc + 3 * v2, *g02 = gc + 4 * v2, *g01 = gc + 5 * v2;
#define Q_LOAD_ROW(N)                                                          \
    for (int64_t u3 = 0; u3 < (N); u3++) {                                     \
        const double a = s0[u3] + s1[u3] + s2[u3];                             \
        ap[lo + u3 * l3] = a;                                                  \
        rp[lo + u3 * l3] = a + s3[u3] + s4[u3] + s5[u3];                       \
        cp[lo + u3 * l3] = c0 * p0[u3] + c1 * p1[u3] + c2 * p2[u3]             \
                         + c3 * p3[u3] + c4 * p4[u3] + c5 * p5[u3];            \
    }
                for (int64_t u0 = 0; u0 < n0; u0++)
                for (int64_t u1 = 0; u1 < n1; u1++) {
                    const int64_t x0 = b0 + u0, x1 = b1 + u1;
                    const double c0 = t01[x0 * v + x1], c5 = g01[x0 * v + x1];
                    const double *p1 = g13 + x1 * v + b3, *p2 = g03 + x0 * v + b3;
                    const double *p3 = t03 + x0 * v + b3, *p4 = t13 + x1 * v + b3;
                    for (int64_t u2 = 0; u2 < n2; u2++) {
                        const int64_t x2 = b2 + u2;
                        const int64_t i0 = x0 * v3 + x1 * v2 + x2 * v + b3;
                        const int64_t i1 = x1 * v3 + x2 * v2 + x0 * v + b3;
                        const int64_t i2 = x2 * v3 + x0 * v2 + x1 * v + b3;
                        const double *s0 = S[0] + i0, *s1 = S[1] + i1, *s2 = S[2] + i2;
                        const double *s3 = S[3] + i0, *s4 = S[4] + i1, *s5 = S[5] + i2;
                        const double c1 = t02[x0 * v + x2], c2 = t12[x1 * v + x2];
                        const double c3 = g12[x1 * v + x2], c4 = g02[x0 * v + x2];
                        const double *p0 = g23 + x2 * v + b3, *p5 = t23 + x2 * v + b3;
                        const int64_t lo = u0 * l0 + u1 * l1 + u2 * l2;
                        if (n3 == QT) {
                            Q_LOAD_ROW(QT)
                        } else {
                            Q_LOAD_ROW(n3)
                        }
                    }
                }
#undef Q_LOAD_ROW
            }
            /* a and r plus their images under x0 <-> x1: the image of the
             * element of buffer pi is the same base element of the buffer
             * of pi with its first two entries exchanged */
            if (mirror)
                for (int pi = 0; pi < 24; pi++) {
                    const int pp = part[pi][5];
                    if (pp < pi) continue;
                    double *a0 = al + pi * Q4, *a1 = al + pp * Q4;
                    double *r0 = rl + pi * Q4, *r1 = rl + pp * Q4;
                    for (int64_t e = 0; e < Q4; e++) {
                        const double sa = a0[e] + a1[e], sr = r0[e] + r1[e];
                        a0[e] = a1[e] = sa;
                        r0[e] = r1[e] = sr;
                    }
                }
            q_tile_energy(al, cl, rl, part, bb, nn, rep, eocc, ev, &e5, &e6);
        }
        free(al);
    }
    free(orb);
    out[0] = e5;
    out[1] = e6;
}

/* The same for the sorted tile quadruples whose largest tile lies in the slab
 * of tiles t0 <= T3 < t1, from the *faces* of the quadruple: with d0 = t0 QT,
 * n = min(v, t1 QT) and nd = n - d0, face p (the pair p carries the slab
 * index) is given by six arrays F[6 p + k] of shape (n, n, n, nd) in the
 * three cyclic layouts of the other pairs q0 < q1 < q2,
 *   a[x] = F[6p][x_q0, x_q1, x_q2, x_p - d0] + F[6p+1][x_q1, x_q2, x_q0, .]
 *        + F[6p+2][x_q2, x_q0, x_q1, .]      (b: F[6p+3 .. 6p+5]),
 * valid where x_p is in the slab and the other indices are below n.  Every
 * permuted position of an element whose largest index is in the slab has
 * that index at some position p and is read from face p.  tc / gc as in
 * zc_sa_q_energy_d (v x v matrices). */
void zc_sa_q_energy_faces_d(int64_t v, int64_t t0, int64_t t1, const double **F,
                            const double *tc, const double *gc, double eocc,
                            const double *ev, double *out)
{
    const int64_t d0 = t0 * QT, v2 = v * v;
    const int64_t n = t1 * QT < v ? t1 * QT : v, nd = n - d0;
    int64_t norb = 0;
    for (int64_t d = t0; d < t1; d++) norb += (d + 1) * (d + 2) * (d + 3) / 6;
    int64_t *orb = malloc(4 * (norb > 0 ? norb : 1) * sizeof(int64_t));
    int64_t cnt = 0;
    for (int64_t d = t0; d < t1; d++)
    for (int64_t a = 0; a <= d; a++)
    for (int64_t b = a; b <= d; b++)
    for (int64_t c = b; c <= d; c++) {
        orb[4 * cnt] = a; orb[4 * cnt + 1] = b; orb[4 * cnt + 2] = c; orb[4 * cnt + 3] = d; cnt++;
    }
    static const int64_t QS[4] = {Q3, Q2, QT, 1};
    int part[24][6];
    q_parts(part);
    /* c[x] = sum over the six pair sets of tc[k][x_p, x_q] gc[k][x_r, x_s] */
    static const int CP[6][4] = {{0, 1, 2, 3}, {0, 2, 1, 3}, {0, 3, 1, 2},
                                 {1, 2, 0, 3}, {1, 3, 0, 2}, {2, 3, 0, 1}};
    double e5 = 0, e6 = 0;
    #pragma omp parallel reduction(+:e5, e6)
    {
        double *al = malloc(3 * 24 * Q4 * sizeof(double));
        double *cl = al + 24 * Q4, *rl = al + 48 * Q4;
        #pragma omp for schedule(dynamic)
        for (int64_t w = 0; w < norb; w++) {
            const int64_t *T = orb + 4 * w;
            int64_t bb[4], nn[4];
            for (int k = 0; k < 4; k++) {
                bb[k] = T[k] * QT;
                nn[k] = v - bb[k] < QT ? v - bb[k] : QT;
            }
            const int rep = T[0] == T[1] || T[1] == T[2] || T[2] == T[3];
            for (int pi = 0; pi < 24; pi++) {
                const int *P = Q_PERM[pi];
                if (rep && q_copy_tile(al, cl, rl, pi, T, nn)) continue;
                /* position k3 holds the base index 3 (the slab tile); the
                 * other positions q[0] < q[1] < q[2] */
                int k3 = 0, q[3], m = 0;
                for (int k = 0; k < 4; k++) {
                    if (P[k] == 3) k3 = k;
                    else q[m++] = k;
                }
                const double *f0 = F[6 * k3], *f1 = F[6 * k3 + 1], *f2 = F[6 * k3 + 2];
                const double *f3 = F[6 * k3 + 3], *f4 = F[6 * k3 + 4], *f5 = F[6 * k3 + 5];
                const int64_t ba = bb[P[q[0]]], bbq = bb[P[q[1]]], bc = bb[P[q[2]]];
                const int64_t na = nn[P[q[0]]], nb = nn[P[q[1]]], nc = nn[P[q[2]]];
                const int64_t la = QS[P[q[0]]], lb = QS[P[q[1]]], lc = QS[P[q[2]]];
                const int64_t n3 = nn[3], off = bb[3] - d0;
                double *ap = al + pi * Q4, *cp = cl + pi * Q4, *rp = rl + pi * Q4;
                int64_t y[4];
                for (int64_t ua = 0; ua < na; ua++)
                for (int64_t ub = 0; ub < nb; ub++)
                for (int64_t uc = 0; uc < nc; uc++) {
                    const int64_t xa = ba + ua, xb = bbq + ub, xc = bc + uc;
                    const double *s0 = f0 + ((xa * n + xb) * n + xc) * nd + off;
                    const double *s1 = f1 + ((xb * n + xc) * n + xa) * nd + off;
                    const double *s2 = f2 + ((xc * n + xa) * n + xb) * nd + off;
                    const double *s3 = f3 + ((xa * n + xb) * n + xc) * nd + off;
                    const double *s4 = f4 + ((xb * n + xc) * n + xa) * nd + off;
                    const double *s5 = f5 + ((xc * n + xa) * n + xb) * nd + off;
                    const int64_t lo = ua * la + ub * lb + uc * lc;
                    y[q[0]] = xa; y[q[1]] = xb; y[q[2]] = xc; y[k3] = bb[3];
                    /* per pair set: the factor without the slab position is a
                     * constant of the row, the other one a row (or column) of
                     * its v x v matrix along the slab index */
                    double cf[6];
                    const double *cq[6];
                    int64_t cs[6];
                    for (int k = 0; k < 6; k++) {
                        const int *c = CP[k];
                        const double *tm = tc + k * v2, *gm = gc + k * v2;
                        if (c[0] == k3 || c[1] == k3) {
                            cf[k] = gm[y[c[2]] * v + y[c[3]]];
                            cq[k] = tm + y[c[0]] * v + y[c[1]];
                            cs[k] = c[0] == k3 ? v : 1;
                        } else {
                            cf[k] = tm[y[c[0]] * v + y[c[1]]];
                            cq[k] = gm + y[c[2]] * v + y[c[3]];
                            cs[k] = c[2] == k3 ? v : 1;
                        }
                    }
                    for (int64_t u3 = 0; u3 < n3; u3++) {
                        const double a = s0[u3] + s1[u3] + s2[u3];
                        ap[lo + u3] = a;
                        rp[lo + u3] = a + s3[u3] + s4[u3] + s5[u3];
                        cp[lo + u3] = cf[0] * cq[0][u3 * cs[0]] + cf[1] * cq[1][u3 * cs[1]]
                                    + cf[2] * cq[2][u3 * cs[2]] + cf[3] * cq[3][u3 * cs[3]]
                                    + cf[4] * cq[4][u3 * cs[4]] + cf[5] * cq[5][u3 * cs[5]];
                    }
                }
            }
            q_tile_energy(al, cl, rl, part, bb, nn, rep, eocc, ev, &e5, &e6);
        }
        free(al);
    }
    free(orb);
    out[0] = e5;
    out[1] = e6;
}

/* Dense [e, x, k] (e < v, x < n, k < K) from rows packed over pairs e >= x:
 *   out[(e n + x) K + k] = src[row[k] ps + tri(e, x)],  ps = v (v + 1) / 2
 * (a quantity symmetric in (e, x), one packed row per k).  Rows in chunks, so
 * that few read streams are open at a time. */
#define QU_CHUNK 16
void zc_sa_q_unpack_d(double *out, int64_t v, int64_t n, int64_t K, const double *src,
                      const int64_t *row)
{
    const int64_t ps = v * (v + 1) / 2, nch = (K + QU_CHUNK - 1) / QU_CHUNK;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t ch = 0; ch < nch; ch++)
    for (int64_t e = 0; e < v; e++) {
        const int64_t k0 = ch * QU_CHUNK;
        const int64_t k1 = k0 + QU_CHUNK < K ? k0 + QU_CHUNK : K;
        for (int64_t x = 0; x < n; x++) {
            const int64_t t = tri_idx(e, x);
            double *o = out + (e * n + x) * K;
            for (int64_t k = k0; k < k1; k++) o[k] = src[row[k] * ps + t];
        }
    }
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
                            const int64_t *zoff, int64_t npg,                  \
                            const int64_t *nQg,                                \
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
            const int64_t *zo = zoff + (P - p0) * npg;                         \
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
 * sc = vspin[c];  Xs[sc] is the GEMM output for the c of class sc. */        \
void zc_breduce_ppl_##SFX(T *R, const int64_t *Rbase, const int64_t *Rcol,     \
                          int64_t nrows, const int64_t *rows, int64_t nvt,     \
                          int64_t A0, int64_t A1, const int64_t *vfree,        \
                          const int64_t *vpair, const int64_t *vspin,          \
                          const int64_t *hoff, const int64_t *ncs,             \
                          const int64_t *coff, const T *const *Xs)             \
{                                                                              \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t ib = 0; ib < nrows; ib++) {                                   \
        T *r = R + Rbase[rows[ib]];                                            \
        for (int64_t A = A0; A < A1; A++) {                                    \
            T acc = 0;                                                         \
            for (int s = 0; s < 3; s++) {                                      \
                const int64_t c = vfree[s * nvt + A], Q = vpair[s * nvt + A];  \
                const int64_t sc = vspin[c];                                   \
                const T *X = Xs[sc];                                           \
                const T v = X[((Q - hoff[sc]) * nrows + ib) * ncs[sc] + (c - coff[sc])]; \
                acc += (s == 1) ? -v : v;                                      \
            }                                                                  \
            r[Rcol[A]] += acc;                                                 \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
/* The same for one slab of the operator: only the terms whose free index c   \
 * has spin sc and whose pair Q lies in the rows [q0, q1) of its block (first  \
 * pair hoff), with X the GEMM output for these rows,                         \
 *   R[I, A] += sum_s rs_s X[(Q - hoff - q0) nrows nc + ib nc + (c - coff)]. */ \
void zc_breduce_ppl_rows_##SFX(T *R, const int64_t *Rbase, const int64_t *Rcol, \
                               int64_t nrows, const int64_t *rows, int64_t nvt, \
                               int64_t A0, int64_t A1, const int64_t *vfree,   \
                               const int64_t *vpair, const int64_t *vspin,     \
                               int64_t sc, int64_t hoff, int64_t q0, int64_t q1, \
                               int64_t nc, int64_t coff, const T *X)           \
{                                                                              \
    const uint64_t nq = (uint64_t)(q1 - q0);                                   \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t ib = 0; ib < nrows; ib++) {                                   \
        T *r = R + Rbase[rows[ib]];                                            \
        for (int64_t A = A0; A < A1; A++) {                                    \
            T acc = 0;                                                         \
            int hit = 0;                                                       \
            for (int s = 0; s < 3; s++) {                                      \
                const int64_t c = vfree[s * nvt + A];                          \
                if (vspin[c] != sc) continue;                                  \
                const uint64_t q = (uint64_t)(vpair[s * nvt + A] - hoff - q0); \
                if (q >= nq) continue;                                         \
                const T v = X[(q * nrows + ib) * nc + (c - coff)];             \
                acc += (s == 1) ? -v : v;                                      \
                hit = 1;                                                       \
            }                                                                  \
            if (hit) r[Rcol[A]] += acc;                                        \
        }                                                                      \
    }                                                                          \
}
DEFINE_BLOCKED(double, d)
DEFINE_BLOCKED(double complex, z)


/* ---- The occupied operator of the T3 equation (hh ladder + F_oo) as a
 * matrix on packed occupied triples, written down directly
 * (zccsdt/cc/t3pack.py occ_operator_direct; docs/factorization.md 2.3):
 *   (O t)[ijk] = P(k/ij) sum_{m<n} W[m,n,i,j] t[mnk] - P(i/jk) sum_m F[m,i] t[mjk]
 * For the rows I = rows[a] (nI of the nto triples; loc[I] = a, -1 for the
 * triples that are not rows) and the columns J = rows[b]:
 *   O[a, b] += sgn_r(I) sgn[k, P] W[P; pair_r(I)]   k = free_r(I), J = tri[k, P]
 *   O[a, b] -= sgn_r(I) sgn[m, Q] F[m, free_r(I)]   Q = pair_r(I), J = tri[m, Q]
 * W: (no, no, no, no) as W[m, n, i, j]; F: (no, no) as F[m, i]; O: (nI, nI),
 * overwritten.  otri, osgn: (no, npo); ofree, opair: (3, nto); pa, pb: the
 * two indices of a pair.  The work is 3 (npo + no) per row. */
#define DEFINE_OCCOP(T, SFX)                                                   \
void zc_occ_operator_##SFX(T *O, int64_t nI, const int64_t *rows,              \
                           const int64_t *loc, int64_t no, int64_t npo,        \
                           int64_t nto, const int64_t *otri, const int8_t *osgn, \
                           const int64_t *ofree, const int64_t *opair,         \
                           const int64_t *pa, const int64_t *pb,               \
                           const T *W, const T *F)                             \
{                                                                              \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t a = 0; a < nI; a++) {                                         \
        T *o = O + a * nI;                                                     \
        const int64_t It = rows[a];                                             \
        memset(o, 0, sizeof(T) * nI);                                          \
        for (int r = 0; r < 3; r++) {                                          \
            const int64_t k = ofree[r * nto + It], Q = opair[r * nto + It];      \
            const int64_t p = pa[Q], q = pb[Q];                                \
            const double sr = (r == 1) ? -1.0 : 1.0;                           \
            for (int64_t P = 0; P < npo; P++) {                                \
                const int8_t s = osgn[k * npo + P];                            \
                if (!s) continue;                                              \
                const int64_t b = loc[otri[k * npo + P]];                      \
                if (b < 0) continue;                                           \
                o[b] += (sr * s) * W[((pa[P] * no + pb[P]) * no + p) * no + q]; \
            }                                                                  \
            for (int64_t m = 0; m < no; m++) {                                 \
                const int8_t s = osgn[m * npo + Q];                            \
                if (!s) continue;                                              \
                const int64_t b = loc[otri[m * npo + Q]];                      \
                if (b < 0) continue;                                           \
                o[b] -= (sr * s) * F[m * no + k];                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
}
DEFINE_OCCOP(double, d)
DEFINE_OCCOP(double complex, z)
