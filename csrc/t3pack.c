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
