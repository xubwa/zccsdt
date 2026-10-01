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
