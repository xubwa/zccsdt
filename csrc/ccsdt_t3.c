/*
 * ccsdt_t3.c -- T3 part of one CCSDT iteration on packed T3, in C.
 *
 * Same algorithm and data layouts as zccsdt/cc/ccsdt_opt.py (see
 * docs/factorization.md).  Python builds the small intermediates and calls:
 *
 *   zc_t3_pass1_*  : T3 -> T1, T2 and T3 dressing of W_vvvo / W_vooo
 *   zc_t3_pass2_*  : ring + drive terms, antisymmetrized into R
 *   zc_t3_ppl_*    : pp ladder (with F_vv folded into W)
 *   zc_t3_update_* : T3 <- T3 + R / D
 * (the hh ladder is a single GEMM, done in Python).
 *
 * Packed T3: T3p[I][A], I = occupied triple i<j<k, A = virtual triple a<b<c.
 * Maps (from zccsdt.cc.t3pack.TripleMap, all C-contiguous):
 *   otri[x * npo + P], osgn[...]  : (free occ x, occ pair P) -> triple, sign
 *   ofree[r * nto + I], opair[...]: triple I -> r-th (free, pair), r = 0,1,2
 *   vtri, vsgn, vfree, vpair      : the same for virtual indices
 * The reduction signs for r = 0, 1, 2 are +1, -1, +1.
 * Batches of occupied pairs are contiguous ranges [p0, p1).
 *
 * Every function exists for double (_d) and double complex (_z).
 */

#include <complex.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef ZC_USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#include <omp.h>

/* stats[]: accumulated wall seconds: 0 gather, 1 gemm, 2 reduce/other,
 * 3 drive GEMM (pass 2) */
#define TIC(k) do { double _t = omp_get_wtime(); (void)0
#define TOC(k) stats[k] += omp_get_wtime() - _t; } while (0)

static const double RSGN[3] = {1.0, -1.0, 1.0};

static void gemm_d(int ta, int tb, int64_t m, int64_t n, int64_t k, double alpha,
                   const double *a, int64_t lda, const double *b, int64_t ldb,
                   double beta, double *c, int64_t ldc)
{
    cblas_dgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans,
                tb ? CblasTrans : CblasNoTrans, m, n, k, alpha, a, lda, b, ldb,
                beta, c, ldc);
}

static void gemm_z(int ta, int tb, int64_t m, int64_t n, int64_t k, double complex alpha,
                   const double complex *a, int64_t lda, const double complex *b,
                   int64_t ldb, double complex beta, double complex *c, int64_t ldc)
{
    cblas_zgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans,
                tb ? CblasTrans : CblasNoTrans, m, n, k, &alpha, a, lda, b, ldb,
                &beta, c, ldc);
}

static void gemv_d(int trans, int64_t m, int64_t n, double alpha, const double *a,
                   int64_t lda, const double *x, double beta, double *y)
{
    cblas_dgemv(CblasRowMajor, trans ? CblasTrans : CblasNoTrans, m, n, alpha,
                a, lda, x, 1, beta, y, 1);
}

static void gemv_z(int trans, int64_t m, int64_t n, double complex alpha,
                   const double complex *a, int64_t lda, const double complex *x,
                   double complex beta, double complex *y)
{
    cblas_zgemv(CblasRowMajor, trans ? CblasTrans : CblasNoTrans, m, n, &alpha,
                a, lda, x, 1, &beta, y, 1);
}

/* Plain row-major GEMM, C = alpha op(A) op(B) + beta C, exported so that
 * Python-side contractions use the same BLAS (and thread pool) as the
 * kernels.  alpha/beta are passed by pointer (complex for _z). */
void zc_gemm_d(int ta, int tb, int64_t m, int64_t n, int64_t k,
               const double *alpha, const double *a, int64_t lda,
               const double *b, int64_t ldb, const double *beta, double *c,
               int64_t ldc)
{
    gemm_d(ta, tb, m, n, k, *alpha, a, lda, b, ldb, *beta, c, ldc);
}

void zc_gemm_z(int ta, int tb, int64_t m, int64_t n, int64_t k,
               const double complex *alpha, const double complex *a, int64_t lda,
               const double complex *b, int64_t ldb, const double complex *beta,
               double complex *c, int64_t ldc)
{
    gemm_z(ta, tb, m, n, k, *alpha, a, lda, b, ldb, *beta, c, ldc);
}

#define DEFINE_T3(T, SFX)                                                      \
                                                                               \
/* T[P - p0, d, m, Q] = osgn * vsgn * T3p[otri[m, P], vtri[d, Q]] */          \
static void gather_half_##SFX(const T *T3p, int64_t nvt, int64_t p0, int64_t nbp, \
                              int64_t no, int64_t nv, int64_t npo, int64_t nvp, \
                              const int64_t *otri, const int8_t *osgn,         \
                              const int64_t *vtri, const int8_t *vsgn, T *out) \
{                                                                              \
    _Pragma("omp parallel for collapse(2) schedule(static)")                   \
    for (int64_t P = 0; P < nbp; P++)                                          \
    for (int64_t d = 0; d < nv; d++) {                                         \
        const int64_t *vt = vtri + d * nvp;                                    \
        const int8_t *vs = vsgn + d * nvp;                                     \
        for (int64_t m = 0; m < no; m++) {                                     \
            T *o = out + ((P * nv + d) * no + m) * nvp;                        \
            int8_t s = osgn[m * npo + p0 + P];                                 \
            if (s == 0) { memset(o, 0, sizeof(T) * nvp); continue; }           \
            const T *row = T3p + otri[m * npo + p0 + P] * nvt;                 \
            if (s > 0) for (int64_t Q = 0; Q < nvp; Q++) o[Q] = vs[Q] * row[vt[Q]]; \
            else       for (int64_t Q = 0; Q < nvp; Q++) o[Q] = -vs[Q] * row[vt[Q]]; \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
/* Pass 1.  For every batch of pairs, with T = T[P, d, m, Q]:                  \
 *   Xacc[(e|j), (i, Q)] += sum_{P,d} GPd[(e|j), P, d] T[P, d, i, Q]           \
 *   Yall[P, d, :]        = sum_{m,Q} T[P, d, m, Q] GmQ[(m, Q), :]              \
 *   r1[(d, m)]          += sum_{P,Q} T[P, d, m, Q] gpp[P, Q]                   \
 *   rA[P, Q]             = sum_{(d,m)} fT[(d, m)] T[P, d, m, Q]                \
 * GPd: (nv + no, npo, nv);  GmQ: (no * nvp, ny);  Xacc: (nv + no, no * nvp);  \
 * Yall: (npo, nv, ny);  r1: (nv * no);  rA: (npo, nvp). */                    \
void zc_t3_pass1_##SFX(const T *T3p, int64_t nto, int64_t nvt,                 \
                       int64_t no, int64_t nv, int64_t npo, int64_t nvp,       \
                       const int64_t *otri, const int8_t *osgn,                \
                       const int64_t *vtri, const int8_t *vsgn,                \
                       int64_t batch,                                          \
                       const T *GPd, const T *GmQ, int64_t ny,                 \
                       const T *gpp, const T *fT,                              \
                       T *Xacc, T *Yall, T *r1, T *rA, double *stats)                         \
{                                                                              \
    (void)nto;                                                                 \
    const int64_t dm = nv * no;                                                \
    T *buf = malloc(sizeof(T) * batch * dm * nvp);                             \
    for (int64_t p0 = 0; p0 < npo; p0 += batch) {                              \
        int64_t nbp = (p0 + batch <= npo) ? batch : npo - p0;                  \
        TIC(0);                                                                \
        gather_half_##SFX(T3p, nvt, p0, nbp, no, nv, npo, nvp, otri, osgn,     \
                          vtri, vsgn, buf);                                    \
        TOC(0);                                                                \
        TIC(1);                                                                \
        gemm_##SFX(0, 0, nv + no, no * nvp, nbp * nv, 1.0, GPd + p0 * nv,      \
                   npo * nv, buf, no * nvp, 1.0, Xacc, no * nvp);              \
        gemm_##SFX(0, 0, nbp * nv, ny, no * nvp, 1.0, buf, no * nvp, GmQ, ny,  \
                   0.0, Yall + p0 * nv * ny, ny);                              \
        TOC(1);                                                                \
        TIC(2);                                                                \
        /* r1 += T[P] (dm x nvp) gpp[P];  rA[P] = fT (dm) . T[P] */          \
        for (int64_t P = 0; P < nbp; P++) {                                    \
            gemv_##SFX(0, dm, nvp, 1.0, buf + P * dm * nvp, nvp,               \
                       gpp + (p0 + P) * nvp, 1.0, r1);                         \
            gemv_##SFX(1, dm, nvp, 1.0, buf + P * dm * nvp, nvp, fT, 0.0,      \
                       rA + (p0 + P) * nvp);                                   \
        }                                                                      \
        TOC(2);                                                                \
    }                                                                          \
    free(buf);                                                                 \
}                                                                              \
                                                                               \
/* Pass 2.  For every batch of pairs:                                          \
 *   Z[P, (a, i), Q]  = Wring[(a, i), (d, m)] T[P, (d, m), Q]                  \
 *   Z[(P, c), (i, Q)] -= Adrv[P, c, :] Bdrv[:, (i, Q)]   (both drive terms:   \
 *       Adrv = [t2mnT | WoT] (npo, nv, nv + no), Bdrv = [WvT2; t2P])           \
 *   R[I, A] += sum_{r: pair_r(I) in batch} rs_r sum_s rs_s                    \
 *              Z[pair_r(I) - p0, a_s(A), free_r(I), Q_s(A)]                   \
 * Wring: (nv * no, nv * no); Adrv: (npo, nv, nv + no); Bdrv: (nv + no,       \
 * no * nvp). */                                                               \
void zc_t3_pass2_##SFX(const T *T3p, int64_t nto, int64_t nvt,                 \
                       int64_t no, int64_t nv, int64_t npo, int64_t nvp,       \
                       const int64_t *otri, const int8_t *osgn,                \
                       const int64_t *ofree, const int64_t *opair,             \
                       const int64_t *vtri, const int8_t *vsgn,                \
                       const int64_t *vfree, const int64_t *vpair,             \
                       int64_t batch,                                          \
                       const T *Wring, const T *Adrv, const T *Bdrv,           \
                       T *R, double *stats)                       \
{                                                                              \
    const int64_t dm = nv * no;                                                \
    T *Tb = malloc(sizeof(T) * batch * dm * nvp);                              \
    T *Z = malloc(sizeof(T) * batch * dm * nvp);                               \
    const int64_t sP = dm * nvp, sa = no * nvp;                                \
    for (int64_t p0 = 0; p0 < npo; p0 += batch) {                              \
        int64_t nbp = (p0 + batch <= npo) ? batch : npo - p0;                  \
        TIC(0);                                                                \
        gather_half_##SFX(T3p, nvt, p0, nbp, no, nv, npo, nvp, otri, osgn,     \
                          vtri, vsgn, Tb);                                     \
        TOC(0);                                                                \
        TIC(1);                                                                \
        for (int64_t P = 0; P < nbp; P++)                                      \
            gemm_##SFX(0, 0, dm, nvp, dm, 1.0, Wring, dm, Tb + P * sP, nvp,    \
                       0.0, Z + P * sP, nvp);                                  \
        TOC(1);                                                                \
        TIC(3);                                                                \
        /* both drive terms in one GEMM: K = nv + no, one pass over Z */      \
        gemm_##SFX(0, 0, nbp * nv, no * nvp, nv + no, -1.0,                    \
                   Adrv + p0 * nv * (nv + no), nv + no, Bdrv, no * nvp, 1.0,   \
                   Z, no * nvp);                                               \
        TOC(3);                                                                \
        TIC(2);                                                                \
        _Pragma("omp parallel for schedule(dynamic, 4)")                       \
        for (int64_t it = 0; it < nto; it++) {                                    \
            T *r = R + it * nvt;                                                \
            for (int rr = 0; rr < 3; rr++) {                                   \
                int64_t P = opair[rr * nto + it] - p0;                          \
                if (P < 0 || P >= nbp) continue;                               \
                const T *z = Z + P * sP + ofree[rr * nto + it] * nvp;           \
                double sg = RSGN[rr];                                          \
                for (int64_t A = 0; A < nvt; A++) {                            \
                    T v = z[vfree[A] * sa + vpair[A]]                          \
                        - z[vfree[nvt + A] * sa + vpair[nvt + A]]              \
                        + z[vfree[2 * nvt + A] * sa + vpair[2 * nvt + A]];     \
                    r[A] += sg * v;                                            \
                }                                                              \
            }                                                                  \
        }                                                                      \
        TOC(2);                                                                \
    }                                                                          \
    free(Tb); free(Z);                                                         \
}                                                                              \
                                                                               \
/* pp ladder: for batches of triples I,                                        \
 *   Tv[Q, I, c] = vsgn[c, Q] T3p[I, vtri[c, Q]]                               \
 *   X = Wpp[Q', Q] Tv[Q, (I, c)]                                              \
 *   R[I, A] += sum_s rs_s X[Q_s(A), I, c_s(A)] */                             \
void zc_t3_ppl_##SFX(const T *T3p, int64_t nto, int64_t nvt, int64_t nv,       \
                     int64_t nvp, const int64_t *vtri, const int8_t *vsgn,     \
                     const int64_t *vfree, const int64_t *vpair,               \
                     int64_t batch, const T *Wpp, T *R, double *stats)                        \
{                                                                              \
    T *Tv = malloc(sizeof(T) * batch * nvp * nv);                              \
    T *X = malloc(sizeof(T) * batch * nvp * nv);                               \
    for (int64_t i0 = 0; i0 < nto; i0 += batch) {                              \
        int64_t nb = (i0 + batch <= nto) ? batch : nto - i0;                   \
        TIC(0);                                                                \
        _Pragma("omp parallel for collapse(2) schedule(static)")               \
        for (int64_t Q = 0; Q < nvp; Q++)                                      \
        for (int64_t ib = 0; ib < nb; ib++) {                                  \
            const T *row = T3p + (i0 + ib) * nvt;                              \
            T *o = Tv + (Q * nb + ib) * nv;                                    \
            for (int64_t c = 0; c < nv; c++)                                   \
                o[c] = vsgn[c * nvp + Q] * row[vtri[c * nvp + Q]];             \
        }                                                                      \
        TOC(0);                                                                \
        TIC(1);                                                                \
        gemm_##SFX(0, 0, nvp, nb * nv, nvp, 1.0, Wpp, nvp, Tv, nb * nv, 0.0,   \
                   X, nb * nv);                                                \
        TOC(1);                                                                \
        TIC(2);                                                                \
        const int64_t sQ = nb * nv;                                            \
        _Pragma("omp parallel for schedule(static)")                           \
        for (int64_t ib = 0; ib < nb; ib++) {                                  \
            const T *x = X + ib * nv;                                          \
            T *r = R + (i0 + ib) * nvt;                                        \
            for (int64_t A = 0; A < nvt; A++)                                  \
                r[A] += x[vpair[A] * sQ + vfree[A]]                            \
                      - x[vpair[nvt + A] * sQ + vfree[nvt + A]]                \
                      + x[vpair[2 * nvt + A] * sQ + vfree[2 * nvt + A]];       \
        }                                                                      \
        TOC(2);                                                                \
    }                                                                          \
    free(Tv); free(X);                                                         \
}                                                                              \
                                                                               \
/* R[I, A] += sum_J O6[I, J] T3p[J, A]  (hh ladder + F_oo operator) */       \
void zc_t3_occop_##SFX(const T *O6, const T *T3p, int64_t nto, int64_t nvt,    \
                       T *R)                                                   \
{                                                                              \
    gemm_##SFX(0, 0, nto, nvt, nto, 1.0, O6, nto, T3p, nvt, 1.0, R, nvt);      \
}                                                                              \
                                                                               \
/* R[I, A] = T3p[I, A] + R[I, A] / (eo[I] - ev[A])  (new T3 written over R) */ \
void zc_t3_update_##SFX(const T *T3p, T *R, int64_t nto, int64_t nvt,          \
                        const double *eo, const double *ev)                    \
{                                                                              \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t it = 0; it < nto; it++) {                                     \
        const T *t = T3p + it * nvt;                                           \
        T *r = R + it * nvt;                                                   \
        const double e = eo[it];                                               \
        for (int64_t A = 0; A < nvt; A++)                                      \
            r[A] = t[A] + r[A] * (1.0 / (e - ev[A]));                          \
    }                                                                          \
}

DEFINE_T3(double, d)
DEFINE_T3(double complex, z)
