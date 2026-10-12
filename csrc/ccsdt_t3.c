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

#define _GNU_SOURCE
#include <complex.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef ZC_USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#include <omp.h>
#include <sys/mman.h>
#include <unistd.h>

/* T3 and its residual R may be views of a mapped scratch file (not enough
 * memory).  ``io`` then describes them (NULL: both in ordinary memory):
 *   io[0]  T3 is mapped
 *   io[1]  file descriptor of R (-1: not mapped), io[2] file offset of R[0]
 * and the kernels tell the operating system what they do (zccsdt/scratch.py):
 * the rows of a batch are announced (large sequential reads instead of page
 * faults scattered over the threads), rows of T3 a batch has read leave the
 * page tables (clean pages that are not mapped are cheap to take back under
 * a memory limit), and R is sent to the file as soon as it holds the new T3.
 * While the passes add to R it is left alone: writing it out after each
 * pass writes it several times, and unmapping pages that are still dirty
 * only costs a fault per page at the next pass (both were measured). */
static uintptr_t page_size(void)
{
    static uintptr_t page = 0;
    if (!page) page = (uintptr_t)sysconf(_SC_PAGESIZE);
    return page;
}

static void will_need(const void *p, size_t nbytes)
{
    const uintptr_t page = page_size();
    const uintptr_t lo = (uintptr_t)p & ~(page - 1);
    const uintptr_t hi = (uintptr_t)p + nbytes;
    if (hi > lo) madvise((void *)lo, hi - lo, MADV_WILLNEED);
}

/* whole pages inside [p, p + nbytes) leave the page tables (the data stays
 * in the page cache or goes to the file) */
static void unmap(const void *p, size_t nbytes)
{
    const uintptr_t page = page_size();
    const uintptr_t lo = ((uintptr_t)p + page - 1) & ~(page - 1);
    const uintptr_t hi = ((uintptr_t)p + nbytes) & ~(page - 1);
    if (hi > lo) madvise((void *)lo, hi - lo, MADV_DONTNEED);
}

/* [p, p + nbytes) of R (first element at base) was written: start writing
 * it to the file, and drop it from the page tables */
static void written(const int64_t *io, const void *base, const void *p, size_t nbytes)
{
    if (!io || io[1] < 0) return;
    sync_file_range((int)io[1], io[2] + ((const char *)p - (const char *)base),
                    (off64_t)nbytes, SYNC_FILE_RANGE_WRITE);
    unmap(p, nbytes);
}

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

/* A list of products in one parallel region (the passes over the blocks of a
 * symmetry label make thousands of products of a millisecond each, which a
 * threaded BLAS does not run at its rate, least of all on many cores):
 *   C_t = alpha A_t B_t + beta C_t,  t < ntask,
 * with tasks[t] = (m, n, k, offset of A_t in a, lda, offset of B_t in b, ldb,
 * offset of C_t in c, ldc), offsets in elements, all matrices row-major.
 * Every product is split over its columns into pieces of about ``cols``
 * columns; a piece is one sequential GEMM, and the threads take the pieces
 * of all tasks from one queue. */
#define DEFINE_GEMM_TASKS(T, SFX)                                              \
void zc_gemm_tasks_##SFX(int64_t ntask, const int64_t *tasks, const T *a,      \
                         const T *b, T *c, const T *alpha, const T *beta,      \
                         int64_t cols)                                         \
{                                                                              \
    if (cols <= 0) cols = 96;                                                  \
    int64_t total = 0;                                                         \
    for (int64_t t = 0; t < ntask; t++) {                                      \
        const int64_t *q = tasks + 9 * t;                                      \
        if (q[0] > 0 && q[1] > 0) total += (q[1] + cols - 1) / cols;           \
    }                                                                          \
    if (total == 0) return;                                                    \
    int64_t *piece = malloc(sizeof(int64_t) * 2 * total);                      \
    int64_t np = 0;                                                            \
    for (int64_t t = 0; t < ntask; t++) {                                      \
        const int64_t *q = tasks + 9 * t;                                      \
        if (q[0] <= 0 || q[1] <= 0) continue;                                  \
        const int64_t n = (q[1] + cols - 1) / cols;                            \
        for (int64_t j = 0; j < n; j++) {                                      \
            piece[2 * np] = t;                                                 \
            piece[2 * np + 1] = j * q[1] / n;                                  \
            np++;                                                              \
        }                                                                      \
    }                                                                          \
    _Pragma("omp parallel for schedule(dynamic, 1)")                           \
    for (int64_t p = 0; p < np; p++) {                                         \
        const int64_t *q = tasks + 9 * piece[2 * p];                           \
        const int64_t c0 = piece[2 * p + 1];                                   \
        const int64_t c1 = (p + 1 < np && piece[2 * p + 2] == piece[2 * p])    \
                           ? piece[2 * p + 3] : q[1];                          \
        if (q[2] > 0)                                                          \
            gemm_##SFX(0, 0, q[0], c1 - c0, q[2], *alpha, a + q[3], q[4],      \
                       b + q[5] + c0, q[6], *beta, c + q[7] + c0, q[8]);       \
        else                                                                   \
            for (int64_t i = 0; i < q[0]; i++)                                 \
                for (int64_t j = c0; j < c1; j++)                              \
                    c[q[7] + i * q[8] + j] *= *beta;                           \
    }                                                                          \
    free(piece);                                                               \
}
DEFINE_GEMM_TASKS(double, d)
DEFINE_GEMM_TASKS(double complex, z)

/* Batched GEMM over slices: C_i = alpha op(A) B_i + beta C_i for i < nb, with
 * B_i = b + i sb (k x n, leading dimension ldb) and C_i = c + i sc (m x n,
 * leading dimension ldc).  One slice per thread, BLAS sequential inside. */
void zc_gemm_slices_d(int ta, int64_t nb, int64_t m, int64_t n, int64_t k,
                      const double *alpha, const double *a, int64_t lda,
                      const double *b, int64_t sb, int64_t ldb, const double *beta,
                      double *c, int64_t sc, int64_t ldc)
{
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < nb; i++)
        gemm_d(ta, 0, m, n, k, *alpha, a, lda, b + i * sb, ldb, *beta, c + i * sc, ldc);
}

/* C_i = A_i B_i for i < nb, each product with matrices of its own:
 * A_i = a + i sa (m x k), B_i = b + i sb (k x n), C_i = c + i sc (m x n),
 * row-major with the leading dimensions given.  One product per thread, BLAS
 * sequential inside: for many products of a few milliseconds each, which a
 * threaded BLAS does not run at its rate. */
void zc_gemm_each_d(int64_t nb, int64_t m, int64_t n, int64_t k,
                    const double *a, int64_t sa, int64_t lda,
                    const double *b, int64_t sb, int64_t ldb,
                    double *c, int64_t sc, int64_t ldc)
{
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < nb; i++)
        gemm_d(0, 0, m, n, k, 1.0, a + i * sa, lda, b + i * sb, ldb, 0.0, c + i * sc, ldc);
}

void zc_gemm_each_z(int64_t nb, int64_t m, int64_t n, int64_t k,
                    const double complex *a, int64_t sa, int64_t lda,
                    const double complex *b, int64_t sb, int64_t ldb,
                    double complex *c, int64_t sc, int64_t ldc)
{
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < nb; i++)
        gemm_z(0, 0, m, n, k, 1.0, a + i * sa, lda, b + i * sb, ldb, 0.0, c + i * sc, ldc);
}

#define DEFINE_T3(T, SFX)                                                      \
                                                                               \
/* T[P - p0, d, m, Q] = osgn * vsgn * T3p[otri[m, P], vtri[d, Q]] */          \
static void gather_half_##SFX(const T *T3p, int64_t nvt, int64_t p0, int64_t nbp, \
                              int64_t no, int64_t nv, int64_t npo, int64_t nvp, \
                              const int64_t *otri, const int8_t *osgn,         \
                              const int64_t *vtri, const int8_t *vsgn, T *out, \
                              const int64_t *io)                               \
{                                                                              \
    const int mapped = io && io[0];                                            \
    if (mapped)                                                                \
        for (int64_t P = 0; P < nbp; P++)                                      \
        for (int64_t m = 0; m < no; m++)                                       \
            if (osgn[m * npo + p0 + P])                                        \
                will_need(T3p + otri[m * npo + p0 + P] * nvt, sizeof(T) * nvt); \
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
    if (mapped)                                                                \
        for (int64_t P = 0; P < nbp; P++)                                      \
        for (int64_t m = 0; m < no; m++)                                       \
            if (osgn[m * npo + p0 + P])                                        \
                unmap(T3p + otri[m * npo + p0 + P] * nvt, sizeof(T) * nvt);    \
}                                                                              \
                                                                               \
/* Pass 1.  For every batch of pairs, with T = T[P, d, m, Q]:                  \
 *   Xacc[(e|j), (i, Q)] += sum_{P,d} GPd[(e|j), P, d] T[P, d, i, Q]           \
 *   Yall[P, d, :]        = sum_{m,Q} T[P, d, m, Q] GmQ[(m, Q), :]              \
 *   r1[(d, m)]          += sum_{P,Q} T[P, d, m, Q] gpp[P, Q]                   \
 *   rA[P, Q]             = sum_{(d,m)} fT[(d, m)] T[P, d, m, Q]                \
 * GPd: (nv + no, npo, nv);  GmQ: (no * nvp, ny);  Xacc: (nv + no, no * nvp);  \
 * Yall: (npo, nv, ny);  r1: (nv * no);  rA: (npo, nvp). */                    \
void zc_t3_pass1w_##SFX(const T *T3p, int64_t nto, int64_t nvt,                \
                       int64_t no, int64_t nv, int64_t npo, int64_t nvp,       \
                       const int64_t *otri, const int8_t *osgn,                \
                       const int64_t *vtri, const int8_t *vsgn,                \
                       int64_t batch,                                          \
                       const T *GPd, const T *GmQ, int64_t ny,                 \
                       const T *gpp, const T *fT,                              \
                       T *Xacc, T *Yall, T *r1, T *rA, const int64_t *io,      \
                       double *stats, T *buf)                                  \
{                                                                              \
    (void)nto;                                                                 \
    const int64_t dm = nv * no;                                                \
    for (int64_t p0 = 0; p0 < npo; p0 += batch) {                              \
        int64_t nbp = (p0 + batch <= npo) ? batch : npo - p0;                  \
        TIC(0);                                                                \
        gather_half_##SFX(T3p, nvt, p0, nbp, no, nv, npo, nvp, otri, osgn,     \
                          vtri, vsgn, buf, io);                                \
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
}                                                                              \
                                                                               \
/* pass 1 with a batch buffer of its own */                                    \
void zc_t3_pass1_##SFX(const T *T3p, int64_t nto, int64_t nvt,                 \
                       int64_t no, int64_t nv, int64_t npo, int64_t nvp,       \
                       const int64_t *otri, const int8_t *osgn,                \
                       const int64_t *vtri, const int8_t *vsgn,                \
                       int64_t batch,                                          \
                       const T *GPd, const T *GmQ, int64_t ny,                 \
                       const T *gpp, const T *fT,                              \
                       T *Xacc, T *Yall, T *r1, T *rA, const int64_t *io,      \
                       double *stats)                                          \
{                                                                              \
    T *buf = malloc(sizeof(T) * batch * nv * no * nvp);                        \
    zc_t3_pass1w_##SFX(T3p, nto, nvt, no, nv, npo, nvp, otri, osgn, vtri, vsgn,\
                       batch, GPd, GmQ, ny, gpp, fT, Xacc, Yall, r1, rA, io,   \
                       stats, buf);                                            \
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
void zc_t3_pass2w_##SFX(const T *T3p, int64_t nto, int64_t nvt,                \
                       int64_t no, int64_t nv, int64_t npo, int64_t nvp,       \
                       const int64_t *otri, const int8_t *osgn,                \
                       const int64_t *ofree, const int64_t *opair,             \
                       const int64_t *vtri, const int8_t *vsgn,                \
                       const int64_t *vfree, const int64_t *vpair,             \
                       int64_t batch,                                          \
                       const T *Wring, const T *Adrv, const T *Bdrv,           \
                       T *R, const int64_t *io, double *stats, T *Tb, T *Z)    \
{                                                                              \
    const int64_t dm = nv * no;                                                \
    const int64_t sP = dm * nvp, sa = no * nvp;                                \
    for (int64_t p0 = 0; p0 < npo; p0 += batch) {                              \
        int64_t nbp = (p0 + batch <= npo) ? batch : npo - p0;                  \
        TIC(0);                                                                \
        gather_half_##SFX(T3p, nvt, p0, nbp, no, nv, npo, nvp, otri, osgn,     \
                          vtri, vsgn, Tb, io);                                 \
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
        const int rmap = io && io[1] >= 0;                                     \
        if (rmap)                         /* the rows of R this batch adds to */ \
            for (int64_t it = 0; it < nto; it++)                               \
                for (int rr = 0; rr < 3; rr++) {                               \
                    int64_t P = opair[rr * nto + it] - p0;                     \
                    if (P >= 0 && P < nbp) {                                   \
                        will_need(R + it * nvt, sizeof(T) * nvt);              \
                        break;                                                 \
                    }                                                          \
                }                                                              \
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
}                                                                              \
                                                                               \
/* pass 2 with batch buffers of its own */                                     \
void zc_t3_pass2_##SFX(const T *T3p, int64_t nto, int64_t nvt,                 \
                       int64_t no, int64_t nv, int64_t npo, int64_t nvp,       \
                       const int64_t *otri, const int8_t *osgn,                \
                       const int64_t *ofree, const int64_t *opair,             \
                       const int64_t *vtri, const int8_t *vsgn,                \
                       const int64_t *vfree, const int64_t *vpair,             \
                       int64_t batch,                                          \
                       const T *Wring, const T *Adrv, const T *Bdrv,           \
                       T *R, const int64_t *io, double *stats)                 \
{                                                                              \
    T *Tb = malloc(sizeof(T) * batch * nv * no * nvp);                         \
    T *Z = malloc(sizeof(T) * batch * nv * no * nvp);                          \
    zc_t3_pass2w_##SFX(T3p, nto, nvt, no, nv, npo, nvp, otri, osgn, ofree,     \
                       opair, vtri, vsgn, vfree, vpair, batch, Wring, Adrv,    \
                       Bdrv, R, io, stats, Tb, Z);                             \
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
                     int64_t batch, const T *Wpp, T *R, const int64_t *io,     \
                     double *stats)                                            \
{                                                                              \
    T *Tv = malloc(sizeof(T) * batch * nvp * nv);                              \
    T *X = malloc(sizeof(T) * batch * nvp * nv);                               \
    for (int64_t i0 = 0; i0 < nto; i0 += batch) {                              \
        int64_t nb = (i0 + batch <= nto) ? batch : nto - i0;                   \
        if (io && io[0]) will_need(T3p + i0 * nvt, sizeof(T) * nb * nvt);      \
        if (io && io[1] >= 0) will_need(R + i0 * nvt, sizeof(T) * nb * nvt);   \
        TIC(0);                                                                \
        _Pragma("omp parallel for collapse(2) schedule(static)")               \
        for (int64_t Q = 0; Q < nvp; Q++)                                      \
        for (int64_t ib = 0; ib < nb; ib++) {                                  \
            const T *row = T3p + (i0 + ib) * nvt;                              \
            T *o = Tv + (Q * nb + ib) * nv;                                    \
            for (int64_t c = 0; c < nv; c++)                                   \
                o[c] = vsgn[c * nvp + Q] * row[vtri[c * nvp + Q]];             \
        }                                                                      \
        if (io && io[0]) unmap(T3p + i0 * nvt, sizeof(T) * nb * nvt);          \
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
/* R[I, A] += sum_J O6[I, J] T3p[J, A]  (hh ladder + F_oo operator), in      \
 * blocks of ncb columns (ncb <= 0: all at once): a GEMM then touches one      \
 * column block of T3p and of R, which is what must be in memory when the two  \
 * are mapped files. */                                                        \
void zc_t3_occop_##SFX(const T *O6, const T *T3p, int64_t nto, int64_t nvt,    \
                       int64_t ncb, const int64_t *io, T *R)                   \
{                                                                              \
    const int tmap = io && io[0], rmap = io && io[1] >= 0;                     \
    if (ncb <= 0 || ncb > nvt) ncb = nvt;                                      \
    for (int64_t c0 = 0; c0 < nvt; c0 += ncb) {                                \
        const int64_t nc = (c0 + ncb <= nvt) ? ncb : nvt - c0;                 \
        for (int64_t it = 0; it < nto && (tmap || rmap); it++) {               \
            if (tmap) will_need(T3p + it * nvt + c0, sizeof(T) * nc);          \
            if (rmap) will_need(R + it * nvt + c0, sizeof(T) * nc);            \
        }                                                                      \
        gemm_##SFX(0, 0, nto, nc, nto, 1.0, O6, nto, T3p + c0, nvt, 1.0,       \
                   R + c0, nvt);                                               \
        for (int64_t it = 0; it < nto && tmap; it++)                           \
            unmap(T3p + it * nvt + c0, sizeof(T) * nc);                        \
    }                                                                          \
}                                                                              \
                                                                               \
/* ---- hh ladder + F_oo, pair-driven (docs/factorization.md 2.3): the same    \
 * operator as zc_t3_occop without its C(o,3) x C(o,3) matrix.  With the       \
 * occupied triples written as (spectator k; pair P),                          \
 *   R[(k; P), A] += sum_{P'} W'[P, P'] T[(k; P'), A],                         \
 *   W'[(i<j), (m<n)] = W[m,n,i,j] - 1/2 (F[m,i] d_nj - F[n,i] d_mj            \
 *                                        - F[m,j] d_ni + F[n,j] d_mi)         \
 * (each F term is met from two spectators, hence the half).  For every        \
 * spectator k the rows of the block of T3 that contain k are listed:          \
 * entries koff[k] <= e < koff[k+1] with the row krow[e] (local), its pair     \
 * kpair[e] and the sign ksgn[e] of (k; pair) -> triple.                        \
 *                                                                             \
 * zc_t3_hhpair_w: the matrices of all spectators, one after the other,        \
 * Wk[woff[k] + x n_k + y] = s_x W'[P_x, P_y] s_y (the signs folded in; with   \
 * ``trans`` the transposed operator, W'[P_y, P_x]: the Lambda equations).     \
 * W: (no, no, no, no) as W[m,n,i,j]; F: (no, no) as F[m,i]. */                 \
void zc_t3_hhpair_w_##SFX(T *Wk, const int64_t *woff, int64_t no,              \
                          const int64_t *koff, const int64_t *kpair,           \
                          const int8_t *ksgn, const int64_t *pa,               \
                          const int64_t *pb, const T *W, const T *F, int trans) \
{                                                                              \
    _Pragma("omp parallel for schedule(dynamic, 1)")                           \
    for (int64_t k = 0; k < no; k++) {                                         \
        const int64_t e0 = koff[k], n = koff[k + 1] - e0;                      \
        T *w = Wk + woff[k];                                                   \
        for (int64_t x = 0; x < n; x++)                                        \
        for (int64_t y = 0; y < n; y++) {                                      \
            const int64_t Pi = kpair[e0 + (trans ? y : x)];                    \
            const int64_t Pm = kpair[e0 + (trans ? x : y)];                    \
            const int64_t i = pa[Pi], j = pb[Pi], m = pa[Pm], nn = pb[Pm];     \
            T v = W[((m * no + nn) * no + i) * no + j];                        \
            if (nn == j) v -= 0.5 * F[m * no + i];                             \
            if (m == j) v += 0.5 * F[nn * no + i];                             \
            if (nn == i) v += 0.5 * F[m * no + j];                             \
            if (m == i) v -= 0.5 * F[nn * no + j];                             \
            w[x * n + y] = (double)(ksgn[e0 + x] * ksgn[e0 + y]) * v;          \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
/* R[row, A] += the operator on T3 (both (nrow, ncol), a block of the packed   \
 * arrays): the threads share the columns in chunks of nc; a thread gathers,   \
 * for one spectator after the other, the rows of its chunk that contain it    \
 * (n_k x nc, in cache), multiplies with Wk (a GEMM of its own) and adds the   \
 * result to the rows of R.  T3 and R are read and written once per           \
 * spectator of a row, i.e. three times, a chunk at a time. */                 \
void zc_t3_hhpair_##SFX(const T *T3, T *R, int64_t nrow, int64_t ncol,         \
                        int64_t no, const int64_t *koff, const int64_t *krow,  \
                        const T *Wk, const int64_t *woff, int64_t nc)          \
{                                                                              \
    (void)nrow;                                                                \
    int64_t nmax = 0;                                                          \
    for (int64_t k = 0; k < no; k++)                                           \
        if (koff[k + 1] - koff[k] > nmax) nmax = koff[k + 1] - koff[k];        \
    if (nmax == 0 || ncol == 0) return;                                        \
    if (nc <= 0) nc = 64;                                                      \
    const int64_t nchunk = (ncol + nc - 1) / nc;                               \
    _Pragma("omp parallel")                                                    \
    {                                                                          \
        T *Y = malloc(sizeof(T) * 2 * nmax * nc);                              \
        T *Z = Y + nmax * nc;                                                  \
        _Pragma("omp for schedule(dynamic, 1)")                                \
        for (int64_t ch = 0; ch < nchunk; ch++) {                              \
            const int64_t c0 = ch * nc;                                        \
            const int64_t n1 = (c0 + nc <= ncol) ? nc : ncol - c0;             \
            for (int64_t k = 0; k < no; k++) {                                 \
                const int64_t e0 = koff[k], n = koff[k + 1] - e0;              \
                if (n == 0) continue;                                          \
                for (int64_t x = 0; x < n; x++)                                \
                    memcpy(Y + x * n1, T3 + krow[e0 + x] * ncol + c0,          \
                           sizeof(T) * n1);                                    \
                gemm_##SFX(0, 0, n, n1, n, 1.0, Wk + woff[k], n, Y, n1, 0.0,   \
                           Z, n1);                                             \
                for (int64_t x = 0; x < n; x++) {                              \
                    T *r = R + krow[e0 + x] * ncol + c0;                       \
                    const T *z = Z + x * n1;                                   \
                    for (int64_t c = 0; c < n1; c++) r[c] += z[c];             \
                }                                                              \
            }                                                                  \
        }                                                                      \
        free(Y);                                                               \
    }                                                                          \
}                                                                              \
                                                                               \
/* R[I, A] = T3p[I, A] + R[I, A] / (eo[I] - ev[A])  (new T3 written over R) */ \
void zc_t3_update_##SFX(const T *T3p, T *R, int64_t nto, int64_t nvt,          \
                        const double *eo, const double *ev, const int64_t *io) \
{                                                                              \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t it = 0; it < nto; it++) {                                     \
        const T *t = T3p + it * nvt;                                           \
        T *r = R + it * nvt;                                                   \
        const double e = eo[it];                                               \
        for (int64_t A = 0; A < nvt; A++)                                      \
            r[A] = t[A] + r[A] * (1.0 / (e - ev[A]));                          \
        if (io && io[0]) unmap(t, sizeof(T) * nvt);                            \
        written(io, R, r, sizeof(T) * nvt);                                    \
    }                                                                          \
}                                                                              \
                                                                               \
/* ---- The passes of a Lambda iteration over the packed Lambda_3 (the        \
 * scheduler of zccsdt/cc/lambda_opt.py: DenseL3.pass1, pass2; docs/lambda.md  \
 * section 9), with the batch buffers of the caller.                           \
 *                                                                             \
 * Pass 1.  With L = L[P, d, m, Q] and T = T[P, d, m, Q], the half-expanded    \
 * Lambda_3 and T3 of a batch of occupied pairs:                               \
 *   Xacc[x, (i, Q)]  += sum_{P,d} GPd[x, P, d] L[P, d, i, Q]                   \
 *   Yall[P, d, y]     = sum_{m,Q} L[P, d, m, Q] GmQ[(m, Q), y]                 \
 *   Dr[(a,i), (g,o)] += sum_{P,Q} L[P, a, i, Q] T[P, g, o, Q]                  \
 *   D1[(g, o)]       += sum_{P,Q} T[P, g, o, Q] l2pp[P, Q]                     \
 * GPd: (nv + no, npo, nv);  GmQ: (no * nvp, ny);  Xacc: (nv + no, no * nvp);  \
 * Yall: (npo, nv, ny);  Dr: (nv * no, nv * no);  D1: (nv * no);               \
 * l2pp: (npo, nvp);  bufL, bufT: (batch, nv, no, nvp) each.                   \
 * stats: 0 gather, 1 the stacked GEMMs, 2 the density. */                    \
void zc_l3_pass1w_##SFX(const T *L3p, const T *T3p, int64_t nto, int64_t nvt,  \
                        int64_t no, int64_t nv, int64_t npo, int64_t nvp,      \
                        const int64_t *otri, const int8_t *osgn,               \
                        const int64_t *vtri, const int8_t *vsgn,               \
                        int64_t batch,                                         \
                        const T *GPd, const T *GmQ, int64_t ny,                \
                        const T *l2pp, T *Xacc, T *Yall, T *Dr, T *D1,         \
                        double *stats, T *bufL, T *bufT)                       \
{                                                                              \
    (void)nto;                                                                 \
    const int64_t dm = nv * no;                                                \
    for (int64_t p0 = 0; p0 < npo; p0 += batch) {                              \
        int64_t nbp = (p0 + batch <= npo) ? batch : npo - p0;                  \
        TIC(0);                                                                \
        gather_half_##SFX(L3p, nvt, p0, nbp, no, nv, npo, nvp, otri, osgn,     \
                          vtri, vsgn, bufL, NULL);                             \
        gather_half_##SFX(T3p, nvt, p0, nbp, no, nv, npo, nvp, otri, osgn,     \
                          vtri, vsgn, bufT, NULL);                             \
        TOC(0);                                                                \
        TIC(1);                                                                \
        gemm_##SFX(0, 0, nv + no, no * nvp, nbp * nv, 1.0, GPd + p0 * nv,      \
                   npo * nv, bufL, no * nvp, 1.0, Xacc, no * nvp);             \
        gemm_##SFX(0, 0, nbp * nv, ny, no * nvp, 1.0, bufL, no * nvp, GmQ, ny, \
                   0.0, Yall + p0 * nv * ny, ny);                              \
        TOC(1);                                                                \
        TIC(2);                                                                \
        for (int64_t P = 0; P < nbp; P++) {                                    \
            gemm_##SFX(0, 1, dm, dm, nvp, 1.0, bufL + P * dm * nvp, nvp,       \
                       bufT + P * dm * nvp, nvp, 1.0, Dr, dm);                 \
            gemv_##SFX(0, dm, nvp, 1.0, bufT + P * dm * nvp, nvp,              \
                       l2pp + (p0 + P) * nvp, 1.0, D1);                        \
        }                                                                      \
        TOC(2);                                                                \
    }                                                                          \
}                                                                              \
                                                                               \
/* Pass 2.  For every batch of pairs:                                          \
 *   Z[P, (a, i), Q]   = sum_(d,m) WL[(a, i), (d, m)] L[P, (d, m), Q]           \
 *                       + sum_x U[(a, i), x] Vp[P, x, Q]                       \
 *   Z[(P, c), (i, Q)] -= sum_k Adrv[P, c, k] Bdrv[k, (i, Q)]                   \
 *   R[I, A] += P(i/P) P(a/Q) Z   (the reduction of zc_t3_pass2w)              \
 * WL: (nv no, nv no);  U: (nv no, nu);  Vp: (npo, nu, nvp);  Adrv: (npo, nv,  \
 * K);  Bdrv: (K, no nvp);  Tb, Z: (batch, nv, no, nvp) each.                  \
 * stats: 0 gather, 1 ring GEMM, 2 reduction, 3 drive GEMM. */                \
void zc_l3_pass2w_##SFX(const T *L3p, int64_t nto, int64_t nvt,                \
                        int64_t no, int64_t nv, int64_t npo, int64_t nvp,      \
                        const int64_t *otri, const int8_t *osgn,               \
                        const int64_t *ofree, const int64_t *opair,            \
                        const int64_t *vtri, const int8_t *vsgn,               \
                        const int64_t *vfree, const int64_t *vpair,            \
                        int64_t batch, const T *WL, const T *U, int64_t nu,    \
                        const T *Vp, const T *Adrv, int64_t K, const T *Bdrv,  \
                        T *R, double *stats, T *Tb, T *Z)                      \
{                                                                              \
    const int64_t dm = nv * no;                                                \
    const int64_t sP = dm * nvp, sa = no * nvp;                                \
    for (int64_t p0 = 0; p0 < npo; p0 += batch) {                              \
        int64_t nbp = (p0 + batch <= npo) ? batch : npo - p0;                  \
        TIC(0);                                                                \
        gather_half_##SFX(L3p, nvt, p0, nbp, no, nv, npo, nvp, otri, osgn,     \
                          vtri, vsgn, Tb, NULL);                               \
        TOC(0);                                                                \
        TIC(1);                                                                \
        for (int64_t P = 0; P < nbp; P++) {                                    \
            gemm_##SFX(0, 0, dm, nvp, dm, 1.0, WL, dm, Tb + P * sP, nvp,       \
                       0.0, Z + P * sP, nvp);                                  \
            if (nu > 0)                                                        \
                gemm_##SFX(0, 0, dm, nvp, nu, 1.0, U, nu,                      \
                           Vp + (p0 + P) * nu * nvp, nvp, 1.0, Z + P * sP, nvp); \
        }                                                                      \
        TOC(1);                                                                \
        TIC(3);                                                                \
        gemm_##SFX(0, 0, nbp * nv, no * nvp, K, -1.0, Adrv + p0 * nv * K, K,   \
                   Bdrv, no * nvp, 1.0, Z, no * nvp);                          \
        TOC(3);                                                                \
        TIC(2);                                                                \
        _Pragma("omp parallel for schedule(dynamic, 4)")                       \
        for (int64_t it = 0; it < nto; it++) {                                 \
            T *r = R + it * nvt;                                               \
            for (int rr = 0; rr < 3; rr++) {                                   \
                int64_t P = opair[rr * nto + it] - p0;                         \
                if (P < 0 || P >= nbp) continue;                               \
                const T *z = Z + P * sP + ofree[rr * nto + it] * nvp;          \
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
}                                                                              \
                                                                               \
/* ---- The passes for the pairs plo <= P < phi alone (in batches from plo):  \
 * what zc_t3_pass1w, zc_t3_pass2w, zc_l3_pass1w, zc_l3_pass2w do for all      \
 * pairs, for a caller that deals the batches out (the ranks of xzccsdt).      \
 * Rows of Yall and rA of other pairs are not touched. */                      \
void zc_t3_pass1r_##SFX(const T *T3p, int64_t nto, int64_t nvt,                \
                       int64_t no, int64_t nv, int64_t npo, int64_t nvp,       \
                       const int64_t *otri, const int8_t *osgn,                \
                       const int64_t *vtri, const int8_t *vsgn,                \
                       int64_t batch, int64_t plo, int64_t phi,                \
                       const T *GPd, const T *GmQ, int64_t ny,                 \
                       const T *gpp, const T *fT,                              \
                       T *Xacc, T *Yall, T *r1, T *rA, const int64_t *io,      \
                       double *stats, T *buf)                                  \
{                                                                              \
    (void)nto;                                                                 \
    const int64_t dm = nv * no;                                                \
    for (int64_t p0 = plo; p0 < phi; p0 += batch) {                            \
        int64_t nbp = (p0 + batch <= phi) ? batch : phi - p0;                  \
        TIC(0);                                                                \
        gather_half_##SFX(T3p, nvt, p0, nbp, no, nv, npo, nvp, otri, osgn,     \
                          vtri, vsgn, buf, io);                                \
        TOC(0);                                                                \
        TIC(1);                                                                \
        gemm_##SFX(0, 0, nv + no, no * nvp, nbp * nv, 1.0, GPd + p0 * nv,      \
                   npo * nv, buf, no * nvp, 1.0, Xacc, no * nvp);              \
        gemm_##SFX(0, 0, nbp * nv, ny, no * nvp, 1.0, buf, no * nvp, GmQ, ny,  \
                   0.0, Yall + p0 * nv * ny, ny);                              \
        TOC(1);                                                                \
        TIC(2);                                                                \
        /* r1 += T[P] (dm x nvp) gpp[P];  rA[P] = fT (dm) . T[P] */            \
        for (int64_t P = 0; P < nbp; P++) {                                    \
            gemv_##SFX(0, dm, nvp, 1.0, buf + P * dm * nvp, nvp,               \
                       gpp + (p0 + P) * nvp, 1.0, r1);                         \
            gemv_##SFX(1, dm, nvp, 1.0, buf + P * dm * nvp, nvp, fT, 0.0,      \
                       rA + (p0 + P) * nvp);                                   \
        }                                                                      \
        TOC(2);                                                                \
    }                                                                          \
}                                                                              \
                                                                               \
void zc_t3_pass2r_##SFX(const T *T3p, int64_t nto, int64_t nvt,                \
                       int64_t no, int64_t nv, int64_t npo, int64_t nvp,       \
                       const int64_t *otri, const int8_t *osgn,                \
                       const int64_t *ofree, const int64_t *opair,             \
                       const int64_t *vtri, const int8_t *vsgn,                \
                       const int64_t *vfree, const int64_t *vpair,             \
                       int64_t batch, int64_t plo, int64_t phi,                \
                       const T *Wring, const T *Adrv, const T *Bdrv,           \
                       T *R, const int64_t *io, double *stats, T *Tb, T *Z)    \
{                                                                              \
    const int64_t dm = nv * no;                                                \
    const int64_t sP = dm * nvp, sa = no * nvp;                                \
    for (int64_t p0 = plo; p0 < phi; p0 += batch) {                            \
        int64_t nbp = (p0 + batch <= phi) ? batch : phi - p0;                  \
        TIC(0);                                                                \
        gather_half_##SFX(T3p, nvt, p0, nbp, no, nv, npo, nvp, otri, osgn,     \
                          vtri, vsgn, Tb, io);                                 \
        TOC(0);                                                                \
        TIC(1);                                                                \
        for (int64_t P = 0; P < nbp; P++)                                      \
            gemm_##SFX(0, 0, dm, nvp, dm, 1.0, Wring, dm, Tb + P * sP, nvp,    \
                       0.0, Z + P * sP, nvp);                                  \
        TOC(1);                                                                \
        TIC(3);                                                                \
        /* both drive terms in one GEMM: K = nv + no, one pass over Z */       \
        gemm_##SFX(0, 0, nbp * nv, no * nvp, nv + no, -1.0,                    \
                   Adrv + p0 * nv * (nv + no), nv + no, Bdrv, no * nvp, 1.0,   \
                   Z, no * nvp);                                               \
        TOC(3);                                                                \
        TIC(2);                                                                \
        const int rmap = io && io[1] >= 0;                                     \
        if (rmap)                         /* the rows of R this batch adds to */\
            for (int64_t it = 0; it < nto; it++)                               \
                for (int rr = 0; rr < 3; rr++) {                               \
                    int64_t P = opair[rr * nto + it] - p0;                     \
                    if (P >= 0 && P < nbp) {                                   \
                        will_need(R + it * nvt, sizeof(T) * nvt);              \
                        break;                                                 \
                    }                                                          \
                }                                                              \
        _Pragma("omp parallel for schedule(dynamic, 4)")                       \
        for (int64_t it = 0; it < nto; it++) {                                 \
            T *r = R + it * nvt;                                               \
            for (int rr = 0; rr < 3; rr++) {                                   \
                int64_t P = opair[rr * nto + it] - p0;                         \
                if (P < 0 || P >= nbp) continue;                               \
                const T *z = Z + P * sP + ofree[rr * nto + it] * nvp;          \
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
}                                                                              \
                                                                               \
void zc_l3_pass1r_##SFX(const T *L3p, const T *T3p, int64_t nto, int64_t nvt,  \
                        int64_t no, int64_t nv, int64_t npo, int64_t nvp,      \
                        const int64_t *otri, const int8_t *osgn,               \
                        const int64_t *vtri, const int8_t *vsgn,               \
                        int64_t batch, int64_t plo, int64_t phi,               \
                        const T *GPd, const T *GmQ, int64_t ny,                \
                        const T *l2pp, T *Xacc, T *Yall, T *Dr, T *D1,         \
                        double *stats, T *bufL, T *bufT)                       \
{                                                                              \
    (void)nto;                                                                 \
    const int64_t dm = nv * no;                                                \
    for (int64_t p0 = plo; p0 < phi; p0 += batch) {                            \
        int64_t nbp = (p0 + batch <= phi) ? batch : phi - p0;                  \
        TIC(0);                                                                \
        gather_half_##SFX(L3p, nvt, p0, nbp, no, nv, npo, nvp, otri, osgn,     \
                          vtri, vsgn, bufL, NULL);                             \
        gather_half_##SFX(T3p, nvt, p0, nbp, no, nv, npo, nvp, otri, osgn,     \
                          vtri, vsgn, bufT, NULL);                             \
        TOC(0);                                                                \
        TIC(1);                                                                \
        gemm_##SFX(0, 0, nv + no, no * nvp, nbp * nv, 1.0, GPd + p0 * nv,      \
                   npo * nv, bufL, no * nvp, 1.0, Xacc, no * nvp);             \
        gemm_##SFX(0, 0, nbp * nv, ny, no * nvp, 1.0, bufL, no * nvp, GmQ, ny, \
                   0.0, Yall + p0 * nv * ny, ny);                              \
        TOC(1);                                                                \
        TIC(2);                                                                \
        for (int64_t P = 0; P < nbp; P++) {                                    \
            gemm_##SFX(0, 1, dm, dm, nvp, 1.0, bufL + P * dm * nvp, nvp,       \
                       bufT + P * dm * nvp, nvp, 1.0, Dr, dm);                 \
            gemv_##SFX(0, dm, nvp, 1.0, bufT + P * dm * nvp, nvp,              \
                       l2pp + (p0 + P) * nvp, 1.0, D1);                        \
        }                                                                      \
        TOC(2);                                                                \
    }                                                                          \
}                                                                              \
                                                                               \
void zc_l3_pass2r_##SFX(const T *L3p, int64_t nto, int64_t nvt,                \
                        int64_t no, int64_t nv, int64_t npo, int64_t nvp,      \
                        const int64_t *otri, const int8_t *osgn,               \
                        const int64_t *ofree, const int64_t *opair,            \
                        const int64_t *vtri, const int8_t *vsgn,               \
                        const int64_t *vfree, const int64_t *vpair,            \
                        int64_t batch, int64_t plo, int64_t phi, const T *WL, const T *U, int64_t nu,\
                        const T *Vp, const T *Adrv, int64_t K, const T *Bdrv,  \
                        T *R, double *stats, T *Tb, T *Z)                      \
{                                                                              \
    const int64_t dm = nv * no;                                                \
    const int64_t sP = dm * nvp, sa = no * nvp;                                \
    for (int64_t p0 = plo; p0 < phi; p0 += batch) {                            \
        int64_t nbp = (p0 + batch <= phi) ? batch : phi - p0;                  \
        TIC(0);                                                                \
        gather_half_##SFX(L3p, nvt, p0, nbp, no, nv, npo, nvp, otri, osgn,     \
                          vtri, vsgn, Tb, NULL);                               \
        TOC(0);                                                                \
        TIC(1);                                                                \
        for (int64_t P = 0; P < nbp; P++) {                                    \
            gemm_##SFX(0, 0, dm, nvp, dm, 1.0, WL, dm, Tb + P * sP, nvp,       \
                       0.0, Z + P * sP, nvp);                                  \
            if (nu > 0)                                                        \
                gemm_##SFX(0, 0, dm, nvp, nu, 1.0, U, nu,                      \
                           Vp + (p0 + P) * nu * nvp, nvp, 1.0, Z + P * sP, nvp);\
        }                                                                      \
        TOC(1);                                                                \
        TIC(3);                                                                \
        gemm_##SFX(0, 0, nbp * nv, no * nvp, K, -1.0, Adrv + p0 * nv * K, K,   \
                   Bdrv, no * nvp, 1.0, Z, no * nvp);                          \
        TOC(3);                                                                \
        TIC(2);                                                                \
        _Pragma("omp parallel for schedule(dynamic, 4)")                       \
        for (int64_t it = 0; it < nto; it++) {                                 \
            T *r = R + it * nvt;                                               \
            for (int rr = 0; rr < 3; rr++) {                                   \
                int64_t P = opair[rr * nto + it] - p0;                         \
                if (P < 0 || P >= nbp) continue;                               \
                const T *z = Z + P * sP + ofree[rr * nto + it] * nvp;          \
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
}

DEFINE_T3(double, d)
DEFINE_T3(double complex, z)

/* ---- CC3 with stored T3 (zccsdt/cc/ccsdt_opt.py, model 'cc3'; docs/ccn.md):
 * the triples equation has the drive terms and the Fock operator only.
 *
 * zc_t3_drive: pass 2 without the ring term - T3 is not read:
 *   Z[P, c, i, Q] = - sum_{e|m} Adrv[P, c, (e|m)] Bdrv[(e|m), (i, Q)]
 *   R[I, A] += the antisymmetrizing reduction of Z, as in zc_t3_pass2w.
 * zc_t3_fdiag: R[I, A] += (ev[A] - eo[I]) T3p[I, A], the diagonal of the
 * Fock operator on packed triples (eo, ev: sums of its diagonal elements
 * over the occupied / virtual triples). */
#define DEFINE_CCN(T, SFX)                                                     \
void zc_t3_drive_##SFX(int64_t nto, int64_t nvt,                               \
                       int64_t no, int64_t nv, int64_t npo, int64_t nvp,       \
                       const int64_t *ofree, const int64_t *opair,             \
                       const int64_t *vfree, const int64_t *vpair,             \
                       int64_t batch, const T *Adrv, const T *Bdrv,            \
                       T *R, const int64_t *io, double *stats)                 \
{                                                                              \
    const int64_t dm = nv * no;                                                \
    const int64_t sP = dm * nvp, sa = no * nvp;                                \
    T *Z = malloc(sizeof(T) * batch * nv * no * nvp);                          \
    for (int64_t p0 = 0; p0 < npo; p0 += batch) {                              \
        int64_t nbp = (p0 + batch <= npo) ? batch : npo - p0;                  \
        TIC(3);                                                                \
        gemm_##SFX(0, 0, nbp * nv, no * nvp, nv + no, -1.0,                    \
                   Adrv + p0 * nv * (nv + no), nv + no, Bdrv, no * nvp, 0.0,   \
                   Z, no * nvp);                                               \
        TOC(3);                                                                \
        TIC(2);                                                                \
        const int rmap = io && io[1] >= 0;                                     \
        if (rmap)                                                              \
            for (int64_t it = 0; it < nto; it++)                               \
                for (int rr = 0; rr < 3; rr++) {                               \
                    int64_t P = opair[rr * nto + it] - p0;                     \
                    if (P >= 0 && P < nbp) {                                   \
                        will_need(R + it * nvt, sizeof(T) * nvt);              \
                        break;                                                 \
                    }                                                          \
                }                                                              \
        _Pragma("omp parallel for schedule(dynamic, 4)")                       \
        for (int64_t it = 0; it < nto; it++) {                                 \
            T *r = R + it * nvt;                                               \
            for (int rr = 0; rr < 3; rr++) {                                   \
                int64_t P = opair[rr * nto + it] - p0;                         \
                if (P < 0 || P >= nbp) continue;                               \
                const T *z = Z + P * sP + ofree[rr * nto + it] * nvp;          \
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
    free(Z);                                                                   \
}                                                                              \
                                                                               \
void zc_t3_fdiag_##SFX(const T *T3p, T *R, int64_t nto, int64_t nvt,           \
                       const double *eo, const double *ev)                     \
{                                                                              \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t it = 0; it < nto; it++) {                                     \
        const T *t = T3p + it * nvt;                                           \
        T *r = R + it * nvt;                                                   \
        const double e = eo[it];                                               \
        for (int64_t A = 0; A < nvt; A++)                                      \
            r[A] += (ev[A] - e) * t[A];                                        \
    }                                                                          \
}

DEFINE_CCN(double, d)
DEFINE_CCN(double complex, z)
