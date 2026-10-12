/* CC3 with stored T3 in the native program (cfour/xzccsdt; docs/native.md,
 * section 17): the drive terms of the triples equation for a range of
 * occupied pairs, with the buffer of the caller - zc_t3_drive of
 * csrc/ccsdt_t3.c (which the Python program calls, for all pairs, with a
 * buffer of its own) in the form of zc_t3_pass2r, so that the batches of
 * pairs can be dealt to MPI ranks.  T3 is not read:
 *
 *   Z[P, c, (i, Q)] = - sum_{e|m} Adrv[P, c, (e|m)] Bdrv[(e|m), (i, Q)]
 *   R[I, A] += the antisymmetrizing reduction of Z (as zc_t3_pass2w)
 *
 * for the pairs plo <= P < phi in batches of `batch'; z: batch nv no nvp
 * elements.  stats[3] += seconds of the products, stats[2] of the reduction.
 * Only added to csrc/: the Python program does not call it. */
#include <complex.h>
#include <stdint.h>
#include <omp.h>

void zc_gemm_d(int ta, int tb, int64_t m, int64_t n, int64_t k,
               const double *alpha, const double *a, int64_t lda,
               const double *b, int64_t ldb, const double *beta, double *c,
               int64_t ldc);
void zc_gemm_z(int ta, int tb, int64_t m, int64_t n, int64_t k,
               const double complex *alpha, const double complex *a, int64_t lda,
               const double complex *b, int64_t ldb, const double complex *beta,
               double complex *c, int64_t ldc);

static const double CC3N_SGN[3] = {1.0, -1.0, 1.0};

#define DEFINE_CC3N(T, SFX)                                                    \
void zc_t3_driver_##SFX(int64_t nto, int64_t nvt, int64_t no, int64_t nv,      \
                        int64_t npo, int64_t nvp,                              \
                        const int64_t *ofree, const int64_t *opair,            \
                        const int64_t *vfree, const int64_t *vpair,            \
                        int64_t batch, int64_t plo, int64_t phi,               \
                        const T *Adrv, const T *Bdrv, T *R, double *stats,     \
                        T *Z)                                                  \
{                                                                              \
    const int64_t sP = nv * no * nvp, sa = no * nvp;                           \
    const T mone = -1.0, zero = 0.0;                                           \
    (void)npo;                                                                 \
    for (int64_t p0 = plo; p0 < phi; p0 += batch) {                            \
        const int64_t nbp = (p0 + batch <= phi) ? batch : phi - p0;            \
        double t0 = omp_get_wtime();                                           \
        zc_gemm_##SFX(0, 0, nbp * nv, no * nvp, nv + no, &mone,                \
                      Adrv + p0 * nv * (nv + no), nv + no, Bdrv, no * nvp,     \
                      &zero, Z, no * nvp);                                     \
        double t1 = omp_get_wtime();                                           \
        stats[3] += t1 - t0;                                                   \
        _Pragma("omp parallel for schedule(dynamic, 4)")                       \
        for (int64_t it = 0; it < nto; it++) {                                 \
            T *r = R + it * nvt;                                               \
            for (int rr = 0; rr < 3; rr++) {                                   \
                const int64_t P = opair[rr * nto + it] - p0;                   \
                if (P < 0 || P >= nbp) continue;                               \
                const T *z = Z + P * sP + ofree[rr * nto + it] * nvp;          \
                const double sg = CC3N_SGN[rr];                                \
                for (int64_t A = 0; A < nvt; A++) {                            \
                    T v = z[vfree[A] * sa + vpair[A]]                          \
                        - z[vfree[nvt + A] * sa + vpair[nvt + A]]              \
                        + z[vfree[2 * nvt + A] * sa + vpair[2 * nvt + A]];     \
                    r[A] += sg * v;                                            \
                }                                                              \
            }                                                                  \
        }                                                                      \
        stats[2] += omp_get_wtime() - t1;                                      \
    }                                                                          \
}

DEFINE_CC3N(double, d)
DEFINE_CC3N(double complex, z)
