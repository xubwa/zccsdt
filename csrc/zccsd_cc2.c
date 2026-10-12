/* Kernels of the Cholesky-based spinor CC2 (zccsdt/zccsd/cc2.py; docs/ccn.md):
 * the doubles amplitudes of one occupied index from the products of the
 * dressed three-index integrals, generated, used and discarded per batch.
 *
 * All arrays are complex128 and row-major.
 */
#include <complex.h>
#include <stdint.h>
#include <omp.h>

typedef double complex zc;

/* M[a, j, b] = sum_c B[c, i, a] B[c, j, b] = (ai|bj)~ for one occupied i and
 * the occupied j = 0 .. nj - 1 of a batch (eoj: their energies), in place:
 *
 *   M[a, j, b] <- (M[a, j, b] - M[b, j, a]) / (ei + eoj[j] - ev[a] - ev[b] + omega)
 *
 * - the amplitudes t2[i j, a b] of CC2 (omega = 0), or the doubles part
 * r2[i j, a b] of an excitation vector at the frequency omega.  M: (nv, nj,
 * nv).  Every j is handled by one thread: no two threads write an element. */
void zc_cc2_amps(zc *M, int64_t nv, int64_t nj, double ei, const double *eoj,
                 const double *ev, double omega)
{
    const int64_t sa = nj * nv;
    #pragma omp parallel for schedule(static)
    for (int64_t j = 0; j < nj; j++) {
        const double eij = ei + eoj[j] + omega;
        for (int64_t a = 0; a < nv; a++) {
            zc *ma = M + a * sa + j * nv;               /* M[a, j, :] */
            const double ea = eij - ev[a];
            ma[a] = 0.0;
            for (int64_t b = 0; b < a; b++) {
                zc *mb = M + b * sa + j * nv + a;       /* M[b, j, a] */
                const zc x = (ma[b] - *mb) / (ea - ev[b]);
                ma[b] = x;
                *mb = -x;
            }
        }
    }
}

/* The same for two batches at once that must stay consistent - none: kept
 * minimal.  The contraction of the amplitudes with a vector over (j, b),
 *   out[a] += sum_{j b} t[a, j, b] f[j, b],   outj[j, b] += sum_a t[a, j, b] g[a],
 * in one pass over t (the term f~[m,e] t2[i m, a e] of the singles
 * equation, both for the index i of the batch and for the j of it). */
void zc_cc2_vec(const zc *t, int64_t nv, int64_t nj, const zc *f, const zc *g,
                zc *out, zc *outj)
{
    const int64_t n = nj * nv;
    #pragma omp parallel
    {
        #pragma omp for schedule(static)
        for (int64_t a = 0; a < nv; a++) {
            const zc *ta = t + a * n;
            zc s = 0.0;
            for (int64_t q = 0; q < n; q++)
                s += ta[q] * f[q];
            out[a] += s;
        }
        #pragma omp for schedule(static)
        for (int64_t q = 0; q < n; q++) {
            zc s = 0.0;
            for (int64_t a = 0; a < nv; a++)
                s += t[a * n + q] * g[a];
            outj[q] += s;
        }
    }
}
