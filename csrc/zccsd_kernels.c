/*
 * zccsd_kernels.c -- kernels of the spinor CCSD amplitude equations
 * (zccsdt/zccsd/amps.py) that numpy does badly.
 *
 * zc_ovvv_t1: the <mb||ef> t1 terms of one occupied m from the integrals
 * packed over e > f,
 *
 *     A[b, e, j] = sum_f <mb||ef> t1[j, f],
 *
 * Om[b, (e>f)] = <mb||ef> in numpy.tril_indices(nv, -1) order.  One row b
 * per thread: the row is unpacked to its antisymmetric (e, f) matrix (real
 * and imaginary parts apart, in the cache of the thread) and multiplied
 * with t1 by dot products; no array of nv^3 elements is made.
 */

#include <complex.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

void zc_ovvv_t1(const double complex *Om, const double complex *t1,
                int64_t nv, int64_t no, double complex *A)
{
    const int64_t npv = nv * (nv - 1) / 2;
    double *tr = malloc(sizeof(double) * no * nv);
    double *ti = malloc(sizeof(double) * no * nv);
    for (int64_t x = 0; x < no * nv; x++) {
        tr[x] = creal(t1[x]);
        ti[x] = cimag(t1[x]);
    }
    #pragma omp parallel
    {
        double *ur = calloc(nv * nv, sizeof(double));
        double *ui = calloc(nv * nv, sizeof(double));
        #pragma omp for schedule(static)
        for (int64_t b = 0; b < nv; b++) {
            const double complex *row = Om + b * npv;
            int64_t p = 0;
            for (int64_t e = 1; e < nv; e++)
                for (int64_t f = 0; f < e; f++, p++) {
                    const double re = creal(row[p]), im = cimag(row[p]);
                    ur[e * nv + f] = re;
                    ui[e * nv + f] = im;
                    ur[f * nv + e] = -re;
                    ui[f * nv + e] = -im;
                }
            for (int64_t e = 0; e < nv; e++) {
                const double *xr = ur + e * nv, *xi = ui + e * nv;
                double complex *out = A + (b * nv + e) * no;
                for (int64_t j = 0; j < no; j++) {
                    const double *yr = tr + j * nv, *yi = ti + j * nv;
                    double sr = 0.0, si = 0.0;
                    #pragma omp simd reduction(+:sr, si)
                    for (int64_t f = 0; f < nv; f++) {
                        sr += xr[f] * yr[f] - xi[f] * yi[f];
                        si += xr[f] * yi[f] + xi[f] * yr[f];
                    }
                    out[j] = sr + si * I;
                }
            }
        }
        free(ur);
        free(ui);
    }
    free(tr);
    free(ti);
}

/*
 * Shuffles between the layouts of the amplitude equations: plain loops over
 * arrays of o^2 v^2 elements that numpy does with fancy indexing on one
 * thread.  Pairs are i > j and a > b in numpy.tril_indices(n, -1) order,
 * (i, j) -> i (i - 1) / 2 + j.
 */

/* R[(i>j), (a>b)] += alpha (x[i,a,j,b] - x[j,a,i,b] - x[i,b,j,a] + x[j,b,i,a])
 * for an array x with the strides si, sa, sj, sb (in elements) of its
 * occupied and virtual indices: P(ij) P(ab) of a ring-type term. */
void zc_asym_pack4(const double complex *x, int64_t si, int64_t sa, int64_t sj,
                   int64_t sb, int64_t no, int64_t nv, double alpha,
                   double complex *R)
{
    const int64_t npv = nv * (nv - 1) / 2;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 1; i < no; i++)
        for (int64_t j = 0; j < i; j++) {
            double complex *r = R + (i * (i - 1) / 2 + j) * npv;
            const double complex *xij = x + i * si + j * sj;
            const double complex *xji = x + j * si + i * sj;
            for (int64_t a = 1; a < nv; a++) {
                const double complex *p = xij + a * sa, *q = xji + a * sa;
                const double complex *s = xij + a * sb, *t = xji + a * sb;
                for (int64_t b = 0; b < a; b++)
                    *r++ += alpha * (p[b * sb] - q[b * sb] - s[b * sa] + t[b * sa]);
            }
        }
}

/* R[p, (a>b)] += alpha (x[p,a,b] - x[p,b,a]) for np rows, with the strides
 * sp, sa, sb of x: P(ab) of a term whose occupied pair is packed already. */
void zc_asym_pack2(const double complex *x, int64_t sp, int64_t sa, int64_t sb,
                   int64_t np, int64_t nv, double alpha, double complex *R)
{
    const int64_t npv = nv * (nv - 1) / 2;
    #pragma omp parallel for schedule(static)
    for (int64_t p = 0; p < np; p++) {
        double complex *r = R + p * npv;
        const double complex *xp = x + p * sp;
        for (int64_t a = 1; a < nv; a++)
            for (int64_t b = 0; b < a; b++)
                *r++ += alpha * (xp[a * sa + b * sb] - xp[b * sa + a * sb]);
    }
}

/* out[p, (a>b)] = x[p, a, b] for a contiguous x of np matrices. */
void zc_pack_tril(const double complex *x, int64_t np, int64_t nv, double complex *out)
{
    const int64_t npv = nv * (nv - 1) / 2;
    #pragma omp parallel for schedule(static)
    for (int64_t p = 0; p < np; p++) {
        double complex *r = out + p * npv;
        const double complex *xp = x + p * nv * nv;
        for (int64_t a = 1; a < nv; a++)
            for (int64_t b = 0; b < a; b++)
                *r++ = xp[a * nv + b];
    }
}

/* x[p, a, b] = -x[p, b, a] = in[p, (a>b)], zero diagonal: the inverse. */
void zc_unpack_tril(const double complex *in, int64_t np, int64_t nv, double complex *x)
{
    const int64_t npv = nv * (nv - 1) / 2;
    #pragma omp parallel for schedule(static)
    for (int64_t p = 0; p < np; p++) {
        const double complex *r = in + p * npv;
        double complex *xp = x + p * nv * nv;
        for (int64_t a = 0; a < nv; a++) {
            xp[a * nv + a] = 0.0;
            for (int64_t b = 0; b < a; b++) {
                xp[a * nv + b] = *r;
                xp[b * nv + a] = -*r++;
            }
        }
    }
}

/* The amplitudes in the layouts of the terms, from t2[i, j, a, b]:
 *   tr[i, a, m, e] = t2[i, m, a, e]   (ring layout)
 *   tp[i, j, (a>b)] = t2[i, j, a, b]  (virtual pairs packed) */
void zc_t2_layouts(const double complex *t2, int64_t no, int64_t nv,
                   double complex *tr, double complex *tp)
{
    const int64_t npv = nv * (nv - 1) / 2;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t i = 0; i < no; i++)
        for (int64_t j = 0; j < no; j++) {
            const double complex *x = t2 + (i * no + j) * nv * nv;
            double complex *p = tp + (i * no + j) * npv;
            for (int64_t a = 0; a < nv; a++) {
                double complex *y = tr + ((i * nv + a) * no + j) * nv;
                for (int64_t b = 0; b < nv; b++)
                    y[b] = x[a * nv + b];
                for (int64_t b = 0; b < a; b++)
                    *p++ = x[a * nv + b];
            }
        }
}

/* t2[i, j, a, b], antisymmetric in both pairs, from R[(i>j), (a>b)] divided
 * by eo[i] + eo[j] - ev[a] - ev[b]. */
void zc_unpack_t2(const double complex *R, const double *eo, const double *ev,
                  int64_t no, int64_t nv, double complex *t2)
{
    const int64_t npv = nv * (nv - 1) / 2, nvv = nv * nv;
    #pragma omp parallel for schedule(dynamic, 1)
    for (int64_t i = 0; i < no; i++) {
        double complex *tii = t2 + (i * no + i) * nvv;
        for (int64_t x = 0; x < nvv; x++)
            tii[x] = 0.0;
        for (int64_t j = 0; j < i; j++) {
            const double complex *r = R + (i * (i - 1) / 2 + j) * npv;
            double complex *tij = t2 + (i * no + j) * nvv;
            double complex *tji = t2 + (j * no + i) * nvv;
            const double eij = eo[i] + eo[j];
            for (int64_t a = 0; a < nv; a++) {
                tij[a * nv + a] = 0.0;
                tji[a * nv + a] = 0.0;
                for (int64_t b = 0; b < a; b++) {
                    const double complex val = *r++ / (eij - ev[a] - ev[b]);
                    tij[a * nv + b] = val;
                    tij[b * nv + a] = -val;
                    tji[a * nv + b] = -val;
                    tji[b * nv + a] = val;
                }
            }
        }
    }
}
