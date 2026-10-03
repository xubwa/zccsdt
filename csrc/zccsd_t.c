/*
 * zccsd_t.c -- the (T) energy of the spinor CCSD from batches of its
 * intermediates (zccsdt/zccsd/triples.py).
 *
 * For virtual a > b > c and occupied i > j > k, with pairs in
 * numpy.tril_indices(n, -1) order,
 *
 *   x_{a,(bc)}[i, (jk)] = sum_e t2[j,k,a,e] <bc||ei> - sum_m t2[i,m,b,c] <ma||jk>
 *   Z[i, (jk)]  = x_{a,(bc)} - x_{b,(ac)} + x_{c,(ab)}
 *   w[ijk]      = Z[i, (jk)] - Z[j, (ik)] + Z[k, (ij)]
 *
 * and the same two antisymmetrizations of the disconnected part
 *
 *   y_{a,(bc)}[i, (jk)] = t1[i,a] <bc||jk> + f[a,i] t2[j,k,b,c]
 *
 * give v = w + (antisymmetrized y), and
 *
 *   E(T) = sum_{a>b>c} sum_{i>j>k} conj(v) w / (e_i + e_j + e_k - e_a - e_b - e_c).
 *
 * zc_t_batch adds up the triples (a, b, c) of one a and a range b0 <= b < b1,
 * c < b, from the GEMM results of the batch:
 *
 *   X1[p - p0, i, (jk)] = x_{a,(bc)},  p = b (b - 1) / 2 + c the pair (b, c)
 *   G1[x, i, y - b0, (jk)] = x_{y,(a x)} for x < b1 and b0 <= y < b1
 *   G2[x - b0, i, y, (jk)] = x_{y,(a x)} for b0 <= x < b1 and y < b0
 *
 * (x_{b,(ac)} = G1[c, b]; x_{c,(ab)} = G1[b, c] or G2[b, c]).  One (b, c) per
 * task: its Z is assembled in the cache of the thread.
 */

#include <complex.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

typedef double complex zc;

/* VpT[(b>c), (j>k)] = <bc||jk>, T2pT[(b>c), (j>k)] = t2[j,k,b,c], t1[i,a],
 * fvo[a,i]; out[0], out[1]: real and imaginary part of the sum. */
void zc_t_batch(const zc *X1, const zc *G1, const zc *G2, int64_t a, int64_t b0, int64_t b1,
                int64_t no, int64_t nv, const zc *t1, const zc *fvo, const zc *VpT,
                const zc *T2pT, const double *eo, const double *ev, double *out)
{
    const int64_t npo = no * (no - 1) / 2, nb = b1 - b0;
    const int64_t p0 = b0 * (b0 - 1) / 2, pa = a * (a - 1) / 2;
    const int64_t n1 = nb * npo, n2 = b0 * npo;     /* columns of G1 and G2 */
    double er = 0.0, ei = 0.0;
    #pragma omp parallel reduction(+:er, ei)
    {
        zc *Z = malloc(sizeof(zc) * no * npo);
        zc *V = malloc(sizeof(zc) * no * npo);
        #pragma omp for schedule(dynamic, 1) collapse(1)
        for (int64_t b = b0 > 1 ? b0 : 1; b < b1; b++)
            for (int64_t c = 0; c < b; c++) {
                const int64_t pbc = b * (b - 1) / 2 + c;
                if (c + 1 < b)                      /* the strided rows of the next c */
                    for (int64_t i = 0; i < no; i++) {
                        const zc *q2 = G1 + ((c + 1) * no + i) * n1 + (b - b0) * npo;
                        const zc *q3 = c + 1 >= b0 ? G1 + (b * no + i) * n1 + (c + 1 - b0) * npo
                                                   : G2 + ((b - b0) * no + i) * n2 + (c + 1) * npo;
                        for (int64_t q = 0; q < npo && q < 16; q += 4) {
                            __builtin_prefetch(q2 + q);
                            __builtin_prefetch(q3 + q);
                        }
                    }
                const zc *x1 = X1 + (pbc - p0) * no * npo;
                const zc *vbc = VpT + pbc * npo, *tbc = T2pT + pbc * npo;
                const zc *vac = VpT + (pa + c) * npo, *tac = T2pT + (pa + c) * npo;
                const zc *vab = VpT + (pa + b) * npo, *tab = T2pT + (pa + b) * npo;
                for (int64_t i = 0; i < no; i++) {
                    const zc *r1 = x1 + i * npo;
                    const zc *r2 = G1 + (c * no + i) * n1 + (b - b0) * npo;
                    const zc *r3 = c >= b0 ? G1 + (b * no + i) * n1 + (c - b0) * npo
                                           : G2 + ((b - b0) * no + i) * n2 + c * npo;
                    const zc ta = t1[i * nv + a], tb = t1[i * nv + b], tc = t1[i * nv + c];
                    const zc fa = fvo[a * no + i], fb = fvo[b * no + i], fc = fvo[c * no + i];
                    zc *z = Z + i * npo, *v = V + i * npo;
                    for (int64_t q = 0; q < npo; q++) {
                        z[q] = r1[q] - r2[q] + r3[q];
                        v[q] = z[q] + ta * vbc[q] + fa * tbc[q] - tb * vac[q] - fb * tac[q]
                               + tc * vab[q] + fc * tab[q];
                    }
                }
                const double eabc = ev[a] + ev[b] + ev[c];
                for (int64_t i = 2; i < no; i++)
                    for (int64_t j = 1; j < i; j++) {
                        const int64_t pij = i * (i - 1) / 2 + j;
                        const zc *zi = Z + i * npo + j * (j - 1) / 2;
                        const zc *zj = Z + j * npo + i * (i - 1) / 2;
                        const zc *vi = V + i * npo + j * (j - 1) / 2;
                        const zc *vj = V + j * npo + i * (i - 1) / 2;
                        const double dij = eo[i] + eo[j] - eabc;
                        for (int64_t k = 0; k < j; k++) {
                            const zc w = zi[k] - zj[k] + Z[k * npo + pij];
                            const zc u = vi[k] - vj[k] + V[k * npo + pij];
                            const zc p = conj(u) * w / (dij + eo[k]);
                            er += creal(p);
                            ei += cimag(p);
                        }
                    }
            }
        free(Z);
        free(V);
    }
    out[0] = er;
    out[1] = ei;
}

/* Op[(b>c), i, e] = <bc||ei> = -conj(<ie||bc>) for one occupied i from the
 * slab ovvv[e, (b>c)] = <ie||bc> of the integrals as the CCSD keeps them:
 * blocks of pairs, so that the rows of Op that are being written stay in
 * the cache while the slab is read by rows. */
void zc_t_ovvv(const zc *slab, int64_t nv, int64_t npv, int64_t no, int64_t i, zc *Op)
{
    enum { BLK = 256 };
    #pragma omp parallel for schedule(static)
    for (int64_t p0 = 0; p0 < npv; p0 += BLK) {
        const int64_t p1 = p0 + BLK < npv ? p0 + BLK : npv;
        for (int64_t e = 0; e < nv; e++) {
            const zc *src = slab + e * npv;
            for (int64_t p = p0; p < p1; p++)
                Op[(p * no + i) * nv + e] = -conj(src[p]);
        }
    }
}
