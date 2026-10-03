/*
 * zccsd_t.c -- the (T) energy of the spinor CCSD from batches of its
 * intermediates (zccsdt/zccsd/triples.py).
 *
 * For virtual a > b > c and occupied i > j > k, with pairs in
 * numpy.tril_indices(n, -1) order,
 *
 *   x_{a,(bc)}[i, (jk)] = sum_e <bc||ei> t2[j,k,a,e] - sum_m t2[i,m,b,c] <ma||jk>
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
 * c < b, from the GEMM results of a batch, matrices whose rows are (pair or
 * virtual, i) and whose columns are (virtual, (jk)):
 *
 *   X1[(p - p0, i), c1 + (jk)] = x_{a,(bc)},  p = b (b - 1) / 2 + c the pair
 *       (b, c); s1 elements per row (several a side by side: c1 picks one)
 *   G1[(x, i), (y - b0, (jk))] = x_{y,(a x)} for x < b1 and y >= b0; n1
 *       elements per row
 *   G2[(x - b0, i), (y, (jk))] = x_{y,(a x)} for b0 <= x < b1 and y < b0; n2
 *       elements per row
 *
 * (x_{b,(ac)} = G1[c, b]; x_{c,(ab)} = G1[b, c] or G2[b, c]).  The pairs
 * (b, c) are the tasks of the threads: Z of one pair is assembled in the
 * cache of its thread.
 *
 * With m3 the three arrays come as the three real products of the "3M"
 * scheme, p1 = Ar Br, p2 = Ai Bi, p3 = (Ar + Ai) (Br + Bi), summed over the
 * particle and the hole term: x = (p1 - p2) + i (p3 - p1 - p2), taken after
 * the three real Z have been put together.
 */

#include <complex.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <omp.h>

typedef double complex zc;

/* VpT[(b>c), (j>k)] = <bc||jk>, T2pT[(b>c), (j>k)] = t2[j,k,b,c], t1[i,a],
 * fvo[a,i]; out[0], out[1]: real and imaginary part of the sum.  X1, G1, G2:
 * one complex array each (m3 = 0) or three real ones. */
void zc_t_batch(const void **X1p, int64_t s1, int64_t c1, const void **G1p, int64_t n1,
                const void **G2p, int64_t n2, int m3, int64_t a, int64_t b0, int64_t b1,
                int64_t no, int64_t nv, const zc *t1, const zc *fvo, const zc *VpT,
                const zc *T2pT, const double *eo, const double *ev, double *out)
{
    const zc *X1 = X1p[0], *G1 = G1p[0], *G2 = G2p[0];
    const int64_t npo = no * (no - 1) / 2;
    const int64_t p0 = b0 * (b0 - 1) / 2, pa = a * (a - 1) / 2;
    /* the pairs (b, c), c < b, of the range are the tasks, so that the
     * threads are busy also when the range holds a few b only; several
     * pairs per task where the arrays of a pair are small (few occupied
     * spinors) */
    const int64_t bmin = b0 > 1 ? b0 : 1;
    const int64_t pbeg = bmin * (bmin - 1) / 2, pend = b1 * (b1 - 1) / 2;
    const int64_t chunk = 1 + 8192 / (no * npo);
    double er = 0.0, ei = 0.0;
    int has_f = 0;                                  /* f[a,i] = 0 for canonical orbitals */
    for (int64_t x = 0; x < no * nv && !has_f; x++)
        has_f = creal(fvo[x]) != 0.0 || cimag(fvo[x]) != 0.0;
    #pragma omp parallel reduction(+:er, ei)
    {
        zc *Z = malloc(sizeof(zc) * no * npo);      /* Z[i, (jk)] */
        double *zp = malloc(sizeof(double) * 3 * npo);
        zc *tx = malloc(sizeof(zc) * 6 * no);       /* t1[k,x] and f[x,k] for x = a, b, c */
        zc *ta = tx, *tb = tx + no, *tc = tx + 2 * no;
        zc *fa = tx + 3 * no, *fb = tx + 4 * no, *fc = tx + 5 * no;
        #pragma omp for schedule(dynamic, chunk)
        for (int64_t pbc = pbeg; pbc < pend; pbc++) {
            int64_t b = (int64_t)((1.0 + sqrt(1.0 + 8.0 * (double)pbc)) / 2.0);
            while (b * (b - 1) / 2 > pbc)
                b--;
            while ((b + 1) * b / 2 <= pbc)
                b++;
            const int64_t c = pbc - b * (b - 1) / 2;
            /* Z[i, (jk)] = x_{a,(bc)} - x_{b,(ac)} + x_{c,(ab)} */
            for (int64_t i = 0; i < no; i++) {
                const int64_t o1 = ((pbc - p0) * no + i) * s1 + c1;
                const int64_t o2 = (c * no + i) * n1 + (b - b0) * npo;
                const int64_t o3 = c >= b0 ? (b * no + i) * n1 + (c - b0) * npo
                                           : ((b - b0) * no + i) * n2 + c * npo;
                zc *z = Z + i * npo;
                if (m3) {
                    for (int k = 0; k < 3; k++) {
                        const double *r1 = (const double *)X1p[k] + o1;
                        const double *r2 = (const double *)G1p[k] + o2;
                        const double *r3 = (const double *)(c >= b0 ? G1p[k] : G2p[k]) + o3;
                        double *d = zp + k * npo;
                        for (int64_t q = 0; q < npo; q++)
                            d[q] = r1[q] - r2[q] + r3[q];
                    }
                    const double *d0 = zp, *d1 = zp + npo, *d2 = zp + 2 * npo;
                    for (int64_t q = 0; q < npo; q++)
                        z[q] = (d0[q] - d1[q]) + (d2[q] - d0[q] - d1[q]) * I;
                } else {
                    const zc *r1 = X1 + o1, *r2 = G1 + o2, *r3 = (c >= b0 ? G1 : G2) + o3;
                    for (int64_t q = 0; q < npo; q++)
                        z[q] = r1[q] - r2[q] + r3[q];
                }
            }
            /* the disconnected part is put together with w, from the columns
             * a, b, c of t1 (and of f) and the rows (bc), (ac), (ab) of
             * <bc||jk> (and of t2) */
            const zc *vbc = VpT + pbc * npo, *tbc = T2pT + pbc * npo;
            const zc *vac = VpT + (pa + c) * npo, *tac = T2pT + (pa + c) * npo;
            const zc *vab = VpT + (pa + b) * npo, *tab = T2pT + (pa + b) * npo;
            for (int64_t k = 0; k < no; k++) {
                ta[k] = t1[k * nv + a];
                tb[k] = t1[k * nv + b];
                tc[k] = t1[k * nv + c];
                fa[k] = fvo[a * no + k];
                fb[k] = fvo[b * no + k];
                fc[k] = fvo[c * no + k];
            }
            /* w[ijk] = Z[i,(jk)] - Z[j,(ik)] + Z[k,(ij)]; the same combination
             * of y[i,(jk)] = t1[i,a] <bc||jk> - t1[i,b] <ac||jk> + t1[i,c] <ab||jk>
             * (and of the f t2 terms) is added to it in u */
            const double eabc = ev[a] + ev[b] + ev[c];
            for (int64_t i = 2; i < no; i++)
                for (int64_t j = 1; j < i; j++) {
                    const int64_t pij = i * (i - 1) / 2 + j;
                    const int64_t oj = j * (j - 1) / 2, oi = i * (i - 1) / 2;
                    const zc *zi = Z + i * npo + oj, *zj = Z + j * npo + oi;
                    const zc *zk = Z + pij;                         /* Z[k, (ij)]: stride npo */
                    const zc tai = ta[i], tbi = tb[i], tci = tc[i];
                    const zc taj = ta[j], tbj = tb[j], tcj = tc[j];
                    const zc vbcij = vbc[pij], vacij = vac[pij], vabij = vab[pij];
                    const double dij = eo[i] + eo[j] - eabc;
                    if (!has_f) {
                        #pragma omp simd reduction(+:er, ei)
                        for (int64_t k = 0; k < j; k++) {
                            const zc w = zi[k] - zj[k] + zk[k * npo];
                            const zc u = w + tai * vbc[oj + k] - tbi * vac[oj + k]
                                         + tci * vab[oj + k]
                                         - taj * vbc[oi + k] + tbj * vac[oi + k]
                                         - tcj * vab[oi + k]
                                         + ta[k] * vbcij - tb[k] * vacij + tc[k] * vabij;
                            const double rd = 1.0 / (dij + eo[k]);
                            er += (creal(u) * creal(w) + cimag(u) * cimag(w)) * rd;
                            ei += (creal(u) * cimag(w) - cimag(u) * creal(w)) * rd;
                        }
                    } else {
                        const zc fai = fa[i], fbi = fb[i], fci = fc[i];
                        const zc faj = fa[j], fbj = fb[j], fcj = fc[j];
                        const zc tbcij = tbc[pij], tacij = tac[pij], tabij = tab[pij];
                        #pragma omp simd reduction(+:er, ei)
                        for (int64_t k = 0; k < j; k++) {
                            const zc w = zi[k] - zj[k] + zk[k * npo];
                            const zc u = w + tai * vbc[oj + k] - tbi * vac[oj + k]
                                         + tci * vab[oj + k]
                                         - taj * vbc[oi + k] + tbj * vac[oi + k]
                                         - tcj * vab[oi + k]
                                         + ta[k] * vbcij - tb[k] * vacij + tc[k] * vabij
                                         + fai * tbc[oj + k] - fbi * tac[oj + k]
                                         + fci * tab[oj + k]
                                         - faj * tbc[oi + k] + fbj * tac[oi + k]
                                         - fcj * tab[oi + k]
                                         + fa[k] * tbcij - fb[k] * tacij + fc[k] * tabij;
                            const double rd = 1.0 / (dij + eo[k]);
                            er += (creal(u) * creal(w) + cimag(u) * cimag(w)) * rd;
                            ei += (creal(u) * cimag(w) - cimag(u) * creal(w)) * rd;
                        }
                    }
                }
        }
        free(Z);
        free(zp);
        free(tx);
    }
    out[0] = er;
    out[1] = ei;
}

/* ------------------------------------------------------------------ operands
 *
 * The two arrays the GEMMs of (T) read, written block by block:
 *
 *   A[(b>c), i, (e | m)] = (<bc||ei> | t2[i,m,b,c])
 *   B[a, (j>k), (e | m)] = (t2[j,k,a,e] | -<ma||jk>)
 *
 * so that x_{a,(bc)}[i, (jk)] = sum over (e | m) of A B is one product for
 * the particle and the hole term.  The kernels fill one part of the last
 * index each: ld elements per row, the part starts at off.  With m3 the
 * arrays are the three real parts of the 3M scheme (Re, Im, Re + Im: out0,
 * out1, out2 double), else complex (out0). */

#define PUT(x, val)                                         \
    do {                                                    \
        const zc v_ = (val);                                \
        if (m3) {                                           \
            ((double *)out0)[x] = creal(v_);                \
            ((double *)out1)[x] = cimag(v_);                \
            ((double *)out2)[x] = creal(v_) + cimag(v_);    \
        } else                                              \
            ((zc *)out0)[x] = v_;                           \
    } while (0)

/* A[p, i, off + e] = <bc||ei> = -conj(<ie||bc>) for one occupied i and np
 * pairs p = (b>c): src[e ld_src + p] = <ie||bc> for the pair p of the block
 * (the slab of the integrals as the CCSD keeps them, from its first pair
 * on).  Blocks of pairs, so that the rows that are being written stay in
 * the cache while the slab is read by rows. */
void zc_t_ovvv(const zc *src, int64_t ld_src, int64_t nv, int64_t np, int64_t no, int64_t i,
               int64_t ld, int64_t off, int m3, void *out0, void *out1, void *out2)
{
    enum { BLK = 256 };
    #pragma omp parallel for schedule(static)
    for (int64_t p0 = 0; p0 < np; p0 += BLK) {
        const int64_t p1 = p0 + BLK < np ? p0 + BLK : np;
        for (int64_t e = 0; e < nv; e++) {
            const zc *row = src + e * ld_src;
            for (int64_t p = p0; p < p1; p++)
                PUT((p * no + i) * ld + off + e, -conj(row[p]));
        }
    }
}

/* B[a - a0, (j>k), off + e] = t2[j,k,a,e] for a0 <= a < a1 from
 * t2p[(j>k), (a>e)]. */
void zc_t_t2u(const zc *t2p, int64_t npo, int64_t nv, int64_t a0, int64_t a1,
              int64_t ld, int64_t off, int m3, void *out0, void *out1, void *out2)
{
    const int64_t npv = nv * (nv - 1) / 2;
    #pragma omp parallel for collapse(2) schedule(static)
    for (int64_t a = a0; a < a1; a++)
        for (int64_t p = 0; p < npo; p++) {
            const zc *r = t2p + p * npv;
            const int64_t x = ((a - a0) * npo + p) * ld + off;
            for (int64_t e = 0; e < a; e++)
                PUT(x + e, r[a * (a - 1) / 2 + e]);
            PUT(x + a, 0.0);
            for (int64_t e = a + 1; e < nv; e++)
                PUT(x + e, -r[e * (e - 1) / 2 + a]);
        }
}

/* A[q - q0, i, off + m] = t2[i,m,b,c] for the pairs q0 <= q = (b>c) < q1 from
 * t2p[(i>m), (b>c)]. */
void zc_t_t2x(const zc *t2p, int64_t no, int64_t npv, int64_t q0, int64_t q1,
              int64_t ld, int64_t off, int m3, void *out0, void *out1, void *out2)
{
    #pragma omp parallel for schedule(static)
    for (int64_t q = q0; q < q1; q++)
        for (int64_t i = 0; i < no; i++) {
            const int64_t x = ((q - q0) * no + i) * ld + off;
            for (int64_t m = 0; m < i; m++)
                PUT(x + m, t2p[(i * (i - 1) / 2 + m) * npv + q]);
            PUT(x + i, 0.0);
            for (int64_t m = i + 1; m < no; m++)
                PUT(x + m, -t2p[(m * (m - 1) / 2 + i) * npv + q]);
        }
}
