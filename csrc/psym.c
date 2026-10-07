/*
 * (P) in symmetry sectors (zccsdt/cc/pentuples_sym.py).
 *
 * Every virtual spinor has an integer label kv that adds (modulo `modulus`,
 * or as it is for 0).  The sorted tuples of range(n) are numbered
 * lexicographically, as in tnpack.c; the classes of the spinors, pairs,
 * triples and quadruples by label are given by the position of each inside
 * its class (posV, posQ, pos3, pos4).
 *
 * zc_p_operands23:  the stacked operands of one block X[(de), (abc)]: rows
 *     of the integrals and of T2 on the pairs of the block, rows of the
 *     packed T3 and T4 on (g | de) and (f | abc) - the sign of the single
 *     index at its place in the sorted tuple, zero where it is one of the
 *     others -, rows of T3 and blocks of the intermediates on the triples.
 * zc_p_pick:      out[r, c] = sg[r] * M[rows[r], cols[c]].
 * zc_p_y4_row:    one row of the second-order quadruples <Q| V T2 |0> / D:
 *     for the virtual quadruples a<b<c<d of total label M the sum over the
 *     six splits (ab | cd) and the six splits of the occupied quadruple
 *     whose first pair has the label of (ab) of sgn t2[p1, (ab)] w[p2, (cd)];
 *     zero elsewhere.
 * zc_p_sector_energy:  the sum over the virtual pentuples a<b<c<d<e of total
 *     label M of Re(conj(z) r) / (eo - ev), r and z each the sum over the ten
 *     splits (xy | rest) of a block X[(de), (abc)] and over the five splits
 *     (x | rest) of a block X[e, (abcd)]: one block per label q of the pair
 *     at off23[q - qmin] with nc23 columns, one per label of the single
 *     index at off14[its class] with nc14 columns.  For one (a, b, c) the
 *     pairs d<e that complete it are those of one label above c (qprs: the
 *     pairs by label); r and z of these are summed split by split, and all
 *     but one of the fifteen splits read the elements of a block one after
 *     the other: the tuples that differ in the largest index of a class, or
 *     in the last pair, follow each other in their class.
 * zc_p_xq_sector:  the intermediate of three T2 for the occupied spinors o
 *     of one label on the sorted triples t of one label,
 *     out[o, t] = 1/2 sum over the splits (pq | r) of sgn * G[(pq), o, r]
 *               + sum over the six orders (a, b, c) of sgn * Z[a, o, c, b]:
 *     G in blocks by the label of (pq), at goff[q - qmin] with gst elements
 *     from one pair to the next and gnd spinors r; Z in blocks by the
 *     classes of a and b, at off[la * nl + lb] with ast elements from one a
 *     to the next, and ncs and nbs members of the classes of c and b.
 */
#include <complex.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define SMALL 16384             /* elements below which a loop stays on one thread */

#define ZC_RE_d(x) (x)
#define ZC_RE_z(x) creal(x)
#define ZC_CONJ_d(x) (x)
#define ZC_CONJ_z(x) conj(x)

static inline int red(int k, int modulus)
{
    return modulus ? ((k % modulus) + modulus) % modulus : k;
}

/* c2[y] = C(y, 2), c3[y] = C(y, 3), c4[y] = C(y, 4) for y <= n */
static int64_t *binomials(int n)
{
    int64_t *cb = malloc(sizeof(int64_t) * 3 * (size_t)(n + 1));
    for (int64_t y = 0; y <= n; y++) {
        cb[y] = y * (y - 1) / 2;
        cb[n + 1 + y] = y * (y - 1) * (y - 2) / 6;
        cb[2 * (n + 1) + y] = cb[n + 1 + y] * (y - 3) / 4;
    }
    return cb;
}

#define DEFINE_PSEC(T, SFX)                                                    \
static void zc_p_expand13_serial_##SFX(const T *row, int n, const int32_t *es, \
        int ne, const int32_t *tri, int64_t nt, double scale, T *out)          \
{                                                                              \
    const int64_t n1 = n - 1;                                                  \
    const int64_t top = (int64_t)n * n1 / 2 * (n - 2) / 3 * (n - 3) / 4 - 1;   \
    for (int x = 0; x < ne; x++) {                                             \
        const int e = es[x];                                                   \
        T *o = out + x * nt;                                                   \
        for (int64_t t = 0; t < nt; t++) {                                     \
            const int a = tri[3 * t], b = tri[3 * t + 1], c = tri[3 * t + 2];  \
            int64_t y0, y1, y2, y3;                                            \
            double sg;                                                         \
            if (e == a || e == b || e == c) { o[t] = 0; continue; }            \
            if (e < a)      { y0 = n1 - e; y1 = n1 - a; y2 = n1 - b; y3 = n1 - c; sg = scale; }  \
            else if (e < b) { y0 = n1 - a; y1 = n1 - e; y2 = n1 - b; y3 = n1 - c; sg = -scale; } \
            else if (e < c) { y0 = n1 - a; y1 = n1 - b; y2 = n1 - e; y3 = n1 - c; sg = scale; }  \
            else            { y0 = n1 - a; y1 = n1 - b; y2 = n1 - c; y3 = n1 - e; sg = -scale; } \
            o[t] = sg * row[top - y0 * (y0 - 1) / 2 * (y0 - 2) / 3 * (y0 - 3) / 4 \
                            - y1 * (y1 - 1) * (y1 - 2) / 6 - y2 * (y2 - 1) / 2 - y3]; \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
static void zc_p_expand12_serial_##SFX(const T *row, int n, const int32_t *es, \
        int ne, const int32_t *prs, int64_t np, double scale, T *out)          \
{                                                                              \
    const int64_t n1 = n - 1;                                                  \
    const int64_t top = (int64_t)n * n1 / 2 * (n - 2) / 3 - 1;                 \
    for (int x = 0; x < ne; x++) {                                             \
        const int e = es[x];                                                   \
        T *o = out + x * np;                                                   \
        for (int64_t t = 0; t < np; t++) {                                     \
            const int a = prs[2 * t], b = prs[2 * t + 1];                      \
            int64_t y0, y1, y2;                                                \
            double sg;                                                         \
            if (e == a || e == b) { o[t] = 0; continue; }                      \
            if (e < a)      { y0 = n1 - e; y1 = n1 - a; y2 = n1 - b; sg = scale; }  \
            else if (e < b) { y0 = n1 - a; y1 = n1 - e; y2 = n1 - b; sg = -scale; } \
            else            { y0 = n1 - a; y1 = n1 - b; y2 = n1 - e; sg = scale; }  \
            o[t] = sg * row[top - y0 * (y0 - 1) * (y0 - 2) / 6 - y1 * (y1 - 1) / 2 - y2]; \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
void zc_p_pick_##SFX(const T *M, int64_t ld, const int64_t *rows,              \
        const double *sg, int64_t nr, const int64_t *cols, int64_t nc, T *out) \
{                                                                              \
    _Pragma("omp parallel for schedule(static) collapse(2) if(nr * nc > SMALL)") \
    for (int64_t r = 0; r < nr; r++)                                           \
    for (int64_t c = 0; c < nc; c++)                                           \
        out[r * nc + c] = sg[r] * M[rows[r] * ld + cols[c]];                   \
}                                                                              \
                                                                               \
/* the stacked operands of one block X[(de), (abc)] of r (L, R) and of z   \
 * (LZ, RZ): for every segment seg[8 s ..] = (kind, first row, rows,         \
 * left source, left indices, right source, right indices, right signs)   \
 * - addresses -, with the factor scale[s] on the left:                     \
 *   0  <de||fm> (rows f of [f, (de)])        t(ijkl; f abc) of a row of T4  \
 *   1  t(in; de) (rows n of [n, (de)])       a block Xq[n, (abc)]           \
 *   2  Xo(n; ijm; de) (rows n)               rows of T3 with their signs    \
 *   3  t(klm; g de) of a row of T3           a block Xv[g, (abc)]           \
 *   4  as 0, into LZ and RZ (the row: of the second-order quadruples)       \
 *   5  one row <de||lm>                      one row t(ijk; abc), into LZ, RZ \
 * rq, prs: the pairs (de) of the block as positions and as indices; ca,    \
 * tri: its triples. */                                                      \
void zc_p_operands23_##SFX(int n, int64_t nseg, const int64_t *seg,            \
        const double *scale, const int64_t *rq, int64_t nr,                    \
        const int32_t *prs, const int64_t *ca, int64_t nc, const int32_t *tri, \
        T *L, T *R, T *LZ, T *RZ)                                              \
{                                                                              \
    const int64_t nQ = (int64_t)n * (n - 1) / 2, nA3 = nQ * (n - 2) / 3;       \
    int64_t rows = 0;                                                          \
    for (int64_t s = 0; s < nseg; s++) rows += seg[8 * s + 2];                 \
    _Pragma("omp parallel for schedule(dynamic, 1) if(rows * nc > SMALL)")     \
    for (int64_t s = 0; s < nseg; s++) {                                       \
        const int64_t *g = seg + 8 * s;                                        \
        const int kind = (int)g[0];                                            \
        const int64_t at = g[1], m = g[2];                                     \
        const double f = scale[s];                                             \
        T *l = (kind < 4 ? L : LZ) + at * nr, *r = (kind < 4 ? R : RZ) + at * nc; \
        if (kind == 3) {                                                       \
            zc_p_expand12_serial_##SFX((const T *)g[3], n, (const int32_t *)g[4], (int)m, prs, nr, f, l); \
        } else if (kind == 5) {                                                \
            const T *x = (const T *)g[3];                                      \
            for (int64_t c = 0; c < nr; c++) l[c] = f * x[rq[c]];              \
        } else {                                                               \
            const T *x = (const T *)g[3];                                      \
            const int64_t *idx = (const int64_t *)g[4];                        \
            for (int64_t i = 0; i < m; i++)                                    \
                for (int64_t c = 0; c < nr; c++) l[i * nr + c] = f * x[idx[i] * nQ + rq[c]]; \
        }                                                                      \
        if (kind == 0 || kind == 4) {                                          \
            zc_p_expand13_serial_##SFX((const T *)g[5], n, (const int32_t *)g[6], (int)m, tri, nc, 1.0, r); \
        } else if (kind == 1 || kind == 3) {                                   \
            memcpy(r, (const T *)g[5], sizeof(T) * (size_t)(m * nc));          \
        } else if (kind == 2) {                                                \
            const T *x = (const T *)g[5];                                      \
            const int64_t *idx = (const int64_t *)g[6];                        \
            const double *sg = (const double *)g[7];                           \
            for (int64_t i = 0; i < m; i++)                                    \
                for (int64_t c = 0; c < nc; c++) r[i * nc + c] = sg[i] * x[idx[i] * nA3 + ca[c]]; \
        } else {                                                               \
            const T *x = (const T *)g[5];                                      \
            for (int64_t c = 0; c < nc; c++) r[c] = x[ca[c]];                  \
        }                                                                      \
    }                                                                          \
}                                                                              \
                                                                               \
void zc_p_y4_row_##SFX(int n, const int *kv, int modulus, int M,               \
        const double *ev, double eo, int kmin, int nk, const int *lid_of,      \
        const int *lstart, const int *lmem, const T *t2, const T *w,           \
        const int64_t *p1, const int64_t *p2, const double *sgn,               \
        const int *klab, T *out)                                               \
{                                                                              \
    int64_t *cb = binomials(n);                                                \
    const int64_t *c2 = cb, *c3 = cb + n + 1, *c4 = cb + 2 * (n + 1);          \
    const int64_t nQ = c2[n], t2r = c2[n] - 1, t4 = c4[n] - 1;                 \
    static const int g22[6][4] = {{0,1,2,3}, {0,2,1,3}, {0,3,1,2},             \
                                  {1,2,0,3}, {1,3,0,2}, {2,3,0,1}};            \
    static const double s22[6] = {1, -1, 1, 1, -1, 1};                         \
    _Pragma("omp parallel for schedule(static) if(c4[n] > SMALL)")             \
    for (int64_t A = 0; A < c4[n]; A++) out[A] = 0;                            \
    _Pragma("omp parallel for schedule(dynamic, 1) if(c4[n] > SMALL)")         \
    for (int a = 0; a < n - 3; a++)                                            \
    for (int b = a + 1; b < n - 2; b++)                                        \
    for (int c = b + 1; c < n - 1; c++) {                                      \
        const int kd = red(M - kv[a] - kv[b] - kv[c], modulus);                \
        if (kd < kmin || kd >= kmin + nk || lid_of[kd - kmin] < 0) continue;   \
        const int l = lid_of[kd - kmin];                                       \
        int lo = lstart[l], hi = lstart[l + 1];                                \
        while (lo < hi) {               /* the first member above c */         \
            int mid = (lo + hi) / 2;                                           \
            if (lmem[mid] > c) hi = mid; else lo = mid + 1;                    \
        }                                                                      \
        const int64_t rk = t4 - c4[n - 1 - a] - c3[n - 1 - b] - c2[n - 1 - c] - (n - 1); \
        for (int t = lo; t < lstart[l + 1]; t++) {                             \
            const int v[4] = {a, b, c, lmem[t]};                               \
            T z = 0;                                                           \
            for (int s = 0; s < 6; s++) {                                      \
                const int p = v[g22[s][0]], q = v[g22[s][1]];                  \
                const int r = v[g22[s][2]], u = v[g22[s][3]];                  \
                const int kq = red(kv[p] + kv[q], modulus);                    \
                const int64_t rk1 = t2r - c2[n - 1 - p] - (n - 1 - q);         \
                const int64_t rk2 = t2r - c2[n - 1 - r] - (n - 1 - u);         \
                T x = 0;                                                       \
                for (int o = 0; o < 6; o++)                                    \
                    if (klab[o] == kq)                                         \
                        x += sgn[o] * t2[p1[o] * nQ + rk1] * w[p2[o] * nQ + rk2]; \
                z += s22[s] * x;                                               \
            }                                                                  \
            out[rk + v[3]] = z / (eo - ev[a] - ev[b] - ev[c] - ev[v[3]]);      \
        }                                                                      \
    }                                                                          \
    free(cb);                                                                  \
}                                                                              \
                                                                               \
void zc_p_sector_energy_##SFX(int n, const int *kv, int modulus, int M,        \
        const double *ev, double eo, int kmin, int nk, const int *lid_of,      \
        int nl, const int *lstart, const int *lmem, int qmin, int nq,          \
        const int64_t *qstart, const int64_t *qlen, const int32_t *qprs,       \
        const int64_t *off23, const int64_t *nc23, const int64_t *off14,       \
        const int64_t *nc14, const int32_t *posV, const int32_t *posQ,         \
        const int32_t *pos3, const int32_t *pos4, const T *r23, const T *z23,  \
        const T *r14, const T *z14, double *e)                                 \
{                                                                              \
    int64_t *cb = binomials(n);                                                \
    const int64_t *c2 = cb, *c3 = cb + n + 1, *c4 = cb + 2 * (n + 1);          \
    const int64_t t2 = c2[n] - 1, t3 = c3[n] - 1, t4 = c4[n] - 1;              \
    int64_t maxq = 1;                                                          \
    for (int q = 0; q < nq; q++) if (qlen[q] > maxq) maxq = qlen[q];           \
    double esum = 0;                                                           \
    _Pragma("omp parallel reduction(+:esum) if(c4[n] * n > 20 * SMALL)")       \
    {                                                                          \
    T *rb = malloc(sizeof(T) * (size_t)maxq), *zb = malloc(sizeof(T) * (size_t)maxq); \
    int64_t *ix = malloc(sizeof(int64_t) * (size_t)(n + 1));                   \
    _Pragma("omp for schedule(dynamic, 1)")                                    \
    for (int a = 0; a < n - 4; a++)                                            \
    for (int b = a + 1; b < n - 3; b++)                                        \
    for (int c = b + 1; c < n - 2; c++) {                                      \
        const int kde = red(M - kv[a] - kv[b] - kv[c], modulus);               \
        const int q = kde - qmin;                                              \
        if (q < 0 || q >= nq || qlen[q] == 0) continue;                        \
        const int32_t *prs = qprs + 2 * qstart[q];                             \
        int64_t r0 = 0, hi = qlen[q];                                          \
        while (r0 < hi) {               /* the first pair with d above c */    \
            int64_t mid = (r0 + hi) / 2;                                       \
            if (prs[2 * mid] > c) hi = mid; else r0 = mid + 1;                 \
        }                                                                      \
        const int64_t nb = qlen[q] - r0;                                       \
        if (nb == 0) continue;                                                 \
        prs += 2 * r0;                                                         \
        memset(rb, 0, sizeof(T) * (size_t)nb);                                 \
        memset(zb, 0, sizeof(T) * (size_t)nb);                                 \
        const int v[3] = {a, b, c};                                            \
        static const int g3[3][3] = {{0, 1, 2}, {0, 2, 1}, {1, 2, 0}};         \
        static const int h3[3][3] = {{0, 1, 2}, {1, 0, 2}, {2, 0, 1}};         \
        static const double s3[3] = {1, -1, 1};                                \
        const int64_t qa = t4 - c4[n - 1 - a] - c3[n - 1 - b] - c2[n - 1 - c] - (n - 1); /* + e: (a b c e) */ \
        /* (de | abc): one column of the block of (de) */                      \
        if (off23[q] >= 0) {                                                   \
            const int64_t st = nc23[q];                                        \
            const int64_t at = off23[q] + r0 * st + pos3[t3 - c3[n - 1 - a] - c2[n - 1 - b] - (n - 1 - c)];    \
            for (int64_t t = 0; t < nb; t++) { rb[t] += r23[at + t * st]; zb[t] += z23[at + t * st]; } \
        }                                                                      \
        /* (xy | z de) and (x | yz de), x, y, z of a, b, c: all pairs in a row */ \
        for (int s = 0; s < 3; s++) {                                          \
            const int x = v[g3[s][0]], y = v[g3[s][1]], z = v[g3[s][2]];       \
            const int kq = red(kv[x] + kv[y], modulus) - qmin;                 \
            if (kq >= 0 && kq < nq && off23[kq] >= 0) {                        \
                const int64_t at = off23[kq] + posQ[t2 - c2[n - 1 - x] - (n - 1 - y)] * nc23[kq] \
                    + pos3[t3 - c3[n - 1 - z] - c2[n - 1 - prs[0]] - (n - 1 - prs[1])]; \
                for (int64_t t = 0; t < nb; t++) { rb[t] += s3[s] * r23[at + t]; zb[t] += s3[s] * z23[at + t]; } \
            }                                                                  \
            const int p = v[h3[s][0]], u = v[h3[s][1]], w = v[h3[s][2]];       \
            const int lp = lid_of[kv[p] - kmin];                               \
            if (off14[lp] >= 0) {                                              \
                const int64_t at = off14[lp] + posV[p] * nc14[lp]              \
                    + pos4[t4 - c4[n - 1 - u] - c3[n - 1 - w] - c2[n - 1 - prs[0]] - (n - 1 - prs[1])]; \
                for (int64_t t = 0; t < nb; t++) { rb[t] += s3[s] * r14[at + t]; zb[t] += s3[s] * z14[at + t]; } \
            }                                                                  \
        }                                                                      \
        /* (xd | yz e) and (d | abc e): the e of one d in a row */             \
        for (int64_t t0 = 0; t0 < nb; ) {                                      \
            const int d = prs[2 * t0], e0 = prs[2 * t0 + 1];                   \
            int64_t t1 = t0 + 1;                                               \
            while (t1 < nb && prs[2 * t1] == d) t1++;                          \
            const int64_t len = t1 - t0;                                       \
            for (int s = 0; s < 3; s++) {                                      \
                const int x = v[h3[s][0]], y = v[h3[s][1]], z = v[h3[s][2]];   \
                const int kq = red(kv[x] + kv[d], modulus) - qmin;             \
                if (kq < 0 || kq >= nq || off23[kq] < 0) continue;             \
                const int64_t at = off23[kq] + posQ[t2 - c2[n - 1 - x] - (n - 1 - d)] * nc23[kq] \
                    + pos3[t3 - c3[n - 1 - y] - c2[n - 1 - z] - (n - 1 - e0)]; \
                for (int64_t u = 0; u < len; u++) {                            \
                    rb[t0 + u] += s3[s] * r23[at + u];                         \
                    zb[t0 + u] += s3[s] * z23[at + u];                         \
                }                                                              \
            }                                                                  \
            const int ld = lid_of[kv[d] - kmin];                               \
            if (off14[ld] >= 0) {                                              \
                const int64_t at = off14[ld] + posV[d] * nc14[ld] + pos4[qa + e0]; \
                for (int64_t u = 0; u < len; u++) { rb[t0 + u] -= r14[at + u]; zb[t0 + u] -= z14[at + u]; } \
            }                                                                  \
            t0 = t1;                                                           \
        }                                                                      \
        /* (xe | yz d) and (e | abc d): the d of one class below one e in a row */ \
        for (int l = 0; l < nl; l++) {                                         \
            const int ke = red(kde - kv[lmem[lstart[l]]], modulus);            \
            if (ke < kmin || ke >= kmin + nk || lid_of[ke - kmin] < 0) continue; \
            const int le = lid_of[ke - kmin];                                  \
            int dlo = lstart[l], dhi = lstart[l + 1];                          \
            while (dlo < dhi) {                                                \
                int mid = (dlo + dhi) / 2;                                     \
                if (lmem[mid] > c) dhi = mid; else dlo = mid + 1;              \
            }                                                                  \
            if (dlo == lstart[l + 1]) continue;                                \
            const int d0 = lmem[dlo];                                          \
            dhi = dlo;                                                         \
            for (int te = lstart[le]; te < lstart[le + 1]; te++) {             \
                const int e1 = lmem[te];                                       \
                while (dhi < lstart[l + 1] && lmem[dhi] < e1) dhi++;           \
                const int len = dhi - dlo;                                     \
                if (len <= 0) continue;                                        \
                const int64_t top = t2 - (n - 1 - e1);     /* the places of the pairs (d e) */ \
                for (int u = 0; u < len; u++)                                  \
                    ix[u] = posQ[top - c2[n - 1 - lmem[dlo + u]]] - r0;        \
                for (int s = 0; s < 3; s++) {                                  \
                    const int x = v[h3[s][0]], y = v[h3[s][1]], z = v[h3[s][2]]; \
                    const int kq = red(kv[x] + kv[e1], modulus) - qmin;        \
                    if (kq < 0 || kq >= nq || off23[kq] < 0) continue;         \
                    const int64_t at = off23[kq] + posQ[t2 - c2[n - 1 - x] - (n - 1 - e1)] * nc23[kq] \
                        + pos3[t3 - c3[n - 1 - y] - c2[n - 1 - z] - (n - 1 - d0)]; \
                    for (int u = 0; u < len; u++) {                            \
                        rb[ix[u]] -= s3[s] * r23[at + u];                      \
                        zb[ix[u]] -= s3[s] * z23[at + u];                      \
                    }                                                          \
                }                                                              \
                if (off14[le] >= 0) {                                          \
                    const int64_t at = off14[le] + posV[e1] * nc14[le] + pos4[qa + d0]; \
                    for (int u = 0; u < len; u++) { rb[ix[u]] += r14[at + u]; zb[ix[u]] += z14[at + u]; } \
                }                                                              \
            }                                                                  \
        }                                                                      \
        const double eabc = eo - ev[a] - ev[b] - ev[c];                        \
        for (int64_t t = 0; t < nb; t++)                                       \
            esum += ZC_RE_##SFX(ZC_CONJ_##SFX(zb[t]) * rb[t]) / (eabc - ev[prs[2 * t]] - ev[prs[2 * t + 1]]); \
    }                                                                          \
    free(rb);                                                                  \
    free(zb);                                                                  \
    free(ix);                                                                  \
    }                                                                          \
    e[0] += esum;                                                              \
    free(cb);                                                                  \
}                                                                              \
                                                                               \
void zc_p_xq_sector_##SFX(int n, const int *kv, int modulus, int kmin,         \
        const int *lid_of, int nl, const int32_t *tri, int64_t nt, int no,     \
        int qmin, int nq, const int64_t *goff, const int64_t *gst,             \
        const int64_t *gnd, const int32_t *posQ, const T *G,                   \
        const int64_t *off, const int64_t *ast, const int64_t *ncs,            \
        const int64_t *nbs, const int32_t *posV, const T *Z, T *out)           \
{                                                                              \
    const int64_t nQ = (int64_t)n * (n - 1) / 2;                               \
    static const int g3[3][3] = {{0, 1, 2}, {0, 2, 1}, {1, 2, 0}};             \
    static const double s3[3] = {0.5, -0.5, 0.5};                              \
    static const int g[6][3] = {{0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0}}; \
    static const double sg[6] = {1, -1, -1, 1, 1, -1};                         \
    _Pragma("omp parallel for schedule(static) if(nt * no > SMALL)")           \
    for (int64_t t = 0; t < nt; t++) {                                         \
        const int v[3] = {tri[3 * t], tri[3 * t + 1], tri[3 * t + 2]};         \
        for (int o = 0; o < no; o++) out[o * nt + t] = 0;                      \
        for (int s = 0; s < 3; s++) {                                          \
            const int p = v[g3[s][0]], q = v[g3[s][1]], r = v[g3[s][2]];       \
            const int kq = red(kv[p] + kv[q], modulus) - qmin;                 \
            if (kq < 0 || kq >= nq || goff[kq] < 0) continue;                  \
            const int64_t rk = nQ - 1 - (int64_t)(n - 1 - p) * (n - 2 - p) / 2 - (n - 1 - q); \
            const int64_t d = gnd[kq];                                         \
            const T *m = G + goff[kq] + posQ[rk] * gst[kq] + posV[r];          \
            for (int o = 0; o < no; o++) out[o * nt + t] += s3[s] * m[o * d];  \
        }                                                                      \
        for (int s = 0; s < 6; s++) {                                          \
            const int a = v[g[s][0]], b = v[g[s][1]], c = v[g[s][2]];          \
            const int64_t k = (int64_t)lid_of[kv[a] - kmin] * nl + lid_of[kv[b] - kmin]; \
            if (off[k] < 0) continue;                                          \
            const int64_t nc = ncs[k], nb = nbs[k];                            \
            const T *z = Z + off[k] + posV[a] * ast[k] + posV[c] * nb + posV[b]; \
            for (int o = 0; o < no; o++) out[o * nt + t] += sg[s] * z[o * nc * nb]; \
        }                                                                      \
    }                                                                          \
}

DEFINE_PSEC(double, d)
DEFINE_PSEC(double complex, z)
