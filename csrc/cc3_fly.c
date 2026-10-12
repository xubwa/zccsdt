/* CC3 without stored triples (zccsdt/cc/cc3fly.py; docs/ccn.md, sections 9
 * and 10): the triples of one batch of occupied triples are made from the
 * drive terms, divided by the orbital-energy denominator, contracted into
 * the singles and doubles equations and discarded.
 *
 * One call of zc_cc3_batch does one batch - the occupied triples (i, j, k),
 * i0 <= i < i1 <= j < k, with the i of one symmetry class - completely.
 * Everything is passed in plain row-major arrays (int64 sizes and tables,
 * double or double complex data): a Fortran driver calls it as the Python
 * scheduler does.  The products go through zc_gemm (csrc/ccsdt_t3.c), one
 * sequential GEMM per piece of a product, the pieces of all products of a
 * stage on one queue of the OpenMP threads.
 *
 * Symmetry classes.  The occupied and the virtual spinors are in the order
 * of their classes 0 .. ncls - 1 (labels that add; ncls = 1: no symmetry).
 * orng, vrng (ncls + 1): the first spinor of every class.  The virtual pairs
 * Q = (a < b) are numbered group by group of the sum of their classes,
 * qoff (2 ncls): the first pair of the groups 0 .. 2 ncls - 2; the virtual
 * triples A = (a < b < c) likewise, aoff (3 ncls - 1).  An amplitude
 * t3[ijk, A] is not zero only if the group of A is the sum of the classes
 * of i, j, k.  The tables of the virtual triples are those of
 * zccsdt.cc.t3pack.TripleMap: vtri[d, Q], vsgn[d, Q] - the triple of the
 * spinor d and the pair Q and its sign (0: d in Q) -, vfree[r, A],
 * vpair[r, A], r = 0, 1, 2 - the three ways (free; pair) of the triple A,
 * with the signs +, -, +.
 *
 * Operands (hi > lo: an occupied pair as [hi][lo]; the other triangle is
 * not read):
 *   At[hi][lo][c][e] = a2[lo hi, e c]          amplitudes of the drive term
 *   Ao[hi][lo][c][m] = <m c||lo hi>~           W_vooo
 *   Bv[e][i][Q]      = <Q||e i>~               W_vvvo
 *   Bo[m][i][Q]      = a2[m i, Q]
 * - nset = 1: the triples of the ground state (a2 = t2); nset = 2: the sum
 * of two drive terms (an excitation vector: t2 with the derivatives of the
 * integrals, r2 with the integrals).  With Z[P, c, f, Q] = - sum_e
 * At[P][c][e] Bv[e][f][Q] - sum_m Ao[P][c][m] Bo[m][f][Q],
 *
 *   t3[ijk, A] = sum_r s_r (Z[(jk), a_r, i, Q_r] - Z[(ik), a_r, j, Q_r]
 *                + Z[(ij), a_r, k, Q_r]) / (eo[i] + eo[j] + eo[k] - ev3[A] + omega).
 *
 * Contractions (T[d, Q] = vsgn[d, Q] t3[ijk, vtri[d, Q]]; (m; P), sign s:
 * the three ways (i; jk) +, (j; ik) -, (k; ij) + of the occupied triple):
 *   Ysq[P][d][y] += s sum_Q  T[d, Q] Gv[m][Q][y]
 *   Xo[x][m][Q]  += s sum_d  Gsq[P][x][d] T[d, Q]
 *   r1[m][d]     += s sum_Q  T[d, Q] gsq[P][Q]
 *   rAsq[P][Q]   += s sum_d  fov[m][d] T[d, Q]
 * (P as [hi][lo]) - pass 1 of the CCSDT step for the rows of this batch.
 *
 * work: ni (3 H + nA_g) elements, ni = i1 - i0, H the number of (d, Q) with
 * the group of the batch, nA_g its virtual triples: zc_cc3_work returns it.
 * stats[0 .. 3] += seconds of the drive products, of the amplitudes, of the
 * gather, of the contractions.
 */
#include <complex.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void zc_gemm_d(int ta, int tb, int64_t m, int64_t n, int64_t k,
               const double *alpha, const double *a, int64_t lda,
               const double *b, int64_t ldb, const double *beta, double *c,
               int64_t ldc);
void zc_gemm_z(int ta, int tb, int64_t m, int64_t n, int64_t k,
               const double complex *alpha, const double complex *a, int64_t lda,
               const double complex *b, int64_t ldb, const double complex *beta,
               double complex *c, int64_t ldc);

/* H and nA_g of the group g, and the offset of every class block (zoff,
 * ncls + 1 entries: block s is [c in class s][Q in group g - s]) */
static int64_t cc3_layout(int64_t ncls, int64_t g, const int64_t *vrng,
                          const int64_t *qoff, int64_t *zoff)
{
    int64_t H = 0;
    for (int64_t s = 0; s < ncls; s++) {
        zoff[s] = H;
        const int64_t h = g - s;
        if (h < 0 || h > 2 * ncls - 2) continue;
        H += (vrng[s + 1] - vrng[s]) * (qoff[h + 1] - qoff[h]);
    }
    zoff[ncls] = H;
    return H;
}

/* elements of work for a batch of ni occupied triples whose classes add up to g */
int64_t zc_cc3_work(int64_t ncls, int64_t g, int64_t ni, const int64_t *vrng,
                    const int64_t *qoff, const int64_t *aoff)
{
    int64_t *zoff = malloc(sizeof(int64_t) * (ncls + 1));
    const int64_t H = cc3_layout(ncls, g, vrng, qoff, zoff);
    free(zoff);
    return ni * (3 * H + aoff[g + 1] - aoff[g]);
}

#define MAXPROD 4

#define DEFINE_CC3(T, SFX)                                                     \
typedef struct {                                                               \
    int64_t m, n, np;                                                          \
    int64_t k[MAXPROD], lda[MAXPROD], ldb[MAXPROD];                            \
    const T *a[MAXPROD], *b[MAXPROD];                                          \
    T *c; int64_t ldc; double alpha;                                           \
} cc3_task_##SFX;                                                              \
                                                                               \
/* c += alpha sum_p a_p b_p for every task; a task is cut into pieces of      \
 * columns or of rows (whichever gives more pieces), a piece is done by one  \
 * thread.  The c of the tasks do not overlap. */                             \
static void cc3_run_##SFX(int64_t ntask, const cc3_task_##SFX *tasks)          \
{                                                                              \
    const int64_t cols = 96, rows = 16;                                        \
    int64_t total = 0;                                                         \
    for (int64_t t = 0; t < ntask; t++) {                                      \
        const cc3_task_##SFX *q = tasks + t;                                   \
        if (q->m <= 0 || q->n <= 0 || q->np == 0) continue;                    \
        const int64_t pc = (q->n + cols - 1) / cols, pr = (q->m + rows - 1) / rows; \
        total += (pc >= pr) ? pc : pr;                                         \
    }                                                                          \
    if (total == 0) return;                                                    \
    int64_t *piece = malloc(sizeof(int64_t) * 4 * total);                      \
    int64_t np = 0;                                                            \
    for (int64_t t = 0; t < ntask; t++) {                                      \
        const cc3_task_##SFX *q = tasks + t;                                   \
        if (q->m <= 0 || q->n <= 0 || q->np == 0) continue;                    \
        const int64_t pc = (q->n + cols - 1) / cols, pr = (q->m + rows - 1) / rows; \
        const int bycol = pc >= pr;                                            \
        const int64_t len = bycol ? q->n : q->m, n = bycol ? pc : pr;          \
        for (int64_t j = 0; j < n; j++) {                                      \
            piece[4 * np] = t;                                                 \
            piece[4 * np + 1] = j * len / n;                                   \
            piece[4 * np + 2] = (j + 1) * len / n;                             \
            piece[4 * np + 3] = bycol;                                         \
            np++;                                                              \
        }                                                                      \
    }                                                                          \
    _Pragma("omp parallel for schedule(dynamic, 1)")                           \
    for (int64_t p = 0; p < np; p++) {                                         \
        const cc3_task_##SFX *q = tasks + piece[4 * p];                        \
        const int64_t x0 = piece[4 * p + 1], x1 = piece[4 * p + 2];            \
        const int bycol = (int)piece[4 * p + 3];                               \
        const T alpha = q->alpha, one = 1.0;                                   \
        for (int64_t s = 0; s < q->np; s++) {                                  \
            if (q->k[s] <= 0) continue;                                        \
            if (bycol)                                                         \
                zc_gemm_##SFX(0, 0, q->m, x1 - x0, q->k[s], &alpha, q->a[s],   \
                              q->lda[s], q->b[s] + x0, q->ldb[s], &one,        \
                              q->c + x0, q->ldc);                              \
            else                                                               \
                zc_gemm_##SFX(0, 0, x1 - x0, q->n, q->k[s], &alpha,            \
                              q->a[s] + x0 * q->lda[s], q->lda[s], q->b[s],    \
                              q->ldb[s], &one, q->c + x0 * q->ldc, q->ldc);    \
        }                                                                      \
    }                                                                          \
    free(piece);                                                               \
}                                                                              \
                                                                               \
void zc_cc3_batch_##SFX(                                                       \
    int64_t no, int64_t nv, int64_t nQ, int64_t nA, int64_t ncls,              \
    const int64_t *orng, const int64_t *vrng, const int64_t *qoff,             \
    const int64_t *aoff, const int64_t *vtri, const int8_t *vsgn,              \
    const int64_t *vfree, const int64_t *vpair,                                \
    const double *eo, const double *ev3, double omega,                         \
    int64_t i0, int64_t i1, int64_t j, int64_t k, int64_t nset,                \
    const T *At1, const T *Ao1, const T *Bv1, const T *Bo1,                    \
    const T *At2, const T *Ao2, const T *Bv2, const T *Bo2,                    \
    const T *Gv, const T *Gsq, const T *gsq, const T *fov,                     \
    T *Ysq, T *Xo, T *r1, T *rAsq, T *work, double *stats)                     \
{                                                                              \
    const int64_t ni = i1 - i0;                                                \
    if (ni <= 0) return;                                                       \
    /* classes of the spinors */                                               \
    int64_t *ocls = malloc(sizeof(int64_t) * (no + nv + ncls + 1));            \
    int64_t *vcls = ocls + no, *zoff = vcls + nv;                              \
    for (int64_t s = 0; s < ncls; s++) {                                       \
        for (int64_t p = orng[s]; p < orng[s + 1]; p++) ocls[p] = s;           \
        for (int64_t p = vrng[s]; p < vrng[s + 1]; p++) vcls[p] = s;           \
    }                                                                          \
    const int64_t ci = ocls[i0], cj = ocls[j], ck = ocls[k];                   \
    const int64_t g = ci + cj + ck;                                            \
    const int64_t H = cc3_layout(ncls, g, vrng, qoff, zoff);                   \
    const int64_t A0 = aoff[g], nAg = aoff[g + 1] - aoff[g];                   \
    if (H == 0 || nAg == 0) { free(ocls); return; }                            \
    T *Z = work, *t3 = work + 3 * ni * H;                                      \
    const int64_t npg = 2 * ncls - 1;                                          \
    double w0 = omp_get_wtime();                                               \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t x = 0; x < 3 * ni; x++)                                       \
        memset(Z + x * H, 0, sizeof(T) * H);                                   \
    /* ---- the drive terms: three pieces [i][H], for (jk; i), (ik; j), (ij; k) */ \
    const T *At[2] = {At1, At2}, *Ao[2] = {Ao1, Ao2};                          \
    const T *Bv[2] = {Bv1, Bv2}, *Bo[2] = {Bo1, Bo2};                          \
    cc3_task_##SFX *tasks = malloc(sizeof(cc3_task_##SFX) * 5 * 3 * ni * ncls); \
    int64_t nt = 0;                                                            \
    for (int64_t i = i0; i < i1; i++)                                          \
    for (int r = 0; r < 3; r++) {                                              \
        const int64_t lo = (r == 0) ? j : i, hi = (r == 2) ? j : k;            \
        const int64_t f = (r == 0) ? i : (r == 1) ? j : k;                     \
        const int64_t cP = ocls[lo] + ocls[hi];                                \
        for (int64_t s = 0; s < ncls; s++) {                                   \
            const int64_t h = g - s, x = cP - s;                               \
            if (h < 0 || h >= npg || x < 0 || x >= ncls) continue;             \
            cc3_task_##SFX *q = tasks + nt;                                    \
            q->m = vrng[s + 1] - vrng[s];                                      \
            q->n = qoff[h + 1] - qoff[h];                                      \
            q->c = Z + (r * ni + (i - i0)) * H + zoff[s];                      \
            q->ldc = q->n;                                                     \
            q->alpha = -1.0;                                                   \
            q->np = 0;                                                         \
            for (int64_t z = 0; z < nset; z++) {                               \
                int64_t p = q->np++;                                           \
                q->k[p] = vrng[x + 1] - vrng[x];                               \
                q->a[p] = At[z] + ((hi * no + lo) * nv + vrng[s]) * nv + vrng[x]; \
                q->lda[p] = nv;                                                \
                q->b[p] = Bv[z] + (vrng[x] * no + f) * nQ + qoff[h];           \
                q->ldb[p] = no * nQ;                                           \
                p = q->np++;                                                   \
                q->k[p] = orng[x + 1] - orng[x];                               \
                q->a[p] = Ao[z] + ((hi * no + lo) * nv + vrng[s]) * no + orng[x]; \
                q->lda[p] = no;                                                \
                q->b[p] = Bo[z] + (orng[x] * no + f) * nQ + qoff[h];           \
                q->ldb[p] = no * nQ;                                           \
            }                                                                  \
            if (q->m > 0 && q->n > 0) nt++;                                    \
        }                                                                      \
    }                                                                          \
    cc3_run_##SFX(nt, tasks);                                                  \
    double w1 = omp_get_wtime();                                               \
    stats[0] += w1 - w0;                                                       \
    /* ---- the amplitudes of the batch: t3[i][A - A0] */                      \
    int64_t *pos = malloc(sizeof(int64_t) * 3 * nAg);                          \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t A = 0; A < nAg; A++)                                          \
        for (int r = 0; r < 3; r++) {                                          \
            const int64_t a = vfree[r * nA + A0 + A], Q = vpair[r * nA + A0 + A]; \
            const int64_t s = vcls[a], h = g - s;                              \
            pos[3 * A + r] = zoff[s] + (a - vrng[s]) * (qoff[h + 1] - qoff[h]) \
                             + Q - qoff[h];                                    \
        }                                                                      \
    const double ejk = eo[j] + eo[k] + omega;                                  \
    _Pragma("omp parallel for collapse(2) schedule(static)")                   \
    for (int64_t i = 0; i < ni; i++)                                           \
    for (int64_t A = 0; A < nAg; A++) {                                        \
        const T *za = Z + i * H, *zb = Z + (ni + i) * H, *zc = Z + (2 * ni + i) * H; \
        const int64_t *p = pos + 3 * A;                                        \
        const T v = (za[p[0]] - zb[p[0]] + zc[p[0]])                           \
                  - (za[p[1]] - zb[p[1]] + zc[p[1]])                           \
                  + (za[p[2]] - zb[p[2]] + zc[p[2]]);                          \
        t3[i * nAg + A] = v / (eo[i0 + i] + ejk - ev3[A0 + A]);                \
    }                                                                          \
    free(pos);                                                                 \
    double w2 = omp_get_wtime();                                               \
    stats[1] += w2 - w1;                                                       \
    /* ---- gathered: T[i][block s][d][Q] over the first piece */              \
    for (int64_t s = 0; s < ncls; s++) {                                       \
        const int64_t h = g - s;                                               \
        if (h < 0 || h >= npg) continue;                                       \
        const int64_t nd = vrng[s + 1] - vrng[s], nq = qoff[h + 1] - qoff[h];  \
        if (nd == 0 || nq == 0) continue;                                      \
        _Pragma("omp parallel for collapse(2) schedule(static)")               \
        for (int64_t i = 0; i < ni; i++)                                       \
        for (int64_t d = 0; d < nd; d++) {                                     \
            T *o = Z + i * H + zoff[s] + d * nq;                               \
            const T *row = t3 + i * nAg - A0;                                  \
            const int64_t *vt = vtri + (vrng[s] + d) * nQ + qoff[h];           \
            const int8_t *vs = vsgn + (vrng[s] + d) * nQ + qoff[h];            \
            for (int64_t q = 0; q < nq; q++)                                   \
                o[q] = vs[q] ? (double)vs[q] * row[vt[q]] : 0.0;               \
        }                                                                      \
    }                                                                          \
    double w3 = omp_get_wtime();                                               \
    stats[2] += w3 - w2;                                                       \
    /* ---- the contractions, one round per i (the targets of a round do not  \
     * overlap; those of two rounds do) */                                     \
    for (int64_t i = i0; i < i1; i++) {                                        \
        nt = 0;                                                                \
        for (int r = 0; r < 3; r++) {                                          \
            const int64_t lo = (r == 0) ? j : i, hi = (r == 2) ? j : k;        \
            const int64_t f = (r == 0) ? i : (r == 1) ? j : k;                 \
            const double sg = (r == 1) ? -1.0 : 1.0;                           \
            const int64_t cP = ocls[lo] + ocls[hi], cf = ocls[f];              \
            const int64_t P = hi * no + lo;                                    \
            for (int64_t s = 0; s < ncls; s++) {                               \
                const int64_t h = g - s;                                       \
                if (h < 0 || h >= npg) continue;                               \
                const int64_t nd = vrng[s + 1] - vrng[s];                      \
                const int64_t nq = qoff[h + 1] - qoff[h];                      \
                if (nd == 0 || nq == 0) continue;                              \
                const T *Tb = Z + (i - i0) * H + zoff[s];                      \
                const int64_t cy = h - cf, cx = cP - s;                        \
                cc3_task_##SFX *q;                                             \
                if (cy >= 0 && cy < ncls && vrng[cy + 1] > vrng[cy]) {         \
                    q = tasks + nt++;                                          \
                    q->m = nd; q->n = vrng[cy + 1] - vrng[cy]; q->np = 1;      \
                    q->k[0] = nq; q->a[0] = Tb; q->lda[0] = nq;                \
                    q->b[0] = Gv + (f * nQ + qoff[h]) * nv + vrng[cy];         \
                    q->ldb[0] = nv;                                            \
                    q->c = Ysq + (P * nv + vrng[s]) * nv + vrng[cy];           \
                    q->ldc = nv; q->alpha = sg;                                \
                }                                                              \
                if (cx >= 0 && cx < ncls && orng[cx + 1] > orng[cx]) {         \
                    q = tasks + nt++;                                          \
                    q->m = orng[cx + 1] - orng[cx]; q->n = nq; q->np = 1;      \
                    q->k[0] = nd;                                              \
                    q->a[0] = Gsq + (P * no + orng[cx]) * nv + vrng[s];        \
                    q->lda[0] = nv; q->b[0] = Tb; q->ldb[0] = nq;              \
                    q->c = Xo + (orng[cx] * no + f) * nQ + qoff[h];            \
                    q->ldc = no * nQ; q->alpha = sg;                           \
                }                                                              \
                if (h == cP) {              /* r1[f][d] += s T[d, Q] gsq[P][Q] */ \
                    q = tasks + nt++;                                          \
                    q->m = nd; q->n = 1; q->np = 1;                            \
                    q->k[0] = nq; q->a[0] = Tb; q->lda[0] = nq;                \
                    q->b[0] = gsq + P * nQ + qoff[h]; q->ldb[0] = 1;           \
                    q->c = r1 + f * nv + vrng[s]; q->ldc = 1; q->alpha = sg;   \
                }                                                              \
                if (s == cf) {              /* rAsq[P][Q] += s fov[f][d] T[d, Q] */ \
                    q = tasks + nt++;                                          \
                    q->m = 1; q->n = nq; q->np = 1;                            \
                    q->k[0] = nd; q->a[0] = fov + f * nv + vrng[s];            \
                    q->lda[0] = nd; q->b[0] = Tb; q->ldb[0] = nq;              \
                    q->c = rAsq + P * nQ + qoff[h]; q->ldc = nq; q->alpha = sg; \
                }                                                              \
            }                                                                  \
        }                                                                      \
        cc3_run_##SFX(nt, tasks);                                              \
    }                                                                          \
    stats[3] += omp_get_wtime() - w3;                                          \
    free(tasks);                                                               \
    free(ocls);                                                                \
}

DEFINE_CC3(double, d)
DEFINE_CC3(double complex, z)
