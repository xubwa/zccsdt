/* CC3 without stored triples with label shifts (cfour/xzccsdt/xzcc3.inc;
 * docs/native.md, section 17): zc_cc3_batch of csrc/cc3_fly.c - which this
 * file repeats, see there for the layouts - for triples, operands and
 * contraction operands that carry a label shift, in classes:
 *
 *   st      of the triples: for the occupied triples of group g the virtual
 *           triples of group g + st (their amplitudes, tables and ev3)
 *   sa[z]   of the amplitudes of the drive set z: At[P][c][e] with class(c) +
 *           class(e) = class(P) + sa, Bo[m][i][Q] with group(Q) = class(m) +
 *           class(i) + sa
 *   sw[z]   of its operators: Ao[P][c][m] with class(c) + class(m) = class(P)
 *           + sw, Bv[e][i][Q] with group(Q) = class(e) + class(i) + sw;
 *           sa[z] + sw[z] = st for every set
 *   sc      of the contraction operands: Gv[m][Q][y] with class(y) + class(m)
 *           = group(Q) + sc, Gsq[P][x][d] with class(x) + class(d) = class(P)
 *           - sc, gsq[P][Q] with group(Q) = class(P) - sc, fov[m][d] with
 *           class(d) = class(m) - sc; the four sums have the shift st + sc
 *
 * All zero: zc_cc3_batch.  The folded EOM-CC3 sigma vector of a vector of
 * shift s: (st, sa, sw, sc) = (s, {0, s}, {s, 0}, 0) for the triples of the
 * vector, (0, {0}, {0}, s) for those of the ground state with H' = [H~, R1].
 * work: ni (3 H + nA) elements, H and nA those of the group g + st
 * (zc_cc3_work(ncls, g + st, ni, ..)); a batch whose g + st is no group
 * returns.  Only added to csrc/: the Python program does not call it.
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
static int64_t cc3s_layout(int64_t ncls, int64_t g, const int64_t *vrng,
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

#define MAXPROD 4

#define DEFINE_CC3S(T, SFX)                                                     \
typedef struct {                                                               \
    int64_t m, n, np;                                                          \
    int64_t k[MAXPROD], lda[MAXPROD], ldb[MAXPROD];                            \
    const T *a[MAXPROD], *b[MAXPROD];                                          \
    T *c; int64_t ldc; double alpha;                                           \
} cc3s_task_##SFX;                                                              \
                                                                               \
/* c += alpha sum_p a_p b_p for every task; a task is cut into pieces of      \
 * columns or of rows (whichever gives more pieces), a piece is done by one  \
 * thread.  The c of the tasks do not overlap. */                             \
static void cc3s_run_##SFX(int64_t ntask, const cc3s_task_##SFX *tasks)          \
{                                                                              \
    const int64_t cols = 96, rows = 16;                                        \
    int64_t total = 0;                                                         \
    for (int64_t t = 0; t < ntask; t++) {                                      \
        const cc3s_task_##SFX *q = tasks + t;                                   \
        if (q->m <= 0 || q->n <= 0 || q->np == 0) continue;                    \
        const int64_t pc = (q->n + cols - 1) / cols, pr = (q->m + rows - 1) / rows; \
        total += (pc >= pr) ? pc : pr;                                         \
    }                                                                          \
    if (total == 0) return;                                                    \
    int64_t *piece = malloc(sizeof(int64_t) * 4 * total);                      \
    int64_t np = 0;                                                            \
    for (int64_t t = 0; t < ntask; t++) {                                      \
        const cc3s_task_##SFX *q = tasks + t;                                   \
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
        const cc3s_task_##SFX *q = tasks + piece[4 * p];                        \
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
void zc_cc3_batchs_##SFX(                                                       \
    int64_t no, int64_t nv, int64_t nQ, int64_t nA, int64_t ncls,              \
    const int64_t *orng, const int64_t *vrng, const int64_t *qoff,             \
    const int64_t *aoff, const int64_t *vtri, const int8_t *vsgn,              \
    const int64_t *vfree, const int64_t *vpair,                                \
    const double *eo, const double *ev3, double omega,                         \
    int64_t i0, int64_t i1, int64_t j, int64_t k, int64_t nset,                \
    const int64_t *sa, const int64_t *sw, int64_t st, int64_t sc,              \
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
    const int64_t g = ci + cj + ck, gv = g + st;                               \
    if (gv < 0 || gv > 3 * ncls - 3) { free(ocls); return; }                   \
    const int64_t H = cc3s_layout(ncls, gv, vrng, qoff, zoff);                   \
    const int64_t A0 = aoff[gv], nAg = aoff[gv + 1] - aoff[gv];                   \
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
    cc3s_task_##SFX *tasks = malloc(sizeof(cc3s_task_##SFX) * 5 * 3 * ni * ncls); \
    int64_t nt = 0;                                                            \
    for (int64_t i = i0; i < i1; i++)                                          \
    for (int r = 0; r < 3; r++) {                                              \
        const int64_t lo = (r == 0) ? j : i, hi = (r == 2) ? j : k;            \
        const int64_t f = (r == 0) ? i : (r == 1) ? j : k;                     \
        const int64_t cP = ocls[lo] + ocls[hi];                                \
        for (int64_t s = 0; s < ncls; s++) {                                   \
            const int64_t h = gv - s;                                          \
            if (h < 0 || h >= npg) continue;                                   \
            cc3s_task_##SFX *q = tasks + nt;                                    \
            q->m = vrng[s + 1] - vrng[s];                                      \
            q->n = qoff[h + 1] - qoff[h];                                      \
            q->c = Z + (r * ni + (i - i0)) * H + zoff[s];                      \
            q->ldc = q->n;                                                     \
            q->alpha = -1.0;                                                   \
            q->np = 0;                                                         \
            for (int64_t z = 0; z < nset; z++) {                               \
                /* e of class cP + sa - s (At of shift sa), m of class cP + sw - s */ \
                const int64_t xe = cP + sa[z] - s, xm = cP + sw[z] - s;        \
                int64_t p;                                                     \
                if (xe >= 0 && xe < ncls) {                                    \
                p = q->np++;                                                   \
                q->k[p] = vrng[xe + 1] - vrng[xe];                             \
                q->a[p] = At[z] + ((hi * no + lo) * nv + vrng[s]) * nv + vrng[xe]; \
                q->lda[p] = nv;                                                \
                q->b[p] = Bv[z] + (vrng[xe] * no + f) * nQ + qoff[h];          \
                q->ldb[p] = no * nQ;                                           \
                }                                                              \
                if (xm >= 0 && xm < ncls) {                                    \
                p = q->np++;                                                   \
                q->k[p] = orng[xm + 1] - orng[xm];                             \
                q->a[p] = Ao[z] + ((hi * no + lo) * nv + vrng[s]) * no + orng[xm]; \
                q->lda[p] = no;                                                \
                q->b[p] = Bo[z] + (orng[xm] * no + f) * nQ + qoff[h];          \
                q->ldb[p] = no * nQ;                                           \
                }                                                              \
            }                                                                  \
            if (q->m > 0 && q->n > 0) nt++;                                    \
        }                                                                      \
    }                                                                          \
    cc3s_run_##SFX(nt, tasks);                                                  \
    double w1 = omp_get_wtime();                                               \
    stats[0] += w1 - w0;                                                       \
    /* ---- the amplitudes of the batch: t3[i][A - A0] */                      \
    int64_t *pos = malloc(sizeof(int64_t) * 3 * nAg);                          \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t A = 0; A < nAg; A++)                                          \
        for (int r = 0; r < 3; r++) {                                          \
            const int64_t a = vfree[r * nA + A0 + A], Q = vpair[r * nA + A0 + A]; \
            const int64_t s = vcls[a], h = gv - s;                             \
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
        const int64_t h = gv - s;                                              \
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
                const int64_t h = gv - s;                                      \
                if (h < 0 || h >= npg) continue;                               \
                const int64_t nd = vrng[s + 1] - vrng[s];                      \
                const int64_t nq = qoff[h + 1] - qoff[h];                      \
                if (nd == 0 || nq == 0) continue;                              \
                const T *Tb = Z + (i - i0) * H + zoff[s];                      \
                const int64_t cy = h - cf + sc, cx = cP - s - sc;              \
                cc3s_task_##SFX *q;                                             \
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
                if (h == cP - sc) {         /* r1[f][d] += s T[d, Q] gsq[P][Q] */ \
                    q = tasks + nt++;                                          \
                    q->m = nd; q->n = 1; q->np = 1;                            \
                    q->k[0] = nq; q->a[0] = Tb; q->lda[0] = nq;                \
                    q->b[0] = gsq + P * nQ + qoff[h]; q->ldb[0] = 1;           \
                    q->c = r1 + f * nv + vrng[s]; q->ldc = 1; q->alpha = sg;   \
                }                                                              \
                if (s == cf - sc) {         /* rAsq[P][Q] += s fov[f][d] T[d, Q] */ \
                    q = tasks + nt++;                                          \
                    q->m = 1; q->n = nq; q->np = 1;                            \
                    q->k[0] = nd; q->a[0] = fov + f * nv + vrng[s];            \
                    q->lda[0] = nd; q->b[0] = Tb; q->ldb[0] = nq;              \
                    q->c = rAsq + P * nQ + qoff[h]; q->ldc = nq; q->alpha = sg; \
                }                                                              \
            }                                                                  \
        }                                                                      \
        cc3s_run_##SFX(nt, tasks);                                              \
    }                                                                          \
    stats[3] += omp_get_wtime() - w3;                                          \
    free(tasks);                                                               \
    free(ocls);                                                                \
}

DEFINE_CC3S(double, d)
DEFINE_CC3S(double complex, z)
