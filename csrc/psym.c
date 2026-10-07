/*
 * (P) in symmetry sectors (zccsdt/cc/pentuples_sym.py).
 *
 * Every virtual spinor has an integer label kv that adds (modulo `modulus`,
 * or as it is for 0).  The sorted tuples of range(n) are numbered
 * lexicographically, as in tnpack.c; the classes of the spinors, pairs,
 * triples and quadruples by label are given by the position of each inside
 * its class (posV, posQ, pos3, pos4).
 *
 * zc_p_expand13:  out[e, t] = t(e, abc) of one packed row of quadruples, for
 *     the e of a list and the sorted triples (abc) of a list - the sign of
 *     e at its place among a<b<c, zero where e is one of them.
 * zc_p_pick:      out[r, c] = sg[r] * M[rows[r], cols[c]].
 * zc_p_sector_energy:  the sum over the virtual pentuples a<b<c<d<e of total
 *     label M of Re(conj(z) r) / (eo - ev), r and z each the sum over the ten
 *     splits (xy | rest) of a block X[(de), (abc)] and over the five splits
 *     (x | rest) of a block X[e, (abcd)]: one block per label q of the pair
 *     at off23[q - qmin] with nc23 columns, one per label of the single
 *     index at off14[its class] with nc14 columns.
 * zc_p_xq_sector:  the T2 T2 <vo||vv> part of the intermediate of three T2:
 *     out[o, t] += scale * sum over the six orders (a, b, c) of the sorted
 *     triple t of sgn * Z[a, o, c, b], Z in blocks by the classes of a and
 *     b: block (la, lb) at off[la * nl + lb], [a, o, c, b] with nc and nb
 *     members of the classes of c and b.
 */
#include <complex.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

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
void zc_p_expand13_##SFX(const T *row, int n, const int32_t *es, int ne,       \
        const int32_t *tri, int64_t nt, T *out)                                \
{                                                                              \
    int64_t *cb = binomials(n);                                                \
    const int64_t *c2 = cb, *c3 = cb + n + 1, *c4 = cb + 2 * (n + 1);          \
    const int64_t top = c4[n] - 1;                                             \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t t = 0; t < nt; t++) {                                         \
        const int a = tri[3 * t], b = tri[3 * t + 1], c = tri[3 * t + 2];      \
        for (int x = 0; x < ne; x++) {                                         \
            const int e = es[x];                                               \
            int v[4];                                                          \
            double sg;                                                         \
            if (e == a || e == b || e == c) { out[x * nt + t] = 0; continue; } \
            if (e < a)      { v[0] = e; v[1] = a; v[2] = b; v[3] = c; sg = 1; }  \
            else if (e < b) { v[0] = a; v[1] = e; v[2] = b; v[3] = c; sg = -1; } \
            else if (e < c) { v[0] = a; v[1] = b; v[2] = e; v[3] = c; sg = 1; }  \
            else            { v[0] = a; v[1] = b; v[2] = c; v[3] = e; sg = -1; } \
            out[x * nt + t] = sg * row[top - c4[n - 1 - v[0]] - c3[n - 1 - v[1]] \
                                       - c2[n - 1 - v[2]] - (n - 1 - v[3])];   \
        }                                                                      \
    }                                                                          \
    free(cb);                                                                  \
}                                                                              \
                                                                               \
void zc_p_pick_##SFX(const T *M, int64_t ld, const int64_t *rows,              \
        const double *sg, int64_t nr, const int64_t *cols, int64_t nc, T *out) \
{                                                                              \
    _Pragma("omp parallel for schedule(static) collapse(2)")                   \
    for (int64_t r = 0; r < nr; r++)                                           \
    for (int64_t c = 0; c < nc; c++)                                           \
        out[r * nc + c] = sg[r] * M[rows[r] * ld + cols[c]];                   \
}                                                                              \
                                                                               \
void zc_p_sector_energy_##SFX(int n, const int *kv, int modulus, int M,        \
        const double *ev, double eo, int kmin, int nk, const int *lid_of,      \
        const int *lstart, const int *lmem, int qmin, int nq,                  \
        const int64_t *off23, const int64_t *nc23, const int64_t *off14,       \
        const int64_t *nc14, const int32_t *posV, const int32_t *posQ,         \
        const int32_t *pos3, const int32_t *pos4, const T *r23, const T *z23,  \
        const T *r14, const T *z14, double *e)                                 \
{                                                                              \
    int64_t *cb = binomials(n);                                                \
    const int64_t *c2 = cb, *c3 = cb + n + 1, *c4 = cb + 2 * (n + 1);          \
    const int64_t t2 = c2[n] - 1, t3 = c3[n] - 1, t4 = c4[n] - 1;              \
    /* the ten pairs of five places with the three that are left, and the     \
     * four that are left of one */                                            \
    static const int g23[10][5] = {{0,1,2,3,4}, {0,2,1,3,4}, {0,3,1,2,4},      \
        {0,4,1,2,3}, {1,2,0,3,4}, {1,3,0,2,4}, {1,4,0,2,3}, {2,3,0,1,4},       \
        {2,4,0,1,3}, {3,4,0,1,2}};                                             \
    static const double s23[10] = {1, -1, 1, -1, 1, -1, 1, 1, -1, 1};          \
    static const int g14[5][5] = {{0,1,2,3,4}, {1,0,2,3,4}, {2,0,1,3,4},       \
        {3,0,1,2,4}, {4,0,1,2,3}};                                             \
    double esum = 0;                                                           \
    _Pragma("omp parallel for schedule(dynamic, 1) reduction(+:esum)")         \
    for (int a = 0; a < n - 4; a++)                                            \
    for (int b = a + 1; b < n - 3; b++)                                        \
    for (int c = b + 1; c < n - 2; c++)                                        \
    for (int d = c + 1; d < n - 1; d++) {                                      \
        const int ke = red(M - kv[a] - kv[b] - kv[c] - kv[d], modulus);        \
        if (ke < kmin || ke >= kmin + nk || lid_of[ke - kmin] < 0) continue;   \
        const int l = lid_of[ke - kmin];                                       \
        int lo = lstart[l], hi = lstart[l + 1];                                \
        while (lo < hi) {               /* the first member above d */         \
            int mid = (lo + hi) / 2;                                           \
            if (lmem[mid] > d) hi = mid; else lo = mid + 1;                    \
        }                                                                      \
        const double eabcd = eo - ev[a] - ev[b] - ev[c] - ev[d];               \
        for (int t = lo; t < lstart[l + 1]; t++) {                             \
            const int v[5] = {a, b, c, d, lmem[t]};                            \
            T r = 0, z = 0;                                                    \
            for (int s = 0; s < 10; s++) {                                     \
                const int p = v[g23[s][0]], q = v[g23[s][1]];                  \
                const int kq = red(kv[p] + kv[q], modulus) - qmin;             \
                if (kq < 0 || kq >= nq || off23[kq] < 0) continue;             \
                const int x = v[g23[s][2]], y = v[g23[s][3]], w = v[g23[s][4]];\
                const int64_t rk2 = t2 - c2[n - 1 - p] - (n - 1 - q);          \
                const int64_t rk3 = t3 - c3[n - 1 - x] - c2[n - 1 - y] - (n - 1 - w); \
                const int64_t at = off23[kq] + posQ[rk2] * nc23[kq] + pos3[rk3];      \
                r += s23[s] * r23[at];                                         \
                z += s23[s] * z23[at];                                         \
            }                                                                  \
            for (int s = 0; s < 5; s++) {                                      \
                const int p = v[g14[s][0]];                                    \
                const int lp = lid_of[kv[p] - kmin];                           \
                if (off14[lp] < 0) continue;                                   \
                const int x = v[g14[s][1]], y = v[g14[s][2]];                  \
                const int w = v[g14[s][3]], u = v[g14[s][4]];                  \
                const int64_t rk4 = t4 - c4[n - 1 - x] - c3[n - 1 - y] - c2[n - 1 - w] - (n - 1 - u); \
                const int64_t at = off14[lp] + posV[p] * nc14[lp] + pos4[rk4]; \
                const double sg = (s & 1) ? -1.0 : 1.0;                        \
                r += sg * r14[at];                                             \
                z += sg * z14[at];                                             \
            }                                                                  \
            esum += ZC_RE_##SFX(ZC_CONJ_##SFX(z) * r) / (eabcd - ev[v[4]]);    \
        }                                                                      \
    }                                                                          \
    e[0] += esum;                                                              \
    free(cb);                                                                  \
}                                                                              \
                                                                               \
void zc_p_xq_sector_##SFX(int n, const int *kv, int kmin, const int *lid_of,   \
        int nl, const int32_t *tri, int64_t nt, int no, const int64_t *off,    \
        const int64_t *ncs, const int64_t *nbs, const int32_t *posV,           \
        const T *Z, double scale, T *out)                                      \
{                                                                              \
    static const int g[6][3] = {{0,1,2}, {0,2,1}, {1,0,2}, {1,2,0}, {2,0,1}, {2,1,0}}; \
    static const double sg[6] = {1, -1, -1, 1, 1, -1};                         \
    _Pragma("omp parallel for schedule(static)")                               \
    for (int64_t t = 0; t < nt; t++) {                                         \
        const int v[3] = {tri[3 * t], tri[3 * t + 1], tri[3 * t + 2]};         \
        for (int s = 0; s < 6; s++) {                                          \
            const int a = v[g[s][0]], b = v[g[s][1]], c = v[g[s][2]];          \
            const int64_t k = (int64_t)lid_of[kv[a] - kmin] * nl + lid_of[kv[b] - kmin]; \
            if (off[k] < 0) continue;                                          \
            const int64_t nc = ncs[k], nb = nbs[k];                            \
            const T *z = Z + off[k] + posV[a] * no * nc * nb + posV[c] * nb + posV[b]; \
            const double f = scale * sg[s];                                    \
            for (int o = 0; o < no; o++) out[o * nt + t] += f * z[o * nc * nb]; \
        }                                                                      \
    }                                                                          \
}

DEFINE_PSEC(double, d)
DEFINE_PSEC(double complex, z)
