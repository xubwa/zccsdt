/*
 * The energies of CCSDT(Q)_Lambda (zccsdt/cc/quadruples_lambda.py,
 * quadruples_lambda_sym.py), double and complex:
 *
 *     E[Q]_L = sum Re(z2 r) / D,   E(Q)_L - E[Q]_L = sum Re(z3 r) / D,
 *
 * r = r1 + r2 the right-hand side of (Q) (r1 of [V,T3], r2 of the T2 T2
 * terms), z2 = <0| Lambda_2 V |Q>, z3 = <0| Lambda_3 V |Q>: the left-hand
 * side is not conjugated, and it is not the right-hand side of anything.
 *
 * zc_ql_energy:         over the packed virtual quadruples of one occupied
 *                       quadruple.
 * zc_ql_sector_energy:  zc_q_sector_energy of csrc/tnpack.c (the same
 *                       arguments and blocks) with the blocks of Lambda_3:
 *
 *     r1 = b1 (22) + c1 (13),  r2 = b2 (22) + c2 (13),
 *     z2 = b3 (22),            z3 = b4 (22) + c3 (13).
 */
#include <complex.h>
#include <stdint.h>

#define ZC_RE_d(x) (x)
#define ZC_RE_z(x) creal(x)

#define DEFINE_QL(T, SFX)                                                      \
void zc_ql_energy_##SFX(const T *z2, const T *z3, const T *r1, const T *r2,    \
                        const double *ev, double eo, int64_t n, double *e)     \
{                                                                              \
    double e0 = 0, e1 = 0;                                                     \
    _Pragma("omp parallel for schedule(static) reduction(+:e0, e1)")           \
    for (int64_t A = 0; A < n; A++) {                                          \
        T r = (r1[A] + r2[A]) / (eo - ev[A]);                                  \
        e0 += ZC_RE_##SFX(z2[A] * r);                                          \
        e1 += ZC_RE_##SFX(z3[A] * r);                                          \
    }                                                                          \
    e[0] += e0;                                                                \
    e[1] += e1;                                                                \
}                                                                              \
                                                                               \
void zc_ql_sector_energy_##SFX(int n, const int *kv, int modulus, int M,       \
        const double *ev, double eo, int kmin, int nk, const int *lid_of,      \
        const int *lstart, const int *lmem, int qmin, int nq,                  \
        const int64_t *off22, const int64_t *nc22, const int64_t *off13,       \
        const int64_t *nc13, const int32_t *posV, const int32_t *posQ,         \
        const int32_t *pos3, const T *b1, const T *b2, const T *b3,            \
        const T *b4, const T *c1, const T *c2, const T *c3, double *e)         \
{                                                                              \
    const int64_t nQ = (int64_t)n * (n - 1) / 2;                               \
    const int64_t nA3 = nQ * (n - 2) / 3;                                      \
    double e0 = 0, e1 = 0;                                                     \
    _Pragma("omp parallel for schedule(dynamic, 1) reduction(+:e0, e1)")       \
    for (int a = 0; a < n - 3; a++)                                            \
    for (int b = a + 1; b < n - 2; b++)                                        \
    for (int c = b + 1; c < n - 1; c++) {                                      \
        int kd = M - kv[a] - kv[b] - kv[c];                                    \
        if (modulus) kd = ((kd % modulus) + modulus) % modulus;                \
        if (kd < kmin || kd >= kmin + nk || lid_of[kd - kmin] < 0) continue;   \
        const int l = lid_of[kd - kmin];                                       \
        int lo = lstart[l], hi = lstart[l + 1];                                \
        while (lo < hi) {               /* the first member above c */         \
            int mid = (lo + hi) / 2;                                           \
            if (lmem[mid] > c) hi = mid; else lo = mid + 1;                    \
        }                                                                      \
        for (int t = lo; t < lstart[l + 1]; t++) {                             \
            const int v[4] = {a, b, c, lmem[t]};                               \
            static const int g22[6][4] = {{0,1,2,3}, {0,2,1,3}, {0,3,1,2},     \
                                          {1,2,0,3}, {1,3,0,2}, {2,3,0,1}};    \
            static const double s22[6] = {1, -1, 1, 1, -1, 1};                 \
            T r1 = 0, r2 = 0, z2 = 0, z3 = 0;                                  \
            for (int s = 0; s < 6; s++) {                                      \
                const int p = v[g22[s][0]], q = v[g22[s][1]];                  \
                const int r = v[g22[s][2]], u = v[g22[s][3]];                  \
                int kq = kv[p] + kv[q];                                        \
                if (modulus) kq = ((kq % modulus) + modulus) % modulus;        \
                if (kq < qmin || kq >= qmin + nq || off22[kq - qmin] < 0)      \
                    continue;                                                  \
                const int64_t rk1 = nQ - 1 - (int64_t)(n - 1 - p) * (n - 2 - p) / 2 - (n - 1 - q); \
                const int64_t rk2 = nQ - 1 - (int64_t)(n - 1 - r) * (n - 2 - r) / 2 - (n - 1 - u); \
                const int64_t x = off22[kq - qmin] + posQ[rk1] * nc22[kq - qmin] + posQ[rk2];      \
                r1 += s22[s] * b1[x];                                          \
                r2 += s22[s] * b2[x];                                          \
                z2 += s22[s] * b3[x];                                          \
                z3 += s22[s] * b4[x];                                          \
            }                                                                  \
            for (int s = 0; s < 4; s++) {                                      \
                const int p = v[s];                                            \
                const int q = v[s == 0], r = v[s <= 1 ? 2 : 1], u = v[s == 3 ? 2 : 3];             \
                const int lp = lid_of[kv[p] - kmin];                           \
                if (off13[lp] < 0) continue;                                   \
                const int64_t y1 = n - 1 - q, y2 = n - 1 - r;                  \
                const int64_t rk = nA3 - 1 - y1 * (y1 - 1) * (y1 - 2) / 6 - y2 * (y2 - 1) / 2 - (n - 1 - u); \
                const int64_t x = off13[lp] + posV[p] * nc13[lp] + pos3[rk];   \
                const double sg = (s & 1) ? -1.0 : 1.0;                        \
                r1 += sg * c1[x];                                              \
                r2 += sg * c2[x];                                              \
                z3 += sg * c3[x];                                              \
            }                                                                  \
            T rr = (r1 + r2) / (eo - ev[v[0]] - ev[v[1]] - ev[v[2]] - ev[v[3]]);                   \
            e0 += ZC_RE_##SFX(z2 * rr);                                        \
            e1 += ZC_RE_##SFX(z3 * rr);                                        \
        }                                                                      \
    }                                                                          \
    e[0] += e0;                                                                \
    e[1] += e1;                                                                \
}

DEFINE_QL(double, d)
DEFINE_QL(double complex, z)
