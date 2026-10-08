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
 *
 * zc_qlg_sector_vectors:  the same sum backwards, for the first derivatives
 * of (Q)_Lambda in symmetry sectors (zccsdt/cc/quadruples_lambda_grad_sym.py),
 * as zc_qg_sector_vectors of csrc/tnpack.c is for (Q).  For every virtual
 * quadruple of the sector, with D = eo - ev,
 *
 *     y = (r1 + r2) / D,   w = (z2 + z3) / D
 *
 * are written, with the signs of the splits, to the places of the blocks
 * that the splits of the quadruple were read from (the other places are not
 * touched: the caller sets the blocks to zero) - nothing is conjugated:
 *
 *     e22y = y,  e22w = w      on two pairs
 *     e13w = w,  e13y = y      on an index and a triple
 *
 * e[0] += sum Re(w y) D (E(Q)_L of the sector), e[1] += sum Re(w y),
 * e[2] += sum Im(w y) D.
 */
#include <complex.h>
#include <stdint.h>

#define ZC_RE_d(x) (x)
#define ZC_RE_z(x) creal(x)
#define ZC_IM_d(x) (0.0)
#define ZC_IM_z(x) cimag(x)

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
}                                                                              \
                                                                               \
void zc_qlg_sector_vectors_##SFX(int n, const int *kv, int modulus, int M,     \
        const double *ev, double eo, int kmin, int nk, const int *lid_of,      \
        const int *lstart, const int *lmem, int qmin, int nq,                  \
        const int64_t *off22, const int64_t *nc22, const int64_t *off13,       \
        const int64_t *nc13, const int32_t *posV, const int32_t *posQ,         \
        const int32_t *pos3, const T *b1, const T *b2, const T *b3,            \
        const T *b4, const T *c1, const T *c2, const T *c3, T *e22y, T *e22w,  \
        T *e13w, T *e13y, double *e)                                           \
{                                                                              \
    const int64_t nQ = (int64_t)n * (n - 1) / 2;                               \
    const int64_t nA3 = nQ * (n - 2) / 3;                                      \
    double e0 = 0, e1 = 0, e2 = 0;                                             \
    _Pragma("omp parallel for schedule(dynamic, 1) reduction(+:e0, e1, e2)")   \
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
            int64_t x22[6], x13[4];                                            \
            T r = 0, z = 0;                                                    \
            for (int s = 0; s < 6; s++) {                                      \
                const int p = v[g22[s][0]], q = v[g22[s][1]];                  \
                const int r_ = v[g22[s][2]], u = v[g22[s][3]];                 \
                int kq = kv[p] + kv[q];                                        \
                x22[s] = -1;                                                   \
                if (modulus) kq = ((kq % modulus) + modulus) % modulus;        \
                if (kq < qmin || kq >= qmin + nq || off22[kq - qmin] < 0)      \
                    continue;                                                  \
                const int64_t rk1 = nQ - 1 - (int64_t)(n - 1 - p) * (n - 2 - p) / 2 - (n - 1 - q);   \
                const int64_t rk2 = nQ - 1 - (int64_t)(n - 1 - r_) * (n - 2 - r_) / 2 - (n - 1 - u); \
                const int64_t x = off22[kq - qmin] + posQ[rk1] * nc22[kq - qmin] + posQ[rk2];        \
                x22[s] = x;                                                    \
                r += s22[s] * (b1[x] + b2[x]);                                 \
                z += s22[s] * (b3[x] + b4[x]);                                 \
            }                                                                  \
            for (int s = 0; s < 4; s++) {                                      \
                const int p = v[s];                                            \
                const int q = v[s == 0], r_ = v[s <= 1 ? 2 : 1], u = v[s == 3 ? 2 : 3];            \
                const int lp = lid_of[kv[p] - kmin];                           \
                x13[s] = -1;                                                   \
                if (off13[lp] < 0) continue;                                   \
                const int64_t y1 = n - 1 - q, y2 = n - 1 - r_;                 \
                const int64_t rk = nA3 - 1 - y1 * (y1 - 1) * (y1 - 2) / 6 - y2 * (y2 - 1) / 2 - (n - 1 - u); \
                const int64_t x = off13[lp] + posV[p] * nc13[lp] + pos3[rk];   \
                const double sg = (s & 1) ? -1.0 : 1.0;                        \
                x13[s] = x;                                                    \
                r += sg * (c1[x] + c2[x]);                                     \
                z += sg * c3[x];                                               \
            }                                                                  \
            const double D = eo - ev[v[0]] - ev[v[1]] - ev[v[2]] - ev[v[3]];   \
            const T y = r / D, w = z / D;                                      \
            const T wy = w * y;                                                \
            e0 += ZC_RE_##SFX(wy) * D;                                         \
            e1 += ZC_RE_##SFX(wy);                                             \
            e2 += ZC_IM_##SFX(wy) * D;                                         \
            for (int s = 0; s < 6; s++) {                                      \
                if (x22[s] < 0) continue;                                      \
                e22y[x22[s]] = s22[s] * y;                                     \
                e22w[x22[s]] = s22[s] * w;                                     \
            }                                                                  \
            for (int s = 0; s < 4; s++) {                                      \
                if (x13[s] < 0) continue;                                      \
                const double sg = (s & 1) ? -1.0 : 1.0;                        \
                e13w[x13[s]] = sg * w;                                         \
                e13y[x13[s]] = sg * y;                                         \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    e[0] += e0;                                                                \
    e[1] += e1;                                                                \
    e[2] += e2;                                                                \
}

DEFINE_QL(double, d)
DEFINE_QL(double complex, z)
