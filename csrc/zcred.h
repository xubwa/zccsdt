/*
 * zcred.h: a sum of classes of the labels as a group index, as xzw of
 * xzdata.f90 does it: modulo `modulus`; with a second label (zc_gB > 0,
 * set by zc_set_red from xzsetmode) the two digits x1 + zc_gB x2 apart,
 * x1 modulo `modulus`, x2 modulo zc_gm2 (|x1| < zc_gB / 2 always).
 */
#ifndef ZCRED_H
#define ZCRED_H
extern int zc_gB, zc_gm2;
static inline int zc_mod(int k, int m) { return ((k % m) + m) % m; }
static inline int zc_red(int k, int modulus)
{
    if (zc_gB > 0) {
        int t = k + zc_gB / 2, x2 = t / zc_gB;
        if (t % zc_gB < 0) x2--;               /* floor */
        return zc_mod(k - zc_gB * x2, modulus) + zc_gB * zc_mod(x2, zc_gm2);
    }
    return zc_mod(k, modulus);
}
#endif
