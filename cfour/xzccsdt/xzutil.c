/*
 * xzutil.c -- what the Fortran scheduler of xzccsdt asks the system:
 * free disk space, the resident set of the process, the BLAS it is linked to.
 */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/statvfs.h>
#include <time.h>
#include <mkl.h>

/* bytes free for this user on the file system of `path' (-1: unknown) */
int64_t xz_free_disk(const char *path)
{
    struct statvfs s;
    if (statvfs(path, &s) != 0)
        return -1;
    return (int64_t)s.f_bavail * (int64_t)s.f_frsize;
}

/* kb[0]: peak resident set (VmHWM), kb[1]: resident set now (VmRSS), kB */
void xz_rss(int64_t *kb)
{
    char line[256];
    FILE *f = fopen("/proc/self/status", "r");
    kb[0] = kb[1] = -1;
    if (!f)
        return;
    while (fgets(line, sizeof line, f)) {
        long long v;
        if (sscanf(line, "VmHWM: %lld", &v) == 1) kb[0] = v;
        if (sscanf(line, "VmRSS: %lld", &v) == 1) kb[1] = v;
    }
    fclose(f);
}

/* threads of MKL, and 1 if this file was compiled for 8-byte BLAS integers
 * (the kernels are compiled with the same flags) */
void xz_blas_info(int64_t *nthreads, int64_t *ilp64)
{
    *nthreads = mkl_get_max_threads();
    *ilp64 = (sizeof(MKL_INT) == 8);
}

/* wall clock and the CPU time of all threads of the process, seconds.  (A
 * thread of the OpenMP runtime that waits for work spins for KMP_BLOCKTIME,
 * 200 ms by default, and that counts as CPU time.) */
void xz_clocks(double *wall, double *cpu)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    *wall = (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &t);
    *cpu = (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}
