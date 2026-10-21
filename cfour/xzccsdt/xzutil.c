/*
 * xzutil.c -- what the Fortran scheduler of xzccsdt asks the system:
 * free disk space, the resident set of the process, the BLAS it is linked to.
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/statvfs.h>
#include <sys/vfs.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <mkl.h>

/* bytes free for this user on the file system of `path' (-1: unknown) */
int64_t xz_free_disk(const char *path)
{
    struct statvfs s;
    if (statvfs(path, &s) != 0)
        return -1;
    return (int64_t)s.f_bavail * (int64_t)s.f_frsize;
}

/* the directory `path' and those above it, as mkdir -p (0: it is there) */
int64_t xz_mkdir(const char *path)
{
    char buf[4096];
    size_t n = strlen(path);
    if (n == 0 || n >= sizeof(buf)) return 1;
    memcpy(buf, path, n + 1);
    for (size_t i = 1; i < n; i++) {
        if (buf[i] != '/') continue;
        buf[i] = 0;
        mkdir(buf, 0755);
        buf[i] = '/';
    }
    mkdir(buf, 0755);
    struct stat st;
    return stat(buf, &st) == 0 && S_ISDIR(st.st_mode) ? 0 : 1;
}

/* the file `from' takes the name `to', replacing a file of that name (0: done) */
int64_t xz_rename(const char *from, const char *to)
{
    return rename(from, to) == 0 ? 0 : 1;
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

/* The standard output of the process to /dev/null (quiet = 1) and back
 * (quiet = 0): around calls of CFOUR routines that print debug lines for
 * every call.  The caller flushes its own unit before and after. */
void xz_stdout(const int64_t *quiet)
{
    static int saved = -1;
    fflush(stdout);
    if (*quiet && saved < 0) {
        int null = open("/dev/null", O_WRONLY);
        if (null < 0)
            return;
        saved = dup(1);
        if (saved >= 0)
            dup2(null, 1);
        close(null);
    } else if (!*quiet && saved >= 0) {
        dup2(saved, 1);
        close(saved);
        saved = -1;
    }
}

/* The file `path' mapped into the address space, read-only and shared: the
 * pages are those of the page cache of the node, one copy for all the
 * processes that map the file.  Returns the address of its first byte (0:
 * failed) and its length in *nbytes.  populate != 0: the page tables are
 * filled now (the file is read if the cache does not hold it), so that no
 * thread of a kernel waits for a page later. */
int64_t xz_map_file(const char *path, int64_t *nbytes, const int64_t *populate)
{
    struct stat st;
    void *p;
    int fd = open(path, O_RDONLY);
    *nbytes = 0;
    if (fd < 0)
        return 0;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        return 0;
    }
    p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED | (*populate ? MAP_POPULATE : 0), fd, 0);
    close(fd);
    if (p == MAP_FAILED)
        return 0;
    *nbytes = (int64_t)st.st_size;
    return (int64_t)(intptr_t)p;
}

/* 1 if the file system of `path' is one of the node (a disk or memory), 0 if
 * it is served over a network (NFS, Lustre, GPFS, BeeGFS, CIFS, Ceph, AFS,
 * a FUSE mount), -1 if that cannot be asked */
int64_t xz_fs_local(const char *path)
{
    struct statfs s;
    if (statfs(path, &s) != 0)
        return -1;
    switch ((unsigned long)s.f_type) {
    case 0x6969UL:        /* NFS */
    case 0x0BD00BD0UL:    /* Lustre */
    case 0x47504653UL:    /* GPFS */
    case 0x19830326UL:    /* BeeGFS */
    case 0xFF534D42UL:    /* CIFS */
    case 0xFE534D42UL:    /* SMB2 */
    case 0x517BUL:        /* SMB */
    case 0x00C36400UL:    /* Ceph */
    case 0x5346414FUL:    /* AFS */
    case 0x65735546UL:    /* FUSE */
        return 0;
    default:
        return 1;
    }
}

/* kB of the process: kb[0] peak resident set, kb[1] resident anonymous
 * memory (its own), kb[2] resident pages of mapped files (shared with the
 * other processes that map them), kb[3] resident shared memory */
void xz_rss4(int64_t *kb)
{
    char line[256];
    FILE *f = fopen("/proc/self/status", "r");
    kb[0] = kb[1] = kb[2] = kb[3] = -1;
    if (!f)
        return;
    while (fgets(line, sizeof line, f)) {
        long long v;
        if (sscanf(line, "VmHWM: %lld", &v) == 1) kb[0] = v;
        if (sscanf(line, "RssAnon: %lld", &v) == 1) kb[1] = v;
        if (sscanf(line, "RssFile: %lld", &v) == 1) kb[2] = v;
        if (sscanf(line, "RssShmem: %lld", &v) == 1) kb[3] = v;
    }
    fclose(f);
}

/* kB of memory that the node can still give (MemAvailable of /proc/meminfo; -1: unknown) */
int64_t xz_mem_avail(void)
{
    FILE *f = fopen("/proc/meminfo", "r");
    char line[256];
    long long kb = -1;
    if (!f) return -1;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "MemAvailable: %lld", &kb) == 1) break;
    fclose(f);
    return (int64_t)kb;
}
