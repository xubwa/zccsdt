/*
 * xzmpi.c -- the message passing of xzccsdt, behind a few C functions that
 * the Fortran scheduler calls: (Q) over ranks (sums of a few numbers), and
 * the CCSDT and Lambda iterations over ranks (T3 to the ranks, sums of the
 * parts of the residual: arrays of the size of T3, moved in chunks).
 *
 * All calls of MPI of the program are in this file, and they are plain
 * MPI-3 calls with int counts (no large-count routines, nothing of one
 * vendor): an array of n double words goes in messages of at most
 * xz_chunk words (2^23 = 64 MB by default, XZ_MPICHUNK words, at most 2^27
 * so that no count in bytes passes 2^31 - 1 either).
 *
 * Compiled with -DXZ_MPI (and an MPI compiler wrapper) these are calls of
 * MPI; without it the same functions describe one rank and the program is
 * the serial one.  The Fortran side never sees an MPI handle or an MPI
 * integer: no dependence on the integer kind of the MPI library.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* words (doubles) of one message of the chunked routines */
#define XZ_CHUNK_MAX ((int64_t)1 << 27)
static int64_t xz_chunk = (int64_t)1 << 23;

void xzmpi_set_chunk(int64_t n)
{
    if (n < 1) n = 1;
    if (n > XZ_CHUNK_MAX) n = XZ_CHUNK_MAX;
    xz_chunk = n;
}

int64_t xzmpi_chunk(void) { return xz_chunk; }

static void chunk_from_env(void)
{
    const char *e = getenv("XZ_MPICHUNK");
    if (e && *e) xzmpi_set_chunk(strtoll(e, NULL, 10));
}

#ifdef XZ_MPI
#include <mpi.h>

void xzmpi_init(void)
{
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
    chunk_from_env();
}

int64_t xzmpi_rank(void)
{
    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    return r;
}

int64_t xzmpi_size(void)
{
    int n;
    MPI_Comm_size(MPI_COMM_WORLD, &n);
    return n;
}

int64_t xzmpi_enabled(void) { return 1; }

/* n bytes from rank 0 to all ranks (in chunks: counts are C ints) */
void xzmpi_bcast(void *buf, int64_t nbytes)
{
    char *p = (char *)buf;
    while (nbytes > 0) {
        int n = nbytes > (1 << 30) ? (1 << 30) : (int)nbytes;
        MPI_Bcast(p, n, MPI_BYTE, 0, MPI_COMM_WORLD);
        p += n;
        nbytes -= n;
    }
}

/* out (on rank 0) = the sum of in over the ranks, n doubles */
void xzmpi_sum(const double *in, double *out, int64_t n)
{
    MPI_Reduce(in, out, (int)n, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}

/* out (on rank 0) = the largest of in over the ranks, n doubles */
void xzmpi_max(const double *in, double *out, int64_t n)
{
    MPI_Reduce(in, out, (int)n, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
}

/* buf[r * n ..] on rank 0 = the n doubles `in' of rank r */
void xzmpi_gather(const double *in, double *buf, int64_t n)
{
    MPI_Gather(in, (int)n, MPI_DOUBLE, buf, (int)n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void xzmpi_barrier(void) { MPI_Barrier(MPI_COMM_WORLD); }

/* ---- arrays of n double words, in messages of at most xz_chunk words ---- */

/* buf of rank 0 to all ranks */
void xzmpi_bcast_w(double *buf, int64_t n)
{
    for (int64_t off = 0; off < n; off += xz_chunk) {
        int m = (int)(n - off < xz_chunk ? n - off : xz_chunk);
        MPI_Bcast(buf + off, m, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
}

/* buf to rank dest / from rank src (the two sides chunk alike) */
void xzmpi_send_w(const double *buf, int64_t n, int64_t dest, int64_t tag)
{
    for (int64_t off = 0; off < n; off += xz_chunk) {
        int m = (int)(n - off < xz_chunk ? n - off : xz_chunk);
        MPI_Send(buf + off, m, MPI_DOUBLE, (int)dest, (int)tag, MPI_COMM_WORLD);
    }
}

void xzmpi_recv_w(double *buf, int64_t n, int64_t src, int64_t tag)
{
    for (int64_t off = 0; off < n; off += xz_chunk) {
        int m = (int)(n - off < xz_chunk ? n - off : xz_chunk);
        MPI_Recv(buf + off, m, MPI_DOUBLE, (int)src, (int)tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}

/* buf of rank 0 = the sum of buf over the ranks, in place (the buf of the
 * other ranks is left as it is) */
void xzmpi_sum0_w(double *buf, int64_t n)
{
    int r;
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    for (int64_t off = 0; off < n; off += xz_chunk) {
        int m = (int)(n - off < xz_chunk ? n - off : xz_chunk);
        if (r == 0)
            MPI_Reduce(MPI_IN_PLACE, buf + off, m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        else
            MPI_Reduce(buf + off, NULL, m, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    }
}

/* buf of every rank = the sum of buf over the ranks, in place */
void xzmpi_allsum_w(double *buf, int64_t n)
{
    for (int64_t off = 0; off < n; off += xz_chunk) {
        int m = (int)(n - off < xz_chunk ? n - off : xz_chunk);
        MPI_Allreduce(MPI_IN_PLACE, buf + off, m, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
}

/* The name of the node of this rank (for the output), at most len - 1 characters. */
void xzmpi_node(char *name, int64_t len)
{
    char buf[MPI_MAX_PROCESSOR_NAME + 1];
    int n = 0;
    MPI_Get_processor_name(buf, &n);
    if (n > len - 1) n = (int)len - 1;
    memcpy(name, buf, (size_t)n);
    name[n] = 0;
}

void xzmpi_finalize(void) { MPI_Finalize(); }

void xzmpi_abort(int64_t code) { MPI_Abort(MPI_COMM_WORLD, (int)code); }

#else

void xzmpi_init(void) { chunk_from_env(); }
int64_t xzmpi_rank(void) { return 0; }
int64_t xzmpi_size(void) { return 1; }
int64_t xzmpi_enabled(void) { return 0; }
void xzmpi_bcast(void *buf, int64_t nbytes) { (void)buf; (void)nbytes; }
void xzmpi_sum(const double *in, double *out, int64_t n) { memcpy(out, in, sizeof(double) * (size_t)n); }
void xzmpi_max(const double *in, double *out, int64_t n) { memcpy(out, in, sizeof(double) * (size_t)n); }
void xzmpi_gather(const double *in, double *buf, int64_t n) { memcpy(buf, in, sizeof(double) * (size_t)n); }
void xzmpi_barrier(void) {}
void xzmpi_bcast_w(double *buf, int64_t n) { (void)buf; (void)n; }
void xzmpi_send_w(const double *buf, int64_t n, int64_t dest, int64_t tag) { (void)buf; (void)n; (void)dest; (void)tag; }
void xzmpi_recv_w(double *buf, int64_t n, int64_t src, int64_t tag) { (void)buf; (void)n; (void)src; (void)tag; }
void xzmpi_sum0_w(double *buf, int64_t n) { (void)buf; (void)n; }
void xzmpi_allsum_w(double *buf, int64_t n) { (void)buf; (void)n; }
void xzmpi_node(char *name, int64_t len) { if (len > 0) name[0] = 0; }
void xzmpi_finalize(void) {}
void xzmpi_abort(int64_t code) { exit((int)code); }

#endif

/* ---- A test of the chunked routines (XZ_MPITEST=n words, with the chunk
 * that XZ_MPICHUNK gives): patterns that depend on the rank and on the
 * index go through the broadcast, a ring of sends and receives, the sum
 * onto rank 0 and the sum on all ranks; returns the number of wrong words
 * on this rank (0 without MPI and with one rank as well). */
int64_t xzmpi_selftest(int64_t n)
{
    int64_t me = xzmpi_rank(), np = xzmpi_size(), bad = 0;
    double *a = malloc(sizeof(double) * (size_t)(2 * n + 2));
    double *b = a + n + 1;
    if (!a) return -1;
    /* broadcast */
    for (int64_t i = 0; i < n; i++) a[i] = me == 0 ? 1.0 + (double)(i % 1000003) : -1.0;
    xzmpi_bcast_w(a, n);
    for (int64_t i = 0; i < n; i++) bad += a[i] != 1.0 + (double)(i % 1000003);
    /* ring: every rank sends to the next one */
    if (np > 1) {
        for (int64_t i = 0; i < n; i++) a[i] = (double)(me * 7 + 1) + 0.5 * (double)(i % 4099);
        const int64_t to = (me + 1) % np, from = (me + np - 1) % np;
        /* rank 0 sends first, then every rank receives and passes its own on */
        if (me == 0) {
            xzmpi_send_w(a, n, to, 11);
            xzmpi_recv_w(b, n, from, 11);
        } else {
            xzmpi_recv_w(b, n, from, 11);
            xzmpi_send_w(a, n, to, 11);
        }
        for (int64_t i = 0; i < n; i++)
            bad += b[i] != (double)(from * 7 + 1) + 0.5 * (double)(i % 4099);
    }
    /* sum onto rank 0: sum_r (r + 1) (i mod 977) */
    for (int64_t i = 0; i < n; i++) a[i] = (double)(me + 1) * (double)(i % 977);
    xzmpi_sum0_w(a, n);
    {
        const double f = me == 0 ? 0.5 * (double)(np * (np + 1)) : (double)(me + 1);
        for (int64_t i = 0; i < n; i++) bad += a[i] != f * (double)(i % 977);
    }
    /* sum on all ranks */
    for (int64_t i = 0; i < n; i++) a[i] = (double)(me + 1) * (double)(i % 1013);
    xzmpi_allsum_w(a, n);
    for (int64_t i = 0; i < n; i++) bad += a[i] != 0.5 * (double)(np * (np + 1)) * (double)(i % 1013);
    free(a);
    return bad;
}
