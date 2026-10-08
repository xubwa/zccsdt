/*
 * xzmpi.c -- the message passing of xzccsdt, behind a few C functions that
 * the Fortran scheduler calls: the (Q) correction is parallel over ranks,
 * nothing else is.
 *
 * Compiled with -DXZ_MPI (and an MPI compiler wrapper) these are calls of
 * MPI; without it the same functions describe one rank and the program is
 * the serial one.  The Fortran side never sees an MPI handle or an MPI
 * integer: no dependence on the integer kind of the MPI library.
 */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef XZ_MPI
#include <mpi.h>

void xzmpi_init(void)
{
    int provided;
    MPI_Init_thread(NULL, NULL, MPI_THREAD_FUNNELED, &provided);
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

void xzmpi_finalize(void) { MPI_Finalize(); }

void xzmpi_abort(int64_t code) { MPI_Abort(MPI_COMM_WORLD, (int)code); }

#else

void xzmpi_init(void) {}
int64_t xzmpi_rank(void) { return 0; }
int64_t xzmpi_size(void) { return 1; }
int64_t xzmpi_enabled(void) { return 0; }
void xzmpi_bcast(void *buf, int64_t nbytes) { (void)buf; (void)nbytes; }
void xzmpi_sum(const double *in, double *out, int64_t n) { memcpy(out, in, sizeof(double) * (size_t)n); }
void xzmpi_max(const double *in, double *out, int64_t n) { memcpy(out, in, sizeof(double) * (size_t)n); }
void xzmpi_gather(const double *in, double *buf, int64_t n) { memcpy(buf, in, sizeof(double) * (size_t)n); }
void xzmpi_barrier(void) {}
void xzmpi_finalize(void) {}
void xzmpi_abort(int64_t code) { exit((int)code); }

#endif
