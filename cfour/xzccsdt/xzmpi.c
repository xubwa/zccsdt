/*
 * xzmpi.c -- the message passing of xzccsdt, behind a few C functions that
 * the Fortran scheduler calls: (Q) over ranks (sums of a few numbers), and
 * the CCSDT and Lambda iterations over ranks (T3 to the ranks, sums of the
 * parts of the residual: arrays of the size of T3, moved in chunks).
 *
 * All calls of MPI of the program are in this file, and they are plain
 * MPI-3 calls with int counts (no large-count routines, nothing of one
 * vendor): an array of n double words goes in messages of at most
 * xz_chunk words (2^23 = 64 MB by default, the option mpichunk: words, at most 2^27
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

/* ---- Slices: the array buf is cut into one range of words per rank,
 * off[r] <= word < off[r + 1] the slice of rank r.  Both exchanges go in
 * nrank - 1 rounds; in round d every rank sends to rank + d and receives
 * from rank - d (MPI_Sendrecv, chunk by chunk, every rank the same number
 * of chunks per round: the longest slice), so all ranks transfer at once
 * and no rank relays what belongs to another.
 *
 * xzmpi_slices_add: every rank has added its own contributions to all of
 * buf; on return the slice of this rank is the sum over the ranks (the rest
 * of buf is left as it is).  tmp: work, at least min(chunk, longest slice)
 * words. */
void xzmpi_slices_add(double *buf, const int64_t *off, double *tmp)
{
    int me, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    int64_t longest = 0;
    for (int r = 0; r < np; r++)
        if (off[r + 1] - off[r] > longest) longest = off[r + 1] - off[r];
    const int64_t mine = off[me + 1] - off[me];
    for (int d = 1; d < np; d++) {
        const int to = (me + d) % np, from = (me + np - d) % np;
        const int64_t ns = off[to + 1] - off[to];
        for (int64_t o = 0; o < longest; o += xz_chunk) {
            int64_t s = ns - o, r = mine - o;
            if (s > xz_chunk) s = xz_chunk;
            if (r > xz_chunk) r = xz_chunk;
            if (s < 0) s = 0;
            if (r < 0) r = 0;
            MPI_Sendrecv(buf + off[to] + (s ? o : 0), (int)s, MPI_DOUBLE, to, 21,
                         tmp, (int)r, MPI_DOUBLE, from, 21, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            double *x = buf + off[me] + o;
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < r; i++) x[i] += tmp[i];
        }
    }
}

/* xzmpi_slices_put: every rank has its own slice of buf; on return every
 * rank has all of buf. */
void xzmpi_slices_put(double *buf, const int64_t *off)
{
    int me, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    int64_t longest = 0;
    for (int r = 0; r < np; r++)
        if (off[r + 1] - off[r] > longest) longest = off[r + 1] - off[r];
    const int64_t mine = off[me + 1] - off[me];
    for (int d = 1; d < np; d++) {
        const int to = (me + d) % np, from = (me + np - d) % np;
        const int64_t nr = off[from + 1] - off[from];
        for (int64_t o = 0; o < longest; o += xz_chunk) {
            int64_t s = mine - o, r = nr - o;
            if (s > xz_chunk) s = xz_chunk;
            if (r > xz_chunk) r = xz_chunk;
            if (s < 0) s = 0;
            if (r < 0) r = 0;
            MPI_Sendrecv(buf + off[me] + (s ? o : 0), (int)s, MPI_DOUBLE, to, 22,
                         buf + off[from] + (r ? o : 0), (int)r, MPI_DOUBLE, from, 22,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
}

/* ---- Segments: pieces of an array between pairs of ranks, all at once (the
 * rows of the EOM-DEA vectors that a rank takes from their owners, and the
 * rows of sigma that it gives back).  n segments; segment s is len[s] words
 * at the word pos[s] of w (Fortran numbering, from 1), sent to (dir[s] = 0)
 * or received from (dir[s] = 1) the rank peer[s] (never this rank).  add = 0:
 * a received segment is stored; add = 1: it is added to what is there
 * (through tmp, ntmp words, at least one chunk or the longest segment; the
 * additions are made one after the other, so segments of several ranks may
 * add to the same words).  The two sides list the segments between two
 * ranks in the same order with the same lengths.  All sends are posted
 * first, then the receives: no rank waits for another before it has posted
 * what it gives. */
void xzmpi_segs(double *w, int64_t n, const int64_t *peer, const int64_t *pos, const int64_t *len,
                const int64_t *dir, int64_t add, double *tmp, int64_t ntmp)
{
    int64_t nsend = 0, nrecv = 0, maxlen = 0;
    for (int64_t s = 0; s < n; s++) {
        int64_t c = (len[s] + xz_chunk - 1) / xz_chunk;
        if (dir[s] == 0) nsend += c; else nrecv += c;
        if (len[s] > maxlen) maxlen = len[s];
    }
    MPI_Request *rq = malloc(sizeof(MPI_Request) * (size_t)(nsend + nrecv + 1));
    int64_t nrq = 0;
    for (int64_t s = 0; s < n; s++) {
        if (dir[s] != 0) continue;
        for (int64_t o = 0; o < len[s]; o += xz_chunk) {
            int m = (int)(len[s] - o < xz_chunk ? len[s] - o : xz_chunk);
            MPI_Isend(w + pos[s] - 1 + o, m, MPI_DOUBLE, (int)peer[s], 31 + (int)add, MPI_COMM_WORLD, &rq[nrq++]);
        }
    }
    if (!add) {
        for (int64_t s = 0; s < n; s++) {
            if (dir[s] == 0) continue;
            for (int64_t o = 0; o < len[s]; o += xz_chunk) {
                int m = (int)(len[s] - o < xz_chunk ? len[s] - o : xz_chunk);
                MPI_Irecv(w + pos[s] - 1 + o, m, MPI_DOUBLE, (int)peer[s], 31, MPI_COMM_WORLD, &rq[nrq++]);
            }
        }
    } else if (nrecv > 0) {
        /* the receives that add: up to nslot at a time, in the order of the list */
        int64_t slot = xz_chunk < maxlen ? xz_chunk : maxlen;
        int nslot = (int)(ntmp / (slot > 0 ? slot : 1));
        if (nslot > 16) nslot = 16;
        if (nslot < 1) nslot = 1;
        MPI_Request *ar = malloc(sizeof(MPI_Request) * (size_t)nslot);
        double **dst = malloc(sizeof(double *) * (size_t)nslot);
        int *cnt = malloc(sizeof(int) * (size_t)nslot);
        int64_t s = 0, o = 0, done = 0, posted = 0;
        for (int k = 0; k < nslot; k++) ar[k] = MPI_REQUEST_NULL;
        while (done < nrecv) {
            for (int k = 0; k < nslot && posted < nrecv; k++) {
                if (ar[k] != MPI_REQUEST_NULL) continue;
                while (s < n && (dir[s] == 0 || o >= len[s])) { s++; o = 0; }
                int m = (int)(len[s] - o < xz_chunk ? len[s] - o : xz_chunk);
                dst[k] = w + pos[s] - 1 + o;
                cnt[k] = m;
                MPI_Irecv(tmp + (int64_t)k * slot, m, MPI_DOUBLE, (int)peer[s], 32, MPI_COMM_WORLD, &ar[k]);
                o += m;
                posted++;
            }
            int k;
            MPI_Waitany(nslot, ar, &k, MPI_STATUS_IGNORE);
            double *x = dst[k], *y = tmp + (int64_t)k * slot;
            const int m = cnt[k];
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < m; i++) x[i] += y[i];
            done++;
        }
        free(ar);
        free(dst);
        free(cnt);
    }
    MPI_Waitall((int)nrq, rq, MPI_STATUSES_IGNORE);
    free(rq);
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

/* ---- A sub-communicator of the ranks of one colour (EOM-DEA with 5p3h: the
 * ranks that share an index of Xv5): xzmpi_sub_split makes it (every rank
 * calls it, colour >= 0; a later call replaces it), xzmpi_sub_allsum_w sums
 * buf over its ranks, in place, in chunks. */
static MPI_Comm xz_sub = MPI_COMM_NULL;

void xzmpi_sub_split(int64_t color)
{
    int r;
    if (xz_sub != MPI_COMM_NULL) MPI_Comm_free(&xz_sub);
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    MPI_Comm_split(MPI_COMM_WORLD, (int)color, r, &xz_sub);
}

void xzmpi_sub_allsum_w(double *buf, int64_t n)
{
    int np;
    if (xz_sub == MPI_COMM_NULL) return;
    MPI_Comm_size(xz_sub, &np);
    if (np < 2) return;
    for (int64_t off = 0; off < n; off += xz_chunk) {
        int m = (int)(n - off < xz_chunk ? n - off : xz_chunk);
        MPI_Allreduce(MPI_IN_PLACE, buf + off, m, MPI_DOUBLE, MPI_SUM, xz_sub);
    }
}

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
void xzmpi_bcast_w(double *buf, int64_t n) { (void)buf; (void)n; }
void xzmpi_send_w(const double *buf, int64_t n, int64_t dest, int64_t tag) { (void)buf; (void)n; (void)dest; (void)tag; }
void xzmpi_recv_w(double *buf, int64_t n, int64_t src, int64_t tag) { (void)buf; (void)n; (void)src; (void)tag; }
void xzmpi_sum0_w(double *buf, int64_t n) { (void)buf; (void)n; }
void xzmpi_allsum_w(double *buf, int64_t n) { (void)buf; (void)n; }
void xzmpi_node(char *name, int64_t len) { if (len > 0) name[0] = 0; }
void xzmpi_slices_add(double *buf, const int64_t *off, double *tmp) { (void)buf; (void)off; (void)tmp; }
void xzmpi_slices_put(double *buf, const int64_t *off) { (void)buf; (void)off; }
void xzmpi_segs(double *w, int64_t n, const int64_t *peer, const int64_t *pos, const int64_t *len,
                const int64_t *dir, int64_t add, double *tmp, int64_t ntmp)
{ (void)w; (void)n; (void)peer; (void)pos; (void)len; (void)dir; (void)add; (void)tmp; (void)ntmp; }
void xzmpi_sub_split(int64_t color) { (void)color; }
void xzmpi_sub_allsum_w(double *buf, int64_t n) { (void)buf; (void)n; }
void xzmpi_finalize(void) {}
void xzmpi_abort(int64_t code) { exit((int)code); }

#endif

/* ---- A test of the chunked routines (option mpitest n words, with the chunk
 * that the option mpichunk gives): patterns that depend on the rank and on the
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
    /* slices of unequal lengths (one of them empty with three or more ranks):
     * the sum of the contributions on the slice of the owner, then the
     * slices of the owners on every rank */
    if (np > 1) {
        int64_t *off = malloc(sizeof(int64_t) * (size_t)(np + 1));
        off[0] = 0;
        for (int64_t r = 0; r < np; r++) {
            int64_t len = (2 * (r + 1) * n) / (np * (np + 1));
            if (np > 2 && r == 1) len = 0;
            off[r + 1] = off[r] + len;
        }
        off[np] = n;
        for (int64_t i = 0; i < n; i++) a[i] = (double)(me + 1) * (double)(i % 991);
        xzmpi_slices_add(a, off, b);
        for (int64_t i = off[me]; i < off[me + 1]; i++)
            bad += a[i] != 0.5 * (double)(np * (np + 1)) * (double)(i % 991);
        for (int64_t i = 0; i < n; i++)
            if (i < off[me] || i >= off[me + 1]) bad += a[i] != (double)(me + 1) * (double)(i % 991);
        for (int64_t i = 0; i < n; i++)
            a[i] = (i >= off[me] && i < off[me + 1]) ? (double)(me + 3) + (double)(i % 983) : -7.0;
        xzmpi_slices_put(a, off);
        for (int64_t r = 0; r < np; r++)
            for (int64_t i = off[r]; i < off[r + 1]; i++)
                bad += a[i] != (double)(r + 3) + (double)(i % 983);
        free(off);
    }
    free(a);
    return bad;
}
