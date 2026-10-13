!
! Explicit interfaces of the C kernels (csrc/) and of the C helpers
! (xzutil.c) that the Fortran scheduler calls.
!
module xzkern
  use iso_c_binding
  implicit none

  interface
#include "xztype.h"
#include "xzkern.inc"
#define XZ_REAL
#include "xztype.h"
#include "xzkern.inc"
#undef XZ_REAL

    ! x(0:n-1) = 0 on all threads (n in double words)
    subroutine zc_vzero_d(n, x) bind(C)
      import
      integer(c_int64_t), value :: n
      type(*), dimension(*) :: x
    end subroutine

    ! elements of the work of a batch of ni occupied triples of group g (csrc/cc3_fly.c)
    function zc_cc3_work(ncls, g, ni, vrng, qoff, aoff) bind(C) result(n)
      import
      integer(c_int64_t), value :: ncls, g, ni
      integer(c_int64_t) :: vrng(*), qoff(*), aoff(*)
      integer(c_int64_t) :: n
    end function

    subroutine zc_tn_zero(p, nbytes) bind(C)
      import
      type(*), dimension(*) :: p
      integer(c_int64_t), value :: nbytes
    end subroutine

    ! out(A) = sum of e over the indices of the sorted k-tuple A
    subroutine zc_tn_tuple_sums(e, n, k, out) bind(C)
      import
      real(c_double) :: e(*), out(*)
      integer(c_int), value :: n, k
    end subroutine

    ! bytes that the file system of the directory `path' has free (-1: unknown)
    function xz_free_disk(path) bind(C) result(nbytes)
      import
      character(kind=c_char) :: path(*)
      integer(c_int64_t) :: nbytes
    end function

    ! the file `from' takes the name `to' (0: done)
    function xz_rename(from, to) bind(C) result(ierr)
      import
      character(kind=c_char) :: from(*), to(*)
      integer(c_int64_t) :: ierr
    end function

    ! the file mapped read-only and shared: its address (0: failed), its bytes
    function xz_map_file(path, nbytes, populate) bind(C) result(addr)
      import
      character(kind=c_char) :: path(*)
      integer(c_int64_t) :: nbytes, populate, addr
    end function

    ! 1: the file system of the directory is one of the node, 0: a network file system, -1: unknown
    function xz_fs_local(path) bind(C) result(l)
      import
      character(kind=c_char) :: path(*)
      integer(c_int64_t) :: l
    end function

    ! kB of the process: peak resident set; resident now: anonymous, mapped files, shared memory
    subroutine xz_rss4(kb) bind(C)
      import
      integer(c_int64_t) :: kb(4)
    end subroutine

    ! kB that the node can still give (-1: unknown)
    function xz_mem_avail() bind(C) result(kb)
      import
      integer(c_int64_t) :: kb
    end function

    ! kB of the process: (1) peak resident set, (2) resident set now
    subroutine xz_rss(kb) bind(C)
      import
      integer(c_int64_t) :: kb(2)
    end subroutine

    ! number of threads of the BLAS and whether its integers have 8 bytes
    subroutine xz_clocks(wall, cpu) bind(C)
      import
      real(c_double), intent(out) :: wall, cpu
    end subroutine
    ! the standard output to /dev/null (1) and back (0)
    subroutine xz_stdout(quiet) bind(C)
      import
      integer(c_int64_t) :: quiet
    end subroutine
    subroutine xz_blas_info(nthreads, ilp64) bind(C)
      import
      integer(c_int64_t) :: nthreads, ilp64
    end subroutine
  end interface
end module xzkern
