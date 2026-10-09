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
    subroutine xz_blas_info(nthreads, ilp64) bind(C)
      import
      integer(c_int64_t) :: nthreads, ilp64
    end subroutine
  end interface
end module xzkern
