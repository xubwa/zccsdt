!
! The state of the Krylov solver of the Lambda equations (xzlkry.inc; docs/
! native.md, section 11).  The equations are linear in lambda: the Jacobi
! step of an iteration, g(x) = R(x) / D, is affine, g(x) = c + M x with M the
! Jacobian divided by the denominators.  The solver keeps pairs (b_i, s_i =
! M b_i) of directions and their images, s_i orthonormal, and takes as the
! next lambda the point of x + span(b) whose step is smallest, plus that
! step: the minimal-residual (GMRES) iterate in the form of Anderson's
! method, with the full history instead of the eight vectors of DIIS.
!
! A pair has two vectors of the layout of a DIIS vector: l1 and l2 on pairs
! (rank 0: the small part, always in the work array) and l3 as stored (the
! big part: in the work array, or records of the file XZLKRY; with slices
! the elements of this rank only).  The directions are kept as the raw
! differences dx_j of the iterates; b = dx T (T upper triangular, small).
! Slots: the columns of the subspace (kcol), the previous iterate and its
! step (kprev: the next difference is taken from it), and a free slot whose
! step region is where the next iteration writes its new lambda (kfree).
!
module xzlkryd
  implicit none
  integer, parameter :: mkmax = 63          ! pairs at most (zc_vmdot: 64 vectors)
  integer(8) :: kp = 0                      ! pairs (slots)
  integer(8) :: km = 0                      ! columns of the subspace
  integer(8) :: kcol(mkmax) = 0             ! their slots, in the order of T
  integer(8) :: kprev = 0                   ! slot of the previous iterate (0: none)
  integer(8) :: kfree = 0                   ! slot of the next iterate
  integer(8) :: kcall = 0                   ! iterations seen
  integer(8) :: knres = 0                   ! restarts (compactions)
  integer(8) :: kkeep = 0                   ! columns kept by a compaction
  logical :: kfile = .false.                ! big parts in the file
  logical :: ksplit = .false.               ! big parts: the slice of this rank
  integer(8) :: ksoff = 0, kslen = 0        ! that slice of the stored l3
  integer(8) :: knch = 0, knrec = 0         ! file: elements of a record, records of a vector
  integer(8) :: jxs(mkmax) = 0, jss(mkmax) = 0   ! small parts of dx and s (rank 0)
  integer(8) :: jxb(mkmax) = 0, jsb(mkmax) = 0   ! big parts in the work array
  integer(8) :: jxs0 = 0, jgs0 = 0          ! small parts of x and of its step
  integer(8) :: jrw = 0                     ! file or slices: the new l3 / the step
  integer(8) :: jbuf = 0                    ! file: one record per slot and kind
  complex(8) :: tmat(mkmax, mkmax) = 0      ! b_k = sum_j dx_j T(j, k)
  complex(8) :: dgs(mkmax) = 0              ! <s_i | g> of the present step
  real(8) :: kthr(4) = 0d0                  ! harmonic Ritz values of the last compaction (the smallest |theta|)
  real(8) :: ktime = 0d0, kbytes = 0d0      ! seconds and bytes of the file
  integer, parameter :: lukry = 74
  character(len=11) :: fkry = 'XZLKRY'
end module xzlkryd
