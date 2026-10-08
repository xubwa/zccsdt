!
! DIIS of the amplitudes (the scheme of src/zccsdt/cc/diis.py: the error
! vector of an iteration is the new vector minus the extrapolated vector of
! the iteration before; the first vector only starts the sequence).
!
! A vector has three parts: T1, T2 on i<j, a<b, and packed T3.  Each of the
! mdiis slots holds a vector x and its error vector e.  The small parts are
! always regions of the work array.  The T3 parts are regions of the work
! array as well (infile = .false.), or records of the direct-access file
! XZDIIS in the run directory (infile = .true.), which are then passed over
! in chunks of nchunk elements.
!
module xzdiisd
  implicit none
  integer, parameter :: maxdiis = 32
  integer(8) :: mdiis = 0             ! slots
  integer(8) :: ncnt = 0              ! vectors in the subspace
  integer(8) :: ord(maxdiis) = 0      ! their slots, oldest first
  integer(8) :: ncall = 0             ! vectors seen
  logical :: infile = .false.
  integer(8) :: nchunk = 0            ! elements of a record of the file
  integer(8) :: nrec = 0              ! records of a T3 part
  integer, parameter :: ludiis = 73
  character(len=6), parameter :: fdiis = 'XZDIIS'
  ! regions (index of the first word) of the parts of the slots
  integer(8) :: jx1(maxdiis), jx2(maxdiis), jx3(maxdiis)
  integer(8) :: je1(maxdiis), je2(maxdiis), je3(maxdiis)
  integer(8) :: jrwork = 0            ! infile: the residual / new T3 of the iteration
  integer(8) :: jchunk1 = 0, jchunk2 = 0
  integer(8) :: jwork2 = 0            ! T2 on pairs
  complex(8) :: hmat(maxdiis, maxdiis) = 0     ! <e_k|e_l> by slots
contains

  ! the slot that the vector of the next iteration goes to
  integer(8) function xzdiis_slot()
    if (ncnt < mdiis) then
      xzdiis_slot = ncnt + 1
    else
      xzdiis_slot = ord(1)
    end if
  end function
end module xzdiisd

! ---------------------------------------------------------------------------
! The coefficients c(1:n) of the extrapolation from the overlaps h of the
! error vectors (n, n; Hermitian): the constrained least-squares problem
!   | 0  1 | |l|   |1|
!   | 1  H | |c| = |0|
! solved through the eigenvectors of the matrix, those with eigenvalues below
! 1e-14 left out.  H is scaled by its largest diagonal element first, and
! taken as real if its imaginary parts are below 1e-10 (overlaps of vectors
! that are symmetric under time reversal: real coefficients keep the
! symmetry).
subroutine xzdiis_solve(n, h, ldh, c)
  implicit none
  integer(8), intent(in) :: n, ldh
  complex(8), intent(in) :: h(ldh, *)
  complex(8), intent(out) :: c(n)
  complex(8) :: b(n + 1, n + 1), work(4 * (n + 1)), x
  real(8) :: ev(n + 1), rwork(3 * (n + 1)), scale, himag
  integer(8) :: i, j, k, info, m
  m = n + 1
  scale = 0d0
  do i = 1, n
    scale = max(scale, abs(h(i, i)))
  end do
  if (scale <= 0d0) scale = 1d0
  himag = 0d0
  do i = 1, n
    do j = 1, n
      himag = max(himag, abs(aimag(h(i, j))) / scale)
    end do
  end do
  b = 0
  do i = 1, n
    b(1, i + 1) = 1
    b(i + 1, 1) = 1
    do j = 1, n
      b(i + 1, j + 1) = h(i, j) / scale
      if (himag < 1d-10) b(i + 1, j + 1) = dble(b(i + 1, j + 1))
    end do
  end do
  call zheev('V', 'U', m, b, m, ev, work, 4 * m, rwork, info)
  if (info /= 0) then
    write(6, '(a,i6)') ' @XZCCSDT-W, DIIS: ZHEEV failed, info ', info
    c = 0
    c(n) = 1
    return
  end if
  ! solution = sum_k v_k conj(v_k(1)) / ev_k
  c = 0
  do k = 1, m
    if (abs(ev(k)) < 1d-14) cycle
    x = conjg(b(1, k)) / ev(k)
    do i = 1, n
      c(i) = c(i) + b(i + 1, k) * x
    end do
  end do
end subroutine xzdiis_solve
