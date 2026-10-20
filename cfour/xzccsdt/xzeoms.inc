!
! EOM-EE-CCSDT: the solver and the driver (type-generic; the sigma vector:
! xzeom.inc; input, shifts, results: xzeomd.F90; docs/native.md, section 16).
!
! The solver is that of CFOUR's two-component EOM-CCSD program xx2ceomcc in
! its design (docs/eom.md, section 9; its algorithm 2, eomene2): right-hand
! vectors only, one orthonormal basis for all roots of a problem, one
! correction vector per root that is not converged per iteration, the
! projected matrix <C_i| H C_j> diagonalized by LAPACK's general solver, the
! eigenpairs sorted by their real parts, a converged root kept by its
! coefficients and recognized by its overlap afterwards, the test on the
! largest element of the residual of the normalized vector, the correction
! r / (omega - D), Gram-Schmidt twice, and - when the subspace is full - its
! compaction to the Ritz vectors of the converged roots and the lowest
! others (roots + 2 in all), C and H C transformed alike.
!
! What is CFOUR's code and what is written here:
!   linked (complex arithmetic)   eomreorder (the sort of the eigenpairs) and
!                                 eomconv (the test and its line of output),
!                                 objects of its program directory x2ceomcc
!   written here                  the same two for real arithmetic, and all
!                                 that touches a vector: CFOUR's routines for
!                                 that read whole vectors of singles and
!                                 doubles through the offsets of its own work
!                                 array (modules soccmem, soccmod), complex.
!
! The vectors of the subspace: in the work array if all of them fit there
! next to what a sigma vector takes, otherwise the files EOMCnnnn and
! EOMHnnnn of the run directory (CFOUR's names), one vector each, read in
! chunks through a buffer of the work array; two vectors are in the work
! array in either case (the new vector and its sigma vector).
!
! More than one rank (xzeomd.F90; docs/native.md, section 16.7): rank 0 runs
! this solver; every rank holds its rows of R3 of every vector of the
! subspace, and every operation on vectors below is a command of rank 0
! that all ranks carry out on their rows (N(xzeserve) on the other ranks),
! the dot products summed onto rank 0.  The singles and doubles of a vector
! are rank 0's.
!

! ---------------------------------------------------------------------------
! The place of the subspace of evcap vectors of nlen elements: the work
! array, or files.  avail: words that are free for it.  If not all msub
! vectors fit in the work array but at least nmem of them do, the subspace
! is that small and stays there (a smaller subspace costs sigma vectors;
! files cost about seven readings of the whole subspace per new vector).
! With files the number of vectors is lowered to what the disk holds (at
! least nmin).
subroutine N(xzevopen)(nlen, msub, nmin, nmem, avail, mayfile)
  use xzdata
  use xzmem
  use xzkern
  use xzeomd
  implicit none
  integer(8), intent(in) :: nlen, nmin, nmem, avail
  integer(8), intent(inout) :: msub
  logical, intent(in) :: mayfile
  integer(8) :: need, free_bytes, per, nfit, masked
  character(len=8) :: arg
  integer :: larg
  evlen = nlen
  evnc = 0
  evnh = 0
  ! (more than one rank: as the dry run of rank 0 planned it - the other
  ! ranks have the memory of that plan)
  masked = msub
  if (ecmd .and. eplan > 0) msub = min(msub, epm(eplan))
  call xzevplace(int(NW, 8), nlen, nmem, avail, mayfile, msub, evfile)
  if (msub /= masked) write(6, '(a,i4,a,i4,a)') '   the work array holds ', msub, &
    ' vectors of the subspace with their sigma vectors (', masked, &
    ' asked for): the subspace is that small and stays in the work array'
  need = 2 * msub * (NW * nlen + xzalign)
  if (ecmd .and. eplan > 0) then
    if (epf(eplan)) evfile = .true.
  end if
  ! a test hook: option eomvec f (files), m (the work array)
  call xzoptget('eomvec', arg, larg)
  if (larg > 0 .and. mayfile .and. .not. xzdry) then
    if (arg(1:1) == 'f' .or. arg(1:1) == 'F') evfile = .true.
  end if
  if (evfile .and. .not. mayfile) then
    write(6, '(/,a)') ' @XZCCSDT-F, the work array is too short for the vectors of the singles-doubles stage of EOM.'
    write(6, '(a,i16,a,i16)') '   words needed: ', need, '   words free: ', avail
    call xzerrex
  end if
  if (.not. evfile) then
    evcap = msub
    evc0 = xzget(msub * NW * nlen, 'EOM: the vectors of the subspace')
    evh0 = xzget(msub * NW * nlen, 'EOM: their sigma vectors')
    if (ecmd) call N(xzevopenw)(msub)
    return
  end if
  ! ---- files: the buffer, the disk
  evnch = min(nlen, 1048576_8, max(avail - 64, 0_8) / max(msub * NW, 1_8))
  if (evnch < min(nlen, 1024_8)) then
    write(6, '(/,a)') ' @XZCCSDT-F, the work array is too short for the buffer of the EOM vectors in files.'
    write(6, '(a,i16,a,i16)') '   words needed: ', msub * NW * min(nlen, 1024_8) + 64, '   words free: ', avail
    call xzerrex
  end if
  per = 2 * 8 * NW * nlen
  ! (over ranks: the ranks of a node may share the run directory - whole vectors)
  if (ecmd) per = 2 * 8 * NW * (xzensd() + xzenbig())
  free_bytes = xz_free_disk('.' // c_null_char)
  if (free_bytes >= 0 .and. per * msub > free_bytes - 1073741824_8) then
    nfit = max(0_8, (free_bytes - 1073741824_8) / max(per, 1_8))
    if (nfit < nmin) then
      write(6, '(/,a)') ' @XZCCSDT-F, the disk of the run directory does not hold the vectors of the EOM subspace.'
      write(6, '(a,i4,a,f12.1,a,f12.1,a)') '   at least ', nmin, ' vectors and their sigma vectors: ', 1d-6 * per * nmin, &
        ' MB;  free: ', 1d-6 * free_bytes, ' MB (1024 MB of it are left alone)'
      call xzerrex
    end if
    write(6, '(a,i4,a,f10.1,a)') ' @XZCCSDT-W, the disk of the run directory holds ', nfit, &
      ' vectors of the EOM subspace with their sigma vectors (', 1d-6 * per, ' MB each): the subspace is that small.'
    msub = nfit
  end if
  evcap = msub
  evbuf = xzget(msub * NW * evnch, 'EOM: buffer of the vectors in files')
  if (ecmd) call N(xzevopenw)(msub)
end subroutine

! More than one rank: the place of the subspace as rank 0 decided it - msub
! vectors, in the work array or in files (evfile), through a buffer of evnch
! elements per vector (rank 0: the command; another rank: its rows of the
! vectors, in its own work array or in its own files EOMCnnnn.<rank>).
subroutine N(xzevopenw)(msub)
  use xzdata
  use xzmem
  use xzeomd
  use xzpar, only: myrank
  implicit none
  integer(8), intent(in) :: msub
  if (myrank == 0) then
    call xzecmd(op_open, dble(msub), merge(1d0, 0d0, evfile), dble(evnch))
    return
  end if
  evoff = xzensd() + esoff(myrank)
  evlen = esoff(myrank + 1) - esoff(myrank)
  evnc = 0
  evnh = 0
  evcap = msub
  if (evfile) then
    evnch = max(1_8, min(evnch, evlen))
    evbuf = xzget(msub * NW * evnch, 'EOM: buffer of the vectors in files')
  else
    evc0 = xzget(msub * NW * evlen, 'EOM: the vectors of the subspace')
    evh0 = xzget(msub * NW * evlen, 'EOM: their sigma vectors')
  end if
end subroutine

! the files are closed and deleted
subroutine N(xzevclose)()
  use xzeomd
  use xzpar, only: myrank
  implicit none
  integer(8) :: j
  integer :: ios
  if (ecmd .and. myrank == 0) call xzecmd(op_close)
  if (evfile) then
    do j = 1, evnc
      close(xzevunit(1, j), status='delete', iostat=ios)
    end do
    do j = 1, evnh
      close(xzevunit(2, j), status='delete', iostat=ios)
    end do
  end if
  evnc = 0
  evnh = 0
  evfile = .false.
end subroutine

! the vector at kx becomes vector j of the basis (kind 1) or of the sigma
! vectors (kind 2)
subroutine N(xzevput)(w, kind, j, kx)
  use xzdata
  use xzeomd
  use xzpar, only: myrank
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer, intent(in) :: kind
  integer(8), intent(in) :: j, kx
  integer(8) :: k
  integer :: ios, u
  real(8) :: t0
  integer(8) :: kl
  if (ecmd .and. myrank == 0) call xzecmd(op_put, dble(kind), dble(j), xzewhich(kx))
  ! (the part of the vector that this rank holds)
  kl = kx + NW * evoff
  if (kind == 1) evlast = j
  if (.not. evfile .or. kind > 2) then
    k = xzevbase(kind) + (j - 1) * NW * evlen
    call N(xzecopy)(evlen, w(kl), w(k))
    return
  end if
  t0 = omp_get_wtime()
  u = xzevunit(kind, j)
  open(unit=u, file=trim(xzevname(kind, j)), form='unformatted', access='stream', status='replace', iostat=ios)
  if (ios == 0) write(u, iostat=ios) w(kl:kl + NW * evlen - 1)
  if (ios == 0) flush(u, iostat=ios)
  if (ios /= 0) then
    write(6, '(/,a,a,a,i8,a)') ' @XZCCSDT-F, writing the file ', trim(xzevname(kind, j)), ' failed (iostat ', ios, '): disk full?'
    write(6, '(a,f12.1,a)') '   a vector of the EOM subspace: ', 8d-6 * NW * evlen, ' MB'
    call xzerrex
  end if
  if (kind == 1) evnc = max(evnc, j)
  if (kind == 2) evnh = max(evnh, j)
  evio = evio + omp_get_wtime() - t0
  evbytes = evbytes + 8d0 * NW * evlen
end subroutine

! vector j of a kind (the part that this rank holds) into the vector at kx
subroutine N(xzevget)(w, kind, j, kx)
  use xzdata
  use xzeomd
  use xzpar, only: myrank
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer, intent(in) :: kind
  integer(8), intent(in) :: j, kx
  integer(8) :: k, kl
  integer :: ios
  real(8) :: t0
  if (ecmd .and. myrank == 0) call xzecmd(op_get, dble(kind), dble(j), xzewhich(kx))
  kl = kx + NW * evoff
  if (.not. evfile .or. kind > 2) then
    k = xzevbase(kind) + (j - 1) * NW * evlen
    call N(xzecopy)(evlen, w(k), w(kl))
    return
  end if
  t0 = omp_get_wtime()
  read(xzevunit(kind, j), pos=1, iostat=ios) w(kl:kl + NW * evlen - 1)
  if (ios /= 0) then
    write(6, '(/,a,a,a,i8)') ' @XZCCSDT-F, reading the file ', trim(xzevname(kind, j)), ' failed, iostat ', ios
    call xzerrex
  end if
  evio = evio + omp_get_wtime() - t0
  evbytes = evbytes + 8d0 * NW * evlen
end subroutine

! the elements off .. off + nc - 1 of the vectors 1 .. n of a kind: their
! addresses (in the work array, or in the buffer after reading them)
subroutine N(xzevchunk)(w, kind, n, off, nc, ptr)
  use xzdata
  use xzeomd
  use xzkern
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer, intent(in) :: kind
  integer(8), intent(in) :: n, off, nc
  integer(c_intptr_t), intent(out) :: ptr(*)
  integer(8) :: j, k
  integer :: ios
  real(8) :: t0
  if (.not. evfile .or. kind > 2) then
    do j = 1, n
      k = xzevbase(kind) + (j - 1) * NW * evlen + NW * off
      ptr(j) = loc(w(k))
    end do
    return
  end if
  t0 = omp_get_wtime()
  do j = 1, n
    k = evbuf + (j - 1) * NW * evnch
    read(xzevunit(kind, j), pos=1 + 8 * NW * off, iostat=ios) w(k:k + NW * nc - 1)
    if (ios /= 0) then
      write(6, '(/,a,a,a,i8)') ' @XZCCSDT-F, reading the file ', trim(xzevname(kind, j)), ' failed, iostat ', ios
      call xzerrex
    end if
    ptr(j) = loc(w(k))
  end do
  evio = evio + omp_get_wtime() - t0
  evbytes = evbytes + 8d0 * NW * nc * n
end subroutine

! res(j) = <x | X_j>, j = 1 .. n, for the vector x at kx
subroutine N(xzevmdot)(w, kind, n, kx, res)
  use xzdata
  use xzeomd
  use xzkern
  use xzpar, only: myrank
  implicit none
  real(8) :: w(*)
  integer, intent(in) :: kind
  integer(8), intent(in) :: n, kx
  ZT, intent(out) :: res(n)
  ZT :: part(n)
  integer(c_intptr_t) :: ptr(n + 1)
  integer(8) :: off, nc, step
  res = 0
  if (n == 0) return
  if (ecmd .and. myrank == 0) call xzecmd(op_mdot, dble(kind), dble(n), xzewhich(kx))
  step = max(evlen, 1_8)
  if (evfile) step = evnch
  do off = 0, evlen - 1, step
    nc = min(step, evlen - off)
    call N(xzevchunk)(w, kind, n, off, nc, ptr)
    call N(zc_vmdot)(nc, n, w(kx + NW * (evoff + off)), ptr, part)
    res = res + part
  end do
  ! (the sum of the parts of the ranks, on rank 0)
  if (ecmd) call N(xzesum)(n, res)
end subroutine

! x (n numbers) of rank 0 becomes the sum of x over the ranks
subroutine N(xzesum)(n, x)
  use xzpar, only: myrank, xzmpi_sum
  implicit none
  integer(8), intent(in) :: n
  ZT, intent(inout) :: x(n)
  real(8) :: a(NW * n), b(NW * n)
  a = transfer(x, a, NW * n)
  b = 0d0
  call xzmpi_sum(a, b, NW * n)
  if (myrank == 0) x = transfer(b, x, n)
end subroutine

! x = [x +] sum_j coef(j) X_j for the vector x at kx
subroutine N(xzevcomb)(w, kind, n, coef, kx, add)
  use xzdata
  use xzeomd
  use xzkern
  use xzpar, only: myrank, xzmpi_bcast
  implicit none
  real(8) :: w(*)
  integer, intent(in) :: kind
  integer(8), intent(in) :: n, kx
  ZT, intent(in) :: coef(n)
  logical, intent(in) :: add
  ZT :: c(n + 1)
  integer(c_intptr_t) :: ptr(n + 1)
  integer(8) :: off, nc, step, nk
  if (ecmd .and. myrank == 0) call xzecmd(op_comb, dble(kind), dble(n), xzewhich(kx), merge(1d0, 0d0, add))
  if (n == 0) then
    if (.not. add) call zc_vzero_d(NW * evlen, w(kx + NW * evoff))
    return
  end if
  c(1:n) = coef
  ! (the coefficients of rank 0)
  if (ecmd) call xzmpi_bcast(c, 8_8 * NW * n)
  c(n + 1) = 1
  nk = n
  if (add) nk = n + 1
  step = max(evlen, 1_8)
  if (evfile) step = evnch
  do off = 0, evlen - 1, step
    nc = min(step, evlen - off)
    call N(xzevchunk)(w, kind, n, off, nc, ptr)
    ptr(n + 1) = loc(w(kx + NW * (evoff + off)))
    call N(zc_vlincomb)(nc, nk, ptr, c, w(kx + NW * (evoff + off)))
  end do
end subroutine

! The first m vectors of both kinds become sum_j X_j q[j][k] (k < m) of the
! n there are; the others are dropped.
subroutine N(xzevtrans)(w, n, m, q)
  use xzdata
  use xzeomd
  use xzkern
  use xzpar, only: myrank, xzmpi_bcast
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: n, m
  ZT :: q(*)                                                   ! [j][k], n x m
  integer(c_intptr_t) :: ptr(n)
  integer(8) :: off, nc, step, j, k
  integer :: kind, ios
  real(8) :: t0
  if (ecmd .and. myrank == 0) call xzecmd(op_trans, dble(n), dble(m))
  if (ecmd) call xzmpi_bcast(q, 8_8 * NW * n * m)
  step = max(evlen, 1_8)
  if (evfile) step = evnch
  do kind = 1, 2
    do off = 0, evlen - 1, step
      nc = min(step, evlen - off)
      call N(xzevchunk)(w, kind, n, off, nc, ptr)
      call N(zc_vtransform)(nc, n, m, ptr, q)
      if (evfile) then
        t0 = omp_get_wtime()
        do j = 1, m
          k = evbuf + (j - 1) * NW * evnch
          write(xzevunit(kind, j), pos=1 + 8 * NW * off, iostat=ios) w(k:k + NW * nc - 1)
          if (ios /= 0) then
            write(6, '(/,a,a,a,i8,a)') ' @XZCCSDT-F, writing the file ', trim(xzevname(kind, j)), ' failed (iostat ', ios, ')'
            call xzerrex
          end if
        end do
        evio = evio + omp_get_wtime() - t0
        evbytes = evbytes + 8d0 * NW * nc * m
      end if
    end do
    if (evfile) then
      do j = m + 1, n
        close(xzevunit(kind, j), status='delete', iostat=ios)
      end do
    end if
  end do
  if (evfile) then
    evnc = m
    evnh = m
  end if
end subroutine

! ---------------------------------------------------------------------------
! The eigenpairs of the projected matrix g (n x n, general), sorted by the
! real parts of the eigenvalues, the vectors normalized.  Real arithmetic: a
! complex pair comes as two vectors, the real and the imaginary part of its
! eigenvector, both with the real part of the eigenvalue (two roots that are
! degenerate but for the rounding come out like that).
subroutine N(xzesmall)(n, g, ldg, ev, y)
  implicit none
  integer(8), intent(in) :: n, ldg
  ZT, intent(in) :: g(ldg, *)
  complex(8), intent(out) :: ev(n)
  ZT, intent(out) :: y(ldg, *)
  ZT :: a(n, n), vr(n, n), vl(1, 1), work(8 * n + 64)
  integer(8) :: info, lwork, i, j
  real(8) :: s
#ifdef XZ_REAL
  real(8) :: wr(n), wi(n), tmp(n), z
  complex(8) :: x
#else
  real(8) :: rwork(2 * n)
  complex(8) :: ctmp(n, n), tmp(n)
#endif
  a = g(1:n, 1:n)
  lwork = 8 * n + 64
#ifdef XZ_REAL
  call dgeev('N', 'V', n, a, n, wr, wi, vl, 1_8, vr, n, work, lwork, info)
  ev = cmplx(wr, wi, 8)
  ! the sort of CFOUR's eomreorder: insertion by the real parts
  do i = 2, n
    x = ev(i)
    tmp = vr(:, i)
    j = i
    do while (j > 1)
      if (dble(x) >= dble(ev(j - 1))) exit
      ev(j) = ev(j - 1)
      vr(:, j) = vr(:, j - 1)
      j = j - 1
    end do
    ev(j) = x
    vr(:, j) = tmp
  end do
  z = 0
#else
  call zgeev('N', 'V', n, a, n, ev, vl, 1_8, vr, n, work, lwork, rwork, info)
  call eomreorder(vr, ev, ctmp, tmp, n)
#endif
  if (info /= 0) then
    write(6, '(/,a,i6)') ' @XZCCSDT-F, EOM: the eigenvalues of the projected matrix failed, info ', info
    call xzerrex
  end if
  do j = 1, n
    s = sqrt(sum(dble(vr(:, j) * CONJ(vr(:, j)))))
    if (s > 0d0) vr(:, j) = vr(:, j) / s
    y(1:n, j) = vr(:, j)
  end do
end subroutine

! The test of convergence of CFOUR's eomconv and its line of output: the
! largest element z of the residual (absolute value; N(xzepmax), over the
! ranks) against the threshold.  Complex arithmetic: eomconv itself, given
! that element.
subroutine N(xzeconv)(z, e, idav, icyc, ieomconv, eomthr)
  implicit none
  integer(8), intent(in) :: icyc, idav
  real(8), intent(in) :: z
  complex(8), intent(in) :: e
  integer(8), intent(out) :: ieomconv
  real(8), intent(in) :: eomthr
#ifdef XZ_REAL
  ieomconv = 1
  if (z > eomthr) ieomconv = 0
  write(6, '("EOM cycle",3X,I4,2X,I4,F22.15,5X,D20.10,5X,D20.10)') icyc, idav, dble(e), z, eomthr
#else
  complex(8) :: c0(1)
  c0(1) = cmplx(z, 0d0, 8)
  call eomconv(c0, e, idav, icyc, 1_8, ieomconv, eomthr)
#endif
end subroutine

! ---------------------------------------------------------------------------
! Operations on the two vectors of the work array (kx, ky: their addresses),
! each on the part that this rank holds; with more than one rank a command
! of rank 0 first, and the numbers summed onto rank 0.
!
! x = 0
subroutine N(xzepzero)(w, kx)
  use xzeomd
  use xzkern
  use xzpar, only: myrank
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kx
  if (ecmd .and. myrank == 0) call xzecmd(op_zero, xzewhich(kx))
  call zc_vzero_d(NW * evlen, w(kx + NW * evoff))
end subroutine

! res = <x | y>; r3only: of the triples alone
subroutine N(xzepdot)(w, kx, ky, r3only, res)
  use xzdata
  use xzeomd
  use xzkern
  use xzpar, only: myrank
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kx, ky
  logical, intent(in) :: r3only
  ZT, intent(out) :: res
  ZT :: r(1)
  integer(8) :: lo, nsd
  if (ecmd .and. myrank == 0) call xzecmd(op_dot, xzewhich(kx), xzewhich(ky), merge(1d0, 0d0, r3only))
  nsd = xzensd()
  lo = evoff
  if (r3only) lo = max(lo, nsd)
  call N(zc_vdot)(evoff + evlen - lo, w(kx + NW * lo), w(ky + NW * lo), r(1))
  if (ecmd) call N(xzesum)(1_8, r)
  res = r(1)
end subroutine

! x <- alpha x
subroutine N(xzepscal)(w, kx, alpha)
  use xzeomd
  use xzpar, only: myrank
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kx
  real(8), intent(in) :: alpha
  if (ecmd .and. myrank == 0) call xzecmd(op_scal, xzewhich(kx), alpha)
  call N(xzevscal)(evlen, alpha, w(kx + NW * evoff))
end subroutine

! z = the largest element of x (absolute value)
subroutine N(xzepmax)(w, kx, z)
  use xzeomd
  use xzpar, only: myrank, xzmpi_max
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kx
  real(8), intent(out) :: z
  real(8) :: a(1), b(1)
  if (ecmd .and. myrank == 0) call xzecmd(op_max, xzewhich(kx))
  call N(xzevmax)(evlen, w(kx + NW * evoff), z)
  if (ecmd) then
    a(1) = z
    b(1) = z
    call xzmpi_max(a, b, 1_8)
    z = b(1)
  end if
end subroutine

! y += alpha x
subroutine N(xzepaxpy)(w, alpha, kx, ky)
  use xzeomd
  use xzpar, only: myrank
  implicit none
  real(8) :: w(*)
  ZT, intent(in) :: alpha
  integer(8), intent(in) :: kx, ky
  real(8), external :: N(xzeimag)
  if (ecmd .and. myrank == 0) call xzecmd(op_axpy, xzewhich(kx), xzewhich(ky), dble(alpha), N(xzeimag)(alpha))
  call N(xzeaxpyc)(evlen, alpha, w(kx + NW * evoff), w(ky + NW * evoff))
end subroutine

! The triples of x: the rows of every rank to all ranks (rank 0 then has the
! whole vector).  One rank: nothing.
subroutine N(xzepcollect)(w, kx)
  use xzdata
  use xzeomd
  use xzpar, only: myrank, xzmpi_slices_put
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kx
  if (.not. ecmd) return
  if (myrank == 0) call xzecmd(op_collect, xzewhich(kx))
  if (edea) then
    ! (EOM-DEA: every rank holds only its rows - the whole vector to rank 0)
    call N(xzepslices)(w, kx, .false.)
    return
  end if
  call xzmpi_slices_put(w(kx + NW * xzensd()), eswoff)
end subroutine

! The vector x of rank 0, all of it, to every rank (the test hook eomsig).
subroutine N(xzepspread)(w, kx)
  use xzdata
  use xzeomd
  use xzpar, only: myrank, xzmpi_bcast_w
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kx
  if (.not. ecmd) return
  if (myrank == 0) call xzecmd(op_spread, xzewhich(kx))
  if (edea) then
    ! (EOM-DEA: every rank its rows of the whole vector of rank 0)
    call N(xzepslices)(w, kx, .true.)
    return
  end if
  call xzmpi_bcast_w(w(kx), NW * (xzensd() + xzenbig()))
end subroutine

! EOM-DEA (the test hook deasig): the rows of r4 of the whole vector x of
! rank 0 to their owners (out), or from them to rank 0 (not out)
subroutine N(xzepslices)(w, kx, out)
  use xzdata
  use xzeomd
  use xzpar, only: myrank, nrank, xzmpi_send_w, xzmpi_recv_w
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kx
  logical, intent(in) :: out
  integer(8) :: r, k
  do r = 1, nrank - 1
    k = kx + NW * (dvsd + esoff(r))
    if (myrank == 0 .and. out) call xzmpi_send_w(w(k), NW * (esoff(r + 1) - esoff(r)), r, 23_8)
    if (myrank == 0 .and. .not. out) call xzmpi_recv_w(w(k), NW * (esoff(r + 1) - esoff(r)), r, 23_8)
    if (myrank == r .and. out) call xzmpi_recv_w(w(k), NW * (esoff(r + 1) - esoff(r)), 0_8, 23_8)
    if (myrank == r .and. .not. out) call xzmpi_send_w(w(k), NW * (esoff(r + 1) - esoff(r)), 0_8, 23_8)
  end do
end subroutine

! The sigma vector of x into y.  More than one rank (and R3): the singles
! and doubles of rank 0 and the rows of R3 of every rank go to all ranks,
! every rank does its batches of the two residuals (N(xzesig)), and the
! rows of sigma_3 go to their owners; sigma_1 and sigma_2 are made by every
! rank, rank 0's are used.
subroutine N(xzepsig)(w, kx, ky, sd)
  use xzdata
  use xzeomd
  use xzdist
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kx, ky
  logical, intent(in) :: sd
  integer(8) :: nsd
  real(8) :: t0
  if (edea) then
    ! (EOM-DEA: the distribution over the ranks is that of N(xzdsig) itself)
    if (ecmd .and. myrank == 0) call xzecmd(op_sigma, xzewhich(kx), xzewhich(ky))
    call N(xzdsig)(w, kx, ky)
    return
  end if
  if (.not. ecmd .or. sd) then
    call N(xzesig)(w, kx, ky, sd)
    return
  end if
  if (myrank == 0) call xzecmd(op_sigma, xzewhich(kx), xzewhich(ky))
  nsd = no * nv + npo * nvp
  t0 = omp_get_wtime()
  call xztic()
  call xzmpi_bcast_w(w(kx), NW * nsd)
  call xzmpi_slices_put(w(kx + NW * nsd), eswoff)
  call xztoc(5, 'MPI: the vector to the ranks')
  dsec(6, 3) = dsec(6, 3) + omp_get_wtime() - t0
  dwords(3) = dwords(3) + dble(NW * (n3s - (esoff(myrank + 1) - esoff(myrank))))
  call N(xzesig)(w, kx, ky, .false.)
  call xzrebal_eom()
end subroutine

! ---------------------------------------------------------------------------
! A rank other than 0 in the problems with R3: it carries out the commands
! of rank 0 (xzeomd: op_...) on its rows of the vectors until rank 0 says
! that the excited states are done.  nsig: the sigma vectors it took part in.
subroutine N(xzeserve)(w, nsig)
  use xzdata
  use xzmem
  use xzkern
  use xzeomd
  use xzdist
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(out) :: nsig
  real(8) :: ctl(6), z, t0
  integer(8) :: ms, mv, nsd, n, m, kx, ky, msub
  integer :: op, kind
  ZT :: res, om
  ZT, allocatable :: c(:), d(:)
  nsd = no * nv + npo * nvp
  nsig = 0
  ms = xzmark()
  mv = ms
  allocate(c(64 * 64), d(64))
  do
    call xzmpi_bcast(ctl, 48_8)
    op = nint(ctl(1))
    kx = ekx(max(1, min(2, nint(ctl(2)))))
    ky = ekx(max(1, min(2, nint(ctl(3)))))
    select case (op)
    case (op_end)
      exit
    case (op_shift)
      if (edea) then
        ! EOM-DEA (xzdeam.inc): the sector of the target, the rows of r4 of this
        ! rank, the batches of rank 0, two whole vectors
        call N(xzdshift)(w, nint(ctl(2), 8), nint(ctl(3), 8), ms, mv)
        cycle
      end if
      ! a problem starts: the layout of its shift, the slices, the batches, two vectors
      call xzeomlay(w, nint(ctl(2), 8))
      call xzeslices()
      call xzdeal_eom(eshift)
      ms = xzmark()
      ekx(1) = xzget(NW * (nsd + n3s), 'EOM: a vector')
      ekx(2) = xzget(NW * (nsd + n3s), 'EOM: its sigma vector')
      mv = xzmark()
      edist = .true.
      ecmd = .true.
      evfile = .false.
      evoff = nsd + esoff(myrank)
      evlen = esoff(myrank + 1) - esoff(myrank)
      write(6, '(a,i5,a,i14,a,i14,a,i14,a,f10.1,a)') '   label shift ', eshift, ' classes: the rows of this rank are the triples ', &
        esoff(myrank), ' to ', esoff(myrank + 1) - 1, ' of ', n3s, ' (', 8d-6 * NW * evlen, ' MB of a vector)'
      flush(6)
    case (op_endshift)
      call xzrel(ms)
    case (op_open)
      msub = nint(ctl(2), 8)
      evfile = ctl(3) == 1d0
      evnch = nint(ctl(4), 8)
      mv = xzmark()
      call N(xzevopenw)(msub)
      ! (the coefficients that rank 0 sends: up to msub, msub x msub for a compaction)
      deallocate(c, d)
      allocate(c(msub * msub), d(msub))
      if (evfile) then
        write(6, '(a,i4,a,f10.1,a)') '   subspace: its rows of at most ', msub, ' vectors and their sigma vectors, in files (', &
          16d-6 * NW * evlen * msub, ' MB)'
      else
        write(6, '(a,i4,a,f10.1,a)') '   subspace: its rows of at most ', msub, ' vectors and their sigma vectors, in the work array (', &
          16d-6 * NW * evlen * msub, ' MB)'
      end if
      flush(6)
    case (op_close)
      call N(xzevclose)()
      call xzrel(mv)
    case (op_zero)
      call N(xzepzero)(w, kx)
    case (op_dot)
      call N(xzepdot)(w, kx, ky, ctl(4) == 1d0, res)
    case (op_scal)
      call N(xzepscal)(w, kx, ctl(3))
    case (op_mdot)
      kind = nint(ctl(2))
      n = nint(ctl(3), 8)
      call N(xzevmdot)(w, kind, n, ekx(nint(ctl(4))), d)
    case (op_comb)
      kind = nint(ctl(2))
      n = nint(ctl(3), 8)
      call N(xzevcomb)(w, kind, n, d, ekx(nint(ctl(4))), ctl(5) == 1d0)
    case (op_put)
      call N(xzevput)(w, nint(ctl(2)), nint(ctl(3), 8), ekx(nint(ctl(4))))
    case (op_sigma)
      t0 = omp_get_wtime()
      call N(xzepsig)(w, kx, ky, .false.)
      teomsig = teomsig + omp_get_wtime() - t0
      nsig = nsig + 1
    case (op_max)
      call N(xzepmax)(w, kx, z)
    case (op_prec)
#ifdef XZ_REAL
      om = ctl(3)
#else
      om = cmplx(ctl(3), ctl(4), 8)
#endif
      call N(xzeprec)(w, kx, om, .false.)
    case (op_trans)
      n = nint(ctl(2), 8)
      m = nint(ctl(3), 8)
      call N(xzevtrans)(w, n, m, c)
    case (op_axpy)
#ifdef XZ_REAL
      om = ctl(4)
#else
      om = cmplx(ctl(4), ctl(5), 8)
#endif
      call N(xzepaxpy)(w, om, kx, ky)
    case (op_collect)
      call N(xzepcollect)(w, kx)
    case (op_spread)
      call N(xzepspread)(w, kx)
    case (op_fold)
      ! EOM-CC4: the sigma vectors that follow have the T4 terms at this frequency (or not)
      efold = ctl(2) == 1d0
      eomega = ctl(3)
    case (op_qopen)
      ! EOM-CC4, macro / micro: the rows of this rank of q and F
      n = nint(ctl(2), 8)
      e4kq = xzget(NW * n * evlen, 'EOM-CC4: the vectors of a root')
      e4kf = xzget(NW * n * evlen, 'EOM-CC4: their sigma vectors with the T4 terms')
    case (op_get)
      call N(xzevget)(w, nint(ctl(2)), nint(ctl(3), 8), ekx(nint(ctl(4))))
    case default
      write(6, '(a,i6)') ' @XZCCSDT-F, EOM: a command of rank 0 that this rank does not know: ', op
      call xzerrex
    end select
  end do
  edist = .false.
  ecmd = .false.
  efold = .false.
  evoff = 0
  deallocate(c, d)
end subroutine

! the largest element of x (absolute value)
subroutine N(xzevmax)(n, x, z)
  implicit none
  integer(8), intent(in) :: n
  ZT, intent(in) :: x(n)
  real(8), intent(out) :: z
  integer(8) :: i
  z = 0d0
  !$omp parallel do reduction(max:z)
  do i = 1, n
    z = max(z, abs(x(i)))
  end do
end subroutine

! x <- alpha x
subroutine N(xzevscal)(n, alpha, x)
  implicit none
  integer(8), intent(in) :: n
  real(8), intent(in) :: alpha
  ZT, intent(inout) :: x(n)
  integer(8) :: i
  !$omp parallel do
  do i = 1, n
    x(i) = alpha * x(i)
  end do
end subroutine

! ---------------------------------------------------------------------------
! x <- x / (omega - D), D the differences of the spinor energies of the
! excitation; denominators below 1e-3 are replaced by 1e-3.
subroutine N(xzeprec)(w, kx, omega, sd)
  use xzdata
  use xzkern
  use xzeomd
  use xzpar, only: myrank
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kx
  ZT, intent(in) :: omega
  logical, intent(in) :: sd
  real(8), external :: N(xzeimag)
  real(8) :: eo2(0:max(npo, 1_8) - 1), ev2(0:max(nvp, 1_8) - 1)
  integer(8) :: nsd, g, ni, na, a, b
  real(8), parameter :: floor = 1d-3
  if (ecmd .and. myrank == 0) call xzecmd(op_prec, xzewhich(kx), dble(omega), N(xzeimag)(omega))
  if (edea) then
    call N(xzdprec)(w, kx, omega)
    return
  end if
  nsd = no * nv + npo * nvp
  ! (the singles and doubles: rank 0's)
  if (myrank == 0) then
    call N(zc_vprecond)(w(kx), no, nv, w(jeps), w(jeps + no), omega, floor)
    call xzepsum(npo, no, w(jopa), w(jopb), w(jeps), eo2)
    call xzepsum(nvp, nv, w(jvpa), w(jvpb), w(jeps + no), ev2)
    if (npo > 0) call N(zc_vprecond)(w(kx + NW * no * nv), npo, nvp, eo2, ev2, omega, floor)
  end if
  if (sd) return
  ! (the triples: the rows of this rank)
  if (ncls == 1) then
    call xzeslrows(0_8, a, b)
    if (b > a) call N(zc_vprecond)(w(kx + NW * (nsd + a * nvt)), b - a, nvt, w(jeo3 + a), w(jev3), omega, floor)
  else
    do g = 0, ntg - 1
      ni = otoff(g + 1) - otoff(g)
      na = xzna(eshift, g)
      if (ni == 0 .or. na == 0) cycle
      call xzeslrows(g, a, b)
      if (b <= a) cycle
      call N(zc_vprecond)(w(kx + NW * (nsd + boff_s(g) + a * na)), b - a, na, w(jeo3 + otoff(g) + a), &
                          w(jev3 + vtoff(g + eshift)), omega, floor)
    end do
  end if
end subroutine

! ---------------------------------------------------------------------------
! The vector at kx orthogonalized against the n vectors of the basis (twice)
! and normalized.  rel: its norm after that, relative to its norm before.
subroutine N(xzeorth)(w, n, kx, rel)
  use xzdata
  use xzeomd
  use xzkern
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: n, kx
  real(8), intent(out) :: rel
  ZT :: d(max(n, 1_8)), s
  real(8) :: nrm0, nrm
  integer :: pass
  call N(xzepdot)(w, kx, kx, .false., s)
  nrm0 = sqrt(dble(s))
  rel = 0d0
  if (nrm0 == 0d0) return
  call N(xzepscal)(w, kx, 1d0 / nrm0)
  nrm = 1d0
  do pass = 1, 2
    if (n > 0) then
      call N(xzevmdot)(w, 1, n, kx, d)
      d(1:n) = -CONJ(d(1:n))
      call N(xzevcomb)(w, 1, n, d, kx, .true.)
    end if
    call N(xzepdot)(w, kx, kx, .false., s)
    nrm = sqrt(dble(s))
    if (pass == 1) rel = nrm
    if (nrm < 1d-12) then
      rel = 0d0
      return
    end if
    call N(xzepscal)(w, kx, 1d0 / nrm)
  end do
end subroutine

! ---------------------------------------------------------------------------
! The sigma vector of the solver: N(xzesig), or - EOM-CC4, macro / micro
! (xzcc4.inc; e4nq of xzeomd > 0) - of the matrix with the T4 terms held on
! the span of the e4nq orthonormal vectors q_c at e4kq, whose sigma vectors
! with the T4 terms F_c are at e4kf:
!   B x = A_PP (x - sum_c q_c (q_c|x)) + sum_c F_c (q_c|x).
subroutine N(xzesigm)(w, kv, ks, sd)
  use xzdata
  use xzmem
  use xzkern
  use xzeomd
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kv, ks
  logical, intent(in) :: sd
  integer(8) :: m, kt, nlen, c
  ZT :: a(128)
  if (e4nq == 0 .or. sd .or. efold) then
    call N(xzepsig)(w, kv, ks, sd)
    return
  end if
  if (ecmd) then
    ! over the ranks: every rank on its rows of q and F (the vector at kv is
    ! the one of the basis that was put last: it comes back from there)
    call N(xzevmdot)(w, 3, e4nq, kv, a)
    a(1:e4nq) = CONJ(a(1:e4nq))
    a(e4nq + 1:2 * e4nq) = -a(1:e4nq)
    call N(xzevcomb)(w, 3, e4nq, a(e4nq + 1), kv, .true.)
    call N(xzepsig)(w, kv, ks, sd)
    call N(xzevcomb)(w, 4, e4nq, a, ks, .true.)
    call N(xzevget)(w, 1, evlast, kv)
    return
  end if
  nlen = no * nv + npo * nvp + n3s
  m = xzmark()
  kt = xzget(NW * nlen, 'EOM-CC4: the vector without its part on q')
  call N(xzecopy)(nlen, w(kv), w(kt))
  do c = 1, e4nq
    call N(zc_vdot)(nlen, w(e4kq + NW * nlen * (c - 1)), w(kv), a(c))
    call N(xzeaxpyc)(nlen, -a(c), w(e4kq + NW * nlen * (c - 1)), w(kt))
  end do
  call N(xzesig)(w, kt, ks, sd)
  do c = 1, e4nq
    call N(xzeaxpyc)(nlen, a(c), w(e4kf + NW * nlen * (c - 1)), w(ks))
  end do
  call xzrel(m)
end subroutine

! ---------------------------------------------------------------------------
! The solver of one problem (one label shift; the singles-doubles block of
! it if sd).
!   nroot      roots wanted: the lowest
!   thr        the largest element of a residual at most
!   maxit      iterations at most
!   msub       vectors of the subspace at most (lowered if the disk is short)
!   nstart     vectors to start from: unit vectors on the singles `singles'
!              (kstart = 0), or the vectors of singles and doubles at kstart
!   kxa, kxb   two vectors of the work array
!   avail      words that are free for the subspace
!   ee, rmx, rnm, wts, cyc   of the roots found: energy, the largest element
!              and the norm of the residual, the weights of r1, r2, r3, the
!              iteration in which it converged
!   nfound     roots returned;  nsig: sigma vectors;  nit: iterations
!   kout       (sd) the vectors of the roots, singles and doubles, go here
!   iures      (not sd) the unit of the file of the vectors (0: none)
!   lead       the largest elements of r1 of every root: place and value
subroutine N(xzesolve)(w, sd, nroot, thr, maxit, msub, nstart, kstart, singles, kxa, kxb, avail, &
                       ee, xmx, xnm, wts, cyc, nfound, nsig, nit, kout, iures, lead, vlead, quiet)
  use xzdata
  use xzmem
  use xzkern
  use xzeomd
  use xzpar, only: nrank
  use omp_lib
  implicit none
  real(8) :: w(*)
  logical, intent(in) :: sd, quiet
  integer(8), intent(in) :: nroot, maxit, nstart, kstart, singles(*), kxa, kxb, avail, kout
  integer(8), intent(inout) :: msub
  real(8), intent(in) :: thr
  complex(8), intent(out) :: ee(nroot)
  real(8), intent(out) :: xmx(nroot), xnm(nroot), wts(3, nroot)
  integer(8), intent(out) :: cyc(nroot), nfound, nsig, nit, lead(nlead, nroot)
  ZT, intent(out) :: vlead(nlead, nroot)
  integer, intent(in) :: iures
  integer(8) :: nsd, nlen, n, n0, nconv, it, k, j, g, c, ntarget, nun, mkeep, nadded, icv, mv, jbest, nkeep, i, nloc
  integer(8) :: nin, nout, mkeep0, nblk
  ZT, allocatable :: gm(:, :), ym(:, :), cr(:, :), coef(:), dots(:), qm(:, :), qr(:), gq(:, :)
  complex(8), allocatable :: ev(:)
  logical, allocatable :: flag(:)
  integer(8), allocatable :: keep(:), ccyc(:)
  real(8) :: rel, z, zn, ov, best, t0
  ZT :: s, om
  logical :: needdiag, stalled, allok
  integer :: ios
  integer(8) :: wpos

  nsd = xzensd()
  nlen = nsd
  if (.not. sd) nlen = nsd + xzenbig()
  ! (the vectors to start from and those returned at kout: singles and
  ! doubles, or - EOM-CC4, evfin / evfout - whole vectors)
  nin = nsd
  nout = nsd
  if (evfin) nin = nlen
  if (evfout) nout = nlen
  nfound = 0
  nsig = 0
  nit = 0
  mv = xzmark()
  ! (more than one rank, with R3: the vector operations are commands to all
  ! ranks; rank 0 holds the singles and doubles and its rows of R3)
  ecmd = edist .and. .not. sd
  nloc = nlen
  if (ecmd) then
    nloc = nsd + esoff(1)
    ekx = [kxa, kxb]
  end if
  evoff = 0
  call N(xzevopen)(nloc, msub, min(msub, nroot + 3), min(msub, 3 * nroot + 4), avail, .not. sd)
  if (.not. quiet) then
    if (ecmd .and. .not. edea) write(6, '(a,i4,a,f10.1,a,f10.1,a)') '   over ', nrank, &
      ' ranks: every rank holds its rows of R3 of every vector (rank 0 with the singles and doubles: ', &
      8d-6 * NW * nloc, ' MB of ', 8d-6 * NW * nlen, ' MB)'
    if (ecmd .and. edea) write(6, '(a,i4,a,f10.1,a,f10.1,a)') '   over ', nrank, &
      ' ranks: every rank holds its rows of r4 of every vector (rank 0 with r2 and r3: ', &
      8d-6 * NW * nloc, ' MB of ', 8d-6 * NW * nlen, ' MB)'
    if (evfile) then
      write(6, '(a,i4,a,f10.1,a)') '   subspace: at most ', msub, ' vectors and their sigma vectors, in the files EOMCnnnn, ' // &
        'EOMHnnnn of the run directory (', 16d-6 * NW * nloc * msub, ' MB),'
      write(6, '(a,i10,a)') '     read through a buffer of ', evnch, ' elements per vector'
    else
      write(6, '(a,i4,a,f10.1,a)') '   subspace: at most ', msub, ' vectors and their sigma vectors, in the work array (', &
        16d-6 * NW * nloc * msub, ' MB)'
    end if
    flush(6)
  end if
  allocate(gm(msub, msub), ym(msub, msub), cr(msub, nroot), coef(msub), dots(msub), ev(msub), flag(msub), &
           keep(msub), ccyc(nroot), qm(msub, msub), qr(msub * msub), gq(msub, msub))
  gm = 0
  n = 0
  ! a compaction keeps mkeep0 Ritz vectors (the roots and two more); an
  ! iteration adds at most nblk corrections, to the lowest roots that are
  ! not converged, so that what is kept and what is added always fit
  mkeep0 = max(1_8, min(nroot + 2, msub - 1))
  nblk = max(1_8, msub - mkeep0)
  ! ---- the vectors to start from
  do g = 1, nstart
    if (n >= msub) exit
    call N(xzepzero)(w, kxa)
    if (kstart > 0) then
      call N(xzecopy)(nin, w(kstart + NW * nin * (g - 1)), w(kxa))
      ! (a whole vector of rank 0: to all ranks)
      if (evfin) call N(xzepspread)(w, kxa)
    else
      call N(xzeone)(w(kxa + NW * singles(g)))
    end if
    call N(xzeorth)(w, n, kxa, rel)
    if (rel < 1d-8) cycle
    call add_vector()
  end do
  if (n == 0) then
    write(6, '(a)') ' @XZCCSDT-W, EOM: no vector to start from.'
    call N(xzevclose)()
    call xzrel(mv)
    ecmd = .false.
    return
  end if

  nconv = 0
  ccyc = 0
  needdiag = .true.
  it = 0
  wpos = 0
  if (iures > 0) inquire(unit=iures, pos=wpos)
  verify: do
  stalled = .false.
  do while (it < maxit)
    it = it + 1
    nit = it
    call N(xzesmall)(n, gm, msub, ev, ym)
    if (nfollow > 0) call follow()
    if (it == 1) then
      if (allocated(eritz0)) deallocate(eritz0)
      allocate(eritz0(n))
      eritz0 = dble(ev(1:n))
    end if
    needdiag = .false.
    ! the converged roots among the eigenvectors, by their overlap
    call mark_converged()
    ntarget = min(nroot, n)
    nun = count(.not. flag(1:ntarget))
    if (nun == 0) exit
    ! compaction: the Ritz vectors of the converged roots and the lowest others
    if (n + min(nun, nblk) > msub) then
      mkeep = min(n, mkeep0)
      call compact()
      call N(xzesmall)(n, gm, msub, ev, ym)
      if (nfollow > 0) call follow()
      call mark_converged()
      ntarget = min(nroot, n)
    end if
    n0 = n
    nadded = 0
    do k = 1, ntarget
      if (flag(k)) cycle
      ! the residual of the normalized Ritz vector
      coef(1:n0) = ym(1:n0, k)
      call N(xzevcomb)(w, 2, n0, coef, kxa, .false.)
#ifdef XZ_REAL
      om = dble(ev(k))
#else
      om = ev(k)
#endif
      coef(1:n0) = -om * ym(1:n0, k)
      call N(xzevcomb)(w, 1, n0, coef, kxa, .true.)
      call N(xzepmax)(w, kxa, z)
      if (quiet) then
        icv = merge(1_8, 0_8, z <= thr)
      else
        call N(xzeconv)(z, ev(k), n0, it, icv, thr)
      end if
      if (icv == 1) then
        nconv = min(nconv + 1, nroot)
        cr(:, nconv) = 0
        cr(1:n0, nconv) = ym(1:n0, k)
        ccyc(nconv) = it
        flag(k) = .true.
        if (.not. quiet) write(6, '(a,i4,a,i4,a)') '   root ', k, ' converged in ', it, ' cycles'
        cycle
      end if
      if (n >= msub .or. nadded >= nblk) cycle
      ! the correction: r / (omega - D), orthogonal to the basis
      call N(xzeprec)(w, kxa, om, sd)
      call N(xzeorth)(w, n, kxa, rel)
      if (rel < 1d-8) cycle
      call add_vector()
      nadded = nadded + 1
      needdiag = .true.
    end do
    if (.not. quiet) flush(6)
    if (nadded == 0 .and. count(.not. flag(1:ntarget)) > 0) then
      if (.not. quiet) write(6, '(a)') ' @XZCCSDT-W, EOM: no correction vector is left (all of them are in the subspace).'
      stalled = .true.
      exit
    end if
  end do

  ! ---- the roots: their vectors, the residuals once more, the weights.  A
  !      root whose residual is above the threshold after all (the vectors of
  !      a nearly degenerate level turn into each other while the subspace
  !      grows, and a converged root is only recognized by its overlap) is
  !      iterated on: the records of the converged roots are dropped.
  if (needdiag) then
    call N(xzesmall)(n, gm, msub, ev, ym)
    if (nfollow > 0) call follow()
    call mark_converged()
  end if
  nfound = min(nroot, n)
  allok = .true.
  do k = 1, nfound
    coef(1:n) = ym(1:n, k)
    call N(xzevcomb)(w, 1, n, coef, kxa, .false.)
    call N(xzevcomb)(w, 2, n, coef, kxb, .false.)
#ifdef XZ_REAL
    om = dble(ev(k))
#else
    om = ev(k)
#endif
    call N(xzepaxpy)(w, -om, kxa, kxb)
    call N(xzepmax)(w, kxb, z)
    call N(xzepdot)(w, kxb, kxb, .false., s)
    ee(k) = ev(k)
    xmx(k) = z
    xnm(k) = sqrt(dble(s))
    if (z > thr) allok = .false.
    cyc(k) = 0
    do c = 1, nconv
      ov = abs(sum(CONJ(cr(1:n, c)) * ym(1:n, k)))
      if (ov > 0.7d0) cyc(k) = ccyc(c)
    end do
    call N(zc_vdot)(xzen1(), w(kxa), w(kxa), s)
    wts(1, k) = dble(s)
    call N(zc_vdot)(nsd - xzen1(), w(kxa + NW * xzen1()), w(kxa + NW * xzen1()), s)
    wts(2, k) = dble(s)
    wts(3, k) = 0d0
    if (.not. sd) then
      call N(xzepdot)(w, kxa, kxa, .true., s)
      wts(3, k) = dble(s)
    end if
    call N(xzelead)(xzen1(), w(kxa), nlead, lead(1, k), vlead(1, k))
    ! (a whole vector: the rows of the other ranks)
    if (kout > 0 .and. evfout) call N(xzepcollect)(w, kxa)
    if (kout > 0) call N(xzecopy)(nout, w(kxa), w(kout + NW * nout * (k - 1)))
    if (iures > 0) then
      ! (the rows of the other ranks)
      call N(xzepcollect)(w, kxa)
      if (k == 1) then
        write(iures, pos=wpos, iostat=ios) eshift, k, nlen, dble(ev(k)), aimag(ev(k)), w(kxa:kxa + NW * nlen - 1)
      else
        write(iures, iostat=ios) eshift, k, nlen, dble(ev(k)), aimag(ev(k)), w(kxa:kxa + NW * nlen - 1)
      end if
      if (ios /= 0) then
        write(6, '(/,a,i8,a)') ' @XZCCSDT-F, writing the file XZEOMR failed (iostat ', ios, '): disk full?'
        call xzerrex
      end if
    end if
  end do
  if (allok .or. stalled .or. it >= maxit) exit verify
  if (.not. quiet) write(6, '(a)') '   (a root is not converged after all: the iterations go on)'
  nconv = 0
  needdiag = .false.
  end do verify
  call N(xzevclose)()
  call xzrel(mv)
  ecmd = .false.

contains

  ! (nfollow = m > 0: the m eigenpairs whose vectors have the largest weight on
  ! the first m vectors of the basis - those the solver started from, or the
  ! followed Ritz vectors after a compaction - go first, in the order of their
  ! energies; the others follow in theirs)
  subroutine follow()
    integer(8) :: jj, kk, m, nsel
    real(8) :: wgt(msub)
    integer(8) :: sel(msub), rest(msub)
    logical :: taken(msub)
    complex(8) :: et(msub)
    ZT :: yt(msub, msub)
    m = min(nfollow, n)
    if (m <= 0) return
    do jj = 1, n
      wgt(jj) = sum(abs(ym(1:m, jj))**2)
    end do
    taken = .false.
    do kk = 1, m
      best = -1d0
      sel(kk) = 1
      do jj = 1, n
        if (taken(jj)) cycle
        if (wgt(jj) > best) then
          best = wgt(jj)
          sel(kk) = jj
        end if
      end do
      taken(sel(kk)) = .true.
    end do
    ! the selected in the order of their energies (ev is sorted: by index)
    nsel = 0
    do jj = 1, n
      if (taken(jj)) then
        nsel = nsel + 1
        sel(nsel) = jj
      end if
    end do
    kk = 0
    do jj = 1, n
      if (.not. taken(jj)) then
        kk = kk + 1
        rest(kk) = jj
      end if
    end do
    sel(m + 1:n) = rest(1:n - m)
    et(1:n) = ev(sel(1:n))
    yt(1:n, 1:n) = ym(1:n, sel(1:n))
    ev(1:n) = et(1:n)
    ym(1:n, 1:n) = yt(1:n, 1:n)
  end subroutine

  ! the vector at kxa joins the basis: its sigma vector, the new row and
  ! column of the projected matrix
  subroutine add_vector()
    real(8) :: ts
    call N(xzevput)(w, 1, n + 1, kxa)
    ts = omp_get_wtime()
    if (efold .and. cc3mode == 2) call N(xzefsend)(w, kxa)
    ! (over the ranks: N(xzepsig); EOM-CC4, macro / micro: the matrix with the T4 terms held)
    call N(xzesigm)(w, kxa, kxb, sd)
    if (sd .and. .not. efold) then
      teomsd = teomsd + omp_get_wtime() - ts
      neomsd = neomsd + 1
    else
      teomsig = teomsig + omp_get_wtime() - ts
      neomsig = neomsig + 1
    end if
    nsig = nsig + 1
    call N(xzevput)(w, 2, n + 1, kxb)
    ts = omp_get_wtime()
    ! <C_j | H c>, j <= n + 1;  <c | H C_j>, j <= n
    call N(xzevmdot)(w, 1, n + 1, kxb, dots)
    gm(1:n + 1, n + 1) = CONJ(dots(1:n + 1))
    if (n > 0) then
      call N(xzevmdot)(w, 2, n, kxa, dots)
      gm(n + 1, 1:n) = dots(1:n)
    end if
    n = n + 1
    teomvec = teomvec + omp_get_wtime() - ts
  end subroutine

  ! flag(j): the eigenvector j is a converged root (the largest overlap with
  ! the coefficients that were kept when it converged)
  subroutine mark_converged()
    integer(8) :: cc, jj
    logical :: kept(max(nroot, 1_8))
    flag = .false.
    do cc = 1, nconv
      best = -1d0
      jbest = 0
      do jj = 1, n
        if (flag(jj)) cycle
        ov = abs(sum(CONJ(cr(1:n, cc)) * ym(1:n, jj)))
        if (ov > best) then
          best = ov
          jbest = jj
        end if
      end do
      if (jbest > 0 .and. best >= 0.95d0) then
        flag(jbest) = .true.
        kept(cc) = .true.
      else
        kept(cc) = .false.
      end if
    end do
    ! (a record that no eigenvector has any more: the root is tested again)
    jj = 0
    do cc = 1, nconv
      if (.not. kept(cc)) cycle
      jj = jj + 1
      if (jj /= cc) then
        cr(:, jj) = cr(:, cc)
        ccyc(jj) = ccyc(cc)
      end if
    end do
    nconv = jj
  end subroutine

  ! the subspace becomes the span of mkeep Ritz vectors: the converged roots,
  ! then the lowest others; an orthonormal basis of it, C and H C alike
  subroutine compact()
    integer(8) :: nk, jj, kk, ii, pass
    ZT :: dd
    real(8) :: nn
    nk = 0
    do jj = 1, n
      if (flag(jj) .and. nk < mkeep) then
        nk = nk + 1
        keep(nk) = jj
      end if
    end do
    do jj = 1, n
      if (.not. flag(jj) .and. nk < mkeep) then
        nk = nk + 1
        keep(nk) = jj
      end if
    end do
    ! the coefficients, orthonormalized (the basis is orthonormal)
    kk = 0
    do jj = 1, nk
      qm(1:n, kk + 1) = ym(1:n, keep(jj))
      do pass = 1, 2
        do ii = 1, kk
          dd = sum(CONJ(qm(1:n, ii)) * qm(1:n, kk + 1))
          qm(1:n, kk + 1) = qm(1:n, kk + 1) - dd * qm(1:n, ii)
        end do
      end do
      nn = sqrt(sum(dble(CONJ(qm(1:n, kk + 1)) * qm(1:n, kk + 1))))
      if (nn < 1d-8) cycle
      qm(1:n, kk + 1) = qm(1:n, kk + 1) / nn
      kk = kk + 1
    end do
    nk = kk
    do jj = 1, n
      do kk = 1, nk
        qr((jj - 1) * nk + kk) = qm(jj, kk)
      end do
    end do
    t0 = omp_get_wtime()
    call N(xzevtrans)(w, n, nk, qr)
    teomvec = teomvec + omp_get_wtime() - t0
    ! the projected matrix and the coefficients of the converged roots in the new basis
    gq(1:n, 1:nk) = matmul(gm(1:n, 1:n), qm(1:n, 1:nk))
    gm = 0
    gm(1:nk, 1:nk) = matmul(transpose(CONJ(qm(1:n, 1:nk))), gq(1:n, 1:nk))
    do kk = 1, nconv
      coef(1:nk) = matmul(transpose(CONJ(qm(1:n, 1:nk))), cr(1:n, kk))
      cr(:, kk) = 0
      cr(1:nk, kk) = coef(1:nk)
    end do
    if (.not. quiet) write(6, '(a,i4,a,i4,a)') '   subspace compacted: ', n, ' -> ', nk, ' vectors'
    n = nk
  end subroutine
end subroutine

! x(1) = 1
subroutine N(xzeone)(x)
  implicit none
  ZT, intent(out) :: x(1)
  x(1) = 1
end subroutine

! y += alpha x
subroutine N(xzeaxpyc)(n, alpha, x, y)
  implicit none
  integer(8), intent(in) :: n
  ZT, intent(in) :: alpha, x(n)
  ZT, intent(inout) :: y(n)
  integer(8) :: i
  !$omp parallel do
  do i = 1, n
    y(i) = y(i) + alpha * x(i)
  end do
end subroutine

! the nl largest elements of x (n of them): their places (from 0) and values
subroutine N(xzelead)(n, x, nl, idx, val)
  implicit none
  integer(8), intent(in) :: n
  integer, intent(in) :: nl
  ZT, intent(in) :: x(0:n-1)
  integer(8), intent(out) :: idx(nl)
  ZT, intent(out) :: val(nl)
  integer(8) :: i
  integer :: k, m, j
  idx = -1
  val = 0
  m = 0
  do i = 0, n - 1
    k = m
    do while (k >= 1)
      if (abs(val(k)) >= abs(x(i))) exit
      k = k - 1
    end do
    if (k >= nl) cycle
    if (m < nl) m = m + 1
    do j = m, k + 2, -1
      val(j) = val(j - 1)
      idx(j) = idx(j - 1)
    end do
    val(k + 1) = x(i)
    idx(k + 1) = i
  end do
end subroutine

! ---------------------------------------------------------------------------
! The driver: the intermediates once, then the states of every label shift
! of the input.  The amplitudes are in the work array (jt1, jt2, jt3).
! A dry run goes through the allocations of the largest problem.
subroutine N(xzeom)(w, ecorr)
  use xzdata
  use xzmem
  use xzkern
  use xzeomd
  use xzdist
  use omp_lib
  implicit none
  real(8) :: w(*)
  real(8), intent(in) :: ecorr
  integer(8) :: m0, ms, ish, s, nsd, nlen, kxa, kxb, kst, nroot, nkeep, nsing, msd, msub, k, peak, sigw, avail
  integer(8) :: nfound, nsig, nit, nsgd, nitd, nst, i, a, r, nprob, pk, top, nloc, mp, ep0n, epwn, kxf
  logical :: dist, worker, pf
  integer(8), allocatable :: singles(:), lead(:, :), cyc(:)
  complex(8), allocatable :: ee(:)
  real(8), allocatable :: rmax(:), rnorm(:), wts(:, :)
  ZT, allocatable :: vlead(:, :)
  real(8) :: t0, t1, tsh, tsg, need
  integer(8) :: free_bytes
  logical :: ok, dense, ck4
  character(len=256) :: arg
  integer :: larg, ios, iures
  real(8), external :: N(xzeimag)
  real(8), parameter :: ev = 27.211386245988d0, cm = 219474.6313632d0

  nsd = no * nv + npo * nvp
  dense = ncls == 1
  ! CC3 without stored triples: the folded problem (xzcc3.inc)
  if (cc3mode == 2) then
    call N(xzeom3)(w, ecorr)
    return
  end if
  ! more than one rank: the sigma vectors with R3 and the vectors of the
  ! solver over the ranks (section 16.7 of docs/native.md); a rank other
  ! than 0 serves rank 0 (N(xzeserve))
  dist = nrank > 1
  worker = xzworker()
  m0 = xzmark()
  call N(xzeomkeep)()
  if (.not. dense) jobase_s = xzget(nto, 'layout of the label shift')
  call xzeomstep()
  nprob = enshift
  if (dense) nprob = 1

  if (xzdry) then
    edist = dist
    call N(xzeomim)(w)
    if (cc4) call N(xz4eomim)(w)
    ep0n = 0
    epwn = 0
    do ish = 1, nprob
      call problem(ish, s, nroot, ok)
      if (.not. ok) cycle
      call xzeomlay(w, s)
      if (dist) call xzeslices()
      nlen = nsd + n3s
      nkeep = nroot + 2
      msd = 8 + 4 * nkeep
      msub = subspace(nroot)
      ms = xzmark()
      efold = cc4
      if (.not. worker) kst = xzget(NW * nkeep * nsd, 'EOM: vectors to start from')
      ! (EOM-CC4: the vectors of the roots; of a root, and its sigma vector, with the T4 terms held; one more)
      ! (over ranks: rank 0 the vectors of the roots, every rank its rows of q and F)
      if (cc4 .and. .not. dist) kxf = xzget(NW * (3 * nroot + 1) * nlen, 'EOM: the vectors of the roots')
      if (cc4 .and. dist .and. .not. worker) kxf = xzget(NW * nroot * (nlen + 2 * (nsd + esoff(1))), 'EOM: the vectors of the roots')
      if (cc4 .and. dist .and. worker) kxf = xzget(2 * NW * nroot * esmax, 'EOM-CC4: the vectors of a root')
      kxa = xzget(NW * nlen, 'EOM: a vector')
      kxb = xzget(NW * nlen, 'EOM: its sigma vector')
      top = xztop
      if (worker) then
        ! (a rank other than 0: its rows of the subspace as rank 0 planned them)
        k = xzget(epw, 'EOM: the subspace')
        call N(xzesig)(w, kxa, kxb, .false.)
      else
        ! (the least: the singles-doubles stage in the work array, or the buffer of the files)
        k = xzget(max(2 * msd * (NW * nsd + xzalign), msub * NW * min(nlen, 1024_8) + 64), 'EOM: the subspace')
        mp = xztop
        pk = xzpeak
        xzpeak = xztop
        call N(xzesig)(w, kxa, kxb, .false.)
        sigw = xzpeak - mp
        xzpeak = max(pk, xzpeak)
        if (dist) then
          ! the subspace that the run will have with this work array: what
          ! the other ranks need for their rows of it
          avail = max(0_8, xzsize - top - sigw)
          nloc = nsd + esoff(1)
          call xzevplace(int(NW, 8), nloc, min(msub, 3 * nroot + 4), avail, .true., msub, pf)
          epm(ish) = msub
          epf(ish) = pf
          if (pf) then
            ep0n = max(ep0n, msub * NW * min(nloc, 1048576_8))
            epwn = max(epwn, msub * NW * min(esmax, 1048576_8) + xzalign)
          else
            ep0n = max(ep0n, 2 * msub * (NW * nloc + xzalign))
            epwn = max(epwn, 2 * msub * (NW * esmax + xzalign))
          end if
        end if
      end if
      efold = .false.
      call xzrel(ms)
    end do
    if (.not. worker) then
      ep0 = ep0n
      if (myrank == 0) epw = epwn
    end if
    edist = .false.
    call xzrel(m0)
    return
  end if

  if (myrank > 0) then
    ! ---- a rank other than 0: the amplitudes of rank 0, the intermediates of
    !      the ground state made here, then the commands of rank 0
    call xzmpi_bcast_w(w(jt1), NW * no * nv)
    call xzmpi_bcast_w(w(jt2), NW * no * no * nv * nv)
    call xzmpi_bcast_w(w(jt3), NW * n3)
    t0 = omp_get_wtime()
    call N(xzeomim)(w)
    if (cc4) call N(xz4eomim)(w)
    teomim = omp_get_wtime() - t0
    write(6, '(/,a,a,a)') '  EOM-EE-', trim(mname), ': the batches of this rank in the sigma vectors, its rows of the vectors'
    write(6, '(a,f10.2,a)') '   intermediates of the ground state: ', teomim, ' s'
    flush(6)
    call N(xzeserve)(w, neomsig)
    call xzdtable(3, neomsig)
    write(6, '(a,i6,a)') '   ', neomsig, ' sigma vectors'
    call xzrel(m0)
    return
  end if

  ! ---- a test hook: option eomtest seed, the sigma vector of a random vector
  !      against the derivative of the amplitude step (before the pp operator
  !      takes the place of <ab||cd>)
  call xzoptget('eomtest', arg, larg)
  if (larg > 0) then
    if (dist) then
      write(6, '(/,a)') ' @XZCCSDT-F, option eomtest (the sigma vector against the derivative of the amplitude step)'
      write(6, '(a)') '             runs with one rank only; option eomsig is the test of a sigma vector over ranks.'
      call xzerrex
    end if
    call N(xzeomtest)(w, arg(1:larg))
    call xzrel(m0)
    return
  end if

  t0 = omp_get_wtime()
  if (dist) then
    ! the converged amplitudes to all ranks; every rank makes the intermediates
    call xzmpi_bcast_w(w(jt1), NW * no * nv)
    call xzmpi_bcast_w(w(jt2), NW * no * no * nv * nv)
    call xzmpi_bcast_w(w(jt3), NW * n3)
  end if
  call N(xzeomim)(w)
  if (cc4) call N(xz4eomim)(w)
  teomim = omp_get_wtime() - t0
  if (cc4) then
    write(6, '(/,a)') '  EOM-EE-CC4: excitation energies and right-hand vectors, the quadruples folded in'
    write(6, '(a)') '  (first the roots of the matrix without the T4 terms - that of EOM-CCSDT at the CC4 amplitudes -,'
    write(6, '(a)') '  then every root with the quadruples at its own frequency)'
  else
  write(6, '(/,a)') '  EOM-EE-CCSDT: excitation energies and right-hand vectors'
  end if
  write(6, '(a)') '  ' // repeat('-', 98)
  write(6, '(a,f10.2,a)') '   intermediates of the ground state (the pp operator in the place of <ab||cd>): ', teomim, ' s'
  write(6, '(a,es9.2,a,i4,a)') '   convergence: the largest element of a residual below ', ethr, &
    ' (ESTATE_CONV, %eomthrs);  at most ', emaxcyc, ' iterations (ESTATE_MAXCYC)'
  if (eguess == 1) then
    write(6, '(a)') '   start: the roots of the singles-doubles block from unit vectors (%xzeomguess 1)'
  else
    write(6, '(a)') '   start: unit vectors on the lowest single excitations (%xzeomguess 0)'
  end if
  if (dense .and. enshift > 1) write(6, '(a,i4,a)') '   no blocks of labels: the states of all shifts are roots of one problem; ', &
    sum(enr(1:enshift)), ' roots (the sum of those asked for)'
  if (dist) write(6, '(a,i4,a)') '   over ', nrank, ' ranks: every rank does its batches of the two residuals of a sigma vector and ' // &
    'holds its rows of R3 of every vector; rank 0 runs the solver'

  ! ---- a test hook: option eomsig file, sigma of the vector of the file (of the
  !      first shift), written to file.out
  call xzoptget('eomsig', arg, larg)
  if (larg > 0) then
    call problem(1_8, s, nroot, ok)
    call xzeomlay(w, s)
    nlen = nsd + n3s
    if (dist) then
      call xzeslices()
      call xzecmd(op_shift, dble(s))
      call xzdeal_eom(eshift)
    end if
    kxa = xzget(NW * nlen, 'EOM: a vector')
    kxb = xzget(NW * nlen, 'EOM: its sigma vector')
    ekx = [kxa, kxb]
    open(unit=74, file=arg(1:larg), form='unformatted', access='stream', status='old', iostat=ios)
    if (ios == 0) read(74, iostat=ios) w(kxa:kxa + NW * nlen - 1)
    close(74)
    if (ios /= 0) then
      write(6, '(a,i12,a)') ' @XZCCSDT-F, option eomsig: the file does not hold a vector of ', nlen, ' elements.'
      call xzerrex
    end if
    call xzoptget('eomsd', length=larg)
    ! (over ranks: the vector to all, sigma_3 from the owners of its rows)
    edist = dist
    ecmd = dist .and. larg == 0
    evoff = 0
    evlen = nlen
    if (dist) evlen = nsd + esoff(1)
    call N(xzepspread)(w, kxa)
    call N(xzepsig)(w, kxa, kxb, larg > 0)
    call N(xzepcollect)(w, kxb)
    ecmd = .false.
    edist = .false.
    if (dist) then
      call xzecmd(op_endshift)
      call xzecmd(op_end)
      call xzdtable(3, 1_8)
    end if
    call xzoptget('eomsig', arg, larg)
    open(unit=74, file=arg(1:larg) // '.out', form='unformatted', access='stream', status='replace')
    write(74) w(kxb:kxb + NW * nlen - 1)
    close(74)
    write(6, '(a,i12,a,a)') ' option eomsig: sigma of a vector of ', nlen, ' elements written to ', arg(1:larg) // '.out'
    call xzrel(m0)
    return
  end if

  ! ---- the file of the vectors of the roots, if the disk of the run directory holds them
  iures = 0
  e4ures = 0
  e4uwant = .false.
  if (ewvec /= 0) then
    need = 0d0
    do ish = 1, nprob
      call problem(ish, s, nroot, ok)
      if (.not. ok) cycle
      call xzeomlay(w, s)
      need = need + 8d0 * NW * dble(nroot) * dble(nsd + n3s + 8)
    end do
    free_bytes = xz_free_disk('.' // c_null_char)
    if (free_bytes >= 0 .and. need > dble(free_bytes) - 1073741824d0) then
      write(6, '(a,f12.1,a,f12.1,a)') ' @XZCCSDT-W, the disk of the run directory does not hold the vectors of the roots (', &
        1d-6 * need, ' MB; free: ', 1d-6 * free_bytes, ' MB): the file XZEOMR is not written.'
    else if (cc4) then
      ! (EOM-CC4: written root by root when the root is done with the T4 terms, N(xz4evec))
      e4uwant = .true.
      write(6, '(a,f12.1,a)') '   the vectors of the roots go to the file XZEOMR (', 1d-6 * need, ' MB; %xzeomvec 0: not written)'
    else
      iures = 78
      open(unit=iures, file='XZEOMR', form='unformatted', access='stream', status='replace', iostat=ios)
      if (ios == 0) write(iures, iostat=ios) 'XZEOMR01', no, nv, int(NW, 8), ncls, labstep
      if (ios /= 0) then
        write(6, '(a)') ' @XZCCSDT-W, the file XZEOMR of the vectors cannot be written: no vectors are kept.'
        iures = 0
      else
        write(6, '(a,f12.1,a)') '   the vectors of the roots go to the file XZEOMR (', 1d-6 * need, ' MB; %xzeomvec 0: not written)'
      end if
    end if
  end if
  nres = 0
  do ish = 1, nprob
    tsh = omp_get_wtime()
    call problem(ish, s, nroot, ok)
    if (.not. ok) cycle
    call xzeomlay(w, s)
    nlen = nsd + n3s
    nkeep = nroot + 2
    allocate(singles(nkeep))
    call xzesingles(eshift, w(jeps), nkeep, singles, nsing)
    write(6, '(/,a)') '  ' // repeat('=', 98)
    if (dense) then
      write(6, '(a,i5,a)') '   EOM-EE-CCSDT on the packed block: ', nroot, ' roots'
    else
      write(6, '(a,i5,a,i4,a,i5,a)') '   EOM-EE-CCSDT, label shift ', eshl(ish), ' (', s, ' classes): ', nroot, ' roots'
    end if
    write(6, '(a,i14,a,i12,a,i14,a,f10.1,a)') '   a vector: ', nlen, ' elements (singles and doubles ', nsd, ', triples ', n3s, &
      ') = ', 8d-6 * NW * nlen, ' MB'
    if (nsing == 0) then
      write(6, '(a)') ' @XZCCSDT-W, no single excitation has this shift: no guess, the shift is left out.'
      deallocate(singles)
      cycle
    end if
    nkeep = min(nkeep, nsing)
    nroot = min(nroot, nsing)
    if (dist) then
      ! the problem to the other ranks: its layout, its slices, its batches
      call xzeslices()
      call xzecmd(op_shift, dble(s))
      call xzdeal_eom(eshift)
    end if
    ms = xzmark()
    kst = xzget(NW * nkeep * nsd, 'EOM: vectors to start from')
    kxf = 0
    if (cc4) kxf = xzget(NW * nroot * nlen, 'EOM: the vectors of the roots')
    kxa = xzget(NW * nlen, 'EOM: a vector')
    kxb = xzget(NW * nlen, 'EOM: its sigma vector')
    allocate(ee(nkeep), rmax(nkeep), rnorm(nkeep), wts(3, nkeep), cyc(nkeep), lead(nlead, nkeep), vlead(nlead, nkeep))
    nst = nroot
    nsgd = 0
    ! ---- EOM-CC4, macro / micro: the roots of this problem of an earlier run
    !      (the file XZE4R.nnn: those without the T4 terms, and those done with them)
    e4rst = .false.
    e4kdone = 0
    ck4 = .false.
    if (cc4) then
      call xzoptget('cc4eom', arg, larg)
      ck4 = .not. (larg > 0 .and. (arg(1:1) == 'd' .or. arg(1:1) == 'D'))
    end if
    if (ck4) call N(xz4er)(w, 'r', ish, merge(0_8, eshl(ish), dense), nroot, nlen, ecorr, kxf, ee, nfound, e4kdone, &
                           e4eclu, e4rayc, e4shotc, e4rst)
    if (e4rst) then
      write(6, '(a,i3.3,a)') '   the roots of the matrix without the T4 terms and the vectors: from the file XZE4R.', ish, &
        ' of an earlier run'
      write(6, '(a,8f14.8)') '     roots / Eh: ', (dble(ee(k)), k = 1, min(nfound, 8_8))
      write(6, '(a,i4,a,i4,a)') '     ', e4kdone, ' of ', min(nroot, nfound), ' roots are done with the T4 terms'
      edist = dist
      eplan = 0
      if (dist) eplan = ish
    end if
    if (.not. e4rst) then
    if (eguess == 1) then
      ! ---- the singles-doubles block of the matrix, from unit vectors
      t1 = omp_get_wtime()
      msd = 8 + 4 * nkeep
      call xzdry_begin()
      call N(xzesig)(w, kxa, kxb, .true.)
      call xzdry_end(peak)
      avail = xzsize - xztop - (peak - xztop)
      call N(xzesolve)(w, .true., nkeep, max(1d-4, ethr), 60_8, msd, nkeep, 0_8, singles, kxa, kxb, avail, &
                       ee, rmax, rnorm, wts, cyc, nfound, nsgd, nitd, kst, 0, lead, vlead, .true.)
      write(6, '(a,i4,a,i5,a,i4,a,f9.2,a)') '   singles-doubles stage: ', nfound, ' roots, ', nsgd, ' sigma vectors, ', nitd, &
        ' iterations, ', omp_get_wtime() - t1, ' s'
      write(6, '(a,8f14.8)') '     roots / Eh: ', (dble(ee(k)), k = 1, min(nfound, 8_8))
      nst = nfound
    end if
    ! ---- the problem with triples
    msub = subspace(nroot)
    edist = dist
    eplan = 0
    if (dist) eplan = ish
    call xzdry_begin()
    call N(xzesig)(w, kxa, kxb, .false.)
    call xzdry_end(peak)
    sigw = peak - xztop
    avail = xzsize - xztop - sigw
    t1 = omp_get_wtime()
    tsg = teomsig
    evio = 0d0
    evbytes = 0d0
    write(6, '(a,f10.1,a,f10.1,a)') '   a sigma vector takes ', 8d-6 * sigw, ' MB of the work array; free besides it: ', &
      8d-6 * avail, ' MB'
    flush(6)
    evfout = cc4
    if (eguess == 1) then
      call N(xzesolve)(w, .false., nroot, ethr, emaxcyc, msub, nst, kst, singles, kxa, kxb, avail, &
                       ee, rmax, rnorm, wts, cyc, nfound, nsig, nit, kxf, iures, lead, vlead, .false.)
    else
      call N(xzesolve)(w, .false., nroot, ethr, emaxcyc, msub, nst, 0_8, singles, kxa, kxb, avail, &
                       ee, rmax, rnorm, wts, cyc, nfound, nsig, nit, kxf, iures, lead, vlead, .false.)
    end if
    evfout = .false.
    t1 = omp_get_wtime() - t1
    tsg = teomsig - tsg
    ! ---- the roots
    write(6, '(/,a,i5,a,i4,a,f10.2,a,f9.2,a)') '   sigma vectors: ', nsig, '   iterations: ', nit, '   time: ', t1, &
      ' s   (', t1 / max(nsig, 1_8), ' s per sigma vector, the solver included)'
    write(6, '(a,f10.2,a,f9.2,a,f10.2,a,f9.2,a)') '   in the sigma vectors: ', tsg, ' s (', &
      tsg / max(nsig, 1_8), ' each);  in the vector operations of the solver: ', t1 - tsg, ' s (', &
      (t1 - tsg) / max(nsig, 1_8), ' per sigma vector)'
    if (evbytes > 0d0) write(6, '(a,f12.1,a,f10.2,a)') '   vectors in files: ', 1d-9 * evbytes, &
      ' GB read and written in ', evio, ' s'
    write(6, '(/,a)') '   root        omega / Eh            eV            cm-1    max |r|      |r|   cycle    R1     R2     R3' // &
                      '          total energy / Eh'
    do k = 1, nfound
      write(6, '(i7,f20.12,f14.8,f16.4,2es10.2,i6,3f7.4,f26.12)') k, dble(ee(k)), dble(ee(k)) * ev, dble(ee(k)) * cm, &
        rmax(k), rnorm(k), cyc(k), wts(:, k), escf + ecorr + dble(ee(k))
      if (abs(aimag(ee(k))) > 1d-10) write(6, '(a,es12.4)') '          imaginary part of omega: ', aimag(ee(k))
      if (rmax(k) > ethr) write(6, '(a)') '          NOT converged'
      if (nres < maxres .and. .not. cc4) then
        nres = nres + 1
        rshift(nres) = eshl(ish)
        if (dense) rshift(nres) = 0
        rroot(nres) = k
        rom(nres) = dble(ee(k))
        rim(nres) = aimag(ee(k))
        rmx(nres) = rmax(k)
        rnm(nres) = rnorm(k)
        rwt(:, nres) = wts(:, k)
        rconv(nres) = rmax(k) <= ethr
        rcyc(nres) = cyc(k)
        rnsig(nres) = nsig
        rli(nres) = 0
        rla(nres) = 0
        rlv(nres) = 0d0
        if (lead(1, k) >= 0) then
          rli(nres) = nfrozen + operm(lead(1, k) / nv) + 1
          rla(nres) = nfrozen + no + vperm(mod(lead(1, k), nv)) + 1
          rlv(nres) = abs(vlead(1, k))
        end if
      end if
    end do
    write(6, '(/,a)') '   the largest elements of R1 (spinors in the order of the SCF: number, label, energy / Eh)'
    do k = 1, nfound
      do r = 1, nlead
        if (lead(r, k) < 0) cycle
        if (abs(vlead(r, k)) < 0.05d0 .and. r > 1) cycle
        i = lead(r, k) / nv
        a = mod(lead(r, k), nv)
        write(6, '(a,i4,a,i5,a,i4,a,f12.6,a,i5,a,i4,a,f12.6,a,2f10.6)') '   root', k, ': ', &
          nfrozen + operm(i) + 1, ' (', lab(nfrozen + operm(i)), ',', w(jeps + i), ')  -> ', &
          nfrozen + no + vperm(a) + 1, ' (', lab(nfrozen + no + vperm(a)), ',', w(jeps + no + a), ')   ', &
          dble(vlead(r, k)), N(xzeimag)(vlead(r, k))
      end do
    end do
    end if
    ! ---- EOM-CC4: the quadruples folded in, root by root (xzcc4.inc)
    if (cc4 .and. nfound > 0) call N(xzeom4f)(w, ish, merge(0_8, eshl(ish), dense), nroot, nfound, ee, kxf, kxa, kxb, &
                                              singles, nlen, ecorr)
    edist = .false.
    eplan = 0
    if (dist) call xzecmd(op_endshift)
    write(6, '(a,f10.2,a)') '   time of this problem: ', omp_get_wtime() - tsh, ' s'
    flush(6)
    deallocate(singles, ee, rmax, rnorm, wts, cyc, lead, vlead)
    call xzrel(ms)
  end do
  if (iures > 0) close(iures)
  if (e4ures > 0) close(e4ures)
  e4ures = 0
  if (dist) then
    call xzecmd(op_end)
    call xzdtable(3, neomsig)
  end if
  call xzrel(m0)

contains

  ! the problem number ish: its shift in classes, its roots; ok: it can be done
  subroutine problem(ip, sc, nr, good)
    integer(8), intent(in) :: ip
    integer(8), intent(out) :: sc, nr
    logical, intent(out) :: good
    good = .true.
    if (dense) then
      sc = 0
      nr = sum(enr(1:enshift))
      return
    end if
    nr = enr(ip)
    sc = eshl(ip) / labstep
    if (mod(eshl(ip), labstep) /= 0 .or. abs(sc) > ntg - 1) then
      good = .false.
      if (.not. xzdry) write(6, '(/,a,i6,a,i4,a)') ' @XZCCSDT-W, EOM: the shift ', eshl(ip), &
        ' is not a difference of sums of the labels of this run (their step: ', labstep, '); it is left out.'
    end if
  end subroutine

  integer(8) function subspace(nr)
    integer(8), intent(in) :: nr
    ! (CFOUR's 20 + roots + 2, and - every root that is not converged adds a
    ! vector per iteration - four vectors per root, as the Python solver: with
    ! two per root the subspace is compacted every iteration (40 roots of H2O:
    ! 2107 sigma vectors, 12 roots not converged in 120 iterations, against 923
    ! with 4 roots + 4); given by %eomndav0: at least 2 roots + 2, so that
    ! every root keeps its vector and adds one (40 roots in 50 vectors: 21
    ! converged in 200 iterations))
    subspace = emaxsub
    if (subspace <= 0) subspace = max(20 + nr + 2, 4 * nr + 8)
    subspace = max(subspace, 2 * nr + 2)
  end function
end subroutine

real(8) function N(xzeimag)(x)
  implicit none
  ZT, intent(in) :: x
#ifdef XZ_REAL
  N(xzeimag) = 0d0
#else
  N(xzeimag) = aimag(x)
#endif
end function

! ---------------------------------------------------------------------------
! option eomtest seed: the sigma vector of a random totally symmetric vector R
! against the derivative of the residual of the amplitude step along R,
!   [8 (Res(T + R) - Res(T - R)) - (Res(T + 2R) - Res(T - 2R))] / 12,
! which is exact for a polynomial of degree 4 (docs/eom.md, section 1).  The
! residual: (new amplitudes - amplitudes) x denominators of a Jacobi step
! (xzstep).  Not planned: takes five more arrays of the size of the vector.
subroutine N(xzeomtest)(w, seedtext)
  use xzdata
  use xzmem
  use xzkern
  use xzeomd
  implicit none
  real(8) :: w(*)
  character(len=*), intent(in) :: seedtext
  integer(8) :: nsd, nlen, kr, kfd, ksg, kt1, kt2, kt3, kt1n, kt2n, kr3, kt2r, ov, k, seed, nseed, i, ic
  integer, allocatable :: sd(:)
  real(8) :: cf(4), cw(4), d1, d2, d3, n1, n2, n3n, scale
  ZT :: s
  integer :: isd
  real(8), allocatable :: rnd(:)
  read(seedtext, *) seed
  ov = no * nv
  nsd = ov + npo * nvp
  call xzeomlay(w, 0_8)
  nlen = nsd + n3s
  kr = xzget(NW * nlen, 'test: R')
  kfd = xzget(NW * nlen, 'test: derivative')
  ksg = xzget(NW * nlen, 'test: sigma')
  kt1 = xzget(NW * ov, 'test: T1')
  kt2 = xzget(NW * ov * ov, 'test: T2')
  kt3 = xzget(NW * n3, 'test: T3')
  kt1n = xzget(NW * ov, 'test: new T1')
  kt2n = xzget(NW * ov * ov, 'test: new T2')
  kr3 = xzget(NW * n3, 'test: new T3')
  kt2r = xzget(NW * ov * ov, 'test: R2')
  ! a random vector of the shift 0
  call random_seed(size=isd)
  allocate(sd(isd))
  sd = int(seed)
  call random_seed(put=sd)
  allocate(rnd(NW * nlen))
  call random_number(rnd)
  scale = 0.05d0
  do i = 1, NW * nlen
    w(kr + i - 1) = scale * (rnd(i) - 0.5d0)
  end do
  deallocate(rnd)
  if (ncls > 1) call N(xzemask)(no, nv, npo, nvp, 0_8, ocls, vcls, w(jopgrp), w(jvpgrp), w(kr), w(kr + NW * ov))
  call N(xzunpack2)(no, nv, npo, nvp, w(jopa), w(jopb), w(jvpa), w(jvpb), w(kr + NW * ov), w(kt2r))
  call zc_vzero_d(NW * nlen, w(kfd))
  cf = [1d0, -1d0, 2d0, -2d0]
  cw = [8d0, -8d0, -1d0, 1d0] / 12d0
  do ic = 1, 4
    call N(xzetaxpy)(ov, cf(ic), w(jt1), w(kr), w(kt1))
    call N(xzetaxpy)(ov * ov, cf(ic), w(jt2), w(kt2r), w(kt2))
    call N(xzetaxpy)(n3, cf(ic), w(jt3), w(kr + NW * nsd), w(kt3))
    call N(xzstepm)(w, kt1, kt2, kt3, kt1n, kt2n, kr3)
    call N(xzetres)(w, cw(ic), w(kt1), w(kt2), w(kt3), w(kt1n), w(kt2n), w(kr3), w(kfd), w(kfd + NW * ov), &
                    w(kfd + NW * nsd))
  end do
  call N(xzeomim)(w)
  call N(xzesig)(w, kr, ksg, .false.)
  call N(xzdiff2)(ov, w(kfd), w(ksg), d1)
  call N(xzdiff2)(npo * nvp, w(kfd + NW * ov), w(ksg + NW * ov), d2)
  call N(xzdiff2)(n3s, w(kfd + NW * nsd), w(ksg + NW * nsd), d3)
  call N(zc_vdot)(ov, w(ksg), w(ksg), s)
  n1 = dble(s)
  call N(zc_vdot)(npo * nvp, w(ksg + NW * ov), w(ksg + NW * ov), s)
  n2 = dble(s)
  call N(zc_vdot)(n3s, w(ksg + NW * nsd), w(ksg + NW * nsd), s)
  n3n = dble(s)
  write(6, '(/,a)') ' option eomtest: sigma of a random vector against the derivative of the amplitude step'
  write(6, '(a,3es14.6)') '   |sigma_1|, |sigma_2|, |sigma_3|:        ', sqrt(n1), sqrt(n2), sqrt(n3n)
  write(6, '(a,3es14.6)') '   |sigma - derivative|, the three ranks:  ', sqrt(d1), sqrt(d2), sqrt(d3)
end subroutine

! c = a + alpha b
subroutine N(xzetaxpy)(n, alpha, a, b, c)
  implicit none
  integer(8), intent(in) :: n
  real(8), intent(in) :: alpha
  ZT, intent(in) :: a(n), b(n)
  ZT, intent(out) :: c(n)
  integer(8) :: i
  !$omp parallel do
  do i = 1, n
    c(i) = a(i) + alpha * b(i)
  end do
end subroutine

! f += cw * residual, residual = (new - old) x denominator of the Jacobi step
subroutine N(xzetres)(w, cw, t1, t2, t3, t1n, t2n, t3n, f1, f2, f3)
  use xzdata
  implicit none
  real(8) :: w(*)
  real(8), intent(in) :: cw
  ZT, intent(in) :: t1(0:nv-1, 0:no-1), t1n(0:nv-1, 0:no-1)
  ZT, intent(in) :: t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1), t2n(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: t3(0:*), t3n(0:*)
  ZT, intent(inout) :: f1(0:nv-1, 0:no-1), f2(0:nvp-1, 0:npo-1), f3(0:*)
  integer(8) :: i, a
  do i = 0, no - 1
    do a = 0, nv - 1
      f1(a, i) = f1(a, i) + cw * (t1n(a, i) - t1(a, i)) * (w(jeps + i) - w(jeps + no + a))
    end do
  end do
  call N(xzetres2)(cw, t2, t2n, w(jopa), w(jopb), w(jvpa), w(jvpb), w(jeps), f2)
  call N(xzetres3)(cw, t3, t3n, w(jeo3), w(jev3), f3)
end subroutine

subroutine N(xzetres2)(cw, t2, t2n, opa, opb, vpa, vpb, eps, f2)
  use xzdata
  implicit none
  real(8), intent(in) :: cw
  ZT, intent(in) :: t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1), t2n(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  real(8), intent(in) :: eps(0:no+nv-1)
  ZT, intent(inout) :: f2(0:nvp-1, 0:npo-1)
  integer(8) :: p, q, ii, jj, aa, bb
  do p = 0, npo - 1
    ii = opa(p)
    jj = opb(p)
    do q = 0, nvp - 1
      aa = vpa(q)
      bb = vpb(q)
      f2(q, p) = f2(q, p) + cw * (t2n(bb, aa, jj, ii) - t2(bb, aa, jj, ii)) &
                            * (eps(ii) + eps(jj) - eps(no + aa) - eps(no + bb))
    end do
  end do
end subroutine

subroutine N(xzetres3)(cw, t3, t3n, eo3, ev3, f3)
  use xzdata
  implicit none
  real(8), intent(in) :: cw
  ZT, intent(in) :: t3(0:*), t3n(0:*)
  real(8), intent(in) :: eo3(0:nto-1), ev3(0:nvt-1)
  ZT, intent(inout) :: f3(0:*)
  integer(8) :: g, it, ia, k, na
  do g = 0, ntg - 1
    na = vtoff(g + 1) - vtoff(g)
    do it = otoff(g), otoff(g + 1) - 1
      do ia = 0, na - 1
        k = boff(g) + (it - otoff(g)) * na + ia
        f3(k) = f3(k) + cw * (t3n(k) - t3(k)) * (eo3(it) - ev3(vtoff(g) + ia))
      end do
    end do
  end do
end subroutine

