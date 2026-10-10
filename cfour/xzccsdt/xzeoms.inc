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

! ---------------------------------------------------------------------------
! The place of the subspace of evcap vectors of nlen elements: the work
! array, or files.  avail: words that are free for it.  With files the
! number of vectors is lowered to what the disk holds (at least nmin).
subroutine N(xzevopen)(nlen, msub, nmin, avail, mayfile)
  use xzdata
  use xzmem
  use xzkern
  use xzeomd
  implicit none
  integer(8), intent(in) :: nlen, nmin, avail
  integer(8), intent(inout) :: msub
  logical, intent(in) :: mayfile
  integer(8) :: need, free_bytes, per, nfit
  character(len=8) :: arg
  integer :: larg
  evlen = nlen
  evnc = 0
  evnh = 0
  need = 2 * msub * (NW * nlen + xzalign)
  evfile = need > avail
  ! a test hook: XZ_EOMVEC=f (files), m (the work array)
  call get_environment_variable('XZ_EOMVEC', arg, larg)
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
end subroutine

! the files are closed and deleted
subroutine N(xzevclose)()
  use xzeomd
  implicit none
  integer(8) :: j
  integer :: ios
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
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer, intent(in) :: kind
  integer(8), intent(in) :: j, kx
  integer(8) :: k
  integer :: ios, u
  real(8) :: t0
  if (.not. evfile) then
    k = merge(evc0, evh0, kind == 1) + (j - 1) * NW * evlen
    call N(xzecopy)(evlen, w(kx), w(k))
    return
  end if
  t0 = omp_get_wtime()
  u = xzevunit(kind, j)
  open(unit=u, file=xzevname(kind, j), form='unformatted', access='stream', status='replace', iostat=ios)
  if (ios == 0) write(u, iostat=ios) w(kx:kx + NW * evlen - 1)
  if (ios == 0) flush(u, iostat=ios)
  if (ios /= 0) then
    write(6, '(/,a,a,a,i8,a)') ' @XZCCSDT-F, writing the file ', xzevname(kind, j), ' failed (iostat ', ios, '): disk full?'
    write(6, '(a,f12.1,a)') '   a vector of the EOM subspace: ', 8d-6 * NW * evlen, ' MB'
    call xzerrex
  end if
  if (kind == 1) evnc = max(evnc, j)
  if (kind == 2) evnh = max(evnh, j)
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
  if (.not. evfile) then
    do j = 1, n
      k = merge(evc0, evh0, kind == 1) + (j - 1) * NW * evlen + NW * off
      ptr(j) = loc(w(k))
    end do
    return
  end if
  t0 = omp_get_wtime()
  do j = 1, n
    k = evbuf + (j - 1) * NW * evnch
    read(xzevunit(kind, j), pos=1 + 8 * NW * off, iostat=ios) w(k:k + NW * nc - 1)
    if (ios /= 0) then
      write(6, '(/,a,a,a,i8)') ' @XZCCSDT-F, reading the file ', xzevname(kind, j), ' failed, iostat ', ios
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
  step = evlen
  if (evfile) step = evnch
  do off = 0, evlen - 1, step
    nc = min(step, evlen - off)
    call N(xzevchunk)(w, kind, n, off, nc, ptr)
    call N(zc_vmdot)(nc, n, w(kx + NW * off), ptr, part)
    res = res + part
  end do
end subroutine

! x = [x +] sum_j coef(j) X_j for the vector x at kx
subroutine N(xzevcomb)(w, kind, n, coef, kx, add)
  use xzdata
  use xzeomd
  use xzkern
  implicit none
  real(8) :: w(*)
  integer, intent(in) :: kind
  integer(8), intent(in) :: n, kx
  ZT, intent(in) :: coef(n)
  logical, intent(in) :: add
  ZT :: c(n + 1)
  integer(c_intptr_t) :: ptr(n + 1)
  integer(8) :: off, nc, step, nk
  if (n == 0) then
    if (.not. add) call zc_vzero_d(NW * evlen, w(kx))
    return
  end if
  c(1:n) = coef
  c(n + 1) = 1
  nk = n
  if (add) nk = n + 1
  step = evlen
  if (evfile) step = evnch
  do off = 0, evlen - 1, step
    nc = min(step, evlen - off)
    call N(xzevchunk)(w, kind, n, off, nc, ptr)
    ptr(n + 1) = loc(w(kx + NW * off))
    call N(zc_vlincomb)(nc, nk, ptr, c, w(kx + NW * off))
  end do
end subroutine

! The first m vectors of both kinds become sum_j X_j q[j][k] (k < m) of the
! n there are; the others are dropped.
subroutine N(xzevtrans)(w, n, m, q)
  use xzdata
  use xzeomd
  use xzkern
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: n, m
  ZT, intent(in) :: q(*)                                       ! [j][k], n x m
  integer(c_intptr_t) :: ptr(n)
  integer(8) :: off, nc, step, j, k
  integer :: kind, ios
  real(8) :: t0
  step = evlen
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
            write(6, '(/,a,a,a,i8,a)') ' @XZCCSDT-F, writing the file ', xzevname(kind, j), ' failed (iostat ', ios, ')'
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
! largest element of the residual c0 (nd elements) against the threshold.
subroutine N(xzeconv)(c0, e, idav, icyc, nd, ieomconv, eomthr)
  implicit none
  integer(8), intent(in) :: nd, icyc, idav
  ZT, intent(in) :: c0(nd)
  complex(8), intent(in) :: e
  integer(8), intent(out) :: ieomconv
  real(8), intent(in) :: eomthr
#ifdef XZ_REAL
  real(8) :: z
  integer(8) :: i
  z = 0d0
  !$omp parallel do reduction(max:z)
  do i = 1, nd
    z = max(z, abs(c0(i)))
  end do
  ieomconv = 1
  if (z > eomthr) ieomconv = 0
  write(6, '("EOM cycle",3X,I4,2X,I4,F22.15,5X,D20.10,5X,D20.10)') icyc, idav, dble(e), z, eomthr
#else
  call eomconv(c0, e, idav, icyc, nd, ieomconv, eomthr)
#endif
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
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kx
  ZT, intent(in) :: omega
  logical, intent(in) :: sd
  real(8) :: eo2(0:max(npo, 1_8) - 1), ev2(0:max(nvp, 1_8) - 1)
  integer(8) :: nsd, g, ni, na
  real(8), parameter :: floor = 1d-3
  nsd = no * nv + npo * nvp
  call N(zc_vprecond)(w(kx), no, nv, w(jeps), w(jeps + no), omega, floor)
  call xzepsum(npo, no, w(jopa), w(jopb), w(jeps), eo2)
  call xzepsum(nvp, nv, w(jvpa), w(jvpb), w(jeps + no), ev2)
  if (npo > 0) call N(zc_vprecond)(w(kx + NW * no * nv), npo, nvp, eo2, ev2, omega, floor)
  if (sd) return
  if (ncls == 1) then
    call N(zc_vprecond)(w(kx + NW * nsd), nto, nvt, w(jeo3), w(jev3), omega, floor)
  else
    do g = 0, ntg - 1
      ni = otoff(g + 1) - otoff(g)
      na = xzna(eshift, g)
      if (ni == 0 .or. na == 0) cycle
      call N(zc_vprecond)(w(kx + NW * (nsd + boff_s(g))), ni, na, w(jeo3 + otoff(g)), &
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
  call N(zc_vdot)(evlen, w(kx), w(kx), s)
  nrm0 = sqrt(dble(s))
  rel = 0d0
  if (nrm0 == 0d0) return
  call N(xzevscal)(evlen, 1d0 / nrm0, w(kx))
  nrm = 1d0
  do pass = 1, 2
    if (n > 0) then
      call N(xzevmdot)(w, 1, n, kx, d)
      d(1:n) = -CONJ(d(1:n))
      call N(xzevcomb)(w, 1, n, d, kx, .true.)
    end if
    call N(zc_vdot)(evlen, w(kx), w(kx), s)
    nrm = sqrt(dble(s))
    if (pass == 1) rel = nrm
    if (nrm < 1d-12) then
      rel = 0d0
      return
    end if
    call N(xzevscal)(evlen, 1d0 / nrm, w(kx))
  end do
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
  integer(8) :: nsd, nlen, n, n0, nconv, it, k, j, g, c, ntarget, nun, mkeep, nadded, icv, mv, jbest, nkeep, i
  ZT, allocatable :: gm(:, :), ym(:, :), cr(:, :), coef(:), dots(:), qm(:, :), qr(:), gq(:, :)
  complex(8), allocatable :: ev(:)
  logical, allocatable :: flag(:)
  integer(8), allocatable :: keep(:), ccyc(:)
  real(8) :: rel, z, zn, ov, best, t0
  ZT :: s, om
  logical :: needdiag
  integer :: ios

  nsd = no * nv + npo * nvp
  nlen = nsd
  if (.not. sd) nlen = nsd + n3s
  nfound = 0
  nsig = 0
  nit = 0
  mv = xzmark()
  call N(xzevopen)(nlen, msub, min(msub, 2 * nroot + 2), avail, .not. sd)
  allocate(gm(msub, msub), ym(msub, msub), cr(msub, nroot), coef(msub), dots(msub), ev(msub), flag(msub), &
           keep(msub), ccyc(nroot), qm(msub, msub), qr(msub * msub), gq(msub, msub))
  gm = 0
  n = 0
  ! ---- the vectors to start from
  do g = 1, nstart
    if (n >= msub) exit
    call zc_vzero_d(NW * nlen, w(kxa))
    if (kstart > 0) then
      call N(xzecopy)(nsd, w(kstart + NW * nsd * (g - 1)), w(kxa))
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
    return
  end if

  nconv = 0
  ccyc = 0
  needdiag = .true.
  do it = 1, maxit
    nit = it
    call N(xzesmall)(n, gm, msub, ev, ym)
    needdiag = .false.
    ! the converged roots among the eigenvectors, by their overlap
    call mark_converged()
    ntarget = min(nroot, n)
    nun = count(.not. flag(1:ntarget))
    if (nun == 0) exit
    ! compaction: the Ritz vectors of the converged roots and the lowest others
    if (n + nun > msub) then
      mkeep = min(n, max(nroot + 2, 1_8), msub - nun)
      call compact()
      call N(xzesmall)(n, gm, msub, ev, ym)
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
      if (quiet) then
        call N(xzevmax)(nlen, w(kxa), z)
        icv = merge(1_8, 0_8, z <= thr)
      else
        call N(xzeconv)(w(kxa), ev(k), n0, it, nlen, icv, thr)
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
      if (n >= msub) cycle
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
      exit
    end if
  end do

  ! ---- the roots: their vectors, the residuals once more, the weights
  if (needdiag) then
    call N(xzesmall)(n, gm, msub, ev, ym)
    call mark_converged()
  end if
  nfound = min(nroot, n)
  do k = 1, nfound
    coef(1:n) = ym(1:n, k)
    call N(xzevcomb)(w, 1, n, coef, kxa, .false.)
    call N(xzevcomb)(w, 2, n, coef, kxb, .false.)
#ifdef XZ_REAL
    om = dble(ev(k))
#else
    om = ev(k)
#endif
    call N(xzeaxpyc)(nlen, -om, w(kxa), w(kxb))
    call N(xzevmax)(nlen, w(kxb), z)
    call N(zc_vdot)(nlen, w(kxb), w(kxb), s)
    ee(k) = ev(k)
    xmx(k) = z
    xnm(k) = sqrt(dble(s))
    cyc(k) = 0
    do c = 1, nconv
      ov = abs(sum(CONJ(cr(1:n, c)) * ym(1:n, k)))
      if (ov > 0.7d0) cyc(k) = ccyc(c)
    end do
    call N(zc_vdot)(no * nv, w(kxa), w(kxa), s)
    wts(1, k) = dble(s)
    call N(zc_vdot)(npo * nvp, w(kxa + NW * no * nv), w(kxa + NW * no * nv), s)
    wts(2, k) = dble(s)
    wts(3, k) = 0d0
    if (.not. sd) then
      call N(zc_vdot)(n3s, w(kxa + NW * nsd), w(kxa + NW * nsd), s)
      wts(3, k) = dble(s)
    end if
    call N(xzelead)(no * nv, w(kxa), nlead, lead(1, k), vlead(1, k))
    if (kout > 0) call N(xzecopy)(nsd, w(kxa), w(kout + NW * nsd * (k - 1)))
    if (iures > 0) then
      write(iures, iostat=ios) eshift, k, nlen, dble(ev(k)), aimag(ev(k)), w(kxa:kxa + NW * nlen - 1)
      if (ios /= 0) then
        write(6, '(/,a,i8,a)') ' @XZCCSDT-F, writing the file XZEOMR failed (iostat ', ios, '): disk full?'
        call xzerrex
      end if
    end if
  end do
  call N(xzevclose)()
  call xzrel(mv)

contains

  ! the vector at kxa joins the basis: its sigma vector, the new row and
  ! column of the projected matrix
  subroutine add_vector()
    real(8) :: ts
    call N(xzevput)(w, 1, n + 1, kxa)
    ts = omp_get_wtime()
    call N(xzesig)(w, kxa, kxb, sd)
    if (sd) then
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
      if (jbest > 0) flag(jbest) = .true.
    end do
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
  use omp_lib
  implicit none
  real(8) :: w(*)
  real(8), intent(in) :: ecorr
  integer(8) :: m0, ms, ish, s, nsd, nlen, kxa, kxb, kst, nroot, nkeep, nsing, msd, msub, k, peak, sigw, avail
  integer(8) :: nfound, nsig, nit, nsgd, nitd, nst, i, a, r, nprob
  integer(8), allocatable :: singles(:), lead(:, :), cyc(:)
  complex(8), allocatable :: ee(:)
  real(8), allocatable :: rmax(:), rnorm(:), wts(:, :)
  ZT, allocatable :: vlead(:, :)
  real(8) :: t0, t1, tsh
  logical :: ok, dense
  character(len=256) :: arg
  integer :: larg, ios, iures
  real(8), external :: N(xzeimag)
  real(8), parameter :: ev = 27.211386245988d0, cm = 219474.6313632d0

  nsd = no * nv + npo * nvp
  dense = ncls == 1
  m0 = xzmark()
  call N(xzeomkeep)()
  if (.not. dense) jobase_s = xzget(nto, 'layout of the label shift')
  call xzeomstep()
  nprob = enshift
  if (dense) nprob = 1

  if (xzdry) then
    call N(xzeomim)(w)
    do ish = 1, nprob
      call problem(ish, s, nroot, ok)
      if (.not. ok) cycle
      call xzeomlay(w, s)
      nlen = nsd + n3s
      nkeep = nroot + 2
      msd = min(64_8, 8 + 4 * nkeep)
      msub = subspace(nroot)
      ms = xzmark()
      kst = xzget(NW * nkeep * nsd, 'EOM: vectors to start from')
      kxa = xzget(NW * nlen, 'EOM: a vector')
      kxb = xzget(NW * nlen, 'EOM: its sigma vector')
      ! (the least: the singles-doubles stage in the work array, or the buffer of the files)
      k = xzget(max(2 * msd * (NW * nsd + xzalign), msub * NW * min(nlen, 1024_8) + 64), 'EOM: the subspace')
      call N(xzesig)(w, kxa, kxb, .false.)
      call xzrel(ms)
    end do
    call xzrel(m0)
    return
  end if

  ! ---- a test hook: XZ_EOMTEST=seed, the sigma vector of a random vector
  !      against the derivative of the amplitude step (before the pp operator
  !      takes the place of <ab||cd>)
  call get_environment_variable('XZ_EOMTEST', arg, larg)
  if (larg > 0) then
    call N(xzeomtest)(w, arg(1:larg))
    call xzrel(m0)
    return
  end if

  t0 = omp_get_wtime()
  call N(xzeomim)(w)
  teomim = omp_get_wtime() - t0
  write(6, '(/,a)') '  EOM-EE-CCSDT: excitation energies and right-hand vectors'
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

  ! ---- a test hook: XZ_EOMSIG=file, sigma of the vector of the file (of the
  !      first shift), written to file.out
  call get_environment_variable('XZ_EOMSIG', arg, larg)
  if (larg > 0) then
    call problem(1_8, s, nroot, ok)
    call xzeomlay(w, s)
    nlen = nsd + n3s
    kxa = xzget(NW * nlen, 'EOM: a vector')
    kxb = xzget(NW * nlen, 'EOM: its sigma vector')
    open(unit=74, file=arg(1:larg), form='unformatted', access='stream', status='old', iostat=ios)
    if (ios == 0) read(74, iostat=ios) w(kxa:kxa + NW * nlen - 1)
    close(74)
    if (ios /= 0) then
      write(6, '(a,i12,a)') ' @XZCCSDT-F, XZ_EOMSIG: the file does not hold a vector of ', nlen, ' elements.'
      call xzerrex
    end if
    call get_environment_variable('XZ_EOMSD', length=larg)
    call N(xzesig)(w, kxa, kxb, larg > 0)
    call get_environment_variable('XZ_EOMSIG', arg, larg)
    open(unit=74, file=arg(1:larg) // '.out', form='unformatted', access='stream', status='replace')
    write(74) w(kxb:kxb + NW * nlen - 1)
    close(74)
    write(6, '(a,i12,a,a)') ' XZ_EOMSIG: sigma of a vector of ', nlen, ' elements written to ', arg(1:larg) // '.out'
    call xzrel(m0)
    return
  end if

  iures = 78
  open(unit=iures, file='XZEOMR', form='unformatted', access='stream', status='replace', iostat=ios)
  if (ios == 0) write(iures, iostat=ios) 'XZEOMR01', no, nv, int(NW, 8), ncls, labstep
  if (ios /= 0) then
    write(6, '(a)') ' @XZCCSDT-W, the file XZEOMR of the vectors cannot be written: no vectors are kept.'
    iures = 0
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
    ms = xzmark()
    kst = xzget(NW * nkeep * nsd, 'EOM: vectors to start from')
    kxa = xzget(NW * nlen, 'EOM: a vector')
    kxb = xzget(NW * nlen, 'EOM: its sigma vector')
    allocate(ee(nkeep), rmax(nkeep), rnorm(nkeep), wts(3, nkeep), cyc(nkeep), lead(nlead, nkeep), vlead(nlead, nkeep))
    nst = nroot
    nsgd = 0
    if (eguess == 1) then
      ! ---- the singles-doubles block of the matrix, from unit vectors
      t1 = omp_get_wtime()
      msd = min(64_8, 8 + 4 * nkeep)
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
    call xzdry_begin()
    call N(xzesig)(w, kxa, kxb, .false.)
    call xzdry_end(peak)
    sigw = peak - xztop
    avail = xzsize - xztop - sigw
    t1 = omp_get_wtime()
    evio = 0d0
    evbytes = 0d0
    if (2 * msub * (NW * nlen + xzalign) <= avail) then
      write(6, '(a,i4,a)') '   subspace: at most ', msub, ' vectors and their sigma vectors, in the work array'
    else
      write(6, '(a,i4,a)') '   subspace: at most ', msub, ' vectors and their sigma vectors, in the files EOMCnnnn, EOMHnnnn'
    end if
    write(6, '(a,f10.1,a,f10.1,a,f10.1,a)') '     (', 16d-6 * NW * nlen * msub, ' MB; a sigma vector takes ', 8d-6 * sigw, &
      ' MB of the work array, free besides: ', 8d-6 * avail, ' MB)'
    flush(6)
    if (eguess == 1) then
      call N(xzesolve)(w, .false., nroot, ethr, emaxcyc, msub, nst, kst, singles, kxa, kxb, avail, &
                       ee, rmax, rnorm, wts, cyc, nfound, nsig, nit, 0_8, iures, lead, vlead, .false.)
    else
      call N(xzesolve)(w, .false., nroot, ethr, emaxcyc, msub, nst, 0_8, singles, kxa, kxb, avail, &
                       ee, rmax, rnorm, wts, cyc, nfound, nsig, nit, 0_8, iures, lead, vlead, .false.)
    end if
    t1 = omp_get_wtime() - t1
    ! ---- the roots
    write(6, '(/,a,i5,a,i4,a,f10.2,a,f9.2,a)') '   sigma vectors: ', nsig, '   iterations: ', nit, '   time: ', t1, &
      ' s   (', t1 / max(nsig, 1_8), ' s per sigma vector, the solver included)'
    if (evbytes > 0d0) write(6, '(a,f12.1,a,f10.2,a)') '   vectors in files: ', 1d-9 * evbytes, &
      ' GB read and written in ', evio, ' s'
    write(6, '(/,a)') '   root        omega / Eh            eV            cm-1    max |r|      |r|   cycle    R1     R2     R3' // &
                      '          total energy / Eh'
    do k = 1, nfound
      write(6, '(i7,f20.12,f14.8,f16.4,2es10.2,i6,3f7.4,f26.12)') k, dble(ee(k)), dble(ee(k)) * ev, dble(ee(k)) * cm, &
        rmax(k), rnorm(k), cyc(k), wts(:, k), escf + ecorr + dble(ee(k))
      if (abs(aimag(ee(k))) > 1d-10) write(6, '(a,es12.4)') '          imaginary part of omega: ', aimag(ee(k))
      if (rmax(k) > ethr) write(6, '(a)') '          NOT converged'
      if (nres < maxres) then
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
    write(6, '(a,f10.2,a)') '   time of this problem: ', omp_get_wtime() - tsh, ' s'
    flush(6)
    deallocate(singles, ee, rmax, rnorm, wts, cyc, lead, vlead)
    call xzrel(ms)
  end do
  if (iures > 0) close(iures)
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
    subspace = emaxsub
    if (subspace <= 0) subspace = 20 + nr + 2
    subspace = max(min(subspace, 64_8), 2 * nr + 2)
    subspace = min(subspace, 64_8)
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
! XZ_EOMTEST=seed: the sigma vector of a random totally symmetric vector R
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
    call N(xzstep)(w, kt1, kt2, kt3, kt1n, kt2n, kr3)
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
  write(6, '(/,a)') ' XZ_EOMTEST: sigma of a random vector against the derivative of the amplitude step'
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
