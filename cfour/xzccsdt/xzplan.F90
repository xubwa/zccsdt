!
! The memory plan: the batch sizes of the CCSDT step and of (Q) and the place
! of the DIIS vectors, chosen for the MAXCOR words of the work array before
! anything is placed.
!
! The words that a set of batch sizes takes are not computed from formulae:
! the scheduler itself is run dry (xzmem: xzdry), through every allocation of
! the layout of the integrals, of the iterations and of (Q), and its peak is
! read off.  The plan and the placement therefore cannot disagree.
!
! The rule (monotone: no batch shrinks when the memory grows):
!   1. the floor: every batch at its smallest, the T3 parts of the DIIS
!      vectors in the file XZDIIS.  If the floor does not fit, the program
!      stops here and says how many words it needs.
!   2. the batches are raised to their targets one after the other, each as
!      far as it fits before the next starts:
!        rows of a slab of the dressed <ab||cd>    up to all pairs a<b
!        occupied triples of a pp-ladder batch     up to 1.3 GB of buffers
!        occupied pairs of a batch of the passes   up to 2 GB of buffers
!        first indices of a batch of W_vvvo terms  up to nv / 8
!   3. the DIIS vectors go to the work array if all of them fit there next
!      to that; otherwise they stay in the file - with fewer vectors if the
!      disk of the run directory does not hold them all.
!   (Q) has the work array to itself after the iterations: Xv for more
!   occupied indices first, then the two products in fewer batches of
!   columns, then more rows per step of the formation of Xv.
!
module xzpland
  implicit none
  integer, parameter :: npar = 8
  ! 1 nqs, 2 nbt, 3 nbp, 4 nba, 5 qxv, 6 qc22, 7 qc13, 8 qrows
  integer(8) :: plo(npar), phi(npar)
  integer(8) :: pslot = 0, pchunk = 0
  logical :: pfile = .true.
  logical :: planq = .false.          ! only (Q) is planned (a rank that does no iterations)
contains
  subroutine setpar(k, v)
    use xzdata
    integer, intent(in) :: k
    integer(8), intent(in) :: v
    select case (k)
    case (1)
      nqs = v
    case (2)
      nbt = v
    case (3)
      nbp = v
    case (4)
      nba = v
    case (5)
      qxv = v
    case (6)
      qc22 = v
    case (7)
      qc13 = v
    case (8)
      qrows = v
    end select
  end subroutine
end module xzpland

! words of the work array that the layout and the iterations take
integer(8) function xzpeak_cc(w)
  use xzdata
  use xzmem
  use xzpland
  implicit none
  real(8) :: w(*)
  real(8) :: e
  logical :: c
  integer(8) :: n, peak, status
  call xzdry_begin(0_8)
  call xzlayout(w, status)
  if (nwe == 2) then
    call xzcc_z(w, pslot, pfile, pchunk, e, c, n)
  else
    call xzcc_d(w, pslot, pfile, pchunk, e, c, n)
  end if
  call xzdry_end(peak)
  xzpeak_cc = peak
end function xzpeak_cc

! words that the layout, the amplitudes and (Q) take
integer(8) function xzpeak_q(w)
  use xzdata
  use xzmem
  implicit none
  real(8) :: w(*)
  real(8) :: e5, e6
  integer(8) :: peak, k, status
  call xzdry_begin(0_8)
  call xzlayout(w, status)
  k = xzget(nwe * no * nv, 'T1')
  k = xzget(nwe * no * no * nv * nv, 'T2')
  k = xzget(nwe * n3, 'T3')
  if (with_q) then
    if (nwe == 2) then
      call xzq_z(w, e5, e6)
    else
      call xzq_d(w, e5, e6)
    end if
  end if
  call xzdry_end(peak)
  xzpeak_q = peak
end function xzpeak_q

! ---------------------------------------------------------------------------
! flooronly: only the check of the floor, which says nothing unless it fails
subroutine xzplan(w, nslot, usefile, nch, flooronly)
  use xzdata
  use xzmem
  use xzkern
  use xzpland
  implicit none
  real(8) :: w(*)
  integer(8), intent(out) :: nslot, nch
  logical, intent(out) :: usefile
  logical, intent(in) :: flooronly
  integer(8), external :: xzpeak_cc, xzpeak_q
  integer(8) :: f, floor_cc, floor_q, need_cc, need_q, free_bytes, per_vec, nfit, k8
  integer :: k
  character(len=8) :: arg
  integer :: larg

  f = no * nv * nvp                       ! elements of a half-expanded batch of one pair
  ! ---- the smallest and the target batch sizes
  plo(1) = min(nv, nqmax)
  phi(1) = nqmax
  plo(2) = 1
  phi(2) = max(1_8, min(nto, 1300000000_8 / (2 * 8 * nwe * nvp * nv)))
  plo(3) = 1
  phi(3) = max(1_8, min(npo, 2000000000_8 / (3 * 8 * nwe * f)))
  plo(4) = 1
  phi(4) = max(1_8, nv / 8)
  plo(5) = 3
  phi(5) = max(3_8, no - 1)
  plo(6) = (nvp + 15) / 16
  phi(6) = max(plo(6), min(nvp, 1073741824_8 / (8 * nwe * nvp)))
  plo(7) = (nvt + 15) / 16
  phi(7) = max(plo(7), min(nvt, 1073741824_8 / (8 * nwe * nv)))
  plo(8) = (nvp + 15) / 16
  phi(8) = max(plo(8), min(nvp, 1073741824_8 / (16 * nwe * nv * nv)))
  ncb = 0
  do k = 1, npar
    call setpar(k, plo(k))
  end do

  ! ---- the floor
  pfile = .true.
  pslot = ndiis
  pchunk = min(max(n3, 1_8), 1048576_8)
  floor_cc = 0
  if (.not. planq) floor_cc = xzpeak_cc(w)
  floor_q = xzpeak_q(w)
  if (flooronly .and. max(floor_cc, floor_q) <= xzsize) return
  write(6, '(/,a,a,a)') '   Memory plan (words of 8 bytes; ', merge('real numbers   ', 'complex numbers', nwe == 1), ')'
  write(6, '(a,i16,a,f10.1,a)') '     the least that CCSDT takes:     ', floor_cc, '  = ', 8d-6 * floor_cc, ' MB'
  if (with_q) write(6, '(a,i16,a,f10.1,a)') '     the least that (Q) takes:       ', floor_q, '  = ', &
    8d-6 * floor_q, ' MB'
  write(6, '(a,i16,a,f10.1,a)') '     the work array has:             ', xzsize, '  = ', 8d-6 * xzsize, ' MB'
  if (max(floor_cc, floor_q) > xzsize) then
    write(6, '(/,a)') ' @XZCCSDT-F, the work array is too short for this calculation.'
    if (nwe == 1) write(6, '(a)') '   (even if all integrals are real; complex integrals take about twice as much)'
    write(6, '(a,i16,a,i16)') '   words needed: ', max(floor_cc, floor_q), '   words there: ', xzsize
    write(6, '(a,i16,a,f12.1,a)') '   Set MEMORY_SIZE to at least ', max(floor_cc, floor_q) + 64, &
      ' words (', 8d-6 * (max(floor_cc, floor_q) + 64), ' MB).'
    call errex
  end if

  ! ---- the batches of the iterations, one after the other
  if (.not. planq) then
  do k = 1, 4
    call raise(k, .true.)
  end do
  end if
  ! ---- the DIIS vectors: the work array if all of them fit, else the file
  pfile = .false.
  if (planq) then
    pslot = 0
  else if (xzpeak_cc(w) > xzsize) then
    pfile = .true.
    per_vec = 2 * 8 * nwe * n3
    free_bytes = xz_free_disk('.' // c_null_char)
    if (free_bytes >= 0 .and. per_vec * pslot > free_bytes - 1073741824_8) then
      nfit = max(0_8, (free_bytes - 1073741824_8) / max(per_vec, 1_8))
      write(6, '(a,i4,a,f10.1,a)') ' @XZCCSDT-W, the disk of the run directory holds ', nfit, &
        ' DIIS vectors of T3 (', 1d-6 * per_vec, ' MB each).'
      ! fewer vectors in the file, or as many as fit in the work array if that is more
      k8 = 0
      pfile = .false.
      do pslot = ndiis - 1, 2, -1
        if (xzpeak_cc(w) <= xzsize) then
          k8 = pslot
          exit
        end if
      end do
      if (k8 >= max(nfit, 2_8)) then
        pslot = k8
        pfile = .false.
      else if (nfit >= 2) then
        pslot = nfit
        pfile = .true.
      else
        pslot = 0
        pfile = .false.
        if (xzpeak_cc(w) > xzsize) then
          ! (the floor had the residual in a region of its own as well)
          pslot = 0
        end if
        write(6, '(a)') ' @XZCCSDT-W, neither the disk nor the work array holds two DIIS vectors:'
        write(6, '(a)') '             the iterations run without DIIS.'
      end if
    end if
  end if
  need_cc = 0
  if (.not. planq) need_cc = xzpeak_cc(w)
  ! ---- (Q)
  if (with_q) then
    do k = 5, 8
      call raise(k, .false.)
    end do
  end if
  need_q = xzpeak_q(w)
  ! a test hook: batch sizes given explicitly, XZ_PLAN="nqs nbt nbp nba qxv qc22 qc13 qrows"
  call get_environment_variable('XZ_PLAN', length=larg)
  if (larg > 0) call explicit()

  nslot = pslot
  usefile = pfile
  nch = pchunk
  if (.not. planq) then
  write(6, '(a,i16,a,f10.1,a)') '     CCSDT with this plan:           ', need_cc, '  = ', 8d-6 * need_cc, ' MB'
  end if
  if (with_q) write(6, '(a,i16,a,f10.1,a)') '     (Q) with this plan:             ', need_q, '  = ', &
    8d-6 * need_q, ' MB'
  if (.not. planq) then
  write(6, '(a,i8,a,i8)') '     rows of a slab of the dressed <ab||cd>:      ', nqs, '  of ', nqmax
  write(6, '(a,i8,a,i8)') '     occupied triples of a pp-ladder batch:       ', nbt, '  of ', nto
  write(6, '(a,i8,a,i8)') '     occupied pairs of a batch of pass 1, pass 2: ', nbp, '  of ', npo
  write(6, '(a,i8,a,i8)') '     first indices of a batch of W_vvvo terms:    ', nba, '  of ', nv
  if (pslot < 2) then
    write(6, '(a)') '     DIIS: none'
  else if (pfile) then
    write(6, '(a,i3,a,f10.1,a)') '     DIIS: ', pslot, ' vectors, T3 parts in the file XZDIIS (', &
      16d-6 * nwe * n3 * pslot, ' MB)'
  else
    write(6, '(a,i3,a)') '     DIIS: ', pslot, ' vectors in the work array'
  end if
  end if
  if (with_q) then
    write(6, '(a,i8,a,i8)') '     (Q): occupied indices with Xv at a time:     ', qxv, '  of ', max(3_8, no - 1)
    write(6, '(a,i8,a,i8)') '     (Q): columns (cd) of a product:              ', qc22, '  of ', nvp
    write(6, '(a,i8,a,i8)') '     (Q): columns (bcd) of a product:             ', qc13, '  of ', nvt
    write(6, '(a,i8,a,i8)') '     (Q): rows (bc) of a step of Xv:              ', qrows, '  of ', nvp
  end if
  if (max(need_cc, need_q) > xzsize) then
    write(6, '(/,a)') ' @XZCCSDT-F, the plan does not fit the work array (XZ_PLAN?).'
    call errex
  end if
  flush(6)

contains

  ! parameter k as large as fits, between its floor and its target
  subroutine raise(k, cc)
    integer, intent(in) :: k
    logical, intent(in) :: cc
    integer(8) :: lo, hi, mid, peak
    lo = plo(k)
    hi = phi(k)
    call setpar(k, hi)
    peak = merge(xzpeak_cc(w), xzpeak_q(w), cc)
    if (peak <= xzsize) return
    do while (hi - lo > 1)
      mid = (lo + hi) / 2
      call setpar(k, mid)
      peak = merge(xzpeak_cc(w), xzpeak_q(w), cc)
      if (peak <= xzsize) then
        lo = mid
      else
        hi = mid
      end if
    end do
    call setpar(k, lo)
  end subroutine

  subroutine explicit()
    character(len=256) :: text
    integer(8) :: v(npar)
    integer :: kk, ios
    call get_environment_variable('XZ_PLAN', text)
    read(text, *, iostat=ios) v
    if (ios /= 0) then
      write(6, '(a)') ' @XZCCSDT-F, XZ_PLAN: eight numbers expected.'
      call errex
    end if
    do kk = 1, npar
      if (v(kk) > 0) call setpar(kk, max(1_8, v(kk)))
    end do
    call get_environment_variable('XZ_DIIS', arg, larg)
    if (larg > 0) then
      pfile = arg(1:1) == 'f' .or. arg(1:1) == 'F'
      if (larg > 1) read(arg(2:larg), *) pslot
    end if
    need_cc = xzpeak_cc(w)
    need_q = xzpeak_q(w)
    write(6, '(a)') '     (batch sizes from XZ_PLAN)'
  end subroutine
end subroutine xzplan
