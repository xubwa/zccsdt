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
! words of the work array that the layout and the iterations (CCSDT, the
! Lambda equations, or both one after the other) take
integer(8) function xzpeak_cc(w)
  use xzdata
  use xzmem
  use xzpland
  use xzdist, only: xzworker
  implicit none
  real(8) :: w(*)
  real(8) :: e
  logical :: c
  integer(8) :: n, peak, status, k, mlay
  call xzdry_begin(0_8)
  call xzlayout(w, status)
  mlay = xzmark()
  if (with_cc) then
    if (nwe == 2) then
      call xzcc_z(w, pslot, pfile, pchunk, e, c, n)
    else
      call xzcc_d(w, pslot, pfile, pchunk, e, c, n)
    end if
  else
    k = xzget(nwe * no * nv, 'T1')
    k = xzget(nwe * no * no * nv * nv, 'T2')
    k = xzget(nwe * n3, 'T3')
  end if
  if (with_lam) then
    if (nwe == 2) then
      call xzlam_z(w, pslot, pfile, pchunk, c, n)
    else
      call xzlam_d(w, pslot, pfile, pchunk, c, n)
    end if
  end if
  ! the excited states (rank 0): after the amplitudes
  if (with_eom .and. (.not. xzworker() .or. cc3mode == 2)) then
    if (nwe == 2) then
      call xzeom_z(w, e)
    else
      call xzeom_d(w, e)
    end if
  end if
  ! the density stage (rank 0): the integrals, the amplitudes and lambda again, then its arrays
  if (with_den .and. .not. xzworker()) then
    call xzrel(mlay)
    k = xzget(nwe * no * nv, 'T1')
    k = xzget(nwe * no * no * nv * nv, 'T2')
    k = xzget(nwe * n3, 'T3')
    k = xzget(nwe * no * nv, 'Lambda1')
    k = xzget(nwe * no * no * nv * nv, 'Lambda2')
    k = xzget(nwe * n3, 'Lambda3')
    if (nwe == 2) then
      call xzden_z(w)
    else
      call xzden_d(w)
    end if
  end if
  call xzdry_end(peak)
  xzpeak_cc = peak
end function xzpeak_cc

! words that the layout, the amplitudes (and lambda) and (Q) take
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
  ! (T3 and Lambda3 of (Q): mapped files, not in the work array, unless option qmap 0)
  if (method /= 'MP2' .and. .not. t3shared) k = xzget(nwe * n3, 'T3')
  if (with_q .and. q_left) then
    k = xzget(nwe * no * no * nv * nv, 'Lambda2')
    if (.not. t3shared) k = xzget(nwe * n3, 'Lambda3')
  end if
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
  use xzdist
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(out) :: nslot, nch
  logical, intent(out) :: usefile
  logical, intent(in) :: flooronly
  integer :: nthr0
  integer(8) :: bpr
  integer(8), external :: xzpeak_cc, xzpeak_q
  integer(8) :: f, floor_cc, floor_q, need_cc, need_q, free_bytes, per_vec, nfit, k8
  integer :: k
  character(len=8) :: arg
  character(len=23) :: what
  real(8) :: frac
  integer :: larg

  what = trim(mname)
  if (with_lam) what = 'CCSDT, Lambda'
  if (with_lam .and. .not. with_cc) what = 'Lambda'
  if (with_den) what = trim(what) // ', density'
  if (with_den .and. .not. (with_cc .or. with_lam)) what = 'the density'
  if (with_eom) what = trim(mname) // ', EOM'
  if (with_eom .and. .not. with_cc) what = 'EOM'
  f = no * nv * nvp                       ! elements of a half-expanded batch of one pair
  ! ---- the smallest and the target batch sizes
  plo(1) = min(nv, nqmax)
  phi(1) = nqmax
  plo(2) = 1
  ! (in blocks of the labels a batch holds the allowed part only: the
  ! targets in triples and pairs grow by the dense fraction of T3, so that the
  ! buffers stay at their sizes in bytes and a batch of the passes has enough
  ! pairs for its tasks)
  frac = 1d0
  if (ncls > 1 .and. nto > 0 .and. nvt > 0) frac = min(1d0, max(1d-3, dble(n3) / (dble(nto) * dble(nvt))))
  phi(2) = max(1_8, min(nto, int(1.3d9 / (2d0 * 8 * nwe * nvp * nv * frac), 8)))
  plo(3) = 1
  phi(3) = max(1_8, min(npo, int(2d9 / (3d0 * 8 * nwe * f * frac), 8)))
  ! (with more than one rank the batches are what is dealt out: at least
  ! bpr of them per rank, option bpr, so that the cost model can balance the ranks)
  if (nrank > 1) then
    bpr = 8
    call xzoptget('bpr', arg, larg)
    if (larg > 0) read(arg(1:larg), *) bpr
    phi(2) = max(1_8, min(phi(2), nto / (max(bpr, 1_8) * nrank)))
    phi(3) = max(1_8, min(phi(3), npo / (max(bpr, 1_8) * nrank)))
  end if
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
  ! CC3 without stored triples: the work of a batch of occupied triples - at
  ! least one triple of the largest group (zc_cc3_work), up to 2 GB or all the
  ! triples of a pair (j, k), as the Python program (cc3fly.Fly)
  plo(9) = 1
  phi(9) = 1
  if (cc3mode == 2) then
    do k8 = 0, ntg - 1
      plo(9) = max(plo(9), zc_cc3_work(ncls, k8, 1_8, vcoff, vpoff, vtoff))
    end do
    phi(9) = max(plo(9), min(int(2d9 / (8 * nwe), 8), plo(9) * max(no - 2, 1_8)))
  end if
  ncb = 0
  do k = 1, npar + 1
    call setpar(k, plo(k))
  end do

  ! ---- the floor (a rank other than 0 has none of its own: rank 0 plans its
  !      memory and it takes that plan, below)
  pfile = .true.
  pslot = ndiis
  pchunk = min(max(n3, 1_8), 1048576_8)
  if (myrank > 0 .and. flooronly) return
  floor_cc = 0
  floor_q = 0
  if (myrank == 0) then
  if (.not. planq) floor_cc = xzpeak_cc(w)
  ! (MP2: the integrals and one array of the size of T2 twice)
  if (method == 'MP2') with_q = .false.
  floor_q = xzpeak_q(w)
  if (flooronly .and. max(floor_cc, floor_q) <= xzsize) return
  end if
  write(6, '(/,a,a,a)') '   Memory plan (words of 8 bytes; ', merge('real numbers   ', 'complex numbers', nwe == 1), ')'
  if (.not. planq .and. myrank == 0) write(6, '(a,a36,i16,a,f10.1,a)') '     the least that ', trim(what) // ' takes:', &
    floor_cc, '  = ', 8d-6 * floor_cc, ' MB'
  if (with_q .and. myrank == 0) write(6, '(a,i16,a,f10.1,a)') '     the least that (Q) takes:       ', floor_q, '  = ', &
    8d-6 * floor_q, ' MB'
  write(6, '(a,i16,a,f10.1,a)') '     the work array has:             ', xzsize, '  = ', 8d-6 * xzsize, ' MB'
  if (max(floor_cc, floor_q) > xzsize) then
    write(6, '(/,a)') ' @XZCCSDT-F, the work array is too short for this calculation.'
    if (nwe == 1) write(6, '(a)') '   (even if all integrals are real; complex integrals take about twice as much)'
    write(6, '(a,i16,a,i16)') '   words needed: ', max(floor_cc, floor_q), '   words there: ', xzsize
    write(6, '(a,i16,a,f12.1,a)') '   Set MEMORY_SIZE to at least ', max(floor_cc, floor_q) + 64, &
      ' words (', 8d-6 * (max(floor_cc, floor_q) + 64), ' MB).'
    call xzerrex
  end if

  ! ---- a rank other than 0: the batch sizes of rank 0, in the work array
  !      that rank 0 planned for it; no DIIS vectors
  if (myrank > 0) then
    do k = 1, npar
      call setpar(k, max(1_8, int(rplan(1 + k), 8)))
    end do
    call setpar(9, max(1_8, int(rplan(16), 8)))
    pslot = 0
    pfile = .false.
    if (slices) then
      ! (its slices of the DIIS vectors, where rank 0 has its own)
      pslot = int(rplan(10), 8)
      pfile = rplan(11) == 1d0
      pchunk = int(rplan(12), 8)
    end if
    need_cc = 0
    if (.not. planq) need_cc = xzpeak_cc(w)
    need_q = xzpeak_q(w)
    nslot = pslot
    usefile = pfile
    nch = pchunk
    pneedq = need_q
    if (.not. planq) write(6, '(a,a43,i16,a,f10.1,a)') '     this rank, ', trim(what) // ' with the plan of rank 0:', &
      need_cc, '  = ', 8d-6 * need_cc, ' MB'
    if (with_q) write(6, '(a,i16,a,f10.1,a)') '     (Q) with the plan of rank 0:    ', need_q, '  = ', 8d-6 * need_q, ' MB'
    if (max(need_cc, need_q) > xzsize) then
      write(6, '(/,a)') ' @XZCCSDT-F, the work array of this rank is too short for the plan of rank 0.'
      write(6, '(a,i16,a,i16)') '   words needed: ', max(need_cc, need_q), '   words there: ', xzsize
      write(6, '(a)') '   (rank 0 plans the memory of all ranks: do the ranks differ in their environment?)'
      call xzerrex
    end if
    flush(6)
    return
  end if

  ! ---- the batches of the iterations, one after the other
  if (.not. planq) then
  do k = 1, 4
    ! (CC3 without stored triples has no batches of pass 1, pass 2 or the pp ladder)
    if (cc3mode == 2 .and. (k == 2 .or. k == 3)) cycle
    call raise(k, .true.)
  end do
  ! CC4: the evaluation of the T4 terms (xzcc4.inc) is a stage of the
  ! iterations with the batches of (Q)
  if (cc4) then
    do k = 5, 8
      call raise(k, .true.)
    end do
  end if
  if (cc3mode == 2) then
    call raise(9, .true.)
    ! a test hook: option cc3w n elements of the work of a batch (at least one triple)
    call xzoptget('cc3w', arg, larg)
    if (larg > 0) then
      read(arg(1:larg), *) k8
      call setpar(9, max(plo(9), k8))
    end if
  end if
  end if
  ! ---- the DIIS vectors: the work array if all of them fit, else the file
  pfile = .false.
  if (planq) then
    pslot = 0
  else if (xzpeak_cc(w) > xzsize) then
    pfile = .true.
    per_vec = 2 * 8 * nwe * n3
    if (slices) per_vec = 2 * 8 * nwe * smax
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
  ! a test hook: batch sizes given explicitly, option plan "nqs nbt nbp nba qxv qc22 qc13 qrows"
  call xzoptget('plan', length=larg)
  if (larg > 0) call explicit()

  nslot = pslot
  usefile = pfile
  nch = pchunk
  pneedq = need_q
  ! what a rank other than 0 takes: the iterations without the DIIS vectors
  ! (with the most threads that a rank has), or (Q)
  pneedw = 0
  if (nrank > 1) then
    pneedw = need_q
    if (.not. planq) then
      nthr0 = omp_get_max_threads()
      if (maxthr > nthr0) call omp_set_num_threads(int(maxthr))
      asworker = .true.
      pneedw = max(pneedw, xzpeak_cc(w))
      asworker = .false.
      if (maxthr > nthr0) call omp_set_num_threads(nthr0)
    end if
  end if
  if (.not. planq) then
  write(6, '(a,a51,i16,a,f10.1,a)') '     ', trim(what) // ' with this plan:', need_cc, '  = ', 8d-6 * need_cc, ' MB'
  end if
  if (with_q) write(6, '(a,i16,a,f10.1,a)') '     (Q) with this plan:             ', need_q, '  = ', &
    8d-6 * need_q, ' MB'
  if (nrank > 1 .and. (.not. planq .or. with_q)) then
    write(6, '(a,i16,a,f10.1,a)') '     a rank other than 0 takes:      ', pneedw, '  = ', 8d-6 * pneedw, ' MB'
    if (.not. planq .and. .not. slices) write(6, '(a)') '       (its own work array: the integrals, the amplitudes, ' // &
      'its part of the residual, the batches; no DIIS vectors)'
    if (.not. planq .and. slices) write(6, '(a,f10.1,a)') '       (its own work array: the integrals, the amplitudes, ' // &
      'the residual, the batches, its slice of every DIIS vector: at most ', 8d-6 * nwe * smax, ' MB of T3)'
    call xznodemem(max(need_cc, need_q), pneedw + 4096)
  end if
  if (with_q .and. t3shared) then
    if (q_left) then
      write(6, '(a)') '       (private to a rank.  Not in the work array: T3 and Lambda3, read where the files XZAMPS'
      write(6, '(a,f10.1,a)') '        and XZLAMB are mapped - one copy in the page cache of a node for all its ranks: ', &
        16d-6 * nwe * n3, ' MB)'
    else
      write(6, '(a)') '       (private to a rank.  Not in the work array: T3, read where the file XZAMPS is mapped -'
      write(6, '(a,f10.1,a)') '        one copy in the page cache of a node for all its ranks: ', 8d-6 * nwe * n3, ' MB)'
    end if
  else if (with_q) then
    write(6, '(a)') '       (private to a rank, T3 included: option qmap 0)'
  end if
  if (.not. planq) then
  write(6, '(a,i8,a,i8)') '     rows of a slab of the dressed <ab||cd>:      ', nqs, '  of ', nqmax
  write(6, '(a,i8,a,i8)') '     occupied triples of a pp-ladder batch:       ', nbt, '  of ', nto
  write(6, '(a,i8,a,i8)') '     occupied pairs of a batch of pass 1, pass 2: ', nbp, '  of ', npo
  write(6, '(a,i8,a,i8)') '     first indices of a batch of W_vvvo terms:    ', nba, '  of ', nv
  if (cc3mode == 2) write(6, '(a,i14,a,f10.1,a,i12,a)') '     CC3: the work of a batch of occupied triples: ', nc3w, &
    ' elements (', 8d-6 * nwe * nc3w, ' MB; one triple: ', plo(9), ')'
  if (.not. (with_cc .or. with_lam)) then
  else if (pslot < 2) then
    write(6, '(a)') '     DIIS: none'
  else if (pfile) then
    write(6, '(a,i3,a,f10.1,a)') '     DIIS: ', pslot, ' vectors, T3 parts in the file XZDIIS (', &
      16d-6 * nwe * n3 * pslot, ' MB)'
    if (slices) write(6, '(a)') '       (every rank the slice of its rows, in its own file XZDIIS.<rank>)'
  else
    write(6, '(a,i3,a)') '     DIIS: ', pslot, ' vectors in the work array'
    if (slices) write(6, '(a)') '       (every rank the slice of its rows, in its own work array)'
  end if
  end if
  if (with_q) then
    write(6, '(a,i8,a,i8)') '     (Q): occupied indices with Xv at a time:     ', qxv, '  of ', max(3_8, no - 1)
    write(6, '(a,i8,a,i8)') '     (Q): columns (cd) of a product:              ', qc22, '  of ', nvp
    write(6, '(a,i8,a,i8)') '     (Q): columns (bcd) of a product:             ', qc13, '  of ', nvt
    write(6, '(a,i8,a,i8)') '     (Q): rows (bc) of a step of Xv:              ', qrows, '  of ', nvp
  end if
  if (cc4) then
    write(6, '(a,i8,a,i8)') '     T4 terms: occupied indices with Xv at a time:', qxv, '  of ', max(3_8, no - 1)
    if (ncls == 1) then
      write(6, '(a,i8,a,i8)') '     T4 terms: columns (cd) of a product:         ', qc22, '  of ', nvp
      write(6, '(a,i8,a,i8)') '     T4 terms: columns (bcd) of a product:        ', qc13, '  of ', nvt
      write(6, '(a,i8,a,i8)') '     T4 terms: rows (bc) of a step of Xv:         ', qrows, '  of ', nvp
    end if
  end if
  if (max(need_cc, need_q) > xzsize) then
    write(6, '(/,a)') ' @XZCCSDT-F, the plan does not fit the work array (option plan?).'
    call xzerrex
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
    call xzoptget('plan', text)
    read(text, *, iostat=ios) v
    if (ios /= 0) then
      write(6, '(a)') ' @XZCCSDT-F, option plan: eight numbers expected.'
      call xzerrex
    end if
    do kk = 1, npar
      if (v(kk) > 0) call setpar(kk, max(1_8, v(kk)))
    end do
    call xzoptget('diis', arg, larg)
    if (larg > 0) then
      pfile = arg(1:1) == 'f' .or. arg(1:1) == 'F'
      if (larg > 1) read(arg(2:larg), *) pslot
    end if
    need_cc = xzpeak_cc(w)
    need_q = xzpeak_q(w)
    write(6, '(a)') '     (batch sizes from option plan)'
  end subroutine
end subroutine xzplan
