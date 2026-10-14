!
! The Krylov solver of the Lambda equations, the part that depends on the
! element type (the scheme and the state: xzlkry.F90; docs/native.md,
! section 11).
!
! An iteration (N(xzlk_upd)), given lambda x and its Jacobi image x + g(x):
!   1. g = (x + g) - x: the step, whose largest element is the test (CC_CONV);
!   2. the difference to the previous iterate, (dx, dg) = (x - x', g - g'),
!      is a new pair: dg orthogonalized against the s_i (classical
!      Gram-Schmidt, a second time if the first took more than half of its
!      norm) and normalized, dx through the coefficients into T;
!   3. y = -S^H g minimizes |g + S y| (S orthonormal), and the next lambda is
!      x + B y + (g + S y): the point of the subspace with the smallest step,
!      plus that step;
!   4. a full subspace is compacted (thick restart) to the harmonic Ritz
!      vectors of M on it with the smallest |theta| - the soft directions,
!      which the Jacobi step contracts least and a short history loses.
! The cost is one Jacobi step (a residual of the Lambda equations) per
! iteration, as with DIIS; the subspace adds passes over its vectors.
!

! ---------------------------------------------------------------------------
! The regions of nslot pairs (0: only the region of the new l3, for a rank
! that does its part of the residual and nothing else).  usefile: the l3
! parts in the file XZLKRY (records of nch elements).
subroutine N(xzlk_alloc)(nslot, usefile, nch)
  use xzdata
  use xzmem
  use xzlkryd
  use xzdist, only: slices, soff, smax, myrank
  implicit none
  integer(8), intent(in) :: nslot, nch
  logical, intent(in) :: usefile
  integer(8) :: k, ns, nal
  integer :: ios
  ns = no * nv + npo * nvp
  ksplit = slices
  ksoff = 0
  kslen = n3
  nal = n3
  if (ksplit) then
    ksoff = soff(myrank)
    kslen = soff(myrank + 1) - soff(myrank)
    nal = max(smax, 1_8)
  end if
  fkry = 'XZLKRY'
  if (myrank > 0) write(fkry, '(a7,i4.4)') 'XZLKRY.', myrank
  kp = min(nslot, int(mkmax, 8))
  if (kp == 1) kp = 2
  km = 0
  kprev = 0
  kfree = 1
  kcall = 0
  knres = 0
  kkeep = max(0_8, min(kp - 2, (kp - 1) / 2))
  kfile = usefile .and. kp > 0
  call xzoptget('lritz', length=ios)
  kdbg = ios > 0 .and. .not. xzdry
  tmat = 0
  kthr = 0d0
  ktime = 0d0
  kbytes = 0d0
  if (kp == 0) then
    jrw = xzget(NW * n3, 'new Lambda3')
    return
  end if
  ! (the small parts are rank 0's)
  if (myrank == 0) then
    jxs0 = xzget(NW * ns, 'Krylov: l1, l2 of lambda')
    jgs0 = xzget(NW * ns, 'Krylov: l1, l2 of its step')
    do k = 1, kp
      jxs(k) = xzget(NW * ns, 'Krylov: l1, l2 of a direction')
      jss(k) = xzget(NW * ns, 'Krylov: l1, l2 of its image')
    end do
  end if
  if (.not. kfile) then
    do k = 1, kp
      jxb(k) = xzget(NW * nal, 'Krylov: l3 of a direction')
      jsb(k) = xzget(NW * nal, 'Krylov: l3 of its image')
    end do
    if (ksplit) jrw = xzget(NW * n3, 'new Lambda3')
  else
    knch = max(1_8, min(nch, nal))
    knrec = (nal + knch - 1) / knch
    jrw = xzget(NW * n3, 'new Lambda3')
    jbuf = xzget(2 * kp * NW * knch, 'Krylov: records of the file XZLKRY')
    if (.not. xzdry) then
      open(unit=lukry, file=trim(fkry), form='unformatted', access='direct', status='replace', &
           recl=8 * NW * knch, iostat=ios)
      if (ios /= 0) then
        write(6, '(/,a,a,a,i8)') ' @XZCCSDT-F, cannot open the file ', trim(fkry), &
          ' of the Krylov vectors of the Lambda equations, iostat ', ios
        call xzerrex
      end if
    end if
  end if
end subroutine

! the region in which the next iteration builds its new l3
integer(8) function N(xzlk_target)()
  use xzlkryd
  implicit none
  if (kfile .or. ksplit .or. kp == 0) then
    N(xzlk_target) = jrw
  else
    N(xzlk_target) = jsb(kfree)
  end if
end function

subroutine N(xzlk_close)()
  use xzlkryd
  implicit none
  if (kfile) close(lukry, status='delete')
  kfile = .false.
end subroutine

! ---------------------------------------------------------------------------
! One record of the l3 part of a vector: rw = 'w' write, 'r' read; kind 1
! the direction dx, 2 its image s.
subroutine N(xzlk_io)(rw, kind, slot, irec, buf, n)
  use xzlkryd
  use omp_lib
  implicit none
  character(len=1), intent(in) :: rw
  integer(8), intent(in) :: kind, slot, irec, n
  ZT :: buf(n)
  integer(8) :: rec
  integer :: ios
  real(8) :: t0
  t0 = omp_get_wtime()
  rec = ((slot - 1) * 2 + kind - 1) * knrec + irec + 1
  if (rw == 'w') then
    write(lukry, rec=rec, iostat=ios) buf
  else
    read(lukry, rec=rec, iostat=ios) buf
  end if
  if (ios /= 0) then
    write(6, '(/,a,a,a,i12,a,i8)') ' @XZCCSDT-F, ', merge('writing', 'reading', rw == 'w'), &
      ' the file ' // trim(fkry) // ' of the Krylov vectors failed: record ', rec, ', iostat ', ios
    write(6, '(a,f12.1,a)') '   The file takes ', 16d-6 * kp * knrec * knch * NW, &
      ' MB when all vectors are there.  Is the disk of the run directory full?'
    call xzerrex
  end if
  ktime = ktime + omp_get_wtime() - t0
  kbytes = kbytes + 8d0 * NW * n
end subroutine

! The elements off .. off + nc - 1 of the l3 part of vector (kind, slot): the
! index of the first in w (the work array; or the buffer of the slot, read
! from the file if rd).
integer(8) function N(xzlk_ptr)(w, kind, slot, off, nc, rd)
  use xzlkryd
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kind, slot, off, nc
  logical, intent(in) :: rd
  integer(8) :: p
  if (.not. kfile) then
    if (kind == 1) then
      p = jxb(slot) + NW * off
    else
      p = jsb(slot) + NW * off
    end if
  else
    p = jbuf + NW * knch * ((kind - 1) * kp + slot - 1)
    if (rd) call N(xzlk_io)('r', kind, slot, off / knch, w(p), nc)
  end if
  N(xzlk_ptr) = p
end function

! the buffer of (kind, slot) back to its record (the file; nothing otherwise)
subroutine N(xzlk_wr)(w, kind, slot, off, nc)
  use xzlkryd
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kind, slot, off, nc
  if (kfile) call N(xzlk_io)('w', kind, slot, off / knch, w(jbuf + NW * knch * ((kind - 1) * kp + slot - 1)), nc)
end subroutine

! numbers of all ranks summed (slices) and those of rank 0 on every rank
subroutine N(xzlk_reduce)(n, x)
  use xzlkryd
  use xzdist, only: xzmpi_allsum_w, xzmpi_bcast
  implicit none
  integer(8), intent(in) :: n
  complex(8), intent(inout) :: x(n)
  if (.not. ksplit .or. n == 0) return
  call xzmpi_allsum_w(x, 2 * n)
  call xzmpi_bcast(x, 16 * n)
end subroutine

! ---------------------------------------------------------------------------
! xs = (l1, l2 on pairs), gs = (l1n - l1, l2n - l2 on pairs)
subroutine N(xzlk_small)(no, nv, npo, nvp, opa, opb, vpa, vpb, l1, l2, l1n, l2n, xs, gs)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(in) :: l1(0:nv-1, 0:no-1), l2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: l1n(0:nv-1, 0:no-1), l2n(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(out) :: xs(0:nv*no+nvp*npo-1), gs(0:nv*no+nvp*npo-1)
  integer(8) :: i, a, p, q, n1
  n1 = no * nv
  do i = 0, no - 1
    do a = 0, nv - 1
      xs(a + nv * i) = l1(a, i)
      gs(a + nv * i) = l1n(a, i) - l1(a, i)
    end do
  end do
  !$omp parallel do private(q)
  do p = 0, npo - 1
    do q = 0, nvp - 1
      xs(n1 + q + nvp * p) = l2(vpb(q), vpa(q), opb(p), opa(p))
      gs(n1 + q + nvp * p) = l2n(vpb(q), vpa(q), opb(p), opa(p)) - xs(n1 + q + nvp * p)
    end do
  end do
end subroutine

! y = alpha x
subroutine N(xzlk_scal)(n, alpha, x, y)
  implicit none
  integer(8), intent(in) :: n
  ZT, intent(in) :: alpha, x(n)
  ZT, intent(out) :: y(n)
  integer(8) :: i
  !$omp parallel do
  do i = 1, n
    y(i) = alpha * x(i)
  end do
end subroutine

! ---------------------------------------------------------------------------
! The vector of slot q of kind kq becomes  cq V_kq(q) + sum_i c(i) V_kind(sl(i))
! [+ e1 E1 + e2 E2 for two work vectors given by their small and l3 parts].
subroutine N(xzlk_combs)(w, kq, q, cq, kind, nsl, sl, c, ne, es, eb, ec)
  use xzdata
  use xzkern
  use xzlkryd
  use xzdist, only: myrank
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kq, q, kind, nsl, sl(*), ne, es(*), eb(*)
  complex(8), intent(in) :: cq, c(*), ec(*)
  integer(c_intptr_t) :: ptr(mkmax + 4)
  ZT :: cz(mkmax + 4)
  integer(8) :: ns, off, nc, step, i, p, pq, nk
  integer(8), external :: N(xzlk_ptr)
  ns = no * nv + npo * nvp
  nk = nsl + 1 + ne
  do i = 1, nsl
#ifdef XZ_REAL
    cz(i) = dble(c(i))
#else
    cz(i) = c(i)
#endif
  end do
#ifdef XZ_REAL
  cz(nsl + 1) = dble(cq)
  if (ne > 0) cz(nsl + 2:nsl + 1 + ne) = dble(ec(1:ne))
#else
  cz(nsl + 1) = cq
  if (ne > 0) cz(nsl + 2:nsl + 1 + ne) = ec(1:ne)
#endif
  if (myrank == 0) then
    do i = 1, nsl
      ptr(i) = loc(w(merge(jxs(sl(i)), jss(sl(i)), kind == 1)))
    end do
    pq = merge(jxs(q), jss(q), kq == 1)
    ptr(nsl + 1) = loc(w(pq))
    do i = 1, ne
      ptr(nsl + 1 + i) = loc(w(es(i)))
    end do
    call N(zc_vlincomb)(ns, nk, ptr, cz, w(pq))
  end if
  step = max(kslen, 1_8)
  if (kfile) step = knch
  do off = 0, kslen - 1, step
    nc = min(step, kslen - off)
    do i = 1, nsl
      p = N(xzlk_ptr)(w, kind, sl(i), off, nc, .true.)
      ptr(i) = loc(w(p))
    end do
    pq = N(xzlk_ptr)(w, kq, q, off, nc, .true.)
    ptr(nsl + 1) = loc(w(pq))
    do i = 1, ne
      ptr(nsl + 1 + i) = loc(w(eb(i) + NW * off))
    end do
    call N(zc_vlincomb)(nc, nk, ptr, cz, w(pq))
    call N(xzlk_wr)(w, kq, q, off, nc)
  end do
end subroutine

! ---------------------------------------------------------------------------
! The new lambda (l1, l2 and, in the region kt3, l3) from lambda and its
! Jacobi image (l1n, l2n, and in the region kr3 the new l3).  norms: |dL1|,
! |dL2|, |dL3| of the full tensors (the Jacobi step); emax: the largest
! element of each - the measure of convergence, as with DIIS.
subroutine N(xzlk_upd)(w, l1, l2, kt3, l1n, l2n, kr3, norms, emax)
  use xzdata
  use xzmem
  use xzkern
  use xzlkryd
  use xzdist, only: myrank, xzmpi_allsum_w, xzmpi_max, xzmpi_bcast
  implicit none
  real(8) :: w(*)
  ZT, intent(inout) :: l1(0:nv-1, 0:no-1), l2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: l1n(0:nv-1, 0:no-1), l2n(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  integer(8), intent(in) :: kt3, kr3
  real(8), intent(out) :: norms(3), emax(3)
  integer(8) :: n1, n2, ns, m3, px, pg, q, f, i, j, off, nc, step, nn, p, ss(2), sb(2)
  integer(c_intptr_t) :: ptr(2 * mkmax + 4)
  ZT :: cz(2 * mkmax + 4), dz
  complex(8) :: res(mkmax + 1, 2), c1(mkmax), c2(mkmax), ct(mkmax), y(mkmax), a(mkmax), one, rqq, rqg
  complex(8) :: red(3)
  real(8) :: s1, s2, s3, nr0, nr1, nrm, mx(1), mr(1)
  logical :: small, used
  integer(8), external :: N(xzlk_ptr)
  one = 1
  n1 = no * nv
  n2 = npo * nvp
  ns = n1 + n2
  m3 = kslen
  px = kt3 + NW * ksoff
  pg = kr3 + NW * ksoff
  small = myrank == 0
  kcall = kcall + 1
  ! ---- 1. the step g, in place of the new vector; its norms and largest elements
  s1 = 0d0
  s2 = 0d0
  emax = 0d0
  if (small) then
    call N(xzlk_small)(no, nv, npo, nvp, w(jopa), w(jopb), w(jvpa), w(jvpb), l1, l2, l1n, l2n, w(jxs0), w(jgs0))
    call N(zc_vdot)(n1, w(jgs0), w(jgs0), dz)
    s1 = dble(dz)
    call N(zc_vdot)(n2, w(jgs0 + NW * n1), w(jgs0 + NW * n1), dz)
    s2 = dble(dz)
    call N(xzamax)(n1, w(jgs0), emax(1))
    call N(xzamax)(n2, w(jgs0 + NW * n1), emax(2))
  end if
  call N(zc_vsub)(m3, w(pg), w(px), w(pg))
  call N(zc_vdot)(m3, w(pg), w(pg), dz)
  s3 = dble(dz)
  call N(xzamax)(m3, w(pg), emax(3))
  if (ksplit) then
    red(1) = s3
    call xzmpi_allsum_w(red, 2_8)
    s3 = dble(red(1))
    mx(1) = emax(3)
    call xzmpi_max(mx, mr, 1_8)
    emax(3) = mr(1)
  end if
  norms = [sqrt(s1), 2d0 * sqrt(s2), 6d0 * sqrt(s3)]
  ! the step as a work vector: small part, l3 part
  ss = [jgs0, jxs0]
  sb = [pg, px]

  ! ---- 2. the new pair from the previous iterate
  if (kprev > 0) then
    q = kprev
    if (small) then
      call N(zc_vsub)(ns, w(jxs0), w(jxs(q)), w(jxs(q)))
      call N(zc_vsub)(ns, w(jgs0), w(jss(q)), w(jss(q)))
    end if
    step = max(m3, 1_8)
    if (kfile) step = knch
    do off = 0, m3 - 1, step
      nc = min(step, m3 - off)
      p = N(xzlk_ptr)(w, 1_8, q, off, nc, .true.)
      call N(zc_vsub)(nc, w(px + NW * off), w(p), w(p))
      call N(xzlk_wr)(w, 1_8, q, off, nc)
      p = N(xzlk_ptr)(w, 2_8, q, off, nc, .true.)
      call N(zc_vsub)(nc, w(pg + NW * off), w(p), w(p))
      call N(xzlk_wr)(w, 2_8, q, off, nc)
    end do
    ! the overlaps of the columns and of dg with dg and with g (one pass)
    kcol(km + 1) = q
    call N(xzlk_mdot2)(w, km + 1, kcol, .true., res)
    c1(1:km) = res(1:km, 1)
    rqq = res(km + 1, 1)
    dgs(1:km) = res(1:km, 2)
    rqg = res(km + 1, 2)
    nr0 = sqrt(max(dble(rqq), 0d0))
    ct(1:km) = c1(1:km)
    ! (the norm after the projection is known before it is done: the
    ! normalization goes into the same pass)
    nr1 = sqrt(max(dble(rqq) - sum(dble(c1(1:km) * conjg(c1(1:km)))), 0d0))
    if (km > 0 .and. nr1 < 0.7d0 * nr0) then
      ! twice (the first projection took more than half of the norm)
      call N(xzlk_combs)(w, 2_8, q, one, 2_8, km, kcol, -c1, 0_8, ss, sb, c1)
      call N(xzlk_mdot2)(w, km + 1, kcol, .false., res)
      c1(1:km) = res(1:km, 1)
      ct(1:km) = ct(1:km) + c1(1:km)
      nr1 = sqrt(max(dble(res(km + 1, 1)) - sum(dble(c1(1:km) * conjg(c1(1:km)))), 0d0))
    end if
    nrm = nr1
    if (nrm > 1d-12 * max(nr0, 1d-300) .and. nrm > 0d0) then
      ! projected and normalized in one pass; the coefficients of dx_q in T
      call N(xzlk_combs)(w, 2_8, q, cmplx(1d0 / nrm, 0d0, 8), 2_8, km, kcol, -c1 / nrm, 0_8, ss, sb, c1)
      do j = 1, km
        tmat(j, km + 1) = -sum(tmat(j, 1:km) * ct(1:km)) / nrm
      end do
      tmat(km + 1, 1:km + 1) = 0
      tmat(km + 1, km + 1) = 1d0 / nrm
      dgs(km + 1) = (rqg - sum(conjg(ct(1:km)) * dgs(1:km))) / nrm
      km = km + 1
    end if
    kprev = 0
  end if

  ! ---- the present iterate and its step become the previous ones
  f = kfree
  if (small) then
    call N(xzlk_scal)(ns, one_z(), w(jxs0), w(jxs(f)))
    call N(xzlk_scal)(ns, one_z(), w(jgs0), w(jss(f)))
  end if
  step = max(m3, 1_8)
  if (kfile) step = knch
  do off = 0, m3 - 1, step
    nc = min(step, m3 - off)
    p = N(xzlk_ptr)(w, 1_8, f, off, nc, .false.)
    call N(xzlk_scal)(nc, one_z(), w(px + NW * off), w(p))
    call N(xzlk_wr)(w, 1_8, f, off, nc)
    if (kfile .or. ksplit) then
      p = N(xzlk_ptr)(w, 2_8, f, off, nc, .false.)
      call N(xzlk_scal)(nc, one_z(), w(pg + NW * off), w(p))
      call N(xzlk_wr)(w, 2_8, f, off, nc)
    end if
  end do
  kprev = f

  ! ---- 3. the next lambda: x + B y + g + S y with y = -S^H g, B = dX T
  y(1:km) = -dgs(1:km)
  if (kdbg .and. small) write(6, '(a,es12.4,a,es12.4)') '      |g| ', sqrt(s1 + s2 + s3), '   |g + S y| ', &
    sqrt(max(s1 + s2 + s3 - sum(dble(dgs(1:km) * conjg(dgs(1:km)))), 0d0))
  do j = 1, km
    a(j) = sum(tmat(j, 1:km) * y(1:km))
  end do
  nn = 2 * km + 2
  do j = 1, km
#ifdef XZ_REAL
    cz(j) = dble(a(j))
    cz(km + j) = dble(y(j))
#else
    cz(j) = a(j)
    cz(km + j) = y(j)
#endif
  end do
  cz(2 * km + 1) = 1
  cz(2 * km + 2) = 1
  if (small) then
    do j = 1, km
      ptr(j) = loc(w(jxs(kcol(j))))
      ptr(km + j) = loc(w(jss(kcol(j))))
    end do
    ptr(2 * km + 1) = loc(w(jxs0))
    ptr(2 * km + 2) = loc(w(jgs0))
    call N(zc_vlincomb)(ns, nn, ptr, cz, w(jxs0))
    call N(xzlk_unpack)(no, nv, npo, nvp, w(jopa), w(jopb), w(jvpa), w(jvpb), w(jxs0), l1, l2)
  end if
  do off = 0, m3 - 1, step
    nc = min(step, m3 - off)
    do j = 1, km
      p = N(xzlk_ptr)(w, 1_8, kcol(j), off, nc, .true.)
      ptr(j) = loc(w(p))
      p = N(xzlk_ptr)(w, 2_8, kcol(j), off, nc, .true.)
      ptr(km + j) = loc(w(p))
    end do
    ptr(2 * km + 1) = loc(w(px + NW * off))
    ptr(2 * km + 2) = loc(w(pg + NW * off))
    call N(zc_vlincomb)(nc, nn, ptr, cz, w(px + NW * off))
  end do

  ! ---- 4. the slot of the next iterate; a full subspace is compacted
  kfree = 0
  do i = 1, kp
    used = i == kprev .or. any(kcol(1:km) == i)
    if (.not. used) then
      kfree = i
      exit
    end if
  end do
  if (kfree == 0) then
    call N(xzlk_compact)(w, kkeep, .true.)
    do i = 1, kp
      used = i == kprev .or. any(kcol(1:km) == i)
      if (.not. used) then
        kfree = i
        exit
      end if
    end do
  end if

contains
  ZT function one_z()
    one_z = 1
  end function
end subroutine

! The overlaps <s_i | s_q> (and, withg, <s_i | g>) for the slots sl(1:n)
! (the last is q) in one pass over the images: res(i, 1), res(i, 2).  The
! step g is the work vector of N(xzlk_upd) (small part at jgs0; l3 part in
! the region of the new l3 or of slot kfree's image).
subroutine N(xzlk_mdot2)(w, n, sl, withg, res)
  use xzdata
  use xzkern
  use xzlkryd
  use xzdist, only: myrank
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: n, sl(*)
  logical, intent(in) :: withg
  complex(8), intent(out) :: res(mkmax + 1, 2)
  integer(c_intptr_t) :: ptr(mkmax + 1), pa(2)
  ZT :: part(2 * mkmax + 2)
  integer(8) :: ns, off, nc, step, i, p, pq, pg
  integer(8), external :: N(xzlk_ptr), N(xzlk_target)
  res = 0
  ns = no * nv + npo * nvp
  if (myrank == 0) then
    do i = 1, n
      ptr(i) = loc(w(jss(sl(i))))
    end do
    pa(1) = loc(w(jss(sl(n))))
    pa(2) = loc(w(jgs0))
    call N(zc_vgram)(ns, merge(2_8, 1_8, withg), pa, n, ptr, part)
    res(1:n, 1) = res(1:n, 1) + CONJ(part(1:n))
    if (withg) res(1:n, 2) = res(1:n, 2) + CONJ(part(n + 1:2 * n))
  end if
  pg = N(xzlk_target)() + NW * ksoff
  step = max(kslen, 1_8)
  if (kfile) step = knch
  do off = 0, kslen - 1, step
    nc = min(step, kslen - off)
    pq = 0
    do i = 1, n
      p = N(xzlk_ptr)(w, 2_8, sl(i), off, nc, .true.)
      ptr(i) = loc(w(p))
      pq = p
    end do
    ! (one pass over the images for both: zc_vgram reads a block of all of them once)
    pa(1) = loc(w(pq))
    pa(2) = loc(w(pg + NW * off))
    call N(zc_vgram)(nc, merge(2_8, 1_8, withg), pa, n, ptr, part)
    res(1:n, 1) = res(1:n, 1) + CONJ(part(1:n))
    if (withg) res(1:n, 2) = res(1:n, 2) + CONJ(part(n + 1:2 * n))
  end do
  call N(xzlk_reduce)(2 * (mkmax + 1_8), res)
end subroutine

! l1, l2 from xs = (l1, l2 on pairs)
subroutine N(xzlk_unpack)(no, nv, npo, nvp, opa, opb, vpa, vpb, xs, l1, l2)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(in) :: xs(0:nv*no+nvp*npo-1)
  ZT, intent(out) :: l1(0:nv-1, 0:no-1), l2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  integer(8) :: i, a
  do i = 0, no - 1
    do a = 0, nv - 1
      l1(a, i) = xs(a + nv * i)
    end do
  end do
  call N(xzunpack2)(no, nv, npo, nvp, opa, opb, vpa, vpb, xs(no * nv), l2)
end subroutine

! ---------------------------------------------------------------------------
! The harmonic Ritz pairs of M on the subspace: M b - theta b orthogonal to
! the images S, i.e. (S^H B) y = (1 / theta) y, S^H B = W T with W = S^H dX.
! On return mu(1:km) = 1 / theta sorted by |mu|, largest first (smallest
! |theta|), and yv(:, k) the vectors (real arithmetic: a complex pair as the
! real and the imaginary part of its vector, both with the pair's mu).
subroutine N(xzlk_harm)(w, mu, yv)
  use xzdata
  use xzkern
  use xzlkryd
  use xzdist, only: myrank
  implicit none
  real(8) :: w(*)
  complex(8), intent(out) :: mu(mkmax), yv(mkmax, mkmax)
  integer(c_intptr_t) :: px1(mkmax), px2(mkmax)
  ZT :: g(mkmax * mkmax)
  complex(8) :: wm(mkmax, mkmax), cm(km, km), x
  integer(8) :: ns, off, nc, step, i, j, k, p, info, lwork, idx(km), t
  real(8) :: am(km)
  integer(8), external :: N(xzlk_ptr)
#ifdef XZ_REAL
  real(8) :: ar(km, km), wr(km), wi(km), vl(1, 1), vr(km, km), work(8 * km + 64)
#else
  complex(8) :: vl(1, 1), vr(km, km), work(8 * km + 64), ev(km)
  real(8) :: rwork(2 * km)
#endif
  mu = 0
  yv = 0
  if (km == 0) return
  ns = no * nv + npo * nvp
  wm = 0
  ! W(i, j) = <s_i | dx_j>: zc_vgram gives g[i * km + j]
  if (myrank == 0) then
    do i = 1, km
      px1(i) = loc(w(jss(kcol(i))))
      px2(i) = loc(w(jxs(kcol(i))))
    end do
    call N(zc_vgram)(ns, km, px1, km, px2, g)
    do i = 1, km
      do j = 1, km
        wm(i, j) = wm(i, j) + g(j + (i - 1) * km)
      end do
    end do
  end if
  step = max(kslen, 1_8)
  if (kfile) step = knch
  do off = 0, kslen - 1, step
    nc = min(step, kslen - off)
    do i = 1, km
      p = N(xzlk_ptr)(w, 2_8, kcol(i), off, nc, .true.)
      px1(i) = loc(w(p))
      p = N(xzlk_ptr)(w, 1_8, kcol(i), off, nc, .true.)
      px2(i) = loc(w(p))
    end do
    call N(zc_vgram)(nc, km, px1, km, px2, g)
    do i = 1, km
      do j = 1, km
        wm(i, j) = wm(i, j) + g(j + (i - 1) * km)
      end do
    end do
  end do
  call N(xzlk_reduce)(int(mkmax, 8) * mkmax, wm)
  ! C = W T
  do i = 1, km
    do k = 1, km
      cm(i, k) = sum(wm(i, 1:km) * tmat(1:km, k))
    end do
  end do
  lwork = 8 * km + 64
#ifdef XZ_REAL
  ar = dble(cm)
  call dgeev('N', 'V', km, ar, km, wr, wi, vl, 1_8, vr, km, work, lwork, info)
  if (info /= 0) then
    mu(1:km) = 0
    do k = 1, km
      yv(k, k) = 1
    end do
    return
  end if
  do k = 1, km
    mu(k) = cmplx(wr(k), wi(k), 8)
  end do
  k = 1
  do while (k <= km)
    if (wi(k) /= 0d0 .and. k < km) then
      yv(1:km, k) = vr(:, k)
      yv(1:km, k + 1) = vr(:, k + 1)
      k = k + 2
    else
      yv(1:km, k) = vr(:, k)
      k = k + 1
    end if
  end do
#else
  call zgeev('N', 'V', km, cm, km, ev, vl, 1_8, vr, km, work, lwork, rwork, info)
  if (info /= 0) then
    mu(1:km) = 0
    do k = 1, km
      yv(k, k) = 1
    end do
    return
  end if
  mu(1:km) = ev
  yv(1:km, 1:km) = vr
#endif
  ! sorted by |mu|, largest first (insertion; a real pair stays together)
  do k = 1, km
    am(k) = abs(mu(k))
    idx(k) = k
  end do
  do i = 2, km
    t = idx(i)
    j = i - 1
    do while (j >= 1)
      if (am(idx(j)) >= am(t)) exit
      idx(j + 1) = idx(j)
      j = j - 1
    end do
    idx(j + 1) = t
  end do
  wm(1:km, 1:km) = yv(1:km, 1:km)
  do k = 1, km
    x = mu(idx(k))
    yv(1:km, k) = wm(1:km, idx(k))
    am(k) = abs(x)
    cm(k, 1) = x
  end do
  mu(1:km) = cm(1:km, 1)
end subroutine

! ---------------------------------------------------------------------------
! The subspace compacted to its nk harmonic Ritz directions of the smallest
! |theta| (thick restart; nk = 0: emptied).  The coefficients orthonormalized
! (S Q stays orthonormal), both kinds transformed in place, T = 1.
subroutine N(xzlk_compact)(w, nkeep, say)
  use xzdata
  use xzkern
  use xzlkryd
  use xzdist, only: myrank, xzmpi_bcast
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: nkeep
  logical, intent(in) :: say
  complex(8) :: mu(mkmax), yv(mkmax, mkmax), qm(mkmax, mkmax), tq(mkmax, mkmax), s
  ZT :: qz(mkmax * mkmax), tz(mkmax * mkmax)
  integer(c_intptr_t) :: ptr(mkmax)
  integer(8) :: nk, k, j, i, ns, off, nc, step, p, kind, mold
  integer(8), external :: N(xzlk_ptr)
  real(8) :: nrm
  knres = knres + 1
  mold = km
  nk = min(nkeep, km)
  if (nk <= 0) then
    km = 0
    tmat = 0
    return
  end if
  call N(xzlk_harm)(w, mu, yv)
#ifdef XZ_REAL
  ! (a complex pair at the edge: both)
  if (nk < km) then
    if (aimag(mu(nk)) /= 0d0 .and. abs(mu(nk + 1) - conjg(mu(nk))) < 1d-12 * abs(mu(nk))) nk = nk + 1
  end if
#endif
  ! the coefficients orthonormalized (modified Gram-Schmidt, twice)
  qm = 0
  j = 0
  do k = 1, nk
    qm(1:km, j + 1) = yv(1:km, k)
    do p = 1, 2
      do i = 1, j
        s = sum(conjg(qm(1:km, i)) * qm(1:km, j + 1))
        qm(1:km, j + 1) = qm(1:km, j + 1) - s * qm(1:km, i)
      end do
    end do
    nrm = sqrt(sum(dble(qm(1:km, j + 1) * conjg(qm(1:km, j + 1)))))
    if (nrm > 1d-10) then
      qm(1:km, j + 1) = qm(1:km, j + 1) / nrm
      j = j + 1
    end if
  end do
  nk = j
  ! (the same numbers on every rank: rank 0's)
  if (ksplit) call xzmpi_bcast(qm, 16_8 * mkmax * mkmax)
  do k = 1, nk
    do i = 1, km
      tq(i, k) = sum(tmat(i, 1:km) * qm(1:km, k))
    end do
  end do
  ! zc_vtransform: vector k of the first nk becomes sum_j x_j q[j * nk + k]
  do j = 1, km
    do k = 1, nk
#ifdef XZ_REAL
      qz(k + (j - 1) * nk) = dble(qm(j, k))
      tz(k + (j - 1) * nk) = dble(tq(j, k))
#else
      qz(k + (j - 1) * nk) = qm(j, k)
      tz(k + (j - 1) * nk) = tq(j, k)
#endif
    end do
  end do
  ns = no * nv + npo * nvp
  if (myrank == 0) then
    do j = 1, km
      ptr(j) = loc(w(jss(kcol(j))))
    end do
    call N(zc_vtransform)(ns, km, nk, ptr, qz)
    do j = 1, km
      ptr(j) = loc(w(jxs(kcol(j))))
    end do
    call N(zc_vtransform)(ns, km, nk, ptr, tz)
  end if
  step = max(kslen, 1_8)
  if (kfile) step = knch
  do kind = 1, 2
    do off = 0, kslen - 1, step
      nc = min(step, kslen - off)
      do j = 1, km
        p = N(xzlk_ptr)(w, kind, kcol(j), off, nc, .true.)
        ptr(j) = loc(w(p))
      end do
      if (kind == 2) then
        call N(zc_vtransform)(nc, km, nk, ptr, qz)
      else
        call N(zc_vtransform)(nc, km, nk, ptr, tz)
      end if
      do j = 1, nk
        call N(xzlk_wr)(w, kind, kcol(j), off, nc)
      end do
    end do
  end do
  km = nk
  tmat = 0
  do k = 1, km
    tmat(k, k) = 1
  end do
  do k = 1, min(4_8, nk)
    kthr(k) = 0d0
    if (abs(mu(k)) > 0d0) kthr(k) = abs(1d0 / mu(k))
  end do
  if (say .and. myrank == 0) then
    write(6, '(a,i3,a,i3,a,4es10.2)') '      Krylov subspace compacted:', mold, ' ->', km, &
      ' directions; |theta| of the kept, smallest:', (kthr(k), k = 1, min(4_8, nk))
    flush(6)
  end if
end subroutine

! ---------------------------------------------------------------------------
! Option lritz n: the n harmonic Ritz values of M (the Jacobian divided by
! the denominators: the Jacobi step contracts a direction by 1 + theta) of
! the smallest |theta| on the subspace as it is, and the largest elements of
! l1 of the first Ritz vector (rank 0 prints; after the iterations).
subroutine N(xzlk_ritz)(w, nprint)
  use xzdata
  use xzlkryd
  use xzdist, only: myrank
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: nprint
  complex(8) :: mu(mkmax), yv(mkmax, mkmax), cb(mkmax), th
  integer(8) :: k, j, i, a, n1, ib(3)
  real(8) :: big(3), v
  ZT :: u
  if (km == 0) return
  call N(xzlk_harm)(w, mu, yv)
  if (myrank /= 0) return
  write(6, '(/,a,i3,a)') '   Krylov subspace of ', km, ' directions: harmonic Ritz values theta of M = D^-1 A'
  write(6, '(a)') '   (the Jacobi step contracts the direction by 1 + theta), smallest |theta| first:'
  do k = 1, min(nprint, km)
    th = 0
    if (abs(mu(k)) > 0d0) th = 1d0 / mu(k)
    write(6, '(a,i4,2f14.6,a,f10.6)') '     ', k, dble(th), aimag(th), '   |1 + theta| ', abs(1d0 + th)
  end do
  ! the l1 part of the first Ritz vector u = dX T y (small parts of rank 0)
  do j = 1, km
    cb(j) = sum(tmat(j, 1:km) * yv(1:km, 1))
  end do
  n1 = no * nv
  big = -1d0
  ib = 0
  do i = 0, n1 - 1
    u = 0
    do j = 1, km
      u = u + cb(j) * zget(jxs(kcol(j)) + NW * i)
    end do
    v = abs(u)
    do k = 1, 3
      if (v > big(k)) then
        big(k + 1:3) = big(k:2)
        ib(k + 1:3) = ib(k:2)
        big(k) = v
        ib(k) = i
        exit
      end if
    end do
  end do
  write(6, '(a)') '   largest l1 elements of the first Ritz vector (i, a: active spinors from 1; their energies):'
  do k = 1, 3
    if (ib(k) < 0 .or. big(k) < 0d0) cycle
    i = ib(k) / nv
    a = mod(ib(k), nv)
    write(6, '(a,i5,i5,2f12.6,f10.4)') '     ', i + 1, a + 1, w(jeps + i), w(jeps + no + a), big(k)
  end do
contains
  ZT function zget(k)
    integer(8), intent(in) :: k
#ifdef XZ_REAL
    zget = w(k)
#else
    zget = cmplx(w(k), w(k + 1), 8)
#endif
  end function
end subroutine

