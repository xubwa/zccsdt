!
! DIIS of the amplitudes, the part that depends on the element type
! (see xzdiis.F90 for the scheme and the storage).
!

! ---------------------------------------------------------------------------
! The regions of nslot slots.  usefile: the T3 parts are in the file XZDIIS
! (records of nch elements), and the work array has one region for the
! residual and two for a record each.
subroutine N(xzdiis_alloc)(nslot, usefile, nch)
  use xzdata
  use xzmem
  use xzdiisd
  use xzdist, only: slices, soff, smax, myrank
  implicit none
  integer(8), intent(in) :: nslot, nch
  logical, intent(in) :: usefile
  integer(8) :: k, n1, n2, nal
  integer :: ios
  n1 = no * nv
  n2 = npo * nvp
  split = slices
  soff3 = 0
  slen3 = n3
  nal = n3
  if (split) then
    ! (every rank takes the regions of the longest slice: one plan for all)
    soff3 = soff(myrank)
    slen3 = soff(myrank + 1) - soff(myrank)
    nal = max(smax, 1_8)
  end if
  fdiis = 'XZDIIS'
  if (myrank > 0) write(fdiis, '(a7,i4.4)') 'XZDIIS.', myrank
  mdiis = nslot
  ncnt = 0
  ncall = 0
  infile = usefile
  jwork2 = xzget(NW * n2, 'DIIS work, T2')
  do k = 1, nslot
    jx1(k) = xzget(NW * n1, 'DIIS vector, T1')
    je1(k) = xzget(NW * n1, 'DIIS error vector, T1')
    jx2(k) = xzget(NW * n2, 'DIIS vector, T2')
    je2(k) = xzget(NW * n2, 'DIIS error vector, T2')
  end do
  if (.not. usefile) then
    do k = 1, nslot
      jx3(k) = xzget(NW * nal, 'DIIS vector, T3')
      je3(k) = xzget(NW * nal, 'DIIS error vector, T3')
    end do
    ! (the first vector is not stored: its residual needs a place all the same;
    ! with slices the residual is whole and the vectors are not)
    if (nslot == 0 .or. split) jrwork = xzget(NW * n3, 'residual of T3')
  else
    nchunk = max(1_8, min(nch, nal))
    nrec = (nal + nchunk - 1) / nchunk
    jrwork = xzget(NW * n3, 'residual of T3')
    jchunk1 = xzget(NW * nchunk, 'DIIS record 1')
    jchunk2 = xzget(NW * nchunk, 'DIIS record 2')
    if (.not. xzdry) then
      open(unit=ludiis, file=trim(fdiis), form='unformatted', access='direct', status='replace', &
           recl=8 * NW * nchunk, iostat=ios)
      if (ios /= 0) then
        write(6, '(/,a,a,a,i8)') ' @XZCCSDT-F, cannot open the file ', trim(fdiis), ' of the DIIS vectors, iostat ', ios
        call xzerrex
      end if
    end if
  end if
end subroutine

! the region in which the next iteration builds the residual of T3
integer(8) function N(xzdiis_target)()
  use xzdiisd
  implicit none
  if (infile .or. mdiis == 0 .or. split) then
    N(xzdiis_target) = jrwork
  else
    N(xzdiis_target) = jx3(xzdiis_slot())
  end if
end function

subroutine N(xzdiis_close)()
  use xzdiisd
  implicit none
  if (infile) close(ludiis, status='delete')
  infile = .false.
end subroutine

! ---------------------------------------------------------------------------
! One record of a T3 part: rw = 'w' write, 'r' read.
subroutine N(xzdiis_io)(rw, slot, kind, irec, buf, n)
  use xzdiisd
  implicit none
  character(len=1), intent(in) :: rw
  integer(8), intent(in) :: slot, kind, irec, n
  ZT :: buf(n)
  integer(8) :: rec
  integer :: ios
  rec = ((slot - 1) * 2 + kind) * nrec + irec + 1
  if (rw == 'w') then
    write(ludiis, rec=rec, iostat=ios) buf
  else
    read(ludiis, rec=rec, iostat=ios) buf
  end if
  if (ios /= 0) then
    write(6, '(/,a,a,a,a,i12,a,i8)') ' @XZCCSDT-F, ', merge('writing', 'reading', rw == 'w'), &
      ' the file ' // trim(fdiis) // ' of the DIIS vectors failed: record ', ' ', rec, ', iostat ', ios
    write(6, '(a,f12.1,a)') '   The file takes ', 16d-6 * mdiis * nrec * nchunk * NW, &
      ' MB when all vectors are there.  Is the disk of the run directory full?'
    call xzerrex
  end if
end subroutine

! ---------------------------------------------------------------------------
! The new vector (t1n, t2n and, in the region kr3, T3) is stored and the
! extrapolated vector is written to t1, t2 and the region kt3.
! norms: |dT1|, |dT2|, |dT3| of the full tensors (new minus previous).
subroutine N(xzdiis_upd)(w, t1, t2, kt3, t1n, t2n, kr3, norms)
  use xzdata
  use xzmem
  use xzkern
  use xzdiisd
  use xzdist, only: myrank, xzmpi_allsum_w, xzmpi_bcast
  implicit none
  real(8) :: w(*)
  ZT, intent(inout) :: t1(0:nv-1, 0:no-1), t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: t1n(0:nv-1, 0:no-1), t2n(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  integer(8), intent(in) :: kt3, kr3
  real(8), intent(out) :: norms(3)
  integer(8) :: n1, n2, slot, k, l, s, ir, nn, off, m3, pr, pt, nrl
  integer(8) :: ptr(maxdiis)
  ZT :: cz(maxdiis), d(maxdiis), dd, one
  complex(8) :: h(maxdiis, maxdiis), c(maxdiis)
  real(8) :: s1, s2, s3, sx, ss(1)
  logical :: small
  one = 1
  n1 = no * nv
  n2 = npo * nvp
  ! the T3 part of this rank: m3 elements of the new vector from w(pr), of the
  ! previous one from w(pt) (all of T3, or the slice of this rank).  The small
  ! parts are rank 0's.
  m3 = slen3
  pr = kr3 + NW * soff3
  pt = kt3 + NW * soff3
  small = myrank == 0
  ncall = ncall + 1
  s1 = 0d0
  s2 = 0d0
  if (small) then
    call N(xzdiff2)(n1, t1n, t1, s1)
    call N(xzdiff2)(nv * nv * no * no, t2n, t2, s2)
  end if

  if (ncall == 1 .or. mdiis < 2) then
    ! the first vector starts the sequence
    s3 = 0d0
    if (infile .or. mdiis == 0 .or. split) then
      nn = max(m3, 1_8)
      if (infile) nn = nchunk
      do off = 0, m3 - 1, nn
        l = min(nn, m3 - off)
        call N(xzdiff2)(l, w(pr + NW * off), w(pt + NW * off), sx)
        s3 = s3 + sx
      end do
    else
      call N(zc_vsub)(n3, w(kr3), w(kt3), w(je3(1)))
      call N(zc_vdot)(n3, w(je3(1)), w(je3(1)), dd)
      s3 = dble(dd)
    end if
    if (split) then
      ss(1) = s3
      call xzmpi_allsum_w(ss, 1_8)
      s3 = ss(1)
    end if
    norms = [sqrt(s1), sqrt(s2), 6d0 * sqrt(s3)]
    if (small) then
      t1 = t1n
      t2 = t2n
    end if
    ptr(1) = loc(w(pr))
    cz(1) = 1
    call N(zc_vlincomb)(m3, 1_8, ptr, cz, w(pt))
    return
  end if

  slot = xzdiis_slot()
  if (ncnt == mdiis) then
    ord(1:ncnt - 1) = ord(2:ncnt)
    ncnt = ncnt - 1
  end if
  ncnt = ncnt + 1
  ord(ncnt) = slot
  ! the small parts of the vector and of its error
  d(1:ncnt) = 0
  if (small) then
  call N(xzdiis_small)(no, nv, npo, nvp, w(jopa), w(jopb), w(jvpa), w(jvpb), t1, t2, t1n, t2n, &
                       w(jx1(slot)), w(jx2(slot)), w(je1(slot)), w(je2(slot)))
  ! the overlaps of the new error vector with all of them
  do k = 1, ncnt
    s = ord(k)
    call N(zc_vdot)(n1, w(je1(slot)), w(je1(s)), dd)
    d(k) = d(k) + dd
    if (k == ncnt) s1 = dble(dd)
    call N(zc_vdot)(n2, w(je2(slot)), w(je2(s)), dd)
    d(k) = d(k) + dd
    if (k == ncnt) s2 = dble(dd)
  end do
  end if
  nrl = (m3 + nchunk - 1) / max(nchunk, 1_8)
  if (.not. infile) then
    if (split) then
      ! (the new vector is in the region of the residual: its slice into the slot)
      ptr(1) = loc(w(pr))
      cz(1) = 1
      call N(zc_vlincomb)(m3, 1_8, ptr, cz, w(jx3(slot)))
    end if
    call N(zc_vsub)(m3, w(pr), w(pt), w(je3(slot)))
    do k = 1, ncnt
      ptr(k) = loc(w(je3(ord(k))))
    end do
    cz(1:ncnt) = 0
    call N(zc_vmdot)(m3, ncnt, w(je3(slot)), ptr, cz)
  else
    cz(1:ncnt) = 0
    do ir = 0, nrl - 1
      off = ir * nchunk
      l = min(nchunk, m3 - off)
      call N(zc_vsub)(l, w(pr + NW * off), w(pt + NW * off), w(jchunk1))
      call N(xzdiis_io)('w', slot, 1_8, ir, w(jchunk1), l)
      call N(xzdiis_io)('w', slot, 0_8, ir, w(pr + NW * off), l)
      do k = 1, ncnt - 1
        call N(xzdiis_io)('r', ord(k), 1_8, ir, w(jchunk2), l)
        call N(zc_vdot)(l, w(jchunk1), w(jchunk2), dd)
        cz(k) = cz(k) + dd
      end do
      call N(zc_vdot)(l, w(jchunk1), w(jchunk1), dd)
      cz(ncnt) = cz(ncnt) + dd
    end do
  end if
  if (split) then
    ! the overlaps are sums over the slices of the ranks; with the small parts
    ! of rank 0 they are the same numbers on every rank
    call xzmpi_allsum_w(cz, NW * ncnt)
  end if
  d(1:ncnt) = d(1:ncnt) + cz(1:ncnt)
  s3 = dble(cz(ncnt))
  if (split) call xzmpi_bcast(d, 8_8 * NW * ncnt)
  norms = [sqrt(abs(s1)), 2d0 * sqrt(abs(s2)), 6d0 * sqrt(abs(s3))]
  do k = 1, ncnt
    hmat(slot, ord(k)) = d(k)
    hmat(ord(k), slot) = conjg(hmat(slot, ord(k)))
  end do
  do k = 1, ncnt
    do l = 1, ncnt
      h(k, l) = hmat(ord(k), ord(l))
    end do
  end do
  call xzdiis_solve(ncnt, h, int(maxdiis, 8), c)
  do k = 1, ncnt
#ifdef XZ_REAL
    cz(k) = dble(c(k))
#else
    cz(k) = c(k)
#endif
  end do
  ! the extrapolated vector
  if (small) then
  do k = 1, ncnt
    ptr(k) = loc(w(jx1(ord(k))))
  end do
  call N(zc_vlincomb)(n1, ncnt, ptr, cz, t1)
  do k = 1, ncnt
    ptr(k) = loc(w(jx2(ord(k))))
  end do
  call N(zc_vlincomb)(n2, ncnt, ptr, cz, w(jwork2))
  call N(xzunpack2)(no, nv, npo, nvp, w(jopa), w(jopb), w(jvpa), w(jvpb), w(jwork2), t2)
  end if
  if (.not. infile) then
    do k = 1, ncnt
      ptr(k) = loc(w(jx3(ord(k))))
    end do
    call N(zc_vlincomb)(m3, ncnt, ptr, cz, w(pt))
  else
    do ir = 0, nrl - 1
      off = ir * nchunk
      l = min(nchunk, m3 - off)
      ptr(1) = loc(w(pr + NW * off))
      d(1) = cz(ncnt)
      call N(zc_vlincomb)(l, 1_8, ptr, d, w(pt + NW * off))
      do k = 1, ncnt - 1
        call N(xzdiis_io)('r', ord(k), 0_8, ir, w(jchunk2), l)
        call N(xzaxpy)(l, cz(k), w(jchunk2), w(pt + NW * off))
      end do
    end do
  end if
end subroutine

! ---------------------------------------------------------------------------
! x1 = t1n, x2 = t2n on pairs, e1 = t1n - t1, e2 = (t2n - t2) on pairs
subroutine N(xzdiis_small)(no, nv, npo, nvp, opa, opb, vpa, vpb, t1, t2, t1n, t2n, x1, x2, e1, e2)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(in) :: t1(0:nv-1, 0:no-1), t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: t1n(0:nv-1, 0:no-1), t2n(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(out) :: x1(0:nv-1, 0:no-1), e1(0:nv-1, 0:no-1)
  ZT, intent(out) :: x2(0:nvp-1, 0:npo-1), e2(0:nvp-1, 0:npo-1)
  integer(8) :: p, q
  x1 = t1n
  e1 = t1n - t1
  do p = 0, npo - 1
    do q = 0, nvp - 1
      x2(q, p) = t2n(vpb(q), vpa(q), opb(p), opa(p))
      e2(q, p) = x2(q, p) - t2(vpb(q), vpa(q), opb(p), opa(p))
    end do
  end do
end subroutine

! t2[i][j][a][b] from its part on i<j, a<b
subroutine N(xzunpack2)(no, nv, npo, nvp, opa, opb, vpa, vpb, x2, t2)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(in) :: x2(0:nvp-1, 0:npo-1)
  ZT, intent(out) :: t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  integer(8) :: p, q, i, j, a, b
  t2 = 0
  do p = 0, npo - 1
    i = opa(p)
    j = opb(p)
    do q = 0, nvp - 1
      a = vpa(q)
      b = vpb(q)
      t2(b, a, j, i) = x2(q, p)
      t2(a, b, j, i) = -x2(q, p)
      t2(b, a, i, j) = -x2(q, p)
      t2(a, b, i, j) = x2(q, p)
    end do
  end do
end subroutine

! s = sum |a - b|**2
subroutine N(xzdiff2)(n, a, b, s)
  implicit none
  integer(8), intent(in) :: n
  ZT, intent(in) :: a(n), b(n)
  real(8), intent(out) :: s
  integer(8) :: i
  s = 0d0
  !$omp parallel do reduction(+:s)
  do i = 1, n
    s = s + dble((a(i) - b(i)) * CONJ(a(i) - b(i)))
  end do
end subroutine

! y += alpha x
subroutine N(xzaxpy)(n, alpha, x, y)
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
