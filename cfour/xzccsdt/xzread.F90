!
! The input of xzccsdt: what xjoda, xsox2chf and xx2cvtran (ABCDTYPE=STANDARD)
! have left in the run directory.
!
!   nmototal   the number of spinors of the SCF
!   scf2ceig   their energies (complex numbers; the real parts are used), in
!              the order frozen, active occupied, active virtual, dropped
!   JOBARC     SCFENEG, NUMDROPA (frozen Kramers pairs), IFLAGS
!   moinhhhh   <ij||kl>  [i<=j, k<=l]
!   moinhhhp   <ij||ka>  [i<=j, k, a]
!   moinhhpp   <ij||ab>  [a, i, b, j]
!   moinphhp   <ia||bj>  [b, i, a, j]
!   PPPH0001.. <ab||ci>  [a<=b, c], one file per occupied i
!   moinpppp   <ab||cd>  [a<=b, c<=d]
!
! Each file is one record of an unformatted sequential file, complex(8), the
! first index fastest; a pair p<=q has the number p + q (q + 1) / 2 (from 0),
! its diagonal is there and is zero.  The integrals are antisymmetrized,
! <pq||rs> = (pr|qs) - (ps|qr).  (moinpphh is the conjugate of moinhhpp and
! is not read.)
!

! ---------------------------------------------------------------------------
! One record of an unformatted sequential file, n double words.
subroutine xzrfile(fname, buf, n)
  implicit none
  character(len=*), intent(in) :: fname
  integer(8), intent(in) :: n
  real(8), intent(out) :: buf(n)
  integer(8) :: nbytes
  integer :: ios
  logical :: there
  inquire(file=fname, exist=there, size=nbytes)
  if (.not. there) then
    write(6, '(/,a,a)') ' @XZCCSDT-F, no file ', trim(fname)
    if (fname == 'moinpppp') write(6, '(a)') &
      '   xx2cvtran writes <ab||cd> only with ABCDTYPE=STANDARD.'
    call errex
  end if
  if (nbytes < 8 * n + 8) then
    write(6, '(/,a,a,a,i16,a,i16,a)') ' @XZCCSDT-F, the file ', trim(fname), ' has ', nbytes, &
      ' bytes; ', 8 * n + 8, ' are expected for the sizes of this run.'
    call errex
  end if
  open(unit=71, file=fname, form='unformatted', access='sequential', status='old', &
       action='read', iostat=ios)
  if (ios == 0) read(71, iostat=ios) buf
  if (ios /= 0) then
    write(6, '(/,a,a,a,i8)') ' @XZCCSDT-F, reading the file ', trim(fname), ' failed, iostat ', ios
    call errex
  end if
  close(71)
end subroutine xzrfile

! ---------------------------------------------------------------------------
! The sizes of the calculation.  no and nv from the lengths of moinhhhh and
! moinhhpp (as the Python reader does), checked against nmototal and JOBARC.
subroutine xzsizes()
  use xzdata
  implicit none
  integer(8) :: nb, po, nel, idrop, k
  logical :: there
  character(len=8), parameter :: need(6) = &
    [character(len=8) :: 'nmototal', 'scf2ceig', 'moinhhhh', 'moinhhhp', 'moinhhpp', 'moinphhp']
  do k = 1, 6
    inquire(file=trim(need(k)), exist=there)
    if (.not. there) then
      write(6, '(/,a,a,a)') ' @XZCCSDT-F, no file ', trim(need(k)), &
        ' (run after xjoda, xsox2catm, xsox2chf and xx2cvtran).'
      call errex
    end if
  end do
  inquire(file='moinhhhh', size=nb)
  nel = (nb - 8) / 16
  po = nint(sqrt(dble(nel)), 8)
  no = nint((sqrt(8d0 * dble(po) + 1d0) - 1d0) / 2d0, 8)
  if (no < 1 .or. (no * (no + 1) / 2) ** 2 /= nel) then
    write(6, '(/,a)') ' @XZCCSDT-F, moinhhhh is not a block on pairs of occupied spinors.'
    call errex
  end if
  inquire(file='moinhhpp', size=nb)
  nel = (nb - 8) / 16
  nv = nint(sqrt(dble(nel)) / dble(no), 8)
  if (nv < 1 .or. (no * nv) ** 2 /= nel) then
    write(6, '(/,a)') ' @XZCCSDT-F, moinhhhh and moinhhpp do not belong together.'
    call errex
  end if
  open(unit=71, file='nmototal', form='unformatted', access='sequential', status='old', action='read')
  read(71) ntot
  close(71)
  call getrec(20, 'JOBARC', 'NUMDROPA', 1_8, idrop)
  nfrozen = 2 * idrop
  ndropped = ntot - nfrozen - no - nv
  if (ndropped < 0) then
    write(6, '(/,a,4i7)') ' @XZCCSDT-F, the integral files are not those of this SCF: ' // &
      'spinors, frozen, occupied, virtual', ntot, nfrozen, no, nv
    call errex
  end if
  call getrec(20, 'JOBARC', 'SCFENEG ', 1_8, escf)
  ny = no + nv
  npo = no * (no - 1) / 2
  nvp = nv * (nv - 1) / 2
  nto = no * (no - 1) * (no - 2) / 6
  nvt = nv * (nv - 1) * (nv - 2) / 6
end subroutine xzsizes

! ---------------------------------------------------------------------------
! The energies of the active spinors, occupied first.
subroutine xzrdeps(eps, buf)
  use xzdata
  implicit none
  real(8), intent(out) :: eps(0:ny-1)
  real(8), intent(out) :: buf(2, 0:ntot-1)
  integer(8) :: p
  call xzrfile('scf2ceig', buf, 2 * ntot)
  do p = 0, ny - 1
    eps(p) = buf(1, nfrozen + p)
  end do
end subroutine xzrdeps

! ---------------------------------------------------------------------------
! The small blocks; buf: 2 * max(po**2, po*no*nv, (no*nv)**2) words.
subroutine xzrdsmall(oooo, ooov, oovv, ovvo, oovvp, gpp, vpa, vpb, opa, opb, buf)
  use xzdata
  implicit none
  complex(8), intent(out) :: oooo(0:no-1, 0:no-1, 0:no-1, 0:no-1)     ! [m][n][i][j]
  complex(8), intent(out) :: ooov(0:nv-1, 0:no-1, 0:no-1, 0:no-1)     ! [m][n][i][e]
  complex(8), intent(out) :: oovv(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)     ! [m][n][e][f]
  complex(8), intent(out) :: ovvo(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)     ! [m][b][e][j]
  complex(8), intent(out) :: oovvp(0:nvp-1, 0:no-1, 0:no-1)           ! [m][n][Q]
  complex(8), intent(out) :: gpp(0:nvp-1, 0:npo-1)                    ! [P][Q]
  integer(8), intent(in) :: vpa(0:nvp-1), vpb(0:nvp-1), opa(0:npo-1), opb(0:npo-1)
  complex(8) :: buf(0:*)
  integer(8) :: po, m, n, i, j, e, f, b, q, p, pmn, pij
  real(8) :: smn, sij

  po = no * (no + 1) / 2
  call xzrfile('moinhhhh', buf, 2 * po * po)
  do m = 0, no - 1
    do n = 0, no - 1
      pmn = min(m, n) + max(m, n) * (max(m, n) + 1) / 2
      smn = merge(1d0, -1d0, m < n)
      if (m == n) smn = 0d0
      do i = 0, no - 1
        do j = 0, no - 1
          pij = min(i, j) + max(i, j) * (max(i, j) + 1) / 2
          sij = merge(1d0, -1d0, i < j)
          if (i == j) sij = 0d0
          oooo(j, i, n, m) = smn * sij * buf(pmn + po * pij)
        end do
      end do
    end do
  end do
  call xzrfile('moinhhhp', buf, 2 * po * no * nv)
  do m = 0, no - 1
    do n = 0, no - 1
      pmn = min(m, n) + max(m, n) * (max(m, n) + 1) / 2
      smn = merge(1d0, -1d0, m < n)
      if (m == n) smn = 0d0
      do i = 0, no - 1
        do e = 0, nv - 1
          ooov(e, i, n, m) = smn * buf(pmn + po * (i + no * e))
        end do
      end do
    end do
  end do
  ! <mn||ef> = file[e, m, f, n]
  call xzrfile('moinhhpp', buf, 2 * no * nv * no * nv)
  do m = 0, no - 1
    do n = 0, no - 1
      do e = 0, nv - 1
        do f = 0, nv - 1
          oovv(f, e, n, m) = buf(e + nv * (m + no * (f + nv * n)))
        end do
      end do
    end do
  end do
  ! <mb||ej> = file[e, m, b, j]
  call xzrfile('moinphhp', buf, 2 * no * nv * no * nv)
  do m = 0, no - 1
    do b = 0, nv - 1
      do e = 0, nv - 1
        do j = 0, no - 1
          ovvo(j, e, b, m) = buf(e + nv * (m + no * (b + nv * j)))
        end do
      end do
    end do
  end do
  do m = 0, no - 1
    do n = 0, no - 1
      do q = 0, nvp - 1
        oovvp(q, n, m) = oovv(vpb(q), vpa(q), n, m)
      end do
    end do
  end do
  do p = 0, npo - 1
    gpp(:, p) = oovvp(:, opb(p), opa(p))
  end do
end subroutine xzrdsmall

! ---------------------------------------------------------------------------
! <my||de> for d<e as [m][y][Q] from the files PPPH: <my||de> = - <de||ym>*.
! buf: pv * nv complex numbers, pv = nv (nv + 1) / 2.
subroutine xzrdovvq(ovvq, vpa, vpb, buf)
  use xzdata
  implicit none
  complex(8), intent(out) :: ovvq(0:nvp-1, 0:nv-1, 0:no-1)     ! [m][y][Q]
  integer(8), intent(in) :: vpa(0:nvp-1), vpb(0:nvp-1)
  complex(8) :: buf(0:nv*(nv+1)/2-1, 0:nv-1)
  integer(8) :: m, y, q, pv
  character(len=8) :: fname
  pv = nv * (nv + 1) / 2
  do m = 0, no - 1
    write(fname, '(a4,i4.4)') 'PPPH', m + 1
    call xzrfile(fname, buf, 2 * pv * nv)
    !$omp parallel do private(y, q)
    do y = 0, nv - 1
      do q = 0, nvp - 1
        ovvq(q, y, m) = -conjg(buf(vpa(q) + vpb(q) * (vpb(q) + 1) / 2, y))
      end do
    end do
  end do
end subroutine xzrdovvq

! ---------------------------------------------------------------------------
! <ab||cd> for a<b, c<d as [Q][Q'] from moinpppp, read into the place where
! it stays: v holds the pv**2 numbers of the file (pv = nv (nv + 1) / 2) and
! at the end the nvp**2 packed ones in its first part.  The strict pairs are
! compacted in the order of the file, then rows and columns are permuted to
! the order of the program.  row: nvp complex numbers, src: 2 * nvp integers.
subroutine xzrdvvvv(v, vpidx, row, src)
  use xzdata
  implicit none
  complex(8) :: v(0:*)
  integer(8), intent(in) :: vpidx(0:nv-1, 0:nv-1)
  complex(8) :: row(0:nvp-1)
  integer(8) :: src(0:nvp-1, 0:1)
  integer(8) :: pv, a, b, c, d, q, qp, kd, ks, r, s, t

  pv = nv * (nv + 1) / 2
  call xzrfile('moinpppp', v, 2 * pv * pv)
  ! the strict pairs, in the order of the file: number a + b (b - 1) / 2.
  ! As a row-major array this is [(cd)][(ab)] = <ab||cd> = <cd||ab>*.
  kd = 0
  do d = 1, nv - 1
    do c = 0, d - 1
      qp = c + d * (d + 1) / 2
      do b = 1, nv - 1
        ks = b * (b + 1) / 2 + pv * qp
        do a = 0, b - 1
          v(kd) = conjg(v(ks + a))
          kd = kd + 1
        end do
      end do
    end do
  end do
  ! src(Q): the place in the order of the file of the pair with the number Q
  do b = 1, nv - 1
    do a = 0, b - 1
      src(vpidx(b, a), 0) = a + b * (b - 1) / 2
    end do
  end do
  ! columns
  !$omp parallel do private(q, qp, row)
  do q = 0, nvp - 1
    do qp = 0, nvp - 1
      row(qp) = v(src(qp, 0) + nvp * q)
    end do
    v(nvp * q:nvp * q + nvp - 1) = row
  end do
  ! rows, along the cycles of the permutation
  src(:, 1) = 0
  do s = 0, nvp - 1
    if (src(s, 1) /= 0) cycle
    src(s, 1) = 1
    if (src(s, 0) == s) cycle
    row = v(nvp * s:nvp * s + nvp - 1)
    r = s
    do
      t = src(r, 0)
      if (t == s) exit
      v(nvp * r:nvp * r + nvp - 1) = v(nvp * t:nvp * t + nvp - 1)
      src(t, 1) = 1
      r = t
    end do
    v(nvp * r:nvp * r + nvp - 1) = row
  end do
end subroutine xzrdvvvv
