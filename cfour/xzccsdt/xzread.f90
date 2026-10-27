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
!   xzlabels   optional, not a file of CFOUR: a symmetry label per spinor
!              (xzrdlab)
!   dgcosym    the block of the SCF of every spinor (xsox2chf; m_j blocks
!              with %dglin in the ZMAT): the labels if there is no xzlabels
!              (xzrddgc)
!
! With a section %vtranblk in the ZMAT xx2cvtran writes instead the files
! mobkhhhh, mobkhhhp, mobkhhpp, mobkphhp, PPBK0001.. and mobkpppp: the same
! integrals as real numbers, only the elements that the labels of xzlabels
! allow and only strict pairs (xzbwrite.f90 of x2cvtran: the format, a
! header and the labels of the file first).  They are used if mobkhhhh is
! there (xzrint, xzrbvvvv): the labels of the files decide which elements
! are there, the labels of this program (or none) how they are stored.
!
! Each integral file is one record of an unformatted sequential file,
! complex(8), the first index fastest; a pair p<=q has the number
! p + q (q + 1) / 2 (from 0), its diagonal is there and is zero.  The
! integrals are antisymmetrized, <pq||rs> = (pr|qs) - (ps|qr).  (moinpphh is
! the conjugate of moinhhpp and is not read.)
!
! The program may have the spinors in another order than CFOUR (the order of
! the classes of their symmetry labels, xzdata.f90): operm[i] and vperm[a]
! are CFOUR's numbers of the occupied spinor i and of the virtual spinor a
! of the program.
!

! ---------------------------------------------------------------------------
! The bytes of data of a file of nbytes bytes that is one record of a
! sequential unformatted file of ifort: a record of more than 2**31 - 9 bytes
! is a chain of sub-records, each with two markers of 4 bytes (a Fortran READ
! takes them as one record; only the lengths have to be counted here).
integer(8) function xzrecbytes(nbytes)
  implicit none
  integer(8), intent(in) :: nbytes
  integer(8), parameter :: sub = 2147483639_8
  xzrecbytes = nbytes - 8 * ((nbytes + sub + 7) / (sub + 8))
end function xzrecbytes

! ---------------------------------------------------------------------------
! One record of an unformatted sequential file, n double words.
subroutine xzrfile(fname, buf, n)
  implicit none
  character(len=*), intent(in) :: fname
  integer(8), intent(in) :: n
  real(8), intent(out) :: buf(n)
  integer(8) :: nbytes
  integer(8), external :: xzrecbytes
  integer :: ios
  logical :: there
  inquire(file=fname, exist=there, size=nbytes)
  if (.not. there) then
    write(6, '(/,a,a)') ' @XZCCSDT-F, no file ', trim(fname)
    if (fname == 'moinpppp') write(6, '(a)') &
      '   xx2cvtran writes <ab||cd> only with ABCDTYPE=STANDARD.'
    call xzerrex
  end if
  if (xzrecbytes(nbytes) /= 8 * n) then
    write(6, '(/,a,a,a,i16,a,i16,a)') ' @XZCCSDT-F, the file ', trim(fname), ' has ', xzrecbytes(nbytes), &
      ' bytes of data; ', 8 * n, ' are expected for the sizes of this run.'
    call xzerrex
  end if
  open(unit=71, file=fname, form='unformatted', access='sequential', status='old', &
       action='read', iostat=ios)
  if (ios == 0) read(71, iostat=ios) buf
  if (ios /= 0) then
    write(6, '(/,a,a,a,i8)') ' @XZCCSDT-F, reading the file ', trim(fname), ' failed, iostat ', ios
    call xzerrex
  end if
  close(71)
end subroutine xzrfile

! ---------------------------------------------------------------------------
! The sizes of the calculation.  no and nv from the lengths of moinhhhh and
! moinhhpp (as the Python reader does), checked against nmototal and JOBARC.
subroutine xzsizes()
  use xzdata
  use xzpar
  implicit none
  integer(8) :: nb, po, nel, idrop, k, ib(1)
  integer(8), external :: xzrecbytes
  real(8) :: rb(1)
  logical :: there
  integer(8) :: hdr(8)
  integer :: ios
  character(len=8) :: need(6)
  need = [character(len=8) :: 'nmototal', 'scf2ceig', 'moinhhhh', 'moinhhhp', 'moinhhpp', 'moinphhp']
  inquire(file='mobkhhhh', exist=blkin)
  if (blkin) need(3:6) = [character(len=8) :: 'mobkhhhh', 'mobkhhhp', 'mobkhhpp', 'mobkphhp']
  do k = 1, 6
    inquire(file=trim(need(k)), exist=there)
    if (.not. there) then
      write(6, '(/,a,a,a)') ' @XZCCSDT-F, no file ', trim(need(k)), &
        ' (run after xjoda, xsox2catm, xsox2chf and xx2cvtran).'
      call xzerrex
    end if
  end do
  if (blkin) then
    open(unit=71, file='mobkhhhh', form='unformatted', access='sequential', status='old', &
         action='read', iostat=ios)
    if (ios == 0) read(71, iostat=ios) hdr
    close(71)
    if (ios /= 0 .or. hdr(1) /= 20261015_8 .or. hdr(2) < 1 .or. hdr(2) > 3) then
      write(6, '(/,a)') ' @XZCCSDT-F, mobkhhhh is not a file of xx2cvtran with %vtranblk.'
      call xzerrex
    end if
    no = hdr(7)
    nv = hdr(8)
  else
    inquire(file='moinhhhh', size=nb)
    nel = xzrecbytes(nb) / 16
    po = nint(sqrt(dble(nel)), 8)
    no = nint((sqrt(8d0 * dble(po) + 1d0) - 1d0) / 2d0, 8)
    if (no < 1 .or. (no * (no + 1) / 2) ** 2 /= nel) then
      write(6, '(/,a)') ' @XZCCSDT-F, moinhhhh is not a block on pairs of occupied spinors.'
      call xzerrex
    end if
    inquire(file='moinhhpp', size=nb)
    nel = xzrecbytes(nb) / 16
    nv = nint(sqrt(dble(nel)) / dble(no), 8)
    if (nv < 1 .or. (no * nv) ** 2 /= nel) then
      write(6, '(/,a)') ' @XZCCSDT-F, moinhhhh and moinhhpp do not belong together.'
      call xzerrex
    end if
  end if
  open(unit=71, file='nmototal', form='unformatted', access='sequential', status='old', action='read')
  read(71) ntot
  close(71)
  ! (JOBARC is the business of rank 0)
  if (myrank == 0) then
    call getrec(20, 'JOBARC', 'NUMDROPA', 1_8, idrop)
    call getrec(20, 'JOBARC', 'SCFENEG ', 1_8, escf)
  end if
  ib(1) = idrop
  rb(1) = escf
  call xzmpi_bcast(ib, 8_8)
  call xzmpi_bcast(rb, 8_8)
  idrop = ib(1)
  escf = rb(1)
  nfrozen = 2 * idrop
  ndropped = ntot - nfrozen - no - nv
  if (ndropped < 0) then
    write(6, '(/,a,4i7)') ' @XZCCSDT-F, the integral files are not those of this SCF: ' // &
      'spinors, frozen, occupied, virtual', ntot, nfrozen, no, nv
    call xzerrex
  end if
  if (blkin) then
    if (hdr(4) /= ntot .or. hdr(6) /= nfrozen) then
      write(6, '(/,a,4i7)') ' @XZCCSDT-F, the files mobk.. are not those of this SCF: spinors, frozen ' // &
        '(files, here)', hdr(4), hdr(6), ntot, nfrozen
      call xzerrex
    end if
    if (myrank == 0) write(6, '(a)') '   integrals: the blocked real files of xx2cvtran (%vtranblk)'
  end if
  ny = no + nv
  npo = no * (no - 1) / 2
  nvp = nv * (nv - 1) / 2
  nto = no * (no - 1) * (no - 2) / 6
  nvt = nv * (nv - 1) * (nv - 2) / 6
end subroutine xzsizes

! ---------------------------------------------------------------------------
! The symmetry labels of the spinors from the file xzlabels, if it is there:
! a text file, lines that start with # are comments; the number of spinors
! and the modulus, then one integer label per spinor of the SCF in the order
! of scf2ceig (frozen, active occupied, active virtual, dropped).  A label
! adds: <pq||rs> is zero unless l_p + l_q = l_r + l_s (modulo the modulus if
! that is not 0) - 2 m_j of a linear molecule (no modulus).  Optionally a
! second label: a line 'second m2', then one more integer per spinor, which
! adds modulo m2 as well (the g/u parity of D2h: 0 and 1, m2 = 2).
!   lab(0:nlab-1), lab2 (modul2 = 0: no second label); there = .false.: no file
subroutine xzrdlab(nlab, lab, modul, lab2, modul2, there)
  implicit none
  integer(8), intent(in) :: nlab
  integer(8), intent(out) :: lab(0:nlab-1), modul, lab2(0:nlab-1), modul2
  logical, intent(out) :: there
  character(len=1024) :: line
  integer(8) :: n, k, v(64), have, have2
  integer :: ios, i, nw, c1
  logical :: first, sec
  modul = 0
  modul2 = 0
  lab2 = 0
  inquire(file='xzlabels', exist=there)
  if (.not. there) return
  open(unit=71, file='xzlabels', form='formatted', status='old', action='read')
  first = .true.
  sec = .false.
  have = 0
  have2 = 0
  n = -1
  do
    read(71, '(a)', iostat=ios) line
    if (ios /= 0) exit
    c1 = verify(line, ' ')
    if (c1 == 0) cycle
    if (line(c1:c1) == '#') cycle
    if (line(c1:min(c1 + 5, len(line))) == 'second') then
      read(line(c1 + 6:), *, iostat=ios) modul2
      if (ios /= 0 .or. modul2 < 1) then
        write(6, '(/,a)') ' @XZCCSDT-F, the file xzlabels: the line ''second'' needs the modulus of the second label.'
        call xzerrex
      end if
      sec = .true.
      cycle
    end if
    ! the integers of the line
    nw = 0
    do i = 1, 64
      read(line, *, iostat=ios) v(1:i)
      if (ios /= 0) exit
      nw = i
    end do
    k = 1
    if (first .and. nw >= 2) then
      n = v(1)
      modul = v(2)
      first = .false.
      k = 3
    end if
    do i = int(k), nw
      if (sec) then
        if (have2 < nlab) lab2(have2) = v(i)
        have2 = have2 + 1
      else
        if (have < nlab) lab(have) = v(i)
        have = have + 1
      end if
    end do
  end do
  close(71)
  if (n /= nlab .or. have /= nlab .or. (sec .and. have2 /= nlab)) then
    write(6, '(/,a,i6,a,i6,a,i6,a,i6,a)') ' @XZCCSDT-F, the file xzlabels: ', n, ' spinors announced, ', have, &
      ' labels (', have2, ' second labels) found; the SCF has ', nlab, ' spinors.'
    call xzerrex
  end if
  if (sec) lab2 = modulo(lab2, modul2)
end subroutine xzrdlab

! ---------------------------------------------------------------------------
! The symmetry labels of the spinors from the file dgcosym of CFOUR's
! two-component SCF (xsox2chf), for a run directory without xzlabels - the
! rule of zccsdt.cfour.read_dgcosym / read_labels.  The file is one record
! of nlab 8-byte integers: the block of the SCF of every spinor in the order
! of scf2ceig; all 1 unless the ZMAT has a section %dglin, with which the
! SCF is done in blocks of m_j (linsym of sox2chf): block 2k - 1 is
! m_j = +(2k - 1)/2, block 2k is m_j = -(2k - 1)/2.  A block number is not a
! label that adds; the label is 2 m_j, +(2k - 1) and -(2k - 1), with no
! modulus.  (This routine is the only place where the numbering of the
! blocks is known; a table for another kind of blocks would give labels and
! a modulus here, and the caller takes both as it takes those of xzlabels.)
!   there = .false.: no file, or one block only (no symmetry in the SCF)
!   nblk: the largest block number of the file (0: no file)
subroutine xzrddgc(nlab, lab, modul, there, nblk)
  implicit none
  integer(8), intent(in) :: nlab
  integer(8), intent(out) :: lab(0:nlab-1), modul, nblk
  logical, intent(out) :: there
  integer(8) :: nbytes, p, k
  integer(8), external :: xzrecbytes
  integer :: ios
  modul = 0
  nblk = 0
  inquire(file='dgcosym', exist=there, size=nbytes)
  if (.not. there) return
  if (xzrecbytes(nbytes) /= 8 * nlab) then
    write(6, '(/,a,i12,a,i8,a)') ' @XZCCSDT-F, the file dgcosym has ', xzrecbytes(nbytes), &
      ' bytes of data; one 8-byte integer for each of the ', nlab, ' spinors of the SCF'
    write(6, '(a,i12,a)') '             is expected (nmototal), ', 8 * nlab, &
      ' bytes.  (The file xzlabels, if it is there, is used instead.)'
    call xzerrex
  end if
  open(unit=71, file='dgcosym', form='unformatted', access='sequential', status='old', &
       action='read', iostat=ios)
  if (ios == 0) read(71, iostat=ios) lab
  if (ios /= 0) then
    write(6, '(/,a,i8)') ' @XZCCSDT-F, reading the file dgcosym failed, iostat ', ios
    call xzerrex
  end if
  close(71)
  nblk = maxval(lab)
  if (minval(lab) < 1) then
    write(6, '(/,a,i12,a)') ' @XZCCSDT-F, the file dgcosym: a block number ', minval(lab), &
      '; the blocks of the SCF are numbered from 1.'
    call xzerrex
  end if
  there = nblk > 1
  if (.not. there) return
  do p = 0, nlab - 1
    k = (lab(p) + 1) / 2
    lab(p) = merge(2 * k - 1, -(2 * k - 1), mod(lab(p), 2_8) == 1)
  end do
end subroutine xzrddgc

! ---------------------------------------------------------------------------
! The energies of the active spinors, occupied first, in the order of the
! program.
subroutine xzrdeps(eps, buf)
  use xzdata
  implicit none
  real(8), intent(out) :: eps(0:ny-1)
  real(8), intent(out) :: buf(2, 0:ntot-1)
  integer(8) :: p
  call xzrfile('scf2ceig', buf, 2 * ntot)
  do p = 0, no - 1
    eps(p) = buf(1, nfrozen + operm(p))
  end do
  do p = 0, nv - 1
    eps(no + p) = buf(1, nfrozen + no + vperm(p))
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
  integer(8) :: po, m, n, i, j, e, f, b, q, p, pmn, pij, fm, fn, fi, fj, fe, ff, fb
  real(8) :: smn, sij

  po = no * (no + 1) / 2
  call xzrint(1_8, 'moinhhhh', buf, 2 * po * po)
  !$omp parallel do private(m, n, i, j, fm, fn, fi, fj, pmn, pij, smn, sij)
  do m = 0, no - 1
    do n = 0, no - 1
      fm = operm(m)
      fn = operm(n)
      pmn = min(fm, fn) + max(fm, fn) * (max(fm, fn) + 1) / 2
      smn = merge(1d0, -1d0, fm < fn)
      if (m == n) smn = 0d0
      do i = 0, no - 1
        do j = 0, no - 1
          fi = operm(i)
          fj = operm(j)
          pij = min(fi, fj) + max(fi, fj) * (max(fi, fj) + 1) / 2
          sij = merge(1d0, -1d0, fi < fj)
          if (i == j) sij = 0d0
          oooo(j, i, n, m) = smn * sij * buf(pmn + po * pij)
        end do
      end do
    end do
  end do
  call xzrint(2_8, 'moinhhhp', buf, 2 * po * no * nv)
  !$omp parallel do private(m, n, i, e, fm, fn, pmn, smn)
  do m = 0, no - 1
    do n = 0, no - 1
      fm = operm(m)
      fn = operm(n)
      pmn = min(fm, fn) + max(fm, fn) * (max(fm, fn) + 1) / 2
      smn = merge(1d0, -1d0, fm < fn)
      if (m == n) smn = 0d0
      do i = 0, no - 1
        do e = 0, nv - 1
          ooov(e, i, n, m) = smn * buf(pmn + po * (operm(i) + no * vperm(e)))
        end do
      end do
    end do
  end do
  ! <mn||ef> = file[e, m, f, n]
  call xzrint(3_8, 'moinhhpp', buf, 2 * no * nv * no * nv)
  !$omp parallel do collapse(2) private(m, n, e, f, fm, fn, fe, ff)
  do m = 0, no - 1
    do n = 0, no - 1
      fm = operm(m)
      fn = operm(n)
      do e = 0, nv - 1
        fe = vperm(e)
        do f = 0, nv - 1
          ff = vperm(f)
          oovv(f, e, n, m) = buf(fe + nv * (fm + no * (ff + nv * fn)))
        end do
      end do
    end do
  end do
  ! <mb||ej> = file[e, m, b, j]
  call xzrint(4_8, 'moinphhp', buf, 2 * no * nv * no * nv)
  !$omp parallel do private(m, b, e, j, fm, fb, fe)
  do m = 0, no - 1
    fm = operm(m)
    do b = 0, nv - 1
      fb = vperm(b)
      do e = 0, nv - 1
        fe = vperm(e)
        do j = 0, no - 1
          ovvo(j, e, b, m) = buf(fe + nv * (fm + no * (fb + nv * operm(j))))
        end do
      end do
    end do
  end do
  !$omp parallel do collapse(2) private(m, n, q)
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
  integer(8) :: m, y, q, pv, fd, fe, fy
  character(len=8) :: fname
  if (blkin) then
    call xzrbovvq(ovvq, vpa, vpb)
    return
  end if
  pv = nv * (nv + 1) / 2
  do m = 0, no - 1
    write(fname, '(a4,i4.4)') 'PPPH', operm(m) + 1
    call xzrint(5_8, fname, buf, 2 * pv * nv)
    !$omp parallel do private(y, q, fd, fe, fy)
    do y = 0, nv - 1
      fy = vperm(y)
      do q = 0, nvp - 1
        fd = vperm(vpa(q))
        fe = vperm(vpb(q))
        if (fd < fe) then
          ovvq(q, y, m) = -conjg(buf(fd + fe * (fe + 1) / 2, fy))
        else
          ovvq(q, y, m) = conjg(buf(fe + fd * (fd + 1) / 2, fy))
        end if
      end do
    end do
  end do
end subroutine xzrdovvq

! ---------------------------------------------------------------------------
! <my||de> for d<e as [m][y][Q] (xzrdovvq) from the blocked files PPBK of
! xx2cvtran, their elements put straight into their places: the file of the
! occupied m has <de||ym> on its strict pairs (de) (CFOUR's order), by
! columns y, the rows that the labels allow in the order of the pairs
! (xzbidx, kind 5).  <my||de> = - <de||ym>*; the pair of the program has the
! two spinors in the other order if sgf = -1.
subroutine xzrbovvq(ovvq, vpa, vpb)
  use xzdata
  implicit none
  complex(8), intent(out) :: ovvq(0:nvp-1, 0:nv-1, 0:no-1)     ! [m][y][Q]
  integer(8), intent(in) :: vpa(0:nvp-1), vpb(0:nvp-1)
  integer(8) :: flab(0:ntot-1), flab0(0:ntot-1), fmod(4), fmod0(4), vl(0:nv-1), ol
  integer(8) :: m, y, q, a, b, c, f, k, kk, cl, cnt, ncl, iocc
  integer(8), allocatable :: inv(:), vinv(:), fid(:), rows(:), roff(:), rep(:), ccl(:)
  real(8), allocatable :: sgf(:), d(:)
  integer(8), external :: xzpclass
  logical, external :: xzbok
  logical :: walk
  integer :: ios
  character(len=8) :: bname
  allocate(inv(0:nvp-1), vinv(0:nv-1), sgf(0:nvp-1), fid(0:nvp-1), rows(0:nvp-1), roff(0:nvp), &
           rep(0:nvp-1), ccl(0:nv-1))
  do q = 0, nvp - 1
    a = vperm(vpa(q))
    b = vperm(vpb(q))
    f = min(a, b) + max(a, b) * (max(a, b) - 1) / 2
    inv(f) = q
    sgf(f) = merge(1d0, -1d0, a < b)
  end do
  do y = 0, nv - 1
    vinv(vperm(y)) = y
  end do
  !$omp parallel do private(m, y)
  do m = 0, no - 1
    do y = 0, nv - 1
      ovvq(:, y, m) = 0
    end do
  end do
  do m = 0, no - 1
    iocc = operm(m)
    write(bname, '(a4,i4.4)') 'PPBK', iocc + 1
    call xzbhead(bname, 5_8, flab, fmod)
    if (m == 0 .or. any(flab /= flab0) .or. any(fmod /= fmod0)) then
      ! (the labels of the files: the same in all of them)
      vl = flab(nfrozen + no:nfrozen + no + nv - 1)
      call xzpcls(nv, vl, fmod, fid, rows, roff, rep, ncl, walk)
      flab0 = flab
      fmod0 = fmod
    end if
    ol = flab(nfrozen + iocc)
    ! the class of the rows of each column c (walk), and their number
    cnt = 0
    do c = 0, nv - 1
      if (walk) then
        ccl(c) = xzpclass(vl(c) + ol, rep, ncl, fmod)
        if (ccl(c) >= 0) cnt = cnt + roff(ccl(c) + 1) - roff(ccl(c))
      else
        do b = 1, nv - 1
          do a = 0, b - 1
            if (xzbok(vl(a) + vl(b) - vl(c) - ol, fmod)) cnt = cnt + 1
          end do
        end do
      end if
    end do
    allocate(d(max(fmod(4) * cnt, 1_8)))
    read(71, iostat=ios) d(1:fmod(4) * cnt)
    close(71)
    if (ios /= 0) then
      write(6, '(/,a,a)') ' @XZCCSDT-F, reading the elements of the file ', trim(bname)
      call xzerrex
    end if
    k = 0
    do c = 0, nv - 1
      y = vinv(c)
      if (walk) then
        cl = ccl(c)
        if (cl < 0) cycle
        do kk = roff(cl), roff(cl + 1) - 1
          call put(rows(kk))
        end do
      else
        do b = 1, nv - 1
          do a = 0, b - 1
            if (xzbok(vl(a) + vl(b) - vl(c) - ol, fmod)) call put(a + b * (b - 1) / 2)
          end do
        end do
      end if
    end do
    deallocate(d)
  end do
  deallocate(inv, vinv, sgf, fid, rows, roff, rep, ccl)
contains
  ! the next element of the file, of the strict pair f and the column y
  subroutine put(f)
    integer(8), intent(in) :: f
    complex(8) :: x
    k = k + 1
    if (fmod(4) == 2) then
      x = cmplx(d(2 * k - 1), d(2 * k), 8)
    else
      x = cmplx(d(k), 0d0, 8)
    end if
    ovvq(inv(f), y, m) = -sgf(f) * conjg(x)
  end subroutine put
end subroutine xzrbovvq

! ---------------------------------------------------------------------------
! <ab||cd> for a<b, c<d from moinpppp, read into the place where it stays: v
! holds the pv**2 numbers of the file (pv = nv (nv + 1) / 2) and at the end
! the blocks [Q][Q'] of the pairs of one group (one group: all pairs) one
! after the other in its first part.  The strict pairs are compacted in the
! order of the file, rows and columns are permuted to the order of the
! program, then the diagonal blocks are moved together.  offmax: the largest
! element outside of them (zero by symmetry if the labels are right).
! row: nvp complex numbers, src: 2 * nvp integers, sg: nvp reals.
subroutine xzrdvvvv(v, vpa, vpb, row, src, sg, offmax)
  use xzdata
  implicit none
  complex(8) :: v(0:*)
  integer(8), intent(in) :: vpa(0:nvp-1), vpb(0:nvp-1)
  complex(8) :: row(0:nvp-1)
  integer(8) :: src(0:nvp-1, 0:1)
  real(8) :: sg(0:nvp-1)
  real(8), intent(out) :: offmax
  integer(8) :: pv, a, b, c, d, q, qp, kd, ks, r, s, t, g, h, n, lo, k

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
  ! here, sg(Q): its sign there
  do q = 0, nvp - 1
    a = vperm(vpa(q))
    b = vperm(vpb(q))
    sg(q) = merge(1d0, -1d0, a < b)
    src(q, 0) = min(a, b) + max(a, b) * (max(a, b) - 1) / 2
  end do
  ! columns
  !$omp parallel do private(q, qp, row)
  do q = 0, nvp - 1
    do qp = 0, nvp - 1
      row(qp) = sg(qp) * v(src(qp, 0) + nvp * q)
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
  !$omp parallel do private(q)
  do q = 0, nvp - 1
    v(nvp * q:nvp * q + nvp - 1) = sg(q) * v(nvp * q:nvp * q + nvp - 1)
  end do
  ! the diagonal blocks of the groups of pairs, one after the other
  offmax = 0d0
  if (npg == 1) return
  kd = 0
  do g = 0, npg - 1
    lo = vpoff(g)
    n = vpoff(g + 1) - lo
    do q = lo, lo + n - 1
      do h = 0, npg - 1
        if (h == g) cycle
        do k = vpoff(h), vpoff(h + 1) - 1
          offmax = max(offmax, abs(v(k + nvp * q)))
        end do
      end do
      do k = 0, n - 1
        v(kd + k) = v(lo + k + nvp * q)
      end do
      kd = kd + n
    end do
  end do
end subroutine xzrdvvvv

! ---------------------------------------------------------------------------
! The file xzvvvv (not a file of CFOUR): <ab||cd> for a<b, c<d in the blocks
! of the pairs of one label sum, the elements between pairs of different
! label sums not stored - what `bench/dea/native_rundir.py` writes from a
! block file of zccsdt.hblocks (docs/eom.md 11.10) in the place of moinpppp,
! whose record of (nv (nv + 1) / 2)^2 complex numbers is 60 GB for 350
! virtual spinors.  If the file is there, moinpppp is not read.  A stream
! file, 8 bytes per number, little-endian:
!   'XZVVVV01', nv, nwf (words per element: 1 real, 2 complex), ngrp
!   cnt(ngrp)       the pairs of each group (their sum: nv (nv - 1) / 2)
!   pair(nvp)       the pairs of the groups, group after group: a + b (b - 1) / 2
!                   for a < b, CFOUR's numbers of the virtual spinors from 0
!   the blocks, group after group: [i][j] = <pair_i || pair_j>, row-major
! The groups need not be those of the program (labels 0: one group of all
! pairs; the groups of the file then are parts of it).
!   there: the file is there; nwf; bmax: the elements of its largest block
subroutine xzvvhdr(there, nwf, bmax)
  use xzdata
  implicit none
  logical, intent(out) :: there
  integer(8), intent(out) :: nwf, bmax
  character(len=8) :: tag
  integer(8) :: hdr(3), nbytes, nexp
  integer(8), allocatable :: cnt(:)
  integer :: ios
  nwf = 0
  bmax = 0
  inquire(file='xzvvvv', exist=there, size=nbytes)
  if (.not. there) return
  open(unit=72, file='xzvvvv', access='stream', form='unformatted', status='old', action='read', iostat=ios)
  if (ios == 0) read(72, iostat=ios) tag, hdr
  if (ios /= 0 .or. tag /= 'XZVVVV01') then
    write(6, '(/,a)') ' @XZCCSDT-F, the file xzvvvv is not a file of <vv||vv> in blocks (XZVVVV01).'
    call xzerrex
  end if
  if (hdr(1) /= nv .or. hdr(2) < 1 .or. hdr(2) > 2 .or. hdr(3) < 1) then
    write(6, '(/,a,3i10,a,i8)') ' @XZCCSDT-F, the file xzvvvv: virtual spinors, words, groups ', hdr, &
      '; this run has virtual spinors ', nv
    call xzerrex
  end if
  nwf = hdr(2)
  allocate(cnt(hdr(3)))
  read(72, iostat=ios) cnt
  close(72)
  nexp = 32 + 8 * hdr(3) + 8 * nvp + 8 * nwf * sum(cnt ** 2)
  if (ios /= 0 .or. sum(cnt) /= nvp .or. minval(cnt) < 0 .or. nbytes /= nexp) then
    write(6, '(/,a,i14,a,i14,a)') ' @XZCCSDT-F, the file xzvvvv has ', nbytes, ' bytes; its header announces ', &
      nexp, ' (or its pairs are not those of this run).'
    call xzerrex
  end if
  bmax = maxval(cnt) ** 2
  deallocate(cnt)
end subroutine xzvvhdr

! ---------------------------------------------------------------------------
! <ab||cd> for a<b, c<d from the file xzvvvv (xzvvhdr) into the diagonal
! blocks of the groups of pairs of the program, [Q][Q'] at vvoff(g), nwe
! words per element (the real parts only if nwe = 1).  Block by block of the
! file: one block in buf, each element to its place with the signs of the
! two pairs (the program may number the spinors in another order, operm /
! vperm).  offmax: the largest element of the file between two groups of the
! program (zero by symmetry if the labels are right); ximag: the largest
! imaginary part (a complex file, nwe = 1).
! qof: 2 nvp integers, sg: nvp reals, buf: nwf bmax words.
subroutine xzrdvvb(v, nwf, vpa, vpb, vpgrp, qof, sg, buf, offmax, ximag)
  use xzdata
  implicit none
  integer(8), intent(in) :: nwf
  real(8) :: v(0:*)
  integer(8), intent(in) :: vpa(0:nvp-1), vpb(0:nvp-1), vpgrp(0:nvp-1)
  integer(8) :: qof(0:nvp-1, 0:1)
  real(8) :: sg(0:nvp-1), buf(0:*)
  real(8), intent(out) :: offmax, ximag
  character(len=8) :: tag
  integer(8) :: hdr(3), q, a, b, k, f, n, k0, i, j, qi, qj, gi, lo, nq, kb, idx
  integer(8) :: pos
  integer(8), allocatable :: cnt(:)
  real(8) :: s, x
  integer :: ios

  offmax = 0d0
  ximag = 0d0
  !$omp parallel do
  do k = 0, nwe * nvv - 1
    v(k) = 0d0
  end do
  ! the pair of the program with CFOUR's number s, and its sign there
  do q = 0, nvp - 1
    a = vperm(vpa(q))
    b = vperm(vpb(q))
    sg(q) = merge(1d0, -1d0, a < b)
    qof(min(a, b) + max(a, b) * (max(a, b) - 1) / 2, 0) = q
  end do
  open(unit=72, file='xzvvvv', access='stream', form='unformatted', status='old', action='read', iostat=ios)
  if (ios == 0) read(72, iostat=ios) tag, hdr
  if (ios == 0) then
    allocate(cnt(hdr(3)))
    read(72, iostat=ios) cnt
  end if
  if (ios == 0) read(72, iostat=ios) qof(:, 1)
  if (ios /= 0) then
    write(6, '(/,a,i8)') ' @XZCCSDT-F, reading the file xzvvvv failed, iostat ', ios
    call xzerrex
  end if
  do k = 0, nvp - 1
    if (qof(k, 1) < 0 .or. qof(k, 1) >= nvp) then
      write(6, '(/,a,i14)') ' @XZCCSDT-F, the file xzvvvv: a pair number out of range, ', qof(k, 1)
      call xzerrex
    end if
    qof(k, 1) = qof(qof(k, 1), 0)
  end do
  pos = 33 + 8 * hdr(3) + 8 * nvp
  k0 = 0
  do f = 1, hdr(3)
    n = cnt(f)
    if (n > 0) then
      read(72, pos=pos, iostat=ios) buf(0:nwf*n*n-1)
      if (ios /= 0) then
        write(6, '(/,a,i6,a,i8)') ' @XZCCSDT-F, reading block ', f, ' of the file xzvvvv failed, iostat ', ios
        call xzerrex
      end if
    end if
    pos = pos + 8 * nwf * n * n
    !$omp parallel do private(i, j, qi, qj, gi, lo, nq, kb, idx, s, x) reduction(max:offmax, ximag) schedule(dynamic, 16)
    do i = 0, n - 1
      qi = qof(k0 + i, 1)
      gi = vpgrp(qi)
      lo = vpoff(gi)
      nq = vpoff(gi + 1) - lo
      do j = 0, n - 1
        qj = qof(k0 + j, 1)
        kb = nwf * (i * n + j)
        if (vpgrp(qj) /= gi) then
          x = abs(buf(kb))
          if (nwf == 2) x = hypot(buf(kb), buf(kb + 1))
          offmax = max(offmax, x)
          cycle
        end if
        s = sg(qi) * sg(qj)
        idx = nwe * (vvoff(gi) + (qi - lo) * nq + (qj - lo))
        v(idx) = s * buf(kb)
        if (nwf == 2) then
          if (nwe == 2) then
            v(idx + 1) = s * buf(kb + 1)
          else
            ximag = max(ximag, abs(buf(kb + 1)))
          end if
        end if
      end do
    end do
    k0 = k0 + n
  end do
  close(72)
  deallocate(cnt)
end subroutine xzrdvvb

! ---------------------------------------------------------------------------
! The largest element of the small blocks and of <ov||vv> that the labels
! forbid: an element <pq||rs> with class(p) + class(q) /= class(r) + class(s)
! (modulo mgrp with a modulus: xzeq).
subroutine xzchklab(oooo, ooov, oovv, ovvo, ovvq, vpgrp, vmax, what)
  use xzdata
  implicit none
  complex(8), intent(in) :: oooo(0:no-1, 0:no-1, 0:no-1, 0:no-1)     ! [m][n][i][j]
  complex(8), intent(in) :: ooov(0:nv-1, 0:no-1, 0:no-1, 0:no-1)     ! [m][n][i][e]
  complex(8), intent(in) :: oovv(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)     ! [m][n][e][f]
  complex(8), intent(in) :: ovvo(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)     ! [m][b][e][j]
  complex(8), intent(in) :: ovvq(0:nvp-1, 0:nv-1, 0:no-1)            ! [m][y][Q]
  integer(8), intent(in) :: vpgrp(0:nvp-1)
  real(8), intent(out) :: vmax
  character(len=*), intent(out) :: what
  integer(8) :: m, n, i, j, e, f, b, y, q, g
  real(8) :: x(5)
  x = 0d0
  ! (<ov||vv>: the pairs Q of the group s(m) + s(y) are allowed, the others
  ! not; the pairs are in the order of their groups)
  !$omp parallel do private(m, n, i, j, e, f, b, y, q, g) reduction(max:x)
  do m = 0, no - 1
    do n = 0, no - 1
      do i = 0, no - 1
        do j = 0, no - 1
          if (.not. xzeq(ocls(m) + ocls(n), ocls(i) + ocls(j))) x(1) = max(x(1), abs(oooo(j, i, n, m)))
        end do
        do e = 0, nv - 1
          if (.not. xzeq(ocls(m) + ocls(n), ocls(i) + vcls(e))) x(2) = max(x(2), abs(ooov(e, i, n, m)))
        end do
      end do
      do e = 0, nv - 1
        do f = 0, nv - 1
          if (.not. xzeq(ocls(m) + ocls(n), vcls(e) + vcls(f))) x(3) = max(x(3), abs(oovv(f, e, n, m)))
        end do
      end do
    end do
    do b = 0, nv - 1
      do e = 0, nv - 1
        do j = 0, no - 1
          if (.not. xzeq(ocls(m) + vcls(b), vcls(e) + ocls(j))) x(4) = max(x(4), abs(ovvo(j, e, b, m)))
        end do
      end do
    end do
    do y = 0, nv - 1
      g = xzw(ocls(m) + vcls(y))
      do q = 0, vpoff(g) - 1
        x(5) = max(x(5), abs(ovvq(q, y, m)))
      end do
      do q = vpoff(g + 1), nvp - 1
        x(5) = max(x(5), abs(ovvq(q, y, m)))
      end do
    end do
  end do
  vmax = maxval(x)
  what = '<oo||oo>'
  if (x(2) == vmax) what = '<oo||ov>'
  if (x(3) == vmax) what = '<oo||vv>'
  if (x(4) == vmax) what = '<ov||vo>'
  if (x(5) == vmax) what = '<ov||vv>'
end subroutine xzchklab

! ---------------------------------------------------------------------------
! The largest imaginary part of n complex numbers.
real(8) function xzimag(n, a)
  implicit none
  integer(8), intent(in) :: n
  real(8), intent(in) :: a(2, n)
  integer(8) :: i
  real(8) :: x
  x = 0d0
  !$omp parallel do reduction(max:x)
  do i = 1, n
    x = max(x, abs(a(2, i)))
  end do
  xzimag = x
end function xzimag

! The real parts of the n complex numbers at src to dst, which may be a
! lower place that overlaps it.
subroutine xztoreal(n, src, dst)
  implicit none
  integer(8), intent(in) :: n
  real(8) :: src(*), dst(*)
  integer(8) :: i
  !dir$ novector
  do i = 1, n
    dst(i) = src(2 * i - 1)
  end do
end subroutine xztoreal

! ---------------------------------------------------------------------------
! The blocked real files of xx2cvtran (%vtranblk): opens fname on unit 71,
! reads and checks the header, returns the labels of the file and their
! modulus; the elements follow on the unit.
subroutine xzbhead(fname, kind, flab, fmod)
  use xzdata
  implicit none
  character(len=*), intent(in) :: fname
  integer(8), intent(in) :: kind
  integer(8), intent(out) :: flab(0:ntot-1), fmod(4)
  integer(8) :: hdr(10)
  integer :: ios
  open(unit=71, file=fname, form='unformatted', access='sequential', status='old', &
       action='read', iostat=ios)
  if (ios /= 0) then
    write(6, '(/,a,a,a)') ' @XZCCSDT-F, no file ', trim(fname), ' (xx2cvtran with %vtranblk)'
    call xzerrex
  end if
  ! version 2: ten numbers, the second label in hdr(9:10); version 1: eight
  hdr = 0
  read(71, iostat=ios) hdr
  if (ios /= 0) then
    rewind(71)
    hdr = 0
    read(71, iostat=ios) hdr(1:8)
  end if
  if (ios /= 0 .or. hdr(1) /= 20261015_8 .or. hdr(2) < 1 .or. hdr(2) > 3 .or. hdr(3) /= kind) then
    write(6, '(/,a,a,a)') ' @XZCCSDT-F, the file ', trim(fname), ' is not a blocked file of xx2cvtran of its kind.'
    call xzerrex
  end if
  if (hdr(4) /= ntot .or. hdr(6) /= nfrozen .or. hdr(7) /= no .or. hdr(8) /= nv) then
    write(6, '(/,a,a,a)') ' @XZCCSDT-F, the file ', trim(fname), ' is not one of this run (sizes).'
    call xzerrex
  end if
  read(71, iostat=ios) flab
  if (ios /= 0) then
    write(6, '(/,a,a)') ' @XZCCSDT-F, reading the labels of the file ', trim(fname)
    call xzerrex
  end if
  ! the modulus, and with a second label its modulus and the base of its
  ! digit in the labels of the file (l1 + base l2)
  ! and the words of an element (version 3: complex)
  fmod = [hdr(5), hdr(9), hdr(10), merge(2_8, 1_8, hdr(2) == 3)]
end subroutine xzbhead

! ---------------------------------------------------------------------------
! An element of a blocked file of xx2cvtran whose labels add up to s (the
! labels of its bra minus those of its ket, as the file has them) is
! allowed: s = 0, modulo the modulus fmod(1), and with a second label (base
! fmod(3)) its digit modulo fmod(2).
logical function xzbok(s, fmod)
  implicit none
  integer(8), intent(in) :: s, fmod(4)
  integer(8) :: x2
  if (fmod(3) > 0) then
    x2 = nint(dble(s) / dble(fmod(3)), 8)
    xzbok = modulo(s - fmod(3) * x2, fmod(1)) == 0 .and. modulo(x2, fmod(2)) == 0
  else if (fmod(1) > 0) then
    xzbok = modulo(s, fmod(1)) == 0
  else
    xzbok = s == 0
  end if
end function xzbok

! ---------------------------------------------------------------------------
! The strict pairs a<b of n indices with the labels vl (the pair number
! f = a + b (b - 1) / 2, the order of the blocked files) by the classes of
! the sums vl(a) + vl(b) under xzbok: fid(f) the class of the pair, the
! pairs of class k rows(roff(k) .. roff(k + 1) - 1) in the order of f,
! rep(k) the sum of one of them.  A blocked file has the rows (ab) of a
! column whose labels add up to t, in the order of f: those of the class k
! with xzbok(rep(k) - t) (xzpclass), or none.  The sums take few values:
! this replaces a test per element by one per column.  walk = .false.: the
! test is not an equivalence on these sums (not a file of xx2cvtran), and
! the callers test every element.
subroutine xzpcls(n, vl, fmod, fid, rows, roff, rep, ncl, walk)
  implicit none
  integer(8), intent(in) :: n, vl(0:n-1), fmod(4)
  integer(8), intent(out) :: fid(0:*), rows(0:*), roff(0:*), rep(0:*), ncl
  logical, intent(out) :: walk
  logical, external :: xzbok
  integer(8) :: a, b, f, i, j, s, ns, np
  integer(8), allocatable :: ssum(:), sid(:)
  np = n * (n - 1) / 2
  allocate(ssum(0:max(np, 1_8) - 1), sid(0:max(np, 1_8) - 1))
  ! the distinct sums, then their classes
  ns = 0
  do b = 1, n - 1
    do a = 0, b - 1
      s = vl(a) + vl(b)
      do j = 0, ns - 1
        if (ssum(j) == s) exit
      end do
      if (j == ns) then
        ssum(ns) = s
        ns = ns + 1
      end if
      fid(a + b * (b - 1) / 2) = j
    end do
  end do
  ncl = 0
  do j = 0, ns - 1
    sid(j) = -1
    do i = 0, j - 1
      if (xzbok(ssum(j) - ssum(i), fmod)) then
        sid(j) = sid(i)
        exit
      end if
    end do
    if (sid(j) < 0) then
      sid(j) = ncl
      rep(ncl) = ssum(j)
      ncl = ncl + 1
    end if
  end do
  walk = .true.
  do j = 0, ns - 1
    do i = 0, ns - 1
      if (xzbok(ssum(j) - ssum(i), fmod) .neqv. (sid(i) == sid(j))) walk = .false.
    end do
  end do
  roff(0:ncl) = 0
  do f = 0, np - 1
    fid(f) = sid(fid(f))
    roff(fid(f) + 1) = roff(fid(f) + 1) + 1
  end do
  do j = 1, ncl
    roff(j) = roff(j) + roff(j - 1)
  end do
  sid(0:ncl - 1) = roff(0:ncl - 1)
  do f = 0, np - 1
    rows(sid(fid(f))) = f
    sid(fid(f)) = sid(fid(f)) + 1
  end do
  deallocate(ssum, sid)
end subroutine xzpcls

! the class of xzpcls that a column whose labels add up to t takes (-1: none)
integer(8) function xzpclass(t, rep, ncl, fmod)
  implicit none
  integer(8), intent(in) :: t, ncl, rep(0:*), fmod(4)
  logical, external :: xzbok
  integer(8) :: k
  xzpclass = -1
  do k = 0, ncl - 1
    if (xzbok(rep(k) - t, fmod)) then
      xzpclass = k
      return
    end if
  end do
end function xzpclass

! ---------------------------------------------------------------------------
! The places in the dense file of the elements of a blocked file, in the
! order of the blocked file (xzbwrite of xx2cvtran): kind 1 to 5 as in the
! header, iocc the occupied spinor of a file PPBK (kind 5, from 0).  With
! fill = .false. only their number cnt.
subroutine xzbidx(kind, iocc, flab, fmod, pos, cnt, fill)
  use xzdata
  implicit none
  integer(8), intent(in) :: kind, iocc, flab(0:ntot-1), fmod(4)
  integer(8) :: pos(*)
  integer(8), intent(out) :: cnt
  logical, intent(in) :: fill
  logical, external :: xzbok
  integer(8) :: i, j, k, l, a, b, c, po, pv, ol(0:no-1), vl(0:nv-1)
  po = no * (no + 1) / 2
  pv = nv * (nv + 1) / 2
  ol = flab(nfrozen:nfrozen + no - 1)
  vl = flab(nfrozen + no:nfrozen + no + nv - 1)
  cnt = 0
  select case (kind)
  case (1)
    do l = 1, no - 1
      do k = 0, l - 1
        do j = 1, no - 1
          do i = 0, j - 1
            if (xzbok(ol(i) + ol(j) - ol(k) - ol(l), fmod)) call put(i + j * (j + 1) / 2 + po * (k + l * (l + 1) / 2))
          end do
        end do
      end do
    end do
  case (2)
    do a = 0, nv - 1
      do k = 0, no - 1
        do j = 1, no - 1
          do i = 0, j - 1
            if (xzbok(ol(i) + ol(j) - ol(k) - vl(a), fmod)) call put(i + j * (j + 1) / 2 + po * (k + no * a))
          end do
        end do
      end do
    end do
  case (3)
    do j = 0, no - 1
      do b = 0, nv - 1
        do i = 0, no - 1
          do a = 0, nv - 1
            if (xzbok(ol(i) + ol(j) - vl(a) - vl(b), fmod)) call put(a + nv * (i + no * (b + nv * j)))
          end do
        end do
      end do
    end do
  case (4)
    do j = 0, no - 1
      do a = 0, nv - 1
        do i = 0, no - 1
          do b = 0, nv - 1
            if (xzbok(ol(i) + vl(a) - vl(b) - ol(j), fmod)) call put(b + nv * (i + no * (a + nv * j)))
          end do
        end do
      end do
    end do
  case (5)
    do c = 0, nv - 1
      do b = 1, nv - 1
        do a = 0, b - 1
          if (xzbok(vl(a) + vl(b) - vl(c) - ol(iocc), fmod)) call put(a + b * (b + 1) / 2 + pv * c)
        end do
      end do
    end do
  end select
contains
  subroutine put(p)
    integer(8), intent(in) :: p
    cnt = cnt + 1
    if (fill) pos(cnt) = p
  end subroutine put
end subroutine xzbidx

! ---------------------------------------------------------------------------
! An integral file of the small blocks or a file PPPH as a dense record of
! complex numbers (n double words, as xzrfile reads it): the file itself,
! or with the blocked files (blkin) the blocked file of the same integrals
! (moinxxxx -> mobkxxxx, PPPHnnnn -> PPBKnnnn), its elements put in their
! places and zero elsewhere.
subroutine xzrint(kind, fname, buf, n)
  use xzdata
  implicit none
  integer(8), intent(in) :: kind, n
  character(len=*), intent(in) :: fname
  complex(8), intent(out) :: buf(0:n/2-1)
  character(len=8) :: bname
  integer(8) :: flab(0:ntot-1), fmod(4), cnt, k, iocc, pos0(1)
  integer(8), allocatable :: pos(:)
  real(8), allocatable :: d(:)
  integer :: ios
  if (.not. blkin) then
    call xzrfile(fname, buf, n)
    return
  end if
  iocc = 0
  if (kind == 5) then
    bname = 'PPBK' // fname(5:8)
    read(fname(5:8), *) iocc
    iocc = iocc - 1
  else
    bname = 'mobk' // fname(5:8)
  end if
  call xzbhead(bname, kind, flab, fmod)
  call xzbidx(kind, iocc, flab, fmod, pos0, cnt, .false.)
  allocate(pos(max(cnt, 1_8)), d(max(fmod(4) * cnt, 1_8)))
  call xzbidx(kind, iocc, flab, fmod, pos, cnt, .true.)
  read(71, iostat=ios) d(1:fmod(4) * cnt)
  close(71)
  if (ios /= 0) then
    write(6, '(/,a,a)') ' @XZCCSDT-F, reading the elements of the file ', trim(bname)
    call xzerrex
  end if
  !$omp parallel do
  do k = 0, n / 2 - 1
    buf(k) = (0d0, 0d0)
  end do
  if (fmod(4) == 2) then
    do k = 1, cnt
      buf(pos(k)) = cmplx(d(2 * k - 1), d(2 * k), 8)
    end do
  else
    do k = 1, cnt
      buf(pos(k)) = cmplx(d(k), 0d0, 8)
    end do
  end if
  deallocate(pos, d)
end subroutine xzrint

! ---------------------------------------------------------------------------
! <ab||cd> for a<b, c<d from the blocked file mobkpppp, straight into the
! blocks [Q][Q'] of the groups of pairs (the layout of xzrdvvvv at its end),
! nwe words an element; nothing of the size of the dense record.  Its
! columns c<d one record each, the rows a<b that the labels of the file
! allow.  offmax: the largest element outside of the blocks of the groups
! of this program; ximag: in real arithmetic, the largest imaginary part
! (complex files, version 3).  inv, ig: nvp integers, sgf: nvp reals,
! col: 2 nvp reals.
subroutine xzrbvvvv(v, vpa, vpb, inv, ig, sgf, col, offmax, ximag)
  use xzdata
  implicit none
  real(8) :: v(0:*)
  integer(8), intent(in) :: vpa(0:nvp-1), vpb(0:nvp-1)
  integer(8) :: inv(0:nvp-1), ig(0:nvp-1)
  real(8) :: sgf(0:nvp-1), col(0:2*nvp-1)
  real(8), intent(out) :: offmax, ximag
  integer(8) :: flab(0:ntot-1), fmod(4), vl(0:nv-1), q, a, b, c, d, f, g, qc, n, lo, k, cnt, fw
  integer(8) :: bb(0:maxgrp), ncl, kk
  integer(8), allocatable :: fid(:), rows(:), roff(:), rep(:)
  logical, external :: xzbok
  logical :: walk
  integer :: ios

  call xzbhead('mobkpppp', 6_8, flab, fmod)
  vl = flab(nfrozen + no:nfrozen + no + nv - 1)
  ! the pair of the program and its sign for each strict pair of the file
  do q = 0, nvp - 1
    a = vperm(vpa(q))
    b = vperm(vpb(q))
    f = min(a, b) + max(a, b) * (max(a, b) - 1) / 2
    inv(f) = q
    sgf(f) = merge(1d0, -1d0, a < b)
  end do
  bb(0) = 0
  do g = 0, npg - 1
    do q = vpoff(g), vpoff(g + 1) - 1
      ig(q) = g
    end do
    bb(g + 1) = bb(g) + (vpoff(g + 1) - vpoff(g)) ** 2
  end do
  ! the rows of the columns by the classes of the sums of their labels (xzpcls)
  allocate(fid(0:nvp-1), rows(0:nvp-1), roff(0:nvp), rep(0:nvp-1))
  call xzpcls(nv, vl, fmod, fid, rows, roff, rep, ncl, walk)
  v(0:nwe * nvv - 1) = 0d0
  offmax = 0d0
  ximag = 0d0
  fw = fmod(4)
  do d = 1, nv - 1
    do c = 0, d - 1
      f = c + d * (d - 1) / 2
      if (walk) then
        cnt = roff(fid(f) + 1) - roff(fid(f))
      else
        cnt = 0
        do b = 1, nv - 1
          do a = 0, b - 1
            if (xzbok(vl(a) + vl(b) - vl(c) - vl(d), fmod)) cnt = cnt + 1
          end do
        end do
      end if
      read(71, iostat=ios) col(0:fw * cnt - 1)
      if (ios /= 0) then
        write(6, '(/,a)') ' @XZCCSDT-F, reading the file mobkpppp failed.'
        call xzerrex
      end if
      qc = inv(f)
      g = ig(qc)
      lo = vpoff(g)
      n = vpoff(g + 1) - lo
      if (walk) then
        do kk = roff(fid(f)), roff(fid(f) + 1) - 1
          call put(rows(kk), kk - roff(fid(f)))
        end do
      else
        k = 0
        do b = 1, nv - 1
          do a = 0, b - 1
            if (.not. xzbok(vl(a) + vl(b) - vl(c) - vl(d), fmod)) cycle
            call put(a + b * (b - 1) / 2, k)
            k = k + 1
          end do
        end do
      end if
    end do
  end do
  close(71)
  deallocate(fid, rows, roff, rep)
contains
  ! the k-th element of the column (cd) = f, the row (ab) = q of the file
  subroutine put(q, k)
    integer(8), intent(in) :: q, k
    integer(8) :: qr, idx
    complex(8) :: x
    ! (the element of row (ab), column (cd) is <ab||cd> = <cd||ab>*; the
    ! place of (cd) here is the row: conjugate, as xzrdvvvv does)
    if (fw == 2) then
      x = sgf(q) * sgf(f) * conjg(cmplx(col(2 * k), col(2 * k + 1), 8))
    else
      x = sgf(q) * sgf(f) * col(k)
    end if
    qr = inv(q)
    if (ig(qr) /= g) then
      offmax = max(offmax, abs(x))
      return
    end if
    idx = bb(g) + (qr - lo) + n * (qc - lo)
    if (nwe == 2) then
      v(2 * idx) = dble(x)
      v(2 * idx + 1) = aimag(x)
    else
      v(idx) = dble(x)
      ximag = max(ximag, abs(aimag(x)))
    end if
  end subroutine put
end subroutine xzrbvvvv
