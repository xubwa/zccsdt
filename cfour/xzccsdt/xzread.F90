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
! the classes of their symmetry labels, xzdata.F90): operm[i] and vperm[a]
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
    if (ios /= 0 .or. hdr(1) /= 20261015_8 .or. hdr(2) /= 1) then
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
! that is not 0) - 2 m_j of a linear molecule (no modulus).
!   lab(0:nlab-1); there = .false.: no file
subroutine xzrdlab(nlab, lab, modul, there)
  implicit none
  integer(8), intent(in) :: nlab
  integer(8), intent(out) :: lab(0:nlab-1), modul
  logical, intent(out) :: there
  character(len=1024) :: line
  integer(8) :: n, k, v(64), have
  integer :: ios, i, nw, c1
  logical :: first
  modul = 0
  inquire(file='xzlabels', exist=there)
  if (.not. there) return
  open(unit=71, file='xzlabels', form='formatted', status='old', action='read')
  first = .true.
  have = 0
  n = -1
  do
    read(71, '(a)', iostat=ios) line
    if (ios /= 0) exit
    c1 = verify(line, ' ')
    if (c1 == 0) cycle
    if (line(c1:c1) == '#') cycle
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
      if (have < nlab) lab(have) = v(i)
      have = have + 1
    end do
  end do
  close(71)
  if (n /= nlab .or. have /= nlab) then
    write(6, '(/,a,i6,a,i6,a,i6,a)') ' @XZCCSDT-F, the file xzlabels: ', n, ' spinors announced, ', have, &
      ' labels found; the SCF has ', nlab, ' spinors.'
    call xzerrex
  end if
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
  integer(8) :: m, n, i, j, e, f, b, y, q
  real(8) :: x(5)
  x = 0d0
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
      do q = 0, nvp - 1
        if (.not. xzeq(ocls(m) + vcls(y), vpgrp(q))) x(5) = max(x(5), abs(ovvq(q, y, m)))
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
  integer(8), intent(out) :: flab(0:ntot-1), fmod
  integer(8) :: hdr(8)
  integer :: ios
  open(unit=71, file=fname, form='unformatted', access='sequential', status='old', &
       action='read', iostat=ios)
  if (ios /= 0) then
    write(6, '(/,a,a,a)') ' @XZCCSDT-F, no file ', trim(fname), ' (xx2cvtran with %vtranblk)'
    call xzerrex
  end if
  read(71, iostat=ios) hdr
  if (ios /= 0 .or. hdr(1) /= 20261015_8 .or. hdr(2) /= 1 .or. hdr(3) /= kind) then
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
  fmod = hdr(5)
end subroutine xzbhead

! ---------------------------------------------------------------------------
! The places in the dense file of the elements of a blocked file, in the
! order of the blocked file (xzbwrite of xx2cvtran): kind 1 to 5 as in the
! header, iocc the occupied spinor of a file PPBK (kind 5, from 0).  With
! fill = .false. only their number cnt.
subroutine xzbidx(kind, iocc, flab, fmod, pos, cnt, fill)
  use xzdata
  implicit none
  integer(8), intent(in) :: kind, iocc, flab(0:ntot-1), fmod
  integer(8) :: pos(*)
  integer(8), intent(out) :: cnt
  logical, intent(in) :: fill
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
            if (ok(ol(i) + ol(j) - ol(k) - ol(l))) call put(i + j * (j + 1) / 2 + po * (k + l * (l + 1) / 2))
          end do
        end do
      end do
    end do
  case (2)
    do a = 0, nv - 1
      do k = 0, no - 1
        do j = 1, no - 1
          do i = 0, j - 1
            if (ok(ol(i) + ol(j) - ol(k) - vl(a))) call put(i + j * (j + 1) / 2 + po * (k + no * a))
          end do
        end do
      end do
    end do
  case (3)
    do j = 0, no - 1
      do b = 0, nv - 1
        do i = 0, no - 1
          do a = 0, nv - 1
            if (ok(ol(i) + ol(j) - vl(a) - vl(b))) call put(a + nv * (i + no * (b + nv * j)))
          end do
        end do
      end do
    end do
  case (4)
    do j = 0, no - 1
      do a = 0, nv - 1
        do i = 0, no - 1
          do b = 0, nv - 1
            if (ok(ol(i) + vl(a) - vl(b) - ol(j))) call put(b + nv * (i + no * (a + nv * j)))
          end do
        end do
      end do
    end do
  case (5)
    do c = 0, nv - 1
      do b = 1, nv - 1
        do a = 0, b - 1
          if (ok(vl(a) + vl(b) - vl(c) - ol(iocc))) call put(a + b * (b + 1) / 2 + pv * c)
        end do
      end do
    end do
  end select
contains
  logical function ok(s)
    integer(8), intent(in) :: s
    if (fmod > 0) then
      ok = modulo(s, fmod) == 0
    else
      ok = s == 0
    end if
  end function ok
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
  integer(8) :: flab(0:ntot-1), fmod, cnt, k, iocc, pos0(1)
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
  allocate(pos(max(cnt, 1_8)), d(max(cnt, 1_8)))
  call xzbidx(kind, iocc, flab, fmod, pos, cnt, .true.)
  read(71, iostat=ios) d(1:cnt)
  close(71)
  if (ios /= 0) then
    write(6, '(/,a,a)') ' @XZCCSDT-F, reading the elements of the file ', trim(bname)
    call xzerrex
  end if
  buf = (0d0, 0d0)
  do k = 1, cnt
    buf(pos(k)) = cmplx(d(k), 0d0, 8)
  end do
  deallocate(pos, d)
end subroutine xzrint

! ---------------------------------------------------------------------------
! <ab||cd> for a<b, c<d from the blocked file mobkpppp, straight into the
! blocks [Q][Q'] of the groups of pairs (the layout of xzrdvvvv at its end),
! nwe words an element; nothing of the size of the dense record.  Its
! columns c<d one record each, the rows a<b that the labels of the file
! allow.  offmax: the largest element outside of the blocks of the groups
! of this program.  inv, ig: nvp integers, sgf, col: nvp reals.
subroutine xzrbvvvv(v, vpa, vpb, inv, ig, sgf, col, offmax)
  use xzdata
  implicit none
  real(8) :: v(0:*)
  integer(8), intent(in) :: vpa(0:nvp-1), vpb(0:nvp-1)
  integer(8) :: inv(0:nvp-1), ig(0:nvp-1)
  real(8) :: sgf(0:nvp-1), col(0:nvp-1)
  real(8), intent(out) :: offmax
  integer(8) :: flab(0:ntot-1), fmod, vl(0:nv-1), q, a, b, c, d, f, g, qr, qc, n, lo, k, cnt, idx
  integer(8) :: bb(0:maxgrp)
  real(8) :: x
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
  v(0:nwe * nvv - 1) = 0d0
  offmax = 0d0
  do d = 1, nv - 1
    do c = 0, d - 1
      cnt = 0
      do b = 1, nv - 1
        do a = 0, b - 1
          if (ok(vl(a) + vl(b) - vl(c) - vl(d))) cnt = cnt + 1
        end do
      end do
      read(71, iostat=ios) col(0:cnt - 1)
      if (ios /= 0) then
        write(6, '(/,a)') ' @XZCCSDT-F, reading the file mobkpppp failed.'
        call xzerrex
      end if
      f = c + d * (d - 1) / 2
      qc = inv(f)
      g = ig(qc)
      lo = vpoff(g)
      n = vpoff(g + 1) - lo
      k = 0
      do b = 1, nv - 1
        do a = 0, b - 1
          if (.not. ok(vl(a) + vl(b) - vl(c) - vl(d))) cycle
          q = a + b * (b - 1) / 2
          x = sgf(q) * sgf(f) * col(k)
          k = k + 1
          qr = inv(q)
          if (ig(qr) /= g) then
            offmax = max(offmax, abs(x))
            cycle
          end if
          idx = bb(g) + (qr - lo) + n * (qc - lo)
          v(nwe * idx) = x
        end do
      end do
    end do
  end do
  close(71)
contains
  logical function ok(s)
    integer(8), intent(in) :: s
    if (fmod > 0) then
      ok = modulo(s, fmod) == 0
    else
      ok = s == 0
    end if
  end function ok
end subroutine xzrbvvvv
