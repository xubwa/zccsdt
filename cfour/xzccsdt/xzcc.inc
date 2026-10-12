!
! The CCSDT iterations (type-generic): second-order start, Jacobi steps
! (xzstep.inc) with DIIS (xzdiis.F90), convergence as in the Python program:
! the energy change below conv_e and |dT1| + |dT2| + |dT3| below conv_t.
! The amplitudes stay in the work array (jt1, jt2, jt3 of xzdata) for what
! follows; everything else that is taken here is released.
!
! With more than one rank (xzdist.F90) rank 0 runs this loop and sends, at
! the start of every iteration, a word that says "one more" and the
! amplitudes; the other ranks (N(xzccwork)) take them into their own T1, T2,
! T3 and do their part of the step.  The DIIS vectors are those of rank 0.
!
subroutine N(xzcc)(w, nslot, usefile, nch, ecorr, converged, ncyc)
  use xzdata
  use xzmem
  use xzkern
  use xzdist
  use xzdiisd, only: ncnt
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: nslot, nch
  logical, intent(in) :: usefile
  real(8), intent(out) :: ecorr
  logical, intent(out) :: converged
  integer(8), intent(out) :: ncyc
  integer(8) :: kt1n, kt2n, kr3, icyc, m0, n, kb1, kb2
  logical :: fromccsd
  integer(8), external :: N(xzdiis_target)
  real(8) :: norms(3), eold, enew, de, normt, t0, t1w, ctl(2), tsg
  ZT :: e
  character(len=256) :: dbg
  integer :: ldbg

  jt1 = xzget(NW * no * nv, 'T1')
  jt2 = xzget(NW * no * no * nv * nv, 'T2')
  jt3 = xzget(NW * n3, 'T3')
  m0 = xzmark()
  kt1n = xzget(NW * no * nv, 'new T1')
  kt2n = xzget(NW * no * no * nv * nv, 'new T2')
  if (xzworker()) then
    ! a rank other than 0: no DIIS vectors, one region for its part of the
    ! residual (with slices: its slices of the DIIS vectors, as on rank 0)
    if (slices) then
      call N(xzdiis_alloc)(nslot, usefile, nch)
    else
      call N(xzdiis_alloc)(0_8, .false., nch)
    end if
    if (.not. xzdry) then
      call N(xzccwork)(w, kt1n, kt2n, ncyc)
      call N(xzdiis_close)()
      call xzrel(m0)
      ecorr = 0d0
      converged = .true.
      return
    end if
  else
    call N(xzdiis_alloc)(nslot, usefile, nch)
  end if
  ! option start ccsd: T1 and T2 of CFOUR's xx2cccsd as the start (its files
  ! t1guess, t2guess of the run directory)
  call xzoptget('start', dbg, ldbg)
  fromccsd = ldbg > 0 .and. (dbg(1:1) == 'c' .or. dbg(1:1) == 'C')
  if (fromccsd) then
    n = xzmark()
    kb1 = xzget(2 * no * nv, 't1guess')
    kb2 = xzget(2 * (no * (no + 1) / 2) * (nv * (nv + 1) / 2), 't2guess')
    if (.not. xzdry) call N(xzguess)(no, nv, w(jeps), w(joovv), w(jt1), w(jt2))
    if (.not. xzdry) call N(xzccsdstart)(no, nv, w(kb1), w(kb2), w(jt1), w(jt2))
    call xzrel(n)
  end if
  if (xzdry) then
    call N(xzstepm)(w, jt1, jt2, jt3, kt1n, kt2n, jt3)
    call xzrel(m0)
    return
  end if

  if (.not. fromccsd) call N(xzguess)(no, nv, w(jeps), w(joovv), w(jt1), w(jt2))
  call zc_vzero_d(NW * n3, w(jt3))
  call N(xzecorr)(no, nv, w(jt1), w(jt2), w(joovv), e)
  eold = dble(e)
  call xzoptget('debug', dbg, ldbg)
  if (ldbg > 0 .and. nrank > 1) then
    write(6, '(/,a)') ' @XZCCSDT-F, option debug (one step from a file) runs with one rank only.'
    call xzerrex
  end if
  if (ldbg > 0) then
    ! one step on amplitudes read from a file, the result written next to it
    n = NW * (no * nv + no * no * nv * nv + n3)
    open(unit=74, file=dbg(1:ldbg), form='unformatted', access='stream', status='old')
    read(74) w(jt1:jt1 + NW * no * nv - 1), w(jt2:jt2 + NW * no * no * nv * nv - 1), &
             w(jt3:jt3 + NW * n3 - 1)
    close(74)
    kr3 = N(xzdiis_target)()
    call N(xzstepm)(w, jt1, jt2, jt3, kt1n, kt2n, kr3)
    open(unit=74, file=dbg(1:ldbg) // '.out', form='unformatted', access='stream', status='replace')
    write(74) w(kt1n:kt1n + NW * no * nv - 1), w(kt2n:kt2n + NW * no * no * nv * nv - 1), &
              w(kr3:kr3 + NW * n3 - 1)
    close(74)
    write(6, '(a,a)') ' option debug: one step written to ', dbg(1:ldbg) // '.out'
    call N(xzdiis_close)()
    call xzrel(m0)
    ecorr = eold
    converged = .false.
    ncyc = 0
    return
  end if

  if (fromccsd) then
    write(6, '(/,a,f20.12)') '  Correlation energy of the start (T1, T2 of t1guess, t2guess):', eold
  else
    write(6, '(/,a,f20.12)') '  Second-order correlation energy:', eold
  end if
  write(6, '(/,a,a,a)') '  ', trim(mname), trim(merge(' iterations (T3 stored)', ' iterations            ', cc3mode == 1))
  write(6, '(a)') '  ' // repeat('-', 98)
  write(6, '(a)') '   cycle     correlation energy      energy change      |dT1|      |dT2|' // &
                  '      |dT3|   DIIS    time/s'
  write(6, '(a)') '  ' // repeat('-', 98)
  converged = .false.
  ncyc = maxcyc
  ! (the other ranks read the integrals after rank 0: wait for them here, so
  ! that the clock of the first broadcast is not that of their reading)
  if (nrank > 1) then
    call xztic()
    call xzmpi_barrier()
    call xztoc(1, 'MPI: ranks reading integrals')
  end if
  do icyc = 1, maxcyc
    t0 = omp_get_wtime()
    if (nrank > 1) then
      ! one more iteration: the amplitudes to the ranks (T3 of the first one is zero)
      call xztic()
      tsg = omp_get_wtime()
      ctl = [1d0, merge(0d0, 1d0, icyc == 1)]
      call xzmpi_bcast(ctl, 16_8)
      call xzmpi_bcast_w(w(jt1), NW * no * nv)
      call xzmpi_bcast_w(w(jt2), NW * no * no * nv * nv)
      if (icyc > 1 .and. .not. slices) call xzmpi_bcast_w(w(jt3), NW * n3)
      call xztoc(1, 'MPI: amplitudes to the ranks')
      dsec(6, 1) = dsec(6, 1) + omp_get_wtime() - tsg
    end if
    kr3 = N(xzdiis_target)()
    call N(xzstepm)(w, jt1, jt2, jt3, kt1n, kt2n, kr3)
    call xzrebal(1)
    tsg = omp_get_wtime()
    call xztic()
    call N(xzdiis_upd)(w, w(jt1), w(jt2), jt3, w(kt1n), w(kt2n), kr3, norms)
    call xztoc(1, 'DIIS')
    dsec(8, 1) = dsec(8, 1) + omp_get_wtime() - tsg
    if (slices) then
      ! the slices of the new T3 to every rank
      tsg = omp_get_wtime()
      call xzmpi_slices_put(w(jt3), swoff)
      call xztoc(1, 'MPI: T3 slices to the ranks')
      dsec(6, 1) = dsec(6, 1) + omp_get_wtime() - tsg
    end if
    call xztic()
    call N(xzecorr)(no, nv, w(jt1), w(jt2), w(joovv), e)
    call xztoc(1, 'energy')
    enew = dble(e)
    de = enew - eold
    eold = enew
    normt = norms(1) + norms(2) + norms(3)
    write(6, '(i8,f23.14,es19.6,3es11.3,i7,f10.2)') icyc, enew, de, norms, ncnt, omp_get_wtime() - t0
    flush(6)
    if (abs(de) < conv_e .and. normt < conv_t) then
      converged = .true.
      ncyc = icyc
      exit
    end if
  end do
  write(6, '(a)') '  ' // repeat('-', 98)
  if (nrank > 1) then
    ! the iterations are over
    ctl = 0d0
    call xzmpi_bcast(ctl, 16_8)
  end if
  ecorr = eold
  call N(xzdiis_close)()
  call xzrel(m0)
end subroutine

! ---------------------------------------------------------------------------
! The iterations of a rank other than 0: as long as rank 0 says "one more",
! its amplitudes into T1, T2, T3 of this rank and the part of this rank of
! the step (xzstep leaves out what only rank 0 does).
subroutine N(xzccwork)(w, kt1n, kt2n, icyc)
  use xzdata
  use xzmem
  use xzkern
  use xzdist
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kt1n, kt2n
  integer(8), intent(out) :: icyc
  integer(8), external :: N(xzdiis_target)
  integer(8) :: kr3
  real(8) :: ctl(2), tsg, t0, norms(3)
  icyc = 0
  write(6, '(/,a,a,a)') '  ', trim(mname), ' iterations: the batches of this rank'
  if (slices) call zc_vzero_d(NW * n3, w(jt3))
  call xzmpi_barrier()
  do
    call xzmpi_bcast(ctl, 16_8)
    if (ctl(1) /= 1d0) exit
    t0 = omp_get_wtime()
    tsg = t0
    call xztic()
    call xzmpi_bcast_w(w(jt1), NW * no * nv)
    call xzmpi_bcast_w(w(jt2), NW * no * no * nv * nv)
    if (slices) then
      ! (T3 is here: the slices of the ranks came at the end of the iteration before)
    else if (ctl(2) == 1d0) then
      call xzmpi_bcast_w(w(jt3), NW * n3)
      dwords(1) = dwords(1) + dble(NW * n3)
    else
      call zc_vzero_d(NW * n3, w(jt3))
    end if
    call xztoc(1, 'MPI: amplitudes from rank 0')
    dsec(6, 1) = dsec(6, 1) + omp_get_wtime() - tsg
    kr3 = N(xzdiis_target)()
    call N(xzstepm)(w, jt1, jt2, jt3, kt1n, kt2n, kr3)
    call xzrebal(1)
    if (slices) then
      ! DIIS on the slice of this rank, then the slices of the new T3 of all ranks
      tsg = omp_get_wtime()
      call xztic()
      call N(xzdiis_upd)(w, w(jt1), w(jt2), jt3, w(kt1n), w(kt2n), kr3, norms)
      call xztoc(1, 'DIIS')
      dsec(8, 1) = dsec(8, 1) + omp_get_wtime() - tsg
      tsg = omp_get_wtime()
      call xzmpi_slices_put(w(jt3), swoff)
      call xztoc(1, 'MPI: T3 slices to the ranks')
      dsec(6, 1) = dsec(6, 1) + omp_get_wtime() - tsg
      dwords(1) = dwords(1) + dble(NW * (n3 - (soff(myrank + 1) - soff(myrank))))
    end if
    icyc = icyc + 1
    write(6, '(a,i5,f12.2,a)') '   cycle ', icyc, omp_get_wtime() - t0, ' s'
    flush(6)
  end do
end subroutine

! ---------------------------------------------------------------------------
! T1 and T2 of a CCSD calculation of CFOUR's xx2cccsd on the same integral
! files as the start: its files t1guess (t1(a, i), one record, complex) and
! t2guess (t2(ab, ij) on a <= b, i <= j, ab = a + b (b - 1) / 2 from 1; the
! diagonals zero), in CFOUR's order of the active spinors (operm, vperm give
! it for the order here).  Without the files, with files of other sizes or -
! in real arithmetic - with amplitudes that are not real, t1 and t2 are left
! as they are (the second-order start) and a warning says so.
subroutine N(xzccsdstart)(no, nv, b1, b2, t1, t2)
  use xzdata, only: operm, vperm
  implicit none
  integer(8), intent(in) :: no, nv
  complex(8) :: b1(0:nv-1, 0:no-1)
  complex(8) :: b2(0:nv*(nv+1)/2-1, 0:no*(no+1)/2-1)
  ZT, intent(inout) :: t1(0:nv-1, 0:no-1)
  ZT, intent(inout) :: t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)      ! [i][j][a][b]
  integer(8) :: i, j, a, b, fi, fj, fa, fb, pij, pab, n1, n2, s1, s2
  integer(8), external :: xzrecbytes
  real(8) :: sij, sab, big
  logical :: e1, e2
  n1 = 2 * no * nv
  n2 = 2 * (no * (no + 1) / 2) * (nv * (nv + 1) / 2)
  inquire(file='t1guess', exist=e1, size=s1)
  inquire(file='t2guess', exist=e2, size=s2)
  if (.not. (e1 .and. e2)) then
    write(6, '(/,a)') ' @XZCCSDT-W, option start ccsd: no files t1guess, t2guess (xx2cccsd writes them); second-order start.'
    return
  end if
  if (xzrecbytes(s1) /= 8 * n1 .or. xzrecbytes(s2) /= 8 * n2) then
    write(6, '(/,a)') ' @XZCCSDT-W, option start ccsd: t1guess, t2guess are not of this calculation (sizes); second-order start.'
    return
  end if
  call xzrfile('t1guess', b1, n1)
  call xzrfile('t2guess', b2, n2)
#ifdef XZ_REAL
  big = max(maxval(abs(aimag(b1))), maxval(abs(aimag(b2))))
  if (big > 1d-8) then
    write(6, '(/,a,es10.2,a)') ' @XZCCSDT-W, option start ccsd: the amplitudes of t1guess, t2guess are not real (', big, &
      '); second-order start.'
    return
  end if
#endif
  !$omp parallel do private(i, j, a, b, fi, fj, fa, fb, pij, pab, sij, sab)
  do i = 0, no - 1
    fi = operm(i)
    do a = 0, nv - 1
      t1(a, i) = b1(vperm(a), fi)
    end do
    do j = 0, no - 1
      fj = operm(j)
      pij = min(fi, fj) + max(fi, fj) * (max(fi, fj) + 1) / 2
      sij = merge(1d0, -1d0, fi < fj)
      if (i == j) sij = 0d0
      do a = 0, nv - 1
        fa = vperm(a)
        do b = 0, nv - 1
          fb = vperm(b)
          pab = min(fa, fb) + max(fa, fb) * (max(fa, fb) + 1) / 2
          sab = merge(1d0, -1d0, fa < fb)
          if (a == b) sab = 0d0
          t2(b, a, j, i) = (sij * sab) * b2(pab, pij)
        end do
      end do
    end do
  end do
  write(6, '(/,a)') '   start: T1 and T2 of CFOUR''s CCSD (files t1guess, t2guess)'
end subroutine

! ---------------------------------------------------------------------------
! The amplitudes in the file XZAMPS of the run directory, the hand-over from
! the iterations to the steps that follow ((Q), on every rank, or a later
! run): an unformatted stream,
!   character(8) 'XZAMPS01'
!   integer(8)   no, nv, words per element (1 real, 2 complex), classes of
!                the labels, elements of T3 as stored
!   real(8)      the CCSDT correlation energy
!   T1 [i][a], T2 [i][j][a][b], T3 as stored (packed; its blocks one after
!   the other), in the order of the spinors of the run that wrote it
! rw = 'w': written from the amplitudes in the work array; 'r': the regions
! are taken from the stack and filled (a dry run: only taken).
subroutine N(xzamps)(w, rw, ecorr)
  use xzdata
  use xzmem
  implicit none
  real(8) :: w(*)
  character(len=1), intent(in) :: rw
  real(8), intent(inout) :: ecorr
  character(len=8) :: tag
  integer(8) :: head(5), n1, n2
  integer :: ios
  logical :: there
  n1 = NW * no * nv
  n2 = NW * no * no * nv * nv
  if (rw == 'w') then
    open(unit=77, file='XZAMPS', form='unformatted', access='stream', status='replace', iostat=ios)
    if (ios == 0) write(77, iostat=ios) 'XZAMPS01', no, nv, int(NW, 8), ncls, n3, ecorr
    if (ios == 0) write(77, iostat=ios) w(jt1:jt1 + n1 - 1)
    if (ios == 0) write(77, iostat=ios) w(jt2:jt2 + n2 - 1)
    if (ios == 0) write(77, iostat=ios) w(jt3:jt3 + NW * n3 - 1)
    if (ios == 0) close(77, iostat=ios)
    if (ios /= 0) then
      write(6, '(/,a,i8,a)') ' @XZCCSDT-F, writing the file XZAMPS failed (iostat ', ios, '): disk full?'
      call xzerrex
    end if
    return
  end if
  jt1 = xzget(n1, 'T1')
  jt2 = xzget(n2, 'T2')
  if (.not. t3shared) jt3 = xzget(NW * n3, 'T3')
  if (xzdry) return
  inquire(file='XZAMPS', exist=there)
  if (.not. there) then
    write(6, '(/,a)') ' @XZCCSDT-F, no file XZAMPS: the amplitudes of a converged CCSDT run of xzccsdt.'
    call xzerrex
  end if
  open(unit=77, file='XZAMPS', form='unformatted', access='stream', status='old', action='read', iostat=ios)
  if (ios == 0) read(77, iostat=ios) tag, head, ecorr
  if (ios /= 0 .or. tag /= 'XZAMPS01') then
    write(6, '(/,a)') ' @XZCCSDT-F, the file XZAMPS is not a file of amplitudes of this program.'
    call xzerrex
  end if
  if (head(1) /= no .or. head(2) /= nv .or. head(3) /= NW .or. head(4) /= ncls .or. head(5) /= n3) then
    write(6, '(/,a)') ' @XZCCSDT-F, the file XZAMPS belongs to another calculation or another mode:'
    write(6, '(a,5i12)') '   occupied, virtual, words per element, classes, T3 in the file: ', head
    write(6, '(a,5i12)') '   and here:                                                      ', no, nv, &
      int(NW, 8), ncls, n3
    write(6, '(a)') '   (the labels and the arithmetic must be those of the run that wrote it: option labels, option real)'
    call xzerrex
  end if
  read(77, iostat=ios) w(jt1:jt1 + n1 - 1)
  if (ios == 0) read(77, iostat=ios) w(jt2:jt2 + n2 - 1)
  if (ios == 0 .and. .not. t3shared) read(77, iostat=ios) w(jt3:jt3 + NW * n3 - 1)
  close(77)
  if (ios /= 0) then
    write(6, '(/,a,i8)') ' @XZCCSDT-F, reading the file XZAMPS failed, iostat ', ios
    call xzerrex
  end if
  ! T3 where the file is mapped (stored there as the program uses it, in every mode)
  if (t3shared) call xzmapt3(w, 'XZAMPS', n1 + n2, NW * n3, jt3)
end subroutine

! ---------------------------------------------------------------------------
! The second-order energy from the integrals in the work array.
subroutine N(xzmp2)(w, e2)
  use xzdata
  use xzmem
  implicit none
  real(8) :: w(*)
  real(8), intent(out) :: e2
  integer(8) :: m0, k1, k2
  ZT :: e
  m0 = xzmark()
  k1 = xzget(NW * no * nv, 'T1')
  k2 = xzget(NW * no * no * nv * nv, 'T2')
  call N(xzguess)(no, nv, w(jeps), w(joovv), w(k1), w(k2))
  call N(xzecorr)(no, nv, w(k1), w(k2), w(joovv), e)
  e2 = dble(e)
  call xzrel(m0)
end subroutine
