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
  integer(8) :: kt1n, kt2n, kr3, icyc, m0, n, kb1, kb2, nsince
  logical :: fromccsd, t4now, fresh, conv, phase4, resumed
  real(8) :: fmicro, dmac, demac, t4s, dprev, kap, dsaved
  integer(8) :: kc4
  character(len=24) :: mark
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
  if (cc4) then
    ! CC4: the T4 terms of the doubles and triples equations (xzcc4.inc)
    jd42 = xzget(NW * npo * nvp, 'CC4: T4 term of R2')
    jd43 = xzget(NW * n3, 'CC4: T4 terms of R3')
  end if
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
    if (cc4) call N(xzt4)(w, jt1, jt2, jt3, jd42, jd43, 0d0)
    call xzrel(m0)
    return
  end if

  if (cc4) then
    call zc_vzero_d(NW * npo * nvp, w(jd42))
    call zc_vzero_d(NW * n3, w(jd43))
  end if
  dsaved = 0d0
  if (cc4rst > 0) then
    ! CC4 from the file XZCC4 of an earlier run: its amplitudes (cc4rst 1: and the T4 terms of its last evaluation)
    call N(xz4ckpt)(w, 'r', kc4, eold, dsaved)
    nt4old = nt4
  else if (cc4start) then
    ! CC4 from the CCSDT amplitudes of the file XZAMPS
    call N(xzamps)(w, 'l', e4ccsdt)
  else
    if (.not. fromccsd) call N(xzguess)(no, nv, w(jeps), w(joovv), w(jt1), w(jt2))
    call zc_vzero_d(NW * n3, w(jt3))
  end if
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

  if (cc4rst > 0) then
    write(6, '(/,a,f20.12)') '  Correlation energy of the start (the CC4 amplitudes of XZCC4):', eold
  else if (cc4start) then
    write(6, '(/,a,f20.12)') '  Correlation energy of the start (the CCSDT amplitudes of XZAMPS):', eold
  else if (fromccsd) then
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
  ! CC4 (docs/native.md, section 19).  phase4: the T4 terms are in the
  ! equations (before that: CCSDT, whose amplitudes are the start).  t4now:
  ! this iteration evaluates them at its amplitudes first.  Macro / micro
  ! (cc4mode 0): after an evaluation the equations of CCSDT with these terms
  ! are iterated (DIIS from new) until the change of the amplitudes is below
  ! fmicro times that of the first step after the evaluation, then the terms
  ! are evaluated again.  Converged when the first step after an evaluation
  ! is below the thresholds - the test of the plain iteration - or, without
  ! that last evaluation, when the first steps of the last two evaluations
  ! (d1, d2: a factor kap = d2 / d1 per evaluation) say that the next one
  ! would be below them: 2 kap d2 < the thresholds (option cc4 confirm: only
  ! the first test).
  phase4 = cc4 .and. (cc4start .or. cc4mode == 2 .or. cc4rst > 0)
  t4now = phase4
  fresh = .false.
  nsince = 0
  if (cc4rst == 0) then
    nt4 = 0
    t4sec = 0d0
  end if
  dmac = 0d0
  demac = 0d0
  dprev = 0d0
  ! (the iterations of an earlier run go on: the T4 terms of its last
  ! evaluation are held, the first step is the first one after it)
  resumed = cc4rst == 1
  if (resumed) then
    t4now = .false.
    fresh = .true.
    dprev = dsaved
  end if
  fmicro = 1d-2
  call xzoptget('cc4micro', dbg, ldbg)
  if (ldbg > 0) read(dbg(1:ldbg), *) fmicro
  if (cc4) then
    if (cc4mode == 0 .or. cc4mode == 3) then
      write(6, '(a,es9.2,a)') '  CC4: macro / micro iterations - the T4 terms (cycles marked T4) are held while the' // &
        ' equations of CCSDT with them are iterated to ', fmicro, ' of the first step'
    else
      write(6, '(a)') '  CC4: the T4 terms are evaluated in every iteration (option cc4)'
    end if
  end if
  ! (the other ranks read the integrals after rank 0: wait for them here, so
  ! that the clock of the first broadcast is not that of their reading)
  if (nrank > 1) then
    call xztic()
    call xzmpi_barrier()
    call xztoc(1, 'MPI: ranks reading integrals')
  end if
  icyc = 0
  do
    if (cc4rst == 2) then
      ! (the converged amplitudes of an earlier run: no iterations)
      converged = .true.
      ncyc = 0
      exit
    end if
    icyc = icyc + 1
    nsince = nsince + 1
    t0 = omp_get_wtime()
    if (nrank > 1) then
      ! one more iteration (2, 3: the T4 terms evaluated first; 2: the DIIS
      ! sequence starts again): the amplitudes to the ranks (T3 of the first
      ! one is zero, or follows: the start of CC4)
      call xztic()
      tsg = omp_get_wtime()
      ctl = [1d0, merge(0d0, 1d0, icyc == 1)]
      if (t4now) ctl(1) = merge(2d0, 3d0, cc4mode == 0 .or. cc4mode == 3 .or. nt4 == 0)
      if (icyc == 1 .and. (cc4start .or. cc4rst > 0)) ctl(2) = 2d0
      ! (4: the T4 term of the doubles equation of an earlier run follows)
      if (icyc == 1 .and. resumed) ctl(1) = 4d0
      call xzmpi_bcast(ctl, 16_8)
      call xzmpi_bcast_w(w(jt1), NW * no * nv)
      call xzmpi_bcast_w(w(jt2), NW * no * no * nv * nv)
      if ((icyc > 1 .and. .not. slices) .or. ctl(2) == 2d0) call xzmpi_bcast_w(w(jt3), NW * n3)
      if (ctl(1) == 4d0) call xzmpi_bcast_w(w(jd42), NW * npo * nvp)
      call xztoc(1, 'MPI: amplitudes to the ranks')
      dsec(6, 1) = dsec(6, 1) + omp_get_wtime() - tsg
    end if
    mark = ' '
    if (icyc == 1 .and. resumed) write(mark, '(a,i3,a)') '   T4', nt4, ' (XZCC4)'
    if (t4now) then
      t4s = omp_get_wtime()
      call xztic()
      call N(xzt4)(w, jt1, jt2, jt3, jd42, jd43, 0d0)
      call xztoc(1, 'CC4: the T4 terms')
      ! (macro / micro: the DIIS sequence starts again with the new terms)
      if (cc4mode == 0 .or. cc4mode == 3 .or. nt4 == 0) call N(xzdiis_restart)()
      nt4 = nt4 + 1
      nt4run = nt4run + 1
      t4sec = t4sec + omp_get_wtime() - t4s
      write(mark, '(a,i3)') '   T4', nt4
      fresh = .true.
      nsince = 1
      ! the checkpoint: these amplitudes and their T4 terms (macro / micro)
      if (cc4mode == 0 .or. cc4mode == 3) call N(xz4ckpt)(w, 'w', -1_8, eold, dprev)
      if (nt4stop > 0 .and. nt4run >= nt4stop) then
        write(6, '(/,a,i4,a)') ' @XZCCSDT-F, option cc4stop: the run stops after its evaluation ', nt4run, ' of the T4 terms.'
        call xzerrex
      end if
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
    write(6, '(i8,f23.14,es19.6,3es11.3,i7,f10.2,a)') icyc, enew, de, norms, ncnt, omp_get_wtime() - t0, trim(mark)
    flush(6)
    conv = abs(de) < conv_e .and. normt < conv_t
    if (.not. cc4) then
      if (conv) then
        converged = .true.
        ncyc = icyc
        exit
      end if
    else if (.not. phase4) then
      if (conv) then
        ! CCSDT is converged: its amplitudes to XZAMPS (for a later run), then CC4 from them
        e4ccsdt = enew
        call N(xzamps)(w, 'w', e4ccsdt)
        call xzkeyw(-1_8)
        write(6, '(a,i4,a)') '  CCSDT converged in ', icyc, ' cycles; its amplitudes written to the file XZAMPS (XZKEY).  CC4:'
        phase4 = .true.
        t4now = .true.
        nsince = 0
      end if
    else
      if (fresh) then
        ! the first step with new T4 terms: the test of convergence of CC4
        if (conv) then
          converged = .true.
          ncyc = icyc
          exit
        end if
        dmac = normt
        demac = abs(de)
      end if
      if (cc4mode == 1 .or. cc4mode == 2) then
        t4now = .true.
      else
        t4now = .not. fresh .and. normt < max(0.3d0 * conv_t, fmicro * dmac) &
                .and. abs(de) < max(0.3d0 * conv_e, fmicro * demac)
        if (t4now) then
          ! the equations with the T4 terms of evaluation nt4 are solved
          kap = 0d0
          if (dprev > 0d0) kap = dmac / dprev
          write(6, '(a,i3,a,f18.12,a,es9.2,a,es9.2,a,f7.4)') '     with the T4 terms of evaluation ', nt4, &
            ': correlation energy ', enew, ';  its first step changed the amplitudes by ', dmac, ', the energy by ', &
            demac, ';  factor ', kap
          if (cc4mode == 0 .and. dprev > 0d0 .and. kap < 0.5d0 .and. 2d0 * kap * dmac < conv_t &
              .and. 2d0 * kap * demac < conv_e) then
            write(6, '(a,es9.2,a)') '     new T4 terms would change the amplitudes by about ', kap * dmac, &
              ' in the first step (the factor of the last two evaluations): converged'
            converged = .true.
            ncyc = icyc
            exit
          end if
          dprev = dmac
        end if
      end if
      fresh = .false.
    end if
    if (timing_run .and. icyc >= maxcyc) exit
    if (nsince >= maxcyc .and. .not. t4now) exit
    if (cc4 .and. nt4 >= maxcyc) exit
  end do
  write(6, '(a)') '  ' // repeat('-', 98)
  if (nrank > 1) then
    ! the iterations are over
    ctl = 0d0
    call xzmpi_bcast(ctl, 16_8)
  end if
  ! CC4 converged: its amplitudes to the file XZCC4 (a later run of these
  ! integrals - the excited states - takes them)
  if (cc4 .and. converged .and. cc4rst /= 2 .and. .not. timing_run) &
    call N(xz4ckpt)(w, 'w', nint(-log10(conv_t), 8), eold, dprev)
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
    if (ctl(1) < 1d0) exit
    t0 = omp_get_wtime()
    tsg = t0
    call xztic()
    call xzmpi_bcast_w(w(jt1), NW * no * nv)
    call xzmpi_bcast_w(w(jt2), NW * no * no * nv * nv)
    if (ctl(2) == 2d0) then
      ! (the start of CC4: the CCSDT amplitudes of rank 0)
      call xzmpi_bcast_w(w(jt3), NW * n3)
    else if (slices) then
      ! (T3 is here: the slices of the ranks came at the end of the iteration before)
    else if (ctl(2) == 1d0) then
      call xzmpi_bcast_w(w(jt3), NW * n3)
      dwords(1) = dwords(1) + dble(NW * n3)
    else
      call zc_vzero_d(NW * n3, w(jt3))
    end if
    call xztoc(1, 'MPI: amplitudes from rank 0')
    dsec(6, 1) = dsec(6, 1) + omp_get_wtime() - tsg
    if (ctl(1) == 4d0) then
      ! CC4, the iterations of an earlier run go on: the T4 term of the doubles
      ! equation from rank 0 (that of the triples equation is rank 0's)
      call xzmpi_bcast_w(w(jd42), NW * npo * nvp)
      call zc_vzero_d(NW * n3, w(jd43))
      call N(xzdiis_restart)()
    else if (ctl(1) >= 2d0) then
      ! CC4: the part of this rank of the T4 terms (2: the DIIS sequence starts again)
      call N(xzt4)(w, jt1, jt2, jt3, jd42, jd43, 0d0)
      call xztoc(1, 'CC4: the T4 terms')
      if (ctl(1) == 2d0) call N(xzdiis_restart)()
    end if
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
  ! ('l': read into the regions that are there)
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
  if (rw /= 'l') then
    jt1 = xzget(n1, 'T1')
    jt2 = xzget(n2, 'T2')
    if (.not. t3shared) jt3 = xzget(NW * n3, 'T3')
  end if
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
