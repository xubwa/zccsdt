!
! The driver of xzccsdt: the job, the sizes, the symmetry labels, the memory
! plan, the integrals, the CCSDT iterations, (Q), the energies.
!
! Two things are decided from the integrals that are read:
!   blocks   the labels of the file xzlabels if it is there, else those of
!            the file dgcosym of CFOUR's SCF if that was done in more than
!            one block (%dglin: 2 m_j), are used - the spinors in the order
!            of their classes, T3 and <ab||cd> on the allowed blocks only -
!            provided no integral that they forbid is larger than 1e-8 (XZ_LABTOL);
!            otherwise the labels are dropped with a warning (one class: the
!            dense case of the same code)
!   real     if no integral has an imaginary part above 1e-10, the real
!            instance of the code runs (half the memory, real kernels)
! The integrals are read for the most favourable case first (labels, real);
! what does not hold shows in the small blocks, which are read first, and the
! integrals are then read again for the next case.
!
subroutine xzdrv(w, maxcor)
  use xzdata
  use xzmem
  use xzkern
  use xzpar
  use xzdist
  use xzpland, only: planq, pneedq, pneedw
  use xzeomd
  use omp_lib
  implicit none
  integer(8), intent(in) :: maxcor
  integer(8) :: mcc, kres
  real(8) :: w(maxcor)
  integer(8) :: iintln, ifltln, iintfp, ialone, ibitwd
  common /machsp/ iintln, ifltln, iintfp, ialone, ibitwd
  integer(8) :: nslot, nch, ncyc, nthr, ilp64, kb(2), nmax, status, nblk
  logical :: usefile, converged, uselab, tryreal, isreal, there, docc
  real(8) :: sync(2)
  real(8) :: ecorr, eq5, eq6, etot, t0, tread, tcc, tq, twall, tlm, tden, lhs, rhs, teom, elow
  character(len=32) :: arg
  integer :: larg

  twall = omp_get_wtime()
  call xzinit(loc(w(1)), maxcor)
  ! (a test hook: XZ_MAXCOR words of the work array are used)
  call get_environment_variable('XZ_MAXCOR', arg, larg)
  if (larg > 0 .and. myrank == 0) then
    read(arg, *) nmax
    xzsize = min(xzsize, nmax)
  end if
  call xzjob()
  if (.not. sizes_set) call xzsizes()
  call xz_blas_info(nthr, ilp64)

  write(6, '(/,a)') '  ' // repeat('=', 78)
  write(6, '(a)') '   XZCCSDT: spinor ' // trim(method) // ' on the integrals of the X2CSO programs'
  write(6, '(a)') '   (Fortran scheduler on the work array of CFOUR, C kernels of zccsdt)'
  write(6, '(a)') '  ' // repeat('=', 78)
  write(6, '(a,i6,a,i6)') '   active spinors: occupied ', no, '   virtual ', nv
  write(6, '(a,i6,a,i6,a,i6)') '   frozen occupied ', nfrozen, '   dropped virtual ', ndropped, &
    '   of ', ntot
  write(6, '(a,i14,a,i14)') '   triples: occupied ', nto, '   virtual ', nvt
  write(6, '(a,es9.1,a,es9.1,a,i5)') '   convergence: energy ', conv_e, '   amplitudes ', conv_t, &
    '   cycles ', maxcyc
  write(6, '(a,f20.12)') '   E(SCF) of JOBARC: ', escf
  write(6, '(a,i14,a,f10.1,a)') '   work array (MAXCOR): ', maxcor, ' words = ', 8d-6 * maxcor, ' MB'
  write(6, '(a,i4,a,i2)') '   threads: ', nthr, '   BLAS integers: bytes ', 4 + 4 * ilp64
  if (xzmpi_enabled() == 1) then
    write(6, '(a,i5,a,i5,a)') '   MPI: rank ', myrank, ' of ', nrank, &
      ' (the iterations and (Q) run over the ranks; a rank other than 0 has a work array of its own,'
    write(6, '(a,i10,a)') '        as large as its part needs; arrays go between the ranks in messages of ', &
      xzmpi_chunk(), ' words)'
  end if
  call xzreuse()
  ! (Q) reads T3 (and Lambda3) where the files are mapped unless XZ_QMAP=0
  t3shared = with_q
  call get_environment_variable('XZ_QMAP', arg, larg)
  if (larg > 0 .and. arg(1:1) == '0') t3shared = .false.
  if (with_q .and. .not. t3shared) write(6, '(a)') '   XZ_QMAP=0: (Q) with a private copy of T3 in the work array of every rank'
  if (t3shared .and. xz_fs_local('.' // c_null_char) == 0) then
    write(6, '(a)') ' @XZCCSDT-W, the run directory is on a network file system: (Q) maps the file XZAMPS there'
    write(6, '(a)') '             (one copy of T3 in the page cache of every node; pages are read over the network'
    write(6, '(a)') '             when they are first needed or after the system has dropped them).  XZ_QMAP=0 reads'
    write(6, '(a)') '             T3 into the work array of every rank instead.'
  end if
  if (ilp64 /= 1) then
    write(6, '(a)') ' @XZCCSDT-F, the kernels were not compiled for the 8-byte integers of this BLAS.'
    call xzerrex
  end if
  if (no < 3 .or. nv < 3) then
    write(6, '(a)') ' @XZCCSDT-F, fewer than three occupied or virtual spinors: no triples.'
    call xzerrex
  end if
  if (with_eom .and. nrank > 1) then
    write(6, '(a)') ' @XZCCSDT-F, the excited states (EOM) run with one rank: the sigma vectors are not distributed.'
    write(6, '(a)') '             Run CALC=CCSDT (without EXCITE) over the ranks, then EXCITE=EOMEE with one rank:'
    write(6, '(a)') '             the second run takes the amplitudes of the file XZAMPS of the first.'
    call xzerrex
  end if

  ! ---- the symmetry labels
  allocate(lab(0:ntot-1), operm(0:no-1), vperm(0:nv-1), ocls(0:no-1), vcls(0:nv-1))
  !      (every rank reads them from the files of its run directory, by the
  !      same rule: 1. xzlabels, 2. dgcosym with more than one block, 3. none)
  nblk = 0
  call xzrdlab(ntot, lab, modulus, there)
  if (there) then
    labfile = 'xzlabels'
    write(6, '(a,i4)') '   symmetry labels: the file xzlabels, modulus ', modulus
  else
    call xzrddgc(ntot, lab, modulus, there, nblk)
    if (there) then
      labfile = 'dgcosym'
      write(6, '(a,i0,a,i0,a,i0)') '   symmetry labels: the file dgcosym of CFOUR''s SCF (m_j blocks), block numbers up to ', &
        nblk, ': labels 2 m_j, |2 m_j| up to ', 2 * ((nblk + 1) / 2) - 1, ', modulus ', modulus
    else if (nblk == 1) then
      write(6, '(a)') '   symmetry labels: none (no file xzlabels; dgcosym: one block, an SCF without %dglin)'
    else
      write(6, '(a)') '   symmetry labels: none (no file xzlabels, no file dgcosym)'
    end if
  end if
  have_labels = there
  uselab = there
  call get_environment_variable('XZ_LABELS', arg, larg)
  if (larg > 0 .and. arg(1:1) == '0' .and. uselab) then
    write(6, '(a)') '   XZ_LABELS=0: the labels are not used'
    uselab = .false.
  end if
  ! (labels with a modulus: the blocks of the CCSDT iterations and of (Q)
  ! take the sums of classes modulo mgrp; Lambda, the density and EOM do not
  ! yet, and run in one block)
  if (uselab .and. modulus /= 0 .and. (with_lam .or. with_den .or. with_eom)) then
    write(6, '(a)') ' @XZCCSDT-W, labels with a modulus are not used for Lambda, the density or EOM yet: one block.'
    uselab = .false.
  end if
  tryreal = .true.
  call get_environment_variable('XZ_REAL', arg, larg)
  if (larg > 0 .and. arg(1:1) == '0') then
    write(6, '(a)') '   XZ_REAL=0: complex arithmetic whatever the integrals'
    tryreal = .false.
  end if

  ! ---- which ranks do what: every rank reads the integrals itself; the
  !      iterations (CCSDT, Lambda) run on all ranks (xzdist.F90: rank 0 sends
  !      the amplitudes, every rank does its batches), (Q) on all ranks from
  !      the files XZAMPS and XZLAMB, the density stage on rank 0
  docc = with_cc
  planq = .not. (with_cc .or. with_lam .or. with_den .or. with_eom)
  if (with_den .and. .not. (with_cc .or. with_lam) .and. myrank > 0) return
  ! (a test hook: XZ_DEN_ACTIVE=1 writes the files of the active spinors alone)
  call get_environment_variable('XZ_DEN_ACTIVE', arg, larg)
  den_active = larg > 0
  if (method == 'MP2' .and. myrank > 0) return
  sync = 0d0

  ! ---- the integrals: for labels and real numbers first; the memory of a
  !      case is checked before it is read (the floor of its plan)
  isreal = tryreal
  t0 = omp_get_wtime()
  do
    call xzsetmode(uselab)
    nwe = merge(1_8, 2_8, isreal)
    call xzslices()
    call xzplan(w, nslot, usefile, nch, .true.)
    call xzrel(0_8)
    call xzlayout(w, status)
    if (status == 0) exit
    if (status == 1) then
      uselab = .false.
      isreal = tryreal
    else
      isreal = .false.
    end if
  end do
  tread = omp_get_wtime() - t0
  write(6, '(/,a,a,a,i4,a)') '   arithmetic: ', merge('real   ', 'complex', isreal), ';  blocks: ', ncls, &
    merge(' classes of the labels', ' class (dense)        ', ncls > 1)
  write(6, '(a,f12.1,a,i16,a)') '   T3 on i<j<k, a<b<c: ', 8d-6 * nwe * n3, ' MB (', n3, ' amplitudes)'
  if (ncls > 1) write(6, '(a,f8.2,a,f8.2,a)') '   of the dense arrays: T3 ', 100d0 * n3 / (nto * nvt), &
    ' %, <ab||cd> ', 100d0 * nvv / (nvp * nvp), ' %'
  write(6, '(a,a)') '   hh ladder and F_oo of T3: ', merge('through occupied pairs (as in CCSD)   ', &
    'one operator on the occupied triples  ', xzhhpair())
  write(6, '(a,f10.2,a)') '   integrals read and sorted: ', tread, ' s'
  flush(6)
  ! (a rank other than 0 has decided from its own files: the labels, their
  ! check against the integrals, real or complex - as rank 0, or not at all)
  if (myrank > 0 .and. rplan(13) > 0d0) then
    if (int(rplan(13), 8) /= ncls .or. int(rplan(14), 8) /= nwe .or. int(rplan(15), 8) /= n3) then
      write(6, '(/,a)') ' @XZCCSDT-F, this rank and rank 0 have not decided alike from the files of their run directories:'
      write(6, '(a,i4,a,i2,a,i16)') '             here    classes ', ncls, '   words per element ', nwe, &
        '   elements of T3 ', n3
      write(6, '(a,i4,a,i2,a,i16)') '             rank 0  classes ', int(rplan(13), 8), '   words per element ', &
        int(rplan(14), 8), '   elements of T3 ', int(rplan(15), 8)
      write(6, '(a)') '             (xzlabels, dgcosym and the integral files must be the same on every node.)'
      call xzerrex
    end if
  end if

  ! ---- MP2: the second-order energy of the integrals as they were read (a
  !      check of the files, e.g. against the number that xx2cvtran prints)
  if (method == 'MP2') then
    if (nwe == 2) then
      call xzmp2_z(w, ecorr)
    else
      call xzmp2_d(w, ecorr)
    end if
    write(6, '(/,a,f25.12)') '   second-order correlation energy  ', ecorr
    write(6, '(a,f25.12)') '   E(MP2)                           ', escf + ecorr
    write(6, '(a,i14,a,f10.1,a)') '   work array used: ', xzpeak, ' words = ', 8d-6 * xzpeak, ' MB'
    write(6, '(a)') '   XZCCSDT finished: MP2 energy done (TOTENERG not written).'
    return
  end if

  ! ---- the memory plan
  call xzplan(w, nslot, usefile, nch, .false.)
  ! (the other ranks allocate what their part takes with this plan and use
  ! its batch sizes: xzrankmem, xzplan)
  if (myrank == 0 .and. nrank > 1 .and. (with_q .or. with_cc .or. with_lam)) then
    rplan = 0d0
    rplan(1) = dble(pneedw)
    rplan(2:9) = dble([nqs, nbt, nbp, nba, qxv, qc22, qc13, qrows])
    rplan(10:12) = dble([nslot, merge(1_8, 0_8, usefile), nch])
    rplan(13:15) = dble([ncls, nwe, n3])
    call xzmpi_bcast(rplan, 128_8)
  end if
  ! (a test hook: XZ_WFAIL, an error on a rank other than 0)
  call get_environment_variable('XZ_WFAIL', length=larg)
  if (larg > 0 .and. myrank > 0) then
    write(6, '(a)') ' @XZCCSDT-F, XZ_WFAIL: a test of the exit of a rank other than 0.'
    call xzerrex
  end if
  ! the batches of the iterations, dealt to the ranks
  if (with_cc .or. with_lam) call xzdeal()
  flush(6)

  ! ---- CCSDT (all ranks: xzdist.F90), or the amplitudes of an earlier run
  tcc = 0d0
  tlm = 0d0
  mcc = xzmark()
  if (docc) then
    t0 = omp_get_wtime()
    if (nwe == 2) then
      call xzcc_z(w, nslot, usefile, nch, ecorr, converged, ncyc)
    else
      call xzcc_d(w, nslot, usefile, nch, ecorr, converged, ncyc)
    end if
    tcc = omp_get_wtime() - t0
    call xzdtable(1, ncyc)
    call xzbdump()
    if (myrank > 0) call xztable(1, 'the CCSDT iterations on this rank', tcc, ncyc)
  end if
  if (docc .and. myrank == 0) then
    if (.not. converged .and. timing_run) then
      write(6, '(/,a,i5,a)') ' @XZCCSDT-W, XZ_NCYC: CCSDT stopped after ', maxcyc, &
        ' cycles; the amplitudes and what follows are NOT converged.'
      converged = .true.
    end if
    if (.not. converged) then
      call get_environment_variable('XZ_DEBUG', length=larg)
      if (larg == 0) write(6, '(/,a,i5,a)') ' @XZCCSDT-F, CCSDT did not converge in ', maxcyc, &
        ' cycles (CC_MAXCYC).'
      call xzmpi_bcast(sync, 16_8)
      if (larg > 0) return
      call xzerrex
    end if
    write(6, '(a,i4,a)') '   CCSDT converged in ', ncyc, ' cycles.'
    call xztable(1, 'the CCSDT iterations', tcc, ncyc)
    ! the amplitudes for the steps that follow, here or in a later run
    t0 = omp_get_wtime()
    if (nwe == 2) then
      call xzamps_z(w, 'w', ecorr)
    else
      call xzamps_d(w, 'w', ecorr)
    end if
    call xzkeyw(-1_8)
    write(6, '(/,a,f10.1,a,f8.2,a)') '   amplitudes written to the file XZAMPS (', &
      8d-6 * nwe * (no * nv + no * no * nv * nv + n3), ' MB, ', omp_get_wtime() - t0, ' s; XZKEY)'
    ! (Q) follows at once: T3 leaves the work array, the file is mapped as on the other ranks
    if (t3shared .and. .not. with_lam) then
      call xzrel(jt3 - xzbase)
      call xzmapt3(w, 'XZAMPS', nwe * (no * nv + no * no * nv * nv), nwe * n3, jt3)
    end if
  else if (.not. docc .and. myrank > 0 .and. with_lam) then
    ! (a rank other than 0 before the Lambda equations: rank 0 sends the amplitudes)
    jt1 = xzget(nwe * no * nv, 'T1')
    jt2 = xzget(nwe * no * no * nv * nv, 'T2')
    jt3 = xzget(nwe * n3, 'T3')
  else if (.not. docc) then
    if (nwe == 2) then
      call xzamps_z(w, 'r', ecorr)
    else
      call xzamps_d(w, 'r', ecorr)
    end if
    write(6, '(/,a,f20.12)') '   amplitudes read from the file XZAMPS; CCSDT correlation energy ', ecorr
  end if
  flush(6)
  ! the amplitudes are there (rank 0 has written XZAMPS, or stops)
  if ((with_cc .or. with_lam) .and. nrank > 1) then
    if (myrank == 0) sync = [ecorr, 1d0]
    call xzmpi_bcast(sync, 16_8)
    if (myrank > 0) then
      if (sync(2) /= 1d0) return
      ecorr = sync(1)
      if (docc .and. with_q .and. .not. with_lam) then
        ! (Q) follows: T1 and T2 of the file, T3 where it is mapped
        call xzrel(mcc)
        if (nwe == 2) then
          call xzamps_z(w, 'r', ecorr)
        else
          call xzamps_d(w, 'r', ecorr)
        end if
      end if
    end if
    sync = 0d0
  end if

  ! ---- the Lambda equations (all ranks), or lambda of an earlier run
  if (with_lam) then
    t0 = omp_get_wtime()
    if (nwe == 2) then
      call xzlam_z(w, nslot, usefile, nch, converged, ncyc)
    else
      call xzlam_d(w, nslot, usefile, nch, converged, ncyc)
    end if
    tlm = omp_get_wtime() - t0
    call xzdtable(2, ncyc)
    call xzbdump()
    if (myrank > 0) call xztable(2, 'the iterations of the Lambda equations on this rank', tlm - tlamim, ncyc)
  end if
  if (with_lam .and. myrank == 0) then
    if (.not. converged .and. timing_run) then
      write(6, '(/,a,i5,a)') ' @XZCCSDT-W, XZ_NCYC: the Lambda equations stopped after ', maxcyc, &
        ' cycles; lambda and what follows are NOT converged.'
      converged = .true.
    end if
    if (.not. converged) then
      write(6, '(/,a,i5,a)') ' @XZCCSDT-F, the Lambda equations of CCSDT did not converge in ', maxcyc, &
        ' cycles (CC_MAXCYC).'
      call xzmpi_bcast(sync, 16_8)
      call xzerrex
    end if
    write(6, '(a,i4,a)') '   The Lambda equations converged in ', ncyc, ' cycles.'
    call xztable(3, 'the intermediates of the Lambda equations', tlamim, 0_8)
    call xztable(2, 'the iterations of the Lambda equations', tlm - tlamim, ncyc)
    t0 = omp_get_wtime()
    if (nwe == 2) then
      call xzlamb_z(w, 'w', ecorr)
    else
      call xzlamb_d(w, 'w', ecorr)
    end if
    call xzkeyw(nint(-log10(conv_t), 8))
    write(6, '(/,a,f10.1,a,f8.2,a)') '   lambda written to the file XZLAMB (', &
      8d-6 * nwe * (no * nv + no * no * nv * nv + n3), ' MB, ', omp_get_wtime() - t0, ' s)'
  end if
  if (with_lam) then
    ! lambda is there (rank 0 has written XZLAMB, or stops)
    if (nrank > 1) then
      if (myrank == 0) sync = [ecorr, 1d0]
      call xzmpi_bcast(sync, 16_8)
      if (myrank > 0 .and. (sync(2) /= 1d0 .or. .not. with_q)) return
      sync = 0d0
    end if
    if (with_q .or. with_den) then
      ! the pp operator took the place of <ab||cd>: the integrals and the
      ! amplitudes are read again for what follows
      call xzrel(0_8)
      call xzlayout(w, status)
      if (status /= 0) then
        write(6, '(/,a)') ' @XZCCSDT-F, the integral files changed during the run.'
        call xzerrex
      end if
      if (nwe == 2) then
        call xzamps_z(w, 'r', ecorr)
      else
        call xzamps_d(w, 'r', ecorr)
      end if
    end if
  end if
  if ((q_left .and. with_q) .or. with_den) then
    if (nwe == 2) then
      call xzlamb_z(w, 'r', ecorr)
    else
      call xzlamb_d(w, 'r', ecorr)
    end if
    if (.not. (with_lam .and. myrank == 0)) write(6, '(a)') '   lambda read from the file XZLAMB'
  end if
  flush(6)

  ! ---- the excited states: EOM-EE-CCSDT (the pp operator takes the place of <ab||cd>)
  teom = 0d0
  if (with_eom) then
    t0 = omp_get_wtime()
    if (nwe == 2) then
      call xzeom_z(w, ecorr)
    else
      call xzeom_d(w, ecorr)
    end if
    teom = omp_get_wtime() - t0
    call get_environment_variable('XZ_EOMTEST', length=larg)
    if (larg > 0) return
    call get_environment_variable('XZ_EOMSIG', length=larg)
    if (larg > 0) return
    call xztable(5, 'EOM: the intermediates and the sigma vectors', teomim + teomsig + teomsd, neomsig)
  end if

  ! ---- the density and the orbital gradient of CCSDT; the files of xx2czvec
  tden = 0d0
  if (with_den) then
    t0 = omp_get_wtime()
    if (nwe == 2) then
      call xzden_z(w)
    else
      call xzden_d(w)
    end if
    tden = omp_get_wtime() - t0
    call xztable(4, 'the density and the orbital gradient', tden, 0_8)
    write(6, '(/,a)') '   unrelaxed density and orbital gradient of CCSDT: the files xfull, dpoo, dpvv, dii0,'
    write(6, '(a)') '   daa0, dpopo, dpvpv written for xx2czvec (the Z-vector equations and denfinal are its).'
    ! what xx2czvec of cfour-x2cso (2026-10) needs to be right: docs/native.md, section 10.4
    lhs = 1d0 + 0.5d0 * dble(nv) * dble(nv - 1) * dble(nv - 2) + dble(nv) * dble(nv + 1) + dble(no) * dble(no + 1)
    rhs = 2d0 * dble(ntot) * dble(ntot)
    if (lhs < rhs) then
      write(6, '(/,a)') ' @XZCCSDT-W, xx2czvec will NOT give the relaxed density of these files: with so few'
      write(6, '(a)') '             active virtual spinors its response integrals are wrong (an unset work'
      write(6, '(a)') '             offset in its source; CFOUR''s own relaxed CC densities have the defect too).'
      write(6, '(a,f16.0,a,f16.0)') '             1 + 3 C(nv,3) + nv (nv+1) + no (no+1) = ', lhs, &
        '   is below   2 N^2 = ', rhs
      write(6, '(a,i6,a,i6,a,i6)') '             (active occupied ', no, ', active virtual ', nv, ', spinors N ', ntot
      write(6, '(a)') '             The files themselves are right; solve the Z-vector equations elsewhere.'
    end if
  end if

  ! ---- (Q), on all ranks
  eq5 = 0d0
  eq6 = 0d0
  tq = 0d0
  if (with_q) then
    t0 = omp_get_wtime()
    if (nwe == 2) then
      call xzq_z(w, eq5, eq6)
    else
      call xzq_d(w, eq5, eq6)
    end if
    tq = omp_get_wtime() - t0
  end if
  if (myrank > 0) return
  if (qsampled) then
    write(6, '(/,a)') '   XZCCSDT finished: a sample of (Q); no energy of (Q), TOTENERG not written.'
    return
  end if

  ! ---- energies
  write(6, '(/,a)') '  ' // repeat('-', 62)
  write(6, '(a,f25.12)') '   E(SCF)                        ', escf
  write(6, '(a,f25.12)') '   CCSDT correlation energy      ', ecorr
  write(6, '(a,f25.12)') '   E(CCSDT)                      ', escf + ecorr
  etot = escf + ecorr
  if (with_q .and. .not. q_left) then
    write(6, '(a,f25.12)') '   E[Q] (fifth order)            ', eq5
    write(6, '(a,f25.12)') '   E(Q)                          ', eq6
    write(6, '(a,f25.12)') '   E(CCSDT[Q])                   ', escf + ecorr + eq5
    write(6, '(a,f25.12)') '   E(CCSDT(Q))                   ', escf + ecorr + eq6
    etot = escf + ecorr + eq6
    if (method == 'CCSDT[Q]' .or. method == '[Q]') etot = escf + ecorr + eq5
  else if (with_q) then
    write(6, '(a,f25.12)') '   E[Q]_L (fifth order)          ', eq5
    write(6, '(a,f25.12)') '   E(Q)_L                        ', eq6
    write(6, '(a,f25.12)') '   E(CCSDT[Q]_L)                 ', escf + ecorr + eq5
    write(6, '(a,f25.12)') '   E(CCSDT(Q)_L)                 ', escf + ecorr + eq6
    etot = escf + ecorr + eq6
    if (method == 'CCSDT[Q]_L' .or. method == '[Q]_L') etot = escf + ecorr + eq5
  end if
  write(6, '(a)') '  ' // repeat('-', 62)
  call putrec(20, 'JOBARC', 'TOTENERG', iintfp, etot)
  write(6, '(a,f25.12)') '   TOTENERG written to JOBARC:   ', etot
  if (with_eom) then
    ! the excited states: the table, the file XZEOM, the lowest state to JOBARC (TOTENER2, as xx2ceomcc)
    write(6, '(/,a)') '   EOM-EE-CCSDT: the states'
    write(6, '(a)') '  ' // repeat('-', 98)
    write(6, '(a)') '   shift  root        omega / Eh            eV            cm-1          total energy / Eh    max |r|'
    open(unit=79, file='XZEOM', form='formatted', status='replace')
    write(79, '(a)') '# EOM-EE-CCSDT of xzccsdt: label shift, root, omega / Eh, total energy / Eh, ' // &
                     'largest element and norm of the residual, weights of R1, R2, R3, converged'
    write(79, '(a,f25.12,a,f25.12)') '# E(SCF) ', escf, '   E(CCSDT) ', escf + ecorr
    elow = huge(1d0)
    do kres = 1, nres
      write(6, '(i8,i6,f20.12,f14.8,f16.4,f27.12,es11.2,a)') rshift(kres), rroot(kres), rom(kres), &
        rom(kres) * 27.211386245988d0, rom(kres) * 219474.6313632d0, escf + ecorr + rom(kres), rmx(kres), &
        merge('                ', '  NOT converged ', rconv(kres))
      write(79, '(2i6,2f25.14,2es12.4,3f10.6,l3,2i7,f10.6)') rshift(kres), rroot(kres), rom(kres), &
        escf + ecorr + rom(kres), rmx(kres), rnm(kres), rwt(:, kres), rconv(kres), rli(kres), rla(kres), rlv(kres)
      if (rconv(kres)) elow = min(elow, rom(kres))
    end do
    close(79)
    write(6, '(a)') '  ' // repeat('-', 98)
    write(6, '(a)') '   (the file XZEOM has this table; the file XZEOMR, if it was written, the vectors)'
    if (elow < huge(1d0)) then
      call putrec(20, 'JOBARC', 'TOTENER2', iintfp, escf + ecorr + elow)
      write(6, '(a,f25.12)') '   TOTENER2 written to JOBARC (the lowest state):   ', escf + ecorr + elow
    end if
  end if
  call xz_rss(kb)
  write(6, '(/,a,f10.2,a,f10.2,a,f10.2,a,f10.2,a)') '   wall times: integrals ', tread, ' s   CCSDT ', tcc, &
    ' s   Lambda ', tlm, ' s   (Q) ', tq, ' s'
  if (with_den) write(6, '(a,f10.2,a)') '               density and gradient ', tden, ' s'
  if (with_eom) write(6, '(a,f10.2,a,i6,a,f10.2,a,i6,a,f10.2,a)') '               EOM ', teom, ' s:  ', neomsig, &
    ' sigma vectors in ', teomsig, ' s, ', neomsd, ' of the singles-doubles block in ', teomsd, ' s'
  write(6, '(a,i14,a,f10.1,a,f8.1,a)') '   work array used: ', xzpeak, ' words = ', 8d-6 * xzpeak, &
    ' MB (', 100d0 * xzpeak / max(xzsize, 1_8), ' % of MAXCOR)'
  write(6, '(a,f10.1,a)') '   peak resident memory of the process: ', 1d-3 * kb(1), ' MB'
  write(6, '(a,f10.2,a)') '   XZCCSDT total wall time: ', omp_get_wtime() - twall, ' s'
  if (with_eom) then
    write(6, '(a)') '   XZCCSDT finished: EOM-EE-CCSDT done.'
  else if (with_den) then
    write(6, '(a)') '   XZCCSDT finished: density of CCSDT done.'
  else
    write(6, '(a)') '   XZCCSDT finished: ' // trim(method) // ' energy done.'
  end if
end subroutine xzdrv

! ---------------------------------------------------------------------------
! What to do, from the ZMAT as joda stored it (the flags in JOBARC), as every
! CFOUR program decides: the program takes no argument.
!
!   CALC (iflags(2))                     EXCITE / PROPS / DERIV_LEV   job
!   CCSDT, CC(n) with EXCITATION=3       -                            CCSDT
!   CCSDT[Q]                             -                            CCSDT, [Q]
!   CCSDT(Q), CC(n-1)(n), EXCITATION=4   -                            CCSDT, (Q) and [Q]
!   CCSDT(Q)_L, CC(n-1)(n)_L, EXC.=4     -                            CCSDT, Lambda, (Q)_Lambda
!   CCSDT                                EXCITE=EOMEE                 CCSDT, EOM-EE-CCSDT
!   CCSDT                                PROPS (not EXCITATION) or    CCSDT, Lambda, the density
!                                        DERIV_LEV > 0                files of xx2czvec
!   MP2 (MBPT(2))                        -                            the MP2 energy (a check)
!   CC3                                  (EXCITE=EOMEE)               not in this version
!
! joda turns EXCITE=EOMEE with CALC=CCSDT (and CC_PROG other than NCC, MRCC)
! into PROPS=EXCITATION, EXCITE=NONE: both forms mean EOM here.  The
! iterations of an earlier run are not repeated when the files XZAMPS
! (XZLAMB) of the run directory belong to these integrals (xzreuse).
! Thresholds: the amplitudes to 10**-CC_CONV, the energy to 10**-(CC_CONV + 2).
subroutine xzjob()
  use xzdata
  use xzeomd, only: xzeominp
  implicit none
  integer(8) :: iflags, iflags2
  common /flags/ iflags(100)
  common /flags2/ iflags2(500)
  character(len=32) :: arg
  integer :: larg
  integer(8) :: level, nexc, iexc, iprop, ideriv
  logical :: eom, den
  logical, save :: done = .false.

  ! (a rank other than 0 is here twice: before its work array and in xzdrv)
  if (done) return
  done = .true.
  if (command_argument_count() > 0) then
    call get_command_argument(1, arg, larg)
    write(6, '(/,a,a,a)') ' @XZCCSDT-F, xzccsdt takes no argument (given: ', trim(arg), &
      ').  The job comes from the ZMAT:'
    write(6, '(a)') '   CALC=CCSDT, CCSDT[Q], CCSDT(Q), CCSDT(Q)_L (also CC(n) with EXCITATION=3, CC(n-1)(n) and'
    write(6, '(a)') '   CC(n-1)(n)_L with EXCITATION=4), MP2; with CALC=CCSDT: EXCITE=EOMEE for EOM-EE-CCSDT, or'
    write(6, '(a)') '   PROPS=FIRST_ORDER (or DERIV_LEV=1) for the density files of xx2czvec.  The amplitudes and'
    write(6, '(a)') '   lambda of an earlier run (files XZAMPS, XZLAMB) are taken when they belong to these integrals.'
    call xzerrex
  end if
  level = iflags(2)
  nexc = iflags2(38)
  iexc = iflags(87)
  iprop = iflags(18)
  ideriv = iflags(3)
  eom = iexc == 3 .or. iprop == 14
  den = .not. eom .and. (iprop > 0 .or. ideriv > 0)
  method = ' '
  if (level == 18 .or. (level == 49 .and. nexc == 3)) method = 'CCSDT'
  if (level == 30) method = 'CCSDT[Q]'
  if (level == 54 .or. (level == 53 .and. nexc == 4)) method = 'CCSDT(Q)'
  if (level == 68 .or. (level == 69 .and. nexc == 4)) method = 'CCSDT(Q)_L'
  if (level == 1 .or. level == 42) method = 'MP2'
  if (level == 32) then
    ! ---- CC3 (and EOM-EE-CC3 with EXCITE=EOMEE): the place of the next method
    write(6, '(/,a)') ' @XZCCSDT-F, CALC=CC3 is not in this version of xzccsdt.'
    call xzerrex
  end if
  if (method == ' ') then
    write(6, '(/,a,i4,a,i3,a)') ' @XZCCSDT-F, CALC of the ZMAT (level ', level, ', EXCITATION ', nexc, &
      ') is not one of CCSDT, CCSDT[Q], CCSDT(Q), CCSDT(Q)_L, MP2.'
    call xzerrex
  end if
  if (eom .or. den) then
    if (method /= 'CCSDT') then
      write(6, '(/,a,a,a)') ' @XZCCSDT-F, ', trim(merge('EXCITE=EOMEE          ', 'PROPS or DERIV_LEV > 0', eom)), &
        ' goes with CALC=CCSDT only (here CALC=' // trim(method) // ').'
      call xzerrex
    end if
    method = merge('CCSDT_EOM    ', 'CCSDT_DENSITY', eom)
  end if
  with_eom = eom
  if (with_eom) call xzeominp()
  with_cc = method /= 'MP2'
  q_left = method == 'CCSDT(Q)_L'
  with_den = den
  with_lam = q_left .or. with_den
  with_q = method == 'CCSDT[Q]' .or. method == 'CCSDT(Q)' .or. q_left
  conv_t = 10d0 ** (-abs(iflags(4)))
  conv_e = 10d0 ** (-abs(iflags(4)))
  maxcyc = iflags(7)
  if (maxcyc <= 0) maxcyc = 50
  ! a timing run: XZ_NCYC cycles of every iteration, converged or not
  call get_environment_variable('XZ_NCYC', arg, larg)
  if (larg > 0) then
    read(arg(1:larg), *) maxcyc
    timing_run = .true.
  end if
  ndiis = 8
end subroutine xzjob

! ---------------------------------------------------------------------------
! The amplitudes (lambda) of an earlier run in the run directory, taken in
! the place of the iterations if they belong to the integrals of this run.
! The file XZKEY, written by rank 0 with XZAMPS (and again with XZLAMB),
! records what they were solved for:
!   XZKEY01
!   active occupied, active virtual, spinors, frozen occupied spinors
!   E(SCF) of JOBARC (SCFENEG)
!   the sum of the spinor energies (scf2ceig, all spinors), and the sum of
!   (p + 1) times the energy of spinor p (the order)
!   CC_CONV of the amplitudes, CC_CONV of lambda (-1: no lambda)
! XZAMPS is taken if XZKEY and XZAMPS are there, its header has these active
! spaces, the numbers above agree (sizes exactly, energies to 1e-9 relative)
! and it was converged at least as tightly (CC_CONV not smaller); XZLAMB if
! XZAMPS is taken, XZLAMB is lambda of these amplitudes (its header, the
! CCSDT energy) and its CC_CONV is not smaller.  The labels and the
! arithmetic (real or complex) are decided again from the same files; a
! file of another mode (XZ_LABELS, XZ_REAL changed) stops the run when it is
! read.  Rank 0 decides, the other ranks are told.
subroutine xzreuse()
  use xzdata
  use xzpar
  use xzdist
  implicit none
  integer(8) :: iflags
  common /flags/ iflags(100)
  logical, save :: done = .false.
  logical :: tok, lok
  real(8) :: dec(2)
  if (done) return
  done = .true.
  tok = .false.
  lok = .false.
  if (myrank == 0 .and. with_cc) call xzkeychk(int(abs(iflags(4)), 8), with_lam, tok, lok)
  dec = [merge(1d0, 0d0, tok), merge(1d0, 0d0, lok)]
  call xzmpi_bcast(dec, 16_8)
  tok = dec(1) == 1d0
  lok = dec(2) == 1d0
  if (.not. with_cc) return
  if (tok) then
    with_cc = .false.
    write(6, '(/,a)') '   amplitudes: the file XZAMPS of an earlier run belongs to these integrals (XZKEY):'
    write(6, '(a)') '               taken, the CCSDT iterations are skipped'
  else
    write(6, '(/,a)') '   amplitudes: solved here (no file XZAMPS of these integrals, or not converged as tightly)'
  end if
  if (with_lam) then
    if (lok) then
      with_lam = .false.
      write(6, '(a)') '   lambda:     the file XZLAMB of an earlier run is lambda of these amplitudes: taken,'
      write(6, '(a)') '               the Lambda iterations are skipped'
    else
      write(6, '(a)') '   lambda:     solved here'
    end if
  end if
end subroutine xzreuse

! the numbers of XZKEY for this run (rank 0)
subroutine xzkeyval(e)
  use xzdata
  implicit none
  real(8), intent(out) :: e(3)
  real(8), allocatable :: buf(:)
  integer(8) :: p
  allocate(buf(2 * ntot))
  call xzrfile('scf2ceig', buf, 2 * ntot)
  e(1) = escf
  e(2) = 0d0
  e(3) = 0d0
  do p = 0, ntot - 1
    e(2) = e(2) + buf(2 * p + 1)
    e(3) = e(3) + dble(p + 1) * buf(2 * p + 1)
  end do
  deallocate(buf)
end subroutine xzkeyval

! XZKEY for the amplitudes just written (lconv < 0) or for lambda (lconv: its CC_CONV)
subroutine xzkeyw(lconv)
  use xzdata
  implicit none
  integer(8), intent(in) :: lconv
  integer(8) :: iflags
  common /flags/ iflags(100)
  real(8) :: e(3)
  integer :: ios
  call xzkeyval(e)
  open(unit=76, file='XZKEY', form='formatted', status='replace', iostat=ios)
  if (ios /= 0) return
  write(76, '(a)') 'XZKEY01'
  write(76, '(4i12)') no, nv, ntot, nfrozen
  write(76, '(es26.17)') e(1)
  write(76, '(2es26.17)') e(2), e(3)
  write(76, '(2i6)') int(abs(iflags(4)), 8), lconv
  close(76)
end subroutine xzkeyw

! the check of XZKEY, XZAMPS and XZLAMB (rank 0)
subroutine xzkeychk(conv, needl, tok, lok)
  use xzdata
  implicit none
  integer(8), intent(in) :: conv
  logical, intent(in) :: needl
  logical, intent(out) :: tok, lok
  character(len=8) :: tag
  integer(8) :: k(4), c(2), head(5), headl(5)
  real(8) :: e(3), ek(3), ea, el
  integer :: ios
  logical :: there
  tok = .false.
  lok = .false.
  inquire(file='XZKEY', exist=there)
  if (.not. there) return
  inquire(file='XZAMPS', exist=there)
  if (.not. there) return
  open(unit=76, file='XZKEY', form='formatted', status='old', action='read', iostat=ios)
  if (ios /= 0) return
  read(76, '(a)', iostat=ios) tag
  if (ios == 0) read(76, *, iostat=ios) k
  if (ios == 0) read(76, *, iostat=ios) ek(1)
  if (ios == 0) read(76, *, iostat=ios) ek(2), ek(3)
  if (ios == 0) read(76, *, iostat=ios) c
  close(76)
  if (ios /= 0 .or. tag /= 'XZKEY01') return
  call xzkeyval(e)
  if (k(1) /= no .or. k(2) /= nv .or. k(3) /= ntot .or. k(4) /= nfrozen) return
  if (abs(e(1) - ek(1)) > 1d-9 * max(1d0, abs(e(1)))) return
  if (abs(e(2) - ek(2)) > 1d-9 * max(1d0, abs(e(2)))) return
  if (abs(e(3) - ek(3)) > 1d-9 * max(1d0, abs(e(3)))) return
  if (c(1) < conv) return
  open(unit=77, file='XZAMPS', form='unformatted', access='stream', status='old', action='read', iostat=ios)
  if (ios == 0) read(77, iostat=ios) tag, head, ea
  close(77)
  if (ios /= 0 .or. tag /= 'XZAMPS01' .or. head(1) /= no .or. head(2) /= nv) return
  tok = .true.
  if (.not. needl .or. c(2) < conv) return
  inquire(file='XZLAMB', exist=there)
  if (.not. there) return
  open(unit=77, file='XZLAMB', form='unformatted', access='stream', status='old', action='read', iostat=ios)
  if (ios == 0) read(77, iostat=ios) tag, headl, el
  close(77)
  if (ios /= 0 .or. tag /= 'XZLAMB01' .or. any(headl /= head) .or. abs(el - ea) > 1d-9) return
  lok = .true.
end subroutine xzkeychk

! ---------------------------------------------------------------------------
! The classes of the spinors and everything that follows from them without
! the work array: the order of the spinors, the groups of pairs and triples,
! the sizes of the blocks of T3 and of <vv||vv>.  uselab = .false.: one class.
subroutine xzsetmode(uselab)
  use xzdata
  implicit none
  logical, intent(in) :: uselab
  integer(8) :: p, i, j, k, g, n, d, sa, si, m, h, t, step, a, b, r

  ncls = 1
  mgrp = 0
  if (uselab) then
    ! (with a modulus: the labels as residues 0 .. modulus - 1)
    if (modulus > 0) lab = modulo(lab, modulus)
    call xzclasses(no, nv, lab(nfrozen), maxcls, ncls, operm, vperm, ocls, vcls, ocoff, vcoff)
    if (ncls == 0) then
      write(6, '(a,i4,a)') ' @XZCCSDT-W, more than ', maxcls, ' classes of labels: the labels are not used.'
      ncls = 1
    else if (modulus > 0 .and. ncls > 1) then
      ! the step of the classes (as in xzclasses): the sums of classes are
      ! taken modulo modulus / step
      step = 0
      do p = nfrozen, nfrozen + no + nv - 1
        a = lab(p) - minval(lab(nfrozen:nfrozen + no + nv - 1))
        b = step
        do while (b /= 0)
          r = mod(a, b)
          a = b
          b = r
        end do
        step = a
      end do
      if (step == 0) step = 1
      if (modulo(modulus, step) /= 0) then
        write(6, '(a,i6,a,i6,a)') ' @XZCCSDT-W, the labels have the step ', step, ', which does not divide the modulus ', &
          modulus, ': the labels are not used.'
        ncls = 1
      else
        mgrp = modulus / step
      end if
    end if
  end if
  if (ncls == 1) mgrp = 0
  if (ncls == 1) then
    do p = 0, no - 1
      operm(p) = p
    end do
    do p = 0, nv - 1
      vperm(p) = p
    end do
    ocls = 0
    vcls = 0
    ocoff = 0
    vcoff = 0
    ocoff(1) = no
    vcoff(1) = nv
  end if
  if (mgrp > 0) then
    ! groups, differences and the sums of (Q) are residues modulo mgrp
    npg = mgrp
    ntg = mgrp
    dlo = 0
    dhi = mgrp - 1
    q4hi = mgrp - 1
    q2hi = mgrp - 1
  else
    npg = 2 * ncls - 1
    ntg = 3 * ncls - 2
    dlo = 1 - ncls
    dhi = ncls - 1
    q4hi = 4 * (ncls - 1)
    q2hi = 2 * (ncls - 1)
  end if
  ! the sizes of the groups
  opoff = 0
  vpoff = 0
  otoff = 0
  vtoff = 0
  do i = 0, no - 1
    do j = i + 1, no - 1
      g = xzw(ocls(i) + ocls(j))
      opoff(g + 1) = opoff(g + 1) + 1
      do k = j + 1, no - 1
        t = xzw(g + ocls(k))
        otoff(t + 1) = otoff(t + 1) + 1
      end do
    end do
  end do
  do i = 0, nv - 1
    do j = i + 1, nv - 1
      g = xzw(vcls(i) + vcls(j))
      vpoff(g + 1) = vpoff(g + 1) + 1
      do k = j + 1, nv - 1
        t = xzw(g + vcls(k))
        vtoff(t + 1) = vtoff(t + 1) + 1
      end do
    end do
  end do
  do g = 0, npg - 1
    opoff(g + 1) = opoff(g + 1) + opoff(g)
    vpoff(g + 1) = vpoff(g + 1) + vpoff(g)
  end do
  do g = 0, ntg - 1
    otoff(g + 1) = otoff(g + 1) + otoff(g)
    vtoff(g + 1) = vtoff(g + 1) + vtoff(g)
  end do
  n3 = 0
  do g = 0, ntg - 1
    boff(g) = n3
    n3 = n3 + (otoff(g + 1) - otoff(g)) * (vtoff(g + 1) - vtoff(g))
  end do
  boff(ntg) = n3
  nvv = 0
  nqmax = 0
  do g = 0, npg - 1
    n = vpoff(g + 1) - vpoff(g)
    vvoff(g) = nvv
    nvv = nvv + n * n
    nqmax = max(nqmax, n)
  end do
  vvoff(npg) = nvv
  ! the composite indices of pass 2
  ndel = 0
  wdoff = 0
  nwd = 0
  do d = dlo, dhi
    do sa = 0, ncls - 1
      si = xzw(sa - d)
      if (si < 0 .or. si >= ncls) cycle
      ndel(d) = ndel(d) + (vcoff(sa + 1) - vcoff(sa)) * (ocoff(si + 1) - ocoff(si))
    end do
    wdoff(d) = nwd
    nwd = nwd + ndel(d) * ndel(d)
  end do
  bdoff = -1
  nbd = 0
  do g = 0, npg - 1
    do si = 0, ncls - 1
      sa = xzw(g - si)
      if (sa < 0 .or. sa >= ncls) cycle
      n = (vcoff(sa + 1) - vcoff(sa) + ocoff(sa + 1) - ocoff(sa)) * (ocoff(si + 1) - ocoff(si)) &
          * (vpoff(g + 1) - vpoff(g))
      if (n == 0) cycle
      bdoff(g, si) = nbd
      nbd = nbd + n
    end do
  end do
  ! the sectors of (Q): M the total class of an occupied quadruple, k the
  ! class of an occupied pair
  nvcmax = 0
  nocmax = 0
  do sa = 0, ncls - 1
    nvcmax = max(nvcmax, vcoff(sa + 1) - vcoff(sa))
    nocmax = max(nocmax, ocoff(sa + 1) - ocoff(sa))
  end do
  natmax = 0
  do g = 0, ntg - 1
    natmax = max(natmax, vtoff(g + 1) - vtoff(g))
  end do
  qn22max = 0
  qn13max = 0
  do m = 0, q4hi
    n = 0
    do g = 0, npg - 1
      h = xzw(m - g)
      if (h < 0 .or. h >= npg) cycle
      n = n + (vpoff(g + 1) - vpoff(g)) * (vpoff(h + 1) - vpoff(h))
    end do
    qn22max = max(qn22max, n)
    n = 0
    do sa = 0, ncls - 1
      h = xzw(m - sa)
      if (h < 0 .or. h >= ntg) cycle
      n = n + (vcoff(sa + 1) - vcoff(sa)) * (vtoff(h + 1) - vtoff(h))
    end do
    qn13max = max(qn13max, n)
  end do
  qxvmax = 0
  qm3max = 0
  qwbmax = 0
  do k = 0, q2hi
    n = 0
    do sa = 0, ncls - 1
      t = xzw(sa + k)
      if (t >= ntg) cycle
      n = n + (vcoff(sa + 1) - vcoff(sa)) * (vtoff(t + 1) - vtoff(t))
      ! the products of this block: pairs (bc) of group g, f of class g - sa, d of class k - (g - sa)
      d = 0
      do g = 0, npg - 1
        si = xzw(g - sa)
        if (si < 0 .or. si >= ncls) cycle
        h = xzw(k - si)
        if (h < 0 .or. h >= ncls) cycle
        d = d + (vpoff(g + 1) - vpoff(g)) * (vcoff(sa + 1) - vcoff(sa)) * (vcoff(h + 1) - vcoff(h))
        qwbmax = max(qwbmax, (vpoff(g + 1) - vpoff(g)) * (vcoff(sa + 1) - vcoff(sa)) &
                             * (vcoff(si + 1) - vcoff(si)))
      end do
      qm3max = max(qm3max, d)
    end do
    qxvmax = max(qxvmax, n)
  end do
end subroutine xzsetmode

! ---------------------------------------------------------------------------
! The arrays that live for the whole run: tables, spinor energies, integrals.
! In a dry run only their places.  status on return (not in a dry run):
!   0  the integrals are there
!   1  an integral that the labels forbid is not zero: read again without them
!   2  real arithmetic was assumed (nwe = 1) and an integral has an imaginary
!      part: read again as complex numbers
subroutine xzlayout(w, status)
  use xzdata
  use xzmem
  implicit none
  real(8) :: w(*)
  integer(8), intent(out) :: status
  integer(8) :: m, mint, kbuf, krow, ksrc, ksg, po, pv, nbuf, k, jold(7), nel(7), ktn
  real(8) :: vmax, offmax, ximag
  real(8), external :: xzimag
  real(8) :: tol_label, tol_imag
  character(len=8) :: what
  character(len=64) :: targ
  integer :: ltarg, ios
  logical, save :: told = .false.
  character(len=24), parameter :: names(7) = [character(len=24) :: '<oo||oo>', '<oo||ov>', '<oo||vv>', &
    '<ov||vo>', '<oo||vv> on a<b', '<oo||vv> on i<j, a<b', '<ov||vv> on a<b']

  status = 0
  ! the largest integral that the labels forbid, and the largest imaginary part, that are
  ! taken as zero: 1e-8 and 1e-10; XZ_LABTOL and XZ_IMAGTOL give other values (a basis
  ! with very large exponents leaves more than 1e-8 in the integrals of CFOUR's chain)
  tol_label = 1d-5
  tol_imag = 1d-5
  call get_environment_variable('XZ_LABTOL', targ, ltarg)
  if (ltarg > 0) then
    read(targ(1:ltarg), *, iostat=ios) tol_label
    if (ios /= 0 .or. tol_label <= 0d0) tol_label = 1d-8
  end if
  call get_environment_variable('XZ_IMAGTOL', targ, ltarg)
  if (ltarg > 0) then
    read(targ(1:ltarg), *, iostat=ios) tol_imag
    if (ios /= 0 .or. tol_imag <= 0d0) tol_imag = 1d-10
  end if
  if ((tol_label /= 1d-8 .or. tol_imag /= 1d-10) .and. .not. xzdry .and. .not. told) write(6, '(a,es9.2,a,es9.2,a)') &
    '   tolerances of the integrals: ', tol_label, ' for what the labels forbid (XZ_LABTOL), ', tol_imag, &
    ' for imaginary parts (XZ_IMAGTOL)'
  if (.not. xzdry) told = .true.
  po = no * (no + 1) / 2
  pv = nv * (nv + 1) / 2
  jotri = xzget(no * npo, 'occupied tables')
  josgn = xzget(no * npo / 8 + 1, 'occupied tables')
  jofree = xzget(3 * nto, 'occupied tables')
  jopair = xzget(3 * nto, 'occupied tables')
  jopa = xzget(npo, 'occupied tables')
  jopb = xzget(npo, 'occupied tables')
  jopidx = xzget(no * no, 'occupied tables')
  jopgrp = xzget(npo, 'occupied tables')
  jvtri = xzget(nv * nvp, 'virtual tables')
  jvsgn = xzget(nv * nvp / 8 + 1, 'virtual tables')
  jvfree = xzget(3 * nvt, 'virtual tables')
  jvpair = xzget(3 * nvt, 'virtual tables')
  jvpa = xzget(nvp, 'virtual tables')
  jvpb = xzget(nvp, 'virtual tables')
  jvpidx = xzget(nv * nv, 'virtual tables')
  jvpgrp = xzget(nvp, 'virtual tables')
  jrows = xzget(max(nto, npo, nv, no, natmax), 'rows')
  jeps = xzget(ny, 'spinor energies')
  jeo3 = xzget(nto, 'spinor energies')
  jev3 = xzget(nvt, 'spinor energies')
  if (ncls > 1) then
    jobase = xzget(nto, 'layout of T3')
    jvcol = xzget(nvt, 'layout of T3')
    josig = xzget(nto, 'layout of T3')
    jvsig = xzget(nvt, 'layout of T3')
    jvci = xzget(nv * nvp / 2 + 1, 'layout of T3')
    jrowpos = xzget(nv * no, 'layout of T3')
  end if
  ! (the passes in blocks: all shifts zero, the layout of T3)
  call xzshctx(0_8, 0_8, 0_8)
  m = xzmark()
  ktn = xzget(max(nto, nvt), 'work of the tables')
  if (.not. xzdry) then
    call xzmaps(no, npo, nto, ncls, ocls, w(jotri), w(josgn), w(jofree), w(jopair), w(jopa), w(jopb), &
                w(jopidx), w(jopgrp), opoff, otoff, w(ktn))
    call xzmaps(nv, nvp, nvt, ncls, vcls, w(jvtri), w(jvsgn), w(jvfree), w(jvpair), w(jvpa), w(jvpb), &
                w(jvpidx), w(jvpgrp), vpoff, vtoff, w(ktn))
    call xziota(max(nto, npo, nv, no, natmax), w(jrows))
    if (ncls > 1) call xzt3lay(no, nv, npo, nvp, nto, nvt, ncls, ocls, vcls, otoff, vtoff, w(jvtri), &
                               w(jvsgn), ocoff, vcoff, boff, n3, w(jobase), w(jvcol), w(josig), &
                               w(jvsig), w(jvci), w(jrowpos))
  end if
  call xzrel(m)

  ! ---- the small blocks and <ov||vv>, read as complex numbers
  mint = xzmark()
  nel = [no ** 4, no ** 3 * nv, no * no * nv * nv, no * no * nv * nv, no * no * nvp, npo * nvp, &
         no * nv * nvp]
  do k = 1, 6
    jold(k) = xzget(2 * nel(k), names(k))
  end do
  m = xzmark()
  nbuf = max(2 * po * po, 2 * po * no * nv, 2 * no * no * nv * nv, 2 * ntot)
  kbuf = xzget(nbuf, 'record of an integral file')
  if (.not. xzdry) then
    call xzrdeps(w(jeps), w(kbuf))
    call xzsum3(no, nto, w(jofree), w(jeps), w(jeo3))
    call xzsum3(nv, nvt, w(jvfree), w(jeps + no), w(jev3))
    call xzrdsmall(w(jold(1)), w(jold(2)), w(jold(3)), w(jold(4)), w(jold(5)), w(jold(6)), w(jvpa), &
                   w(jvpb), w(jopa), w(jopb), w(kbuf))
  end if
  call xzrel(m)
  jold(7) = xzget(2 * nel(7), names(7))
  m = xzmark()
  kbuf = xzget(2 * pv * nv, 'record of a file PPPH')
  if (.not. xzdry) then
    call xzrdovvq(w(jold(7)), w(jvpa), w(jvpb), w(kbuf))
    if (ncls > 1) then
      call xzchklab(w(jold(1)), w(jold(2)), w(jold(3)), w(jold(4)), w(jold(7)), w(jvpgrp), &
                    vmax, what)
      if (vmax > tol_label) then
        write(6, '(/,a)') ' @XZCCSDT-W, the labels of ' // trim(labfile) // ' are not symmetry labels of these spinors:'
        write(6, '(a,a,a,es10.2,a)') '             an element of ', trim(what), ' that they forbid is ', &
          vmax, '.  The labels are not used.'
        write(6, '(a)') '             (Spinors of an SCF without the symmetry mix the labels: a pair of'
        write(6, '(a)') '             Kramers partners has to be rotated into the symmetry first.)'
        status = 1
        return
      end if
    end if
    if (nwe == 1) then
      ximag = 0d0
      do k = 1, 7
        ximag = max(ximag, xzimag(nel(k), w(jold(k))))
      end do
      if (ximag > tol_imag) then
        write(6, '(a,es10.2,a,es9.2,a)') '   the largest imaginary part of the integrals: ', ximag, &
          ' (above the tolerance ', tol_imag, ', XZ_IMAGTOL: complex arithmetic)'
        status = 2
        return
      end if
      if (ximag > 1d-12) write(6, '(a,es10.2,a,es9.2,a)') '   the largest imaginary part of the integrals: ', &
        ximag, ' (below the tolerance ', tol_imag, ': taken as zero)'
    end if
  end if
  call xzrel(m)
  if (nwe == 1) then
    ! the real parts, moved together
    call xzrel(mint)
    do k = 1, 7
      m = xzget(nel(k), names(k))
      if (.not. xzdry) call xztoreal(nel(k), w(jold(k)), w(m))
      jold(k) = m
    end do
  end if
  joooo = jold(1)
  jooov = jold(2)
  joovv = jold(3)
  jovvo = jold(4)
  joovvp = jold(5)
  jgpp = jold(6)
  jovvq = jold(7)

  ! ---- <ab||cd>: the record of the file is read where the blocks stay
  m = xzmark()
  jvvvv = xzget(2 * pv * pv, 'record of moinpppp')
  krow = xzget(2 * nvp, 'row of <vv||vv>')
  ksrc = xzget(2 * nvp, 'permutation of the pairs')
  ksg = xzget(nvp, 'signs of the pairs')
  if (.not. xzdry) then
    call xzrdvvvv(w(jvvvv), w(jvpa), w(jvpb), w(krow), w(ksrc), w(ksg), offmax)
    if (offmax > tol_label) then
      write(6, '(/,a)') ' @XZCCSDT-W, the labels of ' // trim(labfile) // ' are not symmetry labels of these spinors:'
      write(6, '(a,es10.2,a)') '             an element of <vv||vv> that they forbid is ', offmax, &
        '.  The labels are not used.'
      status = 1
      return
    end if
    if (offmax > 1d-8) write(6, '(a,es10.2,a,es9.2,a)') '   the largest element of <vv||vv> that the labels forbid: ', &
      offmax, ' (below the tolerance ', tol_label, ': taken as zero)'
    if (nwe == 1) then
      ximag = xzimag(nvv, w(jvvvv))
      if (ximag > tol_imag) then
        write(6, '(a,es10.2,a,es9.2,a)') '   the largest imaginary part of <vv||vv>: ', ximag, &
          ' (above the tolerance ', tol_imag, ', XZ_IMAGTOL: complex arithmetic)'
        status = 2
        return
      end if
      if (ximag > 1d-12) write(6, '(a,es10.2,a,es9.2,a)') '   the largest imaginary part of <vv||vv>: ', &
        ximag, ' (below the tolerance ', tol_imag, ': taken as zero)'
      call xztoreal(nvv, w(jvvvv), w(jvvvv))
    end if
  end if
  call xzrel(m)
  jvvvv = xzget(nwe * nvv, '<vv||vv> on a<b, c<d')
end subroutine xzlayout

! ---------------------------------------------------------------------------
! The wall and CPU times of the stages of set `set' (xzdata), of ncyc cycles
! that took ttot seconds: per stage the wall time, its share, the CPU time of
! all threads and their ratio, the cores that were busy.
subroutine xztable(set, title, ttot, ncyc)
  use xzdata
  implicit none
  integer, intent(in) :: set
  character(len=*), intent(in) :: title
  real(8), intent(in) :: ttot
  integer(8), intent(in) :: ncyc
  integer :: k
  real(8) :: tsum, csum
  if (ncyc > 0) then
    write(6, '(/,a,a,a,i4,a)') '   Times of ', title, ' (', ncyc, ' cycles): wall, share, CPU of all threads, cores busy'
  else
    write(6, '(/,a,a,a)') '   Times of ', title, ': wall, share, CPU of all threads, cores busy'
  end if
  tsum = 0d0
  csum = 0d0
  do k = 1, ntim(set)
    if (tpart(k, set)) then
      write(6, '(5x,a28,f12.2,a,f6.1,a)') tnam(k, set), twal(k, set), ' s', 100d0 * twal(k, set) / max(ttot, 1d-9), &
        ' %   (inside a kernel)'
    else
      write(6, '(5x,a28,f12.2,a,f6.1,a,f12.2,a,f6.1)') tnam(k, set), twal(k, set), ' s', &
        100d0 * twal(k, set) / max(ttot, 1d-9), ' %', tcpu(k, set), ' s', tcpu(k, set) / max(twal(k, set), 1d-9)
      tsum = tsum + twal(k, set)
      csum = csum + tcpu(k, set)
    end if
  end do
  write(6, '(5x,a28,f12.2,a)') 'other', ttot - tsum, ' s'
  if (ncyc > 0) then
    write(6, '(5x,a28,f12.2,a,f10.2,a,f6.1)') 'total', ttot, ' s;  per cycle ', ttot / ncyc, &
      ' s;  cores busy in the stages ', csum / max(tsum, 1d-9)
  else
    write(6, '(5x,a28,f12.2,a,f6.1)') 'total', ttot, ' s;  cores busy in the stages ', csum / max(tsum, 1d-9)
  end if
end subroutine xztable

! n words to the file `name' (comparisons with the Python program)
subroutine xzdumpw(name, n, x)
  implicit none
  character(len=*), intent(in) :: name
  integer(8), intent(in) :: n
  real(8), intent(in) :: x(n)
  open(unit=78, file=name, form='unformatted', access='stream', status='replace')
  write(78) x
  close(78)
end subroutine xzdumpw

! one record of n complex numbers to the sequential unformatted file `name'
! (n = 0: a record without elements), as CFOUR's programs read it
subroutine xzwrec(name, n, rec)
  implicit none
  character(len=*), intent(in) :: name
  integer(8), intent(in) :: n
  complex(8), intent(in) :: rec(*)
  integer :: ios
  open(unit=79, file=name, form='unformatted', access='sequential', status='replace', iostat=ios)
  if (ios == 0) then
    if (n > 0) then
      write(79, iostat=ios) rec(1:n)
    else
      write(79, iostat=ios)
    end if
  end if
  if (ios == 0) close(79, iostat=ios)
  if (ios /= 0) then
    write(6, '(/,a,a,a,i8,a)') ' @XZCCSDT-F, writing the file ', name, ' failed (iostat ', ios, ').'
    call xzerrex
  end if
end subroutine xzwrec

! ---------------------------------------------------------------------------
! The cost model of the tasks of (Q) (xzpar.F90), in units of one virtual
! quadruple of a sector:
!   qc(M)  an occupied quadruple of total class M: the virtual quadruples
!          a<b<c<d of total class M (one class: all of them)
!   xc(K)  Xv for a pair (x, l) of total class K: gxv times the elements of
!          its products (<bc||ef> on (bc), e times t2 on f, d)
! Measured (docs/native.md, section 14): with qc the times of the groups
! that hold nine tenths of (Q) are met within 5 % (6 threads) to 13 % (24
! threads) in blocks (HBr, 26 + 78), where one cost for every quadruple is
! off by up to 21 %; gxv is 0.0017 to 0.0025 there and 0.0005 for dense
! real T3 - Xv is at most 2 % of the work, so its constant matters little.
subroutine xzqmodel(qc, xc)
  use xzdata
  implicit none
  real(8), intent(out) :: qc(0:4 * maxcls), xc(0:2 * maxcls)
  real(8), parameter :: gxv = 0.0012d0
  real(8) :: ways(0:4, 0:4 * maxcls), new(0:4, 0:4 * maxcls), bin(0:4)
  integer(8) :: c, n, t, m, s, k, ke, q2, kf, kd, tt
  ways = 0d0
  ways(0, 0) = 1d0
  do c = 0, ncls - 1
    n = vcoff(c + 1) - vcoff(c)
    bin(0) = 1d0
    do m = 1, 4
      bin(m) = bin(m - 1) * dble(n - m + 1) / dble(m)
      if (m > n) bin(m) = 0d0
    end do
    new = 0d0
    do t = 0, 4
      do s = 0, 4 * (ncls - 1)
        if (ways(t, s) == 0d0) cycle
        do m = 0, 4 - t
          if (s + c * m > 4 * (ncls - 1)) cycle
          new(t + m, s + c * m) = new(t + m, s + c * m) + ways(t, s) * bin(m)
        end do
      end do
    end do
    ways = new
  end do
  qc = 0d0
  qc(0:4 * (ncls - 1)) = ways(4, 0:4 * (ncls - 1))
  xc = 0d0
  do k = 0, 2 * (ncls - 1)
    do ke = 0, ncls - 1
      tt = ke + k
      if (tt >= ntg) cycle
      if (vtoff(tt + 1) == vtoff(tt)) cycle
      do q2 = 0, npg - 1
        kf = q2 - ke
        kd = k - kf
        if (kf < 0 .or. kf >= ncls .or. kd < 0 .or. kd >= ncls) cycle
        xc(k) = xc(k) + gxv * dble(vpoff(q2 + 1) - vpoff(q2)) * dble(vcoff(ke + 1) - vcoff(ke)) &
                * dble(vcoff(kd + 1) - vcoff(kd)) * dble(vcoff(kf + 1) - vcoff(kf))
      end do
    end do
  end do
end subroutine xzqmodel

! ---------------------------------------------------------------------------
! The words of the work array of a rank other than 0 (called before it is
! allocated): what (Q) takes with the plan of rank 0, which that rank sends
! when it has read the integrals and made its plan.  A job without (Q) on
! these ranks: nothing.
subroutine xzrankmem(maxcor)
  use xzdata
  use xzpar
  use xzdist
  implicit none
  integer(8), intent(inout) :: maxcor
  ! (the broadcasts in the order of rank 0: the sizes, the decision on the
  ! files of an earlier run, then its plan)
  call xzjob()
  call xzsizes()
  sizes_set = .true.
  call xzreuse()
  if (.not. (with_q .or. with_cc .or. with_lam)) then
    maxcor = 4096
    return
  end if
  rplan = 0d0
  call xzmpi_bcast(rplan, 128_8)
  maxcor = min(maxcor, int(rplan(1), 8) + 4096)
end subroutine xzrankmem

! ---------------------------------------------------------------------------
! The start of the ranks (called by every rank after the flags of joda went
! round): the largest number of threads of a rank - the memory plan of a
! rank has buffers per thread, and rank 0 plans for all -, and the test of
! the chunked transfers (XZ_MPITEST=n: arrays of n words through every
! routine, with the chunk of XZ_MPICHUNK).
subroutine xzmpistart()
  use xzpar
  use xzdist
  use xzkern
  implicit none
  real(8) :: a(2), b(2)
  integer(8) :: nthr, ilp64, n
  character(len=32) :: arg
  integer :: larg
  call xz_blas_info(nthr, ilp64)
  a = [dble(nthr), -dble(nthr)]
  b = a
  call xzmpi_max(a, b, 2_8)
  call xzmpi_bcast(b, 16_8)
  maxthr = nint(b(1), 8)
  call xznodes()
  if (myrank == 0 .and. b(1) /= -b(2)) write(6, '(a,i4,a,i4,a)') ' @XZCCSDT-W, the ranks have between ', &
    nint(-b(2)), ' and ', nint(b(1)), ' threads: the memory of every rank is planned for the largest number.'
  call get_environment_variable('XZ_MPITEST', arg, larg)
  if (larg > 0) then
    read(arg(1:larg), *) n
    a = [dble(xzmpi_selftest(n)), 0d0]
    b = a
    call xzmpi_max(a, b, 1_8)
    if (myrank == 0) write(6, '(a,i12,a,i10,a,i4,a,i12)') ' XZ_MPITEST: arrays of ', n, ' words in messages of ', &
      xzmpi_chunk(), ' words over ', nrank, ' ranks; wrong words (largest count of a rank): ', nint(b(1), 8)
    if (myrank == 0 .and. b(1) /= 0d0) call xzerrex
  end if
end subroutine xzmpistart

! ---------------------------------------------------------------------------
! T3 (or Lambda3) of the file `name' where the file is mapped, read-only:
! j3 = the index in w of its first word.  The file holds the header (56
! bytes), nskip words (T1, T2) and n3w words of T3, stored as the program
! uses them (xzamps, xzlamb), so nothing is converted.  The pages are those
! of the page cache: one copy for all the ranks of a node.
subroutine xzmapt3(w, name, nskip, n3w, j3)
  use xzkern
  implicit none
  real(8) :: w(*)
  character(len=*), intent(in) :: name
  integer(8), intent(in) :: nskip, n3w
  integer(8), intent(out) :: j3
  integer(8) :: addr, nbytes, diff
  addr = xz_map_file(name // c_null_char, nbytes, 1_8)
  if (addr == 0 .or. nbytes /= 56 + 8 * (nskip + n3w)) then
    write(6, '(/,a,a,a)') ' @XZCCSDT-F, the file ', name, ' could not be mapped (or has another length than its header says).'
    write(6, '(a,i16,a,i16)') '   bytes mapped: ', nbytes, '   expected: ', 56 + 8 * (nskip + n3w)
    write(6, '(a)') '   XZ_QMAP=0 reads T3 into the work array of every rank instead.'
    call xzerrex
  end if
  diff = addr + 56 + 8 * nskip - loc(w(1))
  if (mod(diff, 8_8) /= 0) then
    write(6, '(/,a,a,a)') ' @XZCCSDT-F, the map of the file ', name, ' is not aligned with the work array; use XZ_QMAP=0.'
    call xzerrex
  end if
  j3 = diff / 8 + 1
end subroutine xzmapt3

! ---------------------------------------------------------------------------
! The exit after an error (every error of the program comes here, its message
! written).  Rank 0: CFOUR's ERREX (status 1, the ERROR banner).  A rank
! other than 0 must not call ERREX - it closes JOBARC as if this rank had
! opened it and writes an empty index (JAINDX) over that of the run
! directory -: it stops all ranks through MPI.
subroutine xzerrex()
  use xzpar
  implicit none
  flush(6)
  if (myrank == 0) then
    call errex
  else
    write(6, '(a,i5,a)') ' @XZCCSDT-F, rank ', myrank, ' stops the run.'
    flush(6)
    call xzmpi_abort(1_8)
  end if
  stop 1
end subroutine xzerrex
