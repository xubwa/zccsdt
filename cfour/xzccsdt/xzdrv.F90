!
! The driver of xzccsdt: the job, the sizes, the symmetry labels, the memory
! plan, the integrals, the CCSDT iterations, (Q), the energies.
!
! Two things are decided from the integrals that are read:
!   blocks   if the file xzlabels is there, its labels are used - the spinors
!            in the order of their classes, T3 and <ab||cd> on the allowed
!            blocks only - provided no integral that they forbid is larger
!            than 1e-8; otherwise the labels are dropped with a warning (one
!            class: the dense case of the same code)
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
  use xzpland, only: planq
  use omp_lib
  implicit none
  integer(8), intent(in) :: maxcor
  real(8) :: w(maxcor)
  integer(8) :: iintln, ifltln, iintfp, ialone, ibitwd
  common /machsp/ iintln, ifltln, iintfp, ialone, ibitwd
  integer(8) :: nslot, nch, ncyc, nthr, ilp64, kb(2), nmax, status
  logical :: usefile, converged, uselab, tryreal, isreal, there, docc
  real(8) :: sync(2)
  real(8) :: ecorr, eq5, eq6, etot, t0, tread, tcc, tq, twall, tlm, tden, lhs, rhs
  character(len=32) :: arg
  integer :: larg

  twall = omp_get_wtime()
  call xzinit(loc(w(1)), maxcor)
  ! (a test hook: XZ_MAXCOR words of the work array are used)
  call get_environment_variable('XZ_MAXCOR', arg, larg)
  if (larg > 0) then
    read(arg, *) nmax
    xzsize = min(xzsize, nmax)
  end if
  call xzjob()
  call xzsizes()
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
  if (xzmpi_enabled() == 1) write(6, '(a,i5,a,i5,a)') '   MPI: rank ', myrank, ' of ', nrank, &
    ' ((Q) is parallel over the ranks; each has a work array of this size)'
  if (ilp64 /= 1) then
    write(6, '(a)') ' @XZCCSDT-F, the kernels were not compiled for the 8-byte integers of this BLAS.'
    call errex
  end if
  if (no < 3 .or. nv < 3) then
    write(6, '(a)') ' @XZCCSDT-F, fewer than three occupied or virtual spinors: no triples.'
    call errex
  end if

  ! ---- the symmetry labels
  allocate(lab(0:ntot-1), operm(0:no-1), vperm(0:nv-1), ocls(0:no-1), vcls(0:nv-1))
  call xzrdlab(ntot, lab, modulus, there)
  have_labels = there
  uselab = there
  if (there) then
    write(6, '(a,i4)') '   symmetry labels: the file xzlabels, modulus ', modulus
    if (modulus /= 0) then
      write(6, '(a)') ' @XZCCSDT-W, labels with a modulus are not used by this program: one block.'
      uselab = .false.
    end if
  else
    write(6, '(a)') '   symmetry labels: none (no file xzlabels)'
  end if
  call get_environment_variable('XZ_LABELS', arg, larg)
  if (larg > 0 .and. arg(1:1) == '0' .and. uselab) then
    write(6, '(a)') '   XZ_LABELS=0: the labels are not used'
    uselab = .false.
  end if
  tryreal = .true.
  call get_environment_variable('XZ_REAL', arg, larg)
  if (larg > 0 .and. arg(1:1) == '0') then
    write(6, '(a)') '   XZ_REAL=0: complex arithmetic whatever the integrals'
    tryreal = .false.
  end if

  ! ---- which ranks do what: the iterations (CCSDT, Lambda) are those of
  !      rank 0; the other ranks wait for them here and then read the
  !      integrals, the amplitudes (file XZAMPS) and lambda (XZLAMB) themselves
  docc = with_cc
  planq = .not. (with_cc .or. with_lam .or. with_den) .or. myrank > 0
  if (with_den .and. myrank > 0) return
  ! (a test hook: XZ_DEN_ACTIVE=1 writes the files of the active spinors alone)
  call get_environment_variable('XZ_DEN_ACTIVE', arg, larg)
  den_active = larg > 0
  if (method == 'MP2' .and. myrank > 0) return
  sync = 0d0
  if ((with_cc .or. with_lam) .and. myrank > 0) then
    call xzmpi_bcast(sync, 16_8)
    if (.not. with_q .or. sync(2) /= 1d0) return
  end if

  ! ---- the integrals: for labels and real numbers first; the memory of a
  !      case is checked before it is read (the floor of its plan)
  isreal = tryreal
  t0 = omp_get_wtime()
  do
    call xzsetmode(uselab)
    nwe = merge(1_8, 2_8, isreal)
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

  ! ---- CCSDT (rank 0), or the amplitudes of an earlier run
  tcc = 0d0
  tlm = 0d0
  if (docc .and. myrank == 0) then
    t0 = omp_get_wtime()
    if (nwe == 2) then
      call xzcc_z(w, nslot, usefile, nch, ecorr, converged, ncyc)
    else
      call xzcc_d(w, nslot, usefile, nch, ecorr, converged, ncyc)
    end if
    tcc = omp_get_wtime() - t0
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
      call errex
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
    write(6, '(/,a,f10.1,a,f8.2,a)') '   amplitudes written to the file XZAMPS (', &
      8d-6 * nwe * (no * nv + no * no * nv * nv + n3), ' MB, ', omp_get_wtime() - t0, ' s)'
  else
    if (nwe == 2) then
      call xzamps_z(w, 'r', ecorr)
    else
      call xzamps_d(w, 'r', ecorr)
    end if
    write(6, '(/,a,f20.12)') '   amplitudes read from the file XZAMPS; CCSDT correlation energy ', ecorr
  end if
  flush(6)

  ! ---- the Lambda equations (rank 0), or lambda of an earlier run
  if (with_lam .and. myrank == 0) then
    t0 = omp_get_wtime()
    if (nwe == 2) then
      call xzlam_z(w, nslot, usefile, nch, converged, ncyc)
    else
      call xzlam_d(w, nslot, usefile, nch, converged, ncyc)
    end if
    tlm = omp_get_wtime() - t0
    if (.not. converged .and. timing_run) then
      write(6, '(/,a,i5,a)') ' @XZCCSDT-W, XZ_NCYC: the Lambda equations stopped after ', maxcyc, &
        ' cycles; lambda and what follows are NOT converged.'
      converged = .true.
    end if
    if (.not. converged) then
      write(6, '(/,a,i5,a)') ' @XZCCSDT-F, the Lambda equations of CCSDT did not converge in ', maxcyc, &
        ' cycles (CC_MAXCYC).'
      call xzmpi_bcast(sync, 16_8)
      call errex
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
    write(6, '(/,a,f10.1,a,f8.2,a)') '   lambda written to the file XZLAMB (', &
      8d-6 * nwe * (no * nv + no * no * nv * nv + n3), ' MB, ', omp_get_wtime() - t0, ' s)'
    if (with_q .or. with_den) then
      ! the pp operator took the place of <ab||cd>: the integrals and the
      ! amplitudes are read again for what follows
      call xzrel(0_8)
      call xzlayout(w, status)
      if (status /= 0) then
        write(6, '(/,a)') ' @XZCCSDT-F, the integral files changed during the run.'
        call errex
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
  if ((with_cc .or. with_lam) .and. myrank == 0) then
    sync = [ecorr, 1d0]
    call xzmpi_bcast(sync, 16_8)
  end if
  flush(6)

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
  call xz_rss(kb)
  write(6, '(/,a,f10.2,a,f10.2,a,f10.2,a,f10.2,a)') '   wall times: integrals ', tread, ' s   CCSDT ', tcc, &
    ' s   Lambda ', tlm, ' s   (Q) ', tq, ' s'
  if (with_den) write(6, '(a,f10.2,a)') '               density and gradient ', tden, ' s'
  write(6, '(a,i14,a,f10.1,a,f8.1,a)') '   work array used: ', xzpeak, ' words = ', 8d-6 * xzpeak, &
    ' MB (', 100d0 * xzpeak / max(xzsize, 1_8), ' % of MAXCOR)'
  write(6, '(a,f10.1,a)') '   peak resident memory of the process: ', 1d-3 * kb(1), ' MB'
  write(6, '(a,f10.2,a)') '   XZCCSDT total wall time: ', omp_get_wtime() - twall, ' s'
  if (with_den) then
    write(6, '(a)') '   XZCCSDT finished: density of CCSDT done.'
  else if (method == 'LAMBDA') then
    write(6, '(a)') '   XZCCSDT finished: lambda of CCSDT done.'
  else
    write(6, '(a)') '   XZCCSDT finished: ' // trim(method) // ' energy done.'
  end if
end subroutine xzdrv

! ---------------------------------------------------------------------------
! What to do: the method (the argument of the command line, else CALC of the
! ZMAT as joda stored it), CC_CONV and CC_MAXCYC.  As in the Python front
! end: the amplitudes to 10**-CC_CONV, the energy to 10**-(CC_CONV + 2).
subroutine xzjob()
  use xzdata
  implicit none
  integer(8) :: iflags, iflags2
  common /flags/ iflags(100)
  common /flags2/ iflags2(500)
  character(len=32) :: arg
  integer :: larg, i, c
  integer(8) :: level, nexc

  call get_command_argument(1, arg, larg)
  method = ' '
  if (larg > 0) then
    do i = 1, larg
      c = iachar(arg(i:i))
      if (c >= iachar('a') .and. c <= iachar('z')) arg(i:i) = achar(c - 32)
    end do
    method = arg(1:larg)
  else
    level = iflags(2)
    nexc = iflags2(38)
    if (level == 18) method = 'CCSDT'
    if (level == 30) method = 'CCSDT[Q]'
    if (level == 54) method = 'CCSDT(Q)'
    if (level == 49 .and. nexc == 3) method = 'CCSDT'
    if (level == 53 .and. nexc == 4) method = 'CCSDT(Q)'
    if (method == ' ') then
      write(6, '(/,a,i4,a,i3,a)') ' @XZCCSDT-F, CALC of the ZMAT (level ', level, ', EXCITATION ', nexc, &
        ') is not CCSDT, CCSDT[Q] or CCSDT(Q).'
      write(6, '(a)') '   Give the method as the argument: xzccsdt ccsdt | "ccsdt(q)"'
      call errex
    end if
  end if
  select case (method)
  case ('CCSDT', 'CCSDT[Q]', 'CCSDT(Q)', '(Q)', '[Q]', 'MP2', 'LAMBDA', 'CCSDT(Q)_L', 'CCSDT[Q]_L', &
        '(Q)_L', '[Q]_L', 'DENSITY', 'CCSDT_DENSITY')
  case default
    write(6, '(/,a,a,a)') ' @XZCCSDT-F, method ', trim(method), &
      ': not one of CCSDT, CCSDT[Q], CCSDT(Q), (Q), [Q], LAMBDA, CCSDT[Q]_L, CCSDT(Q)_L, [Q]_L, (Q)_L, DENSITY, CCSDT_DENSITY, MP2.'
    call errex
  end select
  with_cc = method(1:5) == 'CCSDT'
  larg = len_trim(method)
  q_left = method(max(larg - 1, 1):larg) == '_L'
  with_lam = method == 'LAMBDA' .or. (with_cc .and. q_left)
  with_den = method == 'DENSITY' .or. method == 'CCSDT_DENSITY'
  with_lam = with_lam .or. method == 'CCSDT_DENSITY'
  with_q = method /= 'CCSDT' .and. method /= 'MP2' .and. method /= 'LAMBDA' .and. .not. with_den
  conv_t = 10d0 ** (-abs(iflags(4)))
  conv_e = 10d0 ** (-abs(iflags(4)) - 2)
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
! The classes of the spinors and everything that follows from them without
! the work array: the order of the spinors, the groups of pairs and triples,
! the sizes of the blocks of T3 and of <vv||vv>.  uselab = .false.: one class.
subroutine xzsetmode(uselab)
  use xzdata
  implicit none
  logical, intent(in) :: uselab
  integer(8) :: p, i, j, k, g, n, d, sa, si, m

  ncls = 1
  if (uselab) then
    call xzclasses(no, nv, lab(nfrozen), maxcls, ncls, operm, vperm, ocls, vcls, ocoff, vcoff)
    if (ncls == 0) then
      write(6, '(a,i4,a)') ' @XZCCSDT-W, more than ', maxcls, ' classes of labels: the labels are not used.'
      ncls = 1
    end if
  end if
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
  npg = 2 * ncls - 1
  ntg = 3 * ncls - 2
  ! the sizes of the groups
  opoff = 0
  vpoff = 0
  otoff = 0
  vtoff = 0
  do i = 0, no - 1
    do j = i + 1, no - 1
      g = ocls(i) + ocls(j)
      opoff(g + 1) = opoff(g + 1) + 1
      do k = j + 1, no - 1
        otoff(g + ocls(k) + 1) = otoff(g + ocls(k) + 1) + 1
      end do
    end do
  end do
  do i = 0, nv - 1
    do j = i + 1, nv - 1
      g = vcls(i) + vcls(j)
      vpoff(g + 1) = vpoff(g + 1) + 1
      do k = j + 1, nv - 1
        vtoff(g + vcls(k) + 1) = vtoff(g + vcls(k) + 1) + 1
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
  do d = 1 - ncls, ncls - 1
    do sa = 0, ncls - 1
      si = sa - d
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
      sa = g - si
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
  do m = 0, 4 * (ncls - 1)
    n = 0
    do g = 0, npg - 1
      if (m - g < 0 .or. m - g >= npg) cycle
      n = n + (vpoff(g + 1) - vpoff(g)) * (vpoff(m - g + 1) - vpoff(m - g))
    end do
    qn22max = max(qn22max, n)
    n = 0
    do sa = 0, ncls - 1
      if (m - sa < 0 .or. m - sa >= ntg) cycle
      n = n + (vcoff(sa + 1) - vcoff(sa)) * (vtoff(m - sa + 1) - vtoff(m - sa))
    end do
    qn13max = max(qn13max, n)
  end do
  qxvmax = 0
  qm3max = 0
  qwbmax = 0
  do k = 0, 2 * (ncls - 1)
    n = 0
    do sa = 0, ncls - 1
      if (sa + k >= ntg) cycle
      n = n + (vcoff(sa + 1) - vcoff(sa)) * (vtoff(sa + k + 1) - vtoff(sa + k))
      ! the products of this block: pairs (bc) of group g, f of class g - sa, d of class k - (g - sa)
      d = 0
      do g = 0, npg - 1
        si = g - sa
        if (si < 0 .or. si >= ncls) cycle
        if (k - si < 0 .or. k - si >= ncls) cycle
        d = d + (vpoff(g + 1) - vpoff(g)) * (vcoff(sa + 1) - vcoff(sa)) * (vcoff(k - si + 1) - vcoff(k - si))
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
  real(8), parameter :: tol_label = 1d-8, tol_imag = 1d-10
  character(len=8) :: what
  character(len=24), parameter :: names(7) = [character(len=24) :: '<oo||oo>', '<oo||ov>', '<oo||vv>', &
    '<ov||vo>', '<oo||vv> on a<b', '<oo||vv> on i<j, a<b', '<ov||vv> on a<b']

  status = 0
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
        write(6, '(/,a)') ' @XZCCSDT-W, the labels of xzlabels are not symmetry labels of these spinors:'
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
        status = 2
        return
      end if
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
      write(6, '(/,a)') ' @XZCCSDT-W, the labels of xzlabels are not symmetry labels of these spinors:'
      write(6, '(a,es10.2,a)') '             an element of <vv||vv> that they forbid is ', offmax, &
        '.  The labels are not used.'
      status = 1
      return
    end if
    if (nwe == 1) then
      if (xzimag(nvv, w(jvvvv)) > tol_imag) then
        status = 2
        return
      end if
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
    call errex
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
! that hold nine tenths of (Q) are met within 5 % in blocks (HBr, 26 + 78),
! where one cost for every quadruple is off by up to 21 %; gxv is 0.0017
! there and 0.00075 for dense complex H2O (10 + 106) - Xv is a few per cent
! of the work, so its constant matters little.
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
