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
  real(8) :: ecorr, eq5, eq6, etot, t0, tread, tcc, tq, twall
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
  if (with_q .and. uselab) then
    write(6, '(a)') ' @XZCCSDT-W, (Q) in the blocks of the labels is not there yet: the labels are not used.'
    uselab = .false.
  end if
  tryreal = .true.
  call get_environment_variable('XZ_REAL', arg, larg)
  if (larg > 0 .and. arg(1:1) == '0') then
    write(6, '(a)') '   XZ_REAL=0: complex arithmetic whatever the integrals'
    tryreal = .false.
  end if

  ! ---- which ranks do what: the iterations are those of rank 0; the other
  !      ranks wait for them here and then read the integrals and the
  !      amplitudes (file XZAMPS) themselves
  docc = method(1:5) == 'CCSDT'
  planq = .not. docc .or. myrank > 0
  sync = 0d0
  if (docc .and. myrank > 0) then
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
  write(6, '(a,f10.2,a)') '   integrals read and sorted: ', tread, ' s'
  flush(6)

  ! ---- the memory plan
  call xzplan(w, nslot, usefile, nch, .false.)

  ! ---- CCSDT (rank 0), or the amplitudes of an earlier run
  tcc = 0d0
  if (docc .and. myrank == 0) then
    t0 = omp_get_wtime()
    if (nwe == 2) then
      call xzcc_z(w, nslot, usefile, nch, ecorr, converged, ncyc)
    else
      call xzcc_d(w, nslot, usefile, nch, ecorr, converged, ncyc)
    end if
    tcc = omp_get_wtime() - t0
    if (.not. converged) then
      call get_environment_variable('XZ_DEBUG', length=larg)
      if (larg == 0) write(6, '(/,a,i5,a)') ' @XZCCSDT-F, CCSDT did not converge in ', maxcyc, &
        ' cycles (CC_MAXCYC).'
      call xzmpi_bcast(sync, 16_8)
      if (larg > 0) return
      call errex
    end if
    write(6, '(a,i4,a)') '   CCSDT converged in ', ncyc, ' cycles.'
    call xztimes(tcc, ncyc)
    ! the amplitudes for the steps that follow, here or in a later run
    t0 = omp_get_wtime()
    if (nwe == 2) then
      call xzamps_z(w, 'w', ecorr)
    else
      call xzamps_d(w, 'w', ecorr)
    end if
    write(6, '(/,a,f10.1,a,f8.2,a)') '   amplitudes written to the file XZAMPS (', &
      8d-6 * nwe * (no * nv + no * no * nv * nv + n3), ' MB, ', omp_get_wtime() - t0, ' s)'
    sync = [ecorr, 1d0]
    call xzmpi_bcast(sync, 16_8)
  else
    if (nwe == 2) then
      call xzamps_z(w, 'r', ecorr)
    else
      call xzamps_d(w, 'r', ecorr)
    end if
    write(6, '(/,a,f20.12)') '   amplitudes read from the file XZAMPS; CCSDT correlation energy ', ecorr
  end if
  flush(6)

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
  if (with_q) then
    write(6, '(a,f25.12)') '   E[Q] (fifth order)            ', eq5
    write(6, '(a,f25.12)') '   E(Q)                          ', eq6
    write(6, '(a,f25.12)') '   E(CCSDT[Q])                   ', escf + ecorr + eq5
    write(6, '(a,f25.12)') '   E(CCSDT(Q))                   ', escf + ecorr + eq6
    etot = escf + ecorr + eq6
    if (method == 'CCSDT[Q]' .or. method == '[Q]') etot = escf + ecorr + eq5
  end if
  write(6, '(a)') '  ' // repeat('-', 62)
  call putrec(20, 'JOBARC', 'TOTENERG', iintfp, etot)
  write(6, '(a,f25.12)') '   TOTENERG written to JOBARC:   ', etot
  call xz_rss(kb)
  write(6, '(/,a,f10.2,a,f10.2,a,f10.2,a)') '   wall times: integrals ', tread, ' s   CCSDT ', tcc, &
    ' s   (Q) ', tq, ' s'
  write(6, '(a,i14,a,f10.1,a,f8.1,a)') '   work array used: ', xzpeak, ' words = ', 8d-6 * xzpeak, &
    ' MB (', 100d0 * xzpeak / max(xzsize, 1_8), ' % of MAXCOR)'
  write(6, '(a,f10.1,a)') '   peak resident memory of the process: ', 1d-3 * kb(1), ' MB'
  write(6, '(a,f10.2,a)') '   XZCCSDT total wall time: ', omp_get_wtime() - twall, ' s'
  write(6, '(a)') '   XZCCSDT finished: ' // trim(method) // ' energy done.'
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
  if (method /= 'CCSDT' .and. method /= 'CCSDT[Q]' .and. method /= 'CCSDT(Q)' .and. method /= '(Q)' &
      .and. method /= '[Q]') then
    write(6, '(/,a,a,a)') ' @XZCCSDT-F, method ', trim(method), ': not one of CCSDT, CCSDT[Q], CCSDT(Q), (Q), [Q].'
    call errex
  end if
  with_q = method /= 'CCSDT'
  conv_t = 10d0 ** (-abs(iflags(4)))
  conv_e = 10d0 ** (-abs(iflags(4)) - 2)
  maxcyc = iflags(7)
  if (maxcyc <= 0) maxcyc = 50
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
  integer(8) :: p, i, j, k, g, n, d, sa, si

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
  jrows = xzget(max(nto, npo, nv, no), 'rows')
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
    call xziota(max(nto, npo, nv, no), w(jrows))
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
! The wall times of the stages of the iterations.
subroutine xztimes(tcc, ncyc)
  use xzdata
  implicit none
  real(8), intent(in) :: tcc
  integer(8), intent(in) :: ncyc
  integer :: k
  real(8) :: tsum
  write(6, '(/,a,i4,a)') '   Wall times of the CCSDT iterations (', ncyc, ' cycles)'
  tsum = 0d0
  do k = 1, ntime
    write(6, '(5x,a28,f12.2,a,f6.1,a)') tname(k), tstage(k), ' s', 100d0 * tstage(k) / max(tcc, 1d-9), ' %'
    tsum = tsum + tstage(k)
  end do
  write(6, '(5x,a28,f12.2,a)') 'other', tcc - tsum, ' s'
  write(6, '(5x,a28,f12.2,a,f10.2,a)') 'total', tcc, ' s;  per cycle ', tcc / max(ncyc, 1_8), ' s'
end subroutine xztimes
