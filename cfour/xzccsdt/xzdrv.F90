!
! The driver of xzccsdt: the job, the sizes, the memory plan, the integrals,
! the CCSDT iterations, (Q), the energies.
!
subroutine xzdrv(w, maxcor)
  use xzdata
  use xzmem
  use xzkern
  use omp_lib
  implicit none
  integer(8), intent(in) :: maxcor
  real(8) :: w(maxcor)
  integer(8) :: iintln, ifltln, iintfp, ialone, ibitwd
  common /machsp/ iintln, ifltln, iintfp, ialone, ibitwd
  integer(8) :: nslot, nch, ncyc, nthr, ilp64, kb(2)
  logical :: usefile, converged
  real(8) :: ecorr, eq5, eq6, etot, t0, tread, tcc, tq, twall
  character(len=32) :: arg
  integer :: larg
  integer(8) :: nmax

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
  write(6, '(a,f12.1,a)') '   T3 on i<j<k, a<b<c: ', 16d-6 * nto * nvt, ' MB (complex)'
  write(6, '(a,es9.1,a,es9.1,a,i5)') '   convergence: energy ', conv_e, '   amplitudes ', conv_t, &
    '   cycles ', maxcyc
  write(6, '(a,f20.12)') '   E(SCF) of JOBARC: ', escf
  write(6, '(a,i14,a,f10.1,a)') '   work array (MAXCOR): ', maxcor, ' words = ', 8d-6 * maxcor, ' MB'
  write(6, '(a,i4,a,i2)') '   threads: ', nthr, '   BLAS integers: bytes ', 4 + 4 * ilp64
  if (ilp64 /= 1) then
    write(6, '(a)') ' @XZCCSDT-F, the kernels were not compiled for the 8-byte integers of this BLAS.'
    call errex
  end if
  if (no < 3 .or. nv < 3) then
    write(6, '(a)') ' @XZCCSDT-F, fewer than three occupied or virtual spinors: no triples.'
    call errex
  end if

  ! ---- the memory plan, before anything is placed
  nwe = 2
  call xzplan(w, nslot, usefile, nch)

  ! ---- tables and integrals
  t0 = omp_get_wtime()
  call xzlayout(w)
  tread = omp_get_wtime() - t0
  write(6, '(/,a,f10.2,a)') '   integrals read and sorted: ', tread, ' s'
  flush(6)

  ! ---- CCSDT
  t0 = omp_get_wtime()
  call xzcc_z(w, nslot, usefile, nch, ecorr, converged, ncyc)
  tcc = omp_get_wtime() - t0
  if (.not. converged) then
    call get_environment_variable('XZ_DEBUG', length=ncyc)
    if (ncyc > 0) return
    write(6, '(/,a,i5,a)') ' @XZCCSDT-F, CCSDT did not converge in ', maxcyc, ' cycles (CC_MAXCYC).'
    call errex
  end if
  write(6, '(a,i4,a)') '   CCSDT converged in ', ncyc, ' cycles.'
  call xztimes(tcc, ncyc)

  ! ---- (Q)
  eq5 = 0d0
  eq6 = 0d0
  tq = 0d0
  if (with_q) then
    t0 = omp_get_wtime()
    call xzq_z(w, eq5, eq6)
    tq = omp_get_wtime() - t0
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
    if (method == 'CCSDT[Q]') etot = escf + ecorr + eq5
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
  if (method /= 'CCSDT' .and. method /= 'CCSDT[Q]' .and. method /= 'CCSDT(Q)') then
    write(6, '(/,a,a,a)') ' @XZCCSDT-F, method ', trim(method), ': not one of CCSDT, CCSDT[Q], CCSDT(Q).'
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
! The arrays that live for the whole run: tables, spinor energies, integrals.
! In a dry run only their places.
subroutine xzlayout(w)
  use xzdata
  use xzmem
  implicit none
  real(8) :: w(*)
  integer(8) :: m, kbuf, krow, ksrc, po, pv, nbuf

  po = no * (no + 1) / 2
  pv = nv * (nv + 1) / 2
  jotri = xzget(no * npo, 'occupied tables')
  josgn = xzget(no * npo / 8 + 1, 'occupied tables')
  jofree = xzget(3 * nto, 'occupied tables')
  jopair = xzget(3 * nto, 'occupied tables')
  jopa = xzget(npo, 'occupied tables')
  jopb = xzget(npo, 'occupied tables')
  jopidx = xzget(no * no, 'occupied tables')
  jvtri = xzget(nv * nvp, 'virtual tables')
  jvsgn = xzget(nv * nvp / 8 + 1, 'virtual tables')
  jvfree = xzget(3 * nvt, 'virtual tables')
  jvpair = xzget(3 * nvt, 'virtual tables')
  jvpa = xzget(nvp, 'virtual tables')
  jvpb = xzget(nvp, 'virtual tables')
  jvpidx = xzget(nv * nv, 'virtual tables')
  jrows = xzget(nto, 'rows')
  jeps = xzget(ny, 'spinor energies')
  jeo3 = xzget(nto, 'spinor energies')
  jev3 = xzget(nvt, 'spinor energies')
  joooo = xzget(nwe * no ** 4, '<oo||oo>')
  jooov = xzget(nwe * no ** 3 * nv, '<oo||ov>')
  joovv = xzget(nwe * no * no * nv * nv, '<oo||vv>')
  jovvo = xzget(nwe * no * no * nv * nv, '<ov||vo>')
  joovvp = xzget(nwe * no * no * nvp, '<oo||vv> on a<b')
  jgpp = xzget(nwe * npo * nvp, '<oo||vv> on i<j, a<b')
  jovvq = xzget(nwe * no * nv * nvp, '<ov||vv> on a<b')

  if (.not. xzdry) then
    call xzmaps(no, npo, nto, w(jotri), w(josgn), w(jofree), w(jopair), w(jopa), w(jopb), w(jopidx))
    call xzmaps(nv, nvp, nvt, w(jvtri), w(jvsgn), w(jvfree), w(jvpair), w(jvpa), w(jvpb), w(jvpidx))
    call xziota(nto, w(jrows))
  end if
  m = xzmark()
  nbuf = max(2 * po * po, 2 * po * no * nv, 2 * no * no * nv * nv, 2 * pv * nv, 2 * ntot)
  kbuf = xzget(nbuf, 'record of an integral file')
  if (.not. xzdry) then
    call xzrdeps(w(jeps), w(kbuf))
    call xzsum3(no, nto, w(jeps), w(jeo3))
    call xzsum3(nv, nvt, w(jeps + no), w(jev3))
    call xzrdsmall(w(joooo), w(jooov), w(joovv), w(jovvo), w(joovvp), w(jgpp), w(jvpa), w(jvpb), &
                   w(jopa), w(jopb), w(kbuf))
    call xzrdovvq(w(jovvq), w(jvpa), w(jvpb), w(kbuf))
  end if
  call xzrel(m)
  ! <ab||cd>: the record of the file is read where the packed block stays
  jvvvv = xzget(2 * pv * pv, 'record of moinpppp')
  krow = xzget(2 * nvp, 'row of <vv||vv>')
  ksrc = xzget(2 * nvp, 'permutation of the pairs')
  if (.not. xzdry) call xzrdvvvv(w(jvvvv), w(jvpidx), w(krow), w(ksrc))
  call xzrel(m)
  jvvvv = xzget(nwe * nvp * nvp, '<vv||vv> on a<b, c<d')
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
