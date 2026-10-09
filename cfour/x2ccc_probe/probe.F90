!
! x2cprobe: CFOUR's spinor CCSD residual (hbarso of xx2cccsd, which is a
! sequence of calls of libx2ccc) against the CCSD part of the native CCSDT
! step (cfour/xzccsdt), on the same integral files and the same amplitudes.
! A probe for docs/native_x2ccc.md, not a part of the production program.
!
! What is CFOUR's here: the object files hbarso.o and its helpers of
! $(CFOUR)/x2cccsd and the libraries libx2ccc.a, libx2cint.a, liblibr.a (see
! the Makefile), linked, not copied.  What xx2cccsd's own initialisation
! (setsocc, eomccini, x2c_ccsddrv) does for hbarso is done here by hand in
! cfsetup / cfload: a dozen module variables, the offsets of soccmem into one
! region of the work array, two index tables, the names of the PPPH files,
! and the integral files read by CFOUR's own readers (vreadp, vreadf).
!
! Environment:
!   PROBE_WARM   Jacobi steps of the native CCSD from the second-order start
!                that make the amplitudes of the comparison (default 3)
!   PROBE_REP    timed repetitions, the two codes alternating (default 3)
!   PROBE_T3OPS  1: time the intermediates that only the T3 equation needs
!                as well (W_vvvo, W_vooo, the operands of pass 2, the
!                occupied operator, the pp operator); default 1
!
! Complex arithmetic, no labels (CFOUR's routines know neither real
! arithmetic nor labels).
!
subroutine probedrv(w, maxcor)
  use xzdata
  use xzmem
  use xzkern
  use omp_lib
  implicit none
  integer(8), intent(in) :: maxcor
  real(8) :: w(maxcor)
  integer(8) :: status, kt1n, kt2n, kcf, lcf, nthr, ilp64, nwarm, nrep, k, irep, n1, n2, kb(2)
  integer, parameter :: nt = 12
  real(8) :: tn(nt), tsum(nt), tmin(nt), t0, tcf, tnat, tcfmin, tcfsum, tnatmin, tnatsum
  real(8) :: d1, d2, u1, u2, ecf(2), tread, tcfread
  complex(8) :: e
  logical :: t3ops
  character(len=32) :: arg
  integer :: larg
  character(len=28), parameter :: names(nt) = [character(len=28) :: &
    'T1 dressing', 'R1 operand (<ak||cd> t2)', 'dressed <ab||ij> (with Bv)', 'W_ovvo, F_vv, F_oo', &
    'dressed <ab||cd> rows', 'pp ladder GEMM', 'CCSD part (xzccsd)', &
    'T3 only: W_vvvo terms', 'T3 only: W_vooo, pass-2 ops', 'T3 only: occupied operator', &
    'T3 only: pp operator', ' ']

  call xzinit(loc(w(1)), maxcor)
  call xzsizes()
  call xz_blas_info(nthr, ilp64)
  nwarm = 3
  nrep = 3
  t3ops = .true.
  call get_environment_variable('PROBE_WARM', arg, larg)
  if (larg > 0) read(arg, *) nwarm
  call get_environment_variable('PROBE_REP', arg, larg)
  if (larg > 0) read(arg, *) nrep
  call get_environment_variable('PROBE_T3OPS', arg, larg)
  if (larg > 0) t3ops = arg(1:1) /= '0'
  write(6, '(/,a)') '  X2CPROBE: hbarso of xx2cccsd (libx2ccc) against the CCSD part of xzccsdt'
  write(6, '(a,i5,a,i5,a,i5,a,i5)') '   active spinors: occupied ', no, '   virtual ', nv, '   frozen ', nfrozen, &
    '   dropped ', ndropped
  write(6, '(a,i4,a,i3,a,i3)') '   threads: ', nthr, '   warm-up steps ', nwarm, '   repetitions ', nrep
  if (ilp64 /= 1) call errex

  ! ---- the native side: tables and integrals as xzdrv places them, dense, complex
  allocate(lab(0:ntot-1), operm(0:no-1), vperm(0:nv-1), ocls(0:no-1), vcls(0:nv-1))
  call xzsetmode(.false.)
  nwe = 2
  t0 = omp_get_wtime()
  call xzlayout(w, status)
  tread = omp_get_wtime() - t0
  if (status /= 0) then
    write(6, '(a)') ' @X2CPROBE-F, the native reader refused the integral files.'
    call errex
  end if
  n1 = 2 * no * nv
  n2 = 2 * no * no * nv * nv
  jt1 = xzget(n1, 'T1')
  jt2 = xzget(n2, 'T2')
  kt1n = xzget(n1, 'new T1')
  kt2n = xzget(n2, 'new T2')
  nqs = nvp
  nba = max(1_8, nv / 8)

  ! ---- CFOUR's side: one region of the work array with the layout of soccmem
  call cfsetup(no, nv, nfrozen, lcf)
  kcf = xzget(lcf, 'region of hbarso')
  t0 = omp_get_wtime()
  call cfload(w(kcf), lcf, no, nv, w(jeps))
  tcfread = omp_get_wtime() - t0
  write(6, '(a,f9.2,a,f9.2,a)') '   integrals: native reader ', tread, ' s   CFOUR''s readers (vreadp, vreadf) ', &
    tcfread, ' s'
  write(6, '(a,f10.1,a)') '   region of hbarso (integrals, amplitudes, intermediates, work): ', 8d-6 * lcf, ' MB'

  ! ---- amplitudes: a few Jacobi steps of the native CCSD from second order
  call xzguess_z(no, nv, w(jeps), w(joovv), w(jt1), w(jt2))
  call xzecorr_z(no, nv, w(jt1), w(jt2), w(joovv), e)
  write(6, '(a,f20.12)') '   second-order correlation energy ', dble(e)
  do k = 1, nwarm
    call nccstep(w, jt1, jt2, kt1n, kt2n, .false., tn)
    w(jt1:jt1 + n1 - 1) = w(kt1n:kt1n + n1 - 1)
    call zc_vlincomb_copy(n2, w(kt2n), w(jt2))
    call xzecorr_z(no, nv, w(jt1), w(jt2), w(joovv), e)
    write(6, '(a,i3,a,f20.12)') '   after native CCSD step ', k, ': correlation energy ', dble(e)
  end do

  ! ---- one step of each on these amplitudes, element by element
  call cfput(w(kcf), no, nv, w(jt1), w(jt2))
  flush(6)
  call hbarso(w(kcf))
  flush(6)
  call nccstep(w, jt1, jt2, kt1n, kt2n, .false., tn)
  call cfcmp(w(kcf), no, nv, w(jt1), w(jt2), w(kt1n), w(kt2n), d1, d2, u1, u2, ecf)
  call xzecorr_z(no, nv, w(kt1n), w(kt2n), w(joovv), e)
  write(6, '(/,a)') '   COMPARISON of the new amplitudes of one Jacobi step (T + R / D)'
  write(6, '(a,es10.2,a,es10.2)') '   largest |dT1|, |dT2| of the step:            ', u1, '  ', u2
  write(6, '(a,es10.2,a,es10.2)') '   largest |T1(CFOUR) - T1(native)|, the same T2: ', d1, '  ', d2
  write(6, '(a,f20.14)') '   correlation energy of the new amplitudes, native ', dble(e)
  write(6, '(a,f20.14)') '                                            CFOUR  ', ecf(1)
  if (max(d1, d2) < 1d-10) then
    write(6, '(a)') '   PROBE-AGREE: element-wise below 1e-10'
  else
    write(6, '(a)') '   PROBE-DIFFER: element-wise above 1e-10'
  end if

  ! ---- timing: the two codes alternating
  tsum = 0
  tmin = huge(1d0)
  tcfsum = 0
  tnatsum = 0
  tcfmin = huge(1d0)
  tnatmin = huge(1d0)
  do irep = 1, nrep
    write(6, '(a,i3)') ' PROBE-REP ', irep
    flush(6)
    t0 = omp_get_wtime()
    call hbarso(w(kcf))
    tcf = omp_get_wtime() - t0
    flush(6)
    t0 = omp_get_wtime()
    call nccstep(w, jt1, jt2, kt1n, kt2n, t3ops, tn)
    tnat = omp_get_wtime() - t0
    tsum = tsum + tn
    tmin = min(tmin, tn)
    tcfsum = tcfsum + tcf
    tcfmin = min(tcfmin, tcf)
    tnat = sum(tn(1:7))
    tnatsum = tnatsum + tnat
    tnatmin = min(tnatmin, tnat)
    write(6, '(a,i3,a,f10.4,a,f10.4,a,f10.4)') ' PROBE-TIME rep ', irep, '  hbarso ', tcf, &
      '  native CCSD part ', tnat, '  native T3-only intermediates ', sum(tn(8:11))
  end do
  if (nrep > 0) then
    write(6, '(/,a,i3,a,i4,a)') '   WALL TIMES, seconds (', nrep, ' repetitions, ', nthr, ' threads): mean, least'
    write(6, '(5x,a28,2f11.4)') 'CFOUR hbarso (whole)', tcfsum / nrep, tcfmin
    write(6, '(5x,a28,2f11.4)') 'native CCSD part (sum)', tnatsum / nrep, tnatmin
    do k = 1, 11
      write(6, '(8x,a28,2f11.4)') names(k), tsum(k) / nrep, tmin(k)
    end do
    write(6, '(a,f8.2)') '   PROBE-RATIO hbarso / native CCSD part (means): ', tcfsum / max(tnatsum, 1d-12)
  end if
  call xz_rss(kb)
  write(6, '(a,f10.1,a,f10.1,a)') '   work array used: ', 8d-6 * xzpeak, ' MB   peak resident set: ', 1d-3 * kb(1), ' MB'
  write(6, '(a)') '   X2CPROBE finished.'
end subroutine probedrv

! n double words from src to dst (threads)
subroutine zc_vlincomb_copy(n, src, dst)
  implicit none
  integer(8), intent(in) :: n
  real(8), intent(in) :: src(n)
  real(8), intent(out) :: dst(n)
  integer(8) :: i
  !$omp parallel do
  do i = 1, n
    dst(i) = src(i)
  end do
end subroutine

! ---------------------------------------------------------------------------
! The CCSD part of the native CCSDT step, alone: what xzstep_z does for the
! T1 and T2 equations when there is no T3 (the same routines of xzimd.inc and
! the same kernels in the same order; the passes over T3 left out).  With
! t3ops the intermediates that only the T3 equation needs are formed as well
! (timers 8 to 11); they do not change the result.
!   tn(1:7)   stages of the CCSD part; tn(8:11) the T3-only intermediates
subroutine nccstep(w, kt1, kt2, kt1n, kt2n, t3ops, tn)
  use xzdata
  use xzmem
  use xzkern
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kt1, kt2, kt1n, kt2n
  logical, intent(in) :: t3ops
  real(8), intent(out) :: tn(12)
  integer(8) :: m0, ov, ov2
  integer(8) :: kfoo, kfov, kfvo, kfvv, kpvv, kfevv, kfeoo, kft
  integer(8) :: kooovd, kooood, kovvod, kovood, kk, ktmp
  integer(8) :: kt2p, kt2pp, ktau, ktt
  integer(8) :: kgpd, kgmq, kyall, kr1c, kr1v, kra, kr2, klad, kwkl, kweff, kwo, kadrv, ko6, kwring
  integer(8) :: ka2, kb2, kx2, kd, kwv, kg2, kyb, ku, kxb, kslab
  integer(8) :: q0, nq
  real(8) :: t0
  complex(8) :: one, zero, half
  one = 1
  zero = 0
  half = 0.5d0
  ov = no * nv
  ov2 = ov * ov
  tn = 0
  m0 = xzmark()
  kfoo = xzget(2 * no * no, 'F_oo')
  kfov = xzget(2 * ov, 'F_ov')
  kfvo = xzget(2 * ov, 'F_vo')
  kfvv = xzget(2 * nv * nv, 'F_vv')
  kpvv = xzget(2 * nv * nv, 'F_vv work')
  kfevv = xzget(2 * nv * nv, 'F_vv of T3')
  kfeoo = xzget(2 * no * no, 'F_oo of T3')
  kft = xzget(2 * ov, 'F_ov transposed')
  kooovd = xzget(2 * no * no * ov, 'dressed <oo||ov>')
  kooood = xzget(2 * no ** 4, 'dressed <oo||oo>')
  kovvod = xzget(2 * ov2, 'dressed <ov||vo>')
  kovood = xzget(2 * no * no * ov, 'dressed <ov||oo>')
  kk = xzget(2 * ov2, 'work of the dressing')
  ktmp = xzget(2 * no * no * ov, 'work of W_vooo')
  kt2p = xzget(2 * no * no * nvp, 'T2 [i][j][Q]')
  kt2pp = xzget(2 * npo * nvp, 'T2 [P][Q]')
  ktau = xzget(2 * npo * nvp, 'tau1')
  ktt = xzget(2 * npo * nvp, 'T2 + tau1')
  kgpd = xzget(2 * ny * npo * nv, 'pass-1 operand')
  kgmq = xzget(2 * no * nvp * ny, 'pass-1 operand G')
  kyall = xzget(2 * npo * nv * ny, 'pass-1 result Y')
  kr1c = xzget(2 * ov, 'T3 part of R1')
  kr1v = xzget(2 * no * ny, 'vovv part of R1')
  kra = xzget(2 * npo * nvp, 'T3 part of R2')
  kr2 = xzget(2 * npo * nvp, 'R2')
  klad = xzget(2 * npo * nvp, 'CCSD ladder')
  kwkl = xzget(2 * npo * npo, 'W_oooo')
  kweff = xzget(2 * ov2, 'W_ovvo')
  kwo = xzget(2 * ov * npo, 'W_vooo')
  kadrv = xzget(2 * npo * nv * ny, 'pass-2 operand A')
  ko6 = xzget(2 * nto * nto, 'occupied operator')
  kwring = xzget(2 * ov2, 'pass-2 operand W')
  ka2 = xzget(2 * ov2, 'work A')
  kb2 = xzget(2 * ov2, 'work B')
  kx2 = xzget(2 * ov2, 'work X')
  kd = xzget(2 * ov * nvp, 'dressed <ov||vv>')
  kwv = xzget(2 * nvp * ov, 'W_vvvo')
  kg2 = xzget(2 * npo * ov, 'work [P][e][i]')
  kyb = xzget(2 * nba * nv * nv * no, 'batch of W_vvvo terms Y')
  ku = xzget(2 * nv ** 3, 'dressed <mb||ef> of one m')
  kxb = xzget(2 * nba * no * nv * nv, 'batch of W_vvvo terms X')
  kslab = xzget(2 * nqs * nvp, 'slab of the dressed <ab||cd>')

  ! ---- T1 dressing
  t0 = omp_get_wtime()
  call xzfock_z(no, nv, nvp, w(jeps), w(kt1), w(jooov), w(joovv), w(jovvo), w(jovvq), &
                w(jvpa), w(jvpb), w(kfoo), w(kfov), w(kfvo), w(kfvv), w(kpvv))
  call xzdroo_z(no, nv, w(kt1), w(jooov), w(joovv), w(joooo), w(kooovd), w(kooood))
  call xzdrovvq_z(no, nv, nvp, w(kt1), w(jovvq), w(joovvp), w(kd))
  call xzdrov_z(no, nv, nvp, w(kt1), w(jooov), w(jovvo), w(jovvq), w(kd), w(kooood), &
                w(jvpidx), w(kovvod), w(kovood), w(kk))
  call xzt2forms_z(no, nv, npo, nvp, w(kt1), w(kt2), w(jopa), w(jopb), w(jvpa), w(jvpb), &
                   w(kt2p), w(kt2pp), w(ktau), w(ktt))
  tn(1) = omp_get_wtime() - t0

  ! ---- the <ak||cd> t2 term of R1 (in a CCSDT step: with the pass-1 operands)
  t0 = omp_get_wtime()
  call xzp1ops_z(no, nv, npo, nvp, w(joovv), w(kooovd), w(joovvp), w(kd), w(kfov), &
                 w(jopa), w(jopb), w(kgpd), w(kgmq), w(kft))
  call zc_gemm_z(0_4, 0_4, no, ny, no * nvp, one, w(kt2p), no * nvp, w(kgmq), ny, zero, w(kr1v), ny)
  call zc_vzero_d(2 * ov, w(kr1c))
  call zc_vzero_d(2 * npo * nvp, w(kra))
  tn(2) = omp_get_wtime() - t0

  ! ---- the dressed <ab||ij> (from the bra-dressed <ab||ei>, the start of W_vvvo)
  t0 = omp_get_wtime()
  call xzbv_z(no, nv, npo, nvp, nba, w(kt1), w(ktau), w(jovvq), w(jooov), w(jovvo), &
              w(jopa), w(jopb), w(jvpa), w(jvpb), w(kwv), w(kg2), w(kyb))
  call xzvvoo_z(no, nv, npo, nvp, w(kt1), w(ktau), w(jgpp), w(jooov), w(joooo), w(kwv), &
                w(jopa), w(jopb), w(jvpa), w(jvpb), w(kr2))
  tn(3) = omp_get_wtime() - t0

  if (t3ops) then
    t0 = omp_get_wtime()
    call xzwvoovo_z(no, nv, npo, nvp, w(kt2pp), w(kooovd), w(kfov), w(jopa), w(jopb), w(kwv), w(kg2))
    call xzwvx_z(no, nv, nvp, nba, w(kt2), w(kd), w(jvpa), w(jvpb), w(kwv), w(ka2), w(ku), &
                 w(kxb), w(kyb))
    tn(8) = omp_get_wtime() - t0
    t0 = omp_get_wtime()
    call zc_vzero_d(2 * npo * nv * ny, w(kyall))
    call xzwo_z(no, nv, npo, nvp, w(kt2), w(kt2pp), w(kooovd), w(kovood), w(kd), w(kyall), &
                w(jopa), w(jopb), w(kwo), w(ktmp))
    tn(9) = omp_get_wtime() - t0
  end if

  ! ---- W_ovvo, F_vv, F_oo
  t0 = omp_get_wtime()
  call xzweff_z(no, nv, nvp, w(kt2), w(kt2p), w(joovv), w(joovvp), w(kovvod), w(kfoo), w(kfvv), &
                w(kweff), w(kfevv), w(kfeoo), w(ka2), w(kb2), w(kx2))
  tn(4) = omp_get_wtime() - t0
  if (t3ops) then
    t0 = omp_get_wtime()
    call xzp2ops_z(no, nv, npo, w(kt2), w(kweff), w(kwo), w(jopa), w(jopb), w(kwring), w(kadrv))
    tn(9) = tn(9) + omp_get_wtime() - t0
  end if

  ! ---- the sweep over the rows of the dressed <ab||cd>: the CCSD ladder
  call zc_vzero_d(2 * npo * nvp, w(klad))
  do q0 = 0, nvp - 1, nqs
    nq = min(nqs, nvp - q0)
    t0 = omp_get_wtime()
    call xzdrrows_z(no, nv, npo, nvp, nvp, nq, w(kt1), w(ktau + 2 * q0), &
                    w(jvvvv + 2 * q0 * nvp), w(jovvq), w(jgpp), w(jvpa + q0), &
                    w(jvpb + q0), w(kslab), w(ka2))
    tn(5) = tn(5) + omp_get_wtime() - t0
    if (t3ops) then
      t0 = omp_get_wtime()
      call zc_pair_t1_contract_z(w(kwv + 2 * q0 * ov), w(kslab), w(kt1), nq, nv, nvp, w(jvpidx), no)
      tn(8) = tn(8) + omp_get_wtime() - t0
    end if
    t0 = omp_get_wtime()
    if (npo > 0) call zc_gemm_z(0_4, 1_4, npo, nq, nvp, one, w(ktt), nvp, w(kslab), nvp, &
                                zero, w(klad + 2 * q0), nvp)
    tn(6) = tn(6) + omp_get_wtime() - t0
    if (t3ops) then
      t0 = omp_get_wtime()
      if (npo > 0) call zc_gemm_z(1_4, 0_4, nq, nvp, npo, one, w(kt2pp + 2 * q0), nvp, &
                                  w(jgpp), nvp, one, w(kslab), nvp)
      call zc_pair_one_body_add_z(w(kslab), nq, nvp, w(jvpa + q0), w(jvpb + q0), w(jvpidx), nv, &
                                  w(kfevv), nv, half)
      tn(11) = tn(11) + omp_get_wtime() - t0
    end if
  end do

  ! ---- the CCSD part: new T1 and T2
  t0 = omp_get_wtime()
  call xzccsd_z(no, nv, npo, nvp, w(jeps), w(kt1), w(kt2), w(kt2p), w(kt2pp), w(jgpp), w(kooovd), &
                w(kooood), w(kovvod), w(kweff), w(kfov), w(kfvo), w(kfevv), w(kfeoo), w(kr1v), &
                w(kr1c), w(kra), w(klad), w(jopa), w(jopb), w(jvpa), w(jvpb), w(kr2), w(kwkl), &
                w(kt1n), w(kt2n), w(ka2), w(kb2), w(kx2))
  tn(7) = omp_get_wtime() - t0
  if (t3ops) then
    t0 = omp_get_wtime()
    call xzo6_z(no, npo, nto, w(jofree), w(jopair), w(kwkl), w(kfeoo), w(ko6))
    tn(10) = omp_get_wtime() - t0
  end if
  call xzrel(m0)
end subroutine nccstep
