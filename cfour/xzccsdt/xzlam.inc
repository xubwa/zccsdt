!
! The Lambda equations of CCSDT on packed Lambda_3 (type-generic).  The
! algorithm of src/zccsdt/cc/lambda_opt.py (docs/lambda.md, section 9; in
! blocks of the labels lambda_blk.py, section 9.7): T1-dressed integrals,
! nothing conjugated, l1 and l2 dense, l3 and its residual stored as T3 is.
!
!   once (xzlamim)   what an amplitude iteration makes - the dressed
!                    integrals, pass 1 over T3, W_vvvo, W_vooo, W_ovvo, F_vv,
!                    F_oo, W_oooo, the operator on occupied triples - and two
!                    contractions of T3 (Xq, X2).  The pp operator of the T3
!                    equation, ~<ab||de> + sum_{m<n} t2[mn][ab] <mn||de> +
!                    1/2 A[F_vv], is made in the place of the bare <ab||de>:
!                    after the Lambda equations the work array has no bare
!                    <ab||de> any more.
!   an iteration     pass 1 over l3 and T3 (two stacked GEMMs, the density of
!   (xzlamstep)      l3 with T3), the Gram matrix of the two, pass 2 (ring,
!                    two outer products, four drive products), the ladder
!                    with the transposed pp operator and the contractions
!                    with Xq and X2, the transposed operator on occupied
!                    triples, the residuals of rank 1 and 2, the Jacobi step
!
! T3, l3 and the residual are three regions of the size of T3; the DIIS
! vectors are those of the amplitudes (xzdiis.F90: the work array or the
! file XZDIIS).  Batch sizes: nbp (two half-expanded batches in pass 1 and
! in pass 2), nbt (the gathered batch of the ladder and its product), nba.
!
! With more than one rank (xzdist.F90): rank 0 sends T1, T2, T3 once - every
! rank makes the intermediates itself - and lambda at the start of every
! iteration; the other ranks do their batches of pass 1, pass 2 and the
! ladder.  The DIIS vectors and the residuals of rank 1 and 2 are rank 0's.
!
subroutine N(xzlam)(w, nslot, usefile, nch, converged, ncyc)
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
  logical, intent(out) :: converged
  integer(8), intent(out) :: ncyc
  integer(8) :: kl1n, kl2n, kr3, icyc, m0, ov, kk
  integer(8), external :: N(xzdiis_target)
  real(8) :: norms(3), normt, t0, t1w, ctl(2), tsg
  ZT :: pe
  character(len=256) :: dbg
  integer :: ldbg

  ov = no * nv
  kk = 2 * ny
  converged = .false.
  ncyc = 0
  jl1 = xzget(NW * ov, 'Lambda1')
  jl2 = xzget(NW * ov * ov, 'Lambda2')
  jl3 = xzget(NW * n3, 'Lambda3')
  m0 = xzmark()
  kl1n = xzget(NW * ov, 'new Lambda1')
  kl2n = xzget(NW * ov * ov, 'new Lambda2')
  if (xzworker() .and. .not. slices) then
    call N(xzdiis_alloc)(0_8, .false., nch)
  else
    call N(xzdiis_alloc)(nslot, usefile, nch)
  end if
  ! ---- what is kept of the integrals and of t
  lfov = xzget(NW * ov, 'F_ov')
  lfvv = xzget(NW * nv * nv, 'F_vv of T3')
  lfoo = xzget(NW * no * no, 'F_oo of T3')
  lwov = xzget(NW * ov * ov, 'W_ovvo')
  lwl = xzget(NW * ov * ov, 'ring operator')
  lwkl = xzget(NW * npo * npo, 'W_oooo')
  lo6t = xzget(NW * merge(npo * npo, nto * nto, xzhhpair()), 'occupied operator')
  looovd = xzget(NW * no * no * ov, 'dressed <oo||ov>')
  lt2p = xzget(NW * no * no * nvp, 'T2 [i][j][Q]')
  lgmq = xzget(NW * no * nvp * ny, 'pass-1 operand G')
  lgpd = xzget(NW * ny * npo * nv, 'pass-1 operand')
  lbdrv = xzget(NW * kk * no * nvp, 'pass-2 operand B')
  ladrv = xzget(NW * npo * nv * kk, 'pass-2 operand A')
  lvx = xzget(NW * nv * nv * ov, '~<ej||bg>')
  lxq = xzget(NW * npo * nto * nv, 'Xq')
  lx2 = xzget(NW * nvp * nto * no, 'X2')
  if (xzdry) then
    call N(xzlamim)(w)
    call N(xzlamstep)(w, jl1, jl2, jl3, kl1n, kl2n, jl3, .false.)
    call xzrel(m0)
    return
  end if

  call get_environment_variable('XZ_LDEBUG', dbg, ldbg)
  t0 = omp_get_wtime()
  if (nrank > 1) then
    ! the amplitudes of rank 0 on every rank
    tsg = omp_get_wtime()
    call xzmpi_bcast_w(w(jt1), NW * ov)
    call xzmpi_bcast_w(w(jt2), NW * ov * ov)
    call xzmpi_bcast_w(w(jt3), NW * n3)
    dsec(6, 2) = dsec(6, 2) + omp_get_wtime() - tsg
    if (myrank > 0) dwords(2) = dwords(2) + dble(NW * n3)
  end if
  call N(xzlamim)(w)
  tlamim = omp_get_wtime() - t0
  write(6, '(/,a,f10.2,a)') '   intermediates of the Lambda equations: ', tlamim, ' s'
  if (ldbg > 0) call N(xzlamdump)(w, dbg(1:ldbg))
  if (myrank > 0) then
    call N(xzlamwork)(w, kl1n, kl2n, ncyc)
    call N(xzdiis_close)()
    call xzrel(m0)
    converged = .true.
    return
  end if

  ! the start: conj(t)
  call N(xzlconj)(ov, w(jt1), w(jl1))
  call N(xzlconj)(ov * ov, w(jt2), w(jl2))
  call N(xzlconj)(n3, w(jt3), w(jl3))

  write(6, '(/,a)') '  Lambda equations of CCSDT'
  write(6, '(a)') '  ' // repeat('-', 90)
  write(6, '(a)') '   cycle    1/4 <ab||ij> l2[ij][ab]      |dL1|      |dL2|      |dL3|   DIIS    time/s'
  write(6, '(a)') '  ' // repeat('-', 90)
  ncyc = maxcyc
  do icyc = 1, maxcyc
    t0 = omp_get_wtime()
    if (nrank > 1) then
      ! one more iteration: lambda to the ranks
      call xztic()
      tsg = omp_get_wtime()
      ctl = [1d0, 1d0]
      call xzmpi_bcast(ctl, 16_8)
      call xzmpi_bcast_w(w(jl1), NW * ov)
      call xzmpi_bcast_w(w(jl2), NW * ov * ov)
      if (.not. slices) call xzmpi_bcast_w(w(jl3), NW * n3)
      call xztoc(2, 'MPI: lambda to the ranks')
      dsec(6, 2) = dsec(6, 2) + omp_get_wtime() - tsg
    end if
    kr3 = N(xzdiis_target)()
    call N(xzlamstep)(w, jl1, jl2, jl3, kl1n, kl2n, kr3, ldbg > 0 .and. icyc == 1)
    if (ldbg > 0 .and. icyc == 1) then
      call xzdumpw(dbg(1:ldbg) // '.l1n', NW * ov, w(kl1n))
      call xzdumpw(dbg(1:ldbg) // '.l2n', NW * ov * ov, w(kl2n))
      call xzdumpw(dbg(1:ldbg) // '.l3n', NW * n3, w(kr3))
    end if
    call xztic()
    tsg = omp_get_wtime()
    call N(xzdiis_upd)(w, w(jl1), w(jl2), jl3, w(kl1n), w(kl2n), kr3, norms)
    call xztoc(2, 'DIIS')
    dsec(8, 2) = dsec(8, 2) + omp_get_wtime() - tsg
    if (slices) then
      tsg = omp_get_wtime()
      call xzmpi_slices_put(w(jl3), swoff)
      call xztoc(2, 'MPI: slices of l3 to the ranks')
      dsec(6, 2) = dsec(6, 2) + omp_get_wtime() - tsg
    end if
    call N(xzlpseudo)(no, nv, w(jl2), w(joovv), pe)
    normt = norms(1) + norms(2) + norms(3)
    write(6, '(i8,f28.14,3es11.3,i7,f10.2)') icyc, dble(pe), norms, ncnt, omp_get_wtime() - t0
    flush(6)
    if (normt < conv_t) then
      converged = .true.
      ncyc = icyc
      exit
    end if
  end do
  write(6, '(a)') '  ' // repeat('-', 90)
  if (nrank > 1) then
    ctl = 0d0
    call xzmpi_bcast(ctl, 16_8)
  end if
  call N(xzdiis_close)()
  call xzrel(m0)
end subroutine

! ---------------------------------------------------------------------------
! The Lambda iterations of a rank other than 0: lambda of rank 0 into l1, l2,
! l3 of this rank and its part of the step, as long as rank 0 says "one more".
subroutine N(xzlamwork)(w, kl1n, kl2n, icyc)
  use xzdata
  use xzmem
  use xzkern
  use xzdist
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kl1n, kl2n
  integer(8), intent(out) :: icyc
  integer(8), external :: N(xzdiis_target)
  integer(8) :: kr3, ov
  real(8) :: ctl(2), tsg, t0, norms(3)
  ov = no * nv
  icyc = 0
  write(6, '(/,a)') '  Lambda equations of CCSDT: the batches of this rank'
  ! (with slices l3 does not come from rank 0: the start is conj(t3) here as there)
  if (slices) call N(xzlconj)(n3, w(jt3), w(jl3))
  do
    call xzmpi_bcast(ctl, 16_8)
    if (ctl(1) /= 1d0) exit
    t0 = omp_get_wtime()
    tsg = t0
    call xztic()
    call xzmpi_bcast_w(w(jl1), NW * ov)
    call xzmpi_bcast_w(w(jl2), NW * ov * ov)
    if (.not. slices) then
      call xzmpi_bcast_w(w(jl3), NW * n3)
      dwords(2) = dwords(2) + dble(NW * n3)
    end if
    call xztoc(2, 'MPI: lambda from rank 0')
    dsec(6, 2) = dsec(6, 2) + omp_get_wtime() - tsg
    kr3 = N(xzdiis_target)()
    call N(xzlamstep)(w, jl1, jl2, jl3, kl1n, kl2n, kr3, .false.)
    if (slices) then
      tsg = omp_get_wtime()
      call xztic()
      call N(xzdiis_upd)(w, w(jl1), w(jl2), jl3, w(kl1n), w(kl2n), kr3, norms)
      call xztoc(2, 'DIIS')
      dsec(8, 2) = dsec(8, 2) + omp_get_wtime() - tsg
      tsg = omp_get_wtime()
      call xzmpi_slices_put(w(jl3), swoff)
      call xztoc(2, 'MPI: slices of l3 to the ranks')
      dsec(6, 2) = dsec(6, 2) + omp_get_wtime() - tsg
      dwords(2) = dwords(2) + dble(NW * (n3 - (soff(myrank + 1) - soff(myrank))))
    end if
    icyc = icyc + 1
    write(6, '(a,i5,f12.2,a)') '   cycle ', icyc, omp_get_wtime() - t0, ' s'
    flush(6)
  end do
end subroutine

! ---------------------------------------------------------------------------
! Lambda in the file XZLAMB of the run directory, the hand-over from the
! Lambda equations to what follows ((Q)_Lambda on every rank, or a later
! run): an unformatted stream in the format of XZAMPS,
!   character(8) 'XZLAMB01'
!   integer(8)   no, nv, words per element (1 real, 2 complex), classes of
!                the labels, elements of l3 as stored
!   real(8)      the CCSDT correlation energy of the amplitudes (of XZAMPS)
!   l1 [i][a], l2 [i][j][a][b], l3 as stored (packed i<j<k, a<b<c; its blocks
!   one after the other), in the order of the spinors of the run that wrote it
! rw = 'w': written from the work array; 'r': the regions of l2 and l3 are
! taken from the stack and filled (a dry run: only taken); l1 only for the
! density (with_den).
subroutine N(xzlamb)(w, rw, ecorr)
  use xzdata
  use xzmem
  implicit none
  real(8) :: w(*)
  character(len=1), intent(in) :: rw
  real(8), intent(in) :: ecorr
  character(len=8) :: tag
  integer(8) :: head(5), n1, n2
  real(8) :: e
  integer :: ios
  logical :: there
  n1 = NW * no * nv
  n2 = NW * no * no * nv * nv
  if (rw == 'w') then
    open(unit=77, file='XZLAMB', form='unformatted', access='stream', status='replace', iostat=ios)
    if (ios == 0) write(77, iostat=ios) 'XZLAMB01', no, nv, int(NW, 8), ncls, n3, ecorr
    if (ios == 0) write(77, iostat=ios) w(jl1:jl1 + n1 - 1)
    if (ios == 0) write(77, iostat=ios) w(jl2:jl2 + n2 - 1)
    if (ios == 0) write(77, iostat=ios) w(jl3:jl3 + NW * n3 - 1)
    if (ios == 0) close(77, iostat=ios)
    if (ios /= 0) then
      write(6, '(/,a,i8,a)') ' @XZCCSDT-F, writing the file XZLAMB failed (iostat ', ios, '): disk full?'
      call xzerrex
    end if
    return
  end if
  if (with_den) jl1 = xzget(n1, 'Lambda1')
  jl2 = xzget(n2, 'Lambda2')
  if (.not. t3shared) jl3 = xzget(NW * n3, 'Lambda3')
  if (xzdry) return
  inquire(file='XZLAMB', exist=there)
  if (.not. there) then
    write(6, '(/,a)') ' @XZCCSDT-F, no file XZLAMB: lambda of a converged run of xzccsdt lambda.'
    call xzerrex
  end if
  open(unit=77, file='XZLAMB', form='unformatted', access='stream', status='old', action='read', iostat=ios)
  if (ios == 0) read(77, iostat=ios) tag, head, e
  if (ios /= 0 .or. tag /= 'XZLAMB01') then
    write(6, '(/,a)') ' @XZCCSDT-F, the file XZLAMB is not a file of lambda of this program.'
    call xzerrex
  end if
  if (head(1) /= no .or. head(2) /= nv .or. head(3) /= NW .or. head(4) /= ncls .or. head(5) /= n3) then
    write(6, '(/,a)') ' @XZCCSDT-F, the file XZLAMB belongs to another calculation or another mode:'
    write(6, '(a,5i12)') '   occupied, virtual, words per element, classes, l3 in the file: ', head
    write(6, '(a,5i12)') '   and here:                                                      ', no, nv, &
      int(NW, 8), ncls, n3
    write(6, '(a)') '   (the labels and the arithmetic must be those of the run that wrote it: XZ_LABELS, XZ_REAL)'
    call xzerrex
  end if
  if (abs(e - ecorr) > 1d-9) then
    write(6, '(/,a)') ' @XZCCSDT-F, the file XZLAMB is lambda of other amplitudes than those of XZAMPS:'
    write(6, '(a,2f20.12)') '   CCSDT correlation energies of the two files: ', e, ecorr
    call xzerrex
  end if
  if (with_den) read(77, iostat=ios) w(jl1:jl1 + n1 - 1)
  if (ios == 0) read(77, pos=57 + 8 * n1, iostat=ios) w(jl2:jl2 + n2 - 1)
  if (ios == 0 .and. .not. t3shared) read(77, iostat=ios) w(jl3:jl3 + NW * n3 - 1)
  close(77)
  if (ios /= 0) then
    write(6, '(/,a,i8)') ' @XZCCSDT-F, reading the file XZLAMB failed, iostat ', ios
    call xzerrex
  end if
  if (t3shared) call xzmapt3(w, 'XZLAMB', n1 + n2, NW * n3, jl3)
end subroutine

! ---------------------------------------------------------------------------
! The intermediates.  The kept arrays (l... of xzdata) are placed by the
! caller; what is taken here is released.
subroutine N(xzlamim)(w)
  use xzdata
  use xzmem
  use xzkern
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8) :: ms, md, m1, m2, ov, ov2
  integer(8) :: kfoo, kfvo, kfvv, kpvv, kft, kooood, kovvod, kovood, kk, ktmp, ktmp2, kt2pp, ktau, ktt
  integer(8) :: kyall, kr1c, kra, kwo, ka2, kb2, kx2, kd, kbuf, kwv, kg2, kyb, ku, kxb
  integer(8) :: kgq, kgh, kpib, ktv, kz, h, lo, nqh, i0, nb, kvq
  logical :: dense
  real(8) :: stats(4)
  ZT :: one, zero, half
  one = 1
  zero = 0
  half = 0.5d0
  ov = no * nv
  ov2 = ov * ov
  dense = ncls == 1
  tcur = 3
  ms = xzmark()
  kfoo = xzget(NW * no * no, 'F_oo')
  kfvo = xzget(NW * ov, 'F_vo')
  kfvv = xzget(NW * nv * nv, 'F_vv')
  kpvv = xzget(NW * nv * nv, 'F_vv work')
  kft = xzget(NW * ov, 'F_ov transposed')
  kooood = xzget(NW * no ** 4, 'dressed <oo||oo>')
  kovvod = xzget(NW * ov2, 'dressed <ov||vo>')
  kovood = xzget(NW * no * no * ov, 'dressed <ov||oo>')
  kk = xzget(NW * ov2, 'work of the dressing')
  ktmp = xzget(NW * no * no * ov, 'work of W_vooo')
  ktmp2 = xzget(NW * no * no * ov, 'work of W_vooo 2')
  kt2pp = xzget(NW * npo * nvp, 'T2 [P][Q]')
  ktau = xzget(NW * npo * nvp, 'tau1')
  ktt = xzget(NW * npo * nvp, 'T2 + tau1')
  kyall = xzget(NW * npo * nv * ny, 'pass-1 result Y')
  kr1c = xzget(NW * ov, 'T3 part of R1')
  kra = xzget(NW * npo * nvp, 'T3 part of R2')
  kwo = xzget(NW * ov * npo, 'W_vooo')
  ka2 = xzget(NW * ov2, 'work A')
  kb2 = xzget(NW * ov2, 'work B')
  kx2 = xzget(NW * ov2, 'work X')

  ! ---- T1 dressing
  md = xzmark()
  kd = xzget(NW * ov * nvp, 'dressed <ov||vv>')
  if (.not. xzdry) then
    call xztic()
    call N(xzfock)(no, nv, nvp, w(jeps), w(jt1), w(jooov), w(joovv), w(jovvo), w(jovvq), &
                   w(jvpa), w(jvpb), w(kfoo), w(lfov), w(kfvo), w(kfvv), w(kpvv))
    call N(xzdroo)(no, nv, w(jt1), w(jooov), w(joovv), w(joooo), w(looovd), w(kooood))
    call N(xzdrovvq)(no, nv, nvp, w(jt1), w(jovvq), w(joovvp), w(kd))
    call N(xzdrov)(no, nv, nvp, w(jt1), w(jooov), w(jovvo), w(jovvq), w(kd), w(kooood), &
                   w(jvpidx), w(kovvod), w(kovood), w(kk))
    call N(xzt2forms)(no, nv, npo, nvp, w(jt1), w(jt2), w(jopa), w(jopb), w(jvpa), w(jvpb), &
                      w(lt2p), w(kt2pp), w(ktau), w(ktt))
    ! the operands of pass 1 over T3, in the places of the kept ones
    call N(xzp1ops)(no, nv, npo, nvp, w(joovv), w(looovd), w(joovvp), w(kd), w(lfov), &
                    w(jopa), w(jopb), w(lgpd), w(lgmq), w(kft))
    call xztoc(3, 'T1 dressing, operands')
  end if

  ! ---- pass 1 over T3: the T3 parts of W_vvvo and W_vooo (its result X in
  !      the first rows of the operand B)
  m1 = xzmark()
  if (dense) then
    kbuf = xzget(NW * nbp * ov * nvp, 'pass-1 batch of T3')
    if (.not. xzdry) then
      call zc_vzero_d(NW * ny * no * nvp, w(lbdrv))
      call zc_vzero_d(NW * ov, w(kr1c))
      stats = 0
      call N(zc_t3_pass1w)(w(jt3), nto, nvt, no, nv, npo, nvp, w(jotri), w(josgn), w(jvtri), w(jvsgn), &
                           nbp, w(lgpd), w(lgmq), ny, w(jgpp), w(kft), w(lbdrv), w(kyall), w(kr1c), &
                           w(kra), c_null_ptr, stats, w(kbuf))
    end if
  else
    call N(xzbpass1)(w, jt3, w(lgpd), w(lgmq), w(jgpp), w(lfov), w(lbdrv), w(kyall), w(kr1c), w(kra))
  end if
  if (.not. xzdry) call xztoc(3, 'pass 1 over T3')
  call xzrel(m1)

  ! ---- W_vvvo (without its <ab||cd> part), W_vooo, W_ovvo, F_vv, F_oo
  kwv = xzget(NW * nvp * ov, 'W_vvvo')
  m2 = xzmark()
  kg2 = xzget(NW * npo * ov, 'work [P][e][i]')
  kyb = xzget(NW * nba * nv * nv * no, 'batch of W_vvvo terms Y')
  ku = xzget(NW * nv ** 3, 'dressed <mb||ef> of one m')
  kxb = xzget(NW * nba * no * nv * nv, 'batch of W_vvvo terms X')
  if (.not. xzdry) then
    call N(xzbv)(no, nv, npo, nvp, nba, w(jt1), w(ktau), w(jovvq), w(jooov), w(jovvo), &
                 w(jopa), w(jopb), w(jvpa), w(jvpb), w(kwv), w(kg2), w(kyb))
    call N(zc_add_transposed)(w(kwv), w(lbdrv), nvp, ov, ov, nvp, one)
    call N(xzwvoovo)(no, nv, npo, nvp, w(kt2pp), w(looovd), w(lfov), w(jopa), w(jopb), w(kwv), w(kg2))
    call N(xzwvx)(no, nv, nvp, nba, w(jt2), w(kd), w(jvpa), w(jvpb), w(kwv), w(ka2), w(ku), &
                  w(kxb), w(kyb))
    call xztoc(3, 'W_vvvo')
    call N(xzwo)(no, nv, npo, nvp, w(jt2), w(kt2pp), w(looovd), w(kovood), w(kd), w(kyall), &
                 w(jopa), w(jopb), w(kwo), w(ktmp), w(ktmp2), w(kb2))
    call N(xzweff)(no, nv, nvp, w(jt2), w(lt2p), w(joovv), w(joovvp), w(kovvod), w(kfoo), w(kfvv), &
                   w(lwov), w(lfvv), w(lfoo), w(ka2), w(kb2), w(kx2))
    call xztoc(3, 'W_vooo, W_ovvo, F')
    ! the kept operands that the dressed <ov||vv> and W_vooo go into
    call N(xzlops)(no, nv, npo, nvp, nto, w(jt2), w(kt2pp), w(lfov), w(looovd), w(kooood), w(joovv), &
                   w(joovvp), w(jgpp), w(kd), w(kwo), w(lwov), w(lfoo), w(jopa), w(jopb), w(jvpa), &
                   w(jvpb), w(jofree), w(jopair), w(lgpd), w(lbdrv), w(ladrv), w(lvx), w(lwl), w(lwkl), &
                   w(lo6t), xzhhpair())
    call xztoc(3, 'kept operands, occ. operator')
  end if
  ! W_vvvo moves down to the place of the dressed <ov||vv>
  call xzrel(md)
  kd = xzget(NW * nvp * ov, 'W_vvvo')
  if (.not. xzdry) call N(xzmove)(nvp * ov, w(kwv), w(kd))
  kwv = kd

  ! ---- the pp operator of the T3 equation in the place of the bare
  !      <ab||de>, block by block; its part of W_vvvo on the way
  m2 = xzmark()
  kgq = jovvq
  kgh = jgpp
  kpib = jvpidx
  if (.not. dense) then
    kgq = xzget(NW * ov * nqmax, '<ov||vv> on the pairs of a group')
    kgh = xzget(NW * npo * nqmax, '<oo||vv> on the pairs of a group')
    kpib = xzget(nv * nv, 'pairs of a group')
  end if
  if (.not. xzdry) then
    do h = 0, npg - 1
      lo = vpoff(h)
      nqh = vpoff(h + 1) - lo
      if (nqh == 0) cycle
      if (.not. dense) call N(xzgrpcols)(no, nv, npo, nvp, lo, nqh, w(jovvq), w(jgpp), w(jvpidx), &
                                         w(kgq), w(kgh), w(kpib))
      ! ~<ab||de>, in place
      call N(xzdrrows)(no, nv, npo, nvp, nqh, nqh, w(jt1), w(ktau + NW * lo), w(jvvvv + NW * vvoff(h)), &
                       w(kgq), w(kgh), w(jvpa + lo), w(jvpb + lo), w(jvvvv + NW * vvoff(h)), w(ka2))
      ! W_vvvo[Q][e][i] += sum_f ~<ab||ef> t1[i][f]
      call N(zc_pair_t1_contract)(w(kwv + NW * lo * ov), w(jvvvv + NW * vvoff(h)), w(jt1), nqh, nv, nqh, &
                                  w(kpib), no)
      ! + sum_{m<n} t2[mn][ab] <mn||de> + 1/2 A[F_vv delta]
      if (npo > 0) call N(zc_gemm)(1_4, 0_4, nqh, nqh, npo, one, w(kt2pp + NW * lo), nvp, &
                                   w(jgpp + NW * lo), nvp, one, w(jvvvv + NW * vvoff(h)), nqh)
      call N(zc_pair_one_body_add)(w(jvvvv + NW * vvoff(h)), nqh, nqh, w(jvpa + lo), w(jvpb + lo), &
                                   w(kpib), nv, w(lfvv), nv, half)
    end do
    call N(xzlgmq)(no, nv, nvp, w(lt2p), w(kwv), w(lgmq))
    call xztoc(3, 'pp operator in place')
  end if
  call xzrel(m2)
  call xzrel(md)

  ! ---- the contractions of T3 with <ij||fg> and ~<ei||gh>
  kvq = lbdrv + NW * ny * no * nvp
  if (.not. xzdry) then
    call zc_vzero_d(NW * npo * nto * nv, w(lxq))
    call zc_vzero_d(NW * nvp * nto * no, w(lx2))
  end if
  if (dense) then
    ktv = xzget(NW * nvp * nbt * nv, 'gathered batch of T3')
    kz = xzget(NW * ov * nbt * nv, 'product of X2')
    if (.not. xzdry) then
      do i0 = 0, nto - 1, nbt
        nb = min(nbt, nto - i0)
        call N(zc_gather_vir_half)(w(jt3), nvt, w(jrows + i0), nb, w(jvtri), w(jvsgn), nv, nvp, w(ktv))
        if (npo > 0) call N(zc_gemm)(0_4, 0_4, npo, nb * nv, nvp, one, w(jgpp), nvp, w(ktv), nb * nv, zero, &
                                     w(lxq + NW * i0 * nv), nto * nv)
        call N(zc_gemm)(0_4, 0_4, ov, nb * nv, nvp, one, w(kvq), nvp, w(ktv), nb * nv, zero, w(kz), nb * nv)
        call N(xzlx2)(no, nv, nvp, nto, nb, i0, w(jvpa), w(jvpb), w(kz), w(lx2))
      end do
    end if
  else
    call N(xzlbcx)(w, jt3, w(jgpp), w(kvq), w(lxq), w(lx2))
  end if
  if (.not. xzdry) call xztoc(3, 'Xq, X2 from T3')
  call xzrel(ms)
end subroutine

! ---------------------------------------------------------------------------
! One Jacobi step of the Lambda equations.
!   kl1, kl2, kl3     lambda (in)
!   kl1n, kl2n        the new l1 and l2 (out)
!   kr3               the residual of l3, on return the new l3 (out)
!   dump              write the residuals to files (XZ_LDEBUG)
subroutine N(xzlamstep)(w, kl1, kl2, kl3, kl1n, kl2n, kr3, dump)
  use xzdata
  use xzmem
  use xzkern
  use xzdist
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kl1, kl2, kl3, kl1n, kl2n, kr3
  logical, intent(in) :: dump
  integer(8) :: p0, kex
  real(8) :: tst, tsg
  integer(8) :: m0, m1, ov, ov2, kk
  integer(8) :: kl2p, kl2pp, kyall, kdr, kd1, kg, kdo4, kgoo3, kgvv3, kgvv, kgoo, kdov, ku, kvp
  integer(8) :: kr1, kr2, kr2q, kr1x, kxij, ky1c, kg2, kpmv, ky2t, kbl, kbt, kz, ktv, kxp, kx2b
  integer(8) :: ky2x, ky1u, km, ka2, kb2, kyab, i0, nb, g, ni, na, h, lo, nqh, plo, np, khg, khx
  logical :: dense
  real(8) :: stats(4), t0
  character(len=256) :: dbg
  integer :: ldbg
  ZT :: one, zero, sixth
  one = 1
  zero = 0
  sixth = 1d0 / 6d0
  ov = no * nv
  ov2 = ov * ov
  kk = 2 * ny
  dense = ncls == 1
  tcur = 2
  m0 = xzmark()
  tst = omp_get_wtime()
  kl2p = xzget(NW * no * no * nvp, 'L2 [i][j][Q]')
  kl2pp = xzget(NW * npo * nvp, 'L2 [P][Q]')
  kyall = xzget(NW * npo * nv * ny, 'pass-1 result Y')
  kdr = xzget(NW * ov2, 'density of L3 with T3')
  kd1 = xzget(NW * ov, 'density of L2 with T3')
  kg = xzget(NW * nto * nto, 'Gram matrix')
  kdo4 = xzget(NW * npo * npo, 'Gram matrix on pairs')
  kgoo3 = xzget(NW * no * no, 'G_oo of L3')
  kgvv3 = xzget(NW * nv * nv, 'G_vv of L3')
  kgvv = xzget(NW * nv * nv, 'G_vv')
  kgoo = xzget(NW * no * no, 'G_oo')
  kdov = xzget(NW * ov, 'D_ov')
  ku = xzget(NW * 2 * ov, 'outer products U')
  kvp = xzget(NW * npo * 2 * nvp, 'outer products V')
  kr1 = xzget(NW * ov, 'R1')
  kr2 = xzget(NW * npo * nvp, 'R2')
  kr2q = xzget(NW * npo * nvp, 'R2 of the ladder pass')
  kr1x = xzget(NW * ov, 'R1 of the ladder pass')
  kxij = xzget(NW * no * no * nvp, 'R2 part [i][j][Q]')
  ky1c = xzget(NW * npo * ov, 'Y1')
  kg2 = xzget(NW * npo * ov, 'work [P][e][j]')
  kpmv = xzget(NW * ov, 'Y2 with the pp operator')
  ky2t = xzget(NW * ov, 'trace of Y2')

  ! ---- pass 1: l3 with t2, with the W's and with T3
  if (.not. xzdry) then
    call xztic()
    call N(xzl2forms)(no, nv, npo, nvp, w(kl2), w(jopa), w(jopb), w(jvpa), w(jvpb), w(kl2p), w(kl2pp))
    call zc_vzero_d(NW * ny * no * nvp, w(lbdrv))
    call zc_vzero_d(NW * ov2, w(kdr))
    call zc_vzero_d(NW * ov, w(kd1))
    call xztoc(2, 'L2 on pairs, zeros')
  end if
  m1 = xzmark()
  tsg = omp_get_wtime()
  dkind = 4
  if (dense) then
    kbl = xzget(NW * nbp * ov * nvp, 'pass-1 batch of L3')
    kbt = xzget(NW * nbp * ov * nvp, 'pass-1 batch of T3')
    if (.not. xzdry) then
      stats = 0
      call xztic()
      ! (the rows of the pairs of other ranks)
      if (nrank > 1) call zc_vzero_d(NW * npo * nv * ny, w(kyall))
      do p0 = 0, npo - 1, nbp
        if (.not. xzmine(p0)) cycle
        call xzbtic()
        call N(zc_l3_pass1r)(w(kl3), w(jt3), nto, nvt, no, nv, npo, nvp, w(jotri), w(josgn), w(jvtri), &
                             w(jvsgn), nbp, p0, min(p0 + nbp, npo), w(lgpd), w(lgmq), ny, w(kl2pp), w(lbdrv), &
                             w(kyall), w(kdr), w(kd1), stats, w(kbl), w(kbt))
        call xzbtoc(p0)
      end do
      call xztoc(2, 'pass 1')
      call xztpart(2, 'pass 1: gather', stats(1))
      call xztpart(2, 'pass 1: stacked GEMMs', stats(2))
      call xztpart(2, 'pass 1: density', stats(3))
    end if
  else
    call N(xzlbpass1)(w, kl3, jt3, w(lgpd), w(lgmq), w(kl2pp), w(lbdrv), w(kyall), w(kdr), w(kd1))
  end if
  dkind = 0
  call xzrel(m1)
  if (.not. xzdry) then
    dsec(1, 2) = dsec(1, 2) + omp_get_wtime() - tsg
    if (nrank > 1) then
      ! what the batches of all ranks made, on every rank
      tsg = omp_get_wtime()
      call xztic()
      call xzmpi_barrier()
      call xztoc(2, 'MPI: waiting for the ranks')
      dsec(9, 2) = dsec(9, 2) + omp_get_wtime() - tsg
      tsg = omp_get_wtime()
      call xzmpi_allsum_w(w(lbdrv), NW * ny * no * nvp)
      call xzmpi_allsum_w(w(kyall), NW * npo * nv * ny)
      call xzmpi_allsum_w(w(kdr), NW * ov2)
      call xzmpi_allsum_w(w(kd1), NW * ov)
      call xztoc(2, 'MPI: sums of pass 1')
      dsec(4, 2) = dsec(4, 2) + omp_get_wtime() - tsg
    end if
  end if

  ! ---- the Gram matrix of l3 and T3 (rank 0: only the residuals of rank 1 and 2 need it)
  if (.not. xzdry .and. myrank == 0) then
    tsg = omp_get_wtime()
    call xztic()
    if (dense) then
      call N(zc_gemm)(0_4, 1_4, nto, nto, nvt, one, w(kl3), nvt, w(jt3), nvt, zero, w(kg), nto)
    else
      call zc_vzero_d(NW * nto * nto, w(kg))
      do g = 0, ntg - 1
        ni = otoff(g + 1) - otoff(g)
        na = vtoff(g + 1) - vtoff(g)
        if (ni == 0 .or. na == 0) cycle
        call N(zc_gemm)(0_4, 1_4, ni, ni, na, one, w(kl3 + NW * boff(g)), na, w(jt3 + NW * boff(g)), na, &
                        zero, w(kg + NW * (otoff(g) * nto + otoff(g))), nto)
      end do
    end if
    call N(xzlgramd)(no, npo, nto, w(jotri), w(josgn), w(kg), w(kdo4), w(kgoo3))
    call xztoc(2, 'Gram matrix')
    dsec(8, 2) = dsec(8, 2) + omp_get_wtime() - tsg
  end if
  if (.not. xzdry) then
    call xztic()
    ! the operands of this iteration
    call N(xzlitops)(no, nv, npo, nvp, w(kl1), w(kl2), w(kl2p), w(kl2pp), w(lfov), w(looovd), w(jgpp), &
                     w(kyall), w(jopa), w(jopb), w(lbdrv), w(ladrv), w(kxij), w(ky1c), w(ku), w(kvp), w(kg2))
    call xztoc(2, 'operands of the iteration')
  end if

  ! ---- pass 2: ring, outer products, drive products
  call xztic()
  tsg = omp_get_wtime()
  if (.not. xzdry) call zc_vzero_d(NW * n3, w(kr3))
  m1 = xzmark()
  dkind = 5
  if (dense) then
    kbl = xzget(NW * nbp * ov * nvp, 'pass-2 batch of L3')
    kz = xzget(NW * nbp * ov * nvp, 'pass-2 product')
    if (.not. xzdry) then
      stats = 0
      do p0 = 0, npo - 1, nbp
        if (.not. xzmine(p0)) cycle
        call xzbtic()
        call N(zc_l3_pass2r)(w(kl3), nto, nvt, no, nv, npo, nvp, w(jotri), w(josgn), w(jofree), w(jopair), &
                             w(jvtri), w(jvsgn), w(jvfree), w(jvpair), nbp, p0, min(p0 + nbp, npo), w(lwl), &
                             w(ku), 2_8, w(kvp), w(ladrv), kk, w(lbdrv), w(kr3), stats, w(kbl), w(kz))
        call xzbtoc(p0)
      end do
    end if
  else
    kbl = xzget(NW * 2 * nbd, 'pass-2 operand B in blocks')
    if (.not. xzdry) call N(xzbbdk)(2_8, w(lbdrv), w(kbl))
    call N(xzbpass2)(w, kl3, kr3, w(lwl), w(ladrv), w(kbl), 2_8, 2_8, w(ku), w(kvp))
  end if
  dkind = 0
  call xzrel(m1)
  if (.not. xzdry) then
    call xztoc(2, 'pass 2')
    dsec(3, 2) = dsec(3, 2) + omp_get_wtime() - tsg
  end if
  tsg = omp_get_wtime()

  ! ---- the ladder with the transposed pp operator; Xq and X2
  if (.not. xzdry) then
    call zc_vzero_d(NW * npo * nvp, w(kr2q))
    call zc_vzero_d(NW * ov, w(kr1x))
  end if
  m1 = xzmark()
  dkind = 6
  if (dense) then
    ktv = xzget(NW * nvp * nbt * nv, 'ladder batch of L3')
    kxp = xzget(NW * nvp * nbt * nv, 'ladder product')
    kx2b = xzget(NW * nvp * nbt * no, 'batch of X2')
    if (.not. xzdry) then
      do i0 = 0, nto - 1, nbt
        nb = min(nbt, nto - i0)
        if (.not. xzmine(i0)) cycle
        call xzbtic()
        call xztic()
        call N(zc_gather_vir_half)(w(kl3), nvt, w(jrows + i0), nb, w(jvtri), w(jvsgn), nv, nvp, w(ktv))
        call xztoc(2, 'ladder: gather')
        call xztic()
        call N(zc_gemm)(1_4, 0_4, nvp, nb * nv, nvp, one, w(jvvvv), nvp, w(ktv), nb * nv, zero, &
                        w(kxp), nb * nv)
        call xztoc(2, 'ladder: GEMM')
        call xztic()
        call N(zc_reduce_pair_first_rows)(w(kr3), nvt, w(kxp), w(jrows + i0), nb, nv, 0_8, nvp, &
                                          w(jvfree), w(jvpair))
        ! r2q[P][Q] += Xq[P][(I, e)] Lv[Q][(I, e)];  r1x[a][i] += Lv[(Q, I)][a] X2[(Q, I)][i]
        if (npo > 0) call N(zc_gemm)(0_4, 1_4, npo, nvp, nb * nv, one, w(lxq + NW * i0 * nv), nto * nv, &
                                     w(ktv), nb * nv, one, w(kr2q), nvp)
        call N(xzlx2b)(no, nvp, nto, nb, i0, w(lx2), w(kx2b))
        call N(zc_gemm)(1_4, 0_4, nv, no, nvp * nb, one, w(ktv), nv, w(kx2b), no, one, w(kr1x), no)
        call xztoc(2, 'ladder: reduction, small')
        call xzbtoc(i0)
      end do
    end if
  else
    call N(xzlbladder)(w, kl3, kr3, w(lxq), w(lx2), w(kr2q), w(kr1x))
  end if
  dkind = 0
  call xzrel(m1)
  if (.not. xzdry) then
    dsec(2, 2) = dsec(2, 2) + omp_get_wtime() - tsg
    if (nrank > 1) then
      ! the parts of the residual of l3 and of what the ladder adds to the
      ! residuals of rank 1 and 2, summed onto rank 0
      tsg = omp_get_wtime()
      call xztic()
      call xzmpi_barrier()
      call xztoc(2, 'MPI: waiting for the ranks')
      dsec(10, 2) = dsec(10, 2) + omp_get_wtime() - tsg
      tsg = omp_get_wtime()
      if (.not. slices) call xzmpi_sum0_w(w(kr3), NW * n3)
      call xzmpi_sum0_w(w(kr2q), NW * npo * nvp)
      call xzmpi_sum0_w(w(kr1x), NW * ov)
      call xztoc(2, 'MPI: sum of the residual')
      dsec(5, 2) = dsec(5, 2) + omp_get_wtime() - tsg
    end if
    dsec(7, 2) = dsec(7, 2) + omp_get_wtime() - tst
  end if
  if (slices) then
    ! the rows of this rank: the transposed operator on occupied triples, the
    ! parts of the residual of all ranks; a rank other than 0 then does the
    ! Jacobi step on its rows and is done
    kex = xzget(min(xzmpi_chunk(), NW * max(smax, 1_8)), 'chunk of the exchange')
    if (.not. xzdry) then
      tsg = omp_get_wtime()
      call xztic()
      call N(xzslocc)(w, kl3, kr3, w(lo6t))
      call xztoc(2, 'hh ladder, F_oo')
      dsec(8, 2) = dsec(8, 2) + omp_get_wtime() - tsg
      tsg = omp_get_wtime()
      call xzmpi_slices_add(w(kr3), swoff, w(kex))
      call xztoc(2, 'MPI: residual to its rows')
      dsec(5, 2) = dsec(5, 2) + omp_get_wtime() - tsg
      if (myrank > 0) then
        call N(xzslupd)(w, kl3, kr3)
        call xztoc(2, 'L3 update')
      end if
    end if
  end if
  ! (the rest is rank 0's)
  if (.not. xzdry .and. myrank > 0) then
    call xzrel(m0)
    return
  end if
  tsg = omp_get_wtime()

  ! ---- the transposed operator on the occupied triples
  m1 = xzmark()
  if (xzhhpair()) then
    khg = xzget(NW * npo * no * xzhhcols(), 'hh ladder: L3 by pairs')
    khx = xzget(NW * npo * no * xzhhcols(), 'hh ladder: product')
  end if
  if (.not. xzdry .and. .not. slices) then
    call xztic()
    if (xzhhpair()) then
      call N(xzhhapply)(0_4, no, npo, nto, nvt, xzhhcols(), w(jotri), w(josgn), w(jofree), w(jopair), w(lo6t), &
                        w(kl3), w(kr3), w(khg), w(khx))
    else if (dense) then
      call N(zc_t3_occop)(w(lo6t), w(kl3), nto, nvt, ncb, c_null_ptr, w(kr3))
    else
      call N(xzbocc)(w, kl3, kr3, w(lo6t))
    end if
    call xztoc(2, 'hh ladder, F_oo')
  end if
  call xzrel(m1)

  ! ---- the residuals of rank 1 and 2
  m1 = xzmark()
  ky2x = xzget(NW * nv * nv * ov, 'Y2 on both orders of its pair')
  ky1u = xzget(NW * no * no * ov, 'Y1 on both orders of its pair')
  km = xzget(NW * ov2, 'R2 part [(i, a)][(j, b)]')
  ka2 = xzget(NW * max(ov2, no * no * ov), 'work A')
  kb2 = xzget(NW * max(ov2, no * no * ov), 'work B')
  kyab = xzget(NW * npo * nv * nv, 'R2 part [P][a][b]')
  if (.not. xzdry) then
    call xztic()
    call N(xzldens)(no, nv, nvp, w(kl2), w(jt2), w(lt2p), w(lbdrv), w(kdr), w(kd1), w(kgoo3), w(kgvv3), &
                    w(kgvv), w(kgoo), w(kdov))
    call xztoc(2, 'R1, R2: densities')
    ! Y2 on both orders of its pair, and with the columns of the pp operator
    call zc_vzero_d(NW * ov, w(kpmv))
    if (dense) then
      call N(zc_lam_pair_expand)(w(lbdrv), w(ky2x), nv, no, nv)
      call N(zc_lam_pair_matvec)(w(lbdrv), w(jvvvv), w(kpmv), no, nv)
    else
      call N(zc_lam_pair_expand_tab)(w(lbdrv), w(ky2x), nv, no, nv, nvp, w(jvpidx))
      do h = 0, npg - 1
        lo = vpoff(h)
        nqh = vpoff(h + 1) - lo
        if (nqh > 0) call N(zc_lam_pair_matvec_tab)(w(lbdrv), w(jvvvv + NW * vvoff(h)), w(kpmv), no, nv, &
                                                    nvp, lo, nqh, w(jvpa + lo), w(jvpb + lo))
      end do
    end if
    call xztoc(2, 'R1, R2: Y2 expand, pp matvec')
    call N(xzlr1)(no, nv, npo, nvp, w(kl1), w(kl2), w(kl2p), w(lfov), w(lfvv), w(lfoo), w(lwov), w(lgmq), &
                  w(lgpd), w(lvx), w(joovv), w(looovd), w(lwkl), w(kyall), w(kdr), w(kgvv), w(kgoo), &
                  w(kgvv3), w(kgoo3), w(kdov), w(kdo4), w(kpmv), w(ky2x), w(ky1u), w(kr1x), w(ky2t), &
                  w(jopa), w(jopb), w(kr1))
    call xztoc(2, 'R1')
    ! the Lambda_2 residual: the products with the pp operator and on pairs first
    call N(xzlr2a)(no, nv, npo, nvp, w(kl1), w(kl2), w(kl2p), w(lfvv), w(lfoo), w(lbdrv), w(joovv), &
                   w(joovvp), w(jgpp), w(looovd), w(kyall), w(kgvv), w(kgoo), w(kxij), w(kyab), &
                   w(jopa), w(jopb), w(jvpa), w(jvpb), w(kr2), w(kr2q))
    call xztoc(2, 'R2: part a')
    if (npo > 0) then
      if (dense) then
        call N(zc_gemm)(0_4, 0_4, npo, nvp, nvp, one, w(kl2pp), nvp, w(jvvvv), nvp, one, w(kr2), nvp)
      else
        do h = 0, npg - 1
          lo = vpoff(h)
          nqh = vpoff(h + 1) - lo
          plo = opoff(h)
          np = opoff(h + 1) - plo
          if (nqh > 0 .and. np > 0) call N(zc_gemm)(0_4, 0_4, np, nqh, nqh, one, &
            w(kl2pp + NW * (plo * nvp + lo)), nvp, w(jvvvv + NW * vvoff(h)), nqh, one, &
            w(kr2 + NW * (plo * nvp + lo)), nvp)
        end do
      end if
      call N(zc_gemm)(0_4, 0_4, npo, nvp, npo, one, w(lwkl), npo, w(kl2pp), nvp, one, w(kr2), nvp)
      call N(zc_gemm)(0_4, 0_4, npo, nvp, ov, one, w(ky1c), ov, w(lbdrv + NW * ny * no * nvp), nvp, one, &
                      w(kr2), nvp)
      call N(zc_gemm)(0_4, 0_4, npo, nvp, npo, sixth, w(kdo4), npo, w(jgpp), nvp, one, w(kr2), nvp)
      call N(zc_gemm)(0_4, 0_4, npo, nvp, ov, one, w(kg2), ov, w(lbdrv), nvp, one, w(kr2), nvp)
    end if
    call xztoc(2, 'R2: GEMMs on pairs')
    call N(xzlr2b)(no, nv, npo, nvp, w(kl1), w(kl2), w(lfov), w(lwov), w(lvx), w(joovv), w(looovd), &
                   w(kdr), w(ky2x), w(ky1u), w(jopa), w(jopb), w(jvpa), w(jvpb), w(km), w(ka2), w(kb2), &
                   w(kr2))
    call xztoc(2, 'R2: part b')
    if (dump) then
      call get_environment_variable('XZ_LDEBUG', dbg, ldbg)
      call xzdumpw(dbg(1:ldbg) // '.r1', NW * ov, w(kr1))
      call xzdumpw(dbg(1:ldbg) // '.r2', NW * npo * nvp, w(kr2))
      call xzdumpw(dbg(1:ldbg) // '.r3', NW * n3, w(kr3))
    end if
    ! ---- the Jacobi step
    call xztic()
    call N(xzlupd)(no, nv, npo, nvp, w(jeps), w(kl1), w(kl2), w(kr1), w(kr2), w(jopa), w(jopb), &
                   w(jvpa), w(jvpb), w(kl1n), w(kl2n))
    call xztoc(2, 'update of L1, L2')
    if (slices) then
      call N(xzslupd)(w, kl3, kr3)
    else if (dense) then
      call N(zc_t3_update)(w(kl3), w(kr3), nto, nvt, w(jeo3), w(jev3), c_null_ptr)
    else
      call N(xzbupd)(w, kl3, kr3)
    end if
    call xztoc(2, 'L3 update')
    dsec(8, 2) = dsec(8, 2) + omp_get_wtime() - tsg
    dsec(7, 2) = dsec(7, 2) + omp_get_wtime() - tsg
  end if
  call xzrel(m0)
end subroutine

! ---------------------------------------------------------------------------
! The kept arrays after the intermediates, for comparisons with the Python
! program (XZ_LDEBUG=prefix: one file per array, raw).
subroutine N(xzlamdump)(w, pre)
  use xzdata
  implicit none
  real(8) :: w(*)
  character(len=*), intent(in) :: pre
  integer(8) :: ov
  ov = no * nv
  call xzdumpw(pre // '.fov', NW * ov, w(lfov))
  call xzdumpw(pre // '.fvv', NW * nv * nv, w(lfvv))
  call xzdumpw(pre // '.foo', NW * no * no, w(lfoo))
  call xzdumpw(pre // '.wov', NW * ov * ov, w(lwov))
  call xzdumpw(pre // '.wkl', NW * npo * npo, w(lwkl))
  if (.not. xzhhpair()) call xzdumpw(pre // '.o6t', NW * nto * nto, w(lo6t))
  call xzdumpw(pre // '.gmq', NW * no * nvp * ny, w(lgmq))
  call xzdumpw(pre // '.gpd', NW * ny * npo * nv, w(lgpd))
  call xzdumpw(pre // '.bdrv', NW * 2 * ny * no * nvp, w(lbdrv))
  call xzdumpw(pre // '.adrv', NW * npo * nv * 2 * ny, w(ladrv))
  call xzdumpw(pre // '.vx', NW * nv * nv * ov, w(lvx))
  call xzdumpw(pre // '.xq', NW * npo * nto * nv, w(lxq))
  call xzdumpw(pre // '.x2', NW * nvp * nto * no, w(lx2))
  call xzdumpw(pre // '.wpp', NW * nvv, w(jvvvv))
end subroutine

! ===========================================================================
! The typed parts: loops over elements.

subroutine N(xzlconj)(n, a, b)
  implicit none
  integer(8), intent(in) :: n
  ZT, intent(in) :: a(n)
  ZT, intent(out) :: b(n)
  integer(8) :: i
  !$omp parallel do
  do i = 1, n
    b(i) = CONJ(a(i))
  end do
end subroutine

! 1/4 sum <ab||ij> l2[i][j][a][b]
subroutine N(xzlpseudo)(no, nv, l2, oovv, e)
  implicit none
  integer(8), intent(in) :: no, nv
  ZT, intent(in) :: l2(no * no * nv * nv), oovv(no * no * nv * nv)
  ZT, intent(out) :: e
  integer(8) :: i
  ZT :: s
  s = 0
  !$omp parallel do reduction(+:s)
  do i = 1, no * no * nv * nv
    s = s + CONJ(oovv(i)) * l2(i)
  end do
  e = 0.25d0 * s
end subroutine

! l2p[i][j][Q] and l2pp[P][Q] of l2[i][j][a][b]
subroutine N(xzl2forms)(no, nv, npo, nvp, l2, opa, opb, vpa, vpb, l2p, l2pp)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: l2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(out) :: l2p(0:nvp-1, 0:no-1, 0:no-1), l2pp(0:nvp-1, 0:npo-1)
  integer(8) :: i, j, q, p
  !$omp parallel do collapse(2) private(i, j, q)
  do i = 0, no - 1
    do j = 0, no - 1
      do q = 0, nvp - 1
        l2p(q, j, i) = l2(vpb(q), vpa(q), j, i)
      end do
    end do
  end do
  do p = 0, npo - 1
    l2pp(:, p) = l2p(:, opb(p), opa(p))
  end do
end subroutine

! ---------------------------------------------------------------------------
! The kept operands that are made from the dressed <mb||Q> and W_vooo:
!   gpd[f][P][e] = t2[P][e][f],  gpd[nv+j][P][e] = W_vooo[j][P][e]
!                  = wo[j][e][P] + sum_f f[j][f] t2[P][e][f]
!   bdrv[nv+no+e][i][Q] = ~<ei||Q> = - d[i][e][Q],  bdrv[2nv+no+n][k][Q] = <kn||Q>
!   adrv[P][c][f] = <P||cf>,  adrv[P][c][nv+m] = ~<P||cm> = - ~<P||mc>
!   vx[g][e][j][b] = ~<ej||bg> = - ~<je||bg>
!   wl[(c, k)][(e, m)] = wov[k][e][c][m]
!   wkl[K][P] = ~<K||P> + sum_Q <K||Q> t2[P][Q]
!   o6t: the transpose of the operator of the T3 equation on occupied triples
subroutine N(xzlops)(no, nv, npo, nvp, nto, t2, t2pp, fov, ooovd, ooood, oovv, oovvp, gpp, d, wo, wov, &
                     feoo, opa, opb, vpa, vpb, ofree, opair, gpd, bdrv, adrv, vx, wl, wkl, o6t, pairhh)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp, nto
  ZT, intent(in) :: t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: t2pp(0:nvp-1, 0:npo-1), gpp(0:nvp-1, 0:npo-1)
  ZT, intent(in) :: fov(0:nv-1, 0:no-1)
  ZT, intent(in) :: ooovd(0:nv-1, 0:no-1, 0:no-1, 0:no-1)      ! [m][n][i][e]
  ZT, intent(in) :: ooood(0:no-1, 0:no-1, 0:no-1, 0:no-1)      ! [m][n][i][j]
  ZT, intent(in) :: oovv(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: oovvp(0:nvp-1, 0:no-1, 0:no-1)             ! [m][n][Q]
  ZT, intent(in) :: d(0:nvp-1, 0:nv-1, 0:no-1)                 ! [m][b][Q]
  ZT, intent(in) :: wo(0:npo-1, 0:nv-1, 0:no-1)                ! [m][a][P]
  ZT, intent(in) :: wov(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)        ! [k][e][c][m]
  ZT, intent(in) :: feoo(0:no-1, 0:no-1)
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  integer(8), intent(in) :: ofree(*), opair(*)
  ZT, intent(out) :: gpd(0:nv-1, 0:npo-1, 0:nv+no-1)           ! [x][P][e]
  ZT :: bdrv(0:nvp-1, 0:no-1, 0:2*(nv+no)-1)                   ! [k][i][Q]
  ZT :: adrv(0:2*(nv+no)-1, 0:nv-1, 0:npo-1)                   ! [P][c][k]
  ZT, intent(out) :: vx(0:nv-1, 0:no-1, 0:nv-1, 0:nv-1)        ! [g][e][j][b]
  ZT, intent(out) :: wl(0:no-1, 0:nv-1, 0:no-1, 0:nv-1)        ! [c][k][e][m]
  ZT, intent(out) :: wkl(0:npo-1, 0:npo-1)                     ! [K][P]
  ZT :: o6t(*)                                               ! [I][J], or on pairs [K][P]
  logical, intent(in) :: pairhh
  integer(8) :: p, e, f, j, i, q, n, k, c, m, g, kk, it, jt
  ZT :: s, one, v
  one = 1
  !$omp parallel do private(p, e, f, j, s)
  do p = 0, npo - 1
    do f = 0, nv - 1
      do e = 0, nv - 1
        gpd(e, p, f) = t2(f, e, opb(p), opa(p))
      end do
    end do
    do j = 0, no - 1
      do e = 0, nv - 1
        s = wo(p, e, j)
        do f = 0, nv - 1
          s = s + fov(f, j) * t2(f, e, opb(p), opa(p))
        end do
        gpd(e, p, nv + j) = s
      end do
    end do
  end do
  !$omp parallel do private(e, i)
  do e = 0, nv - 1
    do i = 0, no - 1
      bdrv(:, i, nv + no + e) = -d(:, e, i)
    end do
  end do
  do n = 0, no - 1
    do k = 0, no - 1
      bdrv(:, k, 2 * nv + no + n) = oovvp(:, n, k)
    end do
  end do
  !$omp parallel do private(p, c, f, m)
  do p = 0, npo - 1
    do c = 0, nv - 1
      do f = 0, nv - 1
        adrv(f, c, p) = oovv(f, c, opb(p), opa(p))
      end do
      do m = 0, no - 1
        adrv(nv + m, c, p) = -ooovd(c, m, opb(p), opa(p))
      end do
    end do
  end do
  !$omp parallel do collapse(2) private(g, e, j, q)
  do g = 0, nv - 1
    do e = 0, nv - 1
      vx(:, :, e, g) = 0
    end do
  end do
  !$omp parallel do collapse(2) private(e, j, q)
  do e = 0, nv - 1
    do j = 0, no - 1
      do q = 0, nvp - 1
        vx(vpa(q), j, e, vpb(q)) = -d(q, e, j)
        vx(vpb(q), j, e, vpa(q)) = d(q, e, j)
      end do
    end do
  end do
  !$omp parallel do collapse(2) private(c, k, e, m)
  do c = 0, nv - 1
    do k = 0, no - 1
      do e = 0, nv - 1
        do m = 0, no - 1
          wl(m, e, k, c) = wov(m, c, e, k)
        end do
      end do
    end do
  end do
  do kk = 0, npo - 1
    do p = 0, npo - 1
      wkl(p, kk) = ooood(opb(p), opa(p), opb(kk), opa(kk))
    end do
  end do
  if (npo > 0) call N(zc_gemm)(0_4, 1_4, npo, npo, nvp, one, gpp, nvp, t2pp, nvp, one, wkl, npo)
  if (pairhh) then
    call N(xzwpair)(no, npo, opa, opb, wkl, feoo, o6t)
  else
    call N(xzo6)(no, npo, nto, ofree, opair, wkl, feoo, o6t)
    call N(xzsqtr)(nto, o6t)
  end if
end subroutine

! a square matrix transposed in place
subroutine N(xzsqtr)(n, a)
  implicit none
  integer(8), intent(in) :: n
  ZT, intent(inout) :: a(0:n-1, 0:n-1)
  integer(8) :: i, j
  ZT :: v
  !$omp parallel do private(i, j, v) schedule(dynamic, 16)
  do i = 0, n - 1
    do j = i + 1, n - 1
      v = a(j, i)
      a(j, i) = a(i, j)
      a(i, j) = v
    end do
  end do
end subroutine

! gmq[m][Q][n] = t2[m][n][Q],  gmq[m][Q][no+b] = W_vvvo[Q][b][m]
subroutine N(xzlgmq)(no, nv, nvp, t2p, wv, gmq)
  implicit none
  integer(8), intent(in) :: no, nv, nvp
  ZT, intent(in) :: t2p(0:nvp-1, 0:no-1, 0:no-1)               ! [m][n][Q]
  ZT, intent(in) :: wv(0:no-1, 0:nv-1, 0:nvp-1)                ! [Q][b][m]
  ZT, intent(out) :: gmq(0:no+nv-1, 0:nvp-1, 0:no-1)           ! [m][Q][y]
  integer(8) :: m, q, n, b
  !$omp parallel do private(m, q, n, b)
  do m = 0, no - 1
    do q = 0, nvp - 1
      do n = 0, no - 1
        gmq(n, q, m) = t2p(q, n, m)
      end do
      do b = 0, nv - 1
        gmq(no + b, q, m) = wv(m, b, q)
      end do
    end do
  end do
end subroutine

! x2[(e<f)][i0 + I][i] = z[e][i][I][f] - z[f][i][I][e]
subroutine N(xzlx2)(no, nv, nvp, nto, nb, i0, vpa, vpb, z, x2)
  implicit none
  integer(8), intent(in) :: no, nv, nvp, nto, nb, i0
  integer(8), intent(in) :: vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(in) :: z(0:nv-1, 0:nb-1, 0:no-1, 0:nv-1)          ! [e][i][I][f]
  ZT :: x2(0:no-1, 0:nto-1, 0:nvp-1)                           ! [Q][I][i]
  integer(8) :: q, it, i, e, f
  !$omp parallel do private(q, it, i, e, f)
  do q = 0, nvp - 1
    e = vpa(q)
    f = vpb(q)
    do it = 0, nb - 1
      do i = 0, no - 1
        x2(i, i0 + it, q) = z(f, it, i, e) - z(e, it, i, f)
      end do
    end do
  end do
end subroutine

! x2b[Q][I][i] = x2[Q][i0 + I][i] for the nb triples of a batch
subroutine N(xzlx2b)(no, nvp, nto, nb, i0, x2, x2b)
  implicit none
  integer(8), intent(in) :: no, nvp, nto, nb, i0
  ZT, intent(in) :: x2(0:no-1, 0:nto-1, 0:nvp-1)
  ZT, intent(out) :: x2b(0:no-1, 0:nb-1, 0:nvp-1)
  integer(8) :: q
  !$omp parallel do
  do q = 0, nvp - 1
    x2b(:, :, q) = x2(:, i0:i0 + nb - 1, q)
  end do
end subroutine

! ---------------------------------------------------------------------------
! From the Gram matrix g[I][J] = sum_A l3[I][A] t3[J][A]:
!   do4[P][P'] = 6 sum_m l3[m P] t3[m P']    (the triples that m makes with the pairs)
!   goo3[i][o] = 12 sum_P l3[i P] t3[o P]
subroutine N(xzlgramd)(no, npo, nto, otri, osgn, g, do4, goo3)
  implicit none
  integer(8), intent(in) :: no, npo, nto
  integer(8), intent(in) :: otri(0:npo-1, 0:no-1)
  integer(1), intent(in) :: osgn(0:npo-1, 0:no-1)
  ZT, intent(in) :: g(0:nto-1, 0:nto-1)                        ! [I][J]
  ZT, intent(out) :: do4(0:npo-1, 0:npo-1)                     ! [P][P']
  ZT, intent(out) :: goo3(0:no-1, 0:no-1)                      ! [i][o]
  integer(8) :: m, p, pp, i, o
  ZT :: s
  do4 = 0
  !$omp parallel do private(p, pp, m, s)
  do p = 0, npo - 1
    do pp = 0, npo - 1
      s = 0
      do m = 0, no - 1
        if (osgn(p, m) == 0 .or. osgn(pp, m) == 0) cycle
        s = s + (osgn(p, m) * osgn(pp, m)) * g(otri(pp, m), otri(p, m))
      end do
      do4(pp, p) = 6d0 * s
    end do
  end do
  do i = 0, no - 1
    do o = 0, no - 1
      s = 0
      do p = 0, npo - 1
        if (osgn(p, i) == 0 .or. osgn(p, o) == 0) cycle
        s = s + (osgn(p, i) * osgn(p, o)) * g(otri(p, o), otri(p, i))
      end do
      goo3(o, i) = 12d0 * s
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The operands that change with lambda, after pass 1:
!   xij[i][j][Q] = xacc[nv+j][i][Q]            (taken out before the rows go to l2)
!   bdrv[nv+m][i][Q] = l2[m][i][Q]
!   adrv[P][a][nv+no+e] = l2[P][e][a],  adrv[P][a][2nv+no+n] = y1[P][a][n]
!   y1c[P][a][n] = yall[P][a][n],  g2[P][e][j] = ~<P||ej>
!   u[(a, i)][0] = l1[i][a], u[(a, i)][1] = f[i][a];  vp[P][0][Q] = <P||Q>, vp[P][1][Q] = l2[P][Q]
subroutine N(xzlitops)(no, nv, npo, nvp, l1, l2, l2p, l2pp, fov, ooovd, gpp, yall, opa, opb, bdrv, adrv, &
                       xij, y1c, u, vp, g2)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: l1(0:nv-1, 0:no-1), l2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: l2p(0:nvp-1, 0:no-1, 0:no-1), l2pp(0:nvp-1, 0:npo-1)
  ZT, intent(in) :: fov(0:nv-1, 0:no-1)
  ZT, intent(in) :: ooovd(0:nv-1, 0:no-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: gpp(0:nvp-1, 0:npo-1)
  ZT, intent(in) :: yall(0:no+nv-1, 0:nv-1, 0:npo-1)           ! [P][a][y]
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1)
  ZT :: bdrv(0:nvp-1, 0:no-1, 0:2*(nv+no)-1)
  ZT :: adrv(0:2*(nv+no)-1, 0:nv-1, 0:npo-1)
  ZT, intent(out) :: xij(0:nvp-1, 0:no-1, 0:no-1)              ! [i][j][Q]
  ZT, intent(out) :: y1c(0:no-1, 0:nv-1, 0:npo-1)              ! [P][a][n]
  ZT, intent(out) :: u(0:1, 0:no-1, 0:nv-1)                    ! [a][i][x]
  ZT, intent(out) :: vp(0:nvp-1, 0:1, 0:npo-1)                 ! [P][x][Q]
  ZT, intent(out) :: g2(0:no-1, 0:nv-1, 0:npo-1)               ! [P][e][j]
  integer(8) :: i, j, m, p, a, e, n
  do i = 0, no - 1
    do j = 0, no - 1
      xij(:, j, i) = bdrv(:, i, nv + j)
    end do
  end do
  do m = 0, no - 1
    bdrv(:, :, nv + m) = l2p(:, :, m)
  end do
  !$omp parallel do private(p, a, e, n)
  do p = 0, npo - 1
    do a = 0, nv - 1
      do e = 0, nv - 1
        adrv(nv + no + e, a, p) = l2(a, e, opb(p), opa(p))
      end do
      do n = 0, no - 1
        adrv(2 * nv + no + n, a, p) = yall(n, a, p)
        y1c(n, a, p) = yall(n, a, p)
        g2(n, a, p) = -ooovd(a, n, opb(p), opa(p))
      end do
    end do
    vp(:, 0, p) = gpp(:, p)
    vp(:, 1, p) = l2pp(:, p)
  end do
  do a = 0, nv - 1
    do i = 0, no - 1
      u(0, i, a) = l1(a, i)
      u(1, i, a) = fov(a, i)
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The densities of the residuals of rank 1 and 2:
!   gvv3[a][g] = 4 sum_i dr[a][i][g][i]
!   gvv[e][g] = 1/2 sum_mnf l2[m][n][e][f] t2[m][n][f][g] - gvv3[e][g] / 12
!   goo[m][o] = 1/2 sum_nef l2[m][n][e][f] t2[n][o][e][f] - goo3[m][o] / 12
!   dov[j][h] = d1[h][j] - sum_{i,Q} y2[h][i][Q] t2[j][i][Q]
subroutine N(xzldens)(no, nv, nvp, l2, t2, t2p, y2, dr, d1, goo3, gvv3, gvv, goo, dov)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, nvp
  ZT, intent(in) :: l2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1), t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: t2p(0:nvp-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: y2(0:nvp-1, 0:no-1, 0:nv-1)                ! [h][i][Q]
  ZT, intent(in) :: dr(0:no-1, 0:nv-1, 0:no-1, 0:nv-1)         ! [a][i][g][o]
  ZT, intent(in) :: d1(0:no-1, 0:nv-1)                         ! [g][o]
  ZT, intent(in) :: goo3(0:no-1, 0:no-1)
  ZT, intent(out) :: gvv3(0:nv-1, 0:nv-1), gvv(0:nv-1, 0:nv-1), goo(0:no-1, 0:no-1)
  ZT, intent(out) :: dov(0:nv-1, 0:no-1)                       ! [j][h]
  integer(8) :: a, g, i, e, m, n, f, o, j, h, q
  ZT :: s, mhalf, zero
  mhalf = -0.5d0
  zero = 0
  ! 1/2 sum_mnf l2[m][n][e][f] t2[m][n][f][g] = -1/2 sum_mnf l2[m][n][f][e] t2[m][n][f][g]
  call N(zc_gemm)(1_4, 0_4, nv, nv, no * no * nv, mhalf, l2, nv, t2, nv, zero, gvv, nv)
  ! 1/2 sum_nef l2[m][n][e][f] t2[n][o][e][f] = -1/2 sum_nef l2[m][n][e][f] t2[o][n][e][f]
  call N(zc_gemm)(0_4, 1_4, no, no, no * nv * nv, mhalf, l2, no * nv * nv, t2, no * nv * nv, zero, goo, no)
  !$omp parallel do collapse(2) private(e, g, i, s)
  do e = 0, nv - 1
    do g = 0, nv - 1
      s = 0
      do i = 0, no - 1
        s = s + dr(i, g, i, e)
      end do
      gvv3(g, e) = 4d0 * s
      gvv(g, e) = gvv(g, e) - gvv3(g, e) / 12d0
    end do
  end do
  goo = goo - goo3 / 12d0
  !$omp parallel do collapse(2) private(j, h, i, q, s)
  do j = 0, no - 1
    do h = 0, nv - 1
      s = d1(j, h)
      do i = 0, no - 1
        do q = 0, nvp - 1
          s = s - y2(q, i, h) * t2p(q, i, j)
        end do
      end do
      dov(h, j) = s
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The Lambda_1 residual r1[i][a] (lambda_opt.residuals; the terms in its
! order).  y1u[i][m][a][n] (Y1 on both orders of its pair) is made here for
! the Lambda_2 residual too; y2t: work.
subroutine N(xzlr1)(no, nv, npo, nvp, l1, l2, l2p, fov, fvv, foo, wov, gmq, gpd, vx, oovv, ooovd, wkl, &
                    yall, dr, gvv, goo, gvv3, goo3, dov, do4, pmv, y2x, y1u, r1x, y2t, opa, opb, r1)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: l1(0:nv-1, 0:no-1), l2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: l2p(0:nvp-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: fov(0:nv-1, 0:no-1), fvv(0:nv-1, 0:nv-1), foo(0:no-1, 0:no-1)
  ZT, intent(in) :: wov(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)        ! [k][e][c][m]
  ZT, intent(in) :: gmq(0:no+nv-1, 0:nvp-1, 0:no-1)            ! [m][Q][y]
  ZT, intent(in) :: gpd(0:nv-1, 0:npo-1, 0:nv+no-1)            ! [x][P][e]
  ZT, intent(in) :: vx(0:nv-1, 0:no-1, 0:nv-1, 0:nv-1)         ! [g][e][j][b]
  ZT, intent(in) :: oovv(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: ooovd(0:nv-1, 0:no-1, 0:no-1, 0:no-1)      ! [m][n][i][e]
  ZT, intent(in) :: wkl(0:npo-1, 0:npo-1)                      ! [K][P]
  ZT, intent(in) :: yall(0:no+nv-1, 0:nv-1, 0:npo-1)           ! [P][a][y]
  ZT, intent(in) :: dr(0:no-1, 0:nv-1, 0:no-1, 0:nv-1)         ! [a][i][g][o]
  ZT, intent(in) :: gvv(0:nv-1, 0:nv-1), goo(0:no-1, 0:no-1), gvv3(0:nv-1, 0:nv-1), goo3(0:no-1, 0:no-1)
  ZT, intent(in) :: dov(0:nv-1, 0:no-1)                        ! [j][h]
  ZT, intent(in) :: do4(0:npo-1, 0:npo-1)                      ! [P][P']
  ZT, intent(in) :: pmv(0:nv-1, 0:no-1)                        ! [i][a]
  ZT, intent(in) :: y2x(0:nv-1, 0:no-1, 0:nv-1, 0:nv-1)        ! [g][e][i][a]
  ZT, intent(out) :: y1u(0:no-1, 0:nv-1, 0:no-1, 0:no-1)       ! [i][m][a][n]
  ZT, intent(in) :: r1x(0:no-1, 0:nv-1)                        ! [a][i]
  ZT :: y2t(0:nv-1, 0:no-1)                                    ! [i][e]
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1)
  ZT, intent(out) :: r1(0:nv-1, 0:no-1)                        ! [i][a]
  integer(8) :: i, a, e, m, j, h, g, o, p, pp, n, f, kk
  ZT :: s, v, one
  one = 1

  y1u = 0
  do p = 0, npo - 1
    y1u(:, :, opb(p), opa(p)) = yall(0:no-1, :, p)
    y1u(:, :, opa(p), opb(p)) = -yall(0:no-1, :, p)
  end do
  do i = 0, no - 1
    do e = 0, nv - 1
      s = 0
      do h = 0, nv - 1
        s = s + y2x(e, i, h, h)
      end do
      y2t(e, i) = s
    end do
  end do

  !$omp parallel do private(i, a, e, m, j, h, g, o, f, p, s)
  do i = 0, no - 1
    do a = 0, nv - 1
      s = fov(a, i) + r1x(i, a) - pmv(a, i)
      do e = 0, nv - 1
        s = s + l1(e, i) * fvv(a, e) + 0.5d0 * y2t(e, i) * fvv(a, e) &
              - (fov(e, i) * gvv3(e, a)) / 12d0
      end do
      do m = 0, no - 1
        s = s - foo(m, i) * l1(a, m) - (goo3(m, i) * fov(a, m)) / 12d0
      end do
      do m = 0, no - 1
        do e = 0, nv - 1
          s = s + l1(e, m) * wov(m, a, e, i)
        end do
      end do
      ! l2 with W_vooo
      do p = 0, npo - 1
        do e = 0, nv - 1
          s = s + l2(e, a, opb(p), opa(p)) * gpd(e, p, nv + i)
        end do
      end do
      do j = 0, no - 1
        do h = 0, nv - 1
          s = s + dov(h, j) * oovv(h, a, j, i)
        end do
        do m = 0, no - 1
          s = s - goo(j, m) * ooovd(a, m, j, i)
        end do
      end do
      ! the density of l3 and T3 with <ij||gm>
      do m = 0, no - 1
        do g = 0, nv - 1
          do j = 0, no - 1
            s = s - dr(j, g, m, a) * ooovd(g, m, j, i)
          end do
        end do
      end do
      ! Y1 with W_ovvo
      do m = 0, no - 1
        do e = 0, nv - 1
          do o = 0, no - 1
            s = s - y1u(o, e, m, i) * wov(m, a, e, o)
          end do
        end do
      end do
      r1(a, i) = s
    end do
    ! the sums over two virtual indices, all a at once:
    ! G_vv with ~<ei||ag>, F_vv with Y2
    do g = 0, nv - 1
      do e = 0, nv - 1
        r1(:, i) = r1(:, i) + gvv(g, e) * vx(:, i, e, g) + (0.5d0 * fvv(g, e)) * y2x(:, i, e, g)
      end do
    end do
    ! the density of l3 and T3 with ~<eo||ah>, and Y2 with W_ovvo
    do g = 0, nv - 1
      do e = 0, nv - 1
        do o = 0, no - 1
          r1(:, i) = r1(:, i) + dr(o, g, i, e) * vx(:, o, e, g) - wov(o, g, e, i) * y2x(:, o, e, g)
        end do
      end do
    end do
  end do
  ! l2 with W_vvvo: sum_{j,Q} l2[i][j][Q] gmq[(j, Q)][no + a]
  call N(zc_gemm)(0_4, 0_4, no, nv, no * nvp, one, l2p, no * nvp, gmq(no, 0, 0), no + nv, one, r1, nv)
  ! the Gram matrix on pairs with ~<P'||am>, and Y1 with W_oooo
  do p = 0, npo - 1
    m = opa(p)
    n = opb(p)
    do pp = 0, npo - 1
      v = do4(pp, p) / 6d0
      do a = 0, nv - 1
        r1(a, m) = r1(a, m) - v * ooovd(a, n, opb(pp), opa(pp))
        r1(a, n) = r1(a, n) + v * ooovd(a, m, opb(pp), opa(pp))
      end do
    end do
  end do
  do kk = 0, npo - 1
    i = opa(kk)
    j = opb(kk)
    do p = 0, npo - 1
      v = wkl(p, kk)
      do a = 0, nv - 1
        r1(a, i) = r1(a, i) - yall(j, a, p) * v
        r1(a, j) = r1(a, j) + yall(i, a, p) * v
      end do
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The Lambda_2 residual, first part: r2[P][Q] = <P||Q> + P(ij) xij + P(ab) yab
! + r2q (xij and yab completed here).
subroutine N(xzlr2a)(no, nv, npo, nvp, l1, l2, l2p, fvv, foo, bdrv, oovv, oovvp, gpp, ooovd, yall, &
                     gvv, goo, xij, yab, opa, opb, vpa, vpb, r2, r2q)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: l1(0:nv-1, 0:no-1), l2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: l2p(0:nvp-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: fvv(0:nv-1, 0:nv-1), foo(0:no-1, 0:no-1)
  ZT, intent(in) :: bdrv(0:nvp-1, 0:no-1, 0:2*(nv+no)-1)       ! [k][i][Q]
  ZT, intent(in) :: oovv(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: oovvp(0:nvp-1, 0:no-1, 0:no-1)             ! [m][n][Q]
  ZT, intent(in) :: gpp(0:nvp-1, 0:npo-1)
  ZT, intent(in) :: ooovd(0:nv-1, 0:no-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: yall(0:no+nv-1, 0:nv-1, 0:npo-1)
  ZT, intent(in) :: gvv(0:nv-1, 0:nv-1), goo(0:no-1, 0:no-1)
  ZT :: xij(0:nvp-1, 0:no-1, 0:no-1)                           ! [i][j][Q]
  ZT :: yab(0:nv-1, 0:nv-1, 0:npo-1)                           ! [P][a][b]
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(out) :: r2(0:nvp-1, 0:npo-1)
  ZT, intent(in) :: r2q(0:nvp-1, 0:npo-1)
  integer(8) :: i, j, e, m, o, p, q, a, b, g, m1, n1
  ZT :: s
  !$omp parallel do collapse(2) private(i, j, e, m, o)
  do i = 0, no - 1
    do j = 0, no - 1
      do e = 0, nv - 1
        xij(:, j, i) = xij(:, j, i) + l1(e, i) * bdrv(:, j, nv + no + e)
      end do
      do m = 0, no - 1
        xij(:, j, i) = xij(:, j, i) - l2p(:, m, i) * foo(m, j)
      end do
      do o = 0, no - 1
        xij(:, j, i) = xij(:, j, i) - goo(o, i) * oovvp(:, o, j)
      end do
    end do
  end do
  !$omp parallel do private(p, a, b, m, e, g, m1, n1, s)
  do p = 0, npo - 1
    m1 = opa(p)
    n1 = opb(p)
    do a = 0, nv - 1
      do b = 0, nv - 1
        s = yall(no + b, a, p)
        do m = 0, no - 1
          s = s - l1(a, m) * ooovd(b, m, n1, m1)
        end do
        do e = 0, nv - 1
          s = s + 0.5d0 * l2(e, a, n1, m1) * fvv(b, e) - gvv(e, a) * oovv(e, b, n1, m1)
        end do
        yab(b, a, p) = s
      end do
    end do
  end do
  !$omp parallel do private(p, q, i, j, a, b)
  do p = 0, npo - 1
    i = opa(p)
    j = opb(p)
    do q = 0, nvp - 1
      a = vpa(q)
      b = vpb(q)
      r2(q, p) = gpp(q, p) + r2q(q, p) + xij(q, j, i) - xij(q, i, j) + yab(b, a, p) - yab(a, b, p)
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The Lambda_2 residual, last part: P(ij) P(ab) M[(i, a)][(j, b)],
!   M = l1[i][a] f[j][b] + sum_kc l2[i][k][a][c] wov[j][c][b][k]
!       + sum_ge y2x[g][e][i][a] vx[g][e][j][b] + sum_go dr[a][i][g][o] <jo||bg>
!       + sum_mo y1u[i][m][a][o] ~<jo||bm>
! m4, a2, b2: work (no nv)**2 each.
subroutine N(xzlr2b)(no, nv, npo, nvp, l1, l2, fov, wov, vx, oovv, ooovd, dr, y2x, y1u, opa, opb, vpa, &
                     vpb, m4, a2, b2, r2)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: l1(0:nv-1, 0:no-1), l2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: fov(0:nv-1, 0:no-1)
  ZT, intent(in) :: wov(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)        ! [k][e][c][m]
  ZT, intent(in) :: vx(*)                                      ! [(g, e)][(j, b)]
  ZT, intent(in) :: oovv(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: ooovd(0:nv-1, 0:no-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: dr(*)                                      ! [(a, i)][(g, o)]
  ZT, intent(in) :: y2x(*)                                     ! [(g, e)][(i, a)]
  ZT, intent(in) :: y1u(0:no-1, 0:nv-1, 0:no-1, 0:no-1)        ! [i][m][a][n]
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  ZT :: m4(0:nv-1, 0:no-1, 0:nv-1, 0:no-1)                     ! [(i, a)][(j, b)]
  ZT :: a2(0:nv-1, 0:no-1, 0:nv-1, 0:no-1)
  ZT :: b2(0:nv-1, 0:no-1, 0:no-1, 0:nv-1)
  ZT, intent(inout) :: r2(0:nvp-1, 0:npo-1)
  integer(8) :: i, a, j, b, k, c, m, o, g, p, q, ov
  ZT :: s, one, zero, mone
  one = 1
  zero = 0
  mone = -1
  ov = no * nv
  ! l1 f, and Y1 with ~<jo||bm> = - ~<jo||mb>:
  !   a2[(i, a)][(m, o)] = y1u[i][m][a][o],  b2[(m, o)][(j, b)] = ~<jo||mb>
  !$omp parallel do collapse(2) private(i, a, j, b)
  do i = 0, no - 1
    do a = 0, nv - 1
      do j = 0, no - 1
        do b = 0, nv - 1
          m4(b, j, a, i) = l1(a, i) * fov(b, j)
        end do
      end do
    end do
  end do
  call N(xzly1ops)(no, nv, y1u, ooovd, a2, b2)
  call N(zc_gemm)(0_4, 0_4, ov, ov, no * no, mone, a2, no * no, b2, ov, one, m4, ov)
  ! l2 with W_ovvo: a2[(i, a)][(k, c)] = l2[i][k][a][c],  b2[(k, c)][(j, b)] = wov[j][c][b][k]
  !$omp parallel do collapse(2) private(i, a, k, c)
  do i = 0, no - 1
    do a = 0, nv - 1
      do k = 0, no - 1
        do c = 0, nv - 1
          a2(c, k, a, i) = l2(c, a, k, i)
        end do
      end do
    end do
  end do
  call N(xzlwjb)(no, nv, wov, b2)
  call N(zc_gemm)(0_4, 0_4, ov, ov, ov, one, a2, ov, b2, ov, one, m4, ov)
  ! Y2 with ~<ej||bg>
  call N(zc_gemm)(1_4, 0_4, ov, ov, nv * nv, one, y2x, ov, vx, ov, one, m4, ov)
  ! the density with <jo||bg>: a2[(a, i)][(j, b)] = sum_go dr[(a, i)][(g, o)] <jo||bg>
  call N(xzlgojb)(no, nv, oovv, b2)
  call N(zc_gemm)(0_4, 0_4, ov, ov, ov, one, dr, ov, b2, ov, zero, a2, ov)
  call N(xzladdai)(no, nv, a2, m4)
  !$omp parallel do private(p, q, i, j, a, b)
  do p = 0, npo - 1
    i = opa(p)
    j = opb(p)
    do q = 0, nvp - 1
      a = vpa(q)
      b = vpb(q)
      r2(q, p) = r2(q, p) + m4(b, j, a, i) - m4(b, i, a, j) - m4(a, j, b, i) + m4(a, i, b, j)
    end do
  end do
end subroutine

! a[(i, a)][(m, o)] = y1u[i][m][a][o],  b[(m, o)][(j, b)] = ~<jo||mb>
subroutine N(xzly1ops)(no, nv, y1u, ooovd, a, b)
  implicit none
  integer(8), intent(in) :: no, nv
  ZT, intent(in) :: y1u(0:no-1, 0:nv-1, 0:no-1, 0:no-1)        ! [i][m][a][n]
  ZT, intent(in) :: ooovd(0:nv-1, 0:no-1, 0:no-1, 0:no-1)      ! [j][o][m][b]
  ZT, intent(out) :: a(0:no-1, 0:no-1, 0:nv-1, 0:no-1)         ! [i][a][m][o]
  ZT, intent(out) :: b(0:nv-1, 0:no-1, 0:no-1, 0:no-1)         ! [m][o][j][b]
  integer(8) :: i, c, m, o, j
  !$omp parallel do collapse(2) private(i, c, m, o)
  do i = 0, no - 1
    do c = 0, nv - 1
      do m = 0, no - 1
        do o = 0, no - 1
          a(o, m, c, i) = y1u(o, c, m, i)
        end do
      end do
    end do
  end do
  !$omp parallel do collapse(2) private(m, o, j)
  do m = 0, no - 1
    do o = 0, no - 1
      do j = 0, no - 1
        b(:, j, o, m) = ooovd(:, m, o, j)
      end do
    end do
  end do
end subroutine

! b2[(k, c)][(j, b)] = wov[j][c][b][k]
subroutine N(xzlwjb)(no, nv, wov, b2)
  implicit none
  integer(8), intent(in) :: no, nv
  ZT, intent(in) :: wov(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)        ! [k][e][c][m]
  ZT, intent(out) :: b2(0:nv-1, 0:no-1, 0:nv-1, 0:no-1)        ! [k][c][j][b]
  integer(8) :: k, c, j, b
  !$omp parallel do collapse(2) private(k, c, j, b)
  do k = 0, no - 1
    do c = 0, nv - 1
      do j = 0, no - 1
        do b = 0, nv - 1
          b2(b, j, c, k) = wov(k, b, c, j)
        end do
      end do
    end do
  end do
end subroutine

! b2[(g, o)][(j, b)] = <jo||bg>
subroutine N(xzlgojb)(no, nv, oovv, b2)
  implicit none
  integer(8), intent(in) :: no, nv
  ZT, intent(in) :: oovv(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(out) :: b2(0:nv-1, 0:no-1, 0:no-1, 0:nv-1)        ! [g][o][j][b]
  integer(8) :: g, o, j, b
  !$omp parallel do collapse(2) private(g, o, j, b)
  do g = 0, nv - 1
    do o = 0, no - 1
      do j = 0, no - 1
        do b = 0, nv - 1
          b2(b, j, o, g) = oovv(g, b, o, j)
        end do
      end do
    end do
  end do
end subroutine

! m4[(i, a)][(j, b)] += x[(a, i)][(j, b)]
subroutine N(xzladdai)(no, nv, x, m4)
  implicit none
  integer(8), intent(in) :: no, nv
  ZT, intent(in) :: x(0:no*nv-1, 0:no-1, 0:nv-1)               ! [a][i][(j, b)]
  ZT, intent(inout) :: m4(0:no*nv-1, 0:nv-1, 0:no-1)           ! [i][a][(j, b)]
  integer(8) :: i, a
  !$omp parallel do collapse(2) private(i, a)
  do i = 0, no - 1
    do a = 0, nv - 1
      m4(:, a, i) = m4(:, a, i) + x(:, i, a)
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The Jacobi step of l1 and l2: l + r / D.
subroutine N(xzlupd)(no, nv, npo, nvp, eps, l1, l2, r1, r2, opa, opb, vpa, vpb, l1n, l2n)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  real(8), intent(in) :: eps(0:no+nv-1)
  ZT, intent(in) :: l1(0:nv-1, 0:no-1), l2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: r1(0:nv-1, 0:no-1), r2(0:nvp-1, 0:npo-1)
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(out) :: l1n(0:nv-1, 0:no-1), l2n(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  integer(8) :: i, a, p, q, j, b
  ZT :: v
  do i = 0, no - 1
    do a = 0, nv - 1
      l1n(a, i) = l1(a, i) + r1(a, i) / (eps(i) - eps(no + a))
    end do
  end do
  l2n = l2
  !$omp parallel do private(p, q, i, j, a, b, v)
  do p = 0, npo - 1
    i = opa(p)
    j = opb(p)
    do q = 0, nvp - 1
      a = vpa(q)
      b = vpb(q)
      v = r2(q, p) / (eps(i) + eps(j) - eps(no + a) - eps(no + b))
      l2n(b, a, j, i) = l2n(b, a, j, i) + v
      l2n(a, b, j, i) = l2n(a, b, j, i) - v
      l2n(b, a, i, j) = l2n(b, a, i, j) - v
      l2n(a, b, i, j) = l2n(a, b, i, j) + v
    end do
  end do
end subroutine
