!
! One Jacobi step of CCSDT on packed T3 (type-generic).  The algorithm of
! src/zccsdt/cc/ccsdt_opt.py (update_amps with the dressed <ab||cd> in slabs
! of rows) with the kernels of csrc/ccsdt_t3.c and csrc/t3pack.c:
!
!   T1 dressing of the integrals
!   pass 1     T3 -> T1, T2, and the T3 parts of W_vvvo and W_vooo
!   W_vvvo, W_vooo, W_ovvo, F_vv, F_oo
!   sweep      slabs of rows of the dressed <ab||cd>: their part of W_vvvo,
!              the CCSD ladder, the pp ladder of T3 (+ F_vv)
!   CCSD part  new T1 and T2
!   pass 2     ring and drive terms of T3
!   hh ladder and F_oo: one operator on the occupied triples
!   update     new T3 = T3 + R / D, written over R
!
! Everything that is not a sum over T3 or <ab||cd> is done on dense arrays,
! with the tables of pairs and triples that the layout made.  T3 and
! <ab||cd> are on the blocks of the groups of the symmetry labels: with one
! class (no labels) that is the whole packed array and the dense kernels
! run; with more classes the passes of xzblk.inc.
!
! Every array is a region of the work array w, taken from the stack of
! xzmem.  In a dry run (xzdry) the routine goes through its allocations and
! computes nothing.
!
!   kt1, kt2, kt3     the amplitudes (in)
!   kt1n, kt2n        the new T1 and T2 (out)
!   kr3               the residual of T3, on return the new T3 (out)
!
subroutine N(xzstep)(w, kt1, kt2, kt3, kt1n, kt2n, kr3)
  use xzdata
  use xzmem
  use xzkern
  use xzdist
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kt1, kt2, kt3, kt1n, kt2n, kr3
  integer(8) :: p0, kex
  real(8) :: tst, tsg
  integer(8) :: m0, m1, m2, md, ov, ov2
  integer(8) :: kfoo, kfov, kfvo, kfvv, kpvv, kfevv, kfeoo, kft
  integer(8) :: kooovd, kooood, kovvod, kovood, kk, ktmp, ktmp2
  integer(8) :: kt2p, kt2pp, ktau, ktt
  integer(8) :: kgpd, kyall, kr1c, kr1v, kra, kr2, klad, kwkl
  integer(8) :: kweff, kwo, kadrv, ko6, kwring, ka2, kb2, kx2
  integer(8) :: kd, kxacc, kgmq, kbuf, kwv, kg2, kyb, ku, kxb
  integer(8) :: kslab, ktv, kxp, kbd, ktb, kz, kgq, kpib, kgh, nbdw
  integer(8) :: q0, nq, i0, nb, h, lo, nqh
  integer(8) :: khg, khx
  integer(8) :: kml, kmh, plo, nph, kwk, khh, kee, kzl, kyl
  logical :: dense, pairhh
  real(8) :: stats(4), t0
  ZT :: one, zero, half
  one = 1
  zero = 0
  half = 0.5d0
  ov = no * nv
  ov2 = ov * ov
  dense = ncls == 1
  tcur = 1
  m0 = xzmark()
  tst = omp_get_wtime()

  ! ---- the arrays of the size of T2 and smaller
  kfoo = xzget(NW * no * no, 'F_oo')
  kfov = xzget(NW * ov, 'F_ov')
  kfvo = xzget(NW * ov, 'F_vo')
  kfvv = xzget(NW * nv * nv, 'F_vv')
  kpvv = xzget(NW * nv * nv, 'F_vv work')
  kfevv = xzget(NW * nv * nv, 'F_vv of T3')
  kfeoo = xzget(NW * no * no, 'F_oo of T3')
  kft = xzget(NW * merge(1_8, ov, ccsdonly), 'F_ov transposed')
  kooovd = xzget(NW * no * no * ov, 'dressed <oo||ov>')
  kooood = xzget(NW * no ** 4, 'dressed <oo||oo>')
  kovvod = xzget(NW * ov2, 'dressed <ov||vo>')
  kovood = xzget(NW * merge(1_8, no * no * ov, ccsdonly), 'dressed <ov||oo>')
  kk = xzget(NW * merge(1_8, ov2, ccsdonly), 'work of the dressing')
  ktmp = xzget(NW * merge(1_8, no * no * ov, ccsdonly), 'work of W_vooo')
  ktmp2 = xzget(NW * merge(1_8, no * no * ov, ccsdonly), 'work of W_vooo 2')
  kt2p = xzget(NW * no * no * nvp, 'T2 [i][j][Q]')
  kt2pp = xzget(NW * npo * nvp, 'T2 [P][Q]')
  ktau = xzget(NW * npo * nvp, 'tau1')
  ktt = xzget(NW * npo * nvp, 'T2 + tau1')
  kgpd = xzget(NW * merge(1_8, ny * npo * nv, ccsdonly), 'pass-1 operand')
  kyall = xzget(NW * merge(1_8, npo * nv * ny, ccsdonly), 'pass-1 result Y')
  kr1c = xzget(NW * merge(1_8, ov, ccsdonly), 'T3 part of R1')
  kr1v = xzget(NW * no * ny, 'vovv part of R1')
  kra = xzget(NW * merge(1_8, npo * nvp, ccsdonly), 'T3 part of R2')
  kr2 = xzget(NW * npo * nvp, 'R2')
  klad = xzget(NW * npo * nvp, 'CCSD ladder')
  kwkl = xzget(NW * npo * npo, 'W_oooo')
  kweff = xzget(NW * ov2, 'W_ovvo')
  kwo = xzget(NW * merge(1_8, ov * npo, ccsdonly), 'W_vooo')
  kadrv = xzget(NW * merge(1_8, npo * nv * ny, ccsdonly), 'pass-2 operand A')
  pairhh = xzhhpair()
  ko6 = xzget(NW * merge(npo * npo, nto * nto, pairhh), 'occupied operator')
  if (ccsdonly) then
    ! (CCSD: no operator on the occupied triples)
    call xzrel(ko6)
    ko6 = xzget(1_8, 'occupied operator')
  end if
  kwring = xzget(NW * ov2, 'pass-2 operand W')
  ka2 = xzget(NW * ov2, 'work A')
  kb2 = xzget(NW * ov2, 'work B')
  kx2 = xzget(NW * ov2, 'work X')

  ! ---- T1 dressing
  md = xzmark()
  kd = xzget(NW * ov * nvp, 'dressed <ov||vv>')
  if (.not. xzdry) then
    call xztic()
    call N(xzfock)(no, nv, nvp, w(jeps), w(kt1), w(jooov), w(joovv), w(jovvo), w(jovvq), &
                   w(jvpa), w(jvpb), w(kfoo), w(kfov), w(kfvo), w(kfvv), w(kpvv))
    call xztoc(1, 'dressing: Fock')
    call N(xzdroo)(no, nv, w(kt1), w(jooov), w(joovv), w(joooo), w(kooovd), w(kooood))
    call xztoc(1, 'dressing: <oo||ov>, <oo||oo>')
    if (ccsdonly) then
      ! (CCSD, xzsd.inc: in the blocks of the labels; no dressed <mb||ij>)
      call N(xzsdrovvq)(no, nv, nvp, w(kt1), w(jovvq), w(joovvp), w(kd))
      call xztoc(1, 'dressing: <ov||vv>')
      call N(xzsdrov)(no, nv, nvp, w(kt1), w(jooov), w(jovvo), w(kd), w(jvpa), w(jvpb), w(kovvod))
      call xztoc(1, 'dressing: <ov||vo>, <ov||oo>')
    else
      call N(xzdrovvq)(no, nv, nvp, w(kt1), w(jovvq), w(joovvp), w(kd))
      call xztoc(1, 'dressing: <ov||vv>')
      call N(xzdrov)(no, nv, nvp, w(kt1), w(jooov), w(jovvo), w(jovvq), w(kd), w(kooood), &
                     w(jvpidx), w(kovvod), w(kovood), w(kk))
      call xztoc(1, 'dressing: <ov||vo>, <ov||oo>')
    end if
    call N(xzt2forms)(no, nv, npo, nvp, w(kt1), w(kt2), w(jopa), w(jopb), w(jvpa), w(jvpb), &
                      w(kt2p), w(kt2pp), w(ktau), w(ktt))
    call xztoc(1, 'T2 on pairs, tau')
  end if

  ! ---- pass 1
  kxacc = xzget(NW * merge(1_8, ny * no * nvp, ccsdonly), 'pass-1 result X')
  m1 = xzmark()
  kgmq = xzget(NW * merge(1_8, no * nvp * ny, ccsdonly), 'pass-1 operand G')
  if (.not. xzdry .and. .not. ccsdonly) then
    call xztic()
    call N(xzp1ops)(no, nv, npo, nvp, w(joovv), w(kooovd), w(joovvp), w(kd), w(kfov), &
                    w(jopa), w(jopb), w(kgpd), w(kgmq), w(kft))
    call xztoc(1, 'pass-1 operands')
  end if
  call xztic()
  tsg = omp_get_wtime()
  dkind = 1
  if (ccsdonly) then
    ! CCSD: no T3 (xzsd.inc has no T3 parts; these arrays are not made)
    continue
  else if (dense) then
    kbuf = xzget(NW * nbp * ov * nvp, 'pass-1 batch of T3')
    if (.not. xzdry) then
      call zc_vzero_d(NW * ny * no * nvp, w(kxacc))
      call zc_vzero_d(NW * ov, w(kr1c))
      if (nrank > 1) then
        ! (the rows of the pairs of other ranks)
        call zc_vzero_d(NW * npo * nv * ny, w(kyall))
        call zc_vzero_d(NW * npo * nvp, w(kra))
      end if
      stats = 0
      do p0 = 0, npo - 1, nbp
        if (.not. xzmine(p0)) cycle
        call xzbtic()
        call N(zc_t3_pass1r)(w(kt3), nto, nvt, no, nv, npo, nvp, w(jotri), w(josgn), w(jvtri), w(jvsgn), &
                             nbp, p0, min(p0 + nbp, npo), w(kgpd), w(kgmq), ny, w(jgpp), w(kft), w(kxacc), &
                             w(kyall), w(kr1c), w(kra), c_null_ptr, stats, w(kbuf))
        call xzbtoc(p0)
      end do
    end if
  else
    call N(xzbpass1)(w, kt3, w(kgpd), w(kgmq), w(jgpp), w(kfov), w(kxacc), w(kyall), w(kr1c), w(kra))
  end if
  dkind = 0
  if (.not. xzdry) then
    call xztoc(1, 'pass 1')
    dsec(1, 1) = dsec(1, 1) + omp_get_wtime() - tsg
    if (nrank > 1 .and. .not. ccsdonly) then
      ! what the batches of all ranks made, on every rank (CCSD: no batches)
      tsg = omp_get_wtime()
      tint(1) = tint(1) + tsg - tst
      call xzmpi_barrier()
      call xztoc(1, 'MPI: waiting for the ranks')
      dsec(9, 1) = dsec(9, 1) + omp_get_wtime() - tsg
      tsg = omp_get_wtime()
      call xzmpi_allsum_w(w(kxacc), NW * ny * no * nvp)
      call xzmpi_allsum_w(w(kyall), NW * npo * nv * ny)
      call xzmpi_allsum_w(w(kr1c), NW * ov)
      call xzmpi_allsum_w(w(kra), NW * npo * nvp)
      call xztoc(1, 'MPI: sums of pass 1')
      dsec(4, 1) = dsec(4, 1) + omp_get_wtime() - tsg
      tsync = omp_get_wtime()
    end if
    ! sum_{k,Q} ~<ak||Q> t2[i][k][Q], at the columns no + a
    if (ccsdonly) then
      call N(xzsdr1v)(no, nv, nvp, w(kt2p), w(kd), w(kr1v))
    else
      call N(zc_gemm)(0_4, 0_4, no, ny, no * nvp, one, w(kt2p), no * nvp, w(kgmq), ny, zero, w(kr1v), ny)
    end if
    if (.not. ccsdonly) call N(xzr2t3)(no, nv, npo, nvp, w(jopa), w(jopb), w(jvpa), w(jvpb), w(kyall), w(kxacc), w(kra))
    ! CC4: the T4 term of the doubles equation (xzcc4.inc; the same on every rank)
    if (cc4) call N(xzaxpy)(npo * nvp, one, w(jd42), w(kra))
    call xztoc(1, 'R1, R2 from pass 1')
  end if
  call xzrel(m1)

  ! ---- CCSD: the dressed <ab||ij> (without W_vvvo), W_ovvo, F_vv, F_oo
  !      (xzsd.inc); the dressed <ov||vv> goes
  if (ccsdonly) then
    m2 = xzmark()
    kwk = xzget(NW * npo * npo, 'work [K][P]')
    khh = xzget(NW * no * no * ov, 'work [m][b][i][j]')
    kee = xzget(NW * ov * npo, 'work [m][b][P]')
    if (.not. xzdry) then
      call xztic()
      call N(xzsdvvoo)(no, nv, npo, nvp, w(kt1), w(ktau), w(jgpp), w(jovvq), w(jooov), w(joooo), w(jovvo), &
                       w(jopa), w(jopb), w(jvpa), w(jvpb), w(kr2), w(kwk), w(khh), w(kee))
      call xztoc(1, 'dressed <ab||ij>')
      call N(xzsdweff)(no, nv, nvp, w(kt2), w(kt2p), w(joovv), w(joovvp), w(kovvod), w(kfoo), w(kfvv), &
                       w(kweff), w(kfevv), w(kfeoo), w(ka2), w(kb2), w(kx2))
      call xztoc(1, 'W_ovvo, F_vv, F_oo')
    end if
    call xzrel(md)
    kwv = 0
  else
  ! ---- W_vvvo (without its <ab||cd> part), W_vooo, W_ovvo, F_vv, F_oo
  kwv = xzget(NW * nvp * ov, 'W_vvvo')
  m2 = xzmark()
  kg2 = xzget(NW * npo * ov, 'work [P][e][i]')
  kyb = xzget(NW * nba * nv * nv * no, 'batch of W_vvvo terms Y')
  ku = xzget(NW * nv ** 3, 'dressed <mb||ef> of one m')
  kxb = xzget(NW * nba * no * nv * nv, 'batch of W_vvvo terms X')
  if (.not. xzdry) then
    call xztic()
    call N(xzbv)(no, nv, npo, nvp, nba, w(kt1), w(ktau), w(jovvq), w(jooov), w(jovvo), &
                 w(jopa), w(jopb), w(jvpa), w(jvpb), w(kwv), w(kg2), w(kyb))
    call xztoc(1, 'W_vvvo: bare, tau, t1')
    call N(xzvvoo)(no, nv, npo, nvp, w(kt1), w(ktau), w(jgpp), w(jooov), w(joooo), w(kwv), &
                   w(jopa), w(jopb), w(jvpa), w(jvpb), w(kr2), w(kwkl))
    call xztoc(1, 'dressed <ab||ij>')
    ! (CCSD: W_vvvo, W_vooo and the pass-2 operands are not needed; the part
    ! of W_vvvo that xzvvoo took is all it needs)
    if (.not. ccsdonly) then
    ! (CC3, cc3mode 1: the drive operators are the T1-dressed integrals alone -
    ! no T3 and no T2 parts in W_vvvo and W_vooo)
    if (cc3mode == 0) then
      ! the T3 part: wv[Q][(e, i)] += xacc[(e, i)][Q]
      call N(zc_add_transposed)(w(kwv), w(kxacc), nvp, ov, ov, nvp, one)
      call N(xzwvoovo)(no, nv, npo, nvp, w(kt2pp), w(kooovd), w(kfov), w(jopa), w(jopb), w(kwv), w(kg2))
      call xztoc(1, 'W_vvvo: T3 part, t2 <oo||vo>')
      call N(xzwvx)(no, nv, nvp, nba, w(kt2), w(kd), w(jvpa), w(jvpb), w(kwv), w(ka2), w(ku), &
                    w(kxb), w(kyb))
      call xztoc(1, 'W_vvvo: t2 <ov||vv>')
      call N(xzwo)(no, nv, npo, nvp, w(kt2), w(kt2pp), w(kooovd), w(kovood), w(kd), w(kyall), &
                   w(jopa), w(jopb), w(kwo), w(ktmp), w(ktmp2), w(kb2))
    else
      call N(xzwo3)(no, nv, npo, w(kovood), w(jopa), w(jopb), w(kwo))
    end if
    call xztoc(1, 'W_vooo')
    end if
    call N(xzweff)(no, nv, nvp, w(kt2), w(kt2p), w(joovv), w(joovvp), w(kovvod), w(kfoo), w(kfvv), &
                   w(kweff), w(kfevv), w(kfeoo), w(ka2), w(kb2), w(kx2))
    call xztoc(1, 'W_ovvo, F_vv, F_oo')
    if (.not. ccsdonly) call N(xzp2ops)(no, nv, npo, w(kt2), w(kweff), w(kwo), w(jopa), w(jopb), w(kwring), w(kadrv))
    call xztoc(1, 'pass-2 operands W, A')
  end if
  ! W_vvvo moves down to the place of the dressed <ov||vv>, which goes, as
  ! does the pass-1 result
  call xzrel(md)
  kd = xzget(NW * nvp * ov, 'W_vvvo')
  if (.not. xzdry) then
    call xztic()
    call N(xzmove)(nvp * ov, w(kwv), w(kd))
    call xztoc(1, 'moves')
  end if
  kwv = kd
  end if

  ! ---- the sweep over the rows of the dressed <ab||cd>, block by block
  m2 = xzmark()
  kslab = xzget(NW * nqs * nqmax, 'slab of the dressed <ab||cd>')
  kml = xzget(nv, 'classes of the virtual spinors')
  kmh = xzget(nv, 'classes of the virtual spinors')
  if (.not. xzdry) call N(xzvclsrange)(nv, w(kml), w(kmh))
  if (ccsdonly) then
    kzl = xzget(NW * npo * ov, 'ladder: <mb||Q> (t2 + tau1)')
    kyl = xzget(NW * npo * npo, 'ladder: <K||Q> (t2 + tau1)')
  end if
  kgq = jovvq
  kpib = jvpidx
  if (.not. dense) then
    kgq = xzget(NW * ov * nqmax, '<ov||vv> on the pairs of a group')
    kgh = xzget(NW * npo * nqmax, '<oo||vv> on the pairs of a group')
    kpib = xzget(nv * nv, 'pairs of a group')
  else
    kgh = jgpp
    if (.not. ccsdonly) then
      ktv = xzget(NW * nvp * nbt * nv, 'pp-ladder batch of T3')
      kxp = xzget(NW * nqs * nbt * nv, 'pp-ladder product')
    end if
  end if
  if (.not. xzdry) then
    call xztic()
    call zc_vzero_d(NW * n3, w(kr3))
    call zc_vzero_d(NW * npo * nvp, w(klad))
    ! CC4: the T4 terms of the triples equation (rank 0 has their sum over the ranks)
    if (cc4 .and. myrank == 0) call N(xzaxpy)(n3, one, w(jd43), w(kr3))
    call xztoc(1, 'zero of the residual')
  end if
  tsg = omp_get_wtime()
  dkind = 3
  do h = 0, npg - 1
    lo = vpoff(h)
    nqh = vpoff(h + 1) - lo
    if (nqh == 0) cycle
    if (ccsdonly) then
      ! CCSD: the ladder with the bare <ab||cd> of the group (its occupied
      ! pairs: the others give zero); its T1 parts after the sweep (xzsdladt1)
      plo = opoff(h)
      nph = opoff(h + 1) - plo
      if (.not. xzdry .and. nph > 0 .and. (myrank == 0 .or. slices)) then
        call xztic()
        call N(zc_gemm)(0_4, 1_4, nph, nqh, nqh, one, w(ktt + NW * (lo + nvp * plo)), nvp, &
                        w(jvvvv + NW * vvoff(h)), nqh, zero, w(klad + NW * (lo + nvp * plo)), nvp)
        call xztoc(1, 'CCSD pp ladder (GEMM)')
      end if
      cycle
    end if
    if (.not. dense .and. .not. xzdry) then
      call xztic()
      call N(xzgrpcols1)(no, nv, npo, nvp, h, lo, nqh, w(jovvq), w(jgpp), w(jvpidx), w(kgq), w(kgh), w(kpib))
      call xztoc(1, 'columns of a group')
    end if
    do q0 = 0, nqh - 1, nqs
      nq = min(nqs, nqh - q0)
      if (.not. xzdry) then
        call xztic()
        ! ~<ab||de> = <ab||de> - sum_m t1[m][a] <mb||de> + sum_m t1[m][b] <ma||de>
        !             + sum_{m<n} tau1[mn][ab] <mn||de>
        call N(xzdrrows1)(no, nv, npo, nvp, nqh, nq, w(kt1), w(ktau + NW * (lo + q0)), &
                          w(jvvvv + NW * (vvoff(h) + q0 * nqh)), w(kgq), w(kgh), w(jvpa + lo + q0), &
                          w(jvpb + lo + q0), w(kml), w(kmh), w(kslab), w(ka2))
        ! W_vvvo[Q][e][i] += sum_f ~<ab||ef> t1[i][f]
        if (.not. ccsdonly) &
          call N(zc_pair_t1_contract)(w(kwv + NW * (lo + q0) * ov), w(kslab), w(kt1), nq, nv, nqh, w(kpib), no)
        call xztoc(1, 'dressed <ab||cd> rows')
        call xztic()
        ! the CCSD ladder: lad[P][q] = sum_Q ~<q||Q> (t2 + tau1)[P][Q]
        if (npo > 0 .and. (myrank == 0 .or. slices)) call N(zc_gemm)(0_4, 1_4, npo, nq, nqh, one, w(ktt + NW * lo), nvp, &
                                     w(kslab), nqh, zero, w(klad + NW * (lo + q0)), nvp)
        call xztoc(1, 'CCSD pp ladder (GEMM)')
        if (cc3mode == 0 .and. .not. ccsdonly) then
          call xztic()
          ! the pp operator of T3: + sum_{m<n} t2[mn][ab] <mn||de> + 1/2 A[F_vv delta]
          if (npo > 0) call N(zc_gemm)(1_4, 0_4, nq, nqh, npo, one, w(kt2pp + NW * (lo + q0)), nvp, &
                                       w(jgpp + NW * lo), nvp, one, w(kslab), nqh)
          call N(zc_pair_one_body_add)(w(kslab), nq, nqh, w(jvpa + lo + q0), w(jvpb + lo + q0), w(kpib), nv, &
                                       w(kfevv), nv, half)
          call xztoc(1, 'dressed <ab||cd> rows')
        end if
      end if
      ! (CC3: no pp ladder of T3)
      if (cc3mode /= 0) cycle
      if (xzskip(1) .or. ccsdonly) then
        continue
      else if (dense) then
        if (xzdry) cycle
        do i0 = 0, nto - 1, nbt
          nb = min(nbt, nto - i0)
          if (.not. xzmine(i0)) cycle
          call xzbtic()
          call xztic()
          call N(zc_gather_vir_half)(w(kt3), nvt, w(jrows + i0), nb, w(jvtri), w(jvsgn), nv, nvp, w(ktv))
          call xztoc(1, 'pp ladder: gather')
          call xztic()
          call N(zc_gemm)(0_4, 0_4, nq, nb * nv, nvp, one, w(kslab), nvp, w(ktv), nb * nv, zero, &
                          w(kxp), nb * nv)
          call xztoc(1, 'pp ladder: GEMM')
          call xztic()
          call N(zc_reduce_pair_first_rows)(w(kr3), nvt, w(kxp), w(jrows + i0), nb, nv, q0, q0 + nq, &
                                            w(jvfree), w(jvpair))
          call xztoc(1, 'pp ladder: reduction')
          call xzbtoc(i0)
        end do
      else
        call N(xzbppl)(w, kt3, kr3, h, q0, nq, w(kslab))
      end if
    end do
  end do
  dkind = 0
  if (ccsdonly .and. .not. xzdry .and. (myrank == 0 .or. slices)) then
    call xztic()
    call N(xzsdladt1)(no, nv, npo, nvp, w(kt1), w(ktau), w(ktt), w(jovvq), w(jgpp), w(jopa), w(jopb), &
                      w(jvpa), w(jvpb), w(klad), w(kzl), w(kyl))
    call xztoc(1, 'CCSD ladder: T1 parts')
  end if
  if (.not. xzdry) dsec(2, 1) = dsec(2, 1) + omp_get_wtime() - tsg
  call xzrel(m2)

  ! ---- the CCSD part and the operator on the occupied triples (rank 0; with
  !      slices every rank, for the operator on its rows)
  if (.not. xzdry .and. (myrank == 0 .or. slices)) then
    tsg = omp_get_wtime()
    call xztic()
    if (ccsdonly) then
      call N(xzsdccsd)(no, nv, npo, nvp, w(jeps), w(kt1), w(kt2), w(kt2p), w(kt2pp), w(jgpp), w(kooovd), &
                       w(kooood), w(kovvod), w(kweff), w(kfov), w(kfvo), w(kfevv), w(kfeoo), w(kr1v), &
                       w(klad), w(jopa), w(jopb), w(jvpa), w(jvpb), w(kr2), w(kwkl), &
                       w(kt1n), w(kt2n), w(ka2), w(kb2), w(kx2), w(kwring))
    else
    call N(xzccsd)(no, nv, npo, nvp, w(jeps), w(kt1), w(kt2), w(kt2p), w(kt2pp), w(jgpp), w(kooovd), &
                   w(kooood), w(kovvod), w(kweff), w(kfov), w(kfvo), w(kfevv), w(kfeoo), w(kr1v), &
                   w(kr1c), w(kra), w(klad), w(jopa), w(jopb), w(jvpa), w(jvpb), w(kr2), w(kwkl), &
                   w(kt1n), w(kt2n), w(ka2), w(kb2), w(kx2))
    end if
    call xztoc(1, 'CCSD part')
    call xztic()
    if (cc3mode /= 0) then
      ! (CC3: of the operator only the diagonal Fock part, applied in the update)
    else if (ccsdonly) then
      continue
    else if (pairhh) then
      call N(xzwpair)(no, npo, w(jopa), w(jopb), w(kwkl), w(kfeoo), w(ko6))
    else
      call N(xzo6)(no, npo, nto, w(jofree), w(jopair), w(kwkl), w(kfeoo), w(ko6))
    end if
    call xztoc(1, 'occupied operator')
    dsec(8, 1) = dsec(8, 1) + omp_get_wtime() - tsg
  end if

  ! CCSD: done (T1 and T2 new)
  if (ccsdonly) then
    if (.not. xzdry) dsec(7, 1) = dsec(7, 1) + omp_get_wtime() - tst
    call xzrel(m0)
    return
  end if

  ! ---- pass 2: its right-hand side from W_vvvo, in the place of W_vvvo
  nbdw = ny * no * nvp
  if (.not. dense) nbdw = nbd
  kbd = xzget(NW * nbdw, 'pass-2 operand B')
  if (.not. xzdry) then
    call xztic()
    if (dense) then
      call N(xzbdrv)(no, nv, nvp, w(kwv), w(kt2p), w(kbd))
    else
      call N(xzbbd)(w(kwv), w(kt2p), w(kbd))
    end if
    call xztoc(1, 'pass-2 operand B')
  end if
  call xzrel(md)
  kd = xzget(NW * nbdw, 'pass-2 operand B')
  if (.not. xzdry) then
    call N(xzmove)(nbdw, w(kbd), w(kd))
    call xztoc(1, 'moves')
  end if
  kbd = kd
  tsg = omp_get_wtime()
  ! (the test hook terms: 4 the ring terms of pass 2 left out, 5 its
  ! drive terms; their operands set to zero, dense and blocked alike)
  if (.not. xzdry) then
    if (xzskip(4)) call zc_vzero_d(NW * ov2, w(kwring))
    if (xzskip(5)) call zc_vzero_d(NW * npo * nv * ny, w(kadrv))
  end if
  dkind = 2
  if (xzskip(2)) then
    continue
  else if (dense) then
    ktb = xzget(NW * nbp * ov * nvp, 'pass-2 batch of T3')
    kz = xzget(NW * nbp * ov * nvp, 'pass-2 product')
    if (.not. xzdry) then
      stats = 0
      do p0 = 0, npo - 1, nbp
        if (.not. xzmine(p0)) cycle
        call xzbtic()
        if (cc3mode /= 0) then
          ! CC3: the drive terms alone (T3 is not read)
          call N(zc_t3_driver)(nto, nvt, no, nv, npo, nvp, w(jofree), w(jopair), w(jvfree), w(jvpair), &
                               nbp, p0, min(p0 + nbp, npo), w(kadrv), w(kbd), w(kr3), stats, w(kz))
        else
          call N(zc_t3_pass2r)(w(kt3), nto, nvt, no, nv, npo, nvp, w(jotri), w(josgn), w(jofree), w(jopair), &
                               w(jvtri), w(jvsgn), w(jvfree), w(jvpair), nbp, p0, min(p0 + nbp, npo), &
                               w(kwring), w(kadrv), w(kbd), w(kr3), c_null_ptr, stats, w(ktb), w(kz))
        end if
        call xzbtoc(p0)
      end do
    end if
  else
    noring = cc3mode /= 0
    call N(xzbpass2)(w, kt3, kr3, w(kwring), w(kadrv), w(kbd), 1_8, 0_8, w(kbd), w(kbd))
    noring = .false.
  end if
  dkind = 0
  if (pairhh) then
    khg = xzget(NW * npo * no * xzhhcols(), 'hh ladder: T3 by pairs')
    khx = xzget(NW * npo * no * xzhhcols(), 'hh ladder: product')
  end if
  if (.not. xzdry) then
    call xztoc(1, 'pass 2')
    dsec(3, 1) = dsec(3, 1) + omp_get_wtime() - tsg
    if (nrank > 1 .and. .not. slices) then
      ! the parts of the residual of T3, summed onto rank 0
      tsg = omp_get_wtime()
      tint(2) = tint(2) + tsg - tsync
      call xzmpi_barrier()
      call xztoc(1, 'MPI: waiting for the ranks')
      dsec(10, 1) = dsec(10, 1) + omp_get_wtime() - tsg
      tsg = omp_get_wtime()
      call xzmpi_sum0_w(w(kr3), NW * n3)
      call xztoc(1, 'MPI: sum of the residual')
      dsec(5, 1) = dsec(5, 1) + omp_get_wtime() - tsg
    end if
  end if
  if (slices) then
    ! the rows of this rank: the operator on occupied triples, the parts of
    ! the residual of all ranks, the Jacobi step
    kex = xzget(min(xzmpi_chunk(), NW * max(smax, 1_8)), 'chunk of the exchange')
    if (.not. xzdry) then
      tsg = omp_get_wtime()
      call xztic()
      if (cc3mode /= 0) then
        call N(xzfdiag)(w, kt3, kr3, 0_8, boff, .true.)
      else
        call N(xzslocc)(w, kt3, kr3, w(ko6))
      end if
      call xztoc(1, 'hh ladder, F_oo')
      dsec(8, 1) = dsec(8, 1) + omp_get_wtime() - tsg
      tsg = omp_get_wtime()
      tint(2) = tint(2) + tsg - tsync
      call xzmpi_barrier()
      call xztoc(1, 'MPI: waiting for the ranks')
      dsec(10, 1) = dsec(10, 1) + omp_get_wtime() - tsg
      tsg = omp_get_wtime()
      call xzmpi_slices_add(w(kr3), swoff, w(kex))
      call xztoc(1, 'MPI: residual to its rows')
      dsec(5, 1) = dsec(5, 1) + omp_get_wtime() - tsg
      tsg = omp_get_wtime()
      call N(xzslupd)(w, kt3, kr3)
      call xztoc(1, 'T3 update')
      dsec(8, 1) = dsec(8, 1) + omp_get_wtime() - tsg
    end if
  else if (.not. xzdry .and. myrank == 0) then
    tsg = omp_get_wtime()
    call xztic()
    if (cc3mode /= 0) then
      ! CC3: [F, T3] with the diagonal of F (CFOUR's spinors are canonical)
      call N(xzfdiag)(w, kt3, kr3, 0_8, boff, .false.)
    else if (xzskip(3)) then
      continue
    else if (pairhh) then
      call N(xzhhapply)(1_4, no, npo, nto, nvt, xzhhcols(), w(jotri), w(josgn), w(jofree), w(jopair), w(ko6), &
                        w(kt3), w(kr3), w(khg), w(khx))
    else if (dense) then
      call N(zc_t3_occop)(w(ko6), w(kt3), nto, nvt, ncb, c_null_ptr, w(kr3))
    else
      call N(xzbocc)(w, kt3, kr3, w(ko6))
    end if
    call xztoc(1, 'hh ladder, F_oo')
    call xztic()
    if (dense) then
      call N(zc_t3_update)(w(kt3), w(kr3), nto, nvt, w(jeo3), w(jev3), c_null_ptr)
    else
      call N(xzbupd)(w, kt3, kr3)
    end if
    call xztoc(1, 'T3 update')
    dsec(8, 1) = dsec(8, 1) + omp_get_wtime() - tsg
  end if
  if (.not. xzdry) dsec(7, 1) = dsec(7, 1) + omp_get_wtime() - tst
  call xzrel(m0)
end subroutine

! ---------------------------------------------------------------------------
! Rows of the T1-dressed <ab||de> of one group of pairs (the C kernel does
! the two terms with one t1; the term with tau1 is a GEMM).
!   nqh       pairs of the group; nq: rows
!   tau       tau1[P][Q] from the first row (the whole array, leading dimension nvp)
!   vv        the bare rows [q][Q'] of the block
!   gq, gh    <my||Q'> [m][y][Q'] and <mn||Q'> [P][Q'] on the pairs Q' of the group
!   pa, pb    the pairs of the rows
!   tm        work no * nv
subroutine N(xzdrrows)(no, nv, npo, nvp, nqh, nq, t1, tau, vv, gq, gh, pa, pb, slab, tm)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp, nqh, nq
  ZT, intent(in) :: t1(0:nv-1, 0:no-1)
  ZT, intent(in) :: tau(*), vv(*), gq(*), gh(*)
  integer(8), intent(in) :: pa(*), pb(*)
  ZT :: slab(*)
  ZT :: tm(0:nv-1, 0:no-1)
  ZT :: one
  one = 1
  tm = -t1
  call N(zc_vvvv_dress)(slab, vv, nq, nqh, pa, pb, tm, nv, gq, no, nv, t1, nv, gq, no, nv, tm, tm, 0_8)
  if (npo > 0) call N(zc_gemm)(1_4, 0_4, nq, nqh, npo, one, tau, nvp, gh, nqh, one, slab, nqh)
end subroutine

! ---------------------------------------------------------------------------
! With slices (xzdist.F90): the operator on the occupied triples for the
! rows of this rank, r[I][A] += sum_J o6[I][J] t3[J][A], and the Jacobi step
! on them (r <- t3 + r / D), block by block.
subroutine N(xzslocc)(w, kt3, kr3, o6)
  use xzdata
  use xzkern
  use xzdist
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kt3, kr3
  ZT, intent(in) :: o6(0:nto-1, 0:nto-1)                       ! [I][J]
  integer(8) :: g, ni, na, a, b
  ZT :: one
  one = 1
  do g = 0, ntg - 1
    ni = otoff(g + 1) - otoff(g)
    na = vtoff(g + 1) - vtoff(g)
    if (ni == 0 .or. na == 0) cycle
    call xzslrows(g, a, b)
    if (b <= a) cycle
    call N(zc_gemm)(0_4, 0_4, b - a, na, ni, one, o6(otoff(g), otoff(g) + a), nto, w(kt3 + NW * boff(g)), na, &
                    one, w(kr3 + NW * (boff(g) + a * na)), na)
  end do
end subroutine

subroutine N(xzslupd)(w, kt3, kr3)
  use xzdata
  use xzkern
  use xzdist
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kt3, kr3
  integer(8) :: g, ni, na, a, b
  do g = 0, ntg - 1
    ni = otoff(g + 1) - otoff(g)
    na = vtoff(g + 1) - vtoff(g)
    if (ni == 0 .or. na == 0) cycle
    call xzslrows(g, a, b)
    if (b <= a) cycle
    call N(zc_t3_update)(w(kt3 + NW * (boff(g) + a * na)), w(kr3 + NW * (boff(g) + a * na)), b - a, na, &
                         w(jeo3 + otoff(g) + a), w(jev3 + vtoff(g)), c_null_ptr)
  end do
end subroutine

! ---------------------------------------------------------------------------
! The columns of <my||Q> and <mn||Q> of the pairs Q of one group, and the
! numbers of these pairs inside the group (-1: a pair of another group).
subroutine N(xzgrpcols)(no, nv, npo, nvp, lo, nqh, gq, gpp, vpidx, gqh, gph, pib)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp, lo, nqh
  ZT, intent(in) :: gq(0:nvp-1, 0:no*nv-1), gpp(0:nvp-1, 0:npo-1)
  integer(8), intent(in) :: vpidx(0:nv*nv-1)
  ZT, intent(out) :: gqh(0:nqh-1, 0:no*nv-1), gph(0:nqh-1, 0:npo-1)
  integer(8), intent(out) :: pib(0:nv*nv-1)
  integer(8) :: k
  !$omp parallel do
  do k = 0, no * nv - 1
    gqh(:, k) = gq(lo:lo + nqh - 1, k)
  end do
  do k = 0, npo - 1
    gph(:, k) = gpp(lo:lo + nqh - 1, k)
  end do
  do k = 0, nv * nv - 1
    pib(k) = -1
    if (vpidx(k) >= lo .and. vpidx(k) < lo + nqh) pib(k) = vpidx(k) - lo
  end do
end subroutine
