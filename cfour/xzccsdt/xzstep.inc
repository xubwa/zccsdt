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
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kt1, kt2, kt3, kt1n, kt2n, kr3
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

  ! ---- the arrays of the size of T2 and smaller
  kfoo = xzget(NW * no * no, 'F_oo')
  kfov = xzget(NW * ov, 'F_ov')
  kfvo = xzget(NW * ov, 'F_vo')
  kfvv = xzget(NW * nv * nv, 'F_vv')
  kpvv = xzget(NW * nv * nv, 'F_vv work')
  kfevv = xzget(NW * nv * nv, 'F_vv of T3')
  kfeoo = xzget(NW * no * no, 'F_oo of T3')
  kft = xzget(NW * ov, 'F_ov transposed')
  kooovd = xzget(NW * no * no * ov, 'dressed <oo||ov>')
  kooood = xzget(NW * no ** 4, 'dressed <oo||oo>')
  kovvod = xzget(NW * ov2, 'dressed <ov||vo>')
  kovood = xzget(NW * no * no * ov, 'dressed <ov||oo>')
  kk = xzget(NW * ov2, 'work of the dressing')
  ktmp = xzget(NW * no * no * ov, 'work of W_vooo')
  ktmp2 = xzget(NW * no * no * ov, 'work of W_vooo 2')
  kt2p = xzget(NW * no * no * nvp, 'T2 [i][j][Q]')
  kt2pp = xzget(NW * npo * nvp, 'T2 [P][Q]')
  ktau = xzget(NW * npo * nvp, 'tau1')
  ktt = xzget(NW * npo * nvp, 'T2 + tau1')
  kgpd = xzget(NW * ny * npo * nv, 'pass-1 operand')
  kyall = xzget(NW * npo * nv * ny, 'pass-1 result Y')
  kr1c = xzget(NW * ov, 'T3 part of R1')
  kr1v = xzget(NW * no * ny, 'vovv part of R1')
  kra = xzget(NW * npo * nvp, 'T3 part of R2')
  kr2 = xzget(NW * npo * nvp, 'R2')
  klad = xzget(NW * npo * nvp, 'CCSD ladder')
  kwkl = xzget(NW * npo * npo, 'W_oooo')
  kweff = xzget(NW * ov2, 'W_ovvo')
  kwo = xzget(NW * ov * npo, 'W_vooo')
  kadrv = xzget(NW * npo * nv * ny, 'pass-2 operand A')
  pairhh = xzhhpair()
  ko6 = xzget(NW * merge(npo * npo, nto * nto, pairhh), 'occupied operator')
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
    call N(xzdrovvq)(no, nv, nvp, w(kt1), w(jovvq), w(joovvp), w(kd))
    call xztoc(1, 'dressing: <ov||vv>')
    call N(xzdrov)(no, nv, nvp, w(kt1), w(jooov), w(jovvo), w(jovvq), w(kd), w(kooood), &
                   w(jvpidx), w(kovvod), w(kovood), w(kk))
    call xztoc(1, 'dressing: <ov||vo>, <ov||oo>')
    call N(xzt2forms)(no, nv, npo, nvp, w(kt1), w(kt2), w(jopa), w(jopb), w(jvpa), w(jvpb), &
                      w(kt2p), w(kt2pp), w(ktau), w(ktt))
    call xztoc(1, 'T2 on pairs, tau')
  end if

  ! ---- pass 1
  kxacc = xzget(NW * ny * no * nvp, 'pass-1 result X')
  m1 = xzmark()
  kgmq = xzget(NW * no * nvp * ny, 'pass-1 operand G')
  if (.not. xzdry) then
    call xztic()
    call N(xzp1ops)(no, nv, npo, nvp, w(joovv), w(kooovd), w(joovvp), w(kd), w(kfov), &
                    w(jopa), w(jopb), w(kgpd), w(kgmq), w(kft))
    call xztoc(1, 'pass-1 operands')
  end if
  call xztic()
  if (dense) then
    kbuf = xzget(NW * nbp * ov * nvp, 'pass-1 batch of T3')
    if (.not. xzdry) then
      call zc_vzero_d(NW * ny * no * nvp, w(kxacc))
      call zc_vzero_d(NW * ov, w(kr1c))
      stats = 0
      call N(zc_t3_pass1w)(w(kt3), nto, nvt, no, nv, npo, nvp, w(jotri), w(josgn), w(jvtri), w(jvsgn), &
                           nbp, w(kgpd), w(kgmq), ny, w(jgpp), w(kft), w(kxacc), w(kyall), w(kr1c), &
                           w(kra), c_null_ptr, stats, w(kbuf))
    end if
  else
    call N(xzbpass1)(w, kt3, w(kgpd), w(kgmq), w(jgpp), w(kfov), w(kxacc), w(kyall), w(kr1c), w(kra))
  end if
  if (.not. xzdry) then
    call xztoc(1, 'pass 1')
    ! sum_{k,Q} ~<ak||Q> t2[i][k][Q], at the columns no + a
    call N(zc_gemm)(0_4, 0_4, no, ny, no * nvp, one, w(kt2p), no * nvp, w(kgmq), ny, zero, w(kr1v), ny)
    call N(xzr2t3)(no, nv, npo, nvp, w(jopa), w(jopb), w(jvpa), w(jvpb), w(kyall), w(kxacc), w(kra))
    call xztoc(1, 'R1, R2 from pass 1')
  end if
  call xzrel(m1)

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
    ! the T3 part: wv[Q][(e, i)] += xacc[(e, i)][Q]
    call N(zc_add_transposed)(w(kwv), w(kxacc), nvp, ov, ov, nvp, one)
    call N(xzwvoovo)(no, nv, npo, nvp, w(kt2pp), w(kooovd), w(kfov), w(jopa), w(jopb), w(kwv), w(kg2))
    call xztoc(1, 'W_vvvo: T3 part, t2 <oo||vo>')
    call N(xzwvx)(no, nv, nvp, nba, w(kt2), w(kd), w(jvpa), w(jvpb), w(kwv), w(ka2), w(ku), &
                  w(kxb), w(kyb))
    call xztoc(1, 'W_vvvo: t2 <ov||vv>')
    call N(xzwo)(no, nv, npo, nvp, w(kt2), w(kt2pp), w(kooovd), w(kovood), w(kd), w(kyall), &
                 w(jopa), w(jopb), w(kwo), w(ktmp), w(ktmp2), w(kb2))
    call xztoc(1, 'W_vooo')
    call N(xzweff)(no, nv, nvp, w(kt2), w(kt2p), w(joovv), w(joovvp), w(kovvod), w(kfoo), w(kfvv), &
                   w(kweff), w(kfevv), w(kfeoo), w(ka2), w(kb2), w(kx2))
    call xztoc(1, 'W_ovvo, F_vv, F_oo')
    call N(xzp2ops)(no, nv, npo, w(kt2), w(kweff), w(kwo), w(jopa), w(jopb), w(kwring), w(kadrv))
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

  ! ---- the sweep over the rows of the dressed <ab||cd>, block by block
  m2 = xzmark()
  kslab = xzget(NW * nqs * nqmax, 'slab of the dressed <ab||cd>')
  kgq = jovvq
  kpib = jvpidx
  if (.not. dense) then
    kgq = xzget(NW * ov * nqmax, '<ov||vv> on the pairs of a group')
    kgh = xzget(NW * npo * nqmax, '<oo||vv> on the pairs of a group')
    kpib = xzget(nv * nv, 'pairs of a group')
  else
    kgh = jgpp
    ktv = xzget(NW * nvp * nbt * nv, 'pp-ladder batch of T3')
    kxp = xzget(NW * nqs * nbt * nv, 'pp-ladder product')
  end if
  if (.not. xzdry) then
    call xztic()
    call zc_vzero_d(NW * n3, w(kr3))
    call zc_vzero_d(NW * npo * nvp, w(klad))
    call xztoc(1, 'zero of the residual')
  end if
  do h = 0, npg - 1
    lo = vpoff(h)
    nqh = vpoff(h + 1) - lo
    if (nqh == 0) cycle
    if (.not. dense .and. .not. xzdry) then
      call xztic()
      call N(xzgrpcols)(no, nv, npo, nvp, lo, nqh, w(jovvq), w(jgpp), w(jvpidx), w(kgq), w(kgh), w(kpib))
      call xztoc(1, 'dressed <ab||cd> rows')
    end if
    do q0 = 0, nqh - 1, nqs
      nq = min(nqs, nqh - q0)
      if (.not. xzdry) then
        call xztic()
        ! ~<ab||de> = <ab||de> - sum_m t1[m][a] <mb||de> + sum_m t1[m][b] <ma||de>
        !             + sum_{m<n} tau1[mn][ab] <mn||de>
        call N(xzdrrows)(no, nv, npo, nvp, nqh, nq, w(kt1), w(ktau + NW * (lo + q0)), &
                         w(jvvvv + NW * (vvoff(h) + q0 * nqh)), w(kgq), w(kgh), w(jvpa + lo + q0), &
                         w(jvpb + lo + q0), w(kslab), w(ka2))
        ! W_vvvo[Q][e][i] += sum_f ~<ab||ef> t1[i][f]
        call N(zc_pair_t1_contract)(w(kwv + NW * (lo + q0) * ov), w(kslab), w(kt1), nq, nv, nqh, w(kpib), no)
        call xztoc(1, 'dressed <ab||cd> rows')
        call xztic()
        ! the CCSD ladder: lad[P][q] = sum_Q ~<q||Q> (t2 + tau1)[P][Q]
        if (npo > 0) call N(zc_gemm)(0_4, 1_4, npo, nq, nqh, one, w(ktt + NW * lo), nvp, w(kslab), nqh, &
                                     zero, w(klad + NW * (lo + q0)), nvp)
        call xztoc(1, 'CCSD pp ladder (GEMM)')
        call xztic()
        ! the pp operator of T3: + sum_{m<n} t2[mn][ab] <mn||de> + 1/2 A[F_vv delta]
        if (npo > 0) call N(zc_gemm)(1_4, 0_4, nq, nqh, npo, one, w(kt2pp + NW * (lo + q0)), nvp, &
                                     w(jgpp + NW * lo), nvp, one, w(kslab), nqh)
        call N(zc_pair_one_body_add)(w(kslab), nq, nqh, w(jvpa + lo + q0), w(jvpb + lo + q0), w(kpib), nv, &
                                     w(kfevv), nv, half)
        call xztoc(1, 'dressed <ab||cd> rows')
      end if
      if (dense) then
        if (xzdry) cycle
        do i0 = 0, nto - 1, nbt
          nb = min(nbt, nto - i0)
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
        end do
      else
        call N(xzbppl)(w, kt3, kr3, h, q0, nq, w(kslab))
      end if
    end do
  end do
  call xzrel(m2)

  ! ---- the CCSD part and the operator on the occupied triples
  if (.not. xzdry) then
    call xztic()
    call N(xzccsd)(no, nv, npo, nvp, w(jeps), w(kt1), w(kt2), w(kt2p), w(kt2pp), w(jgpp), w(kooovd), &
                   w(kooood), w(kovvod), w(kweff), w(kfov), w(kfvo), w(kfevv), w(kfeoo), w(kr1v), &
                   w(kr1c), w(kra), w(klad), w(jopa), w(jopb), w(jvpa), w(jvpb), w(kr2), w(kwkl), &
                   w(kt1n), w(kt2n), w(ka2), w(kb2), w(kx2))
    call xztoc(1, 'CCSD part')
    call xztic()
    if (pairhh) then
      call N(xzwpair)(no, npo, w(jopa), w(jopb), w(kwkl), w(kfeoo), w(ko6))
    else
      call N(xzo6)(no, npo, nto, w(jofree), w(jopair), w(kwkl), w(kfeoo), w(ko6))
    end if
    call xztoc(1, 'occupied operator')
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
  if (dense) then
    ktb = xzget(NW * nbp * ov * nvp, 'pass-2 batch of T3')
    kz = xzget(NW * nbp * ov * nvp, 'pass-2 product')
    if (.not. xzdry) then
      stats = 0
      call N(zc_t3_pass2w)(w(kt3), nto, nvt, no, nv, npo, nvp, w(jotri), w(josgn), w(jofree), w(jopair), &
                           w(jvtri), w(jvsgn), w(jvfree), w(jvpair), nbp, w(kwring), w(kadrv), w(kbd), &
                           w(kr3), c_null_ptr, stats, w(ktb), w(kz))
    end if
  else
    call N(xzbpass2)(w, kt3, kr3, w(kwring), w(kadrv), w(kbd), 1_8, 0_8, w(kbd), w(kbd))
  end if
  if (pairhh) then
    khg = xzget(NW * npo * no * xzhhcols(), 'hh ladder: T3 by pairs')
    khx = xzget(NW * npo * no * xzhhcols(), 'hh ladder: product')
  end if
  if (.not. xzdry) then
    call xztoc(1, 'pass 2')
    call xztic()
    if (pairhh) then
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
  end if
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
