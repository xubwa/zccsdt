!
! EOM-EE-CCSDT on packed amplitudes: the intermediates of the ground state
! and the sigma vector (type-generic).  The algorithm of
! src/zccsdt/cc/eom_opt.py (docs/eom.md, sections 2, 3 and 6; docs/native.md,
! section 16):
!
!   sigma = <mu| [Hbar, R] |0> = (Jacobian of the amplitude equations) R
!
! In the T1-dressed formalism the amplitude step is linear in the dressed
! integrals and in T3 and at most quadratic in T2, so
!
!   sigma_3 = residual(R3; W_ring, W_pp, O_hh | W'_vvvo x t2, W'_vooo x t2)
!           + residual(T3; W'_ring, W'_pp, O'_hh | W_vvvo x r2, W_vooo x r2)
!   W'      = W[H'; t2] + (the part of W linear in t2, T3)[H~; r2, R3],
!   H'      = [H~, R1]: the dressed integrals with one index transformed by r1
!
! - the passes of the T3 equation twice, with other operands - and pass 1
! over R3 (all its outputs) and over T3 with H' (its T2 part).  No kernel is
! new; in the blocks of the labels the passes of xzblk.inc run with the label
! shift of R (xzdata: xzshctx).
!
!   once (xzeomim)   what an amplitude step makes, kept (xzdata: efoo ..): the
!                    dressed integrals, W_ovvo, F_vv, F_oo, W_oooo, W_vvvo,
!                    W_vooo, the operator on occupied triples; the pp
!                    operator of the T3 equation in the place of <ab||cd>
!                    (after it the work array has no bare <ab||cd>).
!   a sigma vector   xzesig: the derivative of the pp operator is never
!                    stored - its rows are made slab by slab for the ladder
!                    with T3, as the dressed <ab||cd> of an amplitude step.
!
! A vector is [r1[i][a] | r2[P][Q] on i<j, a<b | R3 in the layout of its
! label shift (the packed block; the blocks of xzeomlay)], one region of the
! work array.  r1 and r2 are dense with zeros outside their shift; sigma_1
! and sigma_2 are masked to it (the integrals are symmetric to the rounding
! of the SCF only).
!

! ---------------------------------------------------------------------------
! The places of the kept arrays (taken from the stack by the caller of
! xzeomim; released with its mark).
subroutine N(xzeomkeep)()
  use xzdata
  use xzmem
  implicit none
  integer(8) :: ov
  ov = no * nv
  efoo = xzget(NW * no * no, 'F_oo')
  efov = xzget(NW * ov, 'F_ov')
  eft = xzget(NW * ov, 'F_ov transposed')
  efvv = xzget(NW * nv * nv, 'F_vv')
  efevv = xzget(NW * nv * nv, 'F_vv of T3')
  efeoo = xzget(NW * no * no, 'F_oo of T3')
  eooovd = xzget(NW * no * no * ov, 'dressed <oo||ov>')
  eooood = xzget(NW * no ** 4, 'dressed <oo||oo>')
  eovvod = xzget(NW * ov * ov, 'dressed <ov||vo>')
  eovood = xzget(NW * no * no * ov, 'dressed <ov||oo>')
  eweff = xzget(NW * ov * ov, 'W_ovvo')
  et2p = xzget(NW * no * no * nvp, 'T2 [i][j][Q]')
  et2pp = xzget(NW * npo * nvp, 'T2 [P][Q]')
  ewkl = xzget(NW * npo * npo, 'W_oooo')
  ekk = xzget(NW * no * no * npo, '<mn||Q> t2[P][Q]')
  ! (CC3: no operator on occupied triples)
  eo6 = xzget(merge(0_8, NW * nto * nto, cc3mode /= 0), 'occupied operator')
  ewo = xzget(NW * ov * npo, 'W_vooo')
  egpd = xzget(NW * ny * npo * nv, 'pass-1 operand')
  egmq = xzget(NW * no * nvp * ny, 'pass-1 operand G')
  ed = xzget(NW * ov * nvp, 'dressed <ov||vv>')
  ebv = xzget(NW * nvp * ov, 'bra-dressed <vv||vo>')
  ewv = xzget(NW * nvp * ov, 'W_vvvo')
  if (cc3mode == 2) then
    ! the operands of the CC3 batches of the ground state (the folded sigma vector)
    e3ao = xzget(NW * no * no * ov, 'CC3: Ao')
    e3bv = xzget(NW * nv * no * nvp, 'CC3: Bv')
    e3gv = xzget(NW * no * nvp * nv, 'CC3: Gv')
  end if
  ! (EOM-CC4: the operands of the T4 terms of the ground state, the bare <ab||cd>)
  if (cc4) call N(xz4eomkeep)()
end subroutine

! ---------------------------------------------------------------------------
! The intermediates of the ground state: one amplitude step without its T3
! residual, its operands kept.
subroutine N(xzeomim)(w)
  use xzdata
  use xzmem
  use xzkern
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8) :: ms, m1, ov, ov2
  integer(8) :: kfvo, kpvv, kk, ktmp, ktmp2, ktau, ktt, kyall, kr1c, kra, ka2, kb2, kx2, kxacc, kbuf
  integer(8) :: kg2, kyb, ku, kxb, kgq, kgh, kpib, h, lo, nqh
  logical :: dense
  real(8) :: stats(4)
  ZT :: one, zero, half
  one = 1
  zero = 0
  half = 0.5d0
  ov = no * nv
  ov2 = ov * ov
  dense = ncls == 1
  tcur = 5
  call xzshctx(0_8, 0_8, 0_8)
  ms = xzmark()
  kfvo = xzget(NW * ov, 'F_vo')
  kpvv = xzget(NW * nv * nv, 'F_vv work')
  kk = xzget(NW * ov2, 'work of the dressing')
  ktmp = xzget(NW * no * no * ov, 'work of W_vooo')
  ktmp2 = xzget(NW * no * no * ov, 'work of W_vooo 2')
  ktau = xzget(NW * npo * nvp, 'tau1')
  ktt = xzget(NW * npo * nvp, 'T2 + tau1')
  kyall = xzget(NW * npo * nv * ny, 'pass-1 result Y')
  kr1c = xzget(NW * ov, 'T3 part of R1')
  kra = xzget(NW * npo * nvp, 'T3 part of R2')
  ka2 = xzget(NW * ov2, 'work A')
  kb2 = xzget(NW * ov2, 'work B')
  kx2 = xzget(NW * ov2, 'work X')
  kxacc = xzget(NW * ny * no * nvp, 'pass-1 result X')
  if (.not. xzdry) then
    call xztic()
    call N(xzfock)(no, nv, nvp, w(jeps), w(jt1), w(jooov), w(joovv), w(jovvo), w(jovvq), &
                   w(jvpa), w(jvpb), w(efoo), w(efov), w(kfvo), w(efvv), w(kpvv))
    call N(xzdroo)(no, nv, w(jt1), w(jooov), w(joovv), w(joooo), w(eooovd), w(eooood))
    call N(xzdrovvq)(no, nv, nvp, w(jt1), w(jovvq), w(joovvp), w(ed))
    call N(xzdrov)(no, nv, nvp, w(jt1), w(jooov), w(jovvo), w(jovvq), w(ed), w(eooood), &
                   w(jvpidx), w(eovvod), w(eovood), w(kk))
    call N(xzt2forms)(no, nv, npo, nvp, w(jt1), w(jt2), w(jopa), w(jopb), w(jvpa), w(jvpb), &
                      w(et2p), w(et2pp), w(ktau), w(ktt))
    call N(xzp1ops)(no, nv, npo, nvp, w(joovv), w(eooovd), w(joovvp), w(ed), w(efov), &
                    w(jopa), w(jopb), w(egpd), w(egmq), w(eft))
    call xztoc(5, 'once: T1 dressing, operands')
  end if

  ! ---- pass 1 over T3: the T3 parts of W_vvvo and W_vooo (not in CC3; CC3
  !      without stored triples has no T3)
  m1 = xzmark()
  if (cc3mode == 2) then
  else if (dense) then
    kbuf = xzget(NW * nbp * ov * nvp, 'pass-1 batch of T3')
    if (.not. xzdry) then
      call zc_vzero_d(NW * ny * no * nvp, w(kxacc))
      call zc_vzero_d(NW * ov, w(kr1c))
      stats = 0
      call N(zc_t3_pass1w)(w(jt3), nto, nvt, no, nv, npo, nvp, w(jotri), w(josgn), w(jvtri), w(jvsgn), &
                           nbp, w(egpd), w(egmq), ny, w(jgpp), w(eft), w(kxacc), w(kyall), w(kr1c), &
                           w(kra), c_null_ptr, stats, w(kbuf))
    end if
  else
    call N(xzbpass1)(w, jt3, w(egpd), w(egmq), w(jgpp), w(efov), w(kxacc), w(kyall), w(kr1c), w(kra))
  end if
  if (.not. xzdry) call xztoc(5, 'once: pass 1 over T3')
  call xzrel(m1)

  ! ---- W_vvvo (without its <ab||cd> part), W_vooo, W_ovvo, F_vv, F_oo, W_oooo
  kg2 = xzget(NW * npo * ov, 'work [P][e][i]')
  kyb = xzget(NW * nba * nv * nv * no, 'batch of W_vvvo terms Y')
  ku = xzget(NW * nv ** 3, 'dressed <mb||ef> of one m')
  kxb = xzget(NW * nba * no * nv * nv, 'batch of W_vvvo terms X')
  if (.not. xzdry) then
    call N(xzbv)(no, nv, npo, nvp, nba, w(jt1), w(ktau), w(jovvq), w(jooov), w(jovvo), &
                 w(jopa), w(jopb), w(jvpa), w(jvpb), w(ebv), w(kg2), w(kyb))
    call N(xzecopy)(nvp * ov, w(ebv), w(ewv))
    ! (CC3: the drive operators are the T1-dressed integrals alone)
    if (cc3mode == 0) then
      call N(zc_add_transposed)(w(ewv), w(kxacc), nvp, ov, ov, nvp, one)
      call N(xzwvoovo)(no, nv, npo, nvp, w(et2pp), w(eooovd), w(efov), w(jopa), w(jopb), w(ewv), w(kg2))
      call N(xzwvx)(no, nv, nvp, nba, w(jt2), w(ed), w(jvpa), w(jvpb), w(ewv), w(ka2), w(ku), &
                    w(kxb), w(kyb))
    end if
    call xztoc(5, 'once: W_vvvo')
    if (cc3mode == 0) then
      call N(xzwo)(no, nv, npo, nvp, w(jt2), w(et2pp), w(eooovd), w(eovood), w(ed), w(kyall), &
                   w(jopa), w(jopb), w(ewo), w(ktmp), w(ktmp2), w(kb2))
    else
      call N(xzwo3)(no, nv, npo, w(eovood), w(jopa), w(jopb), w(ewo))
    end if
    call N(xzweff)(no, nv, nvp, w(jt2), w(et2p), w(joovv), w(joovvp), w(eovvod), w(efoo), w(efvv), &
                   w(eweff), w(efevv), w(efeoo), w(ka2), w(kb2), w(kx2))
    call N(xzewkl)(no, npo, nvp, w(eooood), w(jgpp), w(et2pp), w(jopa), w(jopb), w(ewkl))
    if (npo > 0) call N(zc_gemm)(0_4, 1_4, no * no, npo, nvp, one, w(joovvp), nvp, w(et2pp), nvp, zero, &
                                 w(ekk), npo)
    if (cc3mode == 0) call N(xzo6)(no, npo, nto, w(jofree), w(jopair), w(ewkl), w(efeoo), w(eo6))
    call xztoc(5, 'once: W_vooo, W_ovvo, F, O')
  end if

  ! ---- the pp operator of the T3 equation in the place of the bare
  !      <ab||de>, block by block; its part of W_vvvo on the way
  kgq = jovvq
  kgh = jgpp
  kpib = jvpidx
  if (.not. dense) then
    kgq = xzget(NW * ov * nqmax, '<ov||vv> on the pairs of a group')
    kgh = xzget(NW * npo * nqmax, '<oo||vv> on the pairs of a group')
    kpib = xzget(nv * nv, 'pairs of a group')
  end if
  if (.not. xzdry) then
    ! (EOM-CC4 makes Xv from the bare <ab||cd>: a copy)
    if (cc4) call N(xzecopy)(nvv, w(jvvvv), w(e4vv))
    do h = 0, npg - 1
      lo = vpoff(h)
      nqh = vpoff(h + 1) - lo
      if (nqh == 0) cycle
      if (.not. dense) call N(xzgrpcols)(no, nv, npo, nvp, lo, nqh, w(jovvq), w(jgpp), w(jvpidx), &
                                         w(kgq), w(kgh), w(kpib))
      call N(xzdrrows)(no, nv, npo, nvp, nqh, nqh, w(jt1), w(ktau + NW * lo), w(jvvvv + NW * vvoff(h)), &
                       w(kgq), w(kgh), w(jvpa + lo), w(jvpb + lo), w(jvvvv + NW * vvoff(h)), w(ka2))
      call N(zc_pair_t1_contract)(w(ewv + NW * lo * ov), w(jvvvv + NW * vvoff(h)), w(jt1), nqh, nv, nqh, &
                                  w(kpib), no)
      if (npo > 0) call N(zc_gemm)(1_4, 0_4, nqh, nqh, npo, one, w(et2pp + NW * lo), nvp, &
                                   w(jgpp + NW * lo), nvp, one, w(jvvvv + NW * vvoff(h)), nqh)
      call N(zc_pair_one_body_add)(w(jvvvv + NW * vvoff(h)), nqh, nqh, w(jvpa + lo), w(jvpb + lo), &
                                   w(kpib), nv, w(efevv), nv, half)
    end do
    call xztoc(5, 'once: pp operator in place')
    if (cc3mode == 2) then
      ! the operands of the CC3 batches of the ground state (docs/native.md, section 17)
      call N(xzao3)(no, nv, w(eovood), w(e3ao))
      call N(xzbvq)(no, nv, nvp, w(ewv), w(e3bv))
      call N(xzgvq)(no, nv, nvp, w(egmq), w(e3gv))
      call xztoc(5, 'once: CC3 operands')
    end if
  end if
  call xzrel(ms)
end subroutine

! ---------------------------------------------------------------------------
! The sigma vector.
!   kv     the vector [r1 | r2 on pairs | R3] (in; its label shift: eshift)
!   ks     sigma, the same layout (out)
!   sd     the singles-doubles block only: no R3, none of the terms of T3 in
!          sigma_2, no sigma_3 (the guess stage)
subroutine N(xzesig)(w, kv, ks, sd)
  use xzdata
  use xzmem
  use xzkern
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kv, ks
  logical, intent(in) :: sd
  integer(8) :: ov, ov2, nsd, kr1, kr2pp, kr3, ks1, ks2, ks3, m0, m1, m2, m3, s
  integer(8) :: kr2, kr2p, kdfov, kdfoo, kdfvv, kdfvo, kdfevv, kdfeoo, kdft, kdooov, kdoooo, kdovvo, kdovoo
  integer(8) :: kdweff, ka2, kb2, kx2, kdwkl, kxp, kc1, kr1v, kzz, kyy
  integer(8) :: kdwv, kdwo, kxacc, kyall, kr1c, kra, kgpd, kgmq, kzpp, kbuf, kg2, kyb, ku, kxb, ktmp, ktmp2
  integer(8) :: kzero3, kwring, kadrv, kdo6, kslab, kgq, kgh, kpib, ktv, kxpp, kbd, ktb, kz, nbdw
  integer(8) :: h, hr, lo, lo2, nqh, nqr, q0, nq, i0, nb, p0, x0
  logical :: dense, with3, fold
  real(8) :: stats(4)
  ZT :: one, zero, half, mone
  one = 1
  zero = 0
  half = 0.5d0
  mone = -1
  ov = no * nv
  ov2 = ov * ov
  nsd = ov + npo * nvp
  dense = ncls == 1
  with3 = .not. sd
  ! the folded sigma vector of CC3 without stored triples (efold, at the
  ! frequency eomega): the singles-doubles block and the triples folded in
  ! (N(xzefold)); the vectors are singles and doubles
  fold = sd .and. efold .and. cc3mode == 2
  s = eshift
  if (dense) s = 0
  kr1 = kv
  kr2pp = kv + NW * ov
  kr3 = kv + NW * nsd
  ks1 = ks
  ks2 = ks + NW * ov
  ks3 = ks + NW * nsd
  tcur = 5
  m0 = xzmark()

  ! ---- the arrays of the size of T2 and smaller
  kr2 = xzget(NW * ov2, 'R2 [i][j][a][b]')
  kr2p = xzget(NW * no * no * nvp, 'R2 [i][j][Q]')
  kdfov = xzget(NW * ov, 'F_ov derivative')
  kdft = xzget(NW * ov, 'F_ov derivative transposed')
  kdfoo = xzget(NW * no * no, 'F_oo derivative')
  kdfvv = xzget(NW * nv * nv, 'F_vv derivative')
  kdfvo = xzget(NW * ov, 'F_vo derivative')
  kdfevv = xzget(NW * nv * nv, 'F_vv of T3, derivative')
  kdfeoo = xzget(NW * no * no, 'F_oo of T3, derivative')
  kdooov = xzget(NW * no * no * ov, '<oo||ov> derivative')
  kdoooo = xzget(NW * no ** 4, '<oo||oo> derivative')
  kdovvo = xzget(NW * ov2, '<ov||vo> derivative')
  kdovoo = xzget(NW * no * no * ov, '<ov||oo> derivative')
  kdweff = xzget(NW * ov2, 'W_ovvo derivative')
  kdwkl = xzget(NW * npo * npo, 'W_oooo derivative')
  kxp = xzget(NW * npo * nvp, 'R2 + tau1 derivative')
  kc1 = xzget(NW * npo * npo, 'work [P][K]')
  kr1v = xzget(NW * no * ny, 'vovv part of sigma_1')
  ka2 = xzget(NW * ov2, 'work A')
  kb2 = xzget(NW * ov2, 'work B')
  kx2 = xzget(NW * ov2, 'work X')
  m1 = xzmark()
  kzz = xzget(NW * npo * nv * nv, 'work [P][a][c]')
  kyy = xzget(NW * npo * nv * nv, 'work [P][a][b]')
  if (.not. xzdry) then
    call xztic()
    call N(xzunpack2)(no, nv, npo, nvp, w(jopa), w(jopb), w(jvpa), w(jvpb), w(kr2pp), w(kr2))
    call N(xzer2p)(no, nv, nvp, w(kr2), w(jvpa), w(jvpb), w(kr2p))
    ! H' = [H~, R1]: the Fock matrix and the small blocks
    call N(xzeh1)(no, nv, nvp, w(kr1), w(efov), w(efoo), w(efvv), w(joovv), w(eooovd), w(eovvod), w(ed), &
                  w(jvpa), w(jvpb), w(kdfov), w(kdfoo), w(kdfvv), w(kdfvo), w(kdft))
    call N(xzehs)(no, nv, nvp, w(kr1), w(joovv), w(eooovd), w(eooood), w(eovvod), w(ed), w(jvpidx), &
                  w(kdooov), w(kdoooo), w(kdovvo), w(kdovoo))
    ! the intermediates that are linear in r2: W'_ovvo, F'_vv, F'_oo, W'_oooo
    call N(xzweff)(no, nv, nvp, w(kr2), w(kr2p), w(joovv), w(joovvp), w(kdovvo), w(kdfoo), w(kdfvv), &
                   w(kdweff), w(kdfevv), w(kdfeoo), w(ka2), w(kb2), w(kx2))
    call N(xzewkl)(no, npo, nvp, w(kdoooo), w(jgpp), w(kr2pp), w(jopa), w(jopb), w(kdwkl))
    ! sigma_1
    call N(zc_gemm)(0_4, 0_4, no, ny, no * nvp, one, w(kr2p), no * nvp, w(egmq), ny, zero, w(kr1v), ny)
    call N(xzes1)(no, nv, w(kr1), w(kr2), w(jt2), w(kdfvo), w(kdfov), w(efov), w(efoo), w(efeoo), &
                  w(kdooov), w(eooovd), w(kr1v), w(ks1))
    ! sigma_2: <ab||ij>', the terms of the CCSD equations with r2 and with the derivatives
    call zc_vzero_d(NW * npo * nvp, w(ks2))
    call N(xzes2g)(no, nv, npo, nvp, w(kr1), w(eovood), w(ebv), w(jopa), w(jopb), w(jvpa), w(jvpb), w(ks2))
    call N(xzetau)(no, nv, npo, nvp, w(jt1), w(kr1), w(kr2pp), w(jopa), w(jopb), w(jvpa), w(jvpb), w(kxp))
    ! P(ab) (F_bc r_ijac + F'_bc t_ijac); half of the first is in the ladder operator
    call N(xzefvv)(no, nv, npo, nvp, w(kr2), 0.5d0, -0.5d0, w(jt1), w(kr1), w(efevv), w(jopa), w(jopb), &
                   w(jvpa), w(jvpb), w(ks2), w(kzz), w(kyy))
    call N(xzefvv)(no, nv, npo, nvp, w(jt2), 1d0, 0d0, w(jt1), w(kr1), w(kdfevv), w(jopa), w(jopb), &
                   w(jvpa), w(jvpb), w(ks2), w(kzz), w(kyy))
    call N(xzeb2)(no, nv, npo, nvp, w(kr2), w(kr2p), w(kr2pp), w(ewkl), w(efeoo), w(eovvod), w(eweff), &
                  w(jopa), w(jopb), w(jvpa), w(jvpb), w(ks2), w(ka2), w(kb2), w(kx2))
    call N(xzeb2)(no, nv, npo, nvp, w(jt2), w(et2p), w(et2pp), w(kdwkl), w(kdfeoo), w(kdovvo), w(kdweff), &
                  w(jopa), w(jopb), w(jvpa), w(jvpb), w(ks2), w(ka2), w(kb2), w(kx2))
    call xztoc(5, 'small terms')
  end if
  call xzrel(m1)

  kdwo = 0
  kdwv = 0
  kbd = 0
  if (with3) then
    ! ---- what both residuals take: the operands of pass 2 (the right-hand
    !      sides of the drive terms in blocks: the larger of the two sets)
    nbdw = ny * no * nvp
    if (.not. dense) then
      call xzshctx(0_8, s, s)
      nbdw = nbd
      call xzshctx(s, 0_8, 0_8)
      nbdw = max(nbdw, nbd)
    end if
    kdwo = xzget(NW * ov * npo, 'W_vooo derivative')
    kbd = xzget(NW * nbdw, 'pass-2 operand B')
    kwring = xzget(NW * ov2, 'pass-2 operand W')
    kadrv = xzget(NW * npo * nv * ny, 'pass-2 operand A')
    kdo6 = xzget(merge(0_8, NW * nto * nto, cc3mode /= 0), 'occupied operator, derivative')
    m3 = xzmark()
    ! ---- pass 1 over R3 with the integrals of the ground state, over T3 with H'
    call xzshctx(s, 0_8, 0_8)
    kdwv = xzget(NW * nvp * ov, 'W_vvvo derivative')
    m1 = xzmark()
    kxacc = xzget(NW * ny * no * nvp, 'pass-1 result X')
    kyall = xzget(NW * npo * nv * ny, 'pass-1 result Y')
    kr1c = xzget(NW * ov, 'T3 part of sigma_1')
    kra = xzget(NW * npo * nvp, 'T3 part of sigma_2')
    m2 = xzmark()
    if (.not. xzdry) call zc_vzero_d(NW * n3s, w(ks3))
    if (dense) then
      kbuf = xzget(NW * nbp * ov * nvp, 'pass-1 batch of T3')
      if (.not. xzdry) then
        call zc_vzero_d(NW * ny * no * nvp, w(kxacc))
        call zc_vzero_d(NW * ov, w(kr1c))
        stats = 0
        call N(zc_t3_pass1w)(w(kr3), nto, nvt, no, nv, npo, nvp, w(jotri), w(josgn), w(jvtri), w(jvsgn), &
                             nbp, w(egpd), w(egmq), ny, w(jgpp), w(eft), w(kxacc), w(kyall), w(kr1c), &
                             w(kra), c_null_ptr, stats, w(kbuf))
      end if
    else
      call N(xzbpass1)(w, kr3, w(egpd), w(egmq), w(jgpp), w(efov), w(kxacc), w(kyall), w(kr1c), w(kra))
    end if
    call xzrel(m2)
    if (.not. xzdry) then
      call xztoc(5, 'pass 1 over R3')
      call N(xzeadd1)(no, nv, w(kr1c), w(ks1))
      call N(xzr2t3)(no, nv, npo, nvp, w(jopa), w(jopb), w(jvpa), w(jvpb), w(kyall), w(kxacc), w(kra))
      call N(xzeaxpy)(npo * nvp, w(kra), w(ks2))
      call zc_vzero_d(NW * nvp * ov, w(kdwv))
      ! (CC3: no T3 parts in the derivatives of the drive operators)
      if (cc3mode == 0) then
        call N(zc_add_transposed)(w(kdwv), w(kxacc), nvp, ov, ov, nvp, one)
        call N(xzeyo)(no, nv, npo, w(kyall), w(kdwo))
      end if
      call xztoc(5, 'small terms')
    end if
    ! (of H' only <mn||ei>', <am||ef>' and f'[m][e] are not zero)
    call xzshctx(0_8, s, 0_8)
    kgpd = xzget(NW * ny * npo * nv, 'pass-1 operand, derivative')
    kgmq = xzget(NW * no * nvp * ny, 'pass-1 operand G, derivative')
    kzpp = xzget(NW * npo * nvp, 'zero')
    if (.not. xzdry) then
      call N(xzep1ops)(no, nv, npo, nvp, w(kr1), w(kdooov), w(joovvp), w(jopa), w(jopb), w(kgpd), w(kgmq))
      call zc_vzero_d(NW * npo * nvp, w(kzpp))
      call xztoc(5, 'small terms')
    end if
    if (dense) then
      kbuf = xzget(NW * nbp * ov * nvp, 'pass-1 batch of T3')
      if (.not. xzdry) then
        call zc_vzero_d(NW * ny * no * nvp, w(kxacc))
        call zc_vzero_d(NW * ov, w(kr1c))
        stats = 0
        call N(zc_t3_pass1w)(w(jt3), nto, nvt, no, nv, npo, nvp, w(jotri), w(josgn), w(jvtri), w(jvsgn), &
                             nbp, w(kgpd), w(kgmq), ny, w(kzpp), w(kdft), w(kxacc), w(kyall), w(kr1c), &
                             w(kra), c_null_ptr, stats, w(kbuf))
      end if
    else
      p1_noe = .true.
      call N(xzbpass1)(w, jt3, w(kgpd), w(kgmq), w(kzpp), w(kdfov), w(kxacc), w(kyall), w(kr1c), w(kra))
      p1_noe = .false.
    end if
    if (.not. xzdry) then
      call xztoc(5, 'pass 1 over T3 with H''')
      call N(xzr2t3)(no, nv, npo, nvp, w(jopa), w(jopb), w(jvpa), w(jvpb), w(kyall), w(kxacc), w(kra))
      call N(xzeaxpy)(npo * nvp, w(kra), w(ks2))
    end if
    call xzrel(m1)

    ! ---- the derivatives of W_vvvo (without its W_pp r1 part: in the sweep) and of W_vooo
    kg2 = xzget(NW * npo * ov, 'work [P][e][i]')
    kyb = xzget(NW * nba * nv * nv * no, 'batch of W_vvvo terms Y')
    ku = xzget(NW * nv ** 3, 'dressed <mb||ef> of one m')
    kxb = xzget(NW * nba * no * nv * nv, 'batch of W_vvvo terms X')
    kzero3 = xzget(NW * no * no * ov, 'zero')
    ktmp = xzget(NW * no * no * ov, 'work of W_vooo')
    ktmp2 = xzget(NW * no * no * ov, 'work of W_vooo 2')
    if (.not. xzdry) then
      if (cc3mode == 0) then
        call N(xzewv0)(w, kr1, eweff, kdwv, kyb, ka2)
        ! - f'[m][e] t2[mi][ab], and the T2 terms of W_vvvo with r2
        call zc_vzero_d(NW * no * no * ov, w(kzero3))
        call N(xzwvoovo)(no, nv, npo, nvp, w(et2pp), w(kzero3), w(kdfov), w(jopa), w(jopb), w(kdwv), w(kg2))
        call N(xzwvoovo)(no, nv, npo, nvp, w(kr2pp), w(eooovd), w(efov), w(jopa), w(jopb), w(kdwv), w(kg2))
        call N(xzwvx)(no, nv, nvp, nba, w(kr2), w(ed), w(jvpa), w(jvpb), w(kdwv), w(ka2), w(ku), &
                      w(kxb), w(kyb))
        call xztoc(5, 'W_vvvo derivative')
        call N(xzewo)(no, nv, npo, nvp, w(jt2), w(kr2), w(kr2pp), w(eooovd), w(kdooov), w(kdovoo), w(ed), &
                      w(kr1), w(ekk), w(jopa), w(jopb), w(kdwo), w(ktmp), w(ktmp2), w(kb2))
      else
        ! CC3: <ab||ei>' = - P(ab) r1[m][a] ~<mb||ei> + (W_pp r1 less its F_vv and T2 parts), <ma||ij>'
        call N(xzewv0)(w, kr1, eovvod, kdwv, kyb, ka2)
        call N(xzwvt2out)(no, nv, npo, nvp, w(et2pp), w(kdooov), w(jopa), w(jopb), w(kdwv), w(kg2))
        call xztoc(5, 'W_vvvo derivative')
        call N(xzwo3)(no, nv, npo, w(kdovoo), w(jopa), w(jopb), w(kdwo))
      end if
      call xztoc(5, 'small terms')
    end if
    call xzrel(m1)
  end if
  if (fold) then
    ! ---- CC3, folded: <ab||ei>' (its W_pp r1 part in the sweep below)
    kdwv = xzget(NW * nvp * ov, 'W_vvvo derivative')
    m1 = xzmark()
    kg2 = xzget(NW * npo * ov, 'work [P][e][i]')
    kyb = xzget(NW * nba * nv * nv * no, 'batch of W_vvvo terms Y')
    if (.not. xzdry) then
      call zc_vzero_d(NW * nvp * ov, w(kdwv))
      call N(xzewv0)(w, kr1, eovvod, kdwv, kyb, ka2)
      call N(xzwvt2out)(no, nv, npo, nvp, w(et2pp), w(kdooov), w(jopa), w(jopb), w(kdwv), w(kg2))
      call xztoc(5, 'W_vvvo derivative')
    end if
    call xzrel(m1)
  end if

  ! ---- the sweep over the blocks of the pp operator of the ground state:
  !      its ladders in sigma_2 (with r2 + tau1'; the T2 part of the operator
  !      taken out), W_pp r1 in W'_vvvo, and the pp ladder of R3
  m1 = xzmark()
  kpib = jvpidx
  if (.not. dense) kpib = xzget(nv * nv, 'pairs of a group')
  if (with3 .and. dense) then
    ktv = xzget(NW * nvp * nbt * nv, 'pp-ladder batch of T3')
    kxpp = xzget(NW * nqs * nbt * nv, 'pp-ladder product')
  end if
  call xzshctx(s, 0_8, 0_8)
  do h = 0, npg - 1
    lo = vpoff(h)
    nqh = vpoff(h + 1) - lo
    if (nqh == 0) cycle
    if (.not. xzdry) then
      call xztic()
      if (.not. dense) call N(xzepib)(nv, lo, nqh, w(jvpidx), w(kpib))
      ! sigma_2[P][q] += sum_Q' W_pp[q][Q'] xp[P][Q'] - sum_K (sum_Q' xp[P][Q'] <K||Q'>) t2[K][q]
      if (npo > 0) then
        call N(zc_gemm)(0_4, 1_4, npo, nqh, nqh, one, w(kxp + NW * lo), nvp, w(jvvvv + NW * vvoff(h)), nqh, &
                        one, w(ks2 + NW * lo), nvp)
        call N(zc_gemm)(0_4, 1_4, npo, npo, nqh, one, w(kxp + NW * lo), nvp, w(jgpp + NW * lo), nvp, zero, &
                        w(kc1), npo)
        call N(zc_gemm)(0_4, 0_4, npo, nqh, npo, mone, w(kc1), npo, w(et2pp + NW * lo), nvp, one, &
                        w(ks2 + NW * lo), nvp)
      end if
      if (with3 .or. fold) call N(zc_pair_t1_contract)(w(kdwv + NW * lo * ov), w(jvvvv + NW * vvoff(h)), w(kr1), &
                                                       nqh, nv, nqh, w(kpib), no)
      call xztoc(5, 'pp terms of sigma_2, W_pp r1')
    end if
    ! (CC3: no pp ladder of R3)
    if (.not. with3 .or. cc3mode /= 0) cycle
    do q0 = 0, nqh - 1, nqs
      nq = min(nqs, nqh - q0)
      if (dense) then
        if (xzdry) cycle
        do i0 = 0, nto - 1, nbt
          nb = min(nbt, nto - i0)
          call N(zc_gather_vir_half)(w(kr3), nvt, w(jrows + i0), nb, w(jvtri), w(jvsgn), nv, nvp, w(ktv))
          call N(zc_gemm)(0_4, 0_4, nq, nb * nv, nvp, one, w(jvvvv + NW * q0 * nvp), nvp, w(ktv), nb * nv, zero, &
                          w(kxpp), nb * nv)
          call N(zc_reduce_pair_first_rows)(w(ks3), nvt, w(kxpp), w(jrows + i0), nb, nv, q0, q0 + nq, &
                                            w(jvfree), w(jvpair))
        end do
        call xztoc(5, 'R3: pp ladder')
      else
        call N(xzbppl)(w, kr3, ks3, h, q0, nq, w(jvvvv + NW * (vvoff(h) + q0 * nqh)))
      end if
    end do
  end do
  call xzrel(m1)

  if (with3) then
    ! ---- R3 with the operators of the ground state; the drive terms t2 x W'
    if (.not. xzdry) then
      call xztic()
      call N(xzp2ops)(no, nv, npo, w(jt2), w(eweff), w(kdwo), w(jopa), w(jopb), w(kwring), w(kadrv))
      if (dense) then
        call N(xzbdrv)(no, nv, nvp, w(kdwv), w(et2p), w(kbd))
      else
        call N(xzbbd)(w(kdwv), w(et2p), w(kbd))
      end if
      call xztoc(5, 'small terms')
    end if
    ! (W'_vvvo is in the operand: its place is free for the batches)
    call xzrel(m3)
    if (dense) then
      ktb = xzget(NW * nbp * ov * nvp, 'pass-2 batch of T3')
      kz = xzget(NW * nbp * ov * nvp, 'pass-2 product')
      if (.not. xzdry) then
        stats = 0
        do p0 = 0, npo - 1, nbp
          if (cc3mode /= 0) then
            call N(zc_t3_driver)(nto, nvt, no, nv, npo, nvp, w(jofree), w(jopair), w(jvfree), w(jvpair), &
                                 nbp, p0, min(p0 + nbp, npo), w(kadrv), w(kbd), w(ks3), stats, w(kz))
          else
            call N(zc_t3_pass2r)(w(kr3), nto, nvt, no, nv, npo, nvp, w(jotri), w(josgn), w(jofree), w(jopair), &
                                 w(jvtri), w(jvsgn), w(jvfree), w(jvpair), nbp, p0, min(p0 + nbp, npo), &
                                 w(kwring), w(kadrv), w(kbd), w(ks3), c_null_ptr, stats, w(ktb), w(kz))
          end if
        end do
      end if
    else
      noring = cc3mode /= 0
      call N(xzbpass2)(w, kr3, ks3, w(kwring), w(kadrv), w(kbd), 1_8, 0_8, w(kbd), w(kbd))
      noring = .false.
    end if
    if (.not. xzdry) then
      call xztoc(5, 'R3: pass 2')
      if (cc3mode /= 0) then
        ! CC3: [F, R3] with the diagonal of F
        call N(xzfdiag)(w, kr3, ks3, s, bof_in, .false.)
      else if (dense) then
        call N(zc_t3_occop)(w(eo6), w(kr3), nto, nvt, ncb, c_null_ptr, w(ks3))
      else
        call N(xzbocc)(w, kr3, ks3, w(eo6))
      end if
      call xztoc(5, 'R3: hh ladder, F_oo')
    end if
    call xzrel(m3)
  end if

  ! ---- the sweep over the rows of the derivative of the pp operator,
  !      - P(ab) sum_m r1[m][a] ~<mb||de>  (+ sum_{m<n} r2[mn][ab] <mn||de> + 1/2 A[F'_vv delta]
  !      for the ladder with T3): its ladder with t2 in sigma_2, the pp ladder of T3
  call xzshctx(0_8, s, s)
  m1 = xzmark()
  m2 = xzmark()
  kslab = xzget(NW * nqs * nqmax, 'slab of the pp operator, derivative')
  kgq = ed
  kpib = jvpidx
  if (.not. dense) then
    kgq = xzget(NW * ov * nqmax, '<ov||vv> on the pairs of a group')
    kgh = xzget(NW * npo * nqmax, '<oo||vv> on the pairs of a group')
    kpib = xzget(nv * nv, 'pairs of a group')
  end if
  if (with3 .and. dense) then
    ktv = xzget(NW * nvp * nbt * nv, 'pp-ladder batch of T3')
    kxpp = xzget(NW * nqs * nbt * nv, 'pp-ladder product')
  end if
  do h = 0, npg - 1
    lo = vpoff(h)
    nqh = vpoff(h + 1) - lo
    hr = h + s
    if (nqh == 0 .or. hr < 0 .or. hr >= npg) cycle
    lo2 = vpoff(hr)
    nqr = vpoff(hr + 1) - lo2
    if (nqr == 0) cycle
    if (.not. dense .and. .not. xzdry) then
      call xztic()
      call N(xzgrpcols)(no, nv, npo, nvp, lo, nqh, w(ed), w(jgpp), w(jvpidx), w(kgq), w(kgh), w(kpib))
      call xztoc(5, 'pp operator derivative: rows')
    end if
    do q0 = 0, nqr - 1, nqs
      nq = min(nqs, nqr - q0)
      if (.not. xzdry) then
        call xztic()
        call N(xzedrows)(no, nv, nqh, nq, w(kr1), w(kgq), w(jvpa + lo2 + q0), w(jvpb + lo2 + q0), w(kslab), w(ka2))
        ! sigma_2[P][q] += sum_Q' <q||Q'>' t2[P][Q']
        if (npo > 0) call N(zc_gemm)(0_4, 1_4, npo, nq, nqh, one, w(et2pp + NW * lo), nvp, w(kslab), nqh, one, &
                                     w(ks2 + NW * (lo2 + q0)), nvp)
        if (with3 .and. cc3mode == 0) then
          if (npo > 0) call N(zc_gemm)(1_4, 0_4, nq, nqh, npo, one, w(kr2pp + NW * (lo2 + q0)), nvp, &
                                       w(jgpp + NW * lo), nvp, one, w(kslab), nqh)
          call N(zc_pair_one_body_add)(w(kslab), nq, nqh, w(jvpa + lo2 + q0), w(jvpb + lo2 + q0), w(kpib), nv, &
                                       w(kdfevv), nv, half)
        end if
        call xztoc(5, 'pp operator derivative: rows')
      end if
      ! (CC3: no pp ladder of T3)
      if (.not. with3 .or. cc3mode /= 0) cycle
      if (dense) then
        if (xzdry) cycle
        do i0 = 0, nto - 1, nbt
          nb = min(nbt, nto - i0)
          call N(zc_gather_vir_half)(w(jt3), nvt, w(jrows + i0), nb, w(jvtri), w(jvsgn), nv, nvp, w(ktv))
          call N(zc_gemm)(0_4, 0_4, nq, nb * nv, nvp, one, w(kslab), nvp, w(ktv), nb * nv, zero, &
                          w(kxpp), nb * nv)
          call N(zc_reduce_pair_first_rows)(w(ks3), nvt, w(kxpp), w(jrows + i0), nb, nv, q0, q0 + nq, &
                                            w(jvfree), w(jvpair))
        end do
        call xztoc(5, 'T3: pp ladder')
      else
        call N(xzbppl)(w, jt3, ks3, h, q0, nq, w(kslab))
      end if
    end do
  end do
  call xzrel(m2)

  if (with3) then
    ! ---- T3 with the derivatives of the operators; the drive terms r2 x W
    if (.not. xzdry) then
      call xztic()
      call N(xzp2ops)(no, nv, npo, w(kr2), w(kdweff), w(ewo), w(jopa), w(jopb), w(kwring), w(kadrv))
      if (dense) then
        call N(xzbdrv)(no, nv, nvp, w(ewv), w(kr2p), w(kbd))
      else
        call N(xzbbd)(w(ewv), w(kr2p), w(kbd))
      end if
      if (cc3mode == 0) call N(xzo6)(no, npo, nto, w(jofree), w(jopair), w(kdwkl), w(kdfeoo), w(kdo6))
      call xztoc(5, 'small terms')
    end if
    if (dense) then
      ktb = xzget(NW * nbp * ov * nvp, 'pass-2 batch of T3')
      kz = xzget(NW * nbp * ov * nvp, 'pass-2 product')
      if (.not. xzdry) then
        stats = 0
        do p0 = 0, npo - 1, nbp
          if (cc3mode /= 0) then
            call N(zc_t3_driver)(nto, nvt, no, nv, npo, nvp, w(jofree), w(jopair), w(jvfree), w(jvpair), &
                                 nbp, p0, min(p0 + nbp, npo), w(kadrv), w(kbd), w(ks3), stats, w(kz))
          else
            call N(zc_t3_pass2r)(w(jt3), nto, nvt, no, nv, npo, nvp, w(jotri), w(josgn), w(jofree), w(jopair), &
                                 w(jvtri), w(jvsgn), w(jvfree), w(jvpair), nbp, p0, min(p0 + nbp, npo), &
                                 w(kwring), w(kadrv), w(kbd), w(ks3), c_null_ptr, stats, w(ktb), w(kz))
          end if
        end do
      end if
    else
      noring = cc3mode /= 0
      call N(xzbpass2)(w, jt3, ks3, w(kwring), w(kadrv), w(kbd), 1_8, 0_8, w(kbd), w(kbd))
      noring = .false.
    end if
    if (.not. xzdry) then
      call xztoc(5, 'T3: pass 2')
      if (cc3mode /= 0) then
        ! (CC3: the bare F has no derivative)
      else if (dense) then
        call N(zc_t3_occop)(w(kdo6), w(jt3), nto, nvt, ncb, c_null_ptr, w(ks3))
      else
        call N(xzbocc)(w, jt3, ks3, w(kdo6))
      end if
      call xztoc(5, 'T3: hh ladder, F_oo')
    end if
  end if
  call xzrel(m1)
  call xzshctx(0_8, 0_8, 0_8)
  ! ---- CC3, folded: the triples at the frequency eomega, made per batch
  if (fold) call N(xzefold)(w, kr1, kr2, kr2p, kdwv, kdovoo, kdooov, kdfov, ks1, ks2)
  ! ---- CC4, folded: the T4 terms at the frequency eomega (xzcc4.inc)
  if (efold .and. cc4 .and. with3) call N(xze4)(w, kr1, kr2, kr2p, kr3, kdfov, kdooov, kdoooo, kdovvo, kdovoo, ks2, ks3)

  ! ---- with labels: sigma_1 and sigma_2 on the elements of the shift
  if (.not. dense .and. .not. xzdry) then
    call N(xzemask)(no, nv, npo, nvp, s, ocls, vcls, w(jopgrp), w(jvpgrp), w(ks1), w(ks2))
    call xztoc(5, 'small terms')
  end if
  call xzrel(m0)
end subroutine

! ---------------------------------------------------------------------------
! n elements from a to b (the regions do not overlap)
subroutine N(xzecopy)(n, a, b)
  implicit none
  integer(8), intent(in) :: n
  ZT, intent(in) :: a(n)
  ZT, intent(out) :: b(n)
  integer(8) :: i
  !$omp parallel do
  do i = 1, n
    b(i) = a(i)
  end do
end subroutine

! y += x
subroutine N(xzeaxpy)(n, x, y)
  implicit none
  integer(8), intent(in) :: n
  ZT, intent(in) :: x(n)
  ZT, intent(inout) :: y(n)
  integer(8) :: i
  !$omp parallel do
  do i = 1, n
    y(i) = y(i) + x(i)
  end do
end subroutine

! s1[i][a] += r1c[a][i]
subroutine N(xzeadd1)(no, nv, r1c, s1)
  implicit none
  integer(8), intent(in) :: no, nv
  ZT, intent(in) :: r1c(0:no-1, 0:nv-1)
  ZT, intent(inout) :: s1(0:nv-1, 0:no-1)
  integer(8) :: i, a
  do i = 0, no - 1
    do a = 0, nv - 1
      s1(a, i) = s1(a, i) + r1c(i, a)
    end do
  end do
end subroutine

! wo[m][a][P] = yall[P][a][m]: the part of W_vooo of pass 1
subroutine N(xzeyo)(no, nv, npo, yall, wo)
  implicit none
  integer(8), intent(in) :: no, nv, npo
  ZT, intent(in) :: yall(0:no+nv-1, 0:nv-1, 0:npo-1)           ! [P][d][y]
  ZT, intent(out) :: wo(0:npo-1, 0:nv-1, 0:no-1)               ! [m][a][P]
  integer(8) :: m, a, p
  !$omp parallel do collapse(2) private(m, a, p)
  do m = 0, no - 1
    do a = 0, nv - 1
      do p = 0, npo - 1
        wo(p, a, m) = yall(m, a, p)
      end do
    end do
  end do
end subroutine

! the numbers of the pairs of one group inside it (-1: a pair of another group)
subroutine N(xzepib)(nv, lo, nqh, vpidx, pib)
  implicit none
  integer(8), intent(in) :: nv, lo, nqh
  integer(8), intent(in) :: vpidx(0:nv*nv-1)
  integer(8), intent(out) :: pib(0:nv*nv-1)
  integer(8) :: k
  do k = 0, nv * nv - 1
    pib(k) = -1
    if (vpidx(k) >= lo .and. vpidx(k) < lo + nqh) pib(k) = vpidx(k) - lo
  end do
end subroutine

! r2[i][j][Q] from r2[i][j][a][b]
subroutine N(xzer2p)(no, nv, nvp, r2, vpa, vpb, r2p)
  implicit none
  integer(8), intent(in) :: no, nv, nvp
  ZT, intent(in) :: r2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  integer(8), intent(in) :: vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(out) :: r2p(0:nvp-1, 0:no-1, 0:no-1)
  integer(8) :: i, j, q
  !$omp parallel do collapse(2) private(i, j, q)
  do i = 0, no - 1
    do j = 0, no - 1
      do q = 0, nvp - 1
        r2p(q, j, i) = r2(vpb(q), vpa(q), j, i)
      end do
    end do
  end do
end subroutine

! W_oooo on pairs: wkl[K][P] = oooo[K][P] + sum_Q <K||Q> x2[P][Q]
subroutine N(xzewkl)(no, npo, nvp, oooo, gpp, x2pp, opa, opb, wkl)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, npo, nvp
  ZT, intent(in) :: oooo(0:no-1, 0:no-1, 0:no-1, 0:no-1)       ! [m][n][i][j]
  ZT, intent(in) :: gpp(0:nvp-1, 0:npo-1), x2pp(0:nvp-1, 0:npo-1)
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1)
  ZT, intent(out) :: wkl(0:npo-1, 0:npo-1)                     ! [K][P]
  integer(8) :: kk, p
  ZT :: one
  one = 1
  do kk = 0, npo - 1
    do p = 0, npo - 1
      wkl(p, kk) = oooo(opb(p), opa(p), opb(kk), opa(kk))
    end do
  end do
  if (npo > 0) call N(zc_gemm)(0_4, 1_4, npo, npo, nvp, one, gpp, nvp, x2pp, nvp, one, wkl, npo)
end subroutine

! ---------------------------------------------------------------------------
! The derivative of the dressed Fock matrix along r1 (docs/eom.md, section 2):
!   f'[m][e] = <mn||ef> r1[n][f]
!   f'[m][i] = f[m][e] r1[i][e] + ~<mn||ie> r1[n][e]
!   f'[a][e] = - r1[m][a] f[m][e] - ~<na||ef> r1[n][f]
!   f'[a][i] = - r1[m][a] f[m][i] + f[a][e] r1[i][e] + ~<na||fi> r1[n][f]
subroutine N(xzeh1)(no, nv, nvp, r1, fov, foo, fvv, oovv, ooovd, ovvod, d, vpa, vpb, dfov, dfoo, dfvv, dfvo, dft)
  implicit none
  integer(8), intent(in) :: no, nv, nvp
  ZT, intent(in) :: r1(0:nv-1, 0:no-1)
  ZT, intent(in) :: fov(0:nv-1, 0:no-1), foo(0:no-1, 0:no-1), fvv(0:nv-1, 0:nv-1)
  ZT, intent(in) :: oovv(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)       ! [m][n][e][f]
  ZT, intent(in) :: ooovd(0:nv-1, 0:no-1, 0:no-1, 0:no-1)      ! [m][n][i][e]
  ZT, intent(in) :: ovvod(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)      ! [m][b][e][j]
  ZT, intent(in) :: d(0:nvp-1, 0:nv-1, 0:no-1)                 ! [m][b][Q]
  integer(8), intent(in) :: vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(out) :: dfov(0:nv-1, 0:no-1)                      ! [m][e]
  ZT, intent(out) :: dfoo(0:no-1, 0:no-1)                      ! [m][i]
  ZT, intent(out) :: dfvv(0:nv-1, 0:nv-1)                      ! [a][e]
  ZT, intent(out) :: dfvo(0:no-1, 0:nv-1)                      ! [a][i]
  ZT, intent(out) :: dft(0:no-1, 0:nv-1)                       ! [e][m]
  integer(8) :: m, n, e, f, i, a, b, q
  ZT :: s
  !$omp parallel do private(m, e, n, f, i, s)
  do m = 0, no - 1
    do e = 0, nv - 1
      s = 0
      do n = 0, no - 1
        do f = 0, nv - 1
          s = s + oovv(f, e, n, m) * r1(f, n)
        end do
      end do
      dfov(e, m) = s
      dft(m, e) = s
    end do
    do i = 0, no - 1
      s = 0
      do e = 0, nv - 1
        s = s + fov(e, m) * r1(e, i)
      end do
      do n = 0, no - 1
        do e = 0, nv - 1
          s = s + ooovd(e, i, n, m) * r1(e, n)
        end do
      end do
      dfoo(i, m) = s
    end do
  end do
  dfvv = 0
  !$omp parallel do private(b, i, q)
  do b = 0, nv - 1
    do i = 0, no - 1
      do q = 0, nvp - 1
        dfvv(vpa(q), b) = dfvv(vpa(q), b) - d(q, b, i) * r1(vpb(q), i)
        dfvv(vpb(q), b) = dfvv(vpb(q), b) + d(q, b, i) * r1(vpa(q), i)
      end do
    end do
  end do
  !$omp parallel do private(a, e, m, i, n, f, s)
  do a = 0, nv - 1
    do e = 0, nv - 1
      s = dfvv(e, a)
      do m = 0, no - 1
        s = s - r1(a, m) * fov(e, m)
      end do
      dfvv(e, a) = s
    end do
    do i = 0, no - 1
      s = 0
      do m = 0, no - 1
        s = s - r1(a, m) * foo(i, m)
      end do
      do e = 0, nv - 1
        s = s + fvv(e, a) * r1(e, i)
      end do
      do n = 0, no - 1
        do f = 0, nv - 1
          s = s + ovvod(i, f, a, n) * r1(f, n)
        end do
      end do
      dfvo(i, a) = s
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The derivatives of the small dressed blocks along r1:
!   <mn||ie>' = - <mn||ef> r1[i][f]
!   <mn||ij>' = ~<mn||ie> r1[j][e] - ~<mn||je> r1[i][e]
!   <ma||di>' = ~<ma||de> r1[i][e] + r1[n][a] ~<mn||id>
!   <ma||xy>' = ~<ma||ey> r1[x][e] - ~<ma||ex> r1[y][e] - r1[n][a] ~<mn||xy>
subroutine N(xzehs)(no, nv, nvp, r1, oovv, ooovd, ooood, ovvod, d, vpidx, dooov, doooo, dovvo, dovoo)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, nvp
  ZT, intent(in) :: r1(0:nv-1, 0:no-1)
  ZT, intent(in) :: oovv(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)       ! [m][n][e][f]
  ZT, intent(in) :: ooovd(0:nv-1, 0:no-1, 0:no-1, 0:no-1)      ! [m][n][i][e]
  ZT, intent(in) :: ooood(0:no-1, 0:no-1, 0:no-1, 0:no-1)      ! [m][n][i][j]
  ZT, intent(in) :: ovvod(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)      ! [m][b][e][j]
  ZT, intent(in) :: d(*)                                       ! [m][b][Q]
  integer(8), intent(in) :: vpidx(*)
  ZT, intent(out) :: dooov(0:nv-1, 0:no-1, 0:no-1, 0:no-1)     ! [m][n][i][e]
  ZT, intent(out) :: doooo(0:no-1, 0:no-1, 0:no-1, 0:no-1)     ! [m][n][i][j]
  ZT, intent(out) :: dovvo(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)     ! [m][a][d][i]
  ZT, intent(out) :: dovoo(0:no-1, 0:no-1, 0:nv-1, 0:no-1)     ! [m][a][x][y]
  integer(8) :: m, n, i, j, e, f, a, dd, x, y
  ZT :: s
  !$omp parallel do collapse(2) private(m, n, i, j, e, f, s)
  do m = 0, no - 1
    do n = 0, no - 1
      do i = 0, no - 1
        do e = 0, nv - 1
          s = 0
          do f = 0, nv - 1
            s = s - oovv(f, e, n, m) * r1(f, i)
          end do
          dooov(e, i, n, m) = s
        end do
      end do
      do i = 0, no - 1
        do j = 0, no - 1
          s = 0
          do e = 0, nv - 1
            s = s + ooovd(e, i, n, m) * r1(e, j) - ooovd(e, j, n, m) * r1(e, i)
          end do
          doooo(j, i, n, m) = s
        end do
      end do
    end do
  end do
  !$omp parallel do collapse(2) private(m, a, dd, i, n, x, y, e, s)
  do m = 0, no - 1
    do a = 0, nv - 1
      do dd = 0, nv - 1
        do i = 0, no - 1
          s = 0
          do n = 0, no - 1
            s = s + r1(a, n) * ooovd(dd, i, n, m)
          end do
          dovvo(i, dd, a, m) = s
        end do
      end do
      do x = 0, no - 1
        do y = 0, no - 1
          s = 0
          do e = 0, nv - 1
            s = s + ovvod(y, e, a, m) * r1(e, x) - ovvod(x, e, a, m) * r1(e, y)
          end do
          do n = 0, no - 1
            s = s - r1(a, n) * ooood(y, x, n, m)
          end do
          dovoo(y, x, a, m) = s
        end do
      end do
    end do
  end do
  ! + sum_e ~<ma||de> r1[i][e], from the dressed <ma||Q>
  call N(zc_pair_t1_contract)(dovvo, d, r1, no * nv, nv, nvp, vpidx, no)
end subroutine

! ---------------------------------------------------------------------------
! sigma_1[i][a] = f'[a][i] + f'[k][c] t2[ik][ac] + f[k][c] r2[ik][ac]
!                 - sum_n (F_oo - f)[n][i] r1[n][a]              (the bra of ~<ak||cd> t2[ik][cd] / 2)
!                 + sum_{k,Q} r2[i][k][Q] ~<ak||Q>               (r1v, at the columns no + a)
!                 - 1/2 (<kl||ic>' t2[kl][ac] + ~<kl||ic> r2[kl][ac])
subroutine N(xzes1)(no, nv, r1, r2, t2, dfvo, dfov, fov, foo, feoo, dooov, ooovd, r1v, s1)
  implicit none
  integer(8), intent(in) :: no, nv
  ZT, intent(in) :: r1(0:nv-1, 0:no-1)
  ZT, intent(in) :: r2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1), t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: dfvo(0:no-1, 0:nv-1), dfov(0:nv-1, 0:no-1), fov(0:nv-1, 0:no-1)
  ZT, intent(in) :: foo(0:no-1, 0:no-1), feoo(0:no-1, 0:no-1)
  ZT, intent(in) :: dooov(0:nv-1, 0:no-1, 0:no-1, 0:no-1), ooovd(0:nv-1, 0:no-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: r1v(0:no+nv-1, 0:no-1)                     ! [i][y]
  ZT, intent(out) :: s1(0:nv-1, 0:no-1)
  integer(8) :: i, a, k, c, m, n
  ZT :: s
  !$omp parallel do collapse(2) private(i, a, k, c, m, n, s)
  do i = 0, no - 1
    do a = 0, nv - 1
      s = dfvo(i, a) + r1v(no + a, i)
      do k = 0, no - 1
        do c = 0, nv - 1
          s = s + dfov(c, k) * t2(c, a, k, i) + fov(c, k) * r2(c, a, k, i)
        end do
      end do
      do n = 0, no - 1
        s = s - (feoo(i, n) - foo(i, n)) * r1(a, n)
      end do
      do m = 0, no - 1
        do n = 0, no - 1
          do c = 0, nv - 1
            s = s - 0.5d0 * (dooov(c, i, n, m) * t2(c, a, n, m) + ooovd(c, i, n, m) * r2(c, a, n, m))
          end do
        end do
      end do
      s1(a, i) = s
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! sigma_2[P][Q] += <ab||ij>' = - P(ab) r1[m][a] ~<mb||ij> + P(ij) sum_e bv[Q][e][j] r1[i][e]
! (bv: the bra-dressed <ab||ej>; its ket part comes with the ladder on tau1')
subroutine N(xzes2g)(no, nv, npo, nvp, r1, ovood, bv, opa, opb, vpa, vpb, s2)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: r1(0:nv-1, 0:no-1)
  ZT, intent(in) :: ovood(0:no-1, 0:no-1, 0:nv-1, 0:no-1)      ! [m][b][i][j]
  ZT, intent(in) :: bv(0:no-1, 0:nv-1, 0:nvp-1)                ! [Q][e][i]
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(inout) :: s2(0:nvp-1, 0:npo-1)
  ZT :: x(0:no-1, 0:no-1)
  integer(8) :: p, q, i, j, a, b, m, e
  ZT :: s
  !$omp parallel do private(p, q, i, j, a, b, m, s)
  do p = 0, npo - 1
    i = opa(p)
    j = opb(p)
    do q = 0, nvp - 1
      a = vpa(q)
      b = vpb(q)
      s = 0
      do m = 0, no - 1
        s = s - r1(a, m) * ovood(j, i, b, m) + r1(b, m) * ovood(j, i, a, m)
      end do
      s2(q, p) = s2(q, p) + s
    end do
  end do
  !$omp parallel do private(q, p, i, e, x)
  do q = 0, nvp - 1
    do i = 0, no - 1
      x(:, i) = 0
      do e = 0, nv - 1
        x(:, i) = x(:, i) + bv(:, e, q) * r1(e, i)
      end do
    end do
    do p = 0, npo - 1
      s2(q, p) = s2(q, p) + x(opb(p), opa(p)) - x(opa(p), opb(p))
    end do
  end do
end subroutine

! xp[P][Q] = r2[P][Q] + tau1'[P][Q], tau1' the derivative of
! t1[m][a] t1[n][b] - t1[n][a] t1[m][b] along r1
subroutine N(xzetau)(no, nv, npo, nvp, t1, r1, r2pp, opa, opb, vpa, vpb, xp)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: t1(0:nv-1, 0:no-1), r1(0:nv-1, 0:no-1)
  ZT, intent(in) :: r2pp(0:nvp-1, 0:npo-1)
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(out) :: xp(0:nvp-1, 0:npo-1)
  integer(8) :: p, q, m, n, a, b
  !$omp parallel do private(p, q, m, n, a, b)
  do p = 0, npo - 1
    m = opa(p)
    n = opb(p)
    do q = 0, nvp - 1
      a = vpa(q)
      b = vpb(q)
      xp(q, p) = r2pp(q, p) + r1(a, m) * t1(b, n) - r1(a, n) * t1(b, m) + t1(a, m) * r1(b, n) - t1(a, n) * r1(b, m)
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! sigma_2[P][(a<b)] += P(ab) sum_c z[P][a][c] F[b][c],
!   z[P][a][c] = cx x2[i][j][a][c] + ct tau1'[P][a][c]
! zz, yy: work npo * nv * nv each.
subroutine N(xzefvv)(no, nv, npo, nvp, x2, cx, ct, t1, r1, fvv, opa, opb, vpa, vpb, s2, zz, yy)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: x2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  real(8), intent(in) :: cx, ct
  ZT, intent(in) :: t1(0:nv-1, 0:no-1), r1(0:nv-1, 0:no-1)
  ZT, intent(in) :: fvv(0:nv-1, 0:nv-1)                        ! [b][c]
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(inout) :: s2(0:nvp-1, 0:npo-1)
  ZT :: zz(0:nv-1, 0:nv-1, 0:npo-1)                            ! [P][a][c]
  ZT :: yy(0:nv-1, 0:nv-1, 0:npo-1)                            ! [P][a][b]
  integer(8) :: p, q, i, j, a, c
  ZT :: one, zero
  one = 1
  zero = 0
  if (npo == 0) return
  !$omp parallel do private(p, i, j, a, c)
  do p = 0, npo - 1
    i = opa(p)
    j = opb(p)
    do a = 0, nv - 1
      do c = 0, nv - 1
        zz(c, a, p) = cx * x2(c, a, j, i)
      end do
    end do
    if (ct /= 0d0) then
      do a = 0, nv - 1
        do c = 0, nv - 1
          zz(c, a, p) = zz(c, a, p) + ct * (r1(a, i) * t1(c, j) - r1(a, j) * t1(c, i) &
                                             + t1(a, i) * r1(c, j) - t1(a, j) * r1(c, i))
        end do
      end do
    end if
  end do
  ! yy[(P, a)][b] = sum_c zz[(P, a)][c] F[b][c]
  call N(zc_gemm)(0_4, 1_4, npo * nv, nv, nv, one, zz, nv, fvv, nv, zero, yy, nv)
  !$omp parallel do private(p, q)
  do p = 0, npo - 1
    do q = 0, nvp - 1
      s2(q, p) = s2(q, p) + yy(vpb(q), vpa(q), p) - yy(vpa(q), vpb(q), p)
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The terms of the T2 equation that are bilinear in an amplitude x2 and an
! intermediate, added to sigma_2[P][Q]:
!   1/2 W_klij x_klab - P(ij) F_kj x_ikab + P(ij) P(ab) W_kbcj x_ikac,
!   W_kbcj = (w1 + w2)[k][b][c][j] / 2
! a2, b2, xx: work (no nv)**2 each.
subroutine N(xzeb2)(no, nv, npo, nvp, x2, x2p, x2pp, wkl, foo, w1, w2, opa, opb, vpa, vpb, s2, a2, b2, xx)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: x2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)         ! [i][j][a][b]
  ZT, intent(in) :: x2p(0:nvp-1, 0:no-1, 0:no-1)               ! [i][j][Q]
  ZT, intent(in) :: x2pp(0:nvp-1, 0:npo-1)
  ZT, intent(in) :: wkl(0:npo-1, 0:npo-1)                      ! [K][P]
  ZT, intent(in) :: foo(0:no-1, 0:no-1)                        ! [m][i]
  ZT, intent(in) :: w1(0:no-1, 0:nv-1, 0:nv-1, 0:no-1), w2(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)     ! [k][b][c][j]
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(inout) :: s2(0:nvp-1, 0:npo-1)
  ZT :: a2(0:nv-1, 0:no-1, 0:nv-1, 0:no-1)                     ! [(i, a)][(k, c)]
  ZT :: b2(0:nv-1, 0:no-1, 0:nv-1, 0:no-1)                     ! [(k, c)][(j, b)]
  ZT :: xx(0:nv-1, 0:no-1, 0:nv-1, 0:no-1)                     ! [(i, a)][(j, b)]
  integer(8) :: i, j, k, a, b, c, p, q
  ZT :: s, one, zero
  one = 1
  zero = 0
  if (npo == 0) return
  call N(zc_gemm)(1_4, 0_4, npo, nvp, npo, one, wkl, npo, x2pp, nvp, one, s2, nvp)
  !$omp parallel do private(p, q, i, j, k, s)
  do p = 0, npo - 1
    i = opa(p)
    j = opb(p)
    do q = 0, nvp - 1
      s = 0
      do k = 0, no - 1
        s = s - x2p(q, k, i) * foo(j, k) + x2p(q, k, j) * foo(i, k)
      end do
      s2(q, p) = s2(q, p) + s
    end do
  end do
  !$omp parallel do collapse(2) private(i, a, k, c)
  do i = 0, no - 1
    do a = 0, nv - 1
      do k = 0, no - 1
        do c = 0, nv - 1
          a2(c, k, a, i) = x2(c, a, k, i)
          b2(a, i, c, k) = 0.5d0 * (w1(i, c, a, k) + w2(i, c, a, k))
        end do
      end do
    end do
  end do
  call N(zc_gemm)(0_4, 0_4, no * nv, no * nv, no * nv, one, a2, no * nv, b2, no * nv, zero, xx, no * nv)
  !$omp parallel do private(p, q, i, j, a, b)
  do p = 0, npo - 1
    i = opa(p)
    j = opb(p)
    do q = 0, nvp - 1
      a = vpa(q)
      b = vpb(q)
      s2(q, p) = s2(q, p) + xx(b, j, a, i) - xx(b, i, a, j) - xx(a, j, b, i) + xx(a, i, b, j)
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The operands of pass 1 over T3 with H' = [H~, R1]:
!   gpd[e][P][f] = 0,  gpd[nv+j][P][e] = <mn||ej>' = - <mn||je>'
!   gmq[n][R][m'] = 0, gmq[m][R][no+a] = <am||R>' = sum_n r1[n][a] <mn||R>
subroutine N(xzep1ops)(no, nv, npo, nvp, r1, dooov, oovvp, opa, opb, gpd, gmq)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: r1(0:nv-1, 0:no-1)
  ZT, intent(in) :: dooov(0:nv-1, 0:no-1, 0:no-1, 0:no-1)      ! [m][n][i][e]
  ZT, intent(in) :: oovvp(0:nvp-1, 0:no-1, 0:no-1)             ! [m][n][Q]
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1)
  ZT, intent(out) :: gpd(0:nv-1, 0:npo-1, 0:nv+no-1)           ! [e|j][P][f]
  ZT, intent(out) :: gmq(0:no+nv-1, 0:nvp-1, 0:no-1)           ! [m][R][y]
  integer(8) :: m, n, p, e, j, r, a
  ZT :: s
  !$omp parallel do private(e)
  do e = 0, nv - 1
    gpd(:, :, e) = 0
  end do
  do j = 0, no - 1
    do p = 0, npo - 1
      gpd(:, p, nv + j) = -dooov(:, j, opb(p), opa(p))
    end do
  end do
  !$omp parallel do collapse(2) private(m, r, a, n, s)
  do m = 0, no - 1
    do r = 0, nvp - 1
      gmq(0:no - 1, r, m) = 0
      do a = 0, nv - 1
        s = 0
        do n = 0, no - 1
          s = s + r1(a, n) * oovvp(r, n, m)
        end do
        gmq(no + a, r, m) = s
      end do
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The F_vv part of sum_f W_pp[ab][ef] r1[i][f], taken out of W'_vvvo[Q][e][i]:
!   - 1/2 (F[a][e] r1[i][b] - F[b][e] r1[i][a]) + 1/2 (F r1)[a][i] d(e,b) - 1/2 (F r1)[b][i] d(e,a)
! fr: work no * nv.
subroutine N(xzewvf)(no, nv, nvp, fvv, r1, vpa, vpb, wv, fr)
  implicit none
  integer(8), intent(in) :: no, nv, nvp
  ZT, intent(in) :: fvv(0:nv-1, 0:nv-1)                        ! [a][e]
  ZT, intent(in) :: r1(0:nv-1, 0:no-1)
  integer(8), intent(in) :: vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(inout) :: wv(0:no-1, 0:nv-1, 0:nvp-1)             ! [Q][e][i]
  ZT :: fr(0:no-1, 0:nv-1)                                     ! [a][i]
  integer(8) :: q, a, b, e, i, c
  ZT :: s
  do a = 0, nv - 1
    do i = 0, no - 1
      s = 0
      do c = 0, nv - 1
        s = s + fvv(c, a) * r1(c, i)
      end do
      fr(i, a) = s
    end do
  end do
  !$omp parallel do private(q, a, b, e, i)
  do q = 0, nvp - 1
    a = vpa(q)
    b = vpb(q)
    do e = 0, nv - 1
      do i = 0, no - 1
        wv(i, e, q) = wv(i, e, q) - 0.5d0 * (fvv(e, a) * r1(b, i) - fvv(e, b) * r1(a, i))
      end do
    end do
    do i = 0, no - 1
      wv(i, b, q) = wv(i, b, q) + 0.5d0 * fr(i, a)
      wv(i, a, q) = wv(i, a, q) - 0.5d0 * fr(i, b)
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The derivative of W_vooo[m][a][P], P = (i<j), added to its part of pass 1
! over R3 (which wo holds on entry):
!   - <ma||ij>' - tmp'[m][a][i][j] + tmp'[m][a][j][i] - sum_Q ~<ma||Q> r2[P][Q]
!   + sum_n r1[n][a] kk[m][n][P]                                (- <ma||Q>' t2[P][Q])
!   tmp'[m][b][i][j] = sum_ne (<mn||ie>' t2[j][n][b][e] + ~<mn||ie> r2[j][n][b][e])
! c, a: work no**3 * nv each; b2: work (no nv)**2.
subroutine N(xzewo)(no, nv, npo, nvp, t2, r2, r2pp, ooovd, dooov, dovoo, d, r1, kk, opa, opb, wo, c, a, b2)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1), r2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: r2pp(0:nvp-1, 0:npo-1)
  ZT, intent(in) :: ooovd(0:nv-1, 0:no-1, 0:no-1, 0:no-1), dooov(0:nv-1, 0:no-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: dovoo(0:no-1, 0:no-1, 0:nv-1, 0:no-1)      ! [m][b][i][j]
  ZT, intent(in) :: d(*)
  ZT, intent(in) :: r1(0:nv-1, 0:no-1)
  ZT, intent(in) :: kk(0:npo-1, 0:no-1, 0:no-1)                ! [m][n][P]
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1)
  ZT, intent(inout) :: wo(0:npo-1, 0:nv-1, 0:no-1)             ! [m][a][P]
  ZT :: c(0:no-1, 0:nv-1, 0:no-1, 0:no-1)                      ! [m][i][b][j]
  ZT :: a(0:nv-1, 0:no-1, 0:no-1, 0:no-1)                      ! [m][i][n][e]
  ZT :: b2(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)                     ! [n][e][b][j]
  integer(8) :: m, b, i, j, n, e, p, pass
  ZT :: one, mone, zero, s
  one = 1
  mone = -1
  zero = 0
  if (npo == 0) return
  do pass = 1, 2
    !$omp parallel do collapse(2) private(m, i, n, e)
    do m = 0, no - 1
      do i = 0, no - 1
        do n = 0, no - 1
          do e = 0, nv - 1
            if (pass == 1) then
              a(e, n, i, m) = dooov(e, i, n, m)
            else
              a(e, n, i, m) = ooovd(e, i, n, m)
            end if
          end do
        end do
      end do
    end do
    !$omp parallel do collapse(2) private(n, e, b, j)
    do n = 0, no - 1
      do e = 0, nv - 1
        do b = 0, nv - 1
          do j = 0, no - 1
            if (pass == 1) then
              b2(j, b, e, n) = t2(e, b, n, j)
            else
              b2(j, b, e, n) = r2(e, b, n, j)
            end if
          end do
        end do
      end do
    end do
    call N(zc_gemm)(0_4, 0_4, no * no, nv * no, no * nv, one, a, no * nv, b2, nv * no, zero, c, nv * no)
    !$omp parallel do collapse(2) private(m, b, p, i, j)
    do m = 0, no - 1
      do b = 0, nv - 1
        do p = 0, npo - 1
          i = opa(p)
          j = opb(p)
          wo(p, b, m) = wo(p, b, m) - c(j, b, i, m) + c(i, b, j, m)
        end do
      end do
    end do
  end do
  ! - sum_Q D[(m, a)][Q] r2pp[P][Q]
  call N(zc_gemm)(0_4, 1_4, no * nv, npo, nvp, mone, d, nvp, r2pp, nvp, one, wo, npo)
  !$omp parallel do collapse(2) private(m, b, p, i, j, n, s)
  do m = 0, no - 1
    do b = 0, nv - 1
      do p = 0, npo - 1
        i = opa(p)
        j = opb(p)
        s = -dovoo(j, i, b, m)
        do n = 0, no - 1
          s = s + r1(b, n) * kk(p, n, m)
        end do
        wo(p, b, m) = wo(p, b, m) + s
      end do
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! Rows of <ab||de>' = - P(ab) sum_m r1[m][a] ~<mb||de> for the pairs (a<b) of
! pa, pb and the pairs (d<e) of one group (gq: ~<my||Q'> on these).
! tm: work no * nv.
subroutine N(xzedrows)(no, nv, nqh, nq, r1, gq, pa, pb, slab, tm)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, nqh, nq
  ZT, intent(in) :: r1(0:nv-1, 0:no-1)
  ZT, intent(in) :: gq(*)
  integer(8), intent(in) :: pa(*), pb(*)
  ZT :: slab(*)
  ZT :: tm(0:nv-1, 0:no-1)
  tm = -r1
  call zc_vzero_d(NW * nq * nqh, slab)
  call N(zc_vvvv_dress)(slab, slab, nq, nqh, pa, pb, tm, nv, gq, no, nv, r1, nv, gq, no, nv, tm, tm, 0_8)
end subroutine

! ---------------------------------------------------------------------------
! sigma_1 and sigma_2 on the elements of the label shift s (classes):
! s(a) - s(i) = s, the group of (a<b) less that of (i<j) = s.
subroutine N(xzemask)(no, nv, npo, nvp, s, ocls, vcls, opgrp, vpgrp, s1, s2)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp, s
  integer(8), intent(in) :: ocls(0:no-1), vcls(0:nv-1), opgrp(0:npo-1), vpgrp(0:nvp-1)
  ZT, intent(inout) :: s1(0:nv-1, 0:no-1), s2(0:nvp-1, 0:npo-1)
  integer(8) :: i, a, p, q
  do i = 0, no - 1
    do a = 0, nv - 1
      if (vcls(a) - ocls(i) /= s) s1(a, i) = 0
    end do
  end do
  !$omp parallel do private(p, q)
  do p = 0, npo - 1
    do q = 0, nvp - 1
      if (vpgrp(q) - opgrp(p) /= s) s2(q, p) = 0
    end do
  end do
end subroutine
