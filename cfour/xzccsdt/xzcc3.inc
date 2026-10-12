!
! CC3 (type-generic; docs/native.md, section 17; docs/ccn.md).  The model:
! the singles and doubles equations of CCSDT, the triples equation to the
! lowest order,
!
!   <T| [V~, T2] + [F, T3] |0> = 0,
!
! V~ the T1-dressed two-body part, F the Fock operator of the reference
! (diagonal for CFOUR's canonical spinors).  Two routes, as in the Python
! program (src/zccsdt/cc/ccsdt_opt.py model 'cc3', src/zccsdt/cc/cc3fly.py):
!
!   cc3mode 1   T3 stored: the CCSDT step (xzstep.inc) with the T3 couplings
!               of the triples equation off - no ring, no pp and hh ladder,
!               the drive operators W_vvvo, W_vooo without their T2 and T3
!               parts, [F, T3] by the diagonal of F; the T3 terms of the
!               singles and doubles equations (pass 1) kept.  Its EOM sigma
!               vector is that of EOM-CCSDT with the same switches
!               (xzeom.inc), in all label shifts.
!   cc3mode 2   no T3 (N(xzstep3)): the triples of a batch of occupied triples
!               made from the drive terms, divided by the denominator and
!               contracted into the singles and doubles equations at once
!               (csrc/cc3_fly.c, N(xzfly)); DIIS on T1 and T2.  The excited
!               states: the folded eigenproblem in the space of singles and
!               doubles, a sigma vector at a frequency omega (N(xzefold)),
!               root by root to self-consistency in omega (N(xzeom3)).
!

! ---------------------------------------------------------------------------
! W_vooo of CC3: wo[m][a][P] = - ~<ma||ij>, P = (i<j) (the dressed integral
! alone; the sign as in xzwo)
subroutine N(xzwo3)(no, nv, npo, ovood, opa, opb, wo)
  implicit none
  integer(8), intent(in) :: no, nv, npo
  ZT, intent(in) :: ovood(0:no-1, 0:no-1, 0:nv-1, 0:no-1)      ! [m][b][i][j]
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1)
  ZT, intent(out) :: wo(0:npo-1, 0:nv-1, 0:no-1)               ! [m][a][P]
  integer(8) :: m, b, p
  !$omp parallel do collapse(2) private(m, b, p)
  do m = 0, no - 1
    do b = 0, nv - 1
      do p = 0, npo - 1
        wo(p, b, m) = -ovood(opb(p), opa(p), b, m)
      end do
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! r += (e_abc - e_ijk) t on the blocks of a rank-3 array of label shift s
! (classes; bo: the first element of the block of every group of occupied
! triples): [F, T3] of the triples equation of CC3 with the diagonal of F.
! rows: only the rows of the slice of this rank (xzdist).
subroutine N(xzfdiag)(w, kt3, kr3, s, bo, rows)
  use xzdata
  use xzkern
  use xzdist, only: xzslrows
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kt3, kr3, s
  integer(8), intent(in) :: bo(0:*)
  logical, intent(in) :: rows
  integer(8) :: g, ni, na, a, b
  do g = 0, ntg - 1
    ni = otoff(g + 1) - otoff(g)
    na = xzna(s, g)
    if (ni == 0 .or. na == 0) cycle
    a = 0
    b = ni
    if (rows) call xzslrows(g, a, b)
    if (b <= a) cycle
    call N(zc_t3_fdiag)(w(kt3 + NW * (bo(g) + a * na)), w(kr3 + NW * (bo(g) + a * na)), b - a, na, &
                        w(jeo3 + otoff(g) + a), w(jev3 + vtoff(g + s)))
  end do
end subroutine

! ---------------------------------------------------------------------------
! W'_vvvo -= sum_{m<n} t2[mn][ab] <mn||ei>': the T2 part of W_pp r1, which the
! sweep over the pp operator puts into the derivative of W_vvvo, taken out
! again (the drive operator of CC3 has no T2 part).  dooov[m][n][i][e] =
! <mn||ie>'.  g2: work [P][e][i].
subroutine N(xzwvt2out)(no, nv, npo, nvp, t2pp, dooov, opa, opb, wv, g2)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: t2pp(0:nvp-1, 0:npo-1)
  ZT, intent(in) :: dooov(0:nv-1, 0:no-1, 0:no-1, 0:no-1)      ! [m][n][i][e]
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1)
  ZT :: wv(*)
  ZT :: g2(0:no-1, 0:nv-1, 0:npo-1)                            ! [P][e][i]
  integer(8) :: p, e, i
  ZT :: one, mone
  one = 1
  mone = -1
  if (npo == 0) return
  !$omp parallel do private(p, e, i)
  do p = 0, npo - 1
    do e = 0, nv - 1
      do i = 0, no - 1
        g2(i, e, p) = -dooov(e, i, opb(p), opa(p))
      end do
    end do
  end do
  call N(zc_gemm)(1_4, 0_4, nvp, nv * no, npo, mone, t2pp, nvp, g2, nv * no, one, wv, nv * no)
end subroutine

! ---------------------------------------------------------------------------
! The derivative of W_vvvo along r1, first part (xzeom.inc):
!   W'[Q][e][i] += - P(ab) sum_m r1[m][a] W[m][b][e][i] - (the F_vv part of W_pp r1)
! W: W_ovvo (EOM-CCSDT), or the dressed <mb||ei> (CC3: the drive operator is
! the dressed <ab||ei> alone).  yb: work nba nv nv no; fr: work no nv.
subroutine N(xzewv0)(w, kr1, kw, kdwv, kyb, kfr)
  use xzdata
  use xzkern
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kr1, kw, kdwv, kyb, kfr
  integer(8) :: x0, nb
  ZT :: one, zero, mone
  one = 1
  zero = 0
  mone = -1
  do x0 = 0, nv - 1, nba
    nb = min(nba, nv - x0)
    call N(zc_gemm)(1_4, 0_4, nb, nv * nv * no, no, one, w(kr1 + NW * x0), nv, w(kw), nv * nv * no, zero, &
                    w(kyb), nv * nv * no)
    call N(zc_pair_antisym_batch_add)(w(kdwv), w(kyb), nv, nvp, w(jvpa), w(jvpb), nv * no, x0, x0 + nb, mone)
  end do
  ! the F_vv part of W_pp r1 is taken out again
  call N(xzewvf)(no, nv, nvp, w(efevv), w(kr1), w(jvpa), w(jvpb), w(kdwv), w(kfr))
end subroutine

! ---------------------------------------------------------------------------
! One amplitude step of the model of the run: CCSDT or CC3 with T3 stored
! (xzstep.inc), or CC3 without stored triples (xzstep3; kt3, kr3 unused).
subroutine N(xzstepm)(w, kt1, kt2, kt3, kt1n, kt2n, kr3)
  use xzdata
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kt1, kt2, kt3, kt1n, kt2n, kr3
  if (cc3mode == 2) then
    call N(xzstep3)(w, kt1, kt2, kt1n, kt2n)
  else
    call N(xzstep)(w, kt1, kt2, kt3, kt1n, kt2n, kr3)
  end if
end subroutine

! ---------------------------------------------------------------------------
! One Jacobi step of CC3 without stored triples: the CCSD part of the CCSDT
! step (xzstep.inc: the dressing, the dressed <ab||ij>, W_ovvo, F, the sweep
! over the rows of the dressed <ab||cd> with the CCSD ladder and their part
! of the dressed <ab||ei>), and the T3 terms of the singles and doubles
! equations from the batches of occupied triples (N(xzfly)): the triples
!   t3 = drive[t2; <ab||ei>~, <ma||ij>~] / (e_ijk - e_abc)
! made batch by batch and contracted at once - no array of the size of T3.
! The operands of the batches (docs/ccn.md, section 10):
!   At = t2 [i][j][a][b] as it is; Ao[hi][lo][c][m] = <mc||lo hi>~;
!   Bv[e][i][Q] = <Q||ei>~; Bo = t2[m][i][Q]; Gv[m][Q][y] = ~<ym||Q>;
!   Gsq = the dressed <mn||ie> as it is; gsq = <mn||Q>; fov = f~[m][e].
subroutine N(xzstep3)(w, kt1, kt2, kt1n, kt2n)
  use xzdata
  use xzmem
  use xzkern
  use xzdist
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kt1, kt2, kt1n, kt2n
  integer(8) :: m0, md, m1, m2, ov, ov2
  integer(8) :: kfoo, kfov, kfvo, kfvv, kpvv, kfevv, kfeoo, kft, kooovd, kooood, kovvod, kovood, kk
  integer(8) :: kt2p, kt2pp, ktau, ktt, kgpd, kr1c, kr1v, kra, kr2, klad, kwkl, kweff, ka2, kb2, kx2
  integer(8) :: kao, kgv, kd, kgmq, kwv, kg2, kyb, kslab, kgq, kgh, kpib, kbv, h, lo, nqh, q0, nq
  logical :: dense
  real(8) :: tst
  ZT :: one, zero
  one = 1
  zero = 0
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
  kft = xzget(NW * ov, 'F_ov transposed')
  kooovd = xzget(NW * no * no * ov, 'dressed <oo||ov>')
  kooood = xzget(NW * no ** 4, 'dressed <oo||oo>')
  kovvod = xzget(NW * ov2, 'dressed <ov||vo>')
  kovood = xzget(NW * no * no * ov, 'dressed <ov||oo>')
  kk = xzget(NW * ov2, 'work of the dressing')
  kt2p = xzget(NW * no * no * nvp, 'T2 [i][j][Q]')
  kt2pp = xzget(NW * npo * nvp, 'T2 [P][Q]')
  ktau = xzget(NW * npo * nvp, 'tau1')
  ktt = xzget(NW * npo * nvp, 'T2 + tau1')
  kgpd = xzget(NW * ny * npo * nv, 'pass-1 operand')
  kr1c = xzget(NW * ov, 'T3 part of R1')
  kr1v = xzget(NW * no * ny, 'vovv part of R1')
  kra = xzget(NW * npo * nvp, 'T3 part of R2')
  kr2 = xzget(NW * npo * nvp, 'R2')
  klad = xzget(NW * npo * nvp, 'CCSD ladder')
  kwkl = xzget(NW * npo * npo, 'W_oooo')
  kweff = xzget(NW * ov2, 'W_ovvo')
  ka2 = xzget(NW * ov2, 'work A')
  kb2 = xzget(NW * ov2, 'work B')
  kx2 = xzget(NW * ov2, 'work X')
  kao = xzget(NW * no * no * ov, 'CC3: Ao, the dressed <mc||ij>')
  kgv = xzget(NW * no * nvp * nv, 'CC3: Gv, the dressed <ym||Q>')

  ! ---- T1 dressing
  md = xzmark()
  kd = xzget(NW * ov * nvp, 'dressed <ov||vv>')
  if (.not. xzdry) then
    call xztic()
    call N(xzfock)(no, nv, nvp, w(jeps), w(kt1), w(jooov), w(joovv), w(jovvo), w(jovvq), &
                   w(jvpa), w(jvpb), w(kfoo), w(kfov), w(kfvo), w(kfvv), w(kpvv))
    call N(xzdroo)(no, nv, w(kt1), w(jooov), w(joovv), w(joooo), w(kooovd), w(kooood))
    call N(xzdrovvq)(no, nv, nvp, w(kt1), w(jovvq), w(joovvp), w(kd))
    call N(xzdrov)(no, nv, nvp, w(kt1), w(jooov), w(jovvo), w(jovvq), w(kd), w(kooood), &
                   w(jvpidx), w(kovvod), w(kovood), w(kk))
    call N(xzt2forms)(no, nv, npo, nvp, w(kt1), w(kt2), w(jopa), w(jopb), w(jvpa), w(jvpb), &
                      w(kt2p), w(kt2pp), w(ktau), w(ktt))
    call xztoc(1, 'dressing')
  end if
  ! the operand G of pass 1: the vovv term of R1 and Gv of the batches
  m1 = xzmark()
  kgmq = xzget(NW * no * nvp * ny, 'pass-1 operand G')
  if (.not. xzdry) then
    call xztic()
    call N(xzp1ops)(no, nv, npo, nvp, w(joovv), w(kooovd), w(joovvp), w(kd), w(kfov), &
                    w(jopa), w(jopb), w(kgpd), w(kgmq), w(kft))
    call N(zc_gemm)(0_4, 0_4, no, ny, no * nvp, one, w(kt2p), no * nvp, w(kgmq), ny, zero, w(kr1v), ny)
    call N(xzgvq)(no, nv, nvp, w(kgmq), w(kgv))
    call xztoc(1, 'pass-1 operands, vovv term, Gv')
  end if
  call xzrel(m1)

  ! ---- the dressed <ab||ei> (without its <ab||cd> part: in the sweep), the
  !      dressed <ab||ij>, Ao, W_ovvo, F_vv, F_oo
  kwv = xzget(NW * nvp * ov, 'W_vvvo')
  m2 = xzmark()
  kg2 = xzget(NW * npo * ov, 'work [P][e][i]')
  kyb = xzget(NW * nba * nv * nv * no, 'batch of W_vvvo terms Y')
  if (.not. xzdry) then
    call xztic()
    call N(xzbv)(no, nv, npo, nvp, nba, w(kt1), w(ktau), w(jovvq), w(jooov), w(jovvo), &
                 w(jopa), w(jopb), w(jvpa), w(jvpb), w(kwv), w(kg2), w(kyb))
    call xztoc(1, 'W_vvvo: bare, tau, t1')
    call N(xzvvoo)(no, nv, npo, nvp, w(kt1), w(ktau), w(jgpp), w(jooov), w(joooo), w(kwv), &
                   w(jopa), w(jopb), w(jvpa), w(jvpb), w(kr2), w(kwkl))
    call xztoc(1, 'dressed <ab||ij>')
    call N(xzao3)(no, nv, w(kovood), w(kao))
    call N(xzweff)(no, nv, nvp, w(kt2), w(kt2p), w(joovv), w(joovvp), w(kovvod), w(kfoo), w(kfvv), &
                   w(kweff), w(kfevv), w(kfeoo), w(ka2), w(kb2), w(kx2))
    call xztoc(1, 'Ao, W_ovvo, F_vv, F_oo')
  end if
  ! W_vvvo moves down to the place of the dressed <ov||vv>
  call xzrel(md)
  kd = xzget(NW * nvp * ov, 'W_vvvo')
  if (.not. xzdry) then
    call xztic()
    call N(xzmove)(nvp * ov, w(kwv), w(kd))
    call xztoc(1, 'moves')
  end if
  kwv = kd

  ! ---- the sweep over the rows of the dressed <ab||cd>: its part of the
  !      dressed <ab||ei>, the CCSD ladder
  m2 = xzmark()
  kslab = xzget(NW * nqs * nqmax, 'slab of the dressed <ab||cd>')
  kgq = jovvq
  kgh = jgpp
  kpib = jvpidx
  if (.not. dense) then
    kgq = xzget(NW * ov * nqmax, '<ov||vv> on the pairs of a group')
    kgh = xzget(NW * npo * nqmax, '<oo||vv> on the pairs of a group')
    kpib = xzget(nv * nv, 'pairs of a group')
  end if
  if (.not. xzdry) then
    call zc_vzero_d(NW * npo * nvp, w(klad))
    do h = 0, npg - 1
      lo = vpoff(h)
      nqh = vpoff(h + 1) - lo
      if (nqh == 0) cycle
      call xztic()
      if (.not. dense) call N(xzgrpcols)(no, nv, npo, nvp, lo, nqh, w(jovvq), w(jgpp), w(jvpidx), w(kgq), &
                                         w(kgh), w(kpib))
      do q0 = 0, nqh - 1, nqs
        nq = min(nqs, nqh - q0)
        call N(xzdrrows)(no, nv, npo, nvp, nqh, nq, w(kt1), w(ktau + NW * (lo + q0)), &
                         w(jvvvv + NW * (vvoff(h) + q0 * nqh)), w(kgq), w(kgh), w(jvpa + lo + q0), &
                         w(jvpb + lo + q0), w(kslab), w(ka2))
        call N(zc_pair_t1_contract)(w(kwv + NW * (lo + q0) * ov), w(kslab), w(kt1), nq, nv, nqh, w(kpib), no)
        if (npo > 0 .and. myrank == 0) call N(zc_gemm)(0_4, 1_4, npo, nq, nqh, one, w(ktt + NW * lo), nvp, &
                                                       w(kslab), nqh, zero, w(klad + NW * (lo + q0)), nvp)
      end do
      call xztoc(1, 'dressed <ab||cd> rows, CCSD ladder')
    end do
  end if
  call xzrel(m2)

  ! ---- the batches of occupied triples: the T3 terms of R1 and R2
  kbv = xzget(NW * nv * no * nvp, 'CC3: Bv, the dressed <Q||ei>')
  if (.not. xzdry) then
    call xztic()
    call N(xzbvq)(no, nv, nvp, w(kwv), w(kbv))
    call zc_vzero_d(NW * ov, w(kr1c))
    call zc_vzero_d(NW * npo * nvp, w(kra))
    call xztoc(1, 'CC3: operands')
  end if
  call N(xzfly)(w, 1_8, kt2, kao, kbv, kt2p, kt2, kao, kbv, kt2p, kgv, kooovd, joovvp, kfov, 0d0, &
                kr1c, .true., kra)

  ! ---- the CCSD part, the Jacobi step of T1 and T2 (rank 0)
  if (.not. xzdry .and. myrank == 0) then
    call xztic()
    call N(xzccsd)(no, nv, npo, nvp, w(jeps), w(kt1), w(kt2), w(kt2p), w(kt2pp), w(jgpp), w(kooovd), &
                   w(kooood), w(kovvod), w(kweff), w(kfov), w(kfvo), w(kfevv), w(kfeoo), w(kr1v), &
                   w(kr1c), w(kra), w(klad), w(jopa), w(jopb), w(jvpa), w(jvpb), w(kr2), w(kwkl), &
                   w(kt1n), w(kt2n), w(ka2), w(kb2), w(kx2))
    call xztoc(1, 'CCSD part')
  end if
  if (.not. xzdry) dsec(7, 1) = dsec(7, 1) + omp_get_wtime() - tst
  call xzrel(m0)
end subroutine

! ---------------------------------------------------------------------------
! All batches of occupied triples of CC3 (csrc/cc3_fly.c): the triples of
! the drive terms - nset 1: (At1, Ao1, Bv1, Bo1); 2: their sum with (At2,
! Ao2, Bv2, Bo2) - over the denominator e_ijk - e_abc + omega, contracted
! with the operands (Gv, Gsq, gsqp, fov) into the four sums of pass 1, which
! are then folded into the terms of the equations:
!   kr1   R1 += the singles term ([i][a]; r1t: [a][i]); 0: not wanted
!   kr2   R2[P][Q] += the doubles terms (P = (i<j), Q = (a<b))
! gsqp is <mn||Q> as the native program has it (joovvp, [m][n][Q]): the
! negative of the kernel's gsq[hi][lo][Q] = <lo hi||Q>, so the singles term
! is taken with the other sign.  The batches: per pair (j, k) and class s of
! i the i < j of class s, as many per call as nc3w holds (the plan).  With
! more than one rank every rank does the pairs (j, k) that the cost model
! gives it (the most expensive first, each to the rank with the least so
! far) and the four sums are summed over the ranks.
subroutine N(xzfly)(w, nset, kat1, kao1, kbv1, kbo1, kat2, kao2, kbv2, kbo2, kgv, kgsq, kgsqp, kfov, omega, &
                    kr1, r1t, kr2)
  use xzdata
  use xzmem
  use xzkern
  use xzdist
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: nset, kat1, kao1, kbv1, kbo1, kat2, kao2, kbv2, kbo2, kgv, kgsq, kgsqp, kfov
  integer(8), intent(in) :: kr1, kr2
  real(8), intent(in) :: omega
  logical, intent(in) :: r1t
  integer(8) :: m0, kys, kxo, kras, kr1k, kwk, j, k, s, g, lo, hi, i0, step, npair, p, q, r, best
  integer(8) :: per(0:maxgrp)
  integer(8), allocatable :: own(:), ord(:)
  real(8), allocatable :: cost(:), load(:)
  real(8) :: stats(4), t0
  m0 = xzmark()
  kys = xzget(NW * no * no * nv * nv, 'CC3: the sum Ysq')
  kxo = xzget(NW * no * no * nvp, 'CC3: the sum Xo')
  kras = xzget(NW * no * no * nvp, 'CC3: the sum rAsq')
  kr1k = xzget(NW * no * nv, 'CC3: the sum r1')
  kwk = xzget(NW * nc3w, 'CC3: the work of a batch')
  if (xzdry) then
    call xzrel(m0)
    return
  end if
  call xztic()
  call zc_vzero_d(NW * no * no * nv * nv, w(kys))
  call zc_vzero_d(NW * no * no * nvp, w(kxo))
  call zc_vzero_d(NW * no * no * nvp, w(kras))
  call zc_vzero_d(NW * no * nv, w(kr1k))
  do g = 0, ntg - 1
    per(g) = zc_cc3_work(ncls, g, 1_8, vcoff, vpoff, vtoff)
  end do
  ! the pairs (j < k), p = k (k - 1) / 2 + j, and their ranks
  npair = no * (no - 1) / 2
  allocate(own(0:max(npair, 1_8) - 1))
  own = 0
  if (nrank > 1 .and. npair > 0) then
    allocate(cost(0:npair - 1), ord(npair), load(0:nrank - 1))
    do k = 0, no - 1
      do j = 0, k - 1
        p = k * (k - 1) / 2 + j
        cost(p) = 0d0
        do s = 0, ncls - 1
          hi = min(ocoff(s + 1), j)
          if (hi <= ocoff(s)) cycle
          g = s + ocls(j) + ocls(k)
          cost(p) = cost(p) + dble(hi - ocoff(s)) * dble(per(g))
        end do
      end do
    end do
    ! the most expensive first (a stable insertion sort: every rank the same list)
    do p = 0, npair - 1
      q = p + 1
      do while (q > 1)
        if (cost(ord(q - 1)) >= cost(p)) exit
        ord(q) = ord(q - 1)
        q = q - 1
      end do
      ord(q) = p
    end do
    load = 0d0
    do q = 1, npair
      best = 0
      do r = 1, nrank - 1
        if (load(r) < load(best)) best = r
      end do
      own(ord(q)) = best
      load(best) = load(best) + cost(ord(q))
    end do
    deallocate(cost, ord, load)
  end if
  call xztoc(tcur, 'CC3: the sums zeroed, the pairs dealt')
  stats = 0d0
  t0 = omp_get_wtime()
  do k = 0, no - 1
    do j = 0, k - 1
      if (own(k * (k - 1) / 2 + j) /= myrank) cycle
      do s = 0, ncls - 1
        lo = ocoff(s)
        hi = min(ocoff(s + 1), j)
        if (hi <= lo) cycle
        g = s + ocls(j) + ocls(k)
        if (per(g) == 0) cycle
        step = max(1_8, nc3w / per(g))
        do i0 = lo, hi - 1, step
          call N(zc_cc3_batch)(no, nv, nvp, nvt, ncls, ocoff, vcoff, vpoff, vtoff, w(jvtri), w(jvsgn), &
                               w(jvfree), w(jvpair), w(jeps), w(jev3), omega, i0, min(i0 + step, hi), j, k, nset, &
                               w(kat1), w(kao1), w(kbv1), w(kbo1), w(kat2), w(kao2), w(kbv2), w(kbo2), &
                               w(kgv), w(kgsq), w(kgsqp), w(kfov), w(kys), w(kxo), w(kr1k), w(kras), w(kwk), stats)
        end do
      end do
    end do
  end do
  deallocate(own)
  tfly = tfly + stats
  call xztoc(tcur, 'CC3: the batches')
  if (nrank > 1) then
    call xzmpi_barrier()
    call xztoc(tcur, 'MPI: waiting for the ranks')
    call xzmpi_allsum_w(w(kys), NW * no * no * nv * nv)
    call xzmpi_allsum_w(w(kxo), NW * no * no * nvp)
    call xzmpi_allsum_w(w(kras), NW * no * no * nvp)
    call xzmpi_allsum_w(w(kr1k), NW * no * nv)
    call xztoc(tcur, 'MPI: sums of the CC3 batches')
  end if
  call N(xzfold3)(no, nv, npo, nvp, w(jopa), w(jopb), w(jvpa), w(jvpb), w(kys), w(kxo), w(kr1k), w(kras), &
                  kr1 > 0, r1t, w(max(kr1, 1_8)), w(kr2))
  call xztoc(tcur, 'CC3: the fold')
  call xzrel(m0)
end subroutine

! the four sums folded into the terms of the equations (cc3fly.Fly.run):
!   R1[i][a] -= r1[i][a]   (the sign of gsq, xzfly)
!   R2[P][Q] += rAsq[hi][lo][Q] + Ysq[hi][lo][a][b] - Ysq[hi][lo][b][a] + Xo[hi][lo][Q] - Xo[lo][hi][Q]
subroutine N(xzfold3)(no, nv, npo, nvp, opa, opb, vpa, vpb, ysq, xo, r1k, ras, want1, r1t, r1, r2)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(in) :: ysq(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)        ! [hi][lo][d][y]
  ZT, intent(in) :: xo(0:nvp-1, 0:no-1, 0:no-1)                ! [x][m][Q]
  ZT, intent(in) :: r1k(0:nv-1, 0:no-1)                        ! [m][d]
  ZT, intent(in) :: ras(0:nvp-1, 0:no-1, 0:no-1)               ! [hi][lo][Q]
  logical, intent(in) :: want1, r1t
  ZT :: r1(*)
  ZT, intent(inout) :: r2(0:nvp-1, 0:npo-1)
  integer(8) :: p, q, l, h, a, b, i
  !$omp parallel do private(p, q, l, h, a, b)
  do p = 0, npo - 1
    l = opa(p)
    h = opb(p)
    do q = 0, nvp - 1
      a = vpa(q)
      b = vpb(q)
      r2(q, p) = r2(q, p) + ras(q, l, h) + ysq(b, a, l, h) - ysq(a, b, l, h) + xo(q, l, h) - xo(q, h, l)
    end do
  end do
  if (.not. want1) return
  do i = 0, no - 1
    do a = 0, nv - 1
      if (r1t) then
        r1(1 + i + no * a) = r1(1 + i + no * a) - r1k(a, i)
      else
        r1(1 + a + nv * i) = r1(1 + a + nv * i) - r1k(a, i)
      end if
    end do
  end do
end subroutine

! Ao[hi][lo][c][m] = ovood[m][c][hi][lo], for all (hi, lo): the operand of
! the drive term that pass 2 has as wo[m][c][(lo<hi)] = - ovood[m][c][lo][hi]
! (xzwo, xzwo3; the dressed <mc||lo hi> in the convention of the kernel)
subroutine N(xzao3)(no, nv, ovood, ao)
  implicit none
  integer(8), intent(in) :: no, nv
  ZT, intent(in) :: ovood(0:no-1, 0:no-1, 0:nv-1, 0:no-1)      ! [m][c][x][y]
  ZT, intent(out) :: ao(0:no-1, 0:nv-1, 0:no-1, 0:no-1)        ! [hi][lo][c][m]
  integer(8) :: h, l, c, m
  !$omp parallel do collapse(2) private(h, l, c, m)
  do h = 0, no - 1
    do l = 0, no - 1
      do c = 0, nv - 1
        do m = 0, no - 1
          ao(m, c, l, h) = ovood(l, h, c, m)
        end do
      end do
    end do
  end do
end subroutine

! Bv[e][i][Q] = wv[Q][e][i]
subroutine N(xzbvq)(no, nv, nvp, wv, bv)
  implicit none
  integer(8), intent(in) :: no, nv, nvp
  ZT, intent(in) :: wv(0:no-1, 0:nv-1, 0:nvp-1)                ! [Q][e][i]
  ZT, intent(out) :: bv(0:nvp-1, 0:no-1, 0:nv-1)               ! [e][i][Q]
  integer(8) :: e, i, q, q0
  integer(8), parameter :: nb = 64
  !$omp parallel do private(e, i, q, q0)
  do e = 0, nv - 1
    do q0 = 0, nvp - 1, nb
      do i = 0, no - 1
        do q = q0, min(q0 + nb, nvp) - 1
          bv(q, i, e) = wv(i, e, q)
        end do
      end do
    end do
  end do
end subroutine

! Gv[m][Q][y] = gmq[m][Q][no + y] (the operand G of pass 1: ~<ym||Q>)
subroutine N(xzgvq)(no, nv, nvp, gmq, gv)
  implicit none
  integer(8), intent(in) :: no, nv, nvp
  ZT, intent(in) :: gmq(0:no+nv-1, 0:nvp-1, 0:no-1)
  ZT, intent(out) :: gv(0:nv-1, 0:nvp-1, 0:no-1)
  integer(8) :: m, q
  !$omp parallel do collapse(2) private(m, q)
  do m = 0, no - 1
    do q = 0, nvp - 1
      gv(:, q, m) = gmq(no:no + nv - 1, q, m)
    end do
  end do
end subroutine

! Gv'[m][Q][y] = sum_n r1[n][y] <mn||Q>: the operand Gv of H' = [H~, R1]
subroutine N(xzgv1)(no, nv, nvp, r1, oovvp, gv)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, nvp
  ZT, intent(in) :: r1(0:nv-1, 0:no-1)                         ! [n][y]
  ZT, intent(in) :: oovvp(0:nvp-1, 0:no-1, 0:no-1)             ! [m][n][Q]
  ZT, intent(out) :: gv(0:nv-1, 0:nvp-1, 0:no-1)               ! [m][Q][y]
  integer(8) :: m
  ZT :: one, zero
  one = 1
  zero = 0
  do m = 0, no - 1
    call N(zc_gemm)(1_4, 0_4, nvp, nv, no, one, oovvp(0, 0, m), nvp, r1, nv, zero, gv(0, 0, m), nv)
  end do
end subroutine

! ---------------------------------------------------------------------------
! The folded part of the EOM-CC3 sigma vector of (r1, r2) at the frequency
! eomega (docs/ccn.md, sections 4 and 9; eom_opt.sigma with omega):
!   r3 = (drive[t2; <ab||ei>', <ma||ij>'] + drive[r2; <ab||ei>~, <ma||ij>~])
!        / (omega + e_ijk - e_abc), contracted with the integrals of H~
!        -> sigma_1, sigma_2
!   t3 of the ground state, made again, contracted with the integrals of
!        H' = [H~, R1] (Gv', <mn||ie>', f'[m][e]; gsq = 0) -> sigma_2
! kdwv: <ab||ei>' [Q][e][i]; kdovoo: <ma||xy>' (xzehs); kdooov: <mn||ie>';
! kdfov: f'[m][e]; ks1 [i][a], ks2 [P][Q] accumulated.
subroutine N(xzefold)(w, kr1, kr2, kr2p, kdwv, kdovoo, kdooov, kdfov, ks1, ks2)
  use xzdata
  use xzmem
  use xzkern
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kr1, kr2, kr2p, kdwv, kdovoo, kdooov, kdfov, ks1, ks2
  integer(8) :: m, kao1, kbv1, kgv1
  m = xzmark()
  kao1 = xzget(NW * no * no * no * nv, 'CC3: Ao derivative')
  kbv1 = xzget(NW * nv * no * nvp, 'CC3: Bv derivative')
  kgv1 = xzget(NW * no * nvp * nv, 'CC3: Gv derivative')
  if (.not. xzdry) then
    call xztic()
    call N(xzao3)(no, nv, w(kdovoo), w(kao1))
    call N(xzbvq)(no, nv, nvp, w(kdwv), w(kbv1))
    call N(xzgv1)(no, nv, nvp, w(kr1), w(joovvp), w(kgv1))
    call xztoc(5, 'CC3: operands of a sigma vector')
  end if
  call N(xzfly)(w, 2_8, jt2, kao1, kbv1, et2p, kr2, e3ao, e3bv, kr2p, e3gv, eooovd, joovvp, efov, eomega, &
                ks1, .false., ks2)
  call N(xzfly)(w, 1_8, jt2, e3ao, e3bv, et2p, jt2, e3ao, e3bv, et2p, kgv1, kdooov, joovvp, kdfov, 0d0, &
                0_8, .false., ks2)
  call xzrel(m)
end subroutine

! ---------------------------------------------------------------------------
! EOM-CC3 without stored triples: the folded eigenproblem
!   [A_PP + A_PQ (omega - A_QQ)^-1 A_QP] R_P = omega R_P
! in the space of singles and doubles (docs/ccn.md, section 4), each root to
! self-consistency in omega (the algorithm of ccn.folded_roots):
!   1. the roots of the singles-doubles block A_PP from unit vectors on the
!      lowest single excitations (no triples work), to 1e-4;
!   2. root by root, from the lowest: the solver of xzeoms.inc on the
!      matrix at a fixed omega, started from the vectors of the step before,
!      following the roots with the largest weight on them (nfollow), to a
!      threshold that follows |lambda(omega) - omega| down to ESTATE_CONV;
!      the roots of the singles-doubles stage that lie closer than 5e-4 Eh
!      (XZ_FOLDWIN) to a neighbour form a cluster: solved together at every
!      frequency, the root taken by its place in the cluster (a nearly
!      degenerate level is not split reliably by following one vector);
!      omega replaced by lambda(omega), or by the secant step for
!      lambda(omega) = omega from the last two steps; until |lambda - omega|
!      is below ESTATE_CONV / 100 (XZ_OMEGATOL) with the residual below
!      ESTATE_CONV.
! The totally symmetric states (label shift 0) in blocks of the labels;
! without labels the states of all shifts (the sum of the roots asked for).
! States of other shifts: cc3_stored_eom.
subroutine N(xzeom3)(w, ecorr)
  use xzdata
  use xzmem
  use xzkern
  use xzeomd
  use omp_lib
  implicit none
  real(8) :: w(*)
  real(8), intent(in) :: ecorr
  integer(8) :: m0, nsd, kxa, kxb, kst, kx, nroot, nkeep, nsing, msd, msf, msub, peak, avail, k, ish
  integer(8) :: nfound, nsgd, nitd, nsig, nit, nf1, outer, nsigr, i, a, r, kk, ca, cb, nc, jc
  integer(8), allocatable :: singles(:), lead(:, :), cyc(:), c1(:), l1(:, :)
  complex(8), allocatable :: ee(:), e1(:)
  real(8), allocatable :: rmax(:), rnorm(:), wts(:, :), x1mx(:), x1nm(:), w1(:, :)
  ZT, allocatable :: vlead(:, :), v1(:, :)
  real(8) :: omega, lam, gap, tol, otol, prevo, prevl, slope, nxt, t0, tr, tsg0, cwin
  logical :: haveprev, ok, dense
  character(len=32) :: arg
  integer :: larg, ios
  real(8), external :: N(xzeimag)
  real(8), parameter :: ev = 27.211386245988d0, cm = 219474.6313632d0
  integer(8), parameter :: maxouter = 30

  nsd = no * nv + npo * nvp
  dense = ncls == 1
  m0 = xzmark()
  call N(xzeomkeep)()
  eshift = 0
  n3s = 0
  nroot = 0
  do ish = 1, enshift
    if (dense .or. eshl(ish) == 0) then
      nroot = nroot + enr(ish)
    else if (.not. xzdry) then
      write(6, '(a,i6,a)') ' @XZCCSDT-W, EOM-CC3 without stored triples: the shift ', eshl(ish), &
        ' is left out (the folded problem has the totally symmetric states; cc3_stored_eom has every shift).'
    end if
  end do
  nkeep = max(nroot, 1_8) + 2
  msd = min(64_8, 8 + 4 * nkeep)
  msf = max(12_8, (8 + 4 * nroot) / 2)
  if (xzdry) then
    call N(xzeomim)(w)
    kst = xzget(NW * nkeep * nsd, 'EOM: vectors to start from')
    kx = xzget(NW * nkeep * nsd, 'EOM: the vectors of a cluster of roots')
    kxa = xzget(NW * nsd, 'EOM: a vector')
    kxb = xzget(NW * nsd, 'EOM: its sigma vector')
    k = xzget(2 * min(64_8, max(msd, msf, 3 * nkeep + 4)) * (NW * nsd + xzalign), 'EOM: the subspace')
    efold = .true.
    call N(xzesig)(w, kxa, kxb, .true.)
    efold = .false.
    call xzrel(m0)
    return
  end if
  if (nroot == 0) then
    write(6, '(/,a)') ' @XZCCSDT-W, EOM-CC3 without stored triples: no totally symmetric state asked for (%xzeom).'
    call xzrel(m0)
    return
  end if
  t0 = omp_get_wtime()
  call N(xzeomim)(w)
  teomim = omp_get_wtime() - t0
  otol = ethr / 100d0
  call get_environment_variable('XZ_OMEGATOL', arg, larg)
  if (larg > 0) then
    read(arg(1:larg), *, iostat=ios) otol
    if (ios /= 0) otol = ethr / 100d0
  end if
  ! roots of the singles-doubles stage closer than cwin: one cluster (XZ_FOLDWIN, Eh)
  cwin = 5d-4
  call get_environment_variable('XZ_FOLDWIN', arg, larg)
  if (larg > 0) then
    read(arg(1:larg), *, iostat=ios) cwin
    if (ios /= 0) cwin = 5d-4
  end if
  write(6, '(/,a)') '  EOM-EE-CC3 without stored triples: the folded eigenproblem in the space of singles and doubles'
  write(6, '(a)') '  ' // repeat('-', 98)
  write(6, '(a,f10.2,a)') '   intermediates of the ground state (with the operands of the CC3 batches): ', teomim, ' s'
  write(6, '(a,es9.2,a,es9.2,a,i4,a)') '   convergence: the largest element of a residual below ', ethr, &
    ', |lambda(omega) - omega| below ', otol, ';  at most ', emaxcyc, ' iterations per frequency'
  if (dense) then
    write(6, '(a,i4,a)') '   no blocks of labels: the states of all shifts; ', nroot, ' roots'
  else
    write(6, '(a,i4,a)') '   the totally symmetric states (label shift 0): ', nroot, ' roots'
  end if
  write(6, '(a,i14,a)') '   a vector: ', nsd, ' elements (singles and doubles; the triples are folded in)'
  write(6, '(a)') '   (the vectors of the roots are not written: no file XZEOMR)'
  allocate(singles(nkeep), ee(nkeep), rmax(nkeep), rnorm(nkeep), wts(3, nkeep), cyc(nkeep), lead(nlead, nkeep), &
           vlead(nlead, nkeep), e1(nkeep), x1mx(nkeep), x1nm(nkeep), w1(3, nkeep), c1(nkeep), l1(nlead, nkeep), &
           v1(nlead, nkeep))
  call xzesingles(0_8, w(jeps), nkeep, singles, nsing)
  if (nsing == 0) then
    write(6, '(a)') ' @XZCCSDT-W, no single excitation of shift 0: no guess.'
    call xzrel(m0)
    return
  end if
  nkeep = min(nkeep, nsing)
  nroot = min(nroot, nsing)
  kst = xzget(NW * nkeep * nsd, 'EOM: vectors to start from')
  kx = xzget(NW * nkeep * nsd, 'EOM: the vectors of a cluster of roots')
  kxa = xzget(NW * nsd, 'EOM: a vector')
  kxb = xzget(NW * nsd, 'EOM: its sigma vector')

  ! ---- 1. the singles-doubles block
  tr = omp_get_wtime()
  call xzdry_begin()
  call N(xzesig)(w, kxa, kxb, .true.)
  call xzdry_end(peak)
  avail = xzsize - xztop - (peak - xztop)
  msub = msd
  call N(xzesolve)(w, .true., nkeep, max(1d-4, ethr), 60_8, msub, nkeep, 0_8, singles, kxa, kxb, avail, &
                   ee, rmax, rnorm, wts, cyc, nfound, nsgd, nitd, kst, 0, lead, vlead, .true.)
  write(6, '(a,i4,a,i5,a,i4,a,f9.2,a)') '   singles-doubles stage: ', nfound, ' roots, ', nsgd, ' sigma vectors, ', nitd, &
    ' iterations, ', omp_get_wtime() - tr, ' s'
  write(6, '(a,8f14.8)') '     roots / Eh: ', (dble(ee(k)), k = 1, min(nfound, 8_8))

  ! ---- 2. root by root, to self-consistency in omega
  efold = .true.
  call xzdry_begin()
  call N(xzesig)(w, kxa, kxb, .true.)
  call xzdry_end(peak)
  avail = xzsize - xztop - (peak - xztop)
  write(6, '(a,f10.1,a,f10.1,a)') '   a folded sigma vector takes ', 8d-6 * (peak - xztop), ' MB of the work array; free besides it: ', &
    8d-6 * avail, ' MB'
  write(6, '(/,a)') '   root  step         omega / Eh       lambda(omega) / Eh  |lambda-omega|   max |r|   sigma vectors      time/s'
  nres = 0
  ca = 1
  cb = 0
  nc = 1
  do k = 1, min(nroot, nfound)
    tr = omp_get_wtime()
    tsg0 = teomsig
    ! the cluster of the root: the roots of the singles-doubles stage closer
    ! than cwin to their neighbour, solved together at every frequency and
    ! told apart by their order (a nearly degenerate level is not split by
    ! following one vector); its vectors start the next frequency and root
    if (k > cb) then
      ca = k
      cb = k
      do while (cb < nfound)
        if (dble(ee(cb + 1)) - dble(ee(cb)) >= cwin) exit
        cb = cb + 1
      end do
      nc = cb - ca + 1
      call N(xzecopy)(nc * nsd, w(kst + NW * nsd * (ca - 1)), w(kx))
      if (nc > 1) write(6, '(a,i4,a,i4,a,es9.2,a)') '   (roots', ca, ' to', cb, ' of the singles-doubles stage closer than ', &
        cwin, ' Eh: solved together, by their order)'
    end if
    jc = k - ca + 1
    omega = dble(ee(k))
    gap = 1d0
    haveprev = .false.
    ok = .false.
    nsigr = 0
    prevo = 0d0
    prevl = 0d0
    do outer = 1, maxouter
      tol = min(1d-4, max(ethr, 0.1d0 * gap))
      eomega = omega
      nfollow = nc
      msub = min(64_8, max(msf, 3 * nc + 4))
      t0 = omp_get_wtime()
      call N(xzesolve)(w, .true., nc, tol, emaxcyc, msub, nc, kx, singles, kxa, kxb, avail, &
                       e1, x1mx, x1nm, w1, c1, nf1, nsig, nit, kx, 0, l1, v1, .true.)
      nfollow = 0
      nsigr = nsigr + nsig
      if (nf1 < nc) then
        write(6, '(a,i4,a)') ' @XZCCSDT-W, EOM-CC3: the solver at a frequency found fewer roots than its cluster has (', &
          nf1, '): the root is left.'
        exit
      end if
      lam = dble(e1(jc))
      gap = abs(lam - omega)
      write(6, '(i7,i6,2f22.12,2es12.2,i8,f14.2)') k, outer, omega, lam, gap, x1mx(jc), nsig, omp_get_wtime() - t0
      flush(6)
      if (gap < otol .and. tol <= ethr .and. maxval(x1mx(1:nc)) <= tol) then
        omega = lam
        ok = .true.
        exit
      end if
      ! the next frequency: lambda(omega), or the secant step for lambda(omega) = omega
      nxt = lam
      if (haveprev .and. abs(omega - prevo) > 1d-14) then
        slope = (lam - prevl) / (omega - prevo)
        if (abs(slope) < 0.5d0) nxt = omega + (lam - omega) / (1d0 - slope)
      end if
      prevo = omega
      prevl = lam
      haveprev = .true.
      omega = nxt
    end do
    write(6, '(a,i4,a,f20.12,a,f12.6,a,f14.4,a,i4,a,i6,a,f10.2,a,f8.2,a)') '   root', k, ': omega ', omega, ' Eh = ', &
      omega * ev, ' eV = ', omega * cm, ' cm-1   (', outer, ' frequencies, ', nsigr, ' sigma vectors, ', &
      omp_get_wtime() - tr, ' s; ', (teomsig - tsg0) / max(nsigr, 1_8), ' s per sigma vector)'
    if (.not. ok) write(6, '(a)') '          NOT converged'
    if (nres < maxres) then
      nres = nres + 1
      rshift(nres) = 0
      rroot(nres) = k
      rom(nres) = omega
      rim(nres) = aimag(e1(jc))
      rmx(nres) = x1mx(jc)
      rnm(nres) = x1nm(jc)
      rwt(:, nres) = w1(:, jc)
      rconv(nres) = ok
      rcyc(nres) = outer
      rnsig(nres) = nsigr
      rli(nres) = 0
      rla(nres) = 0
      rlv(nres) = 0d0
      if (l1(1, jc) >= 0) then
        rli(nres) = nfrozen + operm(l1(1, jc) / nv) + 1
        rla(nres) = nfrozen + no + vperm(mod(l1(1, jc), nv)) + 1
        rlv(nres) = abs(v1(1, jc))
      end if
    end if
    do r = 1, nlead
      if (l1(r, jc) < 0) cycle
      if (abs(v1(r, jc)) < 0.05d0 .and. r > 1) cycle
      i = l1(r, jc) / nv
      a = mod(l1(r, jc), nv)
      write(6, '(a,i4,a,i5,a,i4,a,f12.6,a,i5,a,i4,a,f12.6,a,2f10.6)') '   root', k, ': ', &
        nfrozen + operm(i) + 1, ' (', lab(nfrozen + operm(i)), ',', w(jeps + i), ')  -> ', &
        nfrozen + no + vperm(a) + 1, ' (', lab(nfrozen + no + vperm(a)), ',', w(jeps + no + a), ')   ', &
        dble(v1(r, jc)), N(xzeimag)(v1(r, jc))
    end do
  end do
  efold = .false.
  write(6, '(/,a)') '   root        omega / Eh            eV            cm-1    max |r|      |r|   steps    R1     R2' // &
                    '          total energy / Eh'
  do kk = 1, nres
    write(6, '(i7,f20.12,f14.8,f16.4,2es10.2,i6,2f7.4,f26.12)') rroot(kk), rom(kk), rom(kk) * ev, rom(kk) * cm, &
      rmx(kk), rnm(kk), rcyc(kk), rwt(1:2, kk), escf + ecorr + rom(kk)
  end do
  deallocate(singles, ee, rmax, rnorm, wts, cyc, lead, vlead, e1, x1mx, x1nm, w1, c1, l1, v1)
  call xzrel(m0)
end subroutine
