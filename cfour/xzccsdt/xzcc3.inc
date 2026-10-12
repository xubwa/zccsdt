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

! Ao[hi][lo][c][m] = <mc||lo hi>~ = ovood[m][c][lo][hi], for all (hi, lo)
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
          ao(m, c, l, h) = ovood(h, l, c, m)
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
