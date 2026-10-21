!
! EOM-DEA-CCSDT with complete 5p3h operators (type-generic; %xzdea level
! 5p3h; xzdea5d.F90 has the layouts; docs/native.md, section 21).  The
! algorithm and the layouts are those of src/zccsdt/cc/dea5_sector.py
! (docs/eom.md, section 11.11), the reference of every number: the sigma
! vector of R = 2p + 3p1h + 4p2h + 5p3h on the CCSDT ground state, the rows
! to 4p2h those of N(xzdsig) (xzdeas.inc) plus the terms of r5, the 5p3h row
! the T5 equation of CCSDTQP with R in the place of one amplitude, in the
! families of the 61 terms (zccsdt.cc.dea_ref.FAMILIES5):
!
!   pp      sum_{f<g} Wq5(ab, fg) r5(ijk; fg cde)            Wq5 = Wpp + F_vv on pairs / 4
!   hh      the operators on the occupied triples of a group
!   ring    S(i|jk) S(a|bcde) sum_{mf} Wr(m,a,f,i) r5(mjk; f bcde)
!   D1      S(i|jk) S(ab|cde)  - sum_f Wv(ab,f,i) r4(jk; f cde)
!   D2      S(i|jk) S(ab|cde)  sum_m t2(im; ab) XoR5(m; jk; cde)
!   D3      S(k|ij) S(de|abc)  sum_g r3(k; g de) Xv2(g; ij; abc)
!   D4      S(ij|k) S(ab|cde)  sum_{mf} t3(ijm; f ab) Z4(mf; k; cde)
!   D5                S(ab|cde)  sum_f t3(ijk; f ab) XvR5(f; cde)
!   D6                S(ab|cde)  sum_m Xo3(m; ijk; ab) r3(m; cde)
!   D7      S(i|jk) S(de|abc)  sum_{m<n} t3(imn; abc) Y4(mn; jk; de)
!   D8      S(i|jk) S(de|abc)  sum_m t3(jkm; abc) XoRp(m; i; de)
!   E1      S(ij|k) S(a|bcde)  sum_m Wo(m,a,ij) r4(mk; bcde)
!   E1x     S(ij|k) S(a|bcde)  sum_f t2(ij; af) Xv5(f; k; bcde)
!   E2                S(a|bcde)  sum_{m<n} W2(mn; ijk; a) r4(mn; bcde)
!
! and in the rows below the four terms of r5 (sigma3: from the view of the
! pp ladder; sigma4: from that view on (a | bcd) and from the ring).  The
! passes over r5 - pp by batches of rows of a block (zc_t5s_x23), the ring
! by occupied spectator pairs (zc_t5s_x14), hh by blocks, the rows one
! occupied triple at a time - and the intermediates are those of Engine5;
! E1x and the term of r5 in Xv5 go by the index k of Xv5 (Xv5 of one k at a
! time, the rows (l, n, k) of r5 expanded on (h | bcde) for its term of r5,
! the rows of sigma5 of the triples with k made from it) instead of in the
! ring and the rows: over the ranks every k has its ranks and Xv5(.; k) is
! the largest array (o v^5 / classes in all) - docs/native.md, 21.
!

! ---------------------------------------------------------------------------
! The places of the arrays of the ground state of the 5p3h row (after
! N(xzdkeep); released with its mark).
subroutine N(xzd5keep)()
  use xzdata
  use xzmem
  use xzdead
  use xzdead5
  implicit none
  dpos4 = xzget(nv4 / 2 + 1, 'DEA5: the places of the quadruples in their classes')
  dpos4p = xzget(nv4 / 2 + 1, 'DEA5: the places of the quadruples among those of their class and parity')
  dix4p = xzget(nqall / 2 + 1, 'DEA5: the quadruples of a class by parity')
  dwq5 = xzget(NW * nvv, 'DEA5: the pp operator of the 5p3h row')
  dh3 = xzget(NW * max(h3size, 1_8), 'DEA5: the operators on the occupied triples')
  dxo3 = xzget(NW * max(nto * no * nvp, 1_8), 'DEA5: Xo3')
  dw2 = xzget(NW * max(nto * nv * npo, 1_8), 'DEA5: W2')
  db4 = xzget(NW * max(npo * no * no * nv * nv, 1_8), 'DEA5: B4')
  dx2 = xzget(NW * max(x2size, 1_8), 'DEA5: Xv2 with the rows of T3 (the pairs of the triples of this rank)')
end subroutine

! The layout of the arrays of the ground state of the 5p3h row: the
! operators on the triples of a group, the pairs whose Xv2 this rank holds
! (those of its occupied triples, dt0 .. dt1 - 1; all with one rank)
subroutine N(xzd5glay)(w)
  use xzdata
  use xzdead
  use xzdead5
  implicit none
  real(8), target :: w(*)
  integer(8), pointer :: opair(:, :)
  integer(8) :: c, i, r, pq, n
  if (allocated(h3off)) deallocate(h3off, x2loc)
  allocate(h3off(0:ntg), x2loc(0:max(npo, 1_8)-1))
  h3off(0) = 0
  do c = 0, ntg - 1
    h3off(c + 1) = h3off(c) + dnJ(c) * dnJ(c)
  end do
  h3size = h3off(ntg)
  call xzdiv2(w, jopair, max(nto, 1_8), 3_8, opair)
  x2loc = -1
  do i = dt0, dt1 - 1
    do r = 0, 2
      x2loc(opair(i, r)) = 0
    end do
  end do
  n = 0
  do pq = 0, npo - 1
    if (x2loc(pq) < 0) cycle
    x2loc(pq) = n
    n = n + r13p(pq + 1) - r13p(pq)
  end do
  x2size = n
end subroutine

! ---------------------------------------------------------------------------
! The intermediates of the ground state of the 5p3h row (after N(xzdim), the
! arrays of N(xzeomim) and N(xzdkeep) in place; dea5_sector.Ground5): the
! places of the quadruples, Wq5, the operators on the occupied triples, Xo3,
! W2, B4 and Xv2 (Xv of N(xzdxv) with its term of Wpp counted twice) of the
! pairs of x2loc.
subroutine N(xzd5im)(w)
  use xzdata
  use xzmem
  use xzkern
  use xzdead
  use xzdead5
  implicit none
  real(8), target :: w(*)
  integer(8) :: m0, kpib, kw2p, kwt, koa, ktmp, ky, h, lo, nqh, kwrb, kmx, koa2
  ZT :: sc, mone
  m0 = xzmark()
  kpib = xzget(nv * nv, 'DEA5: the pairs of a group')
  kw2p = xzget(NW * max(npo * npo, 1_8), 'DEA5: Whh - F_oo / 2 on pairs')
  kwt = xzget(NW * max(npo * no * no, 1_8), 'DEA5: Whh as [P][m][l]')
  koa = xzget(NW * max(no * nv * npo, 1_8), 'DEA5: <mn||hx> as [x][h][(mn)]')
  ktmp = xzget(NW * nv * nv, 'DEA5: a product of Wr and t2')
  ky = xzget(NW * no * nv * nv, 'DEA5: <am||Q> t3')
  kwrb = xzget(NW * wrsize, 'DEA5: Wr in blocks')
  kmx = xzget(NW * 2 * mxmax, 'DEA5: the products of Xv of one pair')
  koa2 = xzget(NW * no * nv * npo, 'DEA5: <mn||el> as [l][e][(mn)]')
  if (xzdry) then
    call xzd5ptabs(w, .false.)
    call xzrel(m0)
    return
  end if
  call N(xzd5pos4)(w)
  call xzd5ptabs(w, .true.)
  ! Wq5 = (the pp operator of the T3 equation, Wpp + F_vv / 2) - F_vv / 4
  do h = 0, npg - 1
    lo = vpoff(h)
    nqh = vpoff(h + 1) - lo
    if (nqh == 0) cycle
    call N(xzepib)(nv, lo, nqh, w(jvpidx), w(kpib))
    call N(xzecopy)(nqh * nqh, w(jvvvv + NW * vvoff(h)), w(dwq5 + NW * vvoff(h)))
    sc = -0.25d0
    call N(zc_pair_one_body_add)(w(dwq5 + NW * vvoff(h)), nqh, nqh, w(jvpa + lo), w(jvpb + lo), w(kpib), nv, &
                                 w(efevv), nv, sc)
  end do
  ! Whh - F_oo / 2 on pairs, [(mn)][(ij)]
  if (npo > 0) then
    call N(xzecopy)(npo * npo, w(ewkl), w(kw2p))
    sc = -0.5d0
    call N(zc_pair_one_body_add)(w(kw2p), npo, npo, w(jopa), w(jopb), w(jopidx), no, w(efeoo), no, sc)
  end if
  call N(xzd5h3)(w, kw2p)
  call N(xzd5xo3)(w, kwt, koa, ktmp, ky)
  call N(xzd5b4)(w)
  ! Xv2 of the pairs of x2loc: the six-index Xv of N(xzdxv), its term of Wpp counted twice
  dxv2 = .true.
  call N(xzdxv)(w, dvo, kwrb, kmx, koa2)
  dxv2 = .false.
  call xzrel(m0)
end subroutine

! pos4[rank of the quadruple] = its place among the quadruples of its class
subroutine N(xzd5pos4)(w)
  use xzdata
  use xzdead
  use xzdead5
  use iso_c_binding
  implicit none
  real(8), target :: w(*)
  integer(4), pointer :: p4(:), q4(:)
  integer(2), pointer :: quads(:, :)
  integer(8) :: m, t
  call c_f_pointer(c_loc(w(dpos4)), q4, [max(nv4, 1_8)])
  p4(0:max(nv4, 1_8)-1) => q4
  call xzdi2v2(w, dquads, 4_8, max(nqall, 1_8), quads)
  do m = 0, n4c - 1
    !$omp parallel do
    do t = 0, nqa(m) - 1
      p4(xzd5qrank(int(quads(0, qa0(m) + t), 8), int(quads(1, qa0(m) + t), 8), int(quads(2, qa0(m) + t), 8), &
                     int(quads(3, qa0(m) + t), 8), nv)) = int(t, 4)
    end do
  end do
end subroutine

! The operator on the occupied triples of every group c (dea5_sector.Ground5.H):
! sigma5[x] += sum_y H[x][y] r5[y], from W = Whh - F_oo / 2 on pairs
!   H[x][y] = sum over the splits (ab | c3) of x of sign * sign(c3 mn -> y) W[(mn)][(ab)]
subroutine N(xzd5h3)(w, kw2p)
  use xzdata
  use xzdead
  use xzdead5
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: kw2p
  integer(8), pointer :: ofree(:, :), opair(:, :), otri(:, :)
  integer(1), pointer :: osgn(:, :)
  ZT, pointer :: h3(:), w2p(:)
  integer(8) :: c, i, r, c3, pab, pmn, j, nj
  real(8) :: sg
  call xzdiv2(w, jofree, max(nto, 1_8), 3_8, ofree)
  call xzdiv2(w, jopair, max(nto, 1_8), 3_8, opair)
  call xzdiv2(w, jotri, max(npo, 1_8), no, otri)
  call xzdi1v2(w, josgn, max(npo, 1_8), no, osgn)
  call N(xzdv)(w, dh3, max(h3size, 1_8), h3)
  call N(xzdv)(w, kw2p, max(npo * npo, 1_8), w2p)
  h3 = 0
  do c = 0, ntg - 1
    nj = dnJ(c)
    if (nj == 0) cycle
    !$omp parallel do private(r, c3, pab, sg, pmn, j)
    do i = otoff(c), otoff(c + 1) - 1
      do r = 0, 2
        c3 = ofree(i, r)
        pab = opair(i, r)
        sg = merge(-1d0, 1d0, r == 1)
        do pmn = 0, npo - 1
          if (osgn(pmn, c3) == 0) cycle
          j = otri(pmn, c3)
          if (j < otoff(c) .or. j >= otoff(c + 1)) cycle
          h3(h3off(c) + (i - otoff(c)) * nj + j - otoff(c)) = h3(h3off(c) + (i - otoff(c)) * nj + j - otoff(c)) + &
            sg * osgn(pmn, c3) * w2p(pmn * npo + pab)
        end do
      end do
    end do
  end do
end subroutine

! Xo3[I][m][Q] and W2[I][a][(mn)] (dea5_sector.Ground5):
!   Xo3(m; ijk; ab) = P(k/ij) sum_l Whh(lm, ij) t2(kl; ab) - P(i/jk) P(ab) sum_h Wr(m,a,h,i) t2(jk; bh)
!     - sum_h f_mh t3(ijk; h ab) + P(i/jk) sum_{lh} <ml||hi> t3(jkl; h ab) + P(ab) sum_{h<g} <am||hg> t3(ijk; b hg)
!   W2(mn; ijk; a) = P(i/jk) sum_h <mn||hi> t2(jk; ah) + sum_{h<g} <mn||hg> t3(ijk; a hg)
! T3 on (h | Q) is read from T12 (its rows of the triple).
subroutine N(xzd5xo3)(w, kwt, koa, ktmp, ky)
  use xzdata
  use xzkern
  use xzdead
  use xzdead5
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: kwt, koa, ktmp, ky
  integer(8), pointer :: ofree(:, :), opair(:, :), otri(:, :), opa(:), opb(:), vpa(:), vpb(:), opidx(:, :)
  integer(1), pointer :: osgn(:, :)
  ZT, pointer :: wt(:), wkl(:), oa(:), ooovd(:), xo(:), tmp(:), y(:)
  integer(8) :: i, r, x, pyz, yy, zz, m, l, p, qq, g, j, kj, nw2, q, a, b, pq, kx
  real(8) :: sg, sj
  ZT :: one, zero, alpha, mone
  one = 1
  zero = 0
  mone = -1
  nw2 = no * nv
  call xzdiv2(w, jofree, max(nto, 1_8), 3_8, ofree)
  call xzdiv2(w, jopair, max(nto, 1_8), 3_8, opair)
  call xzdiv2(w, jotri, max(npo, 1_8), no, otri)
  call xzdi1v2(w, josgn, max(npo, 1_8), no, osgn)
  call xzdiv(w, jopa, max(npo, 1_8), opa)
  call xzdiv(w, jopb, max(npo, 1_8), opb)
  call xzdiv2(w, jopidx, no, no, opidx)
  call xzdiv(w, jvpa, max(nvp, 1_8), vpa)
  call xzdiv(w, jvpb, max(nvp, 1_8), vpb)
  call N(xzdv)(w, kwt, max(npo * no * no, 1_8), wt)
  call N(xzdv)(w, ewkl, max(npo * npo, 1_8), wkl)
  call N(xzdv)(w, koa, max(no * nv * npo, 1_8), oa)
  call N(xzdv)(w, eooovd, no * no * no * nv, ooovd)
  call N(xzdv)(w, ktmp, nv * nv, tmp)
  call N(xzdv)(w, ky, no * nv * nv, y)
  ! Whh(lm, P) as wt[P][m][l] (antisymmetric in l, m; W_oooo[(lm)][P] of the pairs l < m)
  wt = 0
  do pq = 0, npo - 1
    do m = 0, no - 1
      do l = 0, no - 1
        if (l < m) then
          wt((pq * no + m) * no + l) = wkl(opidx(m, l) * npo + pq)
        else if (l > m) then
          wt((pq * no + m) * no + l) = -wkl(opidx(l, m) * npo + pq)
        end if
      end do
    end do
  end do
  ! <mn||hx> = - ~<mn||xh> as [x][h][(mn)]
  !$omp parallel do collapse(2) private(pq)
  do x = 0, no - 1
    do q = 0, nv - 1
      do pq = 0, npo - 1
        oa((x * nv + q) * npo + pq) = -ooovd(((opa(pq) * no + opb(pq)) * no + x) * nv + q)
      end do
    end do
  end do
  call zc_vzero_d(NW * max(nto * no * nvp, 1_8), w(dxo3))
  call zc_vzero_d(NW * max(nto * nv * npo, 1_8), w(dw2))
  do i = 0, nto - 1
    g = xzd5tgrp(i)
    kx = dxo3 + NW * i * no * nvp
    call N(xzdv)(w, kx, no * nvp, xo)
    do r = 0, 2
      x = ofree(i, r)
      pyz = opair(i, r)
      yy = opa(pyz)
      zz = opb(pyz)
      sg = merge(-1d0, 1d0, r == 1)
      alpha = sg
      ! sum_l Whh(lm, yz) t2(xl; Q)
      call N(zc_gemm)(0_4, 0_4, no, nvp, no, alpha, w(kwt + NW * pyz * no * no), no, w(et2p + NW * x * no * nvp), nvp, &
                      one, w(kx), nvp)
      ! - P(ab) sum_h Wr(m,a,h,x) t2(yz; bh)
      do m = 0, no - 1
        call N(zc_gemm)(0_4, 1_4, nv, nv, nv, one, w(dwr2 + NW * (x * nv * nw2 + m * nv)), nw2, &
                        w(jt2 + NW * (yy * no + zz) * nv * nv), nv, zero, w(ktmp), nv)
        !$omp parallel do private(a, b)
        do q = 0, nvp - 1
          a = vpa(q)
          b = vpb(q)
          xo(m * nvp + q) = xo(m * nvp + q) - sg * (tmp(a * nv + b) - tmp(b * nv + a))
        end do
      end do
      ! sum_{lh} <ml||hx> t3(yzl; h Q)
      do l = 0, no - 1
        sj = osgn(pyz, l)
        if (sj == 0d0) cycle
        j = otri(pyz, l)
        kj = xzd5tgrp(j)
        do p = 0, ncls - 1
          if (t12off(kj, p) < 0) cycle
          qq = kj - p
          alpha = -sg * sj
          call N(zc_gemm)(0_4, 0_4, no, dnQ(qq), dnV(p), alpha, w(eooovd + NW * ((l * no + x) * nv + vcoff(p))), no * no * nv, &
                          w(dt12 + NW * (t12off(kj, p) + (j - otoff(kj)) * dnQ(qq))), dnJ(kj) * dnQ(qq), one, &
                          w(kx + NW * vpoff(qq)), nvp)
        end do
      end do
      ! W2: sum_h t2(yz; a h) <mn||hx>
      alpha = sg
      call N(zc_gemm)(0_4, 0_4, nv, npo, nv, alpha, w(jt2 + NW * (yy * no + zz) * nv * nv), nv, w(koa + NW * x * nv * npo), &
                      npo, one, w(dw2 + NW * i * nv * npo), npo)
    end do
    ! - sum_h f_mh t3(I; h Q);  Y[(m a)][b] = - sum_Q ~<ma||Q> t3(I; b Q);  W2 += t3(I; a Q) <mn||Q>
    call zc_vzero_d(NW * no * nv * nv, w(ky))
    do p = 0, ncls - 1
      if (t12off(g, p) < 0) cycle
      qq = g - p
      call N(zc_gemm)(0_4, 0_4, no, dnQ(qq), dnV(p), mone, w(efov + NW * vcoff(p)), nv, &
                      w(dt12 + NW * (t12off(g, p) + (i - otoff(g)) * dnQ(qq))), dnJ(g) * dnQ(qq), one, &
                      w(kx + NW * vpoff(qq)), nvp)
      call N(zc_gemm)(0_4, 1_4, no * nv, dnV(p), dnQ(qq), mone, w(ed + NW * vpoff(qq)), nvp, &
                      w(dt12 + NW * (t12off(g, p) + (i - otoff(g)) * dnQ(qq))), dnJ(g) * dnQ(qq), one, &
                      w(ky + NW * vcoff(p)), nv)
      call N(zc_gemm)(0_4, 1_4, dnV(p), npo, dnQ(qq), one, &
                      w(dt12 + NW * (t12off(g, p) + (i - otoff(g)) * dnQ(qq))), dnJ(g) * dnQ(qq), &
                      w(jgpp + NW * vpoff(qq)), nvp, one, w(dw2 + NW * (i * nv + vcoff(p)) * npo), npo)
    end do
    !$omp parallel do private(q, a, b)
    do m = 0, no - 1
      do q = 0, nvp - 1
        a = vpa(q)
        b = vpb(q)
        xo(m * nvp + q) = xo(m * nvp + q) + y((m * nv + a) * nv + b) - y((m * nv + b) * nv + a)
      end do
    end do
  end do
end subroutine

! B4(ml g; jk; c) = sum_h <ml||hg> t2(jk; ch) as [(jk)][m][l][g][c]
subroutine N(xzd5b4)(w)
  use xzdata
  use xzkern
  use xzdead
  use xzdead5
  implicit none
  real(8), target :: w(*)
  integer(8), pointer :: opa(:), opb(:)
  integer(8) :: pq, m, l
  ZT :: one, zero
  one = 1
  zero = 0
  call xzdiv(w, jopa, max(npo, 1_8), opa)
  call xzdiv(w, jopb, max(npo, 1_8), opb)
  do pq = 0, npo - 1
    do m = 0, no - 1
      do l = 0, no - 1
        call N(zc_gemm)(1_4, 1_4, nv, nv, nv, one, w(joovv + NW * (m * no + l) * nv * nv), nv, &
                        w(jt2 + NW * (opa(pq) * no + opb(pq)) * nv * nv), nv, zero, &
                        w(db4 + NW * ((pq * no + m) * no + l) * nv * nv), nv)
      end do
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The sector of a target (after xzdsec and xzd5sec): the numbers of the
! quintuples of each parity of every block, the rows of r5 (xzd5store), the
! pool of the quintuple lists, the layouts of the intermediates of a sigma
! vector.
subroutine N(xzd5secw)(w)
  use xzdata
  use xzmem
  use xzkern
  use xzdead
  use xzdead5
  use iso_c_binding
  implicit none
  real(8), target :: w(*)
  integer(8) :: m, n, i, own, big(10), k, lp
  integer(8), allocatable :: cnt(:)
  character(len=32) :: arg
  integer :: larg
  logical :: mine(0:2*n5c-1)
  ! the numbers of the quintuples of each parity (the plan: half of each block)
  q5n = 0
  if (xzdry .or. .not. allocated(dpv)) then
    do m = 0, n5c - 1
      q5n(0, m) = nq5(m)
      if (dtpar >= 0) then
        q5n(0, m) = (nq5(m) + 1) / 2
        q5n(1, m) = (nq5(m) + 1) / 2
      end if
    end do
  else
    if (allocated(d5pv)) deallocate(d5pv)
    allocate(d5pv(0:max(nv, 1_8)-1))
    do i = 0, nv - 1
      d5pv(i) = int(dpv(i), 4)
    end do
    allocate(cnt(0:nv))
    do m = 0, n5c - 1
      if (.not. inblk5(m)) cycle
      if (dtpar < 0) then
        q5n(0, m) = nq5(m)
      else
        q5n(0, m) = zc_t5s_quints_par(int(nv, 4), w(dkv), int(m, 4), 0_c_intptr_t, cnt, loc(d5pv), 0_4)
        q5n(1, m) = nq5(m) - q5n(0, m)
      end if
    end do
    deallocate(cnt)
  end if
  call xzd5store()
  ! the pool of the quintuple lists: the lists of the rows of this rank and the
  ! ten largest others (all of them if that is not less); option dea5pool: its words
  mine = .false.
  do i = dt0, dt1 - 1
    if (blk5(i) >= 0) mine(2 * blk5(i) + r5lp(i)) = .true.
  end do
  own = 0
  big = 0
  do k = 0, 2 * n5c - 1
    if (mine(k)) then
      own = own + q5len(k)
    else
      do lp = 1, 10
        if (q5len(k) > big(lp)) then
          big(lp + 1:10) = big(lp:9)
          big(lp) = q5len(k)
          exit
        end if
      end do
    end if
  end do
  q5cap = min(q5size, own + sum(big))
  call xzoptget('dea5pool', arg, larg)
  if (larg > 0) read(arg(1:larg), *) q5cap
  q5cap = max(q5cap, maxval(q5len))
  q5pool = xzget(max(q5cap, 1_8), 'DEA5: the quintuple lists of the blocks of r5 (a pool)')
  q5top = q5pool
  q5tick = 0
  q5made = 0
  q5at = 0
  call xzd5lay(w)
end subroutine

! ---------------------------------------------------------------------------
! The kernels of csrc/t4sym.c for rank 5 with the tables of the program: a
! block of rows of the quintuples of class m and parity lp (the rows at the
! word address kr, leading dimension ldr) and the products at kx
subroutine N(xzd5x23)(w, mode, kr, ldr, m, lp, nent, row, sg, xoff, s1, s2, kx)
  use xzdata
  use xzkern
  use xzdead
  use xzdead5
  implicit none
  real(8), target :: w(*)
  integer, intent(in) :: mode
  integer(8), intent(in) :: kr, ldr, m, lp, nent, row(*), xoff(*), s1(*), s2(*), kx
  real(8), intent(in) :: sg(*)
  if (nent == 0 .or. q5n(lp, m) == 0) return
  call xzd5q5(w, m, lp)
  call N(zc_t5s_x23)(int(mode, c_int), w(kr), ldr, w(q5at(2 * m + lp)), q5n(lp, m), nent, row, sg, xoff, s1, s2, &
                     int(npg, c_int), int(nv, c_int), w(dkv), w(dposq), w(dpos3), w(kx))
end subroutine

subroutine N(xzd5x14)(w, mode, kr, ldr, m, lp, nent, row, sg, xoff, s1, s2, kx)
  use xzdata
  use xzkern
  use xzdead
  use xzdead5
  implicit none
  real(8), target :: w(*)
  integer, intent(in) :: mode
  integer(8), intent(in) :: kr, ldr, m, lp, nent, row(*), xoff(*), s1(*), s2(*), kx
  real(8), intent(in) :: sg(*)
  if (nent == 0 .or. q5n(lp, m) == 0) return
  call xzd5q5(w, m, lp)
  call N(zc_t5s_x14)(int(mode, c_int), w(kr), ldr, w(q5at(2 * m + lp)), q5n(lp, m), nent, row, sg, xoff, s1, s2, &
                     int(ncls, c_int), int(nv, c_int), w(dkv), w(dposv), w(dpos4), w(kx))
end subroutine

! the parity-split zc_t5s_x14p (xoff, s1, s2 per entry by 2 label + parity of a)
subroutine N(xzd5x14p)(w, mode, kr, ldr, m, lp, nent, row, sg, xoff, s1, s2, kx)
  use xzdata
  use xzkern
  use xzdead
  use xzdead5
  implicit none
  real(8), target :: w(*)
  integer, intent(in) :: mode
  integer(8), intent(in) :: kr, ldr, m, lp, nent, row(*), xoff(*), s1(*), s2(*), kx
  real(8), intent(in) :: sg(*)
  if (nent == 0 .or. q5n(lp, m) == 0) return
  call xzd5q5(w, m, lp)
  call N(zc_t5s_x14p)(int(mode, c_int), w(kr), ldr, w(q5at(2 * m + lp)), q5n(lp, m), nent, row, sg, xoff, s1, s2, &
                      int(ncls, c_int), int(nv, c_int), w(dkv), d5pvz, d5posvp, w(dpos4p), w(kx))
end subroutine

! the row (sign sgn) of the triple i of r5 at kr into X[a of p][(bcde)] at kx
! (mode 0), or back (mode 1): off14 / n14 of its block
subroutine N(xzd5row14)(w, mode, kr, i, sgn, kx)
  use xzdata
  use xzdead
  use xzdead5
  implicit none
  real(8), target :: w(*)
  integer, intent(in) :: mode
  integer(8), intent(in) :: kr, i, kx
  real(8), intent(in) :: sgn
  integer(8) :: row(1), xo(ncls), y1(ncls), y2(ncls), p, m
  real(8) :: sg(1)
  m = blk5(i)
  row(1) = 0
  sg(1) = sgn
  do p = 0, ncls - 1
    xo(p + 1) = off14(m, p)
    y1(p + 1) = max(d5nA4(m - p), 1_8)
    y2(p + 1) = 1
  end do
  call N(xzd5x14)(w, mode, kr, max(r5w(i), 1_8), m, r5lp(i), 1_8, row, sg, xo, y1, y2, kx)
end subroutine

! the row of the triple i as X[(ab) of q][(cde)] of the groups q0 <= q < q1
! (their blocks at kx + off23 - base; the others left out)
subroutine N(xzd5row23p)(w, mode, kr, i, sgn, kx, q0, q1, base)
  use xzdata
  use xzdead
  use xzdead5
  implicit none
  real(8), target :: w(*)
  integer, intent(in) :: mode
  integer(8), intent(in) :: kr, i, kx, q0, q1, base
  real(8), intent(in) :: sgn
  integer(8) :: row(1), xo(npg), y1(npg), y2(npg), q, m
  real(8) :: sg(1)
  m = blk5(i)
  row(1) = 0
  sg(1) = sgn
  do q = 0, npg - 1
    xo(q + 1) = -1
    if (q >= q0 .and. q < q1 .and. off23(m, q) >= 0) xo(q + 1) = off23(m, q) - base
    y1(q + 1) = max(dnK(m - q), 1_8)
    y2(q + 1) = 1
  end do
  call N(xzd5x23)(w, mode, kr, max(r5w(i), 1_8), m, r5lp(i), 1_8, row, sg, xo, y1, y2, kx)
end subroutine

! the row of the pair pq of r4 (at k4, whole r4) on (h | cde) into kx (the
! blocks off13 of its block of r4; n13 elements; zero where nothing goes)
subroutine N(xzd5x4row)(w, k4, pq, kx)
  use xzdata
  use xzkern
  use xzdead
  use xzdead5
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: k4, pq, kx
  integer(8) :: m4, row(1), xo(ncls), y1(ncls), y2(ncls), p
  real(8) :: sg(1)
  m4 = blk4(pq)
  if (m4 < 0) return
  call zc_vzero_d(NW * max(n13(m4), 1_8), w(kx))
  row(1) = pos4(pq)
  sg(1) = 1d0
  do p = 0, ncls - 1
    xo(p + 1) = off13(m4, p)
    y1(p + 1) = max(dnK(m4 - p), 1_8)
    y2(p + 1) = 1
  end do
  call N(xzdx13)(w, 0, k4 + NW * off4(m4), nqa(m4), m4, 1_8, row, sg, xo, y1, y2, kx)
end subroutine

! ---------------------------------------------------------------------------
! The preconditioner of the rows of r5 of this rank (at kc, the part of the
! rank): x <- x / (omega - D), D the energies of the virtual spinors of the
! element less those of the occupied ones, below 1e-3 replaced by 1e-3
subroutine N(xzd5prec)(w, kc, omega)
  use xzdata
  use xzkern
  use xzdead
  use xzdead5
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: kc
  ZT, intent(in) :: omega
  real(8), pointer :: eps(:)
  integer(2), pointer :: q5(:, :)
  integer(8) :: i, m, n, a, a0, a1, e0
  real(8) :: eo(1)
  real(8), allocatable :: ev(:)
  integer(8), parameter :: chunk = 1048576
  real(8), parameter :: floor = 1d-3
  call r8view(jeps, no + nv, eps)
  allocate(ev(0:chunk-1))
  e0 = no
  do i = dt0, dt1 - 1
    m = blk5(i)
    if (m < 0 .or. r5w(i) == 0) cycle
    n = r5w(i)
    eo(1) = eps(xzd5free(i, 0)) + eps(xzd5free(i, 1)) + eps(xzd5free(i, 2))
    call xzd5q5(w, m, r5lp(i))
    call xzdi2v2(w, q5at(2 * m + r5lp(i)), 5_8, max(n, 1_8), q5)
    ! the columns in pieces of at most chunk: their energies, the divisions
    do a0 = 0, n - 1, chunk
      a1 = min(n, a0 + chunk)
      !$omp parallel do
      do a = a0, a1 - 1
        ev(a - a0) = eps(e0 + q5(0, a)) + eps(e0 + q5(1, a)) + eps(e0 + q5(2, a)) + eps(e0 + q5(3, a)) + eps(e0 + q5(4, a))
      end do
      call N(zc_vprecond)(w(kc + NW * (r5own(i) + a0)), 1_8, a1 - a0, eo, ev, omega, floor)
    end do
  end do
  deallocate(ev)
contains
  subroutine r8view(kk, nn, p)
    use iso_c_binding
    integer(8), intent(in) :: kk, nn
    real(8), pointer :: p(:)
    real(8), pointer :: qq(:)
    call c_f_pointer(c_loc(w(kk)), qq, [max(nn, 1_8)])
    p(0:max(nn, 1_8) - 1) => qq
  end subroutine
end subroutine
