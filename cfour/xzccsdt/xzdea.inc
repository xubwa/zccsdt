!
! EOM-DEA-CCSDT on the sector itself (type-generic; xzdead.F90 has the
! input and the tables; docs/native.md, section 20).  The algorithm and the
! layouts are those of src/zccsdt/cc/dea_sector.py (docs/eom.md, section
! 11.9), the reference of every number: the sigma vector of
! R = 2p + 3p1h + 4p2h, r2[(a<b)], r3[i, (a<b<c)], r4[(i<j), (a<b<c<d)], in
! the blocks of the labels, on the ground state of CCSDT (T1-dressed
! integrals, T2, T3), factorized as the T4 equation of CCSDTQ with the
! sector vector in the place of one amplitude:
!
!   4p2h  pp     P(ab/cd) sum_{e<f} Wq(ab,ef) r4(ij; efcd)            Wq = Wpp + F_vv on pairs / 3
!         hh     sum_{m<n} Wh(mn,ij) r4(mn; abcd)                      Wh = Whh - F_oo on pairs
!         ring   P(ij) P(a/bcd) sum_{me} Wr(m,a,e,i) r4(mj; ebcd)
!         22     P(ab/cd) { P(ij) [ - sum_e Wv(ab,e,i) r3(j; cde) + sum_m t(im,ab) XoR(m; j; cd) ]
!                           + sum_{nf} ZR(nf; ab) t(nij; fcd) }
!         13     P(a/bcd) { sum_m Wo(m,a,ij) r3(m; bcd) + sum_e t(ij,ae) XvR(e; bcd)
!                           + sum_m WoR(m,a) t(mij; bcd) + sum_e r2(ae) Xv(e; ij; bcd)
!                           - P(ij) sum_{m<n} YR(mn; j; a) t(mni; bcd) }
!   3p1h, 2p: the rows of dea_sector (its module docstring)
!
! with the intermediates of the sector (linear in R, made per sigma vector:
! XoR, XvR = Z, WoR, YR, WhR, ZR, h) and of the ground state (made once:
! F_vv, F_oo, Wpp, Whh, Wr, Wv, Wo with their T3 parts - those of the EOM-EE
! code, N(xzeomim) -, T3 on (e | cd), T3 by an index and a pair, and the
! six-index Xv(e; kl; bcd) of the T4 equation without T4 with the rows of T3
! above it).  The products go through the kernels of csrc/t4sym.c (a block
! of rows expanded into the splits (ab | cd), (a | bcd), (a | bc) and reduced
! back) and GEMMs; nothing of the size of T4 is made.
!
! Arrays: row-major, indices from 0 (xzdata.F90); views of the work array
! for the loops, the word addresses for the kernels.
!


! ---------------------------------------------------------------------------
! The places of the arrays of the ground state that the sigma vectors use
! (after N(xzeomkeep); released with its mark).
subroutine N(xzdkeep)()
  use xzdata
  use xzmem
  use xzdead
  implicit none
  dkv = xzget(nv / 2 + 1, 'DEA: tables')
  dposv = xzget(nv / 2 + 1, 'DEA: tables')
  dposq = xzget(nv * nv / 2 + 1, 'DEA: tables')
  dpos3 = xzget(nvt / 2 + 1, 'DEA: tables')
  dtri = xzget(dtrisize, 'DEA: tables')
  dquads = xzget(nqall + 1, 'DEA: the quadruples of the classes')
  dwvat = xzget(NW * no * nv * nvp, 'DEA: W_vvvo [i][e][Q]')
  dla13 = xzget(NW * npo * nv * (no + nv), 'DEA: [W_vooo | t2] by pairs')
  dt12 = xzget(NW * t12size, 'DEA: T3 on (e | cd)')
  dt3p = xzget(NW * t3psize, 'DEA: T3 by an index and a pair')
  if (dxvfile) then
    dr13 = xzget(NW * r13pmax, 'DEA: the rows of T3 and Xv of one pair (the file XZDXV)')
  else
    dr13 = xzget(NW * r13loc, 'DEA: the rows of T3 and Xv')
  end if
  dwq4 = xzget(NW * nvv, 'DEA: the pp operator of the 4p2h row')
  dwb = xzget(NW * wbsize, 'DEA: W_pp as [(bc) e][f]')
  dwh = xzget(NW * npo * npo, 'DEA: the hh operator')
  dwr2 = xzget(NW * no * nv * no * nv, 'DEA: the ring operator [(i a)][(m e)]')
  dvo = xzget(NW * vosize, 'DEA: <dm||ef> in blocks')
end subroutine

! ---------------------------------------------------------------------------
! The intermediates of the ground state of the sigma vectors (after
! N(xzeomim), whose arrays they are made from: W_vvvo, W_vooo, W_ovvo, F_vv,
! F_oo, W_oooo, the pp operator of the T3 equation in the place of <ab||cd>,
! the dressed integrals).
subroutine N(xzdim)(w)
  use xzdata
  use xzmem
  use xzkern
  use xzdead
  implicit none
  real(8), target :: w(*)
  integer(8) :: m0, kwrb, kmx, koa, ktmp, kpib
  ZT :: mone
  mone = -1
  m0 = xzmark()
  kwrb = xzget(NW * wrsize, 'DEA: Wr in blocks')
  kmx = xzget(NW * 2 * mxmax, 'DEA: the products of Xv of one pair')
  koa = xzget(NW * no * nv * npo, 'DEA: <mn||el> as [l][e][(mn)]')
  ktmp = xzget(NW * nqmax * nqmax, 'DEA: a block of W_pp')
  kpib = xzget(nv * nv, 'DEA: the pairs of a group')
  if (.not. xzdry) then
    call xzdtabs(w)
    call N(xzdwvat)(no, nv, nvp, w(ewv), w(dwvat))
    call N(xzdla13)(no, nv, npo, ocoff, vcoff, ncls, mix0, w(jopa), w(jopb), w(ewo), w(efov), w(jt2), w(dla13))
    call N(xzdwr2)(no, nv, w(eweff), w(dwr2))
    call N(xzdt3)(w)
    call N(xzdwpp)(w, ktmp, kpib)
    call N(xzecopy)(npo * npo, w(ewkl), w(dwh))
    if (npo > 0) call N(zc_pair_one_body_add)(w(dwh), npo, npo, w(jopa), w(jopb), w(jopidx), no, w(efeoo), no, mone)
    call N(xzdxv)(w, dvo, kwrb, kmx, koa)
  end if
  call xzrel(m0)
end subroutine

! wvat[i][e][Q] = W_vvvo[Q][e][i]
subroutine N(xzdwvat)(no, nv, nvp, wv, wvat)
  implicit none
  integer(8), intent(in) :: no, nv, nvp
  ZT, intent(in) :: wv(0:no-1, 0:nv-1, 0:nvp-1)
  ZT, intent(out) :: wvat(0:nvp-1, 0:nv-1, 0:no-1)
  integer(8) :: i, e
  !$omp parallel do collapse(2)
  do i = 0, no - 1
    do e = 0, nv - 1
      wvat(:, e, i) = wv(i, e, :)
    end do
  end do
end subroutine

! la13[P][a][mix]: Wo(m, a, P) for the m of class c at mix0(c) + (m - o0(c)),
! t2[P][a][e] for the e of class c behind them.  Wo of the T4 equation is
! - W_vooo of the T3 equation of the native program - sum_e f[m][e] t2[P][a][e]
! (N(xzwo) leaves the f term to the T3 equation).
subroutine N(xzdla13)(no, nv, npo, ocoff, vcoff, ncls, mix0, opa, opb, wo, fov, t2, la13)
  implicit none
  integer(8), intent(in) :: no, nv, npo, ncls
  integer(8), intent(in) :: ocoff(0:*), vcoff(0:*), mix0(0:*), opa(0:max(npo, 1_8)-1), opb(0:max(npo, 1_8)-1)
  ZT, intent(in) :: wo(0:npo-1, 0:nv-1, 0:no-1), fov(0:nv-1, 0:no-1), t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(out) :: la13(0:no+nv-1, 0:nv-1, 0:npo-1)
  integer(8) :: p, i, j, a, c, m, e
  ZT :: s
  !$omp parallel do private(i, j, a, c, m, e, s)
  do p = 0, npo - 1
    i = opa(p)
    j = opb(p)
    do a = 0, nv - 1
      do c = 0, ncls - 1
        do m = ocoff(c), ocoff(c + 1) - 1
          s = -wo(p, a, m)
          do e = 0, nv - 1
            s = s - fov(e, m) * t2(e, a, j, i)
          end do
          la13(mix0(c) + m - ocoff(c), a, p) = s
        end do
        do e = vcoff(c), vcoff(c + 1) - 1
          la13(mix0(c) + ocoff(c + 1) - ocoff(c) + e - vcoff(c), a, p) = t2(e, a, j, i)
        end do
      end do
    end do
  end do
end subroutine

! wr2[(i a)][(m e)] = Wr(m, a, e, i) = weff[m][a][e][i]
subroutine N(xzdwr2)(no, nv, weff, wr2)
  implicit none
  integer(8), intent(in) :: no, nv
  ZT, intent(in) :: weff(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)
  ZT, intent(out) :: wr2(0:nv-1, 0:no-1, 0:nv-1, 0:no-1)
  integer(8) :: i, a, m, e
  !$omp parallel do collapse(2) private(m, e)
  do i = 0, no - 1
    do a = 0, nv - 1
      do m = 0, no - 1
        do e = 0, nv - 1
          wr2(e, m, a, i) = weff(i, e, a, m)
        end do
      end do
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! T3 in the forms of the sigma vectors: on (e | cd) (T12), by an index and a
! pair (T3P), and its rows above Xv in R13.
subroutine N(xzdt3)(w)
  use xzdata
  use xzkern
  use xzdead
  implicit none
  real(8), target :: w(*)
  ZT, pointer :: t3(:), t3p(:)
  integer(8), pointer :: otri(:, :), opgrp(:)
  integer(1), pointer :: osgn(:, :)
  integer(8) :: k, p, j, nj, nk, i, c, o, r, pq, jj, mu, m, kk
  integer(8), allocatable :: row(:), xoff(:, :), s1(:, :), s2(:, :)
  real(8), allocatable :: sg(:)
  real(8) :: s
  call N(xzdv)(w, jt3, n3, t3)
  call N(xzdv)(w, dt3p, t3psize, t3p)
  call xzdiv2(w, jotri, npo, no, otri)
  call xzdiv(w, jopgrp, npo, opgrp)
  call xzdi1v2(w, josgn, npo, no, osgn)
  call zc_vzero_d(NW * t12size, w(dt12))
  call zc_vzero_d(NW * t3psize, w(dt3p))
  ! T12: the triples J of group k as [e][J][(cd)]
  do k = 0, n3c - 1
    nj = dnJ(k)
    nk = dnK(k)
    if (nj == 0 .or. nk == 0) cycle
    allocate(row(0:nj-1), sg(0:nj-1), xoff(0:ncls-1, 0:nj-1), s1(0:ncls-1, 0:nj-1), s2(0:ncls-1, 0:nj-1))
    do j = 0, nj - 1
      row(j) = j
      sg(j) = 1d0
      do p = 0, ncls - 1
        xoff(p, j) = -1
        s1(p, j) = 1
        s2(p, j) = 1
        if (t12off(k, p) >= 0) then
          xoff(p, j) = t12off(k, p) + j * dnQ(k - p)
          s1(p, j) = nj * dnQ(k - p)
        end if
      end do
    end do
    call N(xzdx12)(w, 0, jt3 + NW * boff(k), nk, k, nj, row, sg, xoff, s1, s2, dt12)
    deallocate(row, sg, xoff, s1, s2)
  end do
  ! T3P: t(i mn; K) as [(mn) of group c][K]
  !$omp parallel do private(c, o, kk, nk, r, pq, jj, s) schedule(dynamic)
  do i = 0, no - 1
    do c = 0, n2c - 1
      o = t3poff(i, c)
      kk = ocls(i) + c
      if (o < 0 .or. dnJ(kk) == 0) cycle
      nk = dnK(kk)
      do r = 0, dnO2(c) - 1
        pq = opoff(c) + r
        s = osgn(pq, i)
        if (s == 0d0) cycle
        jj = otri(pq, i)
        t3p(o + r * nk:o + (r + 1) * nk - 1) = s * t3(boff(kk) + (jj - otoff(kk)) * nk:boff(kk) + (jj - otoff(kk) + 1) * nk - 1)
      end do
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The kernels of csrc/t4sym.c with the tables of the program: a block of
! rows of a list of triples of group k / quadruples of class M (the array at
! the word address kr, leading dimension ldr) and the products at kx.
subroutine N(xzdx12)(w, mode, kr, ldr, k, nent, row, sg, xoff, s1, s2, kx)
  use xzdata
  use xzkern
  use xzdead
  implicit none
  real(8), target :: w(*)
  integer, intent(in) :: mode
  integer(8), intent(in) :: kr, ldr, k, nent, row(*), xoff(*), s1(*), s2(*), kx
  real(8), intent(in) :: sg(*)
  if (nent == 0 .or. dnK(k) == 0) return
  call N(zc_t4s_x12)(int(mode, c_int), w(kr), ldr, w(dtri + dtrio(k)), dnK(k), nent, row, sg, xoff, s1, s2, &
                     int(ncls, c_int), int(nv, c_int), w(dkv), w(dposv), w(dposq), w(kx))
end subroutine

subroutine N(xzdx13)(w, mode, kr, ldr, m, nent, row, sg, xoff, s1, s2, kx)
  use xzdata
  use xzkern
  use xzdead
  implicit none
  real(8), target :: w(*)
  integer, intent(in) :: mode
  integer(8), intent(in) :: kr, ldr, m, nent, row(*), xoff(*), s1(*), s2(*), kx
  real(8), intent(in) :: sg(*)
  if (nent == 0 .or. nqa(m) == 0) return
  call N(zc_t4s_x13)(int(mode, c_int), w(kr), ldr, w(dquads + qa0(m)), nqa(m), nent, row, sg, xoff, s1, s2, &
                     int(ncls, c_int), int(nv, c_int), w(dkv), w(dposv), w(dpos3), w(kx))
end subroutine

subroutine N(xzdx22)(w, mode, kr, ldr, m, nent, row, sg, xoff, s1, s2, kx)
  use xzdata
  use xzkern
  use xzdead
  implicit none
  real(8), target :: w(*)
  integer, intent(in) :: mode
  integer(8), intent(in) :: kr, ldr, m, nent, row(*), xoff(*), s1(*), s2(*), kx
  real(8), intent(in) :: sg(*)
  if (nent == 0 .or. nqa(m) == 0) return
  call N(zc_t4s_x22)(int(mode, c_int), w(kr), ldr, w(dquads + qa0(m)), nqa(m), nent, row, sg, xoff, s1, s2, &
                     int(npg, c_int), int(nv, c_int), w(dkv), w(dposq), w(kx))
end subroutine

! ---------------------------------------------------------------------------
! The pp operators: of the 4p2h row Wq4 = Wpp + F_vv / 3 on pairs; W_pp as
! [(bc) e][f] for the 4p2h term of XvR and for Xv (from the pp operator of
! the T3 equation, Wpp + F_vv / 2, in the place of <ab||cd>).
subroutine N(xzdwpp)(w, ktmp, kpib)
  use xzdata
  use xzkern
  use xzdead
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: ktmp, kpib
  integer(8) :: h, lo, nqh, mu
  ZT :: sc
  do h = 0, npg - 1
    lo = vpoff(h)
    nqh = vpoff(h + 1) - lo
    if (nqh == 0) cycle
    call N(xzepib)(nv, lo, nqh, w(jvpidx), w(kpib))
    call N(xzecopy)(nqh * nqh, w(jvvvv + NW * vvoff(h)), w(dwq4 + NW * vvoff(h)))
    sc = -1d0 / 6d0
    call N(zc_pair_one_body_add)(w(dwq4 + NW * vvoff(h)), nqh, nqh, w(jvpa + lo), w(jvpb + lo), w(kpib), nv, &
                                 w(efevv), nv, sc)
    call N(xzecopy)(nqh * nqh, w(jvvvv + NW * vvoff(h)), w(ktmp))
    sc = -0.5d0
    call N(zc_pair_one_body_add)(w(ktmp), nqh, nqh, w(jvpa + lo), w(jvpb + lo), w(kpib), nv, w(efevv), nv, sc)
    do mu = 0, ncls - 1
      if (wboff(h, mu) < 0) cycle
      call N(xzdwbblk)(nv, nqh, dnV(mu), dnV(h - mu), vcoff(mu), vcoff(h - mu), w(kpib), w(ktmp), &
                       w(dwb + NW * wboff(h, mu)))
    end do
  end do
end subroutine

! wb[(bc)][e][f] = Wpp[(bc)][(ef)] with the sign of (e, f) (0 for e = f)
subroutine N(xzdwbblk)(nv, nqh, ne, nf, e0, f0, pib, wpp, wb)
  implicit none
  integer(8), intent(in) :: nv, nqh, ne, nf, e0, f0
  integer(8), intent(in) :: pib(0:nv-1, 0:nv-1)
  ZT, intent(in) :: wpp(0:nqh-1, 0:nqh-1)
  ZT, intent(out) :: wb(0:nf-1, 0:ne-1, 0:nqh-1)
  integer(8) :: q, e, f, a, b
  !$omp parallel do private(e, f, a, b)
  do q = 0, nqh - 1
    do e = 0, ne - 1
      a = e0 + e
      do f = 0, nf - 1
        b = f0 + f
        if (a < b) then
          wb(f, e, q) = wpp(pib(b, a), q)
        else if (a > b) then
          wb(f, e, q) = -wpp(pib(a, b), q)
        else
          wb(f, e, q) = 0
        end if
      end do
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The six-index Xv(e; kl; bcd) of the T4 equation without T4, below the rows
! of T3 in R13 (zccsdt.cc.t4engine: residual, 'Xv'; t4engine_sym: tasks_xv):
!   - P(kl) sum_{m<n} <mn||el> t(mnk, bcd)
!   + P(d/bc) [ sum_{mf} <dm||ef> t(m kl; f bc) + P(kl) sum_m Wr(m,d,e,k) t(ml, bc)
!               + 1/2 sum_f Wpp(bc,ef) t(kl, fd) ]
! per pair (k, l) the products of the second line in Mx, reduced over (d | bc).
subroutine N(xzdxv)(w, kvo, kwrb, kmx, koa)
  use xzdata
  use xzkern
  use xzdead
  use xzdead5, only: x2loc, dx2
  use omp_lib
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: kvo, kwrb, kmx, koa
  integer(8), pointer :: opa(:), opb(:), opgrp(:), otri(:, :), vpidx(:, :)
  integer(1), pointer :: osgn(:, :)
  ZT, pointer :: vo(:), wrb(:), oa(:), edv(:), weff(:), ooovd(:), t3(:), r13(:)
  integer(8) :: m, mu, ka, ph, c, o, e, d, f, ee, dd, ff, pq, k, l, kp, kk, nk, a, b, nb, be, jj, kj, ne, nt, kb, r
  integer :: ios
  integer(8), allocatable :: row(:), xoff(:, :), s1(:, :), s2(:, :), blk(:, :)
  real(8), allocatable :: sg(:)
  real(8) :: sgn
  ZT :: one, half, alpha, hw
  logical :: anyb
  integer(8) :: pb, pof
  one = 1
  half = 0.5d0
  ! (dxv2: Xv2 of the pairs of x2loc for the 5p3h row - the term of Wpp counted twice - into dx2)
  hw = half
  if (dxv2) hw = one
  call xzdiv(w, jopa, npo, opa)
  call xzdiv(w, jopb, npo, opb)
  call xzdiv(w, jopgrp, npo, opgrp)
  call xzdiv2(w, jotri, npo, no, otri)
  call xzdi1v2(w, josgn, npo, no, osgn)
  call xzdiv2(w, jvpidx, nv, nv, vpidx)
  call N(xzdv)(w, kvo, vosize, vo)
  call N(xzdv)(w, kwrb, wrsize, wrb)
  call N(xzdv)(w, koa, no * nv * npo, oa)
  call N(xzdv)(w, ed, no * nv * nvp, edv)
  call N(xzdv)(w, eweff, no * nv * nv * no, weff)
  call N(xzdv)(w, eooovd, no * no * no * nv, ooovd)
  call N(xzdv)(w, jt3, n3, t3)
  if (.not. dxvfile .and. .not. dxv2) call zc_vzero_d(NW * r13loc, w(dr13))
  ! <dm||ef> = - <md||ef> as [e of mu][d of ka][f of ph], Wr(c, d, e, m) as [e][d][c]
  !$omp parallel do collapse(3) private(ph, c, o, ee, dd, ff, e, d, f)
  do m = 0, no - 1
    do mu = 0, ncls - 1
      do ka = 0, ncls - 1
        o = vooff(m, mu, ka)
        if (o >= 0) then
          ph = ka + ocls(m) - mu
          do ee = 0, dnV(mu) - 1
            e = vcoff(mu) + ee
            do dd = 0, dnV(ka) - 1
              d = vcoff(ka) + dd
              do ff = 0, dnV(ph) - 1
                f = vcoff(ph) + ff
                if (e < f) then
                  vo(o + (ee * dnV(ka) + dd) * dnV(ph) + ff) = -edv((m * nv + d) * nvp + vpidx(f, e))
                else if (e > f) then
                  vo(o + (ee * dnV(ka) + dd) * dnV(ph) + ff) = edv((m * nv + d) * nvp + vpidx(e, f))
                else
                  vo(o + (ee * dnV(ka) + dd) * dnV(ph) + ff) = 0
                end if
              end do
            end do
          end do
        end if
        o = wroff(m, mu, ka)
        if (o >= 0) then
          c = mu + ocls(m) - ka
          do ee = 0, dnV(mu) - 1
            e = vcoff(mu) + ee
            do dd = 0, dnV(ka) - 1
              d = vcoff(ka) + dd
              do ff = 0, dnO(c) - 1
                f = ocoff(c) + ff
                ! Wr(c, d, e, m) = weff[c][d][e][m]
                wrb(o + (ee * dnV(ka) + dd) * dnO(c) + ff) = weff(((f * nv + d) * nv + e) * no + m)
              end do
            end do
          end do
        end if
      end do
    end do
  end do
  ! <mn||el> = - ~<mn||le> as [l][e][(mn)]
  !$omp parallel do collapse(2) private(pq)
  do l = 0, no - 1
    do e = 0, nv - 1
      do pq = 0, npo - 1
        oa((l * nv + e) * npo + pq) = -ooovd(((opa(pq) * no + opb(pq)) * no + l) * nv + e)
      end do
    end do
  end do
  ! the targets (mu, ka) of the products of one pair, for the parallel loop
  allocate(blk(2, ncls * ncls))
  ! (the pairs of this rank; the chunk of a pair at kb: in the work array, or
  ! in the buffer that goes to the file XZDXV)
  do pq = merge(0_8, dq0, dxv2), merge(npo, dq1, dxv2) - 1
    if (dxv2) then
      if (x2loc(pq) < 0) cycle
    end if
    k = opa(pq)
    l = opb(pq)
    kp = opgrp(pq)
    ! pb: the first word of the chunk of the pair, pof: the element of R13 there
    if (dxv2) then
      pb = dx2 + NW * x2loc(pq)
      pof = r13p(pq)
      call zc_vzero_d(NW * (r13p(pq + 1) - r13p(pq)), w(pb))
    else if (dxvfile) then
      pb = dr13
      pof = r13p(pq)
      call zc_vzero_d(NW * (r13p(pq + 1) - r13p(pq)), w(dr13))
    else
      pb = dr13 + NW * (r13p(pq) - r13p(dq0))
      pof = r13p(pq)
    end if
    kb = pb - NW * pof
    ! the rows t(m kl; K) of the m of class mu
    call N(xzdv)(w, pb, max(r13p(pq + 1) - r13p(pq), 1_8), r13)
    !$omp parallel do private(o, kk, nk, m, r, sgn, jj)
    do mu = 0, ncls - 1
      o = r13off(pq, mu)
      if (o < 0) cycle
      o = o - pof
      kk = kp + mu
      nk = dnK(kk)
      do m = ocoff(mu), ocoff(mu + 1) - 1
        r = m - ocoff(mu)
        sgn = osgn(pq, m)
        if (sgn == 0d0) cycle
        jj = otri(pq, m)
        r13(o + r * nk:o + (r + 1) * nk - 1) = sgn * t3(boff(kk) + (jj - otoff(kk)) * nk:boff(kk) + (jj - otoff(kk) + 1) * nk - 1)
      end do
    end do
    if (mxsize(kp) == 0 .and. all(r13off(pq, :) < 0)) then
      call write_pair()
      cycle
    end if
    ! - P(kl) sum_{m<n} <mn||el> t(mnk, bcd): into the Xv rows directly
    do mu = 0, ncls - 1
      if (r13off(pq, mu) < 0 .or. dnV(mu) == 0) cycle
      kk = kp + mu
      nk = dnK(kk)
      do nb = 1, 2
        if (nb == 1) then
          a = l
          b = k
          alpha = -1
        else
          a = k
          b = l
          alpha = 1
        end if
        c = mu + ocls(a)
        if (c >= n2c .or. dnO2(c) == 0) cycle
        if (t3poff(b, c) < 0) cycle
        call N(zc_gemm)(0_4, 0_4, dnV(mu), nk, dnO2(c), alpha, w(koa + NW * ((a * nv + vcoff(mu)) * npo + opoff(c))), &
                        npo, w(dt3p + NW * t3poff(b, c)), nk, one, &
                        w(kb + NW * (r13off(pq, mu) + dnO(mu) * nk)), nk)
      end do
    end do
    if (mxsize(kp) == 0) then
      call write_pair()
      cycle
    end if
    call zc_vzero_d(NW * 2 * mxsize(kp), w(kmx))
    nt = 0
    do mu = 0, ncls - 1
      do ka = 0, ncls - 1
        if (mxoff(kp, mu, ka) < 0) cycle
        nt = nt + 1
        blk(1, nt) = mu
        blk(2, nt) = ka
      end do
    end do
    !$omp parallel do schedule(dynamic) private(mu, ka, o, be, m, sgn, ph, jj, kj, nb, a, b, c, alpha)
    do jj = 1, nt
      mu = blk(1, jj)
      ka = blk(2, jj)
      o = mxoff(kp, mu, ka)
      be = kp + mu - ka
      ! sum_{mf} <dm||ef> t(m kl; f, bc) as [e, d][(bc)]
      do m = 0, no - 1
        sgn = osgn(pq, m)
        ph = ka + ocls(m) - mu
        if (sgn == 0d0 .or. vooff(m, mu, ka) < 0) cycle
        kj = opgrp(pq) + ocls(m)
        if (t12off(kj, ph) < 0) cycle
        alpha = sgn
        call N(zc_gemm)(0_4, 0_4, dnV(mu) * dnV(ka), dnQ(be), dnV(ph), alpha, w(kvo + NW * vooff(m, mu, ka)), dnV(ph), &
                        w(dt12 + NW * (t12off(kj, ph) + (otri(pq, m) - otoff(kj)) * dnQ(be))), dnJ(kj) * dnQ(be), &
                        one, w(kmx + NW * o), dnQ(be))
      end do
      ! P(kl) sum_m Wr(m, d, e, k) t(ml, bc)
      do nb = 1, 2
        if (nb == 1) then
          a = k
          b = l
          alpha = -1
        else
          a = l
          b = k
          alpha = 1
        end if
        c = mu + ocls(a) - ka
        if (wroff(a, mu, ka) < 0) cycle
        call N(zc_gemm)(0_4, 0_4, dnV(mu) * dnV(ka), dnQ(be), dnO(c), alpha, w(kwrb + NW * wroff(a, mu, ka)), dnO(c), &
                        w(et2p + NW * ((b * no + ocoff(c)) * nvp + vpoff(be))), nvp, one, w(kmx + NW * o), dnQ(be))
      end do
      ! 1/2 sum_f Wpp(bc, ef) t(kl, fd) as [(bc), e][d]
      ph = be - mu
      if (ph >= 0 .and. ph < ncls) then
        if (wboff(be, mu) >= 0) then
          call N(zc_gemm)(0_4, 0_4, dnQ(be) * dnV(mu), dnV(ka), dnV(ph), hw, w(dwb + NW * wboff(be, mu)), dnV(ph), &
                          w(jt2 + NW * (((k * no + l) * nv + vcoff(ph)) * nv + vcoff(ka))), nv, one, &
                          w(kmx + NW * (mxsize(kp) + o)), dnV(ka))
        end if
      end if
    end do
    ! the two sets reduced over (d | bc) into the rows e of Xv
    do mu = 0, ncls - 1
      if (r13off(pq, mu) < 0 .or. dnV(mu) == 0) cycle
      anyb = .false.
      do ka = 0, ncls - 1
        if (mxoff(kp, mu, ka) >= 0) anyb = .true.
      end do
      if (.not. anyb) cycle
      kk = kp + mu
      nk = dnK(kk)
      ne = dnV(mu)
      allocate(row(0:2*ne-1), sg(0:2*ne-1), xoff(0:ncls-1, 0:2*ne-1), s1(0:ncls-1, 0:2*ne-1), s2(0:ncls-1, 0:2*ne-1))
      xoff = -1
      s1 = 1
      s2 = 1
      sg = 1d0
      do e = 0, ne - 1
        row(e) = e
        row(ne + e) = e
        do ka = 0, ncls - 1
          o = mxoff(kp, mu, ka)
          if (o < 0) cycle
          be = kp + mu - ka
          xoff(ka, e) = o + e * dnV(ka) * dnQ(be)
          s1(ka, e) = dnQ(be)
          xoff(ka, ne + e) = mxsize(kp) + o + e * dnV(ka)
          s2(ka, ne + e) = ne * dnV(ka)
        end do
      end do
      call N(xzdx12)(w, 1, kb + NW * (r13off(pq, mu) + dnO(mu) * nk), nk, kk, 2 * ne, row, sg, xoff, s1, s2, kmx)
      deallocate(row, sg, xoff, s1, s2)
    end do
    call write_pair()
  end do
  deallocate(blk)

contains

  ! the chunk of the pair pq to the file XZDXV (dxvfile)
  subroutine write_pair()
    if (.not. dxvfile .or. dxv2) return
    write(81, pos=1 + 8 * NW * (r13p(pq) - r13p(dq0)), iostat=ios) w(dr13:dr13 + NW * (r13p(pq + 1) - r13p(pq)) - 1)
    if (ios /= 0) then
      write(6, '(/,a,i8,a)') ' @XZCCSDT-F, writing the file XZDXV of Xv failed (iostat ', ios, '): disk full?'
      call xzerrex
    end if
  end subroutine
end subroutine
