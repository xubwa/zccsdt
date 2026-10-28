!
! (Q) in the sectors of the symmetry labels (type-generic): the scheduler of
! src/zccsdt/cc/quadruples_sym.py with the kernels zc_q_sector_energy,
! zc_rows_gemm and zc_xv_sector of csrc/tnpack.c, on the amplitudes and
! integrals as the block code has them - T3 in the blocks of the groups of
! triples (its rows are read where they are: no dense T3 is made), <vv||vv>
! in its diagonal pair blocks.
!
! An occupied quadruple of total class M has r and z only on the virtual
! quadruples of total M, and each of the five products of xzq.inc falls into
! blocks:
!   X[(ab)][(cd)]   one block per group q of (ab): the pairs of group q times
!                   those of group M - q; of the stacked rows of a split only
!                   the summed indices of the class that q leaves
!   X[a][(bcd)]     one block per class p of a: the virtual spinors of class p
!                   times the triples of group M - p
! One kernel walks through the virtual quadruples of the sector, sums the
! splits of each from the blocks and adds the energies.  Xv(e; xl; bcd) for
! the e of one class has the triples of one group only: per pair (x, l) one
! block per class of e, one after the other.
!

! ---------------------------------------------------------------------------
! Xv for the pair (x, l): xv = its blocks, the classes of e in their order.
! m3: work (the products of one block), wb: work (a block of <bc||ef>)
subroutine N(xzqsxv)(w, x, l, tri, xv, m3, wb)
  use xzdata
  use xzkern
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: x, l
  integer(4), intent(in) :: tri(0:2, 0:nvt-1)
  ZT :: xv(0:*), m3(0:*), wb(*)
  integer(8) :: kxl, ke, t, ne, nt, q2, kf, kd, nf, nd, nq, n, xoff
  integer(8) :: off(0:npg-1), ndd(0:npg-1)
  ZT :: one, zero
  one = 1
  zero = 0
  kxl = xzw(ocls(x) + ocls(l))
  xoff = 0
  do ke = 0, ncls - 1
    t = xzw(ke + kxl)
    if (t >= ntg) cycle
    ne = vcoff(ke + 1) - vcoff(ke)
    nt = vtoff(t + 1) - vtoff(t)
    if (ne == 0 .or. nt == 0) cycle
    off = -1
    ndd = 0
    n = 0
    do q2 = 0, npg - 1
      kf = xzw(q2 - ke)
      if (kf < 0 .or. kf >= ncls) cycle
      kd = xzw(kxl - kf)
      if (kd < 0 .or. kd >= ncls) cycle
      nf = vcoff(kf + 1) - vcoff(kf)
      nd = vcoff(kd + 1) - vcoff(kd)
      nq = vpoff(q2 + 1) - vpoff(q2)
      if (nf == 0 .or. nd == 0 .or. nq == 0) cycle
      off(q2) = n
      ndd(q2) = nd
      ! <bc||ef> for (bc) of group q2, e of class ke, f of class kf, times t2[x][l][f][d]
      call N(xzqswb)(nv, nq, ne, nf, vcoff(ke), vcoff(kf), vpoff(q2), w(jvpidx), w(jvvvv + NW * vvoff(q2)), wb)
      call N(zc_gemm)(0_4, 0_4, nq * ne, nd, nf, one, wb, nf, &
                      w(jt2 + NW * (((x * no + l) * nv + vcoff(kf)) * nv + vcoff(kd))), nv, zero, m3(n), nd)
      n = n + nq * ne * nd
    end do
    call N(zc_xv_sector)(int(nv, 4), w(jqkv), int(mgrp, 4), tri(0, vtoff(t)), nt, int(ne, 4), 0_4, &
                         int(npg, 4), off, ndd, w(jqposv), w(jqposq), m3, xv(xoff))
    xoff = xoff + ne * nt
  end do
end subroutine

! wb[(bc)][e][f] = <bc||ef> from the block vv[(bc)][(ef)] of a group of pairs
subroutine N(xzqswb)(nv, nq, ne, nf, e0, f0, qlo, vpidx, vv, wb)
  implicit none
  integer(8), intent(in) :: nv, nq, ne, nf, e0, f0, qlo
  integer(8), intent(in) :: vpidx(0:nv-1, 0:nv-1)              ! [p][q]
  ZT, intent(in) :: vv(0:nq-1, 0:nq-1)                         ! [(bc)][(ef)]
  ZT, intent(out) :: wb(0:nf-1, 0:ne-1, 0:nq-1)                ! [(bc)][e][f]
  integer(8) :: b, e, f, ge, gf
  !$omp parallel do private(b, e, f, ge, gf)
  do b = 0, nq - 1
    do e = 0, ne - 1
      ge = e0 + e
      do f = 0, nf - 1
        gf = f0 + f
        if (ge < gf) then
          wb(f, e, b) = vv(vpidx(gf, ge) - qlo, b)
        else if (ge > gf) then
          wb(f, e, b) = -vv(vpidx(ge, gf) - qlo, b)
        else
          wb(f, e, b) = 0
        end if
      end do
    end do
  end do
end subroutine

! the first element of the block of the class ke of e in Xv of a pair of total class kxl
integer(8) function N(xzqsxvoff)(kxl, ke)
  use xzdata
  implicit none
  integer(8), intent(in) :: kxl, ke
  integer(8) :: k, t
  N(xzqsxvoff) = 0
  do k = 0, ke - 1
    t = xzw(k + kxl)
    if (t >= ntg) cycle
    N(xzqsxvoff) = N(xzqsxvoff) + (vcoff(k + 1) - vcoff(k)) * (vtoff(t + 1) - vtoff(t))
  end do
end function

! ---------------------------------------------------------------------------
! One occupied quadruple i<j<k<l: e(1), e(2) = its parts of E[Q] and of
! E(Q) - E[Q].
!   b1, b2, b3   the blocks [(ab)][(cd)] of [V,T3], T2 T2 and z
!   c1, c2       the blocks [a][(bcd)] of [V,T3] and T2 T2
!   la .. rz     stacked operands; lc, rows, sgs: those of the product with rows of T3
! left: (Q)_Lambda (xzq.inc) - b3 then has the blocks of z2 (l2 with <kl||cd>),
! b4 and c3 those of z3 (the products of [V,T3] with l3 and the integrals the
! other way round), l2pp is l2[P][Q].
subroutine N(xzqsquad)(w, i, j, k, l, otri, osgn, opidx, opgrp, t2pp, t2p, t2px, xo, b1, b2, b3, c1, c2, &
                       la, ra, lb, rb, lz, rz, lc, rows, sgs, xv, slot, left, l2pp, b4, c3, e, tq)
  use xzdata
  use xzkern
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: i, j, k, l
  integer(8), intent(in) :: otri(0:npo-1, 0:no-1), opidx(0:no-1, 0:no-1), opgrp(0:npo-1)
  integer(1), intent(in) :: osgn(0:npo-1, 0:no-1)
  ZT, intent(in) :: t2pp(0:nvp-1, 0:npo-1), t2p(0:nvp-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: t2px(0:nv-1, 0:nv-1, 0:npo-1)              ! [P][e][a]
  ZT, intent(in) :: xo(0:nvp-1, 0:no-1, 0:nto-1)
  ZT :: b1(0:*), b2(0:*), b3(0:*), c1(0:*), c2(0:*)
  ZT :: la(0:*), ra(0:*), lb(0:*), rb(0:*), lz(0:*), rz(0:*), lc(0:*)
  integer(8) :: rows(0:*)
  real(8) :: sgs(0:*)
  ZT, intent(in) :: xv(0:qxvmax-1, 0:*)
  integer(8), intent(in) :: slot(0:*)
  logical, intent(in) :: left
  ZT, intent(in) :: l2pp(0:nvp-1, 0:npo-1)
  ZT :: b4(0:*), c3(0:*)
  real(8), intent(out) :: e(2)
  real(8), intent(inout) :: tq(7)
  real(8) :: tl0
  integer(8) :: q(0:3), s, x, m, mt, qq, r, nr, nc, na, nb, nz, ke, ne, nm, n22, n13, p, km, kxl, n
  integer(8) :: jt(0:3), p22(0:5), q22(0:5), xfirst(0:5)
  integer(8) :: off22(0:npg-1), nc22(0:npg-1), off13(0:ncls-1), nc13(0:ncls-1)
  real(8) :: sg, s13(0:3), s22(0:5), t0, eo
  logical :: first
  integer(8), parameter :: f1(0:5) = [0, 0, 0, 1, 1, 2], f2(0:5) = [1, 2, 3, 2, 3, 3]
  integer(8), parameter :: r1(0:5) = [2, 1, 1, 0, 0, 0], r2(0:5) = [3, 3, 2, 3, 2, 1]
  integer(8), parameter :: last(3) = [0, 1, 3]
  integer(8), external :: N(xzqsxvoff)
  ZT :: one, zero, alpha, beta
  one = 1
  zero = 0
  q = [i, j, k, l]
  s13 = [1d0, -1d0, 1d0, -1d0]
  s22 = [1d0, -1d0, 1d0, 1d0, -1d0, 1d0]
  eo = w(jeps + i) + w(jeps + j) + w(jeps + k) + w(jeps + l)
  mt = xzw(ocls(i) + ocls(j) + ocls(k) + ocls(l))
  ! the occupied triples of the splits (i|jkl) and the pairs of the splits (ij|kl)
  jt(0) = otri(opidx(q(3), q(2)), q(1))
  jt(1) = otri(opidx(q(3), q(2)), q(0))
  jt(2) = otri(opidx(q(3), q(1)), q(0))
  jt(3) = otri(opidx(q(2), q(1)), q(0))
  do s = 0, 5
    p22(s) = opidx(q(f2(s)), q(f1(s)))
    q22(s) = opidx(q(r2(s)), q(r1(s)))
    xfirst(s) = q(r1(s))
  end do

  ! ---- the blocks on two pairs
  t0 = omp_get_wtime()
  off22 = -1
  nc22 = 0
  n22 = 0
  do qq = 0, npg - 1
    r = xzw(mt - qq)
    if (r < 0 .or. r >= npg) cycle
    nr = vpoff(qq + 1) - vpoff(qq)
    nc = vpoff(r + 1) - vpoff(r)
    if (nr == 0 .or. nc == 0) cycle
    off22(qq) = n22
    nc22(qq) = nc
    na = 0
    nb = 0
    do s = 0, 3
      x = q(s)
      ke = xzw(qq - ocls(x))
      if (ke < 0 .or. ke >= ncls) cycle
      ne = vcoff(ke + 1) - vcoff(ke)
      nm = ocoff(ke + 1) - ocoff(ke)
      ! - <ab||ex> t(J; e, cd): la[e][(ab)] = sign <xe||ab>*
      if (ne > 0) call N(xzqsab)(nv, nvp, nr, nc, ne, s13(s), .true., vcoff(ke), vpoff(qq), vpoff(r), &
        w(jovvq + NW * nv * nvp * x), w(jt3), w(jobase + jt(s)), w(jvci), w(jvsgn), la(na * nr), ra(na * nc))
      na = na + ne
      ! t(x m, ab) Xo(m; J; cd)
      do m = 0, nm - 1
        lb((nb + m) * nr:(nb + m) * nr + nr - 1) = s13(s) * t2p(vpoff(qq):vpoff(qq) + nr - 1, ocoff(ke) + m, x)
        rb((nb + m) * nc:(nb + m) * nc + nc - 1) = xo(vpoff(r):vpoff(r) + nc - 1, ocoff(ke) + m, jt(s))
      end do
      nb = nb + nm
    end do
    ! t(ij, ab) <cd||kl>
    nz = 0
    do s = 0, 5
      if (opgrp(p22(s)) /= qq) cycle
      if (left) then
        lz(nz * nr:nz * nr + nr - 1) = s22(s) * l2pp(vpoff(qq):vpoff(qq) + nr - 1, p22(s))
        call N(xzqcopy)(nc, 1d0, .false., w(jgpp + NW * (nvp * q22(s) + vpoff(r))), rz(nz * nc))
      else
        lz(nz * nr:nz * nr + nr - 1) = s22(s) * t2pp(vpoff(qq):vpoff(qq) + nr - 1, p22(s))
        call N(xzqconj)(nc, w(jgpp + NW * (nvp * q22(s) + vpoff(r))), rz(nz * nc))
      end if
      nz = nz + 1
    end do
    if (na > 0) then
      call N(zc_gemm)(1_4, 0_4, nr, nc, na, one, la, nr, ra, nc, zero, b1(n22), nc)
    else
      call zc_tn_zero(b1(n22), 8 * NW * nr * nc)
    end if
    if (nb > 0) then
      call N(zc_gemm)(1_4, 0_4, nr, nc, nb, one, lb, nr, rb, nc, zero, b2(n22), nc)
    else
      call zc_tn_zero(b2(n22), 8 * NW * nr * nc)
    end if
    if (nz > 0) then
      call N(zc_gemm)(1_4, 0_4, nr, nc, nz, one, lz, nr, rz, nc, zero, b3(n22), nc)
    else
      call zc_tn_zero(b3(n22), 8 * NW * nr * nc)
    end if
    if (left) then
      ! <xe||ab> l(J; e, cd), in the operands of [V,T3]
      tl0 = omp_get_wtime()
      na = 0
      do s = 0, 3
        x = q(s)
        ke = xzw(qq - ocls(x))
        if (ke < 0 .or. ke >= ncls) cycle
        ne = vcoff(ke + 1) - vcoff(ke)
        if (ne > 0) call N(xzqsab)(nv, nvp, nr, nc, ne, s13(s), .false., vcoff(ke), vpoff(qq), vpoff(r), &
          w(jovvq + NW * nv * nvp * x), w(jl3), w(jobase + jt(s)), w(jvci), w(jvsgn), la(na * nr), ra(na * nc))
        na = na + ne
      end do
      if (na > 0) then
        call N(zc_gemm)(1_4, 0_4, nr, nc, na, one, la, nr, ra, nc, zero, b4(n22), nc)
      else
        call zc_tn_zero(b4(n22), 8 * NW * nr * nc)
      end if
      tq(7) = tq(7) + omp_get_wtime() - tl0
    end if
    n22 = n22 + nr * nc
  end do
  tq(3) = tq(3) + omp_get_wtime() - t0

  ! ---- the blocks on one index and a triple
  t0 = omp_get_wtime()
  off13 = -1
  nc13 = 0
  n13 = 0
  do p = 0, ncls - 1
    r = xzw(mt - p)
    if (r < 0 .or. r >= ntg) cycle
    nr = vcoff(p + 1) - vcoff(p)
    nc = vtoff(r + 1) - vtoff(r)
    if (nr == 0 .or. nc == 0) cycle
    off13(p) = n13
    nc13(p) = nc
    ! <ma||ij> t(m kl; bcd): rows of T3 where they are
    n = 0
    do s = 0, 5
      km = xzw(opgrp(p22(s)) - p)
      if (km < 0 .or. km >= ncls) cycle
      do m = ocoff(km), ocoff(km + 1) - 1
        if (osgn(q22(s), m) == 0) cycle
        call N(xzqlc)(nr, s22(s), w(jooov + NW * (nv * (m + no * (q(f2(s)) + no * q(f1(s)))) + vcoff(p))), &
                      lc(n * nr))
        rows(n) = otri(q22(s), m) - otoff(r)
        sgs(n) = dble(osgn(q22(s), m))
        n = n + 1
      end do
    end do
    if (n > 0) then
      call N(zc_rows_gemm)(w(jt3 + NW * boff(r)), nc, rows, sgs, n, w(jrows), nc, lc, nr, c1(n13))
    else
      call zc_tn_zero(c1(n13), 8 * NW * nr * nc)
    end if
    if (left) then
      ! <ij||ma> l(m kl; bcd): the same rows, of l3
      tl0 = omp_get_wtime()
      n = 0
      do s = 0, 5
        km = xzw(opgrp(p22(s)) - p)
        if (km < 0 .or. km >= ncls) cycle
        do m = ocoff(km), ocoff(km + 1) - 1
          if (osgn(q22(s), m) == 0) cycle
          call N(xzqcopy)(nr, s22(s), .false., &
                          w(jooov + NW * (nv * (m + no * (q(f2(s)) + no * q(f1(s)))) + vcoff(p))), lc(n * nr))
          n = n + 1
        end do
      end do
      if (n > 0) then
        call N(zc_rows_gemm)(w(jl3 + NW * boff(r)), nc, rows, sgs, n, w(jrows), nc, lc, nr, c3(n13))
      else
        call zc_tn_zero(c3(n13), 8 * NW * nr * nc)
      end if
      tq(7) = tq(7) + omp_get_wtime() - tl0
    end if
    ! 2 t(ij, ae) Xv(e; kl; bcd), l in the second pair
    first = .true.
    do m = 1, 3
      s = last(m)
      ke = xzw(opgrp(p22(s)) - p)
      if (ke < 0 .or. ke >= ncls) cycle
      ne = vcoff(ke + 1) - vcoff(ke)
      if (ne == 0) cycle
      kxl = xzw(ocls(xfirst(s)) + ocls(l))
      alpha = 2d0 * s22(s)
      beta = one
      if (first) beta = zero
      call N(zc_gemm)(1_4, 0_4, nr, nc, ne, alpha, t2px(vcoff(p), vcoff(ke), p22(s)), nv, &
                      xv(N(xzqsxvoff)(kxl, ke), slot(xfirst(s))), nc, beta, c2(n13), nc)
      first = .false.
    end do
    if (first) call zc_tn_zero(c2(n13), 8 * NW * nr * nc)
    n13 = n13 + nr * nc
  end do
  tq(4) = tq(4) + omp_get_wtime() - t0

  ! ---- the sum over the virtual quadruples of the sector
  t0 = omp_get_wtime()
  e = 0d0
  if (left) then
    call N(zc_ql_sector_energy)(int(nv, 4), w(jqkv), int(mgrp, 4), int(mt, 4), w(jeps + no), eo, 0_4, int(ncls, 4), &
                                w(jqlid), w(jqlst), w(jqlmem), 0_4, int(npg, 4), off22, nc22, off13, nc13, &
                                w(jqposv), w(jqposq), w(jqpos3), b1, b2, b3, b4, c1, c2, c3, e)
  else
    call N(zc_q_sector_energy)(int(nv, 4), w(jqkv), int(mgrp, 4), int(mt, 4), w(jeps + no), eo, 0_4, int(ncls, 4), &
                               w(jqlid), w(jqlst), w(jqlmem), 0_4, int(npg, 4), off22, nc22, off13, nc13, &
                               w(jqposv), w(jqposq), w(jqpos3), b1, b2, b3, c1, c2, e)
  end if
  tq(5) = tq(5) + omp_get_wtime() - t0
end subroutine

! The operands of the [V,T3] product on two pairs for one split (x | J), the
! e of one class: la[e][(ab)] = sg <xe||ab>* for (ab) of a group, and
! ra[e][(cd)] = t(J; e, cd) for (cd) of a group, from the row of J in its
! block of T3.  cj = .false.: la = sg <xe||ab> (the left side, with l3 for t3).
subroutine N(xzqsab)(nv, nvp, nr, nc, ne, sg, cj, e0, alo, clo, gq, t3, obase, vci, vsgn, la, ra)
  implicit none
  integer(8), intent(in) :: nv, nvp, nr, nc, ne, e0, alo, clo
  real(8), intent(in) :: sg
  logical, intent(in) :: cj
  ZT, intent(in) :: gq(0:nvp-1, 0:nv-1)                        ! [e][Q] of the occupied x
  ZT, intent(in) :: t3(0:*)
  integer(8), intent(in) :: obase(0:0)                         ! of the triple J
  integer(4), intent(in) :: vci(0:nvp-1, 0:nv-1)               ! [e][Q]
  integer(1), intent(in) :: vsgn(0:nvp-1, 0:nv-1)
  ZT, intent(out) :: la(0:nr-1, 0:ne-1), ra(0:nc-1, 0:ne-1)
  integer(8) :: e, c
  !$omp parallel do private(e, c)
  do e = 0, ne - 1
    if (cj) then
      la(:, e) = sg * CONJ(gq(alo:alo + nr - 1, e0 + e))
    else
      la(:, e) = sg * gq(alo:alo + nr - 1, e0 + e)
    end if
    do c = 0, nc - 1
      if (vsgn(clo + c, e0 + e) == 0) then
        ra(c, e) = 0
      else
        ra(c, e) = dble(vsgn(clo + c, e0 + e)) * t3(obase(0) + vci(clo + c, e0 + e))
      end if
    end do
  end do
end subroutine
