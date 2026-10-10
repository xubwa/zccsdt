!
! The passes of a Lambda iteration over Lambda_3 in the blocks of the
! symmetry labels (type-generic): the scheduler of BlockL3 of
! src/zccsdt/cc/lambda_blk.py with the kernels of csrc/t3pack.c.  l3 has the
! blocks of T3 (xzblk.inc: classes, groups, the layout).  Pass 2 is that of
! the T3 equation (xzbpass2 with two parts of the summed index and the outer
! products); the operator on occupied triples and the Jacobi step are xzbocc
! and xzbupd.
!

! ---------------------------------------------------------------------------
! Pass 1: for a batch of pairs P of group gP and a group gQ of Q, on the pairs
! D = (d, m) with s(d) - s(m) = gP - gQ, with L[P][D][Q] and T[P][D][Q] the
! half-expanded l3 and T3:
!   dr[(a, i)][(g, o)] += sum_{P,Q} L[P][(a, i)][Q] T[P][(g, o)][Q]
!   d1[(g, o)]         += sum_{P,Q} T[P][(g, o)][Q] l2pp[P][Q]       (gP = gQ)
! and per class of d and of m the two stacked products of pass 1 of the T3
! equation with L (xzbp1xy):
!   xacc[e|j][m][Q] += sum_{P,d} gpd[e|j][P][d] L,   yall[P][d][j|e] += sum_{m,Q} L gmq[m][Q][j|e]
! xacc, dr and d1 are added to; yall is zeroed here.  A batch is gathered
! for all gQ first; then the densities of one (gQ, P) and the stacked
! products of one (gQ, class of d) are the tasks of two parallel loops, with
! the BLAS sequential inside.
subroutine N(xzlbpass1)(w, kl3, kt3, gpd, gmq, l2pp, xacc, yall, dr, d1)
  use xzdata
  use xzmem
  use xzkern
  use xzdist
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kl3, kt3
  ZT :: gpd(*), gmq(*), xacc(*), yall(*), dr(*), d1(*)
  ZT, intent(in) :: l2pp(0:nvp-1, 0:npo-1)
  integer(8) :: gp, gq, p0, nb, delta, nr, nq, qlo, sd, sm, sy, nd, nm, ne, nj, r0, k
  integer(8) :: m0, mk, kdd, kd0, kdw, nth, nrmax, ig, ngq, it, ntask, me
  integer(8) :: lst(0:npg-1), gkl(0:npg-1), gkt(0:npg-1)
  integer(8), dimension(npg * ncls) :: tgq, tsd, tr0, tks, tkg, tkx, tkgy, tky, ggq, gsd, gr0
  integer(8) :: ngath
  real(8) :: cost(npg * max(ncls, nbp))
  logical :: big(npg * max(ncls, nbp))
  integer :: phase
  ZT :: one, zero
  one = 1
  zero = 0
  nth = omp_get_max_threads()
  nrmax = 0
  do delta = 1 - ncls, ncls - 1
    nrmax = max(nrmax, ndel(delta))
  end do
  m0 = xzmark()
  kdd = xzget(NW * nwd, 'densities of pass 1')
  kd0 = xzget(NW * ndel(0), 'density of L2 with T3')
  kdw = xzget(NW * nth * (nrmax * nrmax + nrmax), 'densities of pass 1, per thread')
  if (.not. xzdry) then
    call zc_vzero_d(NW * npo * nv * ny, yall)
    call zc_vzero_d(NW * nwd, w(kdd))
    call zc_vzero_d(NW * ndel(0), w(kd0))
  end if
  do gp = 0, npg - 1
    do p0 = opoff(gp), opoff(gp + 1) - 1, nbp
      nb = min(nbp, opoff(gp + 1) - p0)
      if (.not. xzdry) then
        if (.not. xzmine(p0)) cycle
      end if
      call xzbtic()
      mk = xzmark()
      ngq = 0
      ntask = 0
      ngath = 0
      do gq = 0, npg - 1
        delta = gp - gq
        if (abs(delta) >= ncls) cycle
        nr = ndel(delta)
        nq = vpoff(gq + 1) - vpoff(gq)
        if (nr == 0 .or. nq == 0) cycle
        lst(ngq) = gq
        ngq = ngq + 1
        gkl(gq) = xzget(NW * nb * nr * nq, 'pass-1 batch of L3')
        gkt(gq) = xzget(NW * nb * nr * nq, 'pass-1 batch of T3')
        r0 = 0
        do sd = 0, ncls - 1
          sm = sd - delta
          if (sm < 0 .or. sm >= ncls) cycle
          nd = vcoff(sd + 1) - vcoff(sd)
          nm = ocoff(sm + 1) - ocoff(sm)
          if (nd > 0 .and. nm > 0) then
            ngath = ngath + 1
            ggq(ngath) = gq
            gsd(ngath) = sd
            gr0(ngath) = r0
          end if
          sy = gp - sd
          if (nd > 0 .and. nm > 0 .and. sy >= 0 .and. sy < ncls) then
            ne = vcoff(sy + 1) - vcoff(sy)
            nj = ocoff(sy + 1) - ocoff(sy)
            if (ne + nj > 0) then
              ntask = ntask + 1
              tgq(ntask) = gq
              tsd(ntask) = sd
              tr0(ntask) = r0
              tks(ntask) = xzget(NW * nb * nd * nm * nq, 'pass-1 batch of L3, a class')
              tkg(ntask) = xzget(NW * (ne + nj) * nb * nd, 'pass-1 operand')
              tkx(ntask) = xzget(NW * (ne + nj) * nm * nq, 'pass-1 product')
              tkgy(ntask) = xzget(NW * nm * nq * (ne + nj), 'pass-1 operand')
              tky(ntask) = xzget(NW * nb * nd * (ne + nj), 'pass-1 product')
            end if
          end if
          r0 = r0 + nd * nm
        end do
      end do
      if (.not. xzdry .and. ngq > 0) then
        call xztic()
        ! the gathers: one task per (gQ, class of d)
        do it = 1, ngath
          sm = gsd(it) - (gp - ggq(it))
          cost(it) = dble(vcoff(gsd(it) + 1) - vcoff(gsd(it))) * (ocoff(sm + 1) - ocoff(sm)) &
                     * (vpoff(ggq(it) + 1) - vpoff(ggq(it)))
        end do
        call xzbig(ngath, cost, big)
        do phase = 1, 2
        !$omp parallel do schedule(dynamic) if(phase == 2) private(it, gq, sd, sm, delta, nr, qlo, nq, nd, nm)
        do it = 1, ngath
          if (big(it) .neqv. (phase == 1)) cycle
          gq = ggq(it)
          sd = gsd(it)
          delta = gp - gq
          sm = sd - delta
          nr = ndel(delta)
          qlo = vpoff(gq)
          nq = vpoff(gq + 1) - qlo
          nd = vcoff(sd + 1) - vcoff(sd)
          nm = ocoff(sm + 1) - ocoff(sm)
          call N(zc_bgather2)(w(kl3), w(jobase), w(josig), w(jotri), w(josgn), npo, w(jvci), &
            w(jvsgn), nvp, nb, w(jrows + p0), nd, w(jrows + vcoff(sd)), nm, w(jrows + ocoff(sm)), &
            qlo, nq, nr * nq, w(gkl(gq) + NW * gr0(it) * nq))
          call N(zc_bgather2)(w(kt3), w(jobase), w(josig), w(jotri), w(josgn), npo, w(jvci), &
            w(jvsgn), nvp, nb, w(jrows + p0), nd, w(jrows + vcoff(sd)), nm, w(jrows + ocoff(sm)), &
            qlo, nq, nr * nq, w(gkt(gq) + NW * gr0(it) * nq))
        end do
        end do
        call xztoc(tcur, 'pass 1: gather')
        ! the densities: one task per (gQ, P), each product in the buffer of
        ! its thread and added to the block of gP - gQ.  (The lock is not what
        ! limits this loop: tasks that own a range of rows of the block and add
        ! to it without a lock were measured 1.7 to 2.3 times slower on 8 to 48
        ! threads - thin products that read the batch of T3 once per task;
        ! docs/native.md, section 13.)
        do ig = 0, ngq - 1
          gq = lst(ig)
          cost(ig * nb + 1:ig * nb + nb) = dble(ndel(gp - gq)) ** 2 * (vpoff(gq + 1) - vpoff(gq))
        end do
        call xzbig(ngq * nb, cost, big)
        do phase = 1, 2
        !$omp parallel do collapse(2) schedule(dynamic) if(phase == 2) private(ig, k, gq, delta, nr, qlo, nq, me)
        do ig = 0, ngq - 1
          do k = 0, nb - 1
            if (big(ig * nb + k + 1) .neqv. (phase == 1)) cycle
            gq = lst(ig)
            delta = gp - gq
            nr = ndel(delta)
            qlo = vpoff(gq)
            nq = vpoff(gq + 1) - qlo
            me = kdw + NW * omp_get_thread_num() * (nrmax * nrmax + nrmax)
            call N(zc_gemm)(0_4, 1_4, nr, nr, nq, one, w(gkl(gq) + NW * k * nr * nq), nq, &
                            w(gkt(gq) + NW * k * nr * nq), nq, zero, w(me), nr)
            if (delta == 0) call N(zc_gemm)(0_4, 0_4, nr, 1_8, nq, one, w(gkt(gq) + NW * k * nr * nq), nq, &
                                            l2pp(qlo, p0 + k), 1_8, zero, w(me + NW * nrmax * nrmax), 1_8)
            !$omp critical (xzlbdens)
            call N(xzlbacc)(nr * nr, w(me), w(kdd + NW * wdoff(delta)))
            if (delta == 0) call N(xzlbacc)(nr, w(me + NW * nrmax * nrmax), w(kd0))
            !$omp end critical (xzlbdens)
          end do
        end do
        end do
        call xztoc(tcur, 'pass 1: density')
        ! the stacked products, per gQ and class of d (and of m)
        do it = 1, ntask
          sm = tsd(it) - (gp - tgq(it))
          sy = gp - tsd(it)
          cost(it) = dble(vcoff(tsd(it) + 1) - vcoff(tsd(it))) * (ocoff(sm + 1) - ocoff(sm)) &
                     * (vpoff(tgq(it) + 1) - vpoff(tgq(it))) &
                     * (8 + vcoff(sy + 1) - vcoff(sy) + ocoff(sy + 1) - ocoff(sy))
        end do
        call xzbig(ntask, cost, big)
        do phase = 1, 2
        !$omp parallel do schedule(dynamic) if(phase == 2) private(it, gq, sd, sm, sy, delta, nr, qlo, nq, nd, nm, ne, nj)
        do it = 1, ntask
          if (big(it) .neqv. (phase == 1)) cycle
          gq = tgq(it)
          sd = tsd(it)
          delta = gp - gq
          sm = sd - delta
          sy = gp - sd
          nr = ndel(delta)
          qlo = vpoff(gq)
          nq = vpoff(gq + 1) - qlo
          nd = vcoff(sd + 1) - vcoff(sd)
          nm = ocoff(sm + 1) - ocoff(sm)
          ne = vcoff(sy + 1) - vcoff(sy)
          nj = ocoff(sy + 1) - ocoff(sy)
          ! L[(P, d)][(m, Q)] of these classes
          call N(xzlbsub)(nb, nr, nq, tr0(it), nd * nm, w(gkl(gq)), w(tks(it)))
          call N(xzbp1xy)(no, nv, npo, nvp, nb, nd, nm, nq, ne, nj, p0, vcoff(sd), ocoff(sm), qlo, &
                          vcoff(sy), ocoff(sy), gpd, gmq, w(tks(it)), w(tkg(it)), w(tkx(it)), w(tkgy(it)), &
                          w(tky(it)), xacc)
        end do
        end do
        do it = 1, ntask
          sd = tsd(it)
          sy = gp - sd
          call N(xzbp1yadd)(no, nv, npo, nb, vcoff(sd + 1) - vcoff(sd), vcoff(sy + 1) - vcoff(sy), &
                            ocoff(sy + 1) - ocoff(sy), p0, vcoff(sd), vcoff(sy), ocoff(sy), w(tky(it)), yall)
        end do
        call xztoc(tcur, 'pass 1: stacked GEMMs')
        call xzbtoc(p0)
      end if
      call xzrel(mk)
    end do
  end do
  if (.not. xzdry) call N(xzlbdr)(w(kdd), w(kd0), dr, d1)
  call xzrel(m0)
end subroutine

! y += x
subroutine N(xzlbacc)(n, x, y)
  implicit none
  integer(8), intent(in) :: n
  ZT, intent(in) :: x(n)
  ZT, intent(inout) :: y(n)
  y = y + x
end subroutine

! sub[P][r][Q] = l[P][r0 + r][Q] for r < n
subroutine N(xzlbsub)(nb, nr, nq, r0, n, l, sub)
  implicit none
  integer(8), intent(in) :: nb, nr, nq, r0, n
  ZT, intent(in) :: l(0:nq-1, 0:nr-1, 0:nb-1)
  ZT, intent(out) :: sub(0:nq-1, 0:n-1, 0:nb-1)
  integer(8) :: p
  !$omp parallel do
  do p = 0, nb - 1
    sub(:, :, p) = l(:, r0:r0 + n - 1, p)
  end do
end subroutine

! the blocks of the densities into dr[(a, i)][(g, o)] and d1[(g, o)]
subroutine N(xzlbdr)(dd, d0, dr, d1)
  use xzdata
  implicit none
  ZT, intent(in) :: dd(0:*), d0(0:*)
  ZT :: dr(0:no*nv-1, 0:no*nv-1), d1(0:no*nv-1)
  integer(8) :: idx(0:no*nv-1), delta, sa, si, a, i, n, r, c
  do delta = 1 - ncls, ncls - 1
    n = 0
    do sa = 0, ncls - 1
      si = sa - delta
      if (si < 0 .or. si >= ncls) cycle
      do a = vcoff(sa), vcoff(sa + 1) - 1
        do i = ocoff(si), ocoff(si + 1) - 1
          idx(n) = a * no + i
          n = n + 1
        end do
      end do
    end do
    do r = 0, n - 1
      do c = 0, n - 1
        dr(idx(c), idx(r)) = dr(idx(c), idx(r)) + dd(wdoff(delta) + r * n + c)
      end do
      if (delta == 0) d1(idx(r)) = d1(idx(r)) + d0(r)
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The ladder pass: for the triples I of a group g and the class sc of the
! free virtual index c, h = g - sc the group of the pair Q, with
! Lv[Q][I][c] = l3[I, Q, c]:
!   R[I][A]   += P(c/Q') sum_Q wpp[h][Q][Q'] Lv[Q][I][c]
!   r2q[P][Q] += sum_{I,e} xq[P][I][e] Lv[Q][I][e]        (P of the group h)
!   r1x[a][i] += sum_{I,Q} Lv[Q][I][a] x2[Q][I][i]        (i of the class sc)
subroutine N(xzlbladder)(w, kl3, kr3, xq, x2, r2q, r1x)
  use xzdata
  use xzmem
  use xzkern
  use xzdist
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kl3, kr3
  ZT, intent(in) :: xq(0:nv-1, 0:nto-1, 0:npo-1)               ! [P][I][e]
  ZT, intent(in) :: x2(0:no-1, 0:nto-1, 0:nvp-1)               ! [Q][I][i]
  ZT :: r2q(0:nvp-1, 0:npo-1), r1x(0:no-1, 0:nv-1)
  integer(8) :: g, sc, h, nc, nqh, qlo, i0, nb, mk, ktv, kxp, ka, plo, np, j0, nj
  real(8) :: t0
  ZT :: one, zero
  one = 1
  zero = 0
  do g = 0, ntg - 1
    if (vtoff(g + 1) == vtoff(g)) cycle
    do i0 = otoff(g), otoff(g + 1) - 1, nbt
      nb = min(nbt, otoff(g + 1) - i0)
      if (.not. xzdry) then
        if (.not. xzmine(i0)) cycle
      end if
      call xzbtic()
      do sc = 0, ncls - 1
        h = g - sc
        if (h < 0 .or. h >= npg) cycle
        nc = vcoff(sc + 1) - vcoff(sc)
        qlo = vpoff(h)
        nqh = vpoff(h + 1) - qlo
        if (nc == 0 .or. nqh == 0) cycle
        plo = opoff(h)
        np = opoff(h + 1) - plo
        j0 = ocoff(sc)
        nj = ocoff(sc + 1) - j0
        mk = xzmark()
        ktv = xzget(NW * nqh * nb * nc, 'ladder batch of L3')
        kxp = xzget(NW * nqh * nb * nc, 'ladder product')
        ka = xzget(NW * max(np * nb * nc, nqh * nb * nj), 'batch of Xq, X2')
        if (.not. xzdry) then
          call xztic()
          call N(zc_bgather_vir)(w(kl3), w(jobase), w(jvcol), w(josig), w(jvsig), w(jvtri), w(jvsgn), nvp, &
                                 nb, w(jrows + i0), qlo, nqh, vcoff(sc), nc, w(ktv))
          call xztoc(2, 'ladder: gather')
          call xztic()
          call N(zc_gemm)(1_4, 0_4, nqh, nb * nc, nqh, one, w(jvvvv + NW * vvoff(h)), nqh, w(ktv), nb * nc, &
                          zero, w(kxp), nb * nc)
          call xztoc(2, 'ladder: GEMM')
          call xztic()
          call N(zc_breduce_ppl_rows)(w(kr3), w(jobase), w(jvcol), nb, w(jrows + i0), nvt, vtoff(g), &
                                      vtoff(g + 1), w(jvfree), w(jvpair), vcls, sc, qlo, 0_8, nqh, &
                                      nc, vcoff(sc), w(kxp))
          if (np > 0) then
            call N(xzlbxq)(nv, nto, np, nb, nc, i0, vcoff(sc), xq(0, 0, plo), w(ka))
            call N(zc_gemm)(0_4, 1_4, np, nqh, nb * nc, one, w(ka), nb * nc, w(ktv), nb * nc, one, &
                            r2q(qlo, plo), nvp)
          end if
          if (nj > 0) then
            call N(xzlbx2)(no, nto, nqh, nb, nj, i0, j0, x2(0, 0, qlo), w(ka))
            call N(zc_gemm)(1_4, 0_4, nc, nj, nqh * nb, one, w(ktv), nc, w(ka), nj, one, &
                            r1x(j0, vcoff(sc)), no)
          end if
          call xztoc(2, 'ladder: reduction, small')
        end if
        call xzrel(mk)
      end do
      if (.not. xzdry) call xzbtoc(i0)
    end do
  end do
end subroutine

! a[P][I][c] = xq[P][i0 + I][c0 + c]
subroutine N(xzlbxq)(nv, nto, np, nb, nc, i0, c0, xq, a)
  implicit none
  integer(8), intent(in) :: nv, nto, np, nb, nc, i0, c0
  ZT, intent(in) :: xq(0:nv-1, 0:nto-1, 0:np-1)
  ZT, intent(out) :: a(0:nc-1, 0:nb-1, 0:np-1)
  integer(8) :: p
  !$omp parallel do
  do p = 0, np - 1
    a(:, :, p) = xq(c0:c0 + nc - 1, i0:i0 + nb - 1, p)
  end do
end subroutine

! a[Q][I][j] = x2[Q][i0 + I][j0 + j]
subroutine N(xzlbx2)(no, nto, nq, nb, nj, i0, j0, x2, a)
  implicit none
  integer(8), intent(in) :: no, nto, nq, nb, nj, i0, j0
  ZT, intent(in) :: x2(0:no-1, 0:nto-1, 0:nq-1)
  ZT, intent(out) :: a(0:nj-1, 0:nb-1, 0:nq-1)
  integer(8) :: q
  !$omp parallel do
  do q = 0, nq - 1
    a(:, :, q) = x2(j0:j0 + nj - 1, i0:i0 + nb - 1, q)
  end do
end subroutine

! ---------------------------------------------------------------------------
! The contractions of T3 that the ladder pass meets (once), with
! Tv[Q][I][f] = t3[I, f, Q] for I of group g, f of class sc, Q of group
! h = g - sc:
!   xq[P][I][f]     += sum_Q gpp[P][Q] Tv[Q][I][f]             (P of group h)
!   x2[(e<f)][I][i]  = A(ef) sum_Q vq[e][i][Q] Tv[Q][I][f]     (s(e) + s(i) = h)
! xq and x2 are zero on entry.
subroutine N(xzlbcx)(w, kt3, gpp, vq, xq, x2)
  use xzdata
  use xzmem
  use xzkern
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kt3
  ZT, intent(in) :: gpp(0:nvp-1, 0:npo-1)
  ZT, intent(in) :: vq(0:nvp-1, 0:no-1, 0:nv-1)                ! [e][i][Q]
  ZT :: xq(0:nv-1, 0:nto-1, 0:npo-1), x2(0:no-1, 0:nto-1, 0:nvp-1)
  integer(8) :: g, sc, h, nc, nqh, qlo, i0, nb, mk, ms, ktv, kz, kv, plo, np, se, si, ne, nj
  ZT :: one, zero
  one = 1
  zero = 0
  do g = 0, ntg - 1
    if (vtoff(g + 1) == vtoff(g)) cycle
    do i0 = otoff(g), otoff(g + 1) - 1, nbt
      nb = min(nbt, otoff(g + 1) - i0)
      do sc = 0, ncls - 1
        h = g - sc
        if (h < 0 .or. h >= npg) cycle
        nc = vcoff(sc + 1) - vcoff(sc)
        qlo = vpoff(h)
        nqh = vpoff(h + 1) - qlo
        if (nc == 0 .or. nqh == 0) cycle
        plo = opoff(h)
        np = opoff(h + 1) - plo
        mk = xzmark()
        ktv = xzget(NW * nqh * nb * nc, 'gathered batch of T3')
        kz = xzget(NW * np * nb * nc, 'product of Xq')
        if (.not. xzdry) then
          call N(zc_bgather_vir)(w(kt3), w(jobase), w(jvcol), w(josig), w(jvsig), w(jvtri), w(jvsgn), nvp, &
                                 nb, w(jrows + i0), qlo, nqh, vcoff(sc), nc, w(ktv))
          if (np > 0) then
            call N(zc_gemm)(0_4, 0_4, np, nb * nc, nqh, one, gpp(qlo, plo), nvp, w(ktv), nb * nc, zero, &
                            w(kz), nb * nc)
            call N(xzlbxqadd)(nv, nto, np, nb, nc, i0, vcoff(sc), w(kz), xq(0, 0, plo))
          end if
        end if
        do se = 0, ncls - 1
          si = h - se
          if (si < 0 .or. si >= ncls) cycle
          ne = vcoff(se + 1) - vcoff(se)
          nj = ocoff(si + 1) - ocoff(si)
          if (ne == 0 .or. nj == 0) cycle
          ms = xzmark()
          kv = xzget(NW * ne * nj * nqh, '~<ei||Q> of a block')
          kz = xzget(NW * ne * nj * nb * nc, 'product of X2')
          if (.not. xzdry) then
            call N(xzlbvq)(no, nv, nvp, ne, nj, nqh, vcoff(se), ocoff(si), qlo, vq, w(kv))
            call N(zc_gemm)(0_4, 0_4, ne * nj, nb * nc, nqh, one, w(kv), nqh, w(ktv), nb * nc, zero, &
                            w(kz), nb * nc)
            call N(xzlbx2add)(no, nv, nvp, nto, ne, nj, nb, nc, vcoff(se), ocoff(si), i0, vcoff(sc), &
                              w(jvpidx), w(kz), x2)
          end if
          call xzrel(ms)
        end do
        call xzrel(mk)
      end do
    end do
  end do
end subroutine

! xq[P][i0 + I][c0 + c] += z[P][I][c]
subroutine N(xzlbxqadd)(nv, nto, np, nb, nc, i0, c0, z, xq)
  implicit none
  integer(8), intent(in) :: nv, nto, np, nb, nc, i0, c0
  ZT, intent(in) :: z(0:nc-1, 0:nb-1, 0:np-1)
  ZT :: xq(0:nv-1, 0:nto-1, 0:np-1)
  integer(8) :: p
  !$omp parallel do
  do p = 0, np - 1
    xq(c0:c0 + nc - 1, i0:i0 + nb - 1, p) = xq(c0:c0 + nc - 1, i0:i0 + nb - 1, p) + z(:, :, p)
  end do
end subroutine

! v[e][i][Q] = vq[e0 + e][j0 + i][qlo + Q]
subroutine N(xzlbvq)(no, nv, nvp, ne, nj, nq, e0, j0, qlo, vq, v)
  implicit none
  integer(8), intent(in) :: no, nv, nvp, ne, nj, nq, e0, j0, qlo
  ZT, intent(in) :: vq(0:nvp-1, 0:no-1, 0:nv-1)
  ZT, intent(out) :: v(0:nq-1, 0:nj-1, 0:ne-1)
  integer(8) :: e
  do e = 0, ne - 1
    v(:, :, e) = vq(qlo:qlo + nq - 1, j0:j0 + nj - 1, e0 + e)
  end do
end subroutine

! x2[(e<f)][i0 + I][j0 + i] += z[e][i][I][f],  x2[(f<e)][..] -= z[e][i][I][f]
! for e of the block from e0 and f of the block from f0
subroutine N(xzlbx2add)(no, nv, nvp, nto, ne, nj, nb, nc, e0, j0, i0, f0, vpidx, z, x2)
  implicit none
  integer(8), intent(in) :: no, nv, nvp, nto, ne, nj, nb, nc, e0, j0, i0, f0
  integer(8), intent(in) :: vpidx(0:nv-1, 0:nv-1)              ! [p][q], p<q
  ZT, intent(in) :: z(0:nc-1, 0:nb-1, 0:nj-1, 0:ne-1)          ! [e][i][I][f]
  ZT :: x2(0:no-1, 0:nto-1, 0:nvp-1)                           ! [Q][I][i]
  integer(8) :: e, f, ea, fa, q, it, i
  do e = 0, ne - 1
    ea = e0 + e
    do f = 0, nc - 1
      fa = f0 + f
      if (ea == fa) cycle
      if (ea < fa) then
        q = vpidx(fa, ea)
        do it = 0, nb - 1
          do i = 0, nj - 1
            x2(j0 + i, i0 + it, q) = x2(j0 + i, i0 + it, q) + z(f, it, i, e)
          end do
        end do
      else
        q = vpidx(ea, fa)
        do it = 0, nb - 1
          do i = 0, nj - 1
            x2(j0 + i, i0 + it, q) = x2(j0 + i, i0 + it, q) - z(f, it, i, e)
          end do
        end do
      end if
    end do
  end do
end subroutine
