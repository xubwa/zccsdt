!
! The T3 passes of a CCSDT step with T3 in the blocks of the symmetry labels
! (type-generic): the scheduler of SpinBlockT3 of src/zccsdt/cc/t3engine.py
! for any number of classes, with the kernels of csrc/t3pack.c.
!
! A class s of spinors has the occupied spinors ocoff(s) .. ocoff(s+1)-1 and
! the virtual ones vcoff(s) .. vcoff(s+1)-1; a group g of pairs or triples has
! the sum g of the classes of its indices.  T3[I][A] is stored for triples I
! and A of the same group.  Every contraction is done per combination of
! classes that the labels allow, on ranges of the dense intermediates.
!
! The batch buffers are taken from the stack inside the loops: a dry run goes
! through the same loops and finds what they take.
!

! ---------------------------------------------------------------------------
! Pass 1: with T[P][d][m][Q] = t3[m, P, d, Q] of a batch of pairs P of group
! gP, d of class sd, m of class sm, Q of group gQ = gP + sm - sd:
!   xacc[e|j][m][Q] += sum_{P,d} gpd[e|j][P][d] T      (e, j of class gP - sd)
!   yall[P][d][j|e] += sum_{m,Q} T gmq[m][Q][j|e]
!   r1c[d][m]       += sum_{P,Q} T gpp[P][Q]           (sd = sm)
!   ra[P][Q]        += sum_{d,m} f[m][d] T
! The four results are zeroed here.
subroutine N(xzbpass1)(w, kt3, gpd, gmq, gpp, fov, xacc, yall, r1c, ra)
  use xzdata
  use xzmem
  use xzkern
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kt3
  ZT :: gpd(*), gmq(*), gpp(0:nvp-1, 0:npo-1), fov(*), xacc(*), yall(*), r1c(*), ra(0:nvp-1, 0:npo-1)
  integer(8) :: gp, gq, p0, nb, sd, sm, sy, nd, nm, nq, qlo, ne, nj, m0, kt, kg, kx, kgy, ky, kacc

  if (.not. xzdry) then
    call zc_vzero_d(NW * ny * no * nvp, xacc)
    call zc_vzero_d(NW * npo * nv * ny, yall)
    call zc_vzero_d(NW * no * nv, r1c)
    call zc_vzero_d(NW * npo * nvp, ra)
  end if
  do gp = 0, npg - 1
    do p0 = opoff(gp), opoff(gp + 1) - 1, nbp
      nb = min(nbp, opoff(gp + 1) - p0)
      do sd = 0, ncls - 1
        do sm = 0, ncls - 1
          gq = gp + sm - sd
          if (gq < 0 .or. gq >= npg) cycle
          nd = vcoff(sd + 1) - vcoff(sd)
          nm = ocoff(sm + 1) - ocoff(sm)
          qlo = vpoff(gq)
          nq = vpoff(gq + 1) - qlo
          if (nd == 0 .or. nm == 0 .or. nq == 0) cycle
          m0 = xzmark()
          kt = xzget(NW * nb * nd * nm * nq, 'pass-1 batch of T3')
          if (.not. xzdry) call N(zc_bgather2)(w(kt3), w(jobase), w(josig), w(jotri), w(josgn), npo, &
            w(jvci), w(jvsgn), nvp, nb, w(jrows + p0), nd, w(jrows + vcoff(sd)), nm, &
            w(jrows + ocoff(sm)), qlo, nq, nd * nm * nq, w(kt))
          sy = gp - sd
          if (sy >= 0 .and. sy < ncls) then
            ne = vcoff(sy + 1) - vcoff(sy)
            nj = ocoff(sy + 1) - ocoff(sy)
            if (ne + nj > 0) then
              kg = xzget(NW * (ne + nj) * nb * nd, 'pass-1 operand')
              kx = xzget(NW * (ne + nj) * nm * nq, 'pass-1 product')
              kgy = xzget(NW * nm * nq * (ne + nj), 'pass-1 operand')
              ky = xzget(NW * nb * nd * (ne + nj), 'pass-1 product')
              if (.not. xzdry) call N(xzbp1xy)(no, nv, npo, nvp, nb, nd, nm, nq, ne, nj, p0, vcoff(sd), &
                ocoff(sm), qlo, vcoff(sy), ocoff(sy), gpd, gmq, w(kt), w(kg), w(kx), w(kgy), w(ky), &
                xacc, yall)
            end if
          end if
          if (sd == sm) then
            kacc = xzget(NW * 2 * nd * nm, 'pass-1 work')
            if (.not. xzdry) call N(xzbp1r)(no, nv, nvp, nb, nd, nm, nq, vcoff(sd), ocoff(sm), &
              w(kt), gpp(qlo, p0), fov, w(kacc), r1c, ra(qlo, p0))
          end if
          call xzrel(m0)
        end do
      end do
    end do
  end do
end subroutine

! the two products of a block of pass 1 and their places in the results
subroutine N(xzbp1xy)(no, nv, npo, nvp, nb, nd, nm, nq, ne, nj, p0, d0, m0, qlo, e0, j0, gpd, gmq, t, &
                      g, x, gy, y, xacc, yall)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp, nb, nd, nm, nq, ne, nj, p0, d0, m0, qlo, e0, j0
  ZT, intent(in) :: gpd(0:nv-1, 0:npo-1, 0:nv+no-1)            ! [e|j][P][d]
  ZT, intent(in) :: gmq(0:no+nv-1, 0:nvp-1, 0:no-1)            ! [m][R][j|e]
  ZT, intent(in) :: t(*)                                       ! [(P, d)][(m, Q)]
  ZT :: g(0:nd-1, 0:nb-1, 0:ne+nj-1)                           ! [r][P][d]
  ZT :: x(0:nq-1, 0:nm-1, 0:ne+nj-1)                           ! [r][m][Q]
  ZT :: gy(0:ne+nj-1, 0:nq-1, 0:nm-1)                          ! [m][Q][c]
  ZT :: y(0:ne+nj-1, 0:nd-1, 0:nb-1)                           ! [P][d][c]
  ZT, intent(inout) :: xacc(0:nvp-1, 0:no-1, 0:nv+no-1)        ! [e|j][m][Q]
  ZT, intent(inout) :: yall(0:no+nv-1, 0:nv-1, 0:npo-1)        ! [P][d][j|e]
  integer(8) :: r, p, m, q, row, col, c
  ZT :: one, zero
  one = 1
  zero = 0
  ! rows: e of the class, then j of the class
  do r = 0, ne + nj - 1
    row = e0 + r
    if (r >= ne) row = nv + j0 + (r - ne)
    do p = 0, nb - 1
      g(:, p, r) = gpd(d0:d0 + nd - 1, p0 + p, row)
    end do
  end do
  call N(zc_gemm)(0_4, 0_4, ne + nj, nm * nq, nb * nd, one, g, nb * nd, t, nm * nq, zero, x, nm * nq)
  !$omp parallel do private(r, row, m)
  do r = 0, ne + nj - 1
    row = e0 + r
    if (r >= ne) row = nv + j0 + (r - ne)
    do m = 0, nm - 1
      xacc(qlo:qlo + nq - 1, m0 + m, row) = xacc(qlo:qlo + nq - 1, m0 + m, row) + x(:, m, r)
    end do
  end do
  ! columns: j of the class, then e of the class
  !$omp parallel do private(m, q, c, col)
  do m = 0, nm - 1
    do q = 0, nq - 1
      do c = 0, nj + ne - 1
        col = j0 + c
        if (c >= nj) col = no + e0 + (c - nj)
        gy(c, q, m) = gmq(col, qlo + q, m0 + m)
      end do
    end do
  end do
  call N(zc_gemm)(0_4, 0_4, nb * nd, nj + ne, nm * nq, one, t, nm * nq, gy, nj + ne, zero, y, nj + ne)
  do p = 0, nb - 1
    do m = 0, nd - 1
      do c = 0, nj + ne - 1
        col = j0 + c
        if (c >= nj) col = no + e0 + (c - nj)
        yall(col, d0 + m, p0 + p) = yall(col, d0 + m, p0 + p) + y(c, m, p)
      end do
    end do
  end do
end subroutine

! the T1 and T2 terms of a block of pass 1 (d and m of the same class)
subroutine N(xzbp1r)(no, nv, nvp, nb, nd, nm, nq, d0, m0, t, gpp, fov, acc, r1c, ra)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, nvp, nb, nd, nm, nq, d0, m0
  ZT, intent(in) :: t(0:nq-1, 0:nd*nm-1, 0:nb-1)               ! [P][(d, m)][Q]
  ZT, intent(in) :: gpp(0:nvp-1, 0:nb-1)                       ! [P][Q], from the first Q of the block
  ZT, intent(in) :: fov(0:nv-1, 0:no-1)                        ! [m][e]
  ZT :: acc(0:nd*nm-1, 0:1)
  ZT, intent(inout) :: r1c(0:no-1, 0:nv-1)                     ! [d][m]
  ZT, intent(inout) :: ra(0:nvp-1, 0:nb-1)                     ! [P][Q], from the first Q of the block
  integer(8) :: p, d, m
  ZT :: one
  one = 1
  acc(:, 0) = 0
  do d = 0, nd - 1
    do m = 0, nm - 1
      acc(d * nm + m, 1) = fov(d0 + d, m0 + m)
    end do
  end do
  do p = 0, nb - 1
    call N(zc_gemm)(0_4, 0_4, nd * nm, 1_8, nq, one, t(0, 0, p), nq, gpp(0, p), 1_8, one, acc(0, 0), 1_8)
    call N(zc_gemm)(0_4, 0_4, 1_8, nq, nd * nm, one, acc(0, 1), nd * nm, t(0, 0, p), nq, one, ra(0, p), nq)
  end do
  do d = 0, nd - 1
    do m = 0, nm - 1
      r1c(m0 + m, d0 + d) = r1c(m0 + m, d0 + d) + acc(d * nm + m, 0)
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The right-hand sides of the drive terms of pass 2, one block per (group gQ
! of Q, class si of i), e and m of the class gQ - si:
!   bd[e][i][Q] = wv[Q][e][i],   bd[ne + m][i][Q] = t2p[m][i][Q]
subroutine N(xzbbd)(wv, t2p, bd)
  use xzdata
  implicit none
  ZT, intent(in) :: wv(0:no-1, 0:nv-1, 0:nvp-1)                ! [Q][e][i]
  ZT, intent(in) :: t2p(0:nvp-1, 0:no-1, 0:no-1)               ! [m][i][Q]
  ZT :: bd(0:*)
  integer(8) :: gq, si, se, e, i, q, m, k, ni, nq, ne, nm, q0, base
  do gq = 0, npg - 1
    do si = 0, ncls - 1
      if (bdoff(gq, si) < 0) cycle
      se = gq - si
      ni = ocoff(si + 1) - ocoff(si)
      ne = vcoff(se + 1) - vcoff(se)
      nm = ocoff(se + 1) - ocoff(se)
      q0 = vpoff(gq)
      nq = vpoff(gq + 1) - q0
      base = bdoff(gq, si)
      !$omp parallel do private(e, i, q, k)
      do e = 0, ne - 1
        do i = 0, ni - 1
          k = base + (e * ni + i) * nq
          do q = 0, nq - 1
            bd(k + q) = wv(ocoff(si) + i, vcoff(se) + e, q0 + q)
          end do
        end do
      end do
      do m = 0, nm - 1
        do i = 0, ni - 1
          k = base + ((ne + m) * ni + i) * nq
          bd(k:k + nq - 1) = t2p(q0:q0 + nq - 1, ocoff(si) + i, ocoff(se) + m)
        end do
      end do
    end do
  end do
end subroutine

! The ring operator on the pairs (a, i) with s(a) - s(i) = delta:
!   wd[(a, i)][(d, m)] = wring[(a, i)][(d, m)]
subroutine N(xzbwd)(delta, wring, wd)
  use xzdata
  implicit none
  integer(8), intent(in) :: delta
  ZT, intent(in) :: wring(0:no*nv-1, 0:no*nv-1)                ! [(a, i)][(d, m)]
  ZT, intent(out) :: wd(0:ndel(delta)-1, 0:ndel(delta)-1)
  integer(8) :: idx(0:ndel(delta)-1), sa, si, a, i, n, r, c
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
  !$omp parallel do private(r, c)
  do r = 0, n - 1
    do c = 0, n - 1
      wd(c, r) = wring(idx(c), idx(r))
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! Pass 2: for a batch of pairs P of group gP and every group gQ of Q with
! |gP - gQ| < ncls, on the pairs (a, i) of delta = gP - gQ:
!   Z[P][(a, i)][Q]  = wd[(a, i)][(d, m)] T[P][(d, m)][Q]
!   Z[P][(a, i)][Q] += u[(a, i)][x] vp[P][x][Q]             (gP = gQ; nu of them)
!   Z[P][c][(i, Q)] -= adrv[P][c][k] bd[k][(i, Q)]
! and the reduction onto the blocks of the residual.  The summed index k of
! the drive terms has npart parts of nv + no values each (virtual, then
! occupied: one part in the T3 equation, two in the Lambda equations); bd is
! in blocks (xzbbd, xzbbdk).
subroutine N(xzbpass2)(w, kt3, kr3, wring, adrv, bd, npart, nu, u, vp)
  use xzdata
  use xzmem
  use xzkern
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kt3, kr3, npart, nu
  ZT, intent(in) :: wring(*), bd(0:*)
  ZT, intent(in) :: adrv(0:npart*(nv+no)-1, 0:nv-1, 0:npo-1)   ! [P][c][k]
  ZT, intent(in) :: u(*), vp(*)
  integer(8) :: gp, gq, p0, nb, off, tmax, m0, kwd, kz, kt, kzoff, ktg, k, g, d, ku0
  integer(8) :: nqg(0:npg-1)
  do g = 0, npg - 1
    nqg(g) = vpoff(g + 1) - vpoff(g)
  end do
  m0 = xzmark()
  kwd = xzget(NW * nwd, 'ring operators of pass 2')
  ku0 = xzget(NW * ndel(0) * max(nu, 1_8), 'outer products of pass 2')
  if (.not. xzdry) then
    do d = 1 - ncls, ncls - 1
      if (ndel(d) > 0) call N(xzbwd)(d, wring, w(kwd + NW * wdoff(d)))
    end do
    if (nu > 0) call N(xzbu0)(nu, u, w(ku0))
  end if
  do gp = 0, npg - 1
    do p0 = opoff(gp), opoff(gp + 1) - 1, nbp
      nb = min(nbp, opoff(gp + 1) - p0)
      off = 0
      tmax = 0
      do gq = 0, npg - 1
        if (abs(gp - gq) >= ncls .or. nqg(gq) == 0) cycle
        off = off + nb * ndel(gp - gq) * nqg(gq)
        tmax = max(tmax, nb * ndel(gp - gq) * nqg(gq))
      end do
      if (off == 0) cycle
      k = xzmark()
      kz = xzget(NW * off, 'pass-2 products')
      kt = xzget(NW * tmax, 'pass-2 batch of T3')
      kzoff = xzget(nb * npg, 'pass-2 offsets')
      ktg = xzget(4 * 3 * nto, 'pass-2 targets')
      if (.not. xzdry) then
        call N(xzbp2batch)(w, kt3, kr3, gp, p0, nb, nqg, w(kwd), adrv, bd, npart, nu, w(ku0), vp, &
                           w(kz), w(kt), w(kzoff), w(ktg))
      end if
      call xzrel(k)
    end do
  end do
  call xzrel(m0)
end subroutine

! the rows of u[(a, i)][x] of the pairs (a, i) with s(a) = s(i), in their order
subroutine N(xzbu0)(nu, u, u0)
  use xzdata
  implicit none
  integer(8), intent(in) :: nu
  ZT, intent(in) :: u(0:nu-1, 0:no-1, 0:nv-1)                  ! [a][i][x]
  ZT, intent(out) :: u0(0:nu-1, 0:*)
  integer(8) :: sa, a, i, n
  n = 0
  do sa = 0, ncls - 1
    do a = vcoff(sa), vcoff(sa + 1) - 1
      do i = ocoff(sa), ocoff(sa + 1) - 1
        u0(:, n) = u(:, i, a)
        n = n + 1
      end do
    end do
  end do
end subroutine

! The right-hand sides of the drive terms in blocks from the dense array
! bdrv[k][i][Q], k in npart parts (virtual, occupied): one block per (group
! gQ of Q, class si of i) with the k of the class gQ - si, part by part.
subroutine N(xzbbdk)(npart, bdrv, bd)
  use xzdata
  implicit none
  integer(8), intent(in) :: npart
  ZT, intent(in) :: bdrv(0:nvp-1, 0:no-1, 0:npart*(nv+no)-1)   ! [k][i][Q]
  ZT :: bd(0:*)
  integer(8) :: gq, si, se, i, ni, nq, ne, nm, q0, base, part, r, row, k
  do gq = 0, npg - 1
    do si = 0, ncls - 1
      if (bdoff(gq, si) < 0) cycle
      se = gq - si
      ni = ocoff(si + 1) - ocoff(si)
      ne = vcoff(se + 1) - vcoff(se)
      nm = ocoff(se + 1) - ocoff(se)
      q0 = vpoff(gq)
      nq = vpoff(gq + 1) - q0
      base = npart * bdoff(gq, si)
      !$omp parallel do collapse(2) private(part, r, row, i, k)
      do part = 0, npart - 1
        do r = 0, ne + nm - 1
          row = part * (nv + no) + vcoff(se) + r
          if (r >= ne) row = part * (nv + no) + nv + ocoff(se) + (r - ne)
          do i = 0, ni - 1
            k = base + ((part * (ne + nm) + r) * ni + i) * nq
            bd(k:k + nq - 1) = bdrv(q0:q0 + nq - 1, ocoff(si) + i, row)
          end do
        end do
      end do
    end do
  end do
end subroutine

! one batch of pass 2
subroutine N(xzbp2batch)(w, kt3, kr3, gp, p0, nb, nqg, wd, adrv, bd, npart, nu, u0, vp, z, t, zoff, tg)
  use xzdata
  use xzkern
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kt3, kr3, gp, p0, nb, npart, nu
  integer(8), intent(in) :: nqg(0:npg-1)
  ZT, intent(in) :: wd(0:*), bd(0:*)
  ZT, intent(in) :: adrv(0:npart*(nv+no)-1, 0:nv-1, 0:npo-1)   ! [P][c][k]
  ZT, intent(in) :: u0(*)
  ZT, intent(in) :: vp(0:nvp-1, 0:max(nu,1_8)-1, 0:*)          ! [P][x][Q]
  ZT :: z(0:*), t(0:*)
  integer(8) :: zoff(0:npg-1, 0:nb-1)                          ! [P][gQ]
  integer(8) :: tg(0:3*nto-1, 0:3)                             ! targets: base, index, pair, (sign)
  integer(8) :: gq, delta, nr, nq, qlo, off, k, sa, si, sc, se, r0, nd, nm, nc, ni, ne, g, nt, it, r, p
  integer(8) :: part, ka, kb
  ZT :: one, zero, mone
  one = 1
  zero = 0
  mone = -1
  zoff = -1
  off = 0
  do k = 0, nb - 1
    do gq = 0, npg - 1
      if (abs(gp - gq) >= ncls .or. nqg(gq) == 0) cycle
      zoff(gq, k) = off
      off = off + ndel(gp - gq) * nqg(gq)
    end do
  end do
  do gq = 0, npg - 1
    if (abs(gp - gq) >= ncls .or. nqg(gq) == 0) cycle
    delta = gp - gq
    nr = ndel(delta)
    if (nr == 0) cycle
    qlo = vpoff(gq)
    nq = nqg(gq)
    ! T[P][(d, m)][Q] on the pairs (d, m) of delta
    r0 = 0
    do sa = 0, ncls - 1
      si = sa - delta
      if (si < 0 .or. si >= ncls) cycle
      nd = vcoff(sa + 1) - vcoff(sa)
      nm = ocoff(si + 1) - ocoff(si)
      if (nd > 0 .and. nm > 0) call N(zc_bgather2)(w(kt3), w(jobase), w(josig), w(jotri), w(josgn), npo, &
        w(jvci), w(jvsgn), nvp, nb, w(jrows + p0), nd, w(jrows + vcoff(sa)), nm, w(jrows + ocoff(si)), &
        qlo, nq, nr * nq, t(r0 * nq))
      r0 = r0 + nd * nm
    end do
    ! the ring term, and the outer products
    do k = 0, nb - 1
      call N(zc_gemm)(0_4, 0_4, nr, nq, nr, one, wd(wdoff(delta)), nr, t(k * nr * nq), nq, zero, &
                      z(zoff(gq, k)), nq)
      if (delta == 0 .and. nu > 0) call N(zc_gemm)(0_4, 0_4, nr, nq, nu, one, u0, nu, vp(qlo, 0, p0 + k), &
                                                   nvp, one, z(zoff(gq, k)), nq)
    end do
    ! the drive terms on the classes (sc, si) of the pairs (c, i)
    r0 = 0
    do sc = 0, ncls - 1
      si = sc - delta
      if (si < 0 .or. si >= ncls) cycle
      nc = vcoff(sc + 1) - vcoff(sc)
      ni = ocoff(si + 1) - ocoff(si)
      se = gp - sc
      if (nc > 0 .and. ni > 0 .and. se >= 0 .and. se < ncls) then
        if (bdoff(gq, si) >= 0) then
          ne = vcoff(se + 1) - vcoff(se)
          nm = ocoff(se + 1) - ocoff(se)
          do part = 0, npart - 1
            ka = part * (nv + no)
            kb = npart * bdoff(gq, si) + part * (ne + nm) * ni * nq
            do k = 0, nb - 1
              p = p0 + k
              if (ne > 0) call N(zc_gemm)(0_4, 0_4, nc, ni * nq, ne, mone, &
                adrv(ka + vcoff(se), vcoff(sc), p), npart * (nv + no), bd(kb), ni * nq, one, &
                z(zoff(gq, k) + r0 * nq), ni * nq)
              if (nm > 0) call N(zc_gemm)(0_4, 0_4, nc, ni * nq, nm, mone, &
                adrv(ka + nv + ocoff(se), vcoff(sc), p), npart * (nv + no), bd(kb + ne * ni * nq), &
                ni * nq, one, z(zoff(gq, k) + r0 * nq), ni * nq)
            end do
          end do
        end if
      end if
      r0 = r0 + nc * ni
    end do
  end do
  ! the reduction: per group of triples, the occupied triples with a pair in
  ! the batch
  do g = 0, ntg - 1
    if (vtoff(g + 1) == vtoff(g)) cycle
    nt = 0
    do r = 0, 2
      call N(xzbtargets)(nto, otoff(g), otoff(g + 1), w(jopair + r * nto), w(jofree + r * nto), w(jobase), &
                         p0, nb, merge(-1d0, 1d0, r == 1), nt, tg(0, 0), tg(0, 1), tg(0, 2), tg(0, 3))
    end do
    if (nt == 0) cycle
    call N(zc_breduce2)(w(kr3), w(jvcol), nt, tg(0, 0), tg(0, 1), tg(0, 2), tg(0, 3), vtoff(g), &
                        vtoff(g + 1), nvt, w(jvfree), w(jvpair), w(jvpgrp), zoff, npg, nqg, vpoff, &
                        w(jrowpos), no, z)
  end do
end subroutine

! the targets of the reduction of pass 2 for one way r of splitting a triple
subroutine N(xzbtargets)(nto, i0, i1, opair, ofree, obase, p0, nb, sg, nt, tbase, tx, tp, tsg)
  implicit none
  integer(8), intent(in) :: nto, i0, i1, p0, nb
  integer(8), intent(in) :: opair(0:nto-1), ofree(0:nto-1), obase(0:nto-1)
  real(8), intent(in) :: sg
  integer(8), intent(inout) :: nt
  integer(8) :: tbase(0:*), tx(0:*), tp(0:*)
  real(8) :: tsg(0:*)
  integer(8) :: it, p
  do it = i0, i1 - 1
    p = opair(it) - p0
    if (p < 0 .or. p >= nb) cycle
    tbase(nt) = obase(it)
    tx(nt) = ofree(it)
    tp(nt) = p
    tsg(nt) = sg
    nt = nt + 1
  end do
end subroutine

! ---------------------------------------------------------------------------
! The pp ladder for the rows q0 .. q0 + nq - 1 (block-local) of the block h of
! the operator, slab[q][Q]: for the groups g of triples and the class
! sc = g - h of the free virtual index,
!   X[q][I][c] = sum_Q slab[q][Q] t3[I, Q, c],  reduced onto the residual.
subroutine N(xzbppl)(w, kt3, kr3, h, q0, nq, slab)
  use xzdata
  use xzmem
  use xzkern
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kt3, kr3, h, q0, nq
  ZT, intent(in) :: slab(*)
  integer(8) :: g, sc, nc, nqh, i0, nb, m0, ktv, kxp
  real(8) :: t0
  ZT :: one, zero
  one = 1
  zero = 0
  nqh = vpoff(h + 1) - vpoff(h)
  do g = 0, ntg - 1
    sc = g - h
    if (sc < 0 .or. sc >= ncls) cycle
    nc = vcoff(sc + 1) - vcoff(sc)
    if (nc == 0 .or. vtoff(g + 1) == vtoff(g)) cycle
    do i0 = otoff(g), otoff(g + 1) - 1, nbt
      nb = min(nbt, otoff(g + 1) - i0)
      m0 = xzmark()
      ktv = xzget(NW * nqh * nb * nc, 'pp-ladder batch of T3')
      kxp = xzget(NW * nq * nb * nc, 'pp-ladder product')
      if (.not. xzdry) then
        t0 = omp_get_wtime()
        call N(zc_bgather_vir)(w(kt3), w(jobase), w(jvcol), w(josig), w(jvsig), w(jvtri), w(jvsgn), nvp, &
                               nb, w(jrows + i0), vpoff(h), nqh, vcoff(sc), nc, w(ktv))
        tstage(7) = tstage(7) + omp_get_wtime() - t0
        t0 = omp_get_wtime()
        call N(zc_gemm)(0_4, 0_4, nq, nb * nc, nqh, one, slab, nqh, w(ktv), nb * nc, zero, w(kxp), nb * nc)
        tstage(8) = tstage(8) + omp_get_wtime() - t0
        t0 = omp_get_wtime()
        call N(zc_breduce_ppl_rows)(w(kr3), w(jobase), w(jvcol), nb, w(jrows + i0), nvt, vtoff(g), &
                                    vtoff(g + 1), w(jvfree), w(jvpair), vcls, sc, vpoff(h), q0, q0 + nq, &
                                    nc, vcoff(sc), w(kxp))
        tstage(9) = tstage(9) + omp_get_wtime() - t0
      end if
      call xzrel(m0)
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The operator on the occupied triples (hh ladder, F_oo), block by block, and
! the Jacobi step: r <- t3 + r / D.
subroutine N(xzbocc)(w, kt3, kr3, o6)
  use xzdata
  use xzkern
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kt3, kr3
  ZT, intent(in) :: o6(0:nto-1, 0:nto-1)                       ! [I][J]
  integer(8) :: g, ni, na
  ZT :: one
  one = 1
  do g = 0, ntg - 1
    ni = otoff(g + 1) - otoff(g)
    na = vtoff(g + 1) - vtoff(g)
    if (ni == 0 .or. na == 0) cycle
    call N(zc_gemm)(0_4, 0_4, ni, na, ni, one, o6(otoff(g), otoff(g)), nto, w(kt3 + NW * boff(g)), na, &
                    one, w(kr3 + NW * boff(g)), na)
  end do
end subroutine

subroutine N(xzbupd)(w, kt3, kr3)
  use xzdata
  use xzkern
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kt3, kr3
  integer(8) :: g, ni, na
  do g = 0, ntg - 1
    ni = otoff(g + 1) - otoff(g)
    na = vtoff(g + 1) - vtoff(g)
    if (ni == 0 .or. na == 0) cycle
    call N(zc_t3_update)(w(kt3 + NW * boff(g)), w(kr3 + NW * boff(g)), ni, na, w(jeo3 + otoff(g)), &
                         w(jev3 + vtoff(g)), c_null_ptr)
  end do
end subroutine
