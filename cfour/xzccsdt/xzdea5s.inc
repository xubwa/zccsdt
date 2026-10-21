!
! EOM-DEA-CCSDT with complete 5p3h: the 5p3h part of a sigma vector
! (type-generic; xzdea5.inc; every step the counterpart of one of
! dea5_sector.Engine5, in the same order where the order matters).
!

! ---------------------------------------------------------------------------
! The terms of r5 in the sigma vector and the 5p3h row, after the rows to
! 4p2h of N(xzdsig) (its intermediates: r2 on all pairs kr2, r3 on (a | bc)
! kx3r, YR kyr, XvR kz, WhR kwhr, summed over the ranks; r3 at k3).  k4, ks4:
! r4 and sigma4 (whole-r4 addresses: the rows of this rank), ks3 sigma3; k5c,
! ks5c: the stored 5p3h part of the vector and of its sigma vector (the rows
! of this rank).
subroutine N(xzd5sig)(w, k3, kr2, kx3r, kyr, kz, kwhr, k4, ks3, ks4, k5c, ks5c)
  use xzdata
  use xzmem
  use xzkern
  use xzdead
  use xzdead5
  use xzpar, only: myrank, nrank, xzmpi_slices_put, xzmpi_slices_add, xzmpi_sum0_w, xzmpi_allsum_w, xzmpi_chunk
  use omp_lib
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: k3, kr2, kx3r, kyr, kz, kwhr, k4, ks3, ks4, k5c, ks5c
  integer(8) :: m0, kr5, ks5, kr5v, ks5v, kr4f, ks4f, ks3x, kg, kg3, kg2, kg5, kd3r, kxorp, kxor5, ky4, kxvr5
  integer(8) :: kx4s, nx4s, pq, m4, ktmp, ntmp, nown5
  integer(8), allocatable :: x4sloc(:), woff(:)
  integer(8) :: row(1), xo(ncls), y1(ncls), y2(ncls), p, r
  real(8) :: sg(1), t0, t1
  ZT :: one
  one = 1
  nown5 = d5f(myrank + 1) - d5f(myrank)
  ! the pairs of the triples of this rank: their rows of sigma4 on (a | bcd) (X4s)
  allocate(x4sloc(0:max(npo, 1_8)-1))
  call N(xzd5x4sl)(w, x4sloc, nx4s)
  m0 = xzmark()
  kr5 = xzget(NW * max(nown5, 1_8), 'DEA5: r5 of the rows of this rank')
  ks5 = xzget(NW * max(nown5, 1_8), 'DEA5: sigma5 of the rows of this rank')
  kr5v = kr5 - NW * d5f(myrank)
  ks5v = ks5 - NW * d5f(myrank)
  if (ddist) then
    kr4f = xzget(NW * max(dsz4, 1_8), 'DEA5: r4 (all rows)')
    ks4f = xzget(NW * max(dsz4, 1_8), 'DEA5: the terms of r5 in sigma4 (all rows)')
  else
    kr4f = k4
    ks4f = ks4
  end if
  ks3x = xzget(NW * max(dsz3, 1_8), 'DEA5: the terms of r5 in sigma3')
  kg = xzget(NW * no * no * no * nv, 'DEA5: G')
  kg3 = xzget(NW * no * no * nv * nv, 'DEA5: G3')
  kg2 = xzget(NW * no * nv * nv * nv, 'DEA5: G2')
  kg5 = xzget(NW * max(nv * nv * npo, 1_8), 'DEA5: G5')
  kd3r = xzget(NW * max(no * nv * nvp, 1_8), 'DEA5: r3 on (h | pair), dense')
  kxorp = xzget(NW * max(no * no * nvp, 1_8), 'DEA5: XoRp')
  kxor5 = xzget(NW * max(xr5size, 1_8), 'DEA5: XoR5')
  ky4 = xzget(NW * max(y4size, 1_8), 'DEA5: Y4')
  kxvr5 = xzget(NW * max(zsize, 1_8), 'DEA5: XvR5')
  kx4s = xzget(NW * max(nx4s, 1_8), 'DEA5: the terms of r5 in sigma4 on (a | bcd)')
  ntmp = 4 * min(xzmpi_chunk(), NW * max(dsz4, 1_8))
  ktmp = 0
  if (ddist) ktmp = xzget(ntmp, 'DEA5: buffer of the sums of sigma4 over the ranks')
  if (.not. xzdry) then
    t0 = omp_get_wtime()
    call N(xzd5pack)(w, k5c, kr5v, .true.)
    call zc_vzero_d(NW * max(nown5, 1_8), w(ks5))
    call zc_vzero_d(NW * max(dsz3, 1_8), w(ks3x))
    call zc_vzero_d(NW * max(nx4s, 1_8), w(kx4s))
    if (ddist) then
      ! r4 of all rows on every rank
      allocate(woff(0:nrank))
      woff = NW * d4e
      call N(xzecopy)(d4own, w(k4 + NW * d4e(myrank)), w(kr4f + NW * d4e(myrank)))
      call xzmpi_slices_put(w(kr4f), woff)
      call zc_vzero_d(NW * max(dsz4, 1_8), w(ks4f))
    end if
    t1 = omp_get_wtime()
    td5(1) = td5(1) + t1 - t0
  end if
  ! the small intermediates
  call N(xzd5g)(w, kr2, kx3r, kwhr, kg, kg3, kg2, kg5, kd3r, kxorp)
  call N(xzd5xvr)(w, kz, kr2, kxvr5)
  call N(xzd5pairs)(w, kr4f, kr2, kx3r, kyr, kg, kg3, kg2, kxorp, kxor5, ky4)
  ! the pp ladder with its thin products (XoR5, sigma3, sigma4 on (a | bcd))
  call N(xzd5pp)(w, kr5v, ks5v, kxor5, ks3x, kx4s, x4sloc)
  if (ddist .and. .not. xzdry) then
    t0 = omp_get_wtime()
    call xzmpi_allsum_w(w(kxor5), NW * max(xr5size, 1_8))
    call xzmpi_allsum_w(w(ky4), NW * max(y4size, 1_8))
    call xzmpi_allsum_w(w(kxorp), NW * max(no * no * nvp, 1_8))
    td5(15) = td5(15) + omp_get_wtime() - t0
  end if
  call N(xzd5ring)(w, kr5v, ks5v, ks4f)
  call N(xzd5hh)(w, kr5v, ks5v)
  call N(xzd5rows)(w, k3, kr4f, kx3r, kxvr5, ky4, kxor5, kxorp, ks5v)
  call N(xzd5zx)(w, kr4f, kr5v, ks5v, kg5)
  if (.not. xzdry) then
    t0 = omp_get_wtime()
    ! sigma4 from r5 on (a | bcd), reduced (sign -1) into the rows of the pairs
    do pq = 0, npo - 1
      if (x4sloc(pq) < 0) cycle
      m4 = blk4(pq)
      row(1) = pos4(pq)
      sg(1) = -1d0
      do p = 0, ncls - 1
        xo(p + 1) = off13(m4, p)
        if (xo(p + 1) >= 0) xo(p + 1) = xo(p + 1) + x4sloc(pq)
        y1(p + 1) = max(dnK(m4 - p), 1_8)
        y2(p + 1) = 1
      end do
      call N(xzdx13)(w, 1, ks4f + NW * off4(m4), nqa(m4), m4, 1_8, row, sg, xo, y1, y2, kx4s)
    end do
    if (ddist) then
      ! the rows of sigma4 to their owners (summed there), sigma3 to rank 0
      call xzmpi_slices_add(w(ks4f), woff, w(ktmp))
      call N(xzeaxpyc)(d4own, one, w(ks4f + NW * d4e(myrank)), w(ks4 + NW * d4e(myrank)))
      if (dsz3 > 0) call xzmpi_sum0_w(w(ks3x), NW * dsz3)
      deallocate(woff)
    end if
    if (dsz3 > 0 .and. (.not. ddist .or. myrank == 0)) call N(xzeaxpyc)(dsz3, one, w(ks3x), w(ks3))
    call N(xzd5pack)(w, ks5c, ks5v, .false.)
    td5(2) = td5(2) + omp_get_wtime() - t0
  end if
  deallocate(x4sloc)
  call xzrel(m0)

contains

  ! (a test hook: option name file) the array at k of n elements to the file
  subroutine dump(name, k, n)
    use xzopt
    character(len=*), intent(in) :: name
    integer(8), intent(in) :: k, n
    character(len=256) :: arg
    integer :: larg
    call xzoptget(name, arg, larg)
    if (larg <= 0) return
    open(unit=75, file=arg(1:larg), form='unformatted', access='stream', status='replace')
    write(75) w(k:k + NW * n - 1)
    close(75)
  end subroutine
end subroutine

! the places of the rows of the pairs of the triples of this rank in X4s
! (r4 on (a | bcd), n13 of their blocks; -1: none), and its elements
subroutine N(xzd5x4sl)(w, x4sloc, n)
  use xzdata
  use xzdead
  use xzdead5
  implicit none
  real(8), target :: w(*)
  integer(8), intent(out) :: x4sloc(0:max(npo, 1_8)-1), n
  integer(8), pointer :: opair(:, :)
  integer(8) :: i, r, pq
  call xzdiv2(w, jopair, max(nto, 1_8), 3_8, opair)
  x4sloc = -1
  do i = dt0, dt1 - 1
    if (blk5(i) < 0) cycle
    do r = 0, 2
      pq = opair(i, r)
      if (blk4(pq) >= 0) x4sloc(pq) = 0
    end do
  end do
  n = 0
  do pq = 0, npo - 1
    if (x4sloc(pq) < 0) cycle
    x4sloc(pq) = n
    n = n + n13(blk4(pq))
  end do
end subroutine

! the occupied index x is this rank's for the split sums (more than one rank:
! mod(x, ranks); one rank: all)
logical function N(xzd5own)(x)
  use xzdead, only: ddist
  use xzpar, only: myrank, nrank
  implicit none
  integer(8), intent(in) :: x
  N(xzd5own) = .not. ddist
  if (ddist) N(xzd5own) = mod(x, int(nrank, 8)) == myrank
end function

! ---------------------------------------------------------------------------
! G(ml; y; c) = <ml||hy> r2(ch), G3(ml; c, g) = <ml||hg> r2(ch), G2(m; c, d, g)
! = <cm||hg> r2(dh), G5(ln; f, b) = <ln||hf> r2(bh) (summed over h; G5 as
! [f][b][(ln)]), r3 dense on (h | pair), and XoRp(m; i; de) but for its terms
! of r4 (dea5_sector.Engine5._pair_imds):
!   P(de) [ - sum_h <md||hi> r2(eh) + sum_{h<g} <dm||hg> r3(i; hg e) - sum_{lhg} <lm||hg> r2(dh) t2(il; eg) ]
!   - sum_h f_mh r3(i; h de) + sum_{lh} <ml||hi> r3(l; h de) + sum_l WhR(lm) t2(il; de)
! (over the ranks XoRp is made by the ranks of i, mod(i, ranks), and summed later)
subroutine N(xzd5g)(w, kr2, kx3r, kwhr, kg, kg3, kg2, kg5, kd3r, kxorp)
  use xzdata
  use xzmem
  use xzkern
  use xzdead
  use xzdead5
  use omp_lib
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: kr2, kx3r, kwhr, kg, kg3, kg2, kg5, kd3r, kxorp
  logical, external :: N(xzd5own)
  integer(8) :: m0, ka, kx, m, l, c, h, g, pq, i, k, p, q, a, b, nw2
  integer(8), pointer :: opa(:), opb(:), vpidx(:, :), vpa(:), vpb(:)
  ZT, pointer :: av(:), edv(:), g5(:), x(:), xorp(:), d3r(:), x3r(:), ovvo(:), ooovd(:)
  ZT :: one, zero, mone
  real(8) :: t0
  one = 1
  zero = 0
  mone = -1
  nw2 = no * nv
  m0 = xzmark()
  ka = xzget(NW * max(nv * nv, no * nw2), 'DEA5: an operand of G2, G5, XoRp')
  kx = xzget(NW * no * no * nv * nv, 'DEA5: the terms of XoRp on (d, e)')
  if (xzdry) then
    call xzrel(m0)
    return
  end if
  t0 = omp_get_wtime()
  call xzdiv(w, jopa, max(npo, 1_8), opa)
  call xzdiv(w, jopb, max(npo, 1_8), opb)
  call xzdiv2(w, jvpidx, nv, nv, vpidx)
  call xzdiv(w, jvpa, max(nvp, 1_8), vpa)
  call xzdiv(w, jvpb, max(nvp, 1_8), vpb)
  call N(xzdv)(w, ka, max(nv * nv, no * nw2), av)
  call N(xzdv)(w, ed, no * nv * nvp, edv)
  call N(xzdv)(w, kg5, max(nv * nv * npo, 1_8), g5)
  call N(xzdv)(w, kx, no * no * nv * nv, x)
  call N(xzdv)(w, kxorp, max(no * no * nvp, 1_8), xorp)
  call N(xzdv)(w, kd3r, max(no * nv * nvp, 1_8), d3r)
  call N(xzdv)(w, kx3r, max(x3size, 1_8), x3r)
  call N(xzdv)(w, eovvod, nw2 * nw2, ovvo)
  call N(xzdv)(w, eooovd, no * no * nw2, ooovd)
  ! G[(m l y)][c] = - ~<ml||yh> r2[c][h]
  call N(zc_gemm)(0_4, 1_4, no * no * no, nv, nv, mone, w(eooovd), nv, w(kr2), nv, zero, w(kg), nv)
  ! G3[m][l] = r2 <ml||..>
  do m = 0, no - 1
    do l = 0, no - 1
      call N(zc_gemm)(0_4, 0_4, nv, nv, nv, one, w(kr2), nv, w(joovv + NW * (m * no + l) * nv * nv), nv, zero, &
                      w(kg3 + NW * (m * no + l) * nv * nv), nv)
    end do
  end do
  ! G2[m][c] = r2 A, A[h][g] = <cm||hg> = - ~<mc||hg>
  do m = 0, no - 1
    do c = 0, nv - 1
      !$omp parallel do private(g)
      do h = 0, nv - 1
        do g = 0, nv - 1
          if (h < g) then
            av(h * nv + g) = -edv((m * nv + c) * nvp + vpidx(g, h))
          else if (h > g) then
            av(h * nv + g) = edv((m * nv + c) * nvp + vpidx(h, g))
          else
            av(h * nv + g) = 0
          end if
        end do
      end do
      call N(zc_gemm)(0_4, 0_4, nv, nv, nv, one, w(kr2), nv, w(ka), nv, zero, w(kg2 + NW * (m * nv + c) * nv * nv), nv)
    end do
  end do
  ! G5[f][b][P] = sum_h <ln||hf> r2[b][h]
  do pq = 0, npo - 1
    call N(zc_gemm)(1_4, 1_4, nv, nv, nv, one, w(joovv + NW * (opa(pq) * no + opb(pq)) * nv * nv), nv, w(kr2), nv, zero, &
                    w(ka), nv)
    !$omp parallel do private(b)
    do a = 0, nv - 1
      do b = 0, nv - 1
        g5((a * nv + b) * npo + pq) = av(a * nv + b)
      end do
    end do
  end do
  ! r3 on (h | pair), dense
  d3r = 0
  do i = 0, no - 1
    k = blk3(i)
    if (k < 0) cycle
    do p = 0, ncls - 1
      if (off12(k, p) < 0) cycle
      q = k - p
      do h = 0, dnV(p) - 1
        d3r((i * nv + vcoff(p) + h) * nvp + vpoff(q):(i * nv + vcoff(p) + h) * nvp + vpoff(q) + dnQ(q) - 1) = &
          x3r(x3off(i) + off12(k, p) + h * dnQ(q):x3off(i) + off12(k, p) + (h + 1) * dnQ(q) - 1)
      end do
    end do
  end do
  ! XoRp of the i of this rank
  xorp = 0
  do i = 0, no - 1
    if (.not. N(xzd5own)(i)) cycle
    do m = 0, no - 1
      ! - sum_h <md||hi> r2[e][h]
      !$omp parallel do private(h)
      do c = 0, nv - 1
        do h = 0, nv - 1
          av(c * nv + h) = ovvo(((m * nv + c) * nv + h) * no + i)
        end do
      end do
      call N(zc_gemm)(0_4, 1_4, nv, nv, nv, mone, w(ka), nv, w(kr2), nv, zero, w(kx + NW * (m * no + i) * nv * nv), nv)
      ! - sum_{lg} G3[l][m][d][g] t2[i][l][e][g]
      do l = 0, no - 1
        call N(zc_gemm)(0_4, 1_4, nv, nv, nv, mone, w(kg3 + NW * (l * no + m) * nv * nv), nv, &
                        w(jt2 + NW * (i * no + l) * nv * nv), nv, one, w(kx + NW * (m * no + i) * nv * nv), nv)
      end do
      ! + sum_Q <dm||Q> r3(i; e Q), <dm||Q> = - ~<md||Q>
      call N(zc_gemm)(0_4, 1_4, nv, nv, nvp, mone, w(ed + NW * m * nv * nvp), nvp, w(kd3r + NW * i * nv * nvp), nvp, one, &
                      w(kx + NW * (m * no + i) * nv * nv), nv)
      !$omp parallel do private(a, b)
      do q = 0, nvp - 1
        a = vpa(q)
        b = vpb(q)
        xorp((m * no + i) * nvp + q) = x(((m * no + i) * nv + a) * nv + b) - x(((m * no + i) * nv + b) * nv + a)
      end do
    end do
    ! - sum_h f_mh r3(i; h Q)
    call N(zc_gemm)(0_4, 0_4, no, nvp, nv, mone, w(efov), nv, w(kd3r + NW * i * nv * nvp), nvp, one, &
                    w(kxorp + NW * i * nvp), no * nvp)
    ! + sum_{lh} <ml||hi> r3(l; h Q), <ml||hi> = - ~<ml||ih>
    !$omp parallel do private(l, h)
    do m = 0, no - 1
      do l = 0, no - 1
        do h = 0, nv - 1
          av((m * no + l) * nv + h) = ooovd(((m * no + l) * no + i) * nv + h)
        end do
      end do
    end do
    call N(zc_gemm)(0_4, 0_4, no, nvp, no * nv, mone, w(ka), no * nv, w(kd3r), nvp, one, w(kxorp + NW * i * nvp), no * nvp)
    ! + sum_l WhR(lm) t2(il; Q)
    call N(zc_gemm)(1_4, 0_4, no, nvp, no, one, w(kwhr), no, w(et2p + NW * i * no * nvp), nvp, one, &
                    w(kxorp + NW * i * nvp), no * nvp)
  end do
  td5(3) = td5(3) + omp_get_wtime() - t0
  call xzrel(m0)
end subroutine

! ---------------------------------------------------------------------------
! Z4(mf; k; cde) = sum_{lh} <ml||fh> r4(lk; h cde) (dea5_sector.Engine5._z4)
! of one k: [m][f][K] per (class of m, class of f) at kout + (z4off(k, mu, fe)
! - z4k(k)).  Made where it is used (the rows of sigma5, family D4): the
! slabs of all k (o^2 v^4 / classes) are not held.
subroutine N(xzd5z4k)(w, kr4f, k, kout)
  use xzdata
  use xzmem
  use xzkern
  use xzdead
  use xzdead5
  use omp_lib
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: kr4f, k, kout
  integer(8) :: m0, kx4, ka, l, pl, m4, ph, mu, fe, nk, mm, f, h
  integer(8), pointer :: opidx(:, :)
  ZT, pointer :: av(:), oovv(:)
  real(8) :: t0
  ZT :: one, alpha
  one = 1
  m0 = xzmark()
  kx4 = xzget(NW * max(x4max, 1_8), 'DEA5: a row of r4 on (h | cde)')
  ka = xzget(NW * max(nocmax * nvcmax * nvcmax, 1_8), 'DEA5: <ml||fh> of a block')
  if (xzdry) then
    call xzrel(m0)
    return
  end if
  t0 = omp_get_wtime()
  call xzdiv2(w, jopidx, no, no, opidx)
  call N(xzdv)(w, ka, max(nocmax * nvcmax * nvcmax, 1_8), av)
  call N(xzdv)(w, joovv, no * no * nv * nv, oovv)
  if (z4k(k + 1) > z4k(k)) call zc_vzero_d(NW * (z4k(k + 1) - z4k(k)), w(kout))
  do l = 0, no - 1
    if (l == k) cycle
    pl = opidx(max(l, k), min(l, k))
    m4 = blk4(pl)
    if (m4 < 0) cycle
    call N(xzd5x4row)(w, kr4f, pl, kx4)
    alpha = merge(1d0, -1d0, l < k)
    do ph = 0, ncls - 1
      if (off13(m4, ph) < 0) cycle
      nk = dnK(m4 - ph)
      do mu = 0, ncls - 1
        fe = mu + ocls(l) - ph
        if (dnO(mu) == 0 .or. dnV(fe) == 0) cycle
        if (z4off(k, mu, fe) < 0) cycle
        ! A[(m f)][h] = <ml||fh>
        !$omp parallel do private(f, h)
        do mm = 0, dnO(mu) - 1
          do f = 0, dnV(fe) - 1
            do h = 0, dnV(ph) - 1
              av((mm * dnV(fe) + f) * dnV(ph) + h) = &
                oovv((((ocoff(mu) + mm) * no + l) * nv + vcoff(fe) + f) * nv + vcoff(ph) + h)
            end do
          end do
        end do
        call N(zc_gemm)(0_4, 0_4, dnO(mu) * dnV(fe), nk, dnV(ph), alpha, w(ka), dnV(ph), w(kx4 + NW * off13(m4, ph)), nk, &
                        one, w(kout + NW * (z4off(k, mu, fe) - z4k(k))), nk)
      end do
    end do
  end do
  td5(4) = td5(4) + omp_get_wtime() - t0
  call xzrel(m0)
end subroutine

! XvR5 = XvR (kz, its term of Wpp once) + P(d/bc) 1/2 sum_f Wpp(bc, ef) r2(fd)
! once more (dea5_sector.Engine5._xvr5), in the layout of XvR (zoffk)
subroutine N(xzd5xvr)(w, kz, kr2, kxvr5)
  use xzdata
  use xzmem
  use xzkern
  use xzdead
  use xzdead5
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: kz, kr2, kxvr5
  integer(8) :: m0, ku, ktu, k, mu, ne, dl, q, ph, nq, e, d, bc, o, p
  integer(8), allocatable :: row(:), xoff(:, :), s1(:, :), s2(:, :)
  real(8), allocatable :: sg(:)
  ZT, pointer :: u(:), tu(:)
  ZT :: one, zero, half
  one = 1
  zero = 0
  half = 0.5d0
  m0 = xzmark()
  ku = xzget(NW * max(usize, 1_8), 'DEA5: U of XvR5')
  ktu = xzget(NW * max(nvcmax ** 3, nqmax * nvcmax * nvcmax), 'DEA5: a product of U')
  if (xzdry) then
    call xzrel(m0)
    return
  end if
  call N(xzecopy)(max(zsize, 1_8), w(kz), w(kxvr5))
  if (dn2 == 0) then
    call xzrel(m0)
    return
  end if
  call N(xzdv)(w, ku, max(usize, 1_8), u)
  call N(xzdv)(w, ktu, max(nvcmax ** 3, nqmax * nvcmax * nvcmax), tu)
  call zc_vzero_d(NW * max(usize, 1_8), w(ku))
  do k = 0, n3c - 1
    if (uoffk(k) < 0) cycle
    mu = k - ds
    ne = dnV(mu)
    do dl = 0, ncls - 1
      q = k - dl
      if (dnV(dl) == 0 .or. dnQ(q) == 0) cycle
      if (wboff(q, mu) < 0) cycle
      ph = q - mu
      nq = dnQ(q)
      call N(zc_gemm)(0_4, 0_4, nq * ne, dnV(dl), dnV(ph), one, w(dwb + NW * wboff(q, mu)), dnV(ph), &
                      w(kr2 + NW * (vcoff(ph) * nv + vcoff(dl))), nv, zero, w(ktu), dnV(dl))
      !$omp parallel do private(d, bc, o)
      do e = 0, ne - 1
        do d = 0, dnV(dl) - 1
          o = uoffk(k) + e * n12(k) + off12(k, dl) + d * nq
          do bc = 0, nq - 1
            u(o + bc) = u(o + bc) + half * tu((bc * ne + e) * dnV(dl) + d)
          end do
        end do
      end do
    end do
  end do
  do k = 0, n3c - 1
    if (uoffk(k) < 0) cycle
    mu = k - ds
    ne = dnV(mu)
    allocate(row(0:ne-1), sg(0:ne-1), xoff(0:ncls-1, 0:ne-1), s1(0:ncls-1, 0:ne-1), s2(0:ncls-1, 0:ne-1))
    do e = 0, ne - 1
      row(e) = e
      sg(e) = 1d0
      do p = 0, ncls - 1
        xoff(p, e) = -1
        s1(p, e) = 1
        s2(p, e) = 1
        if (off12(k, p) >= 0) then
          xoff(p, e) = uoffk(k) + e * n12(k) + off12(k, p)
          s1(p, e) = dnQ(k - p)
        end if
      end do
    end do
    call N(xzdx12)(w, 1, kxvr5 + NW * zoffk(k), dnK(k), k, ne, row, sg, xoff, s1, s2, ku)
    deallocate(row, sg, xoff, s1, s2)
  end do
  call xzrel(m0)
end subroutine

! ---------------------------------------------------------------------------
! The intermediates per occupied pair (j, k) (dea5_sector.Engine5._pair_imds):
! Y4(mn; jk; de) = sum_{h<g} <mn||hg> r4(jk; hg de), the terms of r4 in
! XoRp, and XoR5(m; jk; cde) but for its term of r5:
!   P(jk) sum_{lh} <ml||hj> r4(lk; h cde) + S(c|de) sum_{h<g} <cm||hg> r4(jk; hg de)
!   + 1/2 P(jk) S(c|de) sum_l G(ml; j; c) t2(kl; de) - A(cde) sum_g G2(m; c, d, g) t2(jk; eg)
!   - 1/2 P(jk) S(e|cd) sum_l t2(jl; cd) YR(ml; k; e)
!   + S(c|de) sum_{lg} [ G3(ml; c, g) t3(jkl; g de) + B4(ml g; jk; c) r3(l; g de) ]
! the products on (c | de) reduced once per class of m.  Over the ranks the
! pairs mod(pair, ranks) of this rank (summed later).
subroutine N(xzd5pairs)(w, kr4f, kr2, kx3r, kyr, kg, kg3, kg2, kxorp, kxor5, ky4)
  use xzdata
  use xzmem
  use xzkern
  use xzdead
  use xzdead5
  use xzpar, only: myrank, nrank
  use omp_lib
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: kr4f, kr2, kx3r, kyr, kg, kg3, kg2, kxorp, kxor5, ky4
  integer(8) :: m0, kx22, ku, kyd, kx4, pq, j, k, m4, kp, c, i, l, q, mu, kk, nm, n12k, p, qq, mm, m, iy, yy, zz
  integer(8) :: lq, sj, jj, kj, ph, k3, nq, a, b, d, e, pl, m4l
  integer(8), allocatable :: row(:), xoff(:, :), s1(:, :), s2(:, :)
  real(8), allocatable :: sg(:)
  integer(8), pointer :: opa(:), opb(:), opgrp(:), otri(:, :), opidx(:, :), vpa(:), vpb(:)
  integer(1), pointer :: osgn(:, :)
  ZT, pointer :: u(:), yd(:)
  ZT :: one, zero, mone, alpha
  logical :: have
  real(8) :: t0, sgh, sl
  one = 1
  zero = 0
  mone = -1
  m0 = xzmark()
  kx22 = xzget(NW * max(n22max, 1_8), 'DEA5: a row of r4 on (hg | de)')
  ku = xzget(NW * max(nocmax * maxval(n12), 1_8), 'DEA5: the products of XoR5 on (c | de)')
  kyd = xzget(NW * max(nvcmax * nv * nv, 1_8), 'DEA5: G2 t2')
  kx4 = xzget(NW * max(x4max, 1_8), 'DEA5: a row of r4 on (h | cde)')
  if (xzdry) then
    call xzrel(m0)
    return
  end if
  t0 = omp_get_wtime()
  call xzdiv(w, jopa, max(npo, 1_8), opa)
  call xzdiv(w, jopb, max(npo, 1_8), opb)
  call xzdiv(w, jopgrp, max(npo, 1_8), opgrp)
  call xzdiv2(w, jotri, max(npo, 1_8), no, otri)
  call xzdi1v2(w, josgn, max(npo, 1_8), no, osgn)
  call xzdiv2(w, jopidx, no, no, opidx)
  call xzdiv(w, jvpa, max(nvp, 1_8), vpa)
  call xzdiv(w, jvpb, max(nvp, 1_8), vpb)
  call N(xzdv)(w, kyd, max(nvcmax * nv * nv, 1_8), yd)
  call zc_vzero_d(NW * max(xr5size, 1_8), w(kxor5))
  call zc_vzero_d(NW * max(y4size, 1_8), w(ky4))
  allocate(row(0:nocmax), sg(0:nocmax), xoff(0:ncls-1, 0:nocmax), s1(0:ncls-1, 0:nocmax), s2(0:ncls-1, 0:nocmax))
  do pq = 0, npo - 1
    if (ddist .and. mod(pq, int(nrank, 8)) /= myrank) cycle
    j = opa(pq)
    k = opb(pq)
    m4 = blk4(pq)
    kp = opgrp(pq)
    have = .false.
    if (m4 >= 0) have = n22(m4) > 0
    if (have) then
      ! r4 of the row on (hg | de)
      call zc_vzero_d(NW * n22(m4), w(kx22))
      call x22row()
      ! Y4 = <(mn) of c||(hg) of c> r4(jk; hg de)
      do c = 0, n2c - 1
        if (y4off(pq, c) < 0) cycle
        call N(zc_gemm)(0_4, 0_4, dnO2(c), dnQ(m4 - c), dnQ(c), one, w(jgpp + NW * (opoff(c) * nvp + vpoff(c))), nvp, &
                        w(kx22 + NW * off22(m4, c)), dnQ(m4 - c), zero, w(ky4 + NW * y4off(pq, c)), dnQ(m4 - c))
      end do
      ! XoRp: - sum_l sum_{h<g} <ml||hg> r4(il; hg de), (i, l) = (j, k) and (k, j)
      do iy = 1, 2
        if (iy == 1) then
          i = j
          l = k
          alpha = -1
        else
          i = k
          l = j
          alpha = 1
        end if
        do q = 0, n2c - 1
          if (off22(m4, q) < 0) cycle
          mu = q - ocls(l)
          if (mu < 0 .or. mu >= ncls) cycle
          if (dnO(mu) == 0) cycle
          call N(zc_gemm)(0_4, 0_4, dnO(mu), dnQ(m4 - q), dnQ(q), alpha, w(joovvp + NW * ((ocoff(mu) * no + l) * nvp + vpoff(q))), &
                          no * nvp, w(kx22 + NW * off22(m4, q)), dnQ(m4 - q), one, &
                          w(kxorp + NW * ((ocoff(mu) * no + i) * nvp + vpoff(m4 - q))), no * nvp)
        end do
      end do
    end if
    ! XoR5 rows (m of class mu, jk): the products on (c | de)
    do mu = 0, ncls - 1
      nm = dnO(mu)
      if (nm == 0) cycle
      kk = ds + kp - mu
      if (dnK(kk) == 0) cycle
      n12k = n12(kk)
      if (n12k == 0) cycle
      call N(xzdv)(w, ku, nm * n12k, u)
      call zc_vzero_d(NW * nm * n12k, w(ku))
      do p = 0, ncls - 1
        if (off12(kk, p) < 0) cycle
        q = kk - p
        nq = dnQ(q)
        do mm = 0, nm - 1
          m = ocoff(mu) + mm
          ! (b) sum_{h<g} <cm||hg> r4(jk; hg de), <cm||Q> = - ~<mc||Q>
          if (have) then
            qq = m4 - q
            if (qq >= 0 .and. qq < n2c) then
              if (off22(m4, qq) >= 0) &
                call N(zc_gemm)(0_4, 0_4, dnV(p), nq, dnQ(qq), mone, w(ed + NW * ((m * nv + vcoff(p)) * nvp + vpoff(qq))), nvp, &
                                w(kx22 + NW * off22(m4, qq)), nq, one, w(ku + NW * (mm * n12k + off12(kk, p))), nq)
            end if
          end if
          ! (c) 1/2 P(jk) sum_l G(ml; y; c) t2(zl; Q);  (g) - 1/2 P(jk) sum_l YR(ml; z; c) t2(yl; Q)
          do iy = 1, 2
            if (iy == 1) then
              yy = j
              zz = k
              sgh = 0.5d0
            else
              yy = k
              zz = j
              sgh = -0.5d0
            end if
            alpha = sgh
            call N(zc_gemm)(1_4, 0_4, dnV(p), nq, no, alpha, w(kg + NW * ((m * no * no + yy) * nv + vcoff(p))), no * nv, &
                            w(et2p + NW * (zz * no * nvp + vpoff(q))), nvp, one, w(ku + NW * (mm * n12k + off12(kk, p))), nq)
            alpha = -sgh
            call N(zc_gemm)(1_4, 0_4, dnV(p), nq, no, alpha, w(kyr + NW * ((m * no * no + zz) * nv + vcoff(p))), no * nv, &
                            w(et2p + NW * (yy * no * nvp + vpoff(q))), nvp, one, w(ku + NW * (mm * n12k + off12(kk, p))), nq)
          end do
          ! (h) sum_{lg} G3(ml; c, g) t3(jkl; g Q);  (i) sum_{lg} B4(ml g; jk; c) r3(l; g Q)
          do l = 0, no - 1
            if (osgn(pq, l) /= 0) then
              jj = otri(pq, l)
              kj = xzd5tgrp(jj)
              ph = kj - q
              if (ph >= 0 .and. ph < ncls) then
                if (t12off(kj, ph) >= 0) then
                  alpha = osgn(pq, l)
                  call N(zc_gemm)(0_4, 0_4, dnV(p), nq, dnV(ph), alpha, &
                                  w(kg3 + NW * (((m * no + l) * nv + vcoff(p)) * nv + vcoff(ph))), nv, &
                                  w(dt12 + NW * (t12off(kj, ph) + (jj - otoff(kj)) * nq)), dnJ(kj) * nq, one, &
                                  w(ku + NW * (mm * n12k + off12(kk, p))), nq)
                end if
              end if
            end if
            ph = ds + ocls(l) - q
            k3 = blk3(l)
            if (ph >= 0 .and. ph < ncls .and. k3 >= 0) then
              if (off12(k3, ph) >= 0 .and. dnQ(k3 - ph) == nq) then
                call N(zc_gemm)(1_4, 0_4, dnV(p), nq, dnV(ph), one, &
                                w(db4 + NW * (((pq * no + m) * no + l) * nv * nv + vcoff(ph) * nv + vcoff(p))), nv, &
                                w(kx3r + NW * (x3off(l) + off12(k3, ph))), nq, one, w(ku + NW * (mm * n12k + off12(kk, p))), nq)
              end if
            end if
          end do
          ! (d, e) - A(cde) sum_g G2(m; c, d, g) t2(jk; e g): Yd[c][d][e], - [Yd(c, d, e) - Yd(c, e, d)] on (c | de)
          call N(zc_gemm)(0_4, 1_4, dnV(p) * nv, nv, nv, one, w(kg2 + NW * (m * nv + vcoff(p)) * nv * nv), nv, &
                          w(jt2 + NW * (j * no + k) * nv * nv), nv, zero, w(kyd), nv)
          !$omp parallel do private(lq, a, b)
          do c = 0, dnV(p) - 1
            do lq = 0, nq - 1
              a = vpa(vpoff(q) + lq)
              b = vpb(vpoff(q) + lq)
              u(mm * n12k + off12(kk, p) + c * nq + lq) = u(mm * n12k + off12(kk, p) + c * nq + lq) - &
                (yd((c * nv + a) * nv + b) - yd((c * nv + b) * nv + a))
            end do
          end do
        end do
      end do
      ! reduced over (c | de) into the rows (m of mu, jk)
      do mm = 0, nm - 1
        row(mm) = mm
        sg(mm) = 1d0
        do p = 0, ncls - 1
          xoff(p, mm) = -1
          s1(p, mm) = 1
          s2(p, mm) = 1
          if (off12(kk, p) >= 0) then
            xoff(p, mm) = mm * n12k + off12(kk, p)
            s1(p, mm) = dnQ(kk - p)
          end if
        end do
      end do
      call N(xzdx12)(w, 1, kxor5 + NW * xr5off(ocoff(mu), pq), dnK(kk), kk, nm, row, sg, xoff, s1, s2, ku)
    end do
    ! (a) P(jk) sum_{lh} <ml||hy> r4(lz; h cde), (y, z) = (j, k) and (k, j)
    do iy = 1, 2
      if (iy == 1) then
        yy = j
        zz = k
        sgh = 1d0
      else
        yy = k
        zz = j
        sgh = -1d0
      end if
      do l = 0, no - 1
        if (l == zz) cycle
        pl = opidx(max(l, zz), min(l, zz))
        m4l = blk4(pl)
        if (m4l < 0) cycle
        sl = merge(1d0, -1d0, l < zz)
        call N(xzd5x4row)(w, kr4f, pl, kx4)
        do ph = 0, ncls - 1
          if (off13(m4l, ph) < 0) cycle
          mu = ph + ocls(yy) - ocls(l)
          if (mu < 0 .or. mu >= ncls) cycle
          if (dnO(mu) == 0) cycle
          if (xr5off(ocoff(mu), pq) < 0) cycle
          kk = m4l - ph
          alpha = -sgh * sl
          ! <ml||hy> = - ~<ml||yh>
          call N(zc_gemm)(0_4, 0_4, dnO(mu), dnK(kk), dnV(ph), alpha, w(eooovd + NW * (((ocoff(mu) * no + l) * no + yy) * nv + &
                          vcoff(ph))), no * no * nv, w(kx4 + NW * off13(m4l, ph)), dnK(kk), one, &
                          w(kxor5 + NW * xr5off(ocoff(mu), pq)), dnK(kk))
        end do
      end do
    end do
  end do
  deallocate(row, sg, xoff, s1, s2)
  td5(5) = td5(5) + omp_get_wtime() - t0
  call xzrel(m0)

contains

  subroutine x22row()
    integer(8) :: one_row(1), xo(npg), y1(npg), y2(npg), qx
    real(8) :: one_sg(1)
    one_row(1) = pos4(pq)
    one_sg(1) = 1d0
    do qx = 0, npg - 1
      xo(qx + 1) = off22(m4, qx)
      y1(qx + 1) = max(dnQ(m4 - qx), 1_8)
      y2(qx + 1) = 1
    end do
    call N(xzdx22)(w, 0, kr4f + NW * off4(m4), nqa(m4), m4, 1_8, one_row, one_sg, xo, y1, y2, kx22)
  end subroutine
end subroutine

! ---------------------------------------------------------------------------
! The pp ladder of the 5p3h row: per block the rows of this rank in batches,
! r5 on [(fg)][(row, (cde))] per group of (fg) (zc_t5s_x23), one product with
! the block of Wq5, reduced; from the same view the thin products per row:
! XoR5 (its term of r5), sigma3, sigma4 on (a | bcd) (X4s).
subroutine N(xzd5pp)(w, kr5v, ks5v, kxor5, ks3x, kx4s, x4sloc)
  use xzdata
  use xzmem
  use xzkern
  use xzdead
  use xzdead5
  use omp_lib
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: kr5v, ks5v, kxor5, ks3x, kx4s, x4sloc(0:*)
  integer(8) :: m0, kt, kx, mm, c, nr, ra, rb, nbm, need, r0, r1, nb, r, q, o, nc, i, rr, n_, pjk, mu, p, m4, jj, kk
  integer(8), allocatable :: row(:), xoff(:, :), s1(:, :), s2(:, :)
  real(8), allocatable :: sg(:)
  integer(8), pointer :: opa(:), opb(:), opgrp(:), ofree(:, :), opair(:, :)
  real(8) :: t0, ta, sgn
  ZT :: one, zero, alpha
  one = 1
  zero = 0
  need = 1
  do mm = 0, n5c - 1
    if (.not. inblk5(mm) .or. n23(mm) == 0) cycle
    call xzd5own(mm, ra, rb)
    nr = rb - ra
    if (nr == 0) cycle
    nbm = min(nr, max(1_8, dbatch / n23(mm)))
    need = max(need, nbm * n23(mm))
  end do
  m0 = xzmark()
  kt = xzget(NW * need, 'DEA5: a batch of r5 on (fg | cde)')
  kx = xzget(NW * need, 'DEA5: its product')
  if (xzdry) then
    call xzrel(m0)
    return
  end if
  t0 = omp_get_wtime()
  call xzdiv(w, jopa, max(npo, 1_8), opa)
  call xzdiv(w, jopb, max(npo, 1_8), opb)
  call xzdiv(w, jopgrp, max(npo, 1_8), opgrp)
  call xzdiv2(w, jofree, max(nto, 1_8), 3_8, ofree)
  call xzdiv2(w, jopair, max(nto, 1_8), 3_8, opair)
  do mm = 0, n5c - 1
    if (.not. inblk5(mm) .or. n23(mm) == 0) cycle
    c = mm - ds
    call xzd5own(mm, ra, rb)
    nr = rb - ra
    if (nr == 0) cycle
    nbm = min(nr, max(1_8, dbatch / n23(mm)))
    do r0 = ra, rb - 1, nbm
      r1 = min(rb, r0 + nbm)
      nb = r1 - r0
      allocate(row(0:nb-1), sg(0:nb-1), xoff(0:npg-1, 0:nb-1), s1(0:npg-1, 0:nb-1), s2(0:npg-1, 0:nb-1))
      do r = 0, nb - 1
        row(r) = r0 + r
        sg(r) = 1d0
        do q = 0, npg - 1
          xoff(q, r) = -1
          s1(q, r) = 1
          s2(q, r) = 1
          if (off23(mm, q) >= 0) then
            xoff(q, r) = off23(mm, q) * nb + r * dnK(mm - q)
            s1(q, r) = dnK(mm - q) * nb
          end if
        end do
      end do
      call zc_vzero_d(NW * nb * n23(mm), w(kt))
      ta = omp_get_wtime()
      call N(xzd5x23)(w, 0, kr5v + NW * off5(mm), nq5(mm), mm, nb, row, sg, xoff, s1, s2, kt)
      td5(17) = td5(17) + omp_get_wtime() - ta
      ! the thin products, per row and split (n_ | jk)
      do q = 0, npg - 1
        o = off23(mm, q)
        if (o < 0) cycle
        nc = dnK(mm - q)
        do r = r0, r1 - 1
          i = otoff(c) + r
          do rr = 0, 2
            n_ = ofree(i, rr)
            pjk = opair(i, rr)
            sgn = merge(-1d0, 1d0, rr == 1)
            ! XoR5(m; jk; cde) += sum_{(ef)} <m n_||ef> r5(n_ jk; ef cde)
            mu = q - ocls(n_)
            if (mu >= 0 .and. mu < ncls) then
              if (dnO(mu) > 0 .and. xr5off(ocoff(mu), pjk) >= 0) then
                alpha = sgn
                call N(zc_gemm)(0_4, 0_4, dnO(mu), nc, dnQ(q), alpha, w(joovvp + NW * ((ocoff(mu) * no + n_) * nvp + vpoff(q))), &
                                no * nvp, w(kt + NW * (o * nb + (r - r0) * nc)), nb * nc, one, &
                                w(kxor5 + NW * xr5off(ocoff(mu), pjk)), nc)
              end if
            end if
            ! sigma3(n_; abc) += sum_{e<f} <jk||ef> r5(n_ jk; ef abc)
            kk = mm - q
            if (q == opgrp(pjk) .and. blk3(n_) == kk) then
              alpha = sgn
              call N(zc_gemm)(0_4, 0_4, 1_8, nc, dnQ(q), alpha, w(joovvp + NW * ((opa(pjk) * no + opb(pjk)) * nvp + vpoff(q))), &
                              nvp, w(kt + NW * (o * nb + (r - r0) * nc)), nb * nc, one, &
                              w(ks3x + NW * (off3(kk) + pos3r(n_) * nc)), nc)
            end if
            ! sigma4(jk; a bcd) -= S(a|bcd) sum_{e<f} <a n_||ef> r5(n_ jk; ef bcd), on (a | bcd); <a n_||Q> = - ~<n_ a||Q>
            p = q - ocls(n_)
            m4 = blk4(pjk)
            if (p >= 0 .and. p < ncls .and. m4 >= 0) then
              if (dnV(p) > 0 .and. off13(m4, p) >= 0 .and. x4sloc(pjk) >= 0) then
                alpha = -sgn
                call N(zc_gemm)(0_4, 0_4, dnV(p), nc, dnQ(q), alpha, w(ed + NW * ((n_ * nv + vcoff(p)) * nvp + vpoff(q))), nvp, &
                                w(kt + NW * (o * nb + (r - r0) * nc)), nb * nc, one, &
                                w(kx4s + NW * (x4sloc(pjk) + off13(m4, p))), nc)
              end if
            end if
          end do
        end do
      end do
      if (xzdwant('pp')) then
        ta = omp_get_wtime()
        do q = 0, npg - 1
          o = off23(mm, q)
          if (o < 0) cycle
          nc = dnK(mm - q)
          call N(zc_gemm)(0_4, 0_4, dnQ(q), nb * nc, dnQ(q), one, w(dwq5 + NW * vvoff(q)), dnQ(q), &
                          w(kt + NW * o * nb), nb * nc, zero, w(kx + NW * o * nb), nb * nc)
        end do
        td5(18) = td5(18) + omp_get_wtime() - ta
        ta = omp_get_wtime()
        call N(xzd5x23)(w, 1, ks5v + NW * off5(mm), nq5(mm), mm, nb, row, sg, xoff, s1, s2, kx)
        td5(17) = td5(17) + omp_get_wtime() - ta
      end if
      deallocate(row, sg, xoff, s1, s2)
    end do
  end do
  td5(6) = td5(6) + omp_get_wtime() - t0
  call xzrel(m0)
end subroutine
