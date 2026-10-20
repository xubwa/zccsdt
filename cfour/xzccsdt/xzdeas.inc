!
! EOM-DEA-CCSDT on the sector itself: the sigma vector (type-generic; see
! xzdea.inc and dea_sector.Engine.sigma, of which every step below is the
! counterpart, in the same order).
!

! ---------------------------------------------------------------------------
! out[r][c - c0] = sg * src[rows(r)][c], c0 <= c < c1 (zc_tn_rows with one sign)
subroutine N(xzdgrow)(w, ksrc, ld, n, rows, sg, c0, c1, kout, ldo)
  use xzkern
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: ksrc, ld, n, rows(*), c0, c1, kout, ldo
  integer, intent(in) :: sg
  integer(1) :: s(max(n, 1_8))
  if (n <= 0) return
  s = int(sg, 1)
  call N(zc_tn_rows)(w(ksrc), ld, rows, s, n, c0, c1, w(kout), ldo)
end subroutine

! ---------------------------------------------------------------------------
! The sigma vector of the sector vector at kv into ks ([r2 | r3 | r4] of the
! problem in hand, xzdsec), the masks of the parity applied (N(xzdproj)).
! The pieces in the order of dea_sector.Engine.sigma: the small intermediates,
! the pp ladder with its thin products, the ring with its thin products, the
! hh ladder, the rows of sigma4, the 3p1h and the 2p rows.
subroutine N(xzdsig)(w, kv, ks)
  use xzdata
  use xzmem
  use xzkern
  use xzdead
  use xzpar, only: myrank, xzmpi_bcast_w, xzmpi_slices_put, xzmpi_allsum_w, xzmpi_sum0_w
  use xzeomd, only: eswoff
  use omp_lib
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: kv, ks
  integer(8) :: m0, kr2, kx3r, kx3s, kht, kwhr, kyr, khm, ky3, kwor, kq3, kxor, kzr, ku, kz, ktu, k2, k3, k4
  integer(8) :: s2, s3, s4
  logical, allocatable :: zrhave(:, :)
  real(8) :: t0, t1
  k2 = kv
  k3 = kv + NW * dn2
  k4 = kv + NW * (dn2 + dsz3)
  s2 = ks
  s3 = ks + NW * dn2
  s4 = ks + NW * (dn2 + dsz3)
  m0 = xzmark()
  kr2 = xzget(NW * nv * nv, 'DEA: r2 on all pairs')
  kx3r = xzget(NW * x3size, 'DEA: r3 on (a | bc)')
  kx3s = xzget(NW * x3size, 'DEA: sigma3 on (a | bc)')
  kht = xzget(NW * no * nv, 'DEA: h')
  kwhr = xzget(NW * no * no, 'DEA: WhR')
  kyr = xzget(NW * no * no * no * nv, 'DEA: YR')
  khm = xzget(NW * no * no * nv * nv, 'DEA: H')
  ky3 = xzget(NW * no * nv, 'DEA: Y3')
  kwor = xzget(NW * no * nv, 'DEA: WoR')
  kq3 = xzget(NW * no * nv, 'DEA: Q3')
  kxor = xzget(NW * no * no * nvp, 'DEA: XoR')
  kzr = xzget(NW * zrsize, 'DEA: ZR')
  ku = xzget(NW * usize, 'DEA: U')
  kz = xzget(NW * zsize, 'DEA: XvR')
  ktu = xzget(NW * max(nvcmax ** 3, nqmax * nvcmax * nvcmax), 'DEA: a product of U')
  allocate(zrhave(0:no-1, 0:ncls-1))
  zrhave = .false.
  if (.not. xzdry) then
    t0 = omp_get_wtime()
    ! (over the ranks: r2, r3 of rank 0 and the rows of r4 of every rank to
    ! all ranks; every rank makes the rows of sigma4 of its pairs)
    if (ddist) then
      call xzmpi_bcast_w(w(kv), NW * (dn2 + dsz3))
      call xzmpi_slices_put(w(k4), eswoff)
      tdpart(12) = tdpart(12) + omp_get_wtime() - t0
    end if
    call zc_vzero_d(NW * (dn2 + dsz3 + dsz4), w(ks))
    call N(xzdsmall)(w, k2, k3, kr2, kx3r, kx3s, kht, kwhr, kyr, khm, ky3, kwor, kq3, kxor, kzr, zrhave, ku, kz, ktu)
    ! (the thin products of the pp ladder of every rank are summed into XoR below)
    if (ddist .and. myrank > 0) call zc_vzero_d(NW * no * no * nvp, w(kxor))
    t1 = omp_get_wtime()
    tdpart(1) = tdpart(1) + t1 - t0
  end if
  call N(xzdpp)(w, k4, s2, s4, kx3s, kxor)
  if (ddist .and. .not. xzdry) then
    t0 = omp_get_wtime()
    call xzmpi_allsum_w(w(kxor), NW * no * no * nvp)
    call xzmpi_sum0_w(w(kx3s), NW * x3size)
    if (dn2 > 0) call xzmpi_sum0_w(w(s2), NW * dn2)
    tdpart(12) = tdpart(12) + omp_get_wtime() - t0
  end if
  call N(xzdring)(w, k4, s3, s4, kz)
  if (ddist .and. .not. xzdry) then
    t0 = omp_get_wtime()
    call xzmpi_bcast_w(w(kz), NW * zsize)
    tdpart(12) = tdpart(12) + omp_get_wtime() - t0
  end if
  if (.not. xzdry .and. xzdwant('hh')) then
    t0 = omp_get_wtime()
    call N(xzdhh)(w, k4, s4)
    tdpart(4) = tdpart(4) + omp_get_wtime() - t0
  end if
  call N(xzdrows)(w, k3, kr2, kx3r, kxor, kzr, zrhave, kz, kwor, kyr, s4)
  ! (the 3p1h and 2p rows: rank 0)
  if (.not. xzdry .and. (.not. ddist .or. myrank == 0)) then
    t0 = omp_get_wtime()
    call N(xzdrow3)(w, k3, kr2, kx3r, kx3s, kq3, kwhr, s3)
    t1 = omp_get_wtime()
    tdpart(6) = tdpart(6) + t1 - t0
  end if
  if (.not. ddist .or. myrank == 0) call N(xzdrow2)(w, k2, kx3r, s2)
  if (.not. xzdry .and. (.not. ddist .or. myrank == 0)) tdpart(7) = tdpart(7) + omp_get_wtime() - t1
  call N(xzdproj)(w, ks)
  if (.not. xzdry) ndeasig = ndeasig + 1
  deallocate(zrhave)
  call xzrel(m0)
end subroutine


! ---------------------------------------------------------------------------
! The small intermediates of a sigma vector: r2 on all pairs, r3 on (a | bc),
! h, WhR, YR, H, Y3, WoR, Q3, XoR without its 4p2h term, ZR, U and XvR
! without its 4p2h term.
subroutine N(xzdsmall)(w, k2, k3, kr2, kx3r, kx3s, kht, kwhr, kyr, khm, ky3, kwor, kq3, kxor, kzr, zrhave, ku, kz, ktu)
  use xzdata
  use xzkern
  use xzdead
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: k2, k3, kr2, kx3r, kx3s, kht, kwhr, kyr, khm, ky3, kwor, kq3, kxor, kzr, ku, kz, ktu
  logical, intent(inout) :: zrhave(0:no-1, 0:ncls-1)
  ZT, pointer :: r2v(:), r2(:), ht(:), yr(:), hm(:), y3(:), wor(:), q3(:), xor(:), fov(:), tu(:), u(:)
  integer(8), pointer :: vpa(:), vpb(:)
  integer(8) :: t, q, a, b, k, c, nr, r, p, y, m, n, e, d, ph, mu, dl, kk, qq, ne, nq, bc, o
  integer(8), allocatable :: row(:), xoff(:, :), s1(:, :), s2(:, :)
  real(8), allocatable :: sg(:)
  ZT :: one, zero, mone, half, s
  one = 1
  zero = 0
  mone = -1
  half = 0.5d0
  call N(xzdv)(w, k2, dn2, r2v)
  call N(xzdv)(w, kr2, nv * nv, r2)
  call N(xzdv)(w, kht, no * nv, ht)
  call N(xzdv)(w, kyr, no * no * no * nv, yr)
  call N(xzdv)(w, khm, no * no * nv * nv, hm)
  call N(xzdv)(w, ky3, no * nv, y3)
  call N(xzdv)(w, kwor, no * nv, wor)
  call N(xzdv)(w, kq3, no * nv, q3)
  call N(xzdv)(w, kxor, no * no * nvp, xor)
  call N(xzdv)(w, efov, no * nv, fov)
  call N(xzdv)(w, ktu, max(nvcmax ** 3, nqmax * nvcmax * nvcmax), tu)
  call N(xzdv)(w, ku, usize, u)
  call xzdiv(w, jvpa, nvp, vpa)
  call xzdiv(w, jvpb, nvp, vpb)
  ! r2 on all pairs
  r2(0:nv*nv-1) = 0
  do t = 0, dn2 - 1
    q = vpoff(ds) + t
    a = vpa(q)
    b = vpb(q)
    r2(a * nv + b) = r2v(t)
    r2(b * nv + a) = -r2v(t)
  end do
  ! r3 on (a | bc), row after row
  call zc_vzero_d(NW * x3size, w(kx3r))
  call zc_vzero_d(NW * x3size, w(kx3s))
  do k = 0, n3c - 1
    c = k - ds
    nr = off3(k + 1) - off3(k)
    if (nr == 0) cycle
    nr = nr / dnK(k)
    allocate(row(0:nr-1), sg(0:nr-1), xoff(0:ncls-1, 0:nr-1), s1(0:ncls-1, 0:nr-1), s2(0:ncls-1, 0:nr-1))
    do r = 0, nr - 1
      row(r) = r
      sg(r) = 1d0
      do p = 0, ncls - 1
        xoff(p, r) = -1
        s1(p, r) = 1
        s2(p, r) = 1
        if (off12(k, p) >= 0) then
          xoff(p, r) = x3off(ocoff(c) + r) + off12(k, p)
          s1(p, r) = dnQ(k - p)
        end if
      end do
    end do
    call N(xzdx12)(w, 0, k3 + NW * off3(k), dnK(k), k, nr, row, sg, xoff, s1, s2, kx3r)
    deallocate(row, sg, xoff, s1, s2)
  end do
  ! h[c][m] = sum_{e<f} <cm||ef> r2(ef) (as ht[m][c]); WhR[m][n] = sum_{e<f} <mn||ef> r2(ef)
  if (dn2 > 0) then
    call N(zc_gemm)(0_4, 0_4, no * nv, 1_8, dn2, mone, w(ed + NW * vpoff(ds)), nvp, w(k2), 1_8, zero, w(kht), 1_8)
    call N(zc_gemm)(0_4, 0_4, no * no, 1_8, dn2, one, w(joovvp + NW * vpoff(ds)), nvp, w(k2), 1_8, zero, w(kwhr), 1_8)
  else
    call zc_vzero_d(NW * no * nv, w(kht))
    call zc_vzero_d(NW * no * no, w(kwhr))
  end if
  ! YR(mn; y; d) = sum_{e<f} <mn||ef> r3(y; efd), H[y][m][d][c] = sum_{e<f} <md||ef> r3(y; ef c)
  call zc_vzero_d(NW * no * no * no * nv, w(kyr))
  call zc_vzero_d(NW * no * no * nv * nv, w(khm))
  do y = 0, no - 1
    k = blk3(y)
    if (k < 0) cycle
    do p = 0, ncls - 1
      if (off12(k, p) < 0) cycle
      q = k - p
      call N(zc_gemm)(0_4, 1_4, no * no, dnV(p), dnQ(q), one, w(joovvp + NW * vpoff(q)), nvp, &
                      w(kx3r + NW * (x3off(y) + off12(k, p))), dnQ(q), zero, w(kyr + NW * (y * nv + vcoff(p))), no * nv)
      call N(zc_gemm)(0_4, 1_4, no * nv, dnV(p), dnQ(q), one, w(ed + NW * vpoff(q)), nvp, &
                      w(kx3r + NW * (x3off(y) + off12(k, p))), dnQ(q), zero, &
                      w(khm + NW * (y * no * nv * nv + vcoff(p))), nv)
    end do
  end do
  ! Y3[m][c] = sum_n YR(mn; n; c), WoR, Q3
  do m = 0, no - 1
    do c = 0, nv - 1
      s = 0
      do n = 0, no - 1
        s = s + yr(((m * no + n) * no + n) * nv + c)
      end do
      y3(m * nv + c) = s
    end do
  end do
  do m = 0, no - 1
    do d = 0, nv - 1
      s = -ht(m * nv + d) - y3(m * nv + d)
      do e = 0, nv - 1
        s = s - fov(m * nv + e) * r2(d * nv + e)
      end do
      wor(m * nv + d) = s
      q3(m * nv + d) = y3(m * nv + d) + ht(m * nv + d)
    end do
  end do
  ! XoR(m; j; cd) without its 4p2h term
  call zc_vzero_d(NW * no * no * nvp, w(kxor))
  do n = 0, no - 1
    k = blk3(n)
    if (k < 0) cycle
    do p = 0, ncls - 1
      if (off12(k, p) < 0) cycle
      q = k - p
      do m = 0, no - 1
        ! sum_e <mn||ej> r3(n; e cd), <mn||ej> = - ~<mn||je>
        call N(zc_gemm)(0_4, 0_4, no, dnQ(q), dnV(p), mone, w(eooovd + NW * ((m * no + n) * no * nv + vcoff(p))), nv, &
                        w(kx3r + NW * (x3off(n) + off12(k, p))), dnQ(q), one, w(kxor + NW * (m * no * nvp + vpoff(q))), nvp)
      end do
    end do
  end do
  call N(zc_gemm)(0_4, 0_4, no, no * nvp, no, half, w(kwhr), no, w(et2p), no * nvp, one, w(kxor), no * nvp)
  !$omp parallel do collapse(2) private(q, a, b)
  do m = 0, no - 1
    do y = 0, no - 1
      do q = 0, nvp - 1
        a = vpa(q)
        b = vpb(q)
        xor((m * no + y) * nvp + q) = xor((m * no + y) * nvp + q) + hm(((y * no + m) * nv + b) * nv + a) &
                                      - hm(((y * no + m) * nv + a) * nv + b)
      end do
    end do
  end do
  ! ZR(nf; ab) = sum_{me} <mn||ef> r3(m; eab)
  call zc_vzero_d(NW * zrsize, w(kzr))
  zrhave = .false.
  do m = 0, no - 1
    k = blk3(m)
    if (k < 0) cycle
    do p = 0, ncls - 1
      if (off12(k, p) < 0) cycle
      q = k - p
      do n = 0, no - 1
        ph = ocls(m) + ocls(n) - p
        if (n == m .or. dnV(ph) == 0) cycle
        if (zroff(n, ph) < 0) cycle
        call N(zc_gemm)(1_4, 0_4, dnV(ph), dnQ(q), dnV(p), one, &
                        w(joovv + NW * (((m * no + n) * nv + vcoff(p)) * nv + vcoff(ph))), nv, &
                        w(kx3r + NW * (x3off(m) + off12(k, p))), dnQ(q), one, w(kzr + NW * zroff(n, ph)), dnQ(q))
        zrhave(n, ph) = .true.
      end do
    end do
  end do
  ! U(e; d, (bc)) and XvR(e; bcd) = P(d/bc) U, without its 4p2h term
  call zc_vzero_d(NW * usize, w(ku))
  call zc_vzero_d(NW * zsize, w(kz))
  do m = 0, no - 1
    kk = blk3(m)
    if (kk < 0) cycle
    do ph = 0, ncls - 1
      if (off12(kk, ph) < 0) cycle
      qq = kk - ph
      do mu = 0, ncls - 1
        dl = mu + ph - ocls(m)
        k = dl + qq
        if (dnV(dl) == 0 .or. dnV(mu) == 0 .or. k < 0 .or. k >= n3c) cycle
        if (uoffk(k) < 0 .or. vooff(m, mu, dl) < 0) cycle
        ! sum_f <dm||ef> r3(m; f bc) for every e: [d][(bc)] at U[e] + off12(K, d)
        do e = 0, dnV(mu) - 1
          call N(zc_gemm)(0_4, 0_4, dnV(dl), dnQ(qq), dnV(ph), one, &
                          w(dvo + NW * (vooff(m, mu, dl) + e * dnV(dl) * dnV(ph))), dnV(ph), &
                          w(kx3r + NW * (x3off(m) + off12(kk, ph))), dnQ(qq), one, &
                          w(ku + NW * (uoffk(k) + e * n12(k) + off12(k, dl))), dnQ(qq))
        end do
      end do
    end do
  end do
  if (dn2 > 0) then
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
        ! x[(bc), e, d] = sum_f Wpp(bc, ef) r2(f, d), then U[e][d][(bc)] += x / 2
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
  end if
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
    call N(xzdx12)(w, 1, kz + NW * zoffk(k), dnK(k), k, ne, row, sg, xoff, s1, s2, ku)
    deallocate(row, sg, xoff, s1, s2)
  end do
end subroutine

! ---------------------------------------------------------------------------
! The pp ladder: per block M the rows in batches, r4 on [(ef)][(row, (cd))]
! per group of (ef), one product with the block of Wq4; from the same view the
! 4p2h terms of sigma2, of sigma3 (<cm||ef>) and of XoR (thin products per row).
subroutine N(xzdpp)(w, k4, ks2, ks4, kx3s, kxor)
  use xzdata
  use xzmem
  use xzkern
  use xzdead
  use omp_lib
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: k4, ks2, ks4, kx3s, kxor
  integer(8) :: m0, kt, kx, mm, c, nr, na, nbm, need, r0, r1, nb, r, q, o, nc, pq, u, v, ncol, a, b, p, ka, mu, ra, rb
  integer(8), allocatable :: row(:), xoff(:, :), s1(:, :), s2(:, :)
  real(8), allocatable :: sg(:)
  real(8) :: t0, ta, tb
  logical :: wpp
  integer(8), pointer :: opa(:), opb(:)
  ZT :: one, zero, alpha
  one = 1
  zero = 0
  ! the buffers: the largest batch of any block (of the rows of this rank)
  need = 1
  do mm = 0, n4c - 1
    if (.not. inblk4(mm) .or. n22(mm) == 0) cycle
    call xzdown(mm, ra, rb)
    nr = rb - ra
    if (nr == 0) cycle
    nbm = min(nr, max(1_8, dbatch / n22(mm)))
    need = max(need, nbm * n22(mm))
  end do
  m0 = xzmark()
  kt = xzget(NW * need, 'DEA: a batch of r4 on two pairs')
  kx = xzget(NW * need, 'DEA: its product')
  if (xzdry) then
    call xzrel(m0)
    return
  end if
  t0 = omp_get_wtime()
  wpp = xzdwant('pp')
  call xzdiv(w, jopa, npo, opa)
  call xzdiv(w, jopb, npo, opb)
  do mm = 0, n4c - 1
    if (.not. inblk4(mm) .or. n22(mm) == 0) cycle
    c = mm - ds
    call xzdown(mm, ra, rb)
    nr = rb - ra
    if (nr == 0) cycle
    na = nqa(mm)
    nbm = min(nr, max(1_8, dbatch / n22(mm)))
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
          if (off22(mm, q) >= 0) then
            xoff(q, r) = off22(mm, q) * nb + r * dnQ(mm - q)
            s1(q, r) = dnQ(mm - q) * nb
          end if
        end do
      end do
      call zc_vzero_d(NW * nb * n22(mm), w(kt))
      ta = omp_get_wtime()
      call N(xzdx22)(w, 0, k4 + NW * off4(mm), na, mm, nb, row, sg, xoff, s1, s2, kt)
      tdpart(8) = tdpart(8) + omp_get_wtime() - ta
      ! the thin products, per row
      do q = 0, npg - 1
        o = off22(mm, q)
        if (o < 0) cycle
        nc = dnQ(mm - q)
        ncol = nb * nc
        do r = r0, r1 - 1
          pq = opoff(c) + r
          u = opa(pq)
          v = opb(pq)
          ! T = rows (ef) of group q, columns (cd): at kt + (o nb + (r - r0) nc), leading dimension nb nc
          if (mm - q == ds) then
            ! sigma2(cd) += sum_{(ef)} <uv||ef> r4(uv; ef cd)
            call N(zc_gemm)(0_4, 0_4, 1_8, nc, dnQ(q), one, w(joovvp + NW * ((u * no + v) * nvp + vpoff(q))), dnQ(q), &
                            w(kt + NW * (o * nb + (r - r0) * nc)), ncol, one, w(ks2), nc)
          end if
          do ka = 1, 2
            if (ka == 1) then
              a = u
              b = v
              alpha = 1
            else
              a = v
              b = u
              alpha = -1
            end if
            ! sigma3(a; p bc) -= sg sum_{(ef)} <pb||ef> r4(ab; ef bc): <pb||ef> = - ~<bp||ef>
            p = q - ocls(b)
            if (dnV(p) > 0 .and. blk3(a) >= 0) then
              if (off12(blk3(a), p) >= 0 .and. dnQ(blk3(a) - p) == nc) then
                call N(zc_gemm)(0_4, 0_4, dnV(p), nc, dnQ(q), -alpha, w(ed + NW * ((b * nv + vcoff(p)) * nvp + vpoff(q))), &
                                nvp, w(kt + NW * (o * nb + (r - r0) * nc)), ncol, one, &
                                w(kx3s + NW * (x3off(a) + off12(blk3(a), p))), nc)
              end if
            end if
            ! XoR(m; b; cd) += sg sum_{(ef)} <ma||ef> r4(ab; ef cd), m of class q - class(a)
            mu = q - ocls(a)
            if (dnO(mu) > 0) then
              call N(zc_gemm)(0_4, 0_4, dnO(mu), nc, dnQ(q), alpha, w(joovvp + NW * ((ocoff(mu) * no + a) * nvp + vpoff(q))), &
                              no * nvp, w(kt + NW * (o * nb + (r - r0) * nc)), ncol, one, &
                              w(kxor + NW * ((ocoff(mu) * no + b) * nvp + vpoff(mm - q))), no * nvp)
            end if
          end do
        end do
      end do
      if (wpp) then
        tb = omp_get_wtime()
        do q = 0, npg - 1
          o = off22(mm, q)
          if (o < 0) cycle
          nc = dnQ(mm - q)
          call N(zc_gemm)(0_4, 0_4, dnQ(q), nb * nc, dnQ(q), one, w(dwq4 + NW * vvoff(q)), dnQ(q), &
                          w(kt + NW * o * nb), nb * nc, zero, w(kx + NW * o * nb), nb * nc)
        end do
        tdpart(9) = tdpart(9) + omp_get_wtime() - tb
        ta = omp_get_wtime()
        call N(xzdx22)(w, 1, ks4 + NW * off4(mm), na, mm, nb, row, sg, xoff, s1, s2, kx)
        tdpart(8) = tdpart(8) + omp_get_wtime() - ta
      end if
      deallocate(row, sg, xoff, s1, s2)
    end do
  end do
  tdpart(2) = tdpart(2) + omp_get_wtime() - t0
  call xzrel(m0)
end subroutine

! ---------------------------------------------------------------------------
! The ring: per spectator j the rows (m, j) of r4 on [(m e)][(bcd)], grouped
! by d = class(m) - class(e), times the stacked operator
! [Wr rows (i a) | f_ov | - 1/2 <mj||ek> | 1/2 <mj||ef>]: the ring term of
! sigma4, and the 4p2h terms of sigma3 and of XvR as rows of the same product
! (Engine._ring_pass).
subroutine N(xzdring)(w, k4, ks3, ks4, kz)
  use xzdata
  use xzmem
  use xzkern
  use xzdead
  use omp_lib
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: k4, ks3, ks4, kz
  integer(8) :: m0, kmb, kxb, kwg, j, nbk, nbat, t, b, mbmax, xbmax, wgmax, mo, xo, nk, kk, c, i, x, pq, mm
  integer(8) :: ne, e, l, ci, nc, nr, r, a, cc, nw2
  integer(8), allocatable :: bd(:), bkk(:), bnc(:), bnr(:), bfov(:), bro(:), brv(:), mbase(:, :), ibase(:, :)
  integer(8), allocatable :: bfirst(:), bmo(:), bxo(:), rcode(:), ccode(:)
  integer(8), allocatable :: erow(:), exoff(:, :), es1(:, :), es2(:, :), em(:), sel(:)
  real(8), allocatable :: esg(:)
  integer(8), pointer :: opidx(:, :)
  ZT, pointer :: s3(:), xb(:), z(:)
  real(8) :: t0, ta
  logical :: wring
  ZT :: one, zero
  one = 1
  zero = 0
  allocate(bd(2 * ncls), bkk(2 * ncls), bnc(2 * ncls), bnr(2 * ncls), bfov(2 * ncls), bro(2 * ncls), brv(2 * ncls), &
           mbase(0:no-1, 2 * ncls), ibase(0:no-1, 2 * ncls), bfirst(2 * ncls + 1), bmo(2 * ncls), bxo(2 * ncls))
  ! the buffers: the largest batch of any spectator
  mbmax = 1
  xbmax = 1
  wgmax = 1
  do j = 0, no - 1
    call xzdringblk(j, nbk, bd, bkk, bnc, bnr, bfov, bro, brv, mbase, ibase)
    if (nbk == 0) cycle
    call xzdringbat(nbk, bkk, bnc, bnr, nbat, bfirst, mbmax, xbmax, wgmax)
  end do
  m0 = xzmark()
  kmb = xzget(NW * mbmax, 'DEA: a batch of r4 of the ring')
  kxb = xzget(NW * xbmax, 'DEA: its product')
  kwg = xzget(NW * wgmax, 'DEA: a block of the stacked operator')
  if (xzdry) then
    call xzrel(m0)
    deallocate(bd, bkk, bnc, bnr, bfov, bro, brv, mbase, ibase, bfirst, bmo, bxo)
    return
  end if
  t0 = omp_get_wtime()
  wring = xzdwant('ring')
  nw2 = no * nv
  call xzdiv2(w, jopidx, no, no, opidx)
  call N(xzdv)(w, ks3, dsz3, s3)
  call N(xzdv)(w, kxb, xbmax, xb)
  call N(xzdv)(w, kz, zsize, z)
  allocate(rcode(0:nw2+no+nv), ccode(0:nw2), erow(0:no-1), exoff(0:ncls-1, 0:no-1), es1(0:ncls-1, 0:no-1), &
           es2(0:ncls-1, 0:no-1), em(0:no-1), sel(0:no-1), esg(0:no-1))
  do j = 0, no - 1
    call xzdringblk(j, nbk, bd, bkk, bnc, bnr, bfov, bro, brv, mbase, ibase)
    if (nbk == 0) cycle
    mbmax = 0
    xbmax = 0
    wgmax = 0
    call xzdringbat(nbk, bkk, bnc, bnr, nbat, bfirst, mbmax, xbmax, wgmax)
    do t = 1, nbat
      ! (a batch without a row of this rank: nothing to make)
      if (all(bnr(bfirst(t):bfirst(t + 1) - 1) == 0)) cycle
      ! the places of the blocks of the batch in the operand and the product
      mo = 0
      xo = 0
      do b = bfirst(t), bfirst(t + 1) - 1
        bmo(b) = mo
        bxo(b) = xo
        mo = mo + bnc(b) * dnK(bkk(b))
        xo = xo + bnr(b) * dnK(bkk(b))
      end do
      call zc_vzero_d(NW * mo, w(kmb))
      ! the operand: the rows (m, j) of r4 expanded on (e | bcd), the blocks of
      ! the classes of e of the batch
      ta = omp_get_wtime()
      call entries(.true.)
      tdpart(8) = tdpart(8) + omp_get_wtime() - ta
      ! the products, block by block
      do b = bfirst(t), bfirst(t + 1) - 1
        nk = dnK(bkk(b))
        nc = bnc(b)
        nr = bnr(b)
        if (nr == 0) cycle
        ! the codes of the columns (m nv + e) and of the rows of the stacked operator
        do mm = 0, no - 1
          if (mbase(mm, b) < 0) cycle
          c = ocls(mm) - bd(b)
          do e = 0, dnV(c) - 1
            ccode(mbase(mm, b) + e) = mm * nv + vcoff(c) + e
          end do
        end do
        do i = 0, no - 1
          if (ibase(i, b) < 0) cycle
          c = ocls(i) - bd(b)
          do e = 0, dnV(c) - 1
            rcode(ibase(i, b) + e) = i * nv + vcoff(c) + e
          end do
        end do
        if (bfov(b) >= 0) rcode(bfov(b)) = nw2
        ci = ocls(j) + bd(b)
        if (bro(b) >= 0) then
          do x = 0, dnO(ci) - 1
            rcode(bro(b) + x) = nw2 + 1 + ocoff(ci) + x
          end do
        end if
        if (brv(b) >= 0) then
          do x = 0, dnV(ci) - 1
            rcode(brv(b) + x) = nw2 + 1 + no + vcoff(ci) + x
          end do
        end if
        call N(xzdwgat)(w, j, nr, nc, rcode, ccode, kwg)
        call N(zc_gemm)(0_4, 0_4, nr, nk, nc, one, w(kwg), nc, w(kmb + NW * bmo(b)), nk, zero, w(kxb + NW * bxo(b)), nk)
      end do
      if (wring) then
        ta = omp_get_wtime()
        call entries(.false.)
        tdpart(8) = tdpart(8) + omp_get_wtime() - ta
      end if
      ! the rows of sigma3 and of XvR
      do b = bfirst(t), bfirst(t + 1) - 1
        nk = dnK(bkk(b))
        kk = bkk(b)
        if (bfov(b) >= 0 .and. blk3(j) == kk) then
          s3(off3(kk) + pos3r(j) * nk:off3(kk) + (pos3r(j) + 1) * nk - 1) = &
            s3(off3(kk) + pos3r(j) * nk:off3(kk) + (pos3r(j) + 1) * nk - 1) + &
            xb(bxo(b) + bfov(b) * nk:bxo(b) + (bfov(b) + 1) * nk - 1)
        end if
        ci = ocls(j) + bd(b)
        if (bro(b) >= 0) then
          do x = 0, dnO(ci) - 1
            i = ocoff(ci) + x
            if (blk3(i) /= kk) cycle
            s3(off3(kk) + pos3r(i) * nk:off3(kk) + (pos3r(i) + 1) * nk - 1) = &
              s3(off3(kk) + pos3r(i) * nk:off3(kk) + (pos3r(i) + 1) * nk - 1) + &
              xb(bxo(b) + (bro(b) + x) * nk:bxo(b) + (bro(b) + x + 1) * nk - 1)
          end do
        end if
        if (brv(b) >= 0 .and. zoffk(kk) >= 0) then
          ne = dnV(ci)
          z(zoffk(kk):zoffk(kk) + ne * nk - 1) = z(zoffk(kk):zoffk(kk) + ne * nk - 1) + &
            xb(bxo(b) + brv(b) * nk:bxo(b) + (brv(b) + ne) * nk - 1)
        end if
      end do
    end do
  end do
  deallocate(rcode, ccode, erow, exoff, es1, es2, em, sel, esg)
  deallocate(bd, bkk, bnc, bnr, bfov, bro, brv, mbase, ibase, bfirst, bmo, bxo)
  tdpart(3) = tdpart(3) + omp_get_wtime() - t0
  call xzrel(m0)

contains

  ! the entries of the rows (m, j) of r4 (expand: the operand, the blocks of
  ! mbase) or (i, j) (reduce: the product, ibase) of the batch t, block by block of r4
  subroutine entries(expand)
    logical, intent(in) :: expand
    integer(8) :: n, k, mmb, base, p, bb, mx, px, lx
    n = 0
    do mmb = 0, no - 1
      if (mmb == j) cycle
      px = opidx(max(mmb, j), min(mmb, j))
      mx = blk4(px)
      if (mx < 0) cycle
      exoff(:, n) = -1
      es1(:, n) = 1
      es2(:, n) = 1
      do bb = bfirst(t), bfirst(t + 1) - 1
        if (expand) then
          base = mbase(mmb, bb)
        else
          base = ibase(mmb, bb)
        end if
        if (base < 0) cycle
        lx = ocls(mmb) - bd(bb)
        if (expand) then
          exoff(lx, n) = bmo(bb) + base * dnK(bkk(bb))
        else
          exoff(lx, n) = bxo(bb) + base * dnK(bkk(bb))
        end if
        es1(lx, n) = dnK(bkk(bb))
      end do
      if (all(exoff(:, n) < 0)) cycle
      erow(n) = pos4(px)
      esg(n) = merge(1d0, -1d0, mmb < j)
      em(n) = mx
      n = n + 1
    end do
    ! one call per block of r4
    do mx = 0, n4c - 1
      k = 0
      do p = 0, n - 1
        if (em(p) /= mx) cycle
        sel(k) = p
        k = k + 1
      end do
      if (k == 0) cycle
      call one_block(mx, k, expand)
    end do
  end subroutine

  subroutine one_block(mmx, k, expand)
    integer(8), intent(in) :: mmx, k
    logical, intent(in) :: expand
    integer(8) :: rr(k), xx(ncls, k), y1(ncls, k), y2(ncls, k), p
    real(8) :: gg(k)
    do p = 1, k
      rr(p) = erow(sel(p - 1))
      gg(p) = esg(sel(p - 1))
      xx(:, p) = exoff(:, sel(p - 1))
      y1(:, p) = es1(:, sel(p - 1))
      y2(:, p) = es2(:, sel(p - 1))
    end do
    if (expand) then
      call N(xzdx13)(w, 0, k4 + NW * off4(mmx), nqa(mmx), mmx, k, rr, gg, xx, y1, y2, kmb)
    else
      call N(xzdx13)(w, 1, ks4 + NW * off4(mmx), nqa(mmx), mmx, k, rr, gg, xx, y1, y2, kxb)
    end if
  end subroutine
end subroutine

! A block of the stacked operator of the ring of the spectator j:
! wg[r][c] for the codes of its rows (i nv + a: Wr(m, a, e, i); no nv: f_ov;
! no nv + 1 + k: - 1/2 <mj||ek> = 1/2 ~<mj||ke>; no nv + 1 + no + e': 1/2 <mj||e'f>)
! and of its columns (m nv + e).
subroutine N(xzdwgat)(w, j, nr, nc, rcode, ccode, kwg)
  use xzdata
  use xzdead
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: j, nr, nc, rcode(0:nr-1), ccode(0:nc-1), kwg
  ZT, pointer :: wg(:), wr2(:), fov(:), ooovd(:), oovv(:)
  integer(8) :: r, c, rc, m, e, nw2
  nw2 = no * nv
  call N(xzdv)(w, kwg, nr * nc, wg)
  call N(xzdv)(w, dwr2, nw2 * nw2, wr2)
  call N(xzdv)(w, efov, nw2, fov)
  call N(xzdv)(w, eooovd, no * no * no * nv, ooovd)
  call N(xzdv)(w, joovv, no * no * nv * nv, oovv)
  !$omp parallel do private(rc, c, m, e)
  do r = 0, nr - 1
    rc = rcode(r)
    if (rc < nw2) then
      do c = 0, nc - 1
        wg(r * nc + c) = wr2(rc * nw2 + ccode(c))
      end do
    else if (rc == nw2) then
      do c = 0, nc - 1
        wg(r * nc + c) = fov(ccode(c))
      end do
    else if (rc < nw2 + 1 + no) then
      do c = 0, nc - 1
        m = ccode(c) / nv
        e = mod(ccode(c), nv)
        wg(r * nc + c) = 0.5d0 * ooovd(((m * no + j) * no + rc - nw2 - 1) * nv + e)
      end do
    else
      do c = 0, nc - 1
        m = ccode(c) / nv
        e = mod(ccode(c), nv)
        wg(r * nc + c) = 0.5d0 * oovv(((m * no + j) * nv + rc - nw2 - 1 - no) * nv + e)
      end do
    end if
  end do
end subroutine

! ---------------------------------------------------------------------------
! The hh ladder: per block one product with Wh on the pairs of the block.
subroutine N(xzdhh)(w, k4, ks4)
  use xzdata
  use xzkern
  use xzdead
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: k4, ks4
  integer(8) :: mm, c, p0, nr, ra, rb
  ZT :: one
  one = 1
  do mm = 0, n4c - 1
    if (.not. inblk4(mm)) cycle
    c = mm - ds
    p0 = opoff(c)
    nr = dnO2(c)
    ! (the rows ra .. rb - 1 of this rank, from all rows of the block)
    call xzdown(mm, ra, rb)
    if (rb == ra) cycle
    call N(zc_gemm)(1_4, 0_4, rb - ra, nqa(mm), nr, one, w(dwh + NW * (p0 * npo + p0 + ra)), npo, w(k4 + NW * off4(mm)), &
                    nqa(mm), one, w(ks4 + NW * (off4(mm) + ra * nqa(mm))), nqa(mm))
  end do
end subroutine

! ---------------------------------------------------------------------------
! The rows of sigma4, one occupied pair (i, j) at a time (Engine._rows):
!   X[(ab), (cd)] per group q of (ab), one product of the stacked operands
!     [- Wv(ab, e, x) | t(xm, ab) | ZR(nf; ab)]^T [r3(y; e, cd); XoR(m; y; cd); t(nij; f, cd)]
!   X[a, (bcd)] per class p of a, one product
!     [Wo(m, a, ij) | t(ij, ae) | WoR(m, a) | r2(a, e) | - YR(mn; y; a)]
!     [r3(m; K); XvR(e; K); t(mij; K); Xv(e; ij; K); t(mnx; K)]
!   each reduced once into the row.
subroutine N(xzdrows)(w, k3, kr2, kx3r, kxor, kzr, zrhave, kz, kwor, kyr, ks4)
  use xzdata
  use xzmem
  use xzkern
  use xzdead
  use omp_lib
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: k3, kr2, kx3r, kxor, kzr, kz, kwor, kyr, ks4
  logical, intent(in) :: zrhave(0:no-1, 0:ncls-1)
  integer(8) :: m0, kx22, kx13, klt, krc, klc, krc3, ks13, k22max, k13max, nkmax, mu, kk, s13size
  integer(8) :: mm, c, r, pq, i, j, q, nc, nq, kr, x, y, n, ph, jj, kj, t0p, p, nk, ksum, ec, nb, cc, a, o, e, ii, ra, rb
  integer(8) :: kb13
  integer :: ios
  integer(8), allocatable :: s13off(:), rows(:)
  integer(8) :: one_row(1), xo22(npg), xs1(npg), xs2(npg), xo13(ncls), ys1(ncls), ys2(ncls)
  real(8) :: one_sg(1), t0, ta
  integer(8), pointer :: opa(:), opb(:), otri(:, :)
  integer(1), pointer :: osgn(:, :)
  ZT, pointer :: lc(:), la13(:), wor(:), r2(:), yr(:)
  logical :: w22, w13
  integer :: sgx
  ZT :: one, zero
  one = 1
  zero = 0
  ! the sizes of the stacked operands
  k22max = 2 * (nvcmax + nocmax) + no * nvcmax
  nkmax = 0
  do kk = 0, n3c - 1
    nkmax = max(nkmax, dnK(kk))
  end do
  k13max = 2 * (nocmax + nvcmax)
  do cc = 0, n2c - 1
    k13max = max(k13max, 2 * (nocmax + nvcmax) + 2 * dnO2(cc))
  end do
  allocate(s13off(0:ncls))
  s13size = 0
  do mu = 0, ncls - 1
    s13off(mu) = -1
    kk = mu + ds
    if (dnK(kk) > 0) then
      s13off(mu) = s13size
      s13size = s13size + (dnO(mu) + dnV(mu)) * dnK(kk)
    end if
  end do
  m0 = xzmark()
  kx22 = xzget(NW * max(n22max, 1_8), 'DEA: X[(ab), (cd)] of a row')
  kx13 = xzget(NW * max(n13max, 1_8), 'DEA: X[a, (bcd)] of a row')
  klt = xzget(NW * k22max * nqmax, 'DEA: stacked operand')
  krc = xzget(NW * k22max * nqmax, 'DEA: stacked operand')
  klc = xzget(NW * nvcmax * k13max, 'DEA: stacked operand')
  krc3 = xzget(NW * k13max * nkmax, 'DEA: stacked operand')
  ks13 = xzget(NW * max(s13size, 1_8), 'DEA: [r3; XvR] by classes')
  if (xzdry) then
    call xzrel(m0)
    deallocate(s13off)
    return
  end if
  t0 = omp_get_wtime()
  w22 = xzdwant('x22')
  w13 = xzdwant('x13')
  call xzdiv(w, jopa, npo, opa)
  call xzdiv(w, jopb, npo, opb)
  call xzdiv2(w, jotri, npo, no, otri)
  call xzdi1v2(w, josgn, npo, no, osgn)
  call N(xzdv)(w, klc, nvcmax * k13max, lc)
  call N(xzdv)(w, dla13, npo * nv * (no + nv), la13)
  call N(xzdv)(w, kwor, no * nv, wor)
  call N(xzdv)(w, kr2, nv * nv, r2)
  call N(xzdv)(w, kyr, no * no * no * nv, yr)
  allocate(rows(0:max(nvcmax, nocmax, nkmax, k13max) - 1))
  ! [r3 rows of class mu; XvR rows of class mu] for K = mu + s
  call zc_vzero_d(NW * max(s13size, 1_8), w(ks13))
  do mu = 0, ncls - 1
    if (s13off(mu) < 0) cycle
    kk = mu + ds
    nk = dnK(kk)
    if (off3(kk + 1) > off3(kk)) call N(xzecopy)(dnO(mu) * nk, w(k3 + NW * off3(kk)), w(ks13 + NW * s13off(mu)))
    if (zoffk(kk) >= 0) call N(xzecopy)(dnV(mu) * nk, w(kz + NW * zoffk(kk)), &
                                        w(ks13 + NW * (s13off(mu) + dnO(mu) * nk)))
  end do
  one_sg(1) = 1d0
  do mm = 0, n4c - 1
    if (.not. inblk4(mm)) cycle
    c = mm - ds
    call xzdown(mm, ra, rb)
    do r = ra, rb - 1
      pq = opoff(c) + r
      i = opa(pq)
      j = opb(pq)
      one_row(1) = r
      ! R13 of the pair: in the work array, or read from the file XZDXV
      if (dxvfile) then
        kb13 = dr13 - NW * r13p(pq)
        if (w13 .and. n13(mm) > 0 .and. r13p(pq + 1) > r13p(pq)) then
          ta = omp_get_wtime()
          read(81, pos=1 + 8 * NW * (r13p(pq) - r13p(dq0)), iostat=ios) w(dr13:dr13 + NW * (r13p(pq + 1) - r13p(pq)) - 1)
          if (ios /= 0) then
            write(6, '(/,a,i8,a)') ' @XZCCSDT-F, reading the file XZDXV of Xv failed (iostat ', ios, ').'
            call xzerrex
          end if
          tdpart(12) = tdpart(12) + omp_get_wtime() - ta
        end if
      else
        kb13 = dr13 - NW * r13p(dq0)
      end if
      if (w22 .and. n22(mm) > 0) then
        ta = omp_get_wtime()
        do q = 0, npg - 1
          if (off22(mm, q) < 0) cycle
          nc = dnQ(mm - q)
          nq = dnQ(q)
          kr = 0
          do x = 1, 2
            if (x == 1) then
              ii = i
              y = j
              sgx = 1
            else
              ii = j
              y = i
              sgx = -1
            end if
            ec = q - ocls(ii)
            if (ec < 0 .or. ec >= ncls) cycle
            ! - sg Wv(ab, e, x) with r3(y; e, cd)
            if (dnV(ec) > 0 .and. blk3(y) >= 0) then
              if (off12(blk3(y), ec) >= 0 .and. dnQ(blk3(y) - ec) == nc) then
                do e = 0, dnV(ec) - 1
                  rows(e) = vcoff(ec) + e
                end do
                call N(xzdgrow)(w, dwvat + NW * ii * nv * nvp, nvp, dnV(ec), rows, -sgx, vpoff(q), vpoff(q + 1), &
                                klt + NW * kr * nq, nq)
                do e = 0, dnV(ec) - 1
                  rows(e) = e
                end do
                call N(xzdgrow)(w, kx3r + NW * (x3off(y) + off12(blk3(y), ec)), nc, dnV(ec), rows, 1, 0_8, nc, &
                                krc + NW * kr * nc, nc)
                kr = kr + dnV(ec)
              end if
            end if
            ! sg t(x m, ab) with XoR(m; y; cd), m of class ec
            if (dnO(ec) > 0) then
              do e = 0, dnO(ec) - 1
                rows(e) = ii * no + ocoff(ec) + e
              end do
              call N(xzdgrow)(w, et2p, nvp, dnO(ec), rows, sgx, vpoff(q), vpoff(q + 1), klt + NW * kr * nq, nq)
              do e = 0, dnO(ec) - 1
                rows(e) = (ocoff(ec) + e) * no + y
              end do
              call N(xzdgrow)(w, kxor, nvp, dnO(ec), rows, 1, vpoff(mm - q), vpoff(mm - q + 1), krc + NW * kr * nc, nc)
              kr = kr + dnO(ec)
            end if
          end do
          ! ZR(nf; ab) with t(nij; f, cd)
          do n = 0, no - 1
            ph = q - ds + ocls(n)
            if (ph < 0 .or. ph >= ncls) cycle
            if (zroff(n, ph) < 0) cycle
            if (.not. zrhave(n, ph)) cycle
            sgx = osgn(pq, n)
            if (sgx == 0) cycle
            jj = otri(pq, n)
            kj = ocls(n) + c
            t0p = t12off(kj, ph)
            if (t0p < 0 .or. kj - ph /= mm - q) cycle
            do e = 0, dnV(ph) - 1
              rows(e) = e
            end do
            call N(xzdgrow)(w, kzr + NW * zroff(n, ph), nq, dnV(ph), rows, sgx, 0_8, nq, klt + NW * kr * nq, nq)
            do e = 0, dnV(ph) - 1
              rows(e) = e * dnJ(kj) + jj - otoff(kj)
            end do
            call N(xzdgrow)(w, dt12 + NW * t0p, nc, dnV(ph), rows, 1, 0_8, nc, krc + NW * kr * nc, nc)
            kr = kr + dnV(ph)
          end do
          if (kr == 0) then
            call zc_vzero_d(NW * nq * nc, w(kx22 + NW * off22(mm, q)))
          else
            call N(zc_gemm)(1_4, 0_4, nq, nc, kr, one, w(klt), nq, w(krc), nc, zero, w(kx22 + NW * off22(mm, q)), nc)
          end if
        end do
        do q = 0, npg - 1
          xo22(q + 1) = off22(mm, q)
          xs1(q + 1) = max(dnQ(mm - q), 1_8)
          xs2(q + 1) = 1
        end do
        tdpart(10) = tdpart(10) + omp_get_wtime() - ta
        ta = omp_get_wtime()
        call N(xzdx22)(w, 1, ks4 + NW * off4(mm), nqa(mm), mm, 1_8, one_row, one_sg, xo22, xs1, xs2, kx22)
        tdpart(8) = tdpart(8) + omp_get_wtime() - ta
      end if
      if (w13 .and. n13(mm) > 0) then
        ta = omp_get_wtime()
        do p = 0, ncls - 1
          if (off13(mm, p) < 0) cycle
          nk = dnK(mm - p)
          ksum = 0
          ! the left operand, [a of class p][the parts]
          ! [Wo(m, a, ij) | t(ij, ae)] with [r3(m; K); XvR(e; K)], m and e of class c - p
          mu = c - p
          if (mu >= 0 .and. mu < ncls) then
            if (s13off(mu) >= 0) then
              nb = dnO(mu) + dnV(mu)
              do a = 0, dnV(p) - 1
                lc(a * k13max + ksum:a * k13max + ksum + nb - 1) = &
                  la13((pq * nv + vcoff(p) + a) * (no + nv) + mix0(mu):(pq * nv + vcoff(p) + a) * (no + nv) + mix0(mu) + nb - 1)
              end do
              do e = 0, nb - 1
                rows(e) = e
              end do
              call N(xzdgrow)(w, ks13 + NW * s13off(mu), nk, nb, rows, 1, 0_8, nk, krc3 + NW * ksum * nk, nk)
              ksum = ksum + nb
            end if
          end if
          ! [WoR(m, a) | r2(a, e)] with [t(m ij; K); Xv(e; ij; K)], m and e of class s - p
          mu = ds - p
          if (mu >= 0 .and. mu < ncls) then
            if (r13off(pq, mu) >= 0) then
              nb = dnO(mu) + dnV(mu)
              do a = 0, dnV(p) - 1
                do e = 0, dnO(mu) - 1
                  lc(a * k13max + ksum + e) = wor((ocoff(mu) + e) * nv + vcoff(p) + a)
                end do
                do e = 0, dnV(mu) - 1
                  lc(a * k13max + ksum + dnO(mu) + e) = r2((vcoff(p) + a) * nv + vcoff(mu) + e)
                end do
              end do
              do e = 0, nb - 1
                rows(e) = e
              end do
              call N(xzdgrow)(w, kb13 + NW * r13off(pq, mu), nk, nb, rows, 1, 0_8, nk, krc3 + NW * ksum * nk, nk)
              ksum = ksum + nb
            end if
          end if
          ! - sg YR(mn; y; a) with t(mn x; K)
          do x = 1, 2
            if (x == 1) then
              ii = i
              y = j
              sgx = 1
            else
              ii = j
              y = i
              sgx = -1
            end if
            cc = mm - p - ocls(ii)
            if (cc < 0 .or. cc >= n2c) cycle
            if (t3poff(ii, cc) < 0) cycle
            nb = dnO2(cc)
            do a = 0, dnV(p) - 1
              do e = 0, nb - 1
                o = opoff(cc) + e
                lc(a * k13max + ksum + e) = -sgx * yr(((opa(o) * no + opb(o)) * no + y) * nv + vcoff(p) + a)
              end do
            end do
            do e = 0, nb - 1
              rows(e) = e
            end do
            call N(xzdgrow)(w, dt3p + NW * t3poff(ii, cc), nk, nb, rows, 1, 0_8, nk, krc3 + NW * ksum * nk, nk)
            ksum = ksum + nb
          end do
          if (ksum == 0) then
            call zc_vzero_d(NW * dnV(p) * nk, w(kx13 + NW * off13(mm, p)))
          else
            call N(zc_gemm)(0_4, 0_4, dnV(p), nk, ksum, one, w(klc), k13max, w(krc3), nk, zero, &
                            w(kx13 + NW * off13(mm, p)), nk)
          end if
        end do
        do p = 0, ncls - 1
          xo13(p + 1) = off13(mm, p)
          ys1(p + 1) = max(dnK(mm - p), 1_8)
          ys2(p + 1) = 1
        end do
        tdpart(11) = tdpart(11) + omp_get_wtime() - ta
        ta = omp_get_wtime()
        call N(xzdx13)(w, 1, ks4 + NW * off4(mm), nqa(mm), mm, 1_8, one_row, one_sg, xo13, ys1, ys2, kx13)
        tdpart(8) = tdpart(8) + omp_get_wtime() - ta
      end if
    end do
  end do
  deallocate(rows, s13off)
  tdpart(5) = tdpart(5) + omp_get_wtime() - t0
  call xzrel(m0)
end subroutine

! ---------------------------------------------------------------------------
! sigma3[i; abc] (the 3p1h row; the ring and the pp ladder have added their
! 4p2h terms already):
!   P(a/bc) [ sum_{e<f} Wq3(bc,ef) r3(i; aef) + sum_e Wv(bc,e,i) r2(ae) ]
!   + P(c/ab) [ sum_{me} Wr(m,c,e,i) r3(m; abe) + sum_m t(im,ab) Q3(m, c) ]
!   - sum_m Fo(m,i) r3(m; abc) + sum_{m<n} WhR(mn) t(mni; abc)
subroutine N(xzdrow3)(w, k3, kr2, kx3r, kx3s, kq3, kwhr, ks3)
  use xzdata
  use xzkern
  use xzdead
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: k3, kr2, kx3r, kx3s, kq3, kwhr, ks3
  integer(8) :: i, k, p, q, e, m, ep, km, c, nr, r, nk, nw2, o, t
  integer(8), allocatable :: row(:), xoff(:, :), s1(:, :), s2(:, :)
  real(8), allocatable :: sg(:)
  ZT, allocatable :: wv(:)
  integer(8), pointer :: opa(:), opb(:)
  ZT, pointer :: whr(:)
  ZT :: one, mone
  if (.not. xzdwant('row3')) then
    call zc_vzero_d(NW * dsz3, w(ks3))
    return
  end if
  one = 1
  mone = -1
  nw2 = no * nv
  do i = 0, no - 1
    k = blk3(i)
    if (k < 0) cycle
    do p = 0, ncls - 1
      if (off12(k, p) < 0) cycle
      q = k - p
      o = x3off(i) + off12(k, p)
      ! pp
      call N(zc_gemm)(0_4, 1_4, dnV(p), dnQ(q), dnQ(q), one, w(kx3r + NW * o), dnQ(q), w(jvvvv + NW * vvoff(q)), dnQ(q), &
                      one, w(kx3s + NW * o), dnQ(q))
      e = ds - p
      if (dnV(e) > 0) then
        ! sum_e r2(a, e) Wv(bc, e, i)
        call N(zc_gemm)(0_4, 0_4, dnV(p), dnQ(q), dnV(e), one, w(kr2 + NW * (vcoff(p) * nv + vcoff(e))), nv, &
                        w(dwvat + NW * ((i * nv + vcoff(e)) * nvp + vpoff(q))), nvp, one, w(kx3s + NW * o), dnQ(q))
      end if
      if (dnO(e) > 0) then
        ! sum_m Q3(m, a) t(im, bc)
        call N(zc_gemm)(1_4, 0_4, dnV(p), dnQ(q), dnO(e), one, w(kq3 + NW * (ocoff(e) * nv + vcoff(p))), nv, &
                        w(et2p + NW * ((i * no + ocoff(e)) * nvp + vpoff(q))), nvp, one, w(kx3s + NW * o), dnQ(q))
      end if
      ! ring: sum_{m e} Wr(m, a, e, i) r3(m; e bc)
      do m = 0, no - 1
        ep = ocls(m) + p - ocls(i)
        km = blk3(m)
        if (km < 0 .or. ep < 0 .or. ep >= ncls) cycle
        if (off12(km, ep) < 0 .or. dnQ(km - ep) /= dnQ(q)) cycle
        call N(zc_gemm)(0_4, 0_4, dnV(p), dnQ(q), dnV(ep), one, w(dwr2 + NW * ((i * nv + vcoff(p)) * nw2 + m * nv + vcoff(ep))), &
                        nw2, w(kx3r + NW * (x3off(m) + off12(km, ep))), dnQ(q), one, w(kx3s + NW * o), dnQ(q))
      end do
    end do
  end do
  call xzdiv(w, jopa, npo, opa)
  call xzdiv(w, jopb, npo, opb)
  call N(xzdv)(w, kwhr, no * no, whr)
  allocate(wv(0:max(dnO2(ds), 1_8) - 1))
  do t = 0, dnO2(ds) - 1
    wv(t) = whr(opa(opoff(ds) + t) * no + opb(opoff(ds) + t))
  end do
  do k = 0, n3c - 1
    nr = off3(k + 1) - off3(k)
    if (nr == 0) cycle
    nk = dnK(k)
    nr = nr / nk
    c = k - ds
    allocate(row(0:nr-1), sg(0:nr-1), xoff(0:ncls-1, 0:nr-1), s1(0:ncls-1, 0:nr-1), s2(0:ncls-1, 0:nr-1))
    do r = 0, nr - 1
      row(r) = r
      sg(r) = 1d0
      do p = 0, ncls - 1
        xoff(p, r) = -1
        s1(p, r) = 1
        s2(p, r) = 1
        if (off12(k, p) >= 0) then
          xoff(p, r) = x3off(ocoff(c) + r) + off12(k, p)
          s1(p, r) = dnQ(k - p)
        end if
      end do
    end do
    call N(xzdx12)(w, 1, ks3 + NW * off3(k), nk, k, nr, row, sg, xoff, s1, s2, kx3s)
    deallocate(row, sg, xoff, s1, s2)
    ! - sum_m Fo(m, i) r3(m; abc)
    call N(zc_gemm)(1_4, 0_4, nr, nk, nr, mone, w(efeoo + NW * (ocoff(c) * no + ocoff(c))), no, w(k3 + NW * off3(k)), nk, &
                    one, w(ks3 + NW * off3(k)), nk)
    ! + sum_{m<n} WhR(mn) t(mni; abc)
    if (dn2 > 0 .and. dnO2(ds) > 0) then
      do r = 0, nr - 1
        i = ocoff(c) + r
        if (t3poff(i, ds) < 0) cycle
        call N(zc_gemm)(0_4, 0_4, 1_8, nk, dnO2(ds), one, wv, dnO2(ds), w(dt3p + NW * t3poff(i, ds)), nk, one, &
                        w(ks3 + NW * (off3(k) + r * nk)), nk)
      end do
    end if
  end do
  deallocate(wv)
end subroutine

! ---------------------------------------------------------------------------
! sigma2[ab] = sum_{e<f} Wq2(ab,ef) r2(ef) + sum_{me} f_me r3(m; eab)
!              + P(ab) sum_m sum_{e<f} <bm||ef> r3(m; aef)    (the 4p2h term: the pp ladder)
subroutine N(xzdrow2)(w, k2, kx3r, ks2)
  use xzdata
  use xzmem
  use xzkern
  use xzdead
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: k2, kx3r, ks2
  integer(8) :: m0, kg, m, km, p, e, t, q, a, b
  integer(8), pointer :: vpa(:), vpb(:)
  ZT, pointer :: g(:), s2(:)
  ZT :: one, mone
  m0 = xzmark()
  kg = xzget(NW * nv * nv, 'DEA: G of sigma2')
  if (xzdry) then
    call xzrel(m0)
    return
  end if
  if (.not. xzdwant('row2') .or. dn2 == 0) then
    call zc_vzero_d(NW * dn2, w(ks2))
    call xzrel(m0)
    return
  end if
  one = 1
  mone = -1
  call N(zc_gemm)(0_4, 0_4, dn2, 1_8, dn2, one, w(dwq2), dn2, w(k2), 1_8, one, w(ks2), 1_8)
  call zc_vzero_d(NW * nv * nv, w(kg))
  do m = 0, no - 1
    km = blk3(m)
    if (km < 0) cycle
    p = ocls(m)
    if (km - p == ds .and. off12(km, p) >= 0) then
      call N(zc_gemm)(0_4, 0_4, 1_8, dn2, dnV(p), one, w(efov + NW * (m * nv + vcoff(p))), dnV(p), &
                      w(kx3r + NW * (x3off(m) + off12(km, p))), dn2, one, w(ks2), dn2)
    end if
    do p = 0, ncls - 1
      if (off12(km, p) < 0) cycle
      e = ds - p
      if (dnV(e) == 0) cycle
      ! G[a][b] += sum_{(ef)} r3(m; a, ef) <bm||ef>, <bm||ef> = - ~<mb||ef>
      call N(zc_gemm)(0_4, 1_4, dnV(p), dnV(e), dnQ(km - p), mone, w(kx3r + NW * (x3off(m) + off12(km, p))), dnQ(km - p), &
                      w(ed + NW * ((m * nv + vcoff(e)) * nvp + vpoff(km - p))), nvp, one, w(kg + NW * (vcoff(p) * nv + vcoff(e))), nv)
    end do
  end do
  call N(xzdv)(w, kg, nv * nv, g)
  call N(xzdv)(w, ks2, dn2, s2)
  call xzdiv(w, jvpa, nvp, vpa)
  call xzdiv(w, jvpb, nvp, vpb)
  do t = 0, dn2 - 1
    q = vpoff(ds) + t
    a = vpa(q)
    b = vpb(q)
    s2(t) = s2(t) + g(a * nv + b) - g(b * nv + a)
  end do
  call xzrel(m0)
end subroutine

! ---------------------------------------------------------------------------
! The masks of the parity (%xzdea parity g|u): the elements of a vector whose
! spinors have the other total parity are set to zero (the parity of the
! occupied ones counts negatively).
subroutine N(xzdproj)(w, kx)
  use xzdata
  use xzmem
  use xzdead
  use xzpar, only: myrank
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: kx
  integer(8) :: m0, kp, k, nk, nr, c, r, a, t, mm, q, p0, pr, i0, ra, rb
  logical :: r23
  integer(2), pointer :: tri(:, :), quads(:, :)
  integer(8), pointer :: vpa(:), vpb(:), opa(:), opb(:)
  integer(1), pointer :: cp(:)
  ZT, pointer :: x(:)
  integer(8) :: nmax
  if (dtpar < 0) return
  nmax = 1
  do k = 0, n3c - 1
    nmax = max(nmax, dnK(k))
  end do
  do mm = 0, n4c - 1
    nmax = max(nmax, nqa(mm))
  end do
  m0 = xzmark()
  kp = xzget(nmax / 8 + 1, 'DEA: parities of the columns of a block')
  if (xzdry) then
    call xzrel(m0)
    return
  end if
  call cp1view(w, kp, nmax, cp)
  call N(xzdv)(w, kx, dn2 + dsz3 + dsz4, x)
  call xzdi2v2(w, dquads, 4_8, max(nqall, 1_8), quads)
  call xzdiv(w, jvpa, nvp, vpa)
  call xzdiv(w, jvpb, nvp, vpb)
  call xzdiv(w, jopa, npo, opa)
  call xzdiv(w, jopb, npo, opb)
  r23 = .not. ddist .or. myrank == 0
  do t = 0, dn2 - 1
    if (.not. r23) exit
    q = vpoff(ds) + t
    if (mod(dpv(vpa(q)) + dpv(vpb(q)) + dtpar, 2_8) /= 0) x(t) = 0
  end do
  do k = 0, n3c - 1
    if (.not. r23) exit
    nr = off3(k + 1) - off3(k)
    if (nr == 0) cycle
    nk = dnK(k)
    nr = nr / nk
    c = k - ds
    call xzdi2v2(w, dtri + dtrio(k), 3_8, nk, tri)
    do a = 0, nk - 1
      cp(a) = int(mod(dpv(tri(0, a)) + dpv(tri(1, a)) + dpv(tri(2, a)), 2_8), 1)
    end do
    !$omp parallel do private(a, pr)
    do r = 0, nr - 1
      pr = dpo(ocoff(c) + r) + dtpar
      do a = 0, nk - 1
        if (mod(cp(a) + pr, 2_8) /= 0) x(dn2 + off3(k) + r * nk + a) = 0
      end do
    end do
  end do
  do mm = 0, n4c - 1
    if (.not. inblk4(mm)) cycle
    nk = nqa(mm)
    c = mm - ds
    i0 = qa0(mm)
    !$omp parallel do
    do a = 0, nk - 1
      cp(a) = int(mod(dpv(quads(0, i0 + a)) + dpv(quads(1, i0 + a)) + dpv(quads(2, i0 + a)) + dpv(quads(3, i0 + a)), 2_8), 1)
    end do
    call xzdown(mm, ra, rb)
    !$omp parallel do private(a, pr, p0)
    do r = ra, rb - 1
      p0 = opoff(c) + r
      pr = dpo(opa(p0)) + dpo(opb(p0)) + dtpar
      do a = 0, nk - 1
        if (mod(cp(a) + pr, 2_8) /= 0) x(dn2 + dsz3 + off4(mm) + r * nk + a) = 0
      end do
    end do
  end do
  call xzrel(m0)
contains
  subroutine cp1view(w, k, n, p)
    use iso_c_binding
    real(8), target :: w(*)
    integer(8), intent(in) :: k, n
    integer(1), pointer :: p(:)
    integer(1), pointer :: qq(:)
    call c_f_pointer(c_loc(w(k)), qq, [max(n, 1_8)])
    p(0:max(n, 1_8) - 1) => qq
  end subroutine
end subroutine

! ---------------------------------------------------------------------------
! The preconditioner: x <- x / (omega - D), D = (sum of the energies of the
! virtual spinors) - (those of the occupied ones) of the element; denominators
! below 1e-3 replaced by 1e-3 (zc_vprecond, as the Python solver).
subroutine N(xzdprec)(w, kx, omega)
  use xzdata
  use xzmem
  use xzkern
  use xzdead
  use xzpar, only: myrank
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: kx
  ZT, intent(in) :: omega
  integer(8) :: m0, kc, k, nk, nr, c, r, a, t, mm, q, p0, i0, nmax, ra, rb
  logical :: r23
  real(8), pointer :: cs(:), eps(:)
  integer(2), pointer :: tri(:, :), quads(:, :)
  integer(8), pointer :: vpa(:), vpb(:), opa(:), opb(:)
  real(8), allocatable :: ro(:)
  real(8), parameter :: floor = 1d-3
  nmax = max(1_8, dn2)
  do k = 0, n3c - 1
    nmax = max(nmax, dnK(k))
  end do
  do mm = 0, n4c - 1
    nmax = max(nmax, nqa(mm))
  end do
  m0 = xzmark()
  kc = xzget(nmax, 'DEA: the energies of the columns of a block')
  if (xzdry) then
    call xzrel(m0)
    return
  end if
  call r8view(w, kc, nmax, cs)
  call r8view(w, jeps, no + nv, eps)
  call xzdi2v2(w, dquads, 4_8, max(nqall, 1_8), quads)
  call xzdiv(w, jvpa, nvp, vpa)
  call xzdiv(w, jvpb, nvp, vpb)
  call xzdiv(w, jopa, npo, opa)
  call xzdiv(w, jopb, npo, opb)
  allocate(ro(0:max(npo, no, 1_8) - 1))
  r23 = .not. ddist .or. myrank == 0
  ! r2
  if (dn2 > 0 .and. r23) then
    do t = 0, dn2 - 1
      q = vpoff(ds) + t
      cs(t) = eps(no + vpa(q)) + eps(no + vpb(q))
    end do
    ro(0) = 0d0
    call N(zc_vprecond)(w(kx), 1_8, dn2, ro, cs, omega, floor)
  end if
  ! r3
  do k = 0, n3c - 1
    if (.not. r23) exit
    nr = off3(k + 1) - off3(k)
    if (nr == 0) cycle
    nk = dnK(k)
    nr = nr / nk
    c = k - ds
    call xzdi2v2(w, dtri + dtrio(k), 3_8, nk, tri)
    do a = 0, nk - 1
      cs(a) = eps(no + tri(0, a)) + eps(no + tri(1, a)) + eps(no + tri(2, a))
    end do
    do r = 0, nr - 1
      ro(r) = eps(ocoff(c) + r)
    end do
    call N(zc_vprecond)(w(kx + NW * (dn2 + off3(k))), nr, nk, ro, cs, omega, floor)
  end do
  ! r4
  do mm = 0, n4c - 1
    if (.not. inblk4(mm)) cycle
    nk = nqa(mm)
    c = mm - ds
    i0 = qa0(mm)
    !$omp parallel do
    do a = 0, nk - 1
      cs(a) = eps(no + quads(0, i0 + a)) + eps(no + quads(1, i0 + a)) + eps(no + quads(2, i0 + a)) + &
              eps(no + quads(3, i0 + a))
    end do
    call xzdown(mm, ra, rb)
    if (rb == ra) cycle
    do r = ra, rb - 1
      p0 = opoff(c) + r
      ro(r - ra) = eps(opa(p0)) + eps(opb(p0))
    end do
    call N(zc_vprecond)(w(kx + NW * (dn2 + dsz3 + off4(mm) + ra * nk)), rb - ra, nk, ro, cs, omega, floor)
  end do
  deallocate(ro)
  call xzrel(m0)
contains
  subroutine r8view(w, k, n, p)
    use iso_c_binding
    real(8), target :: w(*)
    integer(8), intent(in) :: k, n
    real(8), pointer :: p(:)
    real(8), pointer :: qq(:)
    call c_f_pointer(c_loc(w(k)), qq, [max(n, 1_8)])
    p(0:max(n, 1_8) - 1) => qq
  end subroutine
end subroutine
