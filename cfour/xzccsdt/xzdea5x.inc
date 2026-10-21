!
! EOM-DEA-CCSDT with complete 5p3h: the rows of sigma5 and the products of
! Xv5 (type-generic; xzdea5.inc, xzdea5s.inc).
!

! ---------------------------------------------------------------------------
! The rows of sigma5, one occupied triple (i<j<k) of this rank at a time
! (Engine5._rows):
!   X[(ab), (cde)] per group q of (ab), one product of the stacked operands
!     D1 - Wv(ab, f, x) | r4(yz; f cde)     D2 t2(xm, ab) | XoR5(m; yz; cde)
!     D3 r3(x; g ab) | Xv2(g; yz; cde)       D7 Y4(mn; yz; ab) | t3(x mn; cde)
!     D8 XoRp(m; x; ab) | t3(yz m; cde)      D4 t3(yz m; f ab) | Z4(mf; x; cde)
!     D5 t3(ijk; f ab) | XvR5(f; cde)        D6 Xo3(m; ijk; ab) | r3(m; cde)
!   over the splits (x | yz) with their signs;
!   X[a, (bcde)] per class p of a: E1 Wo(m, a, xy) | r4(m z; bcde),
!     E2 W2(mn; ijk; a) | r4(mn; bcde)
!   each reduced once into the row (E1x: N(xzd5zx)).
!   kxva >= 0: Xv5 of all k there (N(xzd5zx) before), E1x in the products of E1
subroutine N(xzd5rows)(w, k3, kr4f, kx3r, kxvr5, ky4, kxor5, kxorp, s5o, kxva)
  use xzdata
  use xzmem
  use xzkern
  use xzdead
  use xzdead5
  use xzpar, only: myrank, nrank, xzmpi_allsum_w
  use omp_lib
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: k3, kr4f, kx3r, kxvr5, ky4, kxor5, kxorp, s5o, kxva
  integer(8), allocatable :: xvz(:)
  integer(8) :: m0, kx23, kx14, klt, krc, klc, krc3, kx4t, k23max, k14max, nkmax, na4max, mm, c, ra, rb, r, i, ki
  integer(8) :: q, nc, nq, kr, rr, x, pyz, ph, mu, k3b, o2, cc, m, sj, jj, kj, p, na, ksum, z, pxy, a, e, mt, ii, pm
  integer(8) :: n2omax, aa, m4, kz4, z4max, zx3, zs, kz4x(0:2), xlim, qp0, qp1, base, psz, bsz, nzs, ndist, zbud, ztick
  integer(8), allocatable :: zslot(:), zlast(:), zsl(:), zbk(:), zbmu(:), zbfe(:), zbown(:)
  logical, allocatable :: zneed(:, :)
  logical :: zsh
  integer(8) :: zrq, nst, kst, kx4z, zr, zt, znb
  real(8) :: zflag(1)
  character(len=256) :: zarg
  integer :: lzarg
  logical, allocatable :: zuse(:)
  integer(8), allocatable :: rows(:)
  integer(1), allocatable :: rsg(:)
  integer(8), pointer :: opa(:), opb(:), opgrp(:), ofree(:, :), opair(:, :), otri(:, :), opidx(:, :)
  integer(1), pointer :: osgn(:, :)
  ZT, pointer :: lc(:), la13(:), w2(:)
  real(8) :: t0, ta, sgn
  integer :: sgx
  logical :: wd(8), anyd
  character(len=2) :: dn
  ZT :: one, zero
  one = 1
  zero = 0
  ! (the test hook deaonly: x23 and d1 .. d8 - the families D1 .. D8 of the rows - alone)
  anyd = .false.
  do i = 1, 8
    write(dn, '(a,i1)') 'd', i
    wd(i) = xzdwant(dn)
    anyd = anyd .or. wd(i)
  end do
  if (.not. anyd) wd = .true.
  n2omax = 0
  do cc = 0, n2c - 1
    n2omax = max(n2omax, dnO2(cc))
  end do
  nkmax = 1
  do cc = 0, n3c - 1
    nkmax = max(nkmax, dnK(cc))
  end do
  na4max = 1
  do cc = 0, n4c - 1
    na4max = max(na4max, nqa(cc))
  end do
  k23max = 3 * (2 * nvcmax + 2 * nocmax + n2omax + no * nvcmax) + nvcmax + nocmax
  k14max = 3 * nocmax + n2omax
  if (kxva >= 0) k14max = k14max + 3 * nvcmax
  m0 = xzmark()
  ! X[(ab), (cde)] of a row in parts of the groups of (ab) of at most n14max elements
  ! (each part reduced on its own), then X[a, (bcde)], in one buffer
  xlim = max(n14max, 1_8)
  do mm = 0, n5c - 1
    if (.not. inblk5(mm)) cycle
    do q = 0, npg - 1
      if (off23(mm, q) >= 0) xlim = max(xlim, dnQ(q) * dnK(mm - q))
    end do
  end do
  kx23 = xzget(NW * xlim, 'DEA5: X[(ab), (cde)] (a part) or X[a, (bcde)] of a row')
  kx14 = kx23
  klt = xzget(NW * k23max * nqmax, 'DEA5: stacked operand')
  krc = xzget(NW * k23max * nkmax, 'DEA5: stacked operand')
  klc = xzget(NW * nvcmax * k14max, 'DEA5: stacked operand')
  krc3 = xzget(NW * k14max * na4max, 'DEA5: stacked operand')
  kx4t = xzget(NW * 3 * max(x4max, 1_8), 'DEA5: the rows of r4 of the pairs of a triple on (h | cde)')
  ! Z4 of the three indices of a triple (made here; the slabs of the triple before kept)
  z4max = 1
  do i = 0, no - 1
    z4max = max(z4max, z4k(i + 1) - z4k(i))
  end do
  ! the indices of the triples of every rank (zneed(rank, k))
  allocate(zneed(0:nrank - 1, 0:max(no, 1_8) - 1), zsl(0:max(no, 1_8) - 1))
  zneed = .false.
  do zr = 0, nrank - 1
    do zt = dtb(zr), dtb(zr + 1) - 1
      if (blk5(zt) < 0) cycle
      do rr = 0, 2
        zneed(zr, xzd5free(zt, rr)) = .true.
      end do
    end do
  end do
  ndist = count(zneed(myrank, :))
  zsl = -1
  zs = 0
  do i = 0, no - 1
    if (.not. zneed(myrank, i)) cycle
    zsl(i) = zs
    zs = zs + 1
  end do
  ! over the ranks: the slabs made once over all ranks (N(xzd5z4sh)) when the
  ! slabs of every rank fit in its work array (xzroom: in the plan below the
  ! peak so far) with the rows (l, k) of r4 and the blocks it makes for
  ! others; else (and with one
  ! rank) made where needed: as many slots as there are indices in the
  ! triples of this rank if they fit, at least three (the slabs of a triple),
  ! the least recently used slab replaced
  zsh = .false.
  nst = 0
  if (ddist) then
    allocate(zbk(no * ncls * ncls), zbmu(no * ncls * ncls), zbfe(no * ncls * ncls), zbown(no * ncls * ncls))
    call N(xzd5z4own)(zneed, znb, zbk, zbmu, zbfe, zbown, nst)
    deallocate(zbk, zbmu, zbfe, zbown)
    zrq = NW * (ndist * z4max + max(nst, 1_8) + no * max(x4max, 1_8) + max(nocmax * nvcmax * nvcmax, 1_8)) + 256
    zsh = zrq <= xzroom()
    ! (option dea5z4: the slots of the ranks, not shared)
    call xzoptget('dea5z4', zarg, lzarg)
    if (lzarg > 0) zsh = .false.
    if (.not. xzdry) then
      ! (all ranks or none)
      zflag(1) = merge(1d0, 0d0, zsh)
      call xzmpi_allsum_w(zflag, 1_8)
      zsh = nint(zflag(1)) == nrank
    end if
  end if
  if (zsh) then
    nzs = max(ndist, 1_8)
    kz4 = xzget(NW * nzs * z4max, 'DEA5: Z4 of the indices of the triples of the rank')
    kst = xzget(NW * max(nst, 1_8), 'DEA5: the blocks of Z4 made for other ranks')
    kx4z = xzget(NW * no * max(x4max, 1_8), 'DEA5: the rows (l, k) of r4 on (h | cde) for Z4')
  else
    zbud = xzroom() - NW * (max(x4max, 1_8) + max(nocmax * nvcmax * nvcmax, 1_8)) - 64
    nzs = min(max(ndist, 3_8), max(3_8, zbud / (NW * z4max)))
    ! (option dea5z4: at most that many slots, a test of the replacement)
    call xzoptget('dea5z4', zarg, lzarg)
    if (lzarg > 0) then
      read(zarg(1:lzarg), *) zs
      nzs = max(3_8, min(nzs, zs))
    end if
    kz4 = xzget(NW * nzs * z4max, 'DEA5: Z4 of the indices of the triples (slots)')
  end if
  d5zs = nzs
  d5zd = ndist
  d5zm = 0
  if (xzdry) then
    ! (the transient of making a slab)
    if (zsh) then
      zs = xzget(NW * max(nocmax * nvcmax * nvcmax, 1_8), 'DEA5: <ml||fh> of a block')
    else
      call N(xzd5z4k)(w, kr4f, 0_8, kz4)
    end if
    call xzrel(m0)
    return
  end if
  if (zsh) then
    call N(xzd5z4sh)(w, kr4f, kz4, z4max, zsl, zneed, kst, kx4z)
    d5zm = -1
  end if
  allocate(zslot(0:nzs - 1), zlast(0:nzs - 1), zuse(0:nzs - 1))
  zslot = -1
  zlast = 0
  ztick = 0
  t0 = omp_get_wtime()
  call xzdiv(w, jopa, max(npo, 1_8), opa)
  call xzdiv(w, jopb, max(npo, 1_8), opb)
  call xzdiv(w, jopgrp, max(npo, 1_8), opgrp)
  call xzdiv2(w, jofree, max(nto, 1_8), 3_8, ofree)
  call xzdiv2(w, jopair, max(nto, 1_8), 3_8, opair)
  call xzdiv2(w, jotri, max(npo, 1_8), no, otri)
  call xzdi1v2(w, josgn, max(npo, 1_8), no, osgn)
  call xzdiv2(w, jopidx, no, no, opidx)
  call N(xzdv)(w, klc, nvcmax * k14max, lc)
  call N(xzdv)(w, dla13, max(npo * nv * (no + nv), 1_8), la13)
  call N(xzdv)(w, dw2, max(nto * nv * npo, 1_8), w2)
  allocate(rows(0:max(k23max, k14max, nvcmax, nocmax) - 1), rsg(0:max(nocmax, 1_8) - 1))
  allocate(xvz(0:max(no, 1_8)))
  xvz(0) = 0
  do i = 0, no - 1
    xvz(i + 1) = xvz(i) + xv5k(i)
  end do
  do mm = 0, n5c - 1
    if (.not. inblk5(mm)) cycle
    c = mm - ds
    call xzd5own(mm, ra, rb)
    do r = ra, rb - 1
      i = otoff(c) + r
      ki = xzd5tgrp(i)
      ! the rows of r4 of the three pairs of the triple on (h | cde)
      do rr = 0, 2
        call N(xzd5x4row)(w, kr4f, opair(i, rr), kx4t + NW * rr * max(x4max, 1_8))
      end do
      ! Z4 of its three indices: kept slabs, the others made in the slots not in use
      if (n23(mm) > 0 .and. xzdwant('x23') .and. wd(4) .and. zsh) then
        do rr = 0, 2
          kz4x(rr) = kz4 + NW * zsl(ofree(i, rr)) * z4max
        end do
      else if (n23(mm) > 0 .and. xzdwant('x23') .and. wd(4)) then
        zuse = .false.
        do rr = 0, 2
          zx3 = ofree(i, rr)
          kz4x(rr) = -1
          do zs = 0, nzs - 1
            if (zslot(zs) == zx3) then
              kz4x(rr) = kz4 + NW * zs * z4max
              zuse(zs) = .true.
              ztick = ztick + 1
              zlast(zs) = ztick
            end if
          end do
        end do
        do rr = 0, 2
          if (kz4x(rr) >= 0) cycle
          zx3 = ofree(i, rr)
          ! (a free slot, else the least recently used one not of this triple)
          zs = -1
          do x = 0, nzs - 1
            if (zuse(x)) cycle
            if (zs < 0) then
              zs = x
            else if (zslot(x) < 0 .and. zslot(zs) >= 0) then
              zs = x
            else if ((zslot(x) < 0) .eqv. (zslot(zs) < 0)) then
              if (zlast(x) < zlast(zs)) zs = x
            end if
          end do
          ztick = ztick + 1
          zlast(zs) = ztick
          zuse(zs) = .true.
          zslot(zs) = zx3
          kz4x(rr) = kz4 + NW * zs * z4max
          call N(xzd5z4k)(w, kr4f, zx3, kz4x(rr))
          d5zm = d5zm + 1
        end do
      end if
      if (n23(mm) > 0 .and. xzdwant('x23')) then
        qp0 = 0
        do while (qp0 < npg)
        ! (the part: the groups q from qp0 while their blocks fit in xlim, at least one)
        base = -1
        psz = 0
        qp1 = qp0
        do while (qp1 < npg)
          if (off23(mm, qp1) >= 0) then
            bsz = dnQ(qp1) * dnK(mm - qp1)
            if (base >= 0 .and. psz + bsz > xlim) exit
            if (base < 0) base = off23(mm, qp1)
            psz = psz + bsz
          end if
          qp1 = qp1 + 1
        end do
        if (base < 0) exit
        ta = omp_get_wtime()
        do q = qp0, qp1 - 1
          if (off23(mm, q) < 0) cycle
          nc = dnK(mm - q)
          nq = dnQ(q)
          kr = 0
          do rr = 0, 2
            x = ofree(i, rr)
            pyz = opair(i, rr)
            sgx = merge(-1, 1, rr == 1)
            m4 = blk4(pyz)
            ! D1: - Wv(ab, f, x) with r4(yz; f cde)
            ph = q - ocls(x)
            if (ph >= 0 .and. ph < ncls .and. m4 >= 0 .and. wd(1)) then
              if (dnV(ph) > 0 .and. off13(m4, ph) >= 0) then
                do e = 0, dnV(ph) - 1
                  rows(e) = vcoff(ph) + e
                end do
                call N(xzdgrow)(w, dwvat + NW * x * nv * nvp, nvp, dnV(ph), rows, -sgx, vpoff(q), vpoff(q + 1), &
                                klt + NW * kr * nq, nq)
                do e = 0, dnV(ph) - 1
                  rows(e) = e
                end do
                call N(xzdgrow)(w, kx4t + NW * (rr * max(x4max, 1_8) + off13(m4, ph)), nc, dnV(ph), rows, 1, 0_8, nc, &
                                krc + NW * kr * nc, nc)
                kr = kr + dnV(ph)
              end if
            end if
            ! D2: t2(xm, ab) with XoR5(m; yz; cde), m of class q - class(x)
            mu = q - ocls(x)
            if (mu >= 0 .and. mu < ncls .and. wd(2)) then
              if (dnO(mu) > 0 .and. xr5off(ocoff(mu), pyz) >= 0) then
                do e = 0, dnO(mu) - 1
                  rows(e) = x * no + ocoff(mu) + e
                end do
                call N(xzdgrow)(w, et2p, nvp, dnO(mu), rows, sgx, vpoff(q), vpoff(q + 1), klt + NW * kr * nq, nq)
                do e = 0, dnO(mu) - 1
                  rows(e) = e
                end do
                call N(xzdgrow)(w, kxor5 + NW * xr5off(ocoff(mu), pyz), nc, dnO(mu), rows, 1, 0_8, nc, krc + NW * kr * nc, nc)
                kr = kr + dnO(mu)
              end if
            end if
            ! D3: r3(x; g ab) with Xv2(g; yz; cde)
            ph = ds + ocls(x) - q
            k3b = blk3(x)
            if (ph >= 0 .and. ph < ncls .and. k3b >= 0 .and. wd(3)) then
              if (dnV(ph) > 0 .and. off12(k3b, ph) >= 0 .and. x2loc(pyz) >= 0) then
                o2 = r13off(pyz, ph)
                if (o2 >= 0 .and. opgrp(pyz) + ph == mm - q) then
                  do e = 0, dnV(ph) - 1
                    rows(e) = e
                  end do
                  call N(xzdgrow)(w, kx3r + NW * (x3off(x) + off12(k3b, ph)), nq, dnV(ph), rows, sgx, 0_8, nq, &
                                  klt + NW * kr * nq, nq)
                  call N(xzdgrow)(w, dx2 + NW * (x2loc(pyz) + o2 - r13p(pyz) + dnO(ph) * nc), nc, dnV(ph), rows, 1, 0_8, nc, &
                                  krc + NW * kr * nc, nc)
                  kr = kr + dnV(ph)
                end if
              end if
            end if
            ! D7: Y4(mn; yz; ab) with t3(x mn; cde), (mn) of group mm - q - class(x)
            cc = mm - q - ocls(x)
            if (cc >= 0 .and. cc < n2c .and. wd(7)) then
              if (y4off(pyz, cc) >= 0 .and. t3poff(x, cc) >= 0) then
                do e = 0, dnO2(cc) - 1
                  rows(e) = e
                end do
                call N(xzdgrow)(w, ky4 + NW * y4off(pyz, cc), nq, dnO2(cc), rows, sgx, 0_8, nq, klt + NW * kr * nq, nq)
                call N(xzdgrow)(w, dt3p + NW * t3poff(x, cc), nc, dnO2(cc), rows, 1, 0_8, nc, krc + NW * kr * nc, nc)
                kr = kr + dnO2(cc)
              end if
            end if
            ! D8: XoRp(m; x; ab) with t3(yz m; cde)
            mu = mm - q - opgrp(pyz)
            if (mu >= 0 .and. mu < ncls .and. x2loc(pyz) >= 0 .and. wd(8)) then
              if (dnO(mu) > 0 .and. r13off(pyz, mu) >= 0) then
                do e = 0, dnO(mu) - 1
                  rows(e) = (ocoff(mu) + e) * no + x
                end do
                call N(xzdgrow)(w, kxorp, nvp, dnO(mu), rows, sgx, vpoff(q), vpoff(q + 1), klt + NW * kr * nq, nq)
                do e = 0, dnO(mu) - 1
                  rows(e) = e
                end do
                call N(xzdgrow)(w, dx2 + NW * (x2loc(pyz) + r13off(pyz, mu) - r13p(pyz)), nc, dnO(mu), rows, 1, 0_8, nc, &
                                krc + NW * kr * nc, nc)
                kr = kr + dnO(mu)
              end if
            end if
            ! D4: t3(yz m; f ab) with Z4(mf; x; cde)
            do m = 0, no - 1
              if (.not. wd(4)) exit
              sj = osgn(pyz, m)
              if (sj == 0) cycle
              jj = otri(pyz, m)
              kj = xzd5tgrp(jj)
              ph = kj - q
              if (ph < 0 .or. ph >= ncls) cycle
              if (dnV(ph) == 0 .or. t12off(kj, ph) < 0) cycle
              if (z4off(x, ocls(m), ph) < 0) cycle
              ! (the slab of x: the index of the split, kz4x(rr))
              do e = 0, dnV(ph) - 1
                rows(e) = e * dnJ(kj) + jj - otoff(kj)
              end do
              call N(xzdgrow)(w, dt12 + NW * t12off(kj, ph), nq, dnV(ph), rows, sgx * int(sj), 0_8, nq, klt + NW * kr * nq, nq)
              do e = 0, dnV(ph) - 1
                rows(e) = e
              end do
              call N(xzdgrow)(w, kz4x(rr) + NW * (z4off(x, ocls(m), ph) - z4k(x) + (m - ocoff(ocls(m))) * dnV(ph) * nc), nc, &
                              dnV(ph), rows, 1, 0_8, nc, krc + NW * kr * nc, nc)
              kr = kr + dnV(ph)
            end do
          end do
          ! D5: t3(ijk; f ab) with XvR5(f; cde)
          ph = ki - q
          if (ph >= 0 .and. ph < ncls .and. wd(5)) then
            if (dnV(ph) > 0 .and. t12off(ki, ph) >= 0 .and. ds + ph < n3c) then
              if (zoffk(ds + ph) >= 0) then
                do e = 0, dnV(ph) - 1
                  rows(e) = e * dnJ(ki) + i - otoff(ki)
                end do
                call N(xzdgrow)(w, dt12 + NW * t12off(ki, ph), nq, dnV(ph), rows, 1, 0_8, nq, klt + NW * kr * nq, nq)
                do e = 0, dnV(ph) - 1
                  rows(e) = e
                end do
                call N(xzdgrow)(w, kxvr5 + NW * zoffk(ds + ph), nc, dnV(ph), rows, 1, 0_8, nc, krc + NW * kr * nc, nc)
                kr = kr + dnV(ph)
              end if
            end if
          end if
          ! D6: Xo3(m; ijk; ab) with r3(m; cde)
          cc = mm - q
          mu = cc - ds
          if (mu >= 0 .and. mu < ncls .and. cc < n3c .and. wd(6)) then
            if (dnO(mu) > 0 .and. off3(cc + 1) > off3(cc)) then
              do e = 0, dnO(mu) - 1
                rows(e) = ocoff(mu) + e
              end do
              call N(xzdgrow)(w, dxo3 + NW * i * no * nvp, nvp, dnO(mu), rows, 1, vpoff(q), vpoff(q + 1), klt + NW * kr * nq, nq)
              do e = 0, dnO(mu) - 1
                rows(e) = e
              end do
              call N(xzdgrow)(w, k3 + NW * off3(cc), nc, dnO(mu), rows, 1, 0_8, nc, krc + NW * kr * nc, nc)
              kr = kr + dnO(mu)
            end if
          end if
          if (kr == 0) then
            call zc_vzero_d(NW * nq * nc, w(kx23 + NW * (off23(mm, q) - base)))
          else
            call N(zc_gemm)(1_4, 0_4, nq, nc, kr, one, w(klt), nq, w(krc), nc, zero, w(kx23 + NW * (off23(mm, q) - base)), nc)
          end if
        end do
        td5(10) = td5(10) + omp_get_wtime() - ta
        ta = omp_get_wtime()
        call N(xzd5row23p)(w, 1, s5o + NW * r5own(i), i, 1d0, kx23, qp0, qp1, base)
        td5(17) = td5(17) + omp_get_wtime() - ta
        qp0 = qp1
        end do
      end if
      if (n14(mm) > 0 .and. (xzdwant('x14') .or. (kxva >= 0 .and. xzdwant('e1x')))) then
        ta = omp_get_wtime()
        do p = 0, ncls - 1
          if (off14(mm, p) < 0) cycle
          na = d5nA4(mm - p)
          ksum = 0
          ! E1: sg Wo(m, a, xy) with r4(m z; bcde), m of class group(xy) - p
          do rr = 0, 2
            z = ofree(i, rr)
            pxy = opair(i, rr)
            sgn = merge(-1d0, 1d0, rr == 1)
            mu = opgrp(pxy) - p
            if (mu < 0 .or. mu >= ncls) cycle
            if (dnO(mu) == 0 .and. kxva < 0) cycle
            aa = ds + ocls(z) + mu
            mt = 0
            do e = 0, dnO(mu) - 1
              m = ocoff(mu) + e
              if (m == z) cycle
              pm = opidx(max(m, z), min(m, z))
              if (blk4(pm) /= aa) cycle
              rows(mt) = pos4(pm)
              rsg(mt) = int(merge(1, -1, m < z), 1)
              do a = 0, dnV(p) - 1
                lc(a * k14max + ksum + mt) = sgn * la13((pxy * nv + vcoff(p) + a) * (no + nv) + mix0(mu) + e)
              end do
              mt = mt + 1
            end do
            if (mt > 0 .and. xzdwant('x14')) then
              call N(zc_tn_rows)(w(kr4f + NW * off4(aa)), nqa(aa), rows, rsg, mt, 0_8, na, w(krc3 + NW * ksum * na), na)
              ksum = ksum + mt
            end if
            ! E1x: sg t2(xy; a f) with Xv5(f; z; bcde), f of class mu (the columns of la13 after the m)
            if (kxva >= 0 .and. xzdwant('e1x')) then
              if (dnV(mu) > 0 .and. xv5off(z, mu) >= 0) then
                do a = 0, dnV(p) - 1
                  do e = 0, dnV(mu) - 1
                    lc(a * k14max + ksum + e) = sgn * la13((pxy * nv + vcoff(p) + a) * (no + nv) + mix0(mu) + dnO(mu) + e)
                  end do
                end do
                call N(xzecopy)(dnV(mu) * na, w(kxva + NW * (xvz(z) + xv5off(z, mu))), w(krc3 + NW * ksum * na))
                ksum = ksum + dnV(mu)
              end if
            end if
          end do
          ! E2: W2(mn; ijk; a) with r4(mn; bcde), (mn) of group mm - p - s
          cc = mm - p - ds
          if (cc >= 0 .and. cc < n2c .and. xzdwant('x14')) then
            if (dnO2(cc) > 0) then
              if (blk4(opoff(cc)) == mm - p) then
                do a = 0, dnV(p) - 1
                  do e = 0, dnO2(cc) - 1
                    lc(a * k14max + ksum + e) = w2((i * nv + vcoff(p) + a) * npo + opoff(cc) + e)
                  end do
                end do
                call N(xzecopy)(dnO2(cc) * na, w(kr4f + NW * off4(mm - p)), w(krc3 + NW * ksum * na))
                ksum = ksum + dnO2(cc)
              end if
            end if
          end if
          if (ksum == 0) then
            call zc_vzero_d(NW * dnV(p) * na, w(kx14 + NW * off14(mm, p)))
          else
            call N(zc_gemm)(0_4, 0_4, dnV(p), na, ksum, one, w(klc), k14max, w(krc3), na, zero, w(kx14 + NW * off14(mm, p)), na)
          end if
        end do
        td5(11) = td5(11) + omp_get_wtime() - ta
        ta = omp_get_wtime()
        call N(xzd5row14)(w, 1, s5o + NW * r5own(i), i, 1d0, kx14)
        td5(17) = td5(17) + omp_get_wtime() - ta
      end if
    end do
  end do
  deallocate(rows, rsg, zslot, zlast, zuse, xvz, zneed, zsl)
  td5(9) = td5(9) + omp_get_wtime() - t0
  call xzrel(m0)
end subroutine

! ---------------------------------------------------------------------------
! Xv5 and its products, one index k at a time (Engine5._xv5, the rows of
! Xv5 of the ring, the E1x terms of the rows):
!   Xv5(f; k; bcde) = S(b|cde) [ sum_{lh} <bl||fh> r4(lk; h cde) + sum_{l<n} G5(ln; f, b) t3(lnk; cde) ]
!                     + sum_{l<n} sum_h <ln||fh> r5(lnk; h bcde)
!   sigma5(xyk; a bcde) += sg S(a|bcde) sum_f t2(xy; af) Xv5(f; k; bcde)   (E1x)
! The term of r5: the rows (l, n, k) of r5 on (h | bcde), one product per
! class of h with <ln||fh>.  Over the ranks the k have their ranks
! (xzd5zgroups): with at least as many ranks as occupied spinors the ranks of
! k share the pairs (l, n) of the term of r5 and the pairs (x, y) of E1x,
! Xv5(.; k) is summed over them; a rank takes the rows (l, n, k) from their
! owners and gives its rows (x, y, k) of sigma5 to theirs, one row per round.
subroutine N(xzd5zx)(w, kr4f, k5o, s5o, kg5, kxva)
  use xzdata
  use xzmem
  use xzkern
  use xzdead
  use xzdead5
  use xzpar, only: myrank, nrank, xzmpi_segs, xzmpi_chunk, xzmpi_sub_allsum_w, xzmpi_allsum_w
  use omp_lib
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: kr4f, k5o, s5o, kg5, kxva
  integer(8) :: m0, kxv, ku, kx4z, kx14, kin, kout, ktmp, ntmp, z, fe, aa, p, o, nk, l, pl, m4, ph, cc, f, kz, slot, nslot
  integer(8) :: mrow, nb, nrmax, zq, kxz, tt, xvz(0:no), zmy(0:no), nzmy, nset, lmax
  logical :: zset
  real(8) :: zflag(1)
  integer(8), allocatable :: tset(:, :), nts(:)
  character(len=256) :: barg
  integer :: lbarg
  integer(8) :: ip, l1, l2, i, mx, r, rowt, sgt, nr, g, ng, gme, n13x, nround, ir, it, t3i, x, y, zi, pxy, mu, nitem
  integer(8), allocatable :: row(:), xoff(:, :), s1(:, :), s2(:, :), zlist(:), plist(:, :)
  real(8), allocatable :: sg(:)
  integer(8), pointer :: opidx(:, :), opgrp(:), otri(:, :)
  integer(1), pointer :: osgn(:, :)
  logical :: zx
  real(8) :: t0, ta, sl, sgn
  ZT :: one, alpha
  one = 1
  ntmp = 4 * min(xzmpi_chunk(), NW * max(maxval(r5w), 1_8))
  m0 = xzmark()
  ! over fewer ranks than occupied spinors: the k of a rank (k mod ranks)
  ! at once if their slabs fit (zset): every row with one of them taken
  ! once, walked once for the term of r5 and once for E1x; else slot by slot
  zset = .false.
  zmy = -1
  nzmy = 0
  if (ddist .and. nrank < no) then
    do z = 0, no - 1
      if (mod(z, int(nrank, 8)) /= myrank) cycle
      zmy(z) = nzmy
      nzmy = nzmy + xv5k(z)
    end do
    zset = NW * (nzmy + 2 * max(maxval(r5w), 1_8) + max(nvcmax * n13amax, 1_8) + no * max(x4max, 1_8) + &
                 max(n14max, 1_8)) + ntmp + 256 <= xzroom()
    call xzoptget('dea5xb', barg, lbarg)
    if (lbarg > 0) zset = .false.
    if (.not. xzdry) then
      zflag(1) = merge(1d0, 0d0, zset)
      call xzmpi_allsum_w(zflag, 1_8)
      zset = nint(zflag(1)) == nrank
    end if
  end if
  if (kxva >= 0) then
    kxv = kxva
  else if (zset) then
    kxv = xzget(NW * max(nzmy, 1_8), 'DEA5: Xv5 of the k of this rank')
  else
    kxv = xzget(NW * max(xv5max, 1_8), 'DEA5: Xv5 of one k')
  end if
  ku = xzget(NW * max(nvcmax * n13amax, 1_8), 'DEA5: U of Xv5 on (b | cde)')
  kx4z = xzget(NW * no * max(x4max, 1_8), 'DEA5: the rows (l, k) of r4 on (h | cde)')
  kx14 = xzget(NW * max(n14max, 1_8), 'DEA5: a row of r5 on (a | bcde)')
  if (ddist) then
    ! the rounds of the exchanges in batches of nb rounds: nb rows in and nb
    ! rows out at once, as many as the work array has room for (xzroom)
    mrow = max(maxval(r5w), 1_8)
    nitem = (no - 1) * (no - 2) / 2
    nrmax = nitem
    if (zset) nrmax = nto
    ! (zn: the ranks of every k, xzd5zgroups; not in the plan of other numbers of ranks)
    if (nrank >= no .and. allocated(zn)) then
      nrmax = 0
      do zq = 0, no - 1
        nrmax = max(nrmax, (nitem + zn(zq) - 1) / zn(zq))
      end do
    end if
    nb = (xzroom() - ntmp - 64) / (2 * NW * mrow)
    nb = max(1_8, min(nb, nrmax))
    ! (option dea5xb: at most that many rounds in a batch)
    call xzoptget('dea5xb', barg, lbarg)
    if (lbarg > 0) then
      read(barg(1:lbarg), *) zq
      nb = max(1_8, min(nb, zq))
    end if
    d5xb = nb
    kin = xzget(NW * mrow * nb, 'DEA5: the rows (l, n, k) of r5 from their owners (a batch of rounds)')
    kout = xzget(NW * mrow * nb, 'DEA5: the rows (x, y, k) of sigma5 to their owners (a batch of rounds)')
    ktmp = xzget(ntmp, 'DEA5: buffer of the sums of the rows of sigma5')
  end if
  if (xzdry) then
    call xzrel(m0)
    return
  end if
  t0 = omp_get_wtime()
  call xzdiv2(w, jopidx, no, no, opidx)
  call xzdiv(w, jopgrp, max(npo, 1_8), opgrp)
  call xzdiv2(w, jotri, max(npo, 1_8), no, otri)
  call xzdi1v2(w, josgn, max(npo, 1_8), no, osgn)
  zx = xzdwant('xv5')
  xvz(0) = 0
  do z = 0, no - 1
    xvz(z + 1) = xvz(z) + xv5k(z)
  end do
  allocate(row(0:nvcmax), sg(0:nvcmax), xoff(0:ncls-1, 0:nvcmax), s1(0:ncls-1, 0:nvcmax), s2(0:ncls-1, 0:nvcmax))
  ! the k of this rank, slot by slot (every rank the same number of slots)
  allocate(zlist(no), plist(2, max(npo, 1_8)))
  nitem = (no - 1) * (no - 2) / 2
  nslot = 1
  if (ddist .and. nrank < no .and. .not. zset) nslot = (no + nrank - 1) / nrank
  do slot = 1, nslot
    kz = -1
    ng = 1
    gme = 0
    if (ddist) then
      if (nrank >= no) then
        kz = mod(int(myrank, 8), no)
        ng = zn(kz)
        gme = xzd5zme(kz)
      else
        kz = myrank + (slot - 1) * nrank
        if (kz >= no) kz = -1
      end if
    end if
    do z = 0, no - 1
      if (zset) then
        if (zmy(z) < 0) cycle
      else if (ddist .and. z /= kz) then
        cycle
      end if
      kxz = kxv
      if (kxva >= 0) kxz = kxv + NW * xvz(z)
      if (zset) kxz = kxv + NW * zmy(z)
      call zc_vzero_d(NW * max(xv5k(z), 1_8), w(kxz))
      ! the rows (l, z) of r4 on (h | cde)
      do l = 0, no - 1
        if (l == z) cycle
        call N(xzd5x4row)(w, kr4f, opidx(max(l, z), min(l, z)), kx4z + NW * l * max(x4max, 1_8))
      end do
      ! the terms of r4 and of t3: the classes fe of f dealt to the ranks of z
      do fe = 0, ncls - 1
        if (xv5off(z, fe) < 0) cycle
        if (mod(fe, ng) /= gme) cycle
        aa = ds + ocls(z) + fe
        n13x = n13a(aa)
        if (n13x == 0) cycle
        call zc_vzero_d(NW * dnV(fe) * n13x, w(ku))
        do p = 0, ncls - 1
          o = off13a(aa, p)
          if (o < 0) cycle
          nk = dnK(aa - p)
          do l = 0, no - 1
            if (l == z) cycle
            pl = opidx(max(l, z), min(l, z))
            m4 = blk4(pl)
            if (m4 < 0) cycle
            sl = merge(1d0, -1d0, l < z)
            ph = p - fe + ocls(l)
            if (ph < 0 .or. ph >= ncls) cycle
            if (off13(m4, ph) < 0 .or. vooff(l, fe, p) < 0) cycle
            if (dnK(m4 - ph) /= nk) cycle
            alpha = sl
            ! <bl||fh> = <dm||ef> of the blocks vooff(l, fe, p): [f][b][h]
            call N(zc_gemm)(0_4, 0_4, dnV(fe) * dnV(p), nk, dnV(ph), alpha, w(dvo + NW * vooff(l, fe, p)), dnV(ph), &
                            w(kx4z + NW * (l * max(x4max, 1_8) + off13(m4, ph))), nk, one, w(ku + NW * dnV(fe) * o), nk)
          end do
          cc = aa - p - ocls(z)
          if (cc >= 0 .and. cc < n2c) then
            if (dnO2(cc) > 0 .and. t3poff(z, cc) >= 0) then
              do f = 0, dnV(fe) - 1
                call N(zc_gemm)(0_4, 0_4, dnV(p), nk, dnO2(cc), one, w(kg5 + NW * (((vcoff(fe) + f) * nv + vcoff(p)) * npo + &
                                opoff(cc))), npo, w(dt3p + NW * t3poff(z, cc)), nk, one, &
                                w(ku + NW * (dnV(fe) * o + f * dnV(p) * nk)), nk)
              end do
            end if
          end if
        end do
        do f = 0, dnV(fe) - 1
          row(f) = f
          sg(f) = 1d0
          do p = 0, ncls - 1
            xoff(p, f) = -1
            s1(p, f) = 1
            s2(p, f) = 1
            if (off13a(aa, p) >= 0) then
              xoff(p, f) = dnV(fe) * off13a(aa, p) + f * dnV(p) * dnK(aa - p)
              s1(p, f) = dnK(aa - p)
            end if
          end do
        end do
        call N(xzdx13)(w, 1, kxz + NW * xv5off(z, fe), nqa(aa), aa, dnV(fe), row, sg, xoff, s1, s2, ku)
      end do
      ! the pairs (l < n) without z: of the term of r5 and of E1x
      it = 0
      do l1 = 0, no - 1
        do l2 = l1 + 1, no - 1
          if (l1 == z .or. l2 == z) cycle
          it = it + 1
          plist(1, it) = l1
          plist(2, it) = l2
        end do
      end do
      if (ddist .or. kxva >= 0) cycle
      ! (one rank) the term of r5, then E1x
      do it = 1, nitem
        l1 = plist(1, it)
        l2 = plist(2, it)
        pl = opidx(l2, l1)
        if (osgn(pl, z) == 0) cycle
        t3i = otri(pl, z)
        mx = blk5(t3i)
        if (mx < 0) cycle
        call one_r5(k5o + NW * r5own(t3i), t3i, l1, l2, dble(osgn(pl, z)))
      end do
      do it = 1, nitem
        call one_e1x(it, .true., 0_8)
      end do
    end do
    if (zset) then
      ! (the k of this rank at once) the rows with one of them: lists per rank
      lmax = 0
      allocate(nts(0:nrank - 1), tset(max(nto, 1_8), 0:nrank - 1))
      nts = 0
      do tt = 0, nto - 1
        if (blk5(tt) < 0 .or. r5w(tt) == 0) cycle
        do r = 0, nrank - 1
          if (mod(xzd5free(tt, 0_8), int(nrank, 8)) == r .or. mod(xzd5free(tt, 1_8), int(nrank, 8)) == r .or. &
              mod(xzd5free(tt, 2_8), int(nrank, 8)) == r) then
            nts(r) = nts(r) + 1
            tset(nts(r), r) = tt
          end if
        end do
      end do
      call set_rounds()
      deallocate(nts, tset)
    else if (ddist) then
      ! (over the ranks: every rank takes part in the rounds, with or without a z of this slot)
      z = kz
      call r5_rounds()
      if (nrank >= no) then
        ta = omp_get_wtime()
        call xzmpi_sub_allsum_w(w(kxv), NW * max(xv5k(kz), 1_8))
        td5(16) = td5(16) + omp_get_wtime() - ta
      end if
      call e1x_rounds()
    end if
  end do
  if (kxva >= 0) then
    ! (Xv5 of all k) the term of r5: every row once, into the slabs of its three indices
    do tt = dt0, dt1 - 1
      if (blk5(tt) < 0 .or. r5w(tt) == 0) cycle
      call one_r5all(k5o + NW * r5own(tt), tt)
    end do
  end if
  deallocate(row, sg, xoff, s1, s2, zlist, plist)
  td5(12) = td5(12) + omp_get_wtime() - t0
  call xzrel(m0)

contains

  ! (Xv5 of all k) Xv5(f; z; bcde) += sgn sum_h <l1 l2||fh> r5(T; h bcde) for
  ! the three splits (z | l1 l2) of the triple T, its row at kr expanded once
  subroutine one_r5all(kr, t)
    integer(8), intent(in) :: kr, t
    integer(8) :: mx_, p_, fe_, a_, r_, z_, a1, a2, pq_
    ZT :: al
    mx_ = blk5(t)
    call zc_vzero_d(NW * max(n14(mx_), 1_8), w(kx14))
    call N(xzd5row14)(w, 0, kr, t, 1d0, kx14)
    if (.not. zx) return
    do r_ = 0, 2
      z_ = xzd5free(t, r_)
      a1 = xzd5free(t, merge(1_8, 0_8, r_ == 0))
      a2 = xzd5free(t, merge(1_8, 2_8, r_ == 2))
      pq_ = opidx(a2, a1)
      if (osgn(pq_, z_) == 0) cycle
      al = dble(osgn(pq_, z_))
      do p_ = 0, ncls - 1
        if (off14(mx_, p_) < 0) cycle
        a_ = mx_ - p_
        fe_ = a_ - ds - ocls(z_)
        if (fe_ < 0 .or. fe_ >= ncls) cycle
        if (xv5off(z_, fe_) < 0) cycle
        call N(zc_gemm)(0_4, 0_4, dnV(fe_), d5nA4(a_), dnV(p_), al, &
                        w(joovv + NW * (((a1 * no + a2) * nv + vcoff(fe_)) * nv + vcoff(p_))), nv, &
                        w(kx14 + NW * off14(mx_, p_)), d5nA4(a_), one, w(kxv + NW * (xvz(z_) + xv5off(z_, fe_))), d5nA4(a_))
      end do
    end do
  end subroutine

  ! Xv5(f; z; bcde) += sgn sum_h <l1 l2||fh> r5(T; h bcde) from the row of T at kr
  subroutine one_r5(kr, t, a1, a2, sgn_)
    integer(8), intent(in) :: kr, t, a1, a2
    real(8), intent(in) :: sgn_
    integer(8) :: mx_, p_, fe_, a_
    mx_ = blk5(t)
    call zc_vzero_d(NW * max(n14(mx_), 1_8), w(kx14))
    call N(xzd5row14)(w, 0, kr, t, sgn_, kx14)
    do p_ = 0, ncls - 1
      if (off14(mx_, p_) < 0) cycle
      a_ = mx_ - p_
      fe_ = a_ - ds - ocls(z)
      if (fe_ < 0 .or. fe_ >= ncls) cycle
      if (xv5off(z, fe_) < 0 .or. .not. zx) cycle
      call N(zc_gemm)(0_4, 0_4, dnV(fe_), d5nA4(a_), dnV(p_), one, &
                      w(joovv + NW * (((a1 * no + a2) * nv + vcoff(fe_)) * nv + vcoff(p_))), nv, &
                      w(kx14 + NW * off14(mx_, p_)), d5nA4(a_), one, w(kxz + NW * xv5off(z, fe_)), d5nA4(a_))
    end do
  end subroutine

  ! E1x of the item it (the pair (x, y) of plist) into the row of its triple:
  ! added to sigma5 directly (direct) or made at kd (over the ranks)
  subroutine one_e1x(it_, direct, kd)
    integer(8), intent(in) :: it_, kd
    logical, intent(in) :: direct
    integer(8) :: x_, y_, pxy_, t_, mx_, zi_, p_, mu_, a_
    real(8) :: sg_
    ZT :: al
    x_ = plist(1, it_)
    y_ = plist(2, it_)
    pxy_ = opidx(y_, x_)
    if (osgn(pxy_, z) == 0) return
    t_ = otri(pxy_, z)
    mx_ = blk5(t_)
    if (mx_ < 0) return
    ! the position of z in the triple: the sign of the split (z | xy) is +, -, +
    zi_ = 0
    if (z > x_) zi_ = 1
    if (z > y_) zi_ = 2
    sg_ = merge(-1d0, 1d0, zi_ == 1)
    if (.not. direct) call zc_vzero_d(NW * max(r5w(t_), 1_8), w(kd))
    if (.not. xzdwant('e1x')) return
    call zc_vzero_d(NW * max(n14(mx_), 1_8), w(kx14))
    do p_ = 0, ncls - 1
      if (off14(mx_, p_) < 0) cycle
      mu_ = opgrp(pxy_) - p_
      if (mu_ < 0 .or. mu_ >= ncls) cycle
      if (dnV(mu_) == 0 .or. xv5off(z, mu_) < 0) cycle
      a_ = mx_ - p_
      al = sg_
      call N(zc_gemm)(0_4, 0_4, dnV(p_), d5nA4(a_), dnV(mu_), al, &
                      w(dla13 + NW * ((pxy_ * nv + vcoff(p_)) * (no + nv) + mix0(mu_) + dnO(mu_))), no + nv, &
                      w(kxz + NW * xv5off(z, mu_)), d5nA4(a_), one, w(kx14 + NW * off14(mx_, p_)), d5nA4(a_))
    end do
    if (direct) then
      call N(xzd5row14)(w, 1, s5o + NW * r5own(t_), t_, 1d0, kx14)
    else
      call N(xzd5row14)(w, 1, kd, t_, 1d0, kx14)
    end if
  end subroutine

  ! (the k of this rank at once) the rounds of its rows: in round ir the
  ! ir-th row of the list of every rank, from its owner; the term of r5 of
  ! the rows of this rank, then E1x into rows sent back to their owners
  subroutine set_rounds()
    integer(8) :: ir_, ir0, rr, t_, o_, kb, pass
    integer(8) :: nr_
    nr_ = maxval(nts)
    do pass = 1, 2
      do ir0 = 1, nr_, nb
        if (pass == 2) then
          do ir_ = ir0, min(nr_, ir0 + nb - 1)
            if (ir_ <= nts(myrank)) call one_e1xset(tset(ir_, myrank), kout + NW * mrow * (ir_ - ir0))
          end do
        end if
        ta = omp_get_wtime()
        dsn = 0
        do ir_ = ir0, min(nr_, ir0 + nb - 1)
          kb = merge(kin, kout, pass == 1) + NW * mrow * (ir_ - ir0)
          do rr = 0, nrank - 1
            if (ir_ > nts(rr)) cycle
            t_ = tset(ir_, rr)
            o_ = xzd5owner(t_)
            if (rr /= myrank .and. o_ /= myrank) cycle
            if (pass == 1) then
              if (rr == myrank .and. o_ == myrank) then
                call N(xzecopy)(r5w(t_), w(k5o + NW * r5own(t_)), w(kb))
              else if (rr == myrank) then
                call xzdseg(o_, kb, NW * r5w(t_), 1_8)
              else
                call xzdseg(rr, k5o + NW * r5own(t_), NW * r5w(t_), 0_8)
              end if
            else
              if (rr == myrank .and. o_ == myrank) then
                call N(xzeaxpyc)(r5w(t_), one, w(kb), w(s5o + NW * r5own(t_)))
              else if (rr == myrank) then
                call xzdseg(o_, kb, NW * r5w(t_), 0_8)
              else
                call xzdseg(rr, s5o + NW * r5own(t_), NW * r5w(t_), 1_8)
              end if
            end if
          end do
        end do
        if (pass == 1) then
          call xzmpi_segs(w, dsn, dsp, dspos, dslen, dsdir, 0_8, w, 0_8)
        else
          call xzmpi_segs(w, dsn, dsp, dspos, dslen, dsdir, 1_8, w(ktmp), ntmp)
        end if
        td5(16) = td5(16) + omp_get_wtime() - ta
        if (pass == 1) then
          do ir_ = ir0, min(nr_, ir0 + nb - 1)
            if (ir_ <= nts(myrank)) call one_r5set(kin + NW * mrow * (ir_ - ir0), tset(ir_, myrank))
          end do
        end if
      end do
    end do
  end subroutine

  ! (the k of this rank at once) Xv5(f; z; bcde) += sgn sum_h <l1 l2||fh>
  ! r5(T; h bcde) for the splits (z | l1 l2) of T with z of this rank, the
  ! row at kr expanded once
  subroutine one_r5set(kr, t)
    integer(8), intent(in) :: kr, t
    integer(8) :: mx_, p_, fe_, a_, r_, z_, a1, a2, pq_
    ZT :: al
    mx_ = blk5(t)
    call zc_vzero_d(NW * max(n14(mx_), 1_8), w(kx14))
    call N(xzd5row14)(w, 0, kr, t, 1d0, kx14)
    if (.not. zx) return
    do r_ = 0, 2
      z_ = xzd5free(t, r_)
      if (zmy(z_) < 0) cycle
      a1 = xzd5free(t, merge(1_8, 0_8, r_ == 0))
      a2 = xzd5free(t, merge(1_8, 2_8, r_ == 2))
      pq_ = opidx(a2, a1)
      if (osgn(pq_, z_) == 0) cycle
      al = dble(osgn(pq_, z_))
      do p_ = 0, ncls - 1
        if (off14(mx_, p_) < 0) cycle
        a_ = mx_ - p_
        fe_ = a_ - ds - ocls(z_)
        if (fe_ < 0 .or. fe_ >= ncls) cycle
        if (xv5off(z_, fe_) < 0) cycle
        call N(zc_gemm)(0_4, 0_4, dnV(fe_), d5nA4(a_), dnV(p_), al, &
                        w(joovv + NW * (((a1 * no + a2) * nv + vcoff(fe_)) * nv + vcoff(p_))), nv, &
                        w(kx14 + NW * off14(mx_, p_)), d5nA4(a_), one, w(kxv + NW * (zmy(z_) + xv5off(z_, fe_))), d5nA4(a_))
      end do
    end do
  end subroutine

  ! (the k of this rank at once) E1x of the splits (z | xy) of T with z of
  ! this rank, into one row on (a | bcde), reduced once into the row at kd
  subroutine one_e1xset(t, kd)
    integer(8), intent(in) :: t, kd
    integer(8) :: mx_, p_, mu_, a_, r_, z_, pxy_
    ZT :: al
    mx_ = blk5(t)
    call zc_vzero_d(NW * max(r5w(t), 1_8), w(kd))
    if (.not. xzdwant('e1x')) return
    call zc_vzero_d(NW * max(n14(mx_), 1_8), w(kx14))
    do r_ = 0, 2
      z_ = xzd5free(t, r_)
      if (zmy(z_) < 0) cycle
      pxy_ = opidx(xzd5free(t, merge(1_8, 2_8, r_ == 2)), xzd5free(t, merge(1_8, 0_8, r_ == 0)))
      ! the sign of the split (z | xy): +, -, + for z first, second, third
      al = merge(-1d0, 1d0, r_ == 1)
      do p_ = 0, ncls - 1
        if (off14(mx_, p_) < 0) cycle
        mu_ = opgrp(pxy_) - p_
        if (mu_ < 0 .or. mu_ >= ncls) cycle
        if (dnV(mu_) == 0 .or. xv5off(z_, mu_) < 0) cycle
        a_ = mx_ - p_
        call N(zc_gemm)(0_4, 0_4, dnV(p_), d5nA4(a_), dnV(mu_), al, &
                        w(dla13 + NW * ((pxy_ * nv + vcoff(p_)) * (no + nv) + mix0(mu_) + dnO(mu_))), no + nv, &
                        w(kxv + NW * (zmy(z_) + xv5off(z_, mu_))), d5nA4(a_), one, w(kx14 + NW * off14(mx_, p_)), d5nA4(a_))
      end do
    end do
    call N(xzd5row14)(w, 1, kd, t, 1d0, kx14)
  end subroutine

  ! (over the ranks) the rows (l, n, z) of the items of this rank, one per
  ! round, from their owners; nb rounds at a time
  subroutine r5_rounds()
    integer(8) :: ir_, ir0, ib, rr, zr, ngr, gr, jt, a1, a2, pq_, t_, mx_, o_, kb
    integer(8) :: nr_
    nr_ = 0
    do zr = 0, no - 1
      nr_ = max(nr_, (nitem + zn(zr) - 1) / zn(zr))
    end do
    if (nrank < no) nr_ = nitem
    do ir0 = 1, nr_, nb
      ta = omp_get_wtime()
      dsn = 0
      do ir_ = ir0, min(nr_, ir0 + nb - 1)
        kb = kin + NW * mrow * (ir_ - ir0)
        do rr = 0, nrank - 1
          ! the item of rank rr in this round: its z, its pair
          if (nrank >= no) then
            zr = mod(rr, int(no, 8))
            ngr = zn(zr)
            gr = rr / no
          else
            zr = kz_of(rr)
            ngr = 1
            gr = 0
          end if
          if (zr < 0) cycle
          jt = gr + 1 + (ir_ - 1) * ngr
          if (jt > nitem) cycle
          call item_pair(zr, jt, a1, a2)
          pq_ = opidx(a2, a1)
          if (osgn(pq_, zr) == 0) cycle
          t_ = otri(pq_, zr)
          mx_ = blk5(t_)
          if (mx_ < 0) cycle
          o_ = xzd5owner(t_)
          if (rr /= myrank .and. o_ /= myrank) cycle
          if (rr == myrank .and. o_ == myrank) then
            call N(xzecopy)(r5w(t_), w(k5o + NW * r5own(t_)), w(kb))
          else if (rr == myrank) then
            call xzdseg(o_, kb, NW * r5w(t_), 1_8)
          else
            call xzdseg(rr, k5o + NW * r5own(t_), NW * r5w(t_), 0_8)
          end if
        end do
      end do
      call xzmpi_segs(w, dsn, dsp, dspos, dslen, dsdir, 0_8, w, 0_8)
      td5(16) = td5(16) + omp_get_wtime() - ta
      ! this rank's items
      if (kz < 0) cycle
      do ir_ = ir0, min(nr_, ir0 + nb - 1)
        jt = gme + 1 + (ir_ - 1) * ng
        if (jt > nitem) cycle
        a1 = plist(1, jt)
        a2 = plist(2, jt)
        pq_ = opidx(a2, a1)
        if (osgn(pq_, kz) == 0) cycle
        t_ = otri(pq_, kz)
        if (blk5(t_) < 0) cycle
        call one_r5(kin + NW * mrow * (ir_ - ir0), t_, a1, a2, dble(osgn(pq_, kz)))
      end do
    end do
  end subroutine

  ! (over the ranks) the rows (x, y, z) of E1x of this rank, one per round,
  ! to their owners (added there); nb rounds at a time
  subroutine e1x_rounds()
    integer(8) :: ir_, ir0, rr, zr, ngr, gr, jt, a1, a2, pq_, t_, mx_, o_, nr_, kb
    nr_ = 0
    do zr = 0, no - 1
      nr_ = max(nr_, (nitem + zn(zr) - 1) / zn(zr))
    end do
    if (nrank < no) nr_ = nitem
    do ir0 = 1, nr_, nb
      ! this rank's rows
      if (kz >= 0) then
        do ir_ = ir0, min(nr_, ir0 + nb - 1)
          jt = gme + 1 + (ir_ - 1) * ng
          if (jt <= nitem) call one_e1x(jt, .false., kout + NW * mrow * (ir_ - ir0))
        end do
      end if
      ta = omp_get_wtime()
      dsn = 0
      do ir_ = ir0, min(nr_, ir0 + nb - 1)
        kb = kout + NW * mrow * (ir_ - ir0)
        do rr = 0, nrank - 1
          if (nrank >= no) then
            zr = mod(rr, int(no, 8))
            ngr = zn(zr)
            gr = rr / no
          else
            zr = kz_of(rr)
            ngr = 1
            gr = 0
          end if
          if (zr < 0) cycle
          jt = gr + 1 + (ir_ - 1) * ngr
          if (jt > nitem) cycle
          call item_pair(zr, jt, a1, a2)
          pq_ = opidx(a2, a1)
          if (osgn(pq_, zr) == 0) cycle
          t_ = otri(pq_, zr)
          mx_ = blk5(t_)
          if (mx_ < 0) cycle
          o_ = xzd5owner(t_)
          if (rr /= myrank .and. o_ /= myrank) cycle
          if (rr == myrank .and. o_ == myrank) then
            call N(xzeaxpyc)(r5w(t_), one, w(kb), w(s5o + NW * r5own(t_)))
          else if (rr == myrank) then
            call xzdseg(o_, kb, NW * r5w(t_), 0_8)
          else
            call xzdseg(rr, s5o + NW * r5own(t_), NW * r5w(t_), 1_8)
          end if
        end do
      end do
      call xzmpi_segs(w, dsn, dsp, dspos, dslen, dsdir, 1_8, w(ktmp), ntmp)
      td5(16) = td5(16) + omp_get_wtime() - ta
    end do
  end subroutine

  ! the z of rank rr in this slot (fewer ranks than occupied spinors)
  integer(8) function kz_of(rr)
    integer(8), intent(in) :: rr
    kz_of = rr + (slot - 1) * nrank
    if (kz_of >= no) kz_of = -1
  end function

  ! the jt-th pair (a1 < a2) without zr
  subroutine item_pair(zr, jt, a1, a2)
    integer(8), intent(in) :: zr, jt
    integer(8), intent(out) :: a1, a2
    integer(8) :: b1, b2, cnt
    cnt = 0
    a1 = -1
    a2 = -1
    do b1 = 0, no - 1
      do b2 = b1 + 1, no - 1
        if (b1 == zr .or. b2 == zr) cycle
        cnt = cnt + 1
        if (cnt == jt) then
          a1 = b1
          a2 = b2
          return
        end if
      end do
    end do
  end subroutine
end subroutine
