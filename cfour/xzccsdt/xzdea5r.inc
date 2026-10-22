!
! EOM-DEA-CCSDT with complete 5p3h: the ring and the hh ladder of the 5p3h
! row (type-generic; xzdea5.inc, xzdea5s.inc).
!

! ---------------------------------------------------------------------------
! The ring of the 5p3h row: per spectator pair (u, w) the rows (m, u, w) of r5
! on [(m e)][(bcde)] (zc_t5s_x14p), grouped by d = class(m) - class(e) and by
! the parity of the quadruples (the parity of the row less that of e: two
! products of a quarter of the work and half the size instead of one), times
! the stacked operator [Wr rows (i a) | f_ov | <mu||ei'> | - <mw||ei'>]: the
! ring term of sigma5 and the terms of r5 in sigma4 (Engine5._ring_pass; the
! term of r5 in Xv5 goes with Xv5, N(xzd5zx)).
! Over the ranks the spectator pairs go to the ranks in rounds as the
! spectators of the ring of r4 do (N(xzdring)): a rank takes the rows
! (m, u, w) of its pair from their owners, makes its batches into the rows
! (i, u, w) of a buffer and gives these back to their owners, who add them;
! the rows of sigma4 go to ks4f (all rows, summed over the ranks later).
subroutine N(xzd5ring)(w, k5o, s5o, ks4f)
  use xzdata
  use xzmem
  use xzkern
  use xzdead
  use xzdead5
  use xzpar, only: myrank, nrank, xzmpi_segs, xzmpi_chunk
  use omp_lib
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: k5o, s5o, ks4f
  integer(8) :: m0, kmb, kxb, kwg, pq, u, wq, kp, nbk, nbat, t, b, mbmax, xbmax, wgmax, mo, xo, nk, c, i, x, mm
  integer(8) :: e, ci, nc, nr, a, nw2, nit, it, rk, kin, kout, ktmp, ntmp, ptot, pmax, nround, kr, jme, nsp, q, nj, r0, r
  integer(8) :: pt, ia, kk, pp, pe, nap, rc, ixo
  integer(8), allocatable :: bd(:), bkk(:), bnc(:, :), bnr(:, :), nxr(:, :), xcode(:, :, :), mbase(:, :, :), ibase(:, :, :)
  real(8), allocatable :: icost(:), rload(:), scost(:), rtot(:), rl0(:)
  integer(8), allocatable :: ij(:), ib(:), sord(:), sround(:), rcnt(:), rspec(:, :), njr(:), js(:), lrow(:)
  logical, allocatable :: mine(:, :)
  integer(8), allocatable :: bfirst(:), bmo(:, :), bxo(:, :), rcode(:), ccode(:)
  integer(4), pointer :: ix4(:), q4(:)
  integer(8), pointer :: opidx(:, :), opa(:), opb(:), opgrp(:), otri(:, :)
  integer(1), pointer :: osgn(:, :)
  real(8) :: t0, ta, best, rsum, cf
  logical :: wring
  ZT :: one, zero, alpha
  one = 1
  zero = 0
  allocate(bd(2 * ncls), bkk(2 * ncls), bnc(0:1, 2 * ncls), bnr(0:1, 2 * ncls), nxr(0:1, 2 * ncls), &
           xcode(0:2*no, 0:1, 2 * ncls), mbase(0:no-1, 0:1, 2 * ncls), ibase(0:no-1, 0:1, 2 * ncls), bfirst(2 * ncls + 1), &
           bmo(0:1, 2 * ncls), bxo(0:1, 2 * ncls), lrow(0:no-1))
  call xzdiv(w, jopa, max(npo, 1_8), opa)
  call xzdiv(w, jopb, max(npo, 1_8), opb)
  call xzdiv(w, jopgrp, max(npo, 1_8), opgrp)
  call xzdiv2(w, jotri, max(npo, 1_8), no, otri)
  call xzdi1v2(w, josgn, max(npo, 1_8), no, osgn)
  ! the buffers: the largest batch of any pair; over the ranks the rows of the pair with the most
  mbmax = 1
  xbmax = 1
  wgmax = 1
  pmax = 1
  do pq = 0, npo - 1
    call xzd5ringblk(w, pq, opa(pq), opb(pq), opgrp(pq), nbk, bd, bkk, bnc, bnr, mbase, ibase, nxr, xcode)
    if (nbk == 0) cycle
    call xzd5ringbat(dbatch, nbk, bkk, bnc, bnr, nbat, bfirst, mbmax, xbmax, wgmax)
    call xzd5ringlay(w, pq, lrow, ptot)
    pmax = max(pmax, ptot)
  end do
  ntmp = 4 * min(xzmpi_chunk(), NW * max(maxval(r5w), 1_8))
  m0 = xzmark()
  kmb = xzget(NW * mbmax, 'DEA5: a batch of r5 of the ring')
  kxb = xzget(NW * xbmax, 'DEA5: its product')
  kwg = xzget(NW * wgmax, 'DEA5: a block of the stacked operator')
  if (ddist) then
    kin = xzget(NW * pmax, 'DEA5: the rows (m, u, w) of the spectator pair of the ring')
    kout = xzget(NW * pmax, 'DEA5: the rows (i, u, w) of it, to their owners')
    ktmp = xzget(ntmp, 'DEA5: buffer of the sums of the rows (i, u, w)')
  end if
  if (xzdry) then
    call xzrel(m0)
    deallocate(bd, bkk, bnc, bnr, nxr, xcode, mbase, ibase, bfirst, bmo, bxo, lrow)
    return
  end if
  call c_f_pointer(c_loc(w(dix4p)), q4, [max(nqall, 1_8)])
  ix4(0:max(nqall, 1_8)-1) => q4
  t0 = omp_get_wtime()
  wring = xzdwant('ring')
  nw2 = no * nv
  call xzdiv2(w, jopidx, no, no, opidx)
  allocate(rcode(0:nw2+2*no), ccode(0:nw2))
  allocate(mine(0:max(npo, 1_8)-1, 2 * ncls))
  mine = .true.
  nround = 1
  if (ddist) then
    ! the cost of a batch (pair, blocks): the products, the elements moved by
    ! the expansion and the reduction, and their walks through the quintuples
    ! of the rows (m, u, w), twice; of a pair the sum over its batches
    allocate(icost(npo * 2 * ncls), ij(npo * 2 * ncls), ib(npo * 2 * ncls), rload(0:nrank-1), scost(0:max(npo, 1_8)-1))
    nit = 0
    scost = 0d0
    do pq = 0, npo - 1
      call xzd5ringblk(w, pq, opa(pq), opb(pq), opgrp(pq), nbk, bd, bkk, bnc, bnr, mbase, ibase, nxr, xcode)
      if (nbk == 0) cycle
      mbmax = 0
      xbmax = 0
      wgmax = 0
      call xzd5ringbat(dbatch, nbk, bkk, bnc, bnr, nbat, bfirst, mbmax, xbmax, wgmax)
      do t = 1, nbat
        nit = nit + 1
        ij(nit) = pq
        ib(nit) = t
        icost(nit) = 0d0
        do b = bfirst(t), bfirst(t + 1) - 1
          do pp = 0, 1
            icost(nit) = icost(nit) + dble(bnc(pp, b)) * dble(bnr(pp, b)) * dble(na4p(bkk(b), pp)) + &
                         64d0 * dble(bnc(pp, b) + bnr(pp, b)) * dble(na4p(bkk(b), pp))
          end do
        end do
        do mm = 0, no - 1
          if (osgn(pq, mm) == 0) cycle
          if (blk5(otri(pq, mm)) >= 0) icost(nit) = icost(nit) + 2d0 * 20d0 * dble(r5w(otri(pq, mm)))
        end do
        scost(pq) = scost(pq) + icost(nit)
      end do
    end do
    ! the pairs, the largest first, to the rounds: each to the round with the
    ! least cost so far that has fewer pairs than ranks
    allocate(sord(max(npo, 1_8)), sround(0:max(npo, 1_8)-1), njr(0:max(npo, 1_8)-1), js(max(npo, 1_8)))
    nsp = 0
    do pq = 0, npo - 1
      if (scost(pq) <= 0d0) cycle
      nsp = nsp + 1
      sord(nsp) = pq
    end do
    do q = 1, nsp
      a = q
      do r = q + 1, nsp
        if (scost(sord(r)) > scost(sord(a))) a = r
      end do
      pq = sord(a)
      sord(a) = sord(q)
      sord(q) = pq
    end do
    nround = max(1_8, (nsp + nrank - 1) / nrank)
    allocate(rcnt(nround), rtot(nround), rspec(nround, 0:nrank-1))
    rcnt = 0
    rtot = 0d0
    sround = 0
    do q = 1, nsp
      pq = sord(q)
      kr = 0
      do r = 1, nround
        if (rcnt(r) >= nrank) cycle
        if (kr == 0) then
          kr = r
        else if (rtot(r) < rtot(kr)) then
          kr = r
        end if
      end do
      sround(pq) = kr
      rcnt(kr) = rcnt(kr) + 1
      rtot(kr) = rtot(kr) + scost(pq)
    end do
    ! in a round: one rank per pair, the ranks left one by one to the pair with
    ! the largest cost per rank; the batches of a pair, the largest first, to
    ! the least loaded of its ranks - every rank the same deal
    rspec = -1
    rload = 0d0
    mine = .false.
    rsum = 0d0
    allocate(rl0(0:nrank-1))
    do kr = 1, nround
      rl0 = rload
      nj = 0
      do q = 1, nsp
        if (sround(sord(q)) /= kr) cycle
        nj = nj + 1
        js(nj) = sord(q)
        njr(sord(q)) = 1
      end do
      do r = nj + 1, nrank
        a = 1
        do q = 2, nj
          if (scost(js(q)) / dble(njr(js(q))) > scost(js(a)) / dble(njr(js(a)))) a = q
        end do
        njr(js(a)) = njr(js(a)) + 1
      end do
      r0 = 0
      do q = 1, nj
        pq = js(q)
        rspec(kr, r0:r0 + njr(pq) - 1) = pq
        do
          a = 0
          best = -0.5d0
          do it = 1, nit
            if (ij(it) == pq .and. icost(it) > best) then
              best = icost(it)
              a = it
            end if
          end do
          if (a == 0) exit
          rk = r0 + minloc(rload(r0:r0 + njr(pq) - 1), 1) - 1
          rload(rk) = rload(rk) + icost(a)
          if (rk == myrank) mine(pq, ib(a)) = .true.
          icost(a) = -1d0
        end do
        r0 = r0 + njr(pq)
      end do
      rsum = rsum + maxval(rload - rl0)
    end do
    if (myrank == 0 .and. ndeasig == 0) then
      write(6, '(a,i4,a,i4,a,f8.3,a)') '   the ring of r5 over the ranks: ', nsp, ' spectator pairs in ', nround, &
        ' rounds; the sum over the rounds of the largest cost of a rank is ', rsum / max(sum(rload) / nrank, 1d-300), &
        ' x the mean cost of a rank'
      flush(6)
    end if
  end if
  do kr = 1, nround
    jme = -1
    if (ddist) then
      jme = rspec(kr, myrank)
      ta = omp_get_wtime()
      if (jme >= 0) then
        call xzd5ringlay(w, jme, lrow, ptot)
        call zc_vzero_d(NW * ptot, w(kout))
      end if
      call rows_io(.false.)
      td5(13) = td5(13) + omp_get_wtime() - ta
    end if
    do pq = 0, npo - 1
      if (ddist .and. pq /= jme) cycle
      u = opa(pq)
      wq = opb(pq)
      kp = opgrp(pq)
      call xzd5ringblk(w, pq, u, wq, kp, nbk, bd, bkk, bnc, bnr, mbase, ibase, nxr, xcode)
      if (nbk == 0) cycle
      mbmax = 0
      xbmax = 0
      wgmax = 0
      call xzd5ringbat(dbatch, nbk, bkk, bnc, bnr, nbat, bfirst, mbmax, xbmax, wgmax)
      do t = 1, nbat
        if (.not. mine(pq, t)) cycle
        mo = 0
        xo = 0
        do b = bfirst(t), bfirst(t + 1) - 1
          do pp = 0, 1
            bmo(pp, b) = mo
            bxo(pp, b) = xo
            mo = mo + bnc(pp, b) * na4p(bkk(b), pp)
            xo = xo + bnr(pp, b) * na4p(bkk(b), pp)
          end do
        end do
        call zc_vzero_d(NW * mo, w(kmb))
        ta = omp_get_wtime()
        call entries(.true.)
        td5(17) = td5(17) + omp_get_wtime() - ta
        ta = omp_get_wtime()
        do b = bfirst(t), bfirst(t + 1) - 1
          do pp = 0, 1
            nap = na4p(bkk(b), pp)
            nc = bnc(pp, b)
            nr = bnr(pp, b)
            if (nr == 0 .or. nc == 0 .or. nap == 0) cycle
            ! the codes of the columns (m nv + e) and of the rows of the stacked operator
            do mm = 0, no - 1
              if (mbase(mm, pp, b) < 0) cycle
              c = ocls(mm) - bd(b)
              pe = mod(r5lp(otri(pq, mm)) + pp, 2_8)
              do e = 0, nvp5(c, pe) - 1
                ccode(mbase(mm, pp, b) + e) = mm * nv + d5vl(vlo5(c, pe) + e)
              end do
            end do
            do i = 0, no - 1
              if (ibase(i, pp, b) < 0) cycle
              c = ocls(i) - bd(b)
              pe = mod(r5lp(otri(pq, i)) + pp, 2_8)
              do e = 0, nvp5(c, pe) - 1
                rcode(ibase(i, pp, b) + e) = i * nv + d5vl(vlo5(c, pe) + e)
              end do
            end do
            rcode(nr - nxr(pp, b):nr - 1) = xcode(0:nxr(pp, b) - 1, pp, b)
            call N(xzd5wgat)(w, u, wq, nr, nc, rcode, ccode, kwg)
            call N(zc_gemm)(0_4, 0_4, nr, nap, nc, one, w(kwg), nc, w(kmb + NW * bmo(pp, b)), nap, zero, &
                            w(kxb + NW * bxo(pp, b)), nap)
          end do
        end do
        td5(19) = td5(19) + omp_get_wtime() - ta
        if (wring) then
          ta = omp_get_wtime()
          call entries(.false.)
          td5(17) = td5(17) + omp_get_wtime() - ta
        end if
        ! the rows of sigma4: f_ov (the pair u, w), <mu||ei'> (the pairs i', w), - <mw||ei'> (i', u), on the
        ! quadruples of one parity (their places in the class: dix4p)
        do b = bfirst(t), bfirst(t + 1) - 1
          kk = bkk(b)
          nk = d5nA4(kk)
          do pp = 0, 1
            nap = na4p(kk, pp)
            nr = bnr(pp, b)
            if (nap == 0 .or. bnc(pp, b) == 0) cycle
            ixo = ix4off(kk, pp)
            do x = 0, nxr(pp, b) - 1
              rc = xcode(x, pp, b) - nw2
              if (rc == 0) then
                pt = pq
                cf = 1d0
              else if (rc <= no) then
                i = rc - 1
                pt = opidx(max(i, wq), min(i, wq))
                cf = merge(-0.5d0, 0.5d0, i < wq)
              else
                i = rc - 1 - no
                pt = opidx(max(i, u), min(i, u))
                cf = merge(-0.5d0, 0.5d0, i < u)
              end if
              if (blk4(pt) /= kk) cycle
              call scat4(kxb + NW * (bxo(pp, b) + (nr - nxr(pp, b) + x) * nap), ks4f + NW * (off4(kk) + pos4(pt) * nk), &
                         nap, ixo, cf)
            end do
          end do
        end do
      end do
    end do
    if (ddist) then
      ta = omp_get_wtime()
      call rows_io(.true.)
      td5(13) = td5(13) + omp_get_wtime() - ta
    end if
  end do
  deallocate(rcode, ccode, mine)
  deallocate(bd, bkk, bnc, bnr, nxr, xcode, mbase, ibase, bfirst, bmo, bxo, lrow)
  td5(7) = td5(7) + omp_get_wtime() - t0
  call xzrel(m0)

contains

  ! (over the ranks, round kr) the rows (m, u, w) of the pairs of the round
  ! from their owners to the ranks of the pairs (back: the rows (i, u, w) of
  ! the buffers of these ranks to the owners, added there)
  subroutine rows_io(back)
    logical, intent(in) :: back
    integer(8) :: rr, jr, m, ii, mx, o, go, lo, len, lr(0:no-1), tot
    dsn = 0
    do rr = 0, nrank - 1
      jr = rspec(kr, rr)
      if (jr < 0) cycle
      if (rr == myrank) then
        lr = lrow
      else
        call xzd5ringlay(w, jr, lr, tot)
      end if
      do m = 0, no - 1
        if (osgn(jr, m) == 0) cycle
        ii = otri(jr, m)
        mx = blk5(ii)
        if (mx < 0) cycle
        o = xzd5owner(ii)
        if (rr /= myrank .and. o /= myrank) cycle
        len = r5w(ii)
        go = NW * r5own(ii)
        lo = NW * lr(m)
        if (rr == myrank .and. o == myrank) then
          if (back) then
            call N(xzeaxpyc)(len, one, w(kout + lo), w(s5o + go))
          else
            call N(xzecopy)(len, w(k5o + go), w(kin + lo))
          end if
        else if (rr == myrank) then
          if (back) then
            call xzdseg(o, kout + lo, NW * len, 0_8)
          else
            call xzdseg(o, kin + lo, NW * len, 1_8)
          end if
        else
          if (back) then
            call xzdseg(rr, s5o + go, NW * len, 1_8)
          else
            call xzdseg(rr, k5o + go, NW * len, 0_8)
          end if
        end if
      end do
    end do
    call xzmpi_segs(w, dsn, dsp, dspos, dslen, dsdir, merge(1_8, 0_8, back), w(ktmp), ntmp)
  end subroutine

  ! y[ix[ixo + k]] += cf x[k], k < n: a row of a product on the quadruples of
  ! one parity added to a row of sigma4 (all quadruples of the class)
  subroutine scat4(kxr, kyr, n, ixo_, cf_)
    integer(8), intent(in) :: kxr, kyr, n, ixo_
    real(8), intent(in) :: cf_
    ZT, pointer :: xr(:), yr(:)
    integer(8) :: k
    call N(xzdv)(w, kxr, n, xr)
    call N(xzdv)(w, kyr, nqall, yr)
    !$omp parallel do
    do k = 0, n - 1
      yr(ix4(ixo_ + k)) = yr(ix4(ixo_ + k)) + cf_ * xr(k)
    end do
  end subroutine

  ! the rows (m, u, w) of r5 expanded into the operands (the blocks of mbase:
  ! per class of e and parity of e, the quadruples of the parity of the row
  ! less that of e) or the products reduced into the rows (i, u, w) (ibase):
  ! one call per block and parity of the quintuples (the rows of one list,
  ! equally long: in the part of this rank, or one after the other in the
  ! buffers of the pair over the ranks)
  subroutine entries(expand)
    logical, intent(in) :: expand
    integer(8) :: mmb, base, bb, mx, ii, lx, kr_, m1, ne, mfirst, lp, i1, pq_, pe_
    integer(8) :: rr(no), xx(2 * ncls, no), y1(2 * ncls, no), y2(2 * ncls, no)
    real(8) :: gg(no)
    do m1 = 0, no - 1
      if (osgn(pq, m1) == 0) cycle
      i1 = otri(pq, m1)
      mx = blk5(i1)
      if (mx < 0) cycle
      lp = r5lp(i1)
      ! (the first m of the block and the parity: the others have been done with it)
      mfirst = -1
      do mmb = 0, m1 - 1
        if (osgn(pq, mmb) == 0) cycle
        if (blk5(otri(pq, mmb)) == mx .and. r5lp(otri(pq, mmb)) == lp) mfirst = mmb
      end do
      if (mfirst >= 0) cycle
      ne = 0
      do mmb = m1, no - 1
        if (osgn(pq, mmb) == 0) cycle
        ii = otri(pq, mmb)
        if (blk5(ii) /= mx .or. r5lp(ii) /= lp) cycle
        xx(:, ne + 1) = -1
        y1(:, ne + 1) = 1
        y2(:, ne + 1) = 1
        do bb = bfirst(t), bfirst(t + 1) - 1
          lx = ocls(mmb) - bd(bb)
          do pq_ = 0, 1
            ! (pq_: the parity of the quadruples; e has the parity of the row less it)
            pe_ = mod(lp + pq_, 2_8)
            if (expand) then
              base = mbase(mmb, pq_, bb)
            else
              base = ibase(mmb, pq_, bb)
            end if
            if (base < 0) cycle
            if (expand) then
              xx(2 * lx + pe_ + 1, ne + 1) = bmo(pq_, bb) + base * na4p(bkk(bb), pq_)
            else
              xx(2 * lx + pe_ + 1, ne + 1) = bxo(pq_, bb) + base * na4p(bkk(bb), pq_)
            end if
            y1(2 * lx + pe_ + 1, ne + 1) = na4p(bkk(bb), pq_)
          end do
        end do
        if (all(xx(:, ne + 1) < 0)) cycle
        ne = ne + 1
        gg(ne) = osgn(pq, mmb)
        if (ddist) then
          rr(ne) = (lrow(mmb) - lrow(m1)) / max(r5w(i1), 1_8)
        else
          rr(ne) = (r5own(ii) - r5own(i1)) / max(r5w(i1), 1_8)
        end if
      end do
      if (ne == 0) cycle
      if (ddist) then
        if (expand) then
          kr_ = kin + NW * lrow(m1)
        else
          kr_ = kout + NW * lrow(m1)
        end if
      else
        if (expand) then
          kr_ = k5o + NW * r5own(i1)
        else
          kr_ = s5o + NW * r5own(i1)
        end if
      end if
      ! (the rows of the list before the first m of it, in the part of this rank: negative rows)
      if (.not. ddist) then
        do mmb = 1, ne
          if (rr(mmb) < 0) then
            kr_ = kr_ + NW * rr(mmb) * r5w(i1)
            rr(1:ne) = rr(1:ne) - rr(mmb)
          end if
        end do
      end if
      if (expand) then
        call N(xzd5x14p)(w, 0, kr_, max(r5w(i1), 1_8), mx, lp, ne, rr, gg, xx, y1, y2, kmb)
      else
        call N(xzd5x14p)(w, 1, kr_, max(r5w(i1), 1_8), mx, lp, ne, rr, gg, xx, y1, y2, kxb)
      end if
    end do
  end subroutine
end subroutine

! A block of the stacked operator of the ring of the spectator pair (u, w):
! wg[r][c] for the codes of its rows (i nv + a: Wr(m, a, e, i); no nv: f_ov;
! no nv + 1 + i': <mu||ei'> = - ~<mu||i'e>; no nv + 1 + no + i': - <mw||ei'>)
! and of its columns (m nv + e)
subroutine N(xzd5wgat)(w, u, wq, nr, nc, rcode, ccode, kwg)
  use xzdata
  use xzdead
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: u, wq, nr, nc, rcode(0:nr-1), ccode(0:nc-1), kwg
  ZT, pointer :: wg(:), wr2(:), fov(:), ooovd(:)
  integer(8) :: r, c, rc, m, e, nw2
  nw2 = no * nv
  call N(xzdv)(w, kwg, nr * nc, wg)
  call N(xzdv)(w, dwr2, nw2 * nw2, wr2)
  call N(xzdv)(w, efov, nw2, fov)
  call N(xzdv)(w, eooovd, no * no * no * nv, ooovd)
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
        wg(r * nc + c) = -ooovd(((m * no + u) * no + rc - nw2 - 1) * nv + e)
      end do
    else
      do c = 0, nc - 1
        m = ccode(c) / nv
        e = mod(ccode(c), nv)
        wg(r * nc + c) = ooovd(((m * no + wq) * no + rc - nw2 - 1 - no) * nv + e)
      end do
    end if
  end do
end subroutine

! ---------------------------------------------------------------------------
! The hh ladder of the 5p3h row: per block and parity group of the triples
! (the operator on the occupied triples conserves the parity) one product
! with the operator on the triples of the group.  Over the ranks a rank makes
! its rows from all rows of the block and the group: if other ranks hold some
! of them, in pieces of columns sent to the other ranks that hold rows of it
! (N(xzdhh)).
subroutine N(xzd5hh)(w, k5o, s5o)
  use xzdata
  use xzmem
  use xzkern
  use xzdead
  use xzdead5
  use xzpar, only: myrank, nrank, xzmpi_segs
  use omp_lib
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: k5o, s5o
  integer(8) :: mm, c, nr, m0, kb, kh, nbuf, wc, c0, ncol, r, q, qq, g, ld, na, nm, x, y, nj, i, lp
  integer(8), allocatable :: rows(:), own(:), rk(:)
  logical, allocatable :: holds(:)
  ZT, pointer :: hs(:), h3(:)
  real(8) :: t0, ta
  ZT :: one
  one = 1
  nbuf = 1
  do mm = 0, n5c - 1
    if (.not. inblk5(mm)) cycle
    do g = 0, 1
      call grow(mm, g, na, nm)
      if (nm == 0 .or. nm == na) cycle
      nbuf = max(nbuf, na * max(1_8, min(maxval(r5w), dbatch / max(na, 1_8))))
    end do
  end do
  m0 = xzmark()
  kb = xzget(NW * nbuf, 'DEA5: the rows of a piece of a block of the hh ladder')
  kh = xzget(NW * max(h3size, 1_8), 'DEA5: the operator of the rows of a group of a block')
  if (xzdry .or. .not. xzdwant('hh')) then
    call xzrel(m0)
    return
  end if
  t0 = omp_get_wtime()
  call N(xzdv)(w, dh3, max(h3size, 1_8), h3)
  allocate(rows(0:max(nto, 1_8)-1), own(0:max(nto, 1_8)-1), rk(0:max(nto, 1_8)-1), holds(0:nrank-1))
  do mm = 0, n5c - 1
    if (.not. inblk5(mm)) cycle
    c = mm - ds
    nj = dnJ(c)
    do g = 0, 1
      ! the rows of the group: all of them (rows, their owners rk), those of this rank (own)
      na = 0
      nm = 0
      do i = otoff(c), otoff(c + 1) - 1
        if (r5g(i) /= g) cycle
        rows(na) = i
        rk(na) = xzd5owner(i)
        if (i >= dt0 .and. i < dt1) then
          own(nm) = na
          nm = nm + 1
        end if
        na = na + 1
      end do
      if (nm == 0) cycle
      lp = r5lp(rows(0))
      ld = r5w(rows(0))
      if (ld == 0) cycle
      ! the operator of the rows of this rank from all rows of the group
      call N(xzdv)(w, kh, nm * na, hs)
      do x = 0, nm - 1
        do y = 0, na - 1
          hs(x * na + y) = h3(h3off(c) + (rows(own(x)) - otoff(c)) * nj + rows(y) - otoff(c))
        end do
      end do
      if (nm == na) then
        ! (all rows here: one after the other in the part of this rank)
        call N(zc_gemm)(0_4, 0_4, nm, ld, na, one, w(kh), na, w(k5o + NW * r5own(rows(0))), ld, one, &
                        w(s5o + NW * r5own(rows(0))), ld)
        cycle
      end if
      holds = .false.
      do y = 0, na - 1
        holds(rk(y)) = .true.
      end do
      wc = max(1_8, min(ld, dbatch / na))
      do c0 = 0, ld - 1, wc
        ncol = min(wc, ld - c0)
        ta = omp_get_wtime()
        dsn = 0
        do y = 0, na - 1
          q = rk(y)
          if (q == myrank) then
            call N(xzecopy)(ncol, w(k5o + NW * (r5own(rows(y)) + c0)), w(kb + NW * y * ncol))
            do qq = 0, nrank - 1
              if (qq == myrank .or. .not. holds(qq)) cycle
              call xzdseg(qq, k5o + NW * (r5own(rows(y)) + c0), NW * ncol, 0_8)
            end do
          else
            call xzdseg(q, kb + NW * y * ncol, NW * ncol, 1_8)
          end if
        end do
        call xzmpi_segs(w, dsn, dsp, dspos, dslen, dsdir, 0_8, w, 0_8)
        td5(14) = td5(14) + omp_get_wtime() - ta
        call N(zc_gemm)(0_4, 0_4, nm, ncol, na, one, w(kh), na, w(kb), ncol, one, &
                        w(s5o + NW * (r5own(rows(own(0))) + c0)), ld)
      end do
    end do
  end do
  deallocate(rows, own, rk, holds)
  td5(8) = td5(8) + omp_get_wtime() - t0
  call xzrel(m0)

contains

  ! the rows of the block m and the group gg: all of them, those of this rank
  subroutine grow(m, gg, nall, nmine)
    integer(8), intent(in) :: m, gg
    integer(8), intent(out) :: nall, nmine
    integer(8) :: ii, cc
    cc = m - ds
    nall = 0
    nmine = 0
    do ii = otoff(cc), otoff(cc + 1) - 1
      if (r5g(ii) /= gg) cycle
      nall = nall + 1
      if (ii >= dt0 .and. ii < dt1) nmine = nmine + 1
    end do
  end subroutine
end subroutine
