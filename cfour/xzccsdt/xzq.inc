!
! The (Q) correction of CCSDT(Q) from the converged amplitudes (type-generic):
! the algorithm of src/zccsdt/cc/quadruples.py with the kernels of
! csrc/tnpack.c.
!
!   E[Q] = sum conj(z) r / D,   r = <Q| [V,T3] + 1/2 [[V,T2],T2] |0>,
!                               z = <Q| V T2 |0>
!   E(Q):                       z = <Q| V (T2 + T3) |0>
!
! over i<j<k<l, a<b<c<d; bare integrals, canonical spinors.  For one occupied
! quadruple at a time, r and z over all virtual quadruples are five matrix
! products, each followed by the antisymmetrizing reduction onto a<b<c<d:
!
!   X[(ab)][(cd)]  - sum_e <ab||ei> t(jkl,cde)          (i|jkl), 4 splits   [V,T3]
!                  + sum_m t(im,ab) Xo(m;jkl;cd)        (i|jkl)             T2 T2
!                  + t(ij,ab) <cd||kl>                  (ij|kl), 6 splits   z
!   X[a][(bcd)]    + sum_m <ma||ij> t(mkl,bcd)          (ij|kl), 6 splits   [V,T3]
!                  + 2 sum_e t(ij,ae) Xv(e;kl;bcd)      (ij|kl) with l in the
!                                                       second pair, 3      T2 T2
!
!   Xo(m;jkl;cd) = P(l/jk) [ 1/2 sum_n <mn||jk> t(nl,cd)
!                            + sum_e ( <md||el> t(jk,ce) - <mc||el> t(jk,de) ) ]
!   Xv(e;kl;bcd) = P(d/bc) 1/2 sum_f <bc||ef> t(kl,fd)
!
! Xv is held for the pairs (x, l) of one l at a time, for qxv occupied x (all
! x < l if they fit; otherwise the quadruples of one l are taken in groups by
! the parts of the range of x that their other three indices fall in).  The
! products are done in batches of qc22 columns (cd) and qc13 columns (bcd),
! Xv is formed from qrows rows (bc) of <bc||ef> at a time: the plan
! (xzplan.F90).
!
subroutine N(xzq)(w, eq5, eq6)
  use xzdata
  use xzmem
  use xzkern
  use xzpar
  use omp_lib
  implicit none
  real(8) :: w(*)
  real(8), intent(out) :: eq5, eq6
  integer(8) :: m0, m1, na4, nslot, nxv
  integer(8) :: kt2pp, kt2p, kt2px, kxo, kg, kkm, kb, koo
  integer(8) :: kev4, kz, krt3, krtt, kw22, kw13, kla, kra, klb, krb, klc, krc, kxv, kwx, kmx, kwh
  integer(8) :: i, j, k, l, x, n, size, ng, g1, g2, g3, nquad, ndone
  integer(8) :: lo1, hi1, lo2, hi2, lo3, hi3
  integer(8) :: it, want, nold, nmine, lastl, nlimit, nx, nqnow, nstop
  integer(8) :: slot(0:no-1), xnew(0:no-1), nnew
  logical :: need(0:no-1), first
  real(8) :: e(2), e1(2), t0, tq(6), tstart, cq, cx, costmine, costleft, costdone, tloop, ttask, txv
  real(8) :: estimate, part(6), tot(6), tmx(2)
  integer(8), external :: N(xzqslot)
  integer :: ldbg
  character(len=32) :: arg

  eq5 = 0d0
  eq6 = 0d0
  if (no < 4 .or. nv < 4) return
  call get_environment_variable('XZ_QDEBUG', length=ldbg)
  qdebug = ldbg > 0
  qparts = 0d0
  na4 = nv * (nv - 1) * (nv - 2) * (nv - 3) / 24
  nquad = no * (no - 1) * (no - 2) * (no - 3) / 24
  nslot = max(3_8, min(qxv, no - 1))
  tq = 0d0
  tstart = omp_get_wtime()
  m0 = xzmark()

  ! ---- T2 in three layouts and the intermediate Xo
  kt2pp = xzget(NW * npo * nvp, '(Q): T2 [P][Q]')
  kt2p = xzget(NW * no * no * nvp, '(Q): T2 [i][m][Q]')
  kt2px = xzget(NW * npo * nv * nv, '(Q): T2 [P][e][a]')
  kxo = xzget(NW * nto * no * nvp, '(Q): Xo')
  m1 = xzmark()
  kg = xzget(NW * no * npo * nvp, '(Q): work of Xo')
  kkm = xzget(NW * npo * nv * nv, '(Q): work of Xo')
  kb = xzget(NW * nv * nv, '(Q): work of Xo')
  koo = xzget(NW * npo * no, '(Q): work of Xo')
  if (.not. xzdry) then
    t0 = omp_get_wtime()
    call N(xzqt2)(no, nv, npo, nvp, w(jt2), w(jopa), w(jopb), w(jvpa), w(jvpb), w(kt2pp), w(kt2p), w(kt2px))
    call N(xzqxo)(no, nv, npo, nvp, nto, w(joooo), w(jovvo), w(kt2p), w(kt2px), w(jopa), w(jopb), &
                  w(jvpa), w(jvpb), w(jofree), w(jopair), w(kxo), w(kg), w(kkm), w(kb), w(koo))
    tq(1) = omp_get_wtime() - t0
  end if
  call xzrel(m1)

  ! ---- the work arrays of a quadruple
  if (ncls == 1) then
    kev4 = xzget(na4, '(Q): energies of the virtual quadruples')
    kz = xzget(NW * na4, '(Q): z')
    krt3 = xzget(NW * na4, '(Q): r of [V,T3]')
    krtt = xzget(NW * na4, '(Q): r of T2 T2')
    kw22 = xzget(NW * nvp * qc22, '(Q): product [(ab)][(cd)]')
    kw13 = xzget(NW * nv * qc13, '(Q): product [a][(bcd)]')
    kla = xzget(NW * 4 * nv * nvp, '(Q): stacked operand')
    kra = xzget(NW * 4 * nv * nvp, '(Q): stacked operand')
    klb = xzget(NW * 4 * no * nvp, '(Q): stacked operand')
    krb = xzget(NW * 4 * no * nvp, '(Q): stacked operand')
    klc = xzget(NW * 6 * no * nv, '(Q): stacked operand')
    krc = xzget(NW * 6 * no * nvt, '(Q): stacked operand')
    kwh = xzget(NW * 12 * nvp, '(Q): operands of z')
    nxv = nv * nvt
    kxv = xzget(NW * nslot * nxv, '(Q): Xv')
    kwx = xzget(NW * qrows * nv * nv, '(Q): rows of <vv||vv>')
    kmx = xzget(NW * qrows * nv * nv, '(Q): product of Xv')
  else
    ! in the sectors of the labels (xzqs.inc): the tables of the kernels
    ! (32-bit integers), the blocks of the five products, the operands
    jqkv = xzget(nv / 2 + 1, '(Q): tables')
    jqlid = xzget(ncls / 2 + 1, '(Q): tables')
    jqlst = xzget(ncls / 2 + 2, '(Q): tables')
    jqlmem = xzget(nv / 2 + 1, '(Q): tables')
    jqposv = xzget(nv / 2 + 1, '(Q): tables')
    jqposq = xzget(nvp / 2 + 1, '(Q): tables')
    jqpos3 = xzget(nvt / 2 + 1, '(Q): tables')
    jqtri = xzget(3 * nvt / 2 + 2, '(Q): tables')
    kz = xzget(NW * qn22max, '(Q): blocks of [V,T3] on two pairs')
    krt3 = xzget(NW * qn22max, '(Q): blocks of T2 T2 on two pairs')
    krtt = xzget(NW * qn22max, '(Q): blocks of z')
    kw22 = xzget(NW * qn13max, '(Q): blocks of [V,T3] on an index and a triple')
    kw13 = xzget(NW * qn13max, '(Q): blocks of T2 T2 on an index and a triple')
    kla = xzget(NW * 4 * nvcmax * nqmax, '(Q): stacked operand')
    kra = xzget(NW * 4 * nvcmax * nqmax, '(Q): stacked operand')
    klb = xzget(NW * 4 * nocmax * nqmax, '(Q): stacked operand')
    krb = xzget(NW * 4 * nocmax * nqmax, '(Q): stacked operand')
    kwh = xzget(NW * 12 * nqmax, '(Q): operands of z')
    klc = xzget(NW * 6 * nocmax * nvcmax, '(Q): stacked operand')
    krc = xzget(12 * nocmax, '(Q): rows of T3 of a product')
    nxv = qxvmax
    kxv = xzget(NW * nslot * nxv, '(Q): Xv')
    kwx = xzget(NW * qwbmax, '(Q): block of <vv||vv>')
    kmx = xzget(NW * qm3max, '(Q): products of Xv')
  end if
  if (xzdry) then
    call xzrel(m0)
    return
  end if
  if (ncls == 1) then
    call zc_tn_tuple_sums(w(jeps + no), int(nv, 4), 4_4, w(kev4))
  else
    call xzqstab(nv, nvp, nvt, ncls, vcls, vcoff, vpoff, vtoff, w(jvpidx), w(jvpgrp), w(jvtri), w(jvfree), &
                 w(jqkv), w(jqlid), w(jqlst), w(jqlmem), w(jqposv), w(jqposq), w(jqpos3), w(jqtri))
  end if

  ! ---- the tasks (xzpar.F90): the list, what an earlier run has finished
  cq = dble(nvp) * dble(nvp) * dble(4 * nv + 4 * no + 6) + dble(nv) * dble(nvt) * dble(6 * no + 3 * nv) &
       + 12d0 * dble(na4)
  cx = dble(nvp) * dble(nv) ** 3 + 3d0 * dble(nv) * dble(nvt)
  want = 0
  if (nrank > 1) want = 8 * nrank
  call get_environment_variable('XZ_QTASKS', arg, ldbg)
  if (ldbg > 0) read(arg, *) want
  call xzq_build(no, want, cq, cx)
  qsample = 0d0
  call get_environment_variable('XZ_QSAMPLE', arg, ldbg)
  if (ldbg > 0) read(arg, *) qsample
  if (qsample <= 0d0 .or. qsample >= 1d0) qsample = 0d0
  nstop = -1
  call get_environment_variable('XZ_QSTOP', arg, ldbg)
  if (ldbg > 0) read(arg, *) nstop
  nold = 0
  if (qsample == 0d0) call xzq_restore(no, nv, nwe, ncls, nold)
  costmine = 0d0
  costleft = 0d0
  nmine = 0
  do it = 1, ntask
    if (town(it) /= myrank) cycle
    nmine = nmine + 1
    costmine = costmine + tcost(it)
    if (.not. tdone(it)) costleft = costleft + tcost(it)
  end do
  write(6, '(/,a,i10,a,i14,a)') '   (Q) correction: ', nquad, ' occupied x ', na4, ' virtual quadruples'
  write(6, '(a,i6,a,i5,a,i6,a,i5)') '   tasks: ', ntask, '   ranks: ', nrank, '   tasks of this rank: ', nmine, &
    '   rank ', myrank
  if (nold > 0) write(6, '(a,i6,a)') '   restart: ', nold, ' tasks were finished by an earlier run (files XZQCHK.*)'
  if (qsample > 0d0) write(6, '(a,f7.4,a)') '   XZ_QSAMPLE: the fraction ', qsample, &
    ' of the quadruples of every task is done; the energies are not those of (Q)'
  flush(6)

  first = .true.
  lastl = -1
  costdone = 0d0
  tloop = omp_get_wtime()
  estimate = 0d0
  nqnow = 0
  do it = 1, ntask
    if (town(it) /= myrank .or. tdone(it)) cycle
    l = tl(it)
    if (l /= lastl) slot = -1
    lastl = l
    ttask = omp_get_wtime()
    txv = 0d0
    e = 0d0
    ndone = 0
    nlimit = tnq(it)
    if (qsample > 0d0) nlimit = max(1_8, int(qsample * dble(tnq(it)), 8))
    ! Xv for the x in i0 .. l-1: all at once if they fit, else in three parts
    nx = l - ti0(it)
    size = nx
    if (qxv < nx) size = max(1_8, qxv / 3)
    ng = (nx + size - 1) / size
    groups: do g1 = 0, ng - 1
      do g2 = g1, ng - 1
        do g3 = g2, ng - 1
          need = .false.
          lo1 = ti0(it) + g1 * size
          hi1 = min(l, lo1 + size)
          lo2 = ti0(it) + g2 * size
          hi2 = min(l, lo2 + size)
          lo3 = ti0(it) + g3 * size
          hi3 = min(l, lo3 + size)
          need(lo1:hi1 - 1) = .true.
          need(lo2:hi2 - 1) = .true.
          need(lo3:hi3 - 1) = .true.
          hi1 = min(hi1, ti1(it))
          ! (a group without a quadruple: three indices do not fit)
          n = 0
          do i = lo1, hi1 - 1
            do j = max(i + 1, lo2), hi2 - 1
              n = n + max(0_8, hi3 - max(j + 1, lo3))
            end do
          end do
          if (n == 0) cycle
          do x = 0, l - 1
            if (.not. need(x)) slot(x) = -1
          end do
          t0 = omp_get_wtime()
          nnew = 0
          do x = 0, l - 1
            if (need(x) .and. slot(x) < 0) then
              slot(x) = N(xzqslot)(nslot, l, slot)
              xnew(nnew) = x
              nnew = nnew + 1
              if (ncls > 1) call N(xzqsxv)(w, x, l, w(jqtri), w(kxv + NW * slot(x) * nxv), w(kmx), w(kwx))
            end if
          end do
          ! (dense: the rows of <vv||vv> are expanded once for all new x)
          if (ncls == 1 .and. nnew > 0) call N(xzqxv)(no, nv, nvp, nvt, qrows, nnew, xnew, l, slot, w(jt2), &
                                                      w(jvvvv), w(kxv), w(kwx), w(kmx))
          tq(2) = tq(2) + omp_get_wtime() - t0
          txv = txv + omp_get_wtime() - t0
          do i = lo1, hi1 - 1
            do j = max(i + 1, lo2), hi2 - 1
              do k = max(j + 1, lo3), hi3 - 1
                if (ncls == 1) then
                  call N(xzquad)(w, i, j, k, l, na4, w(jotri), w(josgn), w(kt2pp), w(kt2p), w(kt2px), w(kxo), &
                                 w(kev4), w(kz), w(krt3), w(krtt), w(kw22), w(kw13), w(kla), w(kra), w(klb), &
                                 w(krb), w(klc), w(krc), w(kwh), w(kxv), slot, e1, tq)
                else
                  call N(xzqsquad)(w, i, j, k, l, w(jotri), w(josgn), w(jopidx), w(jopgrp), w(kt2pp), &
                                   w(kt2p), w(kt2px), w(kxo), w(kz), w(krt3), w(krtt), w(kw22), w(kw13), &
                                   w(kla), w(kra), w(klb), w(krb), w(kwh), w(kwh + NW * 6 * nqmax), w(klc), &
                                   w(krc), w(krc + 6 * nocmax), w(kxv), slot, e1, tq)
                end if
                e = e + e1
                ndone = ndone + 1
                if (ndone >= nlimit) exit groups
              end do
            end do
          end do
        end do
      end do
    end do groups
    tsec(it) = omp_get_wtime() - ttask
    nqnow = nqnow + ndone
    if (qsample > 0d0) then
      ! what the whole task would take: Xv as it was, the quadruples in proportion
      estimate = estimate + txv + (tsec(it) - txv) * dble(tnq(it)) / dble(ndone)
      slot = -1
    else
      if (ndone /= tnq(it)) then
        write(6, '(a,i6,2i12)') ' @XZCCSDT-F, (Q): quadruples done and expected in task ', it, ndone, tnq(it)
        call errex
      end if
      te(:, it) = e
      tdone(it) = .true.
      call xzq_save(it, no, nv, nwe, ncls, first)
      ! (a test hook: XZ_QSTOP=n ends the run after n tasks, as an interruption would)
      nstop = nstop - 1
      if (nstop == 0) then
        write(6, '(a)') ' @XZCCSDT-F, XZ_QSTOP: stopped after the task above.'
        flush(6)
        call xzmpi_abort(3_8)
      end if
    end if
    costdone = costdone + tcost(it)
    t0 = omp_get_wtime() - tloop
    write(6, '(a,i6,a,i3,a,i3,a,i3,a,i9,a,f8.2,a,f6.1,a,f10.1,a,f10.1,a)') '   task ', it, '  l ', l, '  i ', &
      ti0(it), ' ..', ti1(it) - 1, '  quadruples ', ndone, '  ', tsec(it), ' s;  done ', &
      100d0 * (costmine - costleft + costdone) / max(costmine, 1d-30), ' %,  ', t0, ' s,  left about ', &
      t0 * (costleft - costdone) / max(costdone, 1d-30), ' s'
    flush(6)
  end do
  call xzrel(m0)

  ! ---- the sum over the ranks
  part = 0d0
  do it = 1, ntask
    if (town(it) /= myrank .or. .not. tdone(it)) cycle
    part(1) = part(1) + te(1, it)
    part(2) = part(2) + te(2, it)
    part(3) = part(3) + dble(tnq(it))
  end do
  part(4) = omp_get_wtime() - tstart
  part(5) = estimate
  part(6) = dble(nqnow)
  call xzmpi_sum(part, tot, 6_8)
  call xzmpi_max(part(4:5), tmx, 2_8)
  eq5 = tot(1)
  eq6 = tot(1) + tot(2)
  qsampled = qsample > 0d0
  if (qdebug) write(6, '(a,4f18.12)') '   (Q) parts: z.[V,T3], z.T2T2, [V,T3].[V,T3], [V,T3].T2T2', qparts
  write(6, '(5x,a28,f12.2,a)') 'Xo', tq(1), ' s'
  write(6, '(5x,a28,f12.2,a)') 'Xv', tq(2), ' s'
  write(6, '(5x,a28,f12.2,a)') 'operands', tq(3), ' s'
  write(6, '(5x,a28,f12.2,a)') 'products (GEMM)', tq(4), ' s'
  write(6, '(5x,a28,f12.2,a)') 'reductions', tq(5), ' s'
  write(6, '(5x,a28,f12.2,a)') 'energy', tq(6), ' s'
  write(6, '(5x,a28,f12.2,a)') 'total of this rank', part(4), ' s'
  if (myrank == 0) then
    if (nrank > 1) write(6, '(5x,a28,f12.2,a,f8.2)') 'slowest rank', tmx(1), ' s;  sum over ranks / slowest ', &
      tot(4) / max(tmx(1), 1d-30)
    if (qsampled) then
      write(6, '(/,a,i10,a,i10,a)') '   sample: ', nint(tot(6), 8), ' of ', nquad, ' quadruples done'
      write(6, '(a,f12.1,a)') '   estimated wall time of the whole (Q), slowest rank: ', tmx(2) + tq(1), ' s'
      write(6, '(a)') '   (from above: Xv was formed for every task in full)'
    else if (nint(tot(3), 8) /= nquad) then
      write(6, '(a,2i12)') ' @XZCCSDT-F, (Q): occupied quadruples done and expected: ', nint(tot(3), 8), nquad
      call errex
    else
      call xzq_cleanup()
    end if
  end if
end subroutine

! a slot of the store of Xv that no x < l holds
integer(8) function N(xzqslot)(nslot, l, slot)
  implicit none
  integer(8), intent(in) :: nslot, l, slot(0:*)
  integer(8) :: s
  do s = 0, nslot - 1
    if (.not. any(slot(0:l - 1) == s)) then
      N(xzqslot) = s
      return
    end if
  end do
  write(6, '(a)') ' @XZCCSDT-F, (Q): no free place for Xv (an error of the schedule).'
  call errex
  N(xzqslot) = 0
end function

! ---------------------------------------------------------------------------
! t2pp[P][Q], t2p[i][m][Q], t2px[P][e][a] = t2[i][j][a][e]
subroutine N(xzqt2)(no, nv, npo, nvp, t2, opa, opb, vpa, vpb, t2pp, t2p, t2px)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)         ! [i][j][a][b]
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(out) :: t2pp(0:nvp-1, 0:npo-1)
  ZT, intent(out) :: t2p(0:nvp-1, 0:no-1, 0:no-1)
  ZT, intent(out) :: t2px(0:nv-1, 0:nv-1, 0:npo-1)             ! [P][e][a]
  integer(8) :: i, j, q, p, a, e
  do i = 0, no - 1
    do j = 0, no - 1
      do q = 0, nvp - 1
        t2p(q, j, i) = t2(vpb(q), vpa(q), j, i)
      end do
    end do
  end do
  do p = 0, npo - 1
    t2pp(:, p) = t2p(:, opb(p), opa(p))
    do e = 0, nv - 1
      do a = 0, nv - 1
        t2px(a, e, p) = t2(e, a, opb(p), opa(p))
      end do
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! xo[(jkl)][m][(cd)]; g: work [l][(jk)][(cd)], km: work [(jk)][c][d],
! b: work [e][d], oo: work [(jk)][n]
subroutine N(xzqxo)(no, nv, npo, nvp, nto, oooo, ovvo, t2p, t2px, opa, opb, vpa, vpb, ofree, opair, &
                    xo, g, km, b, oo)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp, nto
  ZT, intent(in) :: oooo(0:no-1, 0:no-1, 0:no-1, 0:no-1)       ! [m][n][j][k]
  ZT, intent(in) :: ovvo(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)       ! [m][d][e][l]
  ZT, intent(in) :: t2p(0:nvp-1, 0:no-1, 0:no-1)               ! [n][l][Q]
  ZT, intent(in) :: t2px(*)                                    ! [P][c][e] = - t2[j][k][c][e]
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  integer(8), intent(in) :: ofree(0:nto-1, 0:2), opair(0:nto-1, 0:2)
  ZT, intent(out) :: xo(0:nvp-1, 0:no-1, 0:nto-1)
  ZT :: g(0:nvp-1, 0:npo-1, 0:no-1), km(0:nv-1, 0:nv-1, 0:npo-1), b(0:nv-1, 0:nv-1), oo(0:no-1, 0:npo-1)
  integer(8) :: m, l, p, q, n, d, e, it
  ZT :: mone, zero, half
  mone = -1
  zero = 0
  half = 0.5d0
  do m = 0, no - 1
    do p = 0, npo - 1
      do n = 0, no - 1
        oo(n, p) = oooo(opb(p), opa(p), n, m)
      end do
    end do
    do l = 0, no - 1
      do e = 0, nv - 1
        do d = 0, nv - 1
          b(d, e) = ovvo(l, e, d, m)
        end do
      end do
      ! km[(jk)][c][d] = sum_e t2[j][k][c][e] <md||el>
      call N(zc_gemm)(0_4, 0_4, npo * nv, nv, nv, mone, t2px, nv, b, nv, zero, km, nv)
      !$omp parallel do private(p, q)
      do p = 0, npo - 1
        do q = 0, nvp - 1
          g(q, p, l) = km(vpb(q), vpa(q), p) - km(vpa(q), vpb(q), p)
        end do
      end do
      ! + 1/2 sum_n <mn||jk> t2[n][l][(cd)]
      call N(zc_gemm)(0_4, 0_4, npo, nvp, no, half, oo, no, t2p(0, l, 0), no * nvp, mone * mone, &
                      g(0, 0, l), nvp)
    end do
    !$omp parallel do private(it)
    do it = 0, nto - 1
      xo(:, m, it) = g(:, opair(it, 0), ofree(it, 0)) - g(:, opair(it, 1), ofree(it, 1)) &
                     + g(:, opair(it, 2), ofree(it, 2))
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! Xv for the pairs (x, l) of the nnew occupied x of xnew, each into its slot:
!   xv[e][(bcd)] = P(d/bc) 1/2 sum_f <bc||ef> t2[x][l][f][d]
! from nrw rows (bc) of <bc||ef> at a time, which are expanded once for all x.
! wx: work [(bc)][e][f], mx: work [(bc)][e][d]
subroutine N(xzqxv)(no, nv, nvp, nvt, nrw, nnew, xnew, l, slot, t2, vvvv, xv, wx, mx)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, nvp, nvt, nrw, nnew, l
  integer(8), intent(in) :: xnew(0:*), slot(0:*)
  ZT, intent(in) :: t2(0:nv*nv-1, 0:no-1, 0:no-1)              ! [i][j][(a, b)]
  ZT, intent(in) :: vvvv(0:nvp-1, 0:*)
  ZT :: xv(0:nv*nvt-1, 0:*), wx(*), mx(*)
  integer(8) :: b0, nb, k, x
  ZT :: one, zero
  one = 1
  zero = 0
  do k = 0, nnew - 1
    call zc_tn_zero(xv(0, slot(xnew(k))), 8 * NW * nv * nvt)
  end do
  do b0 = 0, nvp - 1, nrw
    nb = min(nrw, nvp - b0)
    call zc_tn_zero(wx, 8 * NW * nb * nv * nv)
    call N(zc_tn_expand)(vvvv(0, b0), nb, nvp, int(nv, 4), 1_4, 1_4, 0_8, nv, 0_8, nv, wx, nv, 1_8, &
                         nv * nv, 1d0)
    do k = 0, nnew - 1
      x = xnew(k)
      call N(zc_gemm)(0_4, 0_4, nb * nv, nv, nv, one, wx, nv, t2(0, l, x), nv, zero, mx, nv)
      call N(zc_tn_reduce)(xv(0, slot(x)), nv, nvt, int(nv, 4), 2_4, 1_4, b0, b0 + nb, 0_8, nv, mx, nv * nv, &
                           1_8, nv, 0.5d0)
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! One occupied quadruple i<j<k<l: e(1), e(2) = its parts of E[Q] and of
! E(Q) - E[Q].
subroutine N(xzquad)(w, i, j, k, l, na4, otri, osgn, t2pp, t2p, t2px, xo, ev4, z, rt3, rtt, w22, w13, &
                     la, ra, lb, rb, lc, rc, wh, xv, slot, e, tq)
  use xzdata
  use xzkern
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: i, j, k, l, na4
  integer(8), intent(in) :: otri(0:npo-1, 0:no-1)
  integer(1), intent(in) :: osgn(0:npo-1, 0:no-1)
  ZT, intent(in) :: t2pp(0:nvp-1, 0:npo-1), t2p(0:nvp-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: t2px(0:nv*nv-1, 0:npo-1)
  ZT, intent(in) :: xo(0:nvp-1, 0:no-1, 0:nto-1)
  real(8), intent(in) :: ev4(*)
  ZT :: z(*), rt3(*), rtt(*), w22(*), w13(*)
  ZT :: la(0:nvp-1, 0:nv-1, 0:3), ra(0:nvp-1, 0:nv-1, 0:3)
  ZT :: lb(0:nvp-1, 0:no-1, 0:3), rb(0:nvp-1, 0:no-1, 0:3)
  ZT :: lc(0:nv-1, 0:no-1, 0:5), rc(0:nvt-1, 0:no-1, 0:5)
  ZT :: wh(0:nvp-1, 0:5, 0:1)
  ZT, intent(in) :: xv(0:nv*nvt-1, 0:*)
  integer(8), intent(in) :: slot(0:*)
  real(8), intent(out) :: e(2)
  real(8), intent(inout) :: tq(6)
  integer(8) :: q(0:3), s, x, jt, m, c0, nc, p1, p2, n
  integer(8) :: rows(0:6*no-1), xfirst(0:5), pfirst(0:5)
  integer(1) :: sgn(0:6*no-1)
  real(8) :: sg, s13(0:3), s22(0:5), t0, eo
  integer(8), external :: xztrank
  integer(8), parameter :: f1(0:5) = [0, 0, 0, 1, 1, 2], f2(0:5) = [1, 2, 3, 2, 3, 3]
  integer(8), parameter :: r1(0:5) = [2, 1, 1, 0, 0, 0], r2(0:5) = [3, 3, 2, 3, 2, 1]
  integer(8), parameter :: last(3) = [0, 1, 3]
  ZT :: one, zero, alpha, beta
  integer(8), external :: xzpidx
  one = 1
  zero = 0
  q = [i, j, k, l]
  s13 = [1d0, -1d0, 1d0, -1d0]
  s22 = [1d0, -1d0, 1d0, 1d0, -1d0, 1d0]
  eo = w(jeps + i) + w(jeps + j) + w(jeps + k) + w(jeps + l)

  ! ---- z = <Q| V T2 |0>: t(ij,ab) <cd||kl> over the six splits (ij|kl)
  t0 = omp_get_wtime()
  do s = 0, 5
    p1 = xzpidx(q(f1(s)), q(f2(s)), no)
    p2 = xzpidx(q(r1(s)), q(r2(s)), no)
    pfirst(s) = p1
    xfirst(s) = q(r1(s))
    wh(:, s, 0) = s22(s) * t2pp(:, p1)
    call N(xzqconj)(nvp, w(jgpp + NW * nvp * p2), wh(0, s, 1))
  end do
  call zc_tn_zero(z, 8 * NW * na4)
  call zc_tn_zero(rt3, 8 * NW * na4)
  call zc_tn_zero(rtt, 8 * NW * na4)
  ! ---- the operands of the splits (i|jkl), stacked along the summed index
  do s = 0, 3
    x = q(s)
    select case (s)
    case (0)
      jt = xztrank(q(1), q(2), q(3), no)
    case (1)
      jt = xztrank(q(0), q(2), q(3), no)
    case (2)
      jt = xztrank(q(0), q(1), q(3), no)
    case default
      jt = xztrank(q(0), q(1), q(2), no)
    end select
    sg = s13(s)
    ! - <ab||ei> = <ie||ab>*:  la[e][(ab)] = sgn conj(<xe||ab>)
    call N(xzqla)(nv * nvp, sg, w(jovvq + NW * nv * nvp * x), la(0, 0, s))
    call zc_tn_zero(ra(0, 0, s), 8 * NW * nv * nvp)
    call N(zc_tn_expand)(w(jt3 + NW * nvt * jt), 1_8, nvt, int(nv, 4), 1_4, 2_4, 0_8, nv, 0_8, nvp, &
                         ra(0, 0, s), nvp, 1_8, 0_8, 1d0)
    lb(:, :, s) = sg * t2p(:, :, x)
    rb(:, :, s) = xo(:, :, jt)
  end do
  ! ---- the splits (ij|kl): <ma||ij> = <ij||ma>* and t(m kl; bcd)
  do s = 0, 5
    do m = 0, no - 1
      call N(xzqlc)(nv, s22(s), w(jooov + NW * nv * (m + no * (q(f2(s)) + no * q(f1(s))))), lc(0, m, s))
      p2 = xzpidx(q(r1(s)), q(r2(s)), no)
      rows(s * no + m) = otri(p2, m)
      sgn(s * no + m) = osgn(p2, m)
    end do
  end do
  call N(zc_tn_rows)(w(jt3), nvt, rows, sgn, 6 * no, 0_8, nvt, rc, nvt)
  tq(3) = tq(3) + omp_get_wtime() - t0

  do c0 = 0, nvp - 1, qc22
    nc = min(qc22, nvp - c0)
    t0 = omp_get_wtime()
    call N(zc_gemm)(1_4, 0_4, nvp, nc, 6_8, one, wh(0, 0, 0), nvp, wh(c0, 0, 1), nvp, zero, w22, nc)
    tq(4) = tq(4) + omp_get_wtime() - t0
    t0 = omp_get_wtime()
    call N(zc_tn_reduce)(z, 1_8, na4, int(nv, 4), 2_4, 2_4, 0_8, nvp, c0, c0 + nc, w22, nc, 1_8, 0_8, 1d0)
    tq(5) = tq(5) + omp_get_wtime() - t0
    t0 = omp_get_wtime()
    call N(zc_gemm)(1_4, 0_4, nvp, nc, 4 * nv, one, la, nvp, ra(c0, 0, 0), nvp, zero, w22, nc)
    tq(4) = tq(4) + omp_get_wtime() - t0
    t0 = omp_get_wtime()
    call N(zc_tn_reduce)(rt3, 1_8, na4, int(nv, 4), 2_4, 2_4, 0_8, nvp, c0, c0 + nc, w22, nc, 1_8, 0_8, 1d0)
    tq(5) = tq(5) + omp_get_wtime() - t0
    t0 = omp_get_wtime()
    call N(zc_gemm)(1_4, 0_4, nvp, nc, 4 * no, one, lb, nvp, rb(c0, 0, 0), nvp, zero, w22, nc)
    tq(4) = tq(4) + omp_get_wtime() - t0
    t0 = omp_get_wtime()
    call N(zc_tn_reduce)(rtt, 1_8, na4, int(nv, 4), 2_4, 2_4, 0_8, nvp, c0, c0 + nc, w22, nc, 1_8, 0_8, 1d0)
    tq(5) = tq(5) + omp_get_wtime() - t0
  end do
  do c0 = 0, nvt - 1, qc13
    nc = min(qc13, nvt - c0)
    t0 = omp_get_wtime()
    call N(zc_gemm)(1_4, 0_4, nv, nc, 6 * no, one, lc, nv, rc(c0, 0, 0), nvt, zero, w13, nc)
    tq(4) = tq(4) + omp_get_wtime() - t0
    t0 = omp_get_wtime()
    call N(zc_tn_reduce)(rt3, 1_8, na4, int(nv, 4), 1_4, 3_4, 0_8, nv, c0, c0 + nc, w13, nc, 1_8, 0_8, 1d0)
    tq(5) = tq(5) + omp_get_wtime() - t0
    t0 = omp_get_wtime()
    ! the splits whose second pair has l: (ij|kl), (ik|jl), (jk|il)
    do n = 1, 3
      s = last(n)
      alpha = 2d0 * s22(s)
      beta = one
      if (n == 1) beta = zero
      call N(zc_gemm)(1_4, 0_4, nv, nc, nv, alpha, t2px(0, pfirst(s)), nv, &
                      xv(c0, slot(xfirst(s))), nvt, beta, w13, nc)
    end do
    tq(4) = tq(4) + omp_get_wtime() - t0
    t0 = omp_get_wtime()
    call N(zc_tn_reduce)(rtt, 1_8, na4, int(nv, 4), 1_4, 3_4, 0_8, nv, c0, c0 + nc, w13, nc, 1_8, 0_8, 1d0)
    tq(5) = tq(5) + omp_get_wtime() - t0
  end do
  t0 = omp_get_wtime()
  e = 0d0
  call N(zc_tn_energy2)(z, rt3, rtt, ev4, eo, na4, e)
  if (qdebug) call N(xzqparts)(na4, z, rt3, rtt, ev4, eo, qparts)
  tq(6) = tq(6) + omp_get_wtime() - t0
end subroutine

! out = conj(a)
subroutine N(xzqconj)(n, a, out)
  implicit none
  integer(8), intent(in) :: n
  ZT, intent(in) :: a(n)
  ZT, intent(out) :: out(n)
  out = CONJ(a)
end subroutine

! out = sg * conj(a)
subroutine N(xzqla)(n, sg, a, out)
  implicit none
  integer(8), intent(in) :: n
  real(8), intent(in) :: sg
  ZT, intent(in) :: a(n)
  ZT, intent(out) :: out(n)
  integer(8) :: i
  !$omp parallel do
  do i = 1, n
    out(i) = sg * CONJ(a(i))
  end do
end subroutine

subroutine N(xzqlc)(n, sg, a, out)
  implicit none
  integer(8), intent(in) :: n
  real(8), intent(in) :: sg
  ZT, intent(in) :: a(n)
  ZT, intent(out) :: out(n)
  out = sg * CONJ(a)
end subroutine

! the parts of the energy of a quadruple (diagnostic, XZ_QDEBUG):
! z.[V,T3], z.T2T2, [V,T3].[V,T3], [V,T3].T2T2
subroutine N(xzqparts)(n, z, rt3, rtt, ev4, eo, s)
  implicit none
  integer(8), intent(in) :: n
  ZT, intent(in) :: z(n), rt3(n), rtt(n)
  real(8), intent(in) :: ev4(n), eo
  real(8), intent(inout) :: s(4)
  integer(8) :: i
  do i = 1, n
    s(1) = s(1) + dble(CONJ(z(i)) * rt3(i)) / (eo - ev4(i))
    s(2) = s(2) + dble(CONJ(z(i)) * rtt(i)) / (eo - ev4(i))
    s(3) = s(3) + dble(CONJ(rt3(i)) * rt3(i)) / (eo - ev4(i))
    s(4) = s(4) + dble(CONJ(rt3(i)) * rtt(i)) / (eo - ev4(i))
  end do
end subroutine
