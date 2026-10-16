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
! (Q)_Lambda (q_left of xzdata; docs/lambda.md, section 5): the same r, with
!
!   E[Q]_L = sum z2 r / D,        z2 = <0| Lambda_2 V |Q>
!   E(Q)_L = sum (z2 + z3) r / D, z3 = <0| Lambda_3 V |Q>
!
! on the left, nothing conjugated: z2 is the product of z with l2 and
! <kl||cd>, z3 the two products of [V,T3] with l3 in the place of T3 and the
! integrals the other way round (the conjugates of the stacked operands),
!
!   X[(ab)][(cd)]  + l(ij,ab) <kl||cd>                  (ij|kl)             z2
!                  + sum_e <ie||ab> l(jkl,cde)          (i|jkl)             z3
!   X[a][(bcd)]    + sum_m <ij||ma> l(mkl,bcd)          (ij|kl)             z3
!
! in the work arrays of [V,T3] after r is there; lambda is at jl2, jl3.
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
  integer(8) :: kl2pp, kz3, kc3
  integer(8) :: i, j, k, l, x, n, size, ng, g1, g2, g3, nquad, ndone
  integer(8) :: lo1, hi1, lo2, hi2, lo3, hi3
  integer(8) :: it, want, nold, nmine, lastl, nlimit, nx, nqnow, nstop
  integer(8) :: slot(0:no-1), xnew(0:no-1), nnew
  logical :: need(0:no-1), first
  real(8) :: e(2), e1(2), t0, tq(7), tstart, costmine, costleft, costdone, tloop, ttask, txv
  real(8) :: estimate, part(6), tot(6), tmx(2), tq0(7), qcm(0:4 * maxcls), xcm(0:2 * maxcls), xvextra, worst
  integer(8) :: nxvs(2), nseen, nrk(16), kb4(4)
  real(8) :: rk(8)
  real(8), allocatable :: rkall(:, :)
  integer :: ios
  logical :: qcost
  integer(8), external :: N(xzqslot)
  integer :: ldbg
  character(len=256) :: arg

  eq5 = 0d0
  eq6 = 0d0
  if (no < 4 .or. nv < 4) return
  call x2c_hasopt('xzccsdt', 'qdebug', ldbg)
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
  kl2pp = kt2pp
  if (q_left) kl2pp = xzget(NW * npo * nvp, '(Q): L2 [P][Q]')
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
    if (q_left) call N(xzqpp)(no, nv, npo, nvp, w(jl2), w(jopa), w(jopb), w(jvpa), w(jvpb), w(kl2pp))
    tq(1) = omp_get_wtime() - t0
  end if
  call xzrel(m1)

  ! ---- the work arrays of a quadruple
  if (ncls == 1) then
    kev4 = xzget(na4, '(Q): energies of the virtual quadruples')
    kz = xzget(NW * na4, '(Q): z')
    kz3 = kz
    kc3 = kz
    if (q_left) kz3 = xzget(NW * na4, '(Q): z of Lambda3')
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
    kz3 = kz
    kc3 = kw13
    if (q_left) then
      kz3 = xzget(NW * qn22max, '(Q): blocks of z of Lambda3 on two pairs')
      kc3 = xzget(NW * qn13max, '(Q): blocks of z of Lambda3 on an index and a triple')
    end if
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
  want = 0
  call x2c_getopt('xzccsdt', 'qtasks', arg, ldbg)
  if (ldbg > 0) read(arg, *) want
  qkind = '(Q)'
  if (q_left) qkind = '(Q)_L'
  call xzqmodel(qcm, xcm)
  call xzq_costs(no, ncls, ocls, qcm, xcm)
  ! (qranks="n1 n2 ..": what the list would be for these numbers of ranks)
  call x2c_getopt('xzccsdt', 'qranks', arg, ldbg)
  if (ldbg > 0 .and. myrank == 0) then
    nrk = 0
    read(arg, *, iostat=ios) nrk
    n = nrank
    do i = 1, 16
      if (nrk(i) < 1) exit
      nrank = nrk(i)
      call xzq_build(no, want)
      call xzq_xvstat(no, nxvs, xvextra, worst)
      write(6, '(a,i5,a,i6,a,i6,a,i6,a,f6.2,a,f6.2,a)') '   qranks: ', nrank, ' ranks: ', ntask, ' tasks, Xv ', &
        nxvs(2), ' of ', nxvs(1), ' pairs; the slowest rank ', 100d0 * worst, ' % above the mean; repeated Xv ', &
        100d0 * xvextra, ' % of the work'
    end do
    nrank = n
  end if
  call xzq_build(no, want)
  call xzq_xvstat(no, nxvs, xvextra, worst)
  qsample = 0d0
  call x2c_getopt('xzccsdt', 'qsample', arg, ldbg)
  if (ldbg > 0) read(arg, *) qsample
  if (qsample <= 0d0 .or. qsample >= 1d0) qsample = 0d0
  nstop = -1
  call x2c_getopt('xzccsdt', 'qstop', arg, ldbg)
  if (ldbg > 0) read(arg, *) nstop
  nold = 0
  ! (all ranks read the checkpoints of an earlier run before any of them writes one)
  call xzmpi_barrier()
  if (qsample == 0d0) call xzq_restore(no, nv, nwe, ncls, nold)
  call xzmpi_barrier()
  costmine = 0d0
  costleft = 0d0
  nmine = 0
  do it = 1, ntask
    if (town(it) /= myrank) cycle
    nmine = nmine + 1
    costmine = costmine + tcost(it)
    if (.not. tdone(it)) costleft = costleft + tcost(it)
  end do
  if (q_left) then
    write(6, '(/,a,i10,a,i14,a)') '   (Q)_Lambda correction: ', nquad, ' occupied x ', na4, ' virtual quadruples'
  else
    write(6, '(/,a,i10,a,i14,a)') '   (Q) correction: ', nquad, ' occupied x ', na4, ' virtual quadruples'
  end if
  write(6, '(a,i6,a,i5,a,i6,a,i5)') '   tasks: ', ntask, '   ranks: ', nrank, '   tasks of this rank: ', nmine, &
    '   rank ', myrank
  write(6, '(a,i6,a,i6,a,i6,a)') '   groups (one largest index l, one Xv): ', no - 3, ';  Xv(x, l): ', nxvs(1), &
    ' pairs, ', nxvs(2), ' formed over all ranks'
  if (nrank > 1) write(6, '(a,f6.2,a,f6.2,a)') '   cost model: the slowest rank ', 100d0 * worst, &
    ' % above the mean; the repeated Xv ', 100d0 * xvextra, ' % of the work'
  if (nold > 0) write(6, '(a,i6,a)') '   restart: ', nold, ' tasks were finished by an earlier run (files XZQCHK.*)'
  if (qsample > 0d0) write(6, '(a,f7.4,a)') '   qsample: the fraction ', qsample, &
    ' of the quadruples of every task is done; the energies are not those of (Q)'
  flush(6)

  call x2c_getopt('xzccsdt', 'qcost', arg, ldbg)
  qcost = ldbg > 0 .and. myrank == 0
  if (qcost) then
    open(unit=78, file=arg(1:ldbg), form='formatted', status='replace')
    write(78, '(a,4i8)') 'sizes ', no, nv, ncls, nwe
    write(78, '(a,64i6)') 'ocls ', ocls(0:no - 1)
    write(78, '(a,400i4)') 'vcls ', vcls(0:nv - 1)
    tq0 = tq
  end if
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
    nseen = 0
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
          if (qcost) then
            write(78, '(a,i4,i6,es12.4)') 'xv ', l, nnew, omp_get_wtime() - t0
            do n = 0, nnew - 1
              write(78, '(a,2i4)') 'x ', xnew(n), l
            end do
            tq0 = tq
          end if
          do i = lo1, hi1 - 1
            do j = max(i + 1, lo2), hi2 - 1
              do k = max(j + 1, lo3), hi3 - 1
                ! (a sample: nlimit of the quadruples of the task, evenly spread)
                nseen = nseen + 1
                if (nlimit < tnq(it)) then
                  if ((nseen * nlimit) / tnq(it) == ((nseen - 1) * nlimit) / tnq(it)) cycle
                end if
                if (ncls == 1) then
                  call N(xzquad)(w, i, j, k, l, na4, w(jotri), w(josgn), w(kt2pp), w(kt2p), w(kt2px), w(kxo), &
                                 w(kev4), w(kz), w(krt3), w(krtt), w(kw22), w(kw13), w(kla), w(kra), w(klb), &
                                 w(krb), w(klc), w(krc), w(kwh), w(kxv), slot, q_left, w(kl2pp), w(kz3), &
                                 e1, tq)
                else
                  call N(xzqsquad)(w, i, j, k, l, w(jotri), w(josgn), w(jopidx), w(jopgrp), w(kt2pp), &
                                   w(kt2p), w(kt2px), w(kxo), w(kz), w(krt3), w(krtt), w(kw22), w(kw13), &
                                   w(kla), w(kra), w(klb), w(krb), w(kwh), w(kwh + NW * 6 * nqmax), w(klc), &
                                   w(krc), w(krc + 6 * nocmax), w(kxv), slot, q_left, w(kl2pp), w(kz3), &
                                   w(kc3), e1, tq)
                end if
                e = e + e1
                ndone = ndone + 1
                if (qcost) then
                  ! (qcost: the times of every quadruple, for the cost model of the tasks)
                  write(78, '(4i4,i5,3es12.4)') i, j, k, l, ocls(i) + ocls(j) + ocls(k) + ocls(l), &
                    tq(3) - tq0(3), tq(4) - tq0(4), tq(5) - tq0(5)
                  tq0 = tq
                end if
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
      estimate = estimate + txv + (tsec(it) - txv) * dble(tnq(it)) / dble(max(ndone, 1_8))
    else
      if (ndone /= tnq(it)) then
        write(6, '(a,i6,2i12)') ' @XZCCSDT-F, (Q): quadruples done and expected in task ', it, ndone, tnq(it)
        call xzerrex
      end if
      te(:, it) = e
      tdone(it) = .true.
      call xzq_save(it, no, nv, nwe, ncls, first)
      ! (a test hook: qstop=n ends the run after n tasks, as an interruption would)
      nstop = nstop - 1
      if (nstop == 0) then
        write(6, '(a)') ' @XZCCSDT-F, qstop: stopped after the task above.'
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
  ! (the memory of the process while the arrays of (Q) are still in use)
  call xz_rss4(kb4)
  call xzrel(m0)
  if (qcost) close(78)

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
  if (q_left) write(6, '(5x,a28,f12.2,a)') 'left side with Lambda_3', tq(7), &
    ' s (in the three lines below)'
  write(6, '(5x,a28,f12.2,a)') 'operands', tq(3), ' s'
  write(6, '(5x,a28,f12.2,a)') 'products (GEMM)', tq(4), ' s'
  write(6, '(5x,a28,f12.2,a)') 'reductions', tq(5), ' s'
  write(6, '(5x,a28,f12.2,a)') 'energy', tq(6), ' s'
  write(6, '(5x,a28,f12.2,a)') 'total of this rank', part(4), ' s'
  ! what every rank did and holds: its own (anonymous) memory and the pages
  ! of mapped files that it has touched (shared with the ranks of its node)
  rk = [part(4), tq(2), tq(1), dble(nmine), part(6), 1d-3 * kb4(2), 1d-3 * kb4(3), 1d-3 * kb4(1)]
  allocate(rkall(8, 0:nrank - 1))
  call xzmpi_gather(rk, rkall, 8_8)
  if (myrank == 0) then
    write(6, '(/,a)') '    rank   tasks  quadruples     total/s      Xv/s      Xo/s   own memory/MB' // &
      '   mapped files/MB   peak resident/MB'
    do n = 0, nrank - 1
      write(6, '(i8,i8,i12,f12.2,2f10.2,f16.1,f18.1,f19.1)') n, nint(rkall(4, n), 8), nint(rkall(5, n), 8), &
        rkall(1, n), rkall(2, n), rkall(3, n), rkall(6, n), rkall(7, n), rkall(8, n)
    end do
    if (nrank > 1) write(6, '(a,f10.2,a,f12.1,a)') '    all ranks: Xv ', sum(rkall(2, :)), ' s;  own memory ', &
      sum(rkall(6, :)), ' MB'
    if (nrank > 1) write(6, '(5x,a28,f12.2,a,f8.2)') 'slowest rank', tmx(1), ' s;  sum over ranks / slowest ', &
      tot(4) / max(tmx(1), 1d-30)
    if (qsampled) then
      write(6, '(/,a,i10,a,i10,a)') '   sample: ', nint(tot(6), 8), ' of ', nquad, ' quadruples done'
      write(6, '(a,f12.1,a)') '   estimated wall time of the whole (Q), slowest rank: ', tmx(2) + tq(1), ' s'
      write(6, '(a)') '   (Xv formed as in the whole calculation; the quadruples of a task in proportion)'
    else if (nint(tot(3), 8) /= nquad) then
      write(6, '(a,2i12)') ' @XZCCSDT-F, (Q): occupied quadruples done and expected: ', nint(tot(3), 8), nquad
      call xzerrex
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
  call xzerrex
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
                     la, ra, lb, rb, lc, rc, wh, xv, slot, left, l2pp, z3, e, tq)
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
  logical, intent(in) :: left                                  ! lambda on the left: (Q)_Lambda
  ZT, intent(in) :: l2pp(0:nvp-1, 0:npo-1)                     ! (left)
  ZT :: z3(*)                                                  ! (left)
  real(8), intent(out) :: e(2)
  real(8), intent(inout) :: tq(7)
  integer(8) :: q(0:3), s, x, jt, m, c0, nc, p1, p2, n, jts(0:3)
  integer(8) :: rows(0:6*no-1), xfirst(0:5), pfirst(0:5)
  real(8) :: tl0
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
    if (left) then
      ! l(ij,ab) <kl||cd>
      wh(:, s, 0) = s22(s) * l2pp(:, p1)
      call N(xzqcopy)(nvp, 1d0, .false., w(jgpp + NW * nvp * p2), wh(0, s, 1))
    else
      wh(:, s, 0) = s22(s) * t2pp(:, p1)
      call N(xzqconj)(nvp, w(jgpp + NW * nvp * p2), wh(0, s, 1))
    end if
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
    jts(s) = jt
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
  if (left) then
    ! ---- z3 = <0| Lambda_3 V |Q>: the two products of [V,T3] with l3 and
    !      the integrals the other way round, in the same work arrays
    tl0 = omp_get_wtime()
    t0 = omp_get_wtime()
    call zc_tn_zero(z3, 8 * NW * na4)
    do s = 0, 3
      call N(xzqcopy)(nv * nvp, s13(s), .false., w(jovvq + NW * nv * nvp * q(s)), la(0, 0, s))
      call zc_tn_zero(ra(0, 0, s), 8 * NW * nv * nvp)
      call N(zc_tn_expand)(w(jl3 + NW * nvt * jts(s)), 1_8, nvt, int(nv, 4), 1_4, 2_4, 0_8, nv, 0_8, nvp, &
                           ra(0, 0, s), nvp, 1_8, 0_8, 1d0)
    end do
    do s = 0, 5
      do m = 0, no - 1
        call N(xzqcopy)(nv, s22(s), .false., w(jooov + NW * nv * (m + no * (q(f2(s)) + no * q(f1(s))))), &
                        lc(0, m, s))
      end do
    end do
    call N(zc_tn_rows)(w(jl3), nvt, rows, sgn, 6 * no, 0_8, nvt, rc, nvt)
    tq(3) = tq(3) + omp_get_wtime() - t0
    do c0 = 0, nvp - 1, qc22
      nc = min(qc22, nvp - c0)
      t0 = omp_get_wtime()
      call N(zc_gemm)(1_4, 0_4, nvp, nc, 4 * nv, one, la, nvp, ra(c0, 0, 0), nvp, zero, w22, nc)
      tq(4) = tq(4) + omp_get_wtime() - t0
      t0 = omp_get_wtime()
      call N(zc_tn_reduce)(z3, 1_8, na4, int(nv, 4), 2_4, 2_4, 0_8, nvp, c0, c0 + nc, w22, nc, 1_8, 0_8, 1d0)
      tq(5) = tq(5) + omp_get_wtime() - t0
    end do
    do c0 = 0, nvt - 1, qc13
      nc = min(qc13, nvt - c0)
      t0 = omp_get_wtime()
      call N(zc_gemm)(1_4, 0_4, nv, nc, 6 * no, one, lc, nv, rc(c0, 0, 0), nvt, zero, w13, nc)
      tq(4) = tq(4) + omp_get_wtime() - t0
      t0 = omp_get_wtime()
      call N(zc_tn_reduce)(z3, 1_8, na4, int(nv, 4), 1_4, 3_4, 0_8, nv, c0, c0 + nc, w13, nc, 1_8, 0_8, 1d0)
      tq(5) = tq(5) + omp_get_wtime() - t0
    end do
    tq(7) = tq(7) + omp_get_wtime() - tl0
  end if
  t0 = omp_get_wtime()
  e = 0d0
  if (left) then
    call N(zc_ql_energy)(z, z3, rt3, rtt, ev4, eo, na4, e)
  else
    call N(zc_tn_energy2)(z, rt3, rtt, ev4, eo, na4, e)
    if (qdebug) call N(xzqparts)(na4, z, rt3, rtt, ev4, eo, qparts)
  end if
  tq(6) = tq(6) + omp_get_wtime() - t0
end subroutine

! l2pp[P][Q] of l2[i][j][a][b]
subroutine N(xzqpp)(no, nv, npo, nvp, l2, opa, opb, vpa, vpb, l2pp)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: l2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(out) :: l2pp(0:nvp-1, 0:npo-1)
  integer(8) :: p, q
  do p = 0, npo - 1
    do q = 0, nvp - 1
      l2pp(q, p) = l2(vpb(q), vpa(q), opb(p), opa(p))
    end do
  end do
end subroutine

! out = sg * a, or sg * conj(a)
subroutine N(xzqcopy)(n, sg, cj, a, out)
  implicit none
  integer(8), intent(in) :: n
  real(8), intent(in) :: sg
  logical, intent(in) :: cj
  ZT, intent(in) :: a(n)
  ZT, intent(out) :: out(n)
  integer(8) :: i
  if (cj) then
    !$omp parallel do if(n > 100000)
    do i = 1, n
      out(i) = sg * CONJ(a(i))
    end do
  else
    !$omp parallel do if(n > 100000)
    do i = 1, n
      out(i) = sg * a(i)
    end do
  end if
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

! the parts of the energy of a quadruple (diagnostic, qdebug):
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
