!
! CC4 without stored quadruples (type-generic; docs/native.md, section 19;
! docs/ccn.md, section 11; the Python program: src/zccsdt/cc/cc4fly.py).
!
! The model: the singles, doubles and triples equations of CCSDTQ, the
! quadruples equation to the lowest order,
!
!   <Q| [V~, T3] + 1/2 [[V~, T2], T2] + [F, T4] |0> = 0,
!
! V~ the T1-dressed two-body part, F the Fock operator of the reference
! (diagonal for CFOUR's canonical spinors).  T4 enters the doubles and the
! triples equations through four sums that are linear in it:
!
!   R2[ij, ab]   += sum_{m<n, e<f} <mn||ef> t4[ijmn, abef]
!   R3[ijk, abc] += sum_{m e} f~[m][e] t4[ijkm, abce]
!                 - P(c/ab) sum_{m, e<f} <cm||ef>~ t4[ijkm, abef]
!                 - P(k/ij) sum_{m<n, e} <mn||ek>~ t4[ijmn, abce]
!
! N(xzt4) makes them for given T1, T2, T3: for one occupied quadruple at a
! time the right-hand side - the quantity r of the (Q) correction (xzq.inc,
! xzqs.inc: five products) with the T1-dressed integrals in the place of the
! bare ones and without the conjugation shortcuts of (Q), which hold for
! bare integrals only -, the row t4 = r / (e_ijkl - e_abcd + omega), its
! contractions into the four sums, and the row is gone: nothing of the size
! of T4 is kept.  The loop over the occupied quadruples is the task list of
! (Q) (xzpar.F90): every rank its tasks and its own sums, which are added.
!
! The iterations (xzcc.inc) take the two sums as fixed parts of the
! residuals of the CCSDT step (xzstep.inc: cc4 of xzdata).
!

! ---------------------------------------------------------------------------
! The T4 terms of the doubles and triples equations for the amplitudes at
! kt1, kt2, kt3:
!   kr2   [P][Q], P = (i<j), Q = (a<b)
!   kr3   as T3 is stored (on a rank other than 0: its part; rank 0 has the sum)
! omega: added to the occupied energies of the denominator.
subroutine N(xzt4)(w, kt1, kt2, kt3, kr2, kr3, omega)
  use xzdata
  use xzmem
  use xzkern
  use xzpar
  use xzdist
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kt1, kt2, kt3, kr2, kr3
  real(8), intent(in) :: omega
  integer(8) :: m0, m1, ov, na4, nslot, nxv, nrb
  integer(8) :: kfov, kooovd, kd, kq1, kq2, ktau, kt2pp, kt2p, kt2px, kxo
  integer(8) :: kfoo, kfvo, kfvv, kpvv, kooood, kovvod, kovood, kk, ktt, kwv, kg2, kyb, kslab, kgq, kgh, kpib, ktm
  integer(8) :: kg, kkm, kb, koo
  integer(8) :: kev4, kz, kw22, kw13, kla, kra, klb, krb, klc, krc, kwh, kxv, kwx, kmx, ky, klk, kbo, ktg
  integer(8) :: ke22, ke13, kg1, kgp, qy12
  integer(8) :: i, j, k, l, x, n, size, ng, g1, g2, g3, h, lo, nqh, q0, nq, qq, g
  integer(8) :: lo1, hi1, lo2, hi2, lo3, hi3, it, lastl, nx, nnew, ndone, nmine
  integer(8) :: slot(0:no-1), xnew(0:no-1)
  logical :: need(0:no-1), dense
  real(8) :: t0, tq(6), tstart, qcm(0:4 * maxcls), xcm(0:2 * maxcls)
  integer(8), external :: N(xzqslot)

  if (no < 4 .or. nv < 4) then
    if (.not. xzdry) then
      call zc_vzero_d(NW * npo * nvp, w(kr2))
      call zc_vzero_d(NW * n3, w(kr3))
    end if
    return
  end if
  ov = no * nv
  dense = ncls == 1
  na4 = nv * (nv - 1) * (nv - 2) * (nv - 3) / 24
  nslot = max(3_8, min(qxv, no - 1))
  tq = 0d0
  tstart = omp_get_wtime()
  m0 = xzmark()

  ! ---- what the quadruples read: the dressed blocks, T2 in three layouts, Xo
  kfov = xzget(NW * ov, 'CC4: dressed f[m][e]')
  kooovd = xzget(NW * no * no * ov, 'CC4: dressed <mn||ie>')
  kd = xzget(NW * ov * nvp, 'CC4: dressed <mb||ef>')
  kq1 = xzget(NW * ov * nvp, 'CC4: dressed <ab||ex> [x][e][Q]')
  kq2 = xzget(NW * no * no * ov, 'CC4: dressed <ma||ij> [i][j][m][a]')
  ktau = xzget(NW * npo * nvp, 'CC4: tau1')
  kt2pp = xzget(NW * npo * nvp, 'CC4: T2 [P][Q]')
  kt2p = xzget(NW * no * no * nvp, 'CC4: T2 [i][m][Q]')
  kt2px = xzget(NW * npo * nv * nv, 'CC4: T2 [P][e][a]')
  kxo = xzget(NW * nto * no * nvp, 'CC4: Xo')
  ktm = xzget(NW * ov, 'CC4: - T1')
  m1 = xzmark()
  kfoo = xzget(NW * no * no, 'CC4: work of the dressing')
  kfvo = xzget(NW * ov, 'CC4: work of the dressing')
  kfvv = xzget(NW * nv * nv, 'CC4: work of the dressing')
  kpvv = xzget(NW * nv * nv, 'CC4: work of the dressing')
  kooood = xzget(NW * no ** 4, 'CC4: dressed <oo||oo>')
  kovvod = xzget(NW * ov * ov, 'CC4: dressed <ov||vo>')
  kovood = xzget(NW * no * no * ov, 'CC4: dressed <ov||oo>')
  kk = xzget(NW * ov * ov, 'CC4: work of the dressing')
  ktt = xzget(NW * npo * nvp, 'CC4: T2 + tau1')
  kwv = xzget(NW * nvp * ov, 'CC4: dressed <ab||ei> [Q][e][i]')
  kg2 = xzget(NW * npo * ov, 'CC4: work [P][e][i]')
  kyb = xzget(NW * nba * nv * nv * no, 'CC4: batch of the dressing')
  kslab = xzget(NW * nqs * nqmax, 'CC4: slab of the dressed <ab||cd>')
  kgq = jovvq
  kgh = jgpp
  kpib = jvpidx
  if (.not. dense) then
    kgq = xzget(NW * ov * nqmax, 'CC4: <ov||vv> on the pairs of a group')
    kgh = xzget(NW * npo * nqmax, 'CC4: <oo||vv> on the pairs of a group')
    kpib = xzget(nv * nv, 'CC4: pairs of a group')
  end if
  kg = xzget(NW * no * npo * nvp, 'CC4: work of Xo')
  kkm = xzget(NW * npo * nv * nv, 'CC4: work of Xo')
  kb = xzget(NW * nv * nv, 'CC4: work of Xo')
  koo = xzget(NW * npo * no, 'CC4: work of Xo')
  if (.not. xzdry) then
    t0 = omp_get_wtime()
    call N(xzfock)(no, nv, nvp, w(jeps), w(kt1), w(jooov), w(joovv), w(jovvo), w(jovvq), &
                   w(jvpa), w(jvpb), w(kfoo), w(kfov), w(kfvo), w(kfvv), w(kpvv))
    call N(xzdroo)(no, nv, w(kt1), w(jooov), w(joovv), w(joooo), w(kooovd), w(kooood))
    call N(xzdrovvq)(no, nv, nvp, w(kt1), w(jovvq), w(joovvp), w(kd))
    call N(xzdrov)(no, nv, nvp, w(kt1), w(jooov), w(jovvo), w(jovvq), w(kd), w(kooood), &
                   w(jvpidx), w(kovvod), w(kovood), w(kk))
    call N(xzt2forms)(no, nv, npo, nvp, w(kt1), w(kt2), w(jopa), w(jopb), w(jvpa), w(jvpb), &
                      w(kt2p), w(kt2pp), w(ktau), w(ktt))
    call N(xzqt2)(no, nv, npo, nvp, w(kt2), w(jopa), w(jopb), w(jvpa), w(jvpb), w(kt2pp), w(kt2p), w(kt2px))
    call N(xz4neg)(ov, w(kt1), w(ktm))
    ! the dressed <ab||ei>: its bra, then + sum_f ~<ab||ef> t1[i][f] from the rows of the dressed <ab||cd>
    call N(xzbv)(no, nv, npo, nvp, nba, w(kt1), w(ktau), w(jovvq), w(jooov), w(jovvo), &
                 w(jopa), w(jopb), w(jvpa), w(jvpb), w(kwv), w(kg2), w(kyb))
    do h = 0, npg - 1
      lo = vpoff(h)
      nqh = vpoff(h + 1) - lo
      if (nqh == 0) cycle
      if (.not. dense) call N(xzgrpcols)(no, nv, npo, nvp, lo, nqh, w(jovvq), w(jgpp), w(jvpidx), w(kgq), &
                                         w(kgh), w(kpib))
      do q0 = 0, nqh - 1, nqs
        nq = min(nqs, nqh - q0)
        call N(xzdrrows)(no, nv, npo, nvp, nqh, nq, w(kt1), w(ktau + NW * (lo + q0)), &
                         w(jvvvv + NW * (vvoff(h) + q0 * nqh)), w(kgq), w(kgh), w(jvpa + lo + q0), &
                         w(jvpb + lo + q0), w(kslab), w(kk))
        call N(zc_pair_t1_contract)(w(kwv + NW * (lo + q0) * ov), w(kslab), w(kt1), nq, nv, nqh, w(kpib), no)
      end do
    end do
    call N(xz4q12)(no, nv, nvp, w(kwv), w(kovood), w(kq1), w(kq2))
    call N(xzqxo)(no, nv, npo, nvp, nto, w(kooood), w(kovvod), w(kt2p), w(kt2px), w(jopa), w(jopb), &
                  w(jvpa), w(jvpb), w(jofree), w(jopair), w(kxo), w(kg), w(kkm), w(kb), w(koo))
    tq(1) = omp_get_wtime() - t0
  end if
  call xzrel(m1)

  ! ---- the work arrays of a quadruple
  nrb = 4 + 6 * no
  if (dense) then
    kev4 = xzget(na4, 'CC4: energies of the virtual quadruples')
    kz = xzget(NW * na4, 'CC4: the row of T4')
    kw22 = xzget(NW * nvp * qc22, 'CC4: product [(ab)][(cd)]')
    kw13 = xzget(NW * nv * qc13, 'CC4: product [a][(bcd)]')
    kla = xzget(NW * 4 * nv * nvp, 'CC4: stacked operand')
    kra = xzget(NW * 4 * nv * nvp, 'CC4: stacked operand')
    klb = xzget(NW * 4 * no * nvp, 'CC4: stacked operand')
    krb = xzget(NW * 4 * no * nvp, 'CC4: stacked operand')
    klc = xzget(NW * 6 * no * nv, 'CC4: stacked operand')
    krc = xzget(NW * 6 * no * nvt, 'CC4: stacked operand')
    kwh = xzget(NW * 12 * nvp, 'CC4: operands of the doubles term')
    ky = xzget(NW * nv * qc22, 'CC4: product [c][(ab)]')
    klk = xzget(NW * nrb * nv, 'CC4: operand of the triples terms')
    kbo = xzget(NW * nrb * qc13, 'CC4: product of the triples terms')
    ktg = xzget(nrb, 'CC4: rows of the triples terms')
    nxv = nv * nvt
    kxv = xzget(NW * nslot * nxv, 'CC4: Xv')
    kwx = xzget(NW * qrows * nv * nv, 'CC4: rows of <vv||vv>')
    kmx = xzget(NW * qrows * nv * nv, 'CC4: product of Xv')
    kg1 = xzget(NW * qrows * nvp, 'CC4: dressed rows of <vv||vv>')
    ke22 = kz
    ke13 = kz
    kgp = kz
  else
    nrb = 4 + 6 * nocmax
    jqkv = xzget(nv / 2 + 1, 'CC4: tables')
    jqlid = xzget(ncls / 2 + 1, 'CC4: tables')
    jqlst = xzget(ncls / 2 + 2, 'CC4: tables')
    jqlmem = xzget(nv / 2 + 1, 'CC4: tables')
    jqposv = xzget(nv / 2 + 1, 'CC4: tables')
    jqposq = xzget(nvp / 2 + 1, 'CC4: tables')
    jqpos3 = xzget(nvt / 2 + 1, 'CC4: tables')
    jqtri = xzget(3 * nvt / 2 + 2, 'CC4: tables')
    kz = xzget(NW * qn22max, 'CC4: blocks of the right-hand side on two pairs')
    ke22 = xzget(NW * qn22max, 'CC4: blocks of T4 on two pairs')
    kw13 = xzget(NW * qn13max, 'CC4: blocks of the right-hand side on an index and a triple')
    ke13 = xzget(NW * qn13max, 'CC4: blocks of T4 on an index and a triple')
    kla = xzget(NW * 4 * nvcmax * nqmax, 'CC4: stacked operand')
    kra = xzget(NW * 4 * nvcmax * nqmax, 'CC4: stacked operand')
    klb = xzget(NW * 4 * nocmax * nqmax, 'CC4: stacked operand')
    krb = xzget(NW * 4 * nocmax * nqmax, 'CC4: stacked operand')
    klc = xzget(NW * 6 * nocmax * nvcmax, 'CC4: stacked operand')
    krc = xzget(12 * nocmax, 'CC4: rows of T3 of a product')
    ! the blocks [c][(ab)] of the term with <cm||ef>: c of class g - q for the pairs of group q
    qy12 = 0
    do g = 0, ntg - 1
      n = 0
      do qq = 0, npg - 1
        n = n + xznvc(g - qq) * (vpoff(qq + 1) - vpoff(qq))
      end do
      qy12 = max(qy12, n)
    end do
    ky = xzget(NW * qy12, 'CC4: blocks [c][(ab)]')
    klk = xzget(NW * nrb * nvcmax, 'CC4: operand of the triples terms')
    kbo = xzget(NW * nrb * natmax, 'CC4: product of the triples terms')
    ktg = xzget(nrb, 'CC4: rows of the triples terms')
    nxv = qxvmax
    kxv = xzget(NW * nslot * nxv, 'CC4: Xv')
    kwx = xzget(NW * qwbmax, 'CC4: block of <vv||vv>')
    kmx = xzget(NW * qm3max, 'CC4: products of Xv')
    kg1 = xzget(NW * ov * nvcmax * nvcmax, 'CC4: <my||ef> of two classes')
    kgp = xzget(NW * npo * nvcmax * nvcmax, 'CC4: <mn||ef> of two classes')
    kev4 = kz
    kw22 = kz
    kwh = kz
  end if
  if (xzdry) then
    call xzrel(m0)
    return
  end if
  call zc_vzero_d(NW * npo * nvp, w(kr2))
  call zc_vzero_d(NW * n3, w(kr3))
  if (dense) then
    call zc_tn_tuple_sums(w(jeps + no), int(nv, 4), 4_4, w(kev4))
  else
    call xzqstab(nv, nvp, nvt, ncls, vcls, vcoff, vpoff, vtoff, w(jvpidx), w(jvpgrp), w(jvtri), w(jvfree), &
                 w(jqkv), w(jqlid), w(jqlst), w(jqlmem), w(jqposv), w(jqposq), w(jqpos3), w(jqtri))
  end if

  ! ---- the tasks of (Q) (xzpar.F90): groups of one largest index l, dealt to the ranks
  call xzqmodel(qcm, xcm)
  call xzq_costs(no, ncls, ocls, qcm, xcm)
  call xzq_build(no, 0_8)
  lastl = -1
  ndone = 0
  nmine = 0
  do it = 1, ntask
    if (town(it) /= myrank) cycle
    nmine = nmine + 1
    l = tl(it)
    if (l /= lastl) slot = -1
    lastl = l
    ! Xv for the x in i0 .. l-1: all at once if they fit, else in three parts
    nx = l - ti0(it)
    size = nx
    if (qxv < nx) size = max(1_8, qxv / 3)
    ng = (nx + size - 1) / size
    do g1 = 0, ng - 1
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
              if (.not. dense) call N(xz4sxv)(w, x, l, w(kt1), w(ktm), w(ktau), w(kt2), w(jqtri), &
                                              w(kxv + NW * slot(x) * nxv), w(kmx), w(kwx), w(kg1), w(kgp))
            end if
          end do
          if (dense .and. nnew > 0) call N(xz4xv)(w, nnew, xnew, l, slot, w(kt1), w(ktau), w(kt2), w(kxv), &
                                                  w(kwx), w(kmx), w(kg1), w(ktm))
          tq(2) = tq(2) + omp_get_wtime() - t0
          do i = lo1, hi1 - 1
            do j = max(i + 1, lo2), hi2 - 1
              do k = max(j + 1, lo3), hi3 - 1
                if (dense) then
                  call zc_tn_zero(w(kz), 8 * NW * na4)
                  call N(xz4fwd)(w, i, j, k, l, na4, w(jotri), w(josgn), w(kt2p), w(kt2px), w(kxo), w(kq1), &
                                 w(kq2), kt3, 1d0, w(kz), w(kw22), w(kw13), w(kla), w(kra), w(klb), w(krb), &
                                 w(klc), w(krc), w(kxv), slot, tq)
                  t0 = omp_get_wtime()
                  call N(zc_cc4_divide)(w(kz), w(kev4), w(jeps + i) + w(jeps + j) + w(jeps + k) + w(jeps + l) &
                                        + omega, na4)
                  tq(5) = tq(5) + omp_get_wtime() - t0
                  call N(xz4back)(w, i, j, k, l, na4, w(jotri), w(josgn), w(kz), w(kw22), w(kw13), w(kwh), &
                                  w(ky), w(klk), w(kbo), w(ktg), w(kfov), w(jgpp), kd, w(kooovd), w(kr2), &
                                  kr3, tq)
                else
                  call N(xz4squad)(w, i, j, k, l, w(jotri), w(josgn), w(jopidx), w(jopgrp), w(kt2p), w(kt2px), &
                                   w(kxo), w(kq1), w(kq2), kt3, w(kz), w(kw13), w(ke22), w(ke13), w(kla), &
                                   w(kra), w(klb), w(krb), w(klc), w(krc), w(krc + 6 * nocmax), w(kxv), slot, &
                                   omega, w(jqtri), w(ky), w(klk), w(kbo), w(ktg), w(kfov), kd, w(kooovd), &
                                   kr2, kr3, tq)
                end if
                ndone = ndone + 1
              end do
            end do
          end do
        end do
      end do
    end do
  end do
  call xzrel(m0)
  ! ---- the sums of the ranks: the doubles term on all of them, the triples term on rank 0
  if (nrank > 1) then
    t0 = omp_get_wtime()
    call xzmpi_allsum_w(w(kr2), NW * npo * nvp)
    call xzmpi_sum0_w(w(kr3), NW * n3)
    tq(1) = tq(1) + omp_get_wtime() - t0
  end if
  write(6, '(a,i10,a,i5,a,f10.2,a,f9.2,a,f9.2,a,3f10.2,a,f10.2,a)') '     T4 terms: ', ndone, ' occupied quadruples, ', &
    nmine, ' tasks, ', omp_get_wtime() - tstart, ' s  (operands ', tq(1), ', Xv ', tq(2), &
    ', right-hand sides: operands, products, reductions ', tq(3), tq(4), tq(5), ', contractions ', tq(6), ')'
  flush(6)
end subroutine

! out = - a
subroutine N(xz4neg)(n, a, out)
  implicit none
  integer(8), intent(in) :: n
  ZT, intent(in) :: a(n)
  ZT, intent(out) :: out(n)
  out = -a
end subroutine

! ---------------------------------------------------------------------------
! The two dressed blocks of the products with T3 in the layouts of (Q):
!   q1[x][e][Q] = - ~<Q||ex>  (wv[Q][e][x]; (Q) has <xe||ab>* there)
!   q2[i][j][m][a] = ~<ma||ij>  (ovood[m][a][i][j]; (Q) has <ij||ma>*)
subroutine N(xz4q12)(no, nv, nvp, wv, ovood, q1, q2)
  implicit none
  integer(8), intent(in) :: no, nv, nvp
  ZT, intent(in) :: wv(0:no-1, 0:nv-1, 0:nvp-1)                ! [Q][e][i]
  ZT, intent(in) :: ovood(0:no-1, 0:no-1, 0:nv-1, 0:no-1)      ! [m][b][i][j]
  ZT, intent(out) :: q1(0:nvp-1, 0:nv-1, 0:no-1)               ! [x][e][Q]
  ZT, intent(out) :: q2(0:nv-1, 0:no-1, 0:no-1, 0:no-1)        ! [i][j][m][a]
  integer(8) :: x, e, q, i, j, m, a
  !$omp parallel do collapse(2) private(x, e, q)
  do x = 0, no - 1
    do e = 0, nv - 1
      do q = 0, nvp - 1
        q1(q, e, x) = -wv(x, e, q)
      end do
    end do
  end do
  !$omp parallel do collapse(2) private(i, j, m, a)
  do i = 0, no - 1
    do j = 0, no - 1
      do m = 0, no - 1
        do a = 0, nv - 1
          q2(a, m, j, i) = ovood(j, i, a, m)
        end do
      end do
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! Xv of N(xzqxv) with the T1-dressed <bc||ef>: the rows of a step are dressed
! (N(xzdrrows): the kernel of the sweep of the CCSDT step) before they are
! expanded.  sl: work nrw * nvp; tm: work no * nv.
subroutine N(xz4xv)(w, nnew, xnew, l, slot, t1, tau, t2, xv, wx, mx, sl, tm)
  use xzdata
  use xzkern
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: nnew, l
  integer(8), intent(in) :: xnew(0:*), slot(0:*)
  ZT, intent(in) :: t1(*), tau(0:nvp-1, 0:*)
  ZT, intent(in) :: t2(0:nv*nv-1, 0:no-1, 0:no-1)              ! [i][j][(a, b)]
  ZT :: xv(0:nv*nvt-1, 0:*), wx(*), mx(*), sl(*), tm(*)
  integer(8) :: b0, nb, k, x
  ZT :: one, zero
  one = 1
  zero = 0
  do k = 0, nnew - 1
    call zc_tn_zero(xv(0, slot(xnew(k))), 8 * NW * nv * nvt)
  end do
  do b0 = 0, nvp - 1, qrows
    nb = min(qrows, nvp - b0)
    call N(xzdrrows)(no, nv, npo, nvp, nvp, nb, t1, tau(b0, 0), w(jvvvv + NW * b0 * nvp), w(jovvq), w(jgpp), &
                     w(jvpa + b0), w(jvpb + b0), sl, tm)
    call zc_tn_zero(wx, 8 * NW * nb * nv * nv)
    call N(zc_tn_expand)(sl, nb, nvp, int(nv, 4), 1_4, 1_4, 0_8, nv, 0_8, nv, wx, nv, 1_8, nv * nv, 1d0)
    do k = 0, nnew - 1
      x = xnew(k)
      call N(zc_gemm)(0_4, 0_4, nb * nv, nv, nv, one, wx, nv, t2(0, l, x), nv, zero, mx, nv)
      call N(zc_tn_reduce)(xv(0, slot(x)), nv, nvt, int(nv, 4), 2_4, 1_4, b0, b0 + nb, 0_8, nv, mx, nv * nv, &
                           1_8, nv, 0.5d0)
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! Xv of N(xzqsxv) (the blocks of the pair (x, l)) with the T1-dressed
! <bc||ef>: the block [(bc)][e][f] of a group of pairs and two classes is
! dressed where it is,
!   - sum_m t1[m][b] <mc||ef> + sum_m t1[m][c] <mb||ef> + sum_{m<n} tau1[mn][bc] <mn||ef>,
! with <my||ef> and <mn||ef> of the two classes (g1, gp: work).  tm = - t1.
subroutine N(xz4sxv)(w, x, l, t1, tm, tau, t2, tri, xv, m3, wb, g1, gp)
  use xzdata
  use xzkern
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: x, l
  ZT, intent(in) :: t1(*), tm(*), tau(0:nvp-1, 0:*)
  ZT, intent(in) :: t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)         ! [i][j][a][b]
  integer(4), intent(in) :: tri(0:2, 0:nvt-1)
  ZT :: xv(0:*), m3(0:*), wb(*), g1(*), gp(*)
  integer(8) :: kxl, ke, t, ne, nt, q2, kf, kd, nf, nd, nq, n, xoff
  integer(8) :: off(0:npg-1), ndd(0:npg-1)
  ZT :: one, zero
  one = 1
  zero = 0
  kxl = ocls(x) + ocls(l)
  xoff = 0
  do ke = 0, ncls - 1
    t = ke + kxl
    if (t >= ntg) cycle
    ne = vcoff(ke + 1) - vcoff(ke)
    nt = vtoff(t + 1) - vtoff(t)
    if (ne == 0 .or. nt == 0) cycle
    off = -1
    ndd = 0
    n = 0
    do q2 = 0, npg - 1
      kf = q2 - ke
      if (kf < 0 .or. kf >= ncls) cycle
      kd = kxl - kf
      if (kd < 0 .or. kd >= ncls) cycle
      nf = vcoff(kf + 1) - vcoff(kf)
      nd = vcoff(kd + 1) - vcoff(kd)
      nq = vpoff(q2 + 1) - vpoff(q2)
      if (nf == 0 .or. nd == 0 .or. nq == 0) cycle
      off(q2) = n
      ndd(q2) = nd
      call N(xzqswb)(nv, nq, ne, nf, vcoff(ke), vcoff(kf), vpoff(q2), w(jvpidx), w(jvvvv + NW * vvoff(q2)), wb)
      call N(xz4g12)(no, nv, npo, nvp, ne, nf, vcoff(ke), vcoff(kf), w(jovvq), w(jgpp), w(jvpidx), g1, gp)
      call N(zc_vvvv_dress)(wb, wb, nq, ne * nf, w(jvpa + vpoff(q2)), w(jvpb + vpoff(q2)), tm, nv, g1, no, nv, &
                            t1, nv, g1, no, nv, tm, tm, 0_8)
      if (npo > 0) call N(zc_gemm)(1_4, 0_4, nq, ne * nf, npo, one, tau(vpoff(q2), 0), nvp, gp, ne * nf, one, &
                                   wb, ne * nf)
      call N(zc_gemm)(0_4, 0_4, nq * ne, nd, nf, one, wb, nf, t2(vcoff(kd), vcoff(kf), l, x), nv, zero, m3(n), nd)
      n = n + nq * ne * nd
    end do
    call N(zc_xv_sector)(int(nv, 4), w(jqkv), 0_4, tri(0, vtoff(t)), nt, int(ne, 4), 0_4, &
                         int(npg, 4), off, ndd, w(jqposv), w(jqposq), m3, xv(xoff))
    xoff = xoff + ne * nt
  end do
end subroutine

! g1[m][y][(e, f)] = <my||ef>, gp[P][(e, f)] = <mn||ef> for e and f of two ranges
subroutine N(xz4g12)(no, nv, npo, nvp, ne, nf, e0, f0, gq, gpp, vpidx, g1, gp)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp, ne, nf, e0, f0
  ZT, intent(in) :: gq(0:nvp-1, 0:no*nv-1)                     ! [m][y][Q]
  ZT, intent(in) :: gpp(0:nvp-1, 0:npo-1)                      ! [P][Q]
  integer(8), intent(in) :: vpidx(0:nv-1, 0:nv-1)              ! [p][q]
  ZT, intent(out) :: g1(0:nf-1, 0:ne-1, 0:no*nv-1), gp(0:nf-1, 0:ne-1, 0:npo-1)
  integer(8) :: r, e, f, ge, gf
  !$omp parallel do private(r, e, f, ge, gf)
  do r = 0, no * nv - 1
    do e = 0, ne - 1
      ge = e0 + e
      do f = 0, nf - 1
        gf = f0 + f
        if (ge < gf) then
          g1(f, e, r) = gq(vpidx(gf, ge), r)
        else if (ge > gf) then
          g1(f, e, r) = -gq(vpidx(ge, gf), r)
        else
          g1(f, e, r) = 0
        end if
      end do
    end do
  end do
  !$omp parallel do private(r, e, f, ge, gf)
  do r = 0, npo - 1
    do e = 0, ne - 1
      ge = e0 + e
      do f = 0, nf - 1
        gf = f0 + f
        if (ge < gf) then
          gp(f, e, r) = gpp(vpidx(gf, ge), r)
        else if (ge > gf) then
          gp(f, e, r) = -gpp(vpidx(ge, gf), r)
        else
          gp(f, e, r) = 0
        end if
      end do
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! One occupied quadruple i<j<k<l without labels, the right-hand side:
!   z[A] += wt * <Q| [V~, T3] + 1/2 [[V~, T2], T2] |0>
! - the five products of N(xzquad) with the dressed blocks q1, q2 (N(xz4q12)),
! Xo and Xv of the dressed integrals; T3 at kt3.
subroutine N(xz4fwd)(w, i, j, k, l, na4, otri, osgn, t2p, t2px, xo, q1, q2, kt3, wt, z, w22, w13, &
                     la, ra, lb, rb, lc, rc, xv, slot, tq)
  use xzdata
  use xzkern
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: i, j, k, l, na4, kt3
  integer(8), intent(in) :: otri(0:npo-1, 0:no-1)
  integer(1), intent(in) :: osgn(0:npo-1, 0:no-1)
  ZT, intent(in) :: t2p(0:nvp-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: t2px(0:nv*nv-1, 0:npo-1)
  ZT, intent(in) :: xo(0:nvp-1, 0:no-1, 0:nto-1)
  ZT, intent(in) :: q1(0:nvp-1, 0:nv-1, 0:no-1)                ! [x][e][Q]
  ZT, intent(in) :: q2(0:nv-1, 0:no-1, 0:no-1, 0:no-1)         ! [i][j][m][a]
  real(8), intent(in) :: wt
  ZT :: z(*), w22(*), w13(*)
  ZT :: la(0:nvp-1, 0:nv-1, 0:3), ra(0:nvp-1, 0:nv-1, 0:3)
  ZT :: lb(0:nvp-1, 0:no-1, 0:3), rb(0:nvp-1, 0:no-1, 0:3)
  ZT :: lc(0:nv-1, 0:no-1, 0:5), rc(0:nvt-1, 0:no-1, 0:5)
  ZT, intent(in) :: xv(0:nv*nvt-1, 0:*)
  integer(8), intent(in) :: slot(0:*)
  real(8), intent(inout) :: tq(6)
  integer(8) :: q(0:3), s, x, jt, m, c0, nc, p2, n
  integer(8) :: rows(0:6*no-1), xfirst(0:5), pfirst(0:5)
  integer(1) :: sgn(0:6*no-1)
  real(8) :: sg, s13(0:3), s22(0:5), t0
  integer(8), external :: xztrank, xzpidx
  integer(8), parameter :: f1(0:5) = [0, 0, 0, 1, 1, 2], f2(0:5) = [1, 2, 3, 2, 3, 3]
  integer(8), parameter :: r1(0:5) = [2, 1, 1, 0, 0, 0], r2(0:5) = [3, 3, 2, 3, 2, 1]
  integer(8), parameter :: last(3) = [0, 1, 3]
  ZT :: one, zero, alpha
  one = 1
  zero = 0
  q = [i, j, k, l]
  s13 = [1d0, -1d0, 1d0, -1d0]
  s22 = [1d0, -1d0, 1d0, 1d0, -1d0, 1d0]
  t0 = omp_get_wtime()
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
    sg = wt * s13(s)
    ! - ~<ab||ex> t(jkl; e, cd)
    call N(xzqcopy)(nv * nvp, sg, .false., q1(0, 0, x), la(0, 0, s))
    call zc_tn_zero(ra(0, 0, s), 8 * NW * nv * nvp)
    call N(zc_tn_expand)(w(kt3 + NW * nvt * jt), 1_8, nvt, int(nv, 4), 1_4, 2_4, 0_8, nv, 0_8, nvp, &
                         ra(0, 0, s), nvp, 1_8, 0_8, 1d0)
    lb(:, :, s) = sg * t2p(:, :, x)
    rb(:, :, s) = xo(:, :, jt)
  end do
  ! ---- the splits (ij|kl): ~<ma||ij> and t(m kl; bcd)
  do s = 0, 5
    pfirst(s) = xzpidx(q(f1(s)), q(f2(s)), no)
    xfirst(s) = q(r1(s))
    p2 = xzpidx(q(r1(s)), q(r2(s)), no)
    do m = 0, no - 1
      lc(:, m, s) = (wt * s22(s)) * q2(:, m, q(f2(s)), q(f1(s)))
      rows(s * no + m) = otri(p2, m)
      sgn(s * no + m) = osgn(p2, m)
    end do
  end do
  call N(zc_tn_rows)(w(kt3), nvt, rows, sgn, 6 * no, 0_8, nvt, rc, nvt)
  tq(3) = tq(3) + omp_get_wtime() - t0

  do c0 = 0, nvp - 1, qc22
    nc = min(qc22, nvp - c0)
    t0 = omp_get_wtime()
    call N(zc_gemm)(1_4, 0_4, nvp, nc, 4 * nv, one, la, nvp, ra(c0, 0, 0), nvp, zero, w22, nc)
    call N(zc_gemm)(1_4, 0_4, nvp, nc, 4 * no, one, lb, nvp, rb(c0, 0, 0), nvp, one, w22, nc)
    tq(4) = tq(4) + omp_get_wtime() - t0
    t0 = omp_get_wtime()
    call N(zc_tn_reduce)(z, 1_8, na4, int(nv, 4), 2_4, 2_4, 0_8, nvp, c0, c0 + nc, w22, nc, 1_8, 0_8, 1d0)
    tq(5) = tq(5) + omp_get_wtime() - t0
  end do
  do c0 = 0, nvt - 1, qc13
    nc = min(qc13, nvt - c0)
    t0 = omp_get_wtime()
    call N(zc_gemm)(1_4, 0_4, nv, nc, 6 * no, one, lc, nv, rc(c0, 0, 0), nvt, zero, w13, nc)
    ! the splits whose second pair has l: (ij|kl), (ik|jl), (jk|il)
    do n = 1, 3
      s = last(n)
      alpha = 2d0 * s22(s) * wt
      call N(zc_gemm)(1_4, 0_4, nv, nc, nv, alpha, t2px(0, pfirst(s)), nv, &
                      xv(c0, slot(xfirst(s))), nvt, one, w13, nc)
    end do
    tq(4) = tq(4) + omp_get_wtime() - t0
    t0 = omp_get_wtime()
    call N(zc_tn_reduce)(z, 1_8, na4, int(nv, 4), 1_4, 3_4, 0_8, nv, c0, c0 + nc, w13, nc, 1_8, 0_8, 1d0)
    tq(5) = tq(5) + omp_get_wtime() - t0
  end do
end subroutine

! ---------------------------------------------------------------------------
! One occupied quadruple without labels, the contractions of its row z of T4
! into the four sums (r2[P][Q]; R3 at kr3), in the batches of columns of the
! products:
!   the row on two pairs, X[(ef)][(ab)], columns (ab):
!     r2[P][(ab)]  += sgn sum_(ef) <P'||ef> X                    six splits (P | P')
!     R3[J]        += reduction (c | ab) of sgn sum_(ef) ~<mc||ef> X    four splits (m | J)
!   the row on an index and a triple, X[e][(abc)], columns (abc):
!     R3[J]        += sgn sum_e f[m][e] X                        four splits (m | J)
!     R3[(k, P)]   += sgn sgn' sum_e ~<P'||ke> X                 six splits, k outside P
! wh: [2][6][nvp]; y: [nv][qc22]; lk: [4 + 6 no][nv]; bo: [4 + 6 no][qc13]; tg: 4 + 6 no rows.
subroutine N(xz4back)(w, i, j, k, l, na4, otri, osgn, z, w22, w13, wh, y, lk, bo, tg, fov, gpp, kd, ooovd, &
                      r2, kr3, tq)
  use xzdata
  use xzkern
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: i, j, k, l, na4, kd, kr3
  integer(8), intent(in) :: otri(0:npo-1, 0:no-1)
  integer(1), intent(in) :: osgn(0:npo-1, 0:no-1)
  ZT, intent(in) :: z(*)
  ZT :: w22(*), w13(*), y(*), bo(*)
  ZT :: wh(0:nvp-1, 0:5, 0:1), lk(0:nv-1, 0:*)
  integer(8) :: tg(0:*)
  ZT, intent(in) :: fov(0:nv-1, 0:no-1)                        ! [m][e]
  ZT, intent(in) :: gpp(0:nvp-1, 0:npo-1)                      ! [P][Q]
  ZT, intent(in) :: ooovd(0:nv-1, 0:no-1, 0:no-1, 0:no-1)      ! [m][n][i][e]
  ZT, intent(inout) :: r2(0:nvp-1, 0:npo-1)
  real(8), intent(inout) :: tq(6)
  integer(8) :: q(0:3), s, c0, nc, p1(0:5), jts(0:3), nrow, kk, mm, nn
  real(8) :: s13(0:3), s22(0:5), t0
  integer(8), external :: xztrank, xzpidx
  integer(8), parameter :: f1(0:5) = [0, 0, 0, 1, 1, 2], f2(0:5) = [1, 2, 3, 2, 3, 3]
  integer(8), parameter :: r1(0:5) = [2, 1, 1, 0, 0, 0], r2i(0:5) = [3, 3, 2, 3, 2, 1]
  ZT :: one, zero, alpha
  one = 1
  zero = 0
  t0 = omp_get_wtime()
  q = [i, j, k, l]
  s13 = [1d0, -1d0, 1d0, -1d0]
  s22 = [1d0, -1d0, 1d0, 1d0, -1d0, 1d0]
  jts(0) = xztrank(q(1), q(2), q(3), no)
  jts(1) = xztrank(q(0), q(2), q(3), no)
  jts(2) = xztrank(q(0), q(1), q(3), no)
  jts(3) = xztrank(q(0), q(1), q(2), no)
  do s = 0, 5
    p1(s) = xzpidx(q(f1(s)), q(f2(s)), no)
    wh(:, s, 1) = s22(s) * gpp(:, xzpidx(q(r1(s)), q(r2i(s)), no))
  end do
  do c0 = 0, nvp - 1, qc22
    nc = min(qc22, nvp - c0)
    call zc_tn_zero(w22, 8 * NW * nvp * nc)
    call N(zc_tn_expand)(z, 1_8, na4, int(nv, 4), 2_4, 2_4, 0_8, nvp, c0, c0 + nc, w22, nc, 1_8, 0_8, 1d0)
    call N(zc_gemm)(0_4, 0_4, 6_8, nc, nvp, one, wh(0, 0, 1), nvp, w22, nc, zero, wh(c0, 0, 0), nvp)
    do s = 0, 3
      alpha = s13(s)
      call N(zc_gemm)(0_4, 0_4, nv, nc, nvp, alpha, w(kd + NW * nvp * nv * q(s)), nvp, w22, nc, zero, y, nc)
      call N(zc_tn_reduce)(w(kr3 + NW * nvt * jts(s)), 1_8, nvt, int(nv, 4), 1_4, 2_4, 0_8, nv, c0, c0 + nc, &
                           y, nc, 1_8, 0_8, 1d0)
    end do
  end do
  do s = 0, 5
    r2(:, p1(s)) = r2(:, p1(s)) + wh(:, s, 0)
  end do
  nrow = 0
  do s = 0, 3
    lk(:, nrow) = s13(s) * fov(:, q(s))
    tg(nrow) = jts(s)
    nrow = nrow + 1
  end do
  do s = 0, 5
    mm = q(r1(s))
    nn = q(r2i(s))
    do kk = 0, no - 1
      if (osgn(p1(s), kk) == 0) cycle
      lk(:, nrow) = (s22(s) * osgn(p1(s), kk)) * ooovd(:, kk, nn, mm)
      tg(nrow) = otri(p1(s), kk)
      nrow = nrow + 1
    end do
  end do
  do c0 = 0, nvt - 1, qc13
    nc = min(qc13, nvt - c0)
    call zc_tn_zero(w13, 8 * NW * nv * nc)
    call N(zc_tn_expand)(z, 1_8, na4, int(nv, 4), 1_4, 3_4, 0_8, nv, c0, c0 + nc, w13, nc, 1_8, 0_8, 1d0)
    call N(zc_gemm)(0_4, 0_4, nrow, nc, nv, one, lk, nv, w13, nc, zero, bo, nc)
    call N(zc_cc4_addrows)(w(kr3 + NW * c0), nvt, tg, nrow, nc, bo)
  end do
  tq(6) = tq(6) + omp_get_wtime() - t0
end subroutine

! ---------------------------------------------------------------------------
! One occupied quadruple i<j<k<l in the sectors of the labels: the blocks of
! the right-hand side (the products of N(xzqsquad) with the dressed blocks),
! the amplitudes written to the places of the blocks (zc_cc4_sector_t4), and
! their contractions into the four sums, block by block.
!   b, e22     the blocks [(ab)][(cd)] of the right-hand side and of T4
!   c, e13     the blocks [a][(bcd)]
!   la .. lc, rows, sgs   the operands of the products (N(xzqsquad))
!   y          the blocks [c][(ab)] of the term with ~<mc||ef>
!   lk, bx, tg operand, product and rows of the terms with f and ~<mn||ke>
subroutine N(xz4squad)(w, i, j, k, l, otri, osgn, opidx, opgrp, t2p, t2px, xo, q1, q2, kt3, b, c, e22, e13, &
                       la, ra, lb, rb, lc, rows, sgs, xv, slot, omega, tri, y, lk, bx, tg, fov, kd, ooovd, &
                       kr2, kr3, tq)
  use xzdata
  use xzkern
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: i, j, k, l, kt3, kd, kr2, kr3
  integer(8), intent(in) :: otri(0:npo-1, 0:no-1), opidx(0:no-1, 0:no-1), opgrp(0:npo-1)
  integer(1), intent(in) :: osgn(0:npo-1, 0:no-1)
  ZT, intent(in) :: t2p(0:nvp-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: t2px(0:nv-1, 0:nv-1, 0:npo-1)              ! [P][e][a]
  ZT, intent(in) :: xo(0:nvp-1, 0:no-1, 0:nto-1)
  ZT, intent(in) :: q1(0:nvp-1, 0:nv-1, 0:no-1)                ! [x][e][Q]
  ZT, intent(in) :: q2(0:nv-1, 0:no-1, 0:no-1, 0:no-1)         ! [i][j][m][a]
  ZT :: b(0:*), c(0:*), e22(0:*), e13(0:*)
  ZT :: la(0:*), ra(0:*), lb(0:*), rb(0:*), lc(0:*), y(0:*), lk(0:*), bx(0:*)
  integer(8) :: rows(0:*), tg(0:*)
  real(8) :: sgs(0:*)
  ZT, intent(in) :: xv(0:qxvmax-1, 0:*)
  integer(8), intent(in) :: slot(0:*)
  real(8), intent(in) :: omega
  integer(4), intent(in) :: tri(0:2, 0:nvt-1)
  ZT, intent(in) :: fov(0:nv-1, 0:no-1)                        ! [m][e]
  ZT, intent(in) :: ooovd(0:nv-1, 0:no-1, 0:no-1, 0:no-1)      ! [m][n][i][e]
  real(8), intent(inout) :: tq(6)
  integer(8) :: q(0:3), s, x, m, mt, qq, r, nr, nc, na, nb, ke, ne, nm, n22, n13, p, km, kxl, n
  integer(8) :: jt(0:3), p22(0:5), q22(0:5), xfirst(0:5), gj, nt, lcc, nl, nyb, kk, mm, nn
  integer(8) :: off22(0:npg-1), nc22(0:npg-1), off13(0:ncls-1), nc13(0:ncls-1), yoff(0:npg-1), ynq(0:npg-1)
  real(8) :: s13(0:3), s22(0:5), t0, eo
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
  eo = w(jeps + i) + w(jeps + j) + w(jeps + k) + w(jeps + l) + omega
  mt = ocls(i) + ocls(j) + ocls(k) + ocls(l)
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
    r = mt - qq
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
      ke = qq - ocls(x)
      if (ke < 0 .or. ke >= ncls) cycle
      ne = vcoff(ke + 1) - vcoff(ke)
      nm = ocoff(ke + 1) - ocoff(ke)
      ! - ~<ab||ex> t(J; e, cd)
      if (ne > 0) call N(xzqsab)(nv, nvp, nr, nc, ne, s13(s), .false., vcoff(ke), vpoff(qq), vpoff(r), &
        q1(0, 0, x), w(kt3), w(jobase + jt(s)), w(jvci), w(jvsgn), la(na * nr), ra(na * nc))
      na = na + ne
      ! t(x m, ab) Xo(m; J; cd)
      do m = 0, nm - 1
        lb((nb + m) * nr:(nb + m) * nr + nr - 1) = s13(s) * t2p(vpoff(qq):vpoff(qq) + nr - 1, ocoff(ke) + m, x)
        rb((nb + m) * nc:(nb + m) * nc + nc - 1) = xo(vpoff(r):vpoff(r) + nc - 1, ocoff(ke) + m, jt(s))
      end do
      nb = nb + nm
    end do
    beta = zero
    if (na > 0) then
      call N(zc_gemm)(1_4, 0_4, nr, nc, na, one, la, nr, ra, nc, zero, b(n22), nc)
      beta = one
    end if
    if (nb > 0) then
      call N(zc_gemm)(1_4, 0_4, nr, nc, nb, one, lb, nr, rb, nc, beta, b(n22), nc)
      beta = one
    end if
    if (beta == zero) call zc_tn_zero(b(n22), 8 * NW * nr * nc)
    n22 = n22 + nr * nc
  end do
  tq(3) = tq(3) + omp_get_wtime() - t0

  ! ---- the blocks on one index and a triple
  t0 = omp_get_wtime()
  off13 = -1
  nc13 = 0
  n13 = 0
  do p = 0, ncls - 1
    r = mt - p
    if (r < 0 .or. r >= ntg) cycle
    nr = vcoff(p + 1) - vcoff(p)
    nc = vtoff(r + 1) - vtoff(r)
    if (nr == 0 .or. nc == 0) cycle
    off13(p) = n13
    nc13(p) = nc
    ! ~<ma||ij> t(m kl; bcd): rows of T3 where they are
    n = 0
    do s = 0, 5
      km = opgrp(p22(s)) - p
      if (km < 0 .or. km >= ncls) cycle
      do m = ocoff(km), ocoff(km + 1) - 1
        if (osgn(q22(s), m) == 0) cycle
        lc(n * nr:n * nr + nr - 1) = s22(s) * q2(vcoff(p):vcoff(p) + nr - 1, m, q(f2(s)), q(f1(s)))
        rows(n) = otri(q22(s), m) - otoff(r)
        sgs(n) = dble(osgn(q22(s), m))
        n = n + 1
      end do
    end do
    first = .true.
    if (n > 0) then
      call N(zc_rows_gemm)(w(kt3 + NW * boff(r)), nc, rows, sgs, n, w(jrows), nc, lc, nr, c(n13))
      first = .false.
    end if
    ! 2 t(ij, ae) Xv(e; kl; bcd), l in the second pair
    do m = 1, 3
      s = last(m)
      ke = opgrp(p22(s)) - p
      if (ke < 0 .or. ke >= ncls) cycle
      ne = vcoff(ke + 1) - vcoff(ke)
      if (ne == 0) cycle
      kxl = ocls(xfirst(s)) + ocls(l)
      alpha = 2d0 * s22(s)
      beta = one
      if (first) beta = zero
      call N(zc_gemm)(1_4, 0_4, nr, nc, ne, alpha, t2px(vcoff(p), vcoff(ke), p22(s)), nv, &
                      xv(N(xzqsxvoff)(kxl, ke), slot(xfirst(s))), nc, beta, c(n13), nc)
      first = .false.
    end do
    if (first) call zc_tn_zero(c(n13), 8 * NW * nr * nc)
    n13 = n13 + nr * nc
  end do
  tq(4) = tq(4) + omp_get_wtime() - t0

  ! ---- the amplitudes of the sector, at the places of the blocks
  t0 = omp_get_wtime()
  call zc_tn_zero(e22, 8 * NW * n22)
  call zc_tn_zero(e13, 8 * NW * n13)
  call N(zc_cc4_sector_t4)(int(nv, 4), w(jqkv), 0_4, int(mt, 4), w(jeps + no), eo, 0_4, int(ncls, 4), &
                           w(jqlid), w(jqlst), w(jqlmem), 0_4, int(npg, 4), off22, nc22, off13, nc13, &
                           w(jqposv), w(jqposq), w(jqpos3), b, c, e22, e13)
  tq(5) = tq(5) + omp_get_wtime() - t0

  ! ---- the four sums
  t0 = omp_get_wtime()
  ! r2[P][(ab)] += sgn sum_(cd) X[(ab)][(cd)] <P'||cd>
  do s = 0, 5
    qq = opgrp(p22(s))
    if (off22(qq) < 0) cycle
    r = mt - qq
    nr = vpoff(qq + 1) - vpoff(qq)
    nc = nc22(qq)
    alpha = s22(s)
    call N(zc_gemm)(0_4, 0_4, nr, 1_8, nc, alpha, e22(off22(qq)), nc, w(jgpp + NW * (nvp * q22(s) + vpoff(r))), &
                    1_8, one, w(kr2 + NW * (nvp * p22(s) + vpoff(qq))), 1_8)
  end do
  ! R3[J] += the reduction (c | ab) of sgn sum_(ef) ~<mc||ef> X[(ab)][(ef)]
  do s = 0, 3
    m = q(s)
    gj = mt - ocls(m)
    if (gj < 0 .or. gj >= ntg) cycle
    nt = vtoff(gj + 1) - vtoff(gj)
    if (nt == 0) cycle
    yoff = -1
    ynq = 0
    nyb = 0
    do qq = 0, npg - 1
      if (off22(qq) < 0) cycle
      lcc = gj - qq
      if (lcc < 0 .or. lcc >= ncls) cycle
      nl = vcoff(lcc + 1) - vcoff(lcc)
      if (nl == 0) cycle
      r = mt - qq
      nr = vpoff(qq + 1) - vpoff(qq)
      nc = nc22(qq)
      yoff(qq) = nyb
      ynq(qq) = nr
      alpha = s13(s)
      call N(zc_gemm)(0_4, 1_4, nl, nr, nc, alpha, w(kd + NW * (nvp * (vcoff(lcc) + nv * m) + vpoff(r))), nvp, &
                      e22(off22(qq)), nc, zero, y(nyb), nr)
      nyb = nyb + nl * nr
    end do
    if (nyb > 0) call N(zc_cc4_red12)(w(kr3 + NW * (boff(gj) + (jt(s) - otoff(gj)) * nt)), tri(0, vtoff(gj)), nt, &
                                     int(nv, 4), w(jqkv), 0_4, 0_4, int(npg, 4), yoff, ynq, w(jqposv), &
                                     w(jqposq), y)
  end do
  ! R3[J] += sgn sum_e f[m][e] X[e][(abc)],  R3[(k, P)] += sgn sgn' sum_e ~<P'||ke> X[e][(abc)]
  do p = 0, ncls - 1
    if (off13(p) < 0) cycle
    r = mt - p
    nr = vcoff(p + 1) - vcoff(p)
    nc = nc13(p)
    n = 0
    do s = 0, 3
      if (ocls(q(s)) /= p) cycle
      lk(n * nr:n * nr + nr - 1) = s13(s) * fov(vcoff(p):vcoff(p) + nr - 1, q(s))
      tg(n) = jt(s) - otoff(r)
      n = n + 1
    end do
    do s = 0, 5
      km = r - opgrp(p22(s))
      if (km < 0 .or. km >= ncls) cycle
      mm = q(r1(s))
      nn = q(r2(s))
      do kk = ocoff(km), ocoff(km + 1) - 1
        if (osgn(p22(s), kk) == 0) cycle
        lk(n * nr:n * nr + nr - 1) = (s22(s) * osgn(p22(s), kk)) * ooovd(vcoff(p):vcoff(p) + nr - 1, kk, nn, mm)
        tg(n) = otri(p22(s), kk) - otoff(r)
        n = n + 1
      end do
    end do
    if (n == 0) cycle
    call N(zc_gemm)(0_4, 0_4, n, nc, nr, one, lk, nr, e13(off13(p)), nc, zero, bx, nc)
    call N(zc_cc4_addrows)(w(kr3 + NW * boff(r)), nc, tg, n, nc, bx)
  end do
  tq(6) = tq(6) + omp_get_wtime() - t0
end subroutine
