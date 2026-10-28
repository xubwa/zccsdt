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
! (Q) (xzpar.f90): every rank its tasks and its own sums, which are added.
! The iterations (xzcc.inc) take the two sums as fixed parts of the
! residuals of the CCSDT step (xzstep.inc: cc4 of xzdata).
!
! Excited states (N(xze4), N(xzeom4)): the folded eigenproblem in the space
! of singles, doubles and triples.  A sigma vector at the frequency omega is
! that of EOM-CCSDT at the CC4 amplitudes (xzeom.inc) plus the T4 terms
!   r4 = ([V', T3] + [V~, R3] + [[V', T2], T2] / 2 + [[V~, R2], T2]) / (omega + e_ijkl - e_abcd)
!        contracted with the integrals of H~, and
!   t4 of the ground state contracted with the integrals of H' = [H~, R1],
! both made in one pass over the occupied quadruples (N(xz4run) with eom).
!
! Label shifts.  Every operand of a product has a shift in classes: (the
! classes of its creators) - (those of its annihilators), 0 for the
! amplitudes and the operators of the ground state, the shift s of the
! vector for r1, r2, R3 and for the integrals of H'.  An occupied quadruple
! of total class M has its row on the virtual quadruples of total class
! M + (the shifts of the two operands of its products).
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
  use xzdist
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kt1, kt2, kt3, kr2, kr3
  real(8), intent(in) :: omega
  integer(8) :: m0, m1, ov, kt2pp
  integer(8) :: kfoo, kfvo, kfvv, kpvv, kooood, kovvod, kovood, kk, ktt, kwv, kg2, kyb, kgq, kgh, kpib, ktau
  integer(8) :: kg, kkm, kb, koo, h, lo, nqh, q0, nq, ndone, nmine
  logical :: dense
  real(8) :: t0, tq(6), tstart

  if (no < 4 .or. nv < 4) then
    if (.not. xzdry) then
      call zc_vzero_d(NW * npo * nvp, w(kr2))
      call zc_vzero_d(NW * n3, w(kr3))
    end if
    return
  end if
  ov = no * nv
  dense = ncls == 1
  tq = 0d0
  tstart = omp_get_wtime()
  m0 = xzmark()

  ! ---- what the quadruples read: the dressed blocks, T2 in three layouts, Xo
  g4fov = xzget(NW * ov, 'CC4: dressed f[m][e]')
  g4ooov = xzget(NW * no * no * ov, 'CC4: dressed <mn||ie>')
  g4d = xzget(NW * ov * nvp, 'CC4: dressed <mb||ef>')
  g4q1 = xzget(NW * ov * nvp, 'CC4: dressed <ab||ex> [x][e][Q]')
  g4q2 = xzget(NW * no * no * ov, 'CC4: dressed <ma||ij> [i][j][m][a]')
  kt2pp = xzget(NW * npo * nvp, 'CC4: T2 [P][Q]')
  g4t2p = xzget(NW * no * no * nvp, 'CC4: T2 [i][m][Q]')
  g4t2px = xzget(NW * npo * nv * nv, 'CC4: T2 [P][e][a]')
  g4xo = xzget(NW * nto * no * nvp, 'CC4: Xo')
  jv4 = xzget(NW * nvv, 'CC4: dressed <ab||cd>')
  m1 = xzmark()
  ktau = xzget(NW * npo * nvp, 'CC4: tau1')
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
                   w(jvpa), w(jvpb), w(kfoo), w(g4fov), w(kfvo), w(kfvv), w(kpvv))
    call N(xzdroo)(no, nv, w(kt1), w(jooov), w(joovv), w(joooo), w(g4ooov), w(kooood))
    call N(xzdrovvq)(no, nv, nvp, w(kt1), w(jovvq), w(joovvp), w(g4d))
    call N(xzdrov)(no, nv, nvp, w(kt1), w(jooov), w(jovvo), w(jovvq), w(g4d), w(kooood), &
                   w(jvpidx), w(kovvod), w(kovood), w(kk))
    call N(xzt2forms)(no, nv, npo, nvp, w(kt1), w(kt2), w(jopa), w(jopb), w(jvpa), w(jvpb), &
                      w(g4t2p), w(kt2pp), w(ktau), w(ktt))
    call N(xzqt2)(no, nv, npo, nvp, w(kt2), w(jopa), w(jopb), w(jvpa), w(jvpb), w(kt2pp), w(g4t2p), w(g4t2px))
    ! the dressed <ab||ei>: its bra, then + sum_f ~<ab||ef> t1[i][f] from the rows of the dressed
    ! <ab||cd>, which are kept for Xv
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
                         w(jvpb + lo + q0), w(jv4 + NW * (vvoff(h) + q0 * nqh)), w(kk))
        call N(zc_pair_t1_contract)(w(kwv + NW * (lo + q0) * ov), w(jv4 + NW * (vvoff(h) + q0 * nqh)), w(kt1), nq, nv, &
                                    nqh, w(kpib), no)
      end do
    end do
    call N(xz4q12)(no, nv, nvp, w(kwv), w(kovood), w(g4q1), w(g4q2))
    call N(xz4xo)(no, nv, npo, nvp, nto, w(kooood), w(kovvod), w(g4t2p), w(g4t2px), w(jopa), w(jopb), &
                  w(jvpa), w(jvpb), w(jofree), w(jopair), w(g4xo), w(kg), w(kkm), w(kb), w(koo), .false.)
    tq(1) = omp_get_wtime() - t0
  end if
  call xzrel(m1)
  if (.not. xzdry) then
    call zc_vzero_d(NW * npo * nvp, w(kr2))
    call zc_vzero_d(NW * n3, w(kr3))
  end if
  call N(xz4run)(w, kt1, kt2, kt3, kr2, kr3, omega, .false., tq, ndone, nmine)
  call xzrel(m0)
  if (xzdry) return
  ! ---- the sums of the ranks: the doubles term on all of them, the triples term on rank 0
  if (nrank > 1) then
    t0 = omp_get_wtime()
    call xzmpi_allsum_w(w(kr2), NW * npo * nvp)
    call xzmpi_sum0_w(w(kr3), NW * n3)
    tq(1) = tq(1) + omp_get_wtime() - t0
  end if
  if (dense) then
    write(6, '(a,i10,a,i5,a,f10.2,a,f9.2,a,f9.2,a,3f10.2,a,f10.2,a)') '     T4 terms: ', ndone, ' occupied quadruples, ', &
      nmine, ' tasks, ', omp_get_wtime() - tstart, ' s  (operands ', tq(1), ', Xv ', tq(2), &
      ', right-hand sides: operands, products, reductions ', tq(3), tq(4), tq(5), ', contractions ', tq(6), ')'
  else
    write(6, '(a,i10,a,i5,a,f10.2,a,f9.2,a,f9.2,a,2f10.2,a,f10.2,a,f10.2,a)') '     T4 terms: ', ndone, ' occupied quadruples, ', &
      nmine, ' tasks, ', omp_get_wtime() - tstart, ' s  (operands ', tq(1), ', Xv ', tq(2), &
      ', right-hand sides: blocks on two pairs, on an index and a triple ', tq(3), tq(4), ', the row of T4 ', tq(5), &
      ', contractions ', tq(6), ')'
  end if
  flush(6)
end subroutine

! ---------------------------------------------------------------------------
! The pass over the occupied quadruples of this rank (the tasks of (Q),
! xzpar.f90), with the operands of the ground state at g4... of xzdata:
!   eom = .false.   t4 of (kt2, kt3) over e_ijkl - e_abcd + omega, contracted
!                   with the integrals of H~: kr2 [P][Q], kr3 (the layout of T3)
!   eom = .true.    the T4 terms of a sigma vector (the operands at x4...,
!                   the shift x4s): r4 at the frequency omega with H~, and t4
!                   of the ground state with H'; kr2, kr3 (the layout of the
!                   shift) are added to.
subroutine N(xz4run)(w, kt1, kt2, kt3, kr2, kr3, omega, eom, tq, ndone, nmine)
  use xzdata
  use xzmem
  use xzkern
  use xzpar
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kt1, kt2, kt3, kr2, kr3
  real(8), intent(in) :: omega
  logical, intent(in) :: eom
  real(8), intent(inout) :: tq(6)
  integer(8), intent(out) :: ndone, nmine
  integer(8) :: m0, na4, nslot, nxv, nrb, nt, s, mt
  integer(8) :: kev4, kz, kw22, kw13, kla, kra, klb, krb, klc, krc, kwh, kxv, kxvs, kwx, kmx, ky, klk, kbo, ktg
  integer(8) :: kl2, qy12
  integer(8) :: i, j, k, l, x, n, size, ng, g1, g2, g3, qq, g, kx
  integer(8) :: lo1, hi1, lo2, hi2, lo3, hi3, it, lastl, nx, nnew
  integer(8) :: slot(0:no-1), xnew(0:no-1), kind(2), ka(2), sw(2)
  integer(8) :: off22(0:npg-1), nc22(0:npg-1), off13(0:ncls-1), nc13(0:ncls-1), n22, n13
  logical :: need(0:no-1), dense
  real(8) :: t0, qcm(0:4 * maxcls), xcm(0:2 * maxcls), eo
  integer(8), external :: N(xzqslot)

  ndone = 0
  nmine = 0
  dense = ncls == 1
  s = 0
  if (eom .and. .not. dense) s = x4s
  na4 = nv * (nv - 1) * (nv - 2) * (nv - 3) / 24
  nslot = max(3_8, min(qxv, no - 1))
  nt = merge(2_8, 1_8, eom)
  m0 = xzmark()
  kxvs = 0
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
    if (eom) kxvs = xzget(NW * nslot * nxv, 'CC4: Xv of the vector')
    kwx = xzget(NW * nt * qrows * nv * nv, 'CC4: rows of <vv||vv>')
    kmx = xzget(NW * qrows * nv * nv, 'CC4: product of Xv')
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
    ! (the right-hand side and then, in its place, the row of T4)
    kz = xzget(NW * qn22max, 'CC4: blocks of the row on two pairs')
    kw13 = xzget(NW * qn13max, 'CC4: blocks of the row on an index and a triple')
    ! (the stacked operands of all blocks on two pairs of a row: four splits, the summed index of every class once)
    kla = xzget(NW * 4 * (nv + no) * nqmax, 'CC4: stacked operands')
    kra = xzget(NW * 4 * (nv + no) * nqmax, 'CC4: stacked operands')
    klc = xzget(NW * (6 * nocmax + 3 * nvcmax) * nvcmax, 'CC4: stacked operand')
    ! the blocks [c][(ab)] of the term with <cm||ef>: c of class g - q for the pairs of group q
    qy12 = 0
    do g = 0, ntg - 1
      n = 0
      do qq = 0, npg - 1
        n = n + xznvc(g - qq) * (vpoff(qq + 1) - vpoff(qq))
      end do
      qy12 = max(qy12, n)
    end do
    ky = xzget(NW * (4 * qy12 + 6 * nqmax), 'CC4: blocks [c][(ab)] of four splits, rows of the doubles term')
    kl2 = xzget(NW * (4 * nv + 6) * nqmax, 'CC4: stacked operands of the contractions')
    klk = xzget(NW * nrb * nvcmax, 'CC4: operand of the triples terms')
    ! Xv of one pair of occupied indices: the blocks of every class of e; with
    ! a shift the triples of other groups (the largest set of any pair class)
    nxv = qxvmax
    if (s /= 0) then
      do kx = 0, 2 * (ncls - 1)
        n = 0
        do g = 0, ncls - 1
          if (g + kx + s < 0 .or. g + kx + s >= ntg) cycle
          n = n + (vcoff(g + 1) - vcoff(g)) * (vtoff(g + kx + s + 1) - vtoff(g + kx + s))
        end do
        nxv = max(nxv, n)
      end do
    end if
    kxv = xzget(NW * nslot * nxv, 'CC4: Xv')
    if (eom) kxvs = xzget(NW * nslot * nxv, 'CC4: Xv of the vector')
    kwx = xzget(NW * nqmax * nvcmax * nvcmax, 'CC4: block of <vv||vv>')
    kmx = xzget(NW * nqmax * nv * nvcmax, 'CC4: products of Xv')
    kev4 = kz
    kw22 = kz
    kwh = kz
  end if
  if (xzdry) then
    call xzrel(m0)
    return
  end if
  if (dense) then
    call zc_tn_tuple_sums(w(jeps + no), int(nv, 4), 4_4, w(kev4))
  else
    call xzqstab(nv, nvp, nvt, ncls, vcls, vcoff, vpoff, vtoff, w(jvpidx), w(jvpgrp), w(jvtri), w(jvfree), &
                 w(jqkv), w(jqlid), w(jqlst), w(jqlmem), w(jqposv), w(jqposq), w(jqpos3), w(jqtri))
  end if

  ! ---- the tasks of (Q) (xzpar.f90): groups of one largest index l, dealt to the ranks
  call xzqmodel(qcm, xcm)
  call xzq_costs(no, ncls, ocls, qcm, xcm)
  call xzq_build(no, 0_8)
  lastl = -1
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
              if (.not. dense) then
                ! Xv of T2 with the dressed <bc||ef>
                kind = [0_8, 0_8]
                ka = [kt2, kt2]
                sw = [0_8, 0_8]
                call N(xz4sxv)(w, x, l, 1_8, kind, ka, sw, 0_8, w(jqtri), w(kxv + NW * slot(x) * nxv), w(kmx), w(kwx))
                if (eom) then
                  ! of the vector: T2 with <bc||ef>' and r2 with the dressed <bc||ef>
                  kind = [1_8, 0_8]
                  ka = [kt2, x4r2]
                  sw = [s, 0_8]
                  call N(xz4sxv)(w, x, l, 2_8, kind, ka, sw, s, w(jqtri), w(kxvs + NW * slot(x) * nxv), w(kmx), w(kwx))
                end if
              end if
            end if
          end do
          if (dense .and. nnew > 0) then
            kind = [0_8, 0_8]
            ka = [kt2, kt2]
            call N(xz4xv)(w, nnew, xnew, l, slot, 1_8, kind, ka, w(kxv), w(kwx), w(kmx))
            if (eom) then
              kind = [1_8, 0_8]
              ka = [kt2, x4r2]
              call N(xz4xv)(w, nnew, xnew, l, slot, 2_8, kind, ka, w(kxvs), w(kwx), w(kmx))
            end if
          end if
          tq(2) = tq(2) + omp_get_wtime() - t0
          do i = lo1, hi1 - 1
            do j = max(i + 1, lo2), hi2 - 1
              do k = max(j + 1, lo3), hi3 - 1
                eo = w(jeps + i) + w(jeps + j) + w(jeps + k) + w(jeps + l)
                mt = ocls(i) + ocls(j) + ocls(k) + ocls(l)
                if (dense .and. eom) then
                  ! ---- r4 of the vector at the frequency, with H~
                  call zc_tn_zero(w(kz), 8 * NW * na4)
                  call N(xz4fwd)(w, i, j, k, l, na4, w(jotri), w(josgn), w(g4t2p), w(g4t2px), w(x4xos), w(x4q1), &
                                 w(x4q2), kt3, 1d0, w(kz), w(kw22), w(kw13), w(kla), w(kra), w(klb), w(krb), &
                                 w(klc), w(krc), w(kxvs), slot, tq)
                  call N(xz4fwd)(w, i, j, k, l, na4, w(jotri), w(josgn), w(x4r2p), w(x4r2px), w(g4xo), w(g4q1), &
                                 w(g4q2), x4r3, 1d0, w(kz), w(kw22), w(kw13), w(kla), w(kra), w(klb), w(krb), &
                                 w(klc), w(krc), w(kxv), slot, tq)
                  call N(zc_cc4_divide)(w(kz), w(kev4), eo + omega, na4)
                  call N(xz4back)(w, i, j, k, l, na4, w(jotri), w(josgn), w(kz), w(kw22), w(kw13), w(kwh), &
                                  w(ky), w(klk), w(kbo), w(ktg), w(g4fov), w(jgpp), .true., g4d, w(g4ooov), &
                                  w(kr2), kr3, tq)
                end if
                if (dense) then
                  ! ---- t4 of the ground state: with H~ (the amplitude equations) or with H'
                  call zc_tn_zero(w(kz), 8 * NW * na4)
                  call N(xz4fwd)(w, i, j, k, l, na4, w(jotri), w(josgn), w(g4t2p), w(g4t2px), w(g4xo), w(g4q1), &
                                 w(g4q2), kt3, 1d0, w(kz), w(kw22), w(kw13), w(kla), w(kra), w(klb), w(krb), &
                                 w(klc), w(krc), w(kxv), slot, tq)
                  t0 = omp_get_wtime()
                  call N(zc_cc4_divide)(w(kz), w(kev4), eo + merge(0d0, omega, eom), na4)
                  tq(5) = tq(5) + omp_get_wtime() - t0
                  if (eom) then
                    call N(xz4back)(w, i, j, k, l, na4, w(jotri), w(josgn), w(kz), w(kw22), w(kw13), w(kwh), &
                                    w(ky), w(klk), w(kbo), w(ktg), w(x4fov), w(jgpp), .false., x4d, w(x4ooov), &
                                    w(kr2), kr3, tq)
                  else
                    call N(xz4back)(w, i, j, k, l, na4, w(jotri), w(josgn), w(kz), w(kw22), w(kw13), w(kwh), &
                                    w(ky), w(klk), w(kbo), w(ktg), w(g4fov), w(jgpp), .true., g4d, w(g4ooov), &
                                    w(kr2), kr3, tq)
                  end if
                end if
                if (.not. dense .and. eom) then
                  call N(xz4slay)(mt, s, off22, nc22, n22, off13, nc13, n13)
                  if (n22 + n13 > 0) then
                    ! H' with T2, T3 (the first products of the blocks); then H~ with r2, R3
                    call N(xz4sfw)(w, i, j, k, l, s, s, 0_8, .true., w(jotri), w(josgn), w(jopidx), w(jopgrp), w(g4t2p), &
                                   w(g4t2px), w(x4xos), w(x4q1), w(x4q2), kt3, jobase, boff, w(kz), w(kw13), w(kla), &
                                   w(kra), w(klc), w(kxvs), nxv, slot, off22, nc22, off13, nc13, tq)
                    call N(xz4sfw)(w, i, j, k, l, s, 0_8, s, .false., w(jotri), w(josgn), w(jopidx), w(jopgrp), w(x4r2p), &
                                   w(x4r2px), w(g4xo), w(g4q1), w(g4q2), x4r3, jobase_s, boff_s, w(kz), w(kw13), &
                                   w(kla), w(kra), w(klc), w(kxv), nxv, slot, off22, nc22, off13, nc13, tq)
                    t0 = omp_get_wtime()
                    call N(zc_cc4_sector_t4i)(int(nv, 4), w(jqkv), 0_4, int(mt + s, 4), w(jeps + no), eo + omega, 0_4, &
                                              int(ncls, 4), w(jqlid), w(jqlst), w(jqlmem), 0_4, int(npg, 4), off22, &
                                              nc22, off13, nc13, w(jqposv), w(jqposq), w(jqpos3), w(kz), w(kw13))
                    tq(5) = tq(5) + omp_get_wtime() - t0
                    call N(xz4sbk)(w, i, j, k, l, s, 0_8, .true., w(jotri), w(josgn), w(jopidx), w(jopgrp), w(kz), &
                                   w(kw13), off22, nc22, off13, nc13, w(jqtri), w(ky), w(kl2), w(klk), &
                                   w(g4fov), g4d, w(g4ooov), kr2, kr3, boff_s, tq)
                  end if
                end if
                if (.not. dense) then
                  call N(xz4slay)(mt, 0_8, off22, nc22, n22, off13, nc13, n13)
                  if (n22 + n13 > 0) then
                    call N(xz4sfw)(w, i, j, k, l, 0_8, 0_8, 0_8, .true., w(jotri), w(josgn), w(jopidx), w(jopgrp), w(g4t2p), &
                                   w(g4t2px), w(g4xo), w(g4q1), w(g4q2), kt3, jobase, boff, w(kz), w(kw13), w(kla), &
                                   w(kra), w(klc), w(kxv), nxv, slot, off22, nc22, off13, nc13, tq)
                    t0 = omp_get_wtime()
                    call N(zc_cc4_sector_t4i)(int(nv, 4), w(jqkv), 0_4, int(mt, 4), w(jeps + no), &
                                              eo + merge(0d0, omega, eom), 0_4, int(ncls, 4), w(jqlid), w(jqlst), &
                                              w(jqlmem), 0_4, int(npg, 4), off22, nc22, off13, nc13, w(jqposv), &
                                              w(jqposq), w(jqpos3), w(kz), w(kw13))
                    tq(5) = tq(5) + omp_get_wtime() - t0
                    if (eom) then
                      call N(xz4sbk)(w, i, j, k, l, 0_8, s, .false., w(jotri), w(josgn), w(jopidx), w(jopgrp), &
                                     w(kz), w(kw13), off22, nc22, off13, nc13, w(jqtri), w(ky), w(kl2), w(klk), &
                                     w(x4fov), x4d, w(x4ooov), kr2, kr3, boff_s, tq)
                    else
                      call N(xz4sbk)(w, i, j, k, l, 0_8, 0_8, .true., w(jotri), w(josgn), w(jopidx), w(jopgrp), &
                                     w(kz), w(kw13), off22, nc22, off13, nc13, w(jqtri), w(ky), w(kl2), w(klk), &
                                     w(g4fov), g4d, w(g4ooov), kr2, kr3, boff, tq)
                    end if
                  end if
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
end subroutine

! ---------------------------------------------------------------------------
! The extrapolation over the macro iterations of CC4 (Anderson, one step of
! history).  A macro iteration is the map G: amplitudes x at which the T4
! terms are evaluated -> the amplitudes g that solve the equations of CCSDT
! with these terms held.  With f = g - x of this and of the macro iteration
! before, the next evaluation is made at
!   x' = g - theta (g - g_before),   theta = <f, f - f_before> / |f - f_before|^2,
! the combination of the two g for which that of the two f is shortest.
! kx: x (N(xz4accget) took it when the T4 terms were evaluated), kg, kf: g
! and f of the macro iteration before (nacc > 0), each [T1 | T2 | T3].  On
! return the amplitudes are x', and kx, kg, kf are ready for the next call;
! theta = 0: no step (the first call, or a theta beyond 1/2 in size).
subroutine N(xz4acc)(w, kx, kg, kf, nacc, theta)
  use xzdata
  use xzkern
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kx, kg, kf
  integer(8), intent(inout) :: nacc
  real(8), intent(out) :: theta
  integer(8) :: n(3), kt(3), o(3), p
  ZT :: s
  real(8) :: dff, dfp, dpp, den
  n = [no * nv, no * no * nv * nv, n3]
  kt = [jt1, jt2, jt3]
  o = [0_8, n(1), n(1) + n(2)]
  theta = 0d0
  ! f = g - x, in the place of x
  do p = 1, 3
    call N(xz4accf)(n(p), w(kt(p)), w(kx + NW * o(p)))
  end do
  if (nacc > 0) then
    dff = 0d0
    dfp = 0d0
    dpp = 0d0
    do p = 1, 3
      call N(zc_vdot)(n(p), w(kx + NW * o(p)), w(kx + NW * o(p)), s)
      dff = dff + dble(s)
      call N(zc_vdot)(n(p), w(kx + NW * o(p)), w(kf + NW * o(p)), s)
      dfp = dfp + dble(s)
      call N(zc_vdot)(n(p), w(kf + NW * o(p)), w(kf + NW * o(p)), s)
      dpp = dpp + dble(s)
    end do
    den = dff - 2d0 * dfp + dpp
    if (den > 0d0) theta = (dff - dfp) / den
    if (abs(theta) > 0.5d0) theta = 0d0
  end if
  ! the amplitudes: g - theta (g - g_before); g and f kept
  do p = 1, 3
    call N(xz4accs)(n(p), theta, w(kt(p)), w(kg + NW * o(p)))
    call N(xzecopy)(n(p), w(kx + NW * o(p)), w(kf + NW * o(p)))
    call N(xzecopy)(n(p), w(kt(p)), w(kx + NW * o(p)))
  end do
  nacc = nacc + 1
end subroutine

! the amplitudes [T1 | T2 | T3] to kx
subroutine N(xz4accget)(w, kx)
  use xzdata
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kx
  call N(xzecopy)(no * nv, w(jt1), w(kx))
  call N(xzecopy)(no * no * nv * nv, w(jt2), w(kx + NW * no * nv))
  call N(xzecopy)(n3, w(jt3), w(kx + NW * (no * nv + no * no * nv * nv)))
end subroutine

! x <- g - x
subroutine N(xz4accf)(n, g, x)
  implicit none
  integer(8), intent(in) :: n
  ZT, intent(in) :: g(n)
  ZT, intent(inout) :: x(n)
  integer(8) :: i
  !$omp parallel do
  do i = 1, n
    x(i) = g(i) - x(i)
  end do
end subroutine

! t <- t - theta (t - g), g <- the t of before
subroutine N(xz4accs)(n, theta, t, g)
  implicit none
  integer(8), intent(in) :: n
  real(8), intent(in) :: theta
  ZT, intent(inout) :: t(n), g(n)
  integer(8) :: i
  ZT :: tk
  !$omp parallel do private(tk)
  do i = 1, n
    tk = t(i)
    t(i) = tk - theta * (tk - g(i))
    g(i) = tk
  end do
end subroutine

! ---------------------------------------------------------------------------
! The file XZCC4 of the run directory: the CC4 iterations for a later run
! (docs/native.md, section 19.5).  Rank 0 writes it after every completed
! evaluation of the T4 terms (macro / micro) and when CC4 has converged; an
! unformatted stream:
!   character(8)  'XZCC4_02'
!   integer(8)    active occupied, active virtual, spinors, frozen occupied
!                 spinors; words per element, classes of the labels, elements
!                 of T3 as stored; T4 evaluations so far; CC_CONV of converged
!                 amplitudes, or -1: the amplitudes at which the last
!                 evaluation was made, followed by its terms; the macro
!                 iterations in the history of the extrapolation (0 or 1)
!   real(8)       the three numbers of XZKEY (xzkeyval); the CCSDT correlation
!                 energy of the start, the correlation energy of the
!                 amplitudes, the change of the amplitudes in the first step
!                 after the evaluation before, the seconds of the evaluations,
!                 the largest factor between the first steps of two evaluations
!   T1 [i][a], T2 [i][j][a][b], T3 as stored
!   if not converged: the T4 terms of R2 [P][Q] and of R3 (as T3); with a
!   history, g and f of the macro iteration before (N(xz4acc); [T1 | T2 | T3] each)
! rw = 'w': written from jt1, jt2, jt3, jd42, jd43 - to XZCC4.new, which then
! takes the name (an interrupted write leaves the file of before); a failure
! is a warning.  'r': into these places; cconv, ecorr, dprev of the file,
! nt4, t4sec and e4ccsdt of xzdata set.
subroutine N(xz4ckpt)(w, rw, cconv, ecorr, dprev)
  use xzdata
  use xzkern
  use omp_lib
  implicit none
  real(8) :: w(*)
  character(len=1), intent(in) :: rw
  integer(8), intent(inout) :: cconv
  real(8), intent(inout) :: ecorr, dprev
  character(len=8) :: tag
  integer(8) :: hd(10), n1, n2, nh, nall
  real(8) :: key(3), st(5), t0
  integer(8) :: ios
  n1 = NW * no * nv
  n2 = NW * no * no * nv * nv
  nall = n1 + n2 + NW * n3
  t0 = omp_get_wtime()
  if (rw == 'w') then
    call xzkeyval(key)
    nh = 0
    if (cconv < 0 .and. acc4) nh = min(a4n, 1_8)
    open(unit=77, file='XZCC4.new', form='unformatted', access='stream', status='replace', iostat=ios)
    if (ios == 0) write(77, iostat=ios) 'XZCC4_02', no, nv, ntot, nfrozen, int(NW, 8), ncls, n3, nt4, cconv, nh, key, &
                                        e4ccsdt, ecorr, dprev, t4sec, a4kmax
    if (ios == 0) write(77, iostat=ios) w(jt1:jt1 + n1 - 1)
    if (ios == 0) write(77, iostat=ios) w(jt2:jt2 + n2 - 1)
    if (ios == 0) write(77, iostat=ios) w(jt3:jt3 + NW * n3 - 1)
    if (ios == 0 .and. cconv < 0) write(77, iostat=ios) w(jd42:jd42 + NW * npo * nvp - 1)
    if (ios == 0 .and. cconv < 0) write(77, iostat=ios) w(jd43:jd43 + NW * n3 - 1)
    if (ios == 0 .and. nh > 0) write(77, iostat=ios) w(a4g:a4g + nall - 1)
    if (ios == 0 .and. nh > 0) write(77, iostat=ios) w(a4f:a4f + nall - 1)
    if (ios == 0) close(77, iostat=ios)
    if (ios == 0) ios = xz_rename('XZCC4.new' // c_null_char, 'XZCC4' // c_null_char)
    if (ios /= 0) then
      write(6, '(a,i8,a)') ' @XZCCSDT-W, the file XZCC4 could not be written (iostat ', ios, &
        '; disk full?): a later run cannot continue from here.'
      close(77, status='delete', iostat=ios)
    else if (cconv < 0) then
      write(6, '(a,i3,a,f10.1,a,f8.2,a)') '     checkpoint: the amplitudes and the T4 terms of evaluation ', nt4, &
        ' written to the file XZCC4 (', 8d-6 * (n1 + n2 + 2 * NW * n3 + NW * npo * nvp + 2 * nh * nall), ' MB, ', &
        omp_get_wtime() - t0, &
        ' s): a run of this ZMAT in this directory continues after it'
    else
      write(6, '(a,f10.1,a,f8.2,a)') '  the converged CC4 amplitudes written to the file XZCC4 (', &
        8d-6 * (n1 + n2 + NW * n3), ' MB, ', omp_get_wtime() - t0, ' s)'
    end if
    flush(6)
    return
  end if
  open(unit=77, file='XZCC4', form='unformatted', access='stream', status='old', action='read', iostat=ios)
  if (ios == 0) read(77, iostat=ios) tag, hd, key, st
  if (ios /= 0 .or. tag /= 'XZCC4_02') then
    write(6, '(/,a)') ' @XZCCSDT-F, the file XZCC4 cannot be read.'
    call xzerrex
  end if
  if (hd(1) /= no .or. hd(2) /= nv .or. hd(5) /= NW .or. hd(6) /= ncls .or. hd(7) /= n3) then
    write(6, '(/,a)') ' @XZCCSDT-F, the file XZCC4 belongs to another calculation or another mode:'
    write(6, '(a,5i12)') '   occupied, virtual, words per element, classes, T3 in the file: ', hd(1), hd(2), hd(5), hd(6), hd(7)
    write(6, '(a,5i12)') '   and here:                                                      ', no, nv, &
      int(NW, 8), ncls, n3
    write(6, '(a)') '   (the labels and the arithmetic must be those of the run that wrote it: option labels, option real;'
    write(6, '(a)') '    remove the file to start again from the CCSDT amplitudes)'
    call xzerrex
  end if
  nt4 = hd(8)
  cconv = hd(9)
  e4ccsdt = st(1)
  ecorr = st(2)
  dprev = st(3)
  t4sec = st(4)
  read(77, iostat=ios) w(jt1:jt1 + n1 - 1)
  if (ios == 0) read(77, iostat=ios) w(jt2:jt2 + n2 - 1)
  if (ios == 0) read(77, iostat=ios) w(jt3:jt3 + NW * n3 - 1)
  if (ios == 0 .and. cconv < 0) read(77, iostat=ios) w(jd42:jd42 + NW * npo * nvp - 1)
  if (ios == 0 .and. cconv < 0) read(77, iostat=ios) w(jd43:jd43 + NW * n3 - 1)
  ! (the history of the extrapolation, if this run extrapolates)
  a4n = 0
  a4kmax = st(5)
  if (ios == 0 .and. cconv < 0 .and. hd(10) > 0 .and. acc4) then
    read(77, iostat=ios) w(a4g:a4g + nall - 1)
    if (ios == 0) read(77, iostat=ios) w(a4f:a4f + nall - 1)
    a4n = hd(10)
  end if
  close(77)
  if (ios /= 0) then
    write(6, '(/,a,i8)') ' @XZCCSDT-F, reading the file XZCC4 failed, iostat ', ios
    call xzerrex
  end if
  write(6, '(/,a,f10.1,a,f8.2,a)') '   the file XZCC4 read (', 8d-6 * (n1 + n2 + NW * n3), ' MB of amplitudes, ', &
    omp_get_wtime() - t0, ' s)'
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
! Xo of N(xzqxo) (xzq.inc) for the given blocks <mn||jk> and <md||el> and
! amplitudes; acc: added to xo.
subroutine N(xz4xo)(no, nv, npo, nvp, nto, oooo, ovvo, t2p, t2px, opa, opb, vpa, vpb, ofree, opair, &
                    xo, g, km, b, oo, acc)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp, nto
  ZT, intent(in) :: oooo(0:no-1, 0:no-1, 0:no-1, 0:no-1)       ! [m][n][j][k]
  ZT, intent(in) :: ovvo(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)       ! [m][d][e][l]
  ZT, intent(in) :: t2p(0:nvp-1, 0:no-1, 0:no-1)               ! [n][l][Q]
  ZT, intent(in) :: t2px(*)                                    ! [P][c][e] = - t2[j][k][c][e]
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  integer(8), intent(in) :: ofree(0:nto-1, 0:2), opair(0:nto-1, 0:2)
  ZT, intent(inout) :: xo(0:nvp-1, 0:no-1, 0:nto-1)
  ZT :: g(0:nvp-1, 0:npo-1, 0:no-1), km(0:nv-1, 0:nv-1, 0:npo-1), b(0:nv-1, 0:nv-1), oo(0:no-1, 0:npo-1)
  logical, intent(in) :: acc
  integer(8) :: m, l, p, q, n, d, e, it
  ZT :: mone, zero, half, one
  mone = -1
  zero = 0
  one = 1
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
      call N(zc_gemm)(0_4, 0_4, npo, nvp, no, half, oo, no, t2p(0, l, 0), no * nvp, one, g(0, 0, l), nvp)
    end do
    if (acc) then
      !$omp parallel do private(it)
      do it = 0, nto - 1
        xo(:, m, it) = xo(:, m, it) + g(:, opair(it, 0), ofree(it, 0)) - g(:, opair(it, 1), ofree(it, 1)) &
                       + g(:, opair(it, 2), ofree(it, 2))
      end do
    else
      !$omp parallel do private(it)
      do it = 0, nto - 1
        xo(:, m, it) = g(:, opair(it, 0), ofree(it, 0)) - g(:, opair(it, 1), ofree(it, 1)) &
                       + g(:, opair(it, 2), ofree(it, 2))
      end do
    end if
  end do
end subroutine

! ---------------------------------------------------------------------------
! Xv of N(xzqxv) (no labels) for the nnew occupied x of xnew, as a sum of
! nterm terms
!   xv[e][(bcd)] = P(d/bc) 1/2 sum_f W[(bc)][e][f] a[x][l][f][d]
!   kind 0   W = the T1-dressed <bc||ef> (jv4 of xzdata)
!   kind 1   W = <bc||ef>' of H' = [H~, R1] (x4vv)
!   ka       the amplitudes a [i][j][a][b] of the term (index in w)
! wx: work nterm * qrows * nv * nv.
subroutine N(xz4xv)(w, nnew, xnew, l, slot, nterm, kind, ka, xv, wx, mx)
  use xzdata
  use xzkern
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: nnew, l, nterm, kind(2), ka(2)
  integer(8), intent(in) :: xnew(0:*), slot(0:*)
  ZT :: xv(0:nv*nvt-1, 0:*), wx(0:qrows*nv*nv-1, *), mx(*)
  integer(8) :: b0, nb, k, x, t, kw
  ZT :: one, zero, beta
  one = 1
  zero = 0
  do k = 0, nnew - 1
    call zc_tn_zero(xv(0, slot(xnew(k))), 8 * NW * nv * nvt)
  end do
  do b0 = 0, nvp - 1, qrows
    nb = min(qrows, nvp - b0)
    do t = 1, nterm
      kw = merge(jv4, x4vv, kind(t) == 0)
      call zc_tn_zero(wx(0, t), 8 * NW * nb * nv * nv)
      call N(zc_tn_expand)(w(kw + NW * b0 * nvp), nb, nvp, int(nv, 4), 1_4, 1_4, 0_8, nv, 0_8, nv, wx(0, t), nv, 1_8, &
                           nv * nv, 1d0)
    end do
    do k = 0, nnew - 1
      x = xnew(k)
      beta = zero
      do t = 1, nterm
        call N(zc_gemm)(0_4, 0_4, nb * nv, nv, nv, one, wx(0, t), nv, w(ka(t) + NW * nv * nv * (l + no * x)), nv, &
                        beta, mx, nv)
        beta = one
      end do
      call N(zc_tn_reduce)(xv(0, slot(x)), nv, nvt, int(nv, 4), 2_4, 1_4, b0, b0 + nb, 0_8, nv, mx, nv * nv, &
                           1_8, nv, 0.5d0)
    end do
  end do
end subroutine

! the first element of the block of the class ke of e in Xv of a pair of
! total class kxl, for an Xv of shift sx: [e of class ke][triples of group ke + kxl + sx]
integer(8) function N(xz4xvoff)(kxl, ke, sx)
  use xzdata
  implicit none
  integer(8), intent(in) :: kxl, ke, sx
  integer(8) :: k, t
  N(xz4xvoff) = 0
  do k = 0, ke - 1
    t = k + kxl + sx
    if (t < 0 .or. t >= ntg) cycle
    N(xz4xvoff) = N(xz4xvoff) + (vcoff(k + 1) - vcoff(k)) * (vtoff(t + 1) - vtoff(t))
  end do
end function

! ---------------------------------------------------------------------------
! Xv of N(xzqsxv) (the blocks of the pair (x, l)) as a sum of nterm terms
!   xv[e][(bcd)] = P(d/bc) 1/2 sum_f W[(bc)][e][f] a[x][l][f][d]
! of the shift sx = sw + (the shift of a): for e of class ke the triples of
! group ke + class(x) + class(l) + sx.
!   kind 0   W = the T1-dressed <bc||ef> (sw = 0): its block of the group of
!            (bc) in the dressed <ab||cd> (jv4 of xzdata)
!   kind 1   W = <bc||ef>' = - sum_m r1[m][b] ~<mc||ef> + sum_m r1[m][c] ~<mb||ef>
!            (shift sw: the group of (bc) less the classes of e and f): its
!            block of the group of (bc) at x4vv
!   ka       the amplitudes a [i][j][a][b] of the term (index in w)
subroutine N(xz4sxv)(w, x, l, nterm, kind, ka, sw, sx, tri, xv, m3, wb)
  use xzdata
  use xzkern
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: x, l, nterm, kind(2), ka(2), sw(2), sx
  integer(4), intent(in) :: tri(0:2, 0:nvt-1)
  ZT :: xv(0:*), m3(0:*), wb(*)
  integer(8) :: kxl, ke, t, ne, nt, q2, qc, kf, kd, nf, nd, nq, n, xoff, it
  integer(8) :: off(0:npg-1), ndd(0:npg-1)
  ZT :: one, zero, beta
  one = 1
  zero = 0
  kxl = ocls(x) + ocls(l)
  xoff = 0
  do ke = 0, ncls - 1
    t = ke + kxl + sx
    if (t < 0 .or. t >= ntg) cycle
    ne = vcoff(ke + 1) - vcoff(ke)
    nt = vtoff(t + 1) - vtoff(t)
    if (ne == 0 .or. nt == 0) cycle
    off = -1
    ndd = 0
    n = 0
    do q2 = 0, npg - 1
      ! (the triple (bc, d) is of group t)
      kd = t - q2
      if (kd < 0 .or. kd >= ncls) cycle
      nd = vcoff(kd + 1) - vcoff(kd)
      nq = vpoff(q2 + 1) - vpoff(q2)
      if (nd == 0 .or. nq == 0) cycle
      beta = zero
      do it = 1, nterm
        kf = q2 - ke - sw(it)
        if (kf < 0 .or. kf >= ncls) cycle
        nf = vcoff(kf + 1) - vcoff(kf)
        if (nf == 0) cycle
        if (kind(it) == 0) then
          call N(xzqswb)(nv, nq, ne, nf, vcoff(ke), vcoff(kf), vpoff(q2), w(jvpidx), w(jv4 + NW * vvoff(q2)), wb)
        else
          ! (the pairs (ef) are of the group q2 - sw)
          qc = q2 - sw(it)
          if (x4vvoff(q2) < 0) cycle
          call N(xz4swb)(nv, nq, vpoff(qc + 1) - vpoff(qc), ne, nf, vcoff(ke), vcoff(kf), vpoff(qc), w(jvpidx), &
                         w(x4vv + NW * x4vvoff(q2)), wb)
        end if
        call N(zc_gemm)(0_4, 0_4, nq * ne, nd, nf, one, wb, nf, &
                        w(ka(it) + NW * (((x * no + l) * nv + vcoff(kf)) * nv + vcoff(kd))), nv, beta, m3(n), nd)
        beta = one
      end do
      if (beta == zero) cycle
      off(q2) = n
      ndd(q2) = nd
      n = n + nq * ne * nd
    end do
    call N(zc_xv_sector)(int(nv, 4), w(jqkv), 0_4, tri(0, vtoff(t)), nt, int(ne, 4), 0_4, &
                         int(npg, 4), off, ndd, w(jqposv), w(jqposq), m3, xv(xoff))
    xoff = xoff + ne * nt
  end do
end subroutine

! wb[(bc)][e][f] = W[(bc)][(ef)] from a block vv[(bc)][(ef)] of nq rows and
! the nqc pairs (ef) of one group, whose first pair is clo (N(xzqswb) for a
! block that is not square)
subroutine N(xz4swb)(nv, nq, nqc, ne, nf, e0, f0, clo, vpidx, vv, wb)
  implicit none
  integer(8), intent(in) :: nv, nq, nqc, ne, nf, e0, f0, clo
  integer(8), intent(in) :: vpidx(0:nv-1, 0:nv-1)              ! [p][q]
  ZT, intent(in) :: vv(0:nqc-1, 0:nq-1)                        ! [(bc)][(ef)]
  ZT, intent(out) :: wb(0:nf-1, 0:ne-1, 0:nq-1)                ! [(bc)][e][f]
  integer(8) :: b, e, f, ge, gf
  !$omp parallel do private(b, e, f, ge, gf)
  do b = 0, nq - 1
    do e = 0, ne - 1
      ge = e0 + e
      do f = 0, nf - 1
        gf = f0 + f
        if (ge < gf) then
          wb(f, e, b) = vv(vpidx(gf, ge) - clo, b)
        else if (ge > gf) then
          wb(f, e, b) = -vv(vpidx(ge, gf) - clo, b)
        else
          wb(f, e, b) = 0
        end if
      end do
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! One occupied quadruple i<j<k<l without labels, the right-hand side:
!   z[A] += wt * <Q| [V~, T3] + 1/2 [[V~, T2], T2] |0>
! - the five products of N(xzquad) with the blocks q1, q2 (N(xz4q12)), Xo and
! Xv of the dressed integrals and the amplitudes t2p, t2px, T3 at kt3 (or
! with the operands of a sigma vector in their places).
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
!     r2[P][(ab)]  += sgn sum_(ef) <P'||ef> X                    six splits (P | P'); if withr2
!     R3[J]        += reduction (c | ab) of sgn sum_(ef) ~<mc||ef> X    four splits (m | J)
!   the row on an index and a triple, X[e][(abc)], columns (abc):
!     R3[J]        += sgn sum_e f[m][e] X                        four splits (m | J)
!     R3[(k, P)]   += sgn sgn' sum_e ~<P'||ke> X                 six splits, k outside P
! wh: [2][6][nvp]; y: [nv][qc22]; lk: [4 + 6 no][nv]; bo: [4 + 6 no][qc13]; tg: 4 + 6 no rows.
subroutine N(xz4back)(w, i, j, k, l, na4, otri, osgn, z, w22, w13, wh, y, lk, bo, tg, fov, gpp, withr2, kd, &
                      ooovd, r2, kr3, tq)
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
  logical, intent(in) :: withr2
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
    if (withr2) call N(zc_gemm)(0_4, 0_4, 6_8, nc, nvp, one, wh(0, 0, 1), nvp, w22, nc, zero, wh(c0, 0, 0), nvp)
    do s = 0, 3
      alpha = s13(s)
      call N(zc_gemm)(0_4, 0_4, nv, nc, nvp, alpha, w(kd + NW * nvp * nv * q(s)), nvp, w22, nc, zero, y, nc)
      call N(zc_tn_reduce)(w(kr3 + NW * nvt * jts(s)), 1_8, nvt, int(nv, 4), 1_4, 2_4, 0_8, nv, c0, c0 + nc, &
                           y, nc, 1_8, 0_8, 1d0)
    end do
  end do
  if (withr2) then
    do s = 0, 5
      r2(:, p1(s)) = r2(:, p1(s)) + wh(:, s, 0)
    end do
  end if
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
! The blocks of the row of an occupied quadruple of total class mt on the
! virtual quadruples of total class mt + st: on two pairs one block per
! group qq of (ab) - at off22(qq), nc22(qq) pairs (cd) of the group
! mt + st - qq -, on an index and a triple one per class p of a (off13, nc13:
! the triples of the group mt + st - p); n22, n13 elements in all.
subroutine N(xz4slay)(mt, st, off22, nc22, n22, off13, nc13, n13)
  use xzdata
  implicit none
  integer(8), intent(in) :: mt, st
  integer(8), intent(out) :: off22(0:npg-1), nc22(0:npg-1), off13(0:ncls-1), nc13(0:ncls-1), n22, n13
  integer(8) :: qq, r, p, nr, nc
  off22 = -1
  nc22 = 0
  n22 = 0
  do qq = 0, npg - 1
    r = mt + st - qq
    if (r < 0 .or. r >= npg) cycle
    nr = vpoff(qq + 1) - vpoff(qq)
    nc = vpoff(r + 1) - vpoff(r)
    if (nr == 0 .or. nc == 0) cycle
    off22(qq) = n22
    nc22(qq) = nc
    n22 = n22 + nr * nc
  end do
  off13 = -1
  nc13 = 0
  n13 = 0
  do p = 0, ncls - 1
    r = mt + st - p
    if (r < 0 .or. r >= ntg) cycle
    nr = vcoff(p + 1) - vcoff(p)
    nc = vtoff(r + 1) - vtoff(r)
    if (nr == 0 .or. nc == 0) cycle
    off13(p) = n13
    nc13(p) = nc
    n13 = n13 + nr * nc
  end do
end subroutine

! ---------------------------------------------------------------------------
! One occupied quadruple i<j<k<l in the sectors of the labels: the five
! products of N(xzqsquad) into the blocks b (two pairs) and c (an index and a
! triple) of the layout of N(xz4slay) for the row shift st.
!   q1, q2, kt3    the operands of the products with the rank-3 array: q1,
!                  q2 of shift s3l, the rank-3 array at kt3 of shift
!                  st - s3l in its layout (kob: obase, bof: its blocks)
!   t2p, t2px, xo, xv   those of the products of two rank-2 arrays: t2p,
!                  t2px of shift s2l, xo and xv of shift st - s2l (nxv: the
!                  elements of a slot of xv)
!   fresh          the blocks are set (the first products of the row), not
!                  added to
! A block on two pairs is one product: the rows of its two kinds of splits
! stacked along the summed index.  A block on an index and a triple is one
! pass over its columns (N(zc_cc4_rowsmm)): the rows of the rank-3 array and
! those of Xv are read where they are.
subroutine N(xz4sfw)(w, i, j, k, l, st, s3l, s2l, fresh, otri, osgn, opidx, opgrp, t2p, t2px, xo, q1, q2, kt3, kob, bof, &
                     b, c, la, ra, lc, xv, nxv, slot, off22, nc22, off13, nc13, tq)
  use xzdata
  use xzkern
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: i, j, k, l, st, s3l, s2l, kt3, kob, bof(0:*), nxv
  logical, intent(in) :: fresh
  integer(8), intent(in) :: otri(0:npo-1, 0:no-1), opidx(0:no-1, 0:no-1), opgrp(0:npo-1)
  integer(1), intent(in) :: osgn(0:npo-1, 0:no-1)
  ZT, intent(in) :: t2p(0:nvp-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: t2px(0:nv-1, 0:nv-1, 0:npo-1)              ! [P][e][a]
  ZT, intent(in) :: xo(0:nvp-1, 0:no-1, 0:nto-1)
  ZT, intent(in) :: q1(0:nvp-1, 0:nv-1, 0:no-1)                ! [x][e][Q]
  ZT, intent(in) :: q2(0:nv-1, 0:no-1, 0:no-1, 0:no-1)         ! [i][j][m][a]
  ZT :: b(0:*), c(0:*)
  ZT :: la(0:*), ra(0:*), lc(0:*)
  ZT, intent(in) :: xv(0:nxv-1, 0:*)
  integer(8), intent(in) :: slot(0:*)
  integer(8), intent(in) :: off22(0:npg-1), nc22(0:npg-1), off13(0:ncls-1), nc13(0:ncls-1)
  real(8), intent(inout) :: tq(6)
  integer(8) :: q(0:3), s, x, m, mt, qq, r, nr, nc, na, ke, ne, nm, p, km, kxl, n, g3, e, xb
  integer(8) :: jt(0:3), p22(0:5), q22(0:5), xfirst(0:5)
  integer(8) :: nas(0:npg-1), loff(0:npg-1), roff(0:npg-1), uo(0:7, 0:npg-1), nl, nrr, u
  integer(c_intptr_t) :: ptr(6 * nocmax + 3 * nvcmax)
  real(8) :: s13(0:3), s22(0:5), t0, alpha
  integer(8), parameter :: f1(0:5) = [0, 0, 0, 1, 1, 2], f2(0:5) = [1, 2, 3, 2, 3, 3]
  integer(8), parameter :: r1(0:5) = [2, 1, 1, 0, 0, 0], r2(0:5) = [3, 3, 2, 3, 2, 1]
  integer(8), parameter :: last(3) = [0, 1, 3]
  integer(8), external :: N(xz4xvoff)
  ZT :: one, zero, beta
  one = 1
  zero = 0
  q = [i, j, k, l]
  s13 = [1d0, -1d0, 1d0, -1d0]
  s22 = [1d0, -1d0, 1d0, 1d0, -1d0, 1d0]
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

  ! ---- the blocks on two pairs.  The stacked operands of all blocks first
  !      (the threads take blocks: gathers and copies), then the products
  t0 = omp_get_wtime()
  !      (uo: the first row of the stack of a block that a split fills - the splits
  !      0 to 3 of the first kind, then those of the second)
  nl = 0
  nrr = 0
  do qq = 0, npg - 1
    nas(qq) = 0
    if (off22(qq) < 0) cycle
    do s = 0, 3
      uo(s, qq) = nas(qq)
      nas(qq) = nas(qq) + xznvc(qq - ocls(q(s)) - s3l)
    end do
    do s = 0, 3
      uo(4 + s, qq) = nas(qq)
      nas(qq) = nas(qq) + xznoc(qq - ocls(q(s)) - s2l)
    end do
    loff(qq) = nl
    roff(qq) = nrr
    nl = nl + nas(qq) * (vpoff(qq + 1) - vpoff(qq))
    nrr = nrr + nas(qq) * nc22(qq)
  end do
  !$omp parallel do collapse(2) schedule(dynamic, 1) private(qq, u, r, nr, nc, na, s, x, ke, ne, km, nm, m)
  do qq = 0, npg - 1
    do u = 0, 7
      if (off22(qq) < 0) cycle
      r = mt + st - qq
      nr = vpoff(qq + 1) - vpoff(qq)
      nc = nc22(qq)
      na = uo(u, qq)
      s = mod(u, 4_8)
      x = q(s)
      if (u < 4) then
        ! - ~<ab||ex> t(J; e, cd): e of the class that the shift of the integral leaves
        ke = qq - ocls(x) - s3l
        ne = xznvc(ke)
        if (ne == 0) cycle
        call N(xzqsab)(nv, nvp, nr, nc, ne, s13(s), .false., vcoff(ke), vpoff(qq), vpoff(r), &
                       q1(0, 0, x), w(kt3), w(kob + jt(s)), w(jvci), w(jvsgn), la(loff(qq) + na * nr), &
                       ra(roff(qq) + na * nc))
      else
        ! t(x m, ab) Xo(m; J; cd)
        km = qq - ocls(x) - s2l
        nm = xznoc(km)
        do m = 0, nm - 1
          la(loff(qq) + (na + m) * nr:loff(qq) + (na + m) * nr + nr - 1) = &
            s13(s) * t2p(vpoff(qq):vpoff(qq) + nr - 1, ocoff(km) + m, x)
          ra(roff(qq) + (na + m) * nc:roff(qq) + (na + m) * nc + nc - 1) = &
            xo(vpoff(r):vpoff(r) + nc - 1, ocoff(km) + m, jt(s))
        end do
      end if
    end do
  end do
  do qq = 0, npg - 1
    if (off22(qq) < 0) cycle
    nr = vpoff(qq + 1) - vpoff(qq)
    nc = nc22(qq)
    if (nas(qq) > 0) then
      beta = one
      if (fresh) beta = zero
      call N(zc_gemm)(1_4, 0_4, nr, nc, nas(qq), one, la(loff(qq)), nr, ra(roff(qq)), nc, beta, b(off22(qq)), nc)
    else if (fresh) then
      call zc_tn_zero(b(off22(qq)), 8 * NW * nr * nc)
    end if
  end do
  tq(3) = tq(3) + omp_get_wtime() - t0

  ! ---- the blocks on one index and a triple
  t0 = omp_get_wtime()
  do p = 0, ncls - 1
    if (off13(p) < 0) cycle
    r = mt + st - p
    nr = vcoff(p + 1) - vcoff(p)
    nc = nc13(p)
    ! ~<ma||ij> t(m kl; bcd): rows of the rank-3 array where they are, in the
    ! block of the occupied triples of group mt + s3l - p
    g3 = mt + s3l - p
    n = 0
    if (g3 >= 0 .and. g3 < ntg) then
      do s = 0, 5
        km = opgrp(p22(s)) + s3l - p
        if (km < 0 .or. km >= ncls) cycle
        do m = ocoff(km), ocoff(km + 1) - 1
          if (osgn(q22(s), m) == 0) cycle
          lc(n * nr:n * nr + nr - 1) = (s22(s) * osgn(q22(s), m)) * q2(vcoff(p):vcoff(p) + nr - 1, m, q(f2(s)), q(f1(s)))
          ptr(n + 1) = loc(w(kt3 + NW * (bof(g3) + (otri(q22(s), m) - otoff(g3)) * nc)))
          n = n + 1
        end do
      end do
    end if
    ! 2 t(ij, ae) Xv(e; kl; bcd), l in the second pair
    do m = 1, 3
      s = last(m)
      ke = opgrp(p22(s)) + s2l - p
      ne = xznvc(ke)
      if (ne == 0) cycle
      kxl = ocls(xfirst(s)) + ocls(l)
      alpha = 2d0 * s22(s)
      xb = N(xz4xvoff)(kxl, ke, st - s2l)
      do e = 0, ne - 1
        lc(n * nr:n * nr + nr - 1) = alpha * t2px(vcoff(p):vcoff(p) + nr - 1, vcoff(ke) + e, p22(s))
        ptr(n + 1) = loc(xv(xb + e * nc, slot(xfirst(s))))
        n = n + 1
      end do
    end do
    if (n > 0) then
      call N(zc_cc4_rowsmm)(ptr, lc, n, nr, nc, merge(1_8, 0_8, fresh), c(off13(p)))
    else if (fresh) then
      call zc_tn_zero(c(off13(p)), 8 * NW * nr * nc)
    end if
  end do
  tq(4) = tq(4) + omp_get_wtime() - t0
end subroutine

! out[r][c] = sg a[r][c], c < nc, for n rows of an array of leading dimension ld
subroutine N(xz4lrows)(n, nc, sg, a, ld, out)
  implicit none
  integer(8), intent(in) :: n, nc, ld
  real(8), intent(in) :: sg
  ZT, intent(in) :: a(0:ld-1, 0:*)
  ZT, intent(out) :: out(0:nc-1, 0:*)
  integer(8) :: r
  do r = 0, n - 1
    out(:, r) = sg * a(0:nc - 1, r)
  end do
end subroutine

! ---------------------------------------------------------------------------
! One occupied quadruple in the sectors of the labels: the contractions of
! the blocks e22, e13 of its row of T4 (row shift st: N(xz4slay)) with
! operators of shift sb into the four sums, which have the shift st + sb -
! r2 at kr2 [P][Q], the rank-3 array at kr3 in the layout bofo of that shift.
!   y          the products of the blocks on two pairs: [c][(ab)] of the four
!              splits of the term with ~<mc||ef>, rows of the doubles term
!   l2         their stacked left operand;  lk: that of the terms with f and
!              ~<mn||ke>
!   withr2     the doubles term (<mn||ef>: not of H')
! A block on two pairs is read once: one product with the rows of ~<mc||ef>
! of the four splits and those of <mn||ef> of the doubles term stacked.  A
! block on an index and a triple is one pass over its columns
! (N(zc_cc4_addmm)): the rows of the rank-3 array are added to where they are.
subroutine N(xz4sbk)(w, i, j, k, l, st, sb, withr2, otri, osgn, opidx, opgrp, e22, e13, off22, nc22, off13, nc13, &
                     tri, y, l2, lk, fov, kd, ooovd, kr2, kr3, bofo, tq)
  use xzdata
  use xzkern
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: i, j, k, l, st, sb, kd, kr2, kr3, bofo(0:*)
  logical, intent(in) :: withr2
  integer(8), intent(in) :: otri(0:npo-1, 0:no-1), opidx(0:no-1, 0:no-1), opgrp(0:npo-1)
  integer(1), intent(in) :: osgn(0:npo-1, 0:no-1)
  ZT, intent(in) :: e22(0:*), e13(0:*)
  integer(8), intent(in) :: off22(0:npg-1), nc22(0:npg-1), off13(0:ncls-1), nc13(0:ncls-1)
  integer(4), intent(in) :: tri(0:2, 0:nvt-1)
  ZT :: y(0:*), l2(0:*), lk(0:*)
  ZT, intent(in) :: fov(0:nv-1, 0:no-1)                        ! [m][e]
  ZT, intent(in) :: ooovd(0:nv-1, 0:no-1, 0:no-1, 0:no-1)      ! [m][n][i][e]
  real(8), intent(inout) :: tq(6)
  integer(8) :: q(0:3), s, mt, qq, r, nr, nc, p, km, n, g3
  integer(8) :: jt(0:3), p22(0:5), q22(0:5), gj, gv, lcc, nl, nyb, kk, mm, nn, mrow
  integer(8) :: yoff(0:npg-1, 0:3), ynq(0:npg-1, 0:3), gvs(0:3), ntv(0:3), r2row(0:5), r2qq(0:5)
  integer(8) :: mrows(0:npg-1), loff(0:npg-1), ybase(0:npg-1), nlo
  integer(c_intptr_t) :: ptr(4 + 6 * nocmax)
  real(8) :: s13(0:3), s22(0:5), t0
  integer(8), parameter :: f1(0:5) = [0, 0, 0, 1, 1, 2], f2(0:5) = [1, 2, 3, 2, 3, 3]
  integer(8), parameter :: r1(0:5) = [2, 1, 1, 0, 0, 0], r2(0:5) = [3, 3, 2, 3, 2, 1]
  ZT :: one, zero
  one = 1
  zero = 0
  t0 = omp_get_wtime()
  q = [i, j, k, l]
  s13 = [1d0, -1d0, 1d0, -1d0]
  s22 = [1d0, -1d0, 1d0, 1d0, -1d0, 1d0]
  mt = ocls(i) + ocls(j) + ocls(k) + ocls(l)
  jt(0) = otri(opidx(q(3), q(2)), q(1))
  jt(1) = otri(opidx(q(3), q(2)), q(0))
  jt(2) = otri(opidx(q(3), q(1)), q(0))
  jt(3) = otri(opidx(q(2), q(1)), q(0))
  do s = 0, 5
    p22(s) = opidx(q(f2(s)), q(f1(s)))
    q22(s) = opidx(q(r2(s)), q(r1(s)))
  end do
  ! the virtual triples of the rows R3[J] of the four splits (m | J): group gv (-1: none)
  do s = 0, 3
    gv = mt - ocls(q(s)) + st + sb
    gvs(s) = -1
    ntv(s) = 0
    if (gv < 0 .or. gv >= ntg) cycle
    ntv(s) = vtoff(gv + 1) - vtoff(gv)
    if (ntv(s) > 0) gvs(s) = gv
  end do
  ! ---- the blocks on two pairs, X[(ab)][(ef)]:
  !      y[c][(ab)] = sgn sum_(ef) ~<mc||ef> X   for R3[J] += its reduction (c | ab), four splits
  !      r2[P][(ab)] += sgn sum_(ef) <P'||ef> X  for the splits whose P' is of the group of (ef)
  !      The rows of the stacked left operands of all blocks first: where they go,
  !      then the copies (the threads take blocks), then the products
  yoff = -1
  ynq = 0
  r2row = -1
  r2qq = -1
  nyb = 0
  nlo = 0
  do qq = 0, npg - 1
    mrows(qq) = 0
    if (off22(qq) < 0) cycle
    r = mt + st - qq
    nr = vpoff(qq + 1) - vpoff(qq)
    mrow = 0
    do s = 0, 3
      if (gvs(s) < 0) cycle
      nl = xznvc(gvs(s) - qq)
      if (nl == 0) cycle
      yoff(qq, s) = nyb + mrow * nr
      ynq(qq, s) = nr
      mrow = mrow + nl
    end do
    if (withr2) then
      do s = 0, 5
        if (opgrp(q22(s)) /= r) cycle
        r2row(s) = nyb + mrow * nr
        r2qq(s) = qq
        mrow = mrow + 1
      end do
    end if
    mrows(qq) = mrow
    loff(qq) = nlo
    ybase(qq) = nyb
    nlo = nlo + mrow * nc22(qq)
    nyb = nyb + mrow * nr
  end do
  !$omp parallel do schedule(dynamic, 1) private(qq, r, nr, nc, mrow, s, lcc, nl)
  do qq = 0, npg - 1
    if (mrows(qq) == 0) cycle
    r = mt + st - qq
    nr = vpoff(qq + 1) - vpoff(qq)
    nc = nc22(qq)
    mrow = 0
    do s = 0, 3
      if (yoff(qq, s) < 0) cycle
      lcc = gvs(s) - qq
      nl = xznvc(lcc)
      call N(xz4lrows)(nl, nc, s13(s), w(kd + NW * (nvp * (vcoff(lcc) + nv * q(s)) + vpoff(r))), nvp, &
                       l2(loff(qq) + mrow * nc))
      mrow = mrow + nl
    end do
    do s = 0, 5
      if (r2qq(s) /= qq) cycle
      call N(xz4lrows)(1_8, nc, s22(s), w(jgpp + NW * (nvp * q22(s) + vpoff(r))), nvp, l2(loff(qq) + mrow * nc))
      mrow = mrow + 1
    end do
  end do
  do qq = 0, npg - 1
    if (mrows(qq) == 0) cycle
    nr = vpoff(qq + 1) - vpoff(qq)
    nc = nc22(qq)
    call N(zc_gemm)(0_4, 1_4, mrows(qq), nr, nc, one, l2(loff(qq)), nc, e22(off22(qq)), nc, zero, y(ybase(qq)), nr)
    do s = 0, 5
      if (r2qq(s) == qq) call N(xzaxpy)(nr, one, y(r2row(s)), w(kr2 + NW * (nvp * p22(s) + vpoff(qq))))
    end do
  end do
  do s = 0, 3
    if (gvs(s) < 0) cycle
    if (all(yoff(:, s) < 0)) cycle
    gj = mt - ocls(q(s))
    call N(zc_cc4_red12)(w(kr3 + NW * (bofo(gj) + (jt(s) - otoff(gj)) * ntv(s))), tri(0, vtoff(gvs(s))), ntv(s), &
                         int(nv, 4), w(jqkv), 0_4, 0_4, int(npg, 4), yoff(0, s), ynq(0, s), w(jqposv), &
                         w(jqposq), y)
  end do
  ! ---- the blocks on an index and a triple, X[e][(abc)]:
  !      R3[J] += sgn sum_e f[m][e] X,  R3[(k, P)] += sgn sgn' sum_e ~<P'||ke> X:
  !      the rows of the occupied triples of group mt - sb - p
  do p = 0, ncls - 1
    if (off13(p) < 0) cycle
    r = mt + st - p
    g3 = mt - sb - p
    if (g3 < 0 .or. g3 >= ntg) cycle
    nr = vcoff(p + 1) - vcoff(p)
    nc = nc13(p)
    n = 0
    do s = 0, 3
      if (ocls(q(s)) - sb /= p) cycle
      lk(n * nr:n * nr + nr - 1) = s13(s) * fov(vcoff(p):vcoff(p) + nr - 1, q(s))
      ptr(n + 1) = loc(w(kr3 + NW * (bofo(g3) + (jt(s) - otoff(g3)) * nc)))
      n = n + 1
    end do
    do s = 0, 5
      km = opgrp(q22(s)) - sb - p
      if (km < 0 .or. km >= ncls) cycle
      mm = q(r1(s))
      nn = q(r2(s))
      do kk = ocoff(km), ocoff(km + 1) - 1
        if (osgn(p22(s), kk) == 0) cycle
        lk(n * nr:n * nr + nr - 1) = (s22(s) * osgn(p22(s), kk)) * ooovd(vcoff(p):vcoff(p) + nr - 1, kk, nn, mm)
        ptr(n + 1) = loc(w(kr3 + NW * (bofo(g3) + (otri(p22(s), kk) - otoff(g3)) * nc)))
        n = n + 1
      end do
    end do
    if (n > 0) call N(zc_cc4_addmm)(ptr, lk, n, nr, nc, e13(off13(p)))
  end do
  tq(6) = tq(6) + omp_get_wtime() - t0
end subroutine

! ---------------------------------------------------------------------------
! EOM-CC4: the places of what the T4 terms of a sigma vector take of the
! ground state (kept with the arrays of N(xzeomkeep)).
subroutine N(xz4eomkeep)()
  use xzdata
  use xzmem
  implicit none
  integer(8) :: ov
  ov = no * nv
  e4vv = xzget(NW * nvv, 'CC4: the dressed <ab||cd>')
  e4q1 = xzget(NW * ov * nvp, 'CC4: dressed <ab||ex> [x][e][Q]')
  e4q2 = xzget(NW * no * no * ov, 'CC4: dressed <ma||ij> [i][j][m][a]')
  e4xo = xzget(NW * nto * no * nvp, 'CC4: Xo')
  e4t2px = xzget(NW * npo * nv * nv, 'CC4: T2 [P][e][a]')
end subroutine

! ---------------------------------------------------------------------------
! EOM-CC4: these arrays, from the amplitudes and the intermediates of
! N(xzeomim) (which has copied the bare <ab||cd> to e4vv before the pp
! operator took its place): the operands of N(xzt4) for the ground state.
! The copy of <ab||cd> is dressed where it is.
subroutine N(xz4eomim)(w)
  use xzdata
  use xzmem
  use xzkern
  implicit none
  real(8) :: w(*)
  integer(8) :: m0, ov, ktau, ktt, kwv, kg2, kyb, kgq, kgh, kpib, kk, kg, kkm, kb, koo, h, lo, nqh, q0, nq
  logical :: dense
  ov = no * nv
  dense = ncls == 1
  m0 = xzmark()
  ktau = xzget(NW * npo * nvp, 'CC4: tau1')
  ktt = xzget(NW * npo * nvp, 'CC4: T2 + tau1')
  kwv = xzget(NW * nvp * ov, 'CC4: dressed <ab||ei> [Q][e][i]')
  kg2 = xzget(NW * npo * ov, 'CC4: work [P][e][i]')
  kyb = xzget(NW * nba * nv * nv * no, 'CC4: batch of the dressing')
  kk = xzget(NW * ov, 'CC4: work of the dressing')
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
    call xztic()
    call N(xzt2forms)(no, nv, npo, nvp, w(jt1), w(jt2), w(jopa), w(jopb), w(jvpa), w(jvpb), &
                      w(et2p), w(et2pp), w(ktau), w(ktt))
    call N(xzqt2)(no, nv, npo, nvp, w(jt2), w(jopa), w(jopb), w(jvpa), w(jvpb), w(et2pp), w(et2p), w(e4t2px))
    call N(xzbv)(no, nv, npo, nvp, nba, w(jt1), w(ktau), w(jovvq), w(jooov), w(jovvo), &
                 w(jopa), w(jopb), w(jvpa), w(jvpb), w(kwv), w(kg2), w(kyb))
    do h = 0, npg - 1
      lo = vpoff(h)
      nqh = vpoff(h + 1) - lo
      if (nqh == 0) cycle
      if (.not. dense) call N(xzgrpcols)(no, nv, npo, nvp, lo, nqh, w(jovvq), w(jgpp), w(jvpidx), w(kgq), &
                                         w(kgh), w(kpib))
      do q0 = 0, nqh - 1, nqs
        nq = min(nqs, nqh - q0)
        call N(xzdrrows)(no, nv, npo, nvp, nqh, nq, w(jt1), w(ktau + NW * (lo + q0)), &
                         w(e4vv + NW * (vvoff(h) + q0 * nqh)), w(kgq), w(kgh), w(jvpa + lo + q0), &
                         w(jvpb + lo + q0), w(e4vv + NW * (vvoff(h) + q0 * nqh)), w(kk))
        call N(zc_pair_t1_contract)(w(kwv + NW * (lo + q0) * ov), w(e4vv + NW * (vvoff(h) + q0 * nqh)), w(jt1), nq, nv, &
                                    nqh, w(kpib), no)
      end do
    end do
    call N(xz4q12)(no, nv, nvp, w(kwv), w(eovood), w(e4q1), w(e4q2))
    call N(xz4xo)(no, nv, npo, nvp, nto, w(eooood), w(eovvod), w(et2p), w(e4t2px), w(jopa), w(jopb), &
                  w(jvpa), w(jvpb), w(jofree), w(jopair), w(e4xo), w(kg), w(kkm), w(kb), w(koo), .false.)
    call xztoc(5, 'once: CC4 operands')
  end if
  call xzrel(m0)
end subroutine

! px[P][e][a] = a2[i][j][a][e], P = (i<j)
subroutine N(xz4px)(no, nv, npo, a2, opa, opb, px)
  implicit none
  integer(8), intent(in) :: no, nv, npo
  ZT, intent(in) :: a2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)         ! [i][j][a][b]
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1)
  ZT, intent(out) :: px(0:nv-1, 0:nv-1, 0:npo-1)
  integer(8) :: p, a, e
  !$omp parallel do private(p, a, e)
  do p = 0, npo - 1
    do e = 0, nv - 1
      do a = 0, nv - 1
        px(a, e, p) = a2(e, a, opb(p), opa(p))
      end do
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The T4 terms of a sigma vector of EOM-CC4 at the frequency eomega (called
! by N(xzesig) with the vector and the derivatives of the small blocks that
! it has made): the operands of H' = [H~, R1] and of the vector,
!   - <ab||ex>' = P(ab) r1[m][a] ~<mb||ex> - sum_f ~<ab||ef> r1[x][f]     x4q1
!   <ma||ij>'                                                             x4q2
!   <mb||ef>' = - sum_n r1[n][b] <mn||ef>                                 x4d
!   <bc||ef>' = - P(bc) sum_m r1[m][b] ~<mc||ef>, for Xv                  x4vv
!   Xo of (<mn||jk>', <md||el>'; t2) + Xo of (the dressed blocks; r2)     x4xos
! then the pass over the occupied quadruples (N(xz4run)); ks2 [P][Q] and ks3
! (the layout of the shift) are added to.
subroutine N(xze4)(w, kr1, kr2, kr2p, kr3, kdfov, kdooov, kdoooo, kdovvo, kdovoo, ks2, ks3)
  use xzdata
  use xzmem
  use xzkern
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kr1, kr2, kr2p, kr3, kdfov, kdooov, kdoooo, kdovvo, kdovoo, ks2, ks3
  integer(8) :: m0, m1, ov, kdwv, kyb, kgq, kgh, kpib, kk, kg, kkm, kb, koo
  integer(8) :: h, hr, lo, lo2, nqh, nqr, x0, nb, m, ndone, nmine, s, nxvv
  logical :: dense
  ZT :: one, zero, mone
  one = 1
  zero = 0
  mone = -1
  ov = no * nv
  dense = ncls == 1
  s = eshift
  if (dense) s = 0
  ! the blocks of <bc||ef>': the pairs (bc) of group h + s, the pairs (ef) of group h
  x4vvoff = -1
  nxvv = 0
  do h = 0, npg - 1
    hr = h + s
    if (hr < 0 .or. hr >= npg) cycle
    nqh = vpoff(h + 1) - vpoff(h)
    nqr = vpoff(hr + 1) - vpoff(hr)
    if (nqh == 0 .or. nqr == 0) cycle
    x4vvoff(hr) = nxvv
    nxvv = nxvv + nqr * nqh
  end do
  m0 = xzmark()
  x4q1 = xzget(NW * ov * nvp, 'CC4: <ab||ex> derivative')
  x4q2 = xzget(NW * no * no * ov, 'CC4: <ma||ij> derivative')
  x4d = xzget(NW * ov * nvp, 'CC4: <mb||ef> derivative')
  x4xos = xzget(NW * nto * no * nvp, 'CC4: Xo of the vector')
  x4r2px = xzget(NW * npo * nv * nv, 'CC4: R2 [P][e][a]')
  x4vv = xzget(NW * nxvv, 'CC4: <ab||cd> derivative')
  m1 = xzmark()
  kdwv = xzget(NW * nvp * ov, 'CC4: <ab||ei> derivative [Q][e][i]')
  kyb = xzget(NW * nba * nv * nv * no, 'CC4: batch of the dressing')
  kk = xzget(NW * ov, 'CC4: work of the dressing')
  kgq = ed
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
    call xztic()
    ! <ab||ei>' = - P(ab) sum_m r1[m][a] ~<mb||ei> + sum_f ~<ab||ef> r1[i][f]
    call zc_vzero_d(NW * nvp * ov, w(kdwv))
    do x0 = 0, nv - 1, nba
      nb = min(nba, nv - x0)
      call N(zc_gemm)(1_4, 0_4, nb, nv * nv * no, no, one, w(kr1 + NW * x0), nv, w(eovvod), nv * nv * no, zero, &
                      w(kyb), nv * nv * no)
      call N(zc_pair_antisym_batch_add)(w(kdwv), w(kyb), nv, nvp, w(jvpa), w(jvpb), nv * no, x0, x0 + nb, mone)
    end do
    do h = 0, npg - 1
      lo = vpoff(h)
      nqh = vpoff(h + 1) - lo
      if (nqh == 0) cycle
      if (.not. dense) call N(xzgrpcols)(no, nv, npo, nvp, lo, nqh, w(ed), w(jgpp), w(jvpidx), w(kgq), w(kgh), w(kpib))
      call N(zc_pair_t1_contract)(w(kdwv + NW * lo * ov), w(e4vv + NW * vvoff(h)), w(kr1), nqh, nv, nqh, w(kpib), no)
      ! the rows of <bc||ef>' on the pairs (ef) of this group
      hr = h + s
      if (hr < 0 .or. hr >= npg) cycle
      if (x4vvoff(hr) < 0) cycle
      lo2 = vpoff(hr)
      nqr = vpoff(hr + 1) - lo2
      call N(xzedrows)(no, nv, nqh, nqr, w(kr1), w(kgq), w(jvpa + lo2), w(jvpb + lo2), w(x4vv + NW * x4vvoff(hr)), w(kk))
    end do
    call N(xz4q12)(no, nv, nvp, w(kdwv), w(kdovoo), w(x4q1), w(x4q2))
    do m = 0, no - 1
      call N(zc_gemm)(1_4, 0_4, nv, nvp, no, mone, w(kr1), nv, w(joovvp + NW * nvp * no * m), nvp, zero, &
                      w(x4d + NW * nvp * nv * m), nvp)
    end do
    call N(xz4px)(no, nv, npo, w(kr2), w(jopa), w(jopb), w(x4r2px))
    call N(xz4xo)(no, nv, npo, nvp, nto, w(kdoooo), w(kdovvo), w(et2p), w(e4t2px), w(jopa), w(jopb), &
                  w(jvpa), w(jvpb), w(jofree), w(jopair), w(x4xos), w(kg), w(kkm), w(kb), w(koo), .false.)
    call N(xz4xo)(no, nv, npo, nvp, nto, w(eooood), w(eovvod), w(kr2p), w(x4r2px), w(jopa), w(jopb), &
                  w(jvpa), w(jvpb), w(jofree), w(jopair), w(x4xos), w(kg), w(kkm), w(kb), w(koo), .true.)
    call xztoc(5, 'CC4: sigma operands')
  end if
  call xzrel(m1)
  x4s = s
  x4fov = kdfov
  x4ooov = kdooov
  x4r1 = kr1
  x4r2 = kr2
  x4r2p = kr2p
  x4r3 = kr3
  g4fov = efov
  g4ooov = eooovd
  g4d = ed
  g4q1 = e4q1
  g4q2 = e4q2
  g4t2p = et2p
  g4t2px = e4t2px
  g4xo = e4xo
  jv4 = e4vv
  call N(xz4run)(w, jt1, jt2, jt3, ks2, ks3, eomega, .true., t4eom, ndone, nmine)
  if (.not. xzdry) call xztoc(5, 'CC4: the T4 terms')
  call xzrel(m0)
end subroutine

! ---------------------------------------------------------------------------
! EOM-CC4, macro / micro: the files of a problem (number ip of the input: a
! label shift) for a later run of the same ZMAT in the run directory
! (docs/native.md, section 19.5), written by rank 0, unformatted streams:
!
!   XZE4R.nnn   the roots.  'XZE4R_01'; integer(8): active occupied, active
!               virtual, spinors, frozen occupied spinors, words per element,
!               classes of the labels, shift in classes, roots asked for,
!               elements of a vector, roots found, roots done with the T4
!               terms, shift in labels, the place in the file XZEOMR after
!               the vectors of the roots that are done; real(8): the three
!               numbers of XZKEY,
!               the CC4 correlation energy; the energies of the roots without
!               the T4 terms; of the last cluster in hand the energies, the
!               Rayleigh quotients, the energies after the first evaluation
!               (64 each); 16 numbers per root that is done (the line of the
!               table of the states); the vectors of the roots found - of
!               the roots that are done, and of the others of their clusters,
!               those with the T4 terms, of the rest the start vectors.
!               Written when the roots without the T4 terms are there and
!               after every root.
!   XZE4Q.nnn   the last evaluation of the T4 terms of the root in hand.
!               'XZE4Q_01'; integer(8): the first seven of XZE4R, elements of
!               a vector, the root, the first root of its cluster, the roots
!               of the cluster, the macro iteration, sigma vectors of the root
!               with the T4 terms and without; real(8): the numbers of XZKEY,
!               the CC4 correlation energy, the frequency of the evaluation,
!               the change of omega in the macro iteration before; q (the
!               orthonormal vectors of the cluster), F (their sigma vectors
!               with the T4 terms).  Written after every evaluation, removed
!               when the root is done.
!
! A file is taken if its integers and the numbers of XZKEY are those of this
! run and its CC4 energy is within 1e-7 Eh.  Both are written to <name>.new,
! which then takes the name.
!
! N(xz4er): rw = 'w' writes the roots (kdone of them done: the last kdone
! lines of the results of xzeomd); 'r' reads - ok: the file was taken; ee,
! nfound, kdone, eclu, rayc, shotc and the vectors at kxf from it, the lines
! of the roots that are done added to the results.
subroutine N(xz4er)(w, rw, ip, lshift, nroot, nlen, ecorr, kxf, ee, nfound, kdone, eclu, rayc, shotc, ok)
  use xzdata
  use xzkern
  use xzeomd
  implicit none
  real(8) :: w(*)
  character(len=1), intent(in) :: rw
  integer(8), intent(in) :: ip, lshift, nroot, nlen, kxf
  real(8), intent(in) :: ecorr
  complex(8) :: ee(*)
  integer(8), intent(inout) :: nfound, kdone
  real(8), intent(inout) :: eclu(64), rayc(64), shotc(64)
  logical, intent(out) :: ok
  character(len=9) :: fn
  character(len=8) :: tag
  integer(8) :: hd(13), hf(13), k, n0, ios
  real(8) :: key(3), kf3(3), ec
  real(8), allocatable :: rec(:, :)
  logical :: there
  ok = .false.
  write(fn, '(a6,i3.3)') 'XZE4R.', ip
  call xzkeyval(key)
  hd = [no, nv, ntot, nfrozen, int(NW, 8), ncls, eshift, nroot, nlen, nfound, kdone, lshift, e4upos]
  if (rw == 'w') then
    n0 = nres - kdone
    allocate(rec(16, max(kdone, 1_8)))
    do k = 1, kdone
      rec(:, k) = [rom(n0 + k), rim(n0 + k), rmx(n0 + k), rnm(n0 + k), rwt(1, n0 + k), rwt(2, n0 + k), rwt(3, n0 + k), &
                   rpp(n0 + k), rshot(n0 + k), rfirst(n0 + k), rlv(n0 + k), merge(1d0, 0d0, rconv(n0 + k)), &
                   dble(rcyc(n0 + k)), dble(rnsig(n0 + k)), dble(rli(n0 + k)), dble(rla(n0 + k))]
    end do
    open(unit=77, file=fn // '.new', form='unformatted', access='stream', status='replace', iostat=ios)
    if (ios == 0) write(77, iostat=ios) 'XZE4R_01', hd, key, ecorr, ee(1:nfound), eclu, rayc, shotc
    if (ios == 0 .and. kdone > 0) write(77, iostat=ios) rec(:, 1:kdone)
    do k = 1, nfound
      if (ios == 0) write(77, iostat=ios) w(kxf + NW * nlen * (k - 1):kxf + NW * nlen * k - 1)
    end do
    if (ios == 0) close(77, iostat=ios)
    if (ios == 0) ios = xz_rename(fn // '.new' // c_null_char, fn // c_null_char)
    if (ios /= 0) then
      write(6, '(a,a,a,i8,a)') ' @XZCCSDT-W, the file ', fn, ' could not be written (iostat ', ios, &
        '; disk full?): a later run cannot continue from here.'
      close(77, status='delete', iostat=ios)
    else
      ok = .true.
    end if
    deallocate(rec)
    return
  end if
  inquire(file=fn, exist=there)
  if (.not. there) return
  open(unit=77, file=fn, form='unformatted', access='stream', status='old', action='read', iostat=ios)
  if (ios == 0) read(77, iostat=ios) tag, hf, kf3, ec
  if (ios /= 0 .or. tag /= 'XZE4R_01' .or. any(hf(1:9) /= hd(1:9)) .or. hf(12) /= lshift &
      .or. any(abs(kf3 - key) > 1d-9 * max(1d0, abs(key))) .or. abs(ec - ecorr) > 1d-7 &
      .or. hf(10) < 1 .or. hf(10) > nroot + 2 .or. hf(11) < 0 .or. hf(11) > hf(10)) then
    close(77, iostat=ios)
    return
  end if
  allocate(rec(16, max(hf(11), 1_8)))
  read(77, iostat=ios) ee(1:hf(10)), eclu, rayc, shotc
  if (ios == 0 .and. hf(11) > 0) read(77, iostat=ios) rec(:, 1:hf(11))
  do k = 1, hf(10)
    if (ios == 0) read(77, iostat=ios) w(kxf + NW * nlen * (k - 1):kxf + NW * nlen * k - 1)
  end do
  close(77)
  if (ios /= 0) then
    deallocate(rec)
    return
  end if
  nfound = hf(10)
  kdone = hf(11)
  e4rpos = hf(13)
  do k = 1, kdone
    if (nres >= maxres) exit
    nres = nres + 1
    rshift(nres) = lshift
    rroot(nres) = k
    rom(nres) = rec(1, k)
    rim(nres) = rec(2, k)
    rmx(nres) = rec(3, k)
    rnm(nres) = rec(4, k)
    rwt(:, nres) = rec(5:7, k)
    rpp(nres) = rec(8, k)
    rshot(nres) = rec(9, k)
    rfirst(nres) = rec(10, k)
    rlv(nres) = rec(11, k)
    rconv(nres) = rec(12, k) == 1d0
    rcyc(nres) = nint(rec(13, k), 8)
    rnsig(nres) = nint(rec(14, k), 8)
    rli(nres) = nint(rec(15, k), 8)
    rla(nres) = nint(rec(16, k), 8)
  end do
  deallocate(rec)
  ok = .true.
end subroutine

! ---------------------------------------------------------------------------
! The file XZEOMR of EOM-CC4 (the format of N(xzeom): a header, then per root
! the shift in classes, the number of the root, the elements of the vector,
! omega - real and imaginary part -, the vector), written root by root when a
! root is done with the T4 terms, so that a later run goes on behind the
! roots that are done (the place is in XZE4R.nnn).
!   'o'   before the roots of a problem: the file of this run is opened, or -
!         roots of this problem were done in an earlier run - the file of
!         that run, which must be there up to their end
!   'w'   the vector at kv of the root k, its energy omega
subroutine N(xz4evec)(w, what, k, nlen, omega, kv)
  use xzdata
  use xzeomd
  implicit none
  real(8) :: w(*)
  character(len=1), intent(in) :: what
  integer(8), intent(in) :: k, nlen, kv
  real(8), intent(in) :: omega
  character(len=8) :: tag
  integer(8) :: hd(5), fsize, ios
  logical :: there, bad
  if (what == 'o') then
    if (e4rst .and. e4kdone > 0) then
      if (e4ures == 0) then
        inquire(file='XZEOMR', exist=there, size=fsize)
        bad = .not. there .or. e4rpos <= 0
        if (.not. bad) then
          e4ures = 78
          open(unit=e4ures, file='XZEOMR', form='unformatted', access='stream', status='old', iostat=ios)
          if (ios == 0) read(e4ures, iostat=ios) tag, hd
          bad = ios /= 0
          if (.not. bad) bad = tag /= 'XZEOMR01' .or. hd(1) /= no .or. hd(2) /= nv .or. hd(3) /= NW .or. hd(4) /= ncls &
                               .or. fsize < e4rpos - 1
          if (bad) close(e4ures, iostat=ios)
        end if
        if (bad) then
          write(6, '(a)') ' @XZCCSDT-W, the file XZEOMR of the earlier run, with the vectors of the roots that are done, is not there:'
          write(6, '(a)') '             no file XZEOMR is written.'
          e4ures = 0
          e4uwant = .false.
          return
        end if
      end if
      e4upos = e4rpos
    else if (e4ures == 0) then
      e4ures = 78
      open(unit=e4ures, file='XZEOMR', form='unformatted', access='stream', status='replace', iostat=ios)
      if (ios == 0) write(e4ures, iostat=ios) 'XZEOMR01', no, nv, int(NW, 8), ncls, labstep
      if (ios == 0) inquire(unit=e4ures, pos=e4upos)
      if (ios /= 0) then
        write(6, '(a)') ' @XZCCSDT-W, the file XZEOMR of the vectors cannot be written: no vectors are kept.'
        e4ures = 0
        e4uwant = .false.
      end if
    end if
    return
  end if
  write(e4ures, pos=e4upos, iostat=ios) eshift, k, nlen, omega, 0d0, w(kv:kv + NW * nlen - 1)
  if (ios == 0) flush(e4ures, iostat=ios)
  if (ios /= 0) then
    write(6, '(/,a,i8,a)') ' @XZCCSDT-F, writing the file XZEOMR failed (iostat ', ios, '): disk full?'
    call xzerrex
  end if
  inquire(unit=e4ures, pos=e4upos)
end subroutine

! N(xz4eq): rw = 'w' writes the evaluation of the macro iteration `outer' of
! the root k (the first of its cluster: ca; nc roots) - q and F from kq, kf
! (more than one rank: the rows of all ranks through the vector at kxa);
! 'r' reads one of this root - ok: taken; outer, omega, dwprev, nev, nsigr
! from it, q and F into their places, q also into the vectors of the cluster
! at kxc (the start of the eigenproblem; the same span); 'd' removes the file.
subroutine N(xz4eq)(w, rw, ip, nlen, ecorr, k, ca, nc, outer, omega, dwprev, nev, nsigr, kxa, kxb, kq, kf, kxc, ok)
  use xzdata
  use xzkern
  use xzeomd
  use omp_lib
  implicit none
  real(8) :: w(*)
  character(len=1), intent(in) :: rw
  integer(8), intent(in) :: ip, nlen, k, ca, nc, kxa, kxb, kq, kf, kxc
  real(8), intent(in) :: ecorr
  integer(8), intent(inout) :: outer, nev, nsigr
  real(8), intent(inout) :: omega, dwprev
  logical, intent(out) :: ok
  character(len=9) :: fn
  character(len=8) :: tag
  integer(8) :: hd(14), hf(14), c, ios, fsize, kv, base
  integer :: kind
  real(8) :: key(3), kf3(3), st(3), t0
  logical :: there, cmd0
  ok = .false.
  write(fn, '(a6,i3.3)') 'XZE4Q.', ip
  if (rw == 'd') then
    open(unit=77, file=fn, status='old', iostat=ios)
    if (ios == 0) close(77, status='delete', iostat=ios)
    return
  end if
  t0 = omp_get_wtime()
  call xzkeyval(key)
  hd = [no, nv, ntot, nfrozen, int(NW, 8), ncls, eshift, nlen, k, ca, nc, outer, nev, nsigr]
  ! (the commands to the ranks as the caller has them when this returns)
  cmd0 = ecmd
  if (edist) then
    ecmd = .true.
    evoff = 0
    evlen = no * nv + npo * nvp + esoff(1)
    ekx = [kxa, kxb]
  end if
  if (rw == 'w') then
    open(unit=77, file=fn // '.new', form='unformatted', access='stream', status='replace', iostat=ios)
    if (ios == 0) write(77, iostat=ios) 'XZE4Q_01', hd, key, ecorr, omega, dwprev
    do kind = 3, 4
      base = merge(kq, kf, kind == 3)
      do c = 1, nc
        kv = base + NW * nlen * (c - 1)
        if (edist) then
          ! (the rows of the other ranks)
          kv = kxa
          call N(xzevget)(w, kind, c, kxa)
          call N(xzepcollect)(w, kxa)
        end if
        if (ios == 0) write(77, iostat=ios) w(kv:kv + NW * nlen - 1)
      end do
    end do
    if (ios == 0) close(77, iostat=ios)
    if (ios == 0) ios = xz_rename(fn // '.new' // c_null_char, fn // c_null_char)
    if (ios /= 0) then
      write(6, '(a,a,a,i8,a)') ' @XZCCSDT-W, the file ', fn, ' could not be written (iostat ', ios, &
        '; disk full?): a later run cannot continue from here.'
      close(77, status='delete', iostat=ios)
    else
      ok = .true.
      write(6, '(a,a,a,f10.1,a,f8.2,a)') '          (checkpoint: q, F and the frequency of this evaluation written to the file ', fn, &
        ', ', 16d-6 * NW * nlen * nc, ' MB, ', omp_get_wtime() - t0, ' s)'
    end if
    ecmd = cmd0
    return
  end if
  inquire(file=fn, exist=there, size=fsize)
  if (there) then
    open(unit=77, file=fn, form='unformatted', access='stream', status='old', action='read', iostat=ios)
    if (ios == 0) read(77, iostat=ios) tag, hf, kf3, st
    there = ios == 0 .and. tag == 'XZE4Q_01' .and. all(hf(1:11) == hd(1:11)) .and. hf(12) >= 1 &
            .and. all(abs(kf3 - key) <= 1d-9 * max(1d0, abs(key))) .and. abs(st(1) - ecorr) <= 1d-7 &
            .and. fsize == 8 * (1 + 14 + 3 + 3 + 2 * nc * NW * nlen)
    if (.not. there) close(77, iostat=ios)
  end if
  if (.not. there) then
    ecmd = cmd0
    return
  end if
  do kind = 3, 4
    base = merge(kq, kf, kind == 3)
    do c = 1, nc
      kv = base + NW * nlen * (c - 1)
      if (edist) kv = kxa
      read(77, iostat=ios) w(kv:kv + NW * nlen - 1)
      if (ios /= 0) then
        write(6, '(/,a,a,a,i8)') ' @XZCCSDT-F, reading the file ', fn, ' failed, iostat ', ios
        call xzerrex
      end if
      if (kind == 3) call N(xzecopy)(nlen, w(kv), w(kxc + NW * nlen * (c - 1)))
      if (edist) then
        ! (to all ranks, every rank its rows)
        call N(xzepspread)(w, kxa)
        call N(xzevput)(w, kind, c, kxa)
      end if
    end do
  end do
  close(77)
  outer = hf(12)
  nev = hf(13)
  nsigr = hf(14)
  omega = st(2)
  dwprev = st(3)
  ecmd = cmd0
  ok = .true.
end subroutine

! ---------------------------------------------------------------------------
! EOM-CC4 for the problem of one label shift (called by N(xzeom) after its
! solver): the folded eigenproblem
!   [A_PP + dA(omega)] R = omega R,   dA(omega) = A_PQ (omega - A_QQ)^-1 A_QP + (the T4 terms with H'),
! in the space of singles, doubles and triples (docs/ccn.md, section 4), root
! by root, started from the roots ee and vectors (at kxf, nlen elements
! each) of A_PP - the matrix of EOM-CCSDT at the CC4 amplitudes.
!
! Macro / micro (the default): a sigma vector with the T4 terms costs as
! much as several evaluations of the T4 terms of the ground state, one
! without them as much as two CCSDT iterations.  So the T4 terms are
! evaluated once per macro iteration, on the present vector q of the root at
! its present frequency (F = [A_PP + dA(omega)] q), and held on that vector
! while the eigenproblem of
!   B x = A_PP (x - q (q|x)) + F (q|x)
! - the folded matrix on q, A_PP beside it - is solved with sigma vectors
! of EOM-CCSDT; its root and vector are the next omega and q.  The error of
! B on the eigenvector is dA times the part of it outside q, so the macro
! iterations converge like the square of the change.  Roots of the start
! closer than option foldwin form a cluster: q is an orthonormal basis of
! their vectors, evaluated together at the frequency of the root.
! Converged when omega changes by less than omegatol - or when the last two
! changes say that the next will (their ratio times the last).
!
! Option cc4eom davidson: the solver on the folded matrix at a fixed omega
! with every sigma vector evaluated in full, omega by the secant step, as
! N(xzeom3) does for CC3 (the check of the macro / micro route).
!
! Per root also: the Rayleigh quotient of the start vector with the folded
! matrix at its own energy, and omega after the first evaluation.
subroutine N(xzeom4f)(w, ip, lshift, nroot, nfound, ee, kxf, kxa, kxb, singles, nlen, ecorr)
  use xzdata
  use xzmem
  use xzkern
  use xzeomd
  use xzpar, only: nrank
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: ip, lshift, nroot, nfound, kxf, kxa, kxb, singles(*), nlen
  complex(8), intent(in) :: ee(*)
  real(8), intent(in) :: ecorr
  integer(8) :: ostart, nres0, nev0, nfw, kdw
  logical :: ckpt, resume, held, fok
  integer(8) :: msf, msub, peak, avail, k, nsig, nit, nf1, outer, nsigr, i, a, r, ca, cb, nc, jc, kxc, ncmax
  integer(8) :: kq, kf, c, d, nev, m0, nloc
  integer(8), allocatable :: c1(:), l1(:, :)
  complex(8), allocatable :: e1(:), evs(:)
  real(8), allocatable :: x1mx(:), x1nm(:), w1(:, :)
  ZT, allocatable :: v1(:, :), gm(:, :), ym(:, :), dd(:)
  real(8) :: omega, lam, gap, tol, otol, prevo, prevl, slope, nxt, t0, tr, tsg0, cwin, shot, ray, dw, dwprev, kap, rel, t4s
  real(8) :: eclu(64), rayc(64), shotc(64)
  logical :: haveprev, ok, davidson
  ZT :: s
  character(len=32) :: arg
  integer :: larg, ios
  real(8), external :: N(xzeimag)
  real(8), parameter :: ev = 27.211386245988d0, cm = 219474.6313632d0
  integer(8), parameter :: maxouter = 30

  otol = ethr / 100d0
  call x2c_getopt('xzccsdt', 'omegatol', arg, larg)
  if (larg > 0) then
    read(arg(1:larg), *, iostat=ios) otol
    if (ios /= 0) otol = ethr / 100d0
  end if
  cwin = 5d-4
  call x2c_getopt('xzccsdt', 'foldwin', arg, larg)
  if (larg > 0) then
    read(arg(1:larg), *, iostat=ios) cwin
    if (ios /= 0) cwin = 5d-4
  end if
  call x2c_getopt('xzccsdt', 'cc4eom', arg, larg)
  davidson = larg > 0 .and. (arg(1:1) == 'd' .or. arg(1:1) == 'D')
  ! (macro / micro: the files XZE4R.nnn, XZE4Q.nnn for a later run)
  ckpt = .not. davidson
  ! the largest cluster
  ncmax = 1
  nc = 1
  do k = 2, nfound
    if (dble(ee(k)) - dble(ee(k - 1)) < cwin) then
      nc = nc + 1
    else
      nc = 1
    end if
    ncmax = max(ncmax, nc)
  end do
  ! (the records of a cluster - XZE4R.nnn - hold 64 roots)
  if (ncmax > 64) then
    write(6, '(/,a,i6,a)') ' @XZCCSDT-F, EOM-CC4: a cluster of ', ncmax, ' nearly degenerate roots; at most 64.'
    call xzerrex
  end if
  msf = max(12_8, (8 + 4 * nroot) / 2)
  m0 = xzmark()
  kq = 0
  kf = 0
  ! (more than one rank: every rank holds its rows of q and F - rank 0 with
  ! the singles and doubles - as of the vectors of the subspace)
  nloc = nlen
  if (edist) nloc = no * nv + npo * nvp + esoff(1)
  if (.not. davidson) then
    kq = xzget(NW * ncmax * nloc, 'EOM-CC4: the vectors of a root')
    kf = xzget(NW * ncmax * nloc, 'EOM-CC4: their sigma vectors with the T4 terms')
    e4kq = kq
    e4kf = kf
    if (edist) call xzecmd(op_qopen, dble(ncmax))
  end if
  allocate(e1(nfound), x1mx(nfound), x1nm(nfound), w1(3, nfound), c1(nfound), l1(nlead, nfound), v1(nlead, nfound), &
           gm(ncmax, ncmax), ym(ncmax, ncmax), evs(ncmax), dd(ncmax))
  efold = .true.
  call xzdry_begin()
  call N(xzesig)(w, kxa, kxb, .false.)
  call xzdry_end(peak)
  efold = .false.
  avail = xzsize - xztop - (peak - xztop)
  write(6, '(/,a)') '   EOM-EE-CC4: the quadruples folded in at the frequency of every root (the roots above are the start)'
  if (davidson) then
    write(6, '(a)') '   option cc4eom davidson: every sigma vector with the T4 terms'
  else
    write(6, '(a)') '   macro / micro: the T4 terms evaluated on the vector of a root, held while the eigenproblem is solved'
    write(6, '(a)') '   with the sigma vectors of EOM-CCSDT'
  end if
  if (edist) write(6, '(a,i4,a)') '   over ', nrank, ' ranks: every rank the occupied quadruples of its tasks in the T4 terms of a sigma' // &
    ' vector, its batches in the rest'
  write(6, '(a,es9.2,a,es9.2)') '   convergence: the largest element of a residual below ', ethr, &
    ', the change of omega (|lambda(omega) - omega|) below ', otol
  write(6, '(a,f10.1,a,f10.1,a)') '   a sigma vector with the T4 terms takes ', 8d-6 * (peak - xztop), &
    ' MB of the work array; free besides it: ', 8d-6 * avail, ' MB'
  if (davidson) then
    write(6, '(/,a)') '   root  step         omega / Eh       lambda(omega) / Eh  |lambda-omega|   max |r|   sigma vectors      time/s'
  else
    write(6, '(/,a)') '   root macro         omega / Eh         new omega / Eh        change      max |r|   sigma vectors: T4, others' // &
      '   time/s: T4, all'
  end if
  ! the roots of an earlier run (N(xzeom) has read the file XZE4R.nnn), or
  ! the roots without the T4 terms to that file
  eclu = 0d0
  rayc = 0d0
  shotc = 0d0
  nres0 = nres
  if (e4rst) then
    eclu = e4eclu
    rayc = e4rayc
    shotc = e4shotc
    nres0 = nres - e4kdone
  else if (ckpt) then
    nfw = nfound
    kdw = 0
  end if
  ! the file XZEOMR: of this run, or - roots of this problem done before - that of the earlier run
  if (e4uwant) call N(xz4evec)(w, 'o', 0_8, 0_8, 0d0, kxf)
  if (ckpt .and. .not. e4rst) call N(xz4er)(w, 'w', ip, lshift, nroot, nlen, ecorr, kxf, ee, nfw, kdw, eclu, rayc, shotc, fok)
  ca = 1
  cb = 0
  nc = 1
  do k = 1, min(nroot, nfound)
    tr = omp_get_wtime()
    tsg0 = teomsig
    ! the cluster of the root (N(xzeom3)): its vectors, in their place
    if (k > cb) then
      ca = k
      cb = k
      do while (cb < nfound)
        if (dble(ee(cb + 1)) - dble(ee(cb)) >= cwin) exit
        cb = cb + 1
      end do
      nc = cb - ca + 1
      if (nc > 1) write(6, '(a,i4,a,i4,a,es9.2,a)') '   (roots', ca, ' to', cb, ' of the start closer than ', &
        cwin, ' Eh: solved together, by their order)'
    end if
    kxc = kxf + NW * nlen * (ca - 1)
    jc = k - ca + 1
    if (k <= e4kdone) then
      write(6, '(a,i4,a,f20.12,a,i3.3,a,a)') '   root', k, ': omega ', rom(nres0 + k), ' Eh   (done in an earlier run: the file XZE4R.', &
        ip, ')', trim(merge('               ', '  NOT converged', rconv(nres0 + k)))
      cycle
    end if
    omega = dble(ee(k))
    ! (a later root of a cluster: from the last solution of the cluster)
    if (k > ca .and. .not. davidson) omega = eclu(jc)
    gap = 1d0
    haveprev = .false.
    ok = .false.
    nsigr = 0
    nev = 0
    prevo = 0d0
    prevl = 0d0
    shot = 0d0
    ray = 0d0
    dwprev = 0d0
    ! (an evaluation of the T4 terms of this root of an earlier run, the file
    ! XZE4Q.nnn: q, F, the frequency - the iterations go on with its eigenproblem)
    ostart = 1
    resume = .false.
    if (ckpt) call N(xz4eq)(w, 'r', ip, nlen, ecorr, k, ca, nc, ostart, omega, dwprev, nev, nsigr, kxa, kxb, kq, kf, kxc, resume)
    nev0 = nev
    do outer = ostart, maxouter
      t0 = omp_get_wtime()
      eomega = omega
      msub = min(64_8, max(msf, 3 * nc + 4))
      if (davidson) then
        tol = min(1d-4, max(ethr, 0.1d0 * gap))
        nfollow = nc
        efold = .true.
        if (edist) call xzecmd(op_fold, 1d0, eomega)
        evfin = .true.
        evfout = .true.
        call N(xzesolve)(w, .false., nc, tol, emaxcyc, msub, nc, kxc, singles, kxa, kxb, avail, &
                         e1, x1mx, x1nm, w1, c1, nf1, nsig, nit, kxc, 0, l1, v1, .true.)
        evfin = .false.
        evfout = .false.
        efold = .false.
        if (edist) call xzecmd(op_fold, 0d0, 0d0)
        nfollow = 0
        nsigr = nsigr + nsig
        nev = nev + nsig
        if (nf1 < nc) then
          write(6, '(a,i4,a)') ' @XZCCSDT-W, EOM-CC4: the solver at a frequency found fewer roots than its cluster has (', &
            nf1, '): the root is left.'
          exit
        end if
        lam = dble(e1(jc))
        if (outer == 1) then
          ray = eritz0(jc)
          shot = lam
        end if
        gap = abs(lam - omega)
        write(6, '(i7,i6,2f22.12,2es12.2,i8,f14.2)') k, outer, omega, lam, gap, x1mx(jc), nsig, omp_get_wtime() - t0
        flush(6)
        if (gap < otol .and. tol <= ethr .and. maxval(x1mx(1:nc)) <= tol) then
          omega = lam
          ok = .true.
          exit
        end if
        nxt = lam
        if (haveprev .and. abs(omega - prevo) > 1d-14) then
          slope = (lam - prevl) / (omega - prevo)
          if (abs(slope) < 0.5d0) nxt = omega + (lam - omega) / (1d0 - slope)
        end if
        prevo = omega
        prevl = lam
        haveprev = .true.
        omega = nxt
        cycle
      end if
      ! ---- macro / micro.  q: an orthonormal basis of the vectors of the
      !      cluster; F = (A_PP + dA(omega)) q: the sigma vectors with the T4
      !      terms.  More than one rank: the vector of rank 0 to all, then every
      !      rank on its rows (the commands of the solver, xzeoms.inc), the T4
      !      terms dealt as those of the ground state.  held: q and F of this
      !      step are those of an earlier run.
      held = resume .and. outer == ostart
      t4s = 0d0
      if (edist) then
        ecmd = .true.
        evoff = 0
        evlen = nloc
        ekx = [kxa, kxb]
      end if
      if (held) then
        write(6, '(a,i3,a,i3.3,a,f18.12,a)') '          (the T4 terms of step ', outer, ' of this root: from the file XZE4Q.', ip, &
          ' of an earlier run, evaluated at omega ', omega, ')'
      else if (edist) then
        do c = 1, nc
          call N(xzecopy)(nlen, w(kxc + NW * nlen * (c - 1)), w(kxa))
          call N(xzepspread)(w, kxa)
          if (c > 1) then
            call N(xzevmdot)(w, 3, c - 1, kxa, dd)
            dd(1:c - 1) = -CONJ(dd(1:c - 1))
            call N(xzevcomb)(w, 3, c - 1, dd, kxa, .true.)
          end if
          call N(xzepdot)(w, kxa, kxa, .false., s)
          call N(xzepscal)(w, kxa, 1d0 / sqrt(dble(s)))
          call N(xzevput)(w, 3, c, kxa)
        end do
        t4s = omp_get_wtime()
        efold = .true.
        call xzecmd(op_fold, 1d0, eomega)
        do c = 1, nc
          call N(xzevget)(w, 3, c, kxa)
          call N(xzepsig)(w, kxa, kxb, .false.)
          call N(xzevput)(w, 4, c, kxb)
        end do
        efold = .false.
        call xzecmd(op_fold, 0d0, 0d0)
        t4s = omp_get_wtime() - t4s
      else
        do c = 1, nc
          call N(xzecopy)(nlen, w(kxc + NW * nlen * (c - 1)), w(kq + NW * nlen * (c - 1)))
          do d = 1, c - 1
            call N(zc_vdot)(nlen, w(kq + NW * nlen * (d - 1)), w(kq + NW * nlen * (c - 1)), s)
            call N(xzeaxpyc)(nlen, -s, w(kq + NW * nlen * (d - 1)), w(kq + NW * nlen * (c - 1)))
          end do
          call N(zc_vdot)(nlen, w(kq + NW * nlen * (c - 1)), w(kq + NW * nlen * (c - 1)), s)
          call N(xzevscal)(nlen, 1d0 / sqrt(dble(s)), w(kq + NW * nlen * (c - 1)))
        end do
        t4s = omp_get_wtime()
        efold = .true.
        do c = 1, nc
          call N(xzesig)(w, kq + NW * nlen * (c - 1), kf + NW * nlen * (c - 1), .false.)
        end do
        efold = .false.
        t4s = omp_get_wtime() - t4s
      end if
      if (.not. held) then
        teomsig = teomsig + t4s
        neomsig = neomsig + nc
        nev = nev + nc
        nt4run = nt4run + nc
        ! the checkpoint: q, F and the frequency of this evaluation
        if (ckpt) call N(xz4eq)(w, 'w', ip, nlen, ecorr, k, ca, nc, outer, omega, dwprev, nev, nsigr, kxa, kxb, kq, kf, kxc, fok)
        if (nt4stop > 0 .and. nt4run >= nt4stop) then
          write(6, '(/,a,i4,a)') ' @XZCCSDT-F, option cc4stop: the run stops after its evaluation ', nt4run, ' of the T4 terms.'
          call xzerrex
        end if
      end if
      if (outer == 1) then
        ! the folded matrix on the start vectors: its eigenvalues (one root: the Rayleigh quotient)
        do c = 1, nc
          if (edist) then
            call N(xzevget)(w, 4, c, kxb)
            call N(xzevmdot)(w, 3, nc, kxb, dd)
            gm(1:nc, c) = CONJ(dd(1:nc))
          else
            do d = 1, nc
              call N(zc_vdot)(nlen, w(kq + NW * nlen * (d - 1)), w(kf + NW * nlen * (c - 1)), s)
              gm(d, c) = s
            end do
          end if
        end do
        call N(xzesmall)(nc, gm, ncmax, evs, ym)
      end if
      ecmd = .false.
      ! the eigenproblem with the T4 terms held on q
      e4nq = nc
      e4kq = kq
      e4kf = kf
      nfollow = nc
      evfin = .true.
      evfout = .true.
      call N(xzesolve)(w, .false., nc, ethr, emaxcyc, msub, nc, kxc, singles, kxa, kxb, avail, &
                       e1, x1mx, x1nm, w1, c1, nf1, nsig, nit, kxc, 0, l1, v1, .true.)
      evfin = .false.
      evfout = .false.
      nfollow = 0
      e4nq = 0
      nsigr = nsigr + nsig
      if (nf1 < nc) then
        write(6, '(a,i4,a)') ' @XZCCSDT-W, EOM-CC4: the solver found fewer roots than the cluster has (', nf1, '): the root is left.'
        exit
      end if
      lam = dble(e1(jc))
      eclu(1:nc) = dble(e1(1:nc))
      if (outer == 1 .and. k == ca) then
        rayc(1:nc) = dble(evs(1:nc))
        shotc(1:nc) = dble(e1(1:nc))
      end if
      ray = rayc(jc)
      shot = shotc(jc)
      dw = abs(lam - omega)
      write(6, '(i7,i6,2f22.12,2es12.2,i10,i8,2f12.2)') k, outer, omega, lam, dw, x1mx(jc), nc, nsig, t4s, omp_get_wtime() - t0
      flush(6)
      omega = lam
      kap = 1d0
      if (outer > 1 .and. dwprev > 0d0) kap = dw / dwprev
      if (maxval(x1mx(1:nc)) <= ethr .and. (dw < otol .or. (kap < 0.5d0 .and. kap * dw < otol))) then
        ok = .true.
        if (dw >= otol) write(6, '(a,es9.2,a)') '          (the next evaluation would change omega by about ', kap * dw, &
          ': the ratio of the last two changes)'
        exit
      end if
      dwprev = dw
    end do
    write(6, '(a,i4,a,f20.12,a,f12.6,a,f14.4,a,i4,a,i5,a,i6,a,f10.2,a)') '   root', k, ': omega ', omega, ' Eh = ', &
      omega * ev, ' eV = ', omega * cm, ' cm-1   (', min(outer, maxouter), ' steps, ', nev, ' sigma vectors with the T4 terms, ', &
      nsigr - merge(nev, 0_8, davidson), ' others, ', omp_get_wtime() - tr, ' s)'
    if (resume) write(6, '(a,i4,a,i4,a)') '          (', nev0, ' of the sigma vectors with the T4 terms in earlier runs, ', nev - nev0, &
      ' in this run; the time is that of this run)'
    write(6, '(a,f18.12,a,f18.12,a,f18.12,a)') '          without the T4 terms ', dble(ee(k)), &
      ';  Rayleigh quotient of the start vector ', ray, ';  after the first step ', shot, ' Eh'
    if (.not. ok) write(6, '(a)') '          NOT converged'
    if (nres < maxres) then
      nres = nres + 1
      rshift(nres) = lshift
      rroot(nres) = k
      rom(nres) = omega
      rim(nres) = aimag(e1(jc))
      rmx(nres) = x1mx(jc)
      rnm(nres) = x1nm(jc)
      rwt(:, nres) = w1(:, jc)
      rconv(nres) = ok
      rcyc(nres) = min(outer, maxouter)
      rnsig(nres) = nev
      rpp(nres) = dble(ee(k))
      rshot(nres) = ray
      rfirst(nres) = shot
      rli(nres) = 0
      rla(nres) = 0
      rlv(nres) = 0d0
      if (l1(1, jc) >= 0) then
        rli(nres) = nfrozen + operm(l1(1, jc) / nv) + 1
        rla(nres) = nfrozen + no + vperm(mod(l1(1, jc), nv)) + 1
        rlv(nres) = abs(v1(1, jc))
      end if
    end if
    do r = 1, nlead
      if (l1(r, jc) < 0) cycle
      if (abs(v1(r, jc)) < 0.05d0 .and. r > 1) cycle
      i = l1(r, jc) / nv
      a = mod(l1(r, jc), nv)
      write(6, '(a,i4,a,i5,a,i4,a,f12.6,a,i5,a,i4,a,f12.6,a,2f10.6)') '   root', k, ': ', &
        nfrozen + operm(i) + 1, ' (', lab(nfrozen + operm(i)), ',', w(jeps + i), ')  -> ', &
        nfrozen + no + vperm(a) + 1, ' (', lab(nfrozen + no + vperm(a)), ',', w(jeps + no + a), ')   ', &
        dble(v1(r, jc)), N(xzeimag)(v1(r, jc))
    end do
    ! the root is done: its vector to the file XZEOMR;
    if (e4uwant) call N(xz4evec)(w, 'w', k, nlen, omega, kxc + NW * nlen * (jc - 1))
    ! the roots and the vectors to the file XZE4R.nnn, the file of its evaluation goes
    if (ckpt) then
      nfw = nfound
      kdw = k
      call N(xz4er)(w, 'w', ip, lshift, nroot, nlen, ecorr, kxf, ee, nfw, kdw, eclu, rayc, shotc, fok)
      call N(xz4eq)(w, 'd', ip, nlen, ecorr, k, ca, nc, outer, omega, dwprev, nev, nsigr, kxa, kxb, kq, kf, kxc, fok)
    end if
  end do
  deallocate(e1, x1mx, x1nm, w1, c1, l1, v1, gm, ym, evs, dd)
  call xzrel(m0)
end subroutine
