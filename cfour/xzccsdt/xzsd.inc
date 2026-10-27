!
! The CCSD step (CALC=CCSD) in the blocks of the symmetry labels
! (type-generic).  xzstep takes these routines in place of those of xzimd.inc
! when there is no T3: the same equations, the same T1-dressed intermediates,
! but every sum runs over the indices that the labels allow, and the
! products of pairs (i, a) are GEMMs of the blocks of one delta s(a) - s(i)
! (xzovlist).  What CCSDT needs of the intermediates and CCSD does not (the
! dressed <ab||ej> as W_vvvo, the dressed <mb||ij>, the operands of pass 1)
! is not made.  With one class (no labels) the blocks are the whole arrays.
!
! The spinors are in the order of their classes, the pairs in the order of
! their groups (xzdata.F90).  t1 conserves the labels: t1[i][a] = 0 unless
! s(a) = s(i), so a sum over a virtual index with t1[i][e] runs over the
! class of i, and so on.  An array that is "blocked" below has only its
! allowed elements set (the others are not read); the others are complete
! (their forbidden elements zero).
!

! ---------------------------------------------------------------------------
! [mlo(a), mhi(a)): the occupied spinors of the class of the virtual a.
subroutine N(xzvclsrange)(nv, mlo, mhi)
  use xzdata, only: ocls, vcls, ocoff, vcoff, opoff, vpoff, ncls, npg, dlo, dhi, xzw, xzcan, xzovlist
  implicit none
  integer(8), intent(in) :: nv
  integer(8), intent(out) :: mlo(0:nv-1), mhi(0:nv-1)
  integer(8) :: a
  do a = 0, nv - 1
    mlo(a) = ocoff(vcls(a))
    mhi(a) = ocoff(vcls(a) + 1)
  end do
end subroutine

! ---------------------------------------------------------------------------
! The columns of <my||Q> and <mn||Q> of the pairs Q of the group h, as
! xzgrpcols, but of <my||Q> only the rows (m, y) with s(m) + s(y) = h (the
! others are zero, and the dressing of the rows of <ab||cd>,
! zc_vvvv_dress_cls, reads no other).
subroutine N(xzgrpcols1)(no, nv, npo, nvp, h, lo, nqh, gq, gpp, vpidx, gqh, gph, pib)
  use xzdata, only: ocls, vcls, ocoff, vcoff, opoff, vpoff, ncls, npg, dlo, dhi, xzw, xzcan, xzovlist
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp, h, lo, nqh
  ZT, intent(in) :: gq(0:nvp-1, 0:no*nv-1), gpp(0:nvp-1, 0:npo-1)
  integer(8), intent(in) :: vpidx(0:nv*nv-1)
  ZT, intent(out) :: gqh(0:nqh-1, 0:no*nv-1), gph(0:nqh-1, 0:npo-1)
  integer(8), intent(out) :: pib(0:nv*nv-1)
  integer(8) :: k, m, y
  !$omp parallel do private(k, m, y)
  do k = 0, no * nv - 1
    m = k / nv
    y = k - m * nv
    if (xzw(ocls(m) + vcls(y)) /= h) cycle
    gqh(:, k) = gq(lo:lo + nqh - 1, k)
  end do
  do k = 0, npo - 1
    gph(:, k) = gpp(lo:lo + nqh - 1, k)
  end do
  do k = 0, nv * nv - 1
    pib(k) = -1
    if (vpidx(k) >= lo .and. vpidx(k) < lo + nqh) pib(k) = vpidx(k) - lo
  end do
end subroutine

! ---------------------------------------------------------------------------
! Rows of the T1-dressed <ab||de> of one group (xzdrrows), the sums over m
! limited to the classes of a and b (mlo, mhi): the same numbers.
subroutine N(xzdrrows1)(no, nv, npo, nvp, nqh, nq, t1, tau, vv, gq, gh, pa, pb, mlo, mhi, slab, tm)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp, nqh, nq
  ZT, intent(in) :: t1(0:nv-1, 0:no-1)
  ZT, intent(in) :: tau(*), vv(*), gq(*), gh(*)
  integer(8), intent(in) :: pa(*), pb(*), mlo(*), mhi(*)
  ZT :: slab(*)
  ZT :: tm(0:nv-1, 0:no-1)
  ZT :: one
  one = 1
  tm = -t1
  call N(zc_vvvv_dress_cls)(slab, vv, nq, nqh, pa, pb, tm, nv, gq, nv, t1, nv, gq, nv, mlo, mhi)
  if (npo > 0) call N(zc_gemm)(1_4, 0_4, nq, nqh, npo, one, tau, nvp, gh, nqh, one, slab, nqh)
end subroutine

! ---------------------------------------------------------------------------
! Dressed <mb||ef> on e<f (xzdrovvq), blocked: D[m][b][Q] for the pairs Q of
! the group s(m) + s(b) only.
subroutine N(xzsdrovvq)(no, nv, nvp, t1, gq, oovvp, d)
  use xzdata, only: ocls, vcls, ocoff, vcoff, opoff, vpoff, ncls, npg, dlo, dhi, xzw, xzcan, xzovlist
  implicit none
  integer(8), intent(in) :: no, nv, nvp
  ZT, intent(in) :: t1(0:nv-1, 0:no-1)
  ZT, intent(in) :: gq(0:nvp-1, 0:nv-1, 0:no-1)                ! [m][y][Q]
  ZT, intent(in) :: oovvp(0:nvp-1, 0:no-1, 0:no-1)             ! [m][n][Q]
  ZT, intent(out) :: d(0:nvp-1, 0:nv-1, 0:no-1)
  integer(8) :: m, b, n, h, lo, hi
  !$omp parallel do collapse(2) private(m, b, n, h, lo, hi) schedule(dynamic, 16)
  do m = 0, no - 1
    do b = 0, nv - 1
      h = xzw(ocls(m) + vcls(b))
      lo = vpoff(h)
      hi = vpoff(h + 1) - 1
      if (hi < lo) cycle
      d(lo:hi, b, m) = gq(lo:hi, b, m)
      do n = ocoff(vcls(b)), ocoff(vcls(b) + 1) - 1
        d(lo:hi, b, m) = d(lo:hi, b, m) - t1(b, n) * oovvp(lo:hi, n, m)
      end do
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! Dressed <mb||ej> (xzdrov without <mb||ij>, which CCSD does not need):
!   <mb||ej> + sum_f ~<mb||ef> t1[j][f] + sum_n t1[n][b] <mn||je>
! d blocked (xzsdrovvq); ovvod complete.
subroutine N(xzsdrov)(no, nv, nvp, t1, ooov, ovvo, d, vpa, vpb, ovvod)
  use xzdata, only: ocls, vcls, ocoff, vcoff, opoff, vpoff, ncls, npg, dlo, dhi, xzw, xzcan, xzovlist
  implicit none
  integer(8), intent(in) :: no, nv, nvp
  ZT, intent(in) :: t1(0:nv-1, 0:no-1)
  ZT, intent(in) :: ooov(0:nv-1, 0:no-1, 0:no-1, 0:no-1)       ! [m][n][i][e]
  ZT, intent(in) :: ovvo(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)       ! [m][b][e][j]
  ZT, intent(in) :: d(0:nvp-1, 0:nv-1, 0:no-1)                 ! dressed [m][b][Q]
  integer(8), intent(in) :: vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(out) :: ovvod(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)     ! [m][b][e][j]
  integer(8) :: m, b, e, f, j, n, h, q
  ZT :: s, x
  !$omp parallel do collapse(2) private(m, b, e, f, j, n, h, q, s, x) schedule(dynamic, 16)
  do m = 0, no - 1
    do b = 0, nv - 1
      do e = 0, nv - 1
        do j = 0, no - 1
          s = ovvo(j, e, b, m)
          do n = ocoff(vcls(b)), ocoff(vcls(b) + 1) - 1
            s = s + t1(b, n) * ooov(e, j, n, m)
          end do
          ovvod(j, e, b, m) = s
        end do
      end do
      h = xzw(ocls(m) + vcls(b))
      do q = vpoff(h), vpoff(h + 1) - 1
        e = vpa(q)
        f = vpb(q)
        x = d(q, b, m)
        do j = ocoff(vcls(f)), ocoff(vcls(f) + 1) - 1
          ovvod(j, e, b, m) = ovvod(j, e, b, m) + x * t1(f, j)
        end do
        do j = ocoff(vcls(e)), ocoff(vcls(e) + 1) - 1
          ovvod(j, f, b, m) = ovvod(j, f, b, m) - x * t1(e, j)
        end do
      end do
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The vovv term of R1 (the columns no + a of r1v[i][y]; the others zero):
!   r1v[i][no + a] = - sum_{k,Q} t2[i][k][Q] D[k][a][Q]
! one GEMM per (k, group of Q): i and a of the class (group - s(k)).
subroutine N(xzsdr1v)(no, nv, nvp, t2p, d, r1v)
  use xzdata, only: ocls, vcls, ocoff, vcoff, opoff, vpoff, ncls, npg, dlo, dhi, xzw, xzcan, xzovlist
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, nvp
  ZT, intent(in) :: t2p(0:nvp-1, 0:no-1, 0:no-1)               ! [i][k][Q]
  ZT, intent(in) :: d(0:nvp-1, 0:nv-1, 0:no-1)                 ! blocked [k][a][Q]
  ZT, intent(out) :: r1v(0:no+nv-1, 0:no-1)                    ! [i][y]
  integer(8) :: k, h, c, ni, na, nq
  ZT :: one, mone
  one = 1
  mone = -1
  r1v = 0
  do k = 0, no - 1
    do h = 0, npg - 1
      c = xzw(h - ocls(k))
      if (c < 0 .or. c >= ncls) cycle
      ni = ocoff(c + 1) - ocoff(c)
      na = vcoff(c + 1) - vcoff(c)
      nq = vpoff(h + 1) - vpoff(h)
      if (ni == 0 .or. na == 0 .or. nq == 0) cycle
      call N(zc_gemm)(0_4, 1_4, ni, na, nq, mone, t2p(vpoff(h), k, ocoff(c)), no * nvp, &
                      d(vpoff(h), vcoff(c), k), nvp, one, r1v(no + vcoff(c), ocoff(c)), no + nv)
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The dressed <ab||ij> on i<j, a<b without its sum_ef ~<ab||ef> t1 t1 part
! (xzbv and xzvvoo without W_vvvo): with bv[Q][e][j] the bra-dressed
! <ab||ej> of xzbv, r2[P][Q] = <ij||ab>* - P(ab) sum_m t1[m][a] <ij||mb>*
! + sum_K <K||P> tau1[K][Q] + P(ij) sum_e t1[i][e] bv[Q][e][j], the last term
! taken apart into its three parts:
!   - P(ij) sum_e t1[i][e] <je||Q>*
!   + sum_K tau1[K][Q] (G[K][i][j] - G[K][j][i]),  G[K][i][j] = - sum_e t1[i][e] <K||je>
!   - P(ab) sum_m t1[m][a] (H[m][b][i][j] - H[m][b][j][i]),  H[m][b][i][j] = sum_e t1[i][e] <mb||ej>
! r2 complete (zero outside the diagonal blocks of the groups).
!   wk   work [K][P];  hh  work [m][b][i][j];  ee  work [m][b][P]
subroutine N(xzsdvvoo)(no, nv, npo, nvp, t1, tau, gpp, gq, ooov, oooo, ovvo, opa, opb, vpa, vpb, &
                       r2, wk, hh, ee)
  use xzdata, only: ocls, vcls, ocoff, vcoff, opoff, vpoff, ncls, npg, dlo, dhi, xzw, xzcan, xzovlist
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: t1(0:nv-1, 0:no-1)
  ZT, intent(in) :: tau(0:nvp-1, 0:npo-1), gpp(0:nvp-1, 0:npo-1)
  ZT, intent(in) :: gq(0:nvp-1, 0:nv-1, 0:no-1)                ! [i][e][Q]
  ZT, intent(in) :: ooov(0:nv-1, 0:no-1, 0:no-1, 0:no-1)       ! [m][n][i][e]
  ZT, intent(in) :: oooo(0:no-1, 0:no-1, 0:no-1, 0:no-1)       ! [m][n][i][j]
  ZT, intent(in) :: ovvo(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)       ! [m][b][e][j]
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(out) :: r2(0:nvp-1, 0:npo-1)
  ZT :: wk(0:npo-1, 0:npo-1), hh(0:no-1, 0:no-1, 0:nv-1, 0:no-1), ee(0:npo-1, 0:nv-1, 0:no-1)
  integer(8) :: m, b, i, j, e, p, kk, k1, k2, h, plo, nph, lo, nqh, q, a
  ZT :: s, one
  one = 1
  !$omp parallel do collapse(2) private(m, b, i, j, e, s)
  do m = 0, no - 1
    do b = 0, nv - 1
      do i = 0, no - 1
        do j = 0, no - 1
          s = 0
          do e = vcoff(ocls(i)), vcoff(ocls(i) + 1) - 1
            s = s + t1(e, i) * ovvo(j, e, b, m)
          end do
          hh(j, i, b, m) = s
        end do
      end do
    end do
  end do
  !$omp parallel do collapse(2) private(m, b, p, i, j)
  do m = 0, no - 1
    do b = 0, nv - 1
      do p = 0, npo - 1
        i = opa(p)
        j = opb(p)
        ee(p, b, m) = CONJ(ooov(b, m, j, i)) + hh(j, i, b, m) - hh(i, j, b, m)
      end do
    end do
  end do
  !$omp parallel do private(kk, k1, k2, p, i, j, e, s)
  do kk = 0, npo - 1
    k1 = opa(kk)
    k2 = opb(kk)
    do p = 0, npo - 1
      i = opa(p)
      j = opb(p)
      s = oooo(j, i, k2, k1)
      do e = vcoff(ocls(i)), vcoff(ocls(i) + 1) - 1
        s = s - t1(e, i) * ooov(e, j, k2, k1)
      end do
      do e = vcoff(ocls(j)), vcoff(ocls(j) + 1) - 1
        s = s + t1(e, j) * ooov(e, i, k2, k1)
      end do
      wk(p, kk) = s
    end do
  end do
  call zc_vzero_d(NW * npo * nvp, r2)
  do h = 0, npg - 1
    plo = opoff(h)
    nph = opoff(h + 1) - plo
    lo = vpoff(h)
    nqh = vpoff(h + 1) - lo
    if (nph == 0 .or. nqh == 0) cycle
    !$omp parallel do collapse(2) private(p, q, i, j, a, b, m, e, s)
    do p = plo, plo + nph - 1
      do q = lo, lo + nqh - 1
        i = opa(p)
        j = opb(p)
        a = vpa(q)
        b = vpb(q)
        s = CONJ(gpp(q, p))
        do m = ocoff(vcls(a)), ocoff(vcls(a) + 1) - 1
          s = s - t1(a, m) * ee(p, b, m)
        end do
        do m = ocoff(vcls(b)), ocoff(vcls(b) + 1) - 1
          s = s + t1(b, m) * ee(p, a, m)
        end do
        do e = vcoff(ocls(i)), vcoff(ocls(i) + 1) - 1
          s = s - t1(e, i) * CONJ(gq(q, e, j))
        end do
        do e = vcoff(ocls(j)), vcoff(ocls(j) + 1) - 1
          s = s + t1(e, j) * CONJ(gq(q, e, i))
        end do
        r2(q, p) = s
      end do
    end do
    call N(zc_gemm)(1_4, 0_4, nph, nqh, nph, one, wk(plo, plo), npo, tau(lo, plo), nvp, one, r2(lo, plo), nvp)
  end do
end subroutine

! ---------------------------------------------------------------------------
! W_ovvo, F_vv and F_oo (xzweff), blocked:
!   weff[m][a][d][i] = ~<ma||di> + sum_ne <mn||de> t2[i][n][a][e]
!     ((m, d) and (i, a) of one delta D, (n, e) of -D: one GEMM per delta)
!   fevv[a][d] = f[a][d] - 1/2 sum_mnf t2[m][n][a][f] <mn||df>   (s(a) = s(d))
!   feoo[m][i] = f[m][i] + sum_{n,Q} <mn||Q> t2[i][n][Q]          (Q of the group s(m) + s(n))
! weff, fevv, feoo complete.  a2, b2, x2: work (no nv)**2 each.
subroutine N(xzsdweff)(no, nv, nvp, t2, t2p, oovv, oovvp, ovvod, foo, fvv, weff, fevv, feoo, a2, b2, x2)
  use xzdata, only: ocls, vcls, ocoff, vcoff, opoff, vpoff, ncls, npg, dlo, dhi, xzw, xzcan, xzovlist
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, nvp
  ZT, intent(in) :: t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)         ! [i][j][a][b]
  ZT, intent(in) :: t2p(0:nvp-1, 0:no-1, 0:no-1)               ! [i][j][Q]
  ZT, intent(in) :: oovv(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)       ! [m][n][e][f]
  ZT, intent(in) :: oovvp(0:nvp-1, 0:no-1, 0:no-1)             ! [m][n][Q]
  ZT, intent(in) :: ovvod(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)      ! [m][b][e][j]
  ZT, intent(in) :: foo(0:no-1, 0:no-1), fvv(0:nv-1, 0:nv-1)
  ZT, intent(out) :: weff(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)      ! [m][a][d][i]
  ZT, intent(out) :: fevv(0:nv-1, 0:nv-1)                      ! [a][d]
  ZT, intent(out) :: feoo(0:no-1, 0:no-1)                      ! [m][i]
  ZT :: a2(*), b2(*), x2(*)
  integer(8), allocatable :: lx(:), ly(:), loff(:)
  integer(8) :: dd, nd, n1, n2, o1, o2, r, c, m, n, i, q, h, c0, na
  ZT :: s, one, zero, mhalf
  one = 1
  zero = 0
  mhalf = -0.5d0
  allocate(lx(0:no*nv-1), ly(0:no*nv-1), loff(dlo:dhi+1))
  call xzovlist(lx, ly, loff)
  call N(xzsdcopy)(no * no * nv * nv, ovvod, weff)
  do dd = dlo, dhi
    if (.not. xzcan(dd)) cycle
    nd = xzw(-dd)
    o1 = loff(dd)
    n1 = loff(dd + 1) - o1
    o2 = loff(nd)
    n2 = loff(nd + 1) - o2
    if (n1 == 0 .or. n2 == 0) cycle
    ! a2[(m, d)][(n, e)] = <mn||de>, b2[(n, e)][(a, i)] = t2[i][n][a][e]
    !$omp parallel do private(r, c)
    do r = 0, n1 - 1
      do c = 0, n2 - 1
        a2(1 + c + n2 * r) = oovv(ly(o2 + c), ly(o1 + r), lx(o2 + c), lx(o1 + r))
      end do
    end do
    !$omp parallel do private(r, c)
    do r = 0, n2 - 1
      do c = 0, n1 - 1
        b2(1 + c + n1 * r) = t2(ly(o2 + r), ly(o1 + c), lx(o2 + r), lx(o1 + c))
      end do
    end do
    call N(zc_gemm)(0_4, 0_4, n1, n1, n2, one, a2, n2, b2, n1, zero, x2, n1)
    ! weff[m][a][d][i] += x2[(m, d)][(a, i)]
    !$omp parallel do private(r, c)
    do r = 0, n1 - 1
      do c = 0, n1 - 1
        weff(lx(o1 + c), ly(o1 + r), ly(o1 + c), lx(o1 + r)) = weff(lx(o1 + c), ly(o1 + r), ly(o1 + c), lx(o1 + r)) &
                                                              + x2(1 + c + n1 * r)
      end do
    end do
  end do
  deallocate(lx, ly, loff)
  fevv = fvv
  do c = 0, ncls - 1
    c0 = vcoff(c)
    na = vcoff(c + 1) - c0
    if (na == 0) cycle
    call N(zc_gemm)(1_4, 0_4, na, na, no * no * nv, mhalf, t2(c0, 0, 0, 0), nv, oovv(c0, 0, 0, 0), nv, &
                    one, fevv(c0, c0), nv)
  end do
  !$omp parallel do private(m, i, n, q, h, s)
  do m = 0, no - 1
    do i = 0, no - 1
      s = 0
      do n = 0, no - 1
        h = xzw(ocls(m) + ocls(n))
        do q = vpoff(h), vpoff(h + 1) - 1
          s = s + oovvp(q, n, m) * t2p(q, n, i)
        end do
      end do
      feoo(i, m) = foo(i, m) + s
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! n elements from src to dst (they do not overlap), on all threads.
subroutine N(xzsdcopy)(n, src, dst)
  implicit none
  integer(8), intent(in) :: n
  ZT, intent(in) :: src(n)
  ZT, intent(out) :: dst(n)
  integer(8) :: i
  !$omp parallel do
  do i = 1, n
    dst(i) = src(i)
  end do
end subroutine

! ---------------------------------------------------------------------------
! The CCSD part of the T1 and T2 equations and the Jacobi step (xzccsd
! without T3), blocked:
!   R1 = f_ai + f_kc t_ikac + 1/2 g_akcd t_ikcd - 1/2 g_klic t_klac
!   R2 = g_abij + 1/2 g_abcd (t + tau1)_ijcd + 1/2 W_klij t_klab
!        + P(ab) F_bc t_ijac - P(ij) F_kj t_ikab + P(ij)P(ab) W_kbcj t_ikac
! On entry r2[P][Q] has the dressed <ab||ij>, lad[P][Q] the ladder with the
! dressed <ab||cd>, r1v[i][no + a] the vovv term of R1 (xzsdr1v).  The ring
! term: x2[(i, a)][(j, b)] = sum_kc t2[i][k][a][c] W[k][b][c][j] couples
! (i, a) of delta D with (k, c) and (j, b) of -D: one GEMM per delta into
! x2 (only the coupled elements are set; the others are not read).
!   wkl   work [K][P];  a2, b2, c2, x2: work (no nv)**2 each
subroutine N(xzsdccsd)(no, nv, npo, nvp, eps, t1, t2, t2p, t2pp, gpp, ooovd, ooood, ovvod, weff, &
                       fov, fvo, fevv, feoo, r1v, lad, opa, opb, vpa, vpb, &
                       r2, wkl, t1n, t2n, a2, b2, c2, x2)
  use xzdata, only: ocls, vcls, ocoff, vcoff, opoff, vpoff, ncls, npg, dlo, dhi, xzw, xzcan, xzovlist
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  real(8), intent(in) :: eps(0:no+nv-1)
  ZT, intent(in) :: t1(0:nv-1, 0:no-1)
  ZT, intent(in) :: t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)         ! [i][j][a][b]
  ZT, intent(in) :: t2p(0:nvp-1, 0:no-1, 0:no-1)               ! [i][j][Q]
  ZT, intent(in) :: t2pp(0:nvp-1, 0:npo-1), gpp(0:nvp-1, 0:npo-1)
  ZT, intent(in) :: ooovd(0:nv-1, 0:no-1, 0:no-1, 0:no-1)      ! [m][n][i][e]
  ZT, intent(in) :: ooood(0:no-1, 0:no-1, 0:no-1, 0:no-1)      ! [m][n][i][j]
  ZT, intent(in) :: ovvod(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)      ! [k][b][c][j]
  ZT, intent(in) :: weff(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)
  ZT, intent(in) :: fov(0:nv-1, 0:no-1), fvo(0:no-1, 0:nv-1)
  ZT, intent(in) :: fevv(0:nv-1, 0:nv-1), feoo(0:no-1, 0:no-1)
  ZT, intent(in) :: r1v(0:no+nv-1, 0:no-1)                     ! [i][y]: the vovv term at y = no + a
  ZT, intent(in) :: lad(0:nvp-1, 0:npo-1)
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(inout) :: r2(0:nvp-1, 0:npo-1)
  ZT :: wkl(0:npo-1, 0:npo-1)                                  ! [K][P]
  ZT, intent(out) :: t1n(0:nv-1, 0:no-1)
  ZT, intent(out) :: t2n(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT :: a2(*), b2(*), c2(*)
  ZT :: x2(0:nv-1, 0:no-1, 0:nv-1, 0:no-1)                     ! [(i, a)][(j, b)]
  integer(8), allocatable :: lx(:), ly(:), loff(:)
  integer(8) :: i, j, k, a, b, c, p, q, kk, m, n, cc, h, plo, nph, lo, nqh
  integer(8) :: dd, nd, n1, n2, o1, o2, r, cl
  ZT :: s, one, zero, v
  real(8) :: dn
  one = 1
  zero = 0

  ! ---- T1
  !$omp parallel do collapse(2) private(i, a, k, c, m, n, cc, s)
  do i = 0, no - 1
    do a = 0, nv - 1
      s = fvo(i, a) + r1v(no + a, i)
      do k = 0, no - 1
        do c = vcoff(ocls(k)), vcoff(ocls(k) + 1) - 1
          s = s + fov(c, k) * t2(c, a, k, i)
        end do
      end do
      do m = 0, no - 1
        do n = 0, no - 1
          cc = xzw(ocls(m) + ocls(n) - vcls(a))
          if (cc < 0 .or. cc >= ncls) cycle
          do c = vcoff(cc), vcoff(cc + 1) - 1
            s = s - 0.5d0 * ooovd(c, i, n, m) * t2(c, a, n, m)
          end do
        end do
      end do
      t1n(a, i) = t1(a, i) + s / (eps(i) - eps(no + a))
    end do
  end do

  ! ---- T2: the ladder, W_klij and the hh term (blocks of the groups)
  !$omp parallel do private(p)
  do p = 0, npo - 1
    r2(:, p) = r2(:, p) + lad(:, p)
  end do
  !$omp parallel do private(kk, p)
  do kk = 0, npo - 1
    do p = 0, npo - 1
      wkl(p, kk) = ooood(opb(p), opa(p), opb(kk), opa(kk))
    end do
  end do
  do h = 0, npg - 1
    plo = opoff(h)
    nph = opoff(h + 1) - plo
    lo = vpoff(h)
    nqh = vpoff(h + 1) - lo
    if (nph == 0 .or. nqh == 0) cycle
    call N(zc_gemm)(0_4, 1_4, nph, nph, nqh, one, gpp(lo, plo), nvp, t2pp(lo, plo), nvp, one, wkl(plo, plo), npo)
    call N(zc_gemm)(1_4, 0_4, nph, nqh, nph, one, wkl(plo, plo), npo, t2pp(lo, plo), nvp, one, r2(lo, plo), nvp)
  end do

  ! ---- the ring term, by delta
  allocate(lx(0:no*nv-1), ly(0:no*nv-1), loff(dlo:dhi+1))
  call xzovlist(lx, ly, loff)
  do dd = dlo, dhi
    if (.not. xzcan(dd)) cycle
    nd = xzw(-dd)
    o1 = loff(dd)
    n1 = loff(dd + 1) - o1
    o2 = loff(nd)
    n2 = loff(nd + 1) - o2
    if (n1 == 0 .or. n2 == 0) cycle
    ! a2[(i, a)][(k, c)] = t2[i][k][a][c], b2[(k, c)][(j, b)] = (~<kb||cj> + weff[k][b][c][j]) / 2
    !$omp parallel do private(r, cl)
    do r = 0, n1 - 1
      do cl = 0, n2 - 1
        a2(1 + cl + n2 * r) = t2(ly(o2 + cl), ly(o1 + r), lx(o2 + cl), lx(o1 + r))
      end do
    end do
    !$omp parallel do private(r, cl)
    do r = 0, n2 - 1
      do cl = 0, n2 - 1
        b2(1 + cl + n2 * r) = 0.5d0 * (ovvod(lx(o2 + cl), ly(o2 + r), ly(o2 + cl), lx(o2 + r)) &
                                       + weff(lx(o2 + cl), ly(o2 + r), ly(o2 + cl), lx(o2 + r)))
      end do
    end do
    call N(zc_gemm)(0_4, 0_4, n1, n2, n2, one, a2, n2, b2, n2, zero, c2, n2)
    !$omp parallel do private(r, cl)
    do r = 0, n1 - 1
      do cl = 0, n2 - 1
        x2(ly(o2 + cl), lx(o2 + cl), ly(o1 + r), lx(o1 + r)) = c2(1 + cl + n2 * r)
      end do
    end do
  end do
  deallocate(lx, ly, loff)

  ! ---- P(ab) F_bc t_ijac - P(ij) F_kj t_ikab and the ring term
  do h = 0, npg - 1
    plo = opoff(h)
    nph = opoff(h + 1) - plo
    lo = vpoff(h)
    nqh = vpoff(h + 1) - lo
    if (nph == 0 .or. nqh == 0) cycle
    !$omp parallel do collapse(2) private(p, q, i, j, a, b, c, k, s)
    do p = plo, plo + nph - 1
      do q = lo, lo + nqh - 1
        i = opa(p)
        j = opb(p)
        a = vpa(q)
        b = vpb(q)
        s = 0
        do c = vcoff(vcls(b)), vcoff(vcls(b) + 1) - 1
          s = s + t2(c, a, j, i) * fevv(c, b)
        end do
        do c = vcoff(vcls(a)), vcoff(vcls(a) + 1) - 1
          s = s - t2(c, b, j, i) * fevv(c, a)
        end do
        do k = ocoff(ocls(j)), ocoff(ocls(j) + 1) - 1
          s = s - t2p(q, k, i) * feoo(j, k)
        end do
        do k = ocoff(ocls(i)), ocoff(ocls(i) + 1) - 1
          s = s + t2p(q, k, j) * feoo(i, k)
        end do
        r2(q, p) = r2(q, p) + s + x2(b, j, a, i) - x2(b, i, a, j) - x2(a, j, b, i) + x2(a, i, b, j)
      end do
    end do
  end do

  ! ---- the Jacobi step of T2
  call N(xzsdcopy)(no * no * nv * nv, t2, t2n)
  !$omp parallel do private(p, q, i, j, a, b, v, dn)
  do p = 0, npo - 1
    i = opa(p)
    j = opb(p)
    do q = 0, nvp - 1
      a = vpa(q)
      b = vpb(q)
      dn = eps(i) + eps(j) - eps(no + a) - eps(no + b)
      v = r2(q, p) / dn
      t2n(b, a, j, i) = t2n(b, a, j, i) + v
      t2n(a, b, j, i) = t2n(a, b, j, i) - v
      t2n(b, a, i, j) = t2n(b, a, i, j) - v
      t2n(a, b, i, j) = t2n(a, b, i, j) + v
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The T1 parts of the CCSD ladder, which xzstep makes with the bare <ab||cd>
! (lad[P][q] = sum_Q <q||Q> tt[P][Q], tt = t2 + tau1): with the dressed
! ~<ab||Q> = <ab||Q> - sum_m t1[m][a] <mb||Q> + sum_m t1[m][b] <ma||Q>
! + sum_K tau1[K][ab] <K||Q>,
!   lad[P][(ab)] += - sum_m t1[m][a] z[P][m][b] + sum_m t1[m][b] z[P][m][a]
!                   + sum_K tau1[K][(ab)] y[K][P]
!   z[P][m][b] = sum_Q <mb||Q> tt[P][Q],  y[K][P] = sum_Q <K||Q> tt[P][Q]
! in the blocks of the groups (z blocked: s(m) + s(b) the group of P).
subroutine N(xzsdladt1)(no, nv, npo, nvp, t1, tau, tt, gq, gpp, opa, opb, vpa, vpb, lad, z, y)
  use xzdata, only: ocls, vcls, ocoff, vcoff, opoff, vpoff, ncls, npg, dlo, dhi, xzw, xzcan, xzovlist
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: t1(0:nv-1, 0:no-1)
  ZT, intent(in) :: tau(0:nvp-1, 0:npo-1), tt(0:nvp-1, 0:npo-1), gpp(0:nvp-1, 0:npo-1)
  ZT, intent(in) :: gq(0:nvp-1, 0:nv-1, 0:no-1)                ! [m][b][Q]
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(inout) :: lad(0:nvp-1, 0:npo-1)
  ZT :: z(0:nv-1, 0:no-1, 0:npo-1), y(0:npo-1, 0:npo-1)
  integer(8) :: h, plo, nph, lo, nqh, m, c, p, q, a, b
  ZT :: one, zero, s
  one = 1
  zero = 0
  do h = 0, npg - 1
    plo = opoff(h)
    nph = opoff(h + 1) - plo
    lo = vpoff(h)
    nqh = vpoff(h + 1) - lo
    if (nph == 0 .or. nqh == 0) cycle
    do m = 0, no - 1
      c = xzw(h - ocls(m))
      if (c < 0 .or. c >= ncls) cycle
      if (vcoff(c + 1) == vcoff(c)) cycle
      call N(zc_gemm)(0_4, 1_4, nph, vcoff(c + 1) - vcoff(c), nqh, one, tt(lo, plo), nvp, &
                      gq(lo, vcoff(c), m), nvp, zero, z(vcoff(c), m, plo), no * nv)
    end do
    call N(zc_gemm)(0_4, 1_4, nph, nph, nqh, one, gpp(lo, plo), nvp, tt(lo, plo), nvp, zero, y(plo, plo), npo)
    !$omp parallel do collapse(2) private(p, q, a, b, m, s)
    do p = plo, plo + nph - 1
      do q = lo, lo + nqh - 1
        a = vpa(q)
        b = vpb(q)
        s = 0
        do m = ocoff(vcls(a)), ocoff(vcls(a) + 1) - 1
          s = s - t1(a, m) * z(b, m, p)
        end do
        do m = ocoff(vcls(b)), ocoff(vcls(b) + 1) - 1
          s = s + t1(b, m) * z(a, m, p)
        end do
        lad(q, p) = lad(q, p) + s
      end do
    end do
    call N(zc_gemm)(1_4, 0_4, nph, nqh, nph, one, y(plo, plo), npo, tau(lo, plo), nvp, one, lad(lo, plo), nvp)
  end do
end subroutine
