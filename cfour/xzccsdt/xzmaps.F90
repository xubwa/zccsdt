!
! Index tables of the pairs and triples of n indices, as the kernels of
! csrc/ccsdt_t3.c and csrc/t3pack.c take them (see xzdata.F90), and the
! layout of T3 in the blocks of the symmetry labels.
!
integer(8) function xztrank(i, j, k, n)
  ! the number of the triple i<j<k of 0..n-1 in lexicographic order
  implicit none
  integer(8), intent(in) :: i, j, k, n
  integer(8) :: c3, c2, x
  c3(x) = x * (x - 1) * (x - 2) / 6
  c2(x) = x * (x - 1) / 2
  xztrank = c3(n) - c3(n - i) + c2(n - i - 1) - c2(n - j) + (k - j - 1)
end function xztrank

! the number of the pair p<q of 0..n-1 in lexicographic order
integer(8) function xzpidx(p, q, n)
  implicit none
  integer(8), intent(in) :: p, q, n
  xzpidx = p * n - p * (p + 1) / 2 + (q - p - 1)
end function xzpidx

! ---------------------------------------------------------------------------
! The tables of n indices with the classes cls (not decreasing; one class:
! all zero).  Pairs and triples are numbered by the sum of the classes of
! their indices (their group) first, lexicographic inside a group.
!   poff(g), toff(g)     the first pair / triple of group g (npg + 1, ntg + 1 numbers)
!   pgrp(P)              the group of a pair
!   tnew                 work: nt integers
subroutine xzmaps(n, np, nt, nc, cls, tri, sgn, free, pair, pa, pb, pidx, pgrp, poff, toff, tnew)
  ! (the groups as in xzsetmode: npg, ntg of them, sums of classes xzw)
  use xzdata, only: xzw, npg, ntg, maxgrp
  implicit none
  integer(8), intent(in) :: n, np, nt, nc
  integer(8), intent(in) :: cls(0:n-1)
  integer(8), intent(out) :: tri(0:np-1, 0:n-1)       ! [x][P]
  integer(1), intent(out) :: sgn(0:np-1, 0:n-1)       ! [x][P]
  integer(8), intent(out) :: free(0:nt-1, 0:2)        ! [r][I]
  integer(8), intent(out) :: pair(0:nt-1, 0:2)        ! [r][I]
  integer(8), intent(out) :: pa(0:np-1), pb(0:np-1)
  integer(8), intent(out) :: pidx(0:n-1, 0:n-1)       ! [p][q]
  integer(8), intent(out) :: pgrp(0:np-1)
  integer(8), intent(out) :: poff(0:npg), toff(0:ntg)
  integer(8) :: tnew(0:nt-1)
  integer(8) :: i, j, k, x, p, it, g, l
  integer(8) :: cnt(0:maxgrp), nxt(0:maxgrp)
  integer(8), external :: xztrank

  ! pairs
  cnt = 0
  do i = 0, n - 1
    do j = i + 1, n - 1
      g = xzw(cls(i) + cls(j))
      cnt(g) = cnt(g) + 1
    end do
  end do
  poff(0) = 0
  do g = 0, npg - 1
    poff(g + 1) = poff(g) + cnt(g)
  end do
  nxt(0:npg-1) = poff(0:npg-1)
  pidx = -1
  do i = 0, n - 1
    do j = i + 1, n - 1
      g = xzw(cls(i) + cls(j))
      p = nxt(g)
      nxt(g) = p + 1
      pa(p) = i
      pb(p) = j
      pgrp(p) = g
      pidx(j, i) = p
    end do
  end do
  ! triples
  cnt = 0
  do i = 0, n - 1
    do j = i + 1, n - 1
      do k = j + 1, n - 1
        g = xzw(cls(i) + cls(j) + cls(k))
        cnt(g) = cnt(g) + 1
      end do
    end do
  end do
  toff(0) = 0
  do g = 0, ntg - 1
    toff(g + 1) = toff(g) + cnt(g)
  end do
  nxt(0:ntg-1) = toff(0:ntg-1)
  l = 0
  do i = 0, n - 1
    do j = i + 1, n - 1
      do k = j + 1, n - 1
        g = xzw(cls(i) + cls(j) + cls(k))
        it = nxt(g)
        nxt(g) = it + 1
        tnew(l) = it
        l = l + 1
        free(it, 0) = i
        free(it, 1) = j
        free(it, 2) = k
        pair(it, 0) = pidx(k, j)
        pair(it, 1) = pidx(k, i)
        pair(it, 2) = pidx(j, i)
      end do
    end do
  end do
  do x = 0, n - 1
    do p = 0, np - 1
      j = pa(p)
      k = pb(p)
      if (x == j .or. x == k) then
        tri(p, x) = 0
        sgn(p, x) = 0
      else if (x < j) then
        tri(p, x) = tnew(xztrank(x, j, k, n))
        sgn(p, x) = 1
      else if (x < k) then
        tri(p, x) = tnew(xztrank(j, x, k, n))
        sgn(p, x) = -1
      else
        tri(p, x) = tnew(xztrank(j, k, x, n))
        sgn(p, x) = 1
      end if
    end do
  end do
end subroutine xzmaps

! ---------------------------------------------------------------------------
! The layout of T3 in the blocks of the groups of triples: block g is a
! matrix [occupied triples of g][virtual triples of g] at boff(g).
!   obase[I], vcol[A]    T3[I][A] = t3(obase[I] + vcol[A])
!   osig, vsig           signs (all one here; the kernels take them)
!   vci[d][Q]            vcol of the triple of d and the pair Q (int32)
!   rowpos[a][i]         the place of (a, i) in the list of the pairs with
!                        the same s(a) - s(i): the classes of a in their
!                        order, a slowest, i fastest
subroutine xzt3lay(no, nv, npo, nvp, nto, nvt, nc, ocls, vcls, otoff, vtoff, vtri, vsgn, ocoff, vcoff, &
                   boff, n3, obase, vcol, osig, vsig, vci, rowpos)
  ! (the groups as in xzsetmode: ntg of them; the differences dlo .. dhi, xzw)
  use xzdata, only: xzw, ntg, dlo, dhi
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp, nto, nvt, nc
  integer(8), intent(in) :: ocls(0:no-1), vcls(0:nv-1)
  integer(8), intent(in) :: otoff(0:ntg), vtoff(0:ntg)
  integer(8), intent(in) :: vtri(0:nvp-1, 0:nv-1)
  integer(1), intent(in) :: vsgn(0:nvp-1, 0:nv-1)
  integer(8), intent(in) :: ocoff(0:nc), vcoff(0:nc)
  integer(8), intent(out) :: boff(0:ntg), n3
  integer(8), intent(out) :: obase(0:nto-1), vcol(0:nvt-1)
  real(8), intent(out) :: osig(0:nto-1), vsig(0:nvt-1)
  integer(4), intent(out) :: vci(0:nvp-1, 0:nv-1)
  integer(8), intent(out) :: rowpos(0:no-1, 0:nv-1)   ! [a][i]
  integer(8) :: g, it, a, i, d, q, delta, sa, si, pos, na
  n3 = 0
  do g = 0, ntg - 1
    boff(g) = n3
    na = vtoff(g + 1) - vtoff(g)
    do it = otoff(g), otoff(g + 1) - 1
      obase(it) = n3 + (it - otoff(g)) * na
    end do
    do a = vtoff(g), vtoff(g + 1) - 1
      vcol(a) = a - vtoff(g)
    end do
    n3 = n3 + (otoff(g + 1) - otoff(g)) * na
  end do
  boff(ntg) = n3
  osig = 1d0
  vsig = 1d0
  do d = 0, nv - 1
    do q = 0, nvp - 1
      vci(q, d) = 0
      if (vsgn(q, d) /= 0) vci(q, d) = int(vcol(vtri(q, d)), 4)
    end do
  end do
  rowpos = -1
  do delta = dlo, dhi
    pos = 0
    do sa = 0, nc - 1
      si = xzw(sa - delta)
      if (si < 0 .or. si >= nc) cycle
      do a = vcoff(sa), vcoff(sa + 1) - 1
        do i = ocoff(si), ocoff(si + 1) - 1
          rowpos(i, a) = pos
          pos = pos + 1
        end do
      end do
    end do
  end do
end subroutine xzt3lay

! e3(I) = sum of e over the indices of the triple I
subroutine xzsum3(n, nt, free, e, e3)
  implicit none
  integer(8), intent(in) :: n, nt
  integer(8), intent(in) :: free(0:nt-1, 0:2)
  real(8), intent(in) :: e(0:n-1)
  real(8), intent(out) :: e3(0:nt-1)
  integer(8) :: it
  do it = 0, nt - 1
    e3(it) = e(free(it, 0)) + e(free(it, 1)) + e(free(it, 2))
  end do
end subroutine xzsum3

subroutine xziota(n, rows)
  implicit none
  integer(8), intent(in) :: n
  integer(8), intent(out) :: rows(0:n-1)
  integer(8) :: i
  do i = 0, n - 1
    rows(i) = i
  end do
end subroutine xziota

! ---------------------------------------------------------------------------
! The classes of the labels of the active spinors and the order of the
! program: class = (label - smallest label) / step, step the greatest common
! divisor of the differences; the occupied and the virtual spinors each in
! the order of their classes, CFOUR's order inside a class.
!   lab      labels of the active spinors, CFOUR's order (occupied first)
!   perm     CFOUR's number of the spinor at a place here (occupied: 0..no-1,
!            virtual: 0..nv-1)
!   nc = 0 on return: more classes than the tables hold
subroutine xzclasses(no, nv, lab, maxcls, nc, operm, vperm, ocls, vcls, ocoff, vcoff)
  implicit none
  integer(8), intent(in) :: no, nv
  integer(8), intent(in) :: lab(0:no+nv-1)
  integer, intent(in) :: maxcls
  integer(8), intent(out) :: nc
  integer(8), intent(out) :: operm(0:no-1), vperm(0:nv-1), ocls(0:no-1), vcls(0:nv-1)
  integer(8), intent(out) :: ocoff(0:maxcls), vcoff(0:maxcls)
  integer(8) :: lmin, step, p, d, a, b, t, s, k
  lmin = minval(lab)
  step = 0
  do p = 0, no + nv - 1
    a = lab(p) - lmin
    b = step
    do while (b /= 0)
      t = mod(a, b)
      a = b
      b = t
    end do
    step = a
  end do
  if (step == 0) step = 1
  d = (maxval(lab) - lmin) / step + 1
  nc = d
  if (d > maxcls) then
    nc = 0
    return
  end if
  k = 0
  ocoff = 0
  do s = 0, nc - 1
    ocoff(s) = k
    do p = 0, no - 1
      if ((lab(p) - lmin) / step == s) then
        operm(k) = p
        ocls(k) = s
        k = k + 1
      end if
    end do
  end do
  ocoff(nc) = k
  k = 0
  vcoff = 0
  do s = 0, nc - 1
    vcoff(s) = k
    do p = 0, nv - 1
      if ((lab(no + p) - lmin) / step == s) then
        vperm(k) = p
        vcls(k) = s
        k = k + 1
      end if
    end do
  end do
  vcoff(nc) = k
end subroutine xzclasses

! ---------------------------------------------------------------------------
! The same with the classes given (0 .. nc - 1, some of them empty): the
! classes c1 + gB c2 of two labels (xzsetmode).
subroutine xzclasses0(no, nv, cc, nc, maxcls, operm, vperm, ocls, vcls, ocoff, vcoff)
  implicit none
  integer(8), intent(in) :: no, nv, nc
  integer(8), intent(in) :: cc(0:no+nv-1)
  integer, intent(in) :: maxcls
  integer(8), intent(out) :: operm(0:no-1), vperm(0:nv-1), ocls(0:no-1), vcls(0:nv-1)
  integer(8), intent(out) :: ocoff(0:maxcls), vcoff(0:maxcls)
  integer(8) :: p, s, k
  k = 0
  ocoff = 0
  do s = 0, nc - 1
    ocoff(s) = k
    do p = 0, no - 1
      if (cc(p) == s) then
        operm(k) = p
        ocls(k) = s
        k = k + 1
      end if
    end do
  end do
  ocoff(nc) = k
  k = 0
  vcoff = 0
  do s = 0, nc - 1
    vcoff(s) = k
    do p = 0, nv - 1
      if (cc(no + p) == s) then
        vperm(k) = p
        vcls(k) = s
        k = k + 1
      end if
    end do
  end do
  vcoff(nc) = k
end subroutine xzclasses0

! ---------------------------------------------------------------------------
! The two digits of the classes for the C kernels (zc_set_red of tnpack.c).
subroutine xzsetred(b, m2)
  use iso_c_binding
  implicit none
  integer(8), intent(in) :: b, m2
  interface
    subroutine zc_set_red(bb, mm) bind(C, name='zc_set_red')
      import :: c_int
      integer(c_int), value :: bb, mm
    end subroutine zc_set_red
  end interface
  call zc_set_red(int(b, c_int), int(m2, c_int))
end subroutine xzsetred

! ---------------------------------------------------------------------------
! The tables of the sector kernels of (Q) (csrc/tnpack.c: zc_q_sector_energy,
! zc_xv_sector), 32-bit integers; the classes of the virtual spinors are
! their labels there.
!   kv[a]          class of a
!   lid[k], lst[k], lmem[a]   class -> itself, its first spinor, the spinors
!   posv[a]        the place of a in its class
!   posq, pos3     the place of a pair / triple in its group, by the number
!                  of the pair / triple in lexicographic order
!   tri[A][3]      the indices of the triple A (the order of the program)
subroutine xzqstab(nv, nvp, nvt, nc, vcls, vcoff, vpoff, vtoff, vpidx, vpgrp, vtri, vfree, &
                   kv, lid, lst, lmem, posv, posq, pos3, tri)
  use xzdata, only: xzw
  implicit none
  integer(8), intent(in) :: nv, nvp, nvt, nc
  integer(8), intent(in) :: vcls(0:nv-1), vcoff(0:nc), vpoff(0:*), vtoff(0:*)
  integer(8), intent(in) :: vpidx(0:nv-1, 0:nv-1), vpgrp(0:nvp-1), vtri(0:nvp-1, 0:nv-1)
  integer(8), intent(in) :: vfree(0:nvt-1, 0:2)
  integer(4), intent(out) :: kv(0:nv-1), lid(0:nc-1), lst(0:nc), lmem(0:nv-1), posv(0:nv-1)
  integer(4), intent(out) :: posq(0:nvp-1), pos3(0:nvt-1), tri(0:2, 0:nvt-1)
  integer(8) :: a, b, c, k, p, t, g
  integer(8), external :: xzpidx, xztrank
  do a = 0, nv - 1
    kv(a) = int(vcls(a), 4)
    lmem(a) = int(a, 4)
    posv(a) = int(a - vcoff(vcls(a)), 4)
  end do
  do k = 0, nc - 1
    lid(k) = int(k, 4)
    lst(k) = int(vcoff(k), 4)
  end do
  lst(nc) = int(vcoff(nc), 4)
  do a = 0, nv - 1
    do b = a + 1, nv - 1
      p = vpidx(b, a)
      posq(xzpidx(a, b, nv)) = int(p - vpoff(vpgrp(p)), 4)
      do c = b + 1, nv - 1
        t = vtri(vpidx(c, b), a)
        g = xzw(vcls(a) + vcls(b) + vcls(c))
        pos3(xztrank(a, b, c, nv)) = int(t - vtoff(g), 4)
      end do
    end do
  end do
  do t = 0, nvt - 1
    tri(0, t) = int(vfree(t, 0), 4)
    tri(1, t) = int(vfree(t, 1), 4)
    tri(2, t) = int(vfree(t, 2), 4)
  end do
end subroutine xzqstab
