!
! Amplitudes in CFOUR's order of the spinors, for tests between runs with
! other labels or arithmetic (XZ_AMPOUT, XZ_AMPIN of xzcc.inc).  A stream
! file:
!
!   no, nv, nto, nvt                       4 integers(8)
!   T1[i][a]                               complex(8), no * nv
!   T2[i][j][a][b]                         complex(8), (no nv)**2
!   T3[I][A], I = i<j<k, A = a<b<c         complex(8), nto * nvt
!
! with CFOUR's numbers of the active spinors (from 0) and the triples in
! lexicographic order.  The program keeps T1 and T2 in its order of the
! spinors (operm, vperm) and T3 on i<j<k, a<b<c in that order, in the
! blocks of the groups of triples (block g at boff(g): [occupied triples
! of g][virtual triples of g]); a triple taken to CFOUR's order is sorted,
! with the sign of the permutation.  Elements of T3 that the labels forbid
! are zero in the file and not read from it.
!
module xzampio
  implicit none
contains
  ! the number of the triple p<q<r of 0 .. n-1 in lexicographic order
  integer(8) pure function xztrank(n, p, q, r)
    integer(8), intent(in) :: n, p, q, r
    integer(8) :: a
    xztrank = 0
    do a = 0, p - 1
      xztrank = xztrank + (n - 1 - a) * (n - 2 - a) / 2
    end do
    do a = p + 1, q - 1
      xztrank = xztrank + (n - 1 - a)
    end do
    xztrank = xztrank + (r - q - 1)
  end function xztrank

  ! sort three numbers, the sign of the permutation
  pure subroutine xzsort3(x, s)
    integer(8), intent(inout) :: x(3)
    real(8), intent(out) :: s
    integer(8) :: t
    s = 1d0
    if (x(1) > x(2)) then
      t = x(1); x(1) = x(2); x(2) = t; s = -s
    end if
    if (x(2) > x(3)) then
      t = x(2); x(2) = x(3); x(3) = t; s = -s
    end if
    if (x(1) > x(2)) then
      t = x(1); x(1) = x(2); x(2) = t; s = -s
    end if
  end subroutine xzsort3
end module xzampio

! ---------------------------------------------------------------------------
! For every stored element of T3 (the place in the array of the program):
! the place in the file (CFOUR's triples) and the sign.  ofree, vfree: the
! indices of the triples of the program (jofree, jvfree).
subroutine xzampmap(ofree, vfree, place, sgn)
  use xzdata
  use xzampio
  implicit none
  integer(8), intent(in) :: ofree(0:nto-1, 0:2), vfree(0:nvt-1, 0:2)
  integer(8), intent(out) :: place(0:n3-1)
  real(8), intent(out) :: sgn(0:n3-1)
  integer(8) :: g, i, a, x(3), y(3), ic, ac, na, idx
  real(8) :: so, sv
  place = -1
  sgn = 0d0
  do g = 0, ntg - 1
    na = vtoff(g + 1) - vtoff(g)
    do i = otoff(g), otoff(g + 1) - 1
      x = [operm(ofree(i, 0)), operm(ofree(i, 1)), operm(ofree(i, 2))]
      call xzsort3(x, so)
      ic = xztrank(no, x(1), x(2), x(3))
      do a = vtoff(g), vtoff(g + 1) - 1
        y = [vperm(vfree(a, 0)), vperm(vfree(a, 1)), vperm(vfree(a, 2))]
        call xzsort3(y, sv)
        ac = xztrank(nv, y(1), y(2), y(3))
        idx = boff(g) + (i - otoff(g)) * na + (a - vtoff(g))
        place(idx) = ac + nvt * ic
        sgn(idx) = so * sv
      end do
    end do
  end do
end subroutine xzampmap

! ---------------------------------------------------------------------------
! write T1, T2, T3 of the program (at kt1, kt2, kt3 of w) to the file
subroutine xzampw(w, kt1, kt2, kt3, fname)
  use xzdata
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kt1, kt2, kt3
  character(len=*), intent(in) :: fname
  complex(8), allocatable :: c1(:,:), c2(:,:,:,:), c3(:)
  integer(8), allocatable :: place(:)
  real(8), allocatable :: sgn(:)
  integer(8) :: i, j, a, b, k, m
  allocate(c1(0:nv-1, 0:no-1), c2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1), c3(0:nto*nvt-1))
  allocate(place(0:n3-1), sgn(0:n3-1))
  call xzampmap(w(jofree), w(jvfree), place, sgn)
  do i = 0, no - 1
    do a = 0, nv - 1
      c1(vperm(a), operm(i)) = elt(kt1, a + nv * i)
    end do
  end do
  do i = 0, no - 1
    do j = 0, no - 1
      do a = 0, nv - 1
        do b = 0, nv - 1
          m = b + nv * (a + nv * (j + no * i))
          c2(vperm(b), vperm(a), operm(j), operm(i)) = elt(kt2, m)
        end do
      end do
    end do
  end do
  c3 = (0d0, 0d0)
  do k = 0, n3 - 1
    if (place(k) >= 0) c3(place(k)) = sgn(k) * elt(kt3, k)
  end do
  open(unit=75, file=fname, form='unformatted', access='stream', status='replace')
  write(75) no, nv, nto, nvt
  write(75) c1, c2, c3
  close(75)
  deallocate(c1, c2, c3, place, sgn)
  write(6, '(a,a)') '   amplitudes written in CFOUR''s order to ', trim(fname)
contains
  complex(8) function elt(k0, m)
    integer(8), intent(in) :: k0, m
    if (nwe == 2) then
      elt = cmplx(w(k0 + 2 * m), w(k0 + 2 * m + 1), 8)
    else
      elt = cmplx(w(k0 + m), 0d0, 8)
    end if
  end function elt
end subroutine xzampw

! ---------------------------------------------------------------------------
! read T1, T2, T3 from the file into the program (at kt1, kt2, kt3 of w);
! imag: the largest imaginary part dropped in real arithmetic; forb: the
! largest element of T3 in the file that the labels of this run forbid
subroutine xzampr(w, kt1, kt2, kt3, fname, imag, forb)
  use xzdata
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kt1, kt2, kt3
  character(len=*), intent(in) :: fname
  real(8), intent(out) :: imag, forb
  complex(8), allocatable :: c1(:,:), c2(:,:,:,:), c3(:)
  integer(8), allocatable :: place(:)
  real(8), allocatable :: sgn(:)
  logical, allocatable :: used(:)
  integer(8) :: i, j, a, b, k, m, h(4)
  integer :: ios
  open(unit=75, file=fname, form='unformatted', access='stream', status='old', iostat=ios)
  if (ios /= 0) then
    write(6, '(/,a,a)') ' @XZCCSDT-F, XZ_AMPIN: no file ', trim(fname)
    call xzerrex
  end if
  read(75) h
  if (h(1) /= no .or. h(2) /= nv .or. h(3) /= nto .or. h(4) /= nvt) then
    write(6, '(/,a,4i8)') ' @XZCCSDT-F, XZ_AMPIN: the file has other sizes: ', h
    call xzerrex
  end if
  allocate(c1(0:nv-1, 0:no-1), c2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1), c3(0:nto*nvt-1))
  read(75) c1, c2, c3
  close(75)
  allocate(place(0:n3-1), sgn(0:n3-1), used(0:nto*nvt-1))
  call xzampmap(w(jofree), w(jvfree), place, sgn)
  imag = 0d0
  do i = 0, no - 1
    do a = 0, nv - 1
      call put(kt1, a + nv * i, c1(vperm(a), operm(i)))
    end do
  end do
  do i = 0, no - 1
    do j = 0, no - 1
      do a = 0, nv - 1
        do b = 0, nv - 1
          m = b + nv * (a + nv * (j + no * i))
          call put(kt2, m, c2(vperm(b), vperm(a), operm(j), operm(i)))
        end do
      end do
    end do
  end do
  used = .false.
  do k = 0, n3 - 1
    if (place(k) >= 0) then
      call put(kt3, k, sgn(k) * c3(place(k)))
      used(place(k)) = .true.
    end if
  end do
  forb = 0d0
  do k = 0, nto * nvt - 1
    if (.not. used(k)) forb = max(forb, abs(c3(k)))
  end do
  deallocate(c1, c2, c3, place, sgn, used)
contains
  subroutine put(k0, mm, z)
    integer(8), intent(in) :: k0, mm
    complex(8), intent(in) :: z
    if (nwe == 2) then
      w(k0 + 2 * mm) = dble(z)
      w(k0 + 2 * mm + 1) = aimag(z)
    else
      w(k0 + mm) = dble(z)
      imag = max(imag, abs(aimag(z)))
    end if
  end subroutine put
end subroutine xzampr

! ---------------------------------------------------------------------------
! the largest |a - b| of n words
real(8) function xzmaxd(n, a, b)
  implicit none
  integer(8), intent(in) :: n
  real(8), intent(in) :: a(n), b(n)
  integer(8) :: i
  xzmaxd = 0d0
  do i = 1, n
    xzmaxd = max(xzmaxd, abs(a(i) - b(i)))
  end do
end function xzmaxd
