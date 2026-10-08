!
! Index tables of the pairs and triples of n indices, as the kernels of
! csrc/ccsdt_t3.c and csrc/t3pack.c take them (see xzdata.F90).
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

subroutine xzmaps(n, np, nt, tri, sgn, free, pair, pa, pb, pidx)
  implicit none
  integer(8), intent(in) :: n, np, nt
  integer(8), intent(out) :: tri(0:np-1, 0:n-1)       ! [x][P]
  integer(1), intent(out) :: sgn(0:np-1, 0:n-1)       ! [x][P]
  integer(8), intent(out) :: free(0:nt-1, 0:2)        ! [r][I]
  integer(8), intent(out) :: pair(0:nt-1, 0:2)        ! [r][I]
  integer(8), intent(out) :: pa(0:np-1), pb(0:np-1)
  integer(8), intent(out) :: pidx(0:n-1, 0:n-1)       ! [p][q]
  integer(8) :: i, j, k, x, p, it
  integer(8), external :: xztrank

  pidx = -1
  p = 0
  do i = 0, n - 1
    do j = i + 1, n - 1
      pa(p) = i
      pb(p) = j
      pidx(j, i) = p
      p = p + 1
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
        tri(p, x) = xztrank(x, j, k, n)
        sgn(p, x) = 1
      else if (x < k) then
        tri(p, x) = xztrank(j, x, k, n)
        sgn(p, x) = -1
      else
        tri(p, x) = xztrank(j, k, x, n)
        sgn(p, x) = 1
      end if
    end do
  end do
  it = 0
  do i = 0, n - 1
    do j = i + 1, n - 1
      do k = j + 1, n - 1
        free(it, 0) = i
        free(it, 1) = j
        free(it, 2) = k
        pair(it, 0) = pidx(k, j)
        pair(it, 1) = pidx(k, i)
        pair(it, 2) = pidx(j, i)
        it = it + 1
      end do
    end do
  end do
end subroutine xzmaps

! e3(I) = sum of e over the indices of the triple I
subroutine xzsum3(n, nt, e, e3)
  implicit none
  integer(8), intent(in) :: n, nt
  real(8), intent(in) :: e(0:n-1)
  real(8), intent(out) :: e3(0:nt-1)
  integer(8) :: i, j, k, it
  it = 0
  do i = 0, n - 1
    do j = i + 1, n - 1
      do k = j + 1, n - 1
        e3(it) = e(i) + e(j) + e(k)
        it = it + 1
      end do
    end do
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
