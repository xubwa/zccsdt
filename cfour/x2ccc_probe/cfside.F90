!
! CFOUR's side of the probe: what hbarso (x2cccsd/hbarso.f90) needs to be
! called outside xx2cccsd.  hbarso takes one argument, the work array, and
! everything else from two modules:
!
!   soccmod (libx2ccc)   no, nu, nmo, ndrop, ndimt1, ndimt2, inonhf, ipgmask,
!                        ncvs, icvs, ppppao, iofftvv; the tables too, tvv
!                        (allocated by settrioff); the names of the PPPH
!                        files (allocated by wppphfname); r1n, r2n, rmax, ...
!                        (results)
!   soccmem (x2cccsd)    the offsets into the work array: integrals hhhh,
!                        pphh, phhp, hhpp, ppph, hhhp, pppp; oe; t1, t2, r1,
!                        r2; ftaa, ftii, ftia, whhhh, wphhp, tau, taut;
!                        tmp1, tmp2, tmp3; (ftaa2, ftii2, ftia2 for a
!                        non-canonical reference)
!   cvwint_module (libx2c)   iconti
!
! and three files of the run directory read in every call: CONTOCCN and
! CONTOCCP (conmasksd) and ZMAT (oldgradmask), next to the PPPH files.
! xx2cccsd sets all this in setsocc, eomccini and x2c_ccsddrv, which also
! compute the AO integrals for the AO-driven ladder; none of that is called
! here.
!
! Layouts of CFOUR (Fortran order, indices from 1; pairs include the
! diagonal: ab = a + b (b - 1) / 2 for a <= b, likewise ij):
!   t1(a, i), t2(ab, ij), r1, r2 likewise; pphh(ab, ij) = <ab||ij>,
!   hhhh(mn, ij), hhhp(mn, i, e), phhp / hhpp(e, m, b, j), pppp(ab, cd),
!   ppph(ef, a) of one occupied spinor at a time, from the file PPPHnnnn.
!

! The module variables for o occupied and v virtual spinors and the layout
! of the region: lcf double words.
subroutine cfsetup(o, v, nfro, lcf)
  use soccmod
  use soccmem
  use cvwint_module, only: iconti
  implicit none
  integer(8), intent(in) :: o, v, nfro
  integer(8), intent(out) :: lcf
  integer(8) :: po, pv, unit, itot
  no = o
  nu = v
  nmo = o + v
  ndrop = nfro
  ndropvir = 0
  po = o * (o + 1) / 2
  pv = v * (v + 1) / 2
  ndimt1 = 2 * o * v
  ndimt2 = 2 * pv * po
  inonhf = 0
  ipgmask = 0
  iconti = 0
  ncvs = 0
  icvs = 0
  ppppao = .false.
  itot = v * (v - 1) * (v - 2) / 6
  iofftvvv = 1
  iofftvv = iofftvvv + 3 * itot
  iofftoo = iofftvv + 2 * pv
  iend = iofftoo + 2 * po
  hhhh = iend
  pphh = hhhh + 2 * po * po
  phhp = pphh + 2 * pv * po
  hhpp = phhp + 2 * o * o * v * v
  ppph = hhpp + 2 * o * o * v * v
  hhhp = ppph + 2 * pv * v              ! one record of a PPPH file
  pppp = hhhp + 2 * po * o * v
  oe = pppp + 2 * pv * pv
  ftaa2 = oe + 2 * (o + v)
  ftii2 = ftaa2 + 2 * v * v
  ftia2 = ftii2 + 2 * o * o
  t1 = ftia2 + 2 * o * v
  t2 = t1 + ndimt1
  r1 = t2 + ndimt2
  r2 = r1 + ndimt1
  ftaa = r2 + ndimt2
  ftii = ftaa + 2 * v * v
  ftia = ftii + 2 * o * o
  whhhh = ftia + 2 * o * v
  wphhp = whhhh + 2 * po * po
  tau = wphhp + 2 * o * o * v * v
  taut = tau + ndimt2
  unit = 2 * max(o ** 4, o * o * v * v, v ** 3)
  tmp1 = taut + ndimt2
  tmp2 = tmp1 + unit
  tmp3 = tmp2 + unit
  iend = tmp3 + unit
  lcf = iend
  lwork = lcf
end subroutine cfsetup

! The tables, the file names and the integrals, by CFOUR's own routines; the
! spinor energies from the native reader (real, occupied first).
subroutine cfload(work, lcf, o, v, eps)
  use soccmod
  use soccmem
  implicit none
  integer(8), intent(in) :: lcf, o, v
  real(8) :: work(lcf)
  real(8), intent(in) :: eps(o + v)
  integer(8) :: k, izero(1)
  logical :: there
  call dset(lcf, 0d0, work)
  call settrioff(work(iofftvvv), work(iofftvv), work(iofftoo))
  call wppphfname
  call vreadf(work(pppp), work(ppph), no, nu)
  call vreadp(work(hhhh), work(phhp), work(pphh), work(hhpp), work(ppph), work(hhhp), no, nu)
  do k = 1, o + v
    work(oe + 2 * (k - 1)) = eps(k)
    work(oe + 2 * (k - 1) + 1) = 0d0
  end do
  ! conmasksd reads these two in every call (xx2cccsd writes them: no
  ! occupied spinor is constrained)
  inquire(file='CONTOCCN', exist=there)
  if (.not. there) then
    izero = 0
    call wfile('CONTOCCN', izero, 1_8)
    call wfile('CONTOCCP', izero, 1_8)
    write(6, '(a)') '   (CONTOCCN, CONTOCCP written: no constrained occupied spinor)'
  end if
end subroutine cfload

! T1 [i][a] and T2 [i][j][a][b] of the native program into t1(a, i) and
! t2(ab, ij) of CFOUR.
subroutine cfput(work, o, v, t1n, t2n)
  use soccmem, only: t1, t2
  implicit none
  integer(8), intent(in) :: o, v
  real(8) :: work(*)
  complex(8), intent(in) :: t1n(0:v-1, 0:o-1), t2n(0:v-1, 0:v-1, 0:o-1, 0:o-1)
  call cfput1(work(t1), work(t2), o, v, t1n, t2n)
end subroutine cfput

subroutine cfput1(c1, c2, o, v, t1n, t2n)
  implicit none
  integer(8), intent(in) :: o, v
  complex(8), intent(out) :: c1(v, o), c2(v * (v + 1) / 2, o * (o + 1) / 2)
  complex(8), intent(in) :: t1n(0:v-1, 0:o-1), t2n(0:v-1, 0:v-1, 0:o-1, 0:o-1)
  integer(8) :: i, j, a, b, ij, ab
  c2 = 0
  do i = 1, o
    do a = 1, v
      c1(a, i) = t1n(a - 1, i - 1)
    end do
  end do
  !$omp parallel do private(i, j, a, b, ij, ab) schedule(dynamic)
  do j = 1, o
    do i = 1, j - 1
      ij = i + j * (j - 1) / 2
      do b = 1, v
        do a = 1, b - 1
          ab = a + b * (b - 1) / 2
          c2(ab, ij) = t2n(b - 1, a - 1, j - 1, i - 1)
        end do
      end do
    end do
  end do
end subroutine cfput1

! The step of hbarso (r1, r2 are its updates dT = R / D) against the new
! amplitudes of the native step.
!   d1, d2   largest |T(CFOUR) + dT(CFOUR) - Tnew(native)| of T1 and T2
!   u1, u2   largest |dT| of the native step
!   ecf      the correlation energy of CFOUR's new amplitudes (real, imaginary)
subroutine cfcmp(work, o, v, t1o, t2o, t1n, t2n, d1, d2, u1, u2, ecf)
  use soccmem, only: t1, t2, r1, r2, pphh
  implicit none
  integer(8), intent(in) :: o, v
  real(8) :: work(*)
  complex(8), intent(in) :: t1o(0:v-1, 0:o-1), t2o(0:v-1, 0:v-1, 0:o-1, 0:o-1)
  complex(8), intent(in) :: t1n(0:v-1, 0:o-1), t2n(0:v-1, 0:v-1, 0:o-1, 0:o-1)
  real(8), intent(out) :: d1, d2, u1, u2, ecf(2)
  call cfcmp1(work(t1), work(t2), work(r1), work(r2), work(pphh), o, v, t1o, t2o, t1n, t2n, d1, d2, u1, u2, ecf)
end subroutine cfcmp

subroutine cfcmp1(c1, c2, s1, s2, g, o, v, t1o, t2o, t1n, t2n, d1, d2, u1, u2, ecf)
  implicit none
  integer(8), intent(in) :: o, v
  complex(8), intent(in) :: c1(v, o), c2(v * (v + 1) / 2, o * (o + 1) / 2)
  complex(8), intent(in) :: s1(v, o), s2(v * (v + 1) / 2, o * (o + 1) / 2)
  complex(8), intent(in) :: g(v * (v + 1) / 2, o * (o + 1) / 2)
  complex(8), intent(in) :: t1o(0:v-1, 0:o-1), t2o(0:v-1, 0:v-1, 0:o-1, 0:o-1)
  complex(8), intent(in) :: t1n(0:v-1, 0:o-1), t2n(0:v-1, 0:v-1, 0:o-1, 0:o-1)
  real(8), intent(out) :: d1, d2, u1, u2, ecf(2)
  integer(8) :: i, j, a, b, ij, ab
  complex(8) :: x, y, e
  d1 = 0
  d2 = 0
  u1 = 0
  u2 = 0
  e = 0
  do i = 1, o
    do a = 1, v
      d1 = max(d1, abs(c1(a, i) + s1(a, i) - t1n(a - 1, i - 1)))
      u1 = max(u1, abs(t1n(a - 1, i - 1) - t1o(a - 1, i - 1)))
    end do
  end do
  do j = 1, o
    do i = 1, j - 1
      ij = i + j * (j - 1) / 2
      do b = 1, v
        do a = 1, b - 1
          ab = a + b * (b - 1) / 2
          x = c2(ab, ij) + s2(ab, ij)
          y = t2n(b - 1, a - 1, j - 1, i - 1)
          d2 = max(d2, abs(x - y))
          u2 = max(u2, abs(y - t2o(b - 1, a - 1, j - 1, i - 1)))
          ! <ij||ab> = conjg(<ab||ij>); tau of the new amplitudes
          e = e + conjg(g(ab, ij)) * (x + (c1(a, i) + s1(a, i)) * (c1(b, j) + s1(b, j)) &
                                        - (c1(b, i) + s1(b, i)) * (c1(a, j) + s1(a, j)))
        end do
      end do
    end do
  end do
  ecf(1) = dble(e)
  ecf(2) = aimag(e)
end subroutine cfcmp1

! hbarso refers to the AO-driven ladder (ABCDTYPE=AOBASIS), which is not
! linked here: the probe sets ppppao = .false.
subroutine t2abcdaoso(work)
  implicit none
  real(8) :: work(*)
  write(6, '(a)') ' @X2CPROBE-F, the AO-driven ladder is not part of the probe.'
  call errex
end subroutine t2abcdaoso
