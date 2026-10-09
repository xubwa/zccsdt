!
! The density stage with frozen occupied and dropped virtual spinors ("held"
! spinors h; type-generic): the orbital gradient between held and active
! spinors, the rotations y that follow from it, and the rows and columns of
! the right-hand side of the Z-vector equations with a held index
! (docs/native.md, section 13; density_opt.py: _frozen_stage, frozen_terms,
! relaxed_rdm1; zvector._relax).
!
! All of it from x[q][r][s] = (h q|r s), one held spinor at a time, q, r, s
! over all spinors in the full order of xzfrz.F90 (active occupied, active
! virtual, frozen, dropped):
!   sweep 1   bf, kf: the blocks of the two-particle density with the
!             T1-dressed integrals <h a||b c> of the active spinors; the Fock
!             matrices of five densities, F[M][h][q] = sum_rs <h s||q r> M[r][s]
!   then      G between held and active spinors and between dropped and
!             frozen ones, y = G / (difference of energies)
!   sweep 2   the Fock matrix of y - (the unrelaxed virtual-occupied density)
!             in the rows and columns of a held spinor and, of y, between the
!             active spinors
! The integrals are transformed twice (once per sweep) and never stored.
!

! ---------------------------------------------------------------------------
! x[q][r][s] = (h q|r s) in the full order from what CFOUR's transformation
! left: conj(v(s, r, q)) in CFOUR's numbers (the real part in real arithmetic).
subroutine N(xzdfrzx)(nall, cf, v, x)
  implicit none
  integer(8), intent(in) :: nall
  integer(8), intent(in) :: cf(0:nall-1)
  complex(8), intent(in) :: v(nall, nall, nall)
  ZT, intent(out) :: x(0:nall-1, 0:nall-1, 0:nall-1)           ! (s, r, q)
  integer(8) :: q, r, s
  !$omp parallel do collapse(2) private(q, r, s)
  do q = 0, nall - 1
    do r = 0, nall - 1
      do s = 0, nall - 1
#ifdef XZ_REAL
        x(s, r, q) = dble(v(cf(s), cf(r), cf(q)))
#else
        x(s, r, q) = conjg(v(cf(s), cf(r), cf(q)))
#endif
      end do
    end do
  end do
end subroutine

! f[q] = sum_{r,s} <h s||q r> m[r][s] = sum_{r,s} ((h q|s r) - (h r|s q)) m[r][s]
subroutine N(xzdfockh)(nall, x, m, f)
  implicit none
  integer(8), intent(in) :: nall
  ZT, intent(in) :: x(0:nall-1, 0:nall-1, 0:nall-1)
  ZT, intent(in) :: m(0:nall-1, 0:nall-1)                      ! (r, s)
  ZT, intent(out) :: f(0:nall-1)
  integer(8) :: q, r, s
  ZT :: t
  !$omp parallel do private(q, r, s, t)
  do q = 0, nall - 1
    t = 0
    do s = 0, nall - 1
      do r = 0, nall - 1
        t = t + (x(r, s, q) - x(q, s, r)) * m(r, s)
      end do
    end do
    f(q) = t
  end do
end subroutine

! The active part of x, T1-dressed, antisymmetrized:
!   xa[a][b][c] = (h a|b c) (conjugated if cj) with its three indices dressed:
!     kind 'y'  an occupied row gains sum_e t1[i][e] (the virtual rows)
!     kind 'x'  a virtual row loses sum_m t1[m][a] (the occupied rows)
!   kinds = 'yxy' (for bf) or 'xyx' (for kf, on the conjugate)
!   tt[a][b][c] = xa[b][a][c] - xa[c][a][b] = the dressed <h a||b c>
subroutine N(xzdfrzdress)(nall, no, nv, cj, x, t1, xa, tt)
  implicit none
  integer(8), intent(in) :: nall, no, nv
  logical, intent(in) :: cj
  ZT, intent(in) :: x(0:nall-1, 0:nall-1, 0:nall-1)
  ZT, intent(in) :: t1(0:nv-1, 0:no-1)
  ZT :: xa(0:no+nv-1, 0:no+nv-1, 0:no+nv-1)                    ! (c, b, a)
  ZT, intent(out) :: tt(0:no+nv-1, 0:no+nv-1, 0:no+nv-1)
  integer(8) :: n, a, b, c, i, e
  n = no + nv
  !$omp parallel do collapse(2) private(a, b)
  do a = 0, n - 1
    do b = 0, n - 1
      if (cj) then
        xa(:, b, a) = CONJ(x(0:n - 1, b, a))
      else
        xa(:, b, a) = x(0:n - 1, b, a)
      end if
    end do
  end do
  ! the last index
  !$omp parallel do collapse(2) private(a, b, i, e)
  do a = 0, n - 1
    do b = 0, n - 1
      if (cj) then
        do e = 0, nv - 1
          do i = 0, no - 1
            xa(no + e, b, a) = xa(no + e, b, a) - t1(e, i) * xa(i, b, a)
          end do
        end do
      else
        do i = 0, no - 1
          do e = 0, nv - 1
            xa(i, b, a) = xa(i, b, a) + t1(e, i) * xa(no + e, b, a)
          end do
        end do
      end if
    end do
  end do
  ! the middle index
  !$omp parallel do private(a, i, e)
  do a = 0, n - 1
    if (cj) then
      do i = 0, no - 1
        do e = 0, nv - 1
          xa(:, i, a) = xa(:, i, a) + t1(e, i) * xa(:, no + e, a)
        end do
      end do
    else
      do e = 0, nv - 1
        do i = 0, no - 1
          xa(:, no + e, a) = xa(:, no + e, a) - t1(e, i) * xa(:, i, a)
        end do
      end do
    end if
  end do
  ! the first index
  if (cj) then
    !$omp parallel do private(e, i)
    do e = 0, nv - 1
      do i = 0, no - 1
        xa(:, :, no + e) = xa(:, :, no + e) - t1(e, i) * xa(:, :, i)
      end do
    end do
  else
    !$omp parallel do private(e, i)
    do i = 0, no - 1
      do e = 0, nv - 1
        xa(:, :, i) = xa(:, :, i) + t1(e, i) * xa(:, :, no + e)
      end do
    end do
  end if
  !$omp parallel do collapse(2) private(a, b, c)
  do a = 0, n - 1
    do b = 0, n - 1
      do c = 0, n - 1
        tt(c, b, a) = xa(c, a, b) - xa(b, a, c)
      end do
    end do
  end do
end subroutine

! tqo[m][Q] = tt[m][g][h], tqv[x][Q] = tt[no+x][g][h] for the virtual pairs Q = (g<h)
subroutine N(xzdfrzq)(no, nv, nvp, vpa, vpb, tt, tqo, tqv)
  implicit none
  integer(8), intent(in) :: no, nv, nvp
  integer(8), intent(in) :: vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(in) :: tt(0:no+nv-1, 0:no+nv-1, 0:no+nv-1)
  ZT, intent(out) :: tqo(0:nvp-1, 0:no-1), tqv(0:nvp-1, 0:nv-1)
  integer(8) :: q, m
  !$omp parallel do private(q, m)
  do q = 0, nvp - 1
    do m = 0, no - 1
      tqo(q, m) = tt(no + vpb(q), no + vpa(q), m)
    end do
    do m = 0, nv - 1
      tqv(q, m) = tt(no + vpb(q), no + vpa(q), no + m)
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! bf[x] = 1/2 sum <h q||r s> g2[x, q, r, s] of one held spinor (density_opt.
! _frozen_stage, the part "B"): tt the integrals dressed 'yxy'.
!   gam[Q][Q'] = sum_{I,d} l3[I, d, Q] t3[I, d, Q'] in its diagonal pair blocks
subroutine N(xzdfrzb)(no, nv, npo, nvp, tt, tqo, tqv, t1, l2, t2pp, c3, z, dai, rl, y4, dk, dab, gam, &
                      opa, opb, opidx, vpa, vpb, vpidx, us, bf)
  use xzdata, only: npg, vpoff, vvoff
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: tt(0:no+nv-1, 0:no+nv-1, 0:no+nv-1), tqo(0:nvp-1, 0:no-1), tqv(0:nvp-1, 0:nv-1)
  ZT, intent(in) :: t1(0:nv-1, 0:no-1), l2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1), t2pp(0:nvp-1, 0:npo-1)
  ZT, intent(in) :: c3(0:nv-1, 0:nvp-1, 0:no-1), z(0:no-1, 0:nv-1, 0:npo-1)
  ZT, intent(in) :: dai(0:nvp-1, 0:no-1, 0:nv-1)
  ZT, intent(in) :: rl(0:no-1, 0:nv-1, 0:no-1, 0:nv-1)         ! Rg[a][j][b][i]
  ZT, intent(in) :: y4(0:npo-1, 0:npo-1), dk(0:no-1, 0:nv-1, 0:no-1, 0:no-1), dab(0:nvp-1, 0:npo-1)
  ZT, intent(in) :: gam(0:*)
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), opidx(0:no-1, 0:no-1)
  integer(8), intent(in) :: vpa(0:nvp-1), vpb(0:nvp-1), vpidx(0:nv-1, 0:nv-1)
  ZT :: us(0:nv-1, 0:npo-1)                                    ! [P][e]
  ZT, intent(out) :: bf(0:no+nv-1)
  ZT :: bi(0:no-1), ba(0:nv-1)
  integer(8) :: i, m, p, pp, j, a, k, e, b, q, h, qlo, nqh, e1, e2, x, y, qq
  ZT :: s, sg, one, g1, g2
  one = 1
  ! US[P][e] = <h e||m n> + sum_Q t2[P][Q] <h e||Q>
  !$omp parallel do private(p, e)
  do p = 0, npo - 1
    do e = 0, nv - 1
      us(e, p) = tt(opb(p), opa(p), no + e)
    end do
  end do
  if (npo > 0 .and. nvp > 0) call N(zc_gemm)(0_4, 1_4, npo, nv, nvp, one, t2pp, nvp, tqv, nvp, one, us, nv)
  !$omp parallel do private(i, m, p, pp, sg, j, a, k, e, b, q, s)
  do i = 0, no - 1
    s = 0
    do m = 0, no - 1
      if (m == i) cycle
      if (i < m) then
        pp = opidx(m, i)
        sg = 1
      else
        pp = opidx(i, m)
        sg = -1
      end if
      do p = 0, npo - 1
        s = s + sg * tt(opb(p), opa(p), m) * y4(pp, p)
      end do
      do q = 0, nvp - 1
        s = s + sg * dab(q, pp) * tqo(q, m)
      end do
    end do
    do j = 0, no - 1
      do a = 0, nv - 1
        do k = 0, no - 1
          s = s + tt(k, no + a, j) * dk(k, a, j, i)
        end do
      end do
    end do
    do e = 0, nv - 1
      do p = 0, npo - 1
        s = s + tt(opb(p), opa(p), no + e) * z(i, e, p)
      end do
      do q = 0, nvp - 1
        s = s - dai(q, i, e) * tqv(q, e)
      end do
    end do
    do a = 0, nv - 1
      do b = 0, nv - 1
        do j = 0, no - 1
          s = s + tt(j, no + b, no + a) * rl(i, b, j, a)
        end do
      end do
    end do
    bi(i) = s
  end do
  !$omp parallel do private(a, m, p, i, b, j, q, e, e1, e2, pp, sg, s)
  do a = 0, nv - 1
    s = 0
    do m = 0, no - 1
      do p = 0, npo - 1
        s = s - tt(opb(p), opa(p), m) * z(m, a, p)
      end do
      do q = 0, nvp - 1
        s = s + dai(q, m, a) * tqo(q, m)
      end do
    end do
    do i = 0, no - 1
      do b = 0, nv - 1
        do j = 0, no - 1
          s = s - tt(j, no + b, i) * rl(i, b, j, a)
        end do
      end do
    end do
    do p = 0, npo - 1
      do e = 0, nv - 1
        s = s + l2(e, a, opb(p), opa(p)) * us(e, p)
      end do
    end do
    ! - 1/2 sum <h e1||e2 m> C3[m][(a e1)][e2]
    do e1 = 0, nv - 1
      if (e1 == a) cycle
      if (a < e1) then
        pp = vpidx(e1, a)
        sg = -0.5d0
      else
        pp = vpidx(a, e1)
        sg = 0.5d0
      end if
      do e2 = 0, nv - 1
        do m = 0, no - 1
          s = s + sg * tt(m, no + e2, no + e1) * c3(e2, pp, m)
        end do
      end do
    end do
    ba(a) = s
  end do
  ! the l3 t3 part of the block abcd: rows of gam
  do h = 0, npg - 1
    qlo = vpoff(h)
    nqh = vpoff(h + 1) - qlo
    !$omp parallel do private(qq, x, y, q, g1, g2) reduction(+:ba)
    do qq = 0, nqh - 1
      x = vpa(qlo + qq)
      y = vpb(qlo + qq)
      g1 = 0
      g2 = 0
      do q = 0, nqh - 1
        g1 = g1 + gam(vvoff(h) + qq * nqh + q) * tqv(qlo + q, x)
        g2 = g2 + gam(vvoff(h) + qq * nqh + q) * tqv(qlo + q, y)
      end do
      ba(y) = ba(y) - g1
      ba(x) = ba(x) + g2
    end do
  end do
  do i = 0, no - 1
    s = bi(i)
    do a = 0, nv - 1
      s = s - ba(a) * t1(a, i)
    end do
    bf(i) = s
  end do
  bf(no:no + nv - 1) = ba
end subroutine

! kf[x] = 1/2 sum <p q||h s> g2[p, q, x, s] of one held spinor (the part "K"):
! tt the conjugated integrals dressed 'xyx'.
subroutine N(xzdfrzk)(no, nv, npo, nvp, tt, tqo, tqv, t1, t2, l2pp, l2p, c3, z, dai, rl, y4, dk, dab, gam, &
                      opa, opb, opidx, vpa, vpb, vpidx, uk, kf)
  use xzdata, only: npg, vpoff, vvoff
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: tt(0:no+nv-1, 0:no+nv-1, 0:no+nv-1), tqo(0:nvp-1, 0:no-1), tqv(0:nvp-1, 0:nv-1)
  ZT, intent(in) :: t1(0:nv-1, 0:no-1), t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1), l2pp(0:nvp-1, 0:npo-1)
  ZT, intent(in) :: l2p(0:nvp-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: c3(0:nv-1, 0:nvp-1, 0:no-1), z(0:no-1, 0:nv-1, 0:npo-1)
  ZT, intent(in) :: dai(0:nvp-1, 0:no-1, 0:nv-1)
  ZT, intent(in) :: rl(0:no-1, 0:nv-1, 0:no-1, 0:nv-1)
  ZT, intent(in) :: y4(0:npo-1, 0:npo-1), dk(0:no-1, 0:nv-1, 0:no-1, 0:no-1), dab(0:nvp-1, 0:npo-1)
  ZT, intent(in) :: gam(0:*)
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), opidx(0:no-1, 0:no-1)
  integer(8), intent(in) :: vpa(0:nvp-1), vpb(0:nvp-1), vpidx(0:nv-1, 0:nv-1)
  ZT :: uk(0:nv-1, 0:npo-1)                                    ! [P][h]
  ZT, intent(out) :: kf(0:no+nv-1)
  ZT :: ki(0:no-1), ka(0:nv-1)
  integer(8) :: i, m, p, pp, j, a, k, e, b, q, h, qlo, nqh, e1, e2, x, y, qq, jp
  ZT :: s, sg, one, zero, g1, g2
  one = 1
  zero = 0
  if (npo > 0 .and. nvp > 0) then
    call N(zc_gemm)(0_4, 1_4, npo, nv, nvp, one, l2pp, nvp, tqv, nvp, zero, uk, nv)
  else
    uk = 0
  end if
  !$omp parallel do private(a, k, p, e, pp, sg, j, b, i, m, q, e1, e2, s)
  do a = 0, nv - 1
    s = 0
    do k = 0, no - 1
      do p = 0, npo - 1
        s = s + tt(opb(p), opa(p), k) * dk(k, a, opb(p), opa(p))
      end do
      do q = 0, nvp - 1
        s = s - 0.5d0 * c3(a, q, k) * tqo(q, k)
      end do
    end do
    do e = 0, nv - 1
      if (e == a) cycle
      if (a < e) then
        pp = vpidx(e, a)
        sg = 1
      else
        pp = vpidx(a, e)
        sg = -1
      end if
      do p = 0, npo - 1
        s = s + sg * tt(opb(p), opa(p), no + e) * dab(pp, p)
      end do
      ! + sum <h e||e2 m>' Dai[e2][m][(a e)]
      do e2 = 0, nv - 1
        do m = 0, no - 1
          s = s + sg * tt(m, no + e2, no + e) * dai(pp, m, e2)
        end do
      end do
    end do
    do j = 0, no - 1
      do b = 0, nv - 1
        do i = 0, no - 1
          s = s - tt(i, no + b, j) * rl(i, a, j, b)
        end do
      end do
    end do
    do p = 0, npo - 1
      do e = 0, nv - 1
        s = s + t2(e, a, opb(p), opa(p)) * uk(e, p)
      end do
    end do
    ka(a) = s
  end do
  !$omp parallel do private(j, k, p, pp, sg, a, jp, e, m, i, b, q, s)
  do j = 0, no - 1
    s = 0
    do k = 0, no - 1
      if (k /= j) then
        if (j < k) then
          pp = opidx(k, j)
          sg = 1
        else
          pp = opidx(j, k)
          sg = -1
        end if
        do p = 0, npo - 1
          s = s + sg * tt(opb(p), opa(p), k) * y4(p, pp)
        end do
        ! - sum <h k||e m>' Zu[j][k][e][m]
        do e = 0, nv - 1
          do m = 0, no - 1
            s = s - sg * tt(m, no + e, k) * z(m, e, pp)
          end do
        end do
      end if
      do q = 0, nvp - 1
        s = s + l2p(q, k, j) * tqo(q, k)
      end do
    end do
    do a = 0, nv - 1
      do p = 0, npo - 1
        s = s - tt(opb(p), opa(p), no + a) * dk(j, a, opb(p), opa(p))
      end do
      do q = 0, nvp - 1
        s = s + 0.5d0 * c3(a, q, j) * tqv(q, a)
      end do
      do b = 0, nv - 1
        do i = 0, no - 1
          s = s + tt(i, no + a, no + b) * rl(i, b, j, a)
        end do
      end do
    end do
    ki(j) = s
  end do
  ! the l3 t3 part of the block abcd: columns of gam
  do h = 0, npg - 1
    qlo = vpoff(h)
    nqh = vpoff(h + 1) - qlo
    !$omp parallel do private(qq, x, y, q, g1, g2) reduction(+:ka)
    do qq = 0, nqh - 1
      x = vpa(qlo + qq)
      y = vpb(qlo + qq)
      g1 = 0
      g2 = 0
      do q = 0, nqh - 1
        g1 = g1 + gam(vvoff(h) + q * nqh + qq) * tqv(qlo + q, x)
        g2 = g2 + gam(vvoff(h) + q * nqh + qq) * tqv(qlo + q, y)
      end do
      ka(y) = ka(y) - g1
      ka(x) = ka(x) + g2
    end do
  end do
  kf(0:no - 1) = ki
  do a = 0, nv - 1
    s = ka(a)
    do i = 0, no - 1
      s = s + ki(i) * t1(a, i)
    end do
    kf(no + a) = s
  end do
end subroutine

! ---------------------------------------------------------------------------
! The five densities of sweep 1, each (r, s) over all spinors, zero outside
! the active ones:  tau^T, conj(tau), A^T, conj(A), Dc
!   tau[i][a] = t1[i][a];  A = d1 - n - tau / 2 (d1[p][q] = den(q, p));
!   Dc = D - n, D the Hermitian unrelaxed density dh(p, q)
subroutine N(xzdfrzm)(nall, no, nv, t1, den, dh, mm)
  implicit none
  integer(8), intent(in) :: nall, no, nv
  ZT, intent(in) :: t1(0:nv-1, 0:no-1)
  ZT, intent(in) :: den(0:no+nv-1, 0:no+nv-1), dh(0:no+nv-1, 0:no+nv-1)
  ZT, intent(out) :: mm(0:nall-1, 0:nall-1, 5)
  integer(8) :: r, s, i, a, n
  ZT :: av
  n = no + nv
  mm = 0
  do i = 0, no - 1
    do a = 0, nv - 1
      mm(no + a, i, 1) = t1(a, i)
      mm(i, no + a, 2) = CONJ(t1(a, i))
    end do
  end do
  do r = 0, n - 1
    do s = 0, n - 1
      ! A[r][s]
      av = den(s, r)
      if (r == s .and. r < no) av = av - 1d0
      if (r < no .and. s >= no) av = av - 0.5d0 * t1(s - no, r)
      mm(s, r, 3) = av
      mm(r, s, 4) = CONJ(av)
      mm(r, s, 5) = dh(r, s)
    end do
    if (r < no) mm(r, r, 5) = mm(r, r, 5) - 1d0
  end do
end subroutine

! ---------------------------------------------------------------------------
! After sweep 1: the orbital gradient between held and active spinors,
!   B = bf + F[tau][h][y] A[x][y] + F[A][h][y] tau[x][y]
!   K = kf + F[tau][y][h] A[y][x] + F[A][y][h] tau[y][x]
!   Gf[h][x] = (B + conj(K)) / 2 + (n_x - n_h) V[h][x],   V = F[Dc]
! (frozen_terms; fo(q, k, h) = F[M_k][h][q] of the five densities), the
! rotations y between a held spinor and the active ones of its kind,
!   y[h][x] = - Gf[h][x] / (e_h - e_x),  y[x][h] its conjugate,
! and m1 = y - Dv (Dv: the virtual-occupied and occupied-virtual blocks of the
! unrelaxed density), the density of sweep 2.  gf(x, h) = Gf[h][x];
! gdf(q, h) = V[h][q] (used between dropped and frozen spinors).
subroutine N(xzdfrzg)(nall, no, nv, nf, nd, ef, t1, den, dh, bf, kf, fo, gf, ym, m1)
  implicit none
  integer(8), intent(in) :: nall, no, nv, nf, nd
  real(8), intent(in) :: ef(0:nall-1)
  ZT, intent(in) :: t1(0:nv-1, 0:no-1)
  ZT, intent(in) :: den(0:no+nv-1, 0:no+nv-1), dh(0:no+nv-1, 0:no+nv-1)
  ZT, intent(in) :: bf(0:no+nv-1, 0:nf+nd-1), kf(0:no+nv-1, 0:nf+nd-1)
  ZT, intent(in) :: fo(0:nall-1, 5, 0:nf+nd-1)
  ZT, intent(out) :: gf(0:no+nv-1, 0:nf+nd-1)
  ZT, intent(out) :: ym(0:nall-1, 0:nall-1), m1(0:nall-1, 0:nall-1)
  integer(8) :: n, h, hp, x, y, i, a
  ZT :: b, k, axy, ayx, tau
  real(8) :: occh, occx, de, gmax
  real(8), parameter :: degenerate = 1d-9
  n = no + nv
  ym = 0
  gmax = 0d0
  do h = 0, nf + nd - 1
    hp = n + h
    occh = merge(1d0, 0d0, h < nf)
    do x = 0, n - 1
      b = bf(x, h)
      k = kf(x, h)
      do y = 0, n - 1
        ! A[x][y], A[y][x]
        axy = den(y, x)
        ayx = den(x, y)
        if (x == y .and. x < no) then
          axy = axy - 1d0
          ayx = ayx - 1d0
        end if
        if (x < no .and. y >= no) axy = axy - 0.5d0 * t1(y - no, x)
        if (y < no .and. x >= no) ayx = ayx - 0.5d0 * t1(x - no, y)
        b = b + fo(y, 1, h) * axy
        k = k + CONJ(fo(y, 2, h)) * ayx
        ! tau[x][y], tau[y][x]
        if (x < no .and. y >= no) b = b + fo(y, 3, h) * t1(y - no, x)
        if (y < no .and. x >= no) k = k + CONJ(fo(y, 4, h)) * t1(x - no, y)
      end do
      occx = merge(1d0, 0d0, x < no)
      gf(x, h) = 0.5d0 * (b + CONJ(k)) + (occx - occh) * fo(x, 5, h)
    end do
    ! the rotations with the active spinors of its kind
    if (h < nf) then
      do i = 0, no - 1
        de = ef(hp) - ef(i)
        if (abs(de) > degenerate) then
          ym(hp, i) = -gf(i, h) / de
          ym(i, hp) = CONJ(ym(hp, i))
        else
          gmax = max(gmax, abs(gf(i, h)))
        end if
      end do
    else
      do a = no, n - 1
        de = ef(hp) - ef(a)
        if (abs(de) > degenerate) then
          ym(hp, a) = -gf(a, h) / de
          ym(a, hp) = CONJ(ym(hp, a))
        else
          gmax = max(gmax, abs(gf(a, h)))
        end if
      end do
    end if
  end do
  if (gmax > 1d-8) then
    write(6, '(/,a)') ' @XZCCSDT-W, a frozen or dropped spinor is degenerate with an active one of its kind and the'
    write(6, '(a,es10.2,a)') '             energy changes with their rotation (orbital gradient ', gmax, &
      '): the relaxed'
    write(6, '(a)') '             density is not defined.  Freeze whole levels.'
  end if
  m1 = ym
  do a = no, n - 1
    do i = 0, no - 1
      m1(a, i) = m1(a, i) - dh(a, i)
      m1(i, a) = m1(i, a) - dh(i, a)
    end do
  end do
end subroutine

! Sweep 2, between the active spinors: fy[a][i] += sum_j (<a h||i j> y[j][h] + <a j||i h> y[h][j])
! for the held spinor hp (full order), j the active spinors of its kind
subroutine N(xzdfrzy)(nall, no, nv, hp, j0, j1, x, ym, fy)
  implicit none
  integer(8), intent(in) :: nall, no, nv, hp, j0, j1
  ZT, intent(in) :: x(0:nall-1, 0:nall-1, 0:nall-1)            ! (s, r, q) = (h q|r s)
  ZT, intent(in) :: ym(0:nall-1, 0:nall-1)
  ZT, intent(inout) :: fy(0:nv-1, 0:no-1)                      ! (a, i)
  integer(8) :: a, i, j, av
  ZT :: s
  !$omp parallel do collapse(2) private(a, i, j, av, s)
  do i = 0, no - 1
    do a = 0, nv - 1
      av = no + a
      s = 0
      do j = j0, j1
        s = s + (x(i, av, j) - x(j, av, i)) * ym(j, hp) + CONJ(x(av, i, j) - x(j, i, av)) * ym(hp, j)
      end do
      fy(a, i) = fy(a, i) + s
    end do
  end do
end subroutine
