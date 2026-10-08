!
! The T1-dressed integrals and the intermediates of a CCSDT iteration that
! are not sums over T3 (type-generic: compiled once per element type, see
! xztype.h).  The equations are those of src/zccsdt/cc/ccsdt_opt.py
! (docs/factorization.md); arrays in the layouts of the C kernels, declared
! with the indices reversed (xzdata.F90).
!
! Dressing: a virtual bra index a picks up - sum_m t1[m][a] <m..||..>, an
! occupied ket index i picks up + sum_e t1[i][e] <..||..e>.
!

! ---------------------------------------------------------------------------
! The dressed Fock matrix (canonical spinors: the bare one is diagonal).
subroutine N(xzfock)(no, nv, nvp, eps, t1, ooov, oovv, ovvo, gq, vpa, vpb, foo, fov, fvo, fvv, pvv)
  implicit none
  integer(8), intent(in) :: no, nv, nvp
  real(8), intent(in) :: eps(0:no+nv-1)
  ZT, intent(in) :: t1(0:nv-1, 0:no-1)                         ! [i][a]
  ZT, intent(in) :: ooov(0:nv-1, 0:no-1, 0:no-1, 0:no-1)       ! [m][n][i][e]
  ZT, intent(in) :: oovv(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)       ! [m][n][e][f]
  ZT, intent(in) :: ovvo(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)       ! [m][b][e][j]
  ZT, intent(in) :: gq(0:nvp-1, 0:nv-1, 0:no-1)                ! [m][y][Q]
  integer(8), intent(in) :: vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(out) :: foo(0:no-1, 0:no-1)                       ! [m][i]
  ZT, intent(out) :: fov(0:nv-1, 0:no-1)                       ! [m][e]
  ZT, intent(out) :: fvo(0:no-1, 0:nv-1)                       ! [a][i]
  ZT, intent(out) :: fvv(0:nv-1, 0:nv-1)                       ! [a][e]
  ZT :: pvv(0:nv-1, 0:nv-1)                                    ! [b][e]
  integer(8) :: m, i, j, a, b, e, q
  ZT :: s

  ! f + sum_ia <r i||s a> t1[i][a]
  do m = 0, no - 1
    do j = 0, no - 1
      s = 0
      do i = 0, no - 1
        do a = 0, nv - 1
          s = s + ooov(a, j, i, m) * t1(a, i)
        end do
      end do
      foo(j, m) = s
    end do
    foo(m, m) = foo(m, m) + eps(m)
    do e = 0, nv - 1
      s = 0
      do i = 0, no - 1
        do a = 0, nv - 1
          s = s + oovv(a, e, i, m) * t1(a, i)
        end do
      end do
      fov(e, m) = s
    end do
  end do
  do b = 0, nv - 1
    do j = 0, no - 1
      s = 0
      do i = 0, no - 1
        do a = 0, nv - 1
          s = s + ovvo(j, a, b, i) * t1(a, i)
        end do
      end do
      fvo(j, b) = s
    end do
  end do
  ! - sum_ia <ib||ea> t1[i][a], from <ib||de> on d<e
  pvv = 0
  do i = 0, no - 1
    do b = 0, nv - 1
      do q = 0, nvp - 1
        pvv(vpa(q), b) = pvv(vpa(q), b) - gq(q, b, i) * t1(vpb(q), i)
        pvv(vpb(q), b) = pvv(vpb(q), b) + gq(q, b, i) * t1(vpa(q), i)
      end do
    end do
  end do
  do b = 0, nv - 1
    pvv(b, b) = pvv(b, b) + eps(no + b)
  end do
  ! (1 - t1^T) f (1 + t1)
  do m = 0, no - 1
    do i = 0, no - 1
      do e = 0, nv - 1
        foo(i, m) = foo(i, m) + fov(e, m) * t1(e, i)
      end do
    end do
  end do
  do a = 0, nv - 1
    do e = 0, nv - 1
      s = pvv(e, a)
      do m = 0, no - 1
        s = s - t1(a, m) * fov(e, m)
      end do
      fvv(e, a) = s
    end do
  end do
  do a = 0, nv - 1
    do i = 0, no - 1
      s = fvo(i, a)
      do e = 0, nv - 1
        s = s + pvv(e, a) * t1(e, i)
      end do
      do m = 0, no - 1
        s = s - t1(a, m) * foo(i, m)
      end do
      fvo(i, a) = s
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! Dressed <mn||ie> and <mn||ij>.
subroutine N(xzdroo)(no, nv, t1, ooov, oovv, oooo, ooovd, ooood)
  implicit none
  integer(8), intent(in) :: no, nv
  ZT, intent(in) :: t1(0:nv-1, 0:no-1)
  ZT, intent(in) :: ooov(0:nv-1, 0:no-1, 0:no-1, 0:no-1)       ! [m][n][i][e]
  ZT, intent(in) :: oovv(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)       ! [m][n][e][f]
  ZT, intent(in) :: oooo(0:no-1, 0:no-1, 0:no-1, 0:no-1)       ! [m][n][i][j]
  ZT, intent(out) :: ooovd(0:nv-1, 0:no-1, 0:no-1, 0:no-1)
  ZT, intent(out) :: ooood(0:no-1, 0:no-1, 0:no-1, 0:no-1)
  integer(8) :: m, n, i, j, e, f
  ZT :: s
  !$omp parallel do collapse(2) private(m, n, i, j, e, f, s)
  do m = 0, no - 1
    do n = 0, no - 1
      do i = 0, no - 1
        do e = 0, nv - 1
          s = ooov(e, i, n, m)
          do f = 0, nv - 1
            s = s + t1(f, i) * oovv(e, f, n, m)
          end do
          ooovd(e, i, n, m) = s
        end do
      end do
      do i = 0, no - 1
        do j = 0, no - 1
          s = oooo(j, i, n, m)
          do f = 0, nv - 1
            s = s + ooovd(f, i, n, m) * t1(f, j) - ooov(f, j, n, m) * t1(f, i)
          end do
          ooood(j, i, n, m) = s
        end do
      end do
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! Dressed <mb||ef> on e<f: D[m][b][Q] = <mb||Q> - sum_n t1[n][b] <mn||Q>.
subroutine N(xzdrovvq)(no, nv, nvp, t1, gq, oovvp, d)
  implicit none
  integer(8), intent(in) :: no, nv, nvp
  ZT, intent(in) :: t1(0:nv-1, 0:no-1)
  ZT, intent(in) :: gq(0:nvp-1, 0:nv-1, 0:no-1)                ! [m][y][Q]
  ZT, intent(in) :: oovvp(0:nvp-1, 0:no-1, 0:no-1)             ! [m][n][Q]
  ZT, intent(out) :: d(0:nvp-1, 0:nv-1, 0:no-1)
  integer(8) :: m, b, n
  !$omp parallel do collapse(2) private(m, b, n)
  do m = 0, no - 1
    do b = 0, nv - 1
      d(:, b, m) = gq(:, b, m)
      do n = 0, no - 1
        d(:, b, m) = d(:, b, m) - t1(b, n) * oovvp(:, n, m)
      end do
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! Dressed <mb||ej> and <mb||ij>.  k: work, the size of ovvo.
subroutine N(xzdrov)(no, nv, nvp, t1, ooov, ovvo, gq, d, ooood, vpidx, ovvod, ovood, k)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, nvp
  ZT, intent(in) :: t1(0:nv-1, 0:no-1)
  ZT, intent(in) :: ooov(0:nv-1, 0:no-1, 0:no-1, 0:no-1)       ! [m][n][i][e]
  ZT, intent(in) :: ovvo(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)       ! [m][b][e][j]
  ZT, intent(in) :: gq(0:nvp-1, 0:nv-1, 0:no-1)                ! bare [m][b][Q]
  ZT, intent(in) :: d(0:nvp-1, 0:nv-1, 0:no-1)                 ! dressed [m][b][Q]
  ZT, intent(in) :: ooood(0:no-1, 0:no-1, 0:no-1, 0:no-1)      ! [m][n][i][j]
  integer(8), intent(in) :: vpidx(*)
  ZT, intent(out) :: ovvod(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)     ! [m][b][e][j]
  ZT, intent(out) :: ovood(0:no-1, 0:no-1, 0:nv-1, 0:no-1)     ! [m][b][i][j]
  ZT :: k(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)
  integer(8) :: m, n, b, e, i, j
  ZT :: s

  ! <mb||ej> + sum_f ~<mb||ef> t1[j][f] + sum_n t1[n][b] <mn||je>
  !$omp parallel do collapse(2) private(m, b, e, j, n, s)
  do m = 0, no - 1
    do b = 0, nv - 1
      do e = 0, nv - 1
        do j = 0, no - 1
          s = ovvo(j, e, b, m)
          do n = 0, no - 1
            s = s + t1(b, n) * ooov(e, j, n, m)
          end do
          ovvod(j, e, b, m) = s
        end do
      end do
    end do
  end do
  call N(zc_pair_t1_contract)(ovvod, d, t1, no * nv, nv, nvp, vpidx, no)
  ! k[m][b][e][j] = <mb||ej> + sum_f <mb||ef> t1[j][f]
  k = ovvo
  call N(zc_pair_t1_contract)(k, gq, t1, no * nv, nv, nvp, vpidx, no)
  ! <mb||ij> = <ij||mb>*, both ket indices and the bra index dressed
  !$omp parallel do collapse(2) private(m, b, i, j, e, n, s)
  do m = 0, no - 1
    do b = 0, nv - 1
      do i = 0, no - 1
        do j = 0, no - 1
          s = CONJ(ooov(b, m, j, i))
          do e = 0, nv - 1
            s = s + t1(e, i) * k(j, e, b, m) - t1(e, j) * ovvo(i, e, b, m)
          end do
          do n = 0, no - 1
            s = s - t1(b, n) * ooood(j, i, n, m)
          end do
          ovood(j, i, b, m) = s
        end do
      end do
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! T2 in the layouts that the steps take: t2p[i][j][Q], t2pp[P][Q], and
! tau1[P][Q] = t1[m][a] t1[n][b] - t1[n][a] t1[m][b], tt = t2pp + tau1.
subroutine N(xzt2forms)(no, nv, npo, nvp, t1, t2, opa, opb, vpa, vpb, t2p, t2pp, tau, tt)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: t1(0:nv-1, 0:no-1)
  ZT, intent(in) :: t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)         ! [i][j][a][b]
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(out) :: t2p(0:nvp-1, 0:no-1, 0:no-1)              ! [i][j][Q]
  ZT, intent(out) :: t2pp(0:nvp-1, 0:npo-1), tau(0:nvp-1, 0:npo-1), tt(0:nvp-1, 0:npo-1)
  integer(8) :: i, j, q, p, m, n
  !$omp parallel do collapse(2) private(i, j, q)
  do i = 0, no - 1
    do j = 0, no - 1
      do q = 0, nvp - 1
        t2p(q, j, i) = t2(vpb(q), vpa(q), j, i)
      end do
    end do
  end do
  !$omp parallel do private(p, q, m, n)
  do p = 0, npo - 1
    m = opa(p)
    n = opb(p)
    do q = 0, nvp - 1
      t2pp(q, p) = t2p(q, n, m)
      tau(q, p) = t1(vpa(q), m) * t1(vpb(q), n) - t1(vpa(q), n) * t1(vpb(q), m)
      tt(q, p) = t2pp(q, p) + tau(q, p)
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The operands of pass 1 (csrc/ccsdt_t3.c):
!   gpd[e][P][f] = <mn||ef>,  gpd[nv+j][P][e] = ~<mn||ej> = - ~<mn||je>
!   gmq[n][R][m'] = <m'n||R>, gmq[m][R][no+a] = ~<am||R> = - D[m][a][R]
!   ft[e][m] = f[m][e]
subroutine N(xzp1ops)(no, nv, npo, nvp, oovv, ooovd, oovvp, d, fov, opa, opb, gpd, gmq, ft)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: oovv(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)       ! [m][n][e][f]
  ZT, intent(in) :: ooovd(0:nv-1, 0:no-1, 0:no-1, 0:no-1)      ! [m][n][i][e]
  ZT, intent(in) :: oovvp(0:nvp-1, 0:no-1, 0:no-1)             ! [m][n][Q]
  ZT, intent(in) :: d(0:nvp-1, 0:nv-1, 0:no-1)                 ! [m][b][Q]
  ZT, intent(in) :: fov(0:nv-1, 0:no-1)
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1)
  ZT, intent(out) :: gpd(0:nv-1, 0:npo-1, 0:nv+no-1)           ! [e|j][P][f]
  ZT, intent(out) :: gmq(0:no+nv-1, 0:nvp-1, 0:no-1)           ! [m][R][y]
  ZT, intent(out) :: ft(0:no-1, 0:nv-1)                        ! [e][m]
  integer(8) :: m, n, p, e, j, r, a
  do e = 0, nv - 1
    do p = 0, npo - 1
      gpd(:, p, e) = oovv(:, e, opb(p), opa(p))
    end do
  end do
  do j = 0, no - 1
    do p = 0, npo - 1
      gpd(:, p, nv + j) = -ooovd(:, j, opb(p), opa(p))
    end do
  end do
  !$omp parallel do private(n, r, m, a)
  do n = 0, no - 1
    do r = 0, nvp - 1
      do m = 0, no - 1
        gmq(m, r, n) = oovvp(r, n, m)
      end do
      do a = 0, nv - 1
        gmq(no + a, r, n) = -d(r, a, n)
      end do
    end do
  end do
  do e = 0, nv - 1
    do m = 0, no - 1
      ft(m, e) = fov(e, m)
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The T3 part of the T2 equation from the results of pass 1, on i<j, a<b:
! ra[P][Q] on entry, ra + rB + rC on return.
subroutine N(xzr2t3)(no, nv, npo, nvp, opa, opb, vpa, vpb, yall, xacc, ra)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(in) :: yall(0:no+nv-1, 0:nv-1, 0:npo-1)           ! [P][d][y]
  ZT, intent(in) :: xacc(0:nvp-1, 0:no-1, 0:nv+no-1)           ! [e|j][i][Q]
  ZT, intent(inout) :: ra(0:nvp-1, 0:npo-1)
  integer(8) :: p, q, m, n, d, e
  !$omp parallel do private(p, q, m, n, d, e)
  do p = 0, npo - 1
    m = opa(p)
    n = opb(p)
    do q = 0, nvp - 1
      d = vpa(q)
      e = vpb(q)
      ra(q, p) = ra(q, p) - yall(no + d, e, p) + yall(no + e, d, p) &
                 + xacc(q, m, nv + n) - xacc(q, n, nv + m)
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The bra-dressed <ab||ei> on a<b, wv[Q][e][i]:
!   - <ie||ab>* + sum_{m<n} tau1[mn][ab] <mn||ei> - P(ab) sum_m t1[m][a] <mb||ei>
! g2: work [P][e][i]; yb: work [nba][nv][nv*no].
subroutine N(xzbv)(no, nv, npo, nvp, nba, t1, tau, gq, ooov, ovvo, opa, opb, vpa, vpb, wv, g2, yb)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp, nba
  ZT, intent(in) :: t1(0:nv-1, 0:no-1)
  ZT, intent(in) :: tau(0:nvp-1, 0:npo-1)
  ZT, intent(in) :: gq(0:nvp-1, 0:nv-1, 0:no-1)                ! [i][e][Q]
  ZT, intent(in) :: ooov(0:nv-1, 0:no-1, 0:no-1, 0:no-1)       ! [m][n][i][e]
  ZT, intent(in) :: ovvo(*)                                    ! [m][b][e][i]
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(out) :: wv(0:no-1, 0:nv-1, 0:nvp-1)               ! [Q][e][i]
  ZT :: g2(0:no-1, 0:nv-1, 0:npo-1), yb(*)
  integer(8) :: q, e, i, p, x0, nb
  ZT :: one, zero, mone
  one = 1
  zero = 0
  mone = -1
  !$omp parallel do private(q, e, i)
  do q = 0, nvp - 1
    do e = 0, nv - 1
      do i = 0, no - 1
        wv(i, e, q) = -CONJ(gq(q, e, i))
      end do
    end do
  end do
  do p = 0, npo - 1
    do e = 0, nv - 1
      do i = 0, no - 1
        g2(i, e, p) = -ooov(e, i, opb(p), opa(p))
      end do
    end do
  end do
  if (npo > 0) call N(zc_gemm)(1_4, 0_4, nvp, nv * no, npo, one, tau, nvp, g2, nv * no, one, wv, nv * no)
  do x0 = 0, nv - 1, nba
    nb = min(nba, nv - x0)
    ! yb[a - x0][b][(e, i)] = sum_m t1[m][a] <mb||ei>
    call N(zc_gemm)(1_4, 0_4, nb, nv * nv * no, no, one, t1(x0, 0), nv, ovvo, nv * nv * no, zero, &
                    yb, nv * nv * no)
    call N(zc_pair_antisym_batch_add)(wv, yb, nv, nvp, vpa, vpb, nv * no, x0, x0 + nb, mone)
  end do
end subroutine

! ---------------------------------------------------------------------------
! The dressed <ab||ij> on i<j, a<b without its sum_ef ~<ab||ef> t1 t1 part
! (which the ladder brings): r2[P][Q].
subroutine N(xzvvoo)(no, nv, npo, nvp, t1, tau, gpp, ooov, oooo, bv, opa, opb, vpa, vpb, r2)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: t1(0:nv-1, 0:no-1)
  ZT, intent(in) :: tau(0:nvp-1, 0:npo-1), gpp(0:nvp-1, 0:npo-1)
  ZT, intent(in) :: ooov(0:nv-1, 0:no-1, 0:no-1, 0:no-1)       ! [i][j][m][b]
  ZT, intent(in) :: oooo(0:no-1, 0:no-1, 0:no-1, 0:no-1)       ! [m][n][i][j]
  ZT, intent(in) :: bv(0:no-1, 0:nv-1, 0:nvp-1)                ! [Q][e][i]
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(out) :: r2(0:nvp-1, 0:npo-1)
  integer(8) :: p, q, pp, i, j, a, b, m, e
  ZT :: s
  !$omp parallel do collapse(2) private(p, q, pp, i, j, a, b, m, e, s)
  do p = 0, npo - 1
    do q = 0, nvp - 1
      i = opa(p)
      j = opb(p)
      a = vpa(q)
      b = vpb(q)
      s = CONJ(gpp(q, p))
      do m = 0, no - 1
        s = s - t1(a, m) * CONJ(ooov(b, m, j, i)) + t1(b, m) * CONJ(ooov(a, m, j, i))
      end do
      do pp = 0, npo - 1
        s = s + tau(q, pp) * oooo(j, i, opb(pp), opa(pp))
      end do
      do e = 0, nv - 1
        s = s + bv(j, e, q) * t1(e, i) - bv(i, e, q) * t1(e, j)
      end do
      r2(q, p) = s
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! W_vvvo += sum_{m<n} t2[mn][ab] (~<mn||ei> - f[m][e] d(n,i) + f[n][e] d(m,i))
subroutine N(xzwvoovo)(no, nv, npo, nvp, t2pp, ooovd, fov, opa, opb, wv, g2)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: t2pp(0:nvp-1, 0:npo-1)
  ZT, intent(in) :: ooovd(0:nv-1, 0:no-1, 0:no-1, 0:no-1)      ! [m][n][i][e]
  ZT, intent(in) :: fov(0:nv-1, 0:no-1)
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1)
  ZT :: wv(*)
  ZT :: g2(0:no-1, 0:nv-1, 0:npo-1)                            ! [P][e][i]
  integer(8) :: p, e, i, m, n
  ZT :: one
  one = 1
  if (npo == 0) return
  do p = 0, npo - 1
    m = opa(p)
    n = opb(p)
    do e = 0, nv - 1
      do i = 0, no - 1
        g2(i, e, p) = -ooovd(e, i, n, m)
      end do
      g2(n, e, p) = g2(n, e, p) - fov(e, m)
      g2(m, e, p) = g2(m, e, p) + fov(e, n)
    end do
  end do
  call N(zc_gemm)(1_4, 0_4, nvp, nv * no, npo, one, t2pp, nvp, g2, nv * no, one, wv, nv * no)
end subroutine

! ---------------------------------------------------------------------------
! W_vvvo -= P(ab) sum_{mf} ~<mb||ef> t2[m][i][a][f], in batches of a:
!   xb[(a, i)][(b, e)] = sum_m a2[(a, i)][(m, f)] u_m[(b, e)][f]
! a2: work (no nv)**2; u: work nv**3; xb, yb: work nba * no * nv**2 each.
subroutine N(xzwvx)(no, nv, nvp, nba, t2, d, vpa, vpb, wv, a2, u, xb, yb)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, nvp, nba
  ZT, intent(in) :: t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)         ! [m][i][a][f]
  ZT, intent(in) :: d(0:nvp-1, 0:nv-1, 0:no-1)                 ! [m][b][Q]
  integer(8), intent(in) :: vpa(0:nvp-1), vpb(0:nvp-1)
  ZT :: wv(*)
  ZT :: a2(0:nv-1, 0:no-1, 0:no-1, 0:nv-1)                     ! [a][i][m][f]
  ZT :: u(0:nv-1, 0:nv-1, 0:nv-1)                              ! [b][e][f]
  ZT :: xb(0:nv-1, 0:nv-1, 0:no-1, 0:nba-1)                    ! [a][i][b][e]
  ZT :: yb(0:no-1, 0:nv-1, 0:nv-1, 0:nba-1)                    ! [a][b][e][i]
  integer(8) :: a, i, m, f, b, q, e, x0, nb
  ZT :: one, zero, mone, beta
  one = 1
  zero = 0
  mone = -1
  !$omp parallel do collapse(2) private(a, i, m, f)
  do a = 0, nv - 1
    do i = 0, no - 1
      do m = 0, no - 1
        do f = 0, nv - 1
          a2(f, m, i, a) = t2(f, a, i, m)
        end do
      end do
    end do
  end do
  do x0 = 0, nv - 1, nba
    nb = min(nba, nv - x0)
    do m = 0, no - 1
      !$omp parallel do private(b, q)
      do b = 0, nv - 1
        u(:, :, b) = 0
        do q = 0, nvp - 1
          u(vpb(q), vpa(q), b) = d(q, b, m)
          u(vpa(q), vpb(q), b) = -d(q, b, m)
        end do
      end do
      beta = one
      if (m == 0) beta = zero
      call N(zc_gemm)(0_4, 1_4, nb * no, nv * nv, nv, one, a2(0, m, 0, x0), no * nv, u, nv, beta, &
                      xb, nv * nv)
    end do
    !$omp parallel do collapse(2) private(a, b, e, i)
    do a = 0, nb - 1
      do b = 0, nv - 1
        do e = 0, nv - 1
          do i = 0, no - 1
            yb(i, e, b, a) = xb(e, b, i, a)
          end do
        end do
      end do
    end do
    call N(zc_pair_antisym_batch_add)(wv, yb, nv, nvp, vpa, vpb, nv * no, x0, x0 + nb, mone)
  end do
end subroutine

! ---------------------------------------------------------------------------
! wo[m][a][P] of the T3 drive term, P = (i<j):
!   - ~<ma||ij> - tmp[m][a][i][j] + tmp[m][a][j][i] - sum_Q D[m][a][Q] t2[P][Q] + yall[P][a][m]
!   tmp[m][b][i][j] = sum_ne ~<mn||ie> t2[j][n][b][e]
! tmp: work no**3 * nv.
subroutine N(xzwo)(no, nv, npo, nvp, t2, t2pp, ooovd, ovood, d, yall, opa, opb, wo, tmp)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)         ! [i][j][a][b]
  ZT, intent(in) :: t2pp(0:nvp-1, 0:npo-1)
  ZT, intent(in) :: ooovd(0:nv-1, 0:no-1, 0:no-1, 0:no-1)      ! [m][n][i][e]
  ZT, intent(in) :: ovood(0:no-1, 0:no-1, 0:nv-1, 0:no-1)      ! [m][b][i][j]
  ZT, intent(in) :: d(*)
  ZT, intent(in) :: yall(0:no+nv-1, 0:nv-1, 0:npo-1)           ! [P][d][y]
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1)
  ZT, intent(out) :: wo(0:npo-1, 0:nv-1, 0:no-1)               ! [m][a][P]
  ZT :: tmp(0:no-1, 0:no-1, 0:nv-1, 0:no-1)                    ! [m][b][i][j]
  integer(8) :: m, b, i, j, n, e, p
  ZT :: s, mone, zero
  mone = -1
  zero = 0
  !$omp parallel do collapse(2) private(m, b, i, j, n, e, s)
  do m = 0, no - 1
    do b = 0, nv - 1
      do i = 0, no - 1
        do j = 0, no - 1
          s = 0
          do n = 0, no - 1
            do e = 0, nv - 1
              s = s + ooovd(e, i, n, m) * t2(e, b, n, j)
            end do
          end do
          tmp(j, i, b, m) = s
        end do
      end do
    end do
  end do
  ! - sum_Q D[(m, a)][Q] t2pp[P][Q]
  call N(zc_gemm)(0_4, 1_4, no * nv, npo, nvp, mone, d, nvp, t2pp, nvp, zero, wo, npo)
  do m = 0, no - 1
    do b = 0, nv - 1
      do p = 0, npo - 1
        i = opa(p)
        j = opb(p)
        wo(p, b, m) = wo(p, b, m) - ovood(j, i, b, m) - tmp(j, i, b, m) + tmp(i, j, b, m) &
                      + yall(m, b, p)
      end do
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! W_ovvo, F_vv and F_oo of the T3 equation:
!   weff[m][a][d][i] = ~<ma||di> + sum_ne <mn||de> t2[i][n][a][e]
!   feov[a][d] = f[a][d] - 1/2 sum_mnf t2[m][n][a][f] <mn||df>
!   feoo[m][i] = f[m][i] + sum_{n,Q} <mn||Q> t2[i][n][Q]
! a2, b2, x2: work (no nv)**2 each.
subroutine N(xzweff)(no, nv, nvp, t2, t2p, oovv, oovvp, ovvod, foo, fvv, weff, fevv, feoo, a2, b2, x2)
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
  ZT :: a2(0:nv-1, 0:no-1, 0:nv-1, 0:no-1)                     ! [(m, d)][(n, e)]
  ZT :: b2(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)                     ! [(n, e)][(a, i)]
  ZT :: x2(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)                     ! [(m, d)][(a, i)]
  integer(8) :: m, n, d, e, a, i, f, q
  ZT :: s, one, zero
  one = 1
  zero = 0
  !$omp parallel do collapse(2) private(m, n, d, e)
  do m = 0, no - 1
    do d = 0, nv - 1
      do n = 0, no - 1
        do e = 0, nv - 1
          a2(e, n, d, m) = oovv(e, d, n, m)
        end do
      end do
    end do
  end do
  !$omp parallel do collapse(2) private(n, e, a, i)
  do n = 0, no - 1
    do e = 0, nv - 1
      do a = 0, nv - 1
        do i = 0, no - 1
          b2(i, a, e, n) = t2(e, a, n, i)
        end do
      end do
    end do
  end do
  call N(zc_gemm)(0_4, 0_4, no * nv, no * nv, no * nv, one, a2, no * nv, b2, no * nv, zero, x2, no * nv)
  !$omp parallel do collapse(2) private(m, a, d, i)
  do m = 0, no - 1
    do a = 0, nv - 1
      do d = 0, nv - 1
        do i = 0, no - 1
          weff(i, d, a, m) = ovvod(i, d, a, m) + x2(i, a, d, m)
        end do
      end do
    end do
  end do
  !$omp parallel do collapse(2) private(a, d, m, n, f, s)
  do a = 0, nv - 1
    do d = 0, nv - 1
      s = 0
      do m = 0, no - 1
        do n = 0, no - 1
          do f = 0, nv - 1
            s = s + t2(f, a, n, m) * oovv(f, d, n, m)
          end do
        end do
      end do
      fevv(d, a) = fvv(d, a) - 0.5d0 * s
    end do
  end do
  do m = 0, no - 1
    do i = 0, no - 1
      s = 0
      do n = 0, no - 1
        do q = 0, nvp - 1
          s = s + oovvp(q, n, m) * t2p(q, n, i)
        end do
      end do
      feoo(i, m) = foo(i, m) + s
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The CCSD part of the T1 and T2 equations (T1-dressed), the T3 parts added,
! and the Jacobi step:
!   R1 = f_ai + f_kc t_ikac + 1/2 g_akcd t_ikcd - 1/2 g_klic t_klac
!   R2 = g_abij + 1/2 g_abcd (t + tau1)_ijcd + 1/2 W_klij t_klab
!        + P(ab) F_bc t_ijac - P(ij) F_kj t_ikab + P(ij)P(ab) W_kbcj t_ikac
! On entry r2[P][Q] has the dressed <ab||ij>, lad[P][Q] the ladder with the
! dressed <ab||cd>, wkl[K][P] is work.  On return wkl[K][P] = W_klij on pairs
! (the operator of the hh ladder of T3 as well).
! a2, b2, x2: work (no nv)**2 each.
subroutine N(xzccsd)(no, nv, npo, nvp, eps, t1, t2, t2p, t2pp, gpp, ooovd, ooood, ovvod, weff, &
                     fov, fvo, fevv, feoo, r1v, r1c, r2t3, lad, opa, opb, vpa, vpb, &
                     r2, wkl, t1n, t2n, a2, b2, x2)
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
  ZT, intent(in) :: r1c(0:no-1, 0:nv-1)                        ! [a][i]: T3 part of R1
  ZT, intent(in) :: r2t3(0:nvp-1, 0:npo-1), lad(0:nvp-1, 0:npo-1)
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(inout) :: r2(0:nvp-1, 0:npo-1)
  ZT :: wkl(0:npo-1, 0:npo-1)                                  ! [K][P]
  ZT, intent(out) :: t1n(0:nv-1, 0:no-1)
  ZT, intent(out) :: t2n(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT :: a2(0:nv-1, 0:no-1, 0:nv-1, 0:no-1)                     ! [(i, a)][(k, c)]
  ZT :: b2(0:nv-1, 0:no-1, 0:nv-1, 0:no-1)                     ! [(k, c)][(j, b)]
  ZT :: x2(0:nv-1, 0:no-1, 0:nv-1, 0:no-1)                     ! [(i, a)][(j, b)]
  integer(8) :: i, j, k, a, b, c, p, q, kk, m, n
  ZT :: s, one, zero, v
  real(8) :: dn
  one = 1
  zero = 0

  ! ---- T1
  !$omp parallel do collapse(2) private(i, a, k, c, m, n, s)
  do i = 0, no - 1
    do a = 0, nv - 1
      s = fvo(i, a) + r1v(no + a, i) + r1c(i, a)
      do k = 0, no - 1
        do c = 0, nv - 1
          s = s + fov(c, k) * t2(c, a, k, i)
        end do
      end do
      do m = 0, no - 1
        do n = 0, no - 1
          do c = 0, nv - 1
            s = s - 0.5d0 * ooovd(c, i, n, m) * t2(c, a, n, m)
          end do
        end do
      end do
      t1n(a, i) = t1(a, i) + s / (eps(i) - eps(no + a))
    end do
  end do

  ! ---- T2: the ladder, W_klij and the hh term
  r2 = r2 + lad
  do kk = 0, npo - 1
    do p = 0, npo - 1
      wkl(p, kk) = ooood(opb(p), opa(p), opb(kk), opa(kk))
    end do
  end do
  if (npo > 0) then
    call N(zc_gemm)(0_4, 1_4, npo, npo, nvp, one, gpp, nvp, t2pp, nvp, one, wkl, npo)
    call N(zc_gemm)(1_4, 0_4, npo, nvp, npo, one, wkl, npo, t2pp, nvp, one, r2, nvp)
  end if
  ! P(ab) F_bc t_ijac  and  - P(ij) F_kj t_ikab
  !$omp parallel do private(p, q, i, j, a, b, c, k, s)
  do p = 0, npo - 1
    i = opa(p)
    j = opb(p)
    do q = 0, nvp - 1
      a = vpa(q)
      b = vpb(q)
      s = 0
      do c = 0, nv - 1
        s = s + t2(c, a, j, i) * fevv(c, b) - t2(c, b, j, i) * fevv(c, a)
      end do
      do k = 0, no - 1
        s = s - t2p(q, k, i) * feoo(j, k) + t2p(q, k, j) * feoo(i, k)
      end do
      r2(q, p) = r2(q, p) + s
    end do
  end do
  ! P(ij) P(ab) W_kbcj t_ikac, W = (~<kb||cj> + weff[k][b][c][j]) / 2
  !$omp parallel do collapse(2) private(i, a, k, c)
  do i = 0, no - 1
    do a = 0, nv - 1
      do k = 0, no - 1
        do c = 0, nv - 1
          a2(c, k, a, i) = t2(c, a, k, i)
          b2(a, i, c, k) = 0.5d0 * (ovvod(i, c, a, k) + weff(i, c, a, k))
        end do
      end do
    end do
  end do
  call N(zc_gemm)(0_4, 0_4, no * nv, no * nv, no * nv, one, a2, no * nv, b2, no * nv, zero, x2, no * nv)
  !$omp parallel do private(p, q, i, j, a, b)
  do p = 0, npo - 1
    i = opa(p)
    j = opb(p)
    do q = 0, nvp - 1
      a = vpa(q)
      b = vpb(q)
      r2(q, p) = r2(q, p) + x2(b, j, a, i) - x2(b, i, a, j) - x2(a, j, b, i) + x2(a, i, b, j)
    end do
  end do
  ! ---- the Jacobi step of T2
  t2n = t2
  !$omp parallel do private(p, q, i, j, a, b, v, dn)
  do p = 0, npo - 1
    i = opa(p)
    j = opb(p)
    do q = 0, nvp - 1
      a = vpa(q)
      b = vpb(q)
      dn = eps(i) + eps(j) - eps(no + a) - eps(no + b)
      v = (r2(q, p) + r2t3(q, p)) / dn
      t2n(b, a, j, i) = t2n(b, a, j, i) + v
      t2n(a, b, j, i) = t2n(a, b, j, i) - v
      t2n(b, a, i, j) = t2n(b, a, i, j) - v
      t2n(a, b, i, j) = t2n(a, b, i, j) + v
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The operator of the hh ladder and of F_oo on occupied triples:
!   (L t)[i][j][k] = P(k/ij) 1/2 sum_mn W[m][n][i][j] t[m][n][k]
!                    - P(i/jk) sum_m F[m][i] t[m][j][k]
! as o6[I][J]: for the splits of I and of J into (index, pair) with the same
! index the element W[pair of J][pair of I], with the same pair
! - F[index of J][index of I], times the two signs.
subroutine N(xzo6)(no, npo, nto, ofree, opair, wkl, feoo, o6)
  implicit none
  integer(8), intent(in) :: no, npo, nto
  integer(8), intent(in) :: ofree(0:nto-1, 0:2), opair(0:nto-1, 0:2)
  ZT, intent(in) :: wkl(0:npo-1, 0:npo-1)                      ! [K][P]
  ZT, intent(in) :: feoo(0:no-1, 0:no-1)                       ! [m][i]
  ZT, intent(out) :: o6(0:nto-1, 0:nto-1)                      ! [I][J]
  integer(8) :: it, jt, r, s
  real(8), parameter :: sg(0:2) = [1d0, -1d0, 1d0]
  ZT :: v
  !$omp parallel do private(it, jt, r, s, v)
  do it = 0, nto - 1
    do jt = 0, nto - 1
      v = 0
      do r = 0, 2
        do s = 0, 2
          if (ofree(it, r) == ofree(jt, s)) v = v + sg(r) * sg(s) * wkl(opair(it, r), opair(jt, s))
          if (opair(it, r) == opair(jt, s)) v = v - sg(r) * sg(s) * feoo(ofree(it, r), ofree(jt, s))
        end do
      end do
      o6(jt, it) = v
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The operands of pass 2 that are small:
!   wring[(a, i)][(d, m)] = weff[m][a][d][i]
!   adrv[P][c][e] = t2[P][e][c],  adrv[P][c][nv + m] = wo[m][c][P]
subroutine N(xzp2ops)(no, nv, npo, t2, weff, wo, opa, opb, wring, adrv)
  implicit none
  integer(8), intent(in) :: no, nv, npo
  ZT, intent(in) :: t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)         ! [i][j][a][b]
  ZT, intent(in) :: weff(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)       ! [m][a][d][i]
  ZT, intent(in) :: wo(0:npo-1, 0:nv-1, 0:no-1)                ! [m][a][P]
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1)
  ZT, intent(out) :: wring(0:no-1, 0:nv-1, 0:no-1, 0:nv-1)     ! [a][i][d][m]
  ZT, intent(out) :: adrv(0:nv+no-1, 0:nv-1, 0:npo-1)          ! [P][c][e|m]
  integer(8) :: a, i, d, m, p, c, e
  !$omp parallel do collapse(2) private(a, i, d, m)
  do a = 0, nv - 1
    do i = 0, no - 1
      do d = 0, nv - 1
        do m = 0, no - 1
          wring(m, d, i, a) = weff(i, d, a, m)
        end do
      end do
    end do
  end do
  !$omp parallel do private(p, c, e, m)
  do p = 0, npo - 1
    do c = 0, nv - 1
      do e = 0, nv - 1
        adrv(e, c, p) = t2(c, e, opb(p), opa(p))
      end do
      do m = 0, no - 1
        adrv(nv + m, c, p) = wo(p, c, m)
      end do
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! bdrv[e][(i, Q)] = wv[Q][e][i],  bdrv[nv + m][(k, Q)] = t2p[m][k][Q]
subroutine N(xzbdrv)(no, nv, nvp, wv, t2p, bdrv)
  implicit none
  integer(8), intent(in) :: no, nv, nvp
  ZT, intent(in) :: wv(0:no-1, 0:nv-1, 0:nvp-1)                ! [Q][e][i]
  ZT, intent(in) :: t2p(0:nvp-1, 0:no-1, 0:no-1)               ! [m][k][Q]
  ZT, intent(out) :: bdrv(0:nvp-1, 0:no-1, 0:nv+no-1)          ! [e|m][i][Q]
  integer(8) :: e, i, q, q0, m
  integer(8), parameter :: nb = 64
  !$omp parallel do private(e, i, q, q0)
  do e = 0, nv - 1
    do q0 = 0, nvp - 1, nb
      do i = 0, no - 1
        do q = q0, min(q0 + nb, nvp) - 1
          bdrv(q, i, e) = wv(i, e, q)
        end do
      end do
    end do
  end do
  do m = 0, no - 1
    bdrv(:, :, nv + m) = t2p(:, :, m)
  end do
end subroutine

! ---------------------------------------------------------------------------
! n elements from src to dst, which may be a lower place that overlaps it.
subroutine N(xzmove)(n, src, dst)
  implicit none
  integer(8), intent(in) :: n
  ZT :: src(*), dst(*)
  integer(8) :: i
  !dir$ novector
  do i = 1, n
    dst(i) = src(i)
  end do
end subroutine

! ---------------------------------------------------------------------------
! The correlation energy of T1 and T2 (canonical spinors: no f_ia term):
!   1/4 sum t2[i][j][a][b] <ij||ab> + 1/2 sum t1[i][a] t1[j][b] <ij||ab>
subroutine N(xzecorr)(no, nv, t1, t2, oovv, e)
  implicit none
  integer(8), intent(in) :: no, nv
  ZT, intent(in) :: t1(0:nv-1, 0:no-1)
  ZT, intent(in) :: t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: oovv(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(out) :: e
  integer(8) :: i, j, a, b
  ZT :: s
  s = 0
  !$omp parallel do collapse(2) private(i, j, a, b) reduction(+:s)
  do i = 0, no - 1
    do j = 0, no - 1
      do a = 0, nv - 1
        do b = 0, nv - 1
          s = s + (0.25d0 * t2(b, a, j, i) + 0.5d0 * t1(a, i) * t1(b, j)) * oovv(b, a, j, i)
        end do
      end do
    end do
  end do
  e = s
end subroutine

! ---------------------------------------------------------------------------
! The second-order amplitudes: t1 = 0, t2 = <ij||ab>* / D.
subroutine N(xzguess)(no, nv, eps, oovv, t1, t2)
  implicit none
  integer(8), intent(in) :: no, nv
  real(8), intent(in) :: eps(0:no+nv-1)
  ZT, intent(in) :: oovv(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(out) :: t1(0:nv-1, 0:no-1)
  ZT, intent(out) :: t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  integer(8) :: i, j, a, b
  t1 = 0
  do i = 0, no - 1
    do j = 0, no - 1
      do a = 0, nv - 1
        do b = 0, nv - 1
          t2(b, a, j, i) = CONJ(oovv(b, a, j, i)) / (eps(i) + eps(j) - eps(no + a) - eps(no + b))
        end do
      end do
    end do
  end do
end subroutine
