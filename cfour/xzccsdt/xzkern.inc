! Interfaces of the C kernels of one element type (included by xzkern.F90
! once per type; csrc/ccsdt_t3.c, csrc/t3pack.c, csrc/tnpack.c).  Arrays are
! assumed-type: regions of the work array are passed as they are.  All
! layouts are those of the C side (row-major, indices from 0).

    subroutine N(zc_gemm)(ta, tb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc) bind(C)
      import
      integer(c_int), value :: ta, tb
      integer(c_int64_t), value :: m, n, k, lda, ldb, ldc
      ZT :: alpha, beta
      type(*), dimension(*) :: a, b, c
    end subroutine

    subroutine N(zc_t3_pass1w)(t3, nto, nvt, no, nv, npo, nvp, otri, osgn, vtri, vsgn, batch, &
                               gpd, gmq, ny, gpp, ft, xacc, yall, r1, ra, io, stats, buf) bind(C)
      import
      integer(c_int64_t), value :: nto, nvt, no, nv, npo, nvp, batch, ny
      type(*), dimension(*) :: t3, otri, osgn, vtri, vsgn, gpd, gmq, gpp, ft, xacc, yall, r1, ra, buf
      type(c_ptr), value :: io
      real(c_double) :: stats(4)
    end subroutine

    subroutine N(zc_t3_pass2w)(t3, nto, nvt, no, nv, npo, nvp, otri, osgn, ofree, opair, &
                               vtri, vsgn, vfree, vpair, batch, wring, adrv, bdrv, r, io, stats, &
                               tb, z) bind(C)
      import
      integer(c_int64_t), value :: nto, nvt, no, nv, npo, nvp, batch
      type(*), dimension(*) :: t3, otri, osgn, ofree, opair, vtri, vsgn, vfree, vpair
      type(*), dimension(*) :: wring, adrv, bdrv, r, tb, z
      type(c_ptr), value :: io
      real(c_double) :: stats(4)
    end subroutine

    subroutine N(zc_t3_occop)(o6, t3, nto, nvt, ncb, io, r) bind(C)
      import
      integer(c_int64_t), value :: nto, nvt, ncb
      type(*), dimension(*) :: o6, t3, r
      type(c_ptr), value :: io
    end subroutine

    subroutine N(zc_t3_update)(t3, r, nto, nvt, eo, ev, io) bind(C)
      import
      integer(c_int64_t), value :: nto, nvt
      type(*), dimension(*) :: t3, r, eo, ev
      type(c_ptr), value :: io
    end subroutine

    ! ---- the passes of a Lambda iteration (csrc/ccsdt_t3.c) and the kernels of
    !      csrc/lambda_t3.c
    subroutine N(zc_l3_pass1w)(l3, t3, nto, nvt, no, nv, npo, nvp, otri, osgn, vtri, vsgn, batch, &
                               gpd, gmq, ny, l2pp, xacc, yall, dr, d1, stats, bufl, buft) bind(C)
      import
      integer(c_int64_t), value :: nto, nvt, no, nv, npo, nvp, batch, ny
      type(*), dimension(*) :: l3, t3, otri, osgn, vtri, vsgn, gpd, gmq, l2pp, xacc, yall, dr, d1, bufl, buft
      real(c_double) :: stats(4)
    end subroutine

    subroutine N(zc_l3_pass2w)(l3, nto, nvt, no, nv, npo, nvp, otri, osgn, ofree, opair, &
                               vtri, vsgn, vfree, vpair, batch, wl, u, nu, vp, adrv, k, bdrv, r, stats, &
                               tb, z) bind(C)
      import
      integer(c_int64_t), value :: nto, nvt, no, nv, npo, nvp, batch, nu, k
      type(*), dimension(*) :: l3, otri, osgn, ofree, opair, vtri, vsgn, vfree, vpair
      type(*), dimension(*) :: wl, u, vp, adrv, bdrv, r, tb, z
      real(c_double) :: stats(4)
    end subroutine

    subroutine N(zc_lam_pair_expand)(y, out, ny, nx, n) bind(C)
      import
      integer(c_int64_t), value :: ny, nx, n
      type(*), dimension(*) :: y, out
    end subroutine

    subroutine N(zc_lam_pair_expand_tab)(y, out, ny, nx, n, nq, pidx) bind(C)
      import
      integer(c_int64_t), value :: ny, nx, n, nq
      type(*), dimension(*) :: y, out, pidx
    end subroutine

    subroutine N(zc_lam_pair_matvec)(y, w, acc, nx, n) bind(C)
      import
      integer(c_int64_t), value :: nx, n
      type(*), dimension(*) :: y, w, acc
    end subroutine

    subroutine N(zc_lam_pair_matvec_tab)(y, w, acc, nx, n, nqt, q0, nq, pa, pb) bind(C)
      import
      integer(c_int64_t), value :: nx, n, nqt, q0, nq
      type(*), dimension(*) :: y, w, acc, pa, pb
    end subroutine

    subroutine N(zc_gather_vir_half)(t3, nvt, rows, nb, vtri, vsgn, nv, nvp, out) bind(C)
      import
      integer(c_int64_t), value :: nvt, nb, nv, nvp
      type(*), dimension(*) :: t3, rows, vtri, vsgn, out
    end subroutine

    subroutine N(zc_reduce_pair_first_rows)(r, nvt, x, rows, nb, nv, q0, q1, vfree, vpair) bind(C)
      import
      integer(c_int64_t), value :: nvt, nb, nv, q0, q1
      type(*), dimension(*) :: r, x, rows, vfree, vpair
    end subroutine

    ! ---- T3 in blocks (csrc/t3pack.c)
    subroutine N(zc_bgather2)(t3, obase, osig, otri, osgn, npo, vci, vcs, nvp, np, pidx, nd, didx, &
                              nm, midx, qlo, nq, ldp, out) bind(C)
      import
      integer(c_int64_t), value :: npo, nvp, np, nd, nm, qlo, nq, ldp
      type(*), dimension(*) :: t3, obase, osig, otri, osgn, vci, vcs, pidx, didx, midx, out
    end subroutine

    subroutine N(zc_breduce2)(r, rcol, ntg, tbase, tx, tp, tsg, a0, a1, nvt, vfree, vpair, vpgrp, zoff, &
                              npg, nqg, qoff, rowpos, no, z) bind(C)
      import
      integer(c_int64_t), value :: ntg, a0, a1, nvt, npg, no
      type(*), dimension(*) :: r, rcol, tbase, tx, tp, tsg, vfree, vpair, vpgrp, zoff, nqg, qoff, rowpos, z
    end subroutine

    subroutine N(zc_bgather_vir)(t3, obase, vcol, osig, vsig, vtri, vsgn, nvp, nrows, rows, qlo, nq, &
                                 c0, nc, out) bind(C)
      import
      integer(c_int64_t), value :: nvp, nrows, qlo, nq, c0, nc
      type(*), dimension(*) :: t3, obase, vcol, osig, vsig, vtri, vsgn, rows, out
    end subroutine

    subroutine N(zc_breduce_ppl_rows)(r, rbase, rcol, nrows, rows, nvt, a0, a1, vfree, vpair, vspin, &
                                      sc, hoff, q0, q1, nc, coff, x) bind(C)
      import
      integer(c_int64_t), value :: nrows, nvt, a0, a1, sc, hoff, q0, q1, nc, coff
      type(*), dimension(*) :: r, rbase, rcol, rows, vfree, vpair, vspin, x
    end subroutine

    subroutine N(zc_pair_antisym_batch_add)(out, y, n, npair, pa, pb, ncol, x0, x1, alpha) bind(C)
      import
      integer(c_int64_t), value :: n, npair, ncol, x0, x1
      type(*), dimension(*) :: out, y, pa, pb
      ZT :: alpha
    end subroutine

    subroutine N(zc_pair_t1_contract)(out, v, t, nq, n, npair, pidx, no) bind(C)
      import
      integer(c_int64_t), value :: nq, n, npair, no
      type(*), dimension(*) :: out, v, t, pidx
    end subroutine

    subroutine N(zc_add_transposed)(out, a, nr, nc, ldo, ldi, alpha) bind(C)
      import
      integer(c_int64_t), value :: nr, nc, ldo, ldi
      type(*), dimension(*) :: out, a
      ZT :: alpha
    end subroutine

    subroutine N(zc_vvvv_dress)(out, bare, nrow, ncol, pa, pb, ta, lda, ga, nma, nxa, &
                                tb, ldb, gb, nmb, nxb, ct, h, np) bind(C)
      import
      integer(c_int64_t), value :: nrow, ncol, lda, nma, nxa, ldb, nmb, nxb, np
      type(*), dimension(*) :: out, bare, pa, pb, ta, ga, tb, gb, ct, h
    end subroutine

    subroutine N(zc_pair_one_body_add)(w, nq, ncol, pa, pb, pidx, n, f, ldf, scale) bind(C)
      import
      integer(c_int64_t), value :: nq, ncol, n, ldf
      type(*), dimension(*) :: w, pa, pb, pidx, f
      ZT :: scale
    end subroutine

    subroutine N(zc_vsub)(n, a, b, out) bind(C)
      import
      integer(c_int64_t), value :: n
      type(*), dimension(*) :: a, b, out
    end subroutine

    subroutine N(zc_vdot)(n, a, b, res) bind(C)
      import
      integer(c_int64_t), value :: n
      type(*), dimension(*) :: a, b
      ZT :: res
    end subroutine

    ! (x: the addresses of the nk vectors)
    subroutine N(zc_vlincomb)(n, nk, x, c, out) bind(C)
      import
      integer(c_int64_t), value :: n, nk
      integer(c_intptr_t) :: x(*)
      ZT :: c(*)
      type(*), dimension(*) :: out
    end subroutine

    subroutine N(zc_vmdot)(n, nk, a, x, res) bind(C)
      import
      integer(c_int64_t), value :: n, nk
      type(*), dimension(*) :: a
      integer(c_intptr_t) :: x(*)
      ZT :: res(*)
    end subroutine

    subroutine N(zc_tn_reduce)(r, nout, sr, n, k1, k2, b0, b1, c0, c1, x, s1, s2, so, scale) bind(C)
      import
      integer(c_int64_t), value :: nout, sr, b0, b1, c0, c1, s1, s2, so
      integer(c_int), value :: n, k1, k2
      type(*), dimension(*) :: r, x
      real(c_double), value :: scale
    end subroutine

    subroutine N(zc_tn_expand)(r, nout, sr, n, k1, k2, b0, b1, c0, c1, x, s1, s2, so, scale) bind(C)
      import
      integer(c_int64_t), value :: nout, sr, b0, b1, c0, c1, s1, s2, so
      integer(c_int), value :: n, k1, k2
      type(*), dimension(*) :: r, x
      real(c_double), value :: scale
    end subroutine

    subroutine N(zc_tn_rows)(m, ld, rows, sgn, nrows, c0, c1, out, ldo) bind(C)
      import
      integer(c_int64_t), value :: ld, nrows, c0, c1, ldo
      type(*), dimension(*) :: m, rows, sgn, out
    end subroutine

    subroutine N(zc_tn_energy2)(z, r1, r2, ev, eo, n, e) bind(C)
      import
      integer(c_int64_t), value :: n
      type(*), dimension(*) :: z, r1, r2, ev
      real(c_double), value :: eo
      real(c_double) :: e(2)
    end subroutine

    ! ---- (Q) in sectors (csrc/tnpack.c)
    subroutine N(zc_q_sector_energy)(n, kv, modulus, m, ev, eo, kmin, nk, lid_of, lstart, lmem, qmin, nq, &
                                     off22, nc22, off13, nc13, posv, posq, pos3, b1, b2, b3, c1, c2, e) bind(C)
      import
      integer(c_int), value :: n, modulus, m, kmin, nk, qmin, nq
      real(c_double), value :: eo
      type(*), dimension(*) :: kv, ev, lid_of, lstart, lmem, off22, nc22, off13, nc13, posv, posq, pos3
      type(*), dimension(*) :: b1, b2, b3, c1, c2
      real(c_double) :: e(2)
    end subroutine

    ! ---- (Q)_Lambda (csrc/qlambda.c)
    subroutine N(zc_ql_energy)(z2, z3, r1, r2, ev, eo, n, e) bind(C)
      import
      integer(c_int64_t), value :: n
      type(*), dimension(*) :: z2, z3, r1, r2, ev
      real(c_double), value :: eo
      real(c_double) :: e(2)
    end subroutine

    subroutine N(zc_ql_sector_energy)(n, kv, modulus, m, ev, eo, kmin, nk, lid_of, lstart, lmem, qmin, nq, &
                                      off22, nc22, off13, nc13, posv, posq, pos3, b1, b2, b3, b4, c1, c2, c3, &
                                      e) bind(C)
      import
      integer(c_int), value :: n, modulus, m, kmin, nk, qmin, nq
      real(c_double), value :: eo
      type(*), dimension(*) :: kv, ev, lid_of, lstart, lmem, off22, nc22, off13, nc13, posv, posq, pos3
      type(*), dimension(*) :: b1, b2, b3, b4, c1, c2, c3
      real(c_double) :: e(2)
    end subroutine

    subroutine N(zc_rows_gemm)(m, ld, rows, sg, nr, cols, nc, l, na, x) bind(C)
      import
      integer(c_int64_t), value :: ld, nr, nc, na
      type(*), dimension(*) :: m, rows, sg, cols, l, x
    end subroutine

    subroutine N(zc_xv_sector)(n, kv, modulus, tri, nt, ne, qmin, nq, off, nd, posv, posq, m3, out) bind(C)
      import
      integer(c_int), value :: n, modulus, ne, qmin, nq
      integer(c_int64_t), value :: nt
      type(*), dimension(*) :: kv, tri, off, nd, posv, posq, m3, out
    end subroutine
