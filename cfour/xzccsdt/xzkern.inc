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

    ! ---- the passes for the pairs plo <= P < phi alone (the batches of a rank)
    subroutine N(zc_t3_pass1r)(t3, nto, nvt, no, nv, npo, nvp, otri, osgn, vtri, vsgn, batch, plo, phi, &
                               gpd, gmq, ny, gpp, ft, xacc, yall, r1, ra, io, stats, buf) bind(C)
      import
      integer(c_int64_t), value :: nto, nvt, no, nv, npo, nvp, batch, plo, phi, ny
      type(*), dimension(*) :: t3, otri, osgn, vtri, vsgn, gpd, gmq, gpp, ft, xacc, yall, r1, ra, buf
      type(c_ptr), value :: io
      real(c_double) :: stats(4)
    end subroutine

    subroutine N(zc_t3_pass2r)(t3, nto, nvt, no, nv, npo, nvp, otri, osgn, ofree, opair, &
                               vtri, vsgn, vfree, vpair, batch, plo, phi, wring, adrv, bdrv, r, io, stats, &
                               tb, z) bind(C)
      import
      integer(c_int64_t), value :: nto, nvt, no, nv, npo, nvp, batch, plo, phi
      type(*), dimension(*) :: t3, otri, osgn, ofree, opair, vtri, vsgn, vfree, vpair
      type(*), dimension(*) :: wring, adrv, bdrv, r, tb, z
      type(c_ptr), value :: io
      real(c_double) :: stats(4)
    end subroutine

    subroutine N(zc_l3_pass1r)(l3, t3, nto, nvt, no, nv, npo, nvp, otri, osgn, vtri, vsgn, batch, plo, phi, &
                               gpd, gmq, ny, l2pp, xacc, yall, dr, d1, stats, bufl, buft) bind(C)
      import
      integer(c_int64_t), value :: nto, nvt, no, nv, npo, nvp, batch, plo, phi, ny
      type(*), dimension(*) :: l3, t3, otri, osgn, vtri, vsgn, gpd, gmq, l2pp, xacc, yall, dr, d1, bufl, buft
      real(c_double) :: stats(4)
    end subroutine

    subroutine N(zc_l3_pass2r)(l3, nto, nvt, no, nv, npo, nvp, otri, osgn, ofree, opair, &
                               vtri, vsgn, vfree, vpair, batch, plo, phi, wl, u, nu, vp, adrv, k, bdrv, r, stats, &
                               tb, z) bind(C)
      import
      integer(c_int64_t), value :: nto, nvt, no, nv, npo, nvp, batch, plo, phi, nu, k
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

    ! ---- vectors of the EOM solver (csrc/t3pack.c): the first m of the n
    !      vectors x (addresses) become sum_j x[j] q[j][k]; x[I][A] /= omega +
    !      eo[I] - ev[A], denominators below floor replaced by it
    subroutine N(zc_vtransform)(len, n, m, x, q) bind(C)
      import
      integer(c_int64_t), value :: len, n, m
      integer(c_intptr_t) :: x(*)
      ZT :: q(*)
    end subroutine

    ! ---- the Krylov solver of the Lambda equations (xzlkry.inc):
    !      res[i][j] = sum conj(x1[i]) x2[j] for the vectors (addresses) x1, x2
    subroutine N(zc_vgram)(n, m1, x1, m2, x2, res) bind(C)
      import
      integer(c_int64_t), value :: n, m1, m2
      integer(c_intptr_t) :: x1(*), x2(*)
      ZT :: res(*)
    end subroutine

    subroutine N(zc_vprecond)(x, ni, na, eo, ev, omega, floor) bind(C)
      import
      integer(c_int64_t), value :: ni, na
      type(*), dimension(*) :: x
      real(c_double) :: eo(*), ev(*)
      ZT :: omega
      real(c_double), value :: floor
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

    ! ---- CC3 (docs/native.md, section 17; docs/ccn.md, sections 9 and 10)
    ! R[I][A] += (ev[A] - eo[I]) T3[I][A]: the diagonal Fock operator (csrc/ccsdt_t3.c)
    subroutine N(zc_t3_fdiag)(t3, r, nto, nvt, eo, ev) bind(C)
      import
      integer(c_int64_t), value :: nto, nvt
      type(*), dimension(*) :: t3, r, eo, ev
    end subroutine

    ! the drive terms of the triples equation for the pairs plo <= P < phi (csrc/cc3_native.c)
    subroutine N(zc_t3_driver)(nto, nvt, no, nv, npo, nvp, ofree, opair, vfree, vpair, batch, plo, phi, &
                               adrv, bdrv, r, stats, z) bind(C)
      import
      integer(c_int64_t), value :: nto, nvt, no, nv, npo, nvp, batch, plo, phi
      type(*), dimension(*) :: ofree, opair, vfree, vpair, adrv, bdrv, r, z
      real(c_double) :: stats(4)
    end subroutine

    ! the same with label shifts of the triples, the operands and the contraction operands (csrc/cc3_shift.c)
    subroutine N(zc_cc3_batchs)(no, nv, nq, na, ncls, orng, vrng, qoff, aoff, vtri, vsgn, vfree, vpair, &
                                eo, ev3, omega, i0, i1, j, k, nset, sa, sw, st, sg, at1, ao1, bv1, bo1, &
                                at2, ao2, bv2, bo2, gv, gsq, gsqp, fov, ysq, xo, r1, rasq, work, stats) bind(C)
      import
      integer(c_int64_t), value :: no, nv, nq, na, ncls, i0, i1, j, k, nset, st, sg
      integer(c_int64_t) :: sa(2), sw(2)
      real(c_double), value :: omega
      type(*), dimension(*) :: orng, vrng, qoff, aoff, vtri, vsgn, vfree, vpair, eo, ev3
      type(*), dimension(*) :: at1, ao1, bv1, bo1, at2, ao2, bv2, bo2, gv, gsq, gsqp, fov
      type(*), dimension(*) :: ysq, xo, r1, rasq, work
      real(c_double) :: stats(4)
    end subroutine

    ! ---- CC4 without stored quadruples (csrc/cc4_native.c; docs/native.md, section 19)
    subroutine N(zc_cc4_divide)(r, ev, eo, n) bind(C)
      import
      integer(c_int64_t), value :: n
      type(*), dimension(*) :: r, ev
      real(c_double), value :: eo
    end subroutine

    subroutine N(zc_cc4_sector_t4)(n, kv, modulus, m, ev, eo, kmin, nk, lid_of, lstart, lmem, qmin, nq, &
                                   off22, nc22, off13, nc13, posv, posq, pos3, b, c, e22, e13) bind(C)
      import
      integer(c_int), value :: n, modulus, m, kmin, nk, qmin, nq
      real(c_double), value :: eo
      type(*), dimension(*) :: kv, ev, lid_of, lstart, lmem, off22, nc22, off13, nc13, posv, posq, pos3
      type(*), dimension(*) :: b, c, e22, e13
    end subroutine

    subroutine N(zc_cc4_red12)(r, tri, nt, n, kv, modulus, qmin, nqg, off, nq, posv, posq, y) bind(C)
      import
      integer(c_int64_t), value :: nt
      integer(c_int), value :: n, modulus, qmin, nqg
      type(*), dimension(*) :: r, tri, kv, off, nq, posv, posq, y
    end subroutine

    subroutine N(zc_cc4_rows_gemm)(m, ld, rows, sg, nr, nc, l, na, alpha, x) bind(C)
      import
      integer(c_int64_t), value :: ld, nr, nc, na
      real(c_double), value :: alpha
      type(*), dimension(*) :: m, rows, sg, l, x
    end subroutine

    subroutine N(zc_cc4_addrows)(r, ld, tgt, nr, nc, x) bind(C)
      import
      integer(c_int64_t), value :: ld, nr, nc
      type(*), dimension(*) :: r, tgt, x
    end subroutine

    ! the same in place (b and c become the amplitudes; the places of no quadruple zero)
    subroutine N(zc_cc4_sector_t4i)(n, kv, modulus, m, ev, eo, kmin, nk, lid_of, lstart, lmem, qmin, nq, &
                                    off22, nc22, off13, nc13, posv, posq, pos3, b, c) bind(C)
      import
      integer(c_int), value :: n, modulus, m, kmin, nk, qmin, nq
      real(c_double), value :: eo
      type(*), dimension(*) :: kv, ev, lid_of, lstart, lmem, off22, nc22, off13, nc13, posv, posq, pos3
      type(*), dimension(*) :: b, c
    end subroutine

    ! x[a][c] = [x[a][c] +] sum_r l[r][a] row[r][c], the rows by their addresses (init 1: without the bracket)
    subroutine N(zc_cc4_rowsmm)(row, l, nr, na, nc, init, x) bind(C)
      import
      integer(c_int64_t), value :: nr, na, nc, init
      integer(c_intptr_t) :: row(*)
      type(*), dimension(*) :: l, x
    end subroutine

    ! row[r][c] += sum_a l[r][a] e[a][c], the rows by their addresses
    subroutine N(zc_cc4_addmm)(row, l, nr, na, nc, e) bind(C)
      import
      integer(c_int64_t), value :: nr, na, nc
      integer(c_intptr_t) :: row(*)
      type(*), dimension(*) :: l, e
    end subroutine

    ! one batch of occupied triples of CC3 without stored triples (csrc/cc3_fly.c)
    subroutine N(zc_cc3_batch)(no, nv, nq, na, ncls, orng, vrng, qoff, aoff, vtri, vsgn, vfree, vpair, &
                               eo, ev3, omega, i0, i1, j, k, nset, at1, ao1, bv1, bo1, at2, ao2, bv2, bo2, &
                               gv, gsq, gsqp, fov, ysq, xo, r1, rasq, work, stats) bind(C)
      import
      integer(c_int64_t), value :: no, nv, nq, na, ncls, i0, i1, j, k, nset
      real(c_double), value :: omega
      type(*), dimension(*) :: orng, vrng, qoff, aoff, vtri, vsgn, vfree, vpair, eo, ev3
      type(*), dimension(*) :: at1, ao1, bv1, bo1, at2, ao2, bv2, bo2, gv, gsq, gsqp, fov
      type(*), dimension(*) :: ysq, xo, r1, rasq, work
      real(c_double) :: stats(4)
    end subroutine

    ! ---- the blocks of a symmetry label of the T4 equation (csrc/t4sym.c), used by EOM-DEA:
    !      a block of rows of a quadruple / triple list expanded into the splits (a | bcd),
    !      (ab | cd), (a | bc) of a product (mode 0), or reduced back from it (mode 1)
    subroutine N(zc_t4s_x13)(mode, r, ldr, quads, na, nent, row, sgn, xoff, s1, s2, ncls, n, kv, posv, pos3, x) bind(C)
      import
      integer(c_int), value :: mode, ncls, n
      integer(c_int64_t), value :: ldr, na, nent
      type(*), dimension(*) :: r, quads, row, sgn, xoff, s1, s2, kv, posv, pos3, x
    end subroutine

    subroutine N(zc_t4s_x22)(mode, r, ldr, quads, na, nent, row, sgn, xoff, s1, s2, nq, n, kv, posq, x) bind(C)
      import
      integer(c_int), value :: mode, nq, n
      integer(c_int64_t), value :: ldr, na, nent
      type(*), dimension(*) :: r, quads, row, sgn, xoff, s1, s2, kv, posq, x
    end subroutine

    subroutine N(zc_t4s_x12)(mode, r, ldr, tri, nk, nent, row, sgn, xoff, s1, s2, ncls, n, kv, posv, posq, x) bind(C)
      import
      integer(c_int), value :: mode, ncls, n
      integer(c_int64_t), value :: ldr, nk, nent
      type(*), dimension(*) :: r, tri, row, sgn, xoff, s1, s2, kv, posv, posq, x
    end subroutine

    ! ---- rank 5 (EOM-DEA with complete 5p3h): a block of rows of a quintuple list expanded
    !      into the splits (ab | cde), (a | bcde) of a product (mode 0), or reduced back (mode 1)
    subroutine N(zc_t5s_x23)(mode, r, ldr, quints, na, nent, row, sgn, xoff, s1, s2, nq, n, kv, posq, pos3, x) bind(C)
      import
      integer(c_int), value :: mode, nq, n
      integer(c_int64_t), value :: ldr, na, nent
      type(*), dimension(*) :: r, quints, row, sgn, xoff, s1, s2, kv, posq, pos3, x
    end subroutine

    subroutine N(zc_t5s_x14)(mode, r, ldr, quints, na, nent, row, sgn, xoff, s1, s2, ncls, n, kv, posv, pos4, x) bind(C)
      import
      integer(c_int), value :: mode, ncls, n
      integer(c_int64_t), value :: ldr, na, nent
      type(*), dimension(*) :: r, quints, row, sgn, xoff, s1, s2, kv, posv, pos4, x
    end subroutine
