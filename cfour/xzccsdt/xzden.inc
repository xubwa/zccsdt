!
! The one-particle density and the virtual-occupied orbital gradient of CCSDT
! from the amplitudes and lambda in the work array (type-generic), and the
! files that CFOUR's xx2czvec reads to solve the Z-vector equations and to
! write the relaxed density (docs/native.md, sections 10 and 13).  The
! equations are those of src/zccsdt/cc/density_opt.py (kernel; docs/lambda.md,
! section 10; in blocks of the labels density_blk.py): T1-dressed integrals,
! nothing conjugated, every block of the two-particle density but the one
! with four virtual indices made and contracted with its dressed integrals.
!
!   passes over the packed arrays (l3 and T3 as the Lambda equations hold them):
!     A  l3 with t2 on both sides (Z, C3) and with T3 (the ring density R3,
!        l2 with T3): pass 1 of a Lambda iteration with other operands
!     B  T3 with l2, Z (one side) and l2, C3 (the other), with l1 and l2:
!        pass 1 of an amplitude iteration with other operands
!     C  l3 with T3 over the occupied triple and one virtual index, between
!        two operands on virtual pairs: t2 (the term l3 t2 t3 of the block
!        ijab) and the dressed <ef||gi> (the l3 t3 part of the block abcd).
!        The matrix Gamma[(e<f)][(g<h)] of density_opt (v^4 / 4) is never
!        formed: (operand . Lv) Tv^T batch by batch, as the ladder of the
!        Lambda equations does with Xq and X2
!     the Gram matrix of l3 and T3 (the l3 t3 part of the block ijkl)
!   then everything else on dense arrays with pairs in the order of their
!   groups; the dressed <ab||de> is made in the place of the bare one (after
!   this stage the work array has no bare <ab||de>).
!
! Batch sizes: nbp (the half-expanded batches of the passes A and B), nbt
! (the two gathered batches of pass C), nba (the bra-dressed <ab||ei>).
! Timers: set 4.  Frozen and dropped spinors: xzdfrz.inc, xzfrz.f90.
!
subroutine N(xzden)(w)
  use xzdata
  use xzmem
  use xzkern
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8) :: m0, m1, m2, ov, ov2, nth
  integer(8) :: kden, kgnh, kfoo, kfov, kfvo, kfvv, kpvv, kooovd, kooood, kovvod, kovood, kk
  integer(8) :: kt2p, kt2pp, ktau, ktt, kl2p, kl2pp, kd, kvvo, kvvoo, kc3, kz, kgpd, kgmq, kxacc, kyall
  integer(8) :: kdr, kb1, kr1c, ka1, kkg, ks13, ko4, kgoo3, kl1t, kg2, kyb, kwkl, kgq, kgh, kpib, ka2
  integer(8) :: kbl, kbt, klv, ktv, ka, knn, kgr, kgab, kgij, kgia, ktr, kla, krl, ky4, kdk, kzu, kt4
  integer(8) :: kdab, kzai, ksa, kw3, ku, kmq, kg1, kh
  integer(8) :: h, lo, nqh, i0, nb, g, ni, na, kgam, kdh, kxa, kxall, kym, nf, nd
  logical :: dense, held
  real(8) :: stats(4)
  character(len=256) :: dbg
  integer :: ldbg
  ZT :: one, zero, mone
  one = 1
  zero = 0
  mone = -1
  ov = no * nv
  ov2 = ov * ov
  dense = ncls == 1
  nth = omp_get_max_threads()
  tcur = 4
  held = nfrozen + ndropped > 0 .and. .not. den_active
  nf = merge(nfrozen, 0_8, held)
  nd = merge(ndropped, 0_8, held)
  m0 = xzmark()
  ! ---- what stays to the end of the stage
  kden = xzget(NW * ny * ny, 'response density')
  kgnh = xzget(NW * ov, 'orbital gradient')
  kfoo = xzget(NW * no * no, 'F_oo')
  kfov = xzget(NW * ov, 'F_ov')
  kfvo = xzget(NW * ov, 'F_vo')
  kfvv = xzget(NW * nv * nv, 'F_vv')
  kooovd = xzget(NW * no * no * ov, 'dressed <oo||ov>')
  kooood = xzget(NW * no ** 4, 'dressed <oo||oo>')
  kovvod = xzget(NW * ov2, 'dressed <ov||vo>')
  kovood = xzget(NW * no * no * ov, 'dressed <ov||oo>')
  kt2p = xzget(NW * no * no * nvp, 'T2 [i][j][Q]')
  kt2pp = xzget(NW * npo * nvp, 'T2 [P][Q]')
  ktau = xzget(NW * npo * nvp, 'tau1')
  kl2p = xzget(NW * no * no * nvp, 'L2 [i][j][Q]')
  kl2pp = xzget(NW * npo * nvp, 'L2 [P][Q]')
  kd = xzget(NW * ov * nvp, 'dressed <ov||vv>')
  kvvo = xzget(NW * nvp * ov, 'dressed <vv||vo>')
  kvvoo = xzget(NW * npo * nvp, 'dressed <vv||oo>')
  kc3 = xzget(NW * no * nvp * nv, 'L3 with T2: C3')
  kz = xzget(NW * npo * ov, 'L3 with T2: Z')
  kxacc = xzget(NW * ny * no * nvp, 'pass result X')
  kyall = xzget(NW * npo * nv * ny, 'pass result Y')
  kdr = xzget(NW * ov2, 'ring density of L3 with T3')
  kb1 = xzget(NW * ov, 'density of L2 with T3')
  ka1 = xzget(NW * npo * nvp, 'L1 with T3')
  kkg = xzget(NW * npo * nvp, 'L3 with T2 and T3')
  ks13 = xzget(NW * ov, 'L3 with T3 and <vv||vo>')
  ko4 = xzget(NW * npo * npo, 'Gram matrix on pairs')
  kgoo3 = xzget(NW * no * no, 'G_oo of L3')
  kgpd = xzget(NW * ny * npo * nv, 'pass operand')
  ! with held spinors: gam[Q][Q'] = sum_{I,d} l3[I, d, Q] t3[I, d, Q'] in its diagonal pair blocks
  kgam = 0
  if (held) kgam = xzget(NW * nvv, 'L3 with T3 on virtual pairs')

  ! ---- T1 dressing; the bra-dressed <ab||ei> and <ab||ij>
  m1 = xzmark()
  kpvv = xzget(NW * nv * nv, 'F_vv work')
  kk = xzget(NW * ov2, 'work of the dressing')
  ktt = xzget(NW * npo * nvp, 'T2 + tau1')
  kg2 = xzget(NW * npo * ov, 'work [P][e][i]')
  kyb = xzget(NW * nba * nv * nv * no, 'batch of the bra dressing')
  kwkl = xzget(NW * npo * npo, 'work [K][P]')
  if (.not. xzdry) then
    call xztic()
    call N(xzfock)(no, nv, nvp, w(jeps), w(jt1), w(jooov), w(joovv), w(jovvo), w(jovvq), &
                   w(jvpa), w(jvpb), w(kfoo), w(kfov), w(kfvo), w(kfvv), w(kpvv))
    call N(xzdroo)(no, nv, w(jt1), w(jooov), w(joovv), w(joooo), w(kooovd), w(kooood))
    call N(xzdrovvq)(no, nv, nvp, w(jt1), w(jovvq), w(joovvp), w(kd))
    call N(xzdrov)(no, nv, nvp, w(jt1), w(jooov), w(jovvo), w(jovvq), w(kd), w(kooood), &
                   w(jvpidx), w(kovvod), w(kovood), w(kk))
    call N(xzt2forms)(no, nv, npo, nvp, w(jt1), w(jt2), w(jopa), w(jopb), w(jvpa), w(jvpb), &
                      w(kt2p), w(kt2pp), w(ktau), w(ktt))
    call N(xzl2forms)(no, nv, npo, nvp, w(jl2), w(jopa), w(jopb), w(jvpa), w(jvpb), w(kl2p), w(kl2pp))
    call N(xzbv)(no, nv, npo, nvp, nba, w(jt1), w(ktau), w(jovvq), w(jooov), w(jovvo), &
                 w(jopa), w(jopb), w(jvpa), w(jvpb), w(kvvo), w(kg2), w(kyb))
    call N(xzvvoo)(no, nv, npo, nvp, w(jt1), w(ktau), w(jgpp), w(jooov), w(joooo), w(kvvo), &
                   w(jopa), w(jopb), w(jvpa), w(jvpb), w(kvvoo), w(kwkl))
    call xztoc(4, 'T1 dressing')
  end if
  call xzrel(m1)

  ! ---- the dressed <ab||de> in the place of the bare one, block by block;
  !      its parts of the dressed <ab||ei> and <ab||ij>
  m1 = xzmark()
  ka2 = xzget(NW * ov, 'work of the dressing')
  kgq = jovvq
  kgh = jgpp
  kpib = jvpidx
  if (.not. dense) then
    kgq = xzget(NW * ov * nqmax, '<ov||vv> on the pairs of a group')
    kgh = xzget(NW * npo * nqmax, '<oo||vv> on the pairs of a group')
    kpib = xzget(nv * nv, 'pairs of a group')
  end if
  if (.not. xzdry) then
    do h = 0, npg - 1
      lo = vpoff(h)
      nqh = vpoff(h + 1) - lo
      if (nqh == 0) cycle
      if (.not. dense) call N(xzgrpcols)(no, nv, npo, nvp, lo, nqh, w(jovvq), w(jgpp), w(jvpidx), &
                                         w(kgq), w(kgh), w(kpib))
      call N(xzdrrows)(no, nv, npo, nvp, nqh, nqh, w(jt1), w(ktau + NW * lo), w(jvvvv + NW * vvoff(h)), &
                       w(kgq), w(kgh), w(jvpa + lo), w(jvpb + lo), w(jvvvv + NW * vvoff(h)), w(ka2))
      ! <ab||ei> += sum_f ~<ab||ef> t1[i][f];  <ab||ij>[P][Q] += sum_Q' ~<Q||Q'> tau1[P][Q']
      call N(zc_pair_t1_contract)(w(kvvo + NW * lo * ov), w(jvvvv + NW * vvoff(h)), w(jt1), nqh, nv, nqh, &
                                  w(kpib), no)
      if (npo > 0) call N(zc_gemm)(0_4, 1_4, npo, nqh, nqh, one, w(ktau + NW * lo), nvp, &
                                   w(jvvvv + NW * vvoff(h)), nqh, one, w(kvvoo + NW * lo), nvp)
    end do
    call xztoc(4, 'dressed <ab||de> in place')
  end if
  call xzrel(m1)

  ! ---- pass A: l3 with t2 (C3, Z) and with T3 (the ring density), l2 with T3
  m1 = xzmark()
  kgmq = xzget(NW * no * nvp * ny, 'pass operand G')
  if (.not. xzdry) then
    call N(xzdopa)(no, nv, npo, nvp, w(jt2), w(kt2p), w(jopa), w(jopb), w(kgpd), w(kgmq))
    call zc_vzero_d(NW * ny * no * nvp, w(kxacc))
    call zc_vzero_d(NW * ov2, w(kdr))
    call zc_vzero_d(NW * ov, w(kb1))
    call xztoc(4, 'operands of the passes')
  end if
  m2 = xzmark()
  if (dense) then
    kbl = xzget(NW * nbp * ov * nvp, 'pass-A batch of L3')
    kbt = xzget(NW * nbp * ov * nvp, 'pass-A batch of T3')
    if (.not. xzdry) then
      stats = 0
      call N(zc_l3_pass1w)(w(jl3), w(jt3), nto, nvt, no, nv, npo, nvp, w(jotri), w(josgn), w(jvtri), &
                           w(jvsgn), nbp, w(kgpd), w(kgmq), ny, w(kl2pp), w(kxacc), w(kyall), w(kdr), &
                           w(kb1), stats, w(kbl), w(kbt))
      call xztoc(4, 'pass A: L3, ring density')
      call xztpart(4, 'pass 1: gather', stats(1))
      call xztpart(4, 'pass 1: stacked GEMMs', stats(2))
      call xztpart(4, 'pass 1: density', stats(3))
    end if
  else
    call N(xzlbpass1)(w, jl3, jt3, w(kgpd), w(kgmq), w(kl2pp), w(kxacc), w(kyall), w(kdr), w(kb1))
  end if
  call xzrel(m2)

  ! ---- pass B: T3 with l2 and Z, with l2 and C3, with l1
  kl1t = xzget(NW * ov, 'L1 transposed')
  kr1c = xzget(NW * ov, 'pass work')
  if (.not. xzdry) then
    call xztic()
    call N(xzdouta)(no, nv, npo, nvp, w(kxacc), w(kyall), w(kc3), w(kz))
    call N(xzdopb)(no, nv, npo, nvp, w(jl1), w(jl2), w(kl2p), w(kz), w(kc3), w(jopa), w(jopb), w(kgpd), &
                   w(kgmq), w(kl1t))
    call xztoc(4, 'operands of the passes')
  end if
  m2 = xzmark()
  if (dense) then
    kbt = xzget(NW * nbp * ov * nvp, 'pass-B batch of T3')
    if (.not. xzdry) then
      call zc_vzero_d(NW * ny * no * nvp, w(kxacc))
      call zc_vzero_d(NW * ov, w(kr1c))
      stats = 0
      call N(zc_t3_pass1w)(w(jt3), nto, nvt, no, nv, npo, nvp, w(jotri), w(josgn), w(jvtri), w(jvsgn), &
                           nbp, w(kgpd), w(kgmq), ny, w(kl2pp), w(kl1t), w(kxacc), w(kyall), w(kr1c), &
                           w(ka1), c_null_ptr, stats, w(kbt))
    end if
  else
    call N(xzbpass1)(w, jt3, w(kgpd), w(kgmq), w(kl2pp), w(jl1), w(kxacc), w(kyall), w(kr1c), w(ka1))
  end if
  if (.not. xzdry) call xztoc(4, 'pass B: T3')
  call xzrel(m1)

  ! ---- pass C: (t2 . Lv) Tv^T and (<ef||gi> . Lv) Tv^T
  m1 = xzmark()
  knn = xzget(NW * ov * nvp, 'L3 with T3 and <vv||vo>, on pairs')
  if (.not. xzdry) then
    call xztic()
    call zc_vzero_d(NW * npo * nvp, w(kkg))
    call zc_vzero_d(NW * ov * nvp, w(knn))
    if (held) call zc_vzero_d(NW * nvv, w(kgam))
  end if
  if (dense) then
    klv = xzget(NW * nvp * nbt * nv, 'pass-C batch of L3')
    ktv = xzget(NW * nvp * nbt * nv, 'pass-C batch of T3')
    ka = xzget(NW * max(npo, ov) * nbt * nv, 'pass-C product')
    if (.not. xzdry) then
      do i0 = 0, nto - 1, nbt
        nb = min(nbt, nto - i0)
        call N(zc_gather_vir_half)(w(jl3), nvt, w(jrows + i0), nb, w(jvtri), w(jvsgn), nv, nvp, w(klv))
        call N(zc_gather_vir_half)(w(jt3), nvt, w(jrows + i0), nb, w(jvtri), w(jvsgn), nv, nvp, w(ktv))
        call xztoc(4, 'pass C: gather')
        if (npo > 0) then
          call N(zc_gemm)(0_4, 0_4, npo, nb * nv, nvp, one, w(kt2pp), nvp, w(klv), nb * nv, zero, &
                          w(ka), nb * nv)
          call N(zc_gemm)(0_4, 1_4, npo, nvp, nb * nv, one, w(ka), nb * nv, w(ktv), nb * nv, one, &
                          w(kkg), nvp)
        end if
        call N(zc_gemm)(1_4, 0_4, ov, nb * nv, nvp, one, w(kvvo), ov, w(klv), nb * nv, zero, w(ka), nb * nv)
        call N(zc_gemm)(0_4, 1_4, ov, nvp, nb * nv, one, w(ka), nb * nv, w(ktv), nb * nv, one, w(knn), nvp)
        if (held) call N(zc_gemm)(0_4, 1_4, nvp, nvp, nb * nv, one, w(klv), nb * nv, w(ktv), nb * nv, one, &
                                  w(kgam), nvp)
        call xztoc(4, 'pass C: GEMMs')
      end do
    end if
  else
    call N(xzdbpassc)(w, jl3, jt3, w(kt2pp), w(kvvo), w(kkg), w(knn), held, w(max(kgam, 1_8)))
  end if
  if (.not. xzdry) then
    call N(xzdfold)(no, nv, nvp, w(jvpa), w(jvpb), w(knn), w(ks13))
    call xztoc(4, 'pass C: fold')
  end if
  call xzrel(m1)

  ! ---- the Gram matrix of l3 and T3 on occupied pairs
  m1 = xzmark()
  kgr = xzget(NW * nto * nto, 'Gram matrix')
  if (.not. xzdry) then
    if (dense) then
      call N(zc_gemm)(0_4, 1_4, nto, nto, nvt, one, w(jl3), nvt, w(jt3), nvt, zero, w(kgr), nto)
    else
      call zc_vzero_d(NW * nto * nto, w(kgr))
      do g = 0, ntg - 1
        ni = otoff(g + 1) - otoff(g)
        na = vtoff(g + 1) - vtoff(g)
        if (ni == 0 .or. na == 0) cycle
        call N(zc_gemm)(0_4, 1_4, ni, ni, na, one, w(jl3 + NW * boff(g)), na, w(jt3 + NW * boff(g)), na, &
                        zero, w(kgr + NW * (otoff(g) * nto + otoff(g))), nto)
      end do
    end if
    call N(xzlgramd)(no, npo, nto, w(jotri), w(josgn), w(kgr), w(ko4), w(kgoo3))
    call xztoc(4, 'Gram matrix')
  end if
  call xzrel(m1)

  ! ---- the one-particle density
  m1 = xzmark()
  kgab = xzget(NW * nv * nv, 'g_ab')
  kgij = xzget(NW * no * no, 'g_ij')
  kgia = xzget(NW * ov, 'g_ia')
  kg1 = xzget(NW * ny * ny, 'work of the density')
  kh = xzget(NW * ny * ny, 'work of the density')
  if (.not. xzdry) then
    call N(xzdg1)(no, nv, npo, nvp, w(jt1), w(jt2), w(jl1), w(jl2), w(kt2p), w(kl2p), w(kdr), w(kb1), &
                  w(kz), w(jopa), w(jopb), w(kgab), w(kgij), w(kgia), w(kg1), w(kh), w(kden))
    call xztoc(4, 'one-particle density')
  end if

  ! ---- the blocks of the two-particle density
  ktr = xzget(NW * ov2, 'T2 [(a, i)][(e, m)]')
  kla = xzget(NW * ov2, 'L2 [(a, j)][(e, m)]')
  krl = xzget(NW * ov2, 'ring density of L2 with T2')
  ky4 = xzget(NW * npo * npo, 'block ijkl')
  kdk = xzget(NW * no * no * ov, 'block ijak')
  kdab = xzget(NW * npo * nvp, 'block ijab')
  m2 = xzmark()
  kzu = xzget(NW * no * no * ov, 'Z on both orders of its pair')
  kt4 = xzget(NW * no * no * ov, 'work of the block ijak')
  kzai = xzget(NW * ov * npo, 'Z [(a, i)][P]')
  ksa = xzget(NW * nth * 2 * ov * nv, 'work of the block aibc, per thread')
  ku = xzget(NW * npo * max(nv, ov), 'work [P][a]')
  if (.not. xzdry) then
    call N(xzdblk)(no, nv, npo, nvp, nth, w(jt2), w(jl1), w(jl2), w(kt2p), w(kt2pp), w(kl2pp), w(kdr), &
                   w(ko4), w(kc3), w(kz), w(kxacc), w(kyall), w(ka1), w(kkg), w(kgab), w(kgij), &
                   w(jopa), w(jopb), w(jopidx), w(jvpa), w(jvpb), w(jvpidx), w(ktr), w(kla), w(krl), &
                   w(ky4), w(kdk), w(kdab), w(kzu), w(kt4), w(kzai), w(ksa), w(ku))
    call xztoc(4, 'blocks of the 2-density')
  end if
  call xzrel(m2)

  ! ---- the orbital gradient
  kmq = xzget(NW * nvp * ov, 'work [Q][e][i]')
  if (.not. xzdry) then
    call N(xzdgrad)(no, nv, npo, nvp, w(jt2), w(jl1), w(kfoo), w(kfov), w(kfvo), w(kfvv), w(kooovd), &
                    w(kooood), w(kovvod), w(kovood), w(kd), w(kvvo), w(kvvoo), w(kl2pp), w(kc3), w(kz), &
                    w(kxacc), w(kdr), w(krl), w(ky4), w(kdk), w(kdab), w(kgab), w(kgij), w(kgia), &
                    w(ks13), w(jopa), w(jopb), w(jopidx), w(jvpa), w(jvpb), w(kla), w(kmq), w(ku), w(kgnh))
    call xztoc(4, 'gradient: small terms')
    ! the block aibc with the dressed <ab||de>
    do h = 0, npg - 1
      lo = vpoff(h)
      nqh = vpoff(h + 1) - lo
      if (nqh == 0) cycle
      call N(zc_gemm)(0_4, 1_4, nqh, ov, nqh, one, w(jvvvv + NW * vvoff(h)), nqh, w(kxacc + NW * lo), nvp, &
                      zero, w(kmq + NW * lo * ov), ov)
    end do
    call N(xzdbra)(no, nv, nvp, w(jvpa), w(jvpb), w(kmq), mone, w(kgnh))
    call xztoc(4, 'gradient: <ab||de>')
    call xzoptget('ddebug', dbg, ldbg)
    if (ldbg > 0) then
      call xzdumpw(dbg(1:ldbg) // '.d1', NW * ny * ny, w(kden))
      call xzdumpw(dbg(1:ldbg) // '.gnh', NW * ov, w(kgnh))
    end if
  end if

  ! ---- the Hermitian unrelaxed density and the right-hand side of the
  !      Z-vector equations between the active spinors
  kdh = xzget(NW * ny * ny, 'Hermitian density')
  kxa = xzget(NW * ov, 'right-hand side of the Z-vector equations')
  kxall = xzget(NW * ntot * ntot, 'right-hand side, rows and columns of held spinors')
  kym = xzget(NW * ntot * ntot, 'rotations with held spinors')
  if (.not. xzdry) then
    call N(xzdxact)(no, nv, w(jeps), w(joovv), w(jovvo), w(kden), w(kgnh), w(kdh), w(kxa))
    call xztoc(4, 'files of xx2czvec')
  end if
  ! ---- frozen and dropped spinors
  if (held) call N(xzdfrz)(w, kden, kdh, kxa, kxall, kym, kl2pp, kl2p, kt2pp, kc3, kz, kxacc, krl, ky4, kdk, &
                           kdab, kgam)
  ka = xzget(2 * max(ny * ny, (nv + nd) * (no + nf)), 'record of a file')
  if (.not. xzdry) then
    call xztic()
    call N(xzdwrite)(no, nv, nf, nd, ntot, w(kdh), w(kxa), w(kxall), w(kym), operm, vperm, w(ka))
    call xztoc(4, 'files of xx2czvec')
  end if
  call xzrel(m0)
end subroutine

! ---------------------------------------------------------------------------
! The stage of the held spinors (xzdfrz.inc): sweep 1, the gradient and the
! rotations, sweep 2; on return x of the active spinors has the Fock matrix
! of the rotations, xall(p, q) the elements of the right-hand side with a
! held index (full order of xzfrz.f90) and ym(p, q) the rotations y[p][q].
subroutine N(xzdfrz)(w, kden, kdh, kxa, kxall, kym, kl2pp, kl2p, kt2pp, kc3, kz, kdai, krl, ky4, kdk, kdab, kgam)
  use xzdata
  use xzmem
  use xzkern
  use xzfrzd
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kden, kdh, kxa, kxall, kym, kl2pp, kl2p, kt2pp, kc3, kz, kdai, krl, ky4, kdk, kdab, kgam
  integer(8) :: m0, kxx, kxd, ktt, ktqo, ktqv, kus, kmm, km1, kbf, kkf, kgf, kfo, kfm, kfy, h, k, n2, hp, j0, j1
  integer :: ldbg
  m0 = xzmark()
  call xzfrzinit(w)
  n2 = nall * nall
  kxx = xzget(NW * nall ** 3, 'integrals of a held spinor')
  kxd = xzget(NW * ny ** 3, 'dressed integrals of a held spinor')
  ktt = xzget(NW * ny ** 3, 'dressed <h a||b c>')
  ktqo = xzget(NW * nvp * no, '<h m||Q>')
  ktqv = xzget(NW * nvp * nv, '<h e||Q>')
  kus = xzget(NW * npo * nv, 'work [P][e]')
  kmm = xzget(NW * 5 * n2, 'densities of the Fock matrices')
  km1 = xzget(NW * n2, 'density of sweep 2')
  kbf = xzget(NW * ny * nheld, 'bf')
  kkf = xzget(NW * ny * nheld, 'kf')
  kgf = xzget(NW * ny * nheld, 'gradient between held and active spinors')
  kfo = xzget(NW * 5 * nall * nheld, 'Fock matrices, rows of held spinors')
  kfm = xzget(NW * nall * nheld, 'Fock matrix of sweep 2')
  kfy = xzget(NW * no * nv, 'Fock matrix of the rotations')
  if (.not. xzdry) then
    call xztic()
    call N(xzdfrzm)(nall, no, nv, w(jt1), w(kden), w(kdh), w(kmm))
    do h = 0, nheld - 1
      call xzfrzone(w, h)
      call N(xzdfrzx)(nall, w(jfcf), w(jfwb), w(kxx))
      call xztoc(4, 'held: integrals')
      do k = 0, 4
        call N(xzdfockh)(nall, w(kxx), w(kmm + NW * k * n2), w(kfo + NW * (h * 5 + k) * nall))
      end do
      call N(xzdfrzdress)(nall, no, nv, .false., w(kxx), w(jt1), w(kxd), w(ktt))
      call N(xzdfrzq)(no, nv, nvp, w(jvpa), w(jvpb), w(ktt), w(ktqo), w(ktqv))
      call N(xzdfrzb)(no, nv, npo, nvp, w(ktt), w(ktqo), w(ktqv), w(jt1), w(jl2), w(kt2pp), w(kc3), w(kz), &
                      w(kdai), w(krl), w(ky4), w(kdk), w(kdab), w(kgam), w(jopa), w(jopb), w(jopidx), &
                      w(jvpa), w(jvpb), w(jvpidx), w(kus), w(kbf + NW * h * ny))
      call N(xzdfrzdress)(nall, no, nv, .true., w(kxx), w(jt1), w(kxd), w(ktt))
      call N(xzdfrzq)(no, nv, nvp, w(jvpa), w(jvpb), w(ktt), w(ktqo), w(ktqv))
      call N(xzdfrzk)(no, nv, npo, nvp, w(ktt), w(ktqo), w(ktqv), w(jt1), w(jt2), w(kl2pp), w(kl2p), w(kc3), &
                      w(kz), w(kdai), w(krl), w(ky4), w(kdk), w(kdab), w(kgam), w(jopa), w(jopb), w(jopidx), &
                      w(jvpa), w(jvpb), w(jvpidx), w(kus), w(kkf + NW * h * ny))
      call xztoc(4, 'held: sweep 1')
    end do
    call xzoptget('ddebug', length=ldbg)
    if (ldbg > 0) then
      call xzdumpw('xd.bf', NW * ny * nheld, w(kbf))
      call xzdumpw('xd.kf', NW * ny * nheld, w(kkf))
    end if
    call N(xzdfrzg)(nall, no, nv, nfrozen, ndropped, w(jfe), w(jt1), w(kden), w(kdh), w(kbf), w(kkf), w(kfo), &
                    w(kgf), w(kym), w(km1))
    call zc_vzero_d(NW * no * nv, w(kfy))
    do h = 0, nheld - 1
      call xzfrzone(w, h)
      call N(xzdfrzx)(nall, w(jfcf), w(jfwb), w(kxx))
      call xztoc(4, 'held: integrals')
      call N(xzdfockh)(nall, w(kxx), w(km1), w(kfm + NW * h * nall))
      hp = ny + h
      j0 = 0
      j1 = no - 1
      if (h >= nfrozen) then
        j0 = no
        j1 = ny - 1
      end if
      call N(xzdfrzy)(nall, no, nv, hp, j0, j1, w(kxx), w(kym), w(kfy))
      call xztoc(4, 'held: sweep 2')
    end do
    call N(xzdfrzfin)(nall, no, nv, nfrozen, ndropped, w(kgf), w(kfo), w(kfm), w(kfy), w(kxa), w(kxall))
  end if
  call xzrel(m0)
end subroutine

! x between the active spinors gains the Fock matrix of the rotations; the
! elements with a held index (p: active virtual or dropped, q: active
! occupied or frozen; full order):
!   x[d][i] = Gf[d][i] + F[m1][d][i]            x[d][f] = V[d][f] + F[m1][d][f]
!   x[a][f] = - conj(Gf[f][a]) + conj(F[m1][f][a])
subroutine N(xzdfrzfin)(nall, no, nv, nf, nd, gf, fo, fm, fy, xa, xall)
  implicit none
  integer(8), intent(in) :: nall, no, nv, nf, nd
  ZT, intent(in) :: gf(0:no+nv-1, 0:nf+nd-1), fo(0:nall-1, 5, 0:nf+nd-1), fm(0:nall-1, 0:nf+nd-1)
  ZT, intent(in) :: fy(0:nv-1, 0:no-1)
  ZT, intent(inout) :: xa(0:nv-1, 0:no-1)
  ZT, intent(out) :: xall(0:nall-1, 0:nall-1)
  integer(8) :: n, d, f, i, a, hd
  n = no + nv
  xa = xa + fy
  xall = 0
  do d = 0, nd - 1
    hd = nf + d
    do i = 0, no - 1
      xall(n + hd, i) = gf(i, hd) + fm(i, hd)
    end do
    do f = 0, nf - 1
      xall(n + hd, n + f) = fo(n + f, 5, hd) + fm(n + f, hd)
    end do
  end do
  do f = 0, nf - 1
    do a = 0, nv - 1
      xall(no + a, n + f) = -CONJ(gf(no + a, f)) + CONJ(fm(no + a, f))
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! Pass C in the blocks of the labels: for the triples I of a group g and the
! class sc of the single virtual index d, h = g - sc the group of the pairs,
! with Lv[Q][I][d] = l3[I, d, Q] and Tv likewise:
!   kg[P][Q']      += sum_{Q,I,d} t2pp[P][Q] Lv[Q][I][d] Tv[Q'][I][d]     (P of the group h)
!   nn[(a,i)][Q']  += sum_{Q,I,d} vvo[Q][a][i] Lv[Q][I][d] Tv[Q'][I][d]   (s(a) + s(i) = h)
subroutine N(xzdbpassc)(w, kl3, kt3, t2pp, vvo, kg, nn, held, gam)
  use xzdata
  use xzmem
  use xzkern
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: kl3, kt3
  ZT, intent(in) :: t2pp(0:nvp-1, 0:npo-1)
  ZT, intent(in) :: vvo(0:no-1, 0:nv-1, 0:nvp-1)               ! [Q][a][i]
  ZT :: kg(0:nvp-1, 0:npo-1), nn(0:nvp-1, 0:no-1, 0:nv-1)      ! [P][Q], [a][i][Q]
  logical, intent(in) :: held
  ZT :: gam(0:*)                                               ! with held spinors: sum Lv Tv^T per block
  integer(8) :: g, sc, h, nc, nqh, qlo, i0, nb, mk, ms, klv, ktv, ka, kv, kx, plo, np, se, si, ne, nj
  ZT :: one, zero
  one = 1
  zero = 0
  do g = 0, ntg - 1
    if (vtoff(g + 1) == vtoff(g)) cycle
    do i0 = otoff(g), otoff(g + 1) - 1, nbt
      nb = min(nbt, otoff(g + 1) - i0)
      do sc = 0, ncls - 1
        h = xzw(g - sc)
        if (h < 0 .or. h >= npg) cycle
        nc = vcoff(sc + 1) - vcoff(sc)
        qlo = vpoff(h)
        nqh = vpoff(h + 1) - qlo
        if (nc == 0 .or. nqh == 0) cycle
        plo = opoff(h)
        np = opoff(h + 1) - plo
        mk = xzmark()
        klv = xzget(NW * nqh * nb * nc, 'pass-C batch of L3')
        ktv = xzget(NW * nqh * nb * nc, 'pass-C batch of T3')
        ka = xzget(NW * np * nb * nc, 'pass-C product')
        if (.not. xzdry) then
          call N(zc_bgather_vir)(w(kl3), w(jobase), w(jvcol), w(josig), w(jvsig), w(jvtri), w(jvsgn), nvp, &
                                 nb, w(jrows + i0), qlo, nqh, vcoff(sc), nc, w(klv))
          call N(zc_bgather_vir)(w(kt3), w(jobase), w(jvcol), w(josig), w(jvsig), w(jvtri), w(jvsgn), nvp, &
                                 nb, w(jrows + i0), qlo, nqh, vcoff(sc), nc, w(ktv))
          call xztoc(4, 'pass C: gather')
          if (held) call N(zc_gemm)(0_4, 1_4, nqh, nqh, nb * nc, one, w(klv), nb * nc, w(ktv), nb * nc, one, &
                                    gam(vvoff(h)), nqh)
          if (np > 0) then
            call N(zc_gemm)(0_4, 0_4, np, nb * nc, nqh, one, t2pp(qlo, plo), nvp, w(klv), nb * nc, zero, &
                            w(ka), nb * nc)
            call N(zc_gemm)(0_4, 1_4, np, nqh, nb * nc, one, w(ka), nb * nc, w(ktv), nb * nc, one, &
                            kg(qlo, plo), nvp)
          end if
        end if
        do se = 0, ncls - 1
          si = xzw(h - se)
          if (si < 0 .or. si >= ncls) cycle
          ne = vcoff(se + 1) - vcoff(se)
          nj = ocoff(si + 1) - ocoff(si)
          if (ne == 0 .or. nj == 0) cycle
          ms = xzmark()
          kv = xzget(NW * ne * nj * nqh, '<ef||gi> of a block')
          ka = xzget(NW * ne * nj * nb * nc, 'pass-C product')
          kx = xzget(NW * ne * nj * nqh, 'pass-C result')
          if (.not. xzdry) then
            call N(xzdbvq)(no, nv, nvp, ne, nj, nqh, vcoff(se), ocoff(si), qlo, vvo, w(kv))
            call N(zc_gemm)(0_4, 0_4, ne * nj, nb * nc, nqh, one, w(kv), nqh, w(klv), nb * nc, zero, &
                            w(ka), nb * nc)
            call N(zc_gemm)(0_4, 1_4, ne * nj, nqh, nb * nc, one, w(ka), nb * nc, w(ktv), nb * nc, zero, &
                            w(kx), nqh)
            call N(xzdbnadd)(no, nv, nvp, ne, nj, nqh, vcoff(se), ocoff(si), qlo, w(kx), nn)
          end if
          call xzrel(ms)
        end do
        if (.not. xzdry) call xztoc(4, 'pass C: GEMMs')
        call xzrel(mk)
      end do
    end do
  end do
end subroutine

! v[e][i][Q] = vvo[qlo + Q][e0 + e][j0 + i]
subroutine N(xzdbvq)(no, nv, nvp, ne, nj, nq, e0, j0, qlo, vvo, v)
  implicit none
  integer(8), intent(in) :: no, nv, nvp, ne, nj, nq, e0, j0, qlo
  ZT, intent(in) :: vvo(0:no-1, 0:nv-1, 0:nvp-1)
  ZT, intent(out) :: v(0:nq-1, 0:nj-1, 0:ne-1)
  integer(8) :: e, i, q
  !$omp parallel do private(e, i, q)
  do e = 0, ne - 1
    do i = 0, nj - 1
      do q = 0, nq - 1
        v(q, i, e) = vvo(j0 + i, e0 + e, qlo + q)
      end do
    end do
  end do
end subroutine

! nn[e0 + e][j0 + i][qlo + Q] += x[e][i][Q]
subroutine N(xzdbnadd)(no, nv, nvp, ne, nj, nq, e0, j0, qlo, x, nn)
  implicit none
  integer(8), intent(in) :: no, nv, nvp, ne, nj, nq, e0, j0, qlo
  ZT, intent(in) :: x(0:nq-1, 0:nj-1, 0:ne-1)
  ZT :: nn(0:nvp-1, 0:no-1, 0:nv-1)
  integer(8) :: e, i
  !$omp parallel do private(e, i)
  do e = 0, ne - 1
    do i = 0, nj - 1
      nn(qlo:qlo + nq - 1, j0 + i, e0 + e) = nn(qlo:qlo + nq - 1, j0 + i, e0 + e) + x(:, i, e)
    end do
  end do
end subroutine

! ===========================================================================
! The typed parts: loops over elements.

! The operands of pass A (the first rows and columns of those of pass 1 of
! the Lambda equations; the others are zero):
!   gpd[f][P][e] = t2[P][e][f],   gmq[m][Q][n] = t2[m][n][Q]
subroutine N(xzdopa)(no, nv, npo, nvp, t2, t2p, opa, opb, gpd, gmq)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1), t2p(0:nvp-1, 0:no-1, 0:no-1)
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1)
  ZT, intent(out) :: gpd(0:nv-1, 0:npo-1, 0:nv+no-1)           ! [x][P][e]
  ZT, intent(out) :: gmq(0:no+nv-1, 0:nvp-1, 0:no-1)           ! [m][Q][y]
  integer(8) :: p, e, f, m, q, n
  !$omp parallel do private(p, e, f)
  do f = 0, nv + no - 1
    do p = 0, npo - 1
      if (f < nv) then
        do e = 0, nv - 1
          gpd(e, p, f) = t2(f, e, opb(p), opa(p))
        end do
      else
        gpd(:, p, f) = 0
      end if
    end do
  end do
  !$omp parallel do private(m, q, n)
  do m = 0, no - 1
    do q = 0, nvp - 1
      do n = 0, no - 1
        gmq(n, q, m) = t2p(q, n, m)
      end do
      gmq(no:no + nv - 1, q, m) = 0
    end do
  end do
end subroutine

! From the results of pass A (density_opt: C3[m][Q][a] = sum_{no,e} l3[m,n,o,Q,e]
! t2[n,o,a,e], Z[P][g][i] = sum_{m,Q} l3[m,P,g,Q] t2[i,m,Q]):
!   c3[m][Q][a] = -2 xacc[a][m][Q],   z[P][g][i] = - yall[P][g][i]
subroutine N(xzdouta)(no, nv, npo, nvp, xacc, yall, c3, z)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: xacc(0:nvp-1, 0:no-1, 0:nv+no-1)           ! [x][i][Q]
  ZT, intent(in) :: yall(0:no+nv-1, 0:nv-1, 0:npo-1)           ! [P][d][y]
  ZT, intent(out) :: c3(0:nv-1, 0:nvp-1, 0:no-1)               ! [m][Q][a]
  ZT, intent(out) :: z(0:no-1, 0:nv-1, 0:npo-1)                ! [P][g][i]
  integer(8) :: m, q, a, p, g
  !$omp parallel do private(m, q, a)
  do m = 0, no - 1
    do q = 0, nvp - 1
      do a = 0, nv - 1
        c3(a, q, m) = -2d0 * xacc(q, m, a)
      end do
    end do
  end do
  !$omp parallel do private(p, g)
  do p = 0, npo - 1
    do g = 0, nv - 1
      z(:, g, p) = -yall(0:no - 1, g, p)
    end do
  end do
end subroutine

! The operands of pass B:
!   gpd[a][P][d] = - l2[P][d][a],  gpd[nv+i][P][d] = 2 Z[P][d][i]
!   gmq[m][Q][k] = l2[k][m][Q],    gmq[m][Q][no+a] = C3[m][Q][a]
!   l1t[e][m] = l1[m][e]
subroutine N(xzdopb)(no, nv, npo, nvp, l1, l2, l2p, z, c3, opa, opb, gpd, gmq, l1t)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: l1(0:nv-1, 0:no-1), l2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1), l2p(0:nvp-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: z(0:no-1, 0:nv-1, 0:npo-1), c3(0:nv-1, 0:nvp-1, 0:no-1)
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1)
  ZT, intent(out) :: gpd(0:nv-1, 0:npo-1, 0:nv+no-1)           ! [x][P][d]
  ZT, intent(out) :: gmq(0:no+nv-1, 0:nvp-1, 0:no-1)           ! [m][Q][y]
  ZT, intent(out) :: l1t(0:no-1, 0:nv-1)
  integer(8) :: p, d, a, i, m, q, k
  !$omp parallel do private(p, d, a, i)
  do p = 0, npo - 1
    do a = 0, nv - 1
      do d = 0, nv - 1
        gpd(d, p, a) = -l2(a, d, opb(p), opa(p))
      end do
    end do
    do i = 0, no - 1
      do d = 0, nv - 1
        gpd(d, p, nv + i) = 2d0 * z(i, d, p)
      end do
    end do
  end do
  !$omp parallel do private(m, q, k)
  do m = 0, no - 1
    do q = 0, nvp - 1
      do k = 0, no - 1
        gmq(k, q, m) = l2p(q, m, k)
      end do
      gmq(no:no + nv - 1, q, m) = c3(:, q, m)
    end do
  end do
  do m = 0, no - 1
    do a = 0, nv - 1
      l1t(m, a) = l1(a, m)
    end do
  end do
end subroutine

! s13[b][y] = sum_a nn[(a, y)][(a b)], the pair with its sign: nn on pairs
! Q' = (x<x') gives s13[x'][y] += nn[(x, y)][Q'], s13[x][y] -= nn[(x', y)][Q']
subroutine N(xzdfold)(no, nv, nvp, vpa, vpb, nn, s13)
  implicit none
  integer(8), intent(in) :: no, nv, nvp
  integer(8), intent(in) :: vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(in) :: nn(0:nvp-1, 0:no-1, 0:nv-1)                ! [a][i][Q]
  ZT, intent(out) :: s13(0:no-1, 0:nv-1)                       ! [b][y]
  integer(8) :: q, y
  s13 = 0
  do q = 0, nvp - 1
    do y = 0, no - 1
      s13(y, vpb(q)) = s13(y, vpb(q)) + nn(q, y, vpa(q))
      s13(y, vpa(q)) = s13(y, vpa(q)) - nn(q, y, vpb(q))
    end do
  end do
end subroutine

! g[x][i] += sign mq[Q][y][i], g[y][i] -= sign mq[Q][x][i] for the pairs Q = (x<y):
! the contraction of an array antisymmetric in two virtual indices, given on
! pairs and already contracted with everything but the second of them
subroutine N(xzdbra)(no, nv, nvp, vpa, vpb, mq, sign, g)
  implicit none
  integer(8), intent(in) :: no, nv, nvp
  integer(8), intent(in) :: vpa(0:nvp-1), vpb(0:nvp-1)
  ZT, intent(in) :: mq(0:no-1, 0:nv-1, 0:nvp-1)                ! [Q][e][i]
  ZT, intent(in) :: sign
  ZT, intent(inout) :: g(0:no-1, 0:nv-1)                       ! [a][i]
  integer(8) :: q
  do q = 0, nvp - 1
    g(:, vpa(q)) = g(:, vpa(q)) + sign * mq(:, vpb(q), q)
    g(:, vpb(q)) = g(:, vpb(q)) - sign * mq(:, vpa(q), q)
  end do
end subroutine

! ---------------------------------------------------------------------------
! The coefficients of the dressed Fock matrix in the Lagrangian and the
! response density (density_opt.kernel, "the blocks of the densities"):
!   g_ab[a][b] = 1/2 sum l2[m][n][a][e] t2[m][n][b][e] + 1/3 sum_m R3[a][m][b][m]
!   g_ij[i][j] = - sum l2[j][m][Q] t2[i][m][Q] - 1/3 sum_e R3[e][j][e][i]
!   g_ia[i][a] = sum l1[m][e] t2[i][m][a][e] + B1[a][i] + sum_{P,g} Z[P][g][i] t2[P][a][g]
!   d1 = (1 - t1) (g1 + n) (1 + t1),  g1 = [[g_ij, g_ia], [l1^T, g_ab]]
! den[p][q] = d1[p][q] = <p+ q>, spinors [occupied, virtual] in the order here.
subroutine N(xzdg1)(no, nv, npo, nvp, t1, t2, l1, l2, t2p, l2p, dr, b1, z, opa, opb, gab, gij, gia, g1, hh, den)
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: t1(0:nv-1, 0:no-1), l1(0:nv-1, 0:no-1)
  ZT, intent(in) :: t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1), l2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: t2p(0:nvp-1, 0:no-1, 0:no-1), l2p(0:nvp-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: dr(0:no-1, 0:nv-1, 0:no-1, 0:nv-1)         ! R3[a][j][b][i]
  ZT, intent(in) :: b1(0:no-1, 0:nv-1)                         ! [a][i]
  ZT, intent(in) :: z(0:no-1, 0:nv-1, 0:npo-1)                 ! [P][g][i]
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1)
  ZT, intent(out) :: gab(0:nv-1, 0:nv-1), gij(0:no-1, 0:no-1), gia(0:nv-1, 0:no-1)
  ZT :: g1(0:no+nv-1, 0:no+nv-1), hh(0:no+nv-1, 0:no+nv-1)     ! work, (p, q) = [p][q]
  ZT, intent(out) :: den(0:no+nv-1, 0:no+nv-1)                 ! den(q, p) = d1[p][q]
  integer(8) :: a, b, i, j, m, e, q, p, g, n
  ZT :: s
  n = no + nv
  !$omp parallel do collapse(2) private(a, b, i, j, e, m, s)
  do a = 0, nv - 1
    do b = 0, nv - 1
      s = 0
      do i = 0, no - 1
        do j = 0, no - 1
          do e = 0, nv - 1
            s = s + l2(e, a, j, i) * t2(e, b, j, i)
          end do
        end do
      end do
      s = 0.5d0 * s
      do m = 0, no - 1
        s = s + dr(m, b, m, a) / 3d0
      end do
      gab(b, a) = s
    end do
  end do
  !$omp parallel do collapse(2) private(i, j, m, q, e, s)
  do i = 0, no - 1
    do j = 0, no - 1
      s = 0
      do m = 0, no - 1
        do q = 0, nvp - 1
          s = s - l2p(q, m, j) * t2p(q, m, i)
        end do
      end do
      do e = 0, nv - 1
        s = s - dr(i, e, j, e) / 3d0
      end do
      gij(j, i) = s
    end do
  end do
  !$omp parallel do collapse(2) private(i, a, m, e, p, g, s)
  do i = 0, no - 1
    do a = 0, nv - 1
      s = b1(i, a)
      do m = 0, no - 1
        do e = 0, nv - 1
          s = s + l1(e, m) * t2(e, a, m, i)
        end do
      end do
      do p = 0, npo - 1
        do g = 0, nv - 1
          s = s + z(i, g, p) * t2(g, a, opb(p), opa(p))
        end do
      end do
      gia(a, i) = s
    end do
  end do
  ! g1 + n, as (p, q)
  g1 = 0
  do i = 0, no - 1
    do j = 0, no - 1
      g1(i, j) = gij(j, i)
    end do
    g1(i, i) = g1(i, i) + 1d0
    do a = 0, nv - 1
      g1(i, no + a) = gia(a, i)
      g1(no + a, i) = l1(a, i)
    end do
  end do
  do a = 0, nv - 1
    do b = 0, nv - 1
      g1(no + a, no + b) = gab(b, a)
    end do
  end do
  ! (1 - t1) from the left: rows i lose sum_a t1[i][a] (row a)
  hh = g1
  do i = 0, no - 1
    do a = 0, nv - 1
      hh(i, :) = hh(i, :) - t1(a, i) * g1(no + a, :)
    end do
  end do
  ! (1 + t1) from the right: columns b gain sum_i (column i) t1[i][b]
  g1 = hh
  do b = 0, nv - 1
    do i = 0, no - 1
      g1(:, no + b) = g1(:, no + b) + hh(:, i) * t1(b, i)
    end do
  end do
  do p = 0, n - 1
    do q = 0, n - 1
      den(q, p) = g1(p, q)
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The blocks of the two-particle density (density_opt.kernel, "the blocks of
! the densities"), in the layouts their contractions take:
!   tr[(a, i)][(e, m)] = t2[i][m][a][e],  la[(a, j)][(e, m)] = l2[j][m][a][e]
!   rl[(a, j)][(b, i)] on return: Rg = sum_{m,e} l2[j][m][a][e] t2[i][m][b][e] + R3
!   dr on return: unchanged (R3)
!   y4[P][P']   the block ijkl: sum_Q l2[P][Q] t2[P'][Q] + O4[P][P']
!   dk[i][j][a][k]  the block ijak, antisymmetric in (i, j)
!   dab[P][Q]   the block ijab
!   xacc rows a < nv on return: the block aibc, Dai[a][i][Q] (on entry E2 of pass B)
subroutine N(xzdblk)(no, nv, npo, nvp, nth, t2, l1, l2, t2p, t2pp, l2pp, dr, do4, c3, z, xacc, yall, a1, &
                     kg, gab, gij, opa, opb, opidx, vpa, vpb, vpidx, tr, la, rl, y4, dk, dab, zu, t4, zai, &
                     sa, u)
  use xzkern
  use omp_lib
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp, nth
  ZT, intent(in) :: t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1), l2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)
  ZT, intent(in) :: l1(0:nv-1, 0:no-1)
  ZT, intent(in) :: t2p(0:nvp-1, 0:no-1, 0:no-1), t2pp(0:nvp-1, 0:npo-1), l2pp(0:nvp-1, 0:npo-1)
  ZT :: dr(0:no-1, 0:nv-1, 0:no-1, 0:nv-1)                     ! R3[a][j][b][i]
  ZT, intent(in) :: do4(0:npo-1, 0:npo-1)                      ! 6 O4[P][P']
  ZT, intent(in) :: c3(0:nv-1, 0:nvp-1, 0:no-1), z(0:no-1, 0:nv-1, 0:npo-1)
  ZT :: xacc(0:nvp-1, 0:no-1, 0:nv+no-1)                       ! E2[a][i][Q] | X10[i][j][Q]
  ZT, intent(in) :: yall(0:no+nv-1, 0:nv-1, 0:npo-1)           ! Y2k[P][d][k] | Y12[P][d][a]
  ZT, intent(in) :: a1(0:nvp-1, 0:npo-1), kg(0:nvp-1, 0:npo-1)
  ZT, intent(in) :: gab(0:nv-1, 0:nv-1), gij(0:no-1, 0:no-1)
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), opidx(0:no-1, 0:no-1)
  integer(8), intent(in) :: vpa(0:nvp-1), vpb(0:nvp-1), vpidx(0:nv-1, 0:nv-1)
  ZT :: tr(0:no-1, 0:nv-1, 0:no-1, 0:nv-1), la(0:no-1, 0:nv-1, 0:no-1, 0:nv-1)
  ZT :: rl(0:no-1, 0:nv-1, 0:no-1, 0:nv-1)
  ZT, intent(out) :: y4(0:npo-1, 0:npo-1)
  ZT, intent(out) :: dk(0:no-1, 0:nv-1, 0:no-1, 0:no-1)        ! [i][j][a][k]
  ZT, intent(out) :: dab(0:nvp-1, 0:npo-1)
  ZT :: zu(0:no-1, 0:no-1, 0:no-1, 0:nv-1)                     ! [(e, m)][(k, j)] = Zu[k][m][e][j]
  ZT :: t4(0:no-1, 0:no-1, 0:no-1, 0:nv-1)                     ! [(a, i)][(k, j)] = T4[i][j][a][k]
  ZT :: zai(0:npo-1, 0:no-1, 0:nv-1)                           ! [a][i][P]
  ZT :: sa(0:nv-1, 0:no-1, 0:nv-1, 0:1, 0:nth-1)               ! per thread: Ca[(e, m)][c], Sa[(b, i)][c]
  ZT :: u(0:nv-1, 0:npo-1)                                     ! [P][a]
  integer(8) :: ov, a, b, c, e, f, i, j, k, m, n, p, q, me, pq
  ZT :: s, one, zero, mone, half, sg
  one = 1
  zero = 0
  mone = -1
  half = 0.5d0
  ov = no * nv
  ! ---- the ring layouts of t2 and l2, and Rl2 = la tr^T
  !$omp parallel do collapse(2) private(a, i, e, m)
  do a = 0, nv - 1
    do i = 0, no - 1
      do e = 0, nv - 1
        do m = 0, no - 1
          tr(m, e, i, a) = t2(e, a, m, i)
          la(m, e, i, a) = l2(e, a, m, i)
        end do
      end do
    end do
  end do
  call N(zc_gemm)(0_4, 1_4, ov, ov, ov, one, la, ov, tr, ov, zero, rl, ov)
  ! ---- ijkl
  if (npo > 0) call N(zc_gemm)(0_4, 1_4, npo, npo, nvp, one, l2pp, nvp, t2pp, nvp, zero, y4, npo)
  y4 = y4 + do4 / 6d0
  ! ---- ijak: on i<j first, then antisymmetrized, with T4
  !      Dk[P][a][k] = - sum_e l1[k][e] t2[P][a][e] - Y2k[P][a][k] - 1/2 sum_Q t2[P][Q] C3[k][Q][a]
  do k = 0, no - 1
    if (npo > 0) call N(zc_gemm)(0_4, 0_4, npo, nv, nvp, one, t2pp, nvp, c3(0, 0, k), nv, zero, u, nv)
    !$omp parallel do private(p, a, e, s)
    do p = 0, npo - 1
      do a = 0, nv - 1
        s = -yall(k, a, p) - 0.5d0 * u(a, p)
        do e = 0, nv - 1
          s = s - l1(e, k) * t2(e, a, opb(p), opa(p))
        end do
        dk(k, a, opb(p), opa(p)) = s
      end do
    end do
  end do
  do i = 0, no - 1
    dk(:, :, i, i) = 0
    do j = i + 1, no - 1
      dk(:, :, i, j) = -dk(:, :, j, i)
    end do
  end do
  ! Zu[k][m][e][j] = Z[(k m)][e][j] with its sign, as [(e, m)][(k, j)]
  !$omp parallel do collapse(2) private(e, m, k, j)
  do e = 0, nv - 1
    do m = 0, no - 1
      do k = 0, no - 1
        if (k < m) then
          zu(:, k, m, e) = z(:, e, opidx(m, k))
        else if (k > m) then
          zu(:, k, m, e) = -z(:, e, opidx(k, m))
        else
          zu(:, k, m, e) = 0
        end if
      end do
    end do
  end do
  ! T4[i][j][a][k] = sum_{m,e} t2[i][m][a][e] Zu[k][m][e][j]
  call N(zc_gemm)(0_4, 0_4, ov, no * no, ov, one, tr, ov, zu, no * no, zero, t4, no * no)
  !$omp parallel do collapse(2) private(i, j, a, k)
  do i = 0, no - 1
    do j = 0, no - 1
      do a = 0, nv - 1
        do k = 0, no - 1
          dk(k, a, j, i) = dk(k, a, j, i) + t4(j, k, i, a) - t4(i, k, j, a)
        end do
      end do
    end do
  end do
  ! ---- ijab
  !      A1 + t2 + Y4^T t2 + (L3 t2 T3) - A(ab) [t2 g_ab + 1/2 Y12] - A(ij) [g_ij t2 + 1/2 X10]
  !      + A(ij) A(ab) sum_{f,n} (1/2 Rl2 + R3)[f][n][a][i] t2[j][n][b][f]
  dab = a1 + t2pp + kg
  if (npo > 0) call N(zc_gemm)(1_4, 0_4, npo, nvp, npo, one, y4, npo, t2pp, nvp, one, dab, nvp)
  ! (1/2 Rl2 + R3 in the place of R3; w3 in the place of la)
  call N(xzdaxpy)(ov * ov, half, rl, dr)
  call N(zc_gemm)(1_4, 1_4, ov, ov, ov, one, dr, ov, tr, ov, zero, la, ov)
  !$omp parallel do private(p, q, i, j, a, b, e, n, s)
  do p = 0, npo - 1
    i = opa(p)
    j = opb(p)
    do q = 0, nvp - 1
      a = vpa(q)
      b = vpb(q)
      s = 0
      do e = 0, nv - 1
        s = s - t2(e, a, j, i) * gab(b, e) + t2(e, b, j, i) * gab(a, e)
      end do
      s = s - 0.5d0 * (yall(no + a, b, p) - yall(no + b, a, p))
      do n = 0, no - 1
        s = s - gij(n, i) * t2p(q, n, j) + gij(n, j) * t2p(q, n, i)
      end do
      s = s - 0.5d0 * (xacc(q, j, nv + i) - xacc(q, i, nv + j))
      s = s + la(j, b, i, a) - la(i, b, j, a) - la(j, a, i, b) + la(i, a, j, b)
      dab(q, p) = dab(q, p) + s
    end do
  end do
  ! Rg = Rl2 + R3 in rl; R3 back in dr
  call N(xzdaxpy)(ov * ov, -half, rl, dr)
  call N(xzdaxpy)(ov * ov, one, dr, rl)
  ! ---- aibc, in the place of E2:
  !      - E2 - sum_m l1[m][a] t2[i][m][Q] - sum_P Z[P][a][i] t2[P][Q] + 1/2 A(bc) S[a][i][b][c],
  !      S[a][i][b][c] = sum_{m,e} t2[i][m][b][e] C3[m][(a e)][c]
  !$omp parallel do collapse(2) private(a, i, m, p)
  do a = 0, nv - 1
    do i = 0, no - 1
      xacc(:, i, a) = -xacc(:, i, a)
      do m = 0, no - 1
        xacc(:, i, a) = xacc(:, i, a) - l1(a, m) * t2p(:, m, i)
      end do
      do p = 0, npo - 1
        zai(p, i, a) = z(i, a, p)
      end do
    end do
  end do
  if (npo > 0) call N(zc_gemm)(0_4, 0_4, ov, nvp, npo, mone, zai, npo, t2pp, nvp, one, xacc, nvp)
  if (nvp > 0) then
    !$omp parallel do private(a, me, e, m, c, sg, pq, q, b, i) schedule(dynamic)
    do a = 0, nv - 1
      me = omp_get_thread_num()
      do e = 0, nv - 1
        if (e == a) then
          sa(:, :, e, 0, me) = 0
          cycle
        end if
        if (a < e) then
          pq = vpidx(e, a)
          sg = 1
        else
          pq = vpidx(a, e)
          sg = -1
        end if
        do m = 0, no - 1
          sa(:, m, e, 0, me) = sg * c3(:, pq, m)
        end do
      end do
      call N(zc_gemm)(0_4, 0_4, ov, nv, ov, one, tr, ov, sa(0, 0, 0, 0, me), nv, zero, sa(0, 0, 0, 1, me), nv)
      do i = 0, no - 1
        do q = 0, nvp - 1
          b = vpa(q)
          c = vpb(q)
          xacc(q, i, a) = xacc(q, i, a) + 0.5d0 * (sa(c, i, b, 1, me) - sa(b, i, c, 1, me))
        end do
      end do
    end do
  end if
end subroutine

! y += alpha x
subroutine N(xzdaxpy)(n, alpha, x, y)
  implicit none
  integer(8), intent(in) :: n
  ZT, intent(in) :: alpha, x(n)
  ZT, intent(inout) :: y(n)
  integer(8) :: i
  !$omp parallel do
  do i = 1, n
    y(i) = y(i) + alpha * x(i)
  end do
end subroutine

! ---------------------------------------------------------------------------
! The orbital gradient Gnh[a][i] (density_opt.kernel, "the gradient"; the
! terms in its order) but for the product of the block aibc with the dressed
! <ab||de>, which the caller adds.  rl: Rg[a][j][b][i]; dai: the block aibc
! (the first nv rows of the result of pass B); dc, mq, u: work.
subroutine N(xzdgrad)(no, nv, npo, nvp, t2, l1, foo, fov, fvo, fvv, ooovd, ooood, ovvod, ovood, d, vvo, &
                      vvoo, l2pp, c3, z, dai, dr, rl, y4, dk, dab, gab, gij, gia, s13, opa, opb, opidx, &
                      vpa, vpb, dc, mq, u, g)
  use xzkern
  implicit none
  integer(8), intent(in) :: no, nv, npo, nvp
  ZT, intent(in) :: t2(0:nv-1, 0:nv-1, 0:no-1, 0:no-1), l1(0:nv-1, 0:no-1)
  ZT, intent(in) :: foo(0:no-1, 0:no-1), fov(0:nv-1, 0:no-1), fvo(0:no-1, 0:nv-1), fvv(0:nv-1, 0:nv-1)
  ZT, intent(in) :: ooovd(0:nv-1, 0:no-1, 0:no-1, 0:no-1)      ! ~<mn||ie> [m][n][i][e]
  ZT, intent(in) :: ooood(0:no-1, 0:no-1, 0:no-1, 0:no-1)      ! ~<mn||ij>
  ZT, intent(in) :: ovvod(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)      ! ~<mb||ej>
  ZT, intent(in) :: ovood(0:no-1, 0:no-1, 0:nv-1, 0:no-1)      ! ~<mb||ij>
  ZT, intent(in) :: d(0:nvp-1, 0:nv-1, 0:no-1)                 ! ~<mb||Q>
  ZT, intent(in) :: vvo(0:no-1, 0:nv-1, 0:nvp-1)               ! ~<Q||gi> [Q][g][i]
  ZT, intent(in) :: vvoo(0:nvp-1, 0:npo-1)                     ! ~<Q||P> [P][Q]
  ZT, intent(in) :: l2pp(0:nvp-1, 0:npo-1)
  ZT, intent(in) :: c3(0:nv-1, 0:nvp-1, 0:no-1), z(0:no-1, 0:nv-1, 0:npo-1)
  ZT, intent(in) :: dai(0:nvp-1, 0:no-1, 0:nv-1)               ! [a][i][Q]
  ZT, intent(in) :: dr(*)
  ZT, intent(in) :: rl(0:no-1, 0:nv-1, 0:no-1, 0:nv-1)         ! Rg[a][j][b][i]
  ZT, intent(in) :: y4(0:npo-1, 0:npo-1)                       ! [P][P']
  ZT, intent(in) :: dk(0:no-1, 0:nv-1, 0:no-1, 0:no-1)         ! [i][j][a][k]
  ZT, intent(in) :: dab(0:nvp-1, 0:npo-1)
  ZT, intent(in) :: gab(0:nv-1, 0:nv-1), gij(0:no-1, 0:no-1), gia(0:nv-1, 0:no-1)
  ZT, intent(in) :: s13(0:no-1, 0:nv-1)
  integer(8), intent(in) :: opa(0:npo-1), opb(0:npo-1), opidx(0:no-1, 0:no-1), vpa(0:nvp-1), vpb(0:nvp-1)
  ZT :: dc(0:no-1, 0:nv-1, 0:no-1, 0:nv-1)                     ! [(e, i)][(g, j)]
  ZT :: mq(0:no-1, 0:nv-1, 0:nvp-1)                            ! [Q][e][i]
  ZT :: u(0:no-1, 0:nv-1, 0:npo-1)                             ! [P][g][i]
  ZT, intent(out) :: g(0:no-1, 0:nv-1)                         ! [a][i]
  integer(8) :: ov, a, b, e, i, j, k, m, n, p, q, x, y, pp
  ZT :: s, one, zero, v, sg
  one = 1
  zero = 0
  ov = no * nv
  ! ---- the terms with the one-particle coefficients
  !$omp parallel do collapse(2) private(a, i, j, b, k, s)
  do a = 0, nv - 1
    do i = 0, no - 1
      s = fvo(i, a)
      do j = 0, no - 1
        s = s + fvo(j, a) * gij(j, i) - gia(a, j) * foo(i, j)
        do k = 0, no - 1
          s = s + ovood(i, k, a, j) * gij(k, j)
        end do
        do b = 0, nv - 1
          s = s + ovvod(i, b, a, j) * gia(b, j)
        end do
      end do
      do b = 0, nv - 1
        s = s + fvv(b, a) * gia(b, i) - gab(a, b) * fvo(i, b)
      end do
      g(i, a) = s
    end do
  end do
  ! ---- ijkl with <am||jk>, ijak with <am||ej> and <mn||ij>, iabj with <em||ij>
  !$omp parallel do collapse(2) private(a, i, m, p, pp, sg, e, j, n, s)
  do a = 0, nv - 1
    do i = 0, no - 1
      s = 0
      do m = 0, no - 1
        if (m /= i) then
          if (i < m) then
            pp = opidx(m, i)
            sg = 1
          else
            pp = opidx(i, m)
            sg = -1
          end if
          do p = 0, npo - 1
            s = s - sg * ovood(opb(p), opa(p), a, m) * y4(pp, p)
          end do
        end if
        do e = 0, nv - 1
          do j = 0, no - 1
            s = s - ovvod(j, e, a, m) * dk(j, e, m, i) - ovood(j, i, e, m) * rl(m, a, j, e)
          end do
        end do
        do n = 0, no - 1
          do j = 0, no - 1
            s = s - 0.5d0 * ooood(j, i, n, m) * dk(j, a, n, m)
          end do
        end do
      end do
      g(i, a) = g(i, a) + s
    end do
  end do
  ! ---- the terms with the dressed <ab||ij> on pairs: with l1, with Z (iajk)
  !      and with C3 (abci); ijab with <mn||ie> and with <am||ef>
  !$omp parallel do private(p, q, m, n, x, y, v, a, i) reduction(+:g)
  do p = 0, npo - 1
    m = opa(p)
    n = opb(p)
    do q = 0, nvp - 1
      x = vpa(q)
      y = vpb(q)
      v = vvoo(q, p)
      g(n, y) = g(n, y) + v * l1(x, m)
      g(m, y) = g(m, y) - v * l1(x, n)
      g(n, x) = g(n, x) - v * l1(y, m)
      g(m, x) = g(m, x) + v * l1(y, n)
      do i = 0, no - 1
        g(i, x) = g(i, x) + v * z(i, y, p)
        g(i, y) = g(i, y) - v * z(i, x, p)
      end do
      do a = 0, nv - 1
        g(m, a) = g(m, a) + 0.5d0 * v * c3(a, q, n)
        g(n, a) = g(n, a) - 0.5d0 * v * c3(a, q, m)
      end do
      v = dab(q, p)
      do i = 0, no - 1
        g(i, x) = g(i, x) - ooovd(y, i, n, m) * v
        g(i, y) = g(i, y) + ooovd(x, i, n, m) * v
      end do
      do a = 0, nv - 1
        g(m, a) = g(m, a) - d(q, a, n) * v
        g(n, a) = g(n, a) + d(q, a, m) * v
      end do
    end do
  end do
  ! ---- <ae||gj> with the density ab and with the block iabj:
  !      Dc[e][i][(g, j)] = Rg[e][j][g][i] - delta(i, j) g_ab[e][g]
  !$omp parallel do collapse(2) private(e, i, b, j)
  do e = 0, nv - 1
    do i = 0, no - 1
      do b = 0, nv - 1
        do j = 0, no - 1
          dc(j, b, i, e) = rl(i, b, j, e)
        end do
        dc(i, b, i, e) = dc(i, b, i, e) - gab(b, e)
      end do
    end do
  end do
  call N(zc_gemm)(0_4, 1_4, nvp, ov, ov, one, vvo, ov, dc, ov, zero, mq, ov)
  call N(xzdbra)(no, nv, nvp, vpa, vpb, mq, one, g)
  ! ---- abcd with <ef||gi>: its l2 t2 part; its l3 t3 part from pass C
  if (npo > 0) call N(zc_gemm)(0_4, 0_4, npo, ov, nvp, one, l2pp, nvp, vvo, ov, zero, u, ov)
  !$omp parallel do collapse(2) private(a, i, p, b, s)
  do a = 0, nv - 1
    do i = 0, no - 1
      s = -s13(i, a)
      do p = 0, npo - 1
        do b = 0, nv - 1
          s = s + t2(b, a, opb(p), opa(p)) * u(i, b, p)
        end do
      end do
      g(i, a) = g(i, a) + s
    end do
  end do
  ! ---- aibc with <em||if>
  !$omp parallel do collapse(2) private(e, m, q, x, y, v, i) reduction(+:g)
  do e = 0, nv - 1
    do m = 0, no - 1
      do q = 0, nvp - 1
        x = vpa(q)
        y = vpb(q)
        v = dai(q, m, e)
        do i = 0, no - 1
          g(i, x) = g(i, x) - ovvod(i, y, e, m) * v
          g(i, y) = g(i, y) + ovvod(i, x, e, m) * v
        end do
      end do
    end do
  end do
end subroutine

! ---------------------------------------------------------------------------
! The hand-over to CFOUR's xx2czvec (docs/native.md, section 10.2), between
! the active spinors: from the response density den(q, p) = d1[p][q] and the
! gradient gnh[a][i],
!   D[p][q] = (d1[q][p] + conj(d1[p][q])) / 2          the unrelaxed density, dh(p, q)
!   R(d)[a][i] = (e_a - e_i) d[a][i] + sum_{b,j} (<aj||ib> d[b][j] + <ab||ij> conj(d[b][j]))
!   x = gnh / 2 - R(D_vo)
subroutine N(xzdxact)(no, nv, eps, oovv, ovvo, den, gnh, dh, x)
  implicit none
  integer(8), intent(in) :: no, nv
  real(8), intent(in) :: eps(0:no+nv-1)
  ZT, intent(in) :: oovv(0:nv-1, 0:nv-1, 0:no-1, 0:no-1)       ! <mn||ef> [m][n][e][f]
  ZT, intent(in) :: ovvo(0:no-1, 0:nv-1, 0:nv-1, 0:no-1)       ! <mb||ej>
  ZT, intent(in) :: den(0:no+nv-1, 0:no+nv-1), gnh(0:no-1, 0:nv-1)
  ZT, intent(out) :: dh(0:no+nv-1, 0:no+nv-1)                  ! (p, q) = D[p][q]
  ZT, intent(out) :: x(0:nv-1, 0:no-1)                         ! (a, i)
  integer(8) :: p, q, a, b, i, j, n
  ZT :: s
  n = no + nv
  do p = 0, n - 1
    do q = 0, n - 1
      dh(p, q) = 0.5d0 * (den(p, q) + CONJ(den(q, p)))
    end do
  end do
  !$omp parallel do collapse(2) private(a, i, b, j, s)
  do a = 0, nv - 1
    do i = 0, no - 1
      s = (eps(no + a) - eps(i)) * dh(no + a, i)
      do b = 0, nv - 1
        do j = 0, no - 1
          s = s + ovvo(i, b, a, j) * dh(no + b, j) + CONJ(oovv(b, a, j, i)) * CONJ(dh(no + b, j))
        end do
      end do
      x(a, i) = 0.5d0 * gnh(i, a) - s
    end do
  end do
end subroutine

! The seven files: xfull (x; rows active virtual then dropped, columns frozen
! then active occupied), dpoo ((D - 1) / 2), dpvv (D / 2), dii0, daa0 (zeros),
! dpopo (y[frozen][active occupied]), dpvpv (y[dropped][active virtual]); one
! record each, complex, the first index fastest, in CFOUR's order of the
! spinors (operm, vperm: CFOUR's number of the spinor in the order here).
! nf = nd = 0: the files of the active spinors alone; xall and ym are then
! not read.
subroutine N(xzdwrite)(no, nv, nf, nd, nall, dh, x, xall, ym, operm, vperm, rec)
  implicit none
  integer(8), intent(in) :: no, nv, nf, nd, nall
  ZT, intent(in) :: dh(0:no+nv-1, 0:no+nv-1), x(0:nv-1, 0:no-1)
  ZT, intent(in) :: xall(0:nall-1, 0:nall-1), ym(0:nall-1, 0:nall-1)
  integer(8), intent(in) :: operm(0:no-1), vperm(0:nv-1)
  complex(8) :: rec(*)
  integer(8) :: a, b, i, j, n, nr, f, d
  n = no + nv
  nr = nv + nd
  do i = 0, no - 1
    do a = 0, nv - 1
      rec(1 + vperm(a) + nr * (nf + operm(i))) = x(a, i)
    end do
    do d = 0, nd - 1
      rec(1 + nv + d + nr * (nf + operm(i))) = xall(n + nf + d, i)
    end do
  end do
  do f = 0, nf - 1
    do a = 0, nv - 1
      rec(1 + vperm(a) + nr * f) = xall(no + a, n + f)
    end do
    do d = 0, nd - 1
      rec(1 + nv + d + nr * f) = xall(n + nf + d, n + f)
    end do
  end do
  call xzwrec('xfull', nr * (no + nf), rec)
  do j = 0, no - 1
    do i = 0, no - 1
      rec(1 + operm(i) + no * operm(j)) = 0.5d0 * dh(i, j)
    end do
    rec(1 + operm(j) + no * operm(j)) = rec(1 + operm(j) + no * operm(j)) - 0.5d0
  end do
  call xzwrec('dpoo', no * no, rec)
  do b = 0, nv - 1
    do a = 0, nv - 1
      rec(1 + vperm(a) + nv * vperm(b)) = 0.5d0 * dh(no + a, no + b)
    end do
  end do
  call xzwrec('dpvv', nv * nv, rec)
  rec(1:max(no * no, nv * nv)) = 0
  call xzwrec('dii0', no * no, rec)
  call xzwrec('daa0', nv * nv, rec)
  do i = 0, no - 1
    do f = 0, nf - 1
      rec(1 + f + nf * operm(i)) = ym(n + f, i)
    end do
  end do
  call xzwrec('dpopo', nf * no, rec)
  do a = 0, nv - 1
    do d = 0, nd - 1
      rec(1 + d + nd * vperm(a)) = ym(n + nf + d, no + a)
    end do
  end do
  call xzwrec('dpvpv', nd * nv, rec)
end subroutine
