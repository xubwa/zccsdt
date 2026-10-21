!
! EOM-DEA-CCSDT with complete 5p3h operators (%xzdea level 5p3h; xzdea5.inc;
! docs/native.md, section 21; the algorithm of src/zccsdt/cc/dea5_sector.py,
! docs/eom.md, section 11.11): what is not type-generic - the layout of the
! 5p3h blocks of a sector vector, of the intermediates of a sigma vector, the
! distribution of the rows of r5 over the ranks.
!
! The 5p3h part of a sector vector of the shift s:
!   r5   one block per class M of the virtual quintuples: rows the occupied
!        triples of group M - s (in their order), columns the quintuples
!        a<b<c<d<e of class M, lexicographic (csrc/t4sym.c: zc_t5s_quints) -
!        the blocks of zccsdt.cc.dea5_sector.Sector5, held in full while a
!        sigma vector is made;
!   the stored part (the vectors of the solver): of every row of a block only
!        the columns whose quintuple has the parity of the target with the
!        triple (%xzdea parity g|u; all columns without a parity), the rows in
!        their order - so that a range of triples is a range of the stored
!        part as of the blocks (r5c, r5w).  Half of r5 with a parity.
! Since the triples are numbered by groups and the blocks go by classes, the
! rows of the blocks in their order are the triples in their order: a range
! of triples is a range of r5.  Over the ranks every rank holds one range of
! triples (dtb; the same for all targets), and its part of a vector is its
! rows of r4 followed by its rows of the stored r5 (the rank-major layout of
! the solver, esoff of xzeomd; rank 0 with r2 and r3 in front).
!
module xzdead5
  implicit none
  ! ---- tables of all sectors (xzd5counts, N(xzd5tabs))
  integer(8) :: n5c = 1                         ! classes of virtual quintuples, 5 ncls - 4
  integer(8), allocatable :: nq5(:)             ! (0:n5c-1) quintuples of class M
  integer(8) :: nv4 = 0                         ! C(nv, 4)
  integer(8), allocatable :: d5tri(:, :)        ! (0:2, 0:nto-1) the three indices of an occupied triple, ascending
  integer(8), allocatable :: d5tgr(:)           ! (0:nto-1) the group of an occupied triple
  integer(8) :: dpos4 = 0                       ! int32 pos4[rank of a quadruple]: its place in its class

  ! ---- the 5p3h blocks of the sector in hand (xzd5sec)
  integer(8), allocatable :: off5(:)            ! (0:n5c) first element of block M in the full r5
  integer(8), allocatable :: blk5(:), pos5r(:)  ! (0:nto-1) the block of a triple, its row there (-1: none)
  logical, allocatable :: inblk5(:)             ! M is a block of r5 (rows and quintuples)
  integer(8), allocatable :: off23(:, :), n23(:)  ! a row of block M as X[(ab) of group q][(cde)]: (M, q)
  integer(8), allocatable :: off14(:, :), n14(:)  ! as X[a of class p][(bcde)]: (M, p)
  integer(8), allocatable :: off13a(:, :), n13a(:) ! a row of the quadruples of any class A as X[b of p][(cde)]
  integer(8), allocatable :: x4off(:)           ! (0:npo) a row of r4 on (h | cde): its place in X4 (the sizes)
  integer(8), allocatable :: r5c(:), r5w(:)     ! (0:nto) the stored row of a triple: first element, elements
  integer(8) :: dsz5 = 0, dsz5c = 0, n23max = 0, n14max = 0, n13amax = 0, nq5max = 0, x4max = 0
  integer(8), allocatable :: q5at(:)            ! (0:n5c-1) the quintuple list of block M (word address; 0: none)
  integer(8), allocatable :: p5at(:)            ! the parity bits of its columns (word address; 0: none)
  integer(8) :: q5size = 0, p5size = 0          ! words of all lists, of all bits

  ! ---- the rows of r5 over the ranks (xzd5ranges, xzd5slices)
  integer(8), allocatable :: dtb(:)             ! (0:nrank) the triples of every rank
  integer(8) :: dt0 = 0, dt1 = 0                ! of this rank
  integer(8), allocatable :: d4e(:)             ! (0:nrank) first element of r4 of every rank (r4)
  integer(8), allocatable :: d5e(:)             ! (0:nrank) first stored element of r5 of every rank
  integer(8), allocatable :: d5f(:)             ! (0:nrank) first element of the full r5 of every rank
  integer(8) :: d5own = 0, d4own = 0            ! of this rank: stored elements of r5, elements of r4

  ! ---- the intermediates of the ground state (N(xzd5im))
  integer(8) :: dwq5 = 0                        ! Wq5 = Wpp + F_vv / 4 on pairs (the blocks of <vv||vv>)
  integer(8) :: dh3 = 0                         ! the operators on the occupied triples of a group (h3off)
  integer(8) :: dxo3 = 0                        ! Xo3[I][m][Q]
  integer(8) :: dw2 = 0                         ! W2[I][a][(mn)]
  integer(8) :: db4 = 0                         ! B4[(jk)][m][l][g][c]
  integer(8) :: dx2 = 0                         ! [t(m kl; K) | Xv2(e; kl; K)] of the pairs of x2loc (layout r13off)
  integer(8), allocatable :: h3off(:)           ! (0:ntg)
  integer(8), allocatable :: x2loc(:)           ! (0:npo-1) the first element of the pair in dx2 (-1: not held)
  integer(8) :: x2size = 0, h3size = 0

  ! ---- the intermediates of a sigma vector (xzd5lay)
  integer(8), allocatable :: z4off(:, :, :)     ! (k, mu, fe) Z4(mf; k; K) as [m of mu][f of fe][K]
  integer(8), allocatable :: z4k(:)             ! (0:no) the first element of the k (k-major)
  integer(8), allocatable :: xr5off(:, :)       ! (m, P) the row (P, m) of XoR5 (-1: none), nK(K) elements
  integer(8), allocatable :: y4off(:, :)        ! (P, c) Y4(mn; P; de) as [(mn) of group c][(de)]
  integer(8), allocatable :: xv5off(:, :)       ! (k, fe) Xv5(f; k; bcde) as [f of fe][quadruples], in the slab of k
  integer(8), allocatable :: xv5k(:)            ! (0:no-1) elements of the slab of k
  integer(8) :: z4size = 0, xr5size = 0, y4size = 0, xv5max = 0

  ! ---- the indices z of Xv5 over the ranks (xzd5zgroups): the ranks of z
  !      are zr0(z) .. zr0(z) + zn(z) - 1; a rank has the z of zmine
  integer(8), allocatable :: zr0(:), zn(:)
  logical, allocatable :: zmine(:)

  ! ---- seconds of the pieces of the 5p3h part of the sigma vectors
  real(8) :: td5(20) = 0d0

contains

  integer(8) function d5nA4(a)
    use xzdead, only: n4c, nqa
    integer(8), intent(in) :: a
    d5nA4 = 0
    if (a >= 0 .and. a < n4c) d5nA4 = nqa(a)
  end function

  integer(8) function d5nq5(m)
    integer(8), intent(in) :: m
    d5nq5 = 0
    if (m >= 0 .and. m < n5c) d5nq5 = nq5(m)
  end function

  ! the number of the quadruple a<b<c<d of 0..n-1 in lexicographic order
  integer(8) function xzd5qrank(a, b, c, d, n)
    integer(8), intent(in) :: a, b, c, d, n
    integer(8) :: y0, y1, y2, nall
    nall = n * (n - 1) * (n - 2) * (n - 3) / 24
    y0 = n - 1 - a
    y1 = n - 1 - b
    y2 = n - 1 - c
    xzd5qrank = nall - 1 - y0 * (y0 - 1) * (y0 - 2) * (y0 - 3) / 24 - y1 * (y1 - 1) * (y1 - 2) / 6 - &
                y2 * (y2 - 1) / 2 - (n - 1 - d)
  end function

  ! the quintuples of every class M: their numbers (zc_t5s_quints); C(nv, 4);
  ! the indices of the occupied triples
  subroutine xzd5counts(w)
    use xzdata, only: nv, ncls, vcls, nto, jofree, ntg, otoff
    use xzdead, only: xzdiv2
    use xzkern
    real(8), target :: w(*)
    integer(4) :: kv(0:max(nv, 1_8)-1)
    integer(8) :: m
    integer(8), pointer :: ofree(:, :)
    if (allocated(d5tri)) deallocate(d5tri, d5tgr)
    allocate(d5tri(0:2, 0:max(nto, 1_8)-1), d5tgr(0:max(nto, 1_8)-1))
    d5tgr = 0
    do m = 0, ntg - 1
      d5tgr(otoff(m):otoff(m + 1) - 1) = m
    end do
    call xzdiv2(w, jofree, max(nto, 1_8), 3_8, ofree)
    do m = 0, nto - 1
      d5tri(0, m) = ofree(m, 0)
      d5tri(1, m) = ofree(m, 1)
      d5tri(2, m) = ofree(m, 2)
    end do
    n5c = 5 * ncls - 4
    if (allocated(nq5)) deallocate(nq5)
    allocate(nq5(0:n5c-1))
    do m = 0, nv - 1
      kv(m) = int(vcls(m), 4)
    end do
    nq5 = 0
    if (nv >= 5) then
      !$omp parallel do schedule(dynamic)
      do m = 0, n5c - 1
        nq5(m) = zc_t5s_quints(int(nv, 4), kv, int(m, 4), 0_c_intptr_t)
      end do
    end if
    nv4 = nv * (nv - 1) * (nv - 2) * (nv - 3) / 24
  end subroutine

  ! The 5p3h blocks of the sector of the shift ds (after xzdsec) and the
  ! layouts of the rows of a block in the products of the passes
  ! (dea5_sector.Sector5, Engine5)
  subroutine xzd5sec()
    use xzdata
    use xzdead
    integer(8) :: m, c, p, q, n, a, i, pq
    if (allocated(off5)) deallocate(off5, blk5, pos5r, inblk5, off23, n23, off14, n14, off13a, n13a, x4off, r5c, r5w, &
                                    q5at, p5at)
    allocate(off5(0:n5c), blk5(0:max(nto, 1_8)-1), pos5r(0:max(nto, 1_8)-1), inblk5(0:n5c-1), off23(0:n5c-1, 0:n2c-1), &
             n23(0:n5c-1), off14(0:n5c-1, 0:ncls-1), n14(0:n5c-1), off13a(0:n4c-1, 0:ncls-1), n13a(0:n4c-1), &
             x4off(0:max(npo, 1_8)), r5c(0:max(nto, 1_8)), r5w(0:max(nto, 1_8)), q5at(0:n5c-1), p5at(0:n5c-1))
    off5(0) = 0
    blk5 = -1
    pos5r = -1
    nq5max = 0
    do m = 0, n5c - 1
      c = m - ds
      inblk5(m) = .false.
      n = 0
      if (c >= 0 .and. c < n3c .and. nq5(m) > 0) n = dnJ(c)
      if (n > 0) then
        inblk5(m) = .true.
        nq5max = max(nq5max, nq5(m))
        do i = otoff(c), otoff(c + 1) - 1
          blk5(i) = m
          pos5r(i) = i - otoff(c)
        end do
      end if
      off5(m + 1) = off5(m) + n * nq5(m)
    end do
    dsz5 = off5(n5c)
    ! (b | cde) of the quadruples of every class A
    n13amax = 0
    do a = 0, n4c - 1
      n = 0
      do p = 0, ncls - 1
        off13a(a, p) = -1
        if (dnV(p) > 0 .and. dnK(a - p) > 0) then
          off13a(a, p) = n
          n = n + dnV(p) * dnK(a - p)
        end if
      end do
      n13a(a) = n
      n13amax = max(n13amax, n)
    end do
    ! one row of a block: X[(ab), (cde)] by the group of (ab), X[a, (bcde)] by the class of a
    off23 = -1
    off14 = -1
    n23 = 0
    n14 = 0
    n23max = 0
    n14max = 0
    do m = 0, n5c - 1
      if (.not. inblk5(m)) cycle
      n = 0
      do q = 0, n2c - 1
        if (dnQ(q) > 0 .and. dnK(m - q) > 0) then
          off23(m, q) = n
          n = n + dnQ(q) * dnK(m - q)
        end if
      end do
      n23(m) = n
      n = 0
      do p = 0, ncls - 1
        if (dnV(p) > 0 .and. d5nA4(m - p) > 0) then
          off14(m, p) = n
          n = n + dnV(p) * d5nA4(m - p)
        end if
      end do
      n14(m) = n
      n23max = max(n23max, n23(m))
      n14max = max(n14max, n14(m))
    end do
    ! r4 on (h | cde), one row (pair) after the other
    x4off(0) = 0
    x4max = 0
    do pq = 0, npo - 1
      n = 0
      if (blk4(pq) >= 0) n = n13(blk4(pq))
      x4off(pq + 1) = x4off(pq) + n
      x4max = max(x4max, n)
    end do
    q5at = 0
    p5at = 0
  end subroutine

  ! The stored 5p3h part (the columns of the parity of every row), from the
  ! counts of the columns of each parity of every block (cnt(0:1, M): with
  ! no parity cnt(0, M) = all, cnt(1, M) = 0)
  subroutine xzd5store(cnt)
    use xzdata, only: nto, otoff
    use xzdead
    integer(8), intent(in) :: cnt(0:1, 0:n5c-1)
    integer(8) :: i, m, b
    r5c(0) = 0
    do i = 0, nto - 1
      m = blk5(i)
      r5w(i) = 0
      if (m >= 0) then
        if (dtpar < 0) then
          r5w(i) = nq5(m)
        else if (.not. allocated(dpo)) then
          ! (the plan, before the parities are read: the counts of the plan)
          r5w(i) = cnt(0, m)
        else
          b = mod(dpo(xzd5free(i, 0)) + dpo(xzd5free(i, 1)) + dpo(xzd5free(i, 2)) + dtpar, 2_8)
          r5w(i) = cnt(b, m)
        end if
      end if
      r5c(i + 1) = r5c(i) + r5w(i)
    end do
    dsz5c = r5c(nto)
  end subroutine

  ! the column parity (0, 1) that the row of the triple i keeps: the parity of
  ! the triple and of the target (-1: all columns)
  integer(8) function xzd5rpar(i)
    use xzdead, only: dtpar, dpo
    integer(8), intent(in) :: i
    xzd5rpar = -1
    if (dtpar < 0) return
    xzd5rpar = mod(dpo(xzd5free(i, 0)) + dpo(xzd5free(i, 1)) + dpo(xzd5free(i, 2)) + dtpar, 2_8)
  end function

  ! the group of the occupied triple i
  integer(8) function xzd5tgrp(i)
    integer(8), intent(in) :: i
    xzd5tgrp = d5tgr(i)
  end function

  ! the r-th index (r = 0, 1, 2; ascending) of the occupied triple i
  integer(8) function xzd5free(i, r)
    use xzdata, only: no
    integer(8), intent(in) :: i, r
    xzd5free = d5tri(r, i)
  end function

  ! The triples of every rank (more than one rank): ranges in the order of the
  ! triples of about equal sums over the shifts sh(1:nsh) of the targets of
  ! the elements of their rows of r5 (the same for all targets); dt0, dt1 of
  ! this rank.  One rank: all.
  subroutine xzd5ranges(nsh, sh)
    use xzdata
    use xzdead, only: n3c
    use xzpar, only: myrank, nrank
    integer(8), intent(in) :: nsh, sh(*)
    real(8), allocatable :: cum(:)
    integer(8) :: r, i, t, m, g
    if (allocated(dtb)) deallocate(dtb)
    allocate(dtb(0:nrank), cum(0:max(nto, 1_8)))
    cum(0) = 0d0
    do g = 0, ntg - 1
      do i = otoff(g), otoff(g + 1) - 1
        cum(i + 1) = cum(i)
        do t = 1, nsh
          m = g + sh(t)
          if (m >= 0 .and. m < n5c) cum(i + 1) = cum(i + 1) + dble(nq5(m))
        end do
      end do
    end do
    dtb(0) = 0
    do r = 1, nrank - 1
      dtb(r) = nto
      do i = 0, nto - 1
        if (cum(i) >= dble(r) * cum(nto) / dble(nrank)) then
          dtb(r) = i
          exit
        end if
      end do
      dtb(r) = max(dtb(r), dtb(r - 1))
    end do
    dtb(nrank) = nto
    dt0 = dtb(myrank)
    dt1 = dtb(myrank + 1)
    deallocate(cum)
  end subroutine

  ! the rank that holds the row of r5 of the triple i
  integer(8) function xzd5owner(i)
    use xzpar, only: nrank
    integer(8), intent(in) :: i
    integer(8) :: r
    xzd5owner = nrank - 1
    do r = 0, nrank - 1
      if (i < dtb(r + 1)) then
        xzd5owner = r
        return
      end if
    end do
  end function

  ! The parts of the vectors of the sector in hand (after xzdslices): of every
  ! rank its rows of r4 and of the stored r5, one after the other - esoff
  ! (xzeomd) becomes the rank-major place of these parts behind r2 and r3;
  ! d4e, d5e the first element of r4, of the stored r5 of every rank (in r4,
  ! in the stored r5); d5f the same in the full r5.  One rank: [r2|r3|r4|r5].
  subroutine xzd5slices(nw)
    use xzdata
    use xzpar, only: myrank, nrank
    use xzeomd, only: esoff, eswoff, esmax
    use xzdead
    integer(8), intent(in) :: nw
    integer(8) :: r
    if (allocated(d4e)) deallocate(d4e, d5e, d5f)
    allocate(d4e(0:nrank), d5e(0:nrank), d5f(0:nrank))
    if (.not. allocated(esoff)) allocate(esoff(0:nrank), eswoff(0:nrank))
    if (.not. allocated(dqb)) then
      allocate(dqb(0:nrank))
      dqb = 0
      dqb(nrank) = npo
    end if
    do r = 0, nrank
      d4e(r) = d5rowoff4(dqb(r))
      d5e(r) = r5c(dtb(r))
      d5f(r) = d5rowoff5(dtb(r))
    end do
    esoff(0) = 0
    esmax = 0
    do r = 0, nrank - 1
      esoff(r + 1) = esoff(r) + (d4e(r + 1) - d4e(r)) + (d5e(r + 1) - d5e(r))
      esmax = max(esmax, esoff(r + 1) - esoff(r))
    end do
    eswoff = nw * esoff
    d4own = d4e(myrank + 1) - d4e(myrank)
    d5own = d5e(myrank + 1) - d5e(myrank)
    dvec5 = dsz5c
  end subroutine

  ! the first element of r4 of the pairs from p on (the order of the blocks)
  integer(8) function d5rowoff4(p)
    use xzdata, only: npo
    use xzdead, only: blk4, pos4, off4, nqa, dsz4
    integer(8), intent(in) :: p
    integer(8) :: q
    d5rowoff4 = dsz4
    do q = p, npo - 1
      if (blk4(q) >= 0) then
        d5rowoff4 = off4(blk4(q)) + pos4(q) * nqa(blk4(q))
        return
      end if
    end do
  end function

  ! the first element of the full r5 of the triples from i on
  integer(8) function d5rowoff5(i)
    use xzdata, only: nto
    integer(8), intent(in) :: i
    integer(8) :: q
    d5rowoff5 = dsz5
    do q = i, nto - 1
      if (blk5(q) >= 0) then
        d5rowoff5 = off5(blk5(q)) + pos5r(q) * nq5(blk5(q))
        return
      end if
    end do
  end function

  ! the rows a <= r < b of the block M of r5 (the triples of group M - ds)
  ! that this rank holds
  subroutine xzd5own(m, a, b)
    use xzdata, only: otoff
    use xzdead, only: ds
    integer(8), intent(in) :: m
    integer(8), intent(out) :: a, b
    integer(8) :: c
    c = m - ds
    a = max(dt0, otoff(c)) - otoff(c)
    b = min(dt1, otoff(c + 1)) - otoff(c)
    if (b < a) b = a
  end subroutine

  ! The blocks of the ring of the spectator pair pq = (u, w) of r5
  ! (dea5_sector.Engine5.ring_tables): for every d = class(m) - class(e)
  ! with quadruples of the class kk = s + group(pq) + d, the columns (m, e) of
  ! the operand - m not u, w; e of class class(m) - d; mbase(m, b) the first
  ! of m (-1: none) - and the rows of the stacked operator: (i, a) as the
  ! columns (ibase), the row of f_ov (d = 0: bfov), the rows i' of <m u||e i'>
  ! for the occupied i' of class class(u) + d (bru) and of - <m w||e i'> for
  ! those of class class(w) + d (brw); -1: none.
  subroutine xzd5ringblk(pq, u, wq, kp, nbk, bd, bkk, bnc, bnr, bfov, bru, brw, mbase, ibase)
    use xzdata
    use xzdead
    integer(8), intent(in) :: pq, u, wq, kp
    integer(8), intent(out) :: nbk
    integer(8), intent(out) :: bd(*), bkk(*), bnc(*), bnr(*), bfov(*), bru(*), brw(*)
    integer(8), intent(out) :: mbase(0:no-1, *), ibase(0:no-1, *)
    integer(8) :: d, kk, m, c, nc, nr, ci
    nbk = 0
    do d = 1 - ncls, ncls - 1
      kk = ds + kp + d
      if (d5nA4(kk) == 0) cycle
      nc = 0
      do m = 0, no - 1
        mbase(m, nbk + 1) = -1
        c = ocls(m) - d
        if (m == u .or. m == wq .or. dnV(c) == 0) cycle
        mbase(m, nbk + 1) = nc
        nc = nc + dnV(c)
      end do
      if (nc == 0) cycle
      nbk = nbk + 1
      nr = 0
      do m = 0, no - 1
        ibase(m, nbk) = -1
        c = ocls(m) - d
        if (m == u .or. m == wq .or. dnV(c) == 0) cycle
        ibase(m, nbk) = nr
        nr = nr + dnV(c)
      end do
      bfov(nbk) = -1
      if (d == 0) then
        bfov(nbk) = nr
        nr = nr + 1
      end if
      bru(nbk) = -1
      ci = ocls(u) + d
      if (dnO(ci) > 0) then
        bru(nbk) = nr
        nr = nr + dnO(ci)
      end if
      brw(nbk) = -1
      ci = ocls(wq) + d
      if (dnO(ci) > 0) then
        brw(nbk) = nr
        nr = nr + dnO(ci)
      end if
      bd(nbk) = d
      bkk(nbk) = kk
      bnc(nbk) = nc
      bnr(nbk) = nr
    end do
    if (pq < 0) return
  end subroutine

  ! The batches of the blocks of the ring of a spectator pair: a block after
  ! the other while the operand and the product of the batch hold at most
  ! nbatch elements (at least one block); bfirst(t) the first block of batch
  ! t; the most that the operand, the product and a gathered operator of a
  ! batch take (mbmax, xbmax, wgmax: maxima so far, updated)
  subroutine xzd5ringbat(nbatch, nbk, bkk, bnc, bnr, nbat, bfirst, mbmax, xbmax, wgmax)
    integer(8), intent(in) :: nbatch, nbk, bkk(*), bnc(*), bnr(*)
    integer(8), intent(out) :: nbat, bfirst(*)
    integer(8), intent(inout) :: mbmax, xbmax, wgmax
    integer(8) :: b, tot, sz, mt, xt
    nbat = 0
    tot = 0
    mt = 0
    xt = 0
    do b = 1, nbk
      sz = (bnc(b) + bnr(b)) * d5nA4(bkk(b))
      if (nbat > 0 .and. b > bfirst(max(nbat, 1_8)) .and. tot + sz > nbatch) then
        mbmax = max(mbmax, mt)
        xbmax = max(xbmax, xt)
        tot = 0
        mt = 0
        xt = 0
        nbat = nbat + 1
        bfirst(nbat) = b
      else if (nbat == 0) then
        nbat = 1
        bfirst(1) = 1
      end if
      tot = tot + sz
      mt = mt + bnc(b) * d5nA4(bkk(b))
      xt = xt + bnr(b) * d5nA4(bkk(b))
      wgmax = max(wgmax, bnc(b) * bnr(b))
    end do
    mbmax = max(mbmax, mt)
    xbmax = max(xbmax, xt)
    bfirst(nbat + 1) = nbk + 1
  end subroutine

  ! the rows (m, u, w) of r5 of the spectator pair pq as a rank takes them for
  ! the ring over the ranks: one after the other in the order of m; lrow(m)
  ! the first element of the row of m (-1: none), tot all elements
  subroutine xzd5ringlay(w, pq, lrow, tot)
    use xzdata
    use xzdead, only: xzdiv2
    real(8), target :: w(*)
    integer(8), intent(in) :: pq
    integer(8), intent(out) :: lrow(0:no-1), tot
    integer(8), pointer :: otri(:, :)
    integer(1), pointer :: osgn(:, :)
    integer(8) :: m, i
    call xzdiv2(w, jotri, max(npo, 1_8), no, otri)
    call xzdi1v2_(w, josgn, max(npo, 1_8), no, osgn)
    tot = 0
    lrow = -1
    do m = 0, no - 1
      if (osgn(pq, m) == 0) cycle
      i = otri(pq, m)
      if (blk5(i) < 0) cycle
      lrow(m) = tot
      tot = tot + nq5(blk5(i))
    end do
  contains
    subroutine xzdi1v2_(w, k, n1, n2, p)
      use xzdead, only: xzdi1v2
      real(8), target :: w(*)
      integer(8), intent(in) :: k, n1, n2
      integer(1), pointer :: p(:, :)
      call xzdi1v2(w, k, n1, n2, p)
    end subroutine
  end subroutine

  ! The layouts of the intermediates of a sigma vector (dea5_sector.Engine5):
  ! Z4 (k, class of m, class of f) -> [m][f][K], K = s + class(k) + class(f) -
  ! class(m), k-major; XoR5 rows (P, m) of nK(s + group(P) - class(m))
  ! elements; Y4 (P, c) -> [(mn) of group c][(de) of group blk4(P) - c]; Xv5
  ! (k, class of f) -> [f][quadruples of class s + class(k) + class(f)], the
  ! slab of k
  subroutine xzd5lay(w)
    use xzdata
    use xzdead
    real(8), target :: w(*)
    integer(8), pointer :: opgrp(:)
    integer(8) :: k, mu, fe, kk, n, pq, m, c, m4, a
    call xzdiv(w, jopgrp, max(npo, 1_8), opgrp)
    if (allocated(z4off)) deallocate(z4off, z4k, xr5off, y4off, xv5off, xv5k)
    allocate(z4off(0:no-1, 0:ncls-1, 0:ncls-1), z4k(0:no), xr5off(0:no-1, 0:max(npo, 1_8)-1), &
             y4off(0:max(npo, 1_8)-1, 0:n2c-1), xv5off(0:no-1, 0:ncls-1), xv5k(0:no-1))
    z4off = -1
    n = 0
    do k = 0, no - 1
      z4k(k) = n
      do mu = 0, ncls - 1
        if (dnO(mu) == 0) cycle
        do fe = 0, ncls - 1
          if (dnV(fe) == 0) cycle
          kk = ds + ocls(k) + fe - mu
          if (dnK(kk) == 0) cycle
          z4off(k, mu, fe) = n
          n = n + dnO(mu) * dnV(fe) * dnK(kk)
        end do
      end do
    end do
    z4k(no) = n
    z4size = n
    xr5off = -1
    n = 0
    do pq = 0, npo - 1
      do m = 0, no - 1
        kk = ds + opgrp(pq) - ocls(m)
        if (dnK(kk) == 0) cycle
        xr5off(m, pq) = n
        n = n + dnK(kk)
      end do
    end do
    xr5size = n
    y4off = -1
    n = 0
    do pq = 0, npo - 1
      m4 = blk4(pq)
      if (m4 < 0) cycle
      do c = 0, n2c - 1
        if (dnO2(c) == 0 .or. dnQ(m4 - c) == 0 .or. dnQ(c) == 0) cycle
        if (off22(m4, c) < 0) cycle
        y4off(pq, c) = n
        n = n + dnO2(c) * dnQ(m4 - c)
      end do
    end do
    y4size = n
    xv5off = -1
    xv5max = 0
    do k = 0, no - 1
      n = 0
      do fe = 0, ncls - 1
        a = ds + ocls(k) + fe
        if (dnV(fe) == 0 .or. d5nA4(a) == 0) cycle
        xv5off(k, fe) = n
        n = n + dnV(fe) * d5nA4(a)
      end do
      xv5k(k) = n
      xv5max = max(xv5max, n)
    end do
  end subroutine

  ! The indices z of Xv5 and of its products (E1x) over the ranks: with at
  ! least as many ranks as occupied spinors the ranks r with mod(r, no) = z
  ! have z (one z each); with fewer ranks the rank mod(z, ranks) has z
  subroutine xzd5zgroups()
    use xzdata, only: no
    use xzpar, only: myrank, nrank
    integer(8) :: z, r
    if (allocated(zr0)) deallocate(zr0, zn, zmine)
    allocate(zr0(0:no-1), zn(0:no-1), zmine(0:no-1))
    do z = 0, no - 1
      if (nrank >= no) then
        zr0(z) = z
        zn(z) = 0
        do r = z, nrank - 1, no
          zn(z) = zn(z) + 1
        end do
        zmine(z) = mod(int(myrank, 8), no) == z
      else
        zr0(z) = mod(z, int(nrank, 8))
        zn(z) = 1
        zmine(z) = zr0(z) == myrank
      end if
    end do
  end subroutine

  ! the ranks of z: the k-th of them (k = 0 .. zn(z) - 1)
  integer(8) function xzd5zrank(z, k)
    use xzdata, only: no
    use xzpar, only: nrank
    integer(8), intent(in) :: z, k
    if (nrank >= no) then
      xzd5zrank = z + k * no
    else
      xzd5zrank = zr0(z)
    end if
  end function

  ! the place of this rank among the ranks of z (0 .. zn(z) - 1)
  integer(8) function xzd5zme(z)
    use xzdata, only: no
    use xzpar, only: myrank, nrank
    integer(8), intent(in) :: z
    xzd5zme = 0
    if (nrank >= no) xzd5zme = myrank / no
  end function
end module xzdead5
