!
! The CCSDT iterations and the Lambda iterations over MPI ranks.
!
! Every rank runs the same step (xzstep, xzlamstep) on its own copy of the
! amplitudes in its own work array: the small intermediates are made by
! every rank from T1 and T2 (they are small, and nothing has to be sent),
! and the three loops that are 95 % of an iteration - pass 1 and pass 2 over
! batches of occupied pairs, the pp ladder over batches of occupied triples
! - do only the batches of this rank.  What the batches of pass 1 produce
! (the T3 parts of W_vvvo and W_vooo, of the T1 and T2 residuals; in the
! Lambda equations also the densities) is summed over the ranks on every
! rank, because the rest of the step needs it; the parts of the residual of
! T3 are summed onto rank 0, which does what is not distributed (the CCSD
! part, the operator on occupied triples, the Jacobi step, DIIS) and sends
! the new amplitudes at the start of the next iteration.  All of it goes
! through the chunked routines of xzmpi.c.
!
! The batches are those of the loops (a batch never leaves the group of its
! pairs or triples; nbp pairs, nbt triples: the memory plan) and are dealt
! out here: by a cost model, the most expensive batch first, each to the
! rank with the least cost so far.  The list depends on the plan and on the
! number of ranks only, so every rank makes the same one.
!
!   kinds of batches   1 pass 1   2 pass 2   3 pp ladder    (T3 equation)
!                      4 pass 1   5 pass 2   6 ladder       (Lambda equations)
!                      7 pass 1   8 pass 2   9 pp ladder    (EOM sigma vector: the residual with R3)
!                     10 pass 1  11 pass 2  12 pp ladder    (EOM sigma vector: the residual with T3
!                                                            and the derivatives of the operators)
!   The kinds of a sigma vector are dealt per label shift (xzdeal_eom): the
!   passes run with the shift of the vector in their operands.
!
! Without MPI, and with one rank, every batch is this rank's and nothing is
! sent: the serial program.
!
module xzdist
  use xzpar, only: myrank, nrank, xzmpi_bcast, xzmpi_bcast_w, xzmpi_send_w, xzmpi_recv_w, xzmpi_barrier, &
                   xzmpi_sum0_w, xzmpi_allsum_w, xzmpi_gather, xzmpi_max, xzmpi_sum, xzmpi_chunk, &
                   xzmpi_slices_add, xzmpi_slices_put
  implicit none
  integer, parameter :: ndk = 12
  character(len=10), parameter :: dkname(ndk) = [character(len=10) :: 'pass 1', 'pass 2', 'pp ladder', &
    'L pass 1', 'L pass 2', 'L ladder', 'R3 pass 1', 'R3 pass 2', 'R3 ladder', 'T3 pass 1', 'T3 pass 2', 'T3 ladder']
  integer, parameter :: ndset = ndk / 3        ! 1 T3 equation, 2 Lambda, 3 and 4 the two residuals of an EOM sigma vector
  integer(8) :: dshift = 0                    ! the label shift (classes) of the vector of the kinds 7 to 12
  integer(8), allocatable :: bown(:, :)       ! (first pair or triple of the batch, kind): its rank (-1: no batch starts there)
  integer(8), allocatable :: bnum(:, :)       ! pairs or triples of the batch
  real(8), allocatable :: bcost(:, :)         ! its cost by the model
  real(8), allocatable :: bsec(:, :)          ! seconds that this rank spent in it (all iterations)
  integer :: dkind = 0                        ! the kind of the loop that runs; 0: every batch is done
  logical :: asworker = .false.               ! a dry run for the memory of a rank other than 0
  logical :: distk(ndk) = .true.              ! option dist: which kinds are dealt out (else all to rank 0)
  real(8) :: rplan(20) = 0d0                  ! the plan of rank 0: words of a worker, the 8 batch sizes, ..;
                                              ! 17: EOM, the words of the subspace of a rank other than 0
  integer(8) :: maxthr = 0                    ! the largest number of threads of a rank
  ! ---- slices (more than one rank; option slices 0: off): every rank owns a range of
  !      rows of T3 - the elements soff(r) <= e < soff(r + 1) of the stored
  !      array, cut at rows -, gets the residual of its rows from all ranks,
  !      does the operator on occupied triples, the Jacobi step and DIIS on
  !      them (its slice of every DIIS vector in its own work array or file)
  !      and sends its slice of the new T3 to every rank
  logical :: slices = .false.
  integer(8), allocatable :: soff(:)          ! (0:nrank), elements
  integer(8), allocatable :: swoff(:)         ! the same in words
  integer(8) :: smax = 0                      ! elements of the longest slice
  real(8), allocatable :: rnode(:, :)         ! (2, rank) on rank 0: a number that names the node of the rank,
                                              ! the MB that the node could give when the program started
  real(8) :: r0extra(2) = 0d0                 ! option r0load: what rank 0 does alone, as a fraction of the
                                              ! distributed work of an iteration (T3 equation, Lambda)
  ! seconds of this rank, (stage, 1 T3 equation / 2 Lambda): 1 pass 1, 2 the
  ! sweep with the pp ladder, 3 pass 2, 4 sums of pass 1, 5 sum of the
  ! residual (with slices: the residual to the owners of its rows),
  ! 6 amplitudes from rank 0 (the slices of the ranks), 7 the whole step,
  ! 8 what only rank 0 does (with slices: what a rank does on its rows),
  ! 9, 10 waiting for the other ranks before the sums of pass 1 and before
  ! the sum of the residual
  integer, parameter :: nds = 10
  ! (set 3: the sigma vectors of EOM - both passes 1, both sweeps, both passes
  ! 2; 6 the slices of the vector to the ranks; 8 the two operators on
  ! occupied triples on its rows; 9 the waiting before the two sums of pass 1)
  real(8) :: dsec(nds, ndset) = 0d0
  real(8) :: dwords(ndset) = 0d0               ! words of amplitudes sent to the ranks (per receiving rank)
  real(8), private :: tb0 = 0d0
  ! ---- the speeds of the ranks, measured (xzrebal).  The nodes of a
  !      partition are not equally fast, the two sockets of a node need not
  !      be, and what a rank does besides its batches (the intermediates; in
  !      the mode without slices what only rank 0 does) takes its time too.
  !      The ranks meet twice in a step: for the sums of pass 1, and for the
  !      residual after the sweep and pass 2.  A batch goes to the rank that
  !      is done with it first: roff seconds of other work before the
  !      meeting, then its batches at rspeed.
  real(8), allocatable :: rspeed(:, :)        ! (rank, kind): model cost per second (1 before a measurement)
  real(8), allocatable :: roff(:, :)          ! (rank, kind): seconds before its batches of this kind
  real(8), allocatable :: rload(:, :)         ! (rank, kind): model cost of its batches
  real(8) :: ksec(ndk) = 0d0                  ! seconds of this rank in its batches, per kind, since the last xzrebal
  real(8) :: tint(2) = 0d0                    ! seconds of this rank up to the two meetings, since the last xzrebal
  real(8) :: tsync = 0d0                      ! (the end of the first meeting)
  real(8) :: rebtol = 0.03d0                  ! deal again when the last rank arrives this far after the mean
  integer(8) :: nrebal(ndset) = 0              ! how often the batches were dealt again
  real(8) :: tmeet(3) = 0d0                   ! EOM: seconds of this rank up to the three meetings of a sigma vector
  ! constants of the cost model: the part of a batch that does not grow with the summed index
  ! (gathers, reductions), in units of one more summed index of its products
  real(8) :: cg1 = 8d0, cg2 = 64d0, cg3 = 16d0, cg4 = 16d0, cg5 = 96d0, cg6 = 16d0
contains

  ! this rank holds no DIIS vectors and does only its batches
  logical function xzworker()
    xzworker = myrank > 0 .or. asworker
  end function

  ! the batch of the running loop that starts at pair or triple `start' is done by this rank
  logical function xzmine(start)
    integer(8), intent(in) :: start
    xzmine = .true.
    if (dkind == 0 .or. nrank == 1 .or. .not. allocated(bown)) return
    xzmine = bown(start, dkind) == myrank
  end function

  ! the clock of a batch starts / its seconds are added
  subroutine xzbtic()
    use omp_lib
    tb0 = omp_get_wtime()
  end subroutine

  subroutine xzbtoc(start)
    use omp_lib
    integer(8), intent(in) :: start
    real(8) :: t
    if (dkind == 0 .or. .not. allocated(bsec)) return
    t = omp_get_wtime() - tb0
    bsec(start, dkind) = bsec(start, dkind) + t
    ksec(dkind) = ksec(dkind) + t
  end subroutine

  ! The cost of one pair (kinds 1, 2, 4, 5) or one triple (3, 6) of the
  ! group g, in operations of the products of its loop.  One class: the same
  ! number for every pair or triple.  Measured: docs/native.md, section 15.
  real(8) function xzbcost1(kind, g)
    use xzdata
    integer, intent(in) :: kind
    integer(8), intent(in) :: g
    integer(8) :: sd, sm, sy, gq, sc, h, st, sw
    real(8) :: nd, nm, nq, ney, dl, nc, nqh, c, np, nj
    c = 0d0
    ! (the sigma vector of EOM: the shift of the rank-3 operand and of the operators)
    st = 0
    sw = 0
    if (kind >= 7 .and. kind <= 9) st = dshift
    if (kind >= 10) sw = dshift
    select case (kind)
    case (1)
      do sd = 0, ncls - 1
        do sm = 0, ncls - 1
          gq = g + sm - sd
          if (gq < 0 .or. gq >= npg) cycle
          nd = vcoff(sd + 1) - vcoff(sd)
          nm = ocoff(sm + 1) - ocoff(sm)
          nq = vpoff(gq + 1) - vpoff(gq)
          sy = g - sd
          ney = 0d0
          if (sy >= 0 .and. sy < ncls) ney = vcoff(sy + 1) - vcoff(sy) + ocoff(sy + 1) - ocoff(sy)
          c = c + nd * nm * nq * (cg1 + ney)
        end do
      end do
    case (2, 5)
      do gq = 0, npg - 1
        if (abs(g - gq) >= ncls) cycle
        dl = ndel(g - gq)
        nq = vpoff(gq + 1) - vpoff(gq)
        c = c + dl * nq * (dl + merge(cg2, cg5, kind == 2))
      end do
    case (3, 6)
      do sc = 0, ncls - 1
        h = g - sc
        if (h < 0 .or. h >= npg) cycle
        nc = vcoff(sc + 1) - vcoff(sc)
        nqh = vpoff(h + 1) - vpoff(h)
        np = opoff(h + 1) - opoff(h)
        nj = ocoff(sc + 1) - ocoff(sc)
        if (kind == 3) then
          c = c + nc * nqh * (nqh + cg3)
        else
          c = c + nc * nqh * (nqh + np + nj + cg6)
        end if
      end do
    case (4)
      do gq = 0, npg - 1
        if (abs(g - gq) >= ncls) cycle
        dl = ndel(g - gq)
        nq = vpoff(gq + 1) - vpoff(gq)
        c = c + dl * nq * (dl + cg4)
      end do
      do sd = 0, ncls - 1
        do sm = 0, ncls - 1
          gq = g + sm - sd
          if (gq < 0 .or. gq >= npg) cycle
          nd = vcoff(sd + 1) - vcoff(sd)
          nm = ocoff(sm + 1) - ocoff(sm)
          nq = vpoff(gq + 1) - vpoff(gq)
          sy = g - sd
          ney = 0d0
          if (sy >= 0 .and. sy < ncls) ney = vcoff(sy + 1) - vcoff(sy) + ocoff(sy + 1) - ocoff(sy)
          c = c + nd * nm * nq * ney
        end do
      end do
    case (7, 10)
      do sd = 0, ncls - 1
        do sm = 0, ncls - 1
          gq = g + sm - sd + st
          if (gq < 0 .or. gq >= npg) cycle
          nd = vcoff(sd + 1) - vcoff(sd)
          nm = ocoff(sm + 1) - ocoff(sm)
          nq = vpoff(gq + 1) - vpoff(gq)
          ney = 0.5d0 * (xznvc(g - sd - sw) + xznoc(g - sd - sw) + xznvc(g - sd + st + sw) + xznoc(g - sd + st + sw))
          c = c + nd * nm * nq * (cg1 + ney)
        end do
      end do
    case (8, 11)
      do gq = 0, npg - 1
        if (abs(g - gq + st + sw) >= ncls) cycle
        dl = ndel(g - gq + st + sw)
        nq = vpoff(gq + 1) - vpoff(gq)
        c = c + dl * nq * (xzndel(g - gq + st) + cg2)
      end do
    case (9, 12)
      do sc = 0, ncls - 1
        h = g + st - sc
        if (h < 0 .or. h >= npg .or. h + sw < 0 .or. h + sw >= npg) cycle
        if (xzna(st, g) == 0 .or. xzna(st + sw, g) == 0) cycle
        nc = vcoff(sc + 1) - vcoff(sc)
        nqh = vpoff(h + 1) - vpoff(h)
        c = c + nc * nqh * (vpoff(h + sw + 1) - vpoff(h + sw) + cg3)
      end do
    end select
    xzbcost1 = c
  end function

  ! The batches of all kinds and their ranks (called by every rank when the
  ! plan is there).  rank 0 prints the balance that the model expects.
  subroutine xzdeal(only)
    use xzdata
    integer, intent(in), optional :: only       ! 1: the kinds of the T3 equation again, 2: those of Lambda
    integer :: kind, kk
    integer(8) :: n, g, s, nb, step, r, k, best, nbat, rk
    integer(8), allocatable :: lst(:)
    real(8), allocatable :: load(:)
    real(8) :: c, tot, worst, r0x
    character(len=32) :: arg
    integer :: larg
    logical :: triples

    n = max(npo, nto, 1_8)
    allocate(lst(n), load(0:nrank - 1))
    if (.not. present(only)) then
    if (allocated(bown)) deallocate(bown, bnum, bcost, bsec, rspeed, rload, roff)
    allocate(bown(0:n - 1, ndk), bnum(0:n - 1, ndk), bcost(0:n - 1, ndk), bsec(0:n - 1, ndk))
    allocate(rspeed(0:nrank - 1, ndk), rload(0:nrank - 1, ndk), roff(0:nrank - 1, ndk))
    roff = 0d0
    tint = 0d0
    bown = -1
    bnum = 0
    bcost = 0d0
    bsec = 0d0
    rspeed = 1d0
    rload = 0d0
    ksec = 0d0
    nrebal = 0
    ! option dist: the kinds that are dealt out, as digits ("123456": all, the default; "0": none)
    distk = .true.
    call xzoptget('dist', arg, larg)
    if (larg > 0) then
      do kind = 1, ndk
        distk(kind) = index(arg(1:larg), achar(48 + kind)) > 0
      end do
    end if
    r0extra = 0d0
    call xzoptget('r0load', arg, larg)
    if (larg > 0) read(arg(1:larg), *, end=10, err=10) r0extra
10  continue
    ! option rebal x: deal again when the slowest rank is the fraction x above the mean (0: never)
    call xzoptget('rebal', arg, larg)
    if (larg > 0) read(arg(1:larg), *, end=11, err=11) rebtol
11  continue
    end if
    if (myrank == 0 .and. nrank > 1 .and. .not. present(only) .and. (with_cc .or. with_lam)) then
      write(6, '(/,a,i4,a)') '   The iterations over ', nrank, ' ranks: batches of pairs and triples by the cost model'
      write(6, '(a)') '     loop        batches   slowest rank above the mean   share of rank 0'
    end if
    ! (the ladder before pass 2: the two end at the same meeting, and pass 2
    ! starts on a rank when its ladder is done)
    do kk = 1, ndk
      kind = kk
      if (mod(kk, 3) == 2) kind = kk + 1
      if (mod(kk, 3) == 0) kind = kk - 1
      if (present(only)) then
        if ((kind - 1) / 3 + 1 /= only) cycle
      else if (kind > 6) then
        ! (the kinds of a sigma vector: per label shift, xzdeal_eom)
        cycle
      end if
      triples = mod(kind, 3) == 0
      r0x = 0d0
      if (kind <= 6) r0x = r0extra(merge(1, 2, kind <= 3))
      step = merge(nbt, nbp, triples)
      nbat = 0
      tot = 0d0
      do g = 0, merge(ntg, npg, triples) - 1
        if (triples) then
          if (kind <= 6 .and. vtoff(g + 1) == vtoff(g)) cycle
          c = xzbcost1(kind, g)
          do s = otoff(g), otoff(g + 1) - 1, step
            nb = min(step, otoff(g + 1) - s)
            nbat = nbat + 1
            lst(nbat) = s
            bnum(s, kind) = nb
            bcost(s, kind) = c * nb
            tot = tot + c * nb
          end do
        else
          c = xzbcost1(kind, g)
          do s = opoff(g), opoff(g + 1) - 1, step
            nb = min(step, opoff(g + 1) - s)
            nbat = nbat + 1
            lst(nbat) = s
            bnum(s, kind) = nb
            bcost(s, kind) = c * nb
            tot = tot + c * nb
          end do
        end if
      end do
      ! the most expensive batch first (a stable insertion sort: every rank the same list)
      do k = 2, nbat
        s = lst(k)
        r = k - 1
        do while (r >= 1)
          if (bcost(lst(r), kind) >= bcost(s, kind)) exit
          lst(r + 1) = lst(r)
          r = r - 1
        end do
        lst(r + 1) = s
      end do
      load = 0d0
      ! (option r0load: a load of rank 0 by hand, a fraction of the loop)
      if (nrank > 1) load(0) = r0x * tot
      ! each batch to the rank that is done with it first: the seconds before
      ! its batches (pass 2: with those of its ladder), then its batches at
      ! its speed
      do k = 1, nbat
        best = 0
        if (distk(kind)) then
          do rk = 1, nrank - 1
            if (roff(rk, kind) + (load(rk) + bcost(lst(k), kind)) / rspeed(rk, kind) < &
                roff(best, kind) + (load(best) + bcost(lst(k), kind)) / rspeed(best, kind)) best = rk
          end do
        end if
        bown(lst(k), kind) = best
        load(best) = load(best) + bcost(lst(k), kind)
      end do
      rload(:, kind) = load
      if (nrank > 1) rload(0, kind) = load(0) - r0x * tot
      ! (pass 2 of a rank starts when its ladder is done)
      if (triples) roff(:, kind - 1) = roff(:, kind) + rload(:, kind) / rspeed(:, kind)
      if (myrank == 0 .and. nrank > 1 .and. .not. present(only) .and. (with_cc .or. with_lam)) then
        worst = maxval(load) / max(sum(load) / nrank, 1d-300) - 1d0
        write(6, '(5x,a10,i9,f18.2,a,f18.2,a)') dkname(kind), nbat, 100d0 * worst, ' %', &
          100d0 * (load(0) - r0x * tot) / max(tot, 1d-300), ' %'
      end if
    end do
    deallocate(lst, load)
  end subroutine

  ! The batches of the two residuals of an EOM sigma vector of the label
  ! shift s (classes), dealt with the speeds that were measured last (called
  ! by every rank when the problem of a shift starts, and from xzrebal_eom).
  ! Between the last two meetings of a sigma vector a rank does the ladder
  ! and pass 2 with R3, then those with T3.
  subroutine xzdeal_eom(s)
    integer(8), intent(in) :: s
    if (.not. allocated(bown)) call xzdeal()
    dshift = s
    roff(:, 7:12) = 0d0
    call xzdeal(3)
    roff(:, 12) = roff(:, 8) + rload(:, 8) / rspeed(:, 8)
    call xzdeal(4)
  end subroutine

  ! After a sigma vector (called by every rank): as xzrebal, for the six
  ! kinds of batches of a sigma vector and its three meetings (the sums of
  ! the two passes 1, sigma_3 to the owners of its rows).
  subroutine xzrebal_eom()
    real(8) :: mine(9), all(9, 0:nrank - 1), msg(1 + 6 * nrank), worst
    integer :: k
    integer(8) :: r
    if (nrank == 1) return
    mine(1:6) = ksec(7:12)
    mine(7:9) = tmeet
    ksec(7:12) = 0d0
    tmeet = 0d0
    all = 0d0
    call xzmpi_gather(mine, all, 9_8)
    msg = 0d0
    if (myrank == 0 .and. rebtol > 0d0) then
      worst = 0d0
      do k = 7, 9
        if (sum(all(k, :)) > 0d0) worst = max(worst, maxval(all(k, :)) * nrank / sum(all(k, :)) - 1d0)
      end do
      if (worst > rebtol .and. minval(all(1:6, :)) > 0d0 .and. minval(rload(:, 7:12)) > 0d0) then
        msg(1) = 1d0
        do r = 0, nrank - 1
          do k = 1, 6
            msg(1 + 6 * r + k) = rload(r, 6 + k) / all(k, r)
          end do
        end do
      end if
    end if
    call xzmpi_bcast(msg, 8_8 * (1 + 6 * nrank))
    if (msg(1) /= 1d0) return
    do r = 0, nrank - 1
      do k = 1, 6
        rspeed(r, 6 + k) = msg(1 + 6 * r + k)
      end do
    end do
    call xzdeal_eom(dshift)
    nrebal(3) = nrebal(3) + 1
  end subroutine

  ! The slices of the stored T3 for the mode that xzsetmode has set: equal
  ! numbers of elements as far as whole rows allow.
  subroutine xzslices()
    use xzdata
    integer(8) :: r, g, na, ni, t, a
    character(len=8) :: arg
    integer :: larg
    slices = .false.
    if (nrank == 1) return
    ! (CC3 without stored triples has no T3 to slice: rank 0 does the DIIS of T1 and T2)
    if (cc3mode == 2) return
    ! (the default with more than one rank; option slices 0: rank 0 sends T3,
    ! takes the sum of the residual and holds the DIIS vectors alone)
    slices = .true.
    call xzoptget('slices', arg, larg)
    if (larg > 0) slices = arg(1:1) /= '0'
    if (.not. slices) return
    ! (the rows of a rank take the operator on occupied triples: not its pair form)
    hhset = 1
    if (allocated(soff)) deallocate(soff, swoff)
    allocate(soff(0:nrank), swoff(0:nrank))
    soff(0) = 0
    do r = 1, nrank - 1
      t = (r * n3) / nrank
      soff(r) = n3
      do g = 0, ntg - 1
        ni = otoff(g + 1) - otoff(g)
        na = vtoff(g + 1) - vtoff(g)
        if (ni == 0 .or. na == 0) cycle
        if (t < boff(g) + ni * na) then
          a = max(0_8, (t - boff(g) + na - 1) / na)
          soff(r) = boff(g) + a * na
          exit
        end if
      end do
      soff(r) = max(soff(r), soff(r - 1))
    end do
    soff(nrank) = n3
    smax = 0
    do r = 0, nrank - 1
      smax = max(smax, soff(r + 1) - soff(r))
    end do
    swoff = nwe * soff
  end subroutine

  ! the rows a <= row < b of the block of the group g that are in the slice
  ! of this rank (all rows without slices)
  subroutine xzslrows(g, a, b)
    use xzdata
    integer(8), intent(in) :: g
    integer(8), intent(out) :: a, b
    integer(8) :: na, ni
    ni = otoff(g + 1) - otoff(g)
    na = vtoff(g + 1) - vtoff(g)
    a = 0
    b = ni
    if (.not. slices .or. na == 0) return
    a = min(ni, max(0_8, (soff(myrank) - boff(g) + na - 1) / na))
    b = min(ni, max(0_8, (soff(myrank + 1) - boff(g)) / na))
    if (soff(myrank + 1) <= boff(g)) b = 0
    if (b < a) b = a
  end subroutine

  ! Every rank tells rank 0 its node and what the node can give (called once,
  ! by all ranks).
  subroutine xznodes()
    use iso_c_binding
    use xzkern, only: xz_mem_avail
    use xzpar, only: xzmpi_node
    character(kind=c_char) :: name(64)
    real(8) :: mine(2)
    integer :: k
    name = c_null_char
    call xzmpi_node(name, 64_8)
    mine = 0d0
    do k = 1, 64
      if (name(k) == c_null_char) exit
      mine(1) = mine(1) * 131d0 + dble(iachar(name(k)))
      mine(1) = mod(mine(1), 9007199254740881d0 / 256d0)
    end do
    mine(2) = 1d-3 * xz_mem_avail()
    if (allocated(rnode)) deallocate(rnode)
    allocate(rnode(2, 0:nrank - 1))
    rnode = 0d0
    call xzmpi_gather(mine, rnode, 2_8)
  end subroutine

  ! rank 0: the work arrays of the ranks of every node (w0 words for rank 0,
  ! ww for each other rank) against what the node could give at the start
  subroutine xznodemem(w0, ww)
    integer(8), intent(in) :: w0, ww
    integer(8) :: r, q, nr
    real(8) :: need
    logical :: first
    if (nrank == 1 .or. .not. allocated(rnode)) return
    write(6, '(a)') '     the work arrays of the ranks, node by node (MB; rank 0: its plan):'
    do r = 0, nrank - 1
      first = .true.
      do q = 0, r - 1
        if (rnode(1, q) == rnode(1, r)) first = .false.
      end do
      if (.not. first) cycle
      need = 0d0
      nr = 0
      do q = r, nrank - 1
        if (rnode(1, q) /= rnode(1, r)) cycle
        nr = nr + 1
        need = need + 8d-6 * merge(w0, ww, q == 0)
      end do
      write(6, '(a,i5,a,i4,a,f12.1,a,f12.1)') '       node of rank ', r, ': ', nr, ' ranks, work arrays ', need, &
        '   memory available at the start ', rnode(2, r)
      if (rnode(2, r) > 0d0 .and. need > rnode(2, r)) then
        write(6, '(a)') ' @XZCCSDT-W, the work arrays of the ranks of this node are larger than the memory that the node'
        write(6, '(a)') '             had available: lower MEMORY_SIZE (rank 0 then puts the DIIS vectors into the file'
        write(6, '(a)') '             XZDIIS) or use fewer ranks per node.'
      end if
    end do
  end subroutine

  ! After a step (called by every rank): the seconds of the ranks in their
  ! batches of the three loops of the set (1: T3 equation, 2: Lambda) and up
  ! to the two meetings go to rank 0.  If at one of the meetings the last
  ! rank came more than rebtol (of the mean) after the mean, rank 0 sends
  ! what it measured - per rank the model cost per second in each loop and
  ! the seconds of everything else before each meeting - and the batches of
  ! the set are dealt again for the next step.
  subroutine xzrebal(set)
    integer, intent(in) :: set
    real(8) :: mine(5), all(5, 0:nrank - 1), msg(1 + 5 * nrank), worst
    integer :: k, kind, k1, k2, k3
    integer(8) :: r
    if (nrank == 1) return
    k1 = 3 * set - 2          ! pass 1
    k2 = 3 * set - 1          ! pass 2
    k3 = 3 * set              ! ladder
    mine(1:3) = ksec(k1:k3)
    mine(4:5) = tint
    ksec(k1:k3) = 0d0
    tint = 0d0
    all = 0d0
    call xzmpi_gather(mine, all, 5_8)
    msg = 0d0
    if (myrank == 0 .and. rebtol > 0d0) then
      worst = 0d0
      do k = 4, 5
        if (sum(all(k, :)) > 0d0) worst = max(worst, maxval(all(k, :)) * nrank / sum(all(k, :)) - 1d0)
      end do
      if (worst > rebtol .and. minval(all(1:3, :)) > 0d0 .and. minval(rload(:, k1:k3)) > 0d0) then
        msg(1) = 1d0
        do r = 0, nrank - 1
          ! speeds in the three loops
          msg(2 + 5 * r) = rload(r, k1) / all(1, r)
          msg(3 + 5 * r) = rload(r, k2) / all(2, r)
          msg(4 + 5 * r) = rload(r, k3) / all(3, r)
          ! the rest before the two meetings
          msg(5 + 5 * r) = max(0d0, all(4, r) - all(1, r))
          msg(6 + 5 * r) = max(0d0, all(5, r) - all(2, r) - all(3, r))
        end do
      end if
    end if
    call xzmpi_bcast(msg, 8_8 * (1 + 5 * nrank))
    if (msg(1) /= 1d0) return
    do r = 0, nrank - 1
      rspeed(r, k1) = msg(2 + 5 * r)
      rspeed(r, k2) = msg(3 + 5 * r)
      rspeed(r, k3) = msg(4 + 5 * r)
      roff(r, k1) = msg(5 + 5 * r)
      roff(r, k3) = msg(6 + 5 * r)
    end do
    call xzdeal(set)
    nrebal(set) = nrebal(set) + 1
  end subroutine

  ! option bcost file: the batches of this rank, their model cost and their
  ! seconds, to file.<rank> (what the cost model is checked against)
  subroutine xzbdump()
    use xzopt, only: xzoptget
    integer :: kind, larg
    integer(8) :: s
    character(len=256) :: arg
    character(len=4) :: tag
    if (.not. allocated(bown)) return
    call xzoptget('bcost', arg, larg)
    if (larg == 0) return
    write(tag, '(i4.4)') myrank
    open(unit=76, file=arg(1:larg) // '.' // tag, form='formatted', status='replace')
    write(76, '(a)') '# kind  first  number  rank  model cost  seconds'
    do kind = 1, ndk
      do s = 0, ubound(bown, 1)
        if (bown(s, kind) < 0) cycle
        if (nrank > 1 .and. bown(s, kind) /= myrank) cycle
        if (bsec(s, kind) == 0d0) cycle
        write(76, '(i3,3i8,es16.6,f14.6)') kind, s, bnum(s, kind), bown(s, kind), bcost(s, kind), bsec(s, kind)
      end do
    end do
    close(76)
  end subroutine

  ! The seconds of the ranks in the stages of the iterations (set 1: the T3
  ! equation, 2: the Lambda equations), printed by rank 0.
  subroutine xzdtable(set, ncyc)
    use xzmem, only: xzpeak
    use xzkern, only: xz_rss
    use xzdata, only: nwe, n3
    integer, intent(in) :: set
    integer(8), intent(in) :: ncyc
    real(8) :: nrw
    real(8) :: mine(nds + 4), all(nds + 4, 0:nrank - 1), rate
    integer(8) :: r, kb(2)
    if (nrank == 1) return
    mine(1:nds) = dsec(:, set)
    mine(nds + 1) = dwords(set)
    mine(nds + 2) = 0d0
    call xz_rss(kb)
    mine(nds + 3) = 8d-6 * xzpeak
    mine(nds + 4) = 1d-3 * kb(1)
    all = 0d0
    call xzmpi_gather(mine, all, int(nds + 4, 8))
    if (myrank > 0) return
    nrw = dble(nwe * n3)
    if (set <= 2) then
      write(6, '(/,a,a,a,i4,a)') '   The ranks in the ', merge('CCSDT ', 'Lambda', set == 1), ' iterations (seconds of ', &
        ncyc, ' cycles)'
    else
      write(6, '(/,a,i6,a)') '   The ranks in the sigma vectors of EOM (seconds of ', ncyc, ' sigma vectors)'
    end if
    write(6, '(a)') '     rank    pass 1  sweep+ladder    pass 2   waiting  sums of pass 1   waiting  sum of residual' // &
      '  amplitudes  ' // merge('on its rows ', 'rank 0 alone', slices .or. set > 2) // '     step   work array/MB   peak resident/MB'
    do r = 0, nrank - 1
      write(6, '(i9,f10.2,f14.2,f10.2,f10.2,f16.2,f10.2,f17.2,f12.2,f14.2,f9.2,f16.1,f19.1)') r, all(1, r), &
        all(2, r), all(3, r), all(9, r), all(4, r), all(10, r), all(5, r), all(6, r), all(8, r), all(7, r), &
        all(nds + 3, r), all(nds + 4, r)
    end do
    write(6, '(a,3(f8.2,a))') '     the slowest rank above the mean: pass 1 ', &
      100d0 * (maxval(all(1, :)) / max(sum(all(1, :)) / nrank, 1d-30) - 1d0), ' %, sweep and ladder ', &
      100d0 * (maxval(all(2, :)) / max(sum(all(2, :)) / nrank, 1d-30) - 1d0), ' %, pass 2 ', &
      100d0 * (maxval(all(3, :)) / max(sum(all(3, :)) / nrank, 1d-30) - 1d0), ' %'
    rate = 0d0
    if (maxval(all(6, 1:)) > 0d0) rate = 8d-9 * all(nds + 1, 1) / maxval(all(6, 1:))
    if (set > 2) then
      ! (the sigma vectors: the vector to a rank, sigma_3 from a rank, in all)
      if (nrebal(3) > 0) then
        write(6, '(a,i4,a)') '     the batches were dealt again ', nrebal(3), ' times (by the measured speeds); at the end, rank by rank'
        write(6, '(a,*(f7.3))') '       speed in pass 1 with R3 (mean 1): ', rspeed(:, 7) * nrank / sum(rspeed(:, 7))
        write(6, '(a,*(f7.3))') '       speed in pass 2 with R3:          ', rspeed(:, 8) * nrank / sum(rspeed(:, 8))
        write(6, '(a,*(f7.3))') '       speed in pass 2 with T3:          ', rspeed(:, 11) * nrank / sum(rspeed(:, 11))
      end if
      write(6, '(a,f10.2,a,f8.3,a)') '     the vectors: ', 8d-9 * all(nds + 1, 1), ' GB into a rank in all, ', rate, &
        ' GB/s into the slowest receiver (all ranks send and receive at once)'
      if (maxval(all(5, :)) > 0d0) write(6, '(a,f8.3,a)') '     sigma_3 to the owners of its rows: ', &
        8d-9 * all(nds + 1, 1) / maxval(all(5, :)), ' GB/s'
      return
    end if
    if (nrebal(set) > 0) then
      write(6, '(a,i4,a)') '     the batches were dealt again ', nrebal(set), &
        ' times (by the measured speeds); at the end, rank by rank'
      write(6, '(a,*(f7.3))') '       speed in pass 1 (mean 1):      ', rspeed(:, 3 * set - 2) * nrank / sum(rspeed(:, 3 * set - 2))
      write(6, '(a,*(f7.3))') '       speed in pass 2:               ', rspeed(:, 3 * set - 1) * nrank / sum(rspeed(:, 3 * set - 1))
      write(6, '(a,*(f7.3))') '       speed in the ladder:           ', rspeed(:, 3 * set) * nrank / sum(rspeed(:, 3 * set))
      write(6, '(a,*(f7.2))') '       other work before the sums / s:', roff(:, 3 * set - 2)
      write(6, '(a,*(f7.2))') '       before the residual / s:       ', roff(:, 3 * set)
    end if
    write(6, '(a,f10.2,a,f8.3,a)') '     amplitudes to a rank: ', 8d-9 * all(nds + 1, 1), ' GB in all, ', rate, &
      ' GB/s into the slowest receiver (all ranks receive at once)'
    if (maxval(all(5, :)) > 0d0) write(6, '(a,f10.2,a,f8.3,a)') '     the residual: ', &
      8d-9 * ncyc * nrw, ' GB per rank in all, ', 8d-9 * ncyc * nrw / maxval(all(5, :)), &
      merge(' GB/s (every rank sends and receives its share at once)', &
            ' GB/s (the sum of all ranks onto rank 0)               ', slices)
  end subroutine
end module xzdist
