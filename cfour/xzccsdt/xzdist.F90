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
!
! Without MPI, and with one rank, every batch is this rank's and nothing is
! sent: the serial program.
!
module xzdist
  use xzpar, only: myrank, nrank, xzmpi_bcast, xzmpi_bcast_w, xzmpi_send_w, xzmpi_recv_w, &
                   xzmpi_sum0_w, xzmpi_allsum_w, xzmpi_gather, xzmpi_max, xzmpi_sum, xzmpi_chunk
  implicit none
  integer, parameter :: ndk = 6
  character(len=10), parameter :: dkname(ndk) = [character(len=10) :: 'pass 1', 'pass 2', 'pp ladder', &
    'L pass 1', 'L pass 2', 'L ladder']
  integer(8), allocatable :: bown(:, :)       ! (first pair or triple of the batch, kind): its rank (-1: no batch starts there)
  integer(8), allocatable :: bnum(:, :)       ! pairs or triples of the batch
  real(8), allocatable :: bcost(:, :)         ! its cost by the model
  real(8), allocatable :: bsec(:, :)          ! seconds that this rank spent in it (all iterations)
  integer :: dkind = 0                        ! the kind of the loop that runs; 0: every batch is done
  logical :: asworker = .false.               ! a dry run for the memory of a rank other than 0
  logical :: distk(ndk) = .true.              ! XZ_DIST: which kinds are dealt out (else all to rank 0)
  real(8) :: rplan(16) = 0d0                  ! the plan of rank 0: words of a worker, the 8 batch sizes
  integer(8) :: maxthr = 0                    ! the largest number of threads of a rank
  real(8) :: r0extra(2) = 0d0                 ! XZ_R0LOAD: what rank 0 does alone, as a fraction of the
                                              ! distributed work of an iteration (T3 equation, Lambda)
  ! seconds of this rank, (stage, 1 T3 equation / 2 Lambda): 1 pass 1, 2 the
  ! sweep with the pp ladder, 3 pass 2, 4 sums of pass 1 (and waiting for the
  ! others), 5 sum of the residual (and waiting), 6 amplitudes from rank 0,
  ! 7 the whole step, 8 what only rank 0 does
  integer, parameter :: nds = 8
  real(8) :: dsec(nds, 2) = 0d0
  real(8) :: dwords(2) = 0d0                  ! words of amplitudes sent to the ranks (per receiving rank)
  real(8), private :: tb0 = 0d0
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
    if (dkind == 0 .or. .not. allocated(bsec)) return
    bsec(start, dkind) = bsec(start, dkind) + omp_get_wtime() - tb0
  end subroutine

  ! The cost of one pair (kinds 1, 2, 4, 5) or one triple (3, 6) of the
  ! group g, in operations of the products of its loop.  One class: the same
  ! number for every pair or triple.  Measured: docs/native.md, section 15.
  real(8) function xzbcost1(kind, g)
    use xzdata
    integer, intent(in) :: kind
    integer(8), intent(in) :: g
    integer(8) :: sd, sm, sy, gq, sc, h
    real(8) :: nd, nm, nq, ney, dl, nc, nqh, c, np, nj
    c = 0d0
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
    end select
    xzbcost1 = c
  end function

  ! The batches of all kinds and their ranks (called by every rank when the
  ! plan is there).  rank 0 prints the balance that the model expects.
  subroutine xzdeal()
    use xzdata
    integer :: kind
    integer(8) :: n, g, s, nb, step, r, k, best, nbat, rk
    integer(8), allocatable :: lst(:)
    real(8), allocatable :: load(:)
    real(8) :: c, tot, worst
    character(len=32) :: arg
    integer :: larg
    logical :: triples

    n = max(npo, nto, 1_8)
    if (allocated(bown)) deallocate(bown, bnum, bcost, bsec)
    allocate(bown(0:n - 1, ndk), bnum(0:n - 1, ndk), bcost(0:n - 1, ndk), bsec(0:n - 1, ndk))
    allocate(lst(n), load(0:nrank - 1))
    bown = -1
    bnum = 0
    bcost = 0d0
    bsec = 0d0
    ! XZ_DIST: the kinds that are dealt out, as digits ("123456": all, the default; "0": none)
    distk = .true.
    call get_environment_variable('XZ_DIST', arg, larg)
    if (larg > 0) then
      do kind = 1, ndk
        distk(kind) = index(arg(1:larg), achar(48 + kind)) > 0
      end do
    end if
    r0extra = 0d0
    call get_environment_variable('XZ_R0LOAD', arg, larg)
    if (larg > 0) read(arg(1:larg), *, end=10, err=10) r0extra
10  continue
    if (myrank == 0 .and. nrank > 1) then
      write(6, '(/,a,i4,a)') '   The iterations over ', nrank, ' ranks: batches of pairs and triples by the cost model'
      write(6, '(a)') '     loop        batches   slowest rank above the mean   share of rank 0'
    end if
    do kind = 1, ndk
      triples = kind == 3 .or. kind == 6
      step = merge(nbt, nbp, triples)
      nbat = 0
      tot = 0d0
      do g = 0, merge(ntg, npg, triples) - 1
        if (triples) then
          if (vtoff(g + 1) == vtoff(g)) cycle
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
      ! (what rank 0 does alone counts as load of rank 0)
      if (nrank > 1) load(0) = r0extra(merge(1, 2, kind <= 3)) * tot
      do k = 1, nbat
        best = 0
        if (distk(kind)) then
          do rk = 1, nrank - 1
            if (load(rk) < load(best)) best = rk
          end do
        end if
        bown(lst(k), kind) = best
        load(best) = load(best) + bcost(lst(k), kind)
      end do
      if (myrank == 0 .and. nrank > 1) then
        worst = maxval(load) / max(sum(load) / nrank, 1d-300) - 1d0
        write(6, '(5x,a10,i9,f18.2,a,f18.2,a)') dkname(kind), nbat, 100d0 * worst, ' %', &
          100d0 * (load(0) - r0extra(merge(1, 2, kind <= 3)) * tot) / max(tot, 1d-300), ' %'
      end if
    end do
    deallocate(lst, load)
  end subroutine

  ! XZ_BCOST=file: the batches of this rank, their model cost and their
  ! seconds, to file.<rank> (what the cost model is checked against)
  subroutine xzbdump()
    integer :: kind, larg
    integer(8) :: s
    character(len=256) :: arg
    character(len=4) :: tag
    if (.not. allocated(bown)) return
    call get_environment_variable('XZ_BCOST', arg, larg)
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
    integer, intent(in) :: set
    integer(8), intent(in) :: ncyc
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
    write(6, '(/,a,a,a,i4,a)') '   The ranks in the ', merge('CCSDT ', 'Lambda', set == 1), ' iterations (seconds of ', &
      ncyc, ' cycles)'
    write(6, '(a)') '     rank    pass 1  sweep+ladder    pass 2  sums of pass 1  sum of residual  amplitudes' // &
      '   rank 0 alone     step   work array/MB   peak resident/MB'
    do r = 0, nrank - 1
      write(6, '(i9,f10.2,f14.2,f10.2,f16.2,f17.2,f12.2,f15.2,f9.2,f16.1,f19.1)') r, all(1, r), all(2, r), &
        all(3, r), all(4, r), all(5, r), all(6, r), all(8, r), all(7, r), all(nds + 3, r), all(nds + 4, r)
    end do
    rate = 0d0
    if (maxval(all(6, 1:)) > 0d0) rate = 8d-9 * all(nds + 1, 1) / maxval(all(6, 1:))
    write(6, '(a,f10.2,a,f8.3,a)') '     amplitudes to a rank: ', 8d-9 * all(nds + 1, 1), ' GB in all, ', rate, &
      ' GB/s (the slowest receiver; rank 0 sends to all at once)'
    write(6, '(a)') '     (sums: the time from the end of the rank''s own batches to the end of the sum,'
    write(6, '(a)') '      i.e. waiting for the slowest rank and the transfer)'
  end subroutine
end module xzdist
