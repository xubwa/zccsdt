!
! The parallel layer of (Q): ranks (xzmpi.c - one rank without MPI) and the
! list of tasks.
!
! A task is a set of occupied quadruples i<j<k<l that share the intermediate
! Xv of T2 with <vv||vv>: a group - the quadruples of one largest index l -
! or a piece of a group: those of one l whose smallest index i lies in a
! range [i0, i1).  A task needs Xv(x, l) for the x in i0 .. l-1 only; it
! reads the amplitudes and the integrals and returns two numbers.
!
! Ranks: static.  Whole groups go to the ranks (the largest task first, each
! to the rank with the least cost so far), so that no Xv is formed twice; a
! group is cut into pieces only as far as the balance of the ranks needs it
! (xzq_build).  The list depends on the number of ranks and on the cost
! model (xzq_costs: the virtual quadruples of the sector of an occupied
! quadruple; the products of Xv), not on anything a rank computes, so every
! rank builds the same list.  Every rank does its own tasks in the order of
! l - the tasks of one l on a rank share their Xv - and one reduction at
! the end adds the energies.  No rank serves another while it computes; the
! price is that the balance is that of the cost model, which the per-rank
! times of the output let one judge.
!
! Restart: every rank appends a line per finished task to its file
! XZQCHK.<rank> (task, its two energies, seconds).  A later run with the
! same task list (the same number of ranks, or the same qtasks) takes
! the finished tasks of all such files as done.  The same lines give the
! progress report; qsample=f does the fraction f of the quadruples of
! every task, evenly spread, and extrapolates the time.
!
module xzpar
  use iso_c_binding
  implicit none
  interface
    subroutine xzmpi_init() bind(C)
    end subroutine
    function xzmpi_rank() bind(C) result(r)
      import
      integer(c_int64_t) :: r
    end function
    function xzmpi_size() bind(C) result(r)
      import
      integer(c_int64_t) :: r
    end function
    function xzmpi_enabled() bind(C) result(r)
      import
      integer(c_int64_t) :: r
    end function
    subroutine xzmpi_bcast(buf, nbytes) bind(C)
      import
      type(*), dimension(*) :: buf
      integer(c_int64_t), value :: nbytes
    end subroutine
    subroutine xzmpi_sum(a, b, n) bind(C)
      import
      real(c_double) :: a(*), b(*)
      integer(c_int64_t), value :: n
    end subroutine
    subroutine xzmpi_max(a, b, n) bind(C)
      import
      real(c_double) :: a(*), b(*)
      integer(c_int64_t), value :: n
    end subroutine
    subroutine xzmpi_gather(a, b, n) bind(C)
      import
      real(c_double) :: a(*), b(*)
      integer(c_int64_t), value :: n
    end subroutine
    subroutine xzmpi_barrier() bind(C)
    end subroutine
    ! arrays of n double words in messages of at most xzmpi_chunk() words:
    ! from rank 0 to all; to / from one rank; the sum over the ranks in place
    ! on rank 0; the sum over the ranks in place on every rank
    subroutine xzmpi_bcast_w(buf, n) bind(C)
      import
      type(*), dimension(*) :: buf
      integer(c_int64_t), value :: n
    end subroutine
    subroutine xzmpi_send_w(buf, n, dest, tag) bind(C)
      import
      type(*), dimension(*) :: buf
      integer(c_int64_t), value :: n, dest, tag
    end subroutine
    subroutine xzmpi_recv_w(buf, n, src, tag) bind(C)
      import
      type(*), dimension(*) :: buf
      integer(c_int64_t), value :: n, src, tag
    end subroutine
    subroutine xzmpi_sum0_w(buf, n) bind(C)
      import
      type(*), dimension(*) :: buf
      integer(c_int64_t), value :: n
    end subroutine
    subroutine xzmpi_allsum_w(buf, n) bind(C)
      import
      type(*), dimension(*) :: buf
      integer(c_int64_t), value :: n
    end subroutine
    ! slices of an array (off: the first word of the slice of every rank, and
    ! the end): the contributions of all ranks summed on the slice of its
    ! owner; the slices of the owners to every rank
    subroutine xzmpi_slices_add(buf, off, tmp) bind(C)
      import
      type(*), dimension(*) :: buf, tmp
      integer(c_int64_t) :: off(*)
    end subroutine
    subroutine xzmpi_slices_put(buf, off) bind(C)
      import
      type(*), dimension(*) :: buf
      integer(c_int64_t) :: off(*)
    end subroutine
    subroutine xzmpi_set_chunk(n) bind(C)
      import
      integer(c_int64_t), value :: n
    end subroutine
    function xzmpi_chunk() bind(C) result(r)
      import
      integer(c_int64_t) :: r
    end function
    function xzmpi_selftest(n) bind(C) result(r)
      import
      integer(c_int64_t), value :: n
      integer(c_int64_t) :: r
    end function
    subroutine xzmpi_node(name, len) bind(C)
      import
      character(kind=c_char) :: name(*)
      integer(c_int64_t), value :: len
    end subroutine
    subroutine xzmpi_finalize() bind(C)
    end subroutine
    subroutine xzmpi_abort(code) bind(C)
      import
      integer(c_int64_t), value :: code
    end subroutine
  end interface

  integer(8) :: myrank = 0, nrank = 1

  ! ---- the tasks of (Q)
  integer(8) :: ntask = 0
  integer(8), allocatable :: tl(:), ti0(:), ti1(:)   ! l and the range of i
  integer(8), allocatable :: tnq(:)                  ! quadruples
  integer(8), allocatable :: town(:)                 ! the rank that does it
  real(8), allocatable :: tcost(:)                   ! estimated cost (arbitrary units)
  real(8), allocatable :: te(:, :)                   ! (2, task): its parts of E[Q] and E(Q) - E[Q]
  real(8), allocatable :: tsec(:)                    ! seconds it took
  logical, allocatable :: tdone(:)
  real(8) :: qsample = 0d0                           ! qsample: fraction of the quadruples (0: all)
  character(len=8) :: qkind = '(Q)'                  ! what is on the left: '(Q)' or '(Q)_L'
  real(8), allocatable :: qcli(:, :)                 ! (i, l): cost of the quadruples i<j<k<l of one i and l
  real(8), allocatable :: xcl(:, :)                  ! (x, l): cost of Xv(x, l)
  real(8), parameter :: qtol = 0.03d0                ! the slowest rank of the model over the mean, minus one
contains

  ! quadruples i<j<k<l with i0 <= i < i1
  integer(8) function xzq_count(l, i0, i1)
    integer(8), intent(in) :: l, i0, i1
    integer(8) :: i
    xzq_count = 0
    do i = i0, i1 - 1
      xzq_count = xzq_count + (l - 1 - i) * (l - 2 - i) / 2
    end do
  end function

  ! The cost tables of the tasks (units: one virtual quadruple of the sector
  ! of an occupied quadruple).  qc(M): the cost of an occupied quadruple of
  ! total class M - the virtual quadruples of that total class; xc(K): the
  ! cost of Xv for a pair (x, l) of total class K.  Measured: docs/native.md,
  ! section 14.
  subroutine xzq_costs(no, ncls, ocls, qc, xc)
    integer(8), intent(in) :: no, ncls, ocls(0:no-1)
    real(8), intent(in) :: qc(0:*), xc(0:*)
    integer(8) :: i, j, k, l
    real(8) :: s
    if (allocated(qcli)) deallocate(qcli, xcl)
    allocate(qcli(0:no-1, 0:no-1), xcl(0:no-1, 0:no-1))
    qcli = 0d0
    xcl = 0d0
    !$omp parallel do private(l, i, j, k, s) schedule(dynamic)
    do l = 3, no - 1
      do i = 0, l - 3
        s = 0d0
        do j = i + 1, l - 2
          do k = j + 1, l - 1
            s = s + qc(ocls(i) + ocls(j) + ocls(k) + ocls(l))
          end do
        end do
        qcli(i, l) = s
      end do
      do i = 0, l - 1
        xcl(i, l) = xc(ocls(i) + ocls(l))
      end do
    end do
  end subroutine

  ! the tasks of the groups l with npc(l) pieces each (cut where the cost of
  ! the quadruples passes k / npc of that of the group); fill = .false.: count
  subroutine xzq_pieces(no, npc, fill, n)
    integer(8), intent(in) :: no, npc(0:no-1)
    logical, intent(in) :: fill
    integer(8), intent(out) :: n
    integer(8) :: l, np, i, i0, k
    real(8) :: acc, tot
    n = 0
    do l = 3, no - 1
      np = max(1_8, min(l - 2, npc(l)))
      tot = sum(qcli(0:l - 3, l))
      i0 = 0
      acc = 0d0
      k = 1
      do i = 0, l - 3
        acc = acc + qcli(i, l)
        if (i == l - 3 .or. (k < np .and. acc >= dble(k) * tot / dble(np))) then
          n = n + 1
          if (fill) then
            tl(n) = l
            ti0(n) = i0
            ti1(n) = i + 1
            tnq(n) = xzq_count(l, i0, i + 1)
            tcost(n) = sum(qcli(i0:i, l)) + sum(xcl(i0:l - 1, l))
          end if
          i0 = i + 1
          k = k + 1
        end if
      end do
    end do
  end subroutine

  ! the tasks to the ranks: the largest first, each to the rank with the
  ! least so far.  worst: the largest load of a rank over the mean, minus one
  subroutine xzq_assign(worst)
    real(8), intent(out) :: worst
    real(8) :: load(0:nrank-1)
    logical :: taken(ntask)
    integer(8) :: k, it, best, r
    load = 0d0
    taken = .false.
    do k = 1, ntask
      best = 0
      do it = 1, ntask
        if (taken(it)) cycle
        if (best == 0) then
          best = it
        else if (tcost(it) > tcost(best)) then
          best = it
        end if
      end do
      taken(best) = .true.
      r = minloc(load, 1) - 1
      town(best) = r
      load(r) = load(r) + tcost(best)
    end do
    worst = maxval(load) * dble(nrank) / max(sum(load), 1d-300) - 1d0
  end subroutine

  subroutine xzq_alloc(n)
    integer(8), intent(in) :: n
    ntask = n
    if (allocated(tl)) deallocate(tl, ti0, ti1, tnq, town, tcost, te, tsec, tdone)
    allocate(tl(n), ti0(n), ti1(n), tnq(n), town(n), tcost(n), te(2, n), tsec(n), tdone(n))
    te = 0d0
    tsec = 0d0
    tdone = .false.
  end subroutine

  ! The task list and the ranks of the tasks (xzq_costs first).
  !   want = 0   the rule: whole groups - all quadruples of one largest index
  !              l, which share Xv(x, l) - go to the ranks, so that no Xv is
  !              formed twice; a group is cut into pieces (ranges of the
  !              smallest index i) only as far as the balance needs it: while
  !              the slowest rank of the cost model is more than qtol above
  !              the mean, the group of the largest task gets one piece more.
  !              The pieces of a group on different ranks each form the Xv
  !              they need (x from their smallest i on): that repeat is
  !              accepted, it is a small part of a piece (section 14).
  !   want > 0   about that many tasks of equal numbers of quadruples
  !              (qtasks: tests of the list and of restart)
  subroutine xzq_build(no, want)
    integer(8), intent(in) :: no, want
    integer(8) :: npc(0:no-1), best(0:no-1), l, n, it, big, total, nql, step
    real(8) :: worst, wbest, target
    npc = 1
    if (want > 0) then
      total = no * (no - 1) * (no - 2) * (no - 3) / 24
      target = dble(total) / dble(want)
      do l = 3, no - 1
        nql = l * (l - 1) * (l - 2) / 6
        npc(l) = min(l - 2, max(1_8, nint(dble(nql) / target, 8)))
      end do
    end if
    call xzq_pieces(no, npc, .false., n)
    call xzq_alloc(n)
    call xzq_pieces(no, npc, .true., n)
    call xzq_assign(worst)
    if (want > 0 .or. nrank == 1) return
    best = npc
    wbest = worst
    do step = 1, no * no
      if (worst <= qtol) exit
      ! one piece more for the group of the largest task that can be cut
      big = 0
      do it = 1, ntask
        if (ti1(it) - ti0(it) < 2) cycle
        if (npc(tl(it)) >= tl(it) - 2) cycle
        if (big == 0) then
          big = it
        else if (tcost(it) > tcost(big)) then
          big = it
        end if
      end do
      if (big == 0) exit
      npc(tl(big)) = npc(tl(big)) + 1
      call xzq_pieces(no, npc, .false., n)
      call xzq_alloc(n)
      call xzq_pieces(no, npc, .true., n)
      call xzq_assign(worst)
      if (worst < wbest) then
        wbest = worst
        best = npc
      end if
    end do
    if (worst > wbest) then
      npc = best
      call xzq_pieces(no, npc, .false., n)
      call xzq_alloc(n)
      call xzq_pieces(no, npc, .true., n)
      call xzq_assign(worst)
    end if
  end subroutine

  ! What the list means for Xv: nxv(1) pairs (x, l) there are, nxv(2) are
  ! formed over all ranks (a rank forms a pair once for all its tasks of
  ! that l), and the cost of the model of the repeats over the whole cost
  subroutine xzq_xvstat(no, nxv, extra, worst)
    integer(8), intent(in) :: no
    integer(8), intent(out) :: nxv(2)
    real(8), intent(out) :: extra, worst
    integer(8) :: l, r, it, lo
    real(8) :: load(0:nrank-1), c
    nxv = 0
    extra = 0d0
    load = 0d0
    do it = 1, ntask
      load(town(it)) = load(town(it)) + tcost(it)
    end do
    do l = 3, no - 1
      nxv(1) = nxv(1) + l
      c = sum(xcl(0:l - 1, l))
      do r = 0, nrank - 1
        lo = l
        do it = 1, ntask
          if (tl(it) == l .and. town(it) == r) lo = min(lo, ti0(it))
        end do
        nxv(2) = nxv(2) + l - lo
        extra = extra + sum(xcl(lo:l - 1, l))
      end do
      extra = extra - c
    end do
    extra = extra / max(sum(load), 1d-300)
    worst = maxval(load) * dble(nrank) / max(sum(load), 1d-300) - 1d0
  end subroutine

  ! a number that names the list (the checkpoint files carry it)
  integer(8) function xzq_listid()
    integer(8) :: it
    xzq_listid = ntask
    do it = 1, ntask
      xzq_listid = mod(xzq_listid * 1000003_8 + tl(it) * 65536 + ti0(it) * 256 + ti1(it), 2147483647_8)
    end do
  end function

  character(len=11) function xzq_file(r)
    integer(8), intent(in) :: r
    write(xzq_file, '(a7,i4.4)') 'XZQCHK.', r
  end function

  ! the first line of a checkpoint file: what the tasks belong to
  subroutine xzq_header(line, no, nv, nwe, ncls)
    character(len=*), intent(out) :: line
    integer(8), intent(in) :: no, nv, nwe, ncls
    write(line, '(a,1x,a,5i8,i12)') 'XZQCHK', trim(qkind), no, nv, ntask, nwe, ncls, xzq_listid()
  end subroutine

  ! The finished tasks of the checkpoint files of an earlier run (all ranks
  ! read the same files).  nold: how many.
  subroutine xzq_restore(no, nv, nwe, ncls, nold)
    integer(8), intent(in) :: no, nv, nwe, ncls
    integer(8), intent(out) :: nold
    character(len=128) :: head, line
    integer(8) :: r, it
    real(8) :: e1, e2, sec
    integer :: ios
    logical :: there
    nold = 0
    call xzq_header(head, no, nv, nwe, ncls)
    do r = 0, 1023
      inquire(file=xzq_file(r), exist=there)
      if (.not. there) cycle
      open(unit=76, file=xzq_file(r), form='formatted', status='old', action='read', iostat=ios)
      if (ios /= 0) cycle
      read(76, '(a)', iostat=ios) line
      if (ios == 0 .and. trim(line) == trim(head)) then
        do
          read(76, *, iostat=ios) it, e1, e2, sec
          if (ios /= 0) exit
          if (it < 1 .or. it > ntask) cycle
          if (.not. tdone(it)) nold = nold + 1
          tdone(it) = .true.
          te(1, it) = e1
          te(2, it) = e2
          tsec(it) = sec
        end do
      end if
      close(76)
    end do
  end subroutine

  ! one more finished task in the file of this rank
  subroutine xzq_save(it, no, nv, nwe, ncls, first)
    integer(8), intent(in) :: it, no, nv, nwe, ncls
    logical, intent(inout) :: first
    character(len=128) :: head
    integer(8) :: k
    integer :: ios
    if (first) then
      open(unit=76, file=xzq_file(myrank), form='formatted', status='replace', iostat=ios)
      call xzq_header(head, no, nv, nwe, ncls)
      if (ios == 0) write(76, '(a)', iostat=ios) trim(head)
      ! (the file is written anew: the tasks that earlier runs finished go
      ! into it again - in the file of every rank; that does no harm)
      do k = 1, ntask
        if (tdone(k) .and. k /= it .and. ios == 0) &
          write(76, '(i8,2es26.17,f12.3)', iostat=ios) k, te(1, k), te(2, k), tsec(k)
      end do
      first = .false.
    else
      open(unit=76, file=xzq_file(myrank), form='formatted', status='old', position='append', iostat=ios)
    end if
    if (ios == 0) write(76, '(i8,2es26.17,f12.3)', iostat=ios) it, te(1, it), te(2, it), tsec(it)
    if (ios == 0) close(76, iostat=ios)
    if (ios /= 0) write(6, '(a)') ' @XZCCSDT-W, the checkpoint of (Q) could not be written (no restart from here).'
  end subroutine

  subroutine xzq_cleanup()
    integer(8) :: r
    logical :: there
    integer :: ios
    do r = 0, 1023
      inquire(file=xzq_file(r), exist=there)
      if (.not. there) cycle
      open(unit=76, file=xzq_file(r), status='old', iostat=ios)
      if (ios == 0) close(76, status='delete', iostat=ios)
    end do
  end subroutine
end module xzpar
