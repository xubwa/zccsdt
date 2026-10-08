!
! The parallel layer of (Q): ranks (xzmpi.c - one rank without MPI) and the
! list of tasks.
!
! A task is a set of occupied quadruples i<j<k<l that share the intermediate
! Xv of T2 with <vv||vv>: the quadruples of one largest index l, or - where
! one l is more than a fair share of a rank - those of one l whose smallest
! index i lies in a range [i0, i1).  A task needs Xv(x, l) for the x in
! i0 .. l-1 only; it reads the amplitudes and the integrals and returns two
! numbers.  The list depends on the number of tasks asked for (one per l for
! one rank; about eight per rank otherwise; XZ_QTASKS), not on anything a
! rank computes, so every rank builds the same list.
!
! Ranks: static.  The tasks are given to the ranks by their estimated cost
! (largest first, each to the rank with the least so far), every rank does
! its own in the order of l - so that the tasks of one l on a rank share
! their Xv - and one reduction at the end adds the energies.  No rank
! serves another while it computes; the price is that the balance is that
! of the cost model (operation counts of the products and of Xv), which the
! per-rank times of the output let one judge.
!
! Restart: every rank appends a line per finished task to its file
! XZQCHK.<rank> (task, its two energies, seconds).  A later run with the
! same task list takes the finished tasks of all such files as done.  The
! same lines give the progress report; XZ_QSAMPLE=f does the fraction f of
! the quadruples of every task and extrapolates the time.
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
  real(8) :: qsample = 0d0                           ! XZ_QSAMPLE: fraction of the quadruples (0: all)
  character(len=8) :: qkind = '(Q)'                  ! what is on the left: '(Q)' ((Q)_L: not yet)
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

  ! The task list for about `want' tasks (0: one per l), their costs and
  ! their ranks.  cq: cost of a quadruple, cx: of Xv for one x.
  subroutine xzq_build(no, want, cq, cx)
    integer(8), intent(in) :: no, want
    real(8), intent(in) :: cq, cx
    integer(8) :: l, total, nql, np, k, i, i0, acc, it, r, best, pass, n
    real(8) :: target
    real(8), allocatable :: load(:)
    logical, allocatable :: taken(:)
    total = no * (no - 1) * (no - 2) * (no - 3) / 24
    target = dble(total) / dble(max(want, 1_8))
    do pass = 1, 2
      n = 0
      do l = 3, no - 1
        nql = l * (l - 1) * (l - 2) / 6
        np = 1
        if (want > 0) np = min(l - 2, max(1_8, nint(dble(nql) / target, 8)))
        i0 = 0
        acc = 0
        k = 1
        do i = 0, l - 3
          acc = acc + (l - 1 - i) * (l - 2 - i) / 2
          if (i == l - 3 .or. (k < np .and. dble(acc) >= dble(k) * dble(nql) / dble(np))) then
            n = n + 1
            if (pass == 2) then
              tl(n) = l
              ti0(n) = i0
              ti1(n) = i + 1
              tnq(n) = xzq_count(l, i0, i + 1)
              tcost(n) = cq * dble(tnq(n)) + cx * dble(l - i0)
            end if
            i0 = i + 1
            k = k + 1
          end if
        end do
      end do
      if (pass == 1) then
        ntask = n
        if (allocated(tl)) deallocate(tl, ti0, ti1, tnq, town, tcost, te, tsec, tdone)
        allocate(tl(n), ti0(n), ti1(n), tnq(n), town(n), tcost(n), te(2, n), tsec(n), tdone(n))
        te = 0d0
        tsec = 0d0
        tdone = .false.
      end if
    end do
    ! the ranks: the largest task first, each to the rank with the least load
    allocate(load(0:nrank-1), taken(ntask))
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
    deallocate(load, taken)
  end subroutine

  character(len=11) function xzq_file(r)
    integer(8), intent(in) :: r
    write(xzq_file, '(a7,i4.4)') 'XZQCHK.', r
  end function

  ! the first line of a checkpoint file: what the tasks belong to
  subroutine xzq_header(line, no, nv, nwe, ncls)
    character(len=*), intent(out) :: line
    integer(8), intent(in) :: no, nv, nwe, ncls
    write(line, '(a,1x,a,5i8)') 'XZQCHK', trim(qkind), no, nv, ntask, nwe, ncls
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
