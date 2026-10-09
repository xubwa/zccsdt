!
! The work array of the program: one array of MAXCOR double words that
! CFOUR's CRAPSI has allocated.  Every array of the calculation is a region
! of it, taken from a stack:
!
!     mark = xzmark()            on entering a stage
!     i = xzget(nwords, 'name')  a region; w(i) is its first word
!     call xzrel(mark)           on leaving the stage: all regions since mark
!
! Regions start on 64-byte boundaries.  A complex number is two words.
!
! In a dry run (xzdry = .true.) nothing may be touched: the scheduler runs
! through its allocations only, and xzpeak is what a real run with the same
! batch sizes takes.  The memory plan is made from such runs (xzplan.F90),
! so the plan and the placement cannot disagree.
!
module xzmem
  implicit none
  integer(8), parameter :: xzalign = 8      ! words: 64 bytes
  integer(8) :: xzbase = 1                  ! index in w of the first aligned word
  integer(8) :: xzsize = 0                  ! words from there on
  integer(8) :: xztop = 0                   ! words taken
  integer(8) :: xzpeak = 0                  ! the most that was ever taken
  logical :: xzdry = .false.
  integer(8), private :: save_top, save_peak
contains

  ! addr: the address of w(1); maxcor: its length in double words
  subroutine xzinit(addr, maxcor)
    integer(8), intent(in) :: addr, maxcor
    integer(8) :: skip
    skip = mod(64_8 - mod(addr, 64_8), 64_8) / 8
    xzbase = 1 + skip
    xzsize = maxcor - skip
    xzsize = xzsize - mod(xzsize, xzalign)
    xztop = 0
    xzpeak = 0
    xzdry = .false.
  end subroutine

  integer(8) function xzget(n, what)
    integer(8), intent(in) :: n
    character(len=*), intent(in) :: what
    integer(8) :: m
    m = max(n, 0_8)
    m = m + mod(xzalign - mod(m, xzalign), xzalign)
    xzget = xzbase + xztop
    if (.not. xzdry .and. xztop + m > xzsize) then
      write(6, '(/,a,a)') ' @XZCCSDT-F, the work array is too short for ', what
      write(6, '(a,i16,a,i16,a)') '   words needed: ', xztop + m, '   words there: ', xzsize, &
        '   (MEMORY_SIZE)'
      write(6, '(a)') '   This is an error of the memory plan, not of the input.'
      call xzerrex
    end if
    xztop = xztop + m
    xzpeak = max(xzpeak, xztop)
  end function

  integer(8) function xzmark()
    xzmark = xztop
  end function

  subroutine xzrel(mark)
    integer(8), intent(in) :: mark
    xztop = mark
  end subroutine

  ! a dry run starts from the present top (or from the mark `start'); xzdry_end
  ! gives its peak (words from the start of the work array) and restores the
  ! state
  subroutine xzdry_begin(start)
    integer(8), intent(in), optional :: start
    save_top = xztop
    save_peak = xzpeak
    if (present(start)) xztop = start
    xzpeak = xztop
    xzdry = .true.
  end subroutine

  subroutine xzdry_end(peak)
    integer(8), intent(out) :: peak
    peak = xzpeak
    xztop = save_top
    xzpeak = save_peak
    xzdry = .false.
  end subroutine
end module xzmem
