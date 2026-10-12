!
! The options of xzccsdt: the section %xzccsdt of the ZMAT.  The program is
! controlled by the ZMAT alone, as CFOUR's programs are - the job and the
! thresholds by CFOUR's keywords (xzjob), the excited states by their
! sections (xzeomd.F90), everything else here; no environment variable is
! read (OMP_NUM_THREADS and those of the MPI library are not ours).
!
!   %xzccsdt
!   name value
!   name value
!   ...
!
! one option per line, the name first (letters of either case), its value
! the rest of the line (a name alone: the value 1); the section ends at an
! empty line, at a line that starts with % or at the end of the file.  A
! name that is not in the list below stops the program with the list.
! Every rank reads the ZMAT of its run directory.
!
! xzoptget(name, value, length) is how the program asks: the value as text
! and its length, 0 if the option is not given (the call has the form of
! get_environment_variable, which it replaced).  The options and their
! defaults: docs/native.md, section 18.
!
module xzopt
  implicit none
  private
  public :: xzoptget
  integer, parameter :: nuser = 24, ndev = 17, nopt = nuser + ndev
  ! the user options first, then the developer and test options
  character(len=10), parameter :: oname(nopt) = [character(len=10) :: &
    'labtol', 'imagtol', 'focktol', 'omegatol', 'foldwin', 'real', 'labels', 'cc3', 'start', 'hh', &
    'qmap', 'slices', 'bpr', 'rebal', 'dist', 'diis', 'plan', 'cc3w', 'eomvec', 'mpichunk', &
    'qtasks', 'qranks', 'cc4', 'cc4micro', &
    'ncyc', 'maxcor', 'debug', 'ddebug', 'ldebug', 'qdebug', 'den_active', 'wfail', 'eomtest', 'eomsig', &
    'eomsd', 'mpitest', 'qsample', 'qstop', 'qcost', 'bcost', 'r0load']
  character(len=256) :: oval(nopt) = ' '
  logical :: oset(nopt) = .false.
  logical :: oread = .false.
contains

  subroutine lower(s)
    character(len=*), intent(inout) :: s
    integer :: i, c
    do i = 1, len(s)
      c = iachar(s(i:i))
      if (c >= iachar('A') .and. c <= iachar('Z')) s(i:i) = achar(c + 32)
    end do
  end subroutine

  ! the section %xzccsdt of the ZMAT, once
  subroutine xzoptread()
    character(len=512) :: line, tag
    character(len=16) :: nm
    integer :: ios, k, i, j
    logical :: there, found
    oread = .true.
    inquire(file='ZMAT', exist=there)
    if (there) then
      open(unit=31, file='ZMAT', form='formatted', status='old', action='read', iostat=ios)
      do while (ios == 0)
        read(31, '(a)', iostat=ios) line
        if (ios /= 0) exit
        tag = adjustl(line)
        call lower(tag)
        if (tag(1:8) /= '%xzccsdt' .or. len_trim(tag) /= 8) cycle
        do
          read(31, '(a)', iostat=ios) line
          if (ios /= 0) exit
          line = adjustl(line)
          if (len_trim(line) == 0 .or. line(1:1) == '%') exit
          i = scan(line, ' ' // achar(9) // '=')
          nm = line(1:i - 1)
          call lower(nm)
          found = .false.
          do k = 1, nopt
            if (nm == oname(k)) then
              found = .true.
              oset(k) = .true.
              oval(k) = adjustl(line(i + 1:))
              if (len_trim(oval(k)) == 0) oval(k) = '1'
            end if
          end do
          if (.not. found) then
            write(6, '(/,a,a,a)') ' @XZCCSDT-F, the section %xzccsdt of the ZMAT: "', trim(nm), '" is not an option.'
            write(6, '(a)') '   The options (one per line, "name value"; docs/native.md, section 18):'
            write(6, '(4x,8a11)') (oname(j), j = 1, nuser)
            write(6, '(a)') '   developer and test options:'
            write(6, '(4x,8a11)') (oname(j), j = nuser + 1, nopt)
            call xzerrex
          end if
        end do
      end do
      close(31)
    end if
  end subroutine

  subroutine xzoptget(name, value, length)
    character(len=*), intent(in) :: name
    character(len=*), intent(out), optional :: value
    integer, intent(out), optional :: length
    integer :: k
    if (.not. oread) call xzoptread()
    if (present(value)) value = ' '
    if (present(length)) length = 0
    do k = 1, nopt
      if (name == oname(k)) then
        if (oset(k)) then
          if (present(value)) value = oval(k)
          if (present(length)) length = len_trim(oval(k))
        end if
        return
      end if
    end do
    write(6, '(/,a,a)') ' @XZCCSDT-F, xzoptget: an option that is not in the list of xzopt.F90: ', name
    call xzerrex
  end subroutine
end module xzopt
