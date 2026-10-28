!
! The options of xzccsdt: the section %xzccsdt of the ZMAT.  The program is
! controlled by the ZMAT alone, as CFOUR's programs are - the job and the
! thresholds by CFOUR's keywords (xzjob), the excited states by their
! sections (xzeomd.f90), everything else here; no environment variable is
! read (OMP_NUM_THREADS and those of the MPI library are not ours).
!
!   %xzccsdt
!   name value
!   name value
!   ...
!
! one option per line, the name first (letters of either case), its value
! the rest of the line after a blank or '=' (a name alone: the value 1) -
! "name value" and CFOUR's "name=value" (X2C-OPTIONS.md of the CFOUR tree,
! libx2c x2c_getopt) alike; lines that start with # or ! are comments; the
! section ends at an empty line, at a line that starts with % or *, or at
! the end of the file.  A name that is not in the list below stops the
! program with the list (vtdense is an option of xx2cvtran in the same
! section: accepted, not used here).  Every rank reads the ZMAT of its run
! directory.  (The program has its own reader rather than x2c_getopt of
! libx2c so that a misspelt option stops the run, and so that it builds
! against a CFOUR tree without x2c_getopt.)
!
! The program asks as CFOUR's programs do (libx2c x2c_getopt.f90), with
! x2c_getopt('xzccsdt', name, value, length) - the value as text and its
! length, 0 if the option is not given (the form of get_environment_variable,
! which it replaced) - and x2c_hasopt('xzccsdt', name, length): the module
! procedures here, which every routine that uses xzdata sees (they take the
! place of the external routines of libx2c, which are not linked: the calls
! are those of CFOUR's zccsdt/, word for word).  xzoptget(name, value,
! length) is the same without the section.  The options and their defaults:
! docs/native.md, section 18.
!
module xzopt
  implicit none
  private
  public :: xzoptget, x2c_getopt, x2c_hasopt
  integer, parameter :: nuser = 28, ndev = 37, nopt = nuser + ndev
  ! the user options first, then the developer and test options
  character(len=10), parameter :: oname(nopt) = [character(len=10) :: &
    'labtol', 'imagtol', 'focktol', 'omegatol', 'foldwin', 'real', 'labels', 'cc3', 'start', 'hh', &
    'qmap', 'slices', 'bpr', 'rebal', 'dist', 'diis', 'plan', 'cc3w', 'eomvec', 'mpichunk', &
    'qtasks', 'qranks', 'cc4', 'cc4micro', 'cc4eom', 'cc4acc', 'lsolver', 'eomvdir', &
    'ncyc', 'maxcor', 'debug', 'ddebug', 'ldebug', 'qdebug', 'den_active', 'wfail', 'eomtest', 'eomsig', &
    'eomsd', 'mpitest', 'qsample', 'qstop', 'qcost', 'bcost', 'r0load', 'cc4stop', 'lkry', 'lritz', 'lprec', 'deasig', 'deaonly', 'deabatch', 'deaxv', &
    'deaplan', 'dea5pool', 'dea5z4', 'dea5xb', 'dea5xa', 'terms', 'ampin', 'ampout', 'eomkeep', 'eomstart', &
    'eomcyc', 'vtdense']
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
          if (len_trim(line) == 0 .or. line(1:1) == '%' .or. line(1:1) == '*') exit
          if (line(1:1) == '#' .or. line(1:1) == '!') cycle
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
            write(6, '(a)') '   The options (one per line, "name value" or "name=value"; docs/native.md, section 18):'
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
    write(6, '(/,a,a)') ' @XZCCSDT-F, xzoptget: an option that is not in the list of xzopt.f90: ', name
    call xzerrex
  end subroutine

  ! the forms of libx2c (x2c_getopt, x2c_hasopt): the section is always %xzccsdt
  subroutine xzoptsec(section)
    character(len=*), intent(in) :: section
    if (section /= 'xzccsdt') then
      write(6, '(/,a,a)') ' @XZCCSDT-F, x2c_getopt: the options of this program are those of %xzccsdt, not %', section
      call xzerrex
    end if
  end subroutine

  subroutine x2c_getopt(section, name, value, length)
    character(len=*), intent(in) :: section, name
    character(len=*), intent(out) :: value
    integer, intent(out), optional :: length
    call xzoptsec(section)
    call xzoptget(name, value, length)
  end subroutine

  subroutine x2c_hasopt(section, name, length)
    character(len=*), intent(in) :: section, name
    integer, intent(out) :: length
    call xzoptsec(section)
    call xzoptget(name, length=length)
  end subroutine
end module xzopt
