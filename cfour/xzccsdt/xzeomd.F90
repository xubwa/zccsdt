!
! EOM-EE-CCSDT (xzeom.inc, xzeoms.inc): what is not type-generic - the input,
! the label shifts and the layout of their vectors, the place of the vectors
! of the solver, the results.
!
! Input, in CFOUR's manner (the keywords and sections of its two-component
! EOM program xx2ceomcc where they exist; docs/eom.md, section 9):
!
!   ESTATE_CONV=N        the largest element of the residual of a normalized
!                        vector below 10**-N (default 5); the section
!                        %eomthrs (a real number on the next line) overrides
!   ESTATE_MAXCYC=N      iterations of the solver at most (default 40)
!   %eomnroot            roots (next line; default 1) - of every shift that
!                        the section %xzeom gives without a number of its own
!   %eomndav0            vectors of the subspace at most (next line; default
!                        20 + roots + 2, CFOUR's; at most 64)
!   %xzeom               a section of this program: lines "shift [roots]"
!                        until a line that is not two integers - the label
!                        shifts of the states, in the labels of the run (2 m_j
!                        from dgcosym: 2 Omega of the state for a closed
!                        shell).  Without it: shift 0.
!   %xzeomvec            0: the vectors of the roots are not written (no
!                        file XZEOMR); 1 (default): they are
!   %xzeomguess          0: unit vectors on the lowest single excitations of
!                        the shift (CFOUR's default guess); 1 (default): the
!                        roots of the singles-doubles block of the matrix
!                        from these unit vectors first
!
module xzeomd
  implicit none
  integer, parameter :: maxsh = 64, maxres = 512, nlead = 6
  integer(8) :: enshift = 0                 ! shifts asked for
  integer(8) :: eshl(maxsh) = 0             ! in labels
  integer(8) :: enr(maxsh) = 1              ! roots of each
  integer(8) :: emaxcyc = 40
  integer(8) :: emaxsub = 0                 ! 0: 20 + roots + 2
  real(8) :: ethr = 1d-5
  integer(8) :: eguess = 1
  integer(8) :: ewvec = 1                   ! the vectors of the roots go to the file XZEOMR
  integer(8) :: labstep = 1                 ! the step of the labels: a class is (label - smallest) / step

  ! ---- the vectors of the subspace (C: the basis, H: its sigma vectors):
  !      regions of the work array, or the files EOMCnnnn / EOMHnnnn of the
  !      run directory read in chunks through a buffer
  logical :: evfile = .false.
  integer(8) :: evlen = 0                   ! elements of a vector
  integer(8) :: evcap = 0                   ! vectors at most
  integer(8) :: evc0 = 0, evh0 = 0          ! in the work array: vector j of C at evc0 + (j-1) * words
  integer(8) :: evbuf = 0, evnch = 0        ! files: the buffer [vector][evnch elements]
  integer(8) :: evnc = 0, evnh = 0          ! files that are open
  real(8) :: evio = 0d0                     ! seconds in reading and writing them
  real(8) :: evbytes = 0d0                  ! bytes read and written

  ! ---- the results, for the table at the end and the file XZEOM
  integer(8) :: nres = 0
  integer(8) :: rshift(maxres) = 0, rroot(maxres) = 0, rnsig(maxres) = 0, rcyc(maxres) = 0
  real(8) :: rom(maxres) = 0d0, rim(maxres) = 0d0, rmx(maxres) = 0d0, rnm(maxres) = 0d0, rwt(3, maxres) = 0d0
  logical :: rconv(maxres) = .false.
  real(8) :: teomim = 0d0, teomsig = 0d0, teomsd = 0d0, teomvec = 0d0
  integer(8) :: neomsig = 0, neomsd = 0

contains

  ! the name of the file of vector j: EOMCnnnn (kind 1), EOMHnnnn (kind 2),
  ! the names of CFOUR's xx2ceomcc (libx2ccc/davhcfname)
  character(len=8) function xzevname(kind, j)
    integer, intent(in) :: kind
    integer(8), intent(in) :: j
    write(xzevname, '(a4,i4.4)') merge('EOMC', 'EOMH', kind == 1), j
  end function xzevname

  integer function xzevunit(kind, j)
    integer, intent(in) :: kind
    integer(8), intent(in) :: j
    xzevunit = 200 + (kind - 1) * 64 + int(j)
  end function xzevunit

  ! The input: the flags of joda and the sections of the ZMAT.
  subroutine xzeominp()
    integer(8) :: iflags
    common /flags/ iflags(100)
    character(len=256) :: line
    integer :: ios, ios2
    integer(8) :: nroot0, a, b
    logical :: there
    emaxcyc = iflags(97)
    if (emaxcyc <= 0) emaxcyc = 40
    ethr = 10d0 ** (-abs(iflags(98)))
    if (iflags(98) == 0) ethr = 1d-5
    nroot0 = 1
    enshift = 0
    emaxsub = 0
    eguess = 1
    ewvec = 1
    inquire(file='ZMAT', exist=there)
    if (.not. there) then
      enshift = 1
      return
    end if
    open(unit=30, file='ZMAT', form='formatted', status='old', action='read', iostat=ios)
    do while (ios == 0)
      read(30, '(a)', iostat=ios) line
      if (ios /= 0) exit
      if (index(line, '%eomnroot') /= 0) then
        read(30, *, iostat=ios2) nroot0
      else if (index(line, '%eomthrs') /= 0) then
        read(30, *, iostat=ios2) ethr
      else if (index(line, '%eomndav0') /= 0) then
        read(30, *, iostat=ios2) emaxsub
      else if (index(line, '%xzeomguess') /= 0) then
        read(30, *, iostat=ios2) eguess
      else if (index(line, '%xzeomvec') /= 0) then
        read(30, *, iostat=ios2) ewvec
      else if (index(line, '%xzeom') /= 0) then
        do
          read(30, '(a)', iostat=ios) line
          if (ios /= 0) exit
          a = 0
          b = -1
          read(line, *, iostat=ios2) a, b
          if (ios2 /= 0) then
            b = -1
            read(line, *, iostat=ios2) a
            if (ios2 /= 0) exit
          end if
          if (enshift < maxsh) then
            enshift = enshift + 1
            eshl(enshift) = a
            enr(enshift) = b
          end if
        end do
      end if
    end do
    close(30)
    if (enshift == 0) then
      enshift = 1
      eshl(1) = 0
      enr(1) = -1
    end if
    do a = 1, enshift
      if (enr(a) <= 0) enr(a) = max(nroot0, 1_8)
    end do
    ! a test hook: XZ_EOMCYC=n iterations at most (ESTATE_MAXCYC is in JOBARC, from xjoda)
    call get_environment_variable('XZ_EOMCYC', line, ios)
    if (ios > 0) read(line(1:ios), *, iostat=ios2) emaxcyc
  end subroutine xzeominp

  ! The layout of a rank-3 array of the label shift s (classes): the blocks
  ! (occupied triples of group g) x (virtual triples of group g + s), one
  ! after the other (xzdata: eshift, boff_s, n3s; obase[I] at jobase_s, which
  ! the caller has taken from the stack).  One class: the packed block.
  subroutine xzeomlay(w, s)
    use xzdata
    use xzmem
    real(8) :: w(*)
    integer(8), intent(in) :: s
    integer(8) :: g
    external :: xzeomfill
    eshift = s
    if (ncls == 1) then
      eshift = 0
      n3s = nto * nvt
      return
    end if
    n3s = 0
    do g = 0, ntg - 1
      boff_s(g) = n3s
      n3s = n3s + (otoff(g + 1) - otoff(g)) * xzna(s, g)
    end do
    boff_s(ntg) = n3s
    if (.not. xzdry) call xzeomfill(nto, s, w(jobase_s))
  end subroutine xzeomlay

  ! The step of the labels of the active spinors (xzclasses), for the shifts
  ! of the input: a shift D in labels is D / step classes.
  subroutine xzeomstep()
    use xzdata
    integer(8) :: lmin, p, a, b, t
    labstep = 1
    if (ncls == 1) return
    lmin = minval(lab(nfrozen:nfrozen + no + nv - 1))
    labstep = 0
    do p = nfrozen, nfrozen + no + nv - 1
      a = lab(p) - lmin
      b = labstep
      do while (b /= 0)
        t = mod(a, b)
        a = b
        b = t
      end do
      labstep = a
    end do
    if (labstep == 0) labstep = 1
  end subroutine xzeomstep

  ! The single excitations of the class shift s (all of them with one class),
  ! by their differences of spinor energies: list(k) = i * nv + a, the place
  ! of the k-th lowest in r1[i][a]; n of them, at most nmax returned.
  subroutine xzesingles(s, eps, nmax, list, n)
    use xzdata
    integer(8), intent(in) :: s, nmax
    real(8), intent(in) :: eps(0:no+nv-1)
    integer(8), intent(out) :: list(nmax), n
    real(8) :: d(nmax), x
    integer(8) :: i, a, k, m
    n = 0
    m = 0
    do i = 0, no - 1
      do a = 0, nv - 1
        if (ncls > 1) then
          if (vcls(a) - ocls(i) /= s) cycle
        end if
        n = n + 1
        x = eps(no + a) - eps(i)
        ! insertion among the nmax lowest
        k = m
        do while (k >= 1)
          if (d(k) <= x) exit
          k = k - 1
        end do
        if (k >= nmax) cycle
        if (m < nmax) m = m + 1
        d(k + 2:m) = d(k + 1:m - 1)
        list(k + 2:m) = list(k + 1:m - 1)
        d(k + 1) = x
        list(k + 1) = i * nv + a
      end do
    end do
  end subroutine xzesingles
end module xzeomd

! obase[I] of the layout of the label shift s (xzeomlay)
subroutine xzeomfill(n, s, obase)
  use xzdata
  implicit none
  integer(8), intent(in) :: n, s
  integer(8), intent(out) :: obase(0:n-1)
  integer(8) :: gg, it
  do gg = 0, ntg - 1
    do it = otoff(gg), otoff(gg + 1) - 1
      obase(it) = boff_s(gg) + (it - otoff(gg)) * xzna(s, gg)
    end do
  end do
end subroutine xzeomfill

! e2(P) = e(pa(P)) + e(pb(P)): the sums of the spinor energies of the pairs
subroutine xzepsum(np, n, pa, pb, e, e2)
  implicit none
  integer(8), intent(in) :: np, n
  integer(8), intent(in) :: pa(0:np-1), pb(0:np-1)
  real(8), intent(in) :: e(0:n-1)
  real(8), intent(out) :: e2(0:np-1)
  integer(8) :: p
  do p = 0, np - 1
    e2(p) = e(pa(p)) + e(pb(p))
  end do
end subroutine xzepsum
