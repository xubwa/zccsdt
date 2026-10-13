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

  ! ---- more than one rank (docs/native.md, section 16.7).  Rank 0 runs the
  !      solver; R3 and sigma_3 of a vector are cut into one slice of rows per
  !      rank (as T3 in the iterations, xzdist.F90, here in the layout of the
  !      label shift), and every rank holds of every vector of the subspace
  !      its own rows only - rank 0 with the singles and doubles in front.
  !      A vector operation of the solver is a command of rank 0 that every
  !      rank carries out on its part: the elements evoff .. evoff + evlen - 1
  !      of a vector of the work array.
  logical :: edist = .false.                ! the sigma vectors with R3 run over the ranks
  logical :: ecmd = .false.                 ! the vector operations of the solver do (not in the singles-doubles stage)
  integer(8), allocatable :: esoff(:)       ! (0:nrank) the slices of R3 of the shift, elements
  integer(8), allocatable :: eswoff(:)      ! the same in words
  integer(8) :: esmax = 0                   ! elements of the longest slice
  integer(8) :: evoff = 0                   ! the first element of a vector that this rank holds
  integer(8) :: ekx(2) = 0                  ! the two vectors of the work array that the commands name
  ! the commands
  integer, parameter :: op_end = 0, op_shift = 1, op_endshift = 2, op_open = 3, op_close = 4, op_zero = 5, &
    op_dot = 6, op_scal = 7, op_mdot = 8, op_comb = 9, op_put = 10, op_sigma = 11, op_max = 12, op_prec = 13, &
    op_trans = 14, op_axpy = 15, op_collect = 16, op_spread = 17, op_fold = 18, op_qopen = 19, op_get = 20
  integer(8) :: evlast = 0                  ! the vector of the basis that was put last
  ! the plan of the subspace (made by the dry run of rank 0, followed by the
  ! run: the other ranks have the memory of the plan): per problem the
  ! vectors and whether they are in files; the words that the subspace takes
  ! on rank 0 and on another rank (the most of all problems)
  integer(8) :: eplan = 0                   ! the problem that runs (0: no plan to follow - one rank)
  integer(8) :: epm(maxsh) = 0
  logical :: epf(maxsh) = .false.
  integer(8) :: ep0 = 0, epw = 0

  ! ---- the results, for the table at the end and the file XZEOM
  integer(8) :: nres = 0
  integer(8) :: rshift(maxres) = 0, rroot(maxres) = 0, rnsig(maxres) = 0, rcyc(maxres) = 0
  integer(8) :: rli(maxres) = 0, rla(maxres) = 0     ! the largest element of R1: spinors i -> a (numbers of the SCF)
  real(8) :: rlv(maxres) = 0d0                       ! its absolute value
  real(8) :: rom(maxres) = 0d0, rim(maxres) = 0d0, rmx(maxres) = 0d0, rnm(maxres) = 0d0, rwt(3, maxres) = 0d0
  logical :: rconv(maxres) = .false.
  real(8) :: teomim = 0d0, teomsig = 0d0, teomsd = 0d0, teomvec = 0d0
  ! EOM-CC4 (xzcc4.inc): the vectors that the solver starts from (evfin) and
  ! returns (evfout) have all three parts; the eigenvalues of its first
  ! projected matrix; per state the root without the T4 terms, the energy
  ! after one evaluation of them, the eigenvalue at the first frequency
  logical :: evfin = .false., evfout = .false.
  ! macro / micro: the T4 terms held on e4nq orthonormal vectors (at e4kq;
  ! their sigma vectors with the T4 terms at e4kf): N(xzesigm).  More than
  ! one rank: every rank its rows of them, as of the vectors of the subspace
  ! (the kinds 3 and 4 of N(xzevput), N(xzevmdot), N(xzevcomb), N(xzevget))
  integer(8) :: e4nq = 0, e4kq = 0, e4kf = 0
  real(8) :: eritz0(64) = 0d0
  ! the files XZE4R.nnn, XZE4Q.nnn of an earlier run (N(xz4er), N(xz4eq)): the
  ! roots of the problem came from there, e4kdone of them done with the T4
  ! terms; of the last cluster in hand: its energies, its Rayleigh quotients
  ! and its energies after the first evaluation
  logical :: e4rst = .false.
  integer(8) :: e4kdone = 0
  ! the file XZEOMR of EOM-CC4, written root by root (N(xz4evec)): wanted (%xzeomvec, the disk),
  ! its unit (0: not open), the place of the next record; the place after the roots of the
  ! problem that an earlier run has done (from XZE4R.nnn)
  logical :: e4uwant = .false.
  integer :: e4ures = 0
  integer(8) :: e4upos = 0, e4rpos = 0
  real(8) :: e4eclu(64) = 0d0, e4rayc(64) = 0d0, e4shotc(64) = 0d0
  real(8) :: rpp(maxres) = 0d0, rshot(maxres) = 0d0, rfirst(maxres) = 0d0
  integer(8) :: neomsig = 0, neomsd = 0

contains

  ! the name of the file of vector j: EOMCnnnn (kind 1), EOMHnnnn (kind 2),
  ! the names of CFOUR's xx2ceomcc (libx2ccc/davhcfname)
  ! (a rank other than 0 holds its rows of the vector: EOMCnnnn.rrrr)
  character(len=13) function xzevname(kind, j)
    use xzpar, only: myrank
    integer, intent(in) :: kind
    integer(8), intent(in) :: j
    xzevname = ' '
    if (myrank == 0) then
      write(xzevname, '(a4,i4.4)') merge('EOMC', 'EOMH', kind == 1), j
    else
      write(xzevname, '(a4,i4.4,a1,i4.4)') merge('EOMC', 'EOMH', kind == 1), j, '.', myrank
    end if
  end function xzevname

  ! the first word of the vectors of a kind that are in the work array: the
  ! basis (1), its sigma vectors (2), q (3) and F (4) of EOM-CC4
  integer(8) function xzevbase(kind)
    integer, intent(in) :: kind
    select case (kind)
    case (1)
      xzevbase = evc0
    case (2)
      xzevbase = evh0
    case (3)
      xzevbase = e4kq
    case default
      xzevbase = e4kf
    end select
  end function xzevbase

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
      boff_s(0) = 0
      boff_s(1) = n3s
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

  ! A command of rank 0 to the other ranks (which wait for it in N(xzeserve)):
  ! the operation and up to five numbers.
  subroutine xzecmd(op, a1, a2, a3, a4, a5)
    use xzpar, only: xzmpi_bcast
    integer, intent(in) :: op
    real(8), intent(in), optional :: a1, a2, a3, a4, a5
    real(8) :: ctl(6)
    ctl = 0d0
    ctl(1) = dble(op)
    if (present(a1)) ctl(2) = a1
    if (present(a2)) ctl(3) = a2
    if (present(a3)) ctl(4) = a3
    if (present(a4)) ctl(5) = a4
    if (present(a5)) ctl(6) = a5
    call xzmpi_bcast(ctl, 48_8)
  end subroutine xzecmd

  ! which of the two vectors of the commands the address kx is
  real(8) function xzewhich(kx)
    integer(8), intent(in) :: kx
    xzewhich = merge(1d0, 2d0, kx == ekx(1))
  end function xzewhich

  ! The slices of a rank-3 array of the layout that xzeomlay has set: equal
  ! numbers of elements as far as whole rows allow (xzdist: xzslices, for the
  ! layout of a label shift).
  subroutine xzeslices()
    use xzdata
    use xzpar, only: nrank
    integer(8) :: r, g, na, ni, t, a
    if (allocated(esoff)) deallocate(esoff, eswoff)
    allocate(esoff(0:nrank), eswoff(0:nrank))
    esoff(0) = 0
    do r = 1, nrank - 1
      t = (r * n3s) / nrank
      esoff(r) = n3s
      do g = 0, ntg - 1
        ni = otoff(g + 1) - otoff(g)
        na = xzna(eshift, g)
        if (ni == 0 .or. na == 0) cycle
        if (t < boff_s(g) + ni * na) then
          a = max(0_8, (t - boff_s(g) + na - 1) / na)
          esoff(r) = boff_s(g) + a * na
          exit
        end if
      end do
      esoff(r) = max(esoff(r), esoff(r - 1))
    end do
    esoff(nrank) = n3s
    esmax = 0
    do r = 0, nrank - 1
      esmax = max(esmax, esoff(r + 1) - esoff(r))
    end do
    eswoff = nwe * esoff
  end subroutine xzeslices

  ! the rows a <= row < b of the block of the group g (layout of the shift)
  ! that this rank holds: all of them unless the problem runs over the ranks
  subroutine xzeslrows(g, a, b)
    use xzdata
    use xzpar, only: myrank, nrank
    integer(8), intent(in) :: g
    integer(8), intent(out) :: a, b
    integer(8) :: na, ni
    ni = otoff(g + 1) - otoff(g)
    na = xzna(eshift, g)
    a = 0
    b = ni
    if (nrank == 1 .or. .not. edist .or. na == 0) return
    a = min(ni, max(0_8, (esoff(myrank) - boff_s(g) + na - 1) / na))
    b = min(ni, max(0_8, (esoff(myrank + 1) - boff_s(g)) / na))
    if (esoff(myrank + 1) <= boff_s(g)) b = 0
    if (b < a) b = a
  end subroutine xzeslrows

  ! The place of a subspace of msub vectors of nloc elements when avail words
  ! are free for it (the rule of N(xzevopen); docs/native.md, section 16.4):
  ! all of them in the work array; else, if at least nmem fit there, that
  ! many in the work array; else files (infile).
  subroutine xzevplace(nw, nloc, nmem, avail, mayfile, msub, infile)
    use xzmem, only: xzalign
    integer(8), intent(in) :: nw, nloc, nmem, avail
    logical, intent(in) :: mayfile
    integer(8), intent(inout) :: msub
    logical, intent(out) :: infile
    integer(8) :: nfit
    infile = 2 * msub * (nw * nloc + xzalign) > avail
    if (infile .and. mayfile) then
      nfit = avail / (2 * (nw * nloc + xzalign))
      if (nfit >= nmem) then
        msub = nfit
        infile = .false.
      end if
    end if
  end subroutine xzevplace

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
