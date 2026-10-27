!
! EOM-DEA-CCSDT on the sector itself (xzdea.inc; docs/native.md, section 20;
! the algorithm of src/zccsdt/cc/dea_sector.py, docs/eom.md, section 11.9):
! what is not type-generic - the input, the parities of the spinors, the
! layout of the intermediates of the ground state and of the blocks of a
! sector vector, the results.
!
! The states have two electrons more than the closed-shell reference:
!   R = sum r2[(a<b)] a+ b+ + sum r3[i, (a<b<c)] a+ b+ c+ i + sum r4[(i<j), (a<b<c<d)] ...
! (2p + 3p1h + 4p2h) on the ground state of CCSDT.  In the blocks of the
! labels (2 m_j) a problem is the target 2 Omega - the sum of the labels of
! the two attached electrons - in classes: the shift s.
!
! Input (ZMAT):
!   CALC=CCSDT and the section
!     %xzdea
!     <target> [roots]           2 Omega of the states (labels of the run), one line per target
!     ...
!     parity g|u                 (optional) only the states of this parity: the
!                                file xzparity (one parity 0 = g, 1 = u per spinor
!                                of the SCF, the format of xzlabels without the modulus)
!     guess n                    (optional) unit vectors on the n lowest pairs (a<b)
!                                to start from; default max(4, roots + 2)
!     level 5p3h                 (optional) R with complete 5p3h (+ r5[(i<j<k), (a<b<c<d<e)];
!                                xzdea5.inc, docs/native.md, section 21); default level 4p2h
!   (an empty line or a line that starts with % ends it); the thresholds as
!   for EOM-EE: ESTATE_CONV (or %eomthrs), ESTATE_MAXCYC, %eomnroot (the roots
!   of a target without a number), %eomndav0 (vectors of the subspace).
!
! The blocks of a sector vector (shift s), one region [r2 | r3 | r4]:
!   r2   the pairs a<b of group s, in their order
!   r3   one block per group K of the virtual triples: rows the occupied i of
!        class K - s, columns the triples of group K
!   r4   one block per class M of the virtual quadruples: rows the occupied
!        pairs of group M - s (in their order), columns the quadruples a<b<c<d
!        of class M, lexicographic (csrc/t4sym.c: zc_t4s_quads)
! The same as the blocks of zccsdt.cc.dea_sector.Sector.
!
module xzdead
  use xzdata, only: maxcls, maxgrp
  implicit none
  integer, parameter :: dmaxt = 32, dmaxres = 256
  ! ---- the input
  logical :: dea_asked = .false.
  integer(8) :: dnt = 0                     ! targets
  integer(8) :: dtgt(dmaxt) = 0             ! 2 Omega (labels)
  integer(8) :: dnr(dmaxt) = 0              ! roots of each (0: %eomnroot)
  integer(8) :: dtpar = -1                  ! parity of the states: -1 all, 0 g, 1 u
  integer(8) :: dnguess = 0                 ! unit vectors to start from (0: default)
  integer(8) :: dlev = 4                    ! the rank of R: 4 (to 4p2h) or 5 (complete 5p3h, xzdea5.inc)
  integer(8), allocatable :: dpo(:), dpv(:) ! parity of the active occupied, virtual spinors (order here)

  ! ---- the layout of the intermediates of the ground state (offsets in
  !      elements; -1: no block).  Groups of pairs n2c = 2 ncls - 1, of
  !      triples n3c = 3 ncls - 2, classes of quadruples n4c = 4 ncls - 3.
  integer(8) :: n2c = 1, n3c = 1, n4c = 1
  integer(8) :: mix0(0:maxcls) = 0          ! the classes of occupied and virtual spinors mixed
  integer(8), allocatable :: t12off(:, :)   ! (k, p) -> T3 on (e | cd): [e of class p][J of group k][(cd) of group k - p]
  integer(8), allocatable :: t3poff(:, :)   ! (i, c) -> t(i mn; K): [(mn) of group c][K = class(i) + c]
  integer(8), allocatable :: r13off(:, :)   ! (P, mu) -> [t(m kl; K) for m of class mu | Xv(e; kl; K) for e of class mu]
  integer(8), allocatable :: mxoff(:, :, :) ! (kP, mu, ka) -> the products of Xv: [e, d, (bc)] (and [(bc), e, d] behind)
  integer(8), allocatable :: mxsize(:)
  integer(8), allocatable :: wboff(:, :)    ! (be, mu) -> W_pp as [(bc) of group be, e of class mu][f]
  integer(8), allocatable :: vooff(:, :, :) ! (m, mu, ka) -> <dm||ef> as [e of mu][d of ka][f]
  integer(8), allocatable :: wroff(:, :, :) ! (m, mu, ka) -> Wr(c, d, e, m) as [e of mu][d of ka][c]
  integer(8) :: dtrio(0:maxgrp) = 0         ! where the triples of a group start in their table (words)
  integer(8) :: dtrisize = 0
  integer(8), allocatable :: nqa(:)         ! quadruples of a class M (all M)
  integer(8), allocatable :: qa0(:)         ! where the quadruples of class M start in the table (quadruples)
  integer(8) :: t12size = 0, t3psize = 0, r13size = 0, wbsize = 0, vosize = 0, wrsize = 0, mxmax = 0, nqall = 0
  ! the arrays of the ground state (word addresses in the work array)
  integer(8) :: dkv = 0, dposv = 0, dposq = 0, dpos3 = 0, dtri = 0, dquads = 0
  integer(8) :: dwvat = 0, dla13 = 0, dt12 = 0, dt3p = 0, dr13 = 0, dwq4 = 0, dwb = 0, dwh = 0, dwr2 = 0
  integer(8) :: dvo = 0                     ! <dm||ef> in the blocks of vooff
  ! R13 (the rows of T3 and Xv) of the pairs dq0 .. dq1 - 1 of this rank -
  ! all pairs with one rank - in the work array (dxvfile false: at dr13) or in
  ! the file XZDXV (XZDXV.<rank>; dr13 a buffer of the largest pair); r13p(P)
  ! the first element of the pair P in the layout of r13off
  logical :: dxvfile = .false.
  logical :: dxv2 = .false.                 ! N(xzdxv) makes Xv2 of the 5p3h row (xzdea5.inc) instead
  integer(8) :: dq0 = 0, dq1 = 0
  integer(8), allocatable :: r13p(:), dqb(:)
  integer(8) :: r13loc = 0, r13pmax = 0
  integer(8) :: dwq2 = 0                    ! the 2p ladder operator of the problem in hand
  ! option deaplan n1 n2 ...: the memory per rank for these numbers of ranks, nothing computed
  logical :: dplan = .false.
  integer(8) :: dplanc = 0, dplann(16) = 0

  ! ---- the sector of the problem in hand (xzdsec)
  integer(8) :: ds = 0                      ! the shift: the class of the pairs of the two attached electrons
  integer(8) :: dn2 = 0, dsz3 = 0, dsz4 = 0 ! elements of r2, r3, r4
  integer(8), allocatable :: off3(:), off4(:)
  integer(8), allocatable :: blk3(:), pos3r(:), blk4(:), pos4(:)
  integer(8), allocatable :: off22(:, :), n22(:), off13(:, :), n13(:), off12(:, :), n12(:), x3off(:)
  integer(8), allocatable :: zroff(:, :), uoffk(:), zoffk(:)
  integer(8) :: zrsize = 0, usize = 0, zsize = 0, x3size = 0, n22max = 0, n13max = 0
  logical, allocatable :: inblk4(:)         ! M is a block of r4 (rows and quadruples)
  ! the occupied pairs of this rank (more than one rank: xzdslices; else all):
  ! its rows of r4, those of the sigma vector that it makes
  integer(8) :: dp0 = 0, dp1 = 0
  integer(8), allocatable :: dpidx(:, :)    ! the occupied pairs: dpidx(j, i) for i < j (a copy of the table jopidx)
  logical :: ddist = .false.                ! the sigma vector over the ranks
  ! the segments of an exchange of rows between the ranks (xzmpi_segs): peer,
  ! word of the work array, words, 0 send / 1 receive
  integer(8) :: dsn = 0
  integer(8), allocatable :: dsp(:), dspos(:), dslen(:), dsdir(:)

  ! ---- the results
  integer(8) :: dnres = 0
  integer(8) :: dres_t(dmaxres) = 0, dres_k(dmaxres) = 0, dres_la(dmaxres) = 0, dres_lb(dmaxres) = 0
  real(8) :: dres_om(dmaxres) = 0d0, dres_mx(dmaxres) = 0d0, dres_nm(dmaxres) = 0d0, dres_wt(3, dmaxres) = 0d0
  real(8) :: dres_lv(dmaxres) = 0d0
  logical :: dres_cv(dmaxres) = .false.
  real(8) :: tdeaim = 0d0, tdeasig = 0d0
  integer(8) :: ndeasig = 0
  real(8) :: tdpart(14) = 0d0               ! seconds of the pieces of the sigma vectors (13, 14: the
                                            ! rows exchanged by the ring, by the hh ladder; in 12 too)
  integer(8) :: dbatch = 0                  ! elements of an operand of a batch of the passes over r4
  integer(8) :: dvec5 = 0                   ! (level 5p3h) stored elements of r5 of a vector (xzdea5d.f90)

contains

  ! sizes of classes and groups (0 outside)
  integer(8) function dnO(c)
    use xzdata, only: ncls, ocoff
    integer(8), intent(in) :: c
    dnO = 0
    if (c >= 0 .and. c < ncls) dnO = ocoff(c + 1) - ocoff(c)
  end function

  integer(8) function dnV(c)
    use xzdata, only: ncls, vcoff
    integer(8), intent(in) :: c
    dnV = 0
    if (c >= 0 .and. c < ncls) dnV = vcoff(c + 1) - vcoff(c)
  end function

  integer(8) function dnQ(q)
    use xzdata, only: npg, vpoff
    integer(8), intent(in) :: q
    dnQ = 0
    if (q >= 0 .and. q < npg) dnQ = vpoff(q + 1) - vpoff(q)
  end function

  integer(8) function dnO2(q)
    use xzdata, only: npg, opoff
    integer(8), intent(in) :: q
    dnO2 = 0
    if (q >= 0 .and. q < npg) dnO2 = opoff(q + 1) - opoff(q)
  end function

  integer(8) function dnK(k)
    use xzdata, only: ntg, vtoff
    integer(8), intent(in) :: k
    dnK = 0
    if (k >= 0 .and. k < ntg) dnK = vtoff(k + 1) - vtoff(k)
  end function

  integer(8) function dnJ(k)
    use xzdata, only: ntg, otoff
    integer(8), intent(in) :: k
    dnJ = 0
    if (k >= 0 .and. k < ntg) dnJ = otoff(k + 1) - otoff(k)
  end function

  ! views of the work array: integer(8) p(0:n-1); p(0:n1-1, 0:n2-1) of
  ! integer(8), integer(1), integer(2) (the first index fastest)
  ! views p(0:n-1) of the real (xzdv_d) and complex (xzdv_z) elements at the
  ! word address k: N(xzdv) of the type-generic sources
  subroutine xzdv_d(w, k, n, p)
    use iso_c_binding
    real(8), target :: w(*)
    integer(8), intent(in) :: k, n
    real(8), pointer :: p(:)
    real(8), pointer :: q(:)
    call c_f_pointer(c_loc(w(k)), q, [max(n, 1_8)])
    p(0:max(n, 1_8) - 1) => q
  end subroutine

  subroutine xzdv_z(w, k, n, p)
    use iso_c_binding
    real(8), target :: w(*)
    integer(8), intent(in) :: k, n
    complex(8), pointer :: p(:)
    complex(8), pointer :: q(:)
    call c_f_pointer(c_loc(w(k)), q, [max(n, 1_8)])
    p(0:max(n, 1_8) - 1) => q
  end subroutine

  subroutine xzdiv(w, k, n, p)
    use iso_c_binding
    real(8), target :: w(*)
    integer(8), intent(in) :: k, n
    integer(8), pointer :: p(:)
    integer(8), pointer :: q(:)
    call c_f_pointer(c_loc(w(k)), q, [max(n, 1_8)])
    p(0:max(n, 1_8) - 1) => q
  end subroutine

  subroutine xzdiv2(w, k, n1, n2, p)
    use iso_c_binding
    real(8), target :: w(*)
    integer(8), intent(in) :: k, n1, n2
    integer(8), pointer :: p(:, :)
    integer(8), pointer :: q(:)
    call c_f_pointer(c_loc(w(k)), q, [max(n1 * n2, 1_8)])
    p(0:max(n1, 1_8) - 1, 0:max(n2, 1_8) - 1) => q
  end subroutine

  subroutine xzdi1v2(w, k, n1, n2, p)
    use iso_c_binding
    real(8), target :: w(*)
    integer(8), intent(in) :: k, n1, n2
    integer(1), pointer :: p(:, :)
    integer(1), pointer :: q(:)
    call c_f_pointer(c_loc(w(k)), q, [max(n1 * n2, 1_8)])
    p(0:max(n1, 1_8) - 1, 0:max(n2, 1_8) - 1) => q
  end subroutine

  subroutine xzdi2v2(w, k, n1, n2, p)
    use iso_c_binding
    real(8), target :: w(*)
    integer(8), intent(in) :: k, n1, n2
    integer(2), pointer :: p(:, :)
    integer(2), pointer :: q(:)
    call c_f_pointer(c_loc(w(k)), q, [max(n1 * n2, 1_8)])
    p(0:max(n1, 1_8) - 1, 0:max(n2, 1_8) - 1) => q
  end subroutine

  ! the pieces of a sigma vector that are added (a test hook, option deaonly:
  ! pp, hh, ring, x22, x13, row3, row2 as dea_sector.Engine.sigma(only=...);
  ! the intermediates are made in full)
  logical function xzdwant(piece)
    use xzopt
    character(len=*), intent(in) :: piece
    character(len=128), save :: v = ' '
    integer, save :: lv = -1
    if (lv < 0) call xzoptget('deaonly', v, lv)
    xzdwant = lv == 0
    if (lv > 0) xzdwant = index(' ' // v(1:lv) // ' ', ' ' // piece // ' ') > 0
  end function

  ! The section %xzdea of the ZMAT (the module header): found if it is there.
  subroutine xzdeainp(found)
    logical, intent(out) :: found
    character(len=256) :: line, tag
    integer :: ios, ios2, i, c
    integer(8) :: a, b
    logical :: there
    found = .false.
    dnt = 0
    dtpar = -1
    dnguess = 0
    dlev = 4
    inquire(file='ZMAT', exist=there)
    if (.not. there) return
    open(unit=32, file='ZMAT', form='formatted', status='old', action='read', iostat=ios)
    do while (ios == 0)
      read(32, '(a)', iostat=ios) line
      if (ios /= 0) exit
      tag = adjustl(line)
      do i = 1, len_trim(tag)
        c = iachar(tag(i:i))
        if (c >= iachar('A') .and. c <= iachar('Z')) tag(i:i) = achar(c + 32)
      end do
      if (tag(1:6) /= '%xzdea' .or. len_trim(tag) /= 6) cycle
      found = .true.
      do
        read(32, '(a)', iostat=ios) line
        if (ios /= 0) exit
        tag = adjustl(line)
        if (len_trim(tag) == 0 .or. tag(1:1) == '%') exit
        do i = 1, len_trim(tag)
          c = iachar(tag(i:i))
          if (c >= iachar('A') .and. c <= iachar('Z')) tag(i:i) = achar(c + 32)
        end do
        if (tag(1:6) == 'parity') then
          tag = adjustl(tag(7:))
          if (tag(1:1) == 'g') then
            dtpar = 0
          else if (tag(1:1) == 'u') then
            dtpar = 1
          else if (tag(1:1) == 'n' .or. tag(1:1) == 'a') then
            dtpar = -1
          else
            write(6, '(/,a,a,a)') ' @XZCCSDT-F, %xzdea: parity "', trim(tag), '"; g, u or none.'
            call xzerrex
          end if
        else if (tag(1:5) == 'level') then
          tag = adjustl(tag(6:))
          if (tag(1:4) == '5p3h' .or. tag(1:1) == '5') then
            dlev = 5
          else if (tag(1:4) == '4p2h' .or. tag(1:1) == '4') then
            dlev = 4
          else
            write(6, '(/,a,a,a)') ' @XZCCSDT-F, %xzdea: level "', trim(tag), '"; 4p2h or 5p3h.'
            call xzerrex
          end if
        else if (tag(1:5) == 'guess') then
          read(tag(6:), *, iostat=ios2) dnguess
          if (ios2 /= 0) then
            write(6, '(/,a,a)') ' @XZCCSDT-F, %xzdea: guess needs a number: ', trim(line)
            call xzerrex
          end if
        else
          a = 0
          b = 0
          read(tag, *, iostat=ios2) a, b
          if (ios2 /= 0) then
            b = 0
            read(tag, *, iostat=ios2) a
          end if
          if (ios2 /= 0) then
            write(6, '(/,a,a)') ' @XZCCSDT-F, %xzdea: a line that is not "target [roots]", "parity g|u", "guess n" or ' // &
              '"level 4p2h|5p3h": ', &
              trim(line)
            call xzerrex
          end if
          if (dnt < dmaxt) then
            dnt = dnt + 1
            dtgt(dnt) = a
            dnr(dnt) = b
          end if
        end if
      end do
      exit
    end do
    close(32)
    if (found .and. dnt == 0) then
      write(6, '(/,a)') ' @XZCCSDT-F, %xzdea: no target (a line "2Omega [roots]").'
      call xzerrex
    end if
  end subroutine xzdeainp

  ! The parities of the active spinors from the file xzparity (needed with
  ! "parity g|u" of %xzdea), in the order of the program.
  subroutine xzdrdpar()
    use xzdata, only: ntot, nfrozen, no, nv, operm, vperm
    integer(8), allocatable :: par(:)
    integer(8) :: n, i
    integer :: ios
    character(len=256) :: line
    logical :: there
    if (allocated(dpo)) deallocate(dpo, dpv)
    allocate(dpo(0:max(no, 1_8) - 1), dpv(0:max(nv, 1_8) - 1))
    dpo = 0
    dpv = 0
    if (dtpar < 0) return
    inquire(file='xzparity', exist=there)
    if (.not. there) then
      write(6, '(/,a)') ' @XZCCSDT-F, %xzdea asks for a parity of the states, and the run directory has no file xzparity'
      write(6, '(a)') '             (one parity per spinor of the SCF: 0 g, 1 u; first line the number of spinors).'
      call xzerrex
    end if
    open(unit=33, file='xzparity', form='formatted', status='old', action='read')
    do
      read(33, '(a)', iostat=ios) line
      if (ios /= 0) exit
      line = adjustl(line)
      if (len_trim(line) == 0 .or. line(1:1) == '#') cycle
      read(line, *, iostat=ios) n
      exit
    end do
    if (ios /= 0 .or. n /= ntot) then
      write(6, '(/,a,i8,a,i8,a)') ' @XZCCSDT-F, the file xzparity: ', n, ' spinors (the SCF has ', ntot, ').'
      call xzerrex
    end if
    allocate(par(0:ntot-1))
    read(33, *, iostat=ios) par
    close(33)
    if (ios /= 0) then
      write(6, '(/,a)') ' @XZCCSDT-F, the file xzparity: one parity per spinor expected.'
      call xzerrex
    end if
    do i = 0, no - 1
      dpo(i) = mod(abs(par(nfrozen + operm(i))), 2_8)
    end do
    do i = 0, nv - 1
      dpv(i) = mod(abs(par(nfrozen + no + vperm(i))), 2_8)
    end do
    deallocate(par)
  end subroutine xzdrdpar

  ! The layout of the intermediates of the ground state (zccsdt.cc.t4engine_sym.Layout._offsets)
  subroutine xzdglay(w)
    use xzdata
    real(8), target :: w(*)
    integer(8), pointer :: opgrp(:), opidx(:, :)
    integer(8) :: k, p, n, i, c, pq, mu, ka, be, kp, m, ph
    call xzdiv(w, jopgrp, npo, opgrp)
    call xzdiv2(w, jopidx, no, no, opidx)
    if (allocated(dpidx)) deallocate(dpidx)
    allocate(dpidx(0:no-1, 0:no-1))
    dpidx = opidx
    n2c = npg
    n3c = ntg
    n4c = 4 * ncls - 3
    if (allocated(t12off)) deallocate(t12off, t3poff, r13off, mxoff, mxsize, wboff, vooff, wroff)
    allocate(t12off(0:n3c-1, 0:ncls-1), t3poff(0:no-1, 0:n2c-1), r13off(0:max(npo, 1_8)-1, 0:ncls-1), &
             mxoff(0:n2c-1, 0:ncls-1, 0:ncls-1), mxsize(0:n2c-1), wboff(0:n2c-1, 0:ncls-1), &
             vooff(0:no-1, 0:ncls-1, 0:ncls-1), wroff(0:no-1, 0:ncls-1, 0:ncls-1))
    do c = 0, ncls - 1
      mix0(c) = ocoff(c) + vcoff(c)
    end do
    t12off = -1
    n = 0
    do k = 0, n3c - 1
      do p = 0, ncls - 1
        if (dnJ(k) > 0 .and. dnK(k) > 0 .and. dnV(p) > 0 .and. dnQ(k - p) > 0) then
          t12off(k, p) = n
          n = n + dnV(p) * dnJ(k) * dnQ(k - p)
        end if
      end do
    end do
    t12size = n
    t3poff = -1
    n = 0
    do i = 0, no - 1
      do c = 0, n2c - 1
        if (dnO2(c) > 0 .and. dnK(ocls(i) + c) > 0) then
          t3poff(i, c) = n
          n = n + dnO2(c) * dnK(ocls(i) + c)
        end if
      end do
    end do
    t3psize = n
    r13off = -1
    n = 0
    if (allocated(r13p)) deallocate(r13p)
    allocate(r13p(0:max(npo, 1_8)))
    r13pmax = 0
    do pq = 0, npo - 1
      r13p(pq) = n
      do mu = 0, ncls - 1
        k = opgrp(pq) + mu
        if (dnO(mu) + dnV(mu) > 0 .and. dnK(k) > 0) then
          r13off(pq, mu) = n
          n = n + (dnO(mu) + dnV(mu)) * dnK(k)
        end if
      end do
    end do
    r13size = n
    r13p(npo) = n
    do pq = 0, npo - 1
      r13pmax = max(r13pmax, r13p(pq + 1) - r13p(pq))
    end do
    dq0 = 0
    dq1 = npo
    r13loc = r13size
    mxoff = -1
    mxmax = 0
    do kp = 0, n2c - 1
      n = 0
      do mu = 0, ncls - 1
        do ka = 0, ncls - 1
          be = kp + mu - ka
          if (dnV(mu) > 0 .and. dnV(ka) > 0 .and. dnQ(be) > 0 .and. dnK(kp + mu) > 0 .and. dnO2(kp) > 0) then
            mxoff(kp, mu, ka) = n
            n = n + dnV(mu) * dnV(ka) * dnQ(be)
          end if
        end do
      end do
      mxsize(kp) = n
      mxmax = max(mxmax, n)
    end do
    vooff = -1
    wroff = -1
    n = 0
    k = 0
    do m = 0, no - 1
      do mu = 0, ncls - 1
        do ka = 0, ncls - 1
          if (dnV(mu) == 0 .or. dnV(ka) == 0) cycle
          ph = ka + ocls(m) - mu
          c = mu + ocls(m) - ka
          if (dnV(ph) > 0) then
            vooff(m, mu, ka) = n
            n = n + dnV(mu) * dnV(ka) * dnV(ph)
          end if
          if (dnO(c) > 0) then
            wroff(m, mu, ka) = k
            k = k + dnV(mu) * dnV(ka) * dnO(c)
          end if
        end do
      end do
    end do
    vosize = n
    wrsize = k
    wboff = -1
    n = 0
    do be = 0, n2c - 1
      do mu = 0, ncls - 1
        if (dnQ(be) > 0 .and. dnV(mu) > 0 .and. dnV(be - mu) > 0) then
          wboff(be, mu) = n
          n = n + dnQ(be) * dnV(mu) * dnV(be - mu)
        end if
      end do
    end do
    wbsize = n
  end subroutine xzdglay

  ! the quadruples of every class M: their numbers (zc_t4s_quads)
  subroutine xzdquadn()
    use xzdata
    use xzkern
    integer(4) :: kv(0:max(nv, 1_8)-1)
    integer(8) :: m
    if (allocated(nqa)) deallocate(nqa, qa0)
    allocate(nqa(0:n4c-1), qa0(0:n4c))
    do m = 0, nv - 1
      kv(m) = int(vcls(m), 4)
    end do
    qa0(0) = 0
    do m = 0, n4c - 1
      nqa(m) = 0
      if (nv >= 4) nqa(m) = zc_t4s_quads(int(nv, 4), kv, int(m, 4), 0_c_intptr_t)
      qa0(m + 1) = qa0(m) + nqa(m)
    end do
    nqall = qa0(n4c)
    ! the triples of every group on a word boundary
    dtrisize = 0
    do m = 0, ntg - 1
      dtrio(m) = dtrisize
      dtrisize = dtrisize + (3 * (vtoff(m + 1) - vtoff(m)) + 3) / 4
    end do
    dtrisize = max(dtrisize, 1_8)
  end subroutine xzdquadn

  ! The tables of the kernels of csrc/t4sym.c (32- and 16-bit integers): the
  ! class of a virtual spinor, its place in its class, the place of a pair
  ! a<b in its group (posq[a nv + b]), of a triple (by its lexicographic
  ! number) in its group, the triples of every group (their indices, in the
  ! order of the program), the quadruples of every class M (lexicographic).
  subroutine xzdtabs(w)
    use xzdata
    use xzkern
    use iso_c_binding
    real(8), target :: w(*)
    integer(8), pointer :: vpidx(:, :), vpgrp(:), vtri(:, :), vfree(:, :)
    integer(4), pointer :: kv(:), posv(:), posq(:, :), pos3(:), q4(:)
    integer(2), pointer :: tri(:, :), quads(:, :), q2(:)
    integer(8) :: a, b, c, p, t, g, m, n
    integer(8), external :: xztrank
    call xzdiv2(w, jvpidx, nv, nv, vpidx)
    call xzdiv(w, jvpgrp, nvp, vpgrp)
    call xzdiv2(w, jvtri, max(nvp, 1_8), nv, vtri)
    call xzdiv2(w, jvfree, max(nvt, 1_8), 3_8, vfree)
    call c_f_pointer(c_loc(w(dkv)), q4, [nv])
    kv(0:nv-1) => q4
    call c_f_pointer(c_loc(w(dposv)), q4, [nv])
    posv(0:nv-1) => q4
    call c_f_pointer(c_loc(w(dposq)), q4, [nv * nv])
    posq(0:nv-1, 0:nv-1) => q4
    call c_f_pointer(c_loc(w(dpos3)), q4, [max(nvt, 1_8)])
    pos3(0:max(nvt, 1_8)-1) => q4
    call xzdi2v2(w, dquads, 4_8, max(nqall, 1_8), quads)
    do a = 0, nv - 1
      kv(a) = int(vcls(a), 4)
      posv(a) = int(a - vcoff(vcls(a)), 4)
    end do
    posq = 0
    do a = 0, nv - 1
      do b = a + 1, nv - 1
        p = vpidx(b, a)
        posq(b, a) = int(p - vpoff(vpgrp(p)), 4)
        do c = b + 1, nv - 1
          t = vtri(vpidx(c, b), a)
          g = vcls(a) + vcls(b) + vcls(c)
          pos3(xztrank(a, b, c, nv)) = int(t - vtoff(g), 4)
        end do
      end do
    end do
    do g = 0, ntg - 1
      if (vtoff(g + 1) == vtoff(g)) cycle
      call xzdi2v2(w, dtri + dtrio(g), 3_8, vtoff(g + 1) - vtoff(g), tri)
      do t = vtoff(g), vtoff(g + 1) - 1
        tri(0, t - vtoff(g)) = int(vfree(t, 0), 2)
        tri(1, t - vtoff(g)) = int(vfree(t, 1), 2)
        tri(2, t - vtoff(g)) = int(vfree(t, 2), 2)
      end do
    end do
    do m = 0, n4c - 1
      if (nqa(m) == 0) cycle
      n = zc_t4s_quads(int(nv, 4), kv, int(m, 4), int(loc(quads(0, qa0(m))), c_intptr_t))
    end do
  end subroutine xzdtabs

  ! The blocks of a sector vector of the shift s (classes) and the tables of
  ! its passes (zccsdt.cc.dea_sector.Sector; Engine._u_tables)
  subroutine xzdsec(s)
    use xzdata
    integer(8), intent(in) :: s
    integer(8) :: k, m, c, p, q, n, i, pq, mu, nr
    ds = s
    if (allocated(off3)) deallocate(off3, off4, blk3, pos3r, blk4, pos4, off22, n22, off13, n13, off12, n12, x3off, &
                                    zroff, uoffk, zoffk, inblk4)
    allocate(off3(0:n3c), off4(0:n4c), blk3(0:no-1), pos3r(0:no-1), blk4(0:max(npo, 1_8)-1), &
             pos4(0:max(npo, 1_8)-1), off22(0:n4c-1, 0:n2c-1), n22(0:n4c-1), off13(0:n4c-1, 0:ncls-1), &
             n13(0:n4c-1), off12(0:n3c-1, 0:ncls-1), n12(0:n3c-1), x3off(0:no), zroff(0:no-1, 0:ncls-1), &
             uoffk(0:n3c-1), zoffk(0:n3c-1), inblk4(0:n4c-1))
    dn2 = dnQ(s)
    dp0 = 0
    dp1 = npo
    ddist = .false.
    dvec5 = 0
    ! r3: the block K has the occupied of class K - s
    off3(0) = 0
    blk3 = -1
    pos3r = -1
    do k = 0, n3c - 1
      c = k - s
      nr = 0
      if (dnK(k) > 0) nr = dnO(c)
      if (nr > 0) then
        do i = ocoff(c), ocoff(c + 1) - 1
          blk3(i) = k
          pos3r(i) = i - ocoff(c)
        end do
      end if
      off3(k + 1) = off3(k) + nr * dnK(k)
    end do
    dsz3 = off3(n3c)
    ! r4: the block M has the occupied pairs of group M - s
    off4(0) = 0
    blk4 = -1
    pos4 = -1
    do m = 0, n4c - 1
      c = m - s
      nr = dnO2(c)
      if (nqa(m) == 0) nr = 0
      inblk4(m) = nr > 0
      if (nr > 0) then
        do pq = opoff(c), opoff(c + 1) - 1
          blk4(pq) = m
          pos4(pq) = pq - opoff(c)
        end do
      end if
      off4(m + 1) = off4(m) + nr * nqa(m)
    end do
    dsz4 = off4(n4c)
    ! the blocks of one row: X[(ab), (cd)] by the group of (ab), X[a, (bcd)] by the class of a
    off22 = -1
    off13 = -1
    n22 = 0
    n13 = 0
    n22max = 0
    n13max = 0
    do m = 0, n4c - 1
      if (.not. inblk4(m)) cycle
      n = 0
      do q = 0, n2c - 1
        if (dnQ(q) > 0 .and. dnQ(m - q) > 0) then
          off22(m, q) = n
          n = n + dnQ(q) * dnQ(m - q)
        end if
      end do
      n22(m) = n
      n = 0
      do p = 0, ncls - 1
        if (dnV(p) > 0 .and. dnK(m - p) > 0) then
          off13(m, p) = n
          n = n + dnV(p) * dnK(m - p)
        end if
      end do
      n13(m) = n
      n22max = max(n22max, n22(m))
      n13max = max(n13max, n13(m))
    end do
    ! a row of r3 (and of U) on (a | bc): [a of class p][(bc) of group K - p]
    off12 = -1
    do k = 0, n3c - 1
      n = 0
      do p = 0, ncls - 1
        if (dnV(p) > 0 .and. dnQ(k - p) > 0) then
          off12(k, p) = n
          n = n + dnV(p) * dnQ(k - p)
        end if
      end do
      n12(k) = n
    end do
    x3off(0) = 0
    do i = 0, no - 1
      n = 0
      if (blk3(i) >= 0) n = n12(blk3(i))
      x3off(i + 1) = x3off(i) + n
    end do
    x3size = x3off(no)
    ! ZR(nf; ab): (n, class of f) -> [f][(ab) of group class(f) + s - class(n)]
    zroff = -1
    n = 0
    do i = 0, no - 1
      do p = 0, ncls - 1
        q = p + s - ocls(i)
        if (dnV(p) > 0 .and. dnQ(q) > 0) then
          zroff(i, p) = n
          n = n + dnV(p) * dnQ(q)
        end if
      end do
    end do
    zrsize = n
    ! U(e; d, (bc)) and Z = XvR(e; bcd) per group K of (bcd), e of class K - s
    uoffk = -1
    zoffk = -1
    usize = 0
    zsize = 0
    do k = 0, n3c - 1
      mu = k - s
      if (dnK(k) > 0 .and. dnV(mu) > 0) then
        uoffk(k) = usize
        usize = usize + dnV(mu) * n12(k)
        zoffk(k) = zsize
        zsize = zsize + dnV(mu) * dnK(k)
      end if
    end do
  end subroutine xzdsec

  ! The rows of r4 of every rank (more than one rank): one range of occupied
  ! pairs per rank - the order of the pairs is that of the blocks of r4, so
  ! that a range of pairs is a range of r4 - chosen so that the parts of a
  ! vector that the ranks hold (rank 0: r2, r3 and its rows of r4) have about
  ! equal numbers of elements.  esoff (xzeomd): the first element of r4 of
  ! every rank, relative to r4, eswoff the same in words; dp0, dp1: the pairs
  ! of this rank.  One rank: all.
  subroutine xzdslices(nw)
    use xzdata
    use xzpar, only: myrank, nrank
    use xzeomd, only: esoff, eswoff, esmax
    integer(8), intent(in) :: nw
    integer(8), allocatable :: pb(:), cum(:)
    integer(8) :: r, p, tot, nsd, t
    if (allocated(esoff)) deallocate(esoff, eswoff)
    allocate(esoff(0:nrank), eswoff(0:nrank), pb(0:nrank), cum(0:max(npo, 1_8)))
    cum(0) = 0
    do p = 0, npo - 1
      t = 0
      if (blk4(p) >= 0) t = nqa(blk4(p))
      cum(p + 1) = cum(p) + t
    end do
    nsd = dn2 + dsz3
    tot = nsd + dsz4
    ! (the pairs of the ranks are those of their parts of R13, the same for all targets: xzdranges)
    pb(0:nrank) = dqb(0:nrank)
    esmax = 0
    do r = 0, nrank
      esoff(r) = cum(pb(r))
    end do
    do r = 0, nrank - 1
      esmax = max(esmax, esoff(r + 1) - esoff(r))
    end do
    eswoff = nw * esoff
    dp0 = pb(myrank)
    dp1 = pb(myrank + 1)
    deallocate(pb, cum)
  end subroutine xzdslices

  ! The pairs of every rank, the same for all targets (more than one rank):
  ! ranges in the order of the pairs, of about equal sums of the elements of
  ! their rows of r4 over the shifts sh(1:nsh) of the targets; dq0, dq1 those
  ! of this rank, r13loc the elements of R13 that it holds.  One rank: all.
  subroutine xzdranges(w, nsh, sh)
    use xzdata
    use xzpar, only: myrank, nrank
    real(8), target :: w(*)
    integer(8), intent(in) :: nsh, sh(*)
    integer(8), pointer :: opgrp(:)
    real(8), allocatable :: cum(:)
    integer(8) :: r, p, t, m
    real(8) :: x
    call xzdiv(w, jopgrp, npo, opgrp)
    if (allocated(dqb)) deallocate(dqb)
    allocate(dqb(0:nrank), cum(0:max(npo, 1_8)))
    cum(0) = 0d0
    do p = 0, npo - 1
      x = 0d0
      do t = 1, nsh
        m = opgrp(p) + sh(t)
        if (m >= 0 .and. m < n4c) x = x + dble(nqa(m))
      end do
      cum(p + 1) = cum(p) + x
    end do
    dqb(0) = 0
    do r = 1, nrank - 1
      dqb(r) = npo
      do p = 0, npo - 1
        if (cum(p) >= dble(r) * cum(npo) / dble(nrank)) then
          dqb(r) = p
          exit
        end if
      end do
      dqb(r) = max(dqb(r), dqb(r - 1))
    end do
    dqb(nrank) = npo
    dq0 = dqb(myrank)
    dq1 = dqb(myrank + 1)
    r13loc = r13p(dq1) - r13p(dq0)
    deallocate(cum)
  end subroutine xzdranges

  ! the rows a <= r < b of the block M (of the pairs of group M - ds) that
  ! this rank makes
  subroutine xzdown(m, a, b)
    use xzdata, only: opoff
    integer(8), intent(in) :: m
    integer(8), intent(out) :: a, b
    integer(8) :: c
    c = m - ds
    a = max(dp0, opoff(c)) - opoff(c)
    b = min(dp1, opoff(c + 1)) - opoff(c)
    if (b < a) b = a
  end subroutine xzdown

  ! The blocks of the ring of the spectator j (dea_sector.Engine.ring_tables):
  ! for every d = class(m) - class(e) with triples (bcd) of the group
  ! kK = class(j) + s + d, the columns (m, e) of the operand - m not j, e of
  ! class class(m) - d; mbase(m, b): the first of m (-1: none) - and the rows
  ! of the stacked operator: (i, a) as the columns (ibase), the row of f_ov
  ! (d = 0: bfov), the rows k of - 1/2 <mj||ek> for the occupied k of class
  ! class(j) + d (bro) and the rows e' of 1/2 <mj||e'f> for its virtual ones
  ! (brv); -1: none.
  subroutine xzdringblk(j, nbk, bd, bkk, bnc, bnr, bfov, bro, brv, mbase, ibase)
    use xzdata
    integer(8), intent(in) :: j
    integer(8), intent(out) :: nbk
    integer(8), intent(out) :: bd(*), bkk(*), bnc(*), bnr(*), bfov(*), bro(*), brv(*)
    integer(8), intent(out) :: mbase(0:no-1, *), ibase(0:no-1, *)
    integer(8) :: d, kk, m, c, nc, nr, ci
    nbk = 0
    do d = 1 - ncls, ncls - 1
      kk = ocls(j) + ds + d
      if (dnK(kk) == 0) cycle
      nc = 0
      do m = 0, no - 1
        mbase(m, nbk + 1) = -1
        c = ocls(m) - d
        if (m == j .or. dnV(c) == 0) cycle
        mbase(m, nbk + 1) = nc
        nc = nc + dnV(c)
      end do
      if (nc == 0) cycle
      nbk = nbk + 1
      nr = 0
      do m = 0, no - 1
        ibase(m, nbk) = -1
        c = ocls(m) - d
        if (m == j .or. dnV(c) == 0) cycle
        ibase(m, nbk) = nr
        nr = nr + dnV(c)
      end do
      bfov(nbk) = -1
      ci = ocls(j) + d
      bro(nbk) = -1
      brv(nbk) = -1
      if (d == 0) then
        bfov(nbk) = nr
        nr = nr + 1
      end if
      if (dnO(ci) > 0) then
        bro(nbk) = nr
        nr = nr + dnO(ci)
      end if
      if (dnV(ci) > 0) then
        brv(nbk) = nr
        nr = nr + dnV(ci)
      end if
      bd(nbk) = d
      bkk(nbk) = kk
      bnc(nbk) = nc
      bnr(nbk) = nr
    end do
  end subroutine xzdringblk

  ! the number of the occupied pair i < j (the tables of the program)
  integer(8) function xzdpidx(i, j)
    use xzdata, only: no
    integer(8), intent(in) :: i, j
    xzdpidx = dpidx(j, i)
  end function

  ! the rank that holds the occupied pair p (its rows of r4; dqb)
  integer(8) function xzdowner(p)
    use xzpar, only: nrank
    integer(8), intent(in) :: p
    integer(8) :: r
    xzdowner = nrank - 1
    do r = 0, nrank - 1
      if (p < dqb(r + 1)) then
        xzdowner = r
        return
      end if
    end do
  end function

  ! the elements of a vector that this rank holds: r2, r3 (rank 0's; a copy
  ! on the other ranks, for the sigma vector) and its rows of r4 (all with
  ! one rank; level 5p3h: and its stored rows of r5, esoff the rank-major
  ! parts of xzd5slices)
  integer(8) function xzdpart()
    use xzpar, only: myrank
    use xzeomd, only: esoff
    xzdpart = dn2 + dsz3 + dsz4 + dvec5
    if (ddist) xzdpart = dn2 + dsz3 + esoff(myrank + 1) - esoff(myrank)
  end function

  ! The rows (m, j) of r4 of the spectator j, m /= j, as a rank takes them
  ! for the ring over the ranks (20.6): block after block of r4, within a
  ! block in the order of m; lrow(m) the row of (m, j) in its block there
  ! (-1: none), jb0(M) the first element of the block M, tot all elements
  subroutine xzdringlay(j, lrow, jb0, tot)
    use xzdata, only: no
    integer(8), intent(in) :: j
    integer(8), intent(out) :: lrow(0:no-1), jb0(0:n4c-1), tot
    integer(8) :: mx, m, px, cnt
    tot = 0
    lrow = -1
    do mx = 0, n4c - 1
      jb0(mx) = tot
      cnt = 0
      do m = 0, no - 1
        if (m == j) cycle
        px = dpidx(max(m, j), min(m, j))
        if (blk4(px) /= mx) cycle
        lrow(m) = cnt
        cnt = cnt + 1
      end do
      tot = tot + cnt * nqa(mx)
    end do
  end subroutine xzdringlay

  ! a segment of the exchange in hand (dsn = 0 starts one)
  subroutine xzdseg(peer, pos, len, dir)
    integer(8), intent(in) :: peer, pos, len, dir
    integer(8), allocatable :: t(:)
    integer(8) :: n
    if (.not. allocated(dsp)) allocate(dsp(1024), dspos(1024), dslen(1024), dsdir(1024))
    if (dsn >= size(dsp, kind=8)) then
      n = 2 * size(dsp, kind=8)
      allocate(t(n))
      t(1:dsn) = dsp(1:dsn)
      call move_alloc(t, dsp)
      allocate(t(n))
      t(1:dsn) = dspos(1:dsn)
      call move_alloc(t, dspos)
      allocate(t(n))
      t(1:dsn) = dslen(1:dsn)
      call move_alloc(t, dslen)
      allocate(t(n))
      t(1:dsn) = dsdir(1:dsn)
      call move_alloc(t, dsdir)
    end if
    dsn = dsn + 1
    dsp(dsn) = peer
    dspos(dsn) = pos
    dslen(dsn) = len
    dsdir(dsn) = dir
  end subroutine xzdseg

  ! The batches of the blocks of the ring of j (Engine._ring_pass): a block
  ! after the other while the operand and the product of the batch hold at
  ! most dbatch elements (at least one block); bfirst(t) the first block of
  ! batch t; and the most that the operand, the product and a gathered
  ! operator of a batch take (mbmax, xbmax, wgmax: maxima so far, updated).
  subroutine xzdringbat(nbk, bkk, bnc, bnr, nbat, bfirst, mbmax, xbmax, wgmax)
    integer(8), intent(in) :: nbk, bkk(*), bnc(*), bnr(*)
    integer(8), intent(out) :: nbat, bfirst(*)
    integer(8), intent(inout) :: mbmax, xbmax, wgmax
    integer(8) :: b, tot, sz, mt, xt
    nbat = 0
    tot = 0
    mt = 0
    xt = 0
    do b = 1, nbk
      sz = (bnc(b) + bnr(b)) * dnK(bkk(b))
      if (nbat > 0 .and. b > bfirst(nbat) .and. tot + sz > dbatch) then
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
      mt = mt + bnc(b) * dnK(bkk(b))
      xt = xt + bnr(b) * dnK(bkk(b))
      wgmax = max(wgmax, bnc(b) * bnr(b))
    end do
    mbmax = max(mbmax, mt)
    xbmax = max(xbmax, xt)
    bfirst(nbat + 1) = nbk + 1
  end subroutine xzdringbat

  ! The states at the end: the table, the file XZDEA, the lowest state to
  ! JOBARC (TOTENER2, as for the excited states)
  subroutine xzdeaout(escf, ecorr)
    real(8), intent(in) :: escf, ecorr
    integer(8) :: iintln, ifltln, iintfp, ialone, ibitwd
    common /machsp/ iintln, ifltln, iintfp, ialone, ibitwd
    integer(8) :: k
    real(8) :: elow, e0
    real(8), parameter :: ev = 27.211386245988d0, cm = 219474.6313632d0
    write(6, '(/,a)') '   EOM-DEA-CCSDT: the states (two electrons attached to the reference)'
    write(6, '(a)') '  ' // repeat('-', 110)
    write(6, '(a)') '   2Omega  root        omega / Eh          total energy / Eh     above the lowest: Eh          cm-1' // &
                    '      max |r|'
    elow = huge(1d0)
    do k = 1, dnres
      if (dres_cv(k)) elow = min(elow, dres_om(k))
    end do
    e0 = elow
    if (e0 == huge(1d0) .and. dnres > 0) e0 = minval(dres_om(1:dnres))
    open(unit=79, file='XZDEA', form='formatted', status='replace')
    write(79, '(a)') '# EOM-DEA-CCSDT of xzccsdt: 2 Omega of the target, root, omega / Eh (E(N+2) - E_CC(N)), ' // &
                     'total energy / Eh, largest element and norm of the residual, weights of 2p, 3p1h, 4p2h, converged, ' // &
                     'the largest element of r2 as the spinors a b (numbers of the SCF) and its absolute value'
    write(79, '(a,f25.12,a,f25.12)') '# E(SCF) ', escf, '   E(CCSDT) ', escf + ecorr
    do k = 1, dnres
      write(6, '(i8,i6,f20.12,f27.12,f20.12,f14.4,es13.2,a)') dres_t(k), dres_k(k), dres_om(k), escf + ecorr + dres_om(k), &
        dres_om(k) - e0, (dres_om(k) - e0) * cm, dres_mx(k), merge('                ', '  NOT converged ', dres_cv(k))
      write(79, '(2i6,2f25.14,2es12.4,3f10.6,l3,2i7,f10.6)') dres_t(k), dres_k(k), dres_om(k), escf + ecorr + dres_om(k), &
        dres_mx(k), dres_nm(k), dres_wt(:, k), dres_cv(k), dres_la(k), dres_lb(k), dres_lv(k)
    end do
    close(79)
    write(6, '(a)') '  ' // repeat('-', 110)
    write(6, '(a)') '   (the file XZDEA has this table)'
    if (elow < huge(1d0)) then
      call putrec(20, 'JOBARC', 'TOTENER2', iintfp, escf + ecorr + elow)
      write(6, '(a,f25.12)') '   TOTENER2 written to JOBARC (the lowest state):   ', escf + ecorr + elow
    end if
  end subroutine xzdeaout
end module xzdead
