!
! What the routines of xzccsdt share: the sizes, the job, and where the
! arrays that live for the whole run are in the work array (the j...: index
! of the first word in w).
!
! Layouts are those of the C kernels: row-major, indices from 0.  An array
! A[i][j][k] of the comments is declared A(0:nk-1, 0:nj-1, 0:ni-1) in the
! Fortran routines and addressed A(k, j, i).
!
! Pairs p<q in the order (0,1), (0,2), .., (1,2), ..: the first index
! slowest; triples p<q<r likewise (lexicographic).
!
! With symmetry labels (ncls > 1 classes) the active spinors are in the
! order of their classes (occupied and virtual each; by CFOUR's order inside
! a class), and pairs and triples are numbered by groups: the sum of the
! classes of their indices first, lexicographic inside a group.  A block of
! an amplitude or of an integral on pairs or triples is then a range.
! Without labels there is one class and one group: the dense case.
!
module xzdata
  use xzopt
  implicit none

  ! ---- sizes
  integer(8) :: no = 0, nv = 0      ! active occupied and virtual spinors
  integer(8) :: ny = 0              ! no + nv
  integer(8) :: npo = 0, nvp = 0    ! pairs i<j, a<b
  integer(8) :: nto = 0, nvt = 0    ! triples i<j<k, a<b<c
  integer(8) :: nfrozen = 0, ndropped = 0, ntot = 0
  integer(8) :: nwe = 2             ! words of an element: 2 complex, 1 real
  integer(8) :: n3 = 0              ! elements of T3 as it is stored (the allowed blocks)

  ! ---- symmetry labels (the file xzlabels, else dgcosym of CFOUR's SCF) and
  !      their classes.  A label is an integer that adds, modulo modulus if
  !      that is not 0; the block code below is for modulus 0 only (the
  !      classes are (label - smallest) / step, a group is a sum of classes)
  integer, parameter :: maxcls = 64, maxgrp = 3 * maxcls
  logical :: have_labels = .false.  ! labels were found and can be used
  character(len=8) :: labfile = ' ' ! the file they came from: xzlabels, dgcosym
  integer(8) :: modulus = 0
  integer(8) :: ncls = 1            ! classes of the labels: (label - smallest) / step
  integer(8) :: npg = 1, ntg = 1    ! groups of pairs (2 ncls - 1) and of triples (3 ncls - 2)
  integer(8) :: ocoff(0:maxcls) = 0, vcoff(0:maxcls) = 0    ! first spinor of a class
  integer(8) :: opoff(0:maxgrp) = 0, vpoff(0:maxgrp) = 0    ! first pair of a group
  integer(8) :: otoff(0:maxgrp) = 0, vtoff(0:maxgrp) = 0    ! first triple of a group
  integer(8) :: boff(0:maxgrp) = 0  ! first element of the block of T3 of a group of triples
  integer(8) :: vvoff(0:maxgrp) = 0 ! first element of the block of <vv||vv> of a group of pairs
  integer(8) :: nvv = 0             ! elements of <vv||vv> on its diagonal pair blocks
  integer(8) :: nqmax = 0           ! pairs a<b of the largest group
  ! the pairs (a, i) with s(a) - s(i) = delta (the ring term of pass 2 couples
  ! them) and the right-hand sides of the drive terms by (group of Q, class of i)
  integer(8) :: ndel(-maxcls:maxcls) = 0    ! pairs (a, i) of a delta
  integer(8) :: wdoff(-maxcls:maxcls) = 0   ! first element of the ring operator of a delta
  integer(8) :: nwd = 0                     ! elements of the ring operators of all deltas
  integer(8) :: bdoff(0:maxgrp, 0:maxcls) = -1      ! first element of a right-hand side (-1: none)
  integer(8) :: nbd = 0                     ! elements of all of them
  ! (small integer arrays, not in the work array: the plan needs them before
  ! anything is placed)
  integer(8), allocatable :: lab(:)         ! labels of all spinors, CFOUR's order (0:ntot-1)
  integer(8), allocatable :: operm(:), vperm(:)     ! CFOUR's number (from 0, among the active
                                            ! occupied or virtual ones) of the spinors in the order here
  integer(8), allocatable :: ocls(:), vcls(:)       ! their classes
  integer(8) :: jopgrp = 0, jvpgrp = 0      ! group of a pair
  integer(8) :: jobase = 0, jvcol = 0       ! T3[I][A] = t3(obase[I] + vcol[A]) in its block
  integer(8) :: josig = 0, jvsig = 0        ! (signs of the layout: all one)
  integer(8) :: jvci = 0                    ! vcol[vtri[d][Q]] (int32)
  integer(8) :: jrowpos = 0                 ! place of (a, i) among the pairs of s(a) - s(i)

  ! ---- label shifts of the operands of the passes in blocks (xzblk.inc), in
  !      classes.  Every array has a shift: (the classes of its creators) -
  !      (those of its annihilators); the amplitudes and the operators of the
  !      ground state have shift 0, which is what these are unless the EOM
  !      code (xzeom.inc) sets them with xzshctx:
  !        sh_t   of the rank-3 operand of a pass (its layout: kob_in, bof_in)
  !        sh_w   of the operators of the residual (ring, pp, hh) / of the
  !               integrals of pass 1
  !        sh_a   of the amplitudes of the drive terms
  !        sh_o   of the result, sh_t + sh_w (its layout: kob_out, bof_out)
  !        sh_b   of the operators of the drive terms, sh_o - sh_a
  !      The layout of a rank-3 array of shift s: the block of the group g of
  !      occupied triples has the virtual triples of group g + s (xzna of them).
  integer(8) :: sh_t = 0, sh_w = 0, sh_a = 0, sh_o = 0, sh_b = 0
  logical :: p1_noe = .false.               ! pass 1: the rows e of its first operand are zero and left out
  integer(8) :: kob_in = 0, kob_out = 0     ! obase of the two layouts (index in w)
  integer(8) :: bof_in(0:maxgrp) = 0, bof_out(0:maxgrp) = 0
  integer(8) :: eshift = 0                  ! the shift of the EOM vectors that are worked on
  integer(8) :: jobase_s = 0                ! obase of their layout
  integer(8) :: boff_s(0:maxgrp) = 0        ! its blocks
  integer(8) :: n3s = 0                     ! elements of a rank-3 array of that shift

  ! ---- the job
  character(len=16) :: method = ' '
  logical :: with_q = .false.
  logical :: with_cc = .false.      ! the CCSDT iterations are part of the job
  logical :: with_lam = .false.     ! the Lambda equations are
  logical :: q_left = .false.       ! (Q) with lambda on the left: (Q)_Lambda
  logical :: with_den = .false.     ! the density and the orbital gradient of CCSDT are (xzden.inc)
  logical :: with_eom = .false.     ! excited states are: EOM-EE-CCSDT (xzeom.inc)
  ! CC3 (xzcc3.inc; docs/native.md, section 17): the model of the iterations and
  ! of the EOM sigma vector.  0 CCSDT; 1 CC3 with T3 stored (the CCSDT step with
  ! the T3 couplings of the triples equation off: no ring, no pp or hh ladder,
  ! the drive operators without their T2 and T3 parts, the diagonal Fock
  ! operator); 2 CC3 without stored T3 (the triples made per batch of occupied
  ! triples and contracted at once: n3 is 0 then, and the EOM vectors are
  ! singles and doubles with the triples folded in at a frequency)
  integer :: cc3mode = 0
  character(len=5) :: mname = 'CCSDT'   ! the model of the iterations, for the output
  logical :: noring = .false.       ! pass 2 in blocks without its ring term (T3 is not read)
  integer(8) :: nc3w = 0            ! elements of the work of one CC3 batch (the plan)
  ! the folded EOM-CC3 sigma vector: on, at the frequency eomega.  nfollow > 0:
  ! the solver follows the nfollow roots with the largest weight on its first
  ! nfollow vectors (those it started from), in the order of their energies
  logical :: efold = .false.
  integer(8) :: nfollow = 0
  real(8) :: eomega = 0d0
  real(8) :: tfly(4) = 0d0          ! seconds of the CC3 kernel: products, amplitudes, gather, contractions
  ! (Q) reads T3 (and Lambda3) where the files XZAMPS (XZLAMB) are mapped,
  ! read-only: one copy in the page cache of a node for all its ranks, not
  ! in the work array (option qmap 0: a private copy in the work array)
  logical :: t3shared = .false.
  logical :: sizes_set = .false.    ! xzsizes was called (a rank other than 0, before its work array)
  logical :: den_active = .false.   ! option den_active: the files of the active spinors alone (a test hook)
  real(8) :: conv_e = 1d-9, conv_t = 1d-7
  integer(8) :: maxcyc = 50
  integer(8) :: ndiis = 8           ! DIIS vectors asked for
  real(8) :: escf = 0d0

  ! ---- tables of the occupied and of the virtual indices
  !   tri[x][P], sgn[x][P]   the triple that the index x and the pair P make,
  !                          and the sign of t[x, P] = sgn * t[triple] (0: x in P)
  !   free[r][I], pair[r][I] the r-th way to write the triple I as (index, pair);
  !                          signs +, -, +
  !   pa[P], pb[P]           the pair P;  pidx[p][q]: its number (-1 unless p<q)
  integer(8) :: jotri, josgn, jofree, jopair, jopa, jopb, jopidx
  integer(8) :: jvtri, jvsgn, jvfree, jvpair, jvpa, jvpb, jvpidx
  integer(8) :: jrows               ! 0, 1, 2, ..: the rows of a batch of triples
  integer(8) :: jeps                ! spinor energies, occupied first (real)
  integer(8) :: jeo3, jev3          ! their sums over the triples (real)

  ! ---- integrals <pq||rs> (bare)
  integer(8) :: joooo               ! [m][n][i][j]
  integer(8) :: jooov               ! [m][n][i][e]
  integer(8) :: joovv               ! [m][n][e][f]
  integer(8) :: jovvo               ! [m][b][e][j]
  integer(8) :: jovvq               ! [m][y][Q]      <my||de>, Q = (d<e)
  integer(8) :: jvvvv               ! [Q][Q']        <ab||cd>, a<b, c<d
  integer(8) :: joovvp              ! [m][n][Q]      <mn||de>
  integer(8) :: jgpp                ! [P][Q]         <mn||de>, m<n

  ! ---- amplitudes
  integer(8) :: jt1                 ! [i][a]
  integer(8) :: jt2                 ! [i][j][a][b]
  integer(8) :: jt3                 ! [I][A] on i<j<k, a<b<c

  ! ---- lambda, in the layouts of the amplitudes
  integer(8) :: jl1 = 0, jl2 = 0, jl3 = 0

  ! ---- what a Lambda iteration needs of the integrals and of t (xzlam.inc;
  !      made once, kept for the iterations)
  integer(8) :: lfov = 0            ! f[m][e], T1-dressed
  integer(8) :: lfvv = 0, lfoo = 0  ! F_vv[a][d], F_oo[m][i] of the T3 equation
  integer(8) :: lwov = 0            ! W_ovvo[k][e][c][m]
  integer(8) :: lwl = 0             ! [(c, k)][(e, m)]: the ring operator of pass 2
  integer(8) :: lwkl = 0            ! W_oooo[K][P] on pairs
  integer(8) :: lo6t = 0            ! the transposed operator on occupied triples
  integer(8) :: lgmq = 0            ! [m][Q][y]: t2[m][n][Q], W_vvvo[Q][b][m]
  integer(8) :: lgpd = 0            ! [x][P][e]: t2[P][e][f], W_vooo[j][P][e]
  integer(8) :: lbdrv = 0           ! [k][i][Q], k < 2 (nv + no): right-hand sides of the drive terms
  integer(8) :: ladrv = 0           ! [P][c][k]: their left-hand sides
  integer(8) :: lvx = 0             ! [g][e][j][b] = ~<ej||bg>
  integer(8) :: lxq = 0             ! [P][I][e] = sum_{f<g} <P||fg> t3[I][e, fg]
  integer(8) :: lx2 = 0             ! [Q][I][i] = A(ef) sum_{g<h} ~<ei||gh> t3[I][f, gh]
  integer(8) :: looovd = 0          ! dressed <mn||ie> [m][n][i][e]
  integer(8) :: lt2p = 0            ! t2[i][j][Q]

  ! ---- what a sigma vector of EOM-CCSDT needs of the ground state
  !      (xzeom.inc: made once by xzeomim, kept for all sigma vectors; the pp
  !      operator of the T3 equation is in the place of <ab||cd>, jvvvv)
  integer(8) :: efoo = 0, efov = 0, efvv = 0        ! the T1-dressed Fock matrix [m][i], [m][e], [a][e]
  integer(8) :: eft = 0                     ! f[e][m]
  integer(8) :: efevv = 0, efeoo = 0        ! F_vv[a][d], F_oo[m][i] of the T3 equation
  integer(8) :: eooovd = 0, eooood = 0      ! dressed <mn||ie>, <mn||ij>
  integer(8) :: eovvod = 0, eovood = 0      ! dressed <mb||ej>, <mb||ij>
  integer(8) :: eweff = 0                   ! W_ovvo[m][a][d][i]
  integer(8) :: ed = 0                      ! dressed <mb||de> [m][b][Q]
  integer(8) :: ebv = 0                     ! bra-dressed <ab||ei> [Q][e][i]
  integer(8) :: ewv = 0, ewo = 0            ! W_vvvo[Q][e][i], W_vooo[m][a][P]
  integer(8) :: et2p = 0, et2pp = 0         ! t2[i][j][Q], t2[P][Q]
  integer(8) :: egpd = 0, egmq = 0          ! the operands of pass 1
  integer(8) :: ewkl = 0                    ! W_oooo[K][P]
  integer(8) :: ekk = 0                     ! [m][n][P] = sum_Q <mn||Q> t2[P][Q]
  integer(8) :: eo6 = 0                     ! the operator on occupied triples [I][J]
  ! (the folded EOM-CC3 sigma vector: the operands of the CC3 batches of the ground state)
  integer(8) :: e3ao = 0                    ! Ao[hi][lo][c][m] = <mc||lo hi>~
  integer(8) :: e3bv = 0                    ! Bv[e][i][Q] = W_vvvo[Q][e][i] (the dressed <ab||ei> of CC3)
  integer(8) :: e3gv = 0                    ! Gv[m][Q][y] = ~<ym||Q>

  ! ---- batch sizes: the memory plan (xzplan.F90)
  integer(8) :: nbp = 1             ! occupied pairs of a batch of pass 1 and pass 2
  integer(8) :: nqs = 1             ! rows of a slab of the dressed <ab||cd>
  integer(8) :: nbt = 1             ! occupied triples of a batch of the pp ladder
  integer(8) :: nba = 1             ! first virtual indices of a batch of the W_vvvo terms
  integer(8) :: ncb = 0             ! columns of a product of the occupied operator
  integer :: hhset = 0              ! the hh ladder of dense T3: 1 operator on triples, 2 through pairs

  ! ---- batch sizes of (Q)
  integer(8) :: qc22 = 1            ! columns (cd) of a product [(ab)][(cd)]
  integer(8) :: qc13 = 1            ! columns (bcd) of a product [a][(bcd)]
  integer(8) :: qxv = 3             ! occupied x whose Xv(x, l) are held at a time
  integer(8) :: qrows = 1           ! rows (bc) of <bc||ef> of a step of the formation of Xv

  ! ---- (Q) in the sectors of the labels: the largest sets of blocks
  integer(8) :: qn22max = 0         ! products [(ab)][(cd)] of one occupied quadruple
  integer(8) :: qn13max = 0         ! products [a][(bcd)]
  integer(8) :: qxvmax = 0          ! Xv of one pair of occupied indices
  integer(8) :: qm3max = 0          ! the products that one block of Xv is made from
  integer(8) :: qwbmax = 0          ! a block of <bc||ef> as [(bc)][e][f]
  integer(8) :: nvcmax = 0, nocmax = 0      ! spinors of the largest class
  integer(8) :: natmax = 0          ! virtual triples of the largest group
  integer(8) :: jqkv = 0, jqlid = 0, jqlst = 0, jqlmem = 0, jqposv = 0, jqposq = 0, jqpos3 = 0, jqtri = 0

  logical :: qdebug = .false.       ! print the parts of (Q) (option qdebug)
  real(8) :: qparts(4) = 0d0
  logical :: qsampled = .false.     ! (Q) ran in sample mode: its energies are not (Q)
  logical :: timing_run = .false.   ! option ncyc: a fixed number of cycles, nothing is converged

  ! ---- wall and CPU times of the stages, by name, in the order of their
  !      first use.  Sets: 1 the CCSDT iterations, 2 the iterations of the
  !      Lambda equations, 3 the intermediates of the Lambda equations (once),
  !      4 the density and the orbital gradient, 5 EOM (the intermediates
  !      once, the sigma vectors)
  integer, parameter :: ntset = 5, ntmax = 64
  character(len=28) :: tnam(ntmax, ntset) = ' '
  real(8) :: twal(ntmax, ntset) = 0d0, tcpu(ntmax, ntset) = 0d0
  logical :: tpart(ntmax, ntset) = .false.      ! a part of another entry: wall only, not summed
  integer :: ntim(ntset) = 0
  real(8) :: tic_w = 0d0, tic_c = 0d0
  integer :: tcur = 1                           ! the set of the step that runs
  real(8) :: tlamim = 0d0                       ! the intermediates of the Lambda equations

contains

  ! virtual triples of the block of the group g of occupied triples in the
  ! layout of a rank-3 array of shift s (0: no such block)
  integer(8) function xzna(s, g)
    integer(8), intent(in) :: s, g
    integer(8) :: h
    h = g + s
    xzna = 0
    if (g < 0 .or. g >= ntg .or. h < 0 .or. h >= ntg) return
    xzna = vtoff(h + 1) - vtoff(h)
  end function xzna

  ! spinors of a class (0 outside the classes): virtual, occupied
  integer(8) function xznvc(s)
    integer(8), intent(in) :: s
    xznvc = 0
    if (s >= 0 .and. s < ncls) xznvc = vcoff(s + 1) - vcoff(s)
  end function xznvc

  integer(8) function xznoc(s)
    integer(8), intent(in) :: s
    xznoc = 0
    if (s >= 0 .and. s < ncls) xznoc = ocoff(s + 1) - ocoff(s)
  end function xznoc

  ! pairs (a, i) with s(a) - s(i) = d (0 outside the range of d)
  integer(8) function xzndel(d)
    integer(8), intent(in) :: d
    xzndel = 0
    if (abs(d) < ncls) xzndel = ndel(d)
  end function xzndel

  ! The shifts of a pass in blocks (see sh_t ..): the layouts of operand and
  ! result, the ring operators - for the pairs (d, m) of delta the block
  ! [(a, i) of delta + sW][(d, m) of delta] - and the right-hand sides of the
  ! drive terms by (group gQ of Q, class si of i): W_vvvo[Q][e][i] with e of
  ! class gQ - si - sb, t2[m][i][Q] with m of class gQ - si - sa.  (0, 0, 0)
  ! gives what xzsetmode made.
  subroutine xzshctx(st, sw, sa)
    integer(8), intent(in) :: st, sw, sa
    integer(8) :: d, g, si, n
    sh_t = st
    sh_w = sw
    sh_a = sa
    sh_o = st + sw
    sh_b = sh_o - sa
    if (ncls == 1) return
    if (st == 0) then
      kob_in = jobase
      bof_in = boff
    else
      kob_in = jobase_s
      bof_in = boff_s
    end if
    if (sh_o == 0) then
      kob_out = jobase
      bof_out = boff
    else
      kob_out = jobase_s
      bof_out = boff_s
    end if
    wdoff = 0
    nwd = 0
    do d = 1 - ncls, ncls - 1
      wdoff(d) = nwd
      nwd = nwd + xzndel(d + sw) * ndel(d)
    end do
    bdoff = -1
    nbd = 0
    do g = 0, npg - 1
      do si = 0, ncls - 1
        n = (xznvc(g - si - sh_b) + xznoc(g - si - sa)) * (ocoff(si + 1) - ocoff(si)) * (vpoff(g + 1) - vpoff(g))
        if (n == 0) cycle
        bdoff(g, si) = nbd
        nbd = nbd + n
      end do
    end do
  end subroutine xzshctx

  ! The hh ladder and F_oo of dense T3 with the operator on occupied triples
  ! (xzo6, one GEMM: the default) or through occupied pairs (xzwpair,
  ! xzhhapply: option hh pair).  The pair form has (o-2)**2 / (9 o) times fewer
  ! operations, but its gather and reduction are bound by the memory: measured
  ! for 26 occupied spinors it is faster on 8 threads and slower on 16 and
  ! more (docs/native.md), so it is an option.  In blocks of the labels the
  ! operator on triples is used in any case.
  logical function xzhhpair()
    character(len=8) :: arg
    integer :: larg
    if (hhset == 0) then
      hhset = 1
      call xzoptget('hh', arg, larg)
      if (larg > 0) then
        if (arg(1:1) == 'p') hhset = 2
        if (arg(1:1) == 'o') hhset = 1
      end if
      if (no < 3) hhset = 1
    end if
    xzhhpair = hhset == 2 .and. ncls == 1
  end function xzhhpair

  ! columns of T3 per product of the pair form
  integer(8) function xzhhcols()
    xzhhcols = max(1_8, min(nvt, 512_8))
  end function xzhhcols

  ! The tasks of a parallel loop that are too large for it: with nth threads
  ! a task of more than three fair shares of the work (cost: operation
  ! counts) would run on one core while the others wait.  The loops over
  ! tasks run these first, one after the other with threaded kernels and a
  ! threaded BLAS, and the rest in parallel with the BLAS sequential inside.
  subroutine xzbig(n, cost, big)
    use omp_lib
    integer(8), intent(in) :: n
    real(8), intent(in) :: cost(n)
    logical, intent(out) :: big(n)
    real(8) :: share
    share = sum(cost(1:n)) / omp_get_max_threads()
    big(1:n) = cost(1:n) > 3d0 * share
  end subroutine xzbig

  ! the clocks start
  subroutine xztic()
    use xzkern
    call xz_clocks(tic_w, tic_c)
  end subroutine xztic

  integer function xztslot(set, name)
    integer, intent(in) :: set
    character(len=*), intent(in) :: name
    integer :: k
    do k = 1, ntim(set)
      if (tnam(k, set) == name) then
        xztslot = k
        return
      end if
    end do
    ntim(set) = min(ntim(set) + 1, ntmax)
    xztslot = ntim(set)
    tnam(xztslot, set) = name
  end function xztslot

  ! the wall and CPU time (all threads of the process) since the last xztic
  ! or xztoc go to the stage `name'; the clocks start again
  subroutine xztoc(set, name)
    use xzkern
    integer, intent(in) :: set
    character(len=*), intent(in) :: name
    integer :: k
    real(8) :: w1, c1
    call xz_clocks(w1, c1)
    k = xztslot(set, name)
    twal(k, set) = twal(k, set) + w1 - tic_w
    tcpu(k, set) = tcpu(k, set) + c1 - tic_c
    tic_w = w1
    tic_c = c1
  end subroutine xztoc

  ! a wall time measured inside a kernel: a part of the stage that the call is in
  subroutine xztpart(set, name, wall)
    integer, intent(in) :: set
    character(len=*), intent(in) :: name
    real(8), intent(in) :: wall
    integer :: k
    k = xztslot(set, name)
    twal(k, set) = twal(k, set) + wall
    tpart(k, set) = .true.
  end subroutine xztpart
end module xzdata
