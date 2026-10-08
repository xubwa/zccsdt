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
  implicit none

  ! ---- sizes
  integer(8) :: no = 0, nv = 0      ! active occupied and virtual spinors
  integer(8) :: ny = 0              ! no + nv
  integer(8) :: npo = 0, nvp = 0    ! pairs i<j, a<b
  integer(8) :: nto = 0, nvt = 0    ! triples i<j<k, a<b<c
  integer(8) :: nfrozen = 0, ndropped = 0, ntot = 0
  integer(8) :: nwe = 2             ! words of an element: 2 complex, 1 real
  integer(8) :: n3 = 0              ! elements of T3 as it is stored (the allowed blocks)

  ! ---- symmetry labels (the file xzlabels) and their classes
  integer, parameter :: maxcls = 64, maxgrp = 3 * maxcls
  logical :: have_labels = .false.  ! the file is there and can be used
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

  ! ---- the job
  character(len=16) :: method = ' '
  logical :: with_q = .false.
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

  ! ---- batch sizes: the memory plan (xzplan.F90)
  integer(8) :: nbp = 1             ! occupied pairs of a batch of pass 1 and pass 2
  integer(8) :: nqs = 1             ! rows of a slab of the dressed <ab||cd>
  integer(8) :: nbt = 1             ! occupied triples of a batch of the pp ladder
  integer(8) :: nba = 1             ! first virtual indices of a batch of the W_vvvo terms
  integer(8) :: ncb = 0             ! columns of a product of the occupied operator

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

  logical :: qdebug = .false.       ! print the parts of (Q) (XZ_QDEBUG)
  real(8) :: qparts(4) = 0d0
  logical :: qsampled = .false.     ! (Q) ran in sample mode: its energies are not (Q)

  ! ---- wall times of the stages of the iterations (seconds)
  integer, parameter :: ntime = 16
  real(8) :: tstage(ntime) = 0d0
  character(len=28), parameter :: tname(ntime) = [character(len=28) :: &
    'T1 dressing', 'pass-1 operands', 'pass 1', 'W_vvvo', 'intermediates', &
    'dressed <ab||cd> rows', 'pp ladder: gather', 'pp ladder: GEMM', 'pp ladder: reduction', &
    'CCSD part', 'occupied operator', 'pass 2', 'hh ladder, F_oo', 'T3 update', 'DIIS', 'energy']
end module xzdata
