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
module xzdata
  implicit none

  ! ---- sizes
  integer(8) :: no = 0, nv = 0      ! active occupied and virtual spinors
  integer(8) :: ny = 0              ! no + nv
  integer(8) :: npo = 0, nvp = 0    ! pairs i<j, a<b
  integer(8) :: nto = 0, nvt = 0    ! triples i<j<k, a<b<c
  integer(8) :: nfrozen = 0, ndropped = 0, ntot = 0
  integer(8) :: nwe = 2             ! words of an element: 2 complex, 1 real

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

  logical :: qdebug = .false.       ! print the parts of (Q) (XZ_QDEBUG)
  real(8) :: qparts(4) = 0d0

  ! ---- wall times of the stages of the iterations (seconds)
  integer, parameter :: ntime = 16
  real(8) :: tstage(ntime) = 0d0
  character(len=28), parameter :: tname(ntime) = [character(len=28) :: &
    'T1 dressing', 'pass-1 operands', 'pass 1', 'W_vvvo', 'intermediates', &
    'dressed <ab||cd> rows', 'pp ladder: gather', 'pp ladder: GEMM', 'pp ladder: reduction', &
    'CCSD part', 'occupied operator', 'pass 2', 'hh ladder, F_oo', 'T3 update', 'DIIS', 'energy']
end module xzdata
