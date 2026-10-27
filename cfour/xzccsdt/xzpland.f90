!
! The state of the memory plan (xzplan.f90): the batch sizes and their
! bounds, what (Q) and a rank other than 0 take with the plan.
! (A file of its own: CFOUR's config/mkdep90.py makes a file that uses a
! module it defines depend on itself, and make warns of a circular dependency.)
!
module xzpland
  implicit none
  integer, parameter :: npar = 8
  ! 1 nqs, 2 nbt, 3 nbp, 4 nba, 5 qxv, 6 qc22, 7 qc13, 8 qrows; 9 nc3w (CC3
  ! without stored triples: elements of the work of a batch of occupied triples)
  integer(8) :: plo(npar + 1), phi(npar + 1)
  integer(8) :: pslot = 0, pchunk = 0
  logical :: pfile = .true.
  ! the Krylov subspace of the Lambda equations: pairs, in the file, records
  integer(8) :: lslot = 3, lchunk = 0
  logical :: lfile = .true.
  logical :: planq = .false.          ! only (Q) is planned (a rank that does no iterations)
  integer(8) :: pneedq = 0            ! words of the work array that (Q) takes with the plan
  integer(8) :: pneedw = 0            ! words that a rank other than 0 takes with the plan (iterations, (Q))
contains
  subroutine setpar(k, v)
    use xzdata
    integer, intent(in) :: k
    integer(8), intent(in) :: v
    select case (k)
    case (1)
      nqs = v
    case (2)
      nbt = v
    case (3)
      nbp = v
    case (4)
      nba = v
    case (5)
      qxv = v
    case (6)
      qc22 = v
    case (7)
      qc13 = v
    case (8)
      qrows = v
    case (9)
      nc3w = v
    end select
  end subroutine
end module xzpland
