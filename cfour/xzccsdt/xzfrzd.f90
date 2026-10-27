!
! The places of the regions of the density stage with a frozen or dropped
! index in the work array (xzfrz.f90).
! (A file of its own: CFOUR's config/mkdep90.py makes a file that uses a
! module it defines depend on itself, and make warns of a circular dependency.)
!
module xzfrzd
  implicit none
  integer(8) :: nheld = 0           ! frozen occupied + dropped virtual spinors
  integer(8) :: nall = 0            ! all spinors
  integer(8) :: nao = 0             ! AO functions (nall / 2)
  integer(8) :: jfao = 0            ! the list of AO integrals
  integer(8) :: jfc = 0, jfca = 0, jfcb = 0   ! spinors [2 nao][nall]; their upper and lower halves
  integer(8) :: jfcf = 0            ! cf(0:nall-1)
  integer(8) :: jfe = 0             ! spinor energies in the full order
  integer(8) :: jfwa = 0, jfwb = 0, jft0 = 0, jft1 = 0, jfc1 = 0   ! work of the transformation
end module xzfrzd
