subroutine xzplan(w, nslot, usefile, nch)
  use xzdata
  use xzmem
  implicit none
  real(8) :: w(*)
  integer(8), intent(out) :: nslot, nch
  logical, intent(out) :: usefile
  integer(8) :: peak
  real(8) :: e
  logical :: c
  nbp = max(1_8, min(npo, 2000000000_8 / (3 * 8 * no * nv * nvp)))
  nqs = nvp
  nbt = max(1_8, min(nto, 1300000000_8 / (2 * 16 * nvp * nv)))
  nba = nv
  ncb = 0
  qc22 = max(1_8, min(nvp, 1073741824_8 / (16 * nvp)))
  qc13 = max(1_8, min(nvt, 1073741824_8 / (16 * nv)))
  qxv = max(3_8, no - 1)
  qrows = max(1_8, min(nvp, 1073741824_8 / (32 * nv * nv)))
  nslot = ndiis
  usefile = .false.
  nch = 0
  call xzdry_begin()
  call xzlayout(w)
  call xzcc_z(w, nslot, usefile, nch, e, c, peak)
  call xzdry_end(peak)
  write(6, '(a,i14,a,i14)') '   plan: words needed ', peak, '  of ', xzsize
  if (peak > xzsize) call errex
end subroutine
