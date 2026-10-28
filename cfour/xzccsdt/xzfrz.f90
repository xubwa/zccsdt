!
! Integrals with one frozen occupied or dropped virtual spinor ("held"
! spinors) for the density stage (xzden.inc, docs/native.md section 13):
! (h q|r s), h one held spinor, q, r, s over ALL spinors of the SCF.  They are
! not in the files of xx2cvtran; they are made here the way CFOUR's own
! density program makes them (x2cccdenocbp/giintccsdp4.f90), by its routines:
!
!   int_mintv2e (libx2cint)       the list of AO integrals of the file IIII,
!                                 in core: 2 words per integral
!   vtranpsog (an object of the   the transformation with the first index in
!   program directory x2czvec)    a small set and three indices over all spinors
!
! Only linked (Makefile: FRZOBJ, -lx2cint); nothing of CFOUR is copied.  All
! memory is regions of the work array.  The spinors are those of JOBARC
! (scf2ccmo), their energies those of scf2ceig.
!
! The order of all spinors here ("full order"): active occupied (in the order
! of the program), active virtual, frozen occupied, dropped virtual; cf(p) is
! CFOUR's number (from 1) of the spinor p of this order.
!
! ---------------------------------------------------------------------------
! The regions that live for the stage, and (not in a dry run) their contents.
subroutine xzfrzinit(w)
  use xzdata
  use xzmem
  use xzfrzd
  implicit none
  real(8) :: w(*)
  integer(8) :: ntri, m, kbuf
  logical :: there
  nheld = nfrozen + ndropped
  nall = ntot
  nao = nall / 2
  ntri = nao * (nao + 1) / 2
  jfcf = xzget(nall, 'order of all spinors')
  jfe = xzget(nall, 'energies of all spinors')
  jfc = xzget(4 * nao * nall, 'spinors of the SCF')
  jfca = xzget(2 * nao * nall, 'spinors of the SCF, upper')
  jfcb = xzget(2 * nao * nall, 'spinors of the SCF, lower')
  ! one held spinor at a time: (h mu|kappa lambda) real and imaginary, then
  ! (h q|kappa lambda) in their place; (h q|r s) of CFOUR in the other region
  jfwa = xzget(4 * nao * ntri * 2, 'half-transformed integrals')
  jfwb = xzget(2 * nall ** 3, 'integrals of a held spinor, as transformed')
  jft0 = xzget(8 * nao * nao, 'work of the transformation')
  jft1 = xzget(4 * nao * nao, 'work of the transformation')
  jfc1 = xzget(4 * nao, 'a held spinor')
  ! the AO integrals: at most ntri (ntri + 1) / 2 of them, 2 words each
  jfao = xzget((ntri + 1) * ntri + 1, 'AO integrals in core')
  if (xzdry) return
  inquire(file='IIII', exist=there)
  if (.not. there) then
    write(6, '(/,a)') ' @XZCCSDT-F, no file IIII (the AO integrals of xsox2catm): the density with frozen or'
    write(6, '(a)') '   dropped spinors needs integrals with one of them, which are transformed here.'
    call xzerrex
  end if
  m = xzmark()
  kbuf = xzget(2 * nall, 'record of scf2ceig')
  call xzfrzset(w(jfcf), w(jfe), w(kbuf), w(jfc), w(jfca), w(jfcb))
  call xzrel(m)
  call xzfrzao(w(jfao), (ntri + 1) * ntri + 1)
end subroutine xzfrzinit

subroutine xzfrzset(cf, e, buf, c, ca, cb)
  use xzdata
  use xzfrzd
  implicit none
  integer(8), intent(out) :: cf(0:nall-1)
  real(8), intent(out) :: e(0:nall-1)
  real(8) :: buf(2, 0:nall-1)
  complex(8) :: c(2 * nao, nall), ca(nao, nall), cb(nao, nall)
  integer(8) :: p, iintfp
  do p = 0, no - 1
    cf(p) = nfrozen + operm(p) + 1
  end do
  do p = 0, nv - 1
    cf(no + p) = nfrozen + no + vperm(p) + 1
  end do
  do p = 0, nfrozen - 1
    cf(ny + p) = p + 1
  end do
  do p = 0, ndropped - 1
    cf(ny + nfrozen + p) = nfrozen + ny + p + 1
  end do
  call xzrfile('scf2ceig', buf, 2 * nall)
  do p = 0, nall - 1
    e(p) = buf(1, cf(p) - 1)
  end do
  call getrec(1, 'JOBARC', 'scf2ccmo', nao * nall * 4, c)
  ca = c(1:nao, :)
  cb = c(nao + 1:2 * nao, :)
end subroutine xzfrzset

subroutine xzfrzao(scr, length)
  use cvwint_module, only: int2e_ncount
  use xzkern
  implicit none
  integer(8), intent(in) :: length
  real(8) :: scr(length)
  flush(6)
  call xz_stdout(1_8)
  call int_mintv2e(scr, length)
  flush(6)
  call xz_stdout(0_8)
  write(6, '(a,i14,a,f10.1,a)') '   AO integrals in core for the held spinors: ', int2e_ncount, ' (', &
    16d-6 * int2e_ncount, ' MB)'
end subroutine xzfrzao

! ---------------------------------------------------------------------------
! (h q|r s) of the held spinor h (0 .. nheld-1: frozen, then dropped), as
! CFOUR's routine leaves them in the region jfwb: conj(v(s, r, q)) with q, r,
! s CFOUR's numbers.
subroutine xzfrzone(w, h)
  use xzdata
  use xzfrzd
  use xzkern
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: h
  integer(8) :: hc, ntri
  ntri = nao * (nao + 1) / 2
  hc = h + 1
  if (h >= nfrozen) hc = nfrozen + ny + (h - nfrozen) + 1
  call xzfrzcol(nao, nall, hc, w(jfca), w(jfcb), w(jfc1), w(jfc1 + 2 * nao))
  ! (CFOUR's routine prints lines for every call)
  flush(6)
  call xz_stdout(1_8)
  call vtranpsog(w(jfao), w(jfao), w(jfwa), w(jfwa + nao * ntri), w(jfwb), w(jfwa), w(jfwb), &
                 w(jft0), w(jft1), w(jfc1), w(jfc1 + 2 * nao), w(jfca), w(jfcb), w(jfca), w(jfcb), &
                 w(jfca), w(jfcb), nao, 1_8, nall, nall, nall)
  flush(6)
  call xz_stdout(0_8)
end subroutine xzfrzone

subroutine xzfrzcol(n, nall, hc, ca, cb, a1, b1)
  implicit none
  integer(8), intent(in) :: n, nall, hc
  complex(8), intent(in) :: ca(n, nall), cb(n, nall)
  complex(8), intent(out) :: a1(n), b1(n)
  a1 = ca(:, hc)
  b1 = cb(:, hc)
end subroutine xzfrzcol
