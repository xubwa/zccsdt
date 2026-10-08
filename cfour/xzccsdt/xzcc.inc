!
! The CCSDT iterations (type-generic): second-order start, Jacobi steps
! (xzstep.inc) with DIIS (xzdiis.F90), convergence as in the Python program:
! the energy change below conv_e and |dT1| + |dT2| + |dT3| below conv_t.
! The amplitudes stay in the work array (jt1, jt2, jt3 of xzdata) for what
! follows; everything else that is taken here is released.
!
subroutine N(xzcc)(w, nslot, usefile, nch, ecorr, converged, ncyc)
  use xzdata
  use xzmem
  use xzkern
  use xzdiisd, only: ncnt
  use omp_lib
  implicit none
  real(8) :: w(*)
  integer(8), intent(in) :: nslot, nch
  logical, intent(in) :: usefile
  real(8), intent(out) :: ecorr
  logical, intent(out) :: converged
  integer(8), intent(out) :: ncyc
  integer(8) :: kt1n, kt2n, kr3, icyc, m0, n
  integer(8), external :: N(xzdiis_target)
  real(8) :: norms(3), eold, enew, de, normt, t0, t1w
  ZT :: e
  character(len=256) :: dbg
  integer :: ldbg

  jt1 = xzget(NW * no * nv, 'T1')
  jt2 = xzget(NW * no * no * nv * nv, 'T2')
  jt3 = xzget(NW * nto * nvt, 'T3')
  m0 = xzmark()
  kt1n = xzget(NW * no * nv, 'new T1')
  kt2n = xzget(NW * no * no * nv * nv, 'new T2')
  call N(xzdiis_alloc)(nslot, usefile, nch)
  if (xzdry) then
    call N(xzstep)(w, jt1, jt2, jt3, kt1n, kt2n, jt3)
    call xzrel(m0)
    return
  end if

  call N(xzguess)(no, nv, w(jeps), w(joovv), w(jt1), w(jt2))
  call zc_vzero_d(NW * nto * nvt, w(jt3))
  call N(xzecorr)(no, nv, w(jt1), w(jt2), w(joovv), e)
  eold = dble(e)
  call get_environment_variable('XZ_DEBUG', dbg, ldbg)
  if (ldbg > 0) then
    ! one step on amplitudes read from a file, the result written next to it
    n = NW * (no * nv + no * no * nv * nv + nto * nvt)
    open(unit=74, file=dbg(1:ldbg), form='unformatted', access='stream', status='old')
    read(74) w(jt1:jt1 + NW * no * nv - 1), w(jt2:jt2 + NW * no * no * nv * nv - 1), &
             w(jt3:jt3 + NW * nto * nvt - 1)
    close(74)
    kr3 = N(xzdiis_target)()
    call N(xzstep)(w, jt1, jt2, jt3, kt1n, kt2n, kr3)
    open(unit=74, file=dbg(1:ldbg) // '.out', form='unformatted', access='stream', status='replace')
    write(74) w(kt1n:kt1n + NW * no * nv - 1), w(kt2n:kt2n + NW * no * no * nv * nv - 1), &
              w(kr3:kr3 + NW * nto * nvt - 1)
    close(74)
    write(6, '(a,a)') ' XZ_DEBUG: one step written to ', dbg(1:ldbg) // '.out'
    call N(xzdiis_close)()
    call xzrel(m0)
    ecorr = eold
    converged = .false.
    ncyc = 0
    return
  end if

  write(6, '(/,a,f20.12)') '  Second-order correlation energy:', eold
  write(6, '(/,a)') '  CCSDT iterations'
  write(6, '(a)') '  ' // repeat('-', 98)
  write(6, '(a)') '   cycle     correlation energy      energy change      |dT1|      |dT2|' // &
                  '      |dT3|   DIIS    time/s'
  write(6, '(a)') '  ' // repeat('-', 98)
  converged = .false.
  ncyc = maxcyc
  do icyc = 1, maxcyc
    t0 = omp_get_wtime()
    kr3 = N(xzdiis_target)()
    call N(xzstep)(w, jt1, jt2, jt3, kt1n, kt2n, kr3)
    t1w = omp_get_wtime()
    call N(xzdiis_upd)(w, w(jt1), w(jt2), jt3, w(kt1n), w(kt2n), kr3, norms)
    tstage(15) = tstage(15) + omp_get_wtime() - t1w
    t1w = omp_get_wtime()
    call N(xzecorr)(no, nv, w(jt1), w(jt2), w(joovv), e)
    tstage(16) = tstage(16) + omp_get_wtime() - t1w
    enew = dble(e)
    de = enew - eold
    eold = enew
    normt = norms(1) + norms(2) + norms(3)
    write(6, '(i8,f23.14,es19.6,3es11.3,i7,f10.2)') icyc, enew, de, norms, ncnt, omp_get_wtime() - t0
    flush(6)
    if (abs(de) < conv_e .and. normt < conv_t) then
      converged = .true.
      ncyc = icyc
      exit
    end if
  end do
  write(6, '(a)') '  ' // repeat('-', 98)
  ecorr = eold
  call N(xzdiis_close)()
  call xzrel(m0)
end subroutine
