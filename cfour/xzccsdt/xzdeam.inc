!
! EOM-DEA-CCSDT on the sector itself: the driver (type-generic; xzdea.inc,
! xzdeas.inc; docs/native.md, section 20).  The intermediates of the ground
! state once, then for every target of %xzdea the sector of its shift, the
! solver of the excited states (N(xzesolve), CFOUR's design: one basis for
! all roots of a problem, the corrections r / (omega - D), the subspace in
! the work array or in files) on the vectors [r2 | r3 | r4].
!

subroutine N(xzdea)(w, ecorr)
  use xzdata
  use xzmem
  use xzkern
  use xzeomd
  use xzdead
  use xzpar, only: myrank, nrank
  use omp_lib
  implicit none
  real(8), target :: w(*)
  real(8), intent(in) :: ecorr
  integer(8) :: m0, ms, it, s, nroot, nlen, kxa, kxb, msub, sigw, avail, nfound, nsig, nit, nst, k, r, a, b, q, t
  integer(8) :: peak, blo, bhi, bmid, btarget, reserve, kpib, ng, nsel
  integer(8), allocatable :: guess(:), lead(:, :), cyc(:)
  complex(8), allocatable :: ee(:)
  real(8), allocatable :: rmax(:), rnorm(:), wts(:, :), dg(:)
  ZT, allocatable :: vlead(:, :)
  integer(8), pointer :: vpa(:), vpb(:)
  real(8), pointer :: eps(:)
  real(8) :: t0, t1, tsg, x
  logical :: ok
  character(len=256) :: arg
  integer :: larg, ios
  ZT :: half, zz
  real(8), parameter :: ev = 27.211386245988d0, cm = 219474.6313632d0

  zz = 0
  m0 = xzmark()
  call N(xzeomkeep)()
  call xzdglay(w)
  call xzdquadn()
  call N(xzdkeep)()
  call xzeomstep()

  if (xzdry) then
    ! (the plan: the intermediates, and every problem with the smallest
    ! batches and the smallest subspace that can run)
    call N(xzeomim)(w)
    call N(xzdim)(w)
    do it = 1, dnt
      call problem(it, s, nroot, ok)
      if (.not. ok) cycle
      call xzdsec(s)
      dbatch = 1
      nlen = dn2 + dsz3 + dsz4
      ms = xzmark()
      dwq2 = xzget(NW * dn2 * dn2, 'DEA: the pp operator of the 2p row')
      kxa = xzget(NW * nlen, 'DEA: a vector')
      kxb = xzget(NW * nlen, 'DEA: its sigma vector')
      msub = subspace(nroot)
      k = xzget(msub * NW * min(nlen, 1048576_8) + 64, 'DEA: the subspace (the buffer of its files)')
      call N(xzdsig)(w, kxa, kxb)
      call N(xzdprec)(w, kxa, zz)
      call xzrel(ms)
    end do
    call xzrel(m0)
    return
  end if

  if (nrank > 1) then
    write(6, '(/,a)') ' @XZCCSDT-F, EOM-DEA runs with one rank in this version.'
    call xzerrex
  end if
  call xzdrdpar()
  t0 = omp_get_wtime()
  call N(xzeomim)(w)
  t1 = omp_get_wtime()
  call N(xzdim)(w)
  tdeaim = omp_get_wtime() - t0
  write(6, '(/,a)') '  EOM-DEA-CCSDT: two electrons attached to the closed-shell reference (R = 2p + 3p1h + 4p2h, the sector itself)'
  write(6, '(a)') '  ' // repeat('-', 98)
  write(6, '(a,f10.2,a,f10.2,a)') '   intermediates of the ground state: ', tdeaim, ' s (those of EOM-EE ', t1 - t0, &
    ' s, then Xv and the forms of T3)'
  write(6, '(a,f10.1,a,f10.1,a,f10.1,a,f10.1,a)') '   of them (MB): Xv with the rows of T3 ', 8d-6 * NW * r13size, &
    ', T3 on (e | cd) ', 8d-6 * NW * t12size, ', T3 by an index and a pair ', 8d-6 * NW * t3psize, ', W_pp blocks ', &
    8d-6 * NW * (nvv + wbsize)
  write(6, '(a,es9.2,a,i4,a)') '   convergence: the largest element of a residual below ', ethr, &
    ' (ESTATE_CONV, %eomthrs);  at most ', emaxcyc, ' iterations (ESTATE_MAXCYC)'
  if (dtpar >= 0) then
    write(6, '(a,a,a)') '   parity of the states: ', merge('g', 'u', dtpar == 0), ' (the file xzparity)'
  else
    write(6, '(a)') '   parity of the states: all (no parity line in %xzdea)'
  end if
  flush(6)

  ! ---- a test hook: option deasig file, sigma of the vector of the file (the
  !      first target), written to file.out
  call xzoptget('deasig', arg, larg)
  if (larg > 0) then
    call problem(1_8, s, nroot, ok)
    call xzdsec(s)
    call wq2()
    nlen = dn2 + dsz3 + dsz4
    dbatch = 134217728_8 / NW
    call xzoptget('deabatch', arg, larg)
    if (larg > 0) read(arg(1:larg), *) dbatch
    call xzoptget('deasig', arg, larg)
    kxa = xzget(NW * nlen, 'DEA: a vector')
    kxb = xzget(NW * nlen, 'DEA: its sigma vector')
    open(unit=74, file=arg(1:larg), form='unformatted', access='stream', status='old', iostat=ios)
    if (ios == 0) read(74, iostat=ios) w(kxa:kxa + NW * nlen - 1)
    close(74)
    if (ios /= 0) then
      write(6, '(a,i12,a)') ' @XZCCSDT-F, option deasig: the file does not hold a vector of ', nlen, ' elements.'
      call xzerrex
    end if
    t0 = omp_get_wtime()
    call N(xzdsig)(w, kxa, kxb)
    write(6, '(a,f10.3,a,i12,a,i12,a,i14,a)') '   option deasig: a sigma vector in ', omp_get_wtime() - t0, ' s (2p ', dn2, &
      ', 3p1h ', dsz3, ', 4p2h ', dsz4, ' elements)'
    open(unit=74, file=arg(1:larg) // '.out', form='unformatted', access='stream', status='replace')
    write(74) w(kxb:kxb + NW * nlen - 1)
    close(74)
    write(6, '(a,a)') '   written to ', arg(1:larg) // '.out'
    call xzrel(m0)
    return
  end if

  dnres = 0
  do it = 1, dnt
    t0 = omp_get_wtime()
    call problem(it, s, nroot, ok)
    if (.not. ok) cycle
    call xzdsec(s)
    nlen = dn2 + dsz3 + dsz4
    write(6, '(/,a)') '  ' // repeat('=', 98)
    write(6, '(a,i5,a,i4,a,i4,a)') '   EOM-DEA-CCSDT, target 2 Omega = ', dtgt(it), ' (the shift ', s, ' classes): ', nroot, ' roots'
    write(6, '(a,i14,a,i10,a,i12,a,i14,a,f10.1,a)') '   a vector: ', nlen, ' elements (2p ', dn2, ', 3p1h ', dsz3, ', 4p2h ', &
      dsz4, ') = ', 8d-6 * NW * nlen, ' MB'
    if (dn2 == 0) then
      write(6, '(a)') ' @XZCCSDT-W, no pair of virtual spinors has this target: it is left out.'
      cycle
    end if
    ms = xzmark()
    call wq2()
    kxa = xzget(NW * nlen, 'DEA: a vector')
    kxb = xzget(NW * nlen, 'DEA: its sigma vector')
    msub = subspace(nroot)
    ! the batches of the passes over r4: as large as the work array leaves them
    ! (at most 1 GB of an operand) next to the buffer of the subspace in files
    ! - the batches first, as for the DIIS vectors and the EOM-EE subspace
    reserve = msub * NW * min(nlen, 1048576_8) + 64
    btarget = 1073741824_8 / (8 * NW)
    call xzoptget('deabatch', arg, larg)
    if (larg > 0) then
      read(arg(1:larg), *) btarget
      blo = btarget
    else
      blo = 1
      bhi = btarget
      if (sigpeak(bhi) + reserve <= xzsize) then
        blo = bhi
      else if (sigpeak(blo) + reserve > xzsize) then
        write(6, '(/,a)') ' @XZCCSDT-F, the work array is too short for the sigma vectors of EOM-DEA.'
        write(6, '(a,i16,a,i16)') '   words needed: ', sigpeak(blo) + reserve, '   words there: ', xzsize
        call xzerrex
      else
        do while (bhi - blo > max(1_8, blo / 64))
          bmid = blo + (bhi - blo) / 2
          if (sigpeak(bmid) + reserve <= xzsize) then
            blo = bmid
          else
            bhi = bmid
          end if
        end do
      end if
    end if
    dbatch = blo
    sigw = sigpeak(dbatch) - xztop
    avail = xzsize - xztop - sigw
    write(6, '(a,f10.1,a,f10.1,a,f10.1,a)') '   a sigma vector takes ', 8d-6 * sigw, ' MB of the work array (batches of ', &
      8d-6 * NW * dbatch, ' MB); free besides it: ', 8d-6 * avail, ' MB'
    ! the start: unit vectors on the lowest pairs (a<b) of the masks
    ng = dnguess
    if (ng <= 0) ng = max(4_8, nroot + 2)
    allocate(guess(ng), dg(0:dn2-1))
    call xzdiv(w, jvpa, nvp, vpa)
    call xzdiv(w, jvpb, nvp, vpb)
    call r8view(jeps, no + nv, eps)
    do t = 0, dn2 - 1
      q = vpoff(s) + t
      dg(t) = eps(no + vpa(q)) + eps(no + vpb(q))
      if (dtpar >= 0) then
        if (mod(dpv(vpa(q)) + dpv(vpb(q)) + dtpar, 2_8) /= 0) dg(t) = huge(1d0)
      end if
    end do
    nsel = 0
    do k = 1, ng
      x = huge(1d0)
      a = -1
      do t = 0, dn2 - 1
        if (dg(t) < x) then
          x = dg(t)
          a = t
        end if
      end do
      if (a < 0) exit
      nsel = nsel + 1
      guess(nsel) = a
      dg(a) = huge(1d0)
    end do
    nst = nsel
    nroot = min(nroot, nst)
    allocate(ee(max(nroot, 1_8)), rmax(max(nroot, 1_8)), rnorm(max(nroot, 1_8)), wts(3, max(nroot, 1_8)), &
             cyc(max(nroot, 1_8)), lead(nlead, max(nroot, 1_8)), vlead(nlead, max(nroot, 1_8)))
    if (nst == 0) then
      write(6, '(a)') ' @XZCCSDT-W, no pair of virtual spinors of this target has the parity asked for: it is left out.'
    else
      write(6, '(a,i4,a)') '   start: unit vectors on the ', nst, ' lowest pairs (a<b) of the target'
      flush(6)
      edea = .true.
      dvsd = dn2 + dsz3
      dv1 = dn2
      dvbig = dsz4
      tsg = tdeasig
      t1 = omp_get_wtime()
      ndeasig = 0
      tdpart = 0d0
      call N(xzesolve)(w, .false., nroot, ethr, emaxcyc, msub, nst, 0_8, guess, kxa, kxb, avail, &
                       ee, rmax, rnorm, wts, cyc, nfound, nsig, nit, 0_8, 0, lead, vlead, .false.)
      edea = .false.
      t1 = omp_get_wtime() - t1
      write(6, '(/,a,i5,a,i4,a,f10.2,a,f9.2,a)') '   sigma vectors: ', nsig, '   iterations: ', nit, '   time: ', t1, &
        ' s   (', t1 / max(nsig, 1_8), ' s per sigma vector, the solver included)'
      write(6, '(a,f8.2,a,f8.2,a,f8.2,a,f8.2,a,f8.2,a,f8.2,a,f8.2,a)') '   per sigma vector (s): intermediates ', &
        tdpart(1) / max(ndeasig, 1_8), ', pp ', tdpart(2) / max(ndeasig, 1_8), ', ring ', tdpart(3) / max(ndeasig, 1_8), &
        ', hh ', tdpart(4) / max(ndeasig, 1_8), ', rows ', tdpart(5) / max(ndeasig, 1_8), ', 3p1h ', &
        tdpart(6) / max(ndeasig, 1_8), ', 2p ', tdpart(7) / max(ndeasig, 1_8)
      write(6, '(a,f8.2,a,f8.2,a,f8.2,a,f8.2,a)') '     (of them: the expansions and reductions ', tdpart(8) / max(ndeasig, 1_8), &
        ', the products of the pp ladder ', tdpart(9) / max(ndeasig, 1_8), ', the products of the rows [(ab),(cd)] ', &
        tdpart(10) / max(ndeasig, 1_8), ', [a,(bcd)] ', tdpart(11) / max(ndeasig, 1_8), ')'
      tdeasig = tdeasig + sum(tdpart(1:7))
      write(6, '(/,a)') '   root        omega / Eh            eV            cm-1    max |r|      |r|   cycle    2p   3p1h   4p2h' // &
                        '          total energy / Eh'
      do k = 1, nfound
        write(6, '(i7,f20.12,f14.8,f16.4,2es10.2,i6,3f7.4,f26.12)') k, dble(ee(k)), dble(ee(k)) * ev, dble(ee(k)) * cm, &
          rmax(k), rnorm(k), cyc(k), wts(:, k), escf + ecorr + dble(ee(k))
        if (abs(aimag(ee(k))) > 1d-10) write(6, '(a,es12.4)') '          imaginary part of omega: ', aimag(ee(k))
        if (rmax(k) > ethr) write(6, '(a)') '          NOT converged'
        if (dnres < dmaxres) then
          dnres = dnres + 1
          dres_t(dnres) = dtgt(it)
          dres_k(dnres) = k
          dres_om(dnres) = dble(ee(k))
          dres_mx(dnres) = rmax(k)
          dres_nm(dnres) = rnorm(k)
          dres_wt(:, dnres) = wts(:, k)
          dres_cv(dnres) = rmax(k) <= ethr
          dres_la(dnres) = 0
          dres_lb(dnres) = 0
          dres_lv(dnres) = 0d0
          if (lead(1, k) >= 0) then
            q = vpoff(s) + lead(1, k)
            dres_la(dnres) = nfrozen + no + vperm(vpa(q)) + 1
            dres_lb(dnres) = nfrozen + no + vperm(vpb(q)) + 1
            dres_lv(dnres) = abs(vlead(1, k))
          end if
        end if
      end do
      write(6, '(/,a)') '   the largest elements of r2 (spinors in the order of the SCF: number, label, energy / Eh)'
      do k = 1, nfound
        do r = 1, nlead
          if (lead(r, k) < 0) cycle
          if (abs(vlead(r, k)) < 0.05d0 .and. r > 1) cycle
          q = vpoff(s) + lead(r, k)
          a = vpa(q)
          b = vpb(q)
          write(6, '(a,i4,a,i5,a,i4,a,f12.6,a,i5,a,i4,a,f12.6,a,f10.6)') '   root', k, ': ', &
            nfrozen + no + vperm(a) + 1, ' (', lab(nfrozen + no + vperm(a)), ',', eps(no + a), ') ', &
            nfrozen + no + vperm(b) + 1, ' (', lab(nfrozen + no + vperm(b)), ',', eps(no + b), ')   ', abs(vlead(r, k))
        end do
      end do
    end if
    write(6, '(a,f10.2,a)') '   time of this target: ', omp_get_wtime() - t0, ' s'
    flush(6)
    deallocate(guess, dg, ee, rmax, rnorm, wts, cyc, lead, vlead)
    call xzrel(ms)
  end do
  call xzrel(m0)

contains

  ! the target number ip: its shift in classes, its roots; good: it can be done
  subroutine problem(ip, sc, nr, good)
    integer(8), intent(in) :: ip
    integer(8), intent(out) :: sc, nr
    logical, intent(out) :: good
    integer(8) :: lmin, p
    good = .true.
    nr = dnr(ip)
    if (nr <= 0) nr = enr(1)
    if (nr <= 0) nr = 1
    ! the class of a pair: (2 Omega - 2 l_min) / step
    lmin = minval(lab(nfrozen:nfrozen + no + nv - 1))
    p = dtgt(ip) - 2 * lmin
    sc = p / labstep
    if (ncls == 1) then
      ! (no labels: all pairs are one class; the target is not a block)
      sc = 0
      return
    end if
    if (mod(p, labstep) /= 0 .or. sc < 0 .or. sc >= npg) then
      good = .false.
      if (.not. xzdry) write(6, '(/,a,i6,a)') ' @XZCCSDT-W, EOM-DEA: no pair of virtual spinors has the target ', dtgt(ip), &
        '; it is left out.'
    end if
  end subroutine

  integer(8) function subspace(nr)
    integer(8), intent(in) :: nr
    subspace = emaxsub
    if (subspace <= 0) subspace = max(20 + nr + 2, 4 * nr + 8)
    subspace = max(subspace, 2 * nr + 2)
  end function

  ! the pp operator of the 2p row, Wpp + F_vv on pairs, of the group of the target
  subroutine wq2()
    integer(8) :: mk
    dwq2 = xzget(NW * dn2 * dn2, 'DEA: the pp operator of the 2p row')
    if (dn2 == 0) return
    mk = xzmark()
    kpib = xzget(nv * nv, 'DEA: the pairs of a group')
    call N(xzepib)(nv, vpoff(ds), dn2, w(jvpidx), w(kpib))
    call N(xzecopy)(dn2 * dn2, w(jvvvv + NW * vvoff(ds)), w(dwq2))
    half = 0.5d0
    call N(zc_pair_one_body_add)(w(dwq2), dn2, dn2, w(jvpa + vpoff(ds)), w(jvpb + vpoff(ds)), w(kpib), nv, w(efevv), &
                                 nv, half)
    call xzrel(mk)
  end subroutine

  ! the most of the work array that a sigma vector takes with batches of nb elements
  integer(8) function sigpeak(nb)
    integer(8), intent(in) :: nb
    integer(8) :: pk, keep
    keep = dbatch
    dbatch = nb
    call xzdry_begin()
    call N(xzdsig)(w, kxa, kxb)
    call N(xzdprec)(w, kxa, zz)
    call xzdry_end(pk)
    dbatch = keep
    sigpeak = pk
  end function

  subroutine r8view(k, n, p)
    use iso_c_binding
    integer(8), intent(in) :: k, n
    real(8), pointer :: p(:)
    real(8), pointer :: qq(:)
    call c_f_pointer(c_loc(w(k)), qq, [max(n, 1_8)])
    p(0:max(n, 1_8) - 1) => qq
  end subroutine
end subroutine
