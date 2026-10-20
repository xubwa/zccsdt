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
  use xzpar, only: myrank, nrank, xzmpi_bcast_w
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
  logical :: ok, dist
  character(len=256) :: arg
  integer :: larg, ios
  ZT :: half, zz
  real(8), parameter :: ev = 27.211386245988d0, cm = 219474.6313632d0

  zz = 0
  m0 = xzmark()
  call xzdglay(w)
  call xzdquadn()
  call xzeomstep()
  ! the pairs of every rank (the same for all targets) - their rows of r4 and
  ! their part of R13 (the rows of T3 and Xv); one rank: all
  allocate(guess(max(dnt, 1_8)))
  nsel = 0
  do it = 1, dnt
    call problem(it, s, nroot, ok)
    if (.not. ok) cycle
    nsel = nsel + 1
    guess(nsel) = s
  end do
  call xzdranges(w, nsel, guess)
  deallocate(guess)
  ! R13 in the work array if all of EOM-DEA fits there with it (with the
  ! smallest batches and the smallest subspace), else in the file XZDXV
  ! (XZDXV.<rank>), read once per sigma vector; option deaxv m / f: the one or the other
  call xzoptget('deaxv', arg, larg)
  if (larg > 0) then
    dxvfile = arg(1:1) == 'f' .or. arg(1:1) == 'F'
  else
    dxvfile = .false.
    if (xzdry) then
      k = xzmark()
      peak = xzpeak
      xzpeak = xztop
      call allocs()
      t = xzpeak
      call xzrel(k)
      xzpeak = peak
    else
      call xzdry_begin()
      call allocs()
      call xzdry_end(t)
    end if
    dxvfile = t > xzsize
  end if
  if (xzdry) then
    ! (the plan: the intermediates, and every problem with the smallest
    ! batches and the smallest subspace that can run)
    call allocs()
    call xzrel(m0)
    return
  end if
  call N(xzeomkeep)()
  call N(xzdkeep)()

  ! more than one rank (docs/native.md, section 20.6): every rank makes the
  ! intermediates and holds its rows of r4 of every vector; rank 0 runs the
  ! solver, the other ranks carry out its commands (N(xzeserve))
  dist = nrank > 1
  call xzdrdpar()
  if (dxvfile) call xvfile('o')
  t0 = omp_get_wtime()
  if (dist) then
    call xzmpi_bcast_w(w(jt1), NW * no * nv)
    call xzmpi_bcast_w(w(jt2), NW * no * no * nv * nv)
    call xzmpi_bcast_w(w(jt3), NW * n3)
  end if
  call N(xzeomim)(w)
  t1 = omp_get_wtime()
  call N(xzdim)(w)
  tdeaim = omp_get_wtime() - t0
  if (myrank > 0) then
    write(6, '(/,a)') '  EOM-DEA-CCSDT: the rows of r4 of this rank in the sigma vectors and the vectors of the solver'
    write(6, '(a,f10.2,a)') '   intermediates of the ground state: ', tdeaim, ' s'
    flush(6)
    edea = .true.
    ndeasig = 0
    tdpart = 0d0
    call N(xzeserve)(w, nsig)
    edea = .false.
    edist = .false.
    ddist = .false.
    write(6, '(a,i6,a)') '   ', nsig, ' sigma vectors'
    write(6, '(a,8f8.2)') '   seconds of them: intermediates, pp, ring, hh, rows, 3p1h, 2p, MPI and XZDXV ', tdpart(1:7), &
      tdpart(12)
    if (dxvfile) call xvfile('c')
    call xzrel(m0)
    return
  end if
  write(6, '(/,a)') '  EOM-DEA-CCSDT: two electrons attached to the closed-shell reference (R = 2p + 3p1h + 4p2h, the sector itself)'
  write(6, '(a)') '  ' // repeat('-', 98)
  write(6, '(a,f10.2,a,f10.2,a)') '   intermediates of the ground state: ', tdeaim, ' s (those of EOM-EE ', t1 - t0, &
    ' s, then Xv and the forms of T3)'
  write(6, '(a,f10.1,a,f10.1,a,f10.1,a,f10.1,a)') '   of them (MB): Xv with the rows of T3 ', 8d-6 * NW * r13loc, &
    ', T3 on (e | cd) ', 8d-6 * NW * t12size, ', T3 by an index and a pair ', 8d-6 * NW * t3psize, ', W_pp blocks ', &
    8d-6 * NW * (nvv + wbsize)
  if (dxvfile) then
    write(6, '(a)') '   Xv with the rows of T3: in the file XZDXV (it does not fit in the work array with the rest), ' // &
      'read pair by pair in every sigma vector'
  else
    write(6, '(a)') '   Xv with the rows of T3: in the work array'
  end if
  if (dist) write(6, '(a,i4,a)') '   over ', nrank, ' ranks: every rank Xv of its pairs (its rows of r4)'
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
    call N(xzdwq2)(w)
    nlen = dn2 + dsz3 + dsz4
    dbatch = 134217728_8 / NW
    call xzoptget('deabatch', arg, larg)
    if (larg > 0) read(arg(1:larg), *) dbatch
    call xzoptget('deasig', arg, larg)
    kxa = xzget(NW * nlen, 'DEA: a vector')
    kxb = xzget(NW * nlen, 'DEA: its sigma vector')
    edea = .true.
    dvsd = dn2 + dsz3
    dv1 = dn2
    dvbig = dsz4
    open(unit=74, file=arg(1:larg), form='unformatted', access='stream', status='old', iostat=ios)
    if (ios == 0) read(74, iostat=ios) w(kxa:kxa + NW * nlen - 1)
    close(74)
    if (ios /= 0) then
      write(6, '(a,i12,a)') ' @XZCCSDT-F, option deasig: the file does not hold a vector of ', nlen, ' elements.'
      call xzerrex
    end if
    t0 = omp_get_wtime()
    if (dist) then
      ! (over the ranks: the vector to all, the sigma vector from the rows of every rank)
      call xzdslices(int(NW, 8))
      ddist = .true.
      call xzecmd(op_shift, dble(s), dble(dbatch))
      ekx = [kxa, kxb]
      edist = .true.
      ecmd = .true.
      evoff = 0
      evlen = dvsd + esoff(1)
      call N(xzepspread)(w, kxa)
      call N(xzepsig)(w, kxa, kxb, .false.)
      call N(xzepcollect)(w, kxb)
      ecmd = .false.
      edist = .false.
      ddist = .false.
      call xzecmd(op_endshift)
      call xzecmd(op_end)
    else
      call N(xzdsig)(w, kxa, kxb)
    end if
    if (dxvfile) call xvfile('c')
    edea = .false.
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
    call N(xzdwq2)(w)
    kxa = xzget(NW * nlen, 'DEA: a vector')
    kxb = xzget(NW * nlen, 'DEA: its sigma vector')
    msub = subspace(nroot)
    ! (over the ranks: the rows of r4 of every rank, before the batches - a
    ! sigma vector of a rank takes the buffers of its rows)
    if (dist) then
      call xzdslices(int(NW, 8))
      ddist = .true.
      write(6, '(a,i4,a,i14,a,i14,a)') '   over ', nrank, ' ranks: rank 0 holds r2, r3 and ', esoff(1), &
        ' elements of r4, the longest part of another rank ', esmax, ' elements'
    end if
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
      if (dist) call xzecmd(op_shift, dble(s), dble(dbatch))
      edist = dist
      call N(xzesolve)(w, .false., nroot, ethr, emaxcyc, msub, nst, 0_8, guess, kxa, kxb, avail, &
                       ee, rmax, rnorm, wts, cyc, nfound, nsig, nit, 0_8, 0, lead, vlead, .false.)
      edist = .false.
      if (dist) call xzecmd(op_endshift)
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
      if (dist) write(6, '(a,f8.2,a)') '     (the vector to the ranks and the sums over them: ', &
        tdpart(12) / max(ndeasig, 1_8), ' s per sigma vector)'
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
    ddist = .false.
    call xzrel(ms)
  end do
  if (dist) call xzecmd(op_end)
  if (dxvfile) call xvfile('c')
  call xzrel(m0)

contains

  ! the allocations of EOM-DEA (a dry run): the arrays of the ground state,
  ! then every target with the smallest batches and the smallest subspace
  subroutine allocs()
    integer(8) :: mk, mp, kk
    mk = xzmark()
    call N(xzeomkeep)()
    call N(xzdkeep)()
    call N(xzeomim)(w)
    call N(xzdim)(w)
    do it = 1, dnt
      call problem(it, s, nroot, ok)
      if (.not. ok) cycle
      call xzdsec(s)
      if (nrank > 1) then
        call xzdslices(int(NW, 8))
        ddist = .true.
      end if
      dbatch = 1
      nlen = dn2 + dsz3 + dsz4
      mp = xzmark()
      call N(xzdwq2)(w)
      kk = xzget(NW * nlen, 'DEA: a vector')
      kk = xzget(NW * nlen, 'DEA: its sigma vector')
      msub = subspace(nroot)
      kk = xzget(msub * NW * min(nlen, 1048576_8) + 64, 'DEA: the subspace (the buffer of its files)')
      call N(xzdsig)(w, kk, kk)
      call N(xzdprec)(w, kk, zz)
      ddist = .false.
      call xzrel(mp)
    end do
    call xzrel(mk)
  end subroutine

  ! the file XZDXV of R13 ('o': opened, 'c': closed and deleted)
  subroutine xvfile(how)
    character, intent(in) :: how
    character(len=16) :: fname
    integer(8) :: free_bytes
    fname = 'XZDXV'
    if (myrank > 0) write(fname, '(a,i4.4)') 'XZDXV.', myrank
    if (how == 'c') then
      close(81, status='delete')
      return
    end if
    free_bytes = xz_free_disk('.' // c_null_char)
    if (free_bytes >= 0 .and. 8d0 * NW * r13loc > dble(free_bytes) - 1073741824d0) then
      write(6, '(/,a,f12.1,a,f12.1,a)') ' @XZCCSDT-F, the disk of the run directory does not hold the file XZDXV of Xv (', &
        8d-6 * NW * r13loc, ' MB; free: ', 1d-6 * free_bytes, ' MB), and the work array does not hold it either.'
      call xzerrex
    end if
    open(unit=81, file=trim(fname), form='unformatted', access='stream', status='replace', iostat=ios)
    if (ios /= 0) then
      write(6, '(/,a,a)') ' @XZCCSDT-F, the file of Xv cannot be opened: ', trim(fname)
      call xzerrex
    end if
  end subroutine

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

! ---------------------------------------------------------------------------
! The pp operator of the 2p row, Wpp + F_vv on pairs, of the group of the
! target (from the pp operator of the T3 equation, Wpp + F_vv / 2)
subroutine N(xzdwq2)(w)
  use xzdata
  use xzmem
  use xzkern
  use xzdead
  implicit none
  real(8), target :: w(*)
  integer(8) :: mk, kpib
  ZT :: half
  dwq2 = xzget(NW * dn2 * dn2, 'DEA: the pp operator of the 2p row')
  if (dn2 == 0 .or. xzdry) return
  mk = xzmark()
  kpib = xzget(nv * nv, 'DEA: the pairs of a group')
  call N(xzepib)(nv, vpoff(ds), dn2, w(jvpidx), w(kpib))
  call N(xzecopy)(dn2 * dn2, w(jvvvv + NW * vvoff(ds)), w(dwq2))
  half = 0.5d0
  call N(zc_pair_one_body_add)(w(dwq2), dn2, dn2, w(jvpa + vpoff(ds)), w(jvpb + vpoff(ds)), w(kpib), nv, w(efevv), &
                               nv, half)
  call xzrel(mk)
end subroutine

! ---------------------------------------------------------------------------
! A rank other than 0, at the command of rank 0 that a target starts
! (N(xzeserve), op_shift): the sector of the shift s, the rows of r4 of this
! rank, the batches nb of rank 0, the 2p operator, two whole vectors; ms, mv
! the marks of the problem and of its vectors.
subroutine N(xzdshift)(w, s, nb, ms, mv)
  use xzdata
  use xzmem
  use xzdead
  use xzeomd
  use xzpar, only: myrank
  implicit none
  real(8), target :: w(*)
  integer(8), intent(in) :: s, nb
  integer(8), intent(out) :: ms, mv
  call xzdsec(s)
  call xzdslices(int(NW, 8))
  ddist = .true.
  dbatch = nb
  dvsd = dn2 + dsz3
  dv1 = dn2
  dvbig = dsz4
  ms = xzmark()
  call N(xzdwq2)(w)
  ekx(1) = xzget(NW * (dvsd + dvbig), 'DEA: a vector')
  ekx(2) = xzget(NW * (dvsd + dvbig), 'DEA: its sigma vector')
  mv = xzmark()
  edist = .true.
  ecmd = .true.
  evfile = .false.
  evoff = dvsd + esoff(myrank)
  evlen = esoff(myrank + 1) - esoff(myrank)
  write(6, '(a,i5,a,i12,a,i12,a,i14,a,f10.1,a)') '   EOM-DEA, the shift ', s, ' classes: the rows of r4 of this rank are the pairs ', &
    dp0, ' to ', dp1 - 1, ' (', evlen, ' elements, ', 8d-6 * NW * evlen, ' MB of a vector)'
  flush(6)
end subroutine
