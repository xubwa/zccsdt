'''Front end of the approximate iterative models CC2, CC3 (and of CCSD), for
spin orbitals and spinors - with zccsdt.CCSDT, CCSDTQ the ladder

    CC2 < CCSD < CC3 < CCSDT < CC4 < CCSDTQ

of ground states and excitation energies (docs/ccn.md; docs/usage.md).

    mycc = zccsdt.CC3(mf).run()         # GHF or X2C reference; RHF / UHF as GHF spin orbitals
    e, res = mycc.eom_ee(nroots=4)      # CC3 linear-response excitation energies
    e_ip, _ = mycc.eom_ip(nroots=2)

The classes are those of zccsdt.ccsdt with another model: the options, the
output, the memory plans, the symmetry labels and eom_ee / eom_ip / eom_ea
of GCCSDT (CC3: T3 stored, in the blocks of the labels where the reference
has them), with the Jacobian of the model in the eigenproblem.
'''

from math import comb

import numpy as np

from zccsdt.ccsdt import GCCSDT, _count, _size


def _not_here(name):
    def method(self, *args, **kwargs):
        raise NotImplementedError('%s: %s is defined for CCSDT amplitudes (zccsdt.CCSDT)' % (self.__class__.__name__, name))
    return method


class _Model(GCCSDT):
    # (the perturbative corrections, Lambda and the densities are those of CCSDT)
    ccsdt_q = _not_here('the (Q) correction')
    ccsdt_q_lambda = _not_here('(Q)_Lambda')
    solve_lambda = _not_here('Lambda')
    make_rdm1 = _not_here('the density matrix')
    eom_left = _not_here('left-hand vectors of the excited states')
    eom_density = _not_here('densities of the excited states')
    eom_properties = _not_here('properties of the excited states')
    eom_quadruples = _not_here('the quadruples correction of the excitation energies')


class _Doubles(_Model):
    '''A model with T1 and T2 only, on the CCSD level of the CCSDT code
    (dense integrals with <ab||cd> on pairs, everything in memory).'''
    _block_iterations = False       # (no T3: nothing is stored in blocks; a label shift selects states by masks)
    _norm_labels = ('|dT1|', '|dT2|', '-')

    def _amplitude_lines(self, eris):
        o, v = eris.nocc, eris.nmo - eris.nocc
        dtype = np.result_type(eris.fock, eris.oovv)
        vpp = getattr(eris, '_vvvv_pp', None)
        return [('amplitudes', 'T1 %s   T2 %s, %s' % (_count(o * v), _count(o * o * v * v),
                                                    'complex' if dtype.kind == 'c' else 'real')),
                ('integrals', 'packed <ab||cd> %s in memory' % (_size(vpp.nbytes) if vpp is not None else '-'))]

    def _make_plan(self, eris):
        return None

    def _print_plan(self, pr, plan, eris):
        pr.kv('memory plan', 'everything in memory (the integrals and intermediates of a CCSD step)')

    def _make_diis_plan(self, eris, plan):
        from zccsdt import disk
        self._diis_plan = None
        return disk.DiisPlan(space=self.diis_space, requested=self.diis_space, with_big=False, case='m')

    def _options(self, plan, callback):
        opts = super()._options(None, callback)
        opts.diis_plan = None
        return opts

    def _print_diagnostics(self, pr):
        t1, t2 = self.t1, self.t2
        o = t1.shape[0]
        pr.line(' diagnostics       T1 %.4f   (|t1| / sqrt(number of correlated electrons))'
                % (np.linalg.norm(t1) / np.sqrt(max(o, 1))))

    def eom_ip(self, nroots=3, **kwargs):
        '''The ``nroots`` lowest ionization energies (Hartree) and the
        zccsdt.cc.eom.Result: the EE problem with a ghost virtual spinor
        (docs/eom.md 5), which costs little without triples.'''
        kwargs.pop('short', None)
        return self._eom_ghost('ip', nroots, kwargs)

    def eom_ea(self, nroots=3, **kwargs):
        '''The ``nroots`` lowest E(N+1) - E(N); as eom_ip().'''
        kwargs.pop('short', None)
        return self._eom_ghost('ea', nroots, kwargs)


class GCC3(_Doubles):
    __doc__ = '''CC3 on a GHF or two-component spinor (X2C) reference: the singles
    and doubles equations of CCSDT, the triples equation
    <T| [V~, T2] + [F, T3] |0> with the T1-dressed integrals V~ and the Fock
    matrix F of the reference as it is (docs/ccn.md).

    store_t3 : bool
        True (default): T3 is stored as in CCSDT - in the blocks of the
        symmetry labels of a linear molecule - and everything of GCCSDT
        that is about the iterations applies (memory plans, scratch files,
        spinors that are not canonical); eom_ee() with label shifts,
        eom_ip(), eom_ea(): the eigenvalues of the CC3 Jacobian with R3
        stored.
        False: no T3.  The triples are made per batch of occupied triples
        from the drive terms, contracted into the singles and doubles
        equations and discarded (zccsdt.cc.cc3fly): o^3 v^4 per iteration
        and the memory of a CCSD step, in the blocks of the labels too.
        (Semicanonical spinors: the front end rotates them.)  eom_ee(): the totally symmetric states, from
        the folded, frequency-dependent eigenproblem in the space of
        singles and doubles (each root self-consistent in omega); the same
        numbers as with T3 stored.

''' + GCCSDT.__doc__
    _title = 'GCC3'
    _subtitle = 'CC3 on spin orbitals or spinors (GHF, X2C), real or complex'
    _method = 'CC3'
    model = 'cc3'
    store_t3 = True
    _block_iterations = True
    _norm_labels = GCCSDT._norm_labels
    _keys = GCCSDT._keys | {'store_t3'}

    def _amplitude_lines(self, eris):
        if self.store_t3:
            return GCCSDT._amplitude_lines(self, eris)
        return _Doubles._amplitude_lines(self, eris) + [('triples', 'not stored: made per batch of occupied triples')]

    def _make_plan(self, eris):
        return GCCSDT._make_plan(self, eris) if self.store_t3 else None

    def _print_plan(self, pr, plan, eris):
        if self.store_t3:
            return GCCSDT._print_plan(self, pr, plan, eris)
        pr.kv('memory plan', 'the integrals and intermediates of a CCSD step in memory; a work array of at most '
              '2000 MB for the triples of a batch')

    def _make_diis_plan(self, eris, plan):
        return GCCSDT._make_diis_plan(self, eris, plan) if self.store_t3 else _Doubles._make_diis_plan(self, eris, plan)

    def _options(self, plan, callback):
        if self.store_t3:
            return GCCSDT._options(self, plan, callback)
        opts = _Doubles._options(self, plan, callback)
        opts.store_t3 = False
        return opts

    def _print_diagnostics(self, pr):
        return GCCSDT._print_diagnostics(self, pr) if self.store_t3 else _Doubles._print_diagnostics(self, pr)

    def eom_ee(self, nroots=3, conv_tol=1e-6, shift=None, **kwargs):
        if self.store_t3:
            return GCCSDT.eom_ee(self, nroots, conv_tol, shift=shift, **kwargs)
        from zccsdt import disk
        from zccsdt.cc import ccn
        from pyscf.lib import logger
        if self.t1 is None:
            raise RuntimeError('GCC3.eom_ee(): no amplitudes (kernel() first)')
        if shift not in (None, 0):
            raise NotImplementedError('CC3 without stored triples: the totally symmetric states (shift 0); states with '
                                      'a label shift with store_t3 = True')
        eris = self._eris if self._eris is not None else self.ao2mo()
        with disk.limit(self.max_disk):
            res = ccn.eom_folded(eris, self.t1, self.t2, nroots, conv_tol, opts=self._options(None, None),
                                 verbose=self.verbose >= logger.NOTE, stdout=self.stdout, **kwargs)
        self.eom, self.e_ee = res, res.e
        return res.e, res

    def _no_fly(self, what):
        raise NotImplementedError('CC3 without stored triples: %s with store_t3 = True' % what)

    def eom_ip(self, nroots=3, **kwargs):
        return GCCSDT.eom_ip(self, nroots, **kwargs) if self.store_t3 else self._no_fly('ionized states')

    def eom_ea(self, nroots=3, **kwargs):
        return GCCSDT.eom_ea(self, nroots, **kwargs) if self.store_t3 else self._no_fly('attached states')


class GCC2(_Doubles):
    __doc__ = '''CC2 on a GHF or two-component spinor (X2C) reference: the singles
    equation of CCSD, the doubles equation <D| V~ + [F, T2] |0> with the
    T1-dressed integrals and the Fock matrix of the reference as it is
    (docs/ccn.md).  On the integrals of the CCSD step (<ab||cd> included):
    the member of the hierarchy, not the o^2 v^3 algorithm of CC2.
    eom_ee(), eom_ip(), eom_ea(): the eigenvalues of the CC2 Jacobian.

''' + GCCSDT.__doc__
    _title = 'GCC2'
    _subtitle = 'CC2 on spin orbitals or spinors (GHF, X2C), real or complex'
    _method = 'CC2'
    model = 'cc2'


class GCCSD(_Doubles):
    __doc__ = '''CCSD on a GHF or two-component spinor (X2C) reference, on the
    CCSD level of the CCSDT code (T1-dressed, <ab||cd> in memory), with
    EOM-CCSD - for the ladder of docs/ccn.md.  The production CCSD(T) of
    spinors is zccsdt.zccsd.DirectZCCSD.

''' + GCCSDT.__doc__
    _title = 'GCCSD'
    _subtitle = 'CCSD on spin orbitals or spinors (GHF, X2C), real or complex'
    _method = 'CCSD'
    model = 'ccsd'


def _for_reference(cls, name, mf, frozen, mo_coeff, mo_occ):
    from pyscf import scf
    from pyscf.scf import dhf, ghf, hf
    from pyscf.x2c import x2c
    if isinstance(mf, hf.KohnShamDFT):
        raise RuntimeError('%s needs an HF object, not a DFT object (mf.to_hf())' % name)
    if isinstance(mf, dhf.DHF):
        raise NotImplementedError('%s on a four-component (DHF) reference' % name)
    if isinstance(mf, (ghf.GHF, x2c.SCF)):
        return cls(mf, frozen, mo_coeff, mo_occ)
    if mo_coeff is not None or mo_occ is not None:
        raise NotImplementedError('%s with given orbitals needs a GHF or X2C reference' % name)
    if frozen is not None and not isinstance(frozen, (int, np.integer)):
        raise NotImplementedError('%s: a list of frozen orbitals needs a GHF or X2C reference '
                                  '(scf.addons.convert_to_ghf(mf): the list is then one of spin orbitals)' % name)
    return cls(scf.addons.convert_to_ghf(mf), None if frozen is None else 2 * int(frozen))


def CC2(mf, frozen=None, mo_coeff=None, mo_occ=None):
    '''CC2 for the reference ``mf``: GCC2 for GHF and two-component (X2C)
    spinors; an RHF, UHF or ROHF reference is taken as GHF spin orbitals
    (``frozen`` then counts orbitals per spin).'''
    return _for_reference(GCC2, 'CC2', mf, frozen, mo_coeff, mo_occ)


def CCSD(mf, frozen=None, mo_coeff=None, mo_occ=None):
    '''CCSD for the reference ``mf`` (GCCSD), as CC2().'''
    return _for_reference(GCCSD, 'CCSD', mf, frozen, mo_coeff, mo_occ)


def CC3(mf, frozen=None, mo_coeff=None, mo_occ=None):
    '''CC3 for the reference ``mf`` (GCC3), as CC2().'''
    return _for_reference(GCC3, 'CC3', mf, frozen, mo_coeff, mo_occ)


class GCC4(_Model):
    __doc__ = '''CC4 on a GHF or two-component spinor (X2C) reference: the singles,
    doubles and triples equations of CCSDTQ, the quadruples equation
    <Q| [V~, T3] + 1/2 [[V~, T2], T2] + [F, T4] |0> (docs/ccn.md, section 11),
    on semicanonical spinors.

    store_t4 : bool
        False (default): no T4 - the quadruples of every occupied quadruple
        are made from the current amplitudes, contracted into the doubles
        and triples equations and discarded (zccsdt.cc.cc4fly).  The
        iterations are those of GCCSDT (T3 in the blocks of the symmetry
        labels, the memory plan with its scratch files, DIIS) with the pass
        over the quadruples in each: its memory is planned with that of the
        step from ``max_memory`` (zccsdt.cc.cc4fly.plans; ``plan_quadruples``
        after kernel()).  True: T4 stored, from the T4 engine of CCSDTQ (the
        reference route, everything in memory, packed layout).
    workers : int or None
        Processes over the occupied quadruples, at most (default: the cores
        of the process; fewer if the memory does not hold their work
        arrays).  Each has one thread.
    eom_ee(): the folded eigenproblem in the space of singles, doubles and
    triples, R4 folded in at the frequency of every root.

''' + GCCSDT.__doc__
    _title = 'GCC4'
    _subtitle = 'CC4 on spin orbitals or spinors (GHF, X2C), real or complex'
    _method = 'CC4'
    _stages = dict(GCCSDT._stages, **{
        'cc4.dress': ('dressing (Q)', 'T1 dressing of the integrals of the quadruples'),
        'cc4.pack': ('T3 packed', 'T3 on the packed block'),
        'cc4.setup': ('setup', 'quadruples: operands'),
        'cc4.pass': ('quadruples', 'quadruples: the pass over the occupied quadruples'),
        'cc4.fork': ('processes', 'quadruples: starting the processes'),
        'cc4.xo': ('Xo', 'quadruples: T2 intermediate Xo'),
        'cc4.xv': ('Xv', 'quadruples: T2 <vv||vv> intermediate'),
        'cc4.rhs': ('T4 rows', 'quadruples: the rows of T4'),
        'cc4.back': ('T4 sums', 'quadruples: T4 in the T2 and T3 equations'),
        'cc4.update': ('update (Q)', 'quadruples: Jacobi step'),
    })
    model = 'cc4'
    store_t4 = False
    workers = None
    _keys = GCCSDT._keys | {'store_t4', 't4', 'workers', 'plan_quadruples', 'quadruples_stats'}

    @property
    def _block_iterations(self):
        # (T4 stored: the reference route on the packed layout)
        return not self.store_t4

    def _options(self, plan, callback):
        opts = GCCSDT._options(self, plan, callback)
        opts.workers = self.workers
        opts.quadruples = getattr(self, '_qplan', None)
        return opts

    def _make_plan(self, eris):
        from zccsdt.cc import cc4fly
        self._qplan = None
        if self.store_t4:
            return GCCSDT._make_plan(self, eris)
        plan, self._qplan = cc4fly.plans(eris, np.result_type(eris.fock, eris.oovv), self._options(None, None))
        return plan

    def _print_plan(self, pr, plan, eris):
        GCCSDT._print_plan(self, pr, plan, eris)
        q = getattr(self, '_qplan', None)
        if q is None:
            return
        o, v = eris.nocc, eris.nmo - eris.nocc
        pr.kv('quadruples', 'not stored: %d occupied x %s virtual, %s; %s for the pass' % (
            comb(o, 4), _count(comb(v, 4)), 'in the sectors of the symmetry labels' if q.sectors else 'packed layout',
            _size(q.need)), indent=3)
        if q.workers > 1:
            how = '%d processes with one thread each' % q.workers
        elif q.sectors:
            how = 'one process'
        else:
            how = 'one process, the threads of the BLAS'
        pr.kv('', '%s; T2 <vv||vv> intermediate for %s' % (
            how, 'all occupied indices' if q.xv >= max(3, o - 1) else '%d of %d occupied indices at a time'
            % (q.xv, max(3, o - 1))), indent=3)

    def _solve(self, eris, plan, t1, t2, t3, callback):
        from zccsdt.cc import ccn as K
        if not self.store_t4:
            res = GCCSDT._solve(self, eris, plan, t1, t2, t3, callback)
            self.plan_quadruples, self.quadruples_stats = res.quadruples, res.stats
        else:
            res = K.cc4_kernel(eris, self.conv_tol, self._conv_tol_normt(), self.max_cycle, self.diis_space,
                               t1=t1, t2=t2, t3=t3, store_t4=True)
        self._eris = res.eris
        self.t4 = getattr(res, 't4', None)
        return res

    def _first_iteration_note(self):
        st = getattr(self, 'quadruples_stats', None)
        if not st or st.get('workers', 1) < 2:
            return ''
        return ('; the last pass over the quadruples: %d processes with %s of private memory and %s of shared '
                'arrays besides' % (st['workers'], _size(st['private']), _size(st['shared'])))

    def eom_ee(self, nroots=3, conv_tol=1e-6, shift=None, **kwargs):
        '''The ``nroots`` lowest excitation energies of CC4 (Hartree) and the
        Solved of zccsdt.cc.ccn.folded_roots: the folded problem, each root
        self-consistent in its frequency (zccsdt.cc.cc4fly.eom).'''
        from zccsdt.cc import cc4fly
        if self.t1 is None:
            raise RuntimeError('GCC4.eom_ee(): no amplitudes (kernel() first)')
        if shift not in (None, 0):
            raise NotImplementedError('CC4: the states of all labels together (no label shifts)')
        kwargs.setdefault('max_memory', self.max_memory)
        kwargs.setdefault('workers', self.workers)
        e, sol = cc4fly.eom(self._eris, self.t1, self.t2, self.t3, nroots, conv_tol, **kwargs)
        self.eom, self.e_ee = sol, e
        return e, sol

    def eom_ip(self, nroots=3, **kwargs):
        raise NotImplementedError('CC4: ionized and attached states are not implemented')

    eom_ea = eom_ip


def CC4(mf, frozen=None, mo_coeff=None, mo_occ=None):
    '''CC4 for the reference ``mf`` (GCC4), as CC2().'''
    return _for_reference(GCC4, 'CC4', mf, frozen, mo_coeff, mo_occ)
