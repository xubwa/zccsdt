#!/usr/bin/env python
'''
CCSDT, CCSDT(Q), CCSDTQ and CCSDTQ(P) in the style of PySCF's coupled-cluster
classes

    from pyscf import gto, scf
    import zccsdt
    mol = gto.M(atom='O 0 0 0; H 0 .757 .587; H 0 -.757 .587', basis='cc-pvtz')
    mf = scf.RHF(mol).run()
    mycc = zccsdt.CCSDT(mf).run()      # RCCSDT for RHF, UCCSDT for UHF / ROHF,
                                       # GCCSDT for GHF and X2C spinors
    e_q = mycc.ccsdt_q()               # (Q) correction (RCCSDT and GCCSDT)

RCCSDT   RHF reference: spin-adapted closed-shell CCSDT on spatial orbitals
         (zccsdt.sa.rccsdt), with (Q) (zccsdt.sa.rccsdt_q)
UCCSDT   UHF or ROHF reference: spin orbitals in spin blocks, integrals in U
         form (zccsdt.eris_spin, zccsdt.cc.solver)
GCCSDT   GHF or two-component (X2C) spinor reference: spin orbitals or
         spinors without spin blocks, real or complex (zccsdt.eris), with
         (Q) (zccsdt.cc.quadruples)
GCCSDTQ  the same references: CCSDTQ (zccsdt.cc.ccsdtq), with (P)
         (zccsdt.cc.pentuples); zccsdt.CCSDTQ(mf).run(), then ccsdtq_p()

The classes follow PySCF's conventions (attributes such as ``conv_tol``,
``max_cycle``, ``diis_space``, ``frozen``, ``verbose``, ``stdout``; kernel(),
run(), dump_flags(); output through pyscf.lib.logger), and print more than
PySCF's CC classes do: the input and the memory plan, a table of the
iterations, diagnostics and the largest amplitudes, and where the time and
the memory went.  What is printed at which level:

    verbose >= 3 (NOTE)    header, input, memory plan, iterations, results,
                           timings and memory (the default of PySCF objects)
    verbose >= 4 (INFO)    in addition the remaining options and the scratch
                           directory, and the parts of each timing stage
    verbose >= 5 (DEBUG)   in addition the time of each stage in every
                           iteration
'''

import os
import sys
import threading
import time
from datetime import datetime

import numpy as np
from pyscf import lib
from pyscf.lib import logger

from zccsdt import memory, timing

__version__ = '0.1.0'
WIDTH = 104

# Names of the timers of the engines: (short name, description).  The
# stages of an iteration, then their parts.
SA_STAGES = {
    'sa.dress': ('dressing', 'T1 dressing of the integral blocks'),
    'sa.imds': ('intermediates', 'intermediates'),
    'sa.Wpp': ('<vv|vv>', '<vv|vv> operator: dressed rows'),
    'sa.pp_ladder': ('pp ladder', 'pp ladder'),
    'sa.ccsd': ('CCSD', 'CCSD part'),
    'sa.t3_pass': ('pair pass', 'pair pass: ring, T3 -> T1, T2, W'),
    'sa.drive_hh': ('k pass', 'k pass: drive, hh ladder'),
    'sa.project': ('projection', 'projection of the T3 residual'),
    'sa.diis': ('DIIS', 'Jacobi step and DIIS'),
    't3p.gather': ('gather', 'gather of the T3 cubes'),
    't3p.ring_gemm': ('ring', 'ring GEMMs'),
    't3p.scatter': ('scatter', 'scatter into the residual'),
    't3p.pass1': ('T3 -> T1, T2, W', 'T3 -> T1, T2, W GEMMs'),
    'dh.gather': ('gather', 'gather of the T3 cubes'),
    'dh.K_gemm': ('drive', 'drive GEMMs'),
    'dh.H_gemm': ('hh ladder', 'hh ladder GEMMs'),
    'dh.scatter': ('scatter', 'scatter into the residual'),
    'pp.gemm': ('GEMMs', 'ladder GEMMs'),
    'pp.pack': ('pack', 'packing of the T3 blocks'),
    'pp.unpack': ('unpack', 'unpacking into the residual'),
    'diis.dot': ('overlaps', 'DIIS overlaps'),
    'diis.lincomb': ('extrapolation', 'DIIS extrapolation'),
}
CC_STAGES = {
    't1_dress': ('dressing', 'T1 dressing of the integrals'),
    'pass1.operands': ('operands', 'pass-1 operands'),
    'pass1': ('pass 1', 'pass 1: T3 -> T1, T2, W'),
    'intermediates': ('intermediates', 'intermediates, CCSD part, <ab||cd> operator'),
    'pass2': ('pass 2', 'pass 2: ring, drive'),
    'pp_ladder': ('pp ladder', 'pp ladder'),
    'hh_ladder': ('hh ladder', 'hh ladder, F_oo'),
    't3_update': ('update', 'T3 update'),
    'diis': ('DIIS', 'DIIS'),
    'pass1.gemm': ('GEMMs', 'GEMMs'),
    'pass1.gather': ('gather', 'gather of T3'),
    'pass1.acc': ('accumulate', 'accumulation of the products'),
    'pass1.r1rA': ('R1, R2', 'R1 and R2 from T3'),
    'pass2.gather': ('gather', 'gather of T3'),
    'pass2.ring': ('ring', 'ring GEMMs'),
    'pass2.drive': ('drive', 'drive GEMMs'),
    'pass2.reduce': ('reduce', 'reduction into the residual'),
    'pass2.gemm': ('GEMMs', 'ring and drive GEMMs'),
    'imd.Wvvvo': ('W_vvvo', 'W_vvvo'),
    'imd.Wvooo': ('W_vooo', 'W_vooo'),
    'imd.ring_F': ('ring operator', 'ring operator and F'),
    'imd.Wpp': ('ladder operator', 'pp ladder operator'),
    'imd.O6': ('hh operator', 'hh ladder operator'),
    'ccsd': ('CCSD', 'CCSD part'),
    'ppl.gemm': ('GEMMs', 'ladder GEMMs'),
    'ppl.gather': ('gather', 'gather of T3'),
    'ppl.reduce': ('reduce', 'reduction into the residual'),
    'dress.vvvv_pp': ('<ab||cd>', 'dressing of the <ab||cd> rows'),
    'dress.ovvv': ('ovvv', 'dressing of ovvv'),
    'diis.dot': ('overlaps', 'DIIS overlaps'),
    'diis.lincomb': ('extrapolation', 'DIIS extrapolation'),
}


Q_STAGES = dict(CC_STAGES, **{
    'ccsdtq.t3': ('CCSDT step', 'CCSDT step: the T1, T2, T3 equations without T4'),
    'ccsdtq.dress': ('dressing', 'T1 dressing of the integrals of the T4 equation'),
    'ccsdtq.t4': ('T4 equation', 'T4 equation, T4 in the T2 and T3 equations'),
    'ccsdtq.update': ('update', 'Jacobi step'),
    'ccsdtq.diis': ('DIIS', 'DIIS'),
    't4.t3pass': ('T3 operands', 'T3 with one integral'),
    't4.imds': ('intermediates', 'intermediates of T2 and T3'),
    't4.xv': ('Xv', 'six-index intermediate without T4'),
    't4.pp': ('pp ladder', 'pp ladder; T4 in Xo, R2, R3'),
    't4.ring': ('ring', 'ring; T4 in Xv, R3'),
    't4.hh': ('hh ladder', 'hh ladder'),
    't4.rows': ('by quadruple', 'products by occupied quadruple'),
})


_PARTS = {'gemm': 'GEMMs', 'gather': 'gather of T3', 'reduce': 'reduction into the residual',
          'drive': 'drive GEMMs', 'ring': 'ring GEMMs', 'acc': 'accumulation',
          'scatter': 'scatter into the residual'}


def _name(stages, key, short=False):
    n = stages.get(key)
    if n is not None:
        return n[0 if short else 1]
    head, _, part = key.partition('.')
    if head == 'dress' and part:
        return part if short else 'dressing of ' + part
    if part in _PARTS:
        return _PARTS[part]
    return key
ROOT_INTS, ROOT_ITER, ROOT_Q, ROOT_P = 'zccsdt.integrals', 'zccsdt.iterations', 'zccsdt.q', 'zccsdt.p'
ROOT_QG, ROOT_QLG = 'zccsdt.qg', 'zccsdt.qlg'


# ------------------------------------------------------------------ helpers

def _size(nbytes):
    '''1234567 -> '1.2 MB'.'''
    for unit, f in (('GB', 1e9), ('MB', 1e6), ('kB', 1e3)):
        if nbytes >= f:
            return '%.1f %s' % (nbytes / f, unit)
    return '%d B' % nbytes


def _count(n):
    return format(int(n), ',')


def _formula(mol):
    '''Molecular formula in Hill order (C, H, then alphabetical).'''
    from collections import Counter
    n = Counter(mol.atom_pure_symbol(i) for i in range(mol.natm))
    order = (['C', 'H'] if 'C' in n else []) + sorted(e for e in n if 'C' not in n or e not in 'CH')
    return ''.join('%s%s' % (e, n[e] if n[e] > 1 else '') for e in order if e in n)


def _value(x):
    '''An amplitude for printing: real as is, complex as re+im i.'''
    if np.iscomplexobj(x) and abs(np.imag(x)) > 1e-9 * max(1.0, abs(x)):
        return '%.8f%+.8fi' % (np.real(x), np.imag(x))
    return '%.8f' % np.real(x)


def _reference_name(mf):
    name = mf.__class__.__name__
    mod = mf.__class__.__module__
    if '.x2c.' in mod and 'X2C' not in name.upper():
        return 'X2C-' + name
    return name


def _revision():
    '''Short git revision of the source tree ('' if not a git checkout).'''
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        import subprocess
        rev = subprocess.run(['git', '-C', here, 'rev-parse', '--short', 'HEAD'],
                             capture_output=True, text=True, timeout=5)
        if rev.returncode != 0:
            return ''
        dirty = subprocess.run(['git', '-C', here, 'status', '--porcelain', '--untracked-files=no'],
                               capture_output=True, text=True, timeout=5)
        return rev.stdout.strip() + ('+' if dirty.stdout.strip() else '')
    except Exception:
        return ''


def _rss_anon():
    '''Resident anonymous memory of this process in bytes (None if unknown).'''
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('RssAnon'):
                    return 1024 * int(line.split()[1])
    except (OSError, ValueError):
        pass
    return None


def _need_text(plan, explicit):
    if explicit is not None:
        return 'given explicitly'
    return 'needs %s next to the integrals' % _size(plan.need)


class _PeakMemory:
    '''Peak resident anonymous memory of the process while the block runs,
    sampled every ``dt`` seconds by a background thread (the arrays of the
    program; mapped scratch files are page cache and not counted).'''

    def __init__(self, dt=0.05):
        self.dt = dt
        self.peak = _rss_anon()
        self._done = threading.Event()

    def _run(self):
        while not self._done.wait(self.dt):
            m = _rss_anon()
            if m is not None and (self.peak is None or m > self.peak):
                self.peak = m

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._done.set()
        self._thread.join()
        m = _rss_anon()
        if m is not None and (self.peak is None or m > self.peak):
            self.peak = m


class _Printer:
    '''Formatting on top of a pyscf.lib.logger.Logger: rules, key-value
    lines and tables, each at a given level.'''

    def __init__(self, log):
        self.log = log

    def _emit(self, level, msg):
        if self.log.verbose >= level:
            logger.flush(self.log, '%s', msg)

    def rule(self, level=logger.NOTE, char='-'):
        self._emit(level, char * WIDTH)

    def line(self, msg='', level=logger.NOTE):
        self._emit(level, msg)

    def kv(self, key, value, level=logger.NOTE, indent=1):
        self._emit(level, '%s%-17s %s' % (' ' * indent, key, value))


class _Given:
    '''In the place of the SCF object when the integrals were made elsewhere
    (GCCSDT.from_integrals): what the front end asks of one.'''
    converged = True
    mo_coeff = None
    mol = None

    def __init__(self, eris, verbose, stdout):
        self.eris = eris
        self.e_tot = float(eris.e_hf)
        self.mo_occ = (np.arange(eris.nmo) < eris.nocc).astype(float)
        self.verbose = verbose
        self.stdout = sys.stdout if stdout is None else stdout


# -------------------------------------------------------------- the classes

class CCSDTBase(lib.StreamObject):
    '''Common part of the CCSDT front ends.

    Attributes:
        verbose : int
            Print level.  Default value equals to :class:`Mole.verbose`
        stdout : file object
            Default value equals to :class:`Mole.stdout`
        max_memory : float or None
            MB for the whole process, as in PySCF.  Default None: 80% of
            what the node and the memory limit of the job (Slurm cgroup)
            leave.  What does not fit (T3, its residual, the <vv|vv>
            operator) goes to scratch files (ZCCSDT_TMPDIR, else PySCF's
            TMPDIR) and the passes are sliced; see docs/factorization.md and
            docs/spin_adapted.md.
        max_disk : float or None
            MB for the scratch files of the process: the disk that the run
            may use in its scratch location.  Default None: the environment
            variable ZCCSDT_MAX_DISK if set, else the free space of the
            scratch directory (statvfs) less a reserve of 2% (at most 1 GB).
            The arrays that the memory plan maps come first; the DIIS
            vectors get the rest (zccsdt.disk): all in the scratch file,
            else some in the memory that the plan leaves, else fewer than
            ``diis_space``, else without T3 - each stated in the log - and
            an error before the first iteration where the disk does not
            hold what the run has to put there.
        frozen : int or list
            Number of frozen core orbitals: spatial orbitals (RCCSDT), per
            spin (UCCSDT), spin orbitals or spinors (GCCSDT).  Default 0.
            set_frozen() freezes the chemical core.  For spin orbitals and
            spinors (GCCSDT, GCCSDTQ) also a list of indices, as in PySCF:
            occupied ones are frozen, virtual ones dropped.
        conv_tol : float
            Convergence threshold of the correlation energy (Eh).  Default 1e-10.
        conv_tol_normt : float
            Convergence threshold of the amplitude residual norm (RCCSDT,
            default 1e-7) or step norm (UCCSDT, GCCSDT, default 1e-8).
        max_cycle : int
            Maximum number of iterations.  Default 100.
        diis_space : int
            Number of DIIS vectors asked for: a maximum.  Default 8.  Fewer
            are used (never fewer than 3 with T3) where ``max_disk`` and the
            memory beyond the plan do not hold them; the log says so.
        diis_t3 : bool or None
            T3 in the DIIS vectors.  Default None: yes, unless neither the
            disk nor the memory holds three of them - then DIIS on T1 and
            T2, T3 by its Jacobi step, stated in the log.  True: yes, or an
            error.  False: DIIS on T1 and T2 only.
        diis_file : str
            HDF5 file of the DIIS vectors (default: a temporary file).
        incore_complete : bool
            Keep everything in memory, the DIIS vectors included, whatever
            max_memory (MemoryError if the node does not hold the DIIS
            vectors).  Default False.
        plan : MemoryPlan
            An explicit memory plan instead of the automatic one.
        callback : function(info) => None
            Called after every iteration with a dict (cycle, e_corr, de,
            norms, normt, diis, wall, cpu).

    Saved results:
        converged : bool
        e_hf : float
            Reference energy.
        e_corr : float
            CCSDT correlation energy.
        e_tot : float
            Total CCSDT energy.
        cycles : int
            Number of iterations performed.
        t1, t2, t3 :
            Amplitudes; t3 in the storage of the engine (may be a mapped
            scratch array when memory is short).
        plan :
            The memory plan used (``plan_used``); ``diis_plan_used``: where
            the DIIS vectors were (zccsdt.disk.DiisPlan).
        timings : list
            Wall time of each iteration.
        peak_memory : int
            Peak resident anonymous memory of the iterations in bytes.
    '''

    conv_tol = 1e-10
    conv_tol_normt = None
    max_cycle = 100
    diis_space = 8
    diis_t3 = None
    diis_file = None
    max_disk = None
    incore_complete = False
    plan = None
    callback = None
    _title = 'CCSDT'
    _subtitle = ''
    _norm_labels = ('|dT1|', '|dT2|', '|dT3|')
    _stages = CC_STAGES

    _keys = {
        'mol', 'verbose', 'stdout', 'max_memory', 'max_disk', 'frozen', 'conv_tol', 'conv_tol_normt',
        'max_cycle', 'diis_space', 'diis_t3', 'diis_file', 'incore_complete', 'plan', 'diis_plan_used',
        'callback', 'mo_coeff', 'mo_occ', 'converged', 'e_hf', 'e_corr', 'cycles',
        't1', 't2', 't3', 'plan_used', 'timings', 'peak_memory', 'e_q', 'e_q5',
    }

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        from pyscf.scf import hf
        if isinstance(mf, hf.KohnShamDFT):
            raise RuntimeError('%s: the first argument is a DFT object; CCSDT needs an HF '
                               'object (mf.to_hf())' % self.__class__.__name__)
        given = isinstance(mf, _Given)
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
            if mo_coeff is None and not given and mf.mol.nelectron > 0:
                mf.run()
                mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ
        self.mol = mf.mol
        self._scf = mf
        self.verbose = mf.verbose if given else self.mol.verbose
        self.stdout = mf.stdout if given else self.mol.stdout
        self.max_memory = None
        self.frozen = frozen
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.converged = False
        self.cycles = None
        self.e_hf = None
        self.e_corr = None
        self.t1 = self.t2 = self.t3 = None
        self.plan_used = None
        self.diis_plan_used = None
        self.timings = []
        self.peak_memory = None
        self.e_q = self.e_q5 = None
        self._eris = None
        self._t0 = None

    # ---- properties and small helpers

    @property
    def e_tot(self):
        return None if self.e_corr is None else self.e_hf + self.e_corr

    @property
    def e_tot_q(self):
        return None if self.e_q is None else self.e_tot + self.e_q

    @property
    def ncore(self):
        '''Number of frozen core orbitals (see ``frozen``).'''
        if self.frozen is None:
            return 0
        if isinstance(self.frozen, (int, np.integer)):
            return int(self.frozen)
        raise NotImplementedError('%s: frozen must be a number of core orbitals'
                                  % self.__class__.__name__)

    def set_frozen(self):
        '''Freeze the chemical core (pyscf.data.elements.chemcore).'''
        from pyscf.data import elements
        if self.mol is None:
            raise RuntimeError('%s: the integrals were given; what is frozen is decided where they are made'
                               % self.__class__.__name__)
        self.frozen = elements.chemcore(self.mol) * self._frozen_unit
        return self

    _frozen_unit = 1

    def _conv_tol_normt(self):
        return self.conv_tol_normt

    def _orbital_lines(self):
        raise NotImplementedError

    def ao2mo(self):
        raise NotImplementedError

    # ---- printing

    def _header(self, pr):
        pr.rule(char='=')
        pr.line(' zccsdt %s: %s' % (self._title, self._subtitle))
        pr.rule(char='=')
        rev = _revision()
        import pyscf
        pr.line(' zccsdt %s%s   PySCF %s   numpy %s   %d threads   %s   %s'
                % (__version__, ' (%s)' % rev if rev else '', pyscf.__version__,
                   np.__version__, lib.num_threads(), os.uname().nodename.split('.')[0],
                   datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        pr = _Printer(log)
        self._header(pr)
        pr.rule()
        mf = self._scf
        if isinstance(mf, _Given):
            pr.kv('reference', 'integrals given%s   E(SCF) = %.12f'
                  % (' (%s)' % mf.eris.source if mf.eris.source else '', mf.e_tot))
        else:
            conv = 'converged' if getattr(mf, 'converged', True) else 'NOT converged'
            pr.kv('reference', '%s   E(SCF) = %.12f   %s'
                  % (_reference_name(mf), mf.e_tot, conv))
            mol = self.mol
            basis = mol.basis if isinstance(mol.basis, str) else 'per atom'
            pr.kv('molecule', '%s   %d electrons   charge %d   2S = %d   basis %s'
                  % (_formula(mol), mol.nelectron, mol.charge, mol.spin, basis))
        for key, value in self._orbital_lines():
            pr.kv(key, value)
        normt = self._conv_tol_normt()
        pr.kv('convergence', '|dE| < %g Eh and %s < %g; at most %d iterations'
              % (self.conv_tol, self._norm_name, normt, self.max_cycle))
        pr.kv('DIIS', '%d vectors%s; %s'
              % (self.diis_space, ', T1 and T2 only (diis_t3 = False)' if self.diis_t3 is False else ', with T3',
                 'all in memory (incore_complete)' if self.incore_complete else
                 'large parts in %s' % (self.diis_file or 'a scratch file')))
        pr.kv('memory', self._budget_text())
        pr.kv('disk', self._disk_text())
        from zccsdt.scratch import scratch_dir
        pr.kv('scratch', scratch_dir(), level=logger.INFO)
        return self

    def _disk_text(self):
        from zccsdt import disk, scratch
        where = scratch.scratch_dir()
        b, free, cap = disk.budget(self.max_disk, where), scratch.disk_free(where), disk.max_disk_bytes(self.max_disk)
        if b is None:
            return 'no limit known for the scratch files'
        held = scratch.allocated()
        if cap is not None:
            return 'budget %s for scratch files (%s %d MB, %s in scratch files now; %s free in %s)' % (
                _size(b), 'max_disk' if self.max_disk is not None else 'ZCCSDT_MAX_DISK', round(cap / 1e6),
                _size(held), _size(free or 0), where)
        return 'budget %s for scratch files = the free space of %s (%s) less a reserve' % (
            _size(b), where, _size(free))

    _norm_name = '|dT|'

    def _budget_text(self):
        b = memory.budget(self.max_memory)
        if b is None:
            return 'no limit known: everything in memory'
        if self.max_memory is not None:
            return 'budget %s (max_memory %d MB, %s resident now)' % (
                _size(b), self.max_memory, _size(memory.resident()))
        avail = memory.available()
        return 'budget %s = %d%% of %s available (node and job limit)' % (
            _size(b), round(100 * memory.FRACTION), _size(avail))

    def _iteration_header(self, pr):
        pr.rule()
        a, b, c = self._norm_labels
        pr.line(' %4s  %20s  %12s  %9s  %9s  %9s  %4s  %9s  %5s'
                % ('iter', 'E_corr / Eh', 'dE / Eh', a, b, c, 'DIIS', 'time / s', 'cores'))
        pr.line(' %4s  %20s  %12s  %9s  %9s  %9s  %4s  %9s  %5s'
                % ('----', '-' * 20, '-' * 12, '-' * 9, '-' * 9, '-' * 9, '----', '-' * 9,
                   '-----'))

    def _iteration_callback(self, pr):
        state = {'snap': timing.snapshot()}

        def report(info):
            n1, n2, n3 = info['norms']
            busy = info['cpu'] / info['wall'] if info['wall'] > 0 else 0.0
            pr.line(' %4d  %20.15f  %12.4e  %9.2e  %9.2e  %9.2e  %4d  %9.2f  %5.1f'
                    % (info['cycle'], info['e_corr'], info['de'], n1, n2, n3, info['diis'],
                       info['wall'], busy))
            if pr.log.verbose >= logger.DEBUG:
                snap = timing.snapshot()
                rows = timing.children(timing.since(state['snap'], snap), (ROOT_ITER,))
                state['snap'] = snap
                parts = ['%s %.2f' % (_name(self._stages, n, True), w)
                         for n, w, c, k in rows if w >= 0.005]
                for k in range(0, len(parts), 5):
                    pr.line('%7s%s' % ('', '   '.join(parts[k:k + 5])), level=logger.DEBUG)
            if self.callback is not None:
                self.callback(info)
        return report

    def _finalize(self):
        '''Results, diagnostics, largest amplitudes, timings and memory.'''
        log = logger.new_logger(self)
        pr = _Printer(log)
        name = self.__class__.__name__
        pr.rule()
        if self.converged:
            pr.line(' %s converged in %d iterations' % (name, self.cycles))
        else:
            pr.line(' %s NOT converged in %d iterations' % (name, self.cycles))
        pr.kv('E(SCF)', '%20.12f' % self.e_hf, indent=3)
        pr.kv('E_corr(CCSDT)', '%20.12f' % self.e_corr, indent=3)
        pr.kv('E(CCSDT)', '%20.12f' % self.e_tot, indent=3)
        for key, value in self._extra_energies():
            pr.kv(key, value, indent=3)
        self._print_diagnostics(pr)
        return self

    def _extra_energies(self):
        return []

    def _print_diagnostics(self, pr):
        pass

    def _print_amplitudes(self, pr, tables):
        '''tables: [(label, [(value, text)])].'''
        tables = [(lab, rows) for lab, rows in tables if rows]
        if not tables:
            return
        pr.line(' largest amplitudes')
        for lab, rows in tables:
            for k, (val, text) in enumerate(rows):
                pr.line('   %-3s %24s   %s' % (lab if k == 0 else '', _value(val), text))

    def _print_timings(self, pr, snap0, peak, what=None):
        '''Where the time of the calculation went (since snap0).'''
        times = timing.since(snap0)
        top = dict(((n,), (w, c, k)) for n, w, c, k in timing.children(times, ()))
        total_w = sum(w for (w, c, k) in top.values())
        total_c = sum(c for (w, c, k) in top.values())
        pr.rule()
        pr.line(' %-50s %10s %10s %6s %8s' % ('timings', 'wall / s', 'CPU / s', 'cores', 'share'))
        it = top.get((ROOT_ITER,))

        def row(label, w, c, share=None, indent=1, level=logger.NOTE):
            # (times that a C kernel reports itself come without CPU time)
            cpu = '%10.2f %6.1f' % (c, c / w) if c > 0 and w > 0 else '%10s %6s' % ('-', '-')
            pr.line(' %-50s %10.2f %s %8s'
                    % ((' ' * (indent - 1)) + label, w, cpu,
                       '' if share is None else '%.1f %%' % share), level=level)

        if (ROOT_INTS,) in top:
            w, c, k = top[(ROOT_INTS,)]
            row('integrals', w, c)
        if it is not None:
            w_it, c_it, k = it
            row('%s iterations (%d)' % (self._title, len(self.timings)), w_it, c_it, 100.0)
            stages = timing.children(times, (ROOT_ITER,))
            covered = 0.0
            for n, w, c, k in stages:
                covered += w
                row(_name(self._stages, n), w, c, 100 * w / w_it if w_it else 0, indent=3)
                for n2, w2, c2, k2 in timing.children(times, (ROOT_ITER, n))[:6]:
                    if w2 >= 0.01 * w_it:
                        row(_name(self._stages, n2), w2, c2, 100 * w2 / w_it, indent=5,
                            level=logger.INFO)
            if w_it - covered > 0.0005 * w_it:
                pr.line(' %-50s %10.2f' % ('  other', w_it - covered))
        if (ROOT_Q,) in top:
            w, c, k = top[(ROOT_Q,)]
            row('(Q) correction', w, c)
        row('total', total_w, total_c)
        if self.timings:
            first = self.timings[0]
            rest = self.timings[1:] or self.timings
            pr.line(' per iteration %.2f s on average, the first %.2f s%s'
                    % (float(np.mean(rest)), first, what or ''))
        if peak is not None:
            need = getattr(self.plan_used, 'need', 0)
            pr.line(' peak memory %s anonymous (arrays; the plan counted %s next to the integrals)'
                    % (_size(peak), _size(need)))

    # ---- driver

    def kernel(self, t1=None, t2=None, t3=None):
        return self.ccsdt(t1, t2, t3)

    def ccsdt(self, t1=None, t2=None, t3=None):
        from zccsdt import disk
        with disk.limit(self.max_disk):     # (every scratch file of the run counts against max_disk)
            return self._ccsdt(t1, t2, t3)

    def _ccsdt(self, t1, t2, t3):
        log = logger.new_logger(self)
        cput0 = log.init_timer()
        if self.verbose >= logger.WARN:
            self.check_sanity()
        self.dump_flags()
        pr = _Printer(log)
        snap0 = timing.snapshot()
        with timing.timer(ROOT_INTS):
            eris = self.ao2mo()
        self._eris = eris
        for key, value in self._amplitude_lines(eris):
            pr.kv(key, value)
        plan = self.plan or self._make_plan(eris)
        self._print_plan(pr, plan, eris)
        # (before the iterations: the disk holds the plan and the DIIS vectors, or the run stops here)
        self._diis_plan = dplan = self._make_diis_plan(eris, plan)
        pr.kv('DIIS vectors', '%s; they take %s of disk and %s of memory'
              % (dplan.text(), _size(dplan.disk), _size(dplan.memory)), indent=3)
        self._iteration_header(pr)
        with _PeakMemory() as peak, timing.timer(ROOT_ITER):
            res = self._solve(eris, plan, t1, t2, t3, self._iteration_callback(pr))
        self._diis_plan = None
        self.converged, self.cycles = bool(res.converged), int(res.ncycle)
        self.e_hf, self.e_corr = float(eris.e_hf), float(np.real(res.e_corr))
        self.t1, self.t2, self.t3 = res.t1, res.t2, res.t3
        self.plan_used = res.plan
        self.diis_plan_used = getattr(res, 'diis_plan', None)
        self.timings = list(res.timings)
        self.peak_memory = peak.peak
        self._finalize()
        self._print_timings(pr, snap0, peak.peak, self._first_iteration_note())
        pr.rule(char='=')
        log.timer(self.__class__.__name__, *cput0)
        return self.e_corr, self.t1, self.t2, self.t3

    def _first_iteration_note(self):
        return ''

    def ccsdt_q(self, t2=None, t3=None):
        raise NotImplementedError('%s: the (Q) correction is implemented for closed shells '
                                  '(RCCSDT) and for spin orbitals or spinors without spin '
                                  'blocks (GCCSDT)' % self.__class__.__name__)


class RCCSDT(CCSDTBase):
    __doc__ = '''Closed-shell CCSDT on an RHF reference, fully spin-adapted (spatial
    orbitals, zccsdt.sa.rccsdt), and its (Q) correction (ccsdt_q()).

    t1[i,a] and t2[i,j,a,b] are spatial amplitudes (PySCF's RCCSD layout);
    t3 holds one v^3 cube per sorted occupied triple i <= j <= k, not all
    equal (zccsdt.sa.t3blocks.Blocks).

''' + CCSDTBase.__doc__

    _title = 'RCCSDT'
    _subtitle = 'closed-shell CCSDT, spin-adapted, on spatial orbitals'
    _norm_labels = ('|R1|', '|R2|', '|R3|')
    _norm_name = '|R|'
    _stages = SA_STAGES

    def _conv_tol_normt(self):
        return 1e-7 if self.conv_tol_normt is None else self.conv_tol_normt

    def _sizes(self):
        occ = np.asarray(self.mo_occ) > 0
        nocc = int(occ.sum()) - self.ncore
        nmo = len(occ) - self.ncore
        return nocc, nmo - nocc

    def _orbital_lines(self):
        nocc, nvir = self._sizes()
        return [('orbitals', '%d functions; %d occupied, %d virtual spatial orbitals, %d frozen'
                 % (len(self.mo_occ), nocc, nvir, self.ncore))]

    def ao2mo(self):
        from zccsdt.sa import rccsdt_ref
        mf = self._scf
        if self.mo_coeff is not mf.mo_coeff or self.mo_occ is not mf.mo_occ:
            mf = mf.copy()
            mf.mo_coeff, mf.mo_occ = self.mo_coeff, self.mo_occ
        return rccsdt_ref.from_scf(mf, ncore=self.ncore, full=False)

    def _amplitude_lines(self, ints):
        from zccsdt.sa.t3blocks import Blocks
        no, nv = ints.nocc, ints.nvir
        nb = Blocks.count(no)
        return [('amplitudes', 'T1 %s   T2 %s   T3 %s (%s: %d blocks of %d^3, i <= j <= k)'
                 % (_count(no * nv), _count(no * no * nv * nv), _count(nb * nv ** 3),
                    _size(8 * nb * nv ** 3), nb, nv)),
                ('integrals', 'blocks transformed from the AO integrals when first needed '
                 '(in the first iteration)')]

    def _make_plan(self, ints):
        from zccsdt.sa import rccsdt
        return rccsdt.make_plan(ints, self.max_memory, self.incore_complete)

    def _make_diis_plan(self, ints, plan):
        from zccsdt.sa import rccsdt
        return rccsdt.diis_plan_for(ints, plan, self.diis_space, self.diis_file, self.incore_complete,
                                    self.diis_t3, self.max_disk)

    def _print_plan(self, pr, plan, ints):
        from zccsdt.sa.rccsdt import pp_slabs
        from zccsdt.sa.t3blocks import Blocks
        no, nv = ints.nocc, ints.nvir
        t3 = 8 * Blocks.count(no) * nv ** 3
        ps, pa = nv * (nv + 1) // 2, nv * (nv - 1) // 2
        pp = 8 * (ps * ps + pa * pa)
        where = {False: 'memory', True: 'scratch file'}
        r3_file = plan.t3_file if plan.r3_file is None else plan.r3_file
        short = plan.t3_file or r3_file or plan.pp_file or plan.pp_bytes is not None \
            or not plan.keep
        pr.kv('memory plan', '%s: %s' % (_need_text(plan, self.plan),
                                         'memory is short' if short else 'everything in memory'))
        pr.kv('T3, residual', '%s, %s (%s each)'
              % (where[bool(plan.t3_file)], where[bool(r3_file)], _size(t3)), indent=3)
        bare = 'bare form in a scratch file' if plan.pp_file else 'bare form in memory'
        if plan.pp_bytes is None:
            pp_text = '%s, dressed form in memory (%s each)' % (bare, _size(pp))
        else:
            n = len(pp_slabs(nv, plan.pp_bytes))
            pp_text = '%s; dressed rows in %d slabs of %s, %s' % (
                bare, n, _size(plan.pp_bytes),
                'batch-wise (rows kept in a scratch file)' if plan.pp_order == 'batch'
                else 'slab-wise')
        pr.kv('<vv|vv>', pp_text, indent=3)
        pr.kv('buffers', 'ladder %d blocks per batch, k pass %s slabs; %s'
              % (min(plan.batch, Blocks.count(no)), _size(min(plan.slab_bytes, 8 * no * no * nv ** 3)),
                 'kept between iterations' if plan.keep else 'given back after each stage'),
              indent=3)

    def _solve(self, ints, plan, t1, t2, t3, callback):
        from zccsdt.sa import rccsdt
        if t1 is not None or t2 is not None or t3 is not None:
            raise NotImplementedError('RCCSDT: restarting from given amplitudes')
        return rccsdt.kernel(ints, conv_tol=self.conv_tol,
                             conv_tol_normt=self._conv_tol_normt(),
                             max_cycle=self.max_cycle, diis_space=self.diis_space,
                             diis_file=self.diis_file, incore_complete=self.incore_complete,
                             max_memory=self.max_memory, plan=plan, diis_t3=self.diis_t3,
                             callback=callback, max_disk=self.max_disk, diis_plan=self._diis_plan)

    def _first_iteration_note(self):
        return ' (with the integral blocks)'

    def _extra_energies(self):
        # same-spin and opposite-spin parts of the correlation energy
        ints = self._eris
        t1, t2 = self.t1, self.t2
        g = ints.block('oovv')
        tau = t2 + np.einsum('ia,jb->ijab', t1, t1)
        e_os = float(np.einsum('ijab,ijab->', g, tau))
        e_ss = float(np.einsum('ijab,ijab->', g - g.transpose(0, 1, 3, 2), tau))
        return [('E_corr(same-spin)', '%20.12f' % e_ss),
                ('E_corr(oppo-spin)', '%20.12f' % e_os)]

    def _print_diagnostics(self, pr):
        from pyscf.cc import ccsd
        t1, t2 = self.t1, self.t2
        pr.line(' diagnostics       T1 %.4f   D1 %.4f   D2 %.4f'
                % (ccsd.get_t1_diagnostic(t1), ccsd.get_d1_diagnostic(t1),
                   ccsd.get_d2_diagnostic(t2)))
        no, nv = t1.shape
        c = self.ncore

        def mo(p):          # 1-based MO number in the full orbital list
            return p + c + 1

        rows1 = [(t1[i, a], '%d -> %d' % (mo(i), mo(no + a)))
                 for i, a in _largest(t1, 5)]
        rows2 = []
        for i, j, a, b in _largest(t2, 40):
            if (i, j, a, b) > (j, i, b, a):      # t2[i,j,a,b] = t2[j,i,b,a]
                continue
            rows2.append((t2[i, j, a, b], '%d %d -> %d %d' % (mo(i), mo(j), mo(no + a), mo(no + b))))
            if len(rows2) == 5:
                break
        rows3 = _largest_t3_blocks(self.t3, no, nv, mo, 5)
        self._print_amplitudes(pr, [('T1', rows1), ('T2', rows2), ('T3', rows3)])

    def ccsdt_q(self, t2=None, t3=None):
        '''(Q) correction of the converged CCSDT amplitudes; returns E(Q)
        and stores e_q, e_q5 (its fifth-order part, [Q]) and e_tot_q.'''
        from zccsdt.sa import rccsdt_q
        from zccsdt.sa.t3blocks import Blocks
        log = logger.new_logger(self)
        pr = _Printer(log)
        t2 = self.t2 if t2 is None else t2
        t3 = self.t3 if t3 is None else t3
        ints = self._eris if self._eris is not None else self.ao2mo()
        no, nv = ints.nocc, ints.nvir
        need, sliced = rccsdt_q.memory_needed(no, nv), rccsdt_q.sliced_memory(no, nv)
        b = memory.budget(self.max_memory)
        snap0 = timing.snapshot()
        pr.line(' (Q) correction: %d unique occupied quadruples; work arrays %s in the block '
                'algorithm, %s in slabs' % (len(rccsdt_q.unique_quadruples(no)), _size(need),
                                             _size(sliced)))
        pr.kv('algorithm', 'block (everything in memory)' if b is None or need <= b else
              'slabs of the largest virtual index (budget %s)' % _size(b), indent=3)
        with _PeakMemory() as peak, timing.timer(ROOT_Q):
            self.e_q5, self.e_q = rccsdt_q.kernel(ints, Blocks(no), t2, t3,
                                                  max_memory=self.max_memory)
        w, c, k = timing.since(snap0)[(ROOT_Q,)]
        pr.kv('E[Q]', '%20.12f   (fifth order)' % self.e_q5, indent=3)
        pr.kv('E(Q)', '%20.12f' % self.e_q, indent=3)
        pr.kv('E(CCSDT(Q))', '%20.12f' % self.e_tot_q, indent=3)
        pr.kv('time', '%.2f s wall, %.2f s CPU (%.1f cores); peak memory %s anonymous'
              % (w, c, c / w if w > 0 else 0, _size(peak.peak or 0)), indent=3)
        pr.rule(char='=')
        return self.e_q


class UCCSDT(CCSDTBase):
    __doc__ = '''CCSDT on a UHF or ROHF reference: spin orbitals in spin blocks
    (alpha first), integrals in U form (zccsdt.eris_spin), the spin-blocked T3
    engine (zccsdt.cc.t3engine.SpinBlockT3).  ``frozen`` counts orbitals per
    spin, as in PySCF's UCCSD.

    t1[i,a], t2[i,j,a,b] are spin-orbital amplitudes (antisymmetric) over the
    active spin orbitals; t3 is packed by spin blocks of i<j<k, a<b<c.

''' + CCSDTBase.__doc__

    _title = 'UCCSDT'
    _subtitle = 'open-shell CCSDT on spin orbitals in spin blocks (U-form integrals)'
    _frozen_unit = 1

    def _conv_tol_normt(self):
        return 1e-8 if self.conv_tol_normt is None else self.conv_tol_normt

    def _spaces(self):
        '''Spatial indices of the active alpha / beta occupied and virtual orbitals.'''
        mo_occ = self.mo_occ
        if np.asarray(mo_occ).ndim == 1:            # RHF / ROHF
            occ_a, occ_b = np.asarray(mo_occ) > 0, np.asarray(mo_occ) > 1
        else:
            occ_a, occ_b = (np.asarray(o) > 0 for o in mo_occ)
        c = self.ncore
        return (np.where(occ_a)[0][c:], np.where(occ_b)[0][c:],
                np.where(~occ_a)[0], np.where(~occ_b)[0])

    def _orbital_lines(self):
        oa, ob, va, vb = self._spaces()
        return [('orbitals', 'occupied %d alpha + %d beta, virtual %d alpha + %d beta; '
                 '%d frozen per spin' % (len(oa), len(ob), len(va), len(vb), self.ncore))]

    def ao2mo(self):
        from zccsdt import eris_spin
        mf = self._scf
        if self.mo_coeff is not mf.mo_coeff or self.mo_occ is not mf.mo_occ:
            mf = mf.copy()
            mf.mo_coeff, mf.mo_occ = self.mo_coeff, self.mo_occ
        return eris_spin.from_scf(mf, frozen=2 * self.ncore, max_memory=self.max_memory)

    def _engine(self, eris):
        from zccsdt.cc import ccsdt_opt
        return ccsdt_opt.engine_for(eris, np.result_type(eris.fock, eris.oovv))

    def _amplitude_lines(self, eris):
        from zccsdt.scratch import is_mapped
        o, v = eris.nocc, eris.nmo - eris.nocc
        eng = self._engine(eris)
        isz = np.dtype(np.result_type(eris.fock, eris.oovv)).itemsize
        lines = [('amplitudes', 'T1 %s   T2 %s   T3 %s (%s, stored spin blocks)'
                  % (_count(o * v), _count(o * o * v * v), _count(eng.size),
                     _size(eng.size * isz)))]
        blocks = {id(b): b for b in eris.vvvv_blocks}
        pp = sum(b.nbytes for b in blocks.values())
        on_file = all(is_mapped(b) or b.size == 0 for b in blocks.values())
        lines.append(('integrals', 'spin blocks; packed <ab||cd> %s in %s'
                      % (_size(pp), 'a scratch file' if on_file else 'memory')))
        return lines

    def _options(self, plan, callback):
        from zccsdt.cc import solver
        return solver.Options(conv_tol=self.conv_tol, conv_tol_normt=self._conv_tol_normt(),
                              max_cycle=self.max_cycle, diis_space=self.diis_space,
                              diis_file=self.diis_file, incore_complete=self.incore_complete,
                              max_memory=self.max_memory, plan=plan, callback=callback,
                              spin_flip=getattr(self, 'spin_flip', False), max_disk=self.max_disk,
                              diis_t3=self.diis_t3, diis_plan=getattr(self, '_diis_plan', None))

    def _make_plan(self, eris):
        from zccsdt.cc import ccsdt_opt
        return ccsdt_opt.plan_for(eris, np.result_type(eris.fock, eris.oovv),
                                  self._options(None, None))

    def _make_diis_plan(self, eris, plan):
        from zccsdt.cc import solver
        self._diis_plan = None
        return solver.diis_plan_for(eris, np.result_type(eris.fock, eris.oovv), self._options(plan, None), plan)

    def _print_plan(self, pr, plan, eris):
        from zccsdt.cc import ccsdt_opt, t3pack
        dtype = np.result_type(eris.fock, eris.oovv)
        isz = np.dtype(dtype).itemsize
        eng = self._engine(eris)
        t3 = eng.size * isz
        where = {False: 'memory', True: 'scratch file'}
        r3_file = plan.t3_file if plan.r3_file is None else plan.r3_file
        short = plan.t3_file or r3_file or plan.pp_file or plan.pp_bytes is not None
        pr.kv('memory plan', '%s: %s' % (_need_text(plan, self.plan),
                                         'memory is short' if short else 'everything in memory'))
        pr.kv('T3, residual', '%s, %s (%s each)'
              % (where[bool(plan.t3_file)], where[bool(r3_file)], _size(t3)), indent=3)
        bare = 'bare form in a scratch file' if plan.pp_file else 'bare form in memory'
        if plan.pp_bytes is None:
            pp_text = '%s, dressed form in memory' % bare
        else:
            om, vm = t3pack.maps_for(eris)
            if ccsdt_opt._is_spin_eris(eris):
                n = len(ccsdt_opt._spin_slabs(eris, vm, isz, plan.pp_bytes)[0])
            else:
                n = len(ccsdt_opt.pair_slabs(vm.npair, vm.npair * isz, plan.pp_bytes))
            pp_text = '%s; dressed rows in %d slabs of %s, %s' % (
                bare, n, _size(plan.pp_bytes),
                'batch-wise (rows kept in a scratch file)' if plan.pp_order == 'batch'
                else 'slab-wise')
        pr.kv('<ab||cd>', pp_text, indent=3)
        mem = plan.mem_mb if plan.mem_mb is not None else 2000.0
        pr.kv('buffers', 'batch buffers up to %s (mem_mb %.0f)%s'
              % (_size(eng.buffer_bytes(mem)), mem,
                 '; one work array kept between the passes' if eng.kept_buffers else ''),
              indent=3)

    def _solve(self, eris, plan, t1, t2, t3, callback):
        from zccsdt.cc import solver
        return solver.kernel(eris, 'ccsdt', self._options(plan, callback), t1=t1, t2=t2, t3=t3)

    def _labels(self, eris):
        oa, ob, va, vb = self._spaces()
        occ = ['%da' % (i + 1) for i in oa] + ['%db' % (i + 1) for i in ob]
        vir = ['%da' % (a + 1) for a in va] + ['%db' % (a + 1) for a in vb]
        return occ, vir

    def _print_diagnostics(self, pr):
        t1, t2 = self.t1, self.t2
        o = t1.shape[0]
        pr.line(' diagnostics       T1 %.4f   (|t1| / sqrt(number of correlated electrons))'
                % (np.linalg.norm(t1) / np.sqrt(max(o, 1))))
        occ, vir = self._labels(self._eris)
        rows1 = [(t1[i, a], '%s -> %s' % (occ[i], vir[a])) for i, a in _largest(t1, 5)]
        rows2 = []
        for i, j, a, b in _largest(t2, 40):
            if i < j and a < b:
                rows2.append((t2[i, j, a, b], '%s %s -> %s %s' % (occ[i], occ[j], vir[a], vir[b])))
            if len(rows2) == 5:
                break
        rows3 = _largest_t3_packed(self.t3, self._eris, occ, vir, 5)
        self._print_amplitudes(pr, [('T1', rows1), ('T2', rows2), ('T3', rows3)])


class GCCSDT(UCCSDT):
    __doc__ = '''CCSDT on a GHF or two-component spinor (X2C) reference: spin
    orbitals or spinors without spin blocks, real or complex
    (zccsdt.eris, the dense T3 engine), and its (Q) correction (ccsdt_q()).
    ``frozen`` counts spin orbitals or spinors, as in PySCF's GCCSD.

    real_spinors : bool
        For a linear molecule on the z axis the spinors of the reference are
        adapted to m_j, each degenerate level by itself; all integrals and
        amplitudes are real then and the arithmetic is three times faster
        (zccsdt.eris.axial_spinors).  Default True; it applies where the
        reference has such spinors - a closed shell, or an open shell whose
        SCF kept the symmetry of the axis.  False: the spinors as they are.

''' + CCSDTBase.__doc__

    _title = 'GCCSDT'
    _subtitle = 'CCSDT on spin orbitals or spinors (GHF, X2C), real or complex'
    _frozen_unit = 2
    real_spinors = True
    symmetry = True         # use the symmetry labels of the spinors where the integrals have them
    _block_iterations = True    # ... in the iterations too: T3 in blocks, the spinors in the order of the labels
    # Lambda and the densities run in the blocks of the labels where the packed block
    # [C(o,3), C(v,3)] has at least this many elements; below it the many small products of
    # the blocks are slower than the few large ones of the packed block (docs/lambda.md 10.5)
    blocks_min = 800_000
    # CCSDT(Q)_Lambda properties (zccsdt.cc.zeta_opt): the stencil of the iterations of zeta
    # ('forward': one amplitude step each, then two, and the exact formula at the end;
    # 'central': two and the exact formula; 'exact': four) and of the derivatives along zeta
    # that are taken once (the Lambda residuals, the densities)
    zeta_stencil = 'forward'
    response_stencil = 'exact'
    _keys = UCCSDT._keys | {'real_spinors', 'symmetry', 'l1', 'l2', 'l3', 'converged_lambda', 'lambda_blocks', 'rdm1_basis', 'q_estimate', 'q_progress', 'rdm1_blocks', 'blocks_min',
                              'e_ql', 'e_ql5', 'e_tot_ql', 'zeta1', 'zeta2', 'zeta3', 'l1_ql', 'l2_ql', 'l3_ql',
                              'converged_zeta', 'converged_lambda_ql', 'ql_gradient_sectors', 'q_gradient_sectors',
                              'zeta_stencil', 'response_stencil', 'ql_info', 'eom', 'e_ee', 'eom_ip_result', 'eom_ea_result'}

    def _axial(self):
        '''The spinors of a linear molecule on the z axis adapted to m_j
        (zccsdt.eris.axial_spinors), with which all integrals are real - if
        ``real_spinors`` is set, the reference has them and what is frozen
        and dropped does not cut a degenerate level; else None.'''
        key = (id(self._scf), id(self.mo_coeff), id(self.mo_occ), str(self.frozen), bool(self.real_spinors))
        if getattr(self, '_axial_cache', (None,))[0] != key:
            out = None
            mf = self._scf
            if (self.real_spinors and not isinstance(mf, _Given) and self._kind() == 'spinor'
                    and self.mo_coeff is mf.mo_coeff and self.mo_occ is mf.mo_occ):
                from zccsdt import eris
                ax = eris.axial_spinors(mf, with_mj=True)
                if ax is not None:
                    part = np.zeros(len(ax[1]), dtype=np.int64)
                    for k, idx in enumerate(self._active()):
                        part[idx] = k
                    first = np.zeros(ax[1].max() + 1, dtype=np.int64)
                    first[ax[1]] = part
                    if (first[ax[1]] == part).all():
                        out = ax[0]
                        self._axial_mj = ax[2]
            self._axial_cache = (key, out)
        return self._axial_cache[1]

    def _active(self):
        '''(frozen occupied, active occupied, active virtual, dropped
        virtual) spin orbitals or spinors, as indices into mo_coeff.
        ``frozen``: a number - the lowest occupied ones - or, as in PySCF, a
        list of indices: occupied ones are frozen, virtual ones dropped.'''
        mo_occ = np.asarray(self.mo_occ)
        occ, vir = np.where(mo_occ > 0)[0], np.where(mo_occ == 0)[0]
        fr = self.frozen
        if fr is None or isinstance(fr, (int, np.integer)):
            n = int(fr or 0)
            return occ[:n], occ[n:], vir, vir[:0]
        fr = np.unique(np.asarray(fr, dtype=np.int64))
        if len(fr) and (fr[0] < 0 or fr[-1] >= len(mo_occ)):
            raise ValueError('%s: frozen = %s for %d orbitals' % (self.__class__.__name__, fr, len(mo_occ)))
        return (np.intersect1d(occ, fr), np.setdiff1d(occ, fr), np.setdiff1d(vir, fr), np.intersect1d(vir, fr))

    @property
    def ncore(self):
        '''Number of frozen occupied spin orbitals or spinors.'''
        return len(self._active()[0])

    def _orbital_lines(self):
        fo, o, v, fv = self._active()
        kind = 'spinors' if self._kind() == 'spinor' else 'spin orbitals'
        lines = [('orbitals', '%d occupied, %d virtual %s; %d frozen%s'
                  % (len(o), len(v), kind, len(fo), ', %d virtual ones dropped' % len(fv) if len(fv) else ''))]
        if self._axial() is not None:
            lines.append(('', 'a linear molecule on the z axis: the spinors adapted to m_j, real integrals '
                          'and amplitudes'))
        return lines

    @classmethod
    def from_integrals(cls, eris, verbose=logger.NOTE, stdout=None):
        '''The same calculation on integrals made elsewhere: an integral
        container (zccsdt.eris.ERIs) of the active spinors, occupied first,
        with <ab||cd> on pairs - zccsdt.dirac.from_dirac() for the files of
        DIRAC, zccsdt.dirac.container().  There is no SCF object and no
        molecule then, and nothing to freeze.'''
        return cls(_Given(eris, verbose, stdout))

    def _kind(self):
        from zccsdt.eris import _orbital_kind
        return 'spinor' if isinstance(self._scf, _Given) else _orbital_kind(self._scf)

    def ao2mo(self):
        from zccsdt import eris
        fo, o, v, fv = self._active()
        if isinstance(self._scf, _Given):
            if len(fo) or len(fv):
                raise RuntimeError('%s: the integrals were given; nothing can be frozen here'
                                   % self.__class__.__name__)
            return self._scf.eris
        # (the columns: frozen occupied first, the dropped virtual ones left out)
        cols = np.concatenate((fo, o, v))
        real = self._axial()
        if real is not None and self.symmetry and self._block_iterations:
            # the spinors of one label next to each other (by energy inside a label): the
            # blocks of the amplitudes are ranges then
            two_m = np.round(2 * self._axial_mj).astype(np.int64)
            o, v = (x[np.argsort(two_m[x], kind='stable')] for x in (np.asarray(o), np.asarray(v)))
            cols = np.concatenate((fo, o, v))
            self._sorted = (o, v)
        out = eris.from_scf(self._scf, mo_coeff=(np.asarray(self.mo_coeff) if real is None else real)[:, cols],
                            mo_occ=np.asarray(self.mo_occ)[cols], frozen=len(fo), packed_vvvv=True,
                            max_memory=self.max_memory, real=real is not None)
        if real is not None:
            # the symmetry label of the axis: 2 m_j, which adds without a modulus
            out.labels = np.round(2 * self._axial_mj[cols[len(fo):]]).astype(np.int64)
            out.use_labels = bool(self.symmetry and self._block_iterations)
        return out

    def _amplitude_lines(self, eris):
        from zccsdt.scratch import is_mapped
        o, v = eris.nocc, eris.nmo - eris.nocc
        eng = self._engine(eris)
        isz = np.dtype(np.result_type(eris.fock, eris.oovv)).itemsize
        vpp = getattr(eris, '_vvvv_pp', None)
        dtype = np.result_type(eris.fock, eris.oovv)
        return [('amplitudes', 'T1 %s   T2 %s   T3 %s (%s, i<j<k, a<b<c), %s'
                 % (_count(o * v), _count(o * o * v * v), _count(eng.size),
                    _size(eng.size * isz), 'complex' if np.iscomplexobj(np.zeros(0, dtype))
                    else 'real')),
                ('integrals', 'packed <ab||cd> %s in %s'
                 % (_size(vpp.nbytes) if vpp is not None else '-',
                    'a scratch file' if is_mapped(vpp) else 'memory'))]

    def _labels(self, eris):
        _, occ, vir, _ = self._active()
        if getattr(eris, 'labels', None) is not None and getattr(self, '_sorted', None) is not None:
            occ, vir = self._sorted                 # (in the order of their labels)
        num = getattr(eris, 'spinors', None)        # (their numbers where the integrals were made)
        if num is not None:
            occ, vir = num[:eris.nocc], num[eris.nocc:]
        return ['%d' % (i + 1) for i in occ], ['%d' % (a + 1) for a in vir]

    # ---- Lambda and one-electron properties (docs/lambda.md)

    def _blocks_pay(self, eris):
        '''Lambda and the densities in the blocks of the symmetry labels, or
        on the packed block?  In blocks from ``blocks_min`` elements of the
        packed block on (measured twice: HBr, N2, HF, 8 to 10 occupied
        spinors - a Lambda iteration in blocks takes 3 times as long as one
        on the packed block with 0.18 million elements and 1.2 to 1.5 times
        with 0.55; 0.9 of it with 0.85 million, 0.6 to 0.9 with 1.2 to 1.4
        and a third with 4.3 million).'''
        from math import comb
        return comb(eris.nocc, 3) * comb(eris.nmo - eris.nocc, 3) >= self.blocks_min

    def _dense_view(self):
        '''(eris, t3) for the codes that take T3 on the packed block in
        lexicographic order: the integrals without their labels in use, the
        blocks of T3 put in their places.'''
        import copy
        from zccsdt.cc import ccsdt_opt
        eris = self._eris if self._eris is not None else self.ao2mo()
        t3 = self.t3
        if getattr(eris, 'use_labels', False):
            t3 = ccsdt_opt.packed_t3(eris, t3)
            eris = copy.copy(eris)
            eris.use_labels = False
            eris.__dict__.pop('_buffers', None)
        return eris, t3

    def _packed_l3(self):
        '''self.l3 on the packed block in lexicographic order, for the codes
        that take it there: self.l3 has the layout of self.t3 - the blocks
        of the symmetry labels if the iterations ran in them -, and its
        blocks are put in their places once (kept until self.l3 is another
        array).'''
        from math import comb
        from zccsdt.cc import lambda_opt
        eris = self._eris if self._eris is not None else self.ao2mo()
        l3 = self.l3
        if not getattr(eris, 'use_labels', False):
            return l3
        no, nv = eris.nocc, eris.nmo - eris.nocc
        if np.size(l3) == comb(no, 3) * comb(nv, 3) != np.size(self.t3):
            return l3               # (given on the packed block)
        kept = getattr(self, '_l3_packed', None)
        if kept is None or kept[0] is not l3:
            kept = self._l3_packed = (l3, lambda_opt.packed_l3(eris, l3).ravel())
        return kept[1]

    def _q_terms(self):
        '''The first derivatives of (Q) for the amplitudes of this object
        (zccsdt.cc.quadruples_grad.Gradient: the sources of the Lambda
        equations of CCSDT(Q) and the densities of (Q)), kept until self.t3
        is another array; None with fewer than four occupied or virtual
        spinors, where (Q) vanishes.  One pass over the occupied quadruples,
        about three to four times the work of ccsdt_q(): in the sectors of
        the symmetry labels where the integrals have them and the blocks
        fit (zccsdt.cc.quadruples_grad_sym; ``q_gradient_sectors`` says
        so), else on the packed block, where the derivatives of the sizes
        of T3 and <vv||vv> go to a scratch file if memory is short.  It sets
        e_q as well.'''
        from zccsdt.cc import ccsdt_opt, quadruples_grad, quadruples_grad_sym
        kept = getattr(self, '_q_gradient', None)
        if kept is not None and kept[0] is self.t3:
            return kept[1]
        self._q_gradient = None                 # (the old one goes before the new one comes)
        eris = self._eris if self._eris is not None else self.ao2mo()
        no, nv = eris.nocc, eris.nmo - eris.nocc
        q = None
        if no >= 4 and nv >= 4:
            from math import comb
            t3 = self.t3
            if np.size(t3) != comb(no, 3) * comb(nv, 3):
                t3 = ccsdt_opt.packed_t3(eris, t3)  # (the blocks of the labels, as the pass reads them)
            ccsdt_opt.release_cached(eris)
            isz = np.dtype(np.result_type(self.t2, t3, eris.oovv)).itemsize
            budget = memory.budget(self.max_memory)
            spl = None
            if self.symmetry and quadruples_grad_sym.available(eris):
                try:
                    spl = quadruples_grad_sym.plan(eris, isz, budget)
                except MemoryError:     # (the dense code can do with less: batches, scratch)
                    pass
            self.q_gradient_sectors = spl is not None
            if spl is not None:
                with timing.timer(ROOT_QG):
                    q = quadruples_grad_sym.kernel(eris, self.t2, t3, plan_=spl)
            else:
                pl = quadruples_grad.plan(no, nv, isz, budget)
                with timing.timer(ROOT_QG):
                    q = quadruples_grad.kernel(eris, self.t2, t3, plan_=pl)
            self.e_q = q.e_q
        self._q_gradient = (self.t3, q)
        return q

    @staticmethod
    def _q_mode(with_q):
        '''``with_q`` of solve_lambda and make_rdm1: False (CCSDT), True
        (CCSDT(Q)) or 'lambda' (CCSDT(Q)_Lambda).'''
        if isinstance(with_q, str):
            if with_q.lower() != 'lambda':
                raise ValueError("with_q: False, True (CCSDT(Q)) or 'lambda' (CCSDT(Q)_Lambda), not %r" % with_q)
            return 'lambda'
        return bool(with_q)

    def _on_packed_block(self, x):
        '''An array in the layout of self.t3 on the packed block in
        lexicographic order (the blocks of the symmetry labels put in their
        places; as it is where self.t3 is on the packed block).'''
        from math import comb
        from zccsdt.cc import ccsdt_opt
        eris = self._eris if self._eris is not None else self.ao2mo()
        no, nv = eris.nocc, eris.nmo - eris.nocc
        if not getattr(eris, 'use_labels', False) or np.size(x) == comb(no, 3) * comb(nv, 3) != np.size(self.t3):
            return x
        return ccsdt_opt.packed_t3(eris, x).ravel()

    def _ql_terms(self):
        '''The first derivatives of (Q)_Lambda for the amplitudes and the
        lambda of CCSDT of this object
        (zccsdt.cc.quadruples_lambda_grad.Gradient: the sources of the two
        linear equations of CCSDT(Q)_Lambda and the densities of the
        correction), kept until self.t3 or self.l3 is another array; None
        with fewer than four occupied or virtual spinors.  Lambda of CCSDT
        is solved for if it is not there.  One pass over the occupied
        quadruples, in the sectors of the symmetry labels where the
        integrals have them and the blocks fit
        (zccsdt.cc.quadruples_lambda_grad_sym; ``ql_gradient_sectors`` says
        so), else on the packed block, where the derivatives of the sizes
        of T3 and <vv||vv> go to a scratch file if memory is short.  It sets
        e_ql and e_tot_ql as well.'''
        from zccsdt.cc import ccsdt_opt, quadruples_lambda_grad, quadruples_lambda_grad_sym
        if getattr(self, 'l3', None) is None or getattr(self, '_l_with_q', False):
            self.solve_lambda()
        kept = getattr(self, '_ql_gradient', None)
        if kept is not None and kept[0] is self.t3 and kept[1] is self.l3:
            return kept[2]
        self._ql_gradient = self._ql_solution = None    # (the old ones go before the new one comes)
        eris = self._eris if self._eris is not None else self.ao2mo()
        no, nv = eris.nocc, eris.nmo - eris.nocc
        q = None
        self.ql_info = {}
        if no >= 4 and nv >= 4:
            from math import comb
            t3, l3 = self.t3, self.l3
            if np.size(t3) != comb(no, 3) * comb(nv, 3):
                t3 = ccsdt_opt.packed_t3(eris, t3)  # (the blocks of the labels, as the pass reads them)
                l3 = self._packed_l3()
            ccsdt_opt.release_cached(eris)
            isz = np.dtype(np.result_type(self.t2, t3, self.l2, l3, eris.oovv)).itemsize
            budget = memory.budget(self.max_memory)
            spl = None
            if self.symmetry and quadruples_lambda_grad_sym.available(eris):
                try:
                    spl = quadruples_lambda_grad_sym.plan(eris, isz, budget)
                except MemoryError:     # (the dense code can do with less: batches, scratch)
                    pass
            self.ql_gradient_sectors = spl is not None
            t0 = time.perf_counter()
            if spl is not None:
                with timing.timer(ROOT_QLG):
                    q = quadruples_lambda_grad_sym.kernel(eris, self.t2, t3, self.l2, l3, plan_=spl)
            else:
                pl = quadruples_lambda_grad.plan(no, nv, isz, budget)
                with timing.timer(ROOT_QLG):
                    q = quadruples_lambda_grad.kernel(eris, self.t2, t3, self.l2, l3, plan_=pl)
            self.ql_info['pass'] = dict(wall=time.perf_counter() - t0, sectors=spl is not None, need=q.plan.need)
            self.e_ql = q.e_q
            self.e_tot_ql = self.e_tot + self.e_ql
        self._ql_gradient = (self.t3, self.l3, q)
        return q

    def _solve_ql(self):
        '''zeta and lambda' of CCSDT(Q)_Lambda (solve_lambda(with_q='lambda')),
        kept until self.t3 or self.l3 is another array: (l1', l2', l3').'''
        from zccsdt.cc import ccsdt_opt, lambda_opt, solver, zeta_opt
        q = self._ql_terms()
        kept = getattr(self, '_ql_solution', None)
        if kept is not None and kept[0] is self.t3 and kept[1] is self.l3:
            return self.l1_ql, self.l2_ql, self.l3_ql
        self._ql_solution = None
        eris = self._eris if self._eris is not None else self.ao2mo()
        if q is None:                   # (no quadruples: CCSDT)
            self.zeta1, self.zeta2, self.zeta3 = (np.zeros_like(x) for x in (self.t1, self.t2, self.t3))
            self.l1_ql, self.l2_ql, self.l3_ql = self.l1, self.l2, self.l3
            self.converged_zeta, self.converged_lambda_ql = True, bool(self.converged_lambda)
            self._ql_solution = (self.t3, self.l3)
            return self.l1_ql, self.l2_ql, self.l3_ql
        opts = solver.Options(
            conv_tol_normt=self._conv_tol_normt(), max_cycle=self.max_cycle, diis_space=self.diis_space,
            incore_complete=self.incore_complete, max_memory=self.max_memory)
        # ---- zeta: the Jacobian of the amplitude equations, T3 as the iterations had it
        t0 = time.perf_counter()
        rz = zeta_opt.kernel(eris, self.t1, self.t2, self.t3, q.q2, q.q3, opts, stencil=self.zeta_stencil)
        if not rz.converged:
            logger.warn(self, 'the equation of zeta of CCSDT(Q)_Lambda did not converge in %d cycles', rz.ncycle)
        self.ql_info['zeta'] = dict(wall=time.perf_counter() - t0, cycles=rz.ncycle, steps=rz.steps,
                                    stencil_error=rz.stencil_error, forward=rz.forward, mapped=rz.plan.mapped,
                                    need=rz.plan.need,
                                    timings=rz.timings)
        ccsdt_opt.release_cached(eris)          # (what the amplitude steps kept)
        # ---- lambda': the Lambda equations with sources, in the blocks of the labels or on
        # the packed block as solve_lambda()
        labels = bool(getattr(eris, 'use_labels', False))
        dtype = np.result_type(self.t1, self.t2, self.t3, eris.oovv, eris.fock)
        blocks = False
        if (self.symmetry and labels and self._blocks_pay(eris) and np.size(self.l3) == np.size(self.t3)
                and lambda_opt.block_engine(eris, dtype) is not None):
            try:
                zeta_opt.lambda_plan(eris, self.t1, dtype, opts, blocks=True)
                blocks = True
            except MemoryError:         # (the packed block then, which says what it needs)
                pass
        if blocks:
            args, l3, z3 = (eris, self.t1, self.t2, self.t3), self.l3, rz.z3
        else:
            dense, t3 = self._dense_view()
            args, l3, z3 = (dense, self.t1, self.t2, t3), self._packed_l3(), self._on_packed_block(rz.z3)
        t0 = time.perf_counter()
        src = zeta_opt.lambda_sources(*args, self.l1, self.l2, l3, (rz.z1, rz.z2, z3), q.s2, q.s3, opts,
                                      blocks=blocks, stencil=self.response_stencil)
        t1 = time.perf_counter()
        res = lambda_opt.kernel(*args, opts, l1=self.l1, l2=self.l2, l3=l3, blocks=blocks, sources=src)
        if not res.converged:
            logger.warn(self, "the Lambda equations of CCSDT(Q)_Lambda (lambda') did not converge in %d cycles",
                        res.ncycle)
        self.ql_info['sources'] = dict(wall=t1 - t0, blocks=blocks)
        self.ql_info['lambda'] = dict(wall=time.perf_counter() - t1, cycles=res.ncycle, setup=res.setup,
                                      timings=res.timings, blocks=blocks, need=res.plan.need)
        del src
        self.zeta1, self.zeta2, self.zeta3 = rz.z1, rz.z2, rz.z3
        self.l1_ql, self.l2_ql = res.l1, res.l2
        self.l3_ql = lambda_opt.blocked_l3(eris, res.l3) if labels and not blocks else res.l3
        self.converged_zeta, self.converged_lambda_ql = bool(rz.converged), bool(res.converged)
        self._ql_solution = (self.t3, self.l3)
        return self.l1_ql, self.l2_ql, self.l3_ql

    def solve_lambda(self, with_q=False):
        '''The Lambda equations of CCSDT for the converged amplitudes
        (zccsdt.cc.lambda_opt), kept as self.l1, self.l2, self.l3: l1 and l2
        dense, l3 in the layout of self.t3 - the packed block [i<j<k, a<b<c]
        or, where the iterations ran in the blocks of the symmetry labels,
        these blocks (zccsdt.cc.lambda_opt.packed_l3 puts them on the packed
        block).  With ``symmetry`` the equations are solved in the blocks
        too (zccsdt.cc.lambda_blk); on the packed block otherwise, if the
        blocks do not fit in ``max_memory``, or if the packed block has
        fewer than ``blocks_min`` elements (it is the faster one then).

        ``with_q``: lambda of CCSDT(Q) - the equations with the derivatives
        of E(Q) with respect to t2 and t3 as constant terms (_q_terms;
        docs/lambda.md 11), for the first derivatives of the CCSDT(Q)
        energy: make_rdm1(with_q=True).  It is not the lambda of
        ccsdt_q_lambda(), which wants that of CCSDT.

        ``with_q='lambda'``: the two linear equations that the first
        derivatives of the CCSDT(Q)_Lambda energy need besides the Lambda
        equations of CCSDT (docs/lambda.md 12; zccsdt.cc.zeta_opt), for
        make_rdm1(with_q='lambda').  self.l1, self.l2, self.l3 are and
        stay lambda of CCSDT - what ccsdt_q_lambda() takes -, solved for if
        they are not there.  Then: the pass over the quadruples (_ql_terms);
        zeta, the response of the CCSDT amplitudes to the quadruples - the
        Jacobian of the amplitude equations on it, self.zeta1, self.zeta2,
        self.zeta3 in the layouts of t1, t2, t3 -; and lambda', the Lambda
        equations with the derivatives of e and the term of zeta as
        constant terms: self.l1_ql, self.l2_ql, self.l3_ql in the layouts
        of l1, l2, l3, which are returned.  ``zeta_stencil`` and
        ``response_stencil`` choose how the derivatives along zeta are
        taken (zccsdt.cc.zeta_opt.points); ``ql_info`` has cycles, amplitude
        steps and times.'''
        from zccsdt.cc import lambda_opt, solver
        if self.t3 is None:
            raise RuntimeError('%s.solve_lambda(): no amplitudes (kernel() first)' % self.__class__.__name__)
        with_q = self._q_mode(with_q)
        if with_q == 'lambda':
            return self._solve_ql()
        eris = self._eris if self._eris is not None else self.ao2mo()
        opts = solver.Options(
            conv_tol_normt=self._conv_tol_normt(), max_cycle=self.max_cycle, diis_space=self.diis_space,
            incore_complete=self.incore_complete, max_memory=self.max_memory, max_disk=self.max_disk,
            diis_t3=self.diis_t3)
        labels = bool(getattr(eris, 'use_labels', False))
        q = self._q_terms() if with_q else None
        sources = None if q is None else (None, q.s2, q.s3)
        plan = None
        if self.symmetry and labels and self._blocks_pay(eris):
            dtype = np.result_type(self.t1, self.t2, self.t3, eris.oovv, eris.fock)
            if lambda_opt.block_engine(eris, dtype) is not None:
                try:
                    plan = lambda_opt.plan_for(eris, self.t1, dtype, opts, blocks=True, sources=sources)
                except MemoryError:     # (the packed block then, which says what it needs)
                    pass
        if plan is not None:
            opts.plan = plan
            res = lambda_opt.kernel(eris, self.t1, self.t2, self.t3, opts, blocks=True, sources=sources)
            l3 = res.l3
        else:
            dense, t3 = self._dense_view()
            res = lambda_opt.kernel(dense, self.t1, self.t2, t3, opts, sources=sources)
            # (in the layout of self.t3)
            l3 = lambda_opt.blocked_l3(eris, res.l3) if labels else res.l3
        if res.diis_plan is not None and res.diis_plan.reduced:
            logger.note(self, ' Lambda equations of CCSDT: DIIS %s', res.diis_plan.text())
        if not res.converged:
            logger.warn(self, 'the Lambda equations of CCSDT did not converge in %d cycles', res.ncycle)
        self.l1, self.l2, self.l3, self.converged_lambda = res.l1, res.l2, l3, bool(res.converged)
        self.lambda_blocks = bool(res.blocks)
        self._l_with_q = bool(with_q)
        return self.l1, self.l2, self.l3

    def make_rdm1(self, relaxed=True, ao_repr=False, tol=1e-11, with_q=False, zvector_solver=None):
        '''The one-particle density of CCSDT, D[p,q] = <q+ p>, in the basis
        of all spinors of the SCF in the order [frozen occupied, active
        occupied, active virtual, dropped virtual] (``ao_repr``: in the AO
        basis): the relaxed one - with the response of the spinors, from the
        orbital gradient and the Z-vector equations of the SCF
        (zccsdt.cc.density_opt, zccsdt.zccsd.zvector) -, or with
        ``relaxed=False`` the one of t and lambda at fixed spinors.  Lambda
        is solved for if it is not there.  The trace with a one-electron
        operator is the derivative of the CCSDT energy.  With ``symmetry``
        and T3 and l3 in the blocks of the symmetry labels, the passes over
        them run in these blocks (zccsdt.cc.density_blk; ``rdm1_blocks``
        says so), on the packed block otherwise, if the blocks do not fit
        in ``max_memory``, or if the packed block has fewer than
        ``blocks_min`` elements.

        ``with_q``: the density of CCSDT(Q) - the trace of the relaxed one
        with a one-electron operator is the derivative of E(CCSDT) + E(Q),
        the spinors following the perturbation as canonical Hartree-Fock
        spinors.  Lambda is that of solve_lambda(with_q=True), solved for
        if the one that is there is not; the density has the derivative of
        E(Q) with respect to the orbital energies on its diagonal, and the
        relaxed one the rotations among the occupied and among the virtual
        spinors besides (zccsdt.cc.quadruples_grad; docs/lambda.md 11).
        The unrelaxed density of CCSDT(Q) is not the derivative of anything
        at fixed spinors: E(Q) needs canonical ones.

        ``with_q='lambda'``: the density of CCSDT(Q)_Lambda - the trace of
        the relaxed one with a one-electron operator is the derivative of
        E(CCSDT) + E(Q)_L.  zeta and lambda' are those of
        solve_lambda(with_q='lambda'), solved for if they are not there;
        the densities of (t, lambda') and the derivative of those of (t,
        lambda) along zeta (``response_stencil``: four or two more passes
        of the density code), and the terms of the correction as for
        CCSDT(Q) (zccsdt.cc.quadruples_lambda_grad; docs/lambda.md 12).

        ``zvector_solver``: another solver of the Z-vector equations
        (density_opt.relaxed_rdm1, ``solver``; zccsdt.cfour.handover).'''
        from zccsdt.cc import density_blk, density_opt
        if isinstance(self._scf, _Given):
            raise RuntimeError('the integrals were given: no SCF to take the response of the spinors from')
        with_q = self._q_mode(with_q)
        kw_q, zeta = {}, None
        if with_q == 'lambda':
            q = self._ql_terms()
            l1, l2, l3 = self._solve_ql()
            if q is not None:
                zeta = (self.zeta1, self.zeta2, self.zeta3)
                kw_q = dict(q=q, zeta=zeta, lams=(self.l1, self.l2, self.l3), stencil=self.response_stencil)
        else:
            if getattr(self, 'l3', None) is None or bool(getattr(self, '_l_with_q', False)) != with_q:
                self.solve_lambda(with_q=with_q)
            q = self._q_terms() if with_q else None
            if q is not None:
                kw_q = dict(q=q)
            l1, l2, l3 = self.l1, self.l2, self.l3
        eris = self._eris if self._eris is not None else self.ao2mo()
        fo, o, v, fv = self._active()
        if getattr(eris, 'labels', None) is not None and getattr(self, '_sorted', None) is not None:
            o, v = self._sorted
        axial = self._axial()
        Cs = np.asarray(self.mo_coeff) if axial is None else axial
        Ca = Cs[:, np.concatenate((o, v))]
        kw = dict(frozen_occ=Cs[:, fo] if len(fo) else None, dropped_vir=Cs[:, fv] if len(fv) else None,
                  tol=tol, max_memory=self.max_memory, solver=zvector_solver, **kw_q)
        # in the blocks of the symmetry labels where T3 and l3 are in them and the plan fits
        plan = None
        if (self.symmetry and getattr(eris, 'use_labels', False) and np.size(l3) == np.size(self.t3)
                and np.size(self.l3) == np.size(self.t3) and self._blocks_pay(eris)
                and density_blk.engine(eris, np.result_type(self.t1, self.t3, l3, eris.oovv,
                                                            eris.fock)) is not None):
            try:
                plan = density_opt.relaxed_rdm1(self._scf, eris, self.t1, self.t2, self.t3, l1, l2,
                                                l3, Ca, blocks=True, plan_only=True, **kw)
            except MemoryError:         # (the packed block then, which says what it needs)
                pass
        if plan is not None:
            rel = density_opt.relaxed_rdm1(self._scf, eris, self.t1, self.t2, self.t3, l1, l2,
                                           l3, Ca, blocks=True, plan=plan, **kw)
        else:
            if zeta is not None:        # (all in the layout of self.t3: on the packed block)
                kw.update(zeta=zeta[:2] + (self._on_packed_block(zeta[2]),),
                          lams=(self.l1, self.l2, self._packed_l3()))
                l3 = self._on_packed_block(l3)
            else:
                l3 = self._packed_l3()
            eris, t3 = self._dense_view()
            rel = density_opt.relaxed_rdm1(self._scf, eris, self.t1, self.t2, t3, l1, l2, l3, Ca, **kw)
        self.rdm1_blocks = plan is not None
        dm = rel.relaxed if relaxed else rel.unrelaxed
        self.rdm1_basis = rel.mo_coeff
        if ao_repr:
            C = rel.mo_coeff
            return C @ dm @ C.conj().T
        return dm

    def ccsdt_q_lambda(self):
        '''CCSDT(Q)_Lambda: the (Q) correction with Lambda2 and Lambda3 on
        the left (zccsdt.cc.quadruples_lambda; in the sectors of the
        symmetry labels where the integrals have them and the blocks fit).
        Lambda is solved for if it is not there.  Returns E(Q)_L and stores
        e_ql, e_ql5 (its fifth-order part, [Q]_L) and e_tot_ql.'''
        from zccsdt.cc import ccsdt_opt, quadruples_lambda, quadruples_lambda_sym
        if getattr(self, 'l3', None) is None or getattr(self, '_l_with_q', False):
            self.solve_lambda()
        eris = self._eris if self._eris is not None else self.ao2mo()
        t3 = ccsdt_opt.packed_t3(eris, self.t3)
        ccsdt_opt.release_cached(eris)
        no, nv = eris.nocc, eris.nmo - eris.nocc
        if no < 4 or nv < 4:
            self.e_ql5 = self.e_ql = 0.0
            return 0.0
        l3 = self._packed_l3()
        isz = np.dtype(np.result_type(self.t2, t3, self.l2, l3, eris.oovv)).itemsize
        budget = memory.budget(self.max_memory)
        spl = None
        if self.symmetry and quadruples_lambda_sym.available(eris):
            try:
                spl = quadruples_lambda_sym.plan(eris, isz, budget)
            except MemoryError:
                pass
        if spl is not None:
            e5, e = quadruples_lambda_sym.kernel(eris, self.t2, t3, self.l2, l3, plan_=spl)
        else:
            pl = quadruples_lambda.plan(no, nv, isz, budget)
            e5, e = quadruples_lambda.kernel(eris, self.t2, t3, self.l2, l3, plan_=pl)
        self.e_ql5, self.e_ql = float(e5), float(e)
        self.e_tot_ql = self.e_tot + self.e_ql
        return self.e_ql

    def eom_ee(self, nroots=3, conv_tol=1e-6, max_space=None, max_cycle=60, guess=None, nguess=None,
               guess_tol=1e-4, shift=None, partners=False):
        '''EOM-EE-CCSDT: the ``nroots`` lowest excitation energies of the
        converged amplitudes and their right-hand vectors (zccsdt.cc.eom;
        docs/eom.md).  Returns (energies in Hartree, zccsdt.cc.eom.Result);
        the result is kept as self.eom, the energies as self.e_ee.

        The states: all of them on the packed block.  Where the iterations
        ran in the blocks of the symmetry labels (a linear molecule with
        ``symmetry``), the states of one label shift: the excitation
        operator changes the sum of the labels of the spinors (2 m_j, as
        the output lists them) by ``shift`` - r1[i,a] where l_a - l_i =
        shift, r2 and r3 likewise with the sums.  Default 0: the totally
        symmetric states, those with the m_j of the ground state.  Each
        shift is an eigenproblem of its own, in the blocks of that shift.
        Spin is not a label here: the three components of a triplet are
        three roots (in a linear molecule in different shifts, unless two
        of them have the same).

                     On the packed block (``symmetry`` off, or spin orbitals of pure
                     spin from an RHF / UHF reference, whose label is 2 m_s) a shift
                     selects the states too - by projection, at the cost of the packed
                     block: the guess and every sigma vector are restricted to the
                     elements of the shift (shift 2 of a closed shell: the M_S = 1
                     components of the triplets, each state once; shift 0: the singlets
                     and the M_S = 0 components).
        shift        a number, or a list of them (or 'all': every shift a single
                     excitation can make): then ``nroots`` roots of each (nroots may be
                     a list too), and the return value is ({shift: energies}, {shift:
                     Result}).  For a reference that is symmetric under time reversal
                     (a closed shell: every spinor of label l has a partner of label -l
                     with its energy) the states of -shift are the time-reversed states
                     of +shift with the same energies: of such a pair in the list only
                     the first is computed and the other gets its energies (a Result
                     without vectors, ``partner_of`` set), unless ``partners`` is True.
        conv_tol     the norm of the residual (H R - omega R) of a normalized vector
        max_space    vectors of the Davidson subspace at most (default 8 + 4 nroots;
                     lowered to what max_memory or max_disk hold)
        guess        [(r1[i,a], r2[(i<j),(a<b)])] to start from; default: the roots of
                     the singles-doubles block from ``nguess`` (default nroots) unit
                     vectors on the lowest single excitations, converged to guess_tol'''
        from zccsdt import disk
        from zccsdt.cc import eom
        if self.t3 is None:
            raise RuntimeError('%s.eom_ee(): no amplitudes (kernel() first)' % self.__class__.__name__)
        eris = self._eris if self._eris is not None else self.ao2mo()

        def run(n, cls, sector=None):
            with disk.limit(self.max_disk):
                return eom.kernel(eris, self.t1, self.t2, self.t3, nroots=n, conv_tol=conv_tol,
                                  max_space=max_space, max_cycle=max_cycle, guess=guess, nguess=nguess,
                                  guess_tol=guess_tol, opts=self._options(None, None),
                                  verbose=self.verbose >= logger.NOTE, stdout=self.stdout, shift=cls, sector=sector)
        if shift is None:
            res = run(nroots, None)
            self.eom, self.e_ee = res, res.e
            return res.e, res
        from zccsdt.cc import t3pack
        blocks = t3pack.classes_of(eris) is not None        # in the blocks of the labels, or by masks
        labels = np.asarray(eris.labels) if blocks else self._state_labels(eris)
        if labels is None:
            raise ValueError('%s.eom_ee(shift=): no symmetry labels for the states - a linear molecule on the z '
                             'axis (2 m_j) or spin orbitals of pure spin (2 m_s) have them' % self.__class__.__name__)
        many = isinstance(shift, (list, tuple, str))
        if isinstance(shift, str):
            if shift != 'all':
                raise ValueError("shift: a number, a list of numbers or 'all'")
            shifts = sorted(eom.label_shifts(eris, labels), key=lambda d: (abs(d), -d))
        else:
            shifts = [int(d) for d in (shift if many else [shift])]
        counts = list(nroots) if isinstance(nroots, (list, tuple)) else [nroots] * len(shifts)
        if len(counts) != len(shifts):
            raise ValueError('nroots: one number, or one per shift')
        classes = [eom.class_shift(eris, d) if blocks else None for d in shifts]
        paired = not partners and eom.time_reversal_symmetric(eris, labels=labels)
        out = {}
        for d, cls, n in zip(shifts, classes, counts):
            if d in out:
                continue
            if paired and d != 0 and -d in out and len(out[-d].e) >= n:
                out[d] = eom.partner(out[-d], d, n)
                if self.verbose >= logger.NOTE:
                    print('EOM-EE: label shift %+d: the time-reversed states of shift %+d, the same energies'
                          % (d, -d), file=self.stdout, flush=True)
                continue
            if self.verbose >= logger.NOTE:
                print('EOM-EE: label shift %+d' % d, file=self.stdout, flush=True)
            # (shift 0: the layout of T3 and the engine as it is)
            out[d] = run(n, cls if cls else None) if blocks else run(n, None, (labels, d))
            out[d].label_shift = d
        if not many:
            res = out[shifts[0]]
            self.eom, self.e_ee = res, res.e
            return res.e, res
        self.eom, self.e_ee = out, {d: r.e for d, r in out.items()}
        return self.e_ee, out

    def _state_labels(self, eris):
        '''A label per active spinor that adds (occupied, then virtual), by
        which states can be told apart on the packed block: 2 m_j of the
        spinors of a linear molecule, 2 m_s of spin orbitals of pure spin;
        None if there is none.'''
        lab = getattr(eris, 'labels', None)
        if lab is not None:
            return None if getattr(eris, 'modulus', 0) else np.asarray(lab)
        if isinstance(self._scf, _Given) or self._kind() != 'ghf':
            return None
        C = np.asarray(self.mo_coeff)
        nao = C.shape[0] // 2
        _, o, v, _ = self._active()
        cols = np.concatenate((o, v))
        a, b = np.linalg.norm(C[:nao, cols], axis=0), np.linalg.norm(C[nao:, cols], axis=0)
        if not np.all((a < 1e-8) | (b < 1e-8)):
            return None
        return np.where(b < 1e-8, 1, -1).astype(np.int64)

    def eom_ip(self, nroots=3, shift=None, partners=False, short=True, **kwargs):
        '''EOM-IP-CCSDT: the ``nroots`` lowest ionization energies (Hartree)
        and the zccsdt.cc.eom.Result; vectors r[i], r[(i<j), a],
        r[(i<j<k), (a<b)] (docs/eom.md, section 5).  The result is kept as
        self.eom_ip_result.

        shift        None: all states, on the packed layout.  Where the iterations ran
                     in the blocks of the symmetry labels, the label of the electron
                     that is removed - minus the label (2 m_j) of its spinor, the change
                     of the sum of the labels of the state - selects the states, which
                     are then computed in the blocks of the labels; a list of them, or
                     'all' (every label an occupied spinor has), as in eom_ee(): the
                     return value is ({shift: energies}, {shift: Result}), and of a
                     pair +-shift of a reference that is symmetric under time reversal
                     one is computed unless ``partners`` is True
        short        False: the EE problem with a ghost virtual spinor at its full
                     length, on the packed block (zccsdt.cc.eom.ghost: the reference
                     route; the cost of an EE calculation)
        the others   as eom_ee().  The vectors of a Result have the ghost index
                     (res.short.strip(res, k) takes it out).'''
        return self._eom_open('ip', nroots, shift, partners, kwargs) if short else self._eom_ghost('ip', nroots, kwargs)

    def eom_ea(self, nroots=3, shift=None, partners=False, short=True, **kwargs):
        '''EOM-EA-CCSDT: the ``nroots`` lowest E(N+1) - E(N) (minus the
        electron affinities; Hartree) and the Result; vectors r[a],
        r[i, (a<b)], r[(i<j), (a<b<c)].  ``shift``: the label of the
        electron that is attached (the label of its spinor); everything
        else as eom_ip().  The result is kept as self.eom_ea_result.'''
        return self._eom_open('ea', nroots, shift, partners, kwargs) if short else self._eom_ghost('ea', nroots, kwargs)

    def _eom_open(self, kind, nroots, shift, partners, kwargs):
        '''IP and EA with short vectors (zccsdt.cc.eom.ghost_short).'''
        from zccsdt import disk
        from zccsdt.cc import eom
        if self.t3 is None:
            raise RuntimeError('%s.eom_%s(): no amplitudes (kernel() first)' % (self.__class__.__name__, kind))
        note = self.verbose >= logger.NOTE
        if shift is None:
            eris, t3 = self._dense_view()
        else:
            eris, t3 = (self._eris if self._eris is not None else self.ao2mo()), self.t3
            eom.class_shift(eris, 0)            # (the labels must be in use)
        sh = eom.ghost_short(eris, self.t1, self.t2, t3, kind)

        def run(n, cls):
            with disk.limit(self.max_disk):
                res = eom.kernel(sh.eris, sh.t1, sh.t2, sh.t3, nroots=n, opts=self._options(None, None),
                                 name=kind.upper(), verbose=note, stdout=self.stdout, shift=cls, **kwargs)
            res.short = sh
            return res
        if shift is None:
            res = run(nroots, sh.class_shift(None))
            setattr(self, 'eom_' + kind + '_result', res)
            return res.e, res
        many = isinstance(shift, (list, tuple, str))
        if isinstance(shift, str):
            if shift != 'all':
                raise ValueError("shift: a number, a list of numbers or 'all'")
            shifts = sorted(sh.labels(), key=lambda d: (abs(d), -d))
        else:
            shifts = [int(d) for d in (shift if many else [shift])]
        counts = list(nroots) if isinstance(nroots, (list, tuple)) else [nroots] * len(shifts)
        if len(counts) != len(shifts):
            raise ValueError('nroots: one number, or one per shift')
        classes = [sh.class_shift(d) for d in shifts]
        paired = not partners and eom.time_reversal_symmetric(eris)
        out = {}
        for d, cls, n in zip(shifts, classes, counts):
            if d in out:
                continue
            if paired and -d in out and len(out[-d].e) >= n:
                out[d] = eom.partner(out[-d], d, n)
                if note:
                    print('EOM-%s: label %+d: the time-reversed states of label %+d, the same energies'
                          % (kind.upper(), d, -d), file=self.stdout, flush=True)
                continue
            if note:
                print('EOM-%s: label %+d of the electron' % (kind.upper(), d), file=self.stdout, flush=True)
            out[d] = run(n, cls)
            out[d].label_shift = d
        res = out[shifts[0]] if not many else out
        setattr(self, 'eom_' + kind + '_result', res)
        if not many:
            return res.e, res
        return {d: r.e for d, r in out.items()}, out

    def _eom_ghost(self, kind, nroots, kwargs):
        from zccsdt import disk
        from zccsdt.cc import eom
        if self.t3 is None:
            raise RuntimeError('%s.eom_%s(): no amplitudes (kernel() first)' % (self.__class__.__name__, kind))
        eris, t3 = self._dense_view()
        eris2, t1, t2, t3, singles = eom.ghost(eris, self.t1, self.t2, t3, kind)
        with disk.limit(self.max_disk):
            res = eom.kernel(eris2, t1, t2, t3, nroots=nroots, opts=self._options(None, None), singles=singles,
                             name=kind.upper(), verbose=self.verbose >= logger.NOTE, stdout=self.stdout, **kwargs)
        setattr(self, 'eom_' + kind + '_result', res)
        return res.e, res

    q_progress = 300.0      # seconds between the progress lines of (Q)

    def ccsdt_q(self, t2=None, t3=None, fraction=None):
        '''(Q) correction of the converged CCSDT amplitudes on spin orbitals
        or spinors (zccsdt.cc.quadruples; canonical orbitals); returns E(Q)
        and stores e_q, e_q5 (its fifth-order part, [Q]) and e_tot_q.

        The correction is a sum over the occupied quadruples; a line with
        the part done and the time left is printed every ``q_progress``
        seconds.  ``fraction`` (e.g. 0.01): only that part of the occupied
        quadruples is summed, evenly spread, and the time of the whole step
        is estimated from it (the time per quadruple scaled up, what is made
        once per group counted once) - returned in seconds and kept as
        ``q_estimate``; the energies of the object are left as they are.'''
        import time
        from math import comb
        from zccsdt.cc import ccsdt_opt, quadruples, quadruples_sym
        log = logger.new_logger(self)
        pr = _Printer(log)
        stride = 1 if not fraction else max(1, int(round(1.0 / fraction)))
        mark = [time.perf_counter()]

        def progress(done, total, seconds):
            if done == total or time.perf_counter() - mark[0] >= self.q_progress:
                mark[0] = time.perf_counter()
                pr.kv('progress', '%d of %d occupied quadruples (%.1f %%) in %.0f s; about %.0f s left'
                      % (done, total, 100.0 * done / total, seconds, seconds * (total - done) / done), indent=3)
            progress.last = (done, total, seconds)
        progress.last = None
        t2 = self.t2 if t2 is None else t2
        t3 = self.t3 if t3 is None else t3
        eris = self._eris if self._eris is not None else self.ao2mo()
        if np.size(t3) != comb(eris.nocc, 3) * comb(eris.nmo - eris.nocc, 3):
            t3 = ccsdt_opt.packed_t3(eris, t3)  # (the blocks of the labels, as (Q) reads them)
        ccsdt_opt.release_cached(eris)          # (what the iterations kept: not part of this budget)
        no, nv = eris.nocc, eris.nmo - eris.nocc
        dtype = np.result_type(t2, t3, eris.oovv)
        snap0 = timing.snapshot()
        pr.line(' (Q) correction: %d occupied x %s virtual quadruples, %s'
                % (comb(no, 4), _count(comb(nv, 4)),
                   'complex' if np.iscomplexobj(np.zeros(0, dtype)) else 'real'))
        if no < 4 or nv < 4:
            self.e_q5 = self.e_q = 0.0
        else:
            budget = memory.budget(self.max_memory)
            xfull = max(3, no - 1)
            spl = None
            if self.symmetry and quadruples_sym.available(eris):
                try:
                    spl = quadruples_sym.plan(eris, np.dtype(dtype).itemsize, budget)
                except MemoryError:     # (the dense code can do with less: batches of columns)
                    pass
            if spl is not None:
                sec = spl.sectors
                pr.kv('symmetry', '%d labels of the virtual spinors%s: the products in blocks'
                      % (len(sec.vlab), ' modulo %d' % sec.modulus if sec.modulus else ' (2 m_j)'), indent=3)
                pr.kv('memory', '%s besides T3 and the integrals' % _size(spl.need), indent=3)
                pr.kv('plan', 'T2 <vv||vv> intermediate for %s' % (
                    'all occupied indices' if spl.dense.xv >= xfull else
                    '%d of %d occupied indices at a time' % (spl.dense.xv, xfull)), indent=3)
                with _PeakMemory() as peak, timing.timer(ROOT_Q):
                    part = quadruples_sym.kernel(eris, t2, t3, plan_=spl, stride=stride, progress=progress)
            else:
                pl = quadruples.plan(no, nv, np.dtype(dtype).itemsize, budget)
                pr.kv('memory', '%s besides T3 and the integrals' % _size(pl.need), indent=3)
                pr.kv('plan', 'products in %d and %d batches of columns; T2 <vv||vv> intermediate for %s'
                      % (-(-comb(nv, 2) // pl.cols22), -(-comb(nv, 3) // pl.cols13),
                         'all occupied indices' if pl.xv >= xfull else
                         '%d of %d occupied indices at a time' % (pl.xv, xfull)), indent=3)
                with _PeakMemory() as peak, timing.timer(ROOT_Q):
                    part = quadruples.kernel(eris, t2, t3, plan_=pl, stride=stride, progress=progress)
            w, c, k = timing.since(snap0)[(ROOT_Q,)]
            if stride > 1:
                done, total, seconds = progress.last
                est = seconds * comb(no, 4) / done + (w - seconds)   # (w - seconds: what is made once)
                self.q_estimate = {'sampled': done, 'quadruples': comb(no, 4), 'seconds': w, 'estimate': est,
                                   'sectors': spl is not None}
                pr.kv('sample', 'every %d-th occupied quadruple: %d of %d in %.1f s (%.3f s each, %.1f cores)'
                      % (stride, done, comb(no, 4), seconds, seconds / done, c / w if w > 0 else 0), indent=3)
                pr.kv('estimate', 'the whole step: about %.0f s (%.2f h); the energies are not set' % (est, est / 3600),
                      indent=3)
                pr.rule(char='=')
                return est
            self.e_q5, self.e_q = part
        pr.kv('E[Q]', '%20.12f   (fifth order)' % self.e_q5, indent=3)
        pr.kv('E(Q)', '%20.12f' % self.e_q, indent=3)
        pr.kv('E(CCSDT(Q))', '%20.12f' % self.e_tot_q, indent=3)
        if no >= 4 and nv >= 4:
            pr.kv('time', '%.2f s wall, %.2f s CPU (%.1f cores); peak memory %s anonymous'
                  % (w, c, c / w if w > 0 else 0, _size(peak.peak or 0)), indent=3)
        pr.rule(char='=')
        return self.e_q


class GCCSDTQ(GCCSDT):
    __doc__ = '''CCSDTQ on a GHF or two-component spinor (X2C) reference: spin
    orbitals or spinors without spin blocks, real or complex
    (zccsdt.cc.ccsdtq; T3 and T4 on their unique blocks).

    kernel() runs CCSDT first (a GCCSDT object with the same settings, kept
    as ``ccsdt_run``: its ccsdt_q() gives CCSDT(Q)) and CCSDTQ from its
    amplitudes; with amplitudes given (t1, t2, t3 packed, optionally t4) it
    starts from those.  Afterwards e_corr, e_tot, t1 .. t4, converged,
    cycles and timings are those of CCSDTQ, e_ccsdt the total energy of the
    CCSDT step.  ccsdtq_p() gives the (P) correction (zccsdt.cc.pentuples):
    e_p and e_tot_p = E(CCSDTQ(P)).  ``frozen`` counts spin orbitals or
    spinors.

''' + CCSDTBase.__doc__

    _title = 'GCCSDTQ'
    _subtitle = 'CCSDTQ on spin orbitals or spinors (GHF, X2C), real or complex'
    _stages = Q_STAGES
    _keys = GCCSDT._keys | {'t4', 'e_ccsdt', 'ccsdt_run', 'e_p', 'diis_t4'}
    diis_t4 = None      # T4 in the DIIS vectors: None - unless nothing holds three of them; True; False

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        GCCSDT.__init__(self, mf, frozen, mo_coeff, mo_occ)
        self.t4 = None
        self.e_ccsdt = None
        self.ccsdt_run = None
        self.e_p = None

    @property
    def e_tot_p(self):
        return None if self.e_p is None else self.e_tot + self.e_p

    def kernel(self, t1=None, t2=None, t3=None, t4=None):
        if t1 is None or t2 is None or t3 is None:
            cc3 = GCCSDT(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
            for key in ('verbose', 'stdout', 'max_memory', 'max_disk', 'conv_tol', 'conv_tol_normt', 'max_cycle',
                        'diis_space', 'diis_t3', 'diis_file', 'incore_complete', 'real_spinors', 'symmetry',
                        '_block_iterations'):
                setattr(cc3, key, getattr(self, key))
            cc3.kernel()
            self.ccsdt_run, self._eris, self.e_ccsdt = cc3, cc3._eris, cc3.e_tot
            t1, t2, t3 = cc3.t1, cc3.t2, cc3.t3
        return self.ccsdtq(t1, t2, t3, t4)

    def ccsdt_q(self, t2=None, t3=None):
        if self.ccsdt_run is None:
            raise RuntimeError('GCCSDTQ.ccsdt_q(): no CCSDT step has been run (kernel() without amplitudes)')
        return self.ccsdt_run.ccsdt_q(t2, t3)

    def ccsdt_q_lambda(self):
        if self.ccsdt_run is None:
            raise RuntimeError('GCCSDTQ.ccsdt_q_lambda(): no CCSDT step has been run (kernel() without amplitudes)')
        return self.ccsdt_run.ccsdt_q_lambda()

    def solve_lambda(self, with_q=False):
        '''Lambda of the CCSDT step (GCCSDT.solve_lambda), with ``with_q``
        that of CCSDT(Q), or with 'lambda' the equations of
        CCSDT(Q)_Lambda: CCSDTQ has no Lambda equations here.'''
        if self.ccsdt_run is None:
            raise RuntimeError('GCCSDTQ.solve_lambda(): no CCSDT step has been run (kernel() without amplitudes)')
        return self.ccsdt_run.solve_lambda(with_q=with_q)

    def make_rdm1(self, relaxed=True, ao_repr=False, tol=1e-11, with_q=False):
        '''The one-particle density of the CCSDT step (GCCSDT.make_rdm1),
        with ``with_q`` that of CCSDT(Q) or, with 'lambda', of
        CCSDT(Q)_Lambda; not of CCSDTQ.'''
        if self.ccsdt_run is None:
            raise RuntimeError('GCCSDTQ.make_rdm1(): no CCSDT step has been run (kernel() without amplitudes)')
        dm = self.ccsdt_run.make_rdm1(relaxed, ao_repr, tol, with_q)
        self.rdm1_basis, self.rdm1_blocks = self.ccsdt_run.rdm1_basis, self.ccsdt_run.rdm1_blocks
        return dm

    def ccsdtq(self, t1, t2, t3, t4=None):
        from zccsdt import disk
        with disk.limit(self.max_disk):     # (every scratch file of the run counts against max_disk)
            return self._ccsdtq(t1, t2, t3, t4)

    def _ccsdtq(self, t1, t2, t3, t4):
        import copy
        from math import comb
        from zccsdt.cc import ccsdtq, ccsdtq_sym, ccsdt_opt
        log = logger.new_logger(self)
        cput0 = log.init_timer()
        pr = _Printer(log)
        self._header(pr)
        eris = self._eris if self._eris is not None else self.ao2mo()
        self._eris = eris
        no, nv = eris.nocc, eris.nmo - eris.nocc
        dtype = np.result_type(t1, t2, eris.oovv, eris.fock)
        isz = np.dtype(dtype).itemsize
        nt4 = comb(no, 4) * comb(nv, 4)
        # in the blocks of the symmetry labels where the integrals have them and the blocks fit
        splan = None
        if self.symmetry and ccsdtq_sym.available(eris):
            try:
                splan = ccsdtq_sym.plan_for(eris, dtype, self.max_memory, self.diis_space, self.incore_complete)
            except MemoryError:         # (the dense solver has plans for less: scratch files)
                pass
        if splan is None and getattr(eris, 'use_labels', False):
            # the dense solver on spinors that are in the order of their labels: T3 packed, no blocks
            t3 = ccsdt_opt.packed_t3(eris, t3)
            ccsdt_opt.release_cached(eris)
            eris = copy.copy(eris)
            eris.use_labels = False
            eris.__dict__.pop('_buffers', None)
            self._eris = eris
        pr.rule()
        if splan is not None:
            pr.kv('amplitudes', 'T4 %s of %s in the blocks of the labels (%s), %s'
                  % (_count(splan.layout.size), _count(nt4), _size(splan.t4), 'complex' if isz == 16 else 'real'))
        else:
            pr.kv('amplitudes', 'T4 %s (%s, i<j<k<l, a<b<c<d), %s; T3 %s'
                  % (_count(nt4), _size(nt4 * isz), 'complex' if isz == 16 else 'real',
                     _count(comb(no, 3) * comb(nv, 3))))
        normt = self._conv_tol_normt()
        pr.kv('convergence', '|dE| < %g Eh and %s < %g; at most %d iterations'
              % (self.conv_tol, self._norm_name, normt, self.max_cycle))
        pr.kv('memory', self._budget_text())
        if splan is not None:
            plan = splan
            pr.kv('memory plan', 'needs %s next to the integrals: everything in memory, in blocks' % _size(plan.need))
        else:
            plan = self.plan if isinstance(self.plan, ccsdtq.Plan) else ccsdtq.plan_for(
                eris, dtype, self.max_memory, self.diis_space, self.incore_complete)
            filed = [name for name, on in (('T4', plan.t4_file), ('the six-index intermediate', plan.xv_file),
                                           ('the residual of T4', plan.r4_file)) if on]
            pr.kv('memory plan', 'needs %s next to the integrals: %s' % (
                _size(plan.need), 'T4, its residual and the six-index intermediate in memory' if not filed
                else 'memory is short - %s in a scratch file' % ', '.join(filed)))
        pr.kv('batches', 'products of the T4 equation in batches of %s%s'
              % (_size(plan.batch_bytes), '' if plan.mem_mb in (None, 2000.0) else
                 '; the CCSDT step with buffers of %.0f MB' % plan.mem_mb), indent=3)
        pr.kv('disk', self._disk_text())
        opts = ccsdtq.Options(conv_tol=self.conv_tol, conv_tol_normt=normt, max_cycle=self.max_cycle,
                              diis_space=self.diis_space, diis_file=self.diis_file,
                              incore_complete=self.incore_complete, max_memory=self.max_memory,
                              max_disk=self.max_disk, diis_t4=self.diis_t4, plan=plan)
        # (before the iterations: the disk holds the plan and the DIIS vectors, or the run stops here)
        dplan = (ccsdtq_sym if splan is not None else ccsdtq).diis_plan_of(eris, dtype, opts, plan)
        pr.kv('DIIS', '%s; they take %s of disk and %s of memory'
              % (dplan.text(), _size(dplan.disk), _size(dplan.memory)), indent=3)
        pr.rule()
        head = ('iter', 'E_corr / Eh', 'dE / Eh', '|dT1|', '|dT2|', '|dT3|', '|dT4|', 'DIIS', 'time / s', 'cores')
        fmt = ' %4s  %20s  %12s  %9s  %9s  %9s  %9s  %4s  %9s  %5s'
        pr.line(fmt % head)
        pr.line(fmt % tuple('-' * n for n in (4, 20, 12, 9, 9, 9, 9, 4, 9, 5)))

        def report(info):
            busy = info['cpu'] / info['wall'] if info['wall'] > 0 else 0.0
            pr.line(' %4d  %20.15f  %12.4e  %9.2e  %9.2e  %9.2e  %9.2e  %4d  %9.2f  %5.1f'
                    % ((info['cycle'], info['e_corr'], info['de']) + tuple(info['norms'])
                       + (info['diis'], info['wall'], busy)))
            if self.callback is not None:
                self.callback(info)
        opts.callback, opts.diis_plan = report, dplan
        snap0 = timing.snapshot()
        with _PeakMemory() as peak, timing.timer(ROOT_ITER):
            res = (ccsdtq_sym if splan is not None else ccsdtq).kernel(eris, opts, t1, t2, t3, t4)
        self.converged, self.cycles = bool(res.converged), int(res.ncycle)
        self.e_hf, self.e_corr = float(eris.e_hf), float(res.e_corr)
        self.t1, self.t2, self.t3, self.t4 = res.t1, res.t2, res.t3, res.t4
        self.plan_used, self.timings, self.peak_memory = res.plan, list(res.timings), peak.peak
        self.diis_plan_used = res.diis_plan
        pr.rule()
        pr.line(' GCCSDTQ %s in %d iterations' % ('converged' if self.converged else 'NOT converged', self.cycles))
        pr.kv('E(SCF)', '%20.12f' % self.e_hf, indent=3)
        pr.kv('E_corr(CCSDTQ)', '%20.12f' % self.e_corr, indent=3)
        pr.kv('E(CCSDTQ)', '%20.12f' % self.e_tot, indent=3)
        if self.e_ccsdt is not None:
            pr.kv('E(CCSDTQ) - E(CCSDT)', '%20.12f' % (self.e_tot - self.e_ccsdt), indent=3)
        self._print_timings(pr, snap0, peak.peak)
        pr.rule(char='=')
        log.timer(self.__class__.__name__, *cput0)
        return self.e_corr, self.t1, self.t2, self.t3, self.t4


    def ccsdtq_p(self, t2=None, t3=None, t4=None):
        '''(P) correction of the converged CCSDTQ amplitudes on spin orbitals
        or spinors (zccsdt.cc.pentuples; canonical orbitals); returns E(P)
        and stores e_p and e_tot_p, the energy of CCSDTQ(P).'''
        from math import comb
        from zccsdt.cc import ccsdt_opt, pentuples, pentuples_sym
        log = logger.new_logger(self)
        pr = _Printer(log)
        t2 = self.t2 if t2 is None else t2
        t3 = self.t3 if t3 is None else t3
        t4 = self.t4 if t4 is None else t4
        if t4 is None:
            raise RuntimeError('GCCSDTQ.ccsdtq_p(): no CCSDTQ amplitudes (kernel() first)')
        eris = self._eris if self._eris is not None else self.ao2mo()
        # (P) reads the packed arrays: the blocks of the labels are put in their places
        from zccsdt.cc import ccsdtq_sym
        if np.size(t3) != comb(eris.nocc, 3) * comb(eris.nmo - eris.nocc, 3):
            t3 = ccsdt_opt.packed_t3(eris, t3)
        if np.size(t4) != comb(eris.nocc, 4) * comb(eris.nmo - eris.nocc, 4):
            t4 = ccsdtq_sym.packed_t4(eris, t4)
        ccsdt_opt.release_cached(eris)          # (what the iterations kept: not part of this budget)
        no, nv = eris.nocc, eris.nmo - eris.nocc
        dtype = np.result_type(t2, t3, t4, eris.oovv)
        snap0 = timing.snapshot()
        pr.line(' (P) correction: %s occupied x %s virtual pentuples, %s'
                % (_count(comb(no, 5)), _count(comb(nv, 5)),
                   'complex' if np.iscomplexobj(np.zeros(0, dtype)) else 'real'))
        if no < 5 or nv < 5:
            self.e_p = 0.0
        else:
            budget = memory.budget(self.max_memory)
            spl = None
            if self.symmetry and pentuples_sym.available(eris):
                try:
                    spl = pentuples_sym.plan(eris, np.dtype(dtype).itemsize, budget)
                except MemoryError:     # (the dense code has plans for less)
                    pass
            if spl is not None:
                sec = spl.sectors
                pr.kv('symmetry', '%d labels of the virtual spinors%s: the products in blocks'
                      % (len(sec.vlab), ' modulo %d' % sec.modulus if sec.modulus else ' (2 m_j)'), indent=3)
                pr.kv('memory', '%s besides T3, T4 and the integrals' % _size(spl.need), indent=3)
                with _PeakMemory() as peak, timing.timer(ROOT_P):
                    self.e_p = pentuples_sym.kernel(eris, t2, t3, t4, plan_=spl)
            else:
                pl = pentuples.plan(no, nv, np.dtype(dtype).itemsize, budget)
                pr.kv('memory', '%s besides T3, T4 and the integrals' % _size(pl.need), indent=3)
                filed = [name for name, on in (('the second-order quadruples', pl.y4_file),
                                               ('the six-index intermediate', pl.xv_file)) if on]
                pr.kv('plan', 'products in %d and %d batches of columns; T2 T2 T2 intermediate for %s%s'
                      % (-(-comb(nv, 3) // pl.cols23), -(-comb(nv, 4) // pl.cols14),
                         'all occupied quadruples' if pl.store_xq else 'the quadruples of one pentuple at a time',
                         '; memory is short - %s in a scratch file' % ' and '.join(filed) if filed else ''),
                      indent=3)
                with _PeakMemory() as peak, timing.timer(ROOT_P):
                    self.e_p = pentuples.kernel(eris, t2, t3, t4, plan_=pl)
            w, c, k = timing.since(snap0)[(ROOT_P,)]
        pr.kv('E(P)', '%20.12f' % self.e_p, indent=3)
        pr.kv('E(CCSDTQ(P))', '%20.12f' % self.e_tot_p, indent=3)
        if no >= 5 and nv >= 5:
            pr.kv('time', '%.2f s wall, %.2f s CPU (%.1f cores); peak memory %s anonymous'
                  % (w, c, c / w if w > 0 else 0, _size(peak.peak or 0)), indent=3)
            parts = timing.children(timing.since(snap0), (ROOT_P,))
            names = {'p.gemm': 'products', 'p.reduce': 'reductions', 'p.operands': 'operands',
                     'p.xq': 'T2 T2 T2 intermediate', 'p.xv': 'Xv', 'p.xo': 'Xo', 'p.y4': 'second-order quadruples',
                     'p.energy': 'energy', 'p.blocks': 'integral blocks'}
            pr.kv('of which', ', '.join('%s %.2f s' % (names.get(n, n), w2) for n, w2, c2, k2 in parts
                                        if w2 >= 0.01 * w), indent=3, level=logger.INFO)
        pr.rule(char='=')
        return self.e_p


def CCSDTQ(mf, frozen=None, mo_coeff=None, mo_occ=None):
    '''CCSDTQ for the reference ``mf``: GCCSDTQ for GHF and two-component
    (X2C) spinors.  An RHF, UHF or ROHF reference is taken as GHF spin
    orbitals (no spin adaptation at this level yet); ``frozen`` then counts
    orbitals per spin, as in their CCSDT classes.'''
    from pyscf import scf
    from pyscf.scf import dhf, ghf, hf
    from pyscf.x2c import x2c
    if isinstance(mf, hf.KohnShamDFT):
        raise RuntimeError('CCSDTQ needs an HF object, not a DFT object (mf.to_hf())')
    if isinstance(mf, dhf.DHF):
        raise NotImplementedError('CCSDTQ on a four-component (DHF) reference')
    if isinstance(mf, (ghf.GHF, x2c.SCF)):
        return GCCSDTQ(mf, frozen, mo_coeff, mo_occ)
    if mo_coeff is not None or mo_occ is not None:
        raise NotImplementedError('CCSDTQ with given orbitals needs a GHF or X2C reference')
    if frozen is not None and not isinstance(frozen, (int, np.integer)):
        raise NotImplementedError('CCSDTQ: a list of frozen orbitals needs a GHF or X2C reference '
                                  '(scf.addons.convert_to_ghf(mf): the list is then one of spin orbitals)')
    return GCCSDTQ(scf.addons.convert_to_ghf(mf), None if frozen is None else 2 * int(frozen))


def CCSDT(mf, frozen=None, mo_coeff=None, mo_occ=None):
    '''The CCSDT class for the kind of reference: RCCSDT for RHF, UCCSDT for
    UHF and ROHF, GCCSDT for GHF and two-component (X2C) spinors.'''
    from pyscf.scf import dhf, ghf, hf, rohf, uhf
    from pyscf.x2c import x2c
    if isinstance(mf, hf.KohnShamDFT):
        raise RuntimeError('CCSDT needs an HF object, not a DFT object (mf.to_hf())')
    if isinstance(mf, dhf.DHF):
        raise NotImplementedError('CCSDT on a four-component (DHF) reference')
    if isinstance(mf, (ghf.GHF, x2c.SCF)):
        return GCCSDT(mf, frozen, mo_coeff, mo_occ)
    if isinstance(mf, (uhf.UHF, rohf.ROHF)):
        return UCCSDT(mf, frozen, mo_coeff, mo_occ)
    if isinstance(mf, hf.RHF):
        return RCCSDT(mf, frozen, mo_coeff, mo_occ)
    raise NotImplementedError('CCSDT for %s' % mf.__class__.__name__)


# ------------------------------------------------------ largest amplitudes

def _largest(x, n):
    '''Index tuples of the n elements of x with the largest magnitude.'''
    a = np.abs(np.asarray(x)).ravel()
    n = min(n, a.size)
    if n == 0:
        return []
    idx = np.argpartition(-a, n - 1)[:n]
    idx = idx[np.argsort(-a[idx], kind='stable')]
    return [tuple(int(k) for k in np.unravel_index(i, np.shape(x))) for i in idx]


def _top_merge(best, values, labels, n):
    best.extend(zip(values, labels))
    best.sort(key=lambda r: -abs(r[0]))
    del best[n:]


def _largest_t3_blocks(t3b, no, nv, mo, n):
    '''Largest elements of the block-stored spin-adapted T3, one block at a
    time (it may be a mapped scratch array).'''
    from zccsdt.sa.t3blocks import Blocks
    if t3b is None:
        return []
    blocks = Blocks(no)
    best = []
    for B in range(blocks.nblk):
        cube = np.asarray(t3b[B])
        i, j, k = blocks.tri[B]
        vals, labs = [], []
        for a, b, c in _largest(cube, 3 * n):
            if (i == j and a > b) or (j == k and b > c):
                continue            # the same element under the pair exchange
            vals.append(float(cube[a, b, c]))
            labs.append('%d %d %d -> %d %d %d' % (mo(i), mo(j), mo(k),
                                                  mo(no + a), mo(no + b), mo(no + c)))
        _top_merge(best, vals, labs, n)
    return best


def _largest_t3_packed(t3, eris, occ, vir, n):
    '''Largest elements of the packed spin-orbital T3 (i<j<k, a<b<c), in
    chunks of rows (it may be a mapped scratch array).'''
    from zccsdt.cc import ccsdt_opt, t3pack
    if t3 is None:
        return []
    om, vm = t3pack.maps_for(eris)
    eng = ccsdt_opt.engine_for(eris, np.asarray(t3[:1]).dtype)
    best = []
    if hasattr(eng, 'L'):                       # spin blocks: the stored groups
        groups = [(eng.L.block(t3, g), om.tri_off[g], vm.tri_off[g]) for g in eng.L.stored
                  if eng.L.nI[g] and eng.L.nA[g]]
    else:
        groups = [(np.asarray(t3).reshape(om.ntri, vm.ntri), 0, 0)]
    for M, I0, A0 in groups:
        step = max(1, (1 << 23) // max(M.shape[1], 1))
        for r0 in range(0, M.shape[0], step):
            chunk = np.asarray(M[r0:r0 + step])
            vals, labs = [], []
            for r, c in _largest(chunk, n):
                I, A = om.triples[I0 + r0 + r], vm.triples[A0 + c]
                vals.append(chunk[r, c])
                labs.append('%s %s %s -> %s %s %s' % tuple([occ[p] for p in I] + [vir[q] for q in A]))
            _top_merge(best, vals, labs, n)
    return best
