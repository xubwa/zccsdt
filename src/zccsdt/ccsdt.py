#!/usr/bin/env python
'''
CCSDT in the style of PySCF's coupled-cluster classes

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
ROOT_INTS, ROOT_ITER, ROOT_Q = 'zccsdt.integrals', 'zccsdt.iterations', 'zccsdt.q'


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
        frozen : int
            Number of frozen core orbitals: spatial orbitals (RCCSDT), per
            spin (UCCSDT), spin orbitals or spinors (GCCSDT).  Default 0.
            set_frozen() freezes the chemical core.
        conv_tol : float
            Convergence threshold of the correlation energy (Eh).  Default 1e-10.
        conv_tol_normt : float
            Convergence threshold of the amplitude residual norm (RCCSDT,
            default 1e-7) or step norm (UCCSDT, GCCSDT, default 1e-8).
        max_cycle : int
            Maximum number of iterations.  Default 100.
        diis_space : int
            Number of DIIS vectors.  Default 8.
        diis_t3 : bool
            Include T3 in the DIIS vectors (default).  False: DIIS on T1 and
            T2, T3 by its Jacobi step (an option, never chosen automatically).
        diis_file : str
            HDF5 file of the DIIS vectors (default: a temporary file).
        incore_complete : bool
            Keep everything in memory, the DIIS vectors included, whatever
            the budget.  Default False.
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
            The memory plan used (``plan_used``).
        timings : list
            Wall time of each iteration.
        peak_memory : int
            Peak resident anonymous memory of the iterations in bytes.
    '''

    conv_tol = 1e-10
    conv_tol_normt = None
    max_cycle = 100
    diis_space = 8
    diis_t3 = True
    diis_file = None
    incore_complete = False
    plan = None
    callback = None
    _title = 'CCSDT'
    _subtitle = ''
    _norm_labels = ('|dT1|', '|dT2|', '|dT3|')
    _stages = CC_STAGES

    _keys = {
        'mol', 'verbose', 'stdout', 'max_memory', 'frozen', 'conv_tol', 'conv_tol_normt',
        'max_cycle', 'diis_space', 'diis_t3', 'diis_file', 'incore_complete', 'plan',
        'callback', 'mo_coeff', 'mo_occ', 'converged', 'e_hf', 'e_corr', 'cycles',
        't1', 't2', 't3', 'plan_used', 'timings', 'peak_memory', 'e_q', 'e_q5',
    }

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        from pyscf.scf import hf
        if isinstance(mf, hf.KohnShamDFT):
            raise RuntimeError('%s: the first argument is a DFT object; CCSDT needs an HF '
                               'object (mf.to_hf())' % self.__class__.__name__)
        if mo_coeff is None:
            mo_coeff = mf.mo_coeff
            if mo_coeff is None and mf.mol.nelectron > 0:
                mf.run()
                mo_coeff = mf.mo_coeff
        if mo_occ is None:
            mo_occ = mf.mo_occ
        self.mol = mf.mol
        self._scf = mf
        self.verbose = self.mol.verbose
        self.stdout = self.mol.stdout
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
              % (self.diis_space, ', with T3' if self.diis_t3 else ', T1 and T2 only (diis_t3 = False)',
                 'all in memory (incore_complete)' if self.incore_complete else
                 'large parts in %s' % (self.diis_file or 'a scratch file')))
        pr.kv('memory', self._budget_text())
        from zccsdt.scratch import scratch_dir
        pr.kv('scratch', scratch_dir(), level=logger.INFO)
        return self

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
        self._iteration_header(pr)
        with _PeakMemory() as peak, timing.timer(ROOT_ITER):
            res = self._solve(eris, plan, t1, t2, t3, self._iteration_callback(pr))
        self.converged, self.cycles = bool(res.converged), int(res.ncycle)
        self.e_hf, self.e_corr = float(eris.e_hf), float(np.real(res.e_corr))
        self.t1, self.t2, self.t3 = res.t1, res.t2, res.t3
        self.plan_used = res.plan
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
                             callback=callback)

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
                              spin_flip=getattr(self, 'spin_flip', False))

    def _make_plan(self, eris):
        from zccsdt.cc import ccsdt_opt
        return ccsdt_opt.plan_for(eris, np.result_type(eris.fock, eris.oovv),
                                  self._options(None, None))

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

''' + CCSDTBase.__doc__

    _title = 'GCCSDT'
    _subtitle = 'CCSDT on spin orbitals or spinors (GHF, X2C), real or complex'
    _frozen_unit = 2

    def _orbital_lines(self):
        occ = np.asarray(self.mo_occ) > 0
        nocc = int(occ.sum()) - self.ncore
        kind = 'spinors' if self._kind() == 'spinor' else 'spin orbitals'
        return [('orbitals', '%d occupied, %d virtual %s; %d frozen'
                 % (nocc, len(occ) - int(occ.sum()), kind, self.ncore))]

    def _kind(self):
        from zccsdt.eris import _orbital_kind
        return _orbital_kind(self._scf)

    def ao2mo(self):
        from zccsdt import eris
        return eris.from_scf(self._scf, mo_coeff=self.mo_coeff, mo_occ=self.mo_occ,
                             frozen=self.ncore, packed_vvvv=True, max_memory=self.max_memory)

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
        occ = np.where(np.asarray(self.mo_occ) > 0)[0][self.ncore:]
        vir = np.where(np.asarray(self.mo_occ) == 0)[0]
        return ['%d' % (i + 1) for i in occ], ['%d' % (a + 1) for a in vir]

    def ccsdt_q(self, t2=None, t3=None):
        '''(Q) correction of the converged CCSDT amplitudes on spin orbitals
        or spinors (zccsdt.cc.quadruples; canonical orbitals); returns E(Q)
        and stores e_q, e_q5 (its fifth-order part, [Q]) and e_tot_q.'''
        from math import comb
        from zccsdt.cc import ccsdt_opt, quadruples
        log = logger.new_logger(self)
        pr = _Printer(log)
        t2 = self.t2 if t2 is None else t2
        t3 = self.t3 if t3 is None else t3
        eris = self._eris if self._eris is not None else self.ao2mo()
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
            pl = quadruples.plan(no, nv, np.dtype(dtype).itemsize, memory.budget(self.max_memory))
            xfull = max(3, no - 1)
            pr.kv('memory', '%s besides T3 and the integrals' % _size(pl.need), indent=3)
            pr.kv('plan', 'products in %d and %d batches of columns; T2 <vv||vv> intermediate for %s'
                  % (-(-comb(nv, 2) // pl.cols22), -(-comb(nv, 3) // pl.cols13),
                     'all occupied indices' if pl.xv >= xfull else
                     '%d of %d occupied indices at a time' % (pl.xv, xfull)), indent=3)
            with _PeakMemory() as peak, timing.timer(ROOT_Q):
                self.e_q5, self.e_q = quadruples.kernel(eris, t2, t3, plan_=pl)
            w, c, k = timing.since(snap0)[(ROOT_Q,)]
        pr.kv('E[Q]', '%20.12f   (fifth order)' % self.e_q5, indent=3)
        pr.kv('E(Q)', '%20.12f' % self.e_q, indent=3)
        pr.kv('E(CCSDT(Q))', '%20.12f' % self.e_tot_q, indent=3)
        if no >= 4 and nv >= 4:
            pr.kv('time', '%.2f s wall, %.2f s CPU (%.1f cores); peak memory %s anonymous'
                  % (w, c, c / w if w > 0 else 0, _size(peak.peak or 0)), indent=3)
        pr.rule(char='=')
        return self.e_q


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
