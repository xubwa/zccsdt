'''Disk space of the scratch location, and where the DIIS vectors go.

The disk is planned like the memory (zccsdt.memory): ``budget(max_disk)`` is
the number of bytes that the scratch files of a run may still take,

  max_disk    MB for all scratch files of the process, the option of the
              front ends and of the solvers next to ``max_memory`` (what the
              open scratch files hold already is subtracted); without it
              the environment variable ZCCSDT_MAX_DISK (MB), for tests and
              for a disk that is shared;
  else        the free space of the file system of the scratch directory
              (os.statvfs: of the directory of ``diis_file`` for the DIIS
              vectors, else ZCCSDT_TMPDIR, else PySCF's lib.param.TMPDIR)
              less a reserve (RESERVE);

and never more than that free space.

A solver first takes from it what its memory plan maps to scratch files (T3
or T4 and their residuals, the packed <ab||cd> and the rows of the ladder
operator; ``require``: an error with the numbers before anything is
allocated if that does not fit).  The DIIS vectors get the rest
(``plan_diis``).  ``diis_space`` is the size of the subspace that is asked
for, a maximum.  One stored vector is a pair - the amplitudes and their error
vector - and with p bytes for the large parts of a pair (2 T3, or 2 (T3 +
T4)) the plan is, in this order of preference:

  (a) all ``diis_space`` pairs in the scratch file, if the disk budget has
      room for them (the default placement);
  (b) a split: as many pairs in the file as the disk budget holds, the others
      in memory - as many as the memory budget leaves beyond the plan of
      the solver;
  (c) a smaller subspace: the pairs that the two budgets hold together,
      n = floor(disk / p) + floor(memory / p) < diis_space, if n is at least
      FLOOR (3; ``diis_space`` if that is smaller);
  (d) the large part left out of DIIS (T3 of CCSDT and of its Lambda
      equations, T4 of CCSDTQ: DIIS on the smaller parts, the large one by
      its Jacobi step; ``diis_t3 = False``) if not even FLOOR pairs fit;
  (e) an error (DiskFullError, an OSError with ENOSPC) before the iterations
      start, with what was needed and what is there.

(c) and (d) are deliberate reductions that the report of the run states
with the numbers; they are not errors, because ``diis_space`` is a maximum.
What the user sets explicitly is kept: ``incore_complete`` (all pairs in
memory: MemoryError if what the node and the job leave does not hold them),
``diis_t3 = True`` (no (d): (e) instead), ``diis_t3 = False`` (no large part
at all).

``dry_run`` gives the plans of a CCSDT calculation from the numbers of
occupied and virtual spinors and the two budgets, without integrals: for
sizing a job (memory and local disk) before it is submitted.
'''

import errno
import os
from dataclasses import dataclass

from zccsdt import scratch

FLOOR = 3               # smallest subspace with the large part in it
RESERVE = (0.02, 1 << 30)   # of the free space that is left alone: this share, at most this many bytes


class DiskFullError(OSError):
    '''The scratch location does not hold what a run has to put there.'''

    def __init__(self, msg):
        super().__init__(errno.ENOSPC, msg)


def _gb(n):
    if n is None:
        return 'no limit'
    for unit, size in (('GB', 1e9), ('MB', 1e6), ('kB', 1e3)):
        if n >= size / 10:
            return '%.1f %s' % (n / size, unit)
    return '%d bytes' % n


def directory(diis_file=None):
    '''The directory of the scratch files (of ``diis_file`` if that is given).'''
    if diis_file:
        return os.path.dirname(os.path.abspath(diis_file))
    return scratch.scratch_dir()


def same_device(a, b):
    '''Are the directories a and b on one file system?'''
    try:
        return os.stat(a).st_dev == os.stat(b).st_dev
    except OSError:
        return True


def max_disk_bytes(max_disk=None):
    '''The limit in bytes: ``max_disk`` (MB), else what scratch.limit() says
    (ZCCSDT_MAX_DISK, or the max_disk of a run that is under way), else None.'''
    if max_disk is not None:
        return int(max_disk * 1e6)
    return scratch.limit()


def budget(max_disk=None, where=None):
    '''Bytes that the scratch files of the run may still take in the
    directory ``where`` (default: the scratch directory): ``max_disk`` (MB,
    for the whole process; default ZCCSDT_MAX_DISK) minus what its open
    scratch files hold, else the free space less the reserve; never more than
    the free space.  None if nothing is known.'''
    free = scratch.disk_free(where or scratch.scratch_dir())
    cap = max_disk_bytes(max_disk)
    if cap is not None:
        left = max(0, cap - scratch.allocated())
        return left if free is None else min(left, free)
    if free is None:
        return None
    return max(0, free - min(int(RESERVE[0] * free), RESERVE[1]))


class limit:
    '''Context of a run with ``max_disk`` (MB): every scratch file that is
    created inside counts against it (scratch.limit).  Nothing changes
    with max_disk = None.'''

    def __init__(self, max_disk=None):
        self.max_disk = max_disk

    def __enter__(self):
        self.set = self.max_disk is not None
        if self.set:
            self.prev = scratch.limit(int(self.max_disk * 1e6))
        return self

    def __exit__(self, *exc):
        if self.set:
            scratch.limit(self.prev)


def require(nbytes, what, max_disk=None, where=None):
    '''Raise DiskFullError unless budget() holds ``nbytes`` for ``what`` (the
    arrays that a memory plan maps to scratch files).  Returns the budget
    that is left afterwards (None: not known).'''
    where = where or scratch.scratch_dir()
    left = budget(max_disk, where)
    if left is None:
        return None
    if nbytes > left:
        cap = max_disk_bytes(max_disk)
        raise DiskFullError(
            'the memory plan puts %s for %s into scratch files in %s, where %s can be taken (%s free%s, '
            '%s in the scratch files of this run already): give the calculation more memory (max_memory), '
            'more disk (max_disk) or another scratch directory (ZCCSDT_TMPDIR / PYSCF_TMPDIR)'
            % (_gb(nbytes), what, where, _gb(left), _gb(scratch.disk_free(where)),
               '' if cap is None else ', max_disk %s' % _gb(cap), _gb(scratch.allocated())))
    return left - nbytes


def file_bytes(sizes, itemsize, incore_size=None):
    '''Bytes of the parts of one DIIS vector that go to the scratch file:
    those with at least ``incore_size`` elements (default:
    zccsdt.cc.diis.INCORE_SIZE) among the parts with ``sizes`` elements.'''
    from zccsdt.cc import diis
    least = diis.INCORE_SIZE if incore_size is None else incore_size
    return sum(int(n) for n in sizes if n >= least) * int(itemsize)


def _option(what):
    '''The option that keeps the large part ``what`` in or out of DIIS
    (None: there is none, DIIS is not done without that part).'''
    return {'T4': 'diis_t4', 'T3': 'diis_t3', 'l3': 'diis_t3'}.get(what)


@dataclass
class DiisPlan:
    '''Where the DIIS vectors of a run are (plan_diis).'''
    space: int                  # pairs (vector, error vector) in the subspace
    requested: int              # ... asked for (diis_space)
    n_file: int = 0             # pairs whose large parts are in the scratch file
    n_memory: int = 0           # ... in memory
    with_big: bool = True       # the large part (T3, T4) is in the vectors
    pair: int = 0               # bytes of the large parts of one pair
    disk_budget: int = None     # bytes that the disk had for them (None: not known)
    memory_budget: int = None   # bytes that the memory had for them (None: no limit)
    what: str = 'T3'            # name of the large part
    where: str = None           # the scratch file or its directory
    case: str = 'a'             # (a) file, (b) split, (c) fewer, (d) without the large part;
                                # 'm': in memory (incore_complete, or small vectors)

    @property
    def disk(self):
        '''Bytes of the vectors in the scratch file.'''
        return self.n_file * self.pair

    @property
    def memory(self):
        '''Bytes of the large parts of the vectors in memory.'''
        return self.n_memory * self.pair

    @property
    def reduced(self):
        '''Is this less than what was asked for?'''
        return self.space < self.requested or (self.case == 'd')

    def text(self):
        '''One line for the report of a run.'''
        if self.case == 'd':
            return ('%d vectors without %s (it takes its Jacobi step): not even %d pairs of %s fit - disk %s, '
                    'memory %s' % (self.space, self.what, min(FLOOR, self.requested), _gb(self.pair),
                                   _gb(self.disk_budget), _gb(self.memory_budget)))
        if not self.with_big:
            return '%d vectors without %s (%s = False), in memory' % (self.space, self.what, _option(self.what))
        if not self.pair:
            return '%d vectors, in memory' % self.space
        parts = []
        if self.n_file:
            parts.append('%d in %s (%s of disk)' % (self.n_file, self.where or 'a scratch file', _gb(self.disk)))
        if self.n_memory:
            parts.append('%d in memory (%s)' % (self.n_memory, _gb(self.memory)))
        text = '%d vectors with %s, %s per pair: %s' % (self.space, self.what, _gb(self.pair), ', '.join(parts))
        if self.space < self.requested:
            text += '; reduced from the %d asked for (diis_space): disk %s, memory %s' % (
                self.requested, _gb(self.disk_budget), _gb(self.memory_budget))
        elif self.n_file and self.disk_budget is not None:
            text += '; disk budget %s' % _gb(self.disk_budget)
        return text


def plan_diis(big, space=8, memory=None, disk=None, floor=FLOOR, with_big=None, incore=False, what='T3',
              where=None):
    '''DiisPlan for vectors whose large parts take ``big`` bytes (0: nothing
    of them goes to a file), a subspace of at most ``space`` pairs, and the
    two budgets in bytes that are left for them: ``disk`` after the mapped
    arrays of the memory plan and ``memory`` beyond that plan (None: no
    limit known).  The rule is that of the module docstring, (a) to (e).

    Without the large part all vectors are in memory (the solvers do not
    put the smaller parts into the file then).

    ``with_big``: None - the large part is in the vectors unless not even
    the floor fits; True - it is, or DiskFullError; False - it is not.
    ``incore``: all pairs in memory (``incore_complete``); MemoryError if
    ``memory`` - then what the node and the job leave, not max_memory - does
    not hold them.  ``floor``: the smallest subspace with the large part.

    The subspace never shrinks when a budget grows, and the vectors never
    take more than either budget.'''
    space = max(int(space), 0)
    pair = 2 * int(big)
    kw = dict(requested=space, pair=pair, disk_budget=disk, memory_budget=memory, what=what, where=where)
    if with_big is False:
        return DiisPlan(space, with_big=False, case='m', **dict(kw, pair=0))
    if pair == 0 or space == 0:
        return DiisPlan(space, n_memory=space, case='m', **kw)
    if incore:
        if memory is not None and space * pair > memory:
            raise MemoryError('incore_complete: %d DIIS vectors with %s take %s of memory (%s per pair of a '
                              'vector and its error vector); %s are left next to the other arrays of the '
                              'calculation.  Without incore_complete they go to the scratch file as far '
                              'as the disk holds them' % (space, what, _gb(space * pair), _gb(pair), _gb(memory)))
        return DiisPlan(space, n_memory=space, case='m', **kw)
    n_file = space if disk is None else min(space, max(int(disk), 0) // pair)
    n_mem = space - n_file if memory is None else min(space - n_file, max(int(memory), 0) // pair)
    n = n_file + n_mem
    if n >= max(1, min(int(floor), space)):
        case = 'a' if n_file == space else 'b' if n == space else 'c'
        return DiisPlan(n, n_file=n_file, n_memory=n_mem, case=case, **kw)
    if with_big is None:
        return DiisPlan(space, with_big=False, case='d', **kw)
    opt = _option(what)
    raise DiskFullError(
        'DIIS with %s%s needs %s per stored vector and its error vector, and %d of them at '
        'least (%d asked for): %s; the scratch location %s has %s for them and the memory %s.  More disk '
        '(max_disk, another scratch directory: ZCCSDT_TMPDIR / PYSCF_TMPDIR, diis_file) or more memory '
        '(max_memory)%s'
        % (what, ' (%s = True)' % opt if opt else '', _gb(pair), min(int(floor), space), space,
           _gb(min(int(floor), space) * pair), where or '', _gb(disk), _gb(memory),
           ', or DIIS without %s (%s = None or False)' % (what, opt) if opt else ''))


def diis_plan(big, space, max_disk=None, diis_file=None, mapped=0, spare=None, with_big=None,
              incore=False, what='T3', floor=FLOOR, memory_tight=False):
    '''The DiisPlan of a solver: ``big`` bytes for the large parts of a
    vector, ``space`` pairs asked for, ``mapped`` bytes that the memory plan
    of the solver is going to put into scratch files (DiskFullError if those
    do not fit), ``spare`` bytes of the memory budget beyond that plan (None:
    no limit known; with ``incore``: what the node leaves).  With
    ``memory_tight`` - the plan had to map its own arrays - no pair goes to
    memory.'''
    here = directory()
    where = directory(diis_file)
    left = require(mapped, 'T3 and the other arrays of the iterations', max_disk, here) if mapped else \
        budget(max_disk, here)
    if not same_device(here, where):        # (a DIIS file elsewhere: its own file system)
        left = budget(max_disk, where)
    if memory_tight and not incore:
        spare = 0
    return plan_diis(big, space, spare, left, floor, with_big, incore, what, diis_file or where)


def dry_run(nocc, nvir, dtype=complex, max_memory=None, max_disk=None, diis_space=8, diis_t3=None,
            incore_complete=False, mem_mb=2000.0, method='ccsdt'):
    '''What a spin-orbital or spinor CCSDT calculation with ``nocc`` occupied
    and ``nvir`` virtual spin orbitals (spinors) does with ``max_memory`` and
    ``max_disk`` (both in MB, for the whole calculation; None: no limit),
    without the integrals: (memory plan, DiisPlan) - ccsdt_opt.plan_memory
    for the dense packed layout (no symmetry labels: the upper bound of T3)
    and plan_diis.  The free space of this machine is not looked at.
    ``print(disk.describe(*disk.dry_run(...)))`` gives the lines of a report.

    For sizing a job before it is submitted: the memory to ask for (--mem)
    and the local disk.'''
    import numpy as np
    from math import comb
    from zccsdt.cc import ccsdt_opt
    if method != 'ccsdt':
        raise NotImplementedError('dry_run: %s' % method)
    isz = np.dtype(dtype).itemsize
    budget_m = None if max_memory is None else int(max_memory * 1e6)
    plan = ccsdt_opt.plan_memory(nocc, nvir, isz, None if incore_complete else budget_m, mem_mb=mem_mb)
    size = comb(nocc, 3) * comb(nvir, 3)
    big = file_bytes([nocc * nvir, comb(nocc, 2) * comb(nvir, 2), size], isz)
    left = None if max_disk is None else int(max_disk * 1e6) - plan.disk
    if left is not None and left < 0:
        raise DiskFullError('the memory plan puts %s into scratch files; max_disk is %s'
                            % (_gb(plan.disk), _gb(int(max_disk * 1e6))))
    spare = None if budget_m is None else max(0, budget_m - plan.need)
    tight = bool(plan.t3_file or plan.r3_file)
    dplan = plan_diis(big, diis_space, 0 if tight and not incore_complete else spare, left, FLOOR,
                      diis_t3, incore_complete, 'T3')
    if not dplan.with_big and (plan.t3_file if plan.r3_file is None else plan.r3_file):
        # (the residual is then an array of the scratch file of T3)
        if left is not None and left < size * isz:
            raise DiskFullError('T3 and its residual (%s each) do not fit into max_disk next to %s of the '
                                'memory plan' % (_gb(size * isz), _gb(plan.disk)))
        plan.disk += size * isz
    return plan, dplan


def describe(plan, dplan):
    '''Lines for a memory plan of CCSDT and a DiisPlan (dry_run).'''
    where = {False: 'memory', True: 'scratch file', None: 'memory'}
    r3 = plan.t3_file if plan.r3_file is None else plan.r3_file
    return ('memory plan  %s of memory, %s of disk: T3 in %s, its residual in %s%s\n'
            'DIIS         %s; memory %s, disk %s'
            % (_gb(plan.need), _gb(plan.disk), where[bool(plan.t3_file)],
               'the DIIS storage' if r3 and dplan.with_big else where[bool(r3)],
               ', <ab||cd> in slabs' if plan.pp_bytes is not None else '',
               dplan.text(), _gb(dplan.memory), _gb(dplan.disk)))
