'''Spin-orbital CCSDT working on packed T3 only.

Same equations as ``zccsdt.cc.ccsdt`` (the ported reference, T1-dressed
formalism), reorganized so that no full o^3 v^3 array and no full n^4
integral tensor is ever formed:

* T1 dressing is done block by block (at most o v^4 work, for vvvv).
* T3 is accessed through half-expanded batches T[m, P, d, Q]
  (t3pack.gather_half) over occupied pairs P:
    pass 1: T3 -> W_vvvo, W_vooo dressing, T3 -> T1 and T3 -> T2;
    pass 2: ring + drive terms accumulated in one half-expanded buffer and
            reduced (antisymmetrized + packed) once.
* pp ladder: batches of occupied triples, F_vv folded into W(ab,de).
* hh ladder and F_oo: one C(o,3) x C(o,3) operator applied to packed T3.
* The dtype of the integrals is kept (real stays real).

When memory is short (MemoryPlan, plan_memory) T3 and its residual are mapped
arrays of a scratch file, and so is the bare packed <ab||de>, whose dressed
form is then built, used and dropped in slabs of rows (_pp_sweep).

See docs/factorization.md for the derivation of each step.
'''

from dataclasses import dataclass

import numpy as np
from pyscf import lib

from zccsdt.blas import gemm
from zccsdt.cc import ccsd, t3pack, t3kern, t3engine, spin_imds
from zccsdt.cc.spintensor import SpinTensor, einsum as s_einsum, pack_pairs, pair_space
from zccsdt.timing import timer

einsum = lib.einsum
STORED = ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv', 'vvvv')


@dataclass
class MemoryPlan:
    '''Where the large arrays live and how the passes are sliced
    (plan_memory).  The default is everything in memory.'''
    t3_file: bool = False       # T3 in a scratch file
    r3_file: bool = None        # its residual in the scratch file (None: as T3)
    pp_file: bool = False       # bare packed <ab||de> in a scratch file
    pp_bytes: int = None        # dressed rows of the packed <ab||de> held at a
                                # time (None: all of them)
    pp_order: str = 'slab'      # sliced pp ladder: for each slab of rows all
                                # occupied triples ('slab'), or for each batch
                                # of triples all slabs ('batch': T3 is read once)
    mem_mb: float = None        # batch buffers of the T3 passes (None: Options.mem_mb)
    need: int = 0               # bytes of ordinary memory the plan takes


_STEPS = [0]                # CCSDT steps taken by this process

# The o v^3-sized arrays of a step in units of f = o v^3 elements (o, v: spin
# orbitals): up to the end of pass 1, afterwards, and what a run that is not
# short of memory keeps between the iterations on top of both.
OVVV_DENSE = (3.5, 1.75, 0.0)
# U/R-form integrals: spin-blocked where the algebra is (dressed ovvv 3/8 f,
# packed 3/16 f), dense where the engine takes dense operands (pass-1 operand
# and result, W_vvvo: f / 2 each); kept: the bare <ab||ei> and the array of
# W_vvvo
OVVV_SPIN = (2.0, 1.25, 1.0)


def overhead():
    '''Bytes that the threaded BLAS (work buffers per thread) and the
    allocator keep once a CCSDT step has run: about 0.7 GB with 8 threads.'''
    import os
    try:
        n = int(os.environ.get('OMP_NUM_THREADS') or len(os.sched_getaffinity(0)))
    except (ValueError, AttributeError):
        n = 8
    return 300_000_000 + 50_000_000 * n


def cached_bytes(eris):
    '''Bytes of ordinary memory that an earlier CCSDT step has left
    allocated: the arrays kept on the integral object between calls (packed
    bare blocks, work buffers) and the buffers of the BLAS threads.'''
    from zccsdt.scratch import is_mapped
    d = eris.__dict__
    arrs = list(d.get('_bare_Q', ())[1:]) + [d.get('_bare_vvvo')]
    for bufs in d.get('_buffers', {}).values():
        arrs += list(bufs)
    for x in d.get('_bare_packed', {}).values():        # U/R form: SpinTensors
        arrs += list(getattr(x, 'blocks', {}).values())
    seen = {id(x): x for x in arrs                      # (blocks shared between spins once)
            if isinstance(x, np.ndarray) and not is_mapped(x)}
    # (and the work array of the spin-blocked engine, which stays allocated)
    return sum(x.nbytes for x in seen.values()) + t3engine.workspace_bytes() \
        + (overhead() if _STEPS[0] else 0)


def release_cached(eris):
    '''Drop what a CCSDT step keeps for the next one (cached_bytes()): the
    arrays on the integral object and the work array of the engine.  For
    what follows the iterations - a perturbative correction plans its own
    memory - and for a rerun they are formed again.'''
    d = eris.__dict__
    for key in ('_buffers', '_bare_Q', '_bare_vvvo', '_bare_packed'):
        d.pop(key, None)
    t3engine.release_workspace()
    try:                        # (glibc: give the freed pages back, they count as resident)
        import ctypes
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except (OSError, AttributeError):
        pass


def sliceable(eris, eng):
    '''Can the dressed packed <ab||de> be used in slabs of rows?  Yes for the
    dense engine and, with the C kernels, for U/R-form integrals; not for
    dense integrals on spin-grouped maps.'''
    if _is_spin_eris(eris):
        return bool(getattr(eng, 'use_c', False))
    return eng.vm.spins is None


def plan_for(eris, dtype, opts):
    '''MemoryPlan of a run with the solver options ``opts``: everything in
    memory with ``incore_complete``, else what fits in ``max_memory`` (MB for
    the whole process; default: 80% of what is available).'''
    from zccsdt import memory
    from zccsdt.scratch import is_mapped
    om, vm = t3pack.maps_for(eris)
    isz = np.dtype(dtype).itemsize
    eng = engine_for(eris, dtype, flip=getattr(opts, 'spin_flip', False))
    spin = _is_spin_eris(eris)
    kw = dict(mem_mb=opts.mem_mb, t3_bytes=eng.size * isz, sliceable=sliceable(eris, eng),
              engine=eng, ovvv=OVVV_SPIN if spin else OVVV_DENSE)
    if spin:        # (the pair blocks are dressed in place, block by block)
        kw['dress_bytes'] = 0
    if spin:        # bare pair blocks (alpha-alpha shared with beta-beta for RHF)
        blocks = {id(b): b for b in eris.vvvv_blocks}
        kw['pp_bytes'] = sum(b.nbytes for b in blocks.values())
        kw['row_bytes'] = max([b.shape[1] * b.itemsize for b in blocks.values()] + [1])
        where = 'file' if all(is_mapped(b) or b.size == 0 for b in blocks.values()) \
            else 'memory'
    else:
        where = 'file' if is_mapped(getattr(eris, '_vvvv_pp', None)) else 'memory'
    if opts.incore_complete:
        return plan_memory(om.n, vm.n, isz, pp_where=where, **kw)
    # arrays kept on the integral object by an earlier run are part of what
    # the plan accounts for: they do not reduce the budget
    cached = cached_bytes(eris)
    budget = memory.budget(opts.max_memory)
    limit = budget if opts.max_memory is not None else memory.available()
    return plan_memory(om.n, vm.n, isz, None if budget is None else budget + cached,
                       None if limit is None else limit + cached, where, **kw)


def plan_memory(nocc, nvir, itemsize=16, budget=None, limit=None, pp_where='memory',
                mem_mb=2000.0, t3_bytes=None, pp_bytes=None, sliceable=True, row_bytes=None,
                engine=None, ovvv=OVVV_DENSE, dress_bytes=None):
    '''MemoryPlan for ``budget`` bytes of ordinary memory beyond what the
    integral object holds already (None: no limit, everything in memory).

    The memory of an iteration, with f = o v^3 elements (o, v: spin orbitals
    or spinors) of ``itemsize`` bytes:

      ovvv-sized  at most 3.5 f at a time: the packed bare ovvv (f / 2), the
               dressed ovvv, the pass-1 operand and result and W_vvvo (f / 2
               each); 1.75 f while the batch buffers are alive
      buffers  two thirds of mem_mb for the half-expanded T3 batches of a
               pass (at least one occupied pair: f)
      pp       v^4 / 4 elements each for the bare packed <ab||de> and its
               dressed form
      T3       2 C(o,3) C(v,3): amplitudes and residual
      small    eight o^2 v^2-sized arrays and the occupied operator, C(o,3)^2
      overhead what the threaded BLAS and the allocator keep (overhead())

    (the DIIS vectors are in the scratch file already).  These are the numbers
    of dense integrals and the dense engine.  ``engine`` (a T3 engine) gives
    the buffers of another one: the spin-blocked engine has one work array
    for all passes, which stays allocated (its buffer_bytes(mem_mb), next to
    everything else), and works with any buffer size.  ``ovvv`` gives the
    ovvv-sized arrays of another form of the integrals (OVVV_SPIN for the
    U/R form: 2 f up to the end of pass 1 and 1.25 f afterwards, and one f
    more that is kept between the iterations when memory is not short), and
    ``dress_bytes`` the transient of dressing the whole pp operator (default:
    that of the dense operator; 0 for the U/R form).

    If that exceeds the
    budget, the buffers shrink.  The bare pp operator goes to a scratch file
    (``pp_where``: it is in 'memory' now, or in a 'file' already) when both
    of its forms do not fit, and its dressed rows are held ``pp_bytes`` at a
    time if they do not fit either.  T3 and its residual stay in memory if
    they fit next to an eighth of the dressed operator (they are passed over
    many times per iteration, the operator once).  Else T3 goes to the
    scratch file first, and the residual only if it does not fit alone: the
    passes only read T3, whose pages are given up at no cost, while pages of
    the residual, which every pass adds to, would have to be written to the
    file and read back.  ``t3_bytes`` / ``pp_bytes`` / ``row_bytes``: sizes of T3, of the
    bare operator and of its longest row if not those of the dense layout
    (spin-blocked forms); ``sliceable``: False if the pp operator has to stay
    in memory (ccsdt_opt.sliceable).  Raises
    MemoryError if the ovvv-sized part and the smallest buffers do not fit
    in ``limit`` (default: the budget).'''
    o, v = nocc, nvir
    nvp = v * (v - 1) // 2
    f = o * v ** 3 * itemsize
    row = max(nvp * itemsize, 1) if row_bytes is None else row_bytes
    pp = nvp * nvp * itemsize if pp_bytes is None else pp_bytes
    nto, nvt = o * (o - 1) * (o - 2) // 6, v * (v - 1) * (v - 2) // 6
    t3 = nto * nvt * itemsize if t3_bytes is None else t3_bytes
    # o^2 v^2-sized arrays, the occupied operator C(o,3)^2 (and a copy of its
    # blocks), and what the BLAS threads and the allocator keep
    small = 8 * o * o * v * v * itemsize + 2 * nto * nto * itemsize + overhead()
    kept = bool(getattr(engine, 'kept_buffers', False))
    f_before, f_beside, f_cached = ovvv

    def bufs(mem):
        if engine is not None:
            return int(engine.buffer_bytes(mem))
        return int(max(mem * 1e6 * 2 / 3, f))

    def floor(mem, extra=0, cached=0.0):
        # the ovvv-sized arrays before the T3 passes, or half of them next to
        # the buffers (and whatever else is alive then: ``extra``).  A work
        # array that is kept is there at all times, and the blocks of the
        # occupied operator (mem / 4) come on top of it.
        a, b = int((f_before + cached) * f), int((f_beside + cached) * f)
        if kept:
            return small + bufs(mem) + max(a, b + max(extra, int(mem * 250_000)))
        return small + max(a, b + max(bufs(mem), extra))

    plan = MemoryPlan(mem_mb=mem_mb, pp_file=pp_where == 'file')
    # the whole-operator dressing has a transient of its own
    dress = min(2_000_000_000, 2 * pp) + f // 2 if dress_bytes is None else dress_bytes
    plan.need = floor(mem_mb, dress, f_cached) + 2 * t3 + pp
    if budget is None or plan.need <= budget:
        return plan
    hard = budget if limit is None else limit
    if pp_where == 'memory' and sliceable:
        budget, hard = budget + pp, hard + pp       # the bare operator can leave memory
    mem_min = 1.5 * f / 1e6 if engine is None else engine.mem_min()

    def base(mem):
        return floor(mem) + (0 if sliceable else pp)

    # the thinnest slab of the dressed operator: v of its rows (those of one
    # first index at most)
    row_min = min(v * row, pp) if sliceable else 0
    if base(mem_min) + row_min > hard:
        raise MemoryError('CCSDT with %d occupied and %d virtual spin orbitals needs about '
                          '%.1f GB of memory for its o v^3 arrays and work buffers; %.1f GB '
                          'are available (max_memory, or what the node and the memory limit '
                          'of the job leave)'
                          % (o, v, (base(mem_min) + row_min) / 1e9, hard / 1e9))
    # work buffers: an eighth of the budget if that is less than asked for
    mem = min(mem_mb, max(budget / 8 * 1.5 / 1e6, mem_min))
    if base(mem) + row_min > budget:
        mem = mem_min
    plan.mem_mb = mem
    # T3 stays in memory if at least an eighth of the dressed pp operator
    # fits beside it: squeezing it in with thin slabs costs more than paging it
    beside = min(max(pp // 8, row_min), pp) if sliceable else 0
    plan.t3_file = base(mem) + 2 * t3 + beside > budget
    plan.r3_file = plan.t3_file and base(mem) + t3 + beside > budget
    held = (0 if plan.t3_file else t3) + (0 if plan.r3_file else t3)
    plan.need = base(mem) + held
    if not sliceable:
        return plan
    rest = max(budget - plan.need, 0)
    if pp_where == 'memory' and 2 * pp <= rest:
        plan.pp_bytes = pp                          # bare and dressed in memory
        plan.need += 2 * pp
    elif pp <= rest:
        plan.pp_file = True                         # all dressed rows at once
        plan.pp_bytes = pp
        plan.need += pp
    elif plan.t3_file:
        # T3 in the file: the ladder runs batch-wise, and the rows of the
        # operator, kept in a scratch file, are passed over once per batch of
        # triples.  So what is left goes to the batch buffers first (up to
        # three times what was asked for) and the slabs stay thin.
        plan.pp_file = True
        plan.pp_order = 'batch'
        plan.pp_bytes = int(min(max(rest // 8, row_min), pp))
        if not kept:
            room = budget - small - int(f_beside * f) - plan.pp_bytes - rest // 8 - held
            want = min(max(room, bufs(mem)), 3 * bufs(mem_mb))
            if want > bufs(mem):
                plan.mem_mb = mem = want * 1.5 / 1e6
        plan.need = base(mem) + plan.pp_bytes + held
    else:
        plan.pp_file = True
        plan.pp_bytes = int(max(rest // 2, row_min))
        plan.need += plan.pp_bytes
        # slab-wise, T3 is gathered again for every slab; batch-wise, the
        # rows of the operator are kept in a scratch file and read once per
        # batch of triples, which is cheap only while the page cache can
        # hold them
        nslab = len(pair_slabs(nvp, row, plan.pp_bytes))
        per_triple = (nvp + plan.pp_bytes // row) * v * itemsize
        nbatch = -(-nto // max(1, int(mem * 1e6 * 2 / 3 // max(per_triple, 1))))
        plan.pp_order = 'batch' if pp <= budget // 4 and 2 * pp * nbatch < 3 * t3 * nslab \
            else 'slab'
    return plan


def place_vvvv(eris, plan):
    '''Move the bare packed <ab||de> of the integral object to a scratch
    file if the plan wants it there and it is still in memory.'''
    from zccsdt.scratch import Scratch, is_mapped, written, unmap
    if not plan.pp_file:
        return
    spin = _is_spin_eris(eris)
    blocks = list(eris.vvvv_blocks) if spin else [getattr(eris, '_vvvv_pp', None)]
    todo = {id(V): V for V in blocks if V is not None and V.size and not is_mapped(V)}
    if not todo:
        return
    scr = Scratch(prefix='zccsdt_vvvv_', what='the packed <ab||de> integrals',
                  hint=' or give the calculation more memory (max_memory)')
    moved = {}
    for n, (key, V) in enumerate(todo.items()):
        M = scr.create('vvvv%d' % n, V.shape, V.dtype)
        step = max(1, (1 << 28) // max(V[0].nbytes, 1))
        for q in range(0, V.shape[0], step):
            M[q:q + step] = V[q:q + step]
            written(M, q * V.shape[1], min(q + step, V.shape[0]) * V.shape[1])
            unmap(M, q * V.shape[1], min(q + step, V.shape[0]) * V.shape[1])
        moved[key] = M
    if spin:        # (blocks shared between spins stay shared)
        eris.vvvv_blocks = [moved.get(id(V), V) for V in blocks]
    else:
        eris._vvvv_pp = moved[id(blocks[0])]
        eris.__dict__.pop('_vvvv_blocks', None)     # cached views of the old array
    # keeps the file(s) open
    eris.__dict__.setdefault('_scratches', []).append(scr)


def pair_slabs(nrow, row_bytes, max_bytes=None):
    '''Row ranges (q0, q1) covering nrow rows of row_bytes each, at most
    max_bytes per range (None: one range).'''
    if max_bytes is None or nrow == 0:
        return [(0, nrow)]
    n = max(1, int(max_bytes // max(row_bytes, 1)))
    return [(q, min(q + n, nrow)) for q in range(0, nrow, n)]


# ---------------------------------------------------------------- integrals

def get_block(eris, key):
    '''Any <pq||rs> block (key like 'vovv') from the stored PySCF-convention
    blocks, via pair antisymmetry and hermiticity <pq||rs> = <rs||pq>^*.'''
    for conj in (False, True):
        k = key[2:] + key[:2] if conj else key
        for swap_bra in (False, True):
            for swap_ket in (False, True):
                s = k
                if swap_bra:
                    s = s[1] + s[0] + s[2:]
                if swap_ket:
                    s = s[:2] + s[3] + s[2]
                if s not in STORED:
                    continue
                b = np.asarray(getattr(eris, s))
                if swap_bra:
                    b = -b.transpose(1, 0, 2, 3)
                if swap_ket:
                    b = -b.transpose(0, 1, 3, 2)
                if conj:
                    b = b.transpose(2, 3, 0, 1).conj()
                return b
    raise KeyError(key)


def _contract_slot(b, M, p):
    '''b with its slot p contracted against M[source, target], using
    contiguous GEMMs (no moveaxis copies).'''
    sh = b.shape
    if p == 3:
        return (b.reshape(-1, sh[3]) @ M).reshape(sh[:3] + (M.shape[1],))
    if p == 0:
        return (M.T @ b.reshape(sh[0], -1)).reshape((M.shape[1],) + sh[1:])
    lead = int(np.prod(sh[:p]))
    out = np.matmul(M.T, b.reshape(lead, sh[p], -1))
    return out.reshape(sh[:p] + (M.shape[1],) + sh[p + 1:])


def dress_block(eris, t1, key):
    '''T1-dressed block of <pq||rs>: a virtual bra index a picks up
    -sum_m t1[m,a] <m..||..>, an occupied ket index i picks up
    +sum_e t1[i,e] <..||..e>.  Other indices are unchanged.

    The dressing is a product of one-index transformations, so it is applied
    slot by slot (ket slots first): each step combines the blocks that differ
    only in the source space of that slot.'''
    alt = {}
    for p, sp in enumerate(key):
        if p < 2 and sp == 'v':
            alt[p] = ('o', -t1)                      # M[m, a]
        elif p >= 2 and sp == 'o':
            alt[p] = ('v', t1.T)                     # M[e, i]
    choices = [(key[p], alt[p][0]) if p in alt else (key[p],) for p in range(4)]
    state = {''.join(c): get_block(eris, ''.join(c))
             for c in np.ndindex(*[len(x) for x in choices])
             for c in [tuple(choices[p][i] for p, i in enumerate(c))]}
    for p in sorted(alt, reverse=True):
        src, M = alt[p]
        new = {}
        for k, blk in state.items():
            if k[p] != key[p]:
                continue
            k_alt = k[:p] + src + k[p + 1:]
            new[k] = blk + _contract_slot(state[k_alt], M, p)
        state = new
    return state[key]


def vvvv_packed(eris, vm):
    '''Undressed <ab||de> packed over a<b, d<e as the list of diagonal pair
    blocks of ``vm`` (one block with everything for maps without spins;
    alpha-alpha, alpha-beta, beta-beta blocks otherwise, the off-diagonal
    blocks vanish by spin conservation).  Cached on the integral object.  The
    stored packed form (generator / HDF5 attribute _vvvv_pp) is in triu
    (lexicographic) pair order.'''
    if hasattr(type(eris), '__dataclass_fields__') and 'vvvv_blocks' in eris.__dataclass_fields__:
        return eris.vvvv_blocks                       # U/R-form integrals
    cached = getattr(eris, '_vvvv_blocks', None)
    if cached is not None and cached[0] is vm:
        return cached[1]
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    n = vm.n
    lex = getattr(eris, '_vvvv_pp', None)
    if vm.spins is None and lex is not None:
        blocks = [lex]
    else:
        blocks = []
        for lo, hi in vm.blocks:
            a, b = md[lo:hi], ne[lo:hi]
            if lex is not None:
                perm = a * n - a * (a + 1) // 2 + (b - a - 1)       # triu index
                blk = lex[perm[:, None], perm]
            else:
                blk = np.asarray(eris.vvvv)[a[:, None], b[:, None], a, b]
            blocks.append(np.ascontiguousarray(blk))
        if vm.spins is None:
            try:
                eris._vvvv_pp = blocks[0]
            except AttributeError:
                pass
    try:
        eris._vvvv_blocks = (vm, blocks)
    except AttributeError:
        pass
    return blocks


def bra_dressed_vvvo_pp(eris, t1, om, vm):
    '''<ab||ei> with only the bra (a, b) T1-dressed, packed over a<b:
    B = <ab||ei> - (L[a,b] - L[b,a]) + sum_{m<n} tau1[mn,ab] <mn||ei>,
    L[a,b,e,i] = sum_m t1[m,a] <mb||ei>.  Shape (npair_v, nvir, nocc).'''
    nocc, nvir = t1.shape
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    dtype = np.result_type(t1, eris.oovv)
    with timer('bv.gather'):
        # <ab||ei> = -<ie||ab>^* from the packed bare ovvv [i, e, (a<b)]
        # (loop order (e, i, Q): contiguous reads)
        G = _bare_Q(eris, om, vm)[0]
        B = np.empty((vm.npair, nvir, nocc), dtype=dtype)
        t3pack.axpby(B.transpose(1, 2, 0), np.asarray(G, dtype=dtype).transpose(1, 0, 2),
                     -1.0, 0.0)
        if np.iscomplexobj(B):
            np.conjugate(B, out=B)
        ovvo = np.ascontiguousarray(get_block(eris, 'ovvo'), dtype=dtype)
    with timer('bv.L'):
        L = gemm(np.asfortranarray(t1.T), ovvo.reshape(nocc, -1))      # [a, (b, e, i)]
        t3pack.pair_antisym_add(B, L, vm, (nvir * nvir * nocc, nvir * nocc, nocc, 1),
                                alpha=-1.0)
    del L
    tau = _tau1_pairs(t1, om, vm)                                    # (mn, ab)
    oovo = np.ascontiguousarray(get_block(eris, 'oovo')[mm, nn], dtype=dtype)
    gemm(np.asfortranarray(tau.T), oovo.reshape(om.npair, -1), B.reshape(vm.npair, -1),
         beta=1.0)
    return B


def _tau1_pairs(t1, om, vm):
    '''tau1[(m<n), (a<b)] = t1[m,a] t1[n,b] - t1[n,a] t1[m,b].'''
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    return t1[mm][:, md] * t1[nn][:, ne] - t1[nn][:, md] * t1[mm][:, ne]


def wvvvo_packed(Bv, Vd, t1, t2, fov, ge, om, vm, xterm=None, inplace=False,
                 oovo_extra=None):
    '''W_vvvo[Q(a<b), e, i] of the T3 drive term (without the T3 dressing):
    the fully T1-dressed <ab||ei> = Bv + sum_f ~<ab||ef> t1[i,f] (Bv: bra-dressed
    vvvo, Vd: dressed packed vvvv), plus the T2 terms with the dressed oovo,
    ovvv and Fock.  oovo_extra (P, e, i) is added to the packed oovo of the
    t2 x oovo GEMM (used to remove the T2 part when Vd includes it).'''
    nocc, nvir = t1.shape
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    W = Bv if inplace else np.array(Bv, order='C', copy=True)
    # sum_f ~<ab||ef> t1[i,f]: sparse in the packed (e<f) index; GEMMs per spin
    # block for spin-grouped pairs, a direct kernel otherwise
    # (Vd = None: left to the caller, slab by slab, see _pp_sweep)
    with timer('wv.vvvv_t1'):
        if Vd is None:
            pass
        elif vm.spins is not None:
            spin_imds.vvvv_t1_blocks(W, Vd, t1, om, vm)
        else:
            for h, (lo, hi) in enumerate(vm.blocks):
                if hi > lo:
                    t3pack.pair_t1_contract(W[lo:hi], Vd[h], t1, vm, block=h)
    # + 0.5 sum_mn ~<mn||ei> t2[mn,ab] - sum_m f[m,e] t2[m,i,a,b]
    #   = sum_{m<n} t2[mn,ab] (~oovo[mn,e,i] - f[m,e] d[n,i] + f[n,e] d[m,i])
    with timer('wv.oovo'):
        t2mn_pp = t2[mm, nn][:, md, ne]
        G = np.array(ge['oovo'][mm, nn], dtype=W.dtype, order='C', copy=True)  # [P, e, i]
        P = np.arange(om.npair)
        G[P, :, nn] -= fov[mm]
        G[P, :, mm] += fov[nn]
        if oovo_extra is not None:
            G += oovo_extra
        gemm(np.asfortranarray(t2mn_pp.T), G.reshape(om.npair, -1),
             W.reshape(vm.npair, -1), beta=1.0)
    # - P(ab) sum_{mf} ~<mb||ef> t2[mi,af]: X[(a,i),(b,e)] = t2[a,i,m,f] ovvv[m,f,b,e]
    with timer('wv.t2ovvv'):
        if xterm is not None:          # callable adding the term to W (spin-blocked)
            xterm(W)
        else:
            # in batches of a: X[(a,i),(b,e)] = sum_m A[(a,i),(m,f)] ovvv[m][(b,e),f]^T,
            # one GEMM per m on ovvv as stored (an eighth of the range of a at
            # a time: two batch-sized arrays instead of two ovvv-sized ones)
            A = np.ascontiguousarray(t2.transpose(2, 1, 0, 3), dtype=W.dtype).reshape(
                nvir * nocc, nocc * nvir)
            ovvv = np.ascontiguousarray(ge['ovvv'], dtype=W.dtype)           # [m, b, e, f]
            Wf = W.reshape(vm.npair, nvir * nocc)
            nb = max(1, nvir // 8)
            X = np.empty((nb * nocc, nvir * nvir), dtype=W.dtype)
            Y = np.empty((nb, nvir, nvir, nocc), dtype=W.dtype)
            for x0 in range(0, nvir, nb):
                n = min(nb, nvir - x0)
                Xn = X[:n * nocc]
                for m in range(nocc):
                    gemm(A[x0 * nocc:(x0 + n) * nocc, m * nvir:(m + 1) * nvir],
                         ovvv[m].reshape(nvir * nvir, nvir).T, Xn, beta=1.0 if m else 0.0)
                # Y[a, b, e, i] = X[a, i, b, e]
                t3pack.axpby(Y[:n], Xn.reshape(n, nocc, nvir, nvir).transpose(0, 2, 3, 1),
                             1.0, 0.0)
                t3pack.pair_antisym_batch_add(Wf, Y[:n].reshape(n, nvir, nvir * nocc), vm, x0,
                                              alpha=-1.0)
            del X, Y, A
    return W


def dressed_vvoo_pp(eris, t1, Bv, om, vm):
    '''Fully T1-dressed <ab||ij> packed as [(i<j), (a<b)], except for the
    sum_ef ~<ab||ef> t1[i,e] t1[j,f] term, which is added in the ladder GEMM
    together with t2 (see ccsd_residual_packed):
      bra-dressed <ab||ij> + sum_e Bv[ab,e,j] t1[i,e] - (i <-> j).'''
    nocc, nvir = t1.shape
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    dtype = np.result_type(t1, eris.oovv)
    # bra-dressed <ab||ij> = <ab||ij> - P(ab) t1[m,a] <mb||ij> + tau1 <mn||ij>
    vvoo = np.ascontiguousarray(get_block(eris, 'vvoo'), dtype=dtype)   # [a,b,i,j]
    G = np.array(vvoo[md, ne][:, mm, nn], order='C', copy=True)         # (Q, P)
    ovoo = np.ascontiguousarray(get_block(eris, 'ovoo')[:, :, mm, nn], dtype=dtype)
    L = gemm(np.asfortranarray(t1.T), ovoo.reshape(nocc, -1))           # [a, (b, P)]
    t3pack.pair_antisym_add(G.reshape(vm.npair, om.npair, 1), L, vm,
                            (nvir * om.npair, om.npair, 1, 0), alpha=-1.0)
    oooo = np.ascontiguousarray(np.asarray(eris.oooo)[mm, nn][:, mm, nn], dtype=dtype)
    gemm(np.asfortranarray(_tau1_pairs(t1, om, vm).T), oooo, G, beta=1.0)
    # ket dressing with Bv: Y[Q, i, j] = sum_e Bv[Q,e,j] t1[i,e]
    Y = np.matmul(t1, Bv)                                                # (Q, i, j)
    G += Y[:, mm, nn] - Y[:, nn, mm]
    return np.ascontiguousarray(G.T)                                     # (P, Q)


def ccsd_residual_packed(t1, t2, fdr, ge, Vd, vvoo_pp, Feff_vv, Feff_oo,
                         Weff_ovvo, om, vm, r1_vovv=None, ladder_t2=None, t2pp=None,
                         goovv_pp=None, tau=None, ladder=None):
    '''CCSD residuals in the T1-dressed formalism (CCSD with T1 = 0 and the
    dressed Hamiltonian; validated against the determinant-space oracle):
      R1 = f_ai + f_kc t_ikac + 1/2 g_akcd t_ikcd - 1/2 g_klic t_klac
      R2 = g_abij + 1/2 g_abcd (t + tau1)_ijcd + 1/2 W_klij t_klab
           + P(ab) F_bc t_ijac - P(ij) F_kj t_ikab + P(ij)P(ab) W_kbcj t_ikac
    with W_klij = g_klij + 1/2 g_klcd t_ijcd, W_kbcj = g_kbcj + 1/2 g_klcd t_jlbd,
    F = Feff (the same effective Fock blocks as in the T3 equation).
    R2 is returned packed as [(i<j), (a<b)].  ladder_t2 = (H, t2pp): the blocks
    Vd[h] include + t2pp[P_h, Q_h]^T H[h] (the T3 ladder operator), whose
    contribution to the ladder is subtracted.  t2pp, goovv_pp (packed (P, Q))
    and tau (tau1 packed) may be passed in when already formed.  ``ladder``
    (packed (P, Q)): the ladder term sum_{d<e} ~<ab||de> (t + tau1)[ij,de],
    computed by the caller (Vd is then not used).'''
    nocc, nvir = t1.shape
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    o, v = slice(0, nocc), slice(nocc, None)
    if t2pp is None:
        t2pp = np.ascontiguousarray(t2[mm, nn][:, md, ne])              # (P, Q)
    if goovv_pp is None:
        goovv_pp = np.ascontiguousarray(ge['oovv'][mm, nn][:, md, ne])
    if tau is None:
        tau = _tau1_pairs(t1, om, vm)
    t2mn = t2[mm, nn]                                                    # (P, a, b)
    # R1
    R1 = fdr[v, o].T + einsum('kc,ikac->ia', fdr[o, v], t2)
    if r1_vovv is None:
        R1 += einsum('akR,ikR->ia', ge['vovv'][:, :, md, ne], t2[:, :, md, ne])
    else:
        R1 += r1_vovv
    R1 -= einsum('Pic,Pac->ia', ge['ooov'][mm, nn], t2mn)
    # R2: driving + ladder (with the t1 t1 part of the vvoo dressing) + hh;
    # ladder as Vd[h] tt^T, one pass over each distinct (possibly shared) block
    R2 = np.array(vvoo_pp, order='C', copy=True)
    npo = R2.shape[0]
    tt = t2pp + tau
    done = set()

    def t2_part(k):         # the T2 part of the ladder operator, taken out again
        if ladder_t2 is not None and ladder_t2[0][k] is not None:
            klo, khi = vm.blocks[k]
            plo, phi = om.blocks[k]
            C1 = tt[:, klo:khi] @ ladder_t2[0][k].T                       # (P, P_k)
            R2[:, klo:khi] -= C1 @ ladder_t2[1][plo:phi, klo:khi]

    if ladder is not None:
        R2 += ladder
        for k, (klo, khi) in enumerate(vm.blocks):
            if khi > klo:
                t2_part(k)
    for h, (lo, hi) in enumerate(vm.blocks):
        if hi == lo or h in done or ladder is not None:
            continue
        hs = [k for k in range(h, len(vm.blocks))
              if Vd[k] is Vd[h] and vm.blocks[k][1] > vm.blocks[k][0]]
        done.update(hs)
        B = np.ascontiguousarray(np.concatenate([tt[:, slice(*vm.blocks[k])] for k in hs],
                                                axis=0), dtype=R2.dtype)  # (k P, nQ)
        X = gemm(Vd[h], B.T)                                             # (nQ, k P)
        for j, k in enumerate(hs):
            klo, khi = vm.blocks[k]
            t3pack.axpby(R2[:, klo:khi], X[:, j * npo:(j + 1) * npo].T)
            t2_part(k)
    Wkl = np.ascontiguousarray(ge['oooo'][mm, nn][:, mm, nn])           # (kl, ij)
    gemm(goovv_pp, t2pp.T, Wkl, beta=1.0)
    gemm(np.asfortranarray(Wkl.T), t2pp, R2, beta=1.0)
    # P(ab) F_bc t_ijac on the packed occupied pairs: Y[P, a, b] = t[P, a, c] F[b, c]
    Y = gemm(np.ascontiguousarray(t2mn).reshape(-1, nvir), np.asarray(Feff_vv).T)
    Y = Y.reshape(len(mm), nvir, nvir)
    R2 += Y[:, md, ne] - Y[:, ne, md]
    # - P(ij) F_kj t_ikab on the packed virtual pairs: Z[i, j, Q] = t[i, k, Q] F[k, j]
    Z = np.einsum('ikQ,kj->ijQ', t2[:, :, md, ne], Feff_oo, optimize=True)
    R2 -= Z[mm, nn] - Z[nn, mm]
    # P(ij) P(ab) W_kbcj t_ikac: X[(i,a), (j,b)] = t[i,k,a,c] W2[k,b,c,j]
    W2 = 0.5 * (ge['ovvo'] + Weff_ovvo)                                  # [k,b,c,j]
    A = np.ascontiguousarray(t2.transpose(0, 2, 1, 3)).reshape(nocc * nvir, -1)  # (ia,kc)
    B = np.ascontiguousarray(W2.transpose(0, 2, 3, 1)).reshape(nocc * nvir, -1)  # (kc,jb)
    X = gemm(A, B)
    ia = mm[:, None] * nvir + md[None, :]
    ib = mm[:, None] * nvir + ne[None, :]
    ja = nn[:, None] * nvir + md[None, :]
    jb = nn[:, None] * nvir + ne[None, :]
    R2 += X[ia, jb] - X[ja, ib] - X[ib, ja] + X[jb, ia]
    return R1, R2


def dressed_vvvv_pp(eris, t1, om, vm):
    '''Packed T1-dressed <ab||de> (list of diagonal pair blocks, see
    vvvv_packed) without the full v^4 array:
    ~<ab||de> = <ab||de> - Y[a,b,de] + Y[b,a,de] + Z[a,b,de],
    Y[x,y,Q] = sum_m t1[m,x] <my||Q>,  Z[a,b,Q] = sum_mn t1[m,a] t1[n,b] <mn||Q>.'''
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    nocc, nvir = t1.shape
    V = vvvv_packed(eris, vm)
    ovvv_Q, oovv_pp = _bare_Q(eris, om, vm)                              # (m, y, Q), (P, Q)
    tau = _tau1_pairs(t1, om, vm)                                        # (mn, ab)
    out = []
    for h, (lo, hi) in enumerate(vm.blocks):
        nq = hi - lo
        blk = np.array(V[h], dtype=np.result_type(V[h], ovvv_Q, t1), order='C', copy=True)
        if nq:
            a, b = md[lo:hi], ne[lo:hi]
            x0, x1 = int(min(a.min(), b.min())), int(max(a.max(), b.max())) + 1
            G = np.ascontiguousarray(ovvv_Q[:, :, lo:hi]).reshape(nocc, -1)
            # blk[(a<b), Q] -= Y[a,b,Q] - Y[b,a,Q], Y built in batches of x
            nb = max(1, int(2e9 // (nvir * nq * blk.itemsize)))
            for xb in range(x0, x1, nb):
                xe = min(xb + nb, x1)
                Y = gemm(np.ascontiguousarray(t1[:, xb:xe].T), G).reshape(xe - xb, nvir, nq)
                t3pack.pair_antisym_batch_add(blk, Y, vm, xb, alpha=-1.0, block=h)
                del Y
            gemm(np.ascontiguousarray(tau[:, lo:hi].T), np.ascontiguousarray(oovv_pp[:, lo:hi]),
                 blk, beta=1.0)
        out.append(blk)
    return out


def _bare_Q(eris, om, vm):
    '''Bare <my||Q> as [m, y, Q] and <mn||Q> as [P, Q], packed over the
    virtual pair Q = (d<e) (and the occupied pair P = (m<n)); cached on the
    integral object.'''
    c = eris.__dict__.get('_bare_Q')
    if c is None or c[0] is not vm:
        md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
        mm, nn = om.pairs[:, 0], om.pairs[:, 1]
        G = np.ascontiguousarray(np.asarray(eris.ovvv)[:, :, md, ne])
        H = np.ascontiguousarray(np.asarray(eris.oovv)[mm, nn][:, md, ne])
        c = eris.__dict__['_bare_Q'] = (vm, G, H)
    return c[1], c[2]


def dressed_vvvv_rows(eris, t1, om, vm, q0, q1, out, t2pp=None):
    '''Rows q0 <= (a<b) < q1 of the packed T1-dressed <ab||de>
    (dressed_vvvv_pp), written to ``out`` ((q1 - q0, npair), C-contiguous);
    plus sum_{m<n} t2[mn,ab] <mn||de>, the T2 part of the T3 ladder operator,
    if t2pp (packed (P, Q)) is given.  One pass over the bare rows, which may
    be a mapped file.'''
    from zccsdt.scratch import need, unmap
    nocc, nvir = t1.shape
    nq, dt = q1 - q0, out.dtype
    V = vvvv_packed(eris, vm)[0]
    G, H = _bare_Q(eris, om, vm)
    C = _tau1_pairs(t1, om, vm)[:, q0:q1]
    if t2pp is not None:
        C = C + t2pp[:, q0:q1]
    assert out.shape == (nq, vm.npair) and out.flags.c_contiguous
    need(V, q0 * vm.npair, q1 * vm.npair)
    t3pack.axpby(out, V[q0:q1], 1.0, 0.0)
    unmap(V, q0 * vm.npair, q1 * vm.npair)
    lib = t3pack._lib
    if t3pack.USE_C and hasattr(lib, 'zc_vvvv_dress_d') and dt in (np.float64, np.complex128):
        # the two one-index terms in one pass (C kernel), the pair term as a GEMM
        tm = np.ascontiguousarray(-t1, dtype=dt)
        tp = np.ascontiguousarray(t1, dtype=dt)
        Gc = np.ascontiguousarray(G, dtype=dt)
        I64, p_ = t3pack._I64, t3pack._p
        pa = np.ascontiguousarray(vm.pa_c[q0:q1])
        pb = np.ascontiguousarray(vm.pb_c[q0:q1])
        getattr(lib, 'zc_vvvv_dress_' + t3pack._sfx(dt))(
            p_(out), p_(out), I64(nq), I64(vm.npair), p_(pa), p_(pb),
            p_(tm), I64(nvir), p_(Gc), I64(nocc), I64(nvir),
            p_(tp), I64(nvir), p_(Gc), I64(nocc), I64(nvir),
            None, None, I64(0))
    else:
        a, b = vm.pairs[q0:q1, 0], vm.pairs[q0:q1, 1]
        out -= np.einsum('mq,mqc->qc', t1[:, a], G[:, b])
        out += np.einsum('mq,mqc->qc', t1[:, b], G[:, a])
    if om.npair:
        gemm(np.asfortranarray(C.T, dtype=dt), np.ascontiguousarray(H, dtype=dt), out, beta=1.0)
    return out


def add_one_body_pairs(W, F, vm, scale, block=0, rows=None):
    '''W[(a<b),(d<e)] += scale * (F_ad d_be - F_bd d_ae - F_ae d_bd + F_be d_ad),
    the antisymmetrized one-body operator F acting on either pair index, for
    W holding pair block ``block`` of ``vm`` (rows and columns block-local),
    or only its rows ``rows`` = (q0, q1) (block-local).

    Grouped by the index F acts on: acting on a, the column pair is {y, b}
    with value +F[a,y] if y < b else -F[a,y]; acting on b, the column pair
    is {y, a} with value -F[b,y] if y < a else +F[b,y].'''
    lo, hi = vm.blocks[block]
    ncol = hi - lo
    q0, q1 = (0, ncol) if rows is None else rows
    assert W.shape == (q1 - q0, ncol)
    lib = t3pack._lib
    if (t3pack.USE_C and hasattr(lib, 'zc_pair_one_body_add_d') and W.flags.c_contiguous
            and W.dtype in (np.float64, np.complex128)):
        Fc = np.ascontiguousarray(F, dtype=W.dtype)
        sc = np.array(scale, dtype=W.dtype)
        getattr(lib, 'zc_pair_one_body_add_' + t3pack._sfx(W.dtype))(
            t3pack._p(W), t3pack._I64(q1 - q0), t3pack._I64(ncol),
            t3pack._p(np.ascontiguousarray(vm.pa_c[lo + q0:lo + q1])),
            t3pack._p(np.ascontiguousarray(vm.pb_c[lo + q0:lo + q1])),
            t3pack._p(np.ascontiguousarray(vm.pidx_blk[block])), t3pack._I64(vm.n),
            t3pack._p(Fc), t3pack._I64(Fc.shape[1]), t3pack._p(sc))
        return W
    a, b = vm.pairs[lo + q0:lo + q1, 0], vm.pairs[lo + q0:lo + q1, 1]
    y = np.arange(vm.n)[None, :]
    rws = np.broadcast_to(np.arange(q1 - q0)[:, None], (q1 - q0, vm.n))
    for keep, act, s_lt, s_gt in ((b, a, 1, -1), (a, b, -1, 1)):
        k = keep[:, None]
        col = vm.pidx[np.minimum(y, k), np.maximum(y, k)] - lo
        valid = (y != k) & (col >= 0) & (col < ncol)
        val = scale * np.where(y < k, s_lt, s_gt) * F[act[:, None], y]
        W[rws[valid], col[valid]] += val[valid]
    return W


def dress_fock(eris, t1):
    '''T1-dressed Fock matrix (same definition as ccsdt._t1_dress).'''
    nocc = t1.shape[0]
    fock = np.asarray(eris.fock)
    dtype = np.result_type(fock, t1)
    n = fock.shape[0]
    sp = {'o': slice(0, nocc), 'v': slice(nocc, n)}
    fp = fock.astype(dtype)
    for r in 'ov':
        for s in 'ov':
            g = get_block(eris, r + 'o' + s + 'v')
            fp[sp[r], sp[s]] += einsum('risa,ia->rs', g, t1)
    x = np.eye(n, dtype=dtype)
    x[nocc:, :nocc] -= t1.T
    y = np.eye(n, dtype=dtype)
    y[:nocc, nocc:] += t1
    return x @ fp @ y.T


# ---------------------------------------------------------------- helpers

# ---------------------------------------------------------------- main step

def _is_spin_eris(eris):
    return type(eris).__name__ == 'SpinERIs'


def _persistent(eris, name):
    '''A list of reusable work buffers kept on the integral object (large
    per-iteration arrays; reuse avoids the page faults of fresh allocations).'''
    return eris.__dict__.setdefault('_buffers', {}).setdefault(name, [None, None, None])


def _spin_dressing(eris, t1, om, vm, t2pp=None, vd_out=None, vvvv=True):
    '''T1 dressing for U/R-form integrals: small blocks dense, ovvv as a
    SpinTensor and packed over its virtual pair ('ovvv_Q'; the dressed vovv is
    minus its bra transpose, as the dressing treats both bra slots alike),
    packed vvvv as pair blocks (plus the T2 part of the T3 ladder operator if
    t2pp is given, see spin_imds.dressed_vvvv_blocks; None without ``vvvv``:
    the blocks are then dressed in slabs of rows, _pp_sweep_spin).'''
    fdr = spin_imds.dress_fock(eris, t1)
    mats = spin_imds._t1_tensors(eris, t1)
    ge, ges = {}, {}
    for k in ('ovoo', 'oooo', 'ovvo', 'oovv', 'ooov', 'oovo'):
        with timer('dress.' + k):
            ge[k] = spin_imds.dress_block(eris, t1, k, mats).to_dense()
    with timer('dress.ovvv'):
        ges['ovvv'], ges['ovvv_Q'] = spin_imds.dressed_ovvv(eris, t1)
    with timer('dress.vvvv_pp'):
        vvvv_pp = spin_imds.dressed_vvvv_blocks(eris, t1, om, vm, t2pp=t2pp, out=vd_out) \
            if vvvv else None
    return fdr, ge, ges, vvvv_pp


def _spin_r1_vovv(eris, ges, t2, vm):
    '''sum_{k, c<d} ~<ak||cd> t2[i,k,c,d] = -sum ~<ka||cd> t2[i,k,c,d] (the
    vovv term of R1), spin-blocked.'''
    o_, v_ = eris.spaces['o'], eris.spaces['v']
    Qs = pair_space('Q', vm)
    T2s = SpinTensor.from_dense(t2, (o_, o_, v_, v_), sig=(-1, -1, 1, 1))
    return -s_einsum('kaQ,ikQ->ia', ges['ovvv_Q'], pack_pairs(T2s, 2, Qs)).to_dense()


_ENGINES = {}


def _engine_key(eris, dtype, flip=False):
    om, vm = t3pack.maps_for(eris)
    return (id(om), id(vm), np.dtype(dtype).str, bool(flip))


def engine_for(eris, dtype, mem_mb=None, flip=False):
    '''T3 engine for these integrals: spin-blocked if the orbitals carry spin
    labels, dense otherwise (cached per maps / dtype).  ``mem_mb``: its buffer
    budget from now on (default: as it is; 2000 for a new engine).'''
    om, vm = t3pack.maps_for(eris)
    if flip and (om.spins is None or 2 * int(np.sum(om.spins == 0)) != om.n
                 or 2 * int(np.sum(vm.spins == 0)) != vm.n):
        raise ValueError('spin_flip (RCCSDT) needs alpha-first orbitals of a closed-shell '
                         'RHF reference (from_scf(..., spin_order=True))')
    key = _engine_key(eris, dtype, flip)
    eng = _ENGINES.get(key)
    if eng is None:
        eng = _ENGINES[key] = t3engine.make_engine(om, vm, dtype,
                                                   2000 if mem_mb is None else mem_mb, flip=flip)
    elif mem_mb is not None:
        eng.set_memory(mem_mb)
    return eng


def _kept_operator(eris, sizes, dtype):
    '''Mapped scratch arrays (n, n) for n in ``sizes`` (a number: one array)
    for the rows of the T3 pp ladder operator of the current iteration, kept
    on the integral object between iterations; a dict by pair block for a
    list of sizes (U/R form: the blocks 0, 1(, 2) in that order).'''
    from zccsdt.scratch import Scratch
    one = np.isscalar(sizes)
    shapes = [(int(n), int(n)) for n in ([sizes] if one else sizes)]
    held = _persistent(eris, 'wpp_file')
    if held[1] is None or [a.shape for a in held[1]] != shapes or held[1][0].dtype != dtype:
        if held[0] is not None:
            held[0].close()
        held[0] = Scratch(prefix='zccsdt_wpp_', what='the pp ladder operator',
                          hint=' or give the calculation more memory (max_memory)')
        held[1] = [held[0].create('wpp%d' % k, sh, dtype) for k, sh in enumerate(shapes)]
    return held[1][0] if one else dict(enumerate(held[1]))


def _pp_sweep(eng, plan, eris, t1, t2pp, goovv_pp, Feff_vv, om, vm, T3p, Wv, R3):
    '''The terms with the packed dressed <ab||de>, one slab of its rows at a
    time (plan.pp_bytes): sum_f ~<ab||ef> t1[i,f] added to Wv, the CCSD ladder
    (returned, packed (P, Q)) and the T3 pp ladder, accumulated in R3.  The
    dressed rows are built from the bare ones (in memory or in a mapped
    file), used and dropped.'''
    from zccsdt import scratch
    dtype = R3.dtype
    nvp = vm.npair
    slabs = pair_slabs(nvp, nvp * dtype.itemsize, plan.pp_bytes)
    max_rows = max(q1 - q0 for q0, q1 in slabs)
    bufs = _persistent(eris, 'vvvv_rows')
    if bufs[0] is None or bufs[0].shape != (max_rows, nvp) or bufs[0].dtype != dtype:
        bufs[0] = None                          # free before allocating again
        bufs[0] = np.empty((max_rows, nvp), dtype=dtype)
    buf = bufs[0]
    tt = np.ascontiguousarray(t2pp + _tau1_pairs(t1, om, vm), dtype=dtype)    # (P, Q)
    gpp = np.ascontiguousarray(goovv_pp, dtype=dtype)
    lad = np.zeros((om.npair, nvp), dtype=dtype)
    calls = [0]
    order = plan.pp_order if len(slabs) > 1 else 'slab'
    # batch-wise the operator is needed once per batch of triples: its rows
    # are kept in a scratch file by the first sweep and read back by the
    # others (a pass over the file instead of the dressing)
    kept = _kept_operator(eris, nvp, dtype) if order == 'batch' else None

    def first():
        # the sweep that also feeds W_vvvo and the CCSD ladder
        for q0, q1 in slabs:
            W = buf[:q1 - q0]
            with timer('dress.vvvv_pp'):
                dressed_vvvv_rows(eris, t1, om, vm, q0, q1, W)
            with timer('wv.vvvv_t1'):
                t3pack.pair_t1_contract(Wv[q0:q1], W, t1, vm)
            with timer('ccsd'):
                lad[:, q0:q1] = gemm(W, tt.T).T
            with timer('imd.Wpp'):      # W(ab,de) = ~<ab||de> + t2 <mn||de> + 1/2 A[F delta]
                gemm(np.asfortranarray(t2pp[:, q0:q1].T), gpp, W, beta=1.0)
                add_one_body_pairs(W, Feff_vv, vm, 0.5, rows=(q0, q1))
                if kept is not None:
                    t3pack.axpby(kept[q0:q1], W, 1.0, 0.0)
                    scratch.written(kept, q0 * nvp, q1 * nvp)
                    scratch.unmap(kept, q0 * nvp, q1 * nvp)
            yield q0, q1, W

    def again():
        # later sweeps (one per batch of triples): the kept rows
        for q0, q1 in slabs:
            W = buf[:q1 - q0]
            with timer('imd.Wpp'):
                scratch.need(kept, q0 * nvp, q1 * nvp)
                t3pack.axpby(W, kept[q0:q1], 1.0, 0.0)
                scratch.unmap(kept, q0 * nvp, q1 * nvp)
            yield q0, q1, W

    def sweep():
        calls[0] += 1
        return first() if calls[0] == 1 else again()

    with timer('pp_ladder'):
        eng.pp_ladder_rows(T3p, sweep, R3, order=order, max_rows=max_rows)
    if calls[0] == 0:                           # no triples: the other terms remain
        for _ in first():
            pass
    return lad


def _spin_slabs(eris, vm, itemsize, max_bytes):
    '''(h, q0, q1) for the slabs of rows of the distinct pair blocks of the
    spin-blocked packed <ab||de> (aa, ab and, unless it is the aa block, bb),
    at most ``max_bytes`` of dressed rows each; and the blocks each serves.'''
    serves = {0: (0, 2) if eris.restricted else (0,), 1: (1,), 2: (2,)}
    out = []
    for h in ((0, 1) if eris.restricted else (0, 1, 2)):
        lo, hi = vm.blocks[h]
        if hi > lo:
            out += [(h, q0, q1) for q0, q1 in pair_slabs(hi - lo, (hi - lo) * itemsize, max_bytes)]
    return out, serves


def _pp_sweep_spin(eng, plan, eris, t1, t2pp, Feff_vv, om, vm, T3p, Wv, R3):
    '''_pp_sweep for U/R-form integrals: the pair blocks of the packed
    dressed <ab||de>, one slab of rows at a time.  The rows include the T2
    part of the T3 ladder operator from the start (one dressing pass); W_vvvo
    and the CCSD ladder are corrected for it as in the step in memory.'''
    from zccsdt import scratch
    dtype = R3.dtype
    slabs, serves = _spin_slabs(eris, vm, dtype.itemsize, plan.pp_bytes)
    nQ = [hi - lo for lo, hi in vm.blocks]
    size = max([(q1 - q0) * nQ[h] for h, q0, q1 in slabs] + [1])
    bufs = _persistent(eris, 'vvvv_rows')
    if bufs[0] is None or bufs[0].size != size or bufs[0].dtype != dtype:
        bufs[0] = None                          # free before allocating again
        bufs[0] = np.empty(size, dtype=dtype)
    C = np.ascontiguousarray(t2pp + _tau1_pairs(t1, om, vm), dtype=dtype)     # (P, Q)
    lad = np.zeros((om.npair, vm.npair), dtype=dtype)
    calls = [0]
    order = plan.pp_order if len(slabs) > 1 else 'slab'
    kept = None
    if order == 'batch':
        hs = sorted({h for h, _, _ in slabs})
        kept = dict(zip(hs, _kept_operator(eris, [nQ[h] for h in hs], dtype).values()))

    def rows(h, q0, q1):
        return bufs[0][:(q1 - q0) * nQ[h]].reshape(q1 - q0, nQ[h])

    def first():
        # the sweep that also feeds W_vvvo and the CCSD ladder
        for h, q0, q1 in slabs:
            W = rows(h, q0, q1)
            with timer('dress.vvvv_pp'):
                spin_imds.dressed_vvvv_rows(eris, t1, om, vm, h, q0, q1, W, C)
            for k in serves[h]:
                klo, khi = vm.blocks[k]
                if khi == klo:
                    continue
                with timer('wv.vvvv_t1'):
                    Vd, where = [None] * 3, [(0, 0)] * 3
                    Vd[k], where[k] = W, (klo + q0, klo + q1)
                    spin_imds.vvvv_t1_blocks(Wv, Vd, t1, om, vm, rows=where)
                with timer('ccsd'):
                    lad[:, klo + q0:klo + q1] = gemm(W, C[:, klo:khi].T).T
            with timer('imd.Wpp'):      # + 1/2 A[F delta]: rows of the ladder operator
                add_one_body_pairs(W, Feff_vv, vm, 0.5, block=h, rows=(q0, q1))
                if kept is not None:
                    K = kept[h]
                    t3pack.axpby(K[q0:q1], W, 1.0, 0.0)
                    scratch.written(K, q0 * nQ[h], q1 * nQ[h])
                    scratch.unmap(K, q0 * nQ[h], q1 * nQ[h])
            yield serves[h], q0, q1, W

    def again(need):
        # later sweeps (one per batch of triples): the kept rows of the
        # blocks that the spin group of the batch needs
        for h, q0, q1 in slabs:
            if need is not None and not need.intersection(serves[h]):
                continue
            W = rows(h, q0, q1)
            with timer('imd.Wpp'):
                K = kept[h]
                scratch.need(K, q0 * nQ[h], q1 * nQ[h])
                t3pack.axpby(W, K[q0:q1], 1.0, 0.0)
                scratch.unmap(K, q0 * nQ[h], q1 * nQ[h])
            yield serves[h], q0, q1, W

    def sweep(need=None):
        calls[0] += 1
        return first() if calls[0] == 1 else again(need)

    with timer('pp_ladder'):
        eng.pp_ladder_rows(T3p, sweep, R3, order=order,
                           max_rows=max([q1 - q0 for _, q0, q1 in slabs] + [1]))
    if calls[0] == 0:                           # no triples: the other terms remain
        for _ in first():
            pass
    return lad


def update_amps(opts, t1, t2, T3p, eris, mem_mb=2000, out=None, plan=None):
    '''One Jacobi step.  T3p is packed (notri, nvtri) or flat; returns
    (t1new, t2new, T3p_new) with T3p_new flat.  ``out``: a flat array of the
    size of T3p for the new T3 (e.g. a mapped file; default: a new array).
    ``plan``: MemoryPlan (default: everything in memory).'''
    from zccsdt.scratch import io_desc
    plan = plan or MemoryPlan()
    if plan.mem_mb is not None:
        mem_mb = plan.mem_mb
    _STEPS[0] += 1
    nocc, nvir = t1.shape
    om, vm = t3pack.maps_for(eris)
    dtype = np.result_type(t1, t2, T3p, eris.oovv, eris.fock)
    eng = engine_for(eris, dtype, mem_mb, flip=getattr(opts, 'spin_flip', False))
    eng.io = io_desc(T3p, out)          # T3 / the new T3 as mapped arrays of a scratch file
    T3p = np.ascontiguousarray(np.asarray(T3p, dtype=dtype).ravel())
    mo_e = eris.mo_energy
    eia = mo_e[:nocc, None] - (mo_e[None, nocc:] + opts.level_shift)
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]

    spin = _is_spin_eris(eris)
    # the dressed packed vvvv in slabs of rows
    sliced = plan.pp_bytes is not None and sliceable(eris, eng)
    # memory is short: no ovvv-sized array is kept for the next iteration
    tight = plan.t3_file or plan.pp_file or plan.pp_bytes is not None
    # (nor what an earlier run with another plan has kept for the operator)
    eris.__dict__.get('_buffers', {}).pop('vvvv' if sliced else 'vvvv_rows', None)
    t2mn = t2[mm, nn]                                   # (P, a, b)
    t2pp_v = np.ascontiguousarray(t2mn[:, md, ne])      # (P, Q)
    with timer('t1_dress'):
        if spin:
            # vvvv_pp then includes the T2 part of the ladder operator,
            # sum_{m<n} t2[mn,ab] <mn||de>; the CCSD ladder and W_vvvo correct for it
            fdr, ge, ges, vvvv_pp = _spin_dressing(eris, t1, om, vm, t2pp=t2pp_v,
                                                   vd_out=_persistent(eris, 'vvvv'),
                                                   vvvv=not sliced)
        else:
            fdr = dress_fock(eris, t1)
            ge = {}
            # (the dressed vovv is minus the bra transpose of the dressed ovvv,
            # as the dressing treats both bra slots alike: not formed)
            for k in ('ovoo', 'oooo', 'ovvo', 'oovv', 'ovvv', 'ooov', 'oovo'):
                with timer('dress.' + k):
                    ge[k] = dress_block(eris, t1, k)
            with timer('dress.vvvv_pp'):
                vvvv_pp = None if sliced else dressed_vvvv_pp(eris, t1, om, vm)
    foo, fov, fvv = fdr[:nocc, :nocc], fdr[:nocc, nocc:], fdr[nocc:, nocc:]
    goovv = ge['oovv']

    # ---- pass 1: quantities linear in T3 that feed T1, T2 and the W's
    t2P = t2[:, :, md, ne]                              # (i, j, Q)
    goovv_pp = goovv[mm, nn][:, md, ne]                 # (P, Q)
    goovv_P = goovv[:, :, md, ne]                       # (m, n, Q)
    goovv_mn = goovv[mm, nn]                            # (P, e, f)
    goovo_mn = ge['oovo'][mm, nn]                       # (P, e, j)
    # stacked integral operands so that pass 1 is two GEMMs per batch
    G_Pd = np.concatenate((goovv_mn.transpose(1, 0, 2),        # [e, P, f]
                           goovo_mn.transpose(2, 0, 1)), axis=0)   # [j, P, e]
    with timer('pass1.operands'):
        G_mQ = np.empty((nocc, vm.npair, nocc + nvir), dtype=dtype)
        t3pack.axpby(G_mQ[:, :, :nocc], goovv_P.transpose(1, 2, 0), 1.0, 0.0)   # [n, R, m']
        if spin:     # [m, R, a] = ~<am||R> = -~<ma||R>, from the packed blocks
            G_mQ[:, :, nocc:] = 0
            o_, v_ = eris.spaces['o'], eris.spaces['v']
            for (sm, sa, h), blk in ges['ovvv_Q'].blocks.items():
                (m0, m1), (a0, a1), (lo, hi) = o_.groups[sm], v_.groups[sa], vm.blocks[h]
                t3pack.axpby(G_mQ[m0:m1, lo:hi, nocc + a0:nocc + a1],
                             np.asarray(blk, dtype=dtype).transpose(0, 2, 1), -1.0, 0.0)
        else:        # [m, R, a] = ~<am||R> = -~<ma||R>
            t3pack.axpby(G_mQ[:, :, nocc:], ge['ovvv'][:, :, md, ne].transpose(0, 2, 1),
                         -1.0, 0.0)
        G_mQ = G_mQ.reshape(nocc * vm.npair, nocc + nvir)
    with timer('pass1'):
        Xacc, Yall, r1, rA = eng.pass1(T3p, G_Pd, G_mQ, goovv_pp, fov)
    # sum ~<ak||Q> t2[i,k,Q] with the pass-1 operand G_mQ[(k,Q), no + a] = ~<ak||Q>,
    # which is then dropped (ovvv-sized arrays are dropped as soon as they
    # have been used: they decide how much memory a run needs at least)
    r1_vovv = gemm(np.ascontiguousarray(t2P).reshape(nocc, -1), G_mQ[:, nocc:])
    del G_mQ
    C0 = -2 * Xacc[nvir:].transpose(1, 0, 2)               # [i, j, Q]
    dWo = Yall[:, :, :nocc].transpose(2, 1, 0)             # [m, a, P(ij)]
    B0 = -2 * Yall[:, :, nocc:].transpose(0, 2, 1)         # [P, a, b]
    r1 = r1.T
    # T3 contributions to the T1 / T2 residuals (r2 packed over i<j, a<b)
    rB = 0.5 * (B0[:, md, ne] - B0[:, ne, md])
    rC = -0.5 * (C0[mm, nn] - C0[nn, mm])
    r2_t3 = rA + rB + rC

    # ---- effective intermediates for the T3 residual
    with timer('intermediates'):
        govoo = ge['ovoo']
        ooov = ge['ooov']
        t_ = timer('imd.Wvvvo'); t_.__enter__()
        with timer('wv.Bv'):
            if spin:
                buf = _persistent(eris, 'bv')
                Bv = spin_imds.bra_dressed_vvvo(eris, t1, om, vm, out=buf[0], keep=not tight)
                buf[0] = None if tight else Bv
            else:
                Bv = bra_dressed_vvvo_pp(eris, t1, om, vm)
        xterm = (lambda W: spin_imds.t2_ovvv_add(W, t2, ges['ovvv'], om, vm)) \
            if spin else None
        with timer('wv.vvoo'):
            vvoo_pp = dressed_vvoo_pp(eris, t1, Bv, om, vm)    # needs Bv before W
        # W_vvvo is built in Bv from here on.  Its T3 part first, so that the
        # pass-1 result can go: Wv[Q, e, i] += Xacc[e, i, Q]
        with timer('wv.dWv'):
            t3pack.add_transposed(Bv.reshape(vm.npair, -1),
                                  Xacc[:nvir].reshape(nvir * nocc, vm.npair))
        del Xacc
        oovo_extra = None
        if spin:     # vvvv_pp includes t2pp^T oovv: remove sum_f oovv[P,(e,f)] t1[i,f]
            with timer('wv.zcorr'):
                Hb = spin_imds.bare_packed(eris)['oovv_PQ']
                Z = np.zeros((om.npair, nvir, nocc), dtype=dtype)
                spin_imds.vvvv_t1_blocks(Z, [Hb.blocks.get((h, h)) for h in range(3)], t1,
                                         om, vm, rows=om.blocks)
                oovo_extra = -Z
        Wv = wvvvo_packed(Bv, vvvv_pp, t1, t2, fov, ge, om, vm, xterm, inplace=True,
                          oovo_extra=oovo_extra)
        del Bv
        t_.__exit__(None, None, None)
        t_ = timer('imd.Wvooo'); t_.__enter__()
        Wvooo = govoo.copy()
        tmp2 = einsum('mnie,jnbe->mbij', ooov, t2)
        Wvooo -= (tmp2 - tmp2.transpose(0, 1, 3, 2)).transpose(0, 1, 3, 2)
        if spin:
            o_, v_ = eris.spaces['o'], eris.spaces['v']
            T2s = SpinTensor.from_dense(t2, (o_, o_, v_, v_), sig=(-1, -1, 1, 1))
            ovvv_t2 = s_einsum('mbef,ijef->mbij', ges['ovvv'], T2s).to_dense()
        else:
            ovvv_t2 = einsum('mbef,ijef->mbij', ge['ovvv'], t2)
            del ge['ovvv']
        Wvooo -= (0.5 * ovvv_t2).transpose(0, 1, 3, 2)
        Wo = Wvooo[:, :, nn, mm] + dWo                   # [m, a, P(ij)]
        t_.__exit__(None, None, None)
        t_ = timer('imd.ring_F'); t_.__enter__()
        Weff_ovvo = ge['ovvo'] + einsum('mnde,inae->madi', goovv, t2)
        Feff_vv = fvv - einsum('Paf,Pdf->ad', t2mn, goovv_mn)
        Feff_oo = foo + einsum('mnP,inP->mi', goovv_P, t2P)
        t_.__exit__(None, None, None)
        # CCSD residuals (T1-dressed) + T3 contributions, before the dressed
        # packed vvvv is turned into the T3 ladder operator in place
        t_ = timer('ccsd'); t_.__enter__()
        ladder_t2 = None
        if spin:
            Hb = spin_imds.bare_packed(eris)['oovv_PQ']
            ladder_t2 = ([Hb.blocks.get((h, h)) for h in range(3)], t2pp_v)
        lad = R3 = None
        if sliced:
            # slab by slab: its part of W_vvvo and of the CCSD ladder, and the
            # T3 pp ladder, which starts the T3 residual
            t_.__exit__(None, None, None)
            R3 = eng.target(out)
            if spin:
                lad = _pp_sweep_spin(eng, plan, eris, t1, t2pp_v, Feff_vv, om, vm, T3p, Wv, R3)
            else:
                lad = _pp_sweep(eng, plan, eris, t1, t2pp_v, goovv_pp, Feff_vv, om, vm, T3p,
                                Wv, R3)
            t_ = timer('ccsd'); t_.__enter__()
        R1, R2 = ccsd_residual_packed(t1, t2, fdr, ge, vvvv_pp, vvoo_pp, Feff_vv,
                                      Feff_oo, Weff_ovvo, om, vm, r1_vovv, ladder_t2,
                                      t2pp=t2pp_v, goovv_pp=goovv_pp, ladder=lad)
        t1new = t1 + (R1 + r1) / eia
        t2new = t2 + _unpack_pairs(R2 + r2_t3, om, vm) / eijab
        del vvoo_pp, R1, R2, lad
        t_.__exit__(None, None, None)
        t_ = timer('imd.Wpp'); t_.__enter__()
        Wpp = vvvv_pp            # in place, per pair block: W(ab,de) = ~<ab||de> + ...
        del vvvv_pp
        for h, (lo, hi) in enumerate(vm.blocks):
            if Wpp is None or any(Wpp[h] is Wpp[k] for k in range(h)):
                continue                  # shared block (RHF beta-beta = alpha-alpha)
            if hi > lo:
                if not spin:              # (spin path: already in the dressed blocks)
                    gemm(np.asfortranarray(t2pp_v[:, lo:hi].T),
                         np.ascontiguousarray(goovv_pp[:, lo:hi]), Wpp[h], beta=1.0)
                # F_vv folded into the ladder: W(ab,de) += 1/2 A[F_ad delta_be]
                add_one_body_pairs(Wpp[h], Feff_vv, vm, 0.5, block=h)
        t_.__exit__(None, None, None)
        t_ = timer('imd.O6'); t_.__enter__()
        Weff_oooo = ge['oooo'] + einsum('mnP,ijP->mnij', goovv_P, t2P)
        O6 = t3pack.occ_operator(om, lambda t: _occ_apply(t, Weff_oooo, Feff_oo))
        t_.__exit__(None, None, None)

    # ---- T3 residual: ring + drive, pp ladder (+F_vv), hh ladder (+F_oo)
    WringT = Weff_ovvo.transpose(1, 3, 2, 0).reshape(nvir * nocc, nvir * nocc)
    del ge, Yall, B0, C0, goovv_P, goovv_mn, goovo_mn, G_Pd
    if spin:
        del ges
    R = eng.residual(T3p, WringT, t2mn, Wv, Wo, t2P, Wpp, O6,
                     out=out if R3 is None else R3, clear=R3 is None)

    with timer('t3_update'):
        eo = mo_e[:nocc][om.triples].sum(axis=1)
        ev = (mo_e[nocc:] + opts.level_shift)[vm.triples].sum(axis=1)
        T3new = eng.update(T3p, R, eo, ev)
    return t1new, t2new, T3new.ravel()


def ccsd_update_amps(opts, t1, t2, eris):
    '''One Jacobi step of CCSD in the T1-dressed formalism (the CCSD part of
    update_amps without T3).'''
    nocc, nvir = t1.shape
    om, vm = t3pack.maps_for(eris)
    mm, nn = om.pairs[:, 0], om.pairs[:, 1]
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    mo_e = eris.mo_energy
    eia = mo_e[:nocc, None] - (mo_e[None, nocc:] + opts.level_shift)
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    spin = _is_spin_eris(eris)
    with timer('t1_dress'):
        if spin:
            fdr, ge, ges, vvvv_pp = _spin_dressing(eris, t1, om, vm)
            Bv = spin_imds.bra_dressed_vvvo(eris, t1, om, vm)
        else:
            fdr = dress_fock(eris, t1)
            ge = {k: dress_block(eris, t1, k)
                  for k in ('oooo', 'ovvo', 'oovv', 'ooov', 'vovv')}
            vvvv_pp = dressed_vvvv_pp(eris, t1, om, vm)
            Bv = bra_dressed_vvvo_pp(eris, t1, om, vm)
    with timer('ccsd'):
        goovv = ge['oovv']
        t2mn = t2[mm, nn]
        Feff_vv = fdr[nocc:, nocc:] - einsum('Paf,Pdf->ad', t2mn, goovv[mm, nn])
        Feff_oo = fdr[:nocc, :nocc] + einsum('mnP,inP->mi', goovv[:, :, md, ne],
                                              t2[:, :, md, ne])
        Weff_ovvo = ge['ovvo'] + einsum('mnde,inae->madi', goovv, t2)
        vvoo_pp = dressed_vvoo_pp(eris, t1, Bv, om, vm)
        r1_vovv = _spin_r1_vovv(eris, ges, t2, vm) if spin else None
        R1, R2 = ccsd_residual_packed(t1, t2, fdr, ge, vvvv_pp, vvoo_pp, Feff_vv,
                                      Feff_oo, Weff_ovvo, om, vm, r1_vovv)
    return t1 + R1 / eia, t2 + _unpack_pairs(R2, om, vm) / eijab


def _unpack_pairs(packed, om, vm):
    '''Full antisymmetric r2[i,j,a,b] from its (i<j, a<b) block.'''
    mm, nn = om.pairs[:, 0][:, None], om.pairs[:, 1][:, None]
    md, ne = vm.pairs[:, 0], vm.pairs[:, 1]
    r2 = np.zeros((om.n, om.n, vm.n, vm.n), dtype=packed.dtype)
    r2[mm, nn, md, ne] = packed
    r2[nn, mm, md, ne] = -packed
    r2[mm, nn, ne, md] = -packed
    r2[nn, mm, ne, md] = packed
    return r2


def _occ_apply(t, Woooo, Foo):
    '''(L t)[J,i,j,k] = P(k/ij) 1/2 sum_mn W[m,n,i,j] t[J,m,n,k]
                      - P(i/jk) sum_m F[m,i] t[J,m,j,k].'''
    x = 0.5 * einsum('mnij,Jmnk->Jijk', Woooo, t)
    out = x - x.transpose(0, 3, 2, 1) - x.transpose(0, 1, 3, 2)
    y = einsum('mi,Jmjk->Jijk', Foo, t)
    out -= y - y.transpose(0, 2, 1, 3) - y.transpose(0, 3, 2, 1)
    return out
