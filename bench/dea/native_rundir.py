'''A run directory of the native program (cfour/xzccsdt) for EOM-DEA on a
Hamiltonian file of O2(2+) (bench/dea/common.py): the integrals of the
reference determinant in CFOUR's file formats, the labels 2 Omega
(xzlabels) and the parities (xzparity), and a JOBARC of CFOUR's xjoda for a
ZMAT that asks for CALC=CCSDT and the double electron attachment (section
%xzdea; docs/native.md, section 20).

The X2Ccorr Hamiltonians of O2 are not made by CFOUR's chain: xjoda runs on
a stand-in molecule (HF / 6-31G of cfour/merge/test, whose GENBAS is
copied) only for JOBARC - joda's flags (CALC, CC_CONV, ESTATE_CONV, ...,
MEMORY_SIZE); the integral files, scf2ceig and nmototal are written here
(zccsdt.cfour.write) for the spinors of the Hamiltonian file, in the order
of their labels, and E(SCF) (SCFENEG) and the frozen pairs (NUMDROPA, 0)
are put into that JOBARC.  Nothing of the stand-in molecule is read by the
native program.

A block file of zccsdt.hblocks (.h5, o2_x2ccorr.py run-blocks) is written
without any n^4 array: the blocks with an occupied index from the label
blocks of the file (common.load_blocks without <vv||vv>), and <ab||cd> not
as moinpppp - (nv (nv + 1) / 2)^2 complex numbers, 60 GB for 350 virtual
spinors - but as the file xzvvvv of the native program (cfour/xzccsdt
xzread.F90, xzvvhdr): one block per sum of the labels of a pair, made
group by group from the blocks of the file (write_vvvv_blocks).

usage: python bench/dea/native_rundir.py <file.npz | file.h5> <outdir> [--freeze n] [--targets 0,2] [--roots 1]
           [--cc-conv 10] [--estate-conv 9] [--mem-gb 8] [--section text] [--cfour-bin dir] [--level 5p3h]
  --freeze n      the first n spinors of the reference frozen (their mean field folded in; common.load)
  --targets       2 Omega of the two attached electrons, one problem each (default 0,2)
  --roots         roots per target (default 1)
  --cc-conv       CC_CONV (default 10)
  --estate-conv   ESTATE_CONV: the largest element of a residual below 10^-N (default 9)
  --mem-gb        MEMORY_SIZE in GB (default 8)
  --xz line       a line of the section %xzccsdt (repeatable)
  --level 5p3h    the line "level 5p3h" in %xzdea: R with complete 5p3h (docs/native.md, section 21)
  --phases seed   every spinor multiplied by a phase exp(i x), x random with this seed: complex
                  integrals with the same labels (a test of the complex arithmetic)
  --moinpppp      a block file written as before: the full container (common.load_blocks) and
                  moinpppp instead of xzvvvv (a check of the two routes on the same integrals)
'''
import os
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.join(HERE, '..', '..', 'cfour', 'merge', 'test')
BIN = '/home/xwang405/apps/cfour-x2cso/bin'

ZMAT = '''stand-in molecule (the integrals are those of {src})
F 0.000000000000 0.000000000000 0.000000000000
H 0.000000000000 0.000000000000 1.732800000000

*CFOUR(CALC=CCSDT,BASIS=SPECIAL
SYM=OFF,COORDINATES=CARTESIAN,UNITS=BOHR,CHARGE=0,MULT=1
RELATIVISTIC=X2C1E,X2CKEY=SOATMX,ABCDTYPE=STANDARD
SCF_CONV=10,SCF_MAXCYC=200
CC_CONV={cc},CC_MAXCYC=100
ESTATE_CONV={es},ESTATE_MAXCYC=120
MEMORY_SIZE={mem},MEM_UNIT=GB)

F:6-31G
H:6-31G

%xzdea
{targets}
parity g{level}
{extra}
'''


def write_parity(directory, par):
    '''The file xzparity: one parity per spinor of the SCF (0 g, 1 u), the
    order of scf2ceig, the format of xzlabels without the modulus.'''
    with open(os.path.join(directory, 'xzparity'), 'w') as f:
        f.write('# parity per spinor (0 g, 1 u; order of scf2ceig); first line: spinors\n')
        f.write('%d\n' % len(par))
        for k in range(0, len(par), 16):
            f.write(' '.join('%d' % x for x in par[k:k + 16]) + '\n')


def write_vvvv_blocks(path, bf, vpos, verbose=True):
    '''The file xzvvvv (docs/native.md section 2; the reader: xzvvhdr / xzrdvvb
    of cfour/xzccsdt/xzread.F90) from the block file ``bf`` (zccsdt.hblocks):
    <ab||cd> = (ac|bd) - (ad|bc) for a<b, c<d in blocks of the pairs of one
    label sum.  ``vpos[x]``: CFOUR's number of the virtual spinor of global
    index x of the block file (-1: not a virtual spinor of the run).
    A group at a time: (ac|bd) on the ordered pairs (a, b) of the group from
    the blocks (g_a g_c | g_b g_d) of the file, then the antisymmetrization
    and the pairs a<b.  Returns (groups, elements, largest block).'''
    import time
    t0 = time.perf_counter()
    lab = [g[0] for g in bf.groups]
    G = len(bf.groups)
    vloc = [np.flatnonzero(vpos[m] >= 0) for m in bf.members]      # local indices of the virtual members
    vnum = [vpos[m][x] for m, x in zip(bf.members, vloc)]           # their CFOUR numbers
    sums = sorted({lab[g1] + lab[g2] for g1 in range(G) for g2 in range(G) if len(vloc[g1]) and len(vloc[g2])})
    dt = np.dtype(np.float64 if bf.dtype.kind == 'f' else np.complex128)
    cnt, pairs, plan = [], [], []
    for L in sums:
        gp = [(g1, g2) for g1 in range(G) for g2 in range(G) if lab[g1] + lab[g2] == L and len(vloc[g1])
              and len(vloc[g2])]
        ra = np.concatenate([np.repeat(vnum[g1], len(vnum[g2])) for g1, g2 in gp])
        rb = np.concatenate([np.tile(vnum[g2], len(vnum[g1])) for g1, g2 in gp])
        strict = np.flatnonzero(ra < rb)
        num = ra[strict] + rb[strict] * (rb[strict] - 1) // 2
        o = np.argsort(num, kind='stable')
        cnt.append(len(o))
        pairs.append(num[o])
        plan.append((L, gp, ra, rb, strict[o]))
    nv = int(vpos.max()) + 1
    assert sum(cnt) == nv * (nv - 1) // 2, 'the pairs of the label sums are not all pairs'
    nwf = 1 if dt.kind == 'f' else 2
    with open(path, 'wb') as f:
        f.write(b'XZVVVV01')
        np.array([nv, nwf, len(cnt)], dtype='<i8').tofile(f)
        np.array(cnt, dtype='<i8').tofile(f)
        np.concatenate(pairs).astype('<i8').tofile(f)
        nel = 0
        for L, gp, ra, rb, sel in plan:
            off = np.concatenate(([0], np.cumsum([len(vnum[g1]) * len(vnum[g2]) for g1, g2 in gp])))
            N = off[-1]
            X = np.zeros((N, N), dtype=dt)                  # X[(a b), (c d)] = (ac|bd)
            for k, (g1, g2) in enumerate(gp):
                for l, (g3, g4) in enumerate(gp):
                    b = (g1, g3, g2, g4)
                    if b not in bf.offset:
                        continue
                    x = np.asarray(bf.block(b))[np.ix_(vloc[g1], vloc[g3], vloc[g2], vloc[g4])]
                    X[off[k]:off[k + 1], off[l]:off[l + 1]] = x.transpose(0, 2, 1, 3).reshape(off[k + 1] - off[k], -1)
            # the column of (d, c) for the column of (c, d)
            where = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(ra, rb))}
            swap = np.array([where[(int(b), int(a))] for a, b in zip(ra[sel], rb[sel])], dtype=np.int64)
            Y = X[sel]
            del X
            B = Y[:, sel] - Y[:, swap]
            del Y
            np.ascontiguousarray(B).astype(dt.newbyteorder('<'), copy=False).tofile(f)
            nel += B.size
            del B
    if verbose:
        print('xzvvvv: %d groups of pairs (label sums %d .. %d), %d elements (%.2f GB), largest block %d pairs; %.1f s'
              % (len(cnt), sums[0], sums[-1], nel, nel * nwf * 8 / 1e9, max(cnt), time.perf_counter() - t0), flush=True)
    return len(cnt), nel, max(cnt)


def main():
    args = sys.argv[1:]
    opt = {'--freeze': '0', '--targets': '0,2', '--roots': '1', '--cc-conv': '10', '--estate-conv': '9',
           '--mem-gb': '8', '--cfour-bin': BIN, '--phases': None, '--level': None}
    xz = []
    while '--xz' in args:
        k = args.index('--xz')
        xz.append(args[k + 1])
        del args[k:k + 2]
    for key in list(opt):
        if key in args:
            k = args.index(key)
            opt[key] = args[k + 1]
            del args[k:k + 2]
    old = '--moinpppp' in args
    if old:
        args.remove('--moinpppp')
    path, out = args[0], args[1]
    from zccsdt import cfour
    blocks = path.endswith('.h5') and not old
    if blocks:
        from zccsdt.eris import BLOCKS
        if opt['--phases'] is not None:
            raise SystemExit('--phases: not with a block file')
        eris, lab, par, info = common.load_blocks(path, sort=True, freeze=int(opt['--freeze']),
                                                  names=[b for b in BLOCKS if b != 'vvvv'])
    else:
        eris, lab, par, info = common.load(path, sort=True, freeze=int(opt['--freeze']))
    no, n = eris.nocc, info['n']
    print('%s: %d spinors, %d occupied, frozen %s; E(reference) %.12f; largest |imag| %.1e, |f_ov| %.1e, '
          'forbidden integral %.1e' % (path, n, no, opt['--freeze'], info['e_ref'], info['imag'], info['fock_ov'],
                                       info['forbidden']))
    # (the files have the spinor energies only: canonical spinors; f_ov of the SCF is left out)
    f = np.asarray(eris.fock)
    eris.fock = np.diag(np.diag(f))
    if opt['--phases'] is not None:
        # (as bench/cfour/labelled_run.py: <pq||rs> picks up conj(f_p f_q) f_r f_s)
        ph = np.exp(1j * np.random.default_rng(int(opt['--phases'])).uniform(0, 2 * np.pi, n))
        sl = {'o': slice(0, no), 'v': slice(no, n)}
        for key in ('oooo', 'ooov', 'oovv', 'ovov', 'ovvo', 'ovvv', 'vvvv'):
            fz = [ph[sl[c]] for c in key]
            x = np.asarray(getattr(eris, key)).astype(np.complex128)
            x *= (fz[0].conj()[:, None, None, None] * fz[1].conj()[None, :, None, None]
                  * fz[2][None, None, :, None] * fz[3][None, None, None, :])
            setattr(eris, key, x)
        eris.fock = np.asarray(eris.fock).astype(np.complex128)
        print('phases on the spinors (seed %s): complex integrals' % opt['--phases'])
    os.makedirs(out, exist_ok=True)
    targets = '\n'.join('%s %s' % (t, opt['--roots']) for t in opt['--targets'].split(','))
    extra = ('%xzccsdt\n' + '\n'.join(xz) + '\n') if xz else ''
    with open(os.path.join(out, 'ZMAT'), 'w') as fz:
        fz.write(ZMAT.format(src=os.path.basename(path), cc=opt['--cc-conv'], es=opt['--estate-conv'],
                             mem=opt['--mem-gb'], targets=targets, extra=extra,
                             level='' if opt['--level'] is None else '\nlevel ' + opt['--level']))
    shutil.copy(os.path.join(TEST, 'GENBAS'), out)
    with open(os.path.join(out, 'xjoda.out'), 'w') as log:
        subprocess.run([os.path.join(opt['--cfour-bin'], 'xjoda')], cwd=out, stdout=log, stderr=subprocess.STDOUT,
                       check=True)
    tmp = os.path.join(out, 'written')
    cfour.write(tmp, eris, nfrozen=0)
    for name in os.listdir(tmp):
        if name not in ('JOBARC', 'JAINDX'):
            shutil.move(os.path.join(tmp, name), os.path.join(out, name))
    shutil.rmtree(tmp)
    if blocks:
        # <ab||cd> as the file xzvvvv: the CFOUR number of every virtual spinor of the block file
        from zccsdt import hblocks
        bf = hblocks.BlockFile(path)
        fr = int(opt['--freeze'])
        act = np.arange(fr, bf.n)
        vpos = np.full(bf.n, -1, dtype=np.int64)
        vpos[act[info['perm'][no:]]] = np.arange(n - no)
        write_vvvv_blocks(os.path.join(out, 'xzvvvv'), bf, vpos)
        bf.close()
    arc = cfour.JobArc(out)
    arc.put('SCFENEG', np.array([eris.e_hf], dtype='<f8'))
    arc.put('NUMDROPA', np.array([0], dtype='<i8'))
    if par is not None:
        write_parity(out, np.asarray(par, dtype=np.int64))
    print('written: %s (labels 2 Omega: occupied %s, virtual %d spinors from %d to %d)'
          % (out, lab[:no], n - no, lab[no:].min(), lab[no:].max()))


if __name__ == '__main__':
    main()
