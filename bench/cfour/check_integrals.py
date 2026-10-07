'''The MO integral files of CFOUR's two-component programs (zccsdt.cfour)
against what can be known without them, in the directory of a run of
xjoda, xsox2catm, xsox2chf, xx2cvtran, xx2cccsd (CALC=CCSD(T)) with the
files kept (moin*, PPPH*, JOBARC, JAINDX, nmototal, scf2ceig, scf2ccmo,
t1guess, t2guess, MOL and the outputs *.out):

1. every block of the container element by element against the integrals
   that PySCF makes for CFOUR's own spinors (scf2ccmo, its coefficients in
   its AO basis; socutils maps the basis) - order, pairs, signs and which
   of the two spinors of a charge distribution is conjugated;
2. the second-order energy against the one xx2cvtran prints;
3. with the converged amplitudes of xx2cccsd (t1guess, t2guess): the CCSD
   correlation energy, the (T) correction, and the T1 equation - all of
   which use the files and none of which needs <ab||cd>.  With moinpppp
   in the directory also the T2 equation.

usage: python bench/cfour/check_integrals.py RUNDIR PYSCF_BASIS
socutils has to be on the path.
'''
import os
import re
import sys
from types import SimpleNamespace

import numpy as np
from pyscf import gto
from pyscf.cc import gccsd, gccsd_t

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import props                                                            # noqa: E402
from zccsdt import cfour, eris as E                                     # noqa: E402
from zccsdt.cc import ccsd                                              # noqa: E402


def printed(path, pattern):
    '''The last number that follows ``pattern`` in a file (None: not there).'''
    out = None
    if os.path.isfile(path):
        with open(path) as f:
            for line in f:
                m = re.search(pattern + r'\s*:?\s*\(?\s*([-+]?\d+\.\d+(?:[EeDd][-+]?\d+)?)', line)
                if m:
                    out = float(m.group(1).replace('D', 'E'))
    return out


def amplitudes(d, no, nv):
    '''t1[i, a] and t2[i, j, a, b] of the files t1guess and t2guess (the
    amplitudes xx2cccsd stopped with): t1(a, i), t2 on pairs a<=b, i<=j.'''
    t1 = cfour._complex(os.path.join(d, 't1guess'), (nv, no)).T
    t2 = cfour._complex(os.path.join(d, 't2guess'), (nv * (nv + 1) // 2, no * (no + 1) // 2))
    t2 = cfour._unpack(cfour._unpack(t2, nv, 0), no, 2)
    return np.ascontiguousarray(t1), np.ascontiguousarray(t2.transpose(2, 3, 0, 1))


def spinors(d, basis, x):
    '''(mol, C): the molecule in CFOUR's frame and the active spinors of its
    SCF in PySCF's spinor basis.'''
    from socutils.tools.label import Labels
    mol = gto.M(atom=props.read_mol(os.path.join(d, 'MOL')), unit='Bohr', basis=basis, verbose=0)
    n2c = mol.nao_2c()
    c = cfour._complex(os.path.join(d, 'scf2ccmo'), (n2c, n2c))         # [AO (alpha, beta), spinor]
    c = Labels.spinor_cfour2pyscf(mol).conj().T @ c
    return mol, c[:, x.nfrozen:x.nfrozen + x.nocc + x.nvir]


def main():
    d, basis = sys.argv[1], sys.argv[2]
    x = cfour.read(d)
    eris = cfour.container(x, packed_vvvv=False)
    no, nv = x.nocc, x.nvir
    print('%s: %d occupied, %d virtual spinors; %d frozen, %d virtual ones dropped; moinpppp %s'
          % (d, no, nv, x.nfrozen, x.ndropped, 'there' if x.pppp is not None else 'not there'))
    worst = 0.0

    # 1. the integrals of CFOUR's spinors from PySCF
    mol, C = spinors(d, basis, x)
    s = mol.intor('int1e_ovlp_spinor')
    print('spinors of scf2ccmo in the basis of PySCF: orthonormal to %.1e' % abs(C.conj().T @ s @ C - np.eye(no + nv)).max())
    g = E._chem_block(mol, 'spinor', C, C, C, C)                        # (pq|rs)
    anti = g.transpose(0, 2, 1, 3) - g.transpose(0, 2, 3, 1)            # <pq||rs>
    sl = {'o': slice(0, no), 'v': slice(no, no + nv)}
    for key in E.BLOCKS:
        blk = getattr(eris, key)
        if blk is None:
            continue
        ref = anti[tuple(sl[c] for c in key)]
        diff = abs(np.asarray(blk) - ref).max()
        worst = max(worst, diff)
        print('  <%s>  max |file - PySCF| %.1e   (max |element| %.3f, imaginary %.3f; against the conjugate %.1e)'
              % (key, diff, abs(ref).max(), abs(ref.imag).max(), abs(np.asarray(blk) - ref.conj()).max()))

    # 2. second order
    e2, ref2 = cfour.mp2(eris), printed(os.path.join(d, 'xx2cvtran.out'), 'mp2 energy')
    print('E(2)      %.15f' % e2 + ('   xx2cvtran %.15f   difference %.1e' % (ref2, e2 - ref2) if ref2 is not None else ''))

    # 3. the amplitudes of xx2cccsd
    if os.path.isfile(os.path.join(d, 't2guess')):
        out = os.path.join(d, 'xx2cccsd.out')
        t1, t2 = amplitudes(d, no, nv)
        ec, ref = ccsd.energy(t1, t2, eris), printed(out, 'CCSD correlation energy')
        print('E(CCSD)   %.15f   imaginary part %.1e' % (ec.real, ec.imag)
              + ('   xx2cccsd %.15f   difference %.1e' % (ref, ec.real - ref) if ref is not None else ''))
        pe = gccsd._PhysicistsERIs()
        for key in ('fock', 'mo_energy', 'oovv', 'ooov', 'ovvv'):
            setattr(pe, key, getattr(eris, key))
        et = gccsd_t.kernel(SimpleNamespace(verbose=0, stdout=sys.stdout), pe, t1, t2, verbose=0)
        ref = printed(out, r'\(T\) correction')
        print('E(T)      %.15f' % et.real + ('   xx2cccsd %.15f   difference %.1e' % (ref, et.real - ref) if ref is not None else ''))
        # the amplitude equations: the T1 equation has no <ab||cd>
        full = eris.vvvv is not None
        if not full:
            eris.vvvv = np.zeros((nv,) * 4, dtype=np.complex128)
        n1, n2 = ccsd.update_amps(SimpleNamespace(level_shift=0.0), t1, t2, eris, alg='dense')
        eo, ev = eris.mo_energy[:no], eris.mo_energy[no:]
        r1 = (n1 - t1) * (eo[:, None] - ev[None, :])
        print('T1 equation with these amplitudes: max |residual| %.1e (max |t1| %.1e)' % (abs(r1).max(), abs(t1).max()))
        if full:
            r2 = (n2 - t2) * (eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :])
            print('T2 equation with these amplitudes: max |residual| %.1e' % abs(r2).max())
        else:
            print('T2 equation: not checked (no moinpppp)')
    print('largest difference of an integral: %.1e' % worst)


if __name__ == '__main__':
    main()
