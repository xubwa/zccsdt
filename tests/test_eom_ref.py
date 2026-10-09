'''The dense reference of EOM-CCSDT (zccsdt.cc.eom_ref), docs/eom.md:

- the sigma vector in its three forms against each other and against the
  commutator <mu| [Hbar, R] |0> of second-quantized operators in a small
  Fock space (tests/fockspace.py), for complex amplitudes that solve nothing;
- three electrons, where CCSDT is exact: all eigenvalues of the EOM matrix
  against the differences of the eigenvalues of H among the determinants
  (Li, spin orbitals and X2C spinors);
- EOM-CCSD against PySCF (EE, IP, EA; real and complex spin orbitals);
- IP and EA of EOM-CCSDT through the ghost orbital against the operators of
  the Fock space with one electron less / more.'''
import functools
from math import comb

import numpy as np
import pytest
from pyscf import cc
from pyscf.cc import eom_gccsd

import fockspace as F
from conftest import three_electron, water_ghf
from test_ccsdtq_ref import _spin_mixed
from test_lambda_ref import _model
from zccsdt import eris as E
from zccsdt.cc import ccsdtq_ref as R, ccsdtq_terms as Q, eom_ref as EOM


@pytest.fixture(scope='module')
def model():
    '''A complex Hamiltonian, complex amplitudes and a complex vector of 3 +
    4 spin orbitals, and the operators of its Fock space.'''
    no, nv = 3, 4
    f, v = _model(no, nv)
    rng = np.random.default_rng(3)
    amps = [F.complex_antisym(rng, no, nv, k, s) for k, s in ((1, 0.2), (2, 0.1), (3, 0.05))]
    r = [F.complex_antisym(rng, no, nv, k) for k in (1, 2, 3)]
    fk = F.Fock(no, nv)
    return no, nv, f, v, amps, r, fk


def _exact_sigma(fk, f, v, amps, rop, bra):
    '''<bra| [Hbar, R] |0> for the excitation operator ``rop`` (a matrix).'''
    ex = F.Left(fk, f, v, amps)
    ref = fk.ref.astype(complex)
    return bra.T @ (ex.hbar(rop @ ref) - rop @ ex.hbar(ref))


@pytest.mark.parametrize('nrank', [2, 3])
def test_sigma_is_the_commutator(model, nrank):
    '''sigma of the differentiated term table, of the five-point formula
    and of the three evaluations in dressed integrals agree to 1e-12 and
    are <mu| [Hbar, R] |0>; with two ranks the same for EOM-CCSD.'''
    no, nv, f, v, amps, r, fk = model
    amps, r = amps[:nrank], r[:nrank]
    s = EOM.sigma(r, amps, f, v, no)
    for how in ('stencil', 'dressed')[:nrank - 1]:
        s2 = EOM.sigma(r, amps, f, v, no, how)
        assert max(abs(a - b).max() for a, b in zip(s, s2)) < 1e-12
    ex = F.Left(fk, f, v, amps)
    rop = sum(fk.excitation(x) for x in r)
    want = ex.tensors(_exact_sigma(fk, f, v, amps, rop, ex.dets))
    assert max(abs(a - b).max() for a, b in zip(s, want)) < 1e-12
    assert all(abs(a).max() > 0.1 for a in s)
    # flat vectors over the unique elements
    x = EOM.pack(r)
    assert x.size == sum(EOM.sizes(no, nv, nrank))
    assert max(abs(a - b).max() for a, b in zip(EOM.unpack(x, no, nv, nrank), r)) < 1e-14
    assert abs(EOM.apply(x, amps, f, v, no) - EOM.pack(s)).max() < 1e-13


def test_derivative_of_the_dressed_integrals(model):
    '''[H~, R1] by the one-index transformation is the derivative of the
    dressing along r1.'''
    from zccsdt.cc import ccsdt, qlgrad_ref
    no, nv, f, v, amps, r, fk = model
    n = no + nv
    ge, fdr = ccsdt._t1_dress(v, f, amps[0], no, n)
    dg, df = EOM.dressed_derivative(ge, fdr, r[0], no)
    wg, wf = qlgrad_ref.directional(lambda a: ccsdt._t1_dress(v, f, a[0], no, n), [amps[0]], [r[0]])
    assert abs(dg - wg).max() < 1e-13 and abs(df - wf).max() < 1e-13
    assert abs(dg[:no, :no, no:, no:]).max() == 0 and abs(dg).max() > 0.01


@pytest.mark.parametrize('kind', ['ip', 'ea'])
def test_ghost_orbital_gives_ip_and_ea(model, kind):
    '''The EE sigma with one more spin orbital that no integral touches,
    among the vectors that have it once, is <mu| [Hbar, R] |0> for the
    operators that remove (add) an electron: r[i] i, r[ij,a] a+ j i,
    r[ijk,ab] a+ b+ k j i (r[a] a+, ..).'''
    from itertools import combinations
    no, nv, f, v, amps, r, fk = model
    a2, f2, g2, no2, sel = EOM.ghost(amps, f, v, no, kind)
    nv2 = f2.shape[0] - no2
    nfree = (no, nv) if kind == 'ip' else (nv, no)
    assert len(sel) == sum(len(list(combinations(range(nfree[0]), k))) * len(list(combinations(range(nfree[1]), k - 1)))
                           for k in (1, 2, 3))
    rng = np.random.default_rng(5)
    x = np.zeros(sum(EOM.sizes(no2, nv2, 3)), dtype=complex)
    x[sel] = rng.standard_normal(len(sel)) + 1j * rng.standard_normal(len(sel))
    s = EOM.apply(x, a2, f2, g2, no2)
    rest = np.setdiff1d(np.arange(x.size), sel)
    assert abs(s[rest]).max() < 1e-14 and abs(s[sel]).max() > 0.1
    # the same operators without the ghost
    ex = F.Left(fk, f, v, amps)
    ops = []
    for k in (1, 2, 3):
        # a+ b+ x+ k j i = (-1)^k a+ b+ k j i x+ (IP; x the last spin orbital)
        sign = (-1) ** k if kind == 'ip' else 1
        if kind == 'ip':
            tup = [(o, vv) for o in combinations(range(no), k) for vv in combinations(range(nv), k - 1)]
        else:           # (the ghost is the first occupied one: the order of the EE elements)
            tup = [(o, vv) for o in combinations(range(no), k - 1) for vv in combinations(range(nv), k)]
        for o, vv in tup:
            ops.append(sign * ex.tau(o, vv))
    rop = sum(c * op for c, op in zip(x[sel], ops))
    bra = np.array([op @ fk.ref for op in ops]).T
    want = _exact_sigma(fk, f, v, amps, rop, bra)
    assert abs(s[sel] - want).max() < 1e-12


@pytest.mark.parametrize('kind', ['ghf', 'x2c'])
def test_three_electrons_are_exact(kind):
    '''Li / STO-3G, three electrons in ten spin orbitals, or in ten X2C
    spinors mixed by complex unitary matrices among the occupied and among
    the virtual ones (complex integrals, a Fock matrix that is not
    diagonal): CCSDT is the full CI, and the 119 eigenvalues of the EOM
    matrix are the differences of the eigenvalues of H among the
    determinants of three electrons.'''
    mf = three_electron(kind, 'sto-3g')
    C = E.rotate_occ_vir(mf.mo_coeff, mf.mo_occ, seed=3) if kind == 'x2c' else mf.mo_coeff
    fock, g, mo_e, no = R.integrals(E.from_scf(mf, mo_coeff=C))
    n = fock.shape[0]
    nex = comb(n, 3) - 1
    assert (no, n) == (3, 10) and (abs(g.imag).max() > 1e-3) == (kind == 'x2c')
    rt = R.kernel(fock, g, mo_e, no, 'ccsdt', conv_tol=1e-13, conv_tol_normt=1e-11)
    assert rt.converged
    w = EOM.eigenvalues(rt.amps, fock, g, no)
    fk = F.Fock(no, n - no)
    H = (F.one_body(fk, fock) + fk.two_body(g)).toarray()
    sector = np.flatnonzero(np.array([bin(x).count('1') for x in range(2 ** n)]) == no)
    e = np.linalg.eigvalsh(H[np.ix_(sector, sector)])
    assert len(w) == nex and abs(w.imag).max() < 1e-9
    assert abs(w.real - (e[1:] - e[0])).max() < 1e-9
    assert abs(w[0]) < 1e-9 and w[1].real > 0.05      # (the other state of the doublet)
    # the diagonal of the zeroth order: the differences of the orbital energies
    d = EOM.diagonal(mo_e, no, 3)
    assert d.shape == (nex,) and abs(d[0] - (mo_e[no] - mo_e[0])) < 1e-13


def _nearest(w, values):
    return max(abs(w - x).min() for x in np.atleast_1d(values))


SOLVERS = (('ee', eom_gccsd.EOMEE), ('ip', eom_gccsd.EOMIP), ('ea', eom_gccsd.EOMEA))


def _gccsd(mf, C):
    mycc = cc.GCCSD(mf, mo_coeff=C)
    mycc.conv_tol, mycc.conv_tol_normt = 1e-12, 1e-10
    mycc.kernel()
    return mycc


@functools.lru_cache(maxsize=None)
def pyscf_roots(basis='sto-3g', nroots=4):
    """{kind: the lowest roots of PySCF's EOM-CCSD} for H2O with real GHF
    spin orbitals."""
    mycc = _gccsd(water_ghf(basis), None)
    out = {}
    for kind, solver in SOLVERS:
        eom = solver(mycc)
        eom.conv_tol, eom.max_space, eom.max_cycle = 1e-11, 60, 200
        out[kind] = eom.kernel(nroots=nroots)[0]
    return out


@pytest.mark.parametrize('cplx', [False, True])
def test_eom_ccsd_against_pyscf(cplx):
    '''H2O / STO-3G, all electrons, 10 + 4 spin orbitals: the lowest roots
    of PySCF's EOM-EE-, IP- and EA-CCSD for a GHF reference are eigenvalues
    of the reference matrix (EE: 310 x 310; IP and EA through the ghost
    orbital) to 1e-8.  With complex spin orbitals (every pair of spin
    orbitals mixed by an SU(2) matrix of its own) the reference gives the
    same eigenvalues from PySCF's complex amplitudes (PySCF's own EOM solver
    is not run there: with these orbitals its EA guess gave four roots of
    1e-16, and its real-eigenvalue filter stopped the EE and IP runs in
    some orders of the calls).'''
    mf = water_ghf('sto-3g')
    C = _spin_mixed(mf)[0] if cplx else mf.mo_coeff
    mycc = _gccsd(mf, C)
    fock, g, mo_e, no = R.integrals(E.from_scf(mf, mo_coeff=C))
    assert (abs(g.imag).max() > 1e-3) == cplx
    amps = [mycc.t1, mycc.t2]
    ref = pyscf_roots()
    for kind, solver in SOLVERS:
        w = EOM.eigenvalues(amps, fock, g, no, kind)
        assert abs(w.imag).max() < 1e-8
        assert _nearest(w, ref[kind]) < 1e-8, (kind, ref[kind])
