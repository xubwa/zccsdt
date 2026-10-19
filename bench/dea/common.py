'''EOM-DEA of O2 from the closed-shell O2(2+) determinant: reading a
Hamiltonian file, the ground states and the levels (docs/eom.md, section 11).

A Hamiltonian file (.npz): h1 (n,n), eri (n,n,n,n) in chemists' notation
(pq|rs), H = ecore + sum h1[p,q] p+ q + 1/2 sum (pq|rs) p+ r+ s q; ecore;
ncore = 10 (the spinors of the reference determinant come first); omega2
(2 Omega of each spinor); parity (0 = g, 1 = u; optional); eps.

Script-level code (a standing decision: no general helper of this kind in
the package).'''
import json
import time

import numpy as np

H2CM = 219474.6313632

LEVELS = ('ccsd', 'ccsd+t3', 'cc3', 'ccsdt', 'cc4', 'ccsdt+t4cc4', 'ccsdt+t4')
TEXT = {
    'ccsd':        ('DEA-EOM-CCSD',            '2p + 3p1h',                       'CCSD (T1, T2)'),
    'ccsd+t3':     ('DEA-EOM-CCSD, T of CCSDT', '2p + 3p1h',                      'CCSDT (T1, T2, T3)'),
    'cc3':         ('DEA-EOM-CC3',             '2p + 3p1h, the 3p1h row of CC3',  'CC3'),
    'ccsdt':       ('DEA-EOM-CCSDT',           '2p + 3p1h + 4p2h',                'CCSDT (T1, T2, T3)'),
    'cc4':         ('DEA-EOM-CC4',             '2p + 3p1h + 4p2h, the 4p2h row of CC4', 'CC4'),
    'ccsdt+t4cc4': ('DEA-EOM-CCSDT, T of CC4', '2p + 3p1h + 4p2h',                'CC4 (T1 .. T3, T4 of CC4)'),
    'ccsdt+t4':    ('DEA-EOM-CCSDT, T of CCSDTQ', '2p + 3p1h + 4p2h',             'CCSDTQ (T1 .. T4)'),
}


def load(path, tol=1e-10, drop_forbidden=False, sort=False, freeze=0):
    '''(eris, labels, parity, info) from a Hamiltonian file: the integral
    container of the reference determinant of the first ncore spinors -
    <pq||rs> = (pr|qs) - (ps|qr), the Fock matrix f = h1 + sum_i <pi||qi>,
    e_hf = ecore + sum_i h1[i,i] + 1/2 sum_ij <ij||ij> - real if the
    integrals are (imaginary parts below ``tol``).
    ``drop_forbidden``: integrals that the labels omega2 (and parity) forbid
    are set to zero; info['forbidden'] is the largest of them in any case.'''
    from zccsdt.eris import BLOCKS, ERIs
    z = np.load(path)
    h1, eri = np.asarray(z['h1']), np.asarray(z['eri'])
    no = int(z['ncore'])
    n = h1.shape[0]
    lab = np.asarray(z['omega2'], dtype=np.int64)
    par = np.asarray(z['parity'], dtype=np.int64) if 'parity' in z.files else None
    ecore = float(z['ecore'])
    if freeze:
        # the first ``freeze`` spinors of the reference frozen: their mean field in h1, their energy in ecore
        c, a = slice(0, freeze), slice(freeze, n)
        vf = np.einsum('pqii->pq', eri[:, :, c, c]) - np.einsum('piiq->pq', eri[:, c, c, :])
        ecore = float((ecore + np.trace(h1[c, c]) + 0.5 * np.trace(vf[c, c])).real)
        h1, eri = (h1 + vf)[a, a], np.ascontiguousarray(eri[a, a, a, a])
        lab, par = lab[freeze:], None if par is None else par[freeze:]
        no, n = no - freeze, n - freeze
    info = {'n': n, 'ncore': no, 'imag': float(max(abs(h1.imag).max(), abs(eri.imag).max()))}
    if sort:
        # the spinors in the order of their labels (by energy inside a label): the layout of the label blocks
        perm = np.concatenate((np.argsort(lab[:no], kind='stable'), no + np.argsort(lab[no:], kind='stable')))
        h1, eri, lab = h1[np.ix_(perm, perm)], eri[np.ix_(perm, perm, perm, perm)], lab[perm]
        par = None if par is None else par[perm]
        info['perm'] = perm
    if info['imag'] < tol:
        h1, eri = h1.real.copy(), eri.real.copy()
    # (pq|rs) is allowed where l_p + l_r = l_q + l_s (and the parities add to an even number)
    ok1 = lab[:, None] == lab[None, :]
    d = lab[:, None] - lab[None, :]                                    # l_p - l_q
    ok2 = d[:, :, None, None] + d[None, None, :, :] == 0
    if par is not None:
        ok1 &= par[:, None] == par[None, :]
        s = (par[:, None] + par[None, :]) % 2
        ok2 &= s[:, :, None, None] == s[None, None, :, :]
    info['forbidden'] = float(max(abs(h1[~ok1]).max(initial=0.0), abs(eri[~ok2]).max(initial=0.0)))
    if drop_forbidden:
        h1 = h1 * ok1
        eri *= ok2
    del ok2
    g = eri.transpose(0, 2, 1, 3)               # <pq|rs> = (pr|qs)
    g = g - g.transpose(0, 1, 3, 2)
    g = 0.5 * (g - g.transpose(1, 0, 2, 3))
    del eri
    o, v = slice(0, no), slice(no, None)
    fock = h1 + np.einsum('piqi->pq', g[:, o, :, o])
    e_hf = ecore + float(np.einsum('ii->', h1[o, o]).real) + 0.5 * float(np.einsum('ijij->', g[o, o, o, o]).real)
    blocks = {name: np.ascontiguousarray(g[tuple(o if c == 'o' else v for c in name)]) for name in BLOCKS}
    del g
    eris = ERIs(nocc=no, fock=fock, mo_energy=fock.diagonal().real.copy(), e_hf=e_hf, e_nuc=0.0, source=str(path), **blocks)
    info['fock_offdiag_oo_vv'] = float(max(abs(fock[o, o] - np.diag(fock.diagonal()[o])).max(),
                                           abs(fock[v, v] - np.diag(fock.diagonal()[v])).max()))
    info['fock_ov'] = float(abs(fock[o, v]).max())
    if info['fock_offdiag_oo_vv'] < 1e-7:
        # canonical spinors: the rounding of the SCF is taken out of f_oo and f_vv, so that
        # the semicanonicalization of CC3 / CC4 (which has no labels here) rotates nothing -
        # it would mix the two spinors +-m_j of a Kramers pair and spoil the labels
        fock[o, o] = np.diag(fock.diagonal()[o])
        fock[v, v] = np.diag(fock.diagonal()[v])
    info['e_ref'] = e_hf
    if sort:
        eris.labels, eris.modulus, eris.use_labels = lab.copy(), 0, True
    return eris, lab, par, info


def ground(eris, model, tight=1e-11, start=None, verbose=False):
    '''The ground state of the reference: dict(e_corr, t1, t2, t3, t4, eris,
    seconds, ncycle).  ``start``: amplitudes of a lower model.'''
    from zccsdt.cc import ccn, ccsdtq, solver
    t0 = time.perf_counter()
    s = start or {}
    kw = {k: s.get(k) for k in ('t1', 't2', 't3')}
    out = {'eris': eris, 't3': None, 't4': None}
    if model in ('ccsd', 'ccsdt', 'cc3'):
        if model == 'ccsd':
            kw.pop('t3')
        r = solver.kernel(eris, model, solver.Options(conv_tol=tight * 1e-2, conv_tol_normt=tight, verbose=verbose,
                                                      incore_complete=True), **kw)
        out.update(t3=getattr(r, 't3', None) if model != 'ccsd' else None)
        if model == 'cc3':
            out['eris'] = r.eris
    elif model == 'ccsdtq' and getattr(eris, 'use_labels', False):
        from zccsdt.cc import ccsdtq_sym
        r = ccsdtq_sym.kernel(eris, ccsdtq.Options(conv_tol=tight * 1e-2, conv_tol_normt=tight, verbose=verbose), **kw)
        out.update(t3=r.t3, t4=r.t4)
    elif model == 'ccsdtq':
        r = ccsdtq.kernel(eris, ccsdtq.Options(conv_tol=tight * 1e-2, conv_tol_normt=tight, verbose=verbose), **kw)
        out.update(t3=r.t3, t4=r.t4)
    elif model == 'cc4':
        r = ccn.cc4_kernel(eris, conv_tol=tight * 1e-2, conv_tol_normt=tight, verbose=verbose, **kw)
        out.update(t3=r.t3, t4=r.t4, eris=r.eris)
    else:
        raise ValueError(model)
    if not r.converged:
        raise RuntimeError('%s did not converge' % model)
    out.update(e_corr=float(np.real(r.e_corr)), t1=r.t1, t2=r.t2, seconds=time.perf_counter() - t0, ncycle=r.ncycle,
               model=model)
    return out


GROUND_OF = {'ccsd': 'ccsd', 'ccsd+t3': 'ccsdt', 'cc3': 'cc3', 'ccsdt': 'ccsdt', 'cc4': 'cc4', 'ccsdt+t4cc4': 'cc4',
             'ccsdt+t4': 'ccsdtq'}
PROBLEM_OF = {'ccsd': dict(rank=3), 'ccsd+t3': dict(rank=3), 'cc3': dict(rank=3, row3='cc3'), 'ccsdt': dict(rank=4),
              'cc4': dict(rank=4, row4='cc4'), 'ccsdt+t4cc4': dict(rank=4), 'ccsdt+t4': dict(rank=4)}


def run_levels(eris, lab, par, levels, targets=(0, 2), conv_tol=1e-9, tight=1e-11, nroots=1, nguess=4, out=None,
               verbose=True, semicanonical_ok=True):
    '''Every level in every block: {level: dict(e_cc, roots {target: [omega ..]}, ..)}.
    Blocks: the sum of the labels of the two electrons = target, parity g.'''
    from zccsdt.cc import dea
    results, states = {}, {}
    for lev in levels:
        gm = GROUND_OF[lev]
        if gm not in states:
            start = states.get({'ccsdt': 'ccsd', 'ccsdtq': 'ccsdt'}.get(gm))
            if start is not None and start['eris'] is not eris:
                start = None
            states[gm] = ground(eris, gm, tight, start)
            if verbose:
                print('ground state %-7s E_corr = %.12f  E = %.11f  (%d iterations, %.1f s)'
                      % (gm.upper(), states[gm]['e_corr'], eris.e_hf + states[gm]['e_corr'], states[gm]['ncycle'],
                         states[gm]['seconds']), flush=True)
        st = states[gm]
        t3 = None if lev == 'ccsd' else st['t3']
        t4 = st['t4'] if lev in ('ccsdt+t4cc4', 'ccsdt+t4') else None
        if lev == 'cc4':
            t4 = st['t4']
        rec = {'name': TEXT[lev][0], 'operator': TEXT[lev][1], 'T': TEXT[lev][2], 'e_cc': eris.e_hf + st['e_corr'],
               'ground_seconds': st['seconds'], 'roots': {}, 'residuals': {}, 'nsigma': {}, 'sigma_seconds': {},
               'sizes': {}, 'weights': {}}
        for tgt in targets:
            if getattr(eris, 'use_labels', False):
                if lev in ('cc3', 'cc4', 'ccsdt+t4cc4'):
                    raise ValueError('%s: on the packed layout' % lev)
                pb = dea.ProblemSym(st['eris'], st['t1'], st['t2'], t3, t4, target=tgt, parity=par, target_parity=0,
                                    rank=PROBLEM_OF[lev]['rank'])
                if verbose:
                    print('  label blocks: T3 %d, T4 %d elements with the ghosts' % pb.block_sizes, flush=True)
            else:
                pb = dea.Problem(st['eris'], st['t1'], st['t2'], t3, t4, labels=lab, target=tgt, parity=par, target_parity=0,
                                 **PROBLEM_OF[lev])
            r = dea.kernel(None, None, None, problem=pb, nroots=nroots, nguess=nguess, conv_tol=conv_tol, verbose=verbose,
                           max_cycle=120)
            if not r.converged.all():
                print('  warning: not converged', r.residuals, flush=True)
            rec['roots'][tgt] = [float(x) for x in r.e]
            rec['residuals'][tgt] = [float(x) for x in r.residuals]
            rec['nsigma'][tgt], rec['sigma_seconds'][tgt] = r.nsigma, r.timings['sigma']
            rec['sizes'][tgt] = [int(x) for x in r.sizes]
            rec['weights'][tgt] = [[float(y) for y in w / w.sum()] for w in r.weights]
            del pb
        e0, e1 = rec['e_cc'] + rec['roots'][targets[0]][0], rec['e_cc'] + rec['roots'][targets[1]][0]
        rec['E'] = {targets[0]: e0, targets[1]: e1}
        rec['split'] = e1 - e0
        if verbose:
            print('LEVEL %-12s E(2 Omega = %d) = %.11f  E(2 Omega = %d) = %.11f  split = %.4e Eh = %.4f cm-1'
                  % (lev, targets[0], e0, targets[1], e1, rec['split'], rec['split'] * H2CM), flush=True)
        results[lev] = rec
        if out:
            with open(out, 'w') as f:
                json.dump(results, f, indent=1)
    return results


def table(results, exact=None):
    '''Lines of the table of the levels; ``exact``: (E0, E1) of the full CI.'''
    lines = ['%-12s %-34s %-22s %17s %17s %12s %9s' % ('level', 'R', 'T', 'E(2 Omega = 0)', 'E(2 Omega = 2)',
                                                        'split / Eh', 'cm-1')]
    for lev, r in results.items():
        e = list(r['E'].values())
        lines.append('%-12s %-34s %-22s %17.11f %17.11f %12.4e %9.4f' % (lev, r['operator'], r['T'], e[0], e[1], r['split'],
                                                                      r['split'] * H2CM))
        if exact is not None:
            lines.append('%-12s %-34s %-22s %17.2e %17.2e %12.2e %9.4f' % ('', '  error against the full CI', '',
                                                                      e[0] - exact[0], e[1] - exact[1],
                                                                      r['split'] - (exact[1] - exact[0]),
                                                                      (r['split'] - (exact[1] - exact[0])) * H2CM))
    if exact is not None:
        lines.append('%-12s %-34s %-22s %17.11f %17.11f %12.4e %9.4f' % ('full CI', '', '', exact[0], exact[1],
                                                                      exact[1] - exact[0], (exact[1] - exact[0]) * H2CM))
    return lines
