'''O2: the Hamiltonian with the four-component two-electron operator
("X2Ccorr", the scheme of the group; it contains spin-spin coupling) for
the spinors of the X2CMP calculation of O2(2+), and its Hartree-Fock
spinors inside the correlated space.

The spinors C are taken back to the four-component picture (large R C,
small X R C with X, R of the molecular X2C step:
socutils.tools.fcidump_rel.x2c_4c_spinors);

    h1     = C4+ (h_D + J_c - K_c) C4      bare four-component one-electron operator plus the
                                           mean field (Coulomb, Gaunt, Breit) of the frozen 1s
                                           spinors (pyscf.scf.DHF with Gaunt and Breit)
    (pq|rs)  no-pair Dirac-Coulomb-Breit integrals of the correlated spinors
             (fcidump_rel.x2c_active_eri_4c)
    ecore  = E_nuc + Tr D_c (h_D + 1/2 (J_c - K_c))

The model-potential mean field of the X2CMP one-electron operator is not
used: nothing is counted twice.

usage:
  python o2_x2ccorr.py check <their_small_x2ccorr.npz> <their_116.npz>
      the assembly for the spinors stored in a file of the other session
      against the h1, eri, ecore of that file
  python o2_x2ccorr.py run <dir> [n ...]
      the spinors of <dir>/o2_2plus_krhf_x2cmp.chk (bench/dea/o2_scf.py): the Hamiltonian of
      the 116 correlated spinors, closed-shell Hartree-Fock for the 10 electrons of O2(2+)
      inside that space (label block by label block), canonical spinors, and the files
      <dir>/o2_hf2plus_n<n>_x2ccorr.npz of the n lowest of them
  python o2_x2ccorr.py run-blocks <dir> [n ...] [--out <dir2>] [--max-memory MB] [--npz] [--direct] [--fold-core]
      the same Hamiltonian without the n^4 array (zccsdt.eri4c, zccsdt.hblocks): integrals in the
      blocks of the labels and parities, the HF and its rotation block by block, the block files
      <dir2 or dir>/o2_hf2plus_n<n>_x2ccorr.h5 (with --npz also the .npz of the dense route);
      --max-memory: MB for the whole process (default: 80 % of what is free); --direct: all eight
      classes of AO integrals (default: five and the mirror relations); the raw integrals of the
      X2CMP spinors go to a scratch file (ZCCSDT_TMPDIR / TMPDIR); the mean field of the frozen 1s
      spinors from the four-component J, K of their density as in run, or with --fold-core from the
      blocks of the frozen spinors transformed with the others (faster; differs by the rounding of
      the coefficients off the labels: 1e-8 in h1 for DYVTZ-SOG)'''
import os
import sys
import time

import numpy as np
from pyscf import lib, scf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import o2_scf  # noqa: E402

T0 = time.time()


def frozen_field(mf, cf):
    '''(hcore4 + vhf of the frozen spinors cf, ecore, the DHF object).'''
    from socutils.tools import fcidump_rel
    xmol, fl, fs = fcidump_rel.x2c_4c_spinors(mf, cf)
    mf4c = scf.DHF(xmol)
    mf4c.with_gaunt = True
    mf4c.with_breit = True
    hcore4 = mf4c.get_hcore()
    if cf.shape[1] == 0:            # nothing frozen: the bare four-component one-electron operator
        return hcore4, float(mf.mol.energy_nuc())
    c4f = np.vstack((fl, fs))
    dm = c4f @ c4f.conj().T
    vj, vk = mf4c.get_jk(xmol, dm)
    vhf = vj - vk
    ecore = (mf.mol.energy_nuc() + np.einsum('ij,ji->', dm, hcore4) + 0.5 * np.einsum('ij,ji->', dm, vhf)).real
    return hcore4 + vhf, float(ecore)


def assemble(mf, hc, c):
    '''(h1, (pq|rs)) of the spinors c (two-component coefficients).'''
    from socutils.tools import fcidump_rel
    n = c.shape[1]
    _, ml, ms = fcidump_rel.x2c_4c_spinors(mf, c)
    c4 = np.vstack((ml, ms))
    h1 = c4.conj().T @ hc @ c4
    eri = fcidump_rel.x2c_active_eri_4c(mf, c, with_full_coulomb=True, with_gaunt=True, with_breit=True)
    return h1, np.asarray(eri).reshape(n, n, n, n)


def check(small, big):
    import runpy
    from pyscf import gto
    from socutils.scf import spinor_hf
    from socutils.somf import x2cmp
    sog = runpy.run_path(o2_scf.BASIS)['O']
    for charge in (0, 2):
        mol = gto.M(atom='O 0 0 0; O 0 0 1.20752', basis={'O': sog}, verbose=0, nucmod='G', charge=charge,
                    spin=2 - charge)
        mf = spinor_hf.KRHF(mol)
        mf.with_x2c = x2cmp.X2CMP(mol, x2cmp='x2cmp', with_gaunt=True, with_breit=True)
        z = np.load(small)
        c = z['mo_coeff']
        c116 = np.load(big)['mo_coeff']
        s = mol.intor('int1e_ovlp_spinor')
        rest = np.eye(len(s)) - c116 @ c116.conj().T @ s
        w, v = np.linalg.eigh(rest.conj().T @ s @ rest)
        cf = rest @ v[:, -4:] / np.sqrt(w[-4:])
        hc, ecore = frozen_field(mf, cf)
        h1, eri = assemble(mf, hc, c)
        print('check (molecule object with charge %d): %s, %d spinors: ecore %.12f (file %.12f, difference %.1e); '
              'max |h1 - file| %.1e; max |eri - file| %.1e   (%.0f s)'
              % (charge, small, c.shape[1], ecore, float(z['ecore']), ecore - float(z['ecore']),
                 abs(h1 - z['h1']).max(), abs(eri - z['eri']).max(), time.time() - T0), flush=True)


def hartree_fock(h1, eri, ecore, no, lab, par, tol=1e-12, max_cycle=100):
    '''Closed-shell Hartree-Fock of the ``no`` electrons in the given
    spinors (the first ``no`` occupied to start), block by block of the
    labels: (U, energies, E, E of the start).  The new spinors are the
    columns of U: occupied by energy, then virtual by energy.'''
    n = h1.shape[0]
    keys = sorted({(int(a), int(b)) for a, b in zip(lab, par)})
    blocks = [np.flatnonzero((lab == a) & (par == b)) for a, b in keys]
    nocc = [int((idx < no).sum()) for idx in blocks]
    D = np.zeros((n, n), dtype=h1.dtype)
    D[np.arange(no), np.arange(no)] = 1.0

    def fock(D):
        return h1 + np.einsum('pqrs,sr->pq', eri, D) - np.einsum('psrq,sr->pq', eri, D)

    def energy(D, F):
        return float(ecore + 0.5 * np.einsum('pq,qp->', h1 + F, D).real)
    F = fock(D)
    e_start = energy(D, F)
    out = np.max([abs(F[np.ix_(a, b)]).max() for i, a in enumerate(blocks) for j, b in enumerate(blocks) if i != j])
    print('HF in the correlated space: E(start) = %.12f; Fock matrix of the start: max |f_ov| %.2e, largest element '
          'between label blocks %.1e' % (e_start, abs(F[:no, no:]).max(), out), flush=True)
    for it in range(max_cycle):
        U = np.zeros((n, n), dtype=h1.dtype)
        eps = np.zeros(n)
        occ = np.zeros(n, bool)
        for idx, k in zip(blocks, nocc):
            w, v = np.linalg.eigh(F[np.ix_(idx, idx)])
            U[np.ix_(idx, idx)] = v
            eps[idx] = w
            occ[idx[:k]] = True
        Dn = U[:, occ] @ U[:, occ].conj().T
        change = abs(Dn - D).max()
        D = Dn
        F = fock(D)
        e = energy(D, F)
        print('  cycle %2d  E = %.13f  |dD| = %.2e' % (it + 1, e, change), flush=True)
        if change < tol:
            break
    else:
        raise RuntimeError('HF not converged')
    order = np.concatenate((np.flatnonzero(occ)[np.argsort(eps[occ], kind='stable')],
                            np.flatnonzero(~occ)[np.argsort(eps[~occ], kind='stable')]))
    U = U[:, order]
    big = U[np.argmax(abs(U), axis=0), np.arange(n)]
    U = U * (abs(big) / big)
    return U, eps[order], e, e_start


def run(outdir, sizes):
    mf = o2_scf.run_scf(os.path.join(outdir, 'o2_2plus_krhf_x2cmp%s.chk' % o2_scf.TAG), verbose=0)
    C, om2, par, io, iv, e = o2_scf.spinors(mf)
    nf = o2_scf.NFROZEN
    cols = np.concatenate((io[nf:], iv))
    c, lab, pr = C[:, cols], om2[cols], par[cols]
    no, n = len(io) - nf, len(cols)
    hc, ecore = frozen_field(mf, C[:, io[:nf]])
    print('four-component frozen-core mean field: ecore = %.12f  (%.0f s)' % (ecore, time.time() - T0), flush=True)
    h1, eri = assemble(mf, hc, c)
    print('integrals of %d spinors (%.0f s): max |Im h1| %.1e, max |Im eri| %.1e; hermiticity h1 %.1e, '
          '(pq|rs) - (rs|pq) %.1e' % (n, time.time() - T0, abs(h1.imag).max(), abs(eri.imag).max(),
                                      abs(h1 - h1.conj().T).max(), abs(eri - eri.transpose(2, 3, 0, 1)).max()), flush=True)
    if max(abs(h1.imag).max(), abs(eri.imag).max()) < 1e-9:
        h1, eri = h1.real.copy(), eri.real.copy()
    U, eps, e_hf, e_start = hartree_fock(h1, eri, ecore, no, lab, pr)
    dev = U - np.eye(n)
    # (the spinors of a label block may come in another order: compare the spaces)
    s_occ = np.linalg.svd(U[:no, :no], compute_uv=False)
    print('HF: E = %.12f, lowering against the determinant of the X2CMP spinors %.3e Eh; smallest singular value '
          'of the occupied-occupied block of U %.10f (largest rotation angle %.2e rad); max |U_vo| %.2e'
          % (e_hf, e_hf - e_start, s_occ.min(), np.arccos(min(1.0, s_occ.min())), abs(U[no:, :no]).max()), flush=True)
    h1 = U.conj().T @ h1 @ U
    for axis in range(4):
        eri = np.moveaxis(np.tensordot(eri, U.conj() if axis in (0, 2) else U, axes=([axis], [0])), -1, axis)
    eri = np.ascontiguousarray(eri)
    # the labels of the new spinors: those of the block of each column
    k = np.argmax(abs(U), axis=0)
    lab, pr = lab[k], pr[k]
    c = c @ U
    f = h1 + np.einsum('pqii->pq', eri[:, :, :no, :no]) - np.einsum('piiq->pq', eri[:, :no, :no, :])
    print('canonical spinors: max off-diagonal Fock element %.1e; energies and labels (2 Omega, parity):'
          % abs(f - np.diag(f.diagonal())).max())
    for p in range(n):
        print('  %4d %14.8f %3d %s%s' % (p + 1, f[p, p].real, lab[p], 'gu'[pr[p]], '  occupied' if p < no else ''))
    for m in sizes:
        m = m or n
        if m < n and abs(eps[m] - eps[m - 1]) < 1e-7:
            raise RuntimeError('%d spinors cut a degenerate level' % m)
        s = slice(0, m)
        e_ref = ecore + np.einsum('ii->', h1[:no, :no]).real + 0.5 * (
            np.einsum('iijj->', eri[:no, :no, :no, :no]) - np.einsum('ijji->', eri[:no, :no, :no, :no])).real
        name = os.path.join(outdir, 'o2_hf2plus%s_n%d_x2ccorr.npz' % (o2_scf.TAG, m))
        np.savez(name, h1=h1[s, s].astype(complex), eri=eri[s, s, s, s].astype(complex), ecore=ecore, nelec=no + 2, ncore=no,
                 nact=4, omega2=lab[s], parity=pr[s], eps=f.diagonal().real[s], mo_coeff=c[:, s],
                 mo_frozen=C[:, io[:nf]], e_hf=e_hf)
        print('n = %3d: %s  E(reference determinant) %.12f; pi*: 2 Omega %s parity %s  (%.0f s)'
              % (m, name, e_ref, lab[no:no + 4], pr[no:no + 4], time.time() - T0), flush=True)


def rss():
    from zccsdt import memory
    return memory.resident() / 1e9


def run_blocks(outdir, sizes, max_memory=None, npz=False, direct=False, dest=None, dhf_field=True):
    '''run() without the n^4 array: zccsdt.eri4c into a block file of the X2CMP
    spinors, the HF and the rotation on its blocks (zccsdt.hblocks).  The
    mean field of the frozen 1s spinors: with ``dhf_field`` by the
    four-component J and K of their density (frozen_field, as run(): 1521 s
    for cc-pV5Z-X2C-TP on 32 cores), else from the same blocks (the frozen
    spinors transformed with the others and folded, hblocks.fold).'''
    import tempfile
    from zccsdt import eri4c, hblocks
    from zccsdt.scratch import scratch_dir
    mf = o2_scf.run_scf(os.path.join(outdir, 'o2_2plus_krhf_x2cmp%s.chk' % o2_scf.TAG), verbose=0)
    C, om2, par, io, iv, e = o2_scf.spinors(mf)
    nf = o2_scf.NFROZEN
    cols = np.concatenate((io[nf:], iv))
    c, lab, pr = C[:, cols], om2[cols], par[cols]
    no, n = len(io) - nf, len(cols)
    if dhf_field:
        hc, ecore = frozen_field(mf, C[:, io[:nf]])
        nfold = 0
    else:
        hc, ecore = frozen_field(mf, C[:, :0])         # the bare four-component operator, E_nuc
        nfold = nf
        cols = np.concatenate((io, iv))
        c, lab, pr = C[:, cols], om2[cols], par[cols]
    print('four-component one-electron operator%s: ecore = %.12f  (%.0f s, RssAnon %.2f GB)'
          % (' and frozen-core mean field' if dhf_field else '', ecore, time.time() - T0, rss()), flush=True)
    _, ml, ms = eri4c.x2c_4c_spinors(mf, c)
    c4 = np.vstack((ml, ms))
    h1 = c4.conj().T @ hc @ c4
    del hc, c4, ml, ms
    real = abs(h1.imag).max() < 1e-9 and abs(c.imag).max() < 1e-12
    if real:
        h1 = h1.real.copy()
    # (the elements of h1 between groups are rounding, as those of (pq|rs) that are not stored; in the
    # dense route they cancel against the latter in the Fock matrix: f_ov 1e-10 there, 5e-9 with them
    # kept here for the 116 spinors of DYVTZ-SOG)
    ok1 = (lab[:, None] == lab[None, :]) & (pr[:, None] == pr[None, :])
    print('h1: largest element between the groups of (2 Omega, parity) %.1e (set to zero)'
          % abs(h1[~ok1]).max(initial=0.0), flush=True)
    h1 = h1 * ok1
    t0 = time.time()
    fd, raw = tempfile.mkstemp(prefix='o2_x2ccorr_raw_', suffix='.h5', dir=scratch_dir())
    os.close(fd)
    bf = None
    name = os.path.join(dest or outdir, 'o2_hf2plus%s_n%d_x2ccorr.h5' % (o2_scf.TAG, n))
    try:
        for buf, layout in eri4c.make(mf, c, lab, pr, direct=direct, real=real, max_memory=max_memory):
            if bf is None:
                groups, members = hblocks.layout_members(layout)
                bf = hblocks.BlockFile.create(raw, groups, members, no + nfold, ecore, h1, buf.dtype)
            forb = hblocks.write_label_buffer(bf, buf, layout)
            print('label blocks written (%.2f GB of blocks in the file); largest element that the parities forbid '
                  '%.1e  (RssAnon %.2f GB)' % (bf.nbytes() / 1e9, forb, rss()), flush=True)
            del buf
        print('integrals of %d spinors in blocks (%.0f s): %s, %d blocks, %.2f GB (dense complex: %.2f GB); '
              'hermiticity h1 %.1e  (RssAnon %.2f GB)'
              % (n + nfold, time.time() - t0, bf.dtype.name, len(bf.blocks), bf.nbytes() / 1e9,
                 16 * (n + nfold) ** 4 / 1e9, abs(h1 - h1.conj().T).max(), rss()), flush=True)
        if nfold:
            t1 = time.time()
            fd, raw2 = tempfile.mkstemp(prefix='o2_x2ccorr_raw_', suffix='.h5', dir=scratch_dir())
            os.close(fd)
            folded = hblocks.fold(bf, nfold, raw2)
            bf.close()
            os.remove(raw)
            bf, raw = folded, raw2
            h1 = bf.h1
            lab, pr = lab[nfold:], pr[nfold:]
            c = c[:, nfold:]
            print('frozen-core mean field from the blocks of the %d frozen spinors: ecore = %.12f  (%.0f s)'
                  % (nfold, bf.ecore, time.time() - t1), flush=True)
        t1 = time.time()
        U, members, eps, e_hf, e_start = hblocks.hartree_fock(bf, h1, no)
        Uf = np.zeros((n, n), dtype=np.result_type(*U.values()))
        for g, m in enumerate(bf.members):
            Uf[np.ix_(m, members[g])] = U[g]
        s_occ = np.linalg.svd(Uf[:no, :no], compute_uv=False)
        print('HF (%.0f s): E = %.12f, lowering against the determinant of the X2CMP spinors %.3e Eh; smallest '
              'singular value of the occupied-occupied block of U %.10f (largest rotation angle %.2e rad); max |U_vo| '
              '%.2e  (RssAnon %.2f GB)' % (time.time() - t1, e_hf, e_hf - e_start, s_occ.min(),
                                           np.arccos(min(1.0, s_occ.min())), abs(Uf[no:, :no]).max(), rss()),
              flush=True)
        h1n = Uf.conj().T @ h1 @ Uf
        cn = c @ Uf
        k = np.argmax(abs(Uf), axis=0)
        labn, prn = lab[k], pr[k]
        t1 = time.time()
        out = hblocks.rotate(bf, U, members, name + '.part', h1n, attrs=dict(e_hf=e_hf, nelec=no + 2, nact=4),
                             arrays=dict(omega2=labn, parity=prn, mo_coeff=cn, mo_frozen=C[:, io[:nf]]))
        bf.close()
        bf = None
        f = out.fock(h1n)
        out.h5['eps'] = f.diagonal().real
        print('rotation of the blocks (%.0f s, RssAnon %.2f GB); canonical spinors: largest off-diagonal Fock '
              'element in a group %.1e; energies and labels (2 Omega, parity):'
              % (time.time() - t1, rss(), abs(f - np.diag(f.diagonal())).max()))
        for p in range(n):
            print('  %4d %14.8f %3d %s%s' % (p + 1, f[p, p].real, labn[p], 'gu'[prn[p]],
                                             '  occupied' if p < no else ''))
        e_ref = out.reference_energy(h1n)
        out.close()
        os.replace(name + '.part', name)
        print('n = %3d: %s  E(reference determinant) %.12f; pi*: 2 Omega %s parity %s  (%.0f s)'
              % (n, name, e_ref, labn[no:no + 4], prn[no:no + 4], time.time() - T0), flush=True)
    finally:
        if bf is not None:
            bf.close()
        os.remove(raw)
    for m in sizes:
        m = m or n
        out = hblocks.BlockFile(name)
        if m < n:
            if abs(out['eps'][m] - out['eps'][m - 1]) < 1e-7:
                raise RuntimeError('%d spinors cut a degenerate level' % m)
            sub = os.path.join(dest or outdir, 'o2_hf2plus%s_n%d_x2ccorr.h5' % (o2_scf.TAG, m))
            arrays = {k: out[k][:m] for k in ('omega2', 'parity', 'eps')}
            arrays.update(mo_coeff=out['mo_coeff'][:, :m], mo_frozen=out['mo_frozen'])
            s = hblocks.subset(out, m, sub, arrays=arrays, attrs={k: out.attr(k) for k in ('e_hf', 'nelec', 'nact')})
            s.close()
            out.close()
            out = hblocks.BlockFile(sub)
            print('n = %3d: %s' % (m, sub), flush=True)
        if npz:
            path = out.path[:-3] + '.npz'
            np.savez(path, h1=out.h1.astype(complex), eri=out.to_dense().astype(complex), ecore=out.ecore,
                     nelec=out.attr('nelec'), ncore=out.ncore, nact=4, omega2=out['omega2'], parity=out['parity'],
                     eps=out['eps'], mo_coeff=out['mo_coeff'], mo_frozen=out['mo_frozen'], e_hf=out.attr('e_hf'))
            print('  dense copy %s' % path, flush=True)
        out.close()
    with open('/proc/self/status') as f:
        hwm = [x for x in f if x.startswith('VmHWM')]
    print('peak memory of the process: %s' % hwm[0].split(':')[1].strip(), flush=True)


def _option(args, key, conv, default=None):
    if key in args:
        k = args.index(key)
        v = conv(args[k + 1])
        del args[k:k + 2]
        return v
    return default


if __name__ == '__main__':
    lib.num_threads(int(os.environ.get('OMP_NUM_THREADS', 1)))
    args = sys.argv[1:]
    if args[0] == 'check':
        check(args[1], args[2])
    elif args[0] == 'run-blocks':
        mm = _option(args, '--max-memory', float)
        npz = '--npz' in args
        if npz:
            args.remove('--npz')
        direct = '--direct' in args
        if direct:
            args.remove('--direct')
        dest = _option(args, '--out', str)
        dhf = '--fold-core' not in args
        if not dhf:
            args.remove('--fold-core')
        sizes = [int(x) for x in args[2:]] or ([0] if o2_scf.TAG else [26, 30, 32, 116])
        run_blocks(args[1], sizes, mm, npz, direct, dest, dhf)
    else:
        run(args[1], [int(x) for x in args[2:]] or ([0] if o2_scf.TAG else [26, 30, 32, 116]))
