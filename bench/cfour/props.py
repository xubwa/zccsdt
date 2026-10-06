'''The one-electron properties of CFOUR's two-component programs with the
densities of zccsdt: CFOUR's own operators (the files xsox2catm writes, with
its picture-change transformation and the response of the X2C decoupling),
read by socutils.prop.cfour, traced with the SCF, the unrelaxed and the
relaxed coupled-cluster density of zccsdt, next to what xsox2cprop prints.
The numbers then differ only by the densities - also for the properties
that live at the heavy nucleus (electric field gradient, contact density,
effective electric field, ...), which the dipole moment says little about.

usage: python bench/cfour/props.py pack MOLECULE BASIS OPERATORS.npz GEOMETRY
         in the directory of a CFOUR run, after xsox2catm: the operators in
         the spinor basis of PySCF, the density of CFOUR's SCF (SCFDM, for
         ZC_DM0) and CFOUR's geometry (its frame, bohr; a file for
         ZC_GEOMETRY of bench/spinors.py)
       python bench/cfour/props.py guess MOLECULE BASIS GEOMETRY
         in the directory of a CFOUR run, after xjoda: the SCF of
         bench/spinors.py in CFOUR's frame (with its environment: ZC_HAM,
         ZC_NUCMOD, ZC_SCF, ZC_AXIAL), its density as the files drguess and
         diguess that xsox2chf starts from - a doublet with the spin on the
         axis of the molecule, which CFOUR's own guess does not give
       python bench/cfour/props.py compare OPERATORS.npz DENSITIES.npz XSOX2CPROP.OUT
         DENSITIES.npz: scf, unrelaxed, relaxed in the spinor AO basis
         (bench/profile_t_relaxed.py with ZC_DENSITIES); the SCF and the
         relaxed coupled-cluster density are compared (what CFOUR prints as
         unrelaxed is not the density of t and lambda at fixed spinors)
socutils with prop/cfour.py has to be on the path (socutils-git).
'''
import os
import re
import sys

import numpy as np
from pyscf import gto

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import spinors                                                          # noqa: E402

# file -> (heading of xsox2cprop, label with {c} component and {a} atom, the
# sign of the electron charge that xsox2cprop applies)
TABLE = {
    'SOX2CDP': ('Electronic Electric Dipole Moments', 'mu {c}', -1),
    'UTM10': ('Electronic G-factor (a.u.) using UT', 'g fac {c}', 1),
    'UTML0': ('Electronic G-factor (a.u.) using UT', 'UTML0{C}', 1),
    'SOM10': ('Electronic G-factor (a.u.)', 'g fac {c}', 1),
    'SOEFG': ('Electronic Electric Field Gradients', 'EFG {c} for atm# {a}', -1),
    'SOX2CEFD': ('Electronic Electric Effective Field', 'E_eff', 1),
    'SOMQM': ('Nuclear Magnetic Quadrupole Moment', 'nMQM', 1),
    'SOM01': ('with Kinetic Balance (KB) method', 'MHC-KB {c} at atm# {a}', -1),
    'UTM01': ('with Unitary Transformation (UT) method', 'MHC-UT {c} at atm# {a}', -1),
    'SODEN2C': ('Electon Contact Density at nucleus (a.u.)', 'Density at atm# {a}', 1),
    'SODEN2CG': ('Electon Contact Density at nucleus (a.u.), derivative formulation',
                 'Density at atm# {a}', 1),
    'SOMOSS2C': ('Electon Effective Density at nucleus', 'Density at atm# {a}', -1),
    'SOSFMO2C': ('Nuclear Schiff Moment Sensitivity Parameter',
                 'NSM parameter {c} at atm# {a}', 1),
    'SOTPNEC': ('Tensor-pseudotensor nucleo-electron current', 'TPNEC parameter z at atm# {a}', 1),
    'SOPSNEC': ('Pseudoscalar-scalar nucleon-electron current Parameter',
                'PSNEC parameter {c} at atm# {a}', 1),
    'SOSPNEC': ('Scalar-pseudoscalar nucleo-electron current', 'SPNEC parameter z at atm# {a}', 1),
    'SOPEDM': ('proton electric dipole moment Parameter', 'pEDM parameter z at atm# 1', 1),
}
SECTIONS = (('scf', 'Hartree-Fock Properties'), ('relaxed', 'Coupled-cluster Properties'),
            ('unrelaxed', 'Unrelaxed Coupled-cl'))
SCFDM = 'SCFDM'         # the SCF density of CFOUR among the operators


def read_mol(path='MOL'):
    '''[(symbol, (x, y, z))] of CFOUR's MOL file: its frame, bohr, its order.'''
    atoms = []
    with open(path) as f:
        for line in f:
            m = re.match(r'^([A-Za-z]{1,2})\s*#\s*\d+\s+(\S+)\s+(\S+)\s+(\S+)\s*$', line)
            if m:
                atoms.append((m.group(1).capitalize(), tuple(float(x) for x in m.groups()[1:])))
    if not atoms:
        raise ValueError('no atoms found in %s' % path)
    return atoms


def items(store, name):
    '''(key, atom, component) of the matrices of a file; the key is
    'FILE', 'FILE|atom' or 'FILE|atom|component' (atoms from 1).'''
    from socutils.prop import cfour
    spec = cfour.meta(name)
    if spec['layout'] == 'single':
        return [(name, None, None)]
    out = []
    for a in range(store.natm):
        if spec['comps']:
            out += [('%s|%d|%s' % (name, a + 1, c), a, c) for c in spec['comps']]
        else:
            out.append(('%s|%d' % (name, a + 1), a, None))
    return out


def pack(mol_name, basis, out, geometry):
    from socutils.prop import cfour
    atoms = read_mol()
    with open(geometry, 'w') as f:
        f.write('\n'.join('%s %.15f %.15f %.15f' % (s, *xyz) for s, xyz in atoms) + '\n')
    mol = gto.M(atom=atoms, unit='Bohr', basis=spinors.basis_of(mol_name, basis), verbose=0,
                spin=spinors.SPIN.get(mol_name, 0))
    store = cfour.CfourPropStore('.', mol, atom_map=list(range(mol.natm)))
    ops = {}
    for name in store.available():
        if not any(name.startswith(k) for k in TABLE):
            continue
        for key, atom, comp in items(store, name):
            ops[key] = store.op(name, atom=atom, comp=comp)
    print('%d operators of CFOUR (%s) for %s in its frame: %s'
          % (len(ops), ' '.join(sorted({k.split('|')[0] for k in ops})), mol_name,
             '; '.join('%s %.6f %.6f %.6f' % (s, *xyz) for s, xyz in atoms)))
    if os.path.isfile('scf2ccmo'):
        # the spinors of CFOUR's SCF (one record, column-major), the lowest
        # occupied: its density, to start the SCF of zccsdt from - a doublet
        # has a Kramers partner, and with weak spin-orbit coupling any
        # direction of its spin, and what is odd under time reversal depends
        # on which of these states the SCF holds
        n2c = mol.nao_2c()
        with open('scf2ccmo', 'rb') as f:
            f.seek(4)
            buf = np.frombuffer(f.read(n2c * n2c * 16), dtype=np.float64)
        c = store.rotation.conj().T @ (buf[0::2] + 1j * buf[1::2]).reshape(n2c, n2c).T
        s1 = mol.intor('int1e_ovlp_spinor')
        co = c[:, :mol.nelectron]
        print("CFOUR's SCF spinors: largest deviation from orthonormality %.1e"
              % abs(c.conj().T @ s1 @ c - np.eye(n2c)).max())
        ops[SCFDM] = co @ co.conj().T
        if os.path.isfile('drguess') and os.path.isfile('diguess'):
            d = store.rotation.conj().T @ cfour_density() @ store.rotation
            print("drguess/diguess of CFOUR against the density of its spinors: %.1e; against "
                  'its transpose: %.1e' % (abs(d - ops[SCFDM]).max(), abs(d - ops[SCFDM].T).max()))
    np.savez(out, **ops)


def record(path, a):
    '''One record of a Fortran sequential file with the doubles ``a``
    (CFOUR's wfile).'''
    a = np.ascontiguousarray(a, dtype=np.float64)
    mark = np.array([a.nbytes], dtype=np.int32).tobytes()
    with open(path, 'wb') as f:
        f.write(mark + a.tobytes() + mark)


def cfour_density(path_r='drguess', path_i='diguess'):
    '''The density of the files drguess and diguess, as a complex matrix in
    CFOUR's basis.'''
    def read(path):
        with open(path, 'rb') as f:
            f.seek(4)
            return np.frombuffer(f.read()[:-4], dtype=np.float64)
    re, im = read(path_r), read(path_i)
    n = int(round(np.sqrt(re.size)))
    return (re + 1j * im).reshape(n, n).T


def guess(mol_name, basis, geometry):
    from socutils.tools.label import Labels
    atoms = read_mol()
    with open(geometry, 'w') as f:
        f.write('\n'.join('%s %.15f %.15f %.15f' % (s, *xyz) for s, xyz in atoms) + '\n')
    os.environ['ZC_GEOMETRY'] = geometry
    mol = spinors.molecule(mol_name, basis)
    mf = spinors.scf(mol, 1e-12, 1e-9)
    # the density C_occ C_occ^+ in CFOUR's basis (alpha, then beta, of the
    # scalar AOs), column-major like its own drguess and diguess (those of a
    # run of xsox2chf, rotated back, are the density of its spinors)
    c = Labels.spinor_cfour2pyscf(mol)
    d = c @ mf.make_rdm1() @ c.conj().T
    record('drguess', d.real.T.ravel())
    record('diguess', d.imag.T.ravel())
    print('SCF of zccsdt for the guess of xsox2chf: E(SCF) %.10f, converged %s; drguess and '
          'diguess written' % (mf.e_tot, mf.converged))


def printed(path):
    '''{section: {(heading, label): value}} of an output of xsox2cprop.'''
    out, section, heading = {}, None, ''
    number = r'[-+]?\d+\.\d*(?:[EeDd][-+]?\d+)?'
    with open(path) as f:
        for line in f:
            text = line.strip()
            banner = re.match(r'^\*\s+(.*?)\s+\*$', text)
            if banner:
                section = next((k for k, s in SECTIONS if banner.group(1).startswith(s)), None)
                if section:
                    out[section] = {}
                continue
            if section is None or not text or text.startswith('***'):
                continue
            m = re.match(r'^(.*?\S)\s+(%s)(?:\s+%s)?$' % (number, number), text)
            if m and not re.match(r'^%s' % number, text):
                label = re.sub(r'\s+', ' ', m.group(1))
                out[section][(heading, label)] = float(m.group(2).replace('D', 'E'))
            elif not re.match(r'^%s' % number, text):
                heading = text
    return out


def expected(key, kind='scf'):
    '''(heading, label, sign) of xsox2cprop for the matrix ``key`` in the
    section of the density ``kind``.'''
    name, *rest = key.split('|')
    if kind != 'scf' and name[:5] in ('UTM10', 'SOM10'):
        # the coupled-cluster sections have one g-factor block, the UT one
        return ('Electronic G-factor (a.u.)' if name[:5] == 'UTM10' else 'not printed',
                'g fac %s' % name[-1].lower(), 1)
    atom = rest[0] if rest else ''
    comp = rest[1] if len(rest) > 1 else ''
    for prefix, (heading, label, sign) in TABLE.items():
        if name == prefix or (name[:-1] == prefix and name[-1] in 'XYZ'):
            if name != prefix:
                comp = name[-1].lower()
            return heading, label.format(c=comp, C=comp.upper(), a=atom), sign
    raise KeyError(key)


def compare(ops_file, den_file, prop_out):
    ops, den = np.load(ops_file), np.load(den_file)
    ref = printed(prop_out)
    kinds = [k for k in ('scf', 'relaxed') if k in den.files and k in ref]
    rows = {}
    for key in sorted(k for k in ops.files if k != SCFDM):
        o = ops[key]
        for kind in kinds:
            heading, label, sign = expected(key, kind)
            ours = sign * np.einsum('pq,qp', o, den[kind])
            theirs = ref[kind].get((heading, label))
            if theirs is None:                          # '... (a.u.)' after the heading
                theirs = next((v for (h, l), v in ref[kind].items()
                               if h.startswith(heading) and l == label), None)
            if theirs is None and heading != 'not printed':     # another heading here
                same = [v for (h, l), v in ref[kind].items() if l == label]
                theirs = same[0] if len(same) == 1 else None
            rows.setdefault(key, {})[kind] = (theirs, ours)
    # the size of a tensor: the largest of the components of an operator at
    # an atom (a component that vanishes by symmetry has only noise, and the
    # noise of a property of the core is large in absolute terms)
    scale = {}
    for key, row in rows.items():
        group = '|'.join(key.split('|')[:2]) if key.count('|') == 2 else key
        if group[:-1] in ('SOX2CDP', 'UTM10', 'SOM10', 'UTML0'):
            group = group[:-1]
        size = max([abs(t) for t, _ in row.values() if t is not None] or [0.0])
        scale[group] = max(scale.get(group, 0.0), size)
        row['group'] = group
    print('the electronic part, as xsox2cprop prints it; left out: what vanishes here (below 1e-6 '
          'of the\nlargest component of the operator at the atom, or below 1e-5 in all)')
    print('%-16s' % 'operator' + ''.join('  %20s %20s %8s' % ('CFOUR ' + k, 'zccsdt', 'diff.')
                                         for k in kinds))
    worst = {}
    for key, row in rows.items():
        size = scale[row['group']]
        text, show, note = '', False, ''
        for kind in kinds:
            theirs, ours = row[kind]
            if theirs is None:
                text += '  %20s %20.10f %8s' % ('not printed', ours.real, '')
                continue
            # a doublet: the two programs may hold different Kramers partners,
            # and what is odd under time reversal then has the other sign
            flip = abs(theirs) > 1e-6 and abs(ours.real + theirs) < 1e-3 * abs(theirs)
            diff = -ours.real - theirs if flip else ours.real - theirs
            text += '  %20.10f %20.10f %8.1e' % (theirs, ours.real, diff)
            if size > 1e-5 and max(abs(theirs), abs(ours)) > 1e-6 * size:
                show = True
            if size > 1e-5:
                worst[kind] = max(worst.get(kind, (0.0, '')), (abs(diff) / size, key))
            if flip:
                note = '  other sign'
            if abs(ours.imag) > 1e-8 * max(size, 1.0):
                note += '  imaginary part %.1e' % ours.imag
        if show:
            print('%-16s' % key + text + note)
    for kind, (rel, key) in worst.items():
        print('largest difference with the %s density, relative to the size of the operator: '
              '%.1e (%s)' % (kind, rel, key))


if __name__ == '__main__':
    if sys.argv[1] == 'pack':
        pack(*sys.argv[2:6])
    elif sys.argv[1] == 'guess':
        guess(*sys.argv[2:5])
    else:
        compare(*sys.argv[2:5])
