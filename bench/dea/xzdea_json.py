'''The file XZDEA of a native EOM-DEA run as JSON: E(SCF), E(CCSDT), per
target the root (omega, total energy, residual, weights, converged), and
E0 = E(2 Omega = 0), E2 = E(2 Omega = 2), split = E2 - E0 (Eh and cm-1).

usage: python bench/dea/xzdea_json.py <XZDEA> <out.json> [key=value ...]   (extra fields, strings)'''
import json
import sys

HARTREE_CM = 219474.6313632


def read(path):
    out = {'states': []}
    for line in open(path):
        if line.startswith('# E(SCF)'):
            w = line.split()
            out['e_scf'] = float(w[2])
            out['e_ccsdt'] = float(w[4])
        elif line.strip() and not line.startswith('#'):
            w = line.split()
            out['states'].append({'two_omega': int(w[0]), 'root': int(w[1]), 'omega': float(w[2]),
                                  'energy': float(w[3]), 'residual_max': float(w[4]), 'residual_norm': float(w[5]),
                                  'weights_2p_3p1h_4p2h': [float(x) for x in w[6:9]], 'converged': w[9] == 'T'})
    e = {s['two_omega']: s['energy'] for s in out['states'] if s['root'] == 1}
    out['E0'], out['E2'] = e[0], e[2]
    out['split_Eh'] = e[2] - e[0]
    out['split_cm'] = (e[2] - e[0]) * HARTREE_CM
    out['all_converged'] = all(s['converged'] for s in out['states'])
    return out


if __name__ == '__main__':
    d = read(sys.argv[1])
    for a in sys.argv[3:]:
        k, v = a.split('=', 1)
        d[k] = v
    with open(sys.argv[2], 'w') as f:
        json.dump(d, f, indent=1)
    print('%s: E0 %.14f  E2 %.14f  split %.10e Eh = %.5f cm-1  converged %s'
          % (sys.argv[1], d['E0'], d['E2'], d['split_Eh'], d['split_cm'], d['all_converged']))
