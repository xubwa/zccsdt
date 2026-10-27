'''The result of a complete-5p3h EOM-DEA run of one case done as one job per
target (native5_mpi.sbatch): the roots of 2 Omega = 0 and 2 from the files
XZDEA of the two run directories, the sigma vectors and iterations from the
outputs, the elapsed time and nodes of each job (sacct), as JSON.

usage: python bench/dea/o2_cv5_json.py <out.json> <rundir_t0> <tag0> <job0> <rundir_t2> <tag2> <job2> [key=value ...]'''
import json
import os
import re
import subprocess
import sys

HARTREE_CM = 219474.6313632


def xzdea(path):
    out = {}
    for line in open(path):
        if line.startswith('# E(SCF)'):
            w = line.split()
            out['e_scf'], out['e_ccsdt'] = float(w[2]), float(w[4])
        elif line.strip() and not line.startswith('#'):
            w = line.split()
            out['state'] = {'two_omega': int(w[0]), 'root': int(w[1]), 'omega': float(w[2]), 'energy': float(w[3]),
                            'residual_max': float(w[4]), 'residual_norm': float(w[5]),
                            'weights_2p_3p1h_4p2h': [float(x) for x in w[6:9]], 'converged': w[9] == 'T'}
    return out


def job(jobid):
    s = subprocess.run(['sacct', '-j', str(jobid), '-X', '-n', '-P', '-o', 'ElapsedRaw,NNodes,NodeList,State'],
                       capture_output=True, text=True).stdout.split('\n')[0].split('|')
    sec, nn = int(s[0]), int(s[1])
    return {'job': int(jobid), 'elapsed_s': sec, 'nodes': nn, 'nodelist': s[2], 'slurm_state': s[3],
            'node_hours': round(sec * nn / 3600.0, 2)}


def run(rundir, tag, jobid):
    d = xzdea(os.path.join(rundir, 'XZDEA.' + tag))
    txt = open(os.path.join(rundir, 'xzccsdt.%s.out' % tag)).read()
    m = re.search(r'sigma vectors:\s+(\d+)\s+iterations:\s+(\d+)\s+time:\s+([\d.]+) s\s+\(\s*([\d.]+) s per sigma', txt)
    d['sigma_vectors'], d['iterations'] = int(m.group(1)), int(m.group(2))
    d['eom_time_s'], d['s_per_sigma'] = float(m.group(3)), float(m.group(4))
    m = re.search(r'5p3h:\s+(\d+) elements in the blocks,\s+(\d+) stored', txt)
    d['r5_elements_blocks'], d['r5_elements_stored'] = int(m.group(1)), int(m.group(2))
    d.update(job(jobid))
    return d


if __name__ == '__main__':
    a = sys.argv
    r0, r2 = run(a[2], a[3], a[4]), run(a[5], a[6], a[7])
    res = {'e_scf': r0['e_scf'], 'e_ccsdt': r0['e_ccsdt'], 'E0': r0['state']['energy'], 'E2': r2['state']['energy']}
    res['split_Eh'] = res['E2'] - res['E0']
    res['split_cm'] = res['split_Eh'] * HARTREE_CM
    res['hartree_to_cm'] = HARTREE_CM
    res['all_converged'] = r0['state']['converged'] and r2['state']['converged']
    res['node_hours'] = round(r0['node_hours'] + r2['node_hours'], 2)
    res['target_0'], res['target_2'] = r0, r2
    for kv in a[8:]:
        k, v = kv.split('=', 1)
        res[k] = v
    with open(a[1], 'w') as f:
        json.dump(res, f, indent=1)
    print('%s: E0 %.14f  E2 %.14f  split %.10e Eh = %.5f cm-1  converged %s  sigma vectors %d + %d  node-hours %.1f'
          % (a[1], res['E0'], res['E2'], res['split_Eh'], res['split_cm'], res['all_converged'],
             r0['sigma_vectors'], r2['sigma_vectors'], res['node_hours']))
