'''The tables of docs/eom.md, section 11.6, from the json files of the runs
(o2_dea.py, o2_strings.py, o2_fold5.py) in a directory.

usage: python tables.py <dir>
File names: {so|xc}_{n26|n30|n32|dz|n116}[...].json as the job scripts write them.'''
import glob
import json
import os
import sys

H2CM = 219474.6313632
SPACES = ('n26', 'n30', 'n32', 'dz', 'n116')
ROWS = [('3p1h', 'CCSD'), ('3p1h', 'CCSDT'), ('3p1h, row at lowest order', 'CC3'), ('4p2h', 'CCSDT'), ('4p2h', 'CCSDTQ'),
        ('4p2h, row at lowest order', 'CC4'), ('5p3h at lowest order, folded', 'CCSDT'),
        ('5p3h at lowest order, folded', 'CCSDTQ'), ('5p3h', 'CCSDT'), ('5p3h', 'CCSDTQ'), ('6p4h', 'CCSDT'),
        ('6p4h', 'CCSDTQ')]
GHOST = {'ccsd': ('3p1h', 'CCSD'), 'ccsd+t3': ('3p1h', 'CCSDT'), 'cc3': ('3p1h, row at lowest order', 'CC3'),
         'ccsdt': ('4p2h', 'CCSDT'), 'ccsdt+t4': ('4p2h', 'CCSDTQ'), 'cc4': ('4p2h, row at lowest order', 'CC4')}
HOLES = {'1': '3p1h', '2': '4p2h', '3': '5p3h', '4': '6p4h'}


def collect(d):
    out = {}                                    # (ham, space, row) -> (E0, E2, source)
    for path in sorted(glob.glob(os.path.join(d, '*.json'))):
        name = os.path.basename(path)[:-5]
        parts = name.split('_')
        ham = 'so' if 'so' in parts else ('xc' if 'xc' in parts else None)
        space = next((s for s in SPACES if s in parts), None)
        if ham is None or space is None:
            continue
        try:
            data = json.load(open(path))
        except ValueError:
            continue
        if name.startswith('str_'):
            for h, r in data.items():
                if '0' in r['E'] and '2' in r['E'] and h in HOLES:
                    out.setdefault((ham, space, (HOLES[h], r['T'].upper())), (r['E']['0'], r['E']['2'], name))
        elif name.startswith('f5'):
            if 'E5' in data and '0' in data['E5'] and '2' in data['E5']:
                out[(ham, space, ('5p3h at lowest order, folded', data['T'].upper()))] = (data['E5']['0'], data['E5']['2'], name)
            elif 'E5' in data:                  # one block per file
                key = (ham, space, ('5p3h at lowest order, folded', data['T'].upper()))
                e = dict(out.get(key + ('part',), {}))
                e.update({k: v for k, v in data['E5'].items()})
                out[key + ('part',)] = e
                if '0' in e and '2' in e:
                    out[key] = (e['0'], e['2'], name)
        else:
            for lev, r in data.items():
                if lev in GHOST and len(r.get('E', {})) == 2:
                    e = list(r['E'].values())
                    out.setdefault((ham, space, GHOST[lev]), (e[0], e[1], name))
    return out


def main():
    res = collect(sys.argv[1])
    for ham, title in (('so', 'spin-orbit only'), ('xc', 'X2Ccorr')):
        print('\n%s: splitting / cm-1\n' % title)
        print('| R | T | ' + ' | '.join(SPACES) + ' |')
        print('|---|---|' + '---|' * len(SPACES))
        for row in ROWS:
            cells = []
            for s in SPACES:
                v = res.get((ham, s, row))
                cells.append('%.4f' % ((v[1] - v[0]) * H2CM) if v else '')
            if any(cells):
                print('| %s | %s | %s |' % (row[0], row[1], ' | '.join(cells)))
        print('\n%s: E(2 Omega = 0), E(2 Omega = 2) / Eh\n' % title)
        for s in SPACES:
            for row in ROWS:
                v = res.get((ham, s, row))
                if v:
                    print('| %s | %s | %s | %.11f | %.11f | %.4e |' % (s, row[0], row[1], v[0], v[1], v[1] - v[0]))


if __name__ == '__main__':
    main()
