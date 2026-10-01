'''Read basis sets from a CFOUR GENBAS file into PySCF's internal format, so
that zccsdt runs use exactly the basis of CFOUR / MRCC reference runs.'''


def read_genbas(path, key):
    '''PySCF basis list for the GENBAS entry ``key`` (e.g. 'N:6-31G').'''
    lines = open(path).read().split('\n')
    for i, line in enumerate(lines):
        if line.strip() == key:
            break
    else:
        raise KeyError(key)
    nums = []
    j = i + 2                                   # skip the comment line
    while j < len(lines):
        s = lines[j].strip()
        if s and ':' in s and len(s.split()) == 1 and s[0].isalpha():
            break                               # next entry
        nums += s.split()
        j += 1
    it = iter(nums)
    nl = int(next(it))
    ls = [int(next(it)) for _ in range(nl)]
    ncon = [int(next(it)) for _ in range(nl)]
    nprim = [int(next(it)) for _ in range(nl)]
    basis = []
    for l, nc, npr in zip(ls, ncon, nprim):
        exps = [float(next(it)) for _ in range(npr)]
        coef = [[float(next(it)) for _ in range(nc)] for _ in range(npr)]
        for c in range(nc):
            prims = [[e, row[c]] for e, row in zip(exps, coef) if row[c] != 0.0]
            basis.append([l] + prims)
    return basis
