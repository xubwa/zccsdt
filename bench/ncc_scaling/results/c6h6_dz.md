c6h6_dz_31782853: seconds per CCSDT iteration = mean of iterations 2-3

All runs (CPU / wall: of the iterations of the window and of (Q), by the timers of the programs; "process": xncc or python as a whole, /usr/bin/time):

| run | threads | memory policy | iterations | s / iteration | cores busy | (Q) s | cores busy | process wall s | process cores busy | peak RSS GB | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ncc_off_t1 | 1 | membind=0 | 3 | 1232.132 | 1.00 |  |  | 2560 | 1.00 | 13.94 | threads seen 1; memory limit 55.879 GiB, optimal 13.087 GiB |
| ncc_off_t2 | 2 | membind=0 | 3 | 631.337 | 1.98 |  |  | 1341 | 1.94 | 13.89 | threads seen 2; memory limit 55.879 GiB, optimal 13.087 GiB |
| ncc_off_t4 | 4 | membind=0 | 3 | 346.332 | 3.91 |  |  | 762 | 3.69 | 13.88 | threads seen 4; memory limit 55.879 GiB, optimal 13.087 GiB |
| ncc_off_t7 | 7 | membind=0 | 3 | 229.541 | 6.91 |  |  | 525 | 6.25 | 13.96 | threads seen 7; memory limit 55.879 GiB, optimal 13.087 GiB |
| ncc_off_t8 | 8 | membind=0 | 3 | 243.836 | 7.50 |  |  | 553 | 6.82 | 14.07 | threads seen 8; memory limit 55.879 GiB, optimal 13.087 GiB |
| ncc_off_t16 | 16 | membind=0 | 3 | 206.291 | 12.42 |  |  | 477 | 11.06 | 14.40 | threads seen 16; memory limit 55.879 GiB, optimal 13.087 GiB |
| ncc_off_t23 | 23 | membind=0 | 3 | 160.981 | 22.41 |  |  | 386 | 19.16 | 14.13 | threads seen 23; memory limit 55.879 GiB, optimal 13.087 GiB |
| ncc_off_t24 | 24 | membind=0 | 3 | 179.976 | 21.26 |  |  | 424 | 18.48 | 14.15 | threads seen 24; memory limit 55.879 GiB, optimal 13.087 GiB |
| ncc_off_t47 | 47 | first touch | 3 | 125.296 | 44.54 |  |  | 314 | 36.40 | 14.67 | threads seen 47; memory limit 55.879 GiB, optimal 13.087 GiB |
| ncc_off_t48 | 48 | first touch | 3 | 135.549 | 36.40 |  |  | 335 | 30.31 | 14.84 | threads seen 48; memory limit 55.879 GiB, optimal 13.087 GiB |
| ncc_off_active_t24 | 24 | membind=0 | 3 | 180.584 | 23.80 |  |  | 425 | 20.65 | 14.14 | OMP_WAIT_POLICY=ACTIVE; threads seen 24; memory limit 55.879 GiB, optimal 13.087 GiB |
| ncc_off_again_t24 | 24 | membind=0 | 3 | 180.625 | 21.26 |  |  | 425 | 18.49 | 14.14 | threads seen 24; memory limit 55.879 GiB, optimal 13.087 GiB |
| ncc_off_inter_t48 | 48 | interleave=all | 3 | 132.509 | 36.55 |  |  | 329 | 30.32 | 14.83 | threads seen 48; memory limit 55.879 GiB, optimal 13.087 GiB |
| ncc_off_mkl1_t8 | 8 | membind=0 | 3 | 247.328 | 7.44 |  |  | 563 | 6.75 | 13.89 | MKL threads 1; threads seen 8; memory limit 55.879 GiB, optimal 13.087 GiB |
| ncc_off_mkl1_t24 | 24 | membind=0 | 3 | 198.261 | 19.77 |  |  | 464 | 17.36 | 14.07 | MKL threads 1; threads seen 24; memory limit 55.879 GiB, optimal 13.087 GiB |
| ncc_off_passive_t8 | 8 | membind=0 | 3 | 244.205 | 7.11 |  |  | 554 | 6.46 | 14.08 | OMP_WAIT_POLICY=PASSIVE; threads seen 8; memory limit 55.879 GiB, optimal 13.087 GiB |
| ncc_off_passive_t24 | 24 | membind=0 | 3 | 181.193 | 19.66 |  |  | 426 | 17.10 | 14.14 | OMP_WAIT_POLICY=PASSIVE; threads seen 24; memory limit 55.879 GiB, optimal 13.087 GiB |
| ncc_on_t1 | 1 | membind=0 | 3 | 48.796 | 1.00 |  |  | 106 | 1.00 | 2.03 | threads seen 1; memory limit 55.879 GiB, optimal 1.201 GiB |
| ncc_on_t2 | 2 | membind=0 | 3 | 27.750 | 1.99 |  |  | 62 | 1.92 | 1.82 | threads seen 2; memory limit 55.879 GiB, optimal 1.201 GiB |
| ncc_on_t4 | 4 | membind=0 | 3 | 18.617 | 3.99 |  |  | 43 | 3.66 | 1.87 | threads seen 4; memory limit 55.879 GiB, optimal 1.201 GiB |
| ncc_on_t7 | 7 | membind=0 | 3 | 15.316 | 6.98 |  |  | 36 | 6.21 | 1.93 | threads seen 7; memory limit 55.879 GiB, optimal 1.201 GiB |
| ncc_on_t8 | 8 | membind=0 | 3 | 15.081 | 7.98 |  |  | 36 | 7.07 | 1.96 | threads seen 8; memory limit 55.879 GiB, optimal 1.201 GiB |
| ncc_on_t16 | 16 | membind=0 | 3 | 15.392 | 15.96 |  |  | 36 | 14.04 | 2.02 | threads seen 16; memory limit 55.879 GiB, optimal 1.201 GiB |
| ncc_on_t23 | 23 | membind=0 | 3 | 15.607 | 22.93 |  |  | 37 | 20.16 | 2.03 | threads seen 23; memory limit 55.879 GiB, optimal 1.201 GiB |
| ncc_on_t24 | 24 | membind=0 | 3 | 15.442 | 23.93 |  |  | 37 | 21.00 | 2.05 | threads seen 24; memory limit 55.879 GiB, optimal 1.201 GiB |
| ncc_on_t47 | 47 | first touch | 3 | 23.134 | 46.74 |  |  | 52 | 42.62 | 2.15 | threads seen 47; memory limit 55.879 GiB, optimal 1.201 GiB |
| ncc_on_t48 | 48 | first touch | 3 | 22.873 | 47.77 |  |  | 52 | 43.50 | 2.16 | threads seen 48; memory limit 55.879 GiB, optimal 1.201 GiB |
| ncc_on_active_t24 | 24 | membind=0 | 3 | 15.380 | 23.93 |  |  | 36 | 20.99 | 2.05 | OMP_WAIT_POLICY=ACTIVE; threads seen 24; memory limit 55.879 GiB, optimal 1.201 GiB |
| ncc_on_again_t24 | 24 | membind=0 | 3 | 15.443 | 23.93 |  |  | 37 | 20.99 | 2.05 | threads seen 24; memory limit 55.879 GiB, optimal 1.201 GiB |
| ncc_on_inter_t48 | 48 | interleave=all | 3 | 22.825 | 47.72 |  |  | 52 | 43.35 | 2.15 | threads seen 48; memory limit 55.879 GiB, optimal 1.201 GiB |
| ncc_on_mkl1_t8 | 8 | membind=0 | 3 | 17.571 | 7.98 |  |  | 41 | 7.18 | 1.89 | MKL threads 1; threads seen 8; memory limit 55.879 GiB, optimal 1.201 GiB |
| ncc_on_mkl1_t24 | 24 | membind=0 | 3 | 24.501 | 23.93 |  |  | 55 | 21.97 | 2.04 | MKL threads 1; threads seen 24; memory limit 55.879 GiB, optimal 1.201 GiB |
| ncc_on_passive_t8 | 8 | membind=0 | 3 | 23.132 | 3.43 |  |  | 52 | 3.23 | 1.96 | OMP_WAIT_POLICY=PASSIVE; threads seen 8; memory limit 55.879 GiB, optimal 1.201 GiB |
| ncc_on_passive_t24 | 24 | membind=0 | 3 | 32.980 | 5.14 |  |  | 72 | 4.89 | 2.05 | OMP_WAIT_POLICY=PASSIVE; threads seen 24; memory limit 55.879 GiB, optimal 1.201 GiB |
| ncc_onq_t8 | 8 | membind=0 | 6 | 15.029 | 7.98 | 1607.50 | 7.54 | 1706 | 7.48 | 56.53 | converged in 6 (CC_CONV=3); threads seen 8; memory limit 55.879 GiB, optimal 606.039 GiB |
| ncc_onq_t16 | 16 | membind=0 | 6 | 15.314 | 15.95 | 1466.58 | 15.02 | 1566 | 14.88 | 57.19 | converged in 6 (CC_CONV=3); threads seen 16; memory limit 55.879 GiB, optimal 606.039 GiB |
| ncc_onq_t24 | 24 | membind=0 | 29 | 15.472 | 23.93 | 1518.11 | 22.67 | 1974 | 22.72 | 57.87 | converged in 29; threads seen 24; memory limit 55.879 GiB, optimal 606.039 GiB |
| ncc_onq_t48 | 48 | first touch | 6 | 24.712 | 47.69 | 1625.94 | 46.68 | 1773 | 46.19 | 59.82 | converged in 6 (CC_CONV=3); threads seen 48; memory limit 55.879 GiB, optimal 606.039 GiB |
| ncc_onq_mem150_t48 | 48 | first touch | 6 | 25.980 | 47.59 | 1980.44 | 44.85 | 2131 | 44.57 | 140.14 | converged in 6 (CC_CONV=3); threads seen 48; memory limit 139.698 GiB, optimal 606.039 GiB |
| zc_rhf_t1 | 1 | membind=0 | 3 | 536.102 | 1.00 |  |  | 1669 | 0.99 | 29.45 | DIIS pairs: file 0, memory 8 |
| zc_rhf_t2 | 2 | membind=0 | 3 | 274.746 | 1.98 |  |  | 862 | 1.96 | 29.51 | DIIS pairs: file 0, memory 8 |
| zc_rhf_t4 | 4 | membind=0 | 3 | 140.771 | 3.93 |  |  | 447 | 3.85 | 29.48 | DIIS pairs: file 0, memory 8 |
| zc_rhf_t8 | 8 | membind=0 | 3 | 72.678 | 7.79 |  |  | 237 | 7.45 | 29.54 | DIIS pairs: file 0, memory 8 |
| zc_rhf_t16 | 16 | membind=0 | 3 | 41.016 | 15.21 |  |  | 139 | 13.88 | 29.56 | DIIS pairs: file 0, memory 8 |
| zc_rhf_t24 | 24 | membind=0 | 3 | 34.373 | 22.56 |  |  | 142 | 16.52 | 29.70 | DIIS pairs: file 0, memory 8 |
| zc_rhf_t48 | 48 | first touch | 3 | 28.331 | 42.31 |  |  | 96 | 37.34 | 30.07 | DIIS pairs: file 0, memory 8 |
| zc_rhf_active_t24 | 24 | membind=0 | 3 | 34.323 | 23.89 |  |  | 114 | 23.22 | 29.66 | OMP_WAIT_POLICY=ACTIVE; DIIS pairs: file 0, memory 8 |
| zc_rhf_again_t24 | 24 | membind=0 | 3 | 34.350 | 22.55 |  |  | 112 | 20.92 | 29.70 | DIIS pairs: file 0, memory 8 |
| zc_rhf_inter_t48 | 48 | interleave=all | 3 | 25.384 | 42.96 |  |  | 90 | 37.49 | 30.12 | DIIS pairs: file 0, memory 8 |
| zc_rhf_passive_t8 | 8 | membind=0 | 3 | 72.797 | 7.79 |  |  | 234 | 7.54 | 29.54 | OMP_WAIT_POLICY=PASSIVE; DIIS pairs: file 0, memory 8 |
| zc_rhf_passive_t24 | 24 | membind=0 | 3 | 34.356 | 22.56 |  |  | 146 | 16.08 | 29.68 | OMP_WAIT_POLICY=PASSIVE; DIIS pairs: file 0, memory 8 |
| zc_rhfq_t8 | 8 | membind=0 | 2 | 72.675 | 7.79 | 2947.58 | 7.76 | 3111 | 7.74 | 22.08 | window of 1; DIIS pairs: file 0, memory 8 |
| zc_rhfq_t16 | 16 | membind=0 | 2 | 40.879 | 15.22 | 1835.23 | 15.21 | 1933 | 15.12 | 22.14 | window of 1; DIIS pairs: file 0, memory 8 |
| zc_rhfq_t24 | 24 | preferred=0 | 25 | 34.358 | 22.53 | 1603.67 | 22.56 | 2494 | 22.43 | 80.98 | converged in 25; DIIS pairs: file 0, memory 8 |
| zc_rhfq_t48 | 48 | first touch | 2 | 27.602 | 41.62 | 1284.06 | 41.43 | 1349 | 41.15 | 22.59 | window of 1; DIIS pairs: file 0, memory 8 |

ncc_off, seconds per CCSDT iteration (ncc):

| threads | 1 | 2 | 4 | 7 | 8 | 16 | 23 | 24 | 47 | 48 |
|---|---|---|---|---|---|---|---|---|---|---|
| seconds per CCSDT iteration | 1232.13 | 631.34 | 346.33 | 229.54 | 243.84 | 206.29 | 160.98 | 179.98 | 125.30 | 135.55 |
| speedup from 1 thread | 1.00 | 1.95 | 3.56 | 5.37 | 5.05 | 5.97 | 7.65 | 6.85 | 9.83 | 9.09 |
| speedup from 8 threads | 0.20 | 0.39 | 0.70 | 1.06 | 1.00 | 1.18 | 1.51 | 1.35 | 1.95 | 1.80 |
| efficiency (speedup from 1 / threads) | 1.00 | 0.98 | 0.89 | 0.77 | 0.63 | 0.37 | 0.33 | 0.29 | 0.21 | 0.19 |

ncc_off_mkl1, seconds per CCSDT iteration (ncc):

| threads | 8 | 24 |
|---|---|---|
| seconds per CCSDT iteration | 247.33 | 198.26 |
| speedup from 1 thread |  |  |
| speedup from 8 threads | 1.00 | 1.25 |
| efficiency (speedup from 1 / threads) |  |  |

ncc_off_passive, seconds per CCSDT iteration (ncc):

| threads | 8 | 24 |
|---|---|---|
| seconds per CCSDT iteration | 244.20 | 181.19 |
| speedup from 1 thread |  |  |
| speedup from 8 threads | 1.00 | 1.35 |
| efficiency (speedup from 1 / threads) |  |  |

ncc_on, seconds per CCSDT iteration (ncc):

| threads | 1 | 2 | 4 | 7 | 8 | 16 | 23 | 24 | 47 | 48 |
|---|---|---|---|---|---|---|---|---|---|---|
| seconds per CCSDT iteration | 48.80 | 27.75 | 18.62 | 15.32 | 15.08 | 15.39 | 15.61 | 15.44 | 23.13 | 22.87 |
| speedup from 1 thread | 1.00 | 1.76 | 2.62 | 3.19 | 3.24 | 3.17 | 3.13 | 3.16 | 2.11 | 2.13 |
| speedup from 8 threads | 0.31 | 0.54 | 0.81 | 0.98 | 1.00 | 0.98 | 0.97 | 0.98 | 0.65 | 0.66 |
| efficiency (speedup from 1 / threads) | 1.00 | 0.88 | 0.66 | 0.46 | 0.40 | 0.20 | 0.14 | 0.13 | 0.04 | 0.04 |

ncc_on_mkl1, seconds per CCSDT iteration (ncc):

| threads | 8 | 24 |
|---|---|---|
| seconds per CCSDT iteration | 17.57 | 24.50 |
| speedup from 1 thread |  |  |
| speedup from 8 threads | 1.00 | 0.72 |
| efficiency (speedup from 1 / threads) |  |  |

ncc_on_passive, seconds per CCSDT iteration (ncc):

| threads | 8 | 24 |
|---|---|---|
| seconds per CCSDT iteration | 23.13 | 32.98 |
| speedup from 1 thread |  |  |
| speedup from 8 threads | 1.00 | 0.70 |
| efficiency (speedup from 1 / threads) |  |  |

ncc_onq, seconds per CCSDT iteration (ncc):

| threads | 8 | 16 | 24 | 48 |
|---|---|---|---|---|
| seconds per CCSDT iteration | 15.03 | 15.31 | 15.47 | 24.71 |
| speedup from 1 thread |  |  |  |  |
| speedup from 8 threads | 1.00 | 0.98 | 0.97 | 0.61 |
| efficiency (speedup from 1 / threads) |  |  |  |  |

zc_rhf, seconds per CCSDT iteration (zccsdt RCCSDT):

| threads | 1 | 2 | 4 | 8 | 16 | 24 | 48 |
|---|---|---|---|---|---|---|---|
| seconds per CCSDT iteration | 536.10 | 274.75 | 140.77 | 72.68 | 41.02 | 34.37 | 28.33 |
| speedup from 1 thread | 1.00 | 1.95 | 3.81 | 7.38 | 13.07 | 15.60 | 18.92 |
| speedup from 8 threads | 0.14 | 0.26 | 0.52 | 1.00 | 1.77 | 2.11 | 2.57 |
| efficiency (speedup from 1 / threads) | 1.00 | 0.98 | 0.95 | 0.92 | 0.82 | 0.65 | 0.39 |

zc_rhf_passive, seconds per CCSDT iteration (zccsdt RCCSDT):

| threads | 8 | 24 |
|---|---|---|
| seconds per CCSDT iteration | 72.80 | 34.36 |
| speedup from 1 thread |  |  |
| speedup from 8 threads | 1.00 | 2.12 |
| efficiency (speedup from 1 / threads) |  |  |

zc_rhfq, seconds per CCSDT iteration (zccsdt RCCSDT):

| threads | 8 | 16 | 24 | 48 |
|---|---|---|---|---|
| seconds per CCSDT iteration | 72.67 | 40.88 | 34.36 | 27.60 |
| speedup from 1 thread |  |  |  |  |
| speedup from 8 threads | 1.00 | 1.78 | 2.12 | 2.63 |
| efficiency (speedup from 1 / threads) |  |  |  |  |

ncc_onq, seconds of (Q) (ncc):

| threads | 8 | 16 | 24 | 48 |
|---|---|---|---|---|
| seconds of (Q) | 1607.50 | 1466.58 | 1518.11 | 1625.94 |
| speedup from 1 thread |  |  |  |  |
| speedup from 8 threads | 1.00 | 1.10 | 1.06 | 0.99 |
| efficiency (speedup from 1 / threads) |  |  |  |  |

zc_rhfq, seconds of (Q) (zccsdt RCCSDT):

| threads | 8 | 16 | 24 | 48 |
|---|---|---|---|---|
| seconds of (Q) | 2947.58 | 1835.23 | 1603.67 | 1284.06 |
| speedup from 1 thread |  |  |  |  |
| speedup from 8 threads | 1.00 | 1.61 | 1.84 | 2.30 |
| efficiency (speedup from 1 / threads) |  |  |  |  |

ncc against zccsdt RCCSDT, seconds per CCSDT iteration:

| threads | ncc, symmetry | ncc, no symmetry | zccsdt | ncc (symmetry) / zccsdt | ncc (no symmetry) / zccsdt |
|---|---|---|---|---|---|
| 1 | 48.796 | 1232.132 | 536.102 | 0.09 | 2.30 |
| 2 | 27.750 | 631.337 | 274.746 | 0.10 | 2.30 |
| 4 | 18.617 | 346.332 | 140.771 | 0.13 | 2.46 |
| 8 | 15.081 | 243.836 | 72.678 | 0.21 | 3.36 |
| 16 | 15.392 | 206.291 | 41.016 | 0.38 | 5.03 |
| 24 | 15.442 | 179.976 | 34.373 | 0.45 | 5.24 |
| 48 | 22.873 | 135.549 | 28.331 | 0.81 | 4.78 |

ncc against zccsdt RCCSDT, seconds of (Q):

| threads | ncc, symmetry | ncc, no symmetry | zccsdt | ncc (symmetry) / zccsdt | ncc (no symmetry) / zccsdt |
|---|---|---|---|---|---|
| 8 | 1607.499 |  | 2947.580 | 0.55 |  |
| 16 | 1466.578 |  | 1835.230 | 0.80 |  |
| 24 | 1518.111 |  | 1603.670 | 0.95 |  |
| 48 | 1625.940 |  | 1284.060 | 1.27 |  |

Energies of the converged runs:

| run | E(SCF) | E(CCSDT) | E(Q) | E(CCSDT(Q)) |
|---|---|---|---|---|
| ncc_onq_t24 | -230.721903098552 | -231.581421583844 | -0.002741985807 | -231.584163569651 |
| zc_rhfq_t24 | -230.721903098547 | -231.581421583844 | -0.002741985807 | -231.584163569652 |

ncc_onq_t24 - zc_rhfq_t24: CCSDT -1.4e-13  CCSDT(Q) 1.2e-12

ncc with and without symmetry, largest difference of the correlation energies of iterations 1-3 (ncc_on_t48, ncc_off_t48): 6.6e-12

