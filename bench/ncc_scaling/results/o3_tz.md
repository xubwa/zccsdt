o3_tz_31782852: seconds per CCSDT iteration = mean of iterations 2-5

All runs (CPU / wall: of the iterations of the window and of (Q), by the timers of the programs; "process": xncc or python as a whole, /usr/bin/time):

| run | threads | memory policy | iterations | s / iteration | cores busy | (Q) s | cores busy | process wall s | process cores busy | peak RSS GB | note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ncc_off_t1 | 1 | membind=0 | 5 | 380.603 | 1.00 |  |  | 1550 | 1.00 | 4.65 | threads seen 1; memory limit 55.879 GiB, optimal 4.274 GiB |
| ncc_off_t2 | 2 | membind=0 | 5 | 215.089 | 1.88 |  |  | 881 | 1.87 | 4.59 | threads seen 2; memory limit 55.879 GiB, optimal 4.274 GiB |
| ncc_off_t4 | 4 | membind=0 | 5 | 131.899 | 3.37 |  |  | 545 | 3.33 | 4.64 | threads seen 4; memory limit 55.879 GiB, optimal 4.274 GiB |
| ncc_off_t7 | 7 | membind=0 | 5 | 66.202 | 6.94 |  |  | 280 | 6.68 | 4.70 | threads seen 7; memory limit 55.879 GiB, optimal 4.274 GiB |
| ncc_off_t8 | 8 | membind=0 | 5 | 75.942 | 6.91 |  |  | 319 | 6.69 | 4.73 | threads seen 8; memory limit 55.879 GiB, optimal 4.274 GiB |
| ncc_off_t16 | 16 | membind=0 | 5 | 53.331 | 13.83 |  |  | 228 | 13.15 | 4.88 | threads seen 16; memory limit 55.879 GiB, optimal 4.274 GiB |
| ncc_off_t23 | 23 | membind=0 | 5 | 39.304 | 22.62 |  |  | 172 | 21.06 | 4.90 | threads seen 23; memory limit 55.879 GiB, optimal 4.274 GiB |
| ncc_off_t24 | 24 | membind=0 | 5 | 57.345 | 15.47 |  |  | 244 | 14.81 | 4.92 | threads seen 24; memory limit 55.879 GiB, optimal 4.274 GiB |
| ncc_off_t47 | 47 | first touch | 5 | 30.711 | 44.99 |  |  | 138 | 41.00 | 5.07 | threads seen 47; memory limit 55.879 GiB, optimal 4.274 GiB |
| ncc_off_t48 | 48 | first touch | 5 | 41.146 | 32.62 |  |  | 179 | 30.57 | 5.17 | threads seen 48; memory limit 55.879 GiB, optimal 4.274 GiB |
| ncc_off_active_t24 | 24 | membind=0 | 5 | 57.335 | 23.80 |  |  | 244 | 22.64 | 4.95 | OMP_WAIT_POLICY=ACTIVE; threads seen 24; memory limit 55.879 GiB, optimal 4.274 GiB |
| ncc_off_again_t8 | 8 | membind=0 | 5 | 76.041 | 6.91 |  |  | 319 | 6.69 | 4.73 | threads seen 8; memory limit 55.879 GiB, optimal 4.274 GiB |
| ncc_off_inter_t48 | 48 | interleave=all | 5 | 42.365 | 32.96 |  |  | 184 | 30.94 | 5.15 | threads seen 48; memory limit 55.879 GiB, optimal 4.274 GiB |
| ncc_off_mkl1_t8 | 8 | membind=0 | 5 | 77.420 | 6.90 |  |  | 326 | 6.68 | 4.62 | MKL threads 1; threads seen 8; memory limit 55.879 GiB, optimal 4.274 GiB |
| ncc_off_mkl1_t24 | 24 | membind=0 | 5 | 62.487 | 15.53 |  |  | 266 | 14.92 | 4.75 | MKL threads 1; threads seen 24; memory limit 55.879 GiB, optimal 4.274 GiB |
| ncc_off_passive_t8 | 8 | membind=0 | 5 | 75.954 | 6.10 |  |  | 319 | 5.91 | 4.72 | OMP_WAIT_POLICY=PASSIVE; threads seen 8; memory limit 55.879 GiB, optimal 4.274 GiB |
| ncc_off_passive_t24 | 24 | membind=0 | 5 | 57.408 | 11.11 |  |  | 244 | 10.65 | 4.92 | OMP_WAIT_POLICY=PASSIVE; threads seen 24; memory limit 55.879 GiB, optimal 4.274 GiB |
| ncc_on_t1 | 1 | membind=0 | 5 | 41.094 | 1.00 |  |  | 170 | 0.99 | 1.19 | threads seen 1; memory limit 55.879 GiB, optimal 860.738 MiB |
| ncc_on_t2 | 2 | membind=0 | 5 | 23.007 | 1.99 |  |  | 96 | 1.97 | 1.10 | threads seen 2; memory limit 55.879 GiB, optimal 860.738 MiB |
| ncc_on_t4 | 4 | membind=0 | 5 | 13.399 | 3.99 |  |  | 57 | 3.85 | 1.17 | threads seen 4; memory limit 55.879 GiB, optimal 860.738 MiB |
| ncc_on_t7 | 7 | membind=0 | 5 | 9.438 | 6.98 |  |  | 41 | 6.61 | 1.22 | threads seen 7; memory limit 55.879 GiB, optimal 860.738 MiB |
| ncc_on_t8 | 8 | membind=0 | 5 | 8.433 | 7.97 |  |  | 37 | 7.50 | 1.25 | threads seen 8; memory limit 55.879 GiB, optimal 860.738 MiB |
| ncc_on_t16 | 16 | membind=0 | 5 | 6.143 | 15.94 |  |  | 28 | 14.58 | 1.35 | threads seen 16; memory limit 55.879 GiB, optimal 860.738 MiB |
| ncc_on_t23 | 23 | membind=0 | 5 | 5.695 | 22.91 |  |  | 26 | 20.77 | 1.33 | threads seen 23; memory limit 55.879 GiB, optimal 860.738 MiB |
| ncc_on_t24 | 24 | membind=0 | 5 | 5.768 | 23.90 |  |  | 26 | 21.69 | 1.41 | threads seen 24; memory limit 55.879 GiB, optimal 860.738 MiB |
| ncc_on_t47 | 47 | first touch | 5 | 9.142 | 46.22 |  |  | 40 | 43.30 | 1.57 | threads seen 47; memory limit 55.879 GiB, optimal 860.738 MiB |
| ncc_on_t48 | 48 | first touch | 5 | 9.979 | 47.32 |  |  | 43 | 44.57 | 1.62 | threads seen 48; memory limit 55.879 GiB, optimal 860.738 MiB |
| ncc_on_active_t24 | 24 | membind=0 | 5 | 5.778 | 23.90 |  |  | 26 | 21.70 | 1.40 | OMP_WAIT_POLICY=ACTIVE; threads seen 24; memory limit 55.879 GiB, optimal 860.738 MiB |
| ncc_on_again_t8 | 8 | membind=0 | 5 | 8.524 | 7.97 |  |  | 37 | 7.50 | 1.25 | threads seen 8; memory limit 55.879 GiB, optimal 860.738 MiB |
| ncc_on_inter_t48 | 48 | interleave=all | 5 | 10.174 | 47.15 |  |  | 44 | 44.40 | 1.60 | threads seen 48; memory limit 55.879 GiB, optimal 860.738 MiB |
| ncc_on_mkl1_t8 | 8 | membind=0 | 5 | 10.085 | 7.97 |  |  | 44 | 7.57 | 1.18 | MKL threads 1; threads seen 8; memory limit 55.879 GiB, optimal 860.738 MiB |
| ncc_on_mkl1_t24 | 24 | membind=0 | 5 | 8.592 | 23.91 |  |  | 38 | 22.37 | 1.32 | MKL threads 1; threads seen 24; memory limit 55.879 GiB, optimal 860.738 MiB |
| ncc_on_passive_t8 | 8 | membind=0 | 5 | 9.485 | 5.33 |  |  | 41 | 5.06 | 1.25 | OMP_WAIT_POLICY=PASSIVE; threads seen 8; memory limit 55.879 GiB, optimal 860.738 MiB |
| ncc_on_passive_t24 | 24 | membind=0 | 5 | 7.687 | 10.00 |  |  | 34 | 9.33 | 1.41 | OMP_WAIT_POLICY=PASSIVE; threads seen 24; memory limit 55.879 GiB, optimal 860.738 MiB |
| ncc_onq_t1 | 1 | membind=0 | 10 | 41.132 | 1.00 | 4579.10 | 1.00 | 4962 | 1.00 | 51.51 | converged in 10 (CC_CONV=3); threads seen 1; memory limit 55.879 GiB, optimal 259.353 GiB |
| ncc_onq_t2 | 2 | membind=0 | 10 | 22.913 | 1.99 | 2690.88 | 1.95 | 2907 | 1.95 | 51.35 | converged in 10 (CC_CONV=3); threads seen 2; memory limit 55.879 GiB, optimal 259.353 GiB |
| ncc_onq_t4 | 4 | membind=0 | 10 | 13.337 | 3.99 | 1616.87 | 3.86 | 1746 | 3.85 | 51.58 | converged in 10 (CC_CONV=3); threads seen 4; memory limit 55.879 GiB, optimal 259.353 GiB |
| ncc_onq_t8 | 8 | membind=0 | 10 | 8.476 | 7.97 | 1057.43 | 7.67 | 1142 | 7.65 | 51.95 | converged in 10 (CC_CONV=3); threads seen 8; memory limit 55.879 GiB, optimal 259.353 GiB |
| ncc_onq_t16 | 16 | membind=0 | 10 | 6.121 | 15.94 | 824.52 | 15.30 | 887 | 15.23 | 52.66 | converged in 10 (CC_CONV=3); threads seen 16; memory limit 55.879 GiB, optimal 259.353 GiB |
| ncc_onq_t24 | 24 | membind=0 | 68 | 5.745 | 23.90 | 862.99 | 22.14 | 1257 | 22.57 | 53.36 | converged in 68; threads seen 24; memory limit 55.879 GiB, optimal 259.353 GiB |
| ncc_onq_t48 | 48 | first touch | 10 | 10.051 | 47.20 | 812.50 | 41.30 | 915 | 41.60 | 55.27 | converged in 10 (CC_CONV=3); threads seen 48; memory limit 55.879 GiB, optimal 259.353 GiB |
| ncc_onq_mem150_t48 | 48 | first touch | 10 | 9.802 | 47.37 | 834.04 | 40.41 | 934 | 40.79 | 143.62 | converged in 10 (CC_CONV=3); threads seen 48; memory limit 139.698 GiB, optimal 259.353 GiB |
| ncc_onq_passive_t24 | 24 | membind=0 | 10 | 7.678 | 9.96 | 864.65 | 12.88 | 942 | 12.58 | 53.30 | converged in 10 (CC_CONV=3); OMP_WAIT_POLICY=PASSIVE; threads seen 24; memory limit 55.879 GiB, optimal 259.353 GiB |
| zc_ghf_t8 | 8 | membind=0 | 0 |  |  |  |  | 78 | 2.13 | 4.48 | window of 0; MemoryError: incore_complete: 8 DIIS vectors with T3 take 160.8 GB of memory (20.1 GB per pair of a vector and its error vector); 139.7 GB are left next to the other arrays of the calculation.  Without incore_complete they go to the scratch file as far as the disk holds them |
| zc_ghf_t24 | 24 | membind=0 | 0 |  |  |  |  | 77 | 3.15 | 4.65 | window of 0; MemoryError: incore_complete: 8 DIIS vectors with T3 take 160.8 GB of memory (20.1 GB per pair of a vector and its error vector); 138.7 GB are left next to the other arrays of the calculation.  Without incore_complete they go to the scratch file as far as the disk holds them |
| zc_rhf_t1 | 1 | membind=0 | 5 | 134.394 | 1.00 |  |  | 688 | 0.99 | 16.01 | DIIS pairs: file 0, memory 8 |
| zc_rhf_t2 | 2 | membind=0 | 5 | 72.209 | 1.97 |  |  | 377 | 1.93 | 16.10 | DIIS pairs: file 0, memory 8 |
| zc_rhf_t4 | 4 | membind=0 | 5 | 37.438 | 3.89 |  |  | 206 | 3.62 | 16.04 | DIIS pairs: file 0, memory 8 |
| zc_rhf_t8 | 8 | membind=0 | 5 | 19.522 | 7.66 |  |  | 106 | 7.23 | 16.11 | DIIS pairs: file 0, memory 8 |
| zc_rhf_t16 | 16 | membind=0 | 5 | 11.738 | 14.73 |  |  | 81 | 10.92 | 16.18 | DIIS pairs: file 0, memory 8 |
| zc_rhf_t24 | 24 | membind=0 | 5 | 9.564 | 20.83 |  |  | 56 | 18.11 | 16.37 | DIIS pairs: file 0, memory 8 |
| zc_rhf_t48 | 48 | first touch | 5 | 8.262 | 38.24 |  |  | 47 | 34.15 | 16.62 | DIIS pairs: file 0, memory 8 |
| zc_rhf_active_t24 | 24 | membind=0 | 5 | 74.400 | 23.93 |  |  | 368 | 23.90 | 16.39 | OMP_WAIT_POLICY=ACTIVE; DIIS pairs: file 0, memory 8 |
| zc_rhf_again_t8 | 8 | membind=0 | 5 | 19.496 | 7.65 |  |  | 107 | 7.14 | 16.12 | DIIS pairs: file 0, memory 8 |
| zc_rhf_inter_t48 | 48 | interleave=all | 5 | 7.697 | 38.75 |  |  | 46 | 33.98 | 16.71 | DIIS pairs: file 0, memory 8 |
| zc_rhf_passive_t8 | 8 | membind=0 | 5 | 19.506 | 7.65 |  |  | 104 | 7.34 | 16.13 | OMP_WAIT_POLICY=PASSIVE; DIIS pairs: file 0, memory 8 |
| zc_rhf_passive_t24 | 24 | membind=0 | 5 | 9.579 | 20.84 |  |  | 54 | 18.98 | 16.36 | OMP_WAIT_POLICY=PASSIVE; DIIS pairs: file 0, memory 8 |
| zc_rhfq_t1 | 1 | membind=0 | 2 | 132.749 | 1.00 | 4634.99 | 1.00 | 4917 | 1.00 | 8.58 | window of 1; DIIS pairs: file 0, memory 8 |
| zc_rhfq_t2 | 2 | membind=0 | 2 | 71.222 | 1.97 | 2574.66 | 1.98 | 2729 | 1.97 | 8.59 | window of 1; DIIS pairs: file 0, memory 8 |
| zc_rhfq_t4 | 4 | membind=0 | 2 | 36.844 | 3.90 | 1191.25 | 3.90 | 1274 | 3.88 | 8.61 | window of 1; DIIS pairs: file 0, memory 8 |
| zc_rhfq_t8 | 8 | membind=0 | 2 | 19.242 | 7.65 | 635.49 | 7.56 | 684 | 7.48 | 8.68 | window of 1; DIIS pairs: file 0, memory 8 |
| zc_rhfq_t16 | 16 | membind=0 | 2 | 11.650 | 14.71 | 406.18 | 14.56 | 438 | 14.34 | 8.75 | window of 1; DIIS pairs: file 0, memory 8 |
| zc_rhfq_t24 | 24 | membind=0 | 38 | 9.583 | 20.79 | 358.89 | 20.99 | 741 | 20.78 | 27.29 | converged in 38; DIIS pairs: file 0, memory 8 |
| zc_rhfq_t48 | 48 | first touch | 2 | 8.128 | 38.88 | 316.95 | 35.60 | 339 | 35.18 | 9.15 | window of 1; DIIS pairs: file 0, memory 8 |
| zc_rhfq_passive_t24 | 24 | membind=0 | 2 | 9.658 | 20.58 | 360.96 | 20.88 | 389 | 20.47 | 8.91 | OMP_WAIT_POLICY=PASSIVE; window of 1; DIIS pairs: file 0, memory 8 |

ncc_off, seconds per CCSDT iteration (ncc):

| threads | 1 | 2 | 4 | 7 | 8 | 16 | 23 | 24 | 47 | 48 |
|---|---|---|---|---|---|---|---|---|---|---|
| seconds per CCSDT iteration | 380.60 | 215.09 | 131.90 | 66.20 | 75.94 | 53.33 | 39.30 | 57.35 | 30.71 | 41.15 |
| speedup from 1 thread | 1.00 | 1.77 | 2.89 | 5.75 | 5.01 | 7.14 | 9.68 | 6.64 | 12.39 | 9.25 |
| speedup from 8 threads | 0.20 | 0.35 | 0.58 | 1.15 | 1.00 | 1.42 | 1.93 | 1.32 | 2.47 | 1.85 |
| efficiency (speedup from 1 / threads) | 1.00 | 0.88 | 0.72 | 0.82 | 0.63 | 0.45 | 0.42 | 0.28 | 0.26 | 0.19 |

ncc_off_mkl1, seconds per CCSDT iteration (ncc):

| threads | 8 | 24 |
|---|---|---|
| seconds per CCSDT iteration | 77.42 | 62.49 |
| speedup from 1 thread |  |  |
| speedup from 8 threads | 1.00 | 1.24 |
| efficiency (speedup from 1 / threads) |  |  |

ncc_off_passive, seconds per CCSDT iteration (ncc):

| threads | 8 | 24 |
|---|---|---|
| seconds per CCSDT iteration | 75.95 | 57.41 |
| speedup from 1 thread |  |  |
| speedup from 8 threads | 1.00 | 1.32 |
| efficiency (speedup from 1 / threads) |  |  |

ncc_on, seconds per CCSDT iteration (ncc):

| threads | 1 | 2 | 4 | 7 | 8 | 16 | 23 | 24 | 47 | 48 |
|---|---|---|---|---|---|---|---|---|---|---|
| seconds per CCSDT iteration | 41.09 | 23.01 | 13.40 | 9.438 | 8.433 | 6.143 | 5.695 | 5.768 | 9.142 | 9.979 |
| speedup from 1 thread | 1.00 | 1.79 | 3.07 | 4.35 | 4.87 | 6.69 | 7.22 | 7.12 | 4.50 | 4.12 |
| speedup from 8 threads | 0.21 | 0.37 | 0.63 | 0.89 | 1.00 | 1.37 | 1.48 | 1.46 | 0.92 | 0.85 |
| efficiency (speedup from 1 / threads) | 1.00 | 0.89 | 0.77 | 0.62 | 0.61 | 0.42 | 0.31 | 0.30 | 0.10 | 0.09 |

ncc_on_mkl1, seconds per CCSDT iteration (ncc):

| threads | 8 | 24 |
|---|---|---|
| seconds per CCSDT iteration | 10.08 | 8.592 |
| speedup from 1 thread |  |  |
| speedup from 8 threads | 1.00 | 1.17 |
| efficiency (speedup from 1 / threads) |  |  |

ncc_on_passive, seconds per CCSDT iteration (ncc):

| threads | 8 | 24 |
|---|---|---|
| seconds per CCSDT iteration | 9.485 | 7.687 |
| speedup from 1 thread |  |  |
| speedup from 8 threads | 1.00 | 1.23 |
| efficiency (speedup from 1 / threads) |  |  |

ncc_onq, seconds per CCSDT iteration (ncc):

| threads | 1 | 2 | 4 | 8 | 16 | 24 | 48 |
|---|---|---|---|---|---|---|---|
| seconds per CCSDT iteration | 41.13 | 22.91 | 13.34 | 8.476 | 6.121 | 5.745 | 10.05 |
| speedup from 1 thread | 1.00 | 1.80 | 3.08 | 4.85 | 6.72 | 7.16 | 4.09 |
| speedup from 8 threads | 0.21 | 0.37 | 0.64 | 1.00 | 1.38 | 1.48 | 0.84 |
| efficiency (speedup from 1 / threads) | 1.00 | 0.90 | 0.77 | 0.61 | 0.42 | 0.30 | 0.09 |

zc_rhf, seconds per CCSDT iteration (zccsdt RCCSDT):

| threads | 1 | 2 | 4 | 8 | 16 | 24 | 48 |
|---|---|---|---|---|---|---|---|
| seconds per CCSDT iteration | 134.39 | 72.21 | 37.44 | 19.52 | 11.74 | 9.564 | 8.262 |
| speedup from 1 thread | 1.00 | 1.86 | 3.59 | 6.88 | 11.45 | 14.05 | 16.27 |
| speedup from 8 threads | 0.15 | 0.27 | 0.52 | 1.00 | 1.66 | 2.04 | 2.36 |
| efficiency (speedup from 1 / threads) | 1.00 | 0.93 | 0.90 | 0.86 | 0.72 | 0.59 | 0.34 |

zc_rhf_passive, seconds per CCSDT iteration (zccsdt RCCSDT):

| threads | 8 | 24 |
|---|---|---|
| seconds per CCSDT iteration | 19.51 | 9.579 |
| speedup from 1 thread |  |  |
| speedup from 8 threads | 1.00 | 2.04 |
| efficiency (speedup from 1 / threads) |  |  |

zc_rhfq, seconds per CCSDT iteration (zccsdt RCCSDT):

| threads | 1 | 2 | 4 | 8 | 16 | 24 | 48 |
|---|---|---|---|---|---|---|---|
| seconds per CCSDT iteration | 132.75 | 71.22 | 36.84 | 19.24 | 11.65 | 9.583 | 8.128 |
| speedup from 1 thread | 1.00 | 1.86 | 3.60 | 6.90 | 11.39 | 13.85 | 16.33 |
| speedup from 8 threads | 0.14 | 0.27 | 0.52 | 1.00 | 1.65 | 2.01 | 2.37 |
| efficiency (speedup from 1 / threads) | 1.00 | 0.93 | 0.90 | 0.86 | 0.71 | 0.58 | 0.34 |

ncc_onq, seconds of (Q) (ncc):

| threads | 1 | 2 | 4 | 8 | 16 | 24 | 48 |
|---|---|---|---|---|---|---|---|
| seconds of (Q) | 4579.10 | 2690.88 | 1616.87 | 1057.43 | 824.52 | 862.99 | 812.50 |
| speedup from 1 thread | 1.00 | 1.70 | 2.83 | 4.33 | 5.55 | 5.31 | 5.64 |
| speedup from 8 threads | 0.23 | 0.39 | 0.65 | 1.00 | 1.28 | 1.23 | 1.30 |
| efficiency (speedup from 1 / threads) | 1.00 | 0.85 | 0.71 | 0.54 | 0.35 | 0.22 | 0.12 |

zc_rhfq, seconds of (Q) (zccsdt RCCSDT):

| threads | 1 | 2 | 4 | 8 | 16 | 24 | 48 |
|---|---|---|---|---|---|---|---|
| seconds of (Q) | 4634.99 | 2574.66 | 1191.25 | 635.49 | 406.18 | 358.89 | 316.95 |
| speedup from 1 thread | 1.00 | 1.80 | 3.89 | 7.29 | 11.41 | 12.91 | 14.62 |
| speedup from 8 threads | 0.14 | 0.25 | 0.53 | 1.00 | 1.56 | 1.77 | 2.01 |
| efficiency (speedup from 1 / threads) | 1.00 | 0.90 | 0.97 | 0.91 | 0.71 | 0.54 | 0.30 |

ncc against zccsdt RCCSDT, seconds per CCSDT iteration:

| threads | ncc, symmetry | ncc, no symmetry | zccsdt | ncc (symmetry) / zccsdt | ncc (no symmetry) / zccsdt |
|---|---|---|---|---|---|
| 1 | 41.094 | 380.603 | 134.394 | 0.31 | 2.83 |
| 2 | 23.007 | 215.089 | 72.209 | 0.32 | 2.98 |
| 4 | 13.399 | 131.899 | 37.438 | 0.36 | 3.52 |
| 8 | 8.433 | 75.942 | 19.522 | 0.43 | 3.89 |
| 16 | 6.143 | 53.331 | 11.738 | 0.52 | 4.54 |
| 24 | 5.768 | 57.345 | 9.564 | 0.60 | 6.00 |
| 48 | 9.979 | 41.146 | 8.262 | 1.21 | 4.98 |

ncc against zccsdt RCCSDT, seconds of (Q):

| threads | ncc, symmetry | ncc, no symmetry | zccsdt | ncc (symmetry) / zccsdt | ncc (no symmetry) / zccsdt |
|---|---|---|---|---|---|
| 1 | 4579.103 |  | 4634.990 | 0.99 |  |
| 2 | 2690.880 |  | 2574.660 | 1.05 |  |
| 4 | 1616.866 |  | 1191.250 | 1.36 |  |
| 8 | 1057.432 |  | 635.490 | 1.66 |  |
| 16 | 824.522 |  | 406.180 | 2.03 |  |
| 24 | 862.986 |  | 358.890 | 2.40 |  |
| 48 | 812.495 |  | 316.950 | 2.56 |  |

Energies of the converged runs:

| run | E(SCF) | E(CCSDT) | E(Q) | E(CCSDT(Q)) |
|---|---|---|---|---|
| ncc_onq_t24 | -224.340411087556 | -225.169291054214 | -0.007352824815 | -225.176643879028 |
| zc_rhfq_t24 | -224.340411087550 | -225.169291054225 | -0.007352824817 | -225.176643879041 |

ncc_onq_t24 - zc_rhfq_t24: CCSDT 1.1e-11  CCSDT(Q) 1.3e-11

ncc with and without symmetry, largest difference of the correlation energies of iterations 1-5 (ncc_on_t48, ncc_off_t48): 6.0e-12

