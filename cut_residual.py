"""Cut-cell rows for the physics loss (pino_jfo.py, CUT=2).

The cut-cell labels (solver2.solve_jfo_cut) use a modified stencil in the rows next to the groove. With CUT=1 those rows
were left out of the JFO residual; CUT=2 evaluates them with the exact cut-cell coefficients instead, so the residual of
a cut-cell solution vanishes on every non-groove node and the groove-edge flux is constrained by the physics loss too.

For every bearing, the rows of solver2.assemble_cut adjacent to the groove are stored as (row, cols[5], coef[5], f, diag),
padded to M rows (padding repeats the first row). Groove nodes enter with P = Ps through the hard constraint.
usage (library): rows, cols, coef, fval, diag = cut_rows_batch(H, LD, tg, theta, zeta)    (numpy, Pool)
"""
import os
import numpy as np
from multiprocessing import Pool

M = 200     # max adjacent rows per bearing on 64 x 64 (about 120); finer grids: 3.5 n_theta


def cut_rows(args):
    H, LD, tg, theta, zeta = args
    from solver2 import assemble_cut
    K, f, gm = assemble_cut(H.astype(np.float64), 2 * float(LD), theta, zeta, float(tg))
    nt, nzi = gm.shape
    M = 200 if nt <= 64 else int(3.5 * nt)
    adj = (np.roll(gm, 1, 0) | np.roll(gm, -1, 0) | np.pad(gm[:, 1:], ((0, 0), (0, 1))) | np.pad(gm[:, :-1], ((0, 0), (1, 0)))) & ~gm
    ridx = np.flatnonzero(adj.ravel())
    assert 0 < len(ridx) <= M, len(ridx)
    K = K.tocsr()
    rows = np.empty(M, np.int64); cols = np.zeros((M, 5), np.int64); coef = np.zeros((M, 5), np.float32)
    fval = np.zeros(M, np.float32); diag = np.zeros(M, np.float32)
    for k, i in enumerate(ridx):
        s, e = K.indptr[i], K.indptr[i + 1]
        c, v = K.indices[s:e], K.data[s:e]
        assert len(c) <= 5
        rows[k] = i; cols[k, :len(c)] = c; coef[k, :len(c)] = v; cols[k, len(c):] = i      # unused slots: coef 0
        fval[k] = f[i]; diag[k] = K[i, i]
    n = len(ridx)
    rows[n:] = rows[0]; cols[n:] = cols[0]; coef[n:] = coef[0]; fval[n:] = fval[0]; diag[n:] = diag[0]
    return rows, cols, coef, fval, diag


def cut_rows_batch(H, LD, tg, theta, zeta, procs=None, cache=None):
    if cache and os.path.exists(cache):
        z = np.load(cache)
        return z["rows"], z["cols"], z["coef"], z["fval"], z["diag"]
    procs = procs or int(os.environ.get("PROCS", 12))
    args = [(H[i], LD[i], tg[i], theta, zeta) for i in range(len(H))]
    if procs == 1:
        out = [cut_rows(x) for x in args]
    else:
        with Pool(procs) as pool:
            out = pool.map(cut_rows, args, chunksize=20)
    rows, cols, coef, fval, diag = (np.stack(a) for a in zip(*out))
    if cache:
        np.savez(cache, rows=rows, cols=cols, coef=coef, fval=fval, diag=diag)
    return rows, cols, coef, fval, diag
