"""JFO dataset with the cut-cell groove solver (solver2.solve_jfo_cut): 10 000 random bearings (geometry from
make_dataset.make_film_grid), on an n x n grid (default 64). The network input G stays the nodal groove mask.
usage: python make_dataset_jfo_cut.py [n_grid] [n_samples]  -> dataset_jfo_cut.npz (64) or dataset_jfo_cut<n>.npz"""
import os, sys, time
import numpy as np
from multiprocessing import Pool
from solver import grid, groove_mask
from solver2 import solve_jfo_cut
from make_dataset import make_film_grid

NG = int(sys.argv[1]) if len(sys.argv) > 1 else 64


def sample(seed):
    th, ze = grid(NG, NG)
    H, LD = make_film_grid(seed, th, ze)
    rng = np.random.default_rng(seed + 9 * 10 ** 6)
    tg = np.radians(rng.uniform(-30, 30)); Ps = float(np.exp(rng.uniform(np.log(0.002), np.log(0.05))))
    t = time.perf_counter()
    P, Tf, it = solve_jfo_cut(H, 2 * LD, th, ze, tg, Ps)
    G = np.zeros_like(H); G[:, 1:-1] = groove_mask(th, ze, tg)
    return H.astype(np.float32), np.float32(LD), G.astype(np.float32), np.float32(Ps), P.astype(np.float32), \
        Tf.astype(np.float32), time.perf_counter() - t


if __name__ == "__main__":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10000
    seeds = range(n) if len(sys.argv) <= 3 else [int(s) for s in open(sys.argv[3]).read().split()]
    with Pool(8) as pool:
        res = pool.map(sample, list(seeds), chunksize=20)
    H, LD, G, Ps, P, Tf, t = map(np.array, zip(*res))
    th, ze = grid(NG, NG)
    prm = np.load("dataset_jfo.npz")["params"][list(seeds)]
    out = os.environ.get("OUT") or ("dataset_jfo_cut.npz" if NG == 64 else f"dataset_jfo_cut{NG}.npz")
    np.savez_compressed(out, H=H, LD=LD, G=G, Ps=Ps, P=P, Tf=Tf, params=prm, theta=th, zeta=ze, seeds=np.array(list(seeds)))
    print(out, H.shape, "median solve %.0f ms" % (1e3 * np.median(t)))
