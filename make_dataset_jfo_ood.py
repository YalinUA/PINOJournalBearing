"""Out-of-range JFO test sets. Same construction as make_dataset_jfo.py, but one parameter is pushed
outside the training range per set (everything else sampled as in training):

  eps    eps ~ U(0.90, 0.95)                         (train 0.1-0.9)
  ld     L/D ~ U(0.15, 0.25) or U(2, 3), 50/50        (train 0.25-2)
  depth  always textured, dimple depth ~ U(0.5, 1.0)  (train 0.05-0.5, textured 50%)
  ps     Ps ~ logU(5e-4, 2e-3) or logU(0.05, 0.2)     (train 2e-3-0.05)

usage: python make_dataset_jfo_ood.py <set> [n]   -> dataset_jfo_ood_<set>.npz (n default 1000)
Seeds start at 2e6 + 1e5 * set_index, disjoint from training/unlabeled seeds.
"""
import os, sys
import numpy as np
from multiprocessing import Pool
from make_dataset import theta, zeta
from solver import film_plain, add_dimples, groove_mask, solve_jfo
import os as _os
SOLVER = _os.environ.get("SOLVER", "staircase")   # "cut": cut-cell groove (solver2.solve_jfo_cut)

SETS = ["eps", "ld", "depth", "ps"]
KIND = sys.argv[1] if len(sys.argv) > 1 else "eps"


def make_film_ood(seed, kind):
    rng = np.random.default_rng(seed)
    eps = rng.uniform(0.90, 0.95) if kind == "eps" else rng.uniform(0.1, 0.9)
    if kind == "ld":
        LD = rng.uniform(0.15, 0.25) if rng.random() < 0.5 else rng.uniform(2.0, 3.0)
    else:
        LD = rng.uniform(0.25, 2.0)
    LR = 2 * LD
    mis = rng.uniform(0, 0.3) if rng.random() < 0.3 else 0.0
    mis = min(mis, 2 * (0.95 - eps))  # keep the film from closing (same rule as training)
    H = film_plain(eps, theta, zeta, mis, rng.uniform(0, 2 * np.pi))
    textured = kind == "depth" or rng.random() < 0.5
    n_dimp = rng.integers(1, 13) if textured else 0
    dlo, dhi = (0.5, 1.0) if kind == "depth" else (0.05, 0.5)
    dimples = [(rng.uniform(0, 2 * np.pi), rng.uniform(0.15, 0.85),
                rng.uniform(0.1, 0.4), rng.uniform(dlo, dhi)) for _ in range(n_dimp)]
    H = add_dimples(H, theta, zeta, LR, dimples)
    return H, LD, eps, mis, n_dimp


def sample(seed):
    H, LD, eps, mis, n_dimp = make_film_ood(seed, KIND)
    rng = np.random.default_rng(seed + 9 * 10 ** 6)
    tg = np.radians(rng.uniform(-30, 30))
    if KIND == "ps":
        lo, hi = (5e-4, 2e-3) if rng.random() < 0.5 else (0.05, 0.2)
    else:
        lo, hi = 0.002, 0.05
    Ps = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    g = groove_mask(theta, zeta, tg)
    if SOLVER == "cut":
        from solver2 import solve_jfo_cut
        P, Tf, it = solve_jfo_cut(H, 2 * LD, theta, zeta, tg, Ps)
    else:
        P, Tf, it = solve_jfo(H, 2 * LD, g, Ps)
    G = np.zeros_like(H); G[:, 1:-1] = g
    return (H.astype(np.float32), np.float32(LD), G.astype(np.float32), np.float32(Ps),
            P.astype(np.float32), Tf.astype(np.float32),
            np.array([eps, LD, mis, n_dimp, np.degrees(tg), Ps, it], np.float32))


def unlabeled_wide(k, with_tg=False):
    """Unlabeled input (no solve) for PINO training: 1/5 each in-range, eps, ld, depth, ps out-of-range.
    Seeds 3e6 + k, disjoint from the OOD test sets."""
    seed = 3 * 10 ** 6 + k
    rng = np.random.default_rng(seed + 9 * 10 ** 6)
    kind = ["in", "eps", "ld", "depth", "ps"][rng.integers(5)]
    H, LD, *_ = make_film_ood(seed, kind)
    tg = np.radians(rng.uniform(-30, 30))
    lo, hi = ((5e-4, 2e-3) if rng.random() < 0.5 else (0.05, 0.2)) if kind == "ps" else (0.002, 0.05)
    Ps = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
    G = np.zeros_like(H); G[:, 1:-1] = groove_mask(theta, zeta, tg)
    return (H, LD, G, Ps, tg) if with_tg else (H, LD, G, Ps)


if __name__ == "__main__":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    s0 = 2 * 10 ** 6 + 10 ** 5 * SETS.index(KIND)
    with Pool(int(os.environ.get("PROCS", 4))) as pool:
        res = pool.map(sample, range(s0, s0 + n), chunksize=25)
    H, LD, G, Ps, P, Tf, prm = map(np.stack, zip(*res))
    out = f"dataset_jfo_cut_ood_{KIND}.npz" if SOLVER == "cut" else f"dataset_jfo_ood_{KIND}.npz"
    np.savez_compressed(out, H=H, LD=LD, G=G, Ps=Ps, P=P, Tf=Tf, params=prm, theta=theta, zeta=zeta)
    print(out, H.shape, "max iters %d" % prm[:, 6].max(), "P max %.3f" % P.max())
