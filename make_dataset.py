"""Generate (film thickness, L/D) -> pressure pairs for journal bearings.

Each sample: eps ~ U(0.1, 0.9), L/D ~ U(0.25, 2), optional axial misalignment,
and with probability 0.5 a random set of 1-12 dimples (surface texture).
Output: dataset.npz with H (N,64,64), LD (N,), P (N,64,64), params.
"""
import sys
import numpy as np
from multiprocessing import Pool
from solver import grid, film_plain, add_dimples, solve

NT = NZ = 64
DEPTH_MAX = float(__import__('os').environ.get('DEPTH_MAX', 0.5))  # max dimple depth / c
theta, zeta = grid(NT, NZ)


def make_film(seed):
    """Random journal-bearing film (eps, L/D, misalignment, dimples); no solve."""
    rng = np.random.default_rng(seed)
    eps = rng.uniform(0.1, 0.9)
    LD = rng.uniform(0.25, 2.0)
    LR = 2 * LD
    mis = rng.uniform(0, 0.3) if rng.random() < 0.3 else 0.0
    mis = min(mis, 2 * (0.95 - eps))  # keep the film from closing
    H = film_plain(eps, theta, zeta, mis, rng.uniform(0, 2 * np.pi))
    n_dimp = rng.integers(1, 13) if rng.random() < 0.5 else 0
    dimples = [(rng.uniform(0, 2 * np.pi), rng.uniform(0.15, 0.85),
                rng.uniform(0.1, 0.4), rng.uniform(0.05, DEPTH_MAX)) for _ in range(n_dimp)]
    H = add_dimples(H, theta, zeta, LR, dimples)
    return H, LD, eps, mis, n_dimp


def make_film_grid(seed, th, ze):
    """Same random draws as make_film, on an arbitrary (theta, zeta) grid."""
    rng = np.random.default_rng(seed)
    eps = rng.uniform(0.1, 0.9); LD = rng.uniform(0.25, 2.0); LR = 2 * LD
    mis = rng.uniform(0, 0.3) if rng.random() < 0.3 else 0.0
    mis = min(mis, 2 * (0.95 - eps))
    H = film_plain(eps, th, ze, mis, rng.uniform(0, 2 * np.pi))
    n_dimp = rng.integers(1, 13) if rng.random() < 0.5 else 0
    dimples = [(rng.uniform(0, 2 * np.pi), rng.uniform(0.15, 0.85), rng.uniform(0.1, 0.4), rng.uniform(0.05, DEPTH_MAX))
               for _ in range(n_dimp)]
    return add_dimples(H, th, ze, LR, dimples), LD


def sample(seed):
    H, LD, eps, mis, n_dimp = make_film(seed)
    P, it = solve(H, 2 * LD)
    return H.astype(np.float32), np.float32(LD), P.astype(np.float32), \
        np.array([eps, LD, mis, n_dimp], np.float32)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    out = sys.argv[2] if len(sys.argv) > 2 else "dataset.npz"
    with Pool(4) as pool:
        res = pool.map(sample, range(n), chunksize=50)
    H, LD, P, prm = map(np.stack, zip(*res))
    np.savez_compressed(out, H=H, LD=LD, P=P, params=prm,
                        theta=theta, zeta=zeta)
    print(out, H.shape, "P max %.3f median-of-max %.3f" % (P.max(), np.median(P.max((1, 2)))))
