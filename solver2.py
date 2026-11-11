"""Higher-order JFO solver: second-order TVD (minmod-limited) upwinding of the void-fraction Couette flux.

solver.py discretizes d(H r)/dtheta (r = 1 - theta_f) with first-order upwinding: face flux F_{i+1/2} = q_i, q = H r.
Here F_{i+1/2} = q_i + s_i / 2 with the limited slope s_i = minmod(q_{i+1} - q_i, q_i - q_{i-1}) (flow towards +theta).
For a fixed choice of minmod branch per node (forward difference, backward difference or zero) the flux is linear in r,
so each step is the same linear active-set system as in solver.py with a modified B matrix. The branch selection and
the active (cavitated) set are iterated together until neither changes. Everything else (Poiseuille term, Couette term
of H, groove, edges) is identical to solver.py, so the two solvers differ only in the cavitated-zone transport.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from solver import assemble, solve_jfo, grid, groove_mask, integrals, film_plain, add_dimples  # noqa: F401


def _B_limited(H, sel):
    """B such that (B r)_i = (F_{i+1/2} - F_{i-1/2}) / dtheta with F_{i+1/2} = q_i + s_i/2, q = H r on interior z nodes.
    sel (n_t, nzi): 0 -> s_i = 0, 1 -> s_i = q_{i+1} - q_i, 2 -> s_i = q_i - q_{i-1}."""
    n_t, n_z = H.shape
    nzi = n_z - 2
    dt = 2 * np.pi / n_t
    Hi = H[:, 1:-1]
    idx = np.arange(n_t * nzi).reshape(n_t, nzi)
    R, C, V = [], [], []

    def add(rows, cols, vals):
        R.append(rows.ravel()); C.append(cols.ravel()); V.append(vals.ravel())

    roll = lambda a, k: np.roll(a, k, axis=0)
    # face i+1/2 contributes +F/dt to row i, face i-1/2 = F_{i-1+1/2} contributes -F/dt to row i
    for sign, shift in ((+1, 0), (-1, 1)):          # shift 0: face i+1/2 (cell i); shift 1: face i-1/2 (cell i-1)
        c = roll(idx, shift)                         # the upwind cell j of this face
        Hj, s = roll(Hi, shift), roll(sel, shift)
        cp, cm = roll(idx, shift - 1), roll(idx, shift + 1)
        Hp, Hm = roll(Hi, shift - 1), roll(Hi, shift + 1)
        w = sign / dt
        add(idx, c, w * Hj)                                         # q_j
        f1, f2 = (s == 1), (s == 2)                                 # + 0.5 (q_{j+1} - q_j)   or   + 0.5 (q_j - q_{j-1})
        add(idx[f1], cp[f1], 0.5 * w * Hp[f1]); add(idx[f1], c[f1], -0.5 * w * Hj[f1])
        add(idx[f2], c[f2], 0.5 * w * Hj[f2]); add(idx[f2], cm[f2], -0.5 * w * Hm[f2])
    N = n_t * nzi
    return sp.csr_matrix((np.concatenate(V), (np.concatenate(R), np.concatenate(C))), shape=(N, N))


def _minmod_sel(q):
    a = np.roll(q, -1, 0) - q          # forward difference
    b = q - np.roll(q, 1, 0)           # backward difference
    sel = np.zeros(q.shape, dtype=np.int8)
    same = a * b > 0
    sel[same & (np.abs(a) <= np.abs(b))] = 1
    sel[same & (np.abs(a) > np.abs(b))] = 2
    return sel


def solve_jfo2(H, L_over_R, groove, Ps, M=None, max_outer=60, cav0=None, return_info=False):
    """Second-order (minmod TVD) JFO solve. Starts from the first-order solution of solver.solve_jfo.
    Returns P, theta_f (full grid) and the number of outer iterations (and info dict if return_info)."""
    n_t, n_z = H.shape
    P1, T1, it1 = solve_jfo(H, L_over_R, groove, Ps, M=M, cav0=cav0)
    K, f = assemble(H, L_over_R, M)
    n = K.shape[0]
    g = groove.ravel(); rows = ~g
    Kc = K.tocsc()
    rhs = (f - Kc[:, g] @ np.full(g.sum(), Ps))[rows]
    Kr = Kc[rows]
    r = (1 - T1[:, 1:-1]).ravel()
    cav = (r > 0) & rows
    Hi = H[:, 1:-1]
    sel = _minmod_sel(Hi * r.reshape(n_t, -1))
    history = set()
    for it in range(1, max_outer + 1):
        B = _B_limited(H, sel).tocsc()
        Br = B[rows]
        pres = rows & ~cav
        A = sp.hstack([Kr[:, pres], -Br[:, cav]]).tocsc()
        u = spla.spsolve(A, rhs)
        P = np.zeros(n); r = np.zeros(n)
        P[pres] = u[:pres.sum()]; r[cav] = u[pres.sum():]; P[g] = Ps
        new = cav.copy()
        new[pres & (P < -1e-12)] = True
        new[cav & (r < -1e-12)] = False
        new_sel = _minmod_sel(Hi * np.clip(r, 0, None).reshape(n_t, -1))
        if np.array_equal(new, cav) and np.array_equal(new_sel, sel):
            break
        key = (new.tobytes(), new_sel.tobytes())
        if key in history:            # cycling between branch choices: stop at the current consistent state
            break
        history.add(key)
        cav, sel = new, new_sel
    Pf = np.zeros((n_t, n_z)); Tf = np.ones((n_t, n_z))
    Pf[:, 1:-1] = np.maximum(P, 0).reshape(n_t, n_z - 2)
    Tf[:, 1:-1] = (1 - np.clip(r, 0, 1)).reshape(n_t, n_z - 2)
    if return_info:
        return Pf, Tf, it, dict(first_order_iters=it1, converged=it < max_outer)
    return Pf, Tf, it


# ---------------------------------------------------------------------------------------------------------------
# Sub-cell (cut-cell) groove: exact groove edges with a Shortley-Weller boundary treatment
# ---------------------------------------------------------------------------------------------------------------
def assemble_cut(H, L_over_R, theta, zeta, theta_g, width=np.radians(12), z_extent=0.8, M=None):
    """K, f as solver.assemble, but rows of nodes next to the groove use the exact distance delta to the groove edge
    (Dirichlet P = Ps there) instead of the neighbouring node:
        theta part: 2/(delta+dt) [ H3_b (P_i - Ps)/delta + H3_o (P_i - P_o)/dt ],  Couette: (H_b - Hf_o)/((delta+dt)/2)
        z part:     2/(delta+dz) [ H3_b (P_i - Ps)/delta + H3_o (P_i - P_o)/dz ]
    H3_b is H^3 at the middle of the cut segment, H_b is H at the groove edge (linear interpolation of H).
    The groove neighbour column receives the boundary coefficient, so eliminating groove nodes with value Ps
    (as solve_jfo_cut does) places Ps at the true edge. Returns K, f, groove mask (interior z nodes)."""
    n_t, n_z = H.shape
    dt = 2 * np.pi / n_t
    dz = L_over_R / (n_z - 1)
    H3 = H ** 3 if M is None else H ** 3 / M
    K0, f0 = assemble(H, L_over_R, M)
    K = K0.tolil()
    f = f0.copy()
    gm = groove_mask(theta, zeta, theta_g, width, z_extent)             # (n_t, n_z-2)
    nzi = n_z - 2
    idx = np.arange(n_t * nzi).reshape(n_t, nzi)
    Hf = 0.5 * (H + np.roll(H, -1, axis=0))
    t_w_edge, t_e_edge = theta_g - width / 2, theta_g + width / 2       # groove spans [t_w_edge, t_e_edge]
    z_lo, z_hi = 0.5 - z_extent / 2, 0.5 + z_extent / 2
    lin = lambda a, b, s: a + s * (b - a)
    for i in range(n_t):
        for jj in range(nzi):
            if gm[i, jj]:
                continue
            j = jj + 1                                                   # full-grid z index
            row = idx[i, jj]
            ip, im = (i + 1) % n_t, (i - 1) % n_t
            # ---- theta direction ----
            for nb, edge, other, sgn in ((ip, t_w_edge, im, +1), (im, t_e_edge, ip, -1)):
                if not gm[nb, jj]:
                    continue
                delta = np.mod(sgn * (edge - theta[i]), 2 * np.pi)
                delta = float(np.clip(delta, 0.02 * dt, dt))
                s = delta / dt
                Hb = lin(H[i, j], H[nb, j], s)                           # H at the groove edge
                H3b = lin(H[i, j], H[nb, j], s / 2) ** 3                 # H^3 mid-segment
                H3o = 0.5 * (H3[i, j] + H3[other, j])                    # face towards the other neighbour
                c = 2 / (delta + dt)
                K[row, row] = K[row, row] - (H3o + 0.5 * (H3[i, j] + H3[nb, j])) / dt ** 2 + c * (H3b / delta + H3o / dt)
                K[row, idx[nb, jj]] = -c * H3b / delta
                K[row, idx[other, jj]] = -c * H3o / dt
                # Couette RHS: -(F_downstream - F_upstream)/vol with the boundary face at the groove edge
                Hf_o = Hf[i, j] if sgn < 0 else Hf[im, j]                # the regular face on the far side
                f[row] = -sgn * (Hb - Hf_o) / ((delta + dt) / 2)
            # ---- z direction ----
            for jn, edge, jo, sgn in ((jj + 1, z_lo, jj - 1, +1), (jj - 1, z_hi, jj + 1, -1)):
                if not (0 <= jn < nzi) or not gm[i, jn]:
                    continue
                delta = float(np.clip(sgn * (edge - zeta[j]) * L_over_R, 0.02 * dz, dz))
                s = delta / dz
                H3b = lin(H[i, j], H[i, jn + 1], s / 2) ** 3
                H3o = 0.5 * (H3[i, j] + H3[i, jo + 1])
                c = 2 / (delta + dz)
                K[row, row] = K[row, row] - (H3o + 0.5 * (H3[i, j] + H3[i, jn + 1])) / dz ** 2 + c * (H3b / delta + H3o / dz)
                K[row, idx[i, jn]] = -c * H3b / delta
                if 0 <= jo < nzi:
                    K[row, idx[i, jo]] = -c * H3o / dz
                # (edge node jo = -1 or nzi is Dirichlet P = 0 and eliminated; its coefficient drops out)
    return K.tocsr(), f, gm


def solve_jfo_cut(H, L_over_R, theta, zeta, theta_g, Ps, width=np.radians(12), z_extent=0.8, M=None,
                  max_iter=300, cav0=None):
    """JFO solve (first-order void transport, as solver.solve_jfo) with the cut-cell groove of assemble_cut."""
    n_t, n_z = H.shape
    K, f, gm = assemble_cut(H, L_over_R, theta, zeta, theta_g, width, z_extent, M)
    nzi = n_z - 2
    dt = 2 * np.pi / n_t
    idx = np.arange(n_t * nzi).reshape(n_t, nzi)
    Hi = H[:, 1:-1]
    B = sp.csr_matrix((np.concatenate([Hi.ravel() / dt, -np.roll(Hi, 1, axis=0).ravel() / dt]),
                       (np.concatenate([idx.ravel(), idx.ravel()]), np.concatenate([idx.ravel(), np.roll(idx, 1, axis=0).ravel()]))),
                      shape=K.shape)
    n = K.shape[0]
    g = gm.ravel(); rows = ~g
    Kc, Bc = K.tocsc(), B.tocsc()
    rhs = (f - Kc[:, g] @ np.full(g.sum(), Ps))[rows]
    Kr, Br = Kc[rows], Bc[rows]
    if cav0 is not None:
        cav = cav0.ravel() & rows
    else:
        P = np.zeros(n); P[rows] = spla.spsolve(Kr[:, rows].tocsc(), rhs)
        cav = (P < 0) & rows
    for it in range(1, max_iter + 1):
        pres = rows & ~cav
        A = sp.hstack([Kr[:, pres], -Br[:, cav]]).tocsc()
        u = spla.spsolve(A, rhs)
        P = np.zeros(n); r = np.zeros(n)
        P[pres] = u[:pres.sum()]; r[cav] = u[pres.sum():]; P[g] = Ps
        new = cav.copy()
        new[pres & (P < -1e-12)] = True
        new[cav & (r < -1e-12)] = False
        if np.array_equal(new, cav):
            break
        cav = new
    Pf = np.zeros((n_t, n_z)); Tf = np.ones((n_t, n_z))
    Pf[:, 1:-1] = np.maximum(P, 0).reshape(n_t, nzi)
    Tf[:, 1:-1] = (1 - np.clip(r, 0, 1)).reshape(n_t, nzi)
    return Pf, Tf, it
