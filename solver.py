"""Finite-difference Reynolds solver for a journal bearing (steady, isothermal, incompressible).

Dimensionless form (theta = x/R in [0, 2pi), zb = z/R in [0, L/R]):
    d/dtheta(H^3 dP/dtheta) + d/dzb(H^3 dP/dzb) = dH/dtheta
with H = h/c and P = p c^2 / (6 mu U R), U = omega R.

Grid: N_t nodes around the circumference (periodic), N_z nodes axially including the
two edges, where P = 0 (ambient). Cavitation uses the Reynolds (Swift-Stieber) condition,
solved exactly as the obstacle problem P >= 0 with a primal-dual active-set method.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def grid(n_t=64, n_z=64):
    theta = 2 * np.pi * np.arange(n_t) / n_t
    zeta = np.linspace(0.0, 1.0, n_z)  # axial coordinate normalised by L
    return theta, zeta


def film_plain(eps, theta, zeta, misalign=0.0, mis_phase=0.0):
    """H = 1 + eps cos(theta) plus an optional linear axial misalignment term."""
    T, Z = np.meshgrid(theta, zeta, indexing="ij")
    return 1.0 + eps * np.cos(T) + misalign * (Z - 0.5) * np.cos(T - mis_phase)


def add_dimples(H, theta, zeta, L_over_R, dimples):
    """dimples: list of (theta0, zeta0, radius/R, depth/c). Spherical-cap-like smooth pockets."""
    T, Z = np.meshgrid(theta, zeta, indexing="ij")
    H = H.copy()
    for t0, z0, r, d in dimples:
        dt = np.angle(np.exp(1j * (T - t0)))  # periodic distance in theta
        dz = (Z - z0) * L_over_R
        rho2 = (dt ** 2 + dz ** 2) / r ** 2
        H += d * np.clip(1.0 - rho2, 0.0, None)
    return H


def assemble(H, L_over_R, M=None):
    """Return sparse K (SPD) and f such that K P = f is the discrete Reynolds equation
    on interior axial nodes (edge nodes are Dirichlet P = 0 and are eliminated).
    M: optional viscosity field mu/mu_ref (same shape as H); flow coefficient H^3/M."""
    n_t, n_z = H.shape
    dt = 2 * np.pi / n_t
    dz = L_over_R / (n_z - 1)
    H3 = H ** 3 if M is None else H ** 3 / M
    # face values (arithmetic mean of H^3)
    Hte = 0.5 * (H3 + np.roll(H3, -1, axis=0))       # face i+1/2 in theta
    Htw = np.roll(Hte, 1, axis=0)                    # face i-1/2
    Hzn = 0.5 * (H3[:, 1:] + H3[:, :-1])             # face j+1/2 for j=0..n_z-2
    # conservative RHS: (H_{i+1/2} - H_{i-1/2}) / dt
    Hf = 0.5 * (H + np.roll(H, -1, axis=0))
    rhs = (Hf - np.roll(Hf, 1, axis=0)) / dt

    nzi = n_z - 2
    idx = np.arange(n_t * nzi).reshape(n_t, nzi)
    jj = np.arange(1, n_z - 1)
    ae = Hte[:, jj] / dt ** 2
    aw = Htw[:, jj] / dt ** 2
    an = Hzn[:, jj] / dz ** 2        # face j+1/2
    as_ = Hzn[:, jj - 1] / dz ** 2   # face j-1/2
    diag = ae + aw + an + as_

    rows, cols, vals = [idx.ravel()], [idx.ravel()], [diag.ravel()]
    rows.append(idx.ravel()); cols.append(np.roll(idx, -1, axis=0).ravel()); vals.append(-ae.ravel())
    rows.append(idx.ravel()); cols.append(np.roll(idx, 1, axis=0).ravel()); vals.append(-aw.ravel())
    rows.append(idx[:, :-1].ravel()); cols.append(idx[:, 1:].ravel()); vals.append(-an[:, :-1].ravel())
    rows.append(idx[:, 1:].ravel()); cols.append(idx[:, :-1].ravel()); vals.append(-as_[:, 1:].ravel())
    K = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(n_t * nzi, n_t * nzi))
    f = -rhs[:, jj].ravel()   # -div(H^3 grad P) = -dH/dtheta
    return K, f


def solve(H, L_over_R, cavitation=True, max_iter=100, M=None):
    """Return P on the full (n_t, n_z) grid and the number of active-set iterations."""
    n_t, n_z = H.shape
    K, f = assemble(H, L_over_R, M)
    n = K.shape[0]
    P = spla.spsolve(K.tocsc(), f)
    it = 0
    if cavitation:
        lam = np.zeros(n)
        active = P < 0
        for it in range(1, max_iter + 1):
            free = ~active
            P = np.zeros(n)
            Kff = K[free][:, free].tocsc()
            P[free] = spla.spsolve(Kff, f[free])
            lam = K @ P - f
            new_active = (lam - 1e3 * P) > 1e-12
            if np.array_equal(new_active, active):
                break
            active = new_active
        P = np.maximum(P, 0.0)
    full = np.zeros((n_t, n_z))
    full[:, 1:-1] = P.reshape(n_t, n_z - 2)
    return full, it


def integrals(P, theta, L_over_R):
    """Dimensionless load components and attitude angle (load W = (6 mu U R^3/c^2) * W_bar)."""
    n_t, n_z = P.shape
    dt = 2 * np.pi / n_t
    zb = np.linspace(0, L_over_R, n_z)
    Pz = np.trapezoid(P, zb, axis=1)
    Wx = -np.sum(Pz * np.cos(theta)) * dt   # along line of centres
    Wy = np.sum(Pz * np.sin(theta)) * dt
    return np.hypot(Wx, Wy), np.degrees(np.arctan2(Wy, Wx))


# ---------------------------------------------------------------------------
# JFO (mass-conserving) cavitation, Giacopini-type complementarity formulation
# ---------------------------------------------------------------------------
def groove_mask(theta, zeta, theta_g=0.0, width=np.radians(12), z_extent=0.8):
    """Axial supply groove: nodes within width/2 of theta_g and |zeta-0.5| < z_extent/2.
    Returned on interior axial nodes, shape (n_t, n_z-2)."""
    T, Z = np.meshgrid(theta, zeta[1:-1], indexing="ij")
    dth = np.abs(np.angle(np.exp(1j * (T - theta_g))))
    return (dth <= width / 2 + 1e-9) & (np.abs(Z - 0.5) <= z_extent / 2)


def assemble_jfo(H, L_over_R, M=None):
    """K, f as in assemble() plus B: discrete d/dtheta(H r) with first-order upwind
    (flow runs toward +theta), acting on the void fraction r = 1 - theta_film.
    JFO system:  K P - B r = f,  P >= 0,  r >= 0,  P * r = 0."""
    K, f = assemble(H, L_over_R, M)
    n_t, n_z = H.shape
    dt = 2 * np.pi / n_t
    nzi = n_z - 2
    idx = np.arange(n_t * nzi).reshape(n_t, nzi)
    Hi = H[:, 1:-1]
    rows = np.concatenate([idx.ravel(), idx.ravel()])
    cols = np.concatenate([idx.ravel(), np.roll(idx, 1, axis=0).ravel()])
    vals = np.concatenate([Hi.ravel() / dt, -np.roll(Hi, 1, axis=0).ravel() / dt])
    B = sp.csr_matrix((vals, (rows, cols)), shape=K.shape)
    return K, B, f


def solve_jfo(H, L_over_R, groove, Ps, M=None, model="jfo", max_iter=300, cav0=None):
    """Solve with an oil-supply groove held at pressure Ps (needed: without a supply the
    mass-conserving problem on a periodic domain has no unique solution).
    model="jfo": mass-conserving JFO.  model="reynolds": Reynolds (Swift-Stieber) condition.
    Returns P and film fraction theta_f on the full grid, and the active-set iteration count."""
    n_t, n_z = H.shape
    K, B, f = assemble_jfo(H, L_over_R, M)
    if model == "reynolds":
        B = B * 0.0
    n = K.shape[0]
    g = groove.ravel()
    Kc, Bc = K.tocsc(), B.tocsc()
    rows = ~g
    rhs = (f - Kc[:, g] @ np.full(g.sum(), Ps))[rows]
    Kr, Br = Kc[rows], Bc[rows]
    if cav0 is not None:          # warm start: cavitation set from e.g. a neural prediction
        cav = cav0.ravel() & rows
    else:                         # initial guess: everything pressurised
        Pfree = spla.spsolve(Kr[:, rows].tocsc(), rhs)
        P = np.zeros(n); P[rows] = Pfree; P[g] = Ps
        cav = (P < 0) & rows
    for it in range(1, max_iter + 1):
        pres = rows & ~cav
        if model == "jfo":
            A = sp.hstack([Kr[:, pres], -Br[:, cav]]).tocsc()
            u = spla.spsolve(A, rhs)
            P = np.zeros(n); r = np.zeros(n)
            P[pres] = u[:pres.sum()]; r[cav] = u[pres.sum():]
            P[g] = Ps
            new = cav.copy()
            new[pres & (P < -1e-12)] = True
            new[cav & (r < -1e-12)] = False
        else:  # Reynolds condition: obstacle problem, multiplier lam = K P - f >= 0 on cavitated nodes
            sub = pres[rows]
            P = np.zeros(n); P[g] = Ps
            P[pres] = spla.spsolve(Kr[sub][:, pres].tocsc(), rhs[sub])
            lam = K @ P - f
            r = np.zeros(n)
            new = cav.copy()
            new[pres & (P < -1e-12)] = True
            new[cav & (lam < -1e-12)] = False
        if np.array_equal(new, cav):
            break
        cav = new
    Pf = np.zeros((n_t, n_z)); Tf = np.ones((n_t, n_z))
    Pf[:, 1:-1] = np.maximum(P, 0).reshape(n_t, n_z - 2)
    Tf[:, 1:-1] = (1 - np.clip(r, 0, 1)).reshape(n_t, n_z - 2)
    return Pf, Tf, it
