"""PINO / FNO for JFO (mass-conserving) cavitation with an oil-supply groove.

Network outputs pressure P and void fraction r = 1 - theta_f (both >= 0 via softplus).
Hard constraints: P = 0 and r = 0 on the axial edges, P = Ps and r = 0 in the groove.
Physics loss: JFO residual K P - B r - f = 0 on all non-groove interior nodes (Jacobi-scaled),
plus complementarity min(P, r) = 0.

usage: python3 pino_jfo.py <tag> <n_labeled> <pde_weight> <epochs> [n_unlabeled]
env: THREADS, WARMUP (fraction of epochs data-only), STEPS (steps per epoch), UNL_RANGE (train|wide),
     WIDTH (channel width, default 32), DATA_ON_CPU (1: keep data on CPU, move batches), CUT (1: cut-cell labels), GEO (1: groove-edge distance input), HARD (1 default; 0 = ablation without hard constraints: standard 2-D Fourier FNO with zero padding in z,
     no edge/groove imposition, boundary and groove conditions enforced only by a soft penalty BC_W * L_bc)
"""
import os, sys, time, json
import numpy as np
import torch
import torch.nn.functional as F
from pino import FNO2d, TWO_PI

torch.set_num_threads(int(os.environ.get("THREADS", 4)))
HARD = os.environ.get("HARD", "1") != "0"
BC_W = float(os.environ.get("BC_W", 1.0))
CUT = os.environ.get("CUT", "0") in ("1", "2")   # labels from the cut-cell groove solver (solver2.solve_jfo_cut)
CUTRES = os.environ.get("CUT", "0") == "2"       # physics loss uses the exact cut-cell rows next to the groove (cut_residual.py)
CUTW = float(os.environ.get("CUTW", 1.0))         # weight of those rows in the residual (CUT=2)
GEO = os.environ.get("GEO", "0") == "1"   # extra input: signed distance to the exact groove edge (sub-cell position)


def jfo_residual(P, r, H, LR, G, cut=None):
    """P, r, H, G: (B, Nt, Nz); LR: (B,). Returns scaled residual, complementarity, scale.
    cut (CUT=2): (rows, cols, coef, fval, diag) of the cut-cell rows next to the groove for this batch (cut_residual.py);
    those rows are then evaluated with the cut-cell stencil instead of being left out."""
    Nt, Nz = H.shape[1:]
    dt = TWO_PI / Nt
    dz = (LR / (Nz - 1))[:, None, None]
    H3 = H ** 3
    Hte = 0.5 * (H3 + torch.roll(H3, -1, 1)); Htw = torch.roll(Hte, 1, 1)
    Hzn = 0.5 * (H3[:, :, 1:] + H3[:, :, :-1])
    Ft = Hte * (torch.roll(P, -1, 1) - P) / dt
    div_t = (Ft - torch.roll(Ft, 1, 1)) / dt
    Fz = Hzn * (P[:, :, 1:] - P[:, :, :-1]) / dz
    div_z = (Fz[:, :, 1:] - Fz[:, :, :-1]) / dz
    KP = -(div_t[:, :, 1:-1] + div_z)
    Hf = 0.5 * (H + torch.roll(H, -1, 1))
    f = -((Hf - torch.roll(Hf, 1, 1)) / dt)[:, :, 1:-1]
    Hr = H * r
    Br = ((Hr - torch.roll(Hr, 1, 1)) / dt)[:, :, 1:-1]
    diag = ((Hte + Htw)[:, :, 1:-1] / dt ** 2 + (Hzn[:, :, 1:] + Hzn[:, :, :-1]) / dz ** 2)
    if CUT and cut is not None:   # exact cut-cell rows next to the groove
        keep = 1 - G[:, :, 1:-1]
    elif CUT:   # rows next to the groove use the cut-cell stencil in the labels: leave them out of the residual
        Gd = torch.maximum(G, torch.maximum(torch.roll(G, 1, 1), torch.roll(G, -1, 1)))
        Gd = torch.maximum(Gd, torch.maximum(F.pad(G[:, :, 1:], (0, 1)), F.pad(G[:, :, :-1], (1, 0))))
        keep = 1 - Gd[:, :, 1:-1]
    else:
        keep = 1 - G[:, :, 1:-1]
    res = (KP - Br - f) / diag * keep
    if CUT and cut is not None:
        rows, cols, coef, fval, cdiag = cut
        B_, M_ = rows.shape
        Pint = P[:, :, 1:-1].reshape(B_, -1)
        KPc = (coef * Pint.gather(1, cols.reshape(B_, -1)).reshape(B_, M_, -1)).sum(-1)
        resc = (KPc - Br.reshape(B_, -1).gather(1, rows) - fval) / cdiag * CUTW
        res = res.reshape(B_, -1).scatter(1, rows, resc).reshape(res.shape)
    comp = torch.minimum(P, r)[:, :, 1:-1] * keep
    scale = torch.linalg.vector_norm(f / diag, dim=(1, 2))
    return res, comp, scale


def pde_loss(P, r, H, LR, G, cut=None):
    res, comp, scale = jfo_residual(P, r, H, LR, G, cut)
    return ((torch.linalg.vector_norm(res, dim=(1, 2)) + torch.linalg.vector_norm(comp, dim=(1, 2))) / scale).mean()


def features(H, LD, G, Ps, theta, zeta, tg=None):
    T, Z = np.meshgrid(theta, zeta, indexing="ij")
    edge = np.zeros_like(T); edge[:, 0] = edge[:, -1] = 1
    n = len(H)
    c = lambda a: np.broadcast_to(a, (n,) + a.shape)
    chans = [H, np.broadcast_to(LD[:, None, None], H.shape), c(np.cos(T)), c(np.sin(T)), c(Z), c(edge),
             G, np.broadcast_to(np.log(Ps)[:, None, None] / 4, H.shape)]
    if GEO:   # signed circumferential distance to the groove edge in cells, clipped to +-4 and scaled to +-1
        dth = np.abs(np.angle(np.exp(1j * (T[None] - np.asarray(tg)[:, None, None]))))
        sd = np.clip((dth - np.radians(12) / 2) / (theta[1] - theta[0]), -4, 4) / 4
        chans.append(np.broadcast_to(sd, H.shape))
    x = np.stack(chans, 1)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(1 - edge, dtype=torch.float32)


def outputs(model, X, interior):
    """Apply hard constraints (HARD=1) or return the raw softplus outputs (HARD=0). Returns P, r (B, Nt, Nz)."""
    if not HARD:
        o = model(X, torch.ones_like(interior))
        return o[:, 0], o[:, 1]
    o = model(X, interior)
    G = X[:, 6]; Ps = torch.exp(4 * X[:, 7, 0, 0])[:, None, None]
    P = (1 - G) * o[:, 0] + G * Ps * interior
    r = (1 - G) * o[:, 1]
    return P, r


def bc_loss(P, r, X):
    """Soft boundary/groove conditions (HARD=0 only): P = r = 0 on the axial edges, P = Ps and r = 0 in the groove."""
    G = X[:, 6]; Ps = torch.exp(4 * X[:, 7, 0, 0])[:, None, None]
    edge = torch.cat([P[:, :, [0, -1]], r[:, :, [0, -1]]], -1)
    g = (G * ((P - Ps) ** 2 + r ** 2)).sum((1, 2)) / G.sum((1, 2)).clamp_min(1)
    return torch.sqrt((edge ** 2).mean((1, 2)) + g).mean()


class OnDev:
    """CPU-resident (pinned) tensor whose indexed slices are returned on the GPU (env DATA_ON_CPU=1, for large grids)."""
    def __init__(s, t, dev):
        s.t, s.dev = t.pin_memory(), dev

    def __getitem__(s, idx):
        return s.t[torch.as_tensor(idx)].to(s.dev, non_blocking=True)


def rel_l2(a, b):
    return (torch.linalg.vector_norm(a - b, dim=(-2, -1)) / torch.linalg.vector_norm(b, dim=(-2, -1)).clamp_min(1e-8))


def rms(a, b):
    return torch.sqrt(((a - b) ** 2).mean(dim=(-2, -1)))


if __name__ == "__main__":
    tag, n_lab, w_pde, epochs = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    n_unl = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    seed = int(os.environ.get("SEED", 0))
    torch.manual_seed(seed)
    dev = torch.device(os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[{tag}] device {dev}, seed {seed}", flush=True)
    d = np.load(os.environ.get("DATA", "dataset_jfo.npz"))
    H, LD, G, Ps, P, Tf = d["H"], d["LD"], d["G"], d["Ps"], d["P"], d["Tf"]
    theta, zeta = d["theta"], d["zeta"]
    idx = np.random.default_rng(0).permutation(len(H))
    tr, va, te = idx[:n_lab], idx[8000:9000], idx[9000:]
    X, interior = features(H, LD, G, Ps, theta, zeta, np.radians(d["params"][:, 4]))
    on_cpu = os.environ.get("DATA_ON_CPU") == "1"
    put = (lambda a: OnDev(torch.as_tensor(a), dev)) if on_cpu else (lambda a: torch.as_tensor(a).to(dev))
    X, interior = put(X), interior.to(dev)
    Ht, LRt, Gt = put(H), put(2 * LD), put(G)
    Yp, Yr = put(P), put(1 - Tf)
    CL = lambda ids: None
    if CUTRES:
        from cut_residual import cut_rows_batch
        cr = cut_rows_batch(H, LD, np.radians(d["params"][:, 4]), theta, zeta,
                            cache=os.path.splitext(os.environ.get("DATA", "dataset_jfo.npz"))[0] + "_cutrows.npz")
        crt = [put(a) for a in cr]
        CL = lambda ids: tuple(a[ids] for a in crt)
    if n_unl:   # unlabeled films with random groove position and supply pressure (no solve)
        from make_dataset import make_film
        from solver import groove_mask
        rng_u = np.random.default_rng(42)
        TGu = []
        Hu, LDu, Gu, Psu = [], [], [], []
        wide = os.environ.get("UNL_RANGE") == "wide"   # unlabeled films also from out-of-range sets
        if wide:
            from make_dataset_jfo_ood import unlabeled_wide
        for k in range(n_unl):
            if wide:
                h, ld, g, ps, tgk = unlabeled_wide(k, with_tg=True)
                Hu.append(h); LDu.append(ld); Gu.append(g); Psu.append(ps); TGu.append(tgk)
                continue
            if len(theta) == 64:
                h, ld, *_ = make_film(10 ** 6 + k)
            else:                                 # same draws on a finer grid
                from make_dataset import make_film_grid
                h, ld = make_film_grid(10 ** 6 + k, theta, zeta)
            tgk = np.radians(rng_u.uniform(-30, 30)); TGu.append(tgk)
            g = np.zeros_like(h); g[:, 1:-1] = groove_mask(theta, zeta, tgk)
            Hu.append(h); LDu.append(ld); Gu.append(g); Psu.append(np.exp(rng_u.uniform(np.log(0.002), np.log(0.05))))
        Hu, LDu, Gu, Psu = [np.array(a, np.float32) for a in (Hu, LDu, Gu, Psu)]
        Xu, _ = features(Hu, LDu, Gu, Psu, theta, zeta, np.array(TGu) if TGu else None)
        Xu = put(Xu)
        Hut, LRu, Gut = put(Hu), put(2 * LDu), put(Gu)
        CU = lambda ids: None
        if CUTRES:
            cu = cut_rows_batch(Hu, LDu, np.array(TGu), theta, zeta,
                                cache=f"unl_cutrows_{'wide' if wide else 'train'}_{n_unl}_{len(theta)}.npz")
            cut_ = [put(a) for a in cu]
            CU = lambda ids: tuple(a[ids] for a in cut_)

    model = FNO2d(cin=9 if GEO else 8, width=int(os.environ.get("WIDTH", 32)), basis="fs" if HARD else "fourier", cout=2, nz=H.shape[2]).to(dev)
    lr = float(os.environ.get("LR", 2e-3))
    bs = 32
    spe = int(os.environ.get("STEPS", 0)) or max(len(tr) // bs, 250 if n_unl else 1)
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=epochs * spe)
    warm = int(float(os.environ.get("WARMUP", 0)) * epochs)
    rng = np.random.default_rng(1 + seed)

    def predict(ids):
        model.eval()
        with torch.no_grad():
            outs = [outputs(model, X[ids[i:i + 250]], interior) for i in range(0, len(ids), 250)]
        model.train()
        return torch.cat([o[0] for o in outs]), torch.cat([o[1] for o in outs])

    t0 = time.time()
    for ep in range(epochs):
        ramp = 1.0 if warm == 0 else min(1.0, max(0.0, (ep - warm) / warm))
        tot = np.zeros(3)
        for _ in range(spe):
            b = rng.choice(tr, bs, replace=len(tr) < bs)
            p, r = outputs(model, X[b], interior)
            l_data = rel_l2(p, Yp[b]).mean() + rms(r, Yr[b]).mean()
            l_pde = pde_loss(p, r, Ht[b], LRt[b], Gt[b], CL(b)) if w_pde else torch.zeros(())
            if n_unl and w_pde:
                u = rng.integers(0, n_unl, bs)
                pu, ru = outputs(model, Xu[u], interior)
                l_pde = l_pde + pde_loss(pu, ru, Hut[u], LRu[u], Gut[u], CU(u))
            loss = l_data + w_pde * ramp * l_pde
            if not HARD:
                loss = loss + BC_W * bc_loss(p, r, X[b]) + (BC_W * bc_loss(pu, ru, Xu[u]) if n_unl and w_pde else 0)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += [loss.item(), l_data.item(), l_pde.item()]
        if ep % 5 == 4 or ep == epochs - 1:
            pv, rv = predict(va)
            print(f"[{tag}] ep {ep+1:3d} data {tot[1]/spe:.4f} pde {tot[2]/spe:.4f} | val P relL2 {rel_l2(pv, Yp[va]).mean():.4f} "
                  f"film-frac RMS {rms(rv, Yr[va]).mean():.4f} residual {pde_loss(pv, rv, Ht[va], LRt[va], Gt[va], CL(va)):.4f}  "
                  f"{time.time()-t0:.0f}s", flush=True)

    torch.save(model.state_dict(), f"{tag}.pt")
    pt, rt = predict(te)
    np.save(f"{tag}_test_pred.npy", np.stack([pt.cpu().numpy(), 1 - rt.cpu().numpy()], 1))
    rel = rel_l2(pt, Yp[te]).cpu().numpy()
    prm = d["params"][te]
    res = {"tag": tag, "seed": seed, "lr": lr, "warmup": os.environ.get("WARMUP", "0"), "n_labeled": n_lab, "n_unlabeled": n_unl, "pde_weight": w_pde,
           "unl_range": os.environ.get("UNL_RANGE", "train"), "hard": int(HARD), "cut": 2 if CUTRES else int(CUT), "geo": int(GEO),
           "test_P_relL2_mean": float(rel.mean()), "test_P_relL2_median": float(np.median(rel)),
           "test_P_relL2_eps_gt_0.8": float(rel[prm[:, 0] > 0.8].mean()),
           "test_P_relL2_textured": float(rel[prm[:, 3] > 0].mean()),
           "test_filmfrac_rms": float(rms(rt, Yr[te]).mean()),
           "test_pde_residual": pde_loss(pt, rt, Ht[te], LRt[te], Gt[te]).item(),
           "test_pde_residual_cutrows": pde_loss(pt, rt, Ht[te], LRt[te], Gt[te], CL(te)).item() if CUTRES else None,
           "train_seconds": time.time() - t0}
    json.dump(res, open(f"{tag}_results.json", "w"), indent=1); print(res)
