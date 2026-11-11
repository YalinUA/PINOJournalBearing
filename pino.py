"""FNO / PINO for the journal-bearing Reynolds equation.

Model: FNO-2D (lift -> 4 Fourier layers -> project), periodic in theta, z zero-padded.
Physics loss: discrete Reynolds operator identical to solver.assemble, written in torch.
Cavitation (Reynolds condition) as complementarity: r = min(P, (K P - f) / diag) = 0.

usage: python3 pino.py <tag> <n_labeled> <pde_weight> <epochs> [n_unlabeled]
"""
import sys, time, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(int(__import__("os").environ.get("THREADS", 4)))
TWO_PI = 2 * np.pi


# ---------- physics ----------
def reynolds_residual(P, H, LR, M=None):
    """P, H, M: (B, Nt, Nz); LR: (B,). Returns complementarity residual on interior nodes
    (B, Nt, Nz-2) in units of P, and the scale ||f/diag|| per sample. M = viscosity ratio field."""
    Nt, Nz = H.shape[1:]
    dt = TWO_PI / Nt
    dz = (LR / (Nz - 1))[:, None, None]
    H3 = H ** 3 if M is None else H ** 3 / M
    Hte = 0.5 * (H3 + torch.roll(H3, -1, 1))
    Htw = torch.roll(Hte, 1, 1)
    Hzn = 0.5 * (H3[:, :, 1:] + H3[:, :, :-1])
    Ft = Hte * (torch.roll(P, -1, 1) - P) / dt
    div_t = (Ft - torch.roll(Ft, 1, 1)) / dt
    Fz = Hzn * (P[:, :, 1:] - P[:, :, :-1]) / dz
    div_z = (Fz[:, :, 1:] - Fz[:, :, :-1]) / dz
    KP = -(div_t[:, :, 1:-1] + div_z)
    Hf = 0.5 * (H + torch.roll(H, -1, 1))
    f = -((Hf - torch.roll(Hf, 1, 1)) / dt)[:, :, 1:-1]
    diag = ((Hte + Htw)[:, :, 1:-1] / dt ** 2 + (Hzn[:, :, 1:] + Hzn[:, :, :-1]) / dz ** 2)
    r = torch.minimum(P[:, :, 1:-1], (KP - f) / diag)
    scale = torch.linalg.vector_norm(f / diag, dim=(1, 2))
    return r, scale


def pde_loss(P, H, LR, M=None):
    r, scale = reynolds_residual(P, H, LR, M)
    return (torch.linalg.vector_norm(r, dim=(1, 2)) / scale).mean()


# ---------- model ----------
class SpectralConv2d(nn.Module):
    def __init__(s, cin, cout, m1, m2):
        super().__init__()
        s.m1, s.m2 = m1, m2
        sc = 1 / (cin * cout)
        s.w1 = nn.Parameter(sc * torch.randn(cin, cout, m1, m2, dtype=torch.cfloat))
        s.w2 = nn.Parameter(sc * torch.randn(cin, cout, m1, m2, dtype=torch.cfloat))

    def forward(s, x):
        B, C, A, Bn = x.shape
        xf = torch.fft.rfft2(x)
        out = torch.zeros(B, s.w1.shape[1], A, Bn // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :s.m1, :s.m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :s.m1, :s.m2], s.w1)
        out[:, :, -s.m1:, :s.m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -s.m1:, :s.m2], s.w2)
        return torch.fft.irfft2(out, s=(A, Bn))


class FourierSineConv2d(nn.Module):
    """Fourier basis along theta (periodic) x sine basis along z (P = 0 at both edges).
    The sine transform acts on interior z nodes 1..Nz-2 and returns zeros on the edges."""
    def __init__(s, cin, cout, m1, m2, nz=64):
        super().__init__()
        s.m1 = m1
        n = nz - 2
        k, j = np.meshgrid(np.arange(1, m2 + 1), np.arange(1, n + 1), indexing="ij")
        S = np.sin(np.pi * k * j / (n + 1))
        s.register_buffer("S", torch.tensor(S, dtype=torch.float32))              # (m2, n)
        s.register_buffer("Si", torch.tensor(S.T * 2 / (n + 1), dtype=torch.float32))
        sc = 1 / (cin * cout)
        s.w = nn.Parameter(sc * torch.randn(cin, cout, m1, m2, dtype=torch.cfloat))

    def forward(s, x):
        B, C, A, Nz = x.shape
        xs = x[..., 1:-1] @ s.S.T                       # (B, C, A, m2) sine coefficients
        xf = torch.fft.rfft(xs, dim=2)[:, :, :s.m1]     # Fourier along theta
        y = torch.einsum("bixy,ioxy->boxy", xf, s.w)
        y = torch.fft.irfft(y, n=A, dim=2)              # (B, cout, A, m2)
        return F.pad(y @ s.Si.T, (1, 1))


class FNO2d(nn.Module):
    def __init__(s, cin=6, width=32, modes=16, layers=4, zpad=8, basis="fourier", cout=1, nz=64):
        super().__init__()
        s.zpad = zpad if basis == "fourier" else 0
        s.lift = nn.Conv2d(cin, width, 1)
        Spec = SpectralConv2d if basis == "fourier" else FourierSineConv2d
        s.spec = nn.ModuleList([Spec(width, width, modes, modes) if basis == "fourier" else Spec(width, width, modes, modes, nz)
                                for _ in range(layers)])
        s.loc = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        s.proj = nn.Sequential(nn.Conv2d(width, 128, 1), nn.GELU(), nn.Conv2d(128, cout, 1))

    def forward(s, x, interior):
        x = s.lift(x)
        if s.zpad:
            x = F.pad(x, (0, s.zpad))  # z is not periodic
        for i, (sp_, lc) in enumerate(zip(s.spec, s.loc)):
            x = sp_(x) + lc(x)
            if i < len(s.spec) - 1:
                x = F.gelu(x)
        if s.zpad:
            x = x[..., :-s.zpad]
        return F.softplus(s.proj(x), beta=20) * interior


# ---------- data ----------
def build_features(H, LD, theta, zeta, M=None):
    T, Z = np.meshgrid(theta, zeta, indexing="ij")
    edge = np.zeros_like(T); edge[:, 0] = edge[:, -1] = 1
    n = len(H)
    c = lambda a: np.broadcast_to(a, (n,) + a.shape)
    chans = [H, np.broadcast_to(LD[:, None, None], H.shape), c(np.cos(T)), c(np.sin(T)), c(Z), c(edge)]
    if M is not None:
        chans.append(np.log(M))       # viscosity ratio enters as log(mu/mu_inlet)
    x = np.stack(chans, 1)
    return torch.tensor(x, dtype=torch.float32), torch.tensor(1 - edge, dtype=torch.float32)


def rel_l2(a, b):
    return (torch.linalg.vector_norm(a - b, dim=(-2, -1)) /
            torch.linalg.vector_norm(b, dim=(-2, -1)).clamp_min(1e-8))


if __name__ == "__main__":
    tag, n_lab, w_pde, epochs = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
    n_unl = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    torch.manual_seed(0)
    visc = __import__("os").environ.get("VISC") == "1"
    d = np.load("dataset_visc.npz" if visc else "dataset.npz")
    H, LD, P, theta, zeta = d["H"], d["LD"], d["P"], d["theta"], d["zeta"]
    Mv = d["M"] if visc else None
    Mt = torch.tensor(Mv) if visc else None
    idx = np.random.default_rng(0).permutation(len(H))      # same split as train_unet.py
    tr, va, te = idx[:n_lab], idx[8000:9000], idx[9000:]
    X, interior = build_features(H, LD, theta, zeta, Mv)
    Ht, LRt, Y = torch.tensor(H), torch.tensor(2 * LD), torch.tensor(P)
    if n_unl:
        from make_dataset import make_film
        films = [make_film(10 ** 6 + k) for k in range(n_unl)]
        Hu = np.stack([f[0] for f in films]).astype(np.float32)
        LDu = np.array([f[1] for f in films], np.float32)
        Xu, _ = build_features(Hu, LDu, theta, zeta)
        Hut, LRu = torch.tensor(Hu), torch.tensor(2 * LDu)

    basis = __import__("os").environ.get("BASIS", "fourier")
    model = FNO2d(cin=X.shape[1], basis=basis)
    bs = 32
    spe = int(__import__("os").environ.get("STEPS", 0)) or max(len(tr) // bs, 250 if n_unl else 1)   # steps per epoch
    opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * spe)
    rng = np.random.default_rng(1)

    def predict(ids):
        model.eval()
        with torch.no_grad():
            out = torch.cat([model(X[ids[i:i + 250]], interior)[:, 0] for i in range(0, len(ids), 250)])
        model.train(); return out

    warm = int(float(__import__("os").environ.get("WARMUP", 0)) * epochs)
    t0 = time.time()
    for ep in range(epochs):
        tot = np.zeros(3)
        for _ in range(spe):
            b = rng.choice(tr, bs, replace=len(tr) < bs)
            pred = model(X[b], interior)[:, 0]
            l_data = rel_l2(pred, Y[b]).mean()
            l_pde = pde_loss(pred, Ht[b], LRt[b], Mt[b] if visc else None) if w_pde else torch.zeros(())
            if n_unl:
                u = rng.integers(0, n_unl, bs)
                l_pde = l_pde + pde_loss(model(Xu[u], interior)[:, 0], Hut[u], LRu[u])
            ramp = 1.0 if warm == 0 else min(1.0, max(0.0, (ep - warm) / warm))  # data-only warm-up, then ramp in PDE
            loss = l_data + w_pde * ramp * l_pde
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            tot += [loss.item(), l_data.item(), l_pde.item()]
        if ep % 5 == 4 or ep == epochs - 1:
            pv = predict(va)
            v = rel_l2(pv, Y[va]).mean().item()
            vr = pde_loss(pv, Ht[va], LRt[va], Mt[va] if visc else None).item()
            tot /= spe
            print(f"[{tag}] ep {ep+1:3d} data {tot[1]:.4f} pde {tot[2]:.4f} | val relL2 {v:.4f} "
                  f"val residual {vr:.4f}  {time.time()-t0:.0f}s", flush=True)

    torch.save(model.state_dict(), f"{tag}.pt")
    pt = predict(te)
    np.save(f"{tag}_test_pred.npy", pt.numpy())
    rel = rel_l2(pt, Y[te]).numpy()
    prm = d["params"][te]
    hi = prm[:, 0] > 0.8
    res = {"tag": tag, "basis": basis, "n_labeled": n_lab, "n_unlabeled": n_unl, "pde_weight": w_pde,
           "test_relL2_mean": float(rel.mean()), "test_relL2_median": float(np.median(rel)),
           "test_relL2_p95": float(np.percentile(rel, 95)),
           "test_relL2_eps_gt_0.8": float(rel[hi].mean()),
           "test_pde_residual": pde_loss(pt, Ht[te], LRt[te], Mt[te] if visc else None).item(),
           "train_seconds": time.time() - t0}
    t1 = time.time(); predict(te); res["infer_ms_per_case"] = (time.time() - t1) / len(te) * 1e3
    json.dump(res, open(f"{tag}_results.json", "w"), indent=1); print(res)
