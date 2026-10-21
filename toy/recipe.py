"""Recipes mixing per-head and whole Muon on q,k in the toy (task back, QK-norm). Metric: first step where big-batch
loss < 0.5 (lower = faster), checked every 5 steps, and loss at step 100. 4 seeds."""
import os, sys, torch, torch.nn.functional as F
os.environ.setdefault("QKNORM", "1")
from run import Model, batch, msign, H
xb, yb, mb, _ = batch(256, torch.Generator().manual_seed(7), "back")
ph = lambda u: torch.cat([msign(c) for c in u.chunk(H, 0)], 0)
grp = lambda u, g: torch.cat([msign(c) for c in u.chunk(H // g, 0)], 0)


def blend(u, a):  # a = share of per-head; same total size as either step
    s = a * ph(u) + (1 - a) * msign(u); return s * (u.shape[0] ** 0.5 / s.norm())


RECIPES = {
    "whole": lambda i, u: msign(u),
    "perhead": lambda i, u: ph(u),
    "pairs (2 heads)": lambda i, u: grp(u, 2),
    "switch@30": lambda i, u: ph(u) if i < 30 else msign(u),
    "switch@50": lambda i, u: ph(u) if i < 50 else msign(u),
    "switch@70": lambda i, u: ph(u) if i < 70 else msign(u),
    "1 in 5 whole": lambda i, u: msign(u) if i % 5 == 0 else ph(u),
    "1 in 2 whole": lambda i, u: msign(u) if i % 2 == 0 else ph(u),
    "blend 50/50": lambda i, u: blend(u, 0.5),
    "blend fade 1->0 by 70": lambda i, u: blend(u, max(0.0, 1 - i / 70)),
}


def run(r, seed):
    torch.manual_seed(seed); m = Model()
    names = [(l, n) for l in (0, 1) for n in "qkvo"]; mats = [getattr(m.layers[l], n).weight for l, n in names]; ids = {id(p) for p in mats}
    adam = torch.optim.AdamW([p for p in m.parameters() if id(p) not in ids], lr=0.01, betas=(0.9, 0.98), weight_decay=0)
    buf = [torch.zeros_like(p) for p in mats]; gen = torch.Generator().manual_seed(seed); hit, l100 = None, None
    for i in range(201):
        if i % 5 == 0:
            with torch.no_grad(): L = F.cross_entropy(m(xb)[mb], yb[mb]).item()
            if i == 100: l100 = L
            if L < 0.5 and hit is None: hit = i
            if hit is not None and i >= 100: break
        x, y, msk, _ = batch(32, gen, "back"); loss = F.cross_entropy(m(x)[msk], y[msk])
        adam.zero_grad(); [setattr(p, "grad", None) for p in mats]; loss.backward(); adam.step()
        with torch.no_grad():
            for p, b, (l, n) in zip(mats, buf, names):
                b.mul_(0.95).add_(p.grad); u = p.grad + 0.95 * b
                p.add_(RECIPES[r](i, u) if n in "qk" else msign(u), alpha=-0.03)
    return hit, l100


import time; t0 = time.time()
for r in RECIPES:
    if r in ("whole", "perhead") and len(sys.argv) > 1: continue
    res = [run(r, s) for s in range(4)]
    hits = [h if h is not None else 999 for h, _ in res]
    print(f"{time.time() - t0:5.0f}s {r:24} solved at {hits}  mean {sum(hits) / 4:6.1f}   loss@100 {' '.join(f'{l:.2f}' for _, l in res)}", flush=True)
