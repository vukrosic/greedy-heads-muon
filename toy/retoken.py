"""Does the token pile-up cause the delay? Steps < 30: change how much of the layer-1 q,k step lands on token-identity
input directions (top-8 of centered token embeddings), keep the step size. Metric: first step with loss < 0.5, 4 seeds."""
import os, time, torch, torch.nn.functional as F
os.environ.setdefault("QKNORM", "1")
from run import Model, batch, msign, H
xb, yb, mb, _ = batch(256, torch.Generator().manual_seed(7), "back")
ph = lambda u: torch.cat([msign(c) for c in u.chunk(H, 0)], 0)
basis = lambda X: torch.linalg.svd(X - X.mean(0), full_matrices=False)[2][:8]


def share(st, B, target):  # set the share of st's energy inside span(B) to target, same total size
    a = st @ B.T @ B; r = st - a; e, ea = st.pow(2).sum(), a.pow(2).sum()
    return a * (target * e / ea).sqrt() + r * ((1 - target) * e / r.pow(2).sum()).sqrt()


def run(seed, rule, target=None, sub="tok", S=30):
    torch.manual_seed(seed); m = Model(); rnd = torch.linalg.qr(torch.randn(64, 8, generator=torch.Generator().manual_seed(99)))[0].T
    names = [(l, n) for l in (0, 1) for n in "qkvo"]; mats = [getattr(m.layers[l], n).weight for l, n in names]; ids = {id(p) for p in mats}
    adam = torch.optim.AdamW([p for p in m.parameters() if id(p) not in ids], lr=0.01, betas=(0.9, 0.98), weight_decay=0)
    buf = [torch.zeros_like(p) for p in mats]; gen = torch.Generator().manual_seed(seed)
    for i in range(200):
        if i % 5 == 0 and i >= 90:
            with torch.no_grad():
                if F.cross_entropy(m(xb)[mb], yb[mb]).item() < 0.5: return i
        x, y, msk, _ = batch(32, gen, "back"); loss = F.cross_entropy(m(x)[msk], y[msk])
        adam.zero_grad(); [setattr(p, "grad", None) for p in mats]; loss.backward(); adam.step()
        with torch.no_grad():
            B = {"tok": basis(m.emb.weight), "pos": basis(m.pos), "rnd": rnd}[sub]
            for p, b, (l, n) in zip(mats, buf, names):
                b.mul_(0.95).add_(p.grad); u = p.grad + 0.95 * b
                st = ph(u) if (n in "qk" and i < S and rule == "perhead") else msign(u)
                if n in "qk" and i < S and l == 0 and target is not None: st = share(st, B, target)
                p.add_(st, alpha=-0.03)
    return 999


R = {"per-head 30, token share cut to even": dict(rule="perhead", target=0.125),
     "whole 30, token share raised to per-head's": dict(rule="whole", target=0.28),
     "per-head 30, random dirs cut to even (control)": dict(rule="perhead", target=0.125, sub="rnd")}
t0 = time.time()
for k, kw in R.items():
    h = [run(s, **kw) for s in range(4)]; print(f"{time.time() - t0:4.0f}s {k:48} {h} mean {sum(h) / 4:.1f}", flush=True)
