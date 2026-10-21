"""Why do 30 early per-head steps delay the toy's takeoff? Each recipe: per-head on some q,k for steps < S, whole after,
optionally edit q,k at step S. Metric: first step with loss < 0.5 (eval 256, every 5 steps), 4 seeds. ~2.5 s per run."""
import os, sys, time, torch, torch.nn.functional as F
os.environ.setdefault("QKNORM", "1")
from run import Model, batch, msign, H
xb, yb, mb, _ = batch(256, torch.Generator().manual_seed(7), "back")
ph = lambda u: torch.cat([msign(c) for c in u.chunk(H, 0)], 0)


def run(seed, S=0, layers=(0, 1), edit=None, every=0):
    torch.manual_seed(seed); m = Model()
    names = [(l, n) for l in (0, 1) for n in "qkvo"]; mats = [getattr(m.layers[l], n).weight for l, n in names]; ids = {id(p) for p in mats}
    W0 = [p.detach().clone() for p in mats]
    adam = torch.optim.AdamW([p for p in m.parameters() if id(p) not in ids], lr=0.01, betas=(0.9, 0.98), weight_decay=0)
    buf = [torch.zeros_like(p) for p in mats]; gen = torch.Generator().manual_seed(seed)
    for i in range(200):
        if edit and (i == S or (every and i % every == 0 and 0 < i < S)):
            with torch.no_grad():
                for j, (l, n) in enumerate(names):
                    if n in "qk": mats[j].copy_(edit(mats[j], W0[j], buf[j]))
        if i % 5 == 0 and i >= 90:
            with torch.no_grad():
                if F.cross_entropy(m(xb)[mb], yb[mb]).item() < 0.5: return i
        x, y, msk, _ = batch(32, gen, "back"); loss = F.cross_entropy(m(x)[msk], y[msk])
        adam.zero_grad(); [setattr(p, "grad", None) for p in mats]; loss.backward(); adam.step()
        with torch.no_grad():
            for p, b, (l, n) in zip(mats, buf, names):
                b.mul_(0.95).add_(p.grad); u = p.grad + 0.95 * b
                p.add_(ph(u) if (n in "qk" and i < S and l in layers) else msign(u), alpha=-0.03)
    return 999


def flat(W, W0, b):  # keep the q,k change since start, but make all its directions equal size
    d = W - W0; return W0 + msign(d) * (d.norm() / d.shape[0] ** 0.5)


R = {"per-head first 50, then flatten": dict(S=50, edit=flat), "per-head first 70, then flatten": dict(S=70, edit=flat),
     "per-head always, flatten every 10": dict(S=200, edit=flat, every=10), "per-head always, flatten every 5": dict(S=200, edit=flat, every=5)}
t0 = time.time()
for k, kw in R.items():
    h = [run(s, **kw) for s in range(4)]; print(f"{time.time() - t0:4.0f}s {k:36} {h} mean {sum(h) / 4:.1f}", flush=True)
