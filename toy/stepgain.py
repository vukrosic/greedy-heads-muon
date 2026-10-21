"""Theory test: per-head overspends on shared directions, good early, bad late. Along a whole-Muon run, every 10 steps,
take the SAME weights and momentum, apply one q,k step with each rule (lr 0.03), measure big-batch loss drop.
Prediction if the theory holds: per-head drops more early, less late (from about the crossover, step 70-90)."""
import os, copy, torch, torch.nn.functional as F
os.environ.setdefault("QKNORM", "1")
from run import Model, batch, msign, H
xb, yb, mb, _ = batch(512, torch.Generator().manual_seed(7), "back")
ph = lambda u: torch.cat([msign(c) for c in u.chunk(H, 0)], 0)
ev = lambda m: F.cross_entropy(m(xb)[mb], yb[mb]).item()
for seed in (0, 1):
    torch.manual_seed(seed); m = Model()
    names = [(l, n) for l in (0, 1) for n in "qkvo"]; mats = [getattr(m.layers[l], n).weight for l, n in names]; ids = {id(p) for p in mats}
    adam = torch.optim.AdamW([p for p in m.parameters() if id(p) not in ids], lr=0.01, betas=(0.9, 0.98), weight_decay=0)
    buf = [torch.zeros_like(p) for p in mats]; gen = torch.Generator().manual_seed(seed); out = []
    for i in range(141):
        x, y, msk, _ = batch(32, gen, "back"); loss = F.cross_entropy(m(x)[msk], y[msk])
        adam.zero_grad(); [setattr(p, "grad", None) for p in mats]; loss.backward(); adam.step()
        with torch.no_grad():
            us = []
            for p, b in zip(mats, buf): b.mul_(0.95).add_(p.grad); us.append(p.grad + 0.95 * b)
            if i % 10 == 0 and i:
                base = ev(m); d = {}
                for r, f in (("whole", msign), ("perhead", ph)):
                    c = copy.deepcopy(m); cm = [getattr(c.layers[l], n).weight for l, n in names]
                    for p, u, (l, n) in zip(cm, us, names):
                        if n in "qk": p.add_(f(u), alpha=-0.03)
                    d[r] = base - ev(c)
                out.append(f"{i}: L{base:.2f} ph-whole {1e3 * (d['perhead'] - d['whole']):+.1f}")
            for p, u in zip(mats, us): p.add_(msign(u), alpha=-0.03)
    print(f"s{seed} (extra loss drop from per-head, x1000) | " + " | ".join(out), flush=True)
