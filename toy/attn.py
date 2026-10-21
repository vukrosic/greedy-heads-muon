"""Whole vs per-head (q,k, always): every 10 steps, best head's attention to offset -4 in layer 1 / layer 2 and how
peaked attention is (mean of the largest attention weight per query, 1/t = uniform)."""
import os, torch, torch.nn.functional as F
os.environ.setdefault("QKNORM", "1")
from run import Model, batch, msign, H
xb, yb, mb, _ = batch(128, torch.Generator().manual_seed(7), "back")
ph = lambda u: torch.cat([msign(c) for c in u.chunk(H, 0)], 0)
for seed in (0, 1):
    for r in ("whole", "perhead"):
        torch.manual_seed(seed); m = Model()
        names = [(l, n) for l in (0, 1) for n in "qkvo"]; mats = [getattr(m.layers[l], n).weight for l, n in names]; ids = {id(p) for p in mats}
        adam = torch.optim.AdamW([p for p in m.parameters() if id(p) not in ids], lr=0.01, betas=(0.9, 0.98), weight_decay=0)
        buf = [torch.zeros_like(p) for p in mats]; gen = torch.Generator().manual_seed(seed); out = []
        for i in range(121):
            if i % 20 == 0:
                keep = []
                with torch.no_grad(): L = F.cross_entropy(m(xb, keep)[mb], yb[mb]).item()
                lb = [torch.diagonal(a, offset=-4, dim1=-2, dim2=-1).mean((0, 2)).max().item() for a in keep]
                pk = [a.max(-1).values.mean().item() for a in keep]
                out.append(f"{i:3}: L{L:.2f} back {lb[0]:.2f}/{lb[1]:.2f} peak {pk[0]:.2f}/{pk[1]:.2f}")
            x, y, msk, _ = batch(32, gen, "back"); loss = F.cross_entropy(m(x)[msk], y[msk])
            adam.zero_grad(); [setattr(p, "grad", None) for p in mats]; loss.backward(); adam.step()
            with torch.no_grad():
                for p, b, (l, n) in zip(mats, buf, names):
                    b.mul_(0.95).add_(p.grad); u = p.grad + 0.95 * b
                    p.add_(ph(u) if n in "qk" and r == "perhead" else msign(u), alpha=-0.03)
        print(f"seed {seed} {r:7} | " + " | ".join(out[1:]), flush=True)
