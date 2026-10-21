"""How much do heads' q,k changes share input directions (0 = separate, 1 = identical)? whole vs per-head, every 10 steps."""
import os, torch, torch.nn.functional as F
os.environ.setdefault("QKNORM", "1")
from run import Model, batch, msign, H
from twins import overlap, ph
for seed in (0, 1):
    for r in ("whole", "perhead"):
        torch.manual_seed(seed); m = Model()
        names = [(l, n) for l in (0, 1) for n in "qkvo"]; mats = [getattr(m.layers[l], n).weight for l, n in names]; ids = {id(p) for p in mats}
        W0 = [p.detach().clone() for p in mats]
        adam = torch.optim.AdamW([p for p in m.parameters() if id(p) not in ids], lr=0.01, betas=(0.9, 0.98), weight_decay=0)
        buf = [torch.zeros_like(p) for p in mats]; gen = torch.Generator().manual_seed(seed); out = []
        for i in range(1, 101):
            x, y, msk, _ = batch(32, gen, "back"); loss = F.cross_entropy(m(x)[msk], y[msk])
            adam.zero_grad(); [setattr(p, "grad", None) for p in mats]; loss.backward(); adam.step()
            with torch.no_grad():
                for p, b, (l, n) in zip(mats, buf, names):
                    b.mul_(0.95).add_(p.grad); u = p.grad + 0.95 * b
                    p.add_(ph(u) if n in "qk" and r == "perhead" else msign(u), alpha=-0.03)
            if i in (10, 30, 50, 100): out.append(f"{i:3}: " + " ".join(f"{overlap(mats[j], W0[j]):.2f}" for j, (l, n) in enumerate(names) if n in "qk"))
        print(f"s{seed} {r:7} (L1q L1k L2q L2k) | " + " | ".join(out), flush=True)
