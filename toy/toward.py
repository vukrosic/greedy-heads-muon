"""Do whole steps move q,k toward the final solution faster? Target D = q,k change of a finished whole run (150 steps, same
seed). Rerun whole and per-head; every 20 steps: progress = <W_t - W0, D> / |D|^2 (1 = reached the solution), averaged over
the 4 q,k matrices."""
import os, torch, torch.nn.functional as F
os.environ.setdefault("QKNORM", "1")
from run import Model, batch, msign, H
ph = lambda u: torch.cat([msign(c) for c in u.chunk(H, 0)], 0)


def train(r, seed, D=None, steps=150):
    torch.manual_seed(seed); m = Model()
    names = [(l, n) for l in (0, 1) for n in "qkvo"]; mats = [getattr(m.layers[l], n).weight for l, n in names]; ids = {id(p) for p in mats}
    qk = [j for j, (l, n) in enumerate(names) if n in "qk"]; W0 = [mats[j].detach().clone() for j in qk]
    adam = torch.optim.AdamW([p for p in m.parameters() if id(p) not in ids], lr=0.01, betas=(0.9, 0.98), weight_decay=0)
    buf = [torch.zeros_like(p) for p in mats]; gen = torch.Generator().manual_seed(seed); out = []
    for i in range(steps + 1):
        if D is not None and i % 20 == 0:
            out.append(sum(((mats[j] - w0) * d).sum().item() / d.norm().item() ** 2 for j, w0, d in zip(qk, W0, D)) / 4)
        x, y, msk, _ = batch(32, gen, "back"); loss = F.cross_entropy(m(x)[msk], y[msk])
        adam.zero_grad(); [setattr(p, "grad", None) for p in mats]; loss.backward(); adam.step()
        with torch.no_grad():
            for p, b, (l, n) in zip(mats, buf, names):
                b.mul_(0.95).add_(p.grad); u = p.grad + 0.95 * b
                p.add_(ph(u) if n in "qk" and r == "perhead" else msign(u), alpha=-0.03)
    return [mats[j].detach() - w0 for j, w0 in zip(qk, W0)] if D is None else out


print("step     " + " ".join(f"{s:5}" for s in range(0, 151, 20)))
for seed in (0, 1):
    D = train("perhead", seed, steps=200)  # per-head's OWN finished solution as the target
    for r in ("whole", "perhead"): print(f"s{seed} {r:7} " + " ".join(f"{v:5.2f}" for v in train(r, seed, D)), flush=True)
