"""Does per-head win early and lose later in the toy? Task back, QK-norm; big-batch loss every 10 steps for q,k rule
whole vs perhead (same seed, same data), printed as perhead - whole (negative = per-head ahead), seeds 0-3."""
import os, torch, torch.nn.functional as F
os.environ.setdefault("QKNORM", "1")
from run import Model, batch, msign
from onestep import rule
xb, yb, mb, _ = batch(2048, torch.Generator().manual_seed(7), "back")


def run(r, seed):
    torch.manual_seed(seed); m = Model()
    names = [(l, n) for l in (0, 1) for n in "qkvo"]; mats = [getattr(m.layers[l], n).weight for l, n in names]; ids = {id(p) for p in mats}
    adam = torch.optim.AdamW([p for p in m.parameters() if id(p) not in ids], lr=0.01, betas=(0.9, 0.98), weight_decay=0)
    buf = [torch.zeros_like(p) for p in mats]; gen = torch.Generator().manual_seed(seed); out = []
    for i in range(151):
        if i % 10 == 0:
            with torch.no_grad(): out.append(F.cross_entropy(m(xb)[mb], yb[mb]).item())
        x, y, msk, _ = batch(32, gen, "back"); loss = F.cross_entropy(m(x)[msk], y[msk])
        adam.zero_grad(); [setattr(p, "grad", None) for p in mats]; loss.backward(); adam.step()
        with torch.no_grad():
            for p, b, (l, n) in zip(mats, buf, names):
                b.mul_(0.95).add_(p.grad); u = p.grad + 0.95 * b
                p.add_(rule(r, u) if n in "qk" else msign(u), alpha=-0.03)
    return out


print("step:      " + " ".join(f"{s:6}" for s in range(0, 151, 10)))
for seed in range(4):
    w, p = run("whole", seed), run("perhead", seed)
    print(f"seed {seed} diff " + " ".join(f"{b - a:+.3f}" for a, b in zip(w, p)))
    print(f"   whole    " + " ".join(f"{a:6.2f}" for a in w))
