"""Straight path or zigzag? Train task back (QK-norm) with q,k rule argv[1]; every 25 steps print, for layer-1 q and k:
cos of this step with the step 10 steps earlier, and distance moved from the start ||W - W0|| (Frobenius)."""
import sys, os, torch, torch.nn.functional as F
os.environ.setdefault("QKNORM", "1")
from run import Model, batch, msign, H
from onestep import rule
r, seed = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 0
torch.manual_seed(seed); m = Model()
names = [(l, n) for l in (0, 1) for n in "qkvo"]; mats = [getattr(m.layers[l], n).weight for l, n in names]; ids = {id(p) for p in mats}
adam = torch.optim.AdamW([p for p in m.parameters() if id(p) not in ids], lr=0.01, betas=(0.9, 0.98), weight_decay=0)
buf = [torch.zeros_like(p) for p in mats]; gen = torch.Generator().manual_seed(seed)
W0 = [p.detach().clone() for p in mats]; hist = []; cs = []; top = []
for i in range(126):
    x, y, msk, _ = batch(32, gen, "back"); loss = F.cross_entropy(m(x)[msk], y[msk])
    adam.zero_grad(); [setattr(p, "grad", None) for p in mats]; loss.backward(); adam.step()
    with torch.no_grad():
        steps = []
        for p, b, (l, n) in zip(mats, buf, names):
            b.mul_(0.95).add_(p.grad); u = p.grad + 0.95 * b
            st = rule(r, u) if n in "qk" else msign(u); p.add_(st, alpha=-0.03); steps.append(st)
        hist.append([steps[0], steps[1]]); sv = torch.linalg.svdvals(steps[0]); top.append((sv[:4].mean() / sv.mean()).item())
        if i >= 10: cs.append(sum(((a * b).sum() / (a.norm() * b.norm())).item() for a, b in zip(hist[-1], hist[-11])) / 2)
    if i % 25 == 0 and i:
        d = sum((mats[j] - W0[j]).norm().item() for j in (0, 1)) / 2
        print(f"{r:8} step {i:3} loss {loss.item():.2f}  cos(step, 10 earlier) {sum(cs[-15:]) / len(cs[-15:]):.3f}  moved {d:.2f}  top-4 dirs vs average size {sum(top[-25:]) / 25:.2f}")
