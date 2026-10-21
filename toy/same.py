"""Are per-head's piled-up directions the same as the momentum's strongest directions (the ones `boost` enlarged)?
Train with per-head q,k (task back, QK-norm); every 25 steps, for layer-0/1 q and k: overlap of the top-4 input and output
directions of the per-head step with the top-4 of the momentum (1 = same 4 directions, random = 4/64 = 0.06)."""
import sys, os, torch, torch.nn.functional as F
os.environ.setdefault("QKNORM", "1")
from run import Model, batch, msign
from onestep import rule
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
torch.manual_seed(seed); m = Model()
names = [(l, n) for l in (0, 1) for n in "qkvo"]; mats = [getattr(m.layers[l], n).weight for l, n in names]; ids = {id(p) for p in mats}
adam = torch.optim.AdamW([p for p in m.parameters() if id(p) not in ids], lr=0.01, betas=(0.9, 0.98), weight_decay=0)
buf = [torch.zeros_like(p) for p in mats]; gen = torch.Generator().manual_seed(seed)
ov = lambda A, B: (A @ B.T).pow(2).sum().item() / A.shape[0]
for i in range(126):
    x, y, msk, _ = batch(32, gen, "back"); loss = F.cross_entropy(m(x)[msk], y[msk])
    adam.zero_grad(); [setattr(p, "grad", None) for p in mats]; loss.backward(); adam.step()
    with torch.no_grad():
        res = []
        for p, b, (l, n) in zip(mats, buf, names):
            b.mul_(0.95).add_(p.grad); u = p.grad + 0.95 * b
            st = rule("perhead", u) if n in "qk" else msign(u); p.add_(st, alpha=-0.03)
            if n in "qk" and i % 25 == 0 and i:
                Us, _, Vs = torch.linalg.svd(st); Uu, _, Vu = torch.linalg.svd(u)
                res.append((ov(Vs[:4], Vu[:4]), ov(Us[:, :4].T, Uu[:, :4].T)))
        if res: print(f"step {i:3}  input-side {sum(r[0] for r in res) / 4:.2f}  output-side {sum(r[1] for r in res) / 4:.2f}")
