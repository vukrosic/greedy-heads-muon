"""Per-head step, but each head's step is made to use input directions the earlier heads did not (Gram-Schmidt, sizes kept).
Does it keep per-head's early lead (loss at step 50) and avoid the delay (first step loss < 0.5)? 4 seeds."""
import os, time, torch, torch.nn.functional as F
os.environ.setdefault("QKNORM", "1")
from run import Model, batch, msign, H
xb, yb, mb, _ = batch(256, torch.Generator().manual_seed(7), "back")
ph = lambda u: torch.cat([msign(c) for c in u.chunk(H, 0)], 0)


def apart(u):
    out, Q = [], torch.zeros(0, u.shape[1])
    for c in u.chunk(H, 0):
        c = c - c @ Q.T @ Q; s = msign(c); out.append(s); Q = torch.cat([Q, torch.linalg.qr(s.T)[0].T], 0)
    return torch.cat(out, 0)


def run(seed, rule, S=200):
    torch.manual_seed(seed); m = Model(); l50 = None
    names = [(l, n) for l in (0, 1) for n in "qkvo"]; mats = [getattr(m.layers[l], n).weight for l, n in names]; ids = {id(p) for p in mats}
    adam = torch.optim.AdamW([p for p in m.parameters() if id(p) not in ids], lr=0.01, betas=(0.9, 0.98), weight_decay=0)
    buf = [torch.zeros_like(p) for p in mats]; gen = torch.Generator().manual_seed(seed)
    for i in range(200):
        if i == 50 or (i % 5 == 0 and i >= 90):
            with torch.no_grad(): L = F.cross_entropy(m(xb)[mb], yb[mb]).item()
            if i == 50: l50 = L
            elif L < 0.5: return l50, i
        x, y, msk, _ = batch(32, gen, "back"); loss = F.cross_entropy(m(x)[msk], y[msk])
        adam.zero_grad(); [setattr(p, "grad", None) for p in mats]; loss.backward(); adam.step()
        with torch.no_grad():
            for p, b, (l, n) in zip(mats, buf, names):
                b.mul_(0.95).add_(p.grad); u = p.grad + 0.95 * b
                f = {"whole": msign, "perhead": ph, "apart": apart}[rule] if (n in "qk" and i < S) else msign
                p.add_(f(u), alpha=-0.03)
    return l50, 999


t0 = time.time()
for k, kw in {"whole": dict(rule="whole"), "per-head": dict(rule="perhead"), "per-head, heads kept apart": dict(rule="apart")}.items():
    r = [run(s, **kw) for s in range(4)]
    print(f"{time.time() - t0:4.0f}s {k:28} loss@50 {sum(a for a, _ in r) / 4:.3f}  solved {[b for _, b in r]} mean {sum(b for _, b in r) / 4:.1f}", flush=True)
