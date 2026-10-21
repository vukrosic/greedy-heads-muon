"""Do per-head heads behave like copies, and does look-back start when they split? Every 10 steps: head sameness =
mean cosine between heads' attention scores (pre-softmax, causal part, centered per query) in each layer; back = best
layer-1/2 head's attention on offset -4; loss. whole vs per-head on q,k."""
import os, sys, torch, torch.nn.functional as F
os.environ.setdefault("QKNORM", "1")
from run import Model, batch, msign, H, D
xb, yb, mb, _ = batch(64, torch.Generator().manual_seed(7), "back")
ph = lambda u: torch.cat([msign(c) for c in u.chunk(H, 0)], 0)
tri = torch.ones(48, 48, dtype=torch.bool).tril()


def probe(m):
    h = m.emb(xb) + m.pos[:48]; same, back = [], []; global top; top = []
    for L in m.layers:
        z = L.norm(h); b, t, _ = z.shape
        q, k = (f(z).view(b, t, H, D // H).transpose(1, 2) for f in (L.q, L.k)); q, k = L.qn(q), L.kn(k)
        s = q @ k.transpose(-1, -2); s = torch.where(tri, s, torch.nan); s = s - s.nanmean(-1, keepdim=True)
        v = torch.nan_to_num(s).flatten(2).transpose(0, 1).reshape(H, -1); v = F.normalize(v, dim=1); c = v @ v.T
        same.append(((c.sum() - H) / (H * (H - 1))).item())
        a = torch.where(tri, s, float("-inf")).softmax(-1)
        if len(back) == 0:
            mo = torch.stack([torch.diagonal(a, offset=-o, dim1=-2, dim2=-1).mean() for o in range(8)]); top.append((mo.argmax().item(), mo.max().item()))
        back.append(torch.diagonal(a, offset=-4, dim1=-2, dim2=-1).mean((0, 2)).max().item())
        h = L(h)
    return same, back


seeds = [int(a) for a in sys.argv[1:]] or [0, 1]
for seed in seeds:
    for r in ("whole", "perhead"):
        torch.manual_seed(seed); m = Model()
        names = [(l, n) for l in (0, 1) for n in "qkvo"]; mats = [getattr(m.layers[l], n).weight for l, n in names]; ids = {id(p) for p in mats}
        adam = torch.optim.AdamW([p for p in m.parameters() if id(p) not in ids], lr=0.01, betas=(0.9, 0.98), weight_decay=0)
        buf = [torch.zeros_like(p) for p in mats]; gen = torch.Generator().manual_seed(seed); out = []
        for i in range(151):
            if i % 10 == 0 and i:
                with torch.no_grad(): s, bk = probe(m); Lo = F.cross_entropy(m(xb)[mb], yb[mb]).item()
                out.append(f"{i}: same {s[0]:.2f}/{s[1]:.2f} back {bk[0]:.2f}/{bk[1]:.2f} L{Lo:.2f} L1 top offset -{top[0][0]} ({top[0][1]:.2f})")
            x, y, msk, _ = batch(32, gen, "back"); loss = F.cross_entropy(m(x)[msk], y[msk])
            adam.zero_grad(); [setattr(p, "grad", None) for p in mats]; loss.backward(); adam.step()
            with torch.no_grad():
                for p, b, (l, n) in zip(mats, buf, names):
                    b.mul_(0.95).add_(p.grad); u = p.grad + 0.95 * b
                    p.add_(ph(u) if n in "qk" and r == "perhead" else msign(u), alpha=-0.03)
        print(f"s{seed} {r}"); print("  " + "\n  ".join(out), flush=True)
