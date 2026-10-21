"""Share of the layer-1 q,k step that reads POSITION (input directions spanned by the top-8 principal directions of the
centered position embeddings) vs TOKEN identity (top-8 of centered token embeddings). whole vs per-head, every 20 steps."""
import os, torch, torch.nn.functional as F
os.environ.setdefault("QKNORM", "1")
from run import Model, batch, msign, H
ph = lambda u: torch.cat([msign(c) for c in u.chunk(H, 0)], 0)
basis = lambda X: torch.linalg.svd(X - X.mean(0), full_matrices=False)[2][:8]
for seed in (0, 1):
    for r in ("whole", "perhead"):
        torch.manual_seed(seed); m = Model()
        names = [(l, n) for l in (0, 1) for n in "qkvo"]; mats = [getattr(m.layers[l], n).weight for l, n in names]; ids = {id(p) for p in mats}
        adam = torch.optim.AdamW([p for p in m.parameters() if id(p) not in ids], lr=0.01, betas=(0.9, 0.98), weight_decay=0)
        buf = [torch.zeros_like(p) for p in mats]; gen = torch.Generator().manual_seed(seed); out = []; acc = [0, 0, 0]
        for i in range(101):
            x, y, msk, _ = batch(32, gen, "back"); loss = F.cross_entropy(m(x)[msk], y[msk])
            adam.zero_grad(); [setattr(p, "grad", None) for p in mats]; loss.backward(); adam.step()
            with torch.no_grad():
                P, Tk = basis(m.pos.detach()), basis(m.emb.weight.detach())
                for j, (p, b, (l, n)) in enumerate(zip(mats, buf, names)):
                    b.mul_(0.95).add_(p.grad); u = p.grad + 0.95 * b
                    st = ph(u) if n in "qk" and r == "perhead" else msign(u)
                    if l == 0 and n in "qk": acc[0] += (st @ P.T).pow(2).sum().item(); acc[1] += (st @ Tk.T).pow(2).sum().item(); acc[2] += st.pow(2).sum().item()
                    p.add_(st, alpha=-0.03)
            if i % 20 == 0 and i: out.append(f"{i:3}: pos {acc[0] / acc[2]:.3f} tok {acc[1] / acc[2]:.3f}"); acc = [0, 0, 0]
        print(f"s{seed} {r:7} | " + " | ".join(out), flush=True)
