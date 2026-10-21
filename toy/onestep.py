"""Same weights, different q,k step rules: which one lowers the true loss most? Train with normal Muon (or the rule in
argv[1]) on task back with QK-norm; at steps 25..125 copy the model, apply ONE q,k step (lr 0.03, from the current
momentum) with each rule, and measure the change of loss on a fixed batch of 2048. Also the same for the true
big-batch gradient's direction (cos of each step with it)."""
import sys, os, copy, torch, torch.nn.functional as F
os.environ.setdefault("QKNORM", "1")
from run import Model, batch, msign, H, D
traj = sys.argv[1] if len(sys.argv) > 1 else "whole"
def rule(r, u):
    if r == "whole": return msign(u)
    if r == "perhead": return torch.cat([msign(c) for c in u.chunk(H, 0)], 0)
    if r == "phw": return msign(torch.cat([msign(c) for c in u.chunk(H, 0)], 0))  # per-head, then flatten the full matrix
    if r == "boost":  # normal Muon, but the momentum's 4 strongest directions get 2.3x (the per-head pile-up, without per-head)
        U, S, Vh = torch.linalg.svd(u, full_matrices=False); f = torch.ones_like(S); f[:4] = 2.3; f = f * (S.numel() / (f ** 2).sum()) ** 0.5
        return (U * f) @ Vh
    if r == "boost2":  # normal Muon plus per-head's own 4 piled-up directions, so they reach ~2.3x; same total size
        P = torch.cat([msign(c) for c in u.chunk(H, 0)], 0); U, S, Vh = torch.linalg.svd(P)
        st = msign(u) + 1.3 * U[:, :4] @ Vh[:4]; return st * (u.shape[0] ** 0.5 / st.norm())
    if r == "shared": return msign(sum(u.chunk(H, 0)) / H).repeat(H, 1)


if __name__ == "__main__":
    torch.manual_seed(0); m = Model()
    names = [(l, n) for l in (0, 1) for n in "qkvo"]; W = lambda mod: [getattr(mod.layers[l], n).weight for l, n in names]
    mats = W(m); ids = {id(p) for p in mats}
    adam = torch.optim.AdamW([p for p in m.parameters() if id(p) not in ids], lr=0.01, betas=(0.9, 0.98), weight_decay=0)
    buf = [torch.zeros_like(p) for p in mats]; gen = torch.Generator().manual_seed(0)
    xb, yb, mb, _ = batch(2048, torch.Generator().manual_seed(7), "back")
    big = lambda mod: F.cross_entropy(mod(xb)[mb], yb[mb]).item()


    for i in range(126):
        x, y, msk, _ = batch(32, gen, "back")
        loss = F.cross_entropy(m(x)[msk], y[msk]); adam.zero_grad(); [setattr(p, "grad", None) for p in mats]; loss.backward(); adam.step()
        with torch.no_grad():
            us = []
            for p, b in zip(mats, buf): b.mul_(0.95).add_(p.grad); us.append(p.grad + 0.95 * b)
        if i % 25 == 0 and i:
            m2 = copy.deepcopy(m); L = F.cross_entropy(m2(xb)[mb], yb[mb]); L.backward(); G = [p.grad for p in W(m2)]
            base = L.item(); out = []
            for r in ("whole", "perhead", "shared"):
                m3 = copy.deepcopy(m)
                with torch.no_grad():
                    cs = []
                    for j, (p, (l, n)) in enumerate(zip(W(m3), names)):
                        if n in "qk": st = rule(r, us[j]); p.add_(st, alpha=-0.03); cs.append(((G[j] * st).sum() / (G[j].norm() * st.norm())).item())
                out.append(f"{r} drop {base - big(m3):+.4f} cos {sum(cs) / len(cs):.3f}")
            print(f"step {i:3} loss {base:.2f} | " + " | ".join(out), flush=True)
        with torch.no_grad():
            for j, (p, (l, n)) in enumerate(zip(mats, names)):
                p.add_(rule(traj, us[j]) if n in "qk" else msign(us[j]), alpha=-0.03)
