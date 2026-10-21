"""When does the look-back (induction) trick appear: Muon vs AdamW.

Task copy: 48 random tokens; a segment of random length k (8-20) at a random start is repeated right after itself.
In the second copy every next token can be found only by looking back at what followed the same token
earlier (the gap k changes per sequence, so a fixed "look back N positions" head cannot solve it).
Copy accuracy = accuracy on those positions.
Task back (control, no copying): predict the token 4 positions back; one head that looks at a fixed position solves it.
Model: 2 layers, attention only, 4 heads, learned positions.
Muon is used on the attention matrices named in --muon_on (Q, K, V, O per layer); embeddings, positions,
output head and norms always use AdamW.
Head scores (copy task): prev = layer-1 attention to the previous token, ind = layer-2 attention to the token
right after the earlier copy of the current token (best head each).

usage: python3 run.py --opt muon --lr 0.03 --adam_lr 0.01 [--width 64] [--task copy] [--muon_on qkvo] [--seed 0]
"""
import argparse, json, time
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(1)
import os
NOISE = float(os.environ.get("NOISE", 0))  # random noise added to q,k momentum before msign (exploration test)
V, T, H = 64, 48, 4
QKN = os.environ.get("QKNORM", "0") == "1"  # RMSNorm on each head's q and k
D = 64


class Layer(nn.Module):
    def __init__(s):
        super().__init__()
        s.q, s.k, s.v, s.o = (nn.Linear(D, D, bias=False) for _ in range(4))
        s.norm = nn.RMSNorm(D)
        s.qn, s.kn = (nn.RMSNorm(D // H), nn.RMSNorm(D // H)) if QKN else (nn.Identity(), nn.Identity())

    def forward(s, x, keep=None):
        b, t, _ = x.shape
        h = s.norm(x)
        q, k, v = (m(h).view(b, t, H, D // H).transpose(1, 2) for m in (s.q, s.k, s.v))
        q, k = s.qn(q), s.kn(k)
        a = (q @ k.transpose(-1, -2)) / (D // H) ** 0.5
        a = a.masked_fill(torch.ones(t, t, dtype=torch.bool).triu(1), float("-inf")).softmax(-1)
        if keep is not None: keep.append(a)
        return x + s.o((a @ v).transpose(1, 2).reshape(b, t, D))


class Model(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb, s.pos = nn.Embedding(V, D), nn.Parameter(torch.randn(T, D) * 0.1)
        s.layers = nn.ModuleList([Layer(), Layer()])
        s.norm, s.head = nn.RMSNorm(D), nn.Linear(D, V, bias=False)

    def forward(s, x, keep=None):
        h = s.emb(x) + s.pos[: x.shape[1]]
        for l in s.layers: h = l(h, keep)
        return s.head(s.norm(h))


def msign(g):
    U, _, Vh = torch.linalg.svd(g, full_matrices=False)
    return U @ Vh


def batch(n, gen, task):
    """returns input, target, mask of scored positions, gap k (copy task)"""
    x = torch.randint(V, (n, T), generator=gen)
    if task == "back":
        y = torch.roll(x, 4, 1); m = torch.zeros(n, T, dtype=torch.bool); m[:, 4:] = True
        return x, y, m, None
    k = torch.randint(8, 21, (n,), generator=gen)
    s = (torch.rand(n, generator=gen) * (T - 2 * k + 1).float()).long()
    m = torch.zeros(n, T - 1, dtype=torch.bool)
    for i in range(n):
        a, kk = s[i].item(), k[i].item()
        x[i, a + kk: a + 2 * kk] = x[i, a: a + kk]
        m[i, a + kk: a + 2 * kk - 1] = True        # input position t predicts x[t+1] = x[t+1-k]
    return x[:, :-1], x[:, 1:], m, k


def head_scores(att, m, k):
    """att: [layer1, layer2] each b x H x t x t"""
    a1, a2 = att
    t = a1.shape[-1]
    prev = torch.diagonal(a1, offset=-1, dim1=-2, dim2=-1).mean((0, 2)).max().item()
    b, pos = m.nonzero(as_tuple=True)
    src = pos - k[b] + 1
    ind = a2[b, :, pos, src].mean(0).max().item()
    return round(prev, 3), round(ind, 3)


def main():
    global D
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt", choices=["muon", "adamw"], default="muon")
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--adam_lr", type=float, default=1e-2, help="AdamW lr for the non-Muon weights in muon runs")
    ap.add_argument("--muon_on", default="qkvo")
    ap.add_argument("--perhead", default="", help="kinds orthogonalized per head, e.g. qk or v (q,k,v: row blocks; o: column blocks)")
    ap.add_argument("--phv", default="", help="disjoint | shared (q,k variants of --perhead)")
    ap.add_argument("--ph_layers", default="01", help="layers where --perhead applies")
    ap.add_argument("--muon_layers", default="01", help="which layers get Muon, e.g. 0 or 1 or 01")
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--task", choices=["copy", "back"], default="copy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--out", default="runs2.jsonl")
    a = ap.parse_args()
    D = a.width

    torch.manual_seed(a.seed); model = Model()
    mats = [getattr(model.layers[int(li)], n).weight for li in a.muon_layers for n in a.muon_on] if a.opt == "muon" else []
    kinds = [(li, n) for li in a.muon_layers for n in a.muon_on] if a.opt == "muon" else []
    ids = {id(p) for p in mats}
    rest = [p for p in model.parameters() if id(p) not in ids]
    if a.opt == "adamw":                                # attention matrices at --lr, everything else at --adam_lr (same split as muon runs)
        attn = {id(getattr(l, n).weight) for l in model.layers for n in "qkvo"}
        groups = [dict(params=[p for p in rest if id(p) in attn], lr=a.lr), dict(params=[p for p in rest if id(p) not in attn], lr=a.adam_lr)]
    else:
        groups = [dict(params=rest, lr=a.adam_lr)]
    adam = torch.optim.AdamW(groups, betas=(0.9, 0.98), weight_decay=0)
    buf = [torch.zeros_like(p) for p in mats]
    gen, gv = torch.Generator().manual_seed(a.seed), torch.Generator().manual_seed(999)
    xv, yv, mv, kv = batch(256, gv, a.task)

    t0, curve, first = time.time(), [], None
    for i in range(a.steps):
        x, y, msk, _ = batch(32, gen, a.task)
        out = model(x)
        loss = F.cross_entropy(out[msk], y[msk]) if a.task == "back" else F.cross_entropy(out.reshape(-1, V), y.reshape(-1))
        adam.zero_grad(); [setattr(p, "grad", None) for p in mats]
        loss.backward()
        adam.step()
        with torch.no_grad():
            for p, b, kd in zip(mats, buf, kinds):
                b.mul_(0.95).add_(p.grad); u = p.grad + 0.95 * b
                if NOISE and kd[1] in "qk": u = u + NOISE * u.norm() / u.numel() ** 0.5 * torch.randn_like(u)
                dm = 1 if kd[1] == "o" else 0
                if kd[1] in a.perhead and kd[0] in a.ph_layers and a.phv == "disjoint":  # head h may only change input columns of its own quarter
                    w = D // H; st = torch.zeros_like(u)
                    for h in range(H): st[h * w:(h + 1) * w, h * w:(h + 1) * w] = msign(u[h * w:(h + 1) * w, h * w:(h + 1) * w])
                elif kd[1] in a.perhead and kd[0] in a.ph_layers and a.phv == "shared":  # every head gets the same step
                    st = msign(sum(u.chunk(H, 0)) / H).repeat(H, 1)
                elif kd[1] in a.perhead and kd[0] in a.ph_layers: st = torch.cat([msign(c) for c in u.chunk(H, dm)], dm)
                else: st = msign(u)
                p.add_(st, alpha=-a.lr)
        if i % 25 == 0 or i == a.steps - 1:
            with torch.no_grad():
                keep = []
                acc = (model(xv, keep).argmax(-1)[mv] == yv[mv]).float().mean().item()
            row = [i, round(acc, 3)]
            if a.task == "copy": row += list(head_scores(keep, mv, kv))
            else: row += [[round(torch.diagonal(at, offset=-4, dim1=-2, dim2=-1).mean((0, 2))[h].item(), 2) for h in range(H)] for at in keep]
            curve.append(row)
            if first is None and acc > 0.5: first = i
    r = dict(task=a.task, width=a.width, opt=a.opt, lr=a.lr, perhead=a.perhead + a.phv + (":" + a.ph_layers if a.perhead and a.ph_layers != "01" else ""), muon_on=a.muon_on if a.opt == "muon" else "", muon_layers=a.muon_layers if a.opt == "muon" else "",
             adam_lr=a.adam_lr, seed=a.seed, steps=a.steps,
             first_step_over_half=first, final_acc=curve[-1][1], sec=round(time.time() - t0, 1), curve=curve)
    with open(a.out, "a") as f: f.write(json.dumps(r) + "\n")
    print(json.dumps({k: v for k, v in r.items() if k != "curve"}))


if __name__ == "__main__":
    main()
