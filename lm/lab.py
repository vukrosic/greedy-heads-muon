"""Quick Muon experiments: one small byte LM, train with Muon or AdamW, save, then probe.

python3 lab.py train --opt muon --tag base            trains, saves ckpt/<tag>_<opt>.pt, prints val loss
python3 lab.py probe NAME --tag base                  runs probe NAME on both saved models
Model: byte LM, 4 layers, width 64, 4 heads, RoPE, QK-norm, pre-norm RMSNorm, squared-ReLU MLP, tied embeddings, T=64.
Muon: msign (exact SVD) of Nesterov momentum 0.95 on attention and MLP matrices, step x sqrt(max(1,rows/cols)).
AdamW: same matrices with AdamW. Embeddings and norm weights: AdamW 3e-3 in both.
Extra training switches (all default off) are listed in TRAIN_FLAGS.
"""
import argparse, json, math, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
TEXT = os.environ.get("DATA", os.path.join(HERE, "data", "corpus.txt"))  # any plain-text file; see prepare_data.py
CK = os.path.join(HERE, "ckpt"); os.makedirs(CK, exist_ok=True)
T, H = 64, int(os.environ.get("HEADS", 4))
ACT = os.environ.get("ACT", "relu2")
OBLOCK = os.environ.get("OBLOCK", "0") == "1"  # o is block-diagonal: each head writes only to its own D/H residual coordinates
VNORM = os.environ.get("VNORM", "0") == "1"  # RMSNorm on each head's v (gives v heads their own norm)
if not os.path.exists(TEXT): sys.exit(f"no corpus at {TEXT}: run python3 prepare_data.py or set DATA=your_text_file")
data = torch.frombuffer(bytearray(open(TEXT, "rb").read()), dtype=torch.uint8).long()
cut = int(0.9 * len(data)); TR, VA = data[:cut], data[cut:]
_gv = torch.Generator().manual_seed(12345)
VAL = [torch.stack([VA[i:i + T + 1] for i in idx]) for idx in torch.randint(len(VA) - T - 1, (int(os.environ.get("VALN", 4)), 32), generator=_gv)]


def rope(x, base=10000):
    s, d = x.shape[-2], x.shape[-1]
    th = 1.0 / base ** (torch.arange(0, d, 2).float() / d)
    ang = torch.outer(torch.arange(s).float(), th); c, sn = ang.cos(), ang.sin()
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack([a * c - b * sn, b * c + a * sn], -1).flatten(-2)


class Block(nn.Module):
    def __init__(s, D, qknorm=True):
        super().__init__()
        s.D, s.dk = D, D // H
        s.q, s.k, s.v, s.o = (nn.Linear(D, D, bias=False) for _ in range(4))
        s.up, s.down = nn.Linear(D, 4 * D, bias=False), nn.Linear(4 * D, D, bias=False)
        s.n1, s.n2 = nn.RMSNorm(D), nn.RMSNorm(D)
        s.qn, s.kn = (nn.RMSNorm(s.dk), nn.RMSNorm(s.dk)) if qknorm else (nn.Identity(), nn.Identity())
        s.vn = nn.RMSNorm(s.dk) if VNORM else nn.Identity()

    def forward(s, x, cap=None):
        b, t, _ = x.shape
        h = s.n1(x)
        q, k, v = (m(h).view(b, t, H, s.dk).transpose(1, 2) for m in (s.q, s.k, s.v))
        q, k, v = rope(s.qn(q)), rope(s.kn(k)), s.vn(v)
        if cap is None:
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            a = (q @ k.transpose(-1, -2)) / math.sqrt(s.dk)
            a = a.masked_fill(torch.ones(t, t, dtype=torch.bool).triu(1), float("-inf"))
            cap.setdefault("logit", []).append(a.amax().item()); a = a.softmax(-1)
            cap.setdefault("att", []).append(a); o = a @ v
        oin = o.transpose(1, 2).reshape(b, t, s.D)
        x = x + (F.linear(oin, s.o.weight * torch.block_diag(*[torch.ones(s.dk, s.dk)] * H)) if OBLOCK else s.o(oin))
        z = s.up(s.n2(x)); u = F.relu(z).square() if ACT == "relu2" else F.relu(z) if ACT == "relu" else F.gelu(z) if ACT == "gelu" else F.silu(z)
        if cap is not None: cap.setdefault("mlp", []).append(u)
        return x + s.down(u)


class LM(nn.Module):
    def __init__(s, D=64, L=4, qknorm=True, std=0.02):
        super().__init__()
        s.D = D
        s.emb = nn.Embedding(256, D)
        s.blocks = nn.ModuleList([Block(D, qknorm) for _ in range(L)])
        s.norm = nn.RMSNorm(D)
        for m in s.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, std=std)

    def forward(s, x, skip=(), cap=None):
        x = s.emb(x) * math.sqrt(s.D)
        for i, blk in enumerate(s.blocks):
            if cap is not None: cap.setdefault("resid", []).append(x)
            if i not in skip: x = blk(x, cap)
        if cap is not None: cap.setdefault("resid", []).append(x)
        return s.norm(x) @ s.emb.weight.T


def mats_of(m): return [p for n, p in m.named_parameters() if p.ndim == 2 and n != "emb.weight"]


def val_loss(m, **kw):
    with torch.no_grad():
        return sum(F.cross_entropy(m(b[:, :-1], **kw).reshape(-1, 256), b[:, 1:].reshape(-1)).item() for b in VAL) / len(VAL)


POWER = 0.0
FILT = None
CUR = None  # this step's raw gradient, for the "caut" filter
HEAD = None  # (k, h): the momentum's top k directions get strength h, the rest 1 (no renormalization)


def msign(g):
    try: U, S, Vh = torch.linalg.svd(g, full_matrices=False)
    except torch._C._LinAlgError: U, S, Vh = (y.float() for y in torch.linalg.svd(g.double(), full_matrices=False))
    if FILT:  # spectral filter f(s), rescaled to the same Frobenius norm as U V^T
        kind, c = FILT; e = c * S[0]
        if kind == "caut":  # direction i of the momentum keeps full weight only if this step's gradient agrees with it
            a_ = ((U.T @ CUR) * Vh).sum(1); f = torch.where(a_ > 0, torch.ones_like(S), torch.full_like(S, c))
        else: f = S / (S ** 2 + e ** 2).sqrt() if kind == "soft" else S / (S ** 2 + e ** 2)
        return (U * (f * (len(S) ** .5 / f.norm()))) @ Vh
    if HEAD:
        f = torch.ones_like(S); f[:HEAD[0]] = HEAD[1]; return (U * f) @ Vh
    if POWER:  # U S^p V^T, rescaled to the same Frobenius norm as U V^T
        Sp = S ** POWER; return (U * (Sp * (len(S) ** .5 / Sp.norm()))) @ Vh
    return U @ Vh


def ipow(A, eps=1e-6):
    e, Q = torch.linalg.eigh(A.double()); e = e.clamp_min(eps * e.max())
    return ((Q * e ** -0.25) @ Q.T).float()


def lowp(x, kind):
    """Round per row (weights) or per token (activations): 'fp8' = e4m3, 'int8' / 'int6' / ... = symmetric integer."""
    if kind == "fp8":
        sc = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / 448
        return (x / sc).to(torch.float8_e4m3fn).float() * sc
    L = 2 ** (int(kind[3:]) - 1) - 1
    sc = x.abs().amax(-1, keepdim=True).clamp_min(1e-12) / L
    return (x / sc).round().clamp(-L, L) * sc


def canaries(text, C=64):
    """40 strings: first 20 get planted, last 20 stay unseen. Random bytes, or real passages from the val text."""
    if not text: return torch.randint(256, (40, C + 1), generator=torch.Generator().manual_seed(999))
    st = torch.randint(len(VA) - C - 1, (40,), generator=torch.Generator().manual_seed(999))
    return torch.stack([VA[j:j + C + 1] for j in st])


def train(a, model=None, log=None):
    torch.manual_seed(a.seed)
    m = model or LM(a.width, a.layers, not a.no_qknorm, a.init_std)
    mats = mats_of(m); ids = {id(p) for p in mats}
    other = [p for p in m.parameters() if id(p) not in ids]
    if a.muon_emb: mats = mats + [m.emb.weight]; other = [p for p in other if p is not m.emb.weight]
    ao = torch.optim.AdamW(other, lr=a.adam_lr, betas=(0.9, 0.95), weight_decay=0)
    am = torch.optim.AdamW(mats, lr=a.lr, betas=(0.9, 0.95), weight_decay=a.wd) if a.opt == "adamw" else None
    buf = [torch.zeros_like(p) for p in mats]
    train.kind = [n.split(".")[-2] for n, p in m.named_parameters() if p.ndim == 2 and n != "emb.weight"] + (["emb"] if a.muon_emb else [])
    train.rin = torch.ones(len(mats)); lins = [mod for blk in m.blocks for mod in (blk.q, blk.k, blk.v, blk.o, blk.up, blk.down)]
    for j, mod in enumerate(lins): mod.register_forward_pre_hook(lambda mod, inp, j=j: train.rin.__setitem__(j, inp[0].detach().pow(2).mean().sqrt().item()))
    train.n0 = [p.norm().item() for p in mats]; train.mult = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.mult.split(",") if kv}
    if a.budget_fixed: train.fixed = torch.tensor(json.load(open(a.budget_fixed)))
    g = torch.Generator().manual_seed(a.seed + 1 + 1000 * a.dseed)
    t0 = time.time()
    for i in range(a.steps):
        sc = min(1, (i + 1) / max(1, a.warmup)) * (min(1, 2 * (1 - i / a.steps)) if not a.const else 1)
        src = TR[:a.data_kb * 1024] if a.data_kb else TR
        C = a.ctx or T
        b = torch.stack([src[j:j + C + 1] for j in torch.randint(len(src) - C - 1, (a.batch,), generator=g)])
        if a.canary:
            if i == 0: train.can = canaries(a.canary_text, C)
            if i % a.canary == 0 and i < a.canary_until * a.steps: b[0] = train.can[(i // a.canary) % 20]
        if a.mix_kb:
            sub = TR[:a.mix_kb * 1024]; k = a.batch // 2
            b[:k] = torch.stack([sub[j:j + C + 1] for j in torch.randint(len(sub) - C - 1, (k,), generator=g)])
        if a.spike and a.steps // 2 <= i < a.steps // 2 + a.spike: b = torch.randint(256, b.shape, generator=g)
        y = b[:, 1:]
        if a.label_noise: y = torch.where(torch.rand(y.shape, generator=g) < a.label_noise, torch.randint(256, y.shape, generator=g), y)
        qat = (a.qat and i >= int((1 - a.qat) * a.steps)) or a.trainq
        if a.trainq and i == 0:
            for mod in m.modules():
                if isinstance(mod, nn.Linear): mod.register_forward_pre_hook(lambda mod, inp: (inp[0] + (lowp(inp[0], a.trainq) - inp[0]).detach(),))
        if qat:
            from probes import _q
            keep = [p.detach().clone() for p in mats_of(m)]
            with torch.no_grad():
                for p in mats_of(m): p.copy_(lowp(p, a.trainq) if a.trainq else _q(p, 3))
        if a.teacher:
            if i == 0 or not hasattr(train, "tch"): train.tch = load(*(a.teacher.split(":") if ":" in a.teacher else (a.teacher, "muon"))).eval()
            with torch.no_grad(): pt = train.tch(b[:, :-1]).softmax(-1)
            loss = -(pt * m(b[:, :-1]).log_softmax(-1)).sum(-1).mean()
        elif a.decor and i < a.decor_until * a.steps:  # + mean squared correlation between MLP neurons (raises activation rank)
            cap = {}; loss = F.cross_entropy(m(b[:, :-1], cap=cap).reshape(-1, 256), y.reshape(-1))
            for u in cap["mlp"]:
                z = u.reshape(-1, u.shape[-1]); z = (z - z.mean(0)) / (z.std(0) + 1e-4); c = z.T @ z / len(z)
                loss = loss + a.decor * ((c - torch.eye(len(c))) ** 2).mean() / len(cap["mlp"])
        else:
            loss = F.cross_entropy(m(b[:, :-1]).reshape(-1, 256), y.reshape(-1))
        if not torch.isfinite(loss): break
        m.zero_grad(); loss.backward()
        if qat:
            with torch.no_grad():
                for p, k in zip(mats_of(m), keep): p.copy_(k)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        for gr in ao.param_groups: gr["lr"] = a.adam_lr * sc
        ao.step()
        if a.switch and am is None and i >= int((1 - a.switch) * a.steps):
            am = torch.optim.AdamW(mats, lr=1e-2, betas=(0.9, 0.95), weight_decay=a.wd); a.lr = 1e-2
        if am is not None:
            for gr in am.param_groups: gr["lr"] = a.lr * sc
            am.step()
        else:
            with torch.no_grad():
                us = []
                for p, bf in zip(mats, buf):
                    gg = msign(p.grad) if a.ortho_first else p.grad
                    bf.mul_(a.beta).add_(gg); us.append(gg + a.beta * bf if not a.no_nesterov else bf)
                if a.slow:  # slow EMA (normalized), scaled to the fast buffer's 1/(1-beta) size
                    if i == 0: train.slow = [torch.zeros_like(p) for p in mats]
                    for j, (p, sl) in enumerate(zip(mats, train.slow)):
                        sl.mul_(a.slow_beta).add_(p.grad, alpha=1 - a.slow_beta); us[j] = us[j] + a.slow * min(1, i / 200) * sl / (1 - a.beta)
                if getattr(train, "cb", None): train.cb(m, mats, us, i, a)
                nm = torch.tensor([u.norm() / (u.numel() ** .5 if a.budget_rms else 1) for u in us]); bs = (nm / nm.mean()) ** a.budget
                if a.noise:  # LANTON-style signal: EMA of |g_t - g_(t-1)|^2 per matrix
                    gn = torch.tensor([(p.grad - pg).norm().item() ** 2 for p, pg in zip(mats, train.pg)]) if i else torch.tensor([p.grad.norm().item() ** 2 for p in mats])
                    train.H = 0.95 * train.H + 0.05 * gn if i else gn; train.pg = [p.grad.clone() for p in mats]
                    bs = (train.H.sqrt() / train.H.sqrt().mean()) ** a.noise
                if a.bclamp: bs = bs.clamp(1 / a.bclamp, a.bclamp)
                if a.budget_fixed: bs = train.fixed
                if a.inrms: bs = bs * (train.rin / train.rin.mean()) ** -a.inrms
                if a.budget_log: train.rlog = (train.rlog if i else []) + ([[i] + train.rin.tolist()] if i % 25 == 0 else [])
                if a.budget_on: bs = torch.tensor([b_ if k in a.budget_on.split(",") else 1.0 for b_, k in zip(bs.tolist(), train.kind)])
                for j, (p, k) in enumerate(zip(mats, train.kind)):
                    bs[j] *= train.mult.get(k, 1.0) * ((p.norm() / train.n0[j]).item() if k in a.rel.split(",") else 1.0)
                if a.hyper:  # cos(gradient now, step taken last time) > 0: last step too short along it; < 0: overshot
                    if i == 0: train.hm = torch.zeros(len(mats)); train.last = [None] * len(mats)
                    elif i > 0:
                        cs = torch.tensor([(F.cosine_similarity(p.grad.flatten(), l.flatten(), 0).item() if l is not None else 0.0) for p, l in zip(mats, train.last)])
                        train.hm = (train.hm + a.hyper * cs).clamp(-math.log(a.hyper_clip), math.log(a.hyper_clip))
                    bs = bs * train.hm.exp()
                    if a.budget_log and i % 25 == 0: train.hlog = (train.hlog if i else []) + [[i] + train.hm.exp().tolist()]
                if a.budget_log:
                    train.bsum = train.bsum + bs if i else bs.clone()
                    if i % 25 == 0: train.traj = (train.traj if i else []) + [[i] + bs.tolist() + nm.tolist()]
                for j, (p, u, s_) in enumerate(zip(mats, us, bs)):
                    p.mul_(1 - a.lr * sc * a.wd)
                    if a.shampoo >= 0:  # L^-1/4 u R^-1/4 with L, R = EMA of u u^T, u^T u; b2 = 0 is exactly U V^T
                        if i == 0: train.sh = [[torch.zeros(q.shape[0], q.shape[0]), torch.zeros(q.shape[1], q.shape[1])] for q in mats]
                        Ls, Rs = train.sh[j]; st = u if not a.sh_grad else p.grad
                        Ls.mul_(a.shampoo).add_(st @ st.T, alpha=1 - a.shampoo); Rs.mul_(a.shampoo).add_(st.T @ st, alpha=1 - a.shampoo)
                        D = ipow(Ls) @ u @ ipow(Rs); p.add_(D * (min(u.shape) ** .5 / D.norm()), alpha=-a.lr * sc * s_.item() * max(1, p.shape[0] / p.shape[1]) ** 0.5); continue
                    global CUR, FILT, POWER; CUR = p.grad
                    if a.power_until < 1: POWER = a.power if i < a.power_until * a.steps else 0.0
                    if a.power_from > 0: POWER = a.power if i >= a.power_from * a.steps else 0.0
                    if a.filt_late: FILT = (a.filt.split(":")[0], float(a.filt.split(":")[1])) if i >= (1 - a.filt_late) * a.steps else None
                    kd = train.kind[j]; nb = dict((x.split("=")[0], int(x.split("=")[1])) for x in a.blocks.split(",") if x)
                    if a.qkpair and kd in ("q", "k"):
                        if kd == "q":  # q comes right before k in mats: do both now, stash k's
                            uk = us[j + 1]; qc, kc = u.chunk(H, 0), uk.chunk(H, 0); pr = [msign(torch.cat([x, y], 0)) for x, y in zip(qc, kc)]
                            upd = torch.cat([x[:x.shape[0] // 2] for x in pr], 0); train.kst = torch.cat([x[x.shape[0] // 2:] for x in pr], 0)
                        else: upd = train.kst
                    elif kd in nb:
                        dm = 0 if kd != "down" else 1; upd = torch.cat([msign(c) for c in u.chunk(nb[kd], dm)], dm)
                        if a.ph_spec: upd = upd / torch.linalg.matrix_norm(upd, 2) ** a.ph_spec
                    elif a.perhead and kd in a.perhead.split(",") and a.geo_from * a.steps <= i < a.ph_until * a.steps:  # q, k, v: head = block of output rows; o: block of input columns
                        dm = 0 if kd != "o" else 1; ph = [msign(c) for c in u.chunk(H, dm)]
                        if a.ph_mode not in ("both", "flat") and not a.ph_mode.startswith(("pow:", "spec:")):
                            wh = msign(u).chunk(H, dm); tot = sum(w.norm() ** 2 for w in wh) ** .5
                            ph = [w * (tot / H ** .5 / w.norm()) for w in wh] if a.ph_mode == "size" else [p_ * (w.norm() / p_.norm()) for p_, w in zip(ph, wh)]
                        upd = torch.cat(ph, dm) if a.ph_mode != "flat" else msign(torch.cat(ph, dm))  # flat = per-head, then msign of the whole matrix
                        if a.ph_mode.startswith("spec:"):  # per-head's singular-value spread, placed on the momentum's own directions (top = biggest on momentum's top, rand = shuffled)
                            S_ = torch.linalg.svdvals(upd); U_, _, V_ = torch.linalg.svd(u, full_matrices=False)
                            if a.ph_mode == "spec:rand": S_ = S_[torch.randperm(len(S_))]
                            upd = U_ @ torch.diag(S_) @ V_
                        if a.ph_mode.startswith("pow:"):  # lopsidedness dial: per-head directions, singular values ^ pw (0 = flat, 1 = per-head), same size
                            U_, S_, V_ = torch.linalg.svd(upd, full_matrices=False); D_ = U_ @ torch.diag(S_ ** float(a.ph_mode[4:])) @ V_; upd = D_ * (upd.norm() / D_.norm())
                        if a.ph_spec: upd = upd / torch.linalg.matrix_norm(upd, 2) ** a.ph_spec
                    else:
                        uk = "sign" if i < a.early_sign * a.steps and (not a.early_on or kd in a.early_on.split(",")) else a.update
                        upd = msign(u) if uk == "msign" else u * (min(u.shape) ** .5 / u.norm()) if uk == "raw" else u.sign() * (min(u.shape) / u.numel()) ** .5
                    if kd in a.roweq.split(",") and a.geo_from * a.steps <= i < a.roweq_until * a.steps:
                        tgt = p.detach().norm(dim=1, keepdim=True) ** a.rowpow if a.rowpow else 1.0
                        r = upd / upd.norm(dim=1, keepdim=True).clamp_min(1e-8) * tgt; upd = r * (upd.norm() / r.norm())
                    if kd in a.tframe.split(","):  # alternate polar and equal rows: equal-norm tight frame near the momentum
                        for _ in range(a.tf_iters): upd = msign(upd); upd = upd / upd.norm(dim=1, keepdim=True).clamp_min(1e-8) * (upd.norm() / upd.shape[0] ** .5)
                    if kd in a.coleq.split(","): upd = upd / upd.norm(dim=0, keepdim=True).clamp_min(1e-8) * (upd.norm() / upd.shape[1] ** .5)
                    if a.hyper: train.last[j] = upd
                    p.add_(upd, alpha=-a.lr * sc * s_.item() * max(1, p.shape[0] / p.shape[1]) ** 0.5)
                for j, (p, k) in enumerate(zip(mats, train.kind)):
                    if k in a.renorm.split(","): p.mul_(train.n0[j] / p.norm())
        if a.ema:
            with torch.no_grad():
                if i == 0: train.ema = [p.detach().clone() for p in m.parameters()]
                for e, p in zip(train.ema, m.parameters()): e.lerp_(p, 1 - a.ema)
        if a.bf16:
            with torch.no_grad():
                for p in m.parameters(): p.copy_(p.bfloat16().float())
        if log is not None and (i % 50 == 0 or i == a.steps - 1 or (a.spike and i in (a.steps // 2 + a.spike, a.steps // 2 + a.spike + 10, a.steps // 2 + a.spike + 25))): log.append((i, round(val_loss(m), 4)))
    if a.budget_log:
        names = [n for n, p in m.named_parameters() if p.ndim == 2 and n != "emb.weight"]
        avg = (train.bsum / a.steps).tolist(); json.dump(avg, open(a.budget_log, "w")); json.dump(train.traj, open(a.budget_log.replace(".json", "_traj.json"), "w")); json.dump(train.rlog, open(a.budget_log.replace(".json", "_rin.json"), "w"))
        if a.hyper: json.dump(train.hlog, open(a.budget_log.replace(".json", "_hyper.json"), "w"))
        print("mean step multiplier:", {n.replace("blocks.", "").replace(".weight", ""): round(v, 2) for n, v in zip(names, avg)})
    if a.ema:
        import copy; me = copy.deepcopy(m)
        with torch.no_grad():
            for e, p in zip(train.ema, me.parameters()): p.copy_(e)
        print("ema val", round(val_loss(me), 4), "final", round(val_loss(m), 4))
    return m, time.time() - t0


TRAIN_FLAGS = ["--wd", "--no_qknorm", "--muon_emb", "--const", "--no_nesterov", "--beta", "--batch", "--layers", "--width"]


def train_args(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("name", nargs="?")
    ap.add_argument("--opt", default="muon", choices=["muon", "adamw"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--adam_lr", type=float, default=3e-2)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--beta", type=float, default=0.95)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_qknorm", action="store_true"); ap.add_argument("--muon_emb", action="store_true")
    ap.add_argument("--const", action="store_true"); ap.add_argument("--no_nesterov", action="store_true")
    ap.add_argument("--tag", default="base")
    ap.add_argument("--canary_text", action="store_true", help="canaries are real val passages instead of random bytes")
    ap.add_argument("--canary_until", type=float, default=1.0, help="canaries only in this first share of training")
    ap.add_argument("--canary", type=int, default=0, help="every N steps one batch row is one of 20 random 64-byte canaries")
    ap.add_argument("--power", type=float, default=0.0, help="Muon update U S^p V^T instead of U V^T")
    ap.add_argument("--bf16", action="store_true", help="weights rounded to bf16 after every step (no fp32 master copy)")
    ap.add_argument("--mix_kb", type=int, default=0, help="half of every batch comes from the first N KB (an upsampled subset)")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--budget", type=float, default=0.0, help="Muon step of each matrix x (its momentum size / mean size)^budget")
    ap.add_argument("--budget_rms", action="store_true", help="budget uses per-entry RMS of the momentum instead of its total norm")
    ap.add_argument("--budget_log", default="", help="save each matrix's mean budget multiplier to this json")
    ap.add_argument("--budget_fixed", default="", help="fixed per-matrix multipliers from a json (no adaptation)")
    ap.add_argument("--inrms", type=float, default=0.0, help="Muon step of each matrix x (its input RMS / mean input RMS)^-inrms")
    ap.add_argument("--update", default="msign", choices=["msign", "sign", "raw"], help="sign = elementwise sign of momentum, same Frobenius size as msign")
    ap.add_argument("--filt", default="", help="spectral filter kind:c, soft = s/sqrt(s^2+(c s_max)^2), newton = s/(s^2+(c s_max)^2)")
    ap.add_argument("--ortho_first", action="store_true", help="momentum accumulates msign(gradient) instead of the gradient")
    ap.add_argument("--shampoo", type=float, default=-1.0, help="Shampoo-style step with memory b2 (0 = Muon)")
    ap.add_argument("--sh_grad", action="store_true", help="Shampoo statistics from raw gradients instead of momentum")
    ap.add_argument("--noise", type=float, default=0.0, help="multiplier x (gradient-noise size / mean)^noise instead of momentum size")
    ap.add_argument("--ph_mode", default="both", help="both = per-head msign; size = whole msign, heads rescaled equal; shape = per-head msign, heads rescaled to whole-msign sizes")
    ap.add_argument("--ph_spec", type=float, default=0.0, help="per-head update divided by (its spectral norm)^p, p = this (1 = same spectral norm as Muon)")
    ap.add_argument("--blocks", default="", help="kind=n: orthogonalize n blocks of the hidden side separately (up: rows, down: columns), e.g. up=4,down=4")
    ap.add_argument("--qkpair", action="store_true", help="orthogonalize [q_h; k_h] of each head jointly (32x64 blocks)")
    ap.add_argument("--coleq", default="", help="kinds whose Muon step gets equal column lengths, same Frobenius size")
    ap.add_argument("--rowpow", type=float, default=0, help="roweq rows get length ~ |W_row|^rowpow (0 = equal)")
    ap.add_argument("--roweq_until", type=float, default=1.0, help="apply roweq only in the first this fraction of steps")
    ap.add_argument("--ph_until", type=float, default=1.0, help="per-head only in the first this fraction of steps")
    ap.add_argument("--geo_from", type=float, default=0.0, help="per-head and roweq start only after this fraction of steps")
    ap.add_argument("--early_sign", type=float, default=0.0, help="first this fraction of steps: sign steps instead of Muon")
    ap.add_argument("--early_on", default="", help="kinds that use early_sign (default all)")
    ap.add_argument("--power_from", type=float, default=0.0, help="use --power only from this fraction of steps on")
    ap.add_argument("--head", default="", help="k:h, top k momentum directions get strength h (SAMuon-like head shrink)")
    ap.add_argument("--power_until", type=float, default=1.0, help="use --power only in the first this fraction of steps")
    ap.add_argument("--decor", type=float, default=0.0, help="weight of an MLP neuron decorrelation penalty")
    ap.add_argument("--decor_until", type=float, default=0.125, help="penalty only in the first this fraction of steps")
    ap.add_argument("--tframe", default="", help="kinds whose step alternates polar and equal-row normalization")
    ap.add_argument("--tf_iters", type=int, default=5)
    ap.add_argument("--roweq", default="", help="kinds whose Muon step gets equal row lengths, same Frobenius size (U-NorMuon / Aurora), e.g. up")
    ap.add_argument("--perhead", default="", help="kinds whose update is orthogonalized per attention head, e.g. q,k,v,o")
    ap.add_argument("--hyper", type=float, default=0.0, help="per-matrix lr: log-multiplier += h * cos(this gradient, last step); feedback line search")
    ap.add_argument("--hyper_clip", type=float, default=10.0, help="hyper multiplier clamped to [1/c, c]")
    ap.add_argument("--slow", type=float, default=0.0, help="AdEMAMix-style slow momentum weight alpha (Muon input = Nesterov + alpha * slow EMA)")
    ap.add_argument("--slow_beta", type=float, default=0.999)
    ap.add_argument("--filt_late", type=float, default=0.0, help="spectral filter only in the last share of steps")
    ap.add_argument("--bclamp", type=float, default=0.0, help="budget multipliers clamped to [1/c, c]")
    ap.add_argument("--budget_on", default="", help="budget only on these matrix kinds, e.g. q,k")
    ap.add_argument("--rel", default="", help="matrix kinds whose step x (current norm / initial norm)")
    ap.add_argument("--renorm", default="", help="matrix kinds rescaled back to their initial norm after every step")
    ap.add_argument("--mult", default="", help="fixed step multipliers per kind, e.g. q=4,k=4")
    ap.add_argument("--trainq", default="", help="train with weights and linear inputs rounded every step: fp8, int8, int6, int4 (straight-through)")
    ap.add_argument("--ema", type=float, default=0.0, help="also keep an EMA of the weights with this decay, print its val")
    ap.add_argument("--teacher", default="", help="distill from ckpt/<teacher>_muon.pt (soft targets)")
    ap.add_argument("--qat", type=float, default=0.0, help="last share of steps: forward with 3-bit weights (straight-through)")
    ap.add_argument("--label_noise", type=float, default=0.0, help="share of training targets replaced by random bytes")
    ap.add_argument("--init_std", type=float, default=0.1)
    ap.add_argument("--ctx", type=int, default=0, help="training sequence length (default 64)")
    ap.add_argument("--spike", type=int, default=0, help="N batches of random bytes at mid-training")
    ap.add_argument("--data_kb", type=int, default=0, help="train only on the first N KB (repeated data)")
    ap.add_argument("--init", default="", help="start from ckpt/<init>_<opt>.pt (fresh optimizer state)")
    ap.add_argument("--dseed", type=int, default=0, help="changes only the batch order, same init")
    ap.add_argument("--switch", type=float, default=0.0, help="last share of steps: matrices switch from Muon to AdamW 1e-2")
    a = ap.parse_args(argv)
    if a.lr is None: a.lr = 0.02 if a.opt == "muon" else 1e-2
    return a


def load(tag, opt):
    d = torch.load(os.path.join(CK, f"{tag}_{opt}.pt"))
    m = LM(d["width"], d["layers"], d["qknorm"]); m.load_state_dict(d["sd"]); return m


if __name__ == "__main__":
    a = train_args(sys.argv[1:]); POWER = a.power
    if a.head: HEAD = (int(a.head.split(":")[0]), float(a.head.split(":")[1]))
    if a.filt: FILT = (a.filt.split(":")[0], float(a.filt.split(":")[1]))
    if a.cmd == "train":
        log = []
        m, sec = train(a, model=load(a.init, a.opt) if a.init else None, log=log)
        torch.save(dict(sd=m.state_dict(), width=a.width, layers=a.layers, qknorm=not a.no_qknorm), os.path.join(CK, f"{a.tag}_{a.opt}.pt"))
        r = dict(tag=a.tag, opt=a.opt, lr=a.lr, val=log[-1][1], sec=round(sec, 1), curve=log)
        open(os.path.join(HERE, "train.jsonl"), "a").write(json.dumps(r) + "\n")
        print(json.dumps({k: v for k, v in r.items() if k != "curve"}), log[::2] if not a.spike else log)
    elif a.cmd == "probe":
        import probes
        out = {o: getattr(probes, a.name)(load(a.tag, o)) for o in ("muon", "adamw") if os.path.exists(os.path.join(CK, f"{a.tag}_{o}.pt"))}
        open(os.path.join(HERE, "probes.jsonl"), "a").write(json.dumps(dict(probe=a.name, tag=a.tag, **out)) + "\n")
        for o, v in out.items(): print(o, v)
