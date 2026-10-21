# Results: per-head vs whole-matrix Muon on attention q,k

**Whole-matrix Muon** orthogonalizes each q / k matrix as one piece (polar factor U V^T of the momentum).
**Per-head Muon** takes the polar factor of each head's block (rows of that head) separately and stacks them.
Per-head is the greedier rule: each head gets its own best-aligned step.

For a square q (64 x 64, 4 heads of 16 rows), every head block of the whole-matrix step already has 16 orthonormal rows, so both rules give every head the same step size and the same total (Frobenius) size.
The differences are direction and spectral size: whole-matrix Muon makes all 64 rows of the step mutually orthogonal, so the 4 heads must split the 64 input directions between them; per-head steps drop that constraint, and stacked they can reach spectral norm up to sqrt(heads) instead of 1.

All numbers are from exact-SVD orthogonalization (no Newton-Schulz error).

- [Toy model](#toy-2-layer-attention-only-model-task-back)
- [Byte LM](#lm-byte-level-4-layers-width-64)
- [Literature](#literature-on-per-head-muon-vs-these-results)
- [How to run](#how-to-run)

## Toy: 2-layer attention-only model, task "back"

Task back: predict the token 4 positions back (48 tokens, vocab 64). One head that looks at a fixed offset solves it.
Model: 2 layers, attention only, 4 heads, width 64, learned positions, QK-norm.
Muon (Nesterov momentum 0.95, lr 0.03) on q, k, v, o; AdamW on the rest. Only the q,k rule changes between runs.
"Solved step" = first step where big-batch loss < 0.5 (checked every 5 steps), mean of 4 seeds unless noted. Lower = faster.

### Per-head wins early, loses later (`curve.py`)

Big-batch loss, per-head minus whole (negative = per-head ahead), 4 seeds:
- steps 10-70: per-head ahead on all 4 seeds (best -0.033 to -0.064 around step 50-60).
- crossover at step 70-90; then per-head far behind during whole's takeoff (+0.96 to +2.15 at step 120).
- step 150: both near zero loss (per-head reaches it about 20 steps later).

Same shape as the LM below and as arXiv 2605.08933 Fig 1. The toy's crossover comes when the look-back pattern has to be learned.

### The per-head step pushes a few directions much harder (`path.py RULE SEED`, 125 steps)

| q,k rule | final loss, seeds 0-3 | step's top-4 directions vs average size (layer-1 q) |
|---|---|---|
| whole Muon | 0.05 0.02 0.04 0.42 | 1.00 |
| per-head | 1.17 1.95 2.01 2.10 | 2.3 |
| per-head, then msign of the full matrix (`phw`) | 0.04 0.02 0.07 0.18 | 1.00 |
| whole, momentum's top-4 directions x2.3, same total size (`boost`) | 1.42 1.78 1.72 2.14 | 2.3 by construction |

Flattening per-head's step (`phw`) restores full speed; adding a similar pile-up to whole Muon (`boost`) creates the slowdown.
Follow-up (`same.py`, rule `boost2`): per-head's top-4 piled directions overlap the momentum's top-4 by 0.4-0.58 (random 0.06), so `boost` is only partly the same thing. Boosting per-head's OWN top-4 inside whole Muon (to 2.1x, same total size): final loss 0.48 0.84 0.96 1.25 (seeds 0-3), between whole and per-head. The pile-up explains part of the slowdown, not all; `phw` still fully restores speed.

Ruled out on the way: per-step gain (at the same weights a per-head step lowers big-batch loss MORE: `onestep.py`), exploration (`NOISE=` on q,k momentum hurts both rules), coverage (`run.py --phv disjoint`, each head limited to its own input quarter, is slowest), path length (per-head moves farther and straighter: `path.py`).

### Recipes mixing per-head and whole (`recipe.py`)

Solved step: whole 121.2 | per-head 138.8 | pairs of heads 136.2 | switch per-head -> whole at step 30: 137.5, at 50: 142.5, at 70: 142.5 | 1 in 5 steps whole 137.5 | 1 in 2 whole 132.5 | blend 50/50 133.8 | blend fading per-head 1 -> 0 by step 70: 137.5.

No recipe beats whole: any per-head share delays the takeoff, roughly in proportion to how much per-head is used. The early per-head lead (steps 10-70) does not carry over after a switch.

### Why early per-head steps delay the toy (`why_delay.py`, `attn.py`, `posshare.py`, `toward.py`)

- Solved step: whole 121.2; per-head first 10 steps 126.2; first 30: 137.5; first 30 then reset q,k to init 126.2; first 30 then flatten the q,k change (same size, all directions equal) 122.5; first 50 then flatten 131.2; first 70 then flatten 140.0; per-head always with flatten every 5 or 10 steps 142.5; per-head first 30 in layer 1 only 128.8, layer 2 only 125.0.
- Attention (`attn.py`): steps 0-80 identical in both rules (no look-back yet, best head's weight on offset -4 = 0.06-0.07); look-back starts at about step 80-100 with whole, later with per-head. The early loss drop is not look-back.
- Layer-1 q,k step share on input directions (`posshare.py`, 8 of 64 dims; whole = exactly 0.125 everywhere because its step is orthogonal): per-head puts 0.28 on token-identity directions (2.3x) and 0.12-0.14 on position directions at steps 20-40; token share falls and position share rises to 0.21-0.22 by step 100.
- Progress toward a final solution (`toward.py`): inconclusive, each rule moves fastest toward its own end point.
- After about step 50, flattening no longer recovers the lost steps (not explained).

### Cause found: per-head makes heads crowd onto the same input directions (`retoken.py`, `twins.py`, `overlap_t.py`, `apart.py`)

| test | solved step |
|---|---|
| whole | 121.2 |
| per-head first 30 | 137.5 |
| per-head 30, token-direction share of layer-1 q,k step cut to even (0.125) | 136.2 (no fix) |
| whole 30, token share raised to 0.28 | 126.2 |
| per-head 30, random 8 dirs cut to even (control) | 137.5 |
| per-head 30, then heads' q,k changes made to use separate input directions (sizes kept) | 126.2 (most of the fix) |
| per-head 30, then even out directions inside each head only | 133.8 (small) |

Head overlap of the q,k change row spaces (`overlap_t.py`, mean over head pairs; random 16-dim subspaces in 64 dims = 0.25): whole 0.13 at step 10 -> 0.17 at 30 -> 0.25 at 100; per-head 0.38 at step 10, 0.33-0.36 at 100, all four matrices, both seeds.

Optimizer "per-head, heads kept apart" (`apart.py`: per-head msign, each head first projected off earlier heads' step directions):

| rule | loss at step 50 | solved step, seeds 0-3 | seeds 4-11 |
|---|---|---|---|
| whole | 3.534 | 121.2 | 115.0 (loss@50 3.525) |
| per-head | 3.487 | 138.8 | |
| per-head, heads kept apart | 3.578 | 118.8 | 113.1 (loss@50 3.571) |

Reading: the early lead and the late delay are one mechanism. Per-head lets all heads push the same useful directions at once (crowding above random), which lowers loss early; the look-back solution needs heads to differ, so crowded heads must first separate. Removing the crowding removes both the lead and the delay. Whole Muon pushes heads apart (overlap below random) from the start.

### Hypotheses tested and ruled out

**1. Token pile-up.** Per-head piles 2.3x of its step onto token-identity input directions while the task needs position. Cutting that share to even (`retoken.py`) gives 136.2 vs 137.5: no fix. Raising whole's token share to 0.28 costs only 5 steps (126.2). Not the cause.

**2. "Heads must unlearn being copies"** (`same_heads.py`). Head sameness = mean cosine between heads' centered pre-softmax attention scores. Seed 0 / seed 1, layer 1 / layer 2:
- Step 30-50: whole 0.03/0.04 and 0.05/0.18; per-head 0.07/0.04 and 0.09/0.36. Both near zero in layer 1: early per-head heads do not act as copies.
- Sameness rises in both rules as the task is learned (the back-4 solution has every head looking at offset -4): whole s0 at 130 0.45/0.88, s1 at 140 0.72/0.37; per-head s0 at 120 0.73/0.87.
- Per-head is slower to form layer-1 look-back (s0 at 120: 0.14 vs whole 0.51; s1 at 120: 0.20 vs 0.83).

The weight-space crowding (row-space overlap 0.38 vs 0.13) is real and removing it removes most of the delay, but it does not show up as heads behaving alike. The link between weight crowding and the slower look-back is still unknown.

**3. "Per-head steps get worse late"** (`stepgain.py`). Along a whole-Muon run, same weights and momentum, one q,k step each (lr 0.03), big-batch loss drop, per-head minus whole (x1000): seed 0 +0.2 at step 10, +1.6 at 50, +1.5 at 80, +7.2 at 100, +10.0 at 110 (takeoff), +0.2 at 140; seed 1 the same shape (+10.1 at 110). Per-head's single step is better at every point, most during the look-back takeoff. Per-head loses because of where its own earlier steps took the weights (a path effect), matching reset or flatten at step 30 removing the delay.

## LM: byte-level, 4 layers, width 64

Model: byte LM, 4 layers, width 64, 4 heads, RoPE, QK-norm, pre-norm RMSNorm, squared-ReLU MLP, tied embeddings, context 64.
Muon: exact-SVD msign of Nesterov momentum 0.95 on attention and MLP matrices; embeddings and norms AdamW 3e-3.
Data: an 8.4 MB English text corpus (not released; `lm/prepare_data.py` builds a public substitute, so absolute losses will differ).
"Seed" = `--seed` (init and batch order) in the flat, run-length and dial tables; "batch order" = `--dseed` (same init, different batches) in the QK-norm, head-count, early-gain and VNORM tables. Every run length is its own run (the lr schedule spans the run).

### The per-head gain is the pile-up (`--ph_mode flat`)

Val loss, seeds 0-2 (100 steps about 4 s, 200 steps about 7 s on a laptop CPU):

| q,k rule | 100 steps | 200 steps |
|---|---|---|
| Muon | 2.8374 2.8173 2.8341 | 2.5674 2.5394 2.5541 |
| per-head q,k (`--perhead q,k`) | 2.8243 2.7993 2.8059 | 2.5349 2.5195 2.5137 |
| per-head q,k, then msign of the full matrix (`--perhead q,k --ph_mode flat`) | 2.8387 2.8167 2.8355 | 2.5671 2.5401 2.5584 |

Per-head wins on every seed already at 100 steps (-0.013 to -0.028; at 200: -0.020 to -0.040). Flattening the full matrix removes the whole gain. Same intervention as the toy's `phw`: in the toy the heads' shared oversized directions slow learning, in the LM they are exactly what helps.

### Does Muon overtake per-head with longer training?

Gap = per-head minus Muon val loss:

| steps | seed 0 | seed 1 |
|---|---|---|
| 100 | -0.013 | -0.018 |
| 200 | -0.032 | -0.020 |
| 400 | -0.038 | -0.025 |
| 800 | -0.016 | -0.036 |
| 1600 | -0.013 | -0.017 |
| 3200 | -0.008 | -0.012 |
| 6400 | +0.006 | +0.002 |

- 3200 steps: Muon 1.7812 / 1.7850, per-head 1.7730 / 1.7728; seed 0 in-run gap 0.000 at steps 2000-2400.
- 6400 steps: Muon 1.6941 / 1.6932, per-head 1.6996 / 1.6953; in-run gap +0.005 / -0.004 at step 4000.
- Muon is ahead on both seeds at 6400 but by less than the ~0.01 batch-order spread: consistent with a crossover near 4000-6400 steps, not yet shown beyond noise.
- Compressed setups do not reach the late regime in 400 steps: lr 0.04 gap -0.045 / -0.039 and growing; width 32 -0.035 / -0.020; `--data_kb 8` overfits (val 5.2-5.5), not usable.

### QK-norm decides whether per-head helps (val set 4x bigger: `VALN=16`, batch orders `--dseed`)

400 steps (Muon 2.2642 / 2.2592):

| per-head on | val |
|---|---|
| q, k | 2.2228 / 2.235 |
| q, k, v, o | 2.2569 |
| v, o | 2.2799 (worse) |
| q, k without QK-norm (Muon 2.3264) | 2.313 |

1600 steps, lr 0.014, batch orders 0 / 1 / 2:

| run | QK-norm | val | mean vs Muon |
|---|---|---|---|
| Muon | yes | 1.8615 / 1.8548 / 1.8765 | 1.8643 |
| per-head q, k | yes | 1.8494 / 1.8479 / 1.8700 | -0.0085, better on all 3 |
| per-head, step divided by its spectral norm (Muon's spectral size) | yes | 1.8635 / 1.8577 / 1.8761 | +0.0015 |
| Muon, q / k step x2 | yes | 1.8582 / 1.8545 / 1.8743 | -0.002 |
| Muon, q / k step x3 | yes | 1.8552 / 1.8552 / - | -0.003 |
| Muon | no | 1.943 / 1.9368 | 1.940 |
| per-head q, k | no | 1.9786 / 1.9633 | +0.031 (worse) |
| per-head, divided by spectral norm | no | 1.9097 / 1.8895 | -0.040 |
| Muon, q / k step x0.6 | no | 1.9086 / 1.8937 | -0.039 |
| Muon, q / k step x0.4 | no | 1.9044 / 1.8893 | -0.043 |

- Both start ahead (-0.05 at step 200). With QK-norm the lead shrinks to about -0.01 at 1600; without QK-norm it flips at step 600 and ends +0.03.
- Without QK-norm the per-head step has 2x the spectral norm (1.5 vs 0.76 per unit lr), W_q's top singular value ends 7.0 vs 4.7, max attention logits 520 vs 270 in the last layer. q / k simply want 0.4-0.6x Muon's step there; dividing per-head by its spectral norm is that smaller step (it equals the x0.6 control).
- With QK-norm the gain needs both parts: per-head directions at Muon's spectral size give nothing; a bigger whole-matrix step (x2, x3) gives about a quarter of it.

### More heads, bigger gain (`HEADS=2/4/8`, lr 0.02, `--beta 0.9`, 1600 steps, batch orders 0 / 1)

| heads (head size) | Muon | per-head q, k | gain | Muon q, k step x sqrt(heads) | per-head at Muon's spectral size |
|---|---|---|---|---|---|
| 2 (32) | 1.8683 / 1.8559 | 1.863 / 1.8509 | -0.005 | | |
| 4 (16) | 1.853 / 1.839 | 1.8424 / 1.8374 | -0.0065 | x2: 1.8582 / 1.8545 at momentum 0.95, lr 0.014 (a quarter of the gain) | +0.0015 (momentum 0.95) |
| 8 (8) | 1.8867 / 1.8624 | 1.8616 / 1.8477 | -0.020 | x2.8: 1.8754 / 1.8547 (-0.0095) | 1.878 / 1.8551 (-0.008) |

Whole-matrix Muon gets worse with more, smaller heads; per-head recovers most of it. At 8 heads both parts (each head's own directions, and the larger stacked step) give about half each.
Head groups at 8 heads: whole matrix 1.8867 / 1.8624, 2 groups of 4 heads 1.871 / 1.8597, 4 groups of 2 heads 1.8582 / 1.8486, per-head 1.8616 / 1.8477. Groups of 2 match single heads.

### The gain is made early

Batch order 0: per-head the whole run 1.8424, first half only 1.8414, first 200 steps only 1.8449 (Muon 1.853). First 400 steps only, batch orders 1 / 2: 1.8349 / 1.858 (whole run 1.8374 / 1.8586, Muon 1.839 / 1.871): the full gain on both.
Per-head v,o only in the first 200 steps: 1.8541 / 1.8438, no gain (whole-run per-head v,o hurts).

### Lopsidedness dial and direction control (`--ph_mode pow:a`, `spec:top`, `spec:rand`)

`pow:a` = per-head step's own directions, singular values raised to the power a, same Frobenius size (0 = flat, 1 = per-head). 1600 steps, seed 0, gap vs Muon (1.8844):

| a | 100 | 200 | 400 | 800 | 1200 | 1599 |
|---|---|---|---|---|---|---|
| 0 | +0.002 | -0.006 | -0.015 | -0.003 | -0.013 | -0.010 |
| 0.5 | -0.006 | -0.031 | -0.016 | -0.001 | -0.009 | -0.003 |
| 1 | -0.010 | -0.037 | -0.032 | -0.014 | -0.016 | -0.009 |
| 2 | -0.016 | -0.046 | -0.038 | -0.022 | -0.025 | -0.016 |
| 3 | -0.017 | -0.043 | -0.038 | -0.025 | -0.021 | -0.009 |

More lopsided = bigger early lead (monotone to a=2). Late: a=2 still best at 1599; a=3 loses its lead fastest (-0.043 -> -0.009). One seed, ~0.01 noise (a=1 should equal per-head; it gives -0.009 vs -0.013 from SVD float drift). "More lopsided = worse finish" is not supported by 1600 steps.

Same lopsidedness on other directions, 400 steps (Muon 2.2706 / 2.2771), gap vs Muon:
- per-head's own directions: -0.038 / -0.025
- per-head's singular-value spread on the momentum's own top directions (`spec:top`): -0.019 / -0.012
- the same spread on shuffled momentum directions (`spec:rand`): +0.011 / +0.006 (worse than Muon)

Lopsidedness helps only along the right directions; per-head's own (head-shared) directions beat the momentum's top directions by about 2x.

### Why q,k and not v,o? Two explanations tested, both failed

Other 400-step screens (momentum 0.9, Muon 2.2186): per-head q, k 2.180; [q_h; k_h] jointly per head 2.195; per-head v, o 2.231 (per-head v, o at Muon's spectral size 2.254, v, o step x0.5 2.250: the per-head directions there are neutral, the size change hurts); MLP split into 4 blocks 2.250, 16 blocks 2.283.

The one-step whitening gain (sum of block nuclear norms / whole nuclear norm, the gain in arXiv 2605.08933's theory) does not predict which matrices gain from splitting: 1.23-1.30 for q, k, 1.32 for v, 1.65-1.72 for up / down, with the same spectral cost (about 2) for all. The largest-gain matrices are the ones where splitting hurts.

**1. "Units need their own norm."** RMSNorm on each head's v (`VNORM=1`), so v heads are normed like q,k heads. 1600 steps, batch orders 0 / 1:

| VNORM=1 | val |
|---|---|
| Muon | 1.8515 / 1.8529 |
| per-head q,k | 1.8452 / 1.8394 |
| per-head q,k,v | 1.8488 / 1.8471 |
| per-head q,k,v,o | 1.8615 / 1.8591 |

Per-head v still hurts (+0.004 / +0.008 vs q,k only) with its own norm.

**2. "q,k heads need the same input directions, v heads need different ones."** Plain Muon: from each step's momentum, overlap of the heads' top 4 input directions (o: output directions; random = 0.0625) and the cosine between each head's block of the whole-matrix polar step and that head's own polar step (1 = whole-matrix Muon costs the head nothing):

| step (4 heads) | q overlap / cos | k | v | o |
|---|---|---|---|---|
| 50 | 0.448 / 0.703 | 0.407 / 0.701 | 0.566 / 0.682 | 0.553 / 0.697 |
| 200 | 0.295 / 0.676 | 0.336 / 0.703 | 0.451 / 0.683 | 0.361 / 0.721 |
| 800 | 0.269 / 0.717 | 0.281 / 0.738 | 0.490 / 0.725 | 0.324 / 0.764 |
| 1599 | 0.291 / 0.736 | 0.285 / 0.753 | 0.408 / 0.748 | 0.223 / 0.785 |

8 heads: same picture, cos 0.60-0.76. v heads overlap the most, not q,k, and whole-matrix Muon bends every kind's per-head step by the same amount (cos about 0.7). The coupling is equally strong for q, k, v, o; what differs is how the network responds to it. Open: why decoupled steps help heads whose output is a softmax score (q,k) and hurt heads whose output is mixed and added (v,o).

Under whole-matrix Muon every head already gets exactly the same step size (spread 0.000 at every check): the coupling is direction, not magnitude. Head usefulness is more lopsided under Muon (loss increase when one head is removed: layer 1 0.021-0.357 vs 0.025-0.216 per-head; layer 3 0.022-0.072 vs 0.033-0.047).

(The probes behind this section, the logit check, whitening gain and head-overlap measurements, are not in this repo; the training runs are all `lm/lab.py`.)

## Literature on per-head Muon vs these results

| source | what they did / claim | fits ours? |
|---|---|---|
| Grouping heads, arXiv 2605.08933 (GPT-2 Small, modded-nanogpt) | Fig 1: head-wise faster early, full-matrix better later. Final val: full-QKV 3.2780, head-wise QKV 3.2799, QK random groups of 6 heads 3.2722 (best); V grouping no consistent gain. Theory: gain = sum of per-head nuclear norms - full nuclear norm (better first-order alignment), cost = larger update Frobenius norm (needs Newton-Schulz error / noise; zero for exact full-rank polar). | Early lead, shrinking lead, q,k not v, per-step greedier (`onestep.py`, `stepgain.py`): yes. Reversal: within noise at 6400 steps on the LM, clear on the toy. Their cost is zero with exact SVD, yet the lead still shrinks, so the cost here is the spectral pile-up / head crowding, not Frobenius. |
| Y. Zhang, arXiv 2608.16760, Sec 3 | Hessian of Q and K is block-diagonal by head; MLP/V/O blocks are per row. Recommends head-wise Muon for Q,K; says DeepSeek tested and adopted it. | Explains q,k yes / v,o no. Nothing on the toy reversal or the shrinking lead. |
| DeepSeek-V4.1-Flash, arXiv 2609.19969 Sec 2.5 | Head-wise Muon on Q,K only; "we observe" it beats vanilla; no numbers. | Q,K only: yes. |
| Kimi K3, arXiv 2607.24653 Sec 2.5 | Per-head on Q,K,V; reason: big-gradient heads dominate, small heads under-normalized; no ablation numbers. | With exact polar and H*dk <= d every head block of the full-matrix step already has equal size, so this reason needs inexact Newton-Schulz; exact SVD here still gains. V per-head: no gain here. |
| modded-nanogpt PR 253 (merged) | K and V split into head-pair groups: loss 3.2788 vs 3.2800 at 1480 steps. | Small K gain; V part differs from ours. |

Not found in this literature: the dependence on QK-norm (without it per-head loses, +0.031), the gain growing with head count, the gain being made in the first ~400 steps, and the head-crowding mechanism.

## How to run

Toy scripts run from `toy/` (they import `run.py`; QK-norm is on by default in each). CPU, seconds to a minute each.

| script | what it prints | command |
|---|---|---|
| `run.py` | one training run (task copy or back), accuracy curve | `QKNORM=1 python3 run.py --opt muon --lr 0.03 --task back --perhead qk --steps 150` |
| `curve.py` | per-head minus whole loss every 10 steps, 4 seeds | `python3 curve.py` |
| `onestep.py` | one-step loss drop of each q,k rule along a run | `python3 onestep.py whole` |
| `path.py` | loss, step direction, distance moved, pile-up size | `python3 path.py perhead 0` (rules: whole, perhead, phw, boost, boost2, shared) |
| `same.py` | overlap of per-head's top directions with the momentum's | `python3 same.py 0` |
| `recipe.py` | solved step for per-head / whole mixes | `python3 recipe.py` |
| `why_delay.py` | solved step after per-head for S steps, then reset / flatten | `python3 why_delay.py` |
| `attn.py` | look-back attention and peakedness every 10 steps | `python3 attn.py` |
| `posshare.py` | step share on token vs position input directions | `python3 posshare.py` |
| `toward.py` | progress toward a finished run's solution | `python3 toward.py` |
| `retoken.py` | solved step with the token share changed | `python3 retoken.py` |
| `twins.py` | solved step after splitting heads apart / evening out inside heads | `python3 twins.py` |
| `overlap_t.py` | head row-space overlap every 10 steps | `python3 overlap_t.py` |
| `apart.py` | per-head with heads kept apart vs whole vs per-head | `python3 apart.py` |
| `same_heads.py` | head sameness and look-back every 10 steps | `python3 same_heads.py 0 1` |
| `stepgain.py` | per-head minus whole one-step gain along a whole run | `python3 stepgain.py` |

LM runs from `lm/` after `python3 prepare_data.py` (or `DATA=your_file.txt`):

| run | command |
|---|---|
| Muon | `python3 lab.py train --opt muon --steps 400 --seed 0` |
| per-head q,k | `python3 lab.py train --opt muon --steps 400 --seed 0 --perhead q,k` |
| per-head, then flatten | `python3 lab.py train --opt muon --steps 200 --perhead q,k --ph_mode flat` |
| lopsidedness dial | `python3 lab.py train --opt muon --steps 1600 --perhead q,k --ph_mode pow:2` |
| spread on momentum directions | `python3 lab.py train --opt muon --steps 400 --perhead q,k --ph_mode spec:top` (or `spec:rand`) |
| no QK-norm | add `--no_qknorm`; more heads: `HEADS=8`; bigger val set: `VALN=16`; other lr / width: `--lr 0.014 --width 128` |

`lab.py` has many other switches from unrelated experiments (all off by default); `python3 lab.py -h` lists them. The `probe` command needs a probes module that is not included.
