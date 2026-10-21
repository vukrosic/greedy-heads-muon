"""Post charts: per-head Muon's lead over normal Muon (val loss), by run length and inside the 6400-step run."""
import json, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, numpy as np

BG, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C = ["#2a78d6", "#eb6834"]
plt.rcParams.update({"font.size": 13})

def style(ax):
    ax.set_facecolor(BG); ax.axhline(0, color=MUTED, lw=1.2)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)

# 1) lead by run length (each length its own run); lead = normal Muon loss - per-head loss
steps = [100, 200, 400, 800, 1600, 3200, 6400]
gap = {"seed 1": [-0.013, -0.032, -0.038, -0.016, -0.013, -0.008, +0.006],
       "seed 2": [-0.018, -0.020, -0.025, -0.036, -0.017, -0.012, +0.002]}
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150, facecolor=BG); style(ax)
for (k, v), c in zip(gap.items(), C):
    ax.plot(steps, [-x for x in v], color=c, lw=2.5, marker="o", ms=7, label=k)
ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks(steps); ax.set_xticklabels([str(s) for s in steps])
ax.set_xlabel("how long the model was trained (steps)", color=MUTED)
ax.set_ylabel("per-head's lead (lower loss)", color=MUTED)
ax.set_title("Per-head Muon leads early, normal Muon catches up", color=INK, loc="left", fontsize=15)
ax.text(105, 0.041, "above 0: per-head Muon is better", color=MUTED, fontsize=11)
ax.text(105, -0.0085, "below 0: normal Muon is better", color=MUTED, fontsize=11)
ax.set_ylim(-0.011, 0.045)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig("lead_by_length.png", facecolor=BG); plt.close(fig)

# 2) inside the 6400-step runs: lead at each point of training (long_run_curves.json)
D = json.load(open("long_run_curves.json"))
pairs = {k: ({"curve": D[k]["muon"]}, {"curve": D[k]["perhead"]}) for k in ("seed 1", "seed 2")}
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=150, facecolor=BG); style(ax)
for (k, (m, p)), c in zip(pairs.items(), C):
    s = np.array([x[0] for x in m["curve"]]); d = np.array([x[1] for x in m["curve"]]) - np.array([x[1] for x in p["curve"]])
    w = 7; sm = np.convolve(d, np.ones(w) / w, mode="valid"); ss = s[w // 2: w // 2 + len(sm)]
    ax.plot(s[2:], d[2:], color=c, lw=0.8, alpha=0.25)
    ax.plot(ss, sm, color=c, lw=2.5, label=k)
ax.set_xlabel("training step (one 6400-step run)", color=MUTED)
ax.set_ylabel("per-head's lead (lower loss)", color=MUTED)
ax.set_title("Inside the longest run: the lead fades to a tie", color=INK, loc="left", fontsize=15)
ax.text(2300, 0.033, "above 0: per-head Muon is better", color=MUTED, fontsize=11)
ax.text(150, -0.013, "below 0: normal Muon is better", color=MUTED, fontsize=11)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig("lead_in_long_run.png", facecolor=BG); plt.close(fig)

