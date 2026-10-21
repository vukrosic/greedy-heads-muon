"""Draw gap.png: per-head minus whole-matrix Muon val loss vs run length (LM, two batch orders). Numbers from RESULTS.md."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

steps = [100, 200, 400, 800, 1600, 3200, 6400]
gap = {"seed 0": [-0.013, -0.032, -0.038, -0.016, -0.013, -0.008, +0.006],
       "seed 1": [-0.018, -0.020, -0.025, -0.036, -0.017, -0.012, +0.002]}
fig, ax = plt.subplots(figsize=(7, 4), dpi=150, facecolor="#fcfcfb"); ax.set_facecolor("#fcfcfb")
ax.axhline(0, color="#52514e", lw=1)
for (k, v), c in zip(gap.items(), ["#2a78d6", "#eb6834"]):
    ax.plot(steps, v, color=c, lw=2, marker="o", ms=6, label=k)
ax.set_xscale("log"); ax.minorticks_off(); ax.set_xticks(steps); ax.set_xticklabels([str(s) for s in steps])
ax.set_xlabel("training steps (each length is its own run)", color="#52514e")
ax.set_ylabel("val loss: per-head minus Muon", color="#52514e")
ax.set_title("Per-head q,k leads early; the lead shrinks with training", color="#0b0b0b", loc="left")
ax.text(110, 0.003, "above 0 = whole-matrix Muon better", color="#52514e", fontsize=8)
ax.text(110, -0.043, "below 0 = per-head better", color="#52514e", fontsize=8)
ax.set_ylim(-0.046, 0.012)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color("#c3c2b7")
ax.tick_params(colors="#52514e"); ax.grid(axis="y", color="#e6e5e1", lw=0.8); ax.set_axisbelow(True)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout(); fig.savefig("gap.png", facecolor=fig.get_facecolor())
