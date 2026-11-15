"""Step 4: one figure for the two findings. Reads per_patient.csv and
cn_composition.csv written by 02_reproduce.py."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pp = pd.read_csv("per_patient.csv", index_col=0)               # (12 patients, ...)
comp = pd.read_csv("cn_composition.csv", index_col=0)          # (8 CNs, 16 types)
m2_cn = comp["M2"].idxmax()

panels = [
    (f"share_{m2_cn}", f"Share of cells in the M2-macrophage\nneighborhood ({m2_cn}, {comp.loc[m2_cn, 'M2']:.0%} M2)"),
    ("type_CD8", "CD8 T cells as a share of all cells"),
]
fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
for ax, (col, title) in zip(axes, panels):
    for i, (grp, colour) in enumerate([("N", "#4C78A8"), ("Y", "#E45756")]):
        vals = pp.loc[pp["Relapse"] == grp, col]
        ax.scatter([i] * len(vals), vals, color=colour, s=40, alpha=0.8)
        ax.hlines(vals.median(), i - 0.2, i + 0.2, color="black", lw=2)
        for name, v in vals.items():
            ax.annotate(name, (i + 0.06, v), fontsize=7, color="gray")
    ax.set_xticks([0, 1], ["No relapse (n=4)", "Relapse (n=8)"])
    ax.set_xlim(-0.5, 1.7)
    ax.set_title(title, fontsize=10)
fig.suptitle("Per patient, from the public MIBI per-cell table (Zenodo 15109931)", fontsize=9)
fig.tight_layout()
fig.savefig("findings.png", dpi=180)
print("wrote findings.png")
