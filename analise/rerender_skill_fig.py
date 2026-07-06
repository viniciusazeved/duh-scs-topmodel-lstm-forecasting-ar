"""Re-render da figura de skill da forcante (Fig. skill) em EN neutro (nit: master EN).
Le os CSVs ja salvos (analise/skill_{gfs,ifs}.csv), sem reprocessar h5/parquet, e salva em
overleaf/figuras/skill_forcante_58585000.png. Mesmos dados/curvas do skill_fig.py; so os
rotulos e titulos passam de PT para EN (identicos aos da legenda .tex do Paper 2)."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

A = Path(r"D:/TTD_SCS_LSTM/forecasting_v2/analise")
OUT = Path(r"D:/Artigo_JOH/artigo_forecasting/overleaf/figuras/skill_forcante_58585000.png")
sk = {f: pd.read_csv(A / f"skill_{f}.csv") for f in ("gfs", "ifs")}

rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                 "font.size": 9, "axes.linewidth": 0.8, "savefig.dpi": 300})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.6, 1]})
C = {"gfs": "#1f4e79", "ifs": "#c55a11"}
for f in ("gfs", "ifs"):
    d = sk[f]
    ax1.plot(d.lead_h, d.pearson, "-", color=C[f], lw=1.6, label=f"{f.upper()} (Pearson)")
    ax1.plot(d.lead_h, d.spearman, "--", color=C[f], lw=1.1, alpha=0.7, label=f"{f.upper()} (Spearman)")
ax1.axvspan(0, 24, color="0.85", alpha=0.5, zorder=0)
ax1.text(12, 0.86, "forecasting\nhorizon (≤24 h)", ha="center", va="top", fontsize=7, color="0.35")
ax1.set_xlabel("Lead time (h)"); ax1.set_ylabel("Correlation (forecast vs observed)")
ax1.set_xlim(0, 168); ax1.set_ylim(0, 0.9); ax1.legend(frameon=False, fontsize=6.5, ncol=1, loc="lower left")
for s in ("top", "right"): ax1.spines[s].set_visible(False)
ax1.set_title("(a) Correlation by lead", fontsize=8.5, loc="left")

for f in ("gfs", "ifs"):
    ax2.plot(sk[f].lead_h, sk[f].bias_ratio, "-", color=C[f], lw=1.4, label=f.upper())
ax2.axhline(1.0, color="0.4", lw=0.8, ls=":")
ax2.set_xlabel("Lead time (h)"); ax2.set_ylabel("Bias (forecast / observed)")
ax2.set_xlim(0, 168); ax2.set_ylim(0.7, 1.9); ax2.legend(frameon=False, fontsize=7)
for s in ("top", "right"): ax2.spines[s].set_visible(False)
ax2.set_title("(b) Magnitude bias", fontsize=8.5, loc="left")
fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("salvo:", OUT)
