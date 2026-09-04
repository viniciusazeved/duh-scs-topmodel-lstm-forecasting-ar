"""Figura Elsevier-style do probe: captura de pico e correlacao por lead/produto."""
from __future__ import annotations
from pathlib import Path
import polars as pl
import matplotlib.pyplot as plt
from matplotlib import rcParams

OUT = Path(r"C:/Users/vinic/AppData/Local/Temp/claude/D--Artigo-JOH/aa28a107-7cab-498d-b037-52aed9439a3c/scratchpad")
met = pl.read_csv(OUT / "wn_peak_metrics.csv")

rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
})

produtos = {
    "gfs":         ("GraphCast-GFS (atual)",       "#000000", "o", "-"),
    "wn_det_hres": ("WeatherNext HRES (determ.)",  "#0072B2", "s", "-"),
    "wn2_mean":    ("WeatherNext 2 (media ens.)",  "#009E73", "^", "-"),
    "wn2_ens_p90": ("WeatherNext 2 (ensemble p90)","#E69F00", "D", "--"),
    "wn2_ens_max": ("WeatherNext 2 (ensemble max)","#D55E00", "v", ":"),
}
leads = [6, 12, 18, 24]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), dpi=200)

for key, (lab, col, mk, ls) in produtos.items():
    sub = met.filter(pl.col("produto") == key).sort("lead_h")
    x = sub["lead_h"].to_list()
    ax1.plot(x, sub["captura_pico_max"].to_list(), color=col, marker=mk, ls=ls,
             ms=4, lw=1.2, label=lab)
    ax2.plot(x, sub["corr"].to_list(), color=col, marker=mk, ls=ls, ms=4, lw=1.2, label=lab)

ax1.axhline(1.0, color="grey", lw=0.7, ls="--", zorder=0)
ax1.set_ylabel("Captura de pico  (max prev / max obs)")
ax1.set_xlabel("lead time (h)")
ax1.set_title("(a) Magnitude do pico de chuva", fontsize=8, loc="left")
ax1.set_xticks(leads)
ax1.text(6.2, 1.05, "pico observado", color="grey", fontsize=6.5)

ax2.set_ylabel("Correlacao com chuva observada")
ax2.set_xlabel("lead time (h)")
ax2.set_title("(b) Sincronia (timing)", fontsize=8, loc="left")
ax2.set_xticks(leads)
ax2.set_ylim(0.3, 0.8)

ax1.legend(fontsize=6, frameon=False, loc="upper right", ncol=1)
fig.suptitle("Rio Preto | janela de teste do Paper 2 (nov/2024-mar/2025) | forcante 6h",
             fontsize=8.5, y=1.02)
fig.tight_layout()
png = OUT / "wn_peak_figura.png"
fig.savefig(png, bbox_inches="tight", dpi=200)
print(f"salvo: {png}")
