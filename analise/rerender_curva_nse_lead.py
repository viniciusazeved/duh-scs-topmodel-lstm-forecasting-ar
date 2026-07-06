"""Re-render da Fig. 2 (curva NSE x lead) do Paper 2 com o benchmark de PERSISTENCIA (B2,
Opcao A) e SEM titulo interpretativo em PT (nit: figura neutra, master EN).

Espelha a figura do consolida_ar.py (mesmas curvas/estilos das 6 configs AR), acrescenta a
curva de persistencia ingenua (outputs/persistencia_baseline.json) e salva direto em
overleaf/figuras/curva_nse_lead_AR.png. Nenhum numero novo: le os mesmos results.json.
"""
import json, glob
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

BASE = r"D:/TTD_SCS_LSTM/forecasting_v2/outputs/grade_ar"
PERS = r"D:/TTD_SCS_LSTM/forecasting_v2/outputs/persistencia_baseline.json"
OUT = Path(r"D:/Artigo_JOH/artigo_forecasting/overleaf/figuras/curva_nse_lead_AR.png")
H = ["1h", "3h", "6h", "12h", "24h"]; HN = [1, 3, 6, 12, 24]

data = defaultdict(lambda: defaultdict(dict))
for f in glob.glob(BASE + "/*/seed*/results.json"):
    d = json.load(open(f)); name = d["model_name"]; sd = d["seed"]
    for h in H:
        data[name][sd][h] = d["test_by_horizon"][h]["nse"]

def curve(name):
    seeds = sorted(data[name])
    M = np.array([[data[name][s][h] for h in H] for s in seeds])
    return M.mean(0), M.std(0)

pers = json.load(open(PERS))["naive"]
pers_y = [pers[h] for h in H]

rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                 "font.size": 9, "axes.linewidth": 0.8, "savefig.dpi": 300})
fig, ax = plt.subplots(figsize=(6.4, 4.2))
estilo = {
    "LSTM_Lumped": ("Lumped", "#7f7f7f", "--"), "LSTM_Lumped_AR": ("Lumped + AR", "#7f7f7f", "-"),
    "LSTM_DUH_Base_Topmodel": ("Topmodel", "#1f4e79", "--"), "LSTM_DUH_Base_Topmodel_AR": ("Topmodel + AR", "#1f4e79", "-"),
    "LSTM_DUH_Base_SCS": ("SCS", "#c55a11", "--"), "LSTM_DUH_Base_SCS_AR": ("SCS + AR", "#c55a11", "-"),
}
# benchmark de persistencia ingenua (sem treino): preto, pontilhado, marcador losango
ax.plot(HN, pers_y, ":", color="black", lw=1.8, marker="D", ms=4, label="Naive persistence", zorder=5)
for name, (lab, col, ls) in estilo.items():
    if name not in data: continue
    m, s = curve(name)
    ax.plot(HN, m, ls, color=col, lw=1.8 if ls == "-" else 1.3, label=lab,
            marker="o" if ls == "-" else "s", ms=4, alpha=0.95 if ls == "-" else 0.6)
    ax.fill_between(HN, m - s, m + s, color=col, alpha=0.08)
ax.set_xlabel("Lead time (h)"); ax.set_ylabel("NSE (test)")
ax.set_xticks(HN); ax.set_xlim(0, 25); ax.set_ylim(0.3, 1.0)
ax.legend(frameon=False, fontsize=7.5, ncol=4, loc="lower left", columnspacing=1.0)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
fig.tight_layout(); fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("salvo:", OUT)
print("persistencia:", dict(zip(H, [round(v, 3) for v in pers_y])))
