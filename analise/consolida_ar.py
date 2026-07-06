"""Consolida a grade AR: curva NSE x lead (divisao de trabalho no horizonte) + Wilcoxon do AR."""
import json, glob, warnings; warnings.filterwarnings("ignore")
import numpy as np
from collections import defaultdict
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

base = r"D:/TTD_SCS_LSTM/forecasting_v2/outputs/grade_ar"
OUT = Path(r"C:/Users/vinic/AppData/Local/Temp/claude/D--Artigo-JOH/bfcf1311-2eda-4171-b909-bd7d34f659a0/scratchpad")
H = ["1h","3h","6h","12h","24h"]; HN = [1,3,6,12,24]

data = defaultdict(lambda: defaultdict(dict))  # data[name][seed][h] = nse
for f in glob.glob(base + "/*/seed*/results.json"):
    d = json.load(open(f)); name = d["model_name"]; sd = d["seed"]
    for h in H:
        data[name][sd][h] = d["test_by_horizon"][h]["nse"]

def curve(name):
    seeds = sorted(data[name])
    M = np.array([[data[name][s][h] for h in H] for s in seeds])  # (nseed, nh)
    return M.mean(0), M.std(0), M

# --- Wilcoxon: AR agrega no NSE@6h? pares (arquitetura, seed) AR - baseline ---
pares = [("LSTM_Lumped","LSTM_Lumped_AR"), ("LSTM","LSTM_AR"),
         ("LSTM_DUH_Base_SCS","LSTM_DUH_Base_SCS_AR"),
         ("LSTM_DUH_Base_Topmodel","LSTM_DUH_Base_Topmodel_AR"),
         ("LSTM_DUH_Base_Topmodel_PeOnly","LSTM_DUH_Base_Topmodel_PeOnly_AR")]
print("=== AR vs baseline (NSE@6h), por arquitetura ===")
deltas_all = []
for base_n, ar_n in pares:
    ss = sorted(set(data[base_n]) & set(data[ar_n]))
    b = np.array([data[base_n][s]["6h"] for s in ss]); a = np.array([data[ar_n][s]["6h"] for s in ss])
    deltas_all += list(a - b)
    print(f"  {base_n:34s} {b.mean():.3f} -> {ar_n:37s} {a.mean():.3f}  (dAR={a.mean()-b.mean():+.3f})")
deltas_all = np.array(deltas_all)
w, p = wilcoxon(deltas_all)
print(f"\nWilcoxon pareado AR-baseline (n={len(deltas_all)}): dAR medio {deltas_all.mean():+.3f}, "
      f"mediana {np.median(deltas_all):+.3f}, W={w:.1f}, p={p:.2e}")

# --- delta AR por horizonte (decai com o lead?) ---
print("\n=== ganho medio do AR por horizonte (media das 5 arquiteturas) ===")
for i,h in enumerate(H):
    ds = []
    for bn, an in pares:
        ss = sorted(set(data[bn]) & set(data[an]))
        ds += [data[an][s][h]-data[bn][s][h] for s in ss]
    print(f"  {h:>3}: dAR medio {np.mean(ds):+.3f}")

# --- figura: curva NSE x lead ---
rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],"font.size":9,
                 "axes.linewidth":0.8,"savefig.dpi":300})
fig, ax = plt.subplots(figsize=(6.4,4.2))
estilo = {  # (label, cor, ls)
    "LSTM_Lumped": ("Lumped","#7f7f7f","--"), "LSTM_Lumped_AR": ("Lumped + AR","#7f7f7f","-"),
    "LSTM_DUH_Base_Topmodel": ("Topmodel","#1f4e79","--"), "LSTM_DUH_Base_Topmodel_AR": ("Topmodel + AR","#1f4e79","-"),
    "LSTM_DUH_Base_SCS": ("SCS","#c55a11","--"), "LSTM_DUH_Base_SCS_AR": ("SCS + AR","#c55a11","-"),
}
for name,(lab,col,ls) in estilo.items():
    if name not in data: continue
    m,s,_ = curve(name)
    ax.plot(HN, m, ls, color=col, lw=1.8 if ls=="-" else 1.3, label=lab,
            marker="o" if ls=="-" else "s", ms=4, alpha=0.95 if ls=="-" else 0.6)
    ax.fill_between(HN, m-s, m+s, color=col, alpha=0.08)
ax.set_xlabel("Lead time (h)"); ax.set_ylabel("NSE (teste)")
ax.set_xticks(HN); ax.set_xlim(0,25); ax.set_ylim(0.3,0.95)
ax.legend(frameon=False, fontsize=7.5, ncol=3, loc="lower left", columnspacing=1.0)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.set_title("Divisão de trabalho no horizonte: AR domina o curto, física sustenta o longo",
             fontsize=8.5, loc="left")
fig.tight_layout(); fig.savefig(OUT/"curva_nse_lead_AR.png", dpi=300, bbox_inches="tight")
print("\nfigura:", OUT/"curva_nse_lead_AR.png")
