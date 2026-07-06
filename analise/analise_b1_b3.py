"""B1 (teto por lead) + B3 (dp na Tab.2 + Wilcoxon SCS vs TOPMODEL) da revisao Paper 2.
Le teste_isolado_gc_bylead.json (guard 6h == canonico, ja validado). Ganhos PAREADOS por
semente (teto[s]-semfut[s], gfs[s]-semfut[s]) e so entao media +/- dp entre as 5 sementes.
"""
import json, os
from collections import defaultdict
import numpy as np
from scipy.stats import wilcoxon

GC = r"D:/TTD_SCS_LSTM/forecasting_v2/outputs"
data = json.load(open(os.path.join(GC, "teste_isolado_gc_bylead.json")))
rows = data["rows"]
LEADS = ["1h", "3h", "6h", "12h", "24h"]
print(f"device={data['device']}  guard max|diff|6h={data['max_abs_diff_6h_vs_canon']:.2e}  n={len(rows)}\n")

# agrupa por celula (model_type, use_ar, train_mode); guarda ganhos por semente e lead
cells = defaultdict(lambda: {"teto": defaultdict(list), "gfs": defaultdict(list), "semfut": defaultdict(list)})
for r in rows:
    key = (r["model_type"], r["use_ar"], r["train_mode"])
    for h in LEADS:
        cells[key]["teto"][h].append(r["nse"]["teto"][h] - r["nse"]["semfut"][h])   # teto-prev pareado
        cells[key]["gfs"][h].append(r["nse"]["gfs"][h] - r["nse"]["semfut"][h])     # gap-prod pareado
        cells[key]["semfut"][h].append(r["nse"]["semfut"][h])                        # baseline

def short(key):
    mt, ar, tm = key
    g = "TOPMODEL" if "topmodel" in mt else "SCS"
    return f"{g:8s} AR={'on ' if ar else 'off'} train={tm:4s}"

order = sorted(cells, key=lambda k: ("topmodel" in k[0], k[1], k[2]))

print("=== B1: TETO DE PREVISIBILIDADE (teto-semfut) POR LEAD -- media+/-dp entre 5 sementes ===")
print(f"{'cell':30s} " + " ".join(f"{h:>14s}" for h in LEADS))
for key in order:
    c = cells[key]["teto"]
    cells_str = " ".join(f"{np.mean(c[h]):+.3f}+/-{np.std(c[h],ddof=1):.3f}" for h in LEADS)
    print(f"{short(key):30s} {cells_str}")

print("\n=== gap-produto GraphCast (gfs-semfut) POR LEAD -- media+/-dp ===")
print(f"{'cell':30s} " + " ".join(f"{h:>14s}" for h in LEADS))
for key in order:
    c = cells[key]["gfs"]
    cells_str = " ".join(f"{np.mean(c[h]):+.3f}+/-{np.std(c[h],ddof=1):.3f}" for h in LEADS)
    print(f"{short(key):30s} {cells_str}")

print("\n=== baseline NSE sem-fut por lead (contexto) ===")
for key in order:
    c = cells[key]["semfut"]
    print(f"{short(key):30s} " + " ".join(f"{np.mean(c[h]):.3f}" for h in LEADS))

# --- B1 destaque: melhor celula (TOPMODEL AR gfs) e as duas TOPMODEL AR ---
print("\n=== B1 DESTAQUE: teto por lead, celulas TOPMODEL AR-on ===")
for key in order:
    mt, ar, tm = key
    if "topmodel" in mt and ar:
        c = cells[key]["teto"]
        print(f"  {short(key)}: " + "  ".join(f"{h} {np.mean(c[h]):+.3f}+/-{np.std(c[h],ddof=1):.3f}" for h in ["6h","12h","24h"]))

# ================= B3 =================
print("\n\n=== B3: Tab.2 @6h -- teto-prev e gap-prod media+/-dp (para a coluna de dispersao) ===")
print(f"{'cell':30s} {'sem-fut':>8s} {'teto-prev@6h':>18s} {'gap-prod@6h':>18s}")
for key in order:
    ct, cg, cs = cells[key]["teto"]["6h"], cells[key]["gfs"]["6h"], cells[key]["semfut"]["6h"]
    print(f"{short(key):30s} {np.mean(cs):8.3f} "
          f"{np.mean(ct):+.3f}+/-{np.std(ct,ddof=1):.3f}   {np.mean(cg):+.3f}+/-{np.std(cg,ddof=1):.3f}")

# Wilcoxon pareado SCS vs TOPMODEL no ganho de teto @6h. Par = mesma (AR, train_mode, seed).
print("\n=== B3: Wilcoxon pareado SCS vs TOPMODEL no ganho de teto @6h ===")
scs_gains, top_gains = [], []
for r in rows:
    pass
# reconstruir por semente: preciso dos ganhos individuais alinhados por (ar, train, seed)
by_seed = {}  # (model_type_short, ar, train, seed) -> teto_gain@6h
for r in rows:
    g = "TOP" if "topmodel" in r["model_type"] else "SCS"
    key = (g, r["use_ar"], r["train_mode"], r["seed"])
    by_seed[key] = r["nse"]["teto"]["6h"] - r["nse"]["semfut"]["6h"]

pairs_scs, pairs_top = [], []
for (g, ar, tm, sd) in list(by_seed):
    if g != "SCS":
        continue
    ktop = ("TOP", ar, tm, sd)
    if ktop in by_seed:
        pairs_scs.append(by_seed[("SCS", ar, tm, sd)])
        pairs_top.append(by_seed[ktop])
pairs_scs, pairs_top = np.array(pairs_scs), np.array(pairs_top)
d = pairs_top - pairs_scs
w, p = wilcoxon(pairs_top, pairs_scs)
print(f"  n pares = {len(d)} (4 celulas AR/train x 5 sementes)")
print(f"  SCS teto-gain@6h medio  = {pairs_scs.mean():+.4f} (dp {pairs_scs.std(ddof=1):.4f})")
print(f"  TOP teto-gain@6h medio  = {pairs_top.mean():+.4f} (dp {pairs_top.std(ddof=1):.4f})")
print(f"  diff (TOP-SCS) media    = {d.mean():+.4f}, mediana {np.median(d):+.4f}")
print(f"  pares com TOP>SCS       = {(d>0).sum()}/{len(d)}")
print(f"  Wilcoxon signed-rank: W = {w:.1f}, p = {p:.3e}")

# padrao qualitativo (as 4 SCS ~0 vs as 4 TOP >0), media por celula
scs_cell_means = [np.mean(cells[k]["teto"]["6h"]) for k in order if "topmodel" not in k[0]]
top_cell_means = [np.mean(cells[k]["teto"]["6h"]) for k in order if "topmodel" in k[0]]
print(f"\n  padrao qualitativo @6h: 4 celulas SCS teto-gain = {[f'{x:+.3f}' for x in scs_cell_means]}")
print(f"                          4 celulas TOP teto-gain = {[f'{x:+.3f}' for x in top_cell_means]}")
