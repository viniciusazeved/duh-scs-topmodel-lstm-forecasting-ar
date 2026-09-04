"""Estende a analise de skill da forcante (Fig. skill do Paper 2) ao HRES-init
(WeatherNext Graph, GraphCast_operational). Reproduz EXATAMENTE a metodologia de
skill_forcante.py/skill_fig.py e recomputa GFS+IFS para conferir consistencia com o
texto (§2.2). Salva analise/skill_hres.csv e imprime as estatisticas de resumo.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

TELEM = Path(r"D:/TTD_SCS_LSTM/ablacao_skill/data/dataset_58585000_telem.h5")
FORC = Path(r"D:/Graph_Cast/data/forcante")
A = Path(r"D:/TTD_SCS_LSTM/forecasting_v2/analise")
COD = "58585000"

# --- obs de referencia: telem lumped (media 245 ott), IDENTICA a skill_forcante.py ---
with h5py.File(TELEM, "r") as f:
    precip = np.concatenate([f[s]["precipitation"][:] for s in ("train", "val", "test")])
    ts = np.concatenate([f[s]["timestamps"][:] for s in ("train", "val", "test")]).astype(np.int64)
o = np.argsort(ts); ts, precip = ts[o], precip[o]
_, u = np.unique(ts, return_index=True); ts, precip = ts[u], precip[u]
s_obs = pd.Series(precip.mean(1), index=pd.to_datetime(ts, unit="s", utc=True)).sort_index()
obs_6h = s_obs.resample("6h", label="right", closed="right").mean()
print(f"telem obs: {len(s_obs)} h | {s_obs.index.min()} -> {s_obs.index.max()}")


def skill(fonte):
    df = pd.read_parquet(FORC / f"{COD}__graphcast_{fonte}.parquet")[
        ["lead_time_h", "valid_time", "p_graphcast_mmh"]].copy()
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    rows = []
    allpairs = []
    for lead, g in df.groupby("lead_time_h"):
        if lead == 0:
            continue
        j = pd.concat({"pred": g.set_index("valid_time")["p_graphcast_mmh"], "obs": obs_6h},
                      axis=1, sort=True).dropna()
        if len(j) < 30:
            continue
        rows.append((int(lead), j["pred"].corr(j["obs"]),
                     j["pred"].corr(j["obs"], method="spearman"),
                     j["pred"].mean() / max(j["obs"].mean(), 1e-9)))
        allpairs.append(j)
    sk = pd.DataFrame(rows, columns=["lead_h", "pearson", "spearman", "bias_ratio"])
    J = pd.concat(allpairs)
    sub72 = sk[sk["lead_h"] <= 72]
    resumo = {
        "pearson_med_72h": sub72.pearson.median(),
        "spearman_med_72h": sub72.spearman.median(),
        "pearson_mean_24h": sk[sk.lead_h <= 24].pearson.mean(),
        "max_pred": J["pred"].max(),
        "max_obs": J["obs"].max(),
        "mean_pred": J["pred"].mean(),
        "mean_obs": J["obs"].mean(),
    }
    return sk, resumo


print(f"\n{'fonte':6} {'P_med72':>8} {'S_med72':>8} {'P_m24':>7} {'maxpred':>8} {'maxobs':>7} "
      f"{'mpred':>7} {'mobs':>7}")
for fonte in ("gfs", "ifs", "hres"):
    p = FORC / f"{COD}__graphcast_{fonte}.parquet"
    if not p.exists():
        print(f"{fonte:6}  [FALTA parquet: {p.name}]")
        continue
    sk, r = skill(fonte)
    if fonte == "hres":
        sk.to_csv(A / "skill_hres.csv", index=False)
    print(f"{fonte:6} {r['pearson_med_72h']:8.3f} {r['spearman_med_72h']:8.3f} "
          f"{r['pearson_mean_24h']:7.3f} {r['max_pred']:8.2f} {r['max_obs']:7.2f} "
          f"{r['mean_pred']:7.3f} {r['mean_obs']:7.3f}")
print("\n(GFS deve bater o texto §2.2: P_med72≈0.481, S≈0.646, P_m24≈0.533, maxpred≈4.4, maxobs≈9.8, "
      "mpred≈0.370, mobs≈0.309)")
print("skill_hres.csv salvo em analise/ (se o parquet HRES existia).")
