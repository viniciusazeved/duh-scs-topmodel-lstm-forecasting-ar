"""
Skill da forcante GraphCast (GFS e IFS) na bacia 58585000 (rio Preto / Manuel Duarte).

Recalculado LIMPO: previsao p_graphcast_mmh (parquet por bacia, NOAA AIWP)
vs chuva observada de referencia = telem lumped (media das 245 ottobacias),
o mesmo produto observado do artigo de simulacao.

Nao usa nada pronto da pasta Graph_Cast alem do produto extraido (input do nosso eixo GC).
Saida: correlacao Pearson/Spearman, vies e RMSE por lead_time_h, GFS x IFS.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

TELEM = Path(r"D:/TTD_SCS_LSTM/ablacao_skill/data/dataset_58585000_telem.h5")
FORC = Path(r"D:/Graph_Cast/data/forcante")
COD = "58585000"

# ---- 1. chuva observada de referencia: telem lumped (media area das 245 ott) ----
with h5py.File(TELEM, "r") as f:
    precip = np.concatenate([f[s]["precipitation"][:] for s in ("train", "val", "test")])  # (T,245)
    ts = np.concatenate([f[s]["timestamps"][:] for s in ("train", "val", "test")]).astype(np.int64)
# ordena por tempo e remove duplicatas
o = np.argsort(ts)
ts, precip = ts[o], precip[o]
_, uniq = np.unique(ts, return_index=True)
ts, precip = ts[uniq], precip[uniq]
obs_lumped = precip.mean(axis=1)  # mm/h lumped
idx = pd.to_datetime(ts, unit="s", utc=True)
s_obs = pd.Series(obs_lumped, index=idx).sort_index()
print(f"telem: {len(s_obs)} h | {s_obs.index.min()} -> {s_obs.index.max()}")

# reamostra observado para 6h terminando em valid_time (bin (t-6,t], rotulo t)
obs_6h = s_obs.resample("6h", label="right", closed="right").mean()

# ---- 2. forcante GraphCast por fonte ----
def skill_por_lead(fonte):
    df = pd.read_parquet(FORC / f"{COD}__graphcast_{fonte}.parquet")
    df = df[["lead_time_h", "valid_time", "p_graphcast_mmh"]].copy()
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    out = []
    for lead, g in df.groupby("lead_time_h"):
        if lead == 0:
            continue  # lead 0 = analise (init), nao e previsao
        g = g.set_index("valid_time")["p_graphcast_mmh"]
        # casa previsao (valid_time) com observado 6h no mesmo valid_time
        j = pd.concat({"pred": g, "obs": obs_6h}, axis=1).dropna()
        if len(j) < 30:
            continue
        r = j["pred"].corr(j["obs"])               # Pearson
        rho = j["pred"].corr(j["obs"], method="spearman")
        bias = j["pred"].mean() / max(j["obs"].mean(), 1e-9)
        rmse = float(np.sqrt(((j["pred"] - j["obs"]) ** 2).mean()))
        out.append((int(lead), len(j), r, rho, bias, rmse, j["obs"].mean(), j["pred"].mean()))
    return pd.DataFrame(out, columns=["lead_h", "n", "pearson", "spearman", "bias_ratio", "rmse", "obs_mm_h", "pred_mm_h"])

for fonte in ("gfs", "ifs"):
    print(f"\n===== GraphCast-{fonte.upper()} | bacia {COD} =====")
    sk = skill_por_lead(fonte)
    pd.set_option("display.float_format", lambda x: f"{x:8.3f}")
    print(sk.to_string(index=False))
    # leads de interesse (<=72h): mediana da correlacao
    sub = sk[sk["lead_h"] <= 72]
    print(f"  -> Pearson mediano (leads<=72h): {sub['pearson'].median():.3f} | "
          f"Spearman mediano: {sub['spearman'].median():.3f}")
