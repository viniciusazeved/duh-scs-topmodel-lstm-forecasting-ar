"""Skill da forcante GraphCast (GFS/IFS) na bacia 58585000 + figura estilo Elsevier."""
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, h5py
import matplotlib.pyplot as plt
from matplotlib import rcParams

TELEM = Path(r"D:/TTD_SCS_LSTM/ablacao_skill/data/dataset_58585000_telem.h5")
FORC = Path(r"D:/Graph_Cast/data/forcante"); COD = "58585000"
OUT = Path(r"C:/Users/vinic/AppData/Local/Temp/claude/D--Artigo-JOH/bfcf1311-2eda-4171-b909-bd7d34f659a0/scratchpad")

with h5py.File(TELEM, "r") as f:
    precip = np.concatenate([f[s]["precipitation"][:] for s in ("train", "val", "test")])
    ts = np.concatenate([f[s]["timestamps"][:] for s in ("train", "val", "test")]).astype(np.int64)
o = np.argsort(ts); ts, precip = ts[o], precip[o]
_, u = np.unique(ts, return_index=True); ts, precip = ts[u], precip[u]
s_obs = pd.Series(precip.mean(1), index=pd.to_datetime(ts, unit="s", utc=True)).sort_index()
obs_6h = s_obs.resample("6h", label="right", closed="right").mean()

def skill(fonte):
    df = pd.read_parquet(FORC / f"{COD}__graphcast_{fonte}.parquet")[["lead_time_h", "valid_time", "p_graphcast_mmh"]]
    df["valid_time"] = pd.to_datetime(df["valid_time"], utc=True)
    rows = []
    for lead, g in df.groupby("lead_time_h"):
        if lead == 0: continue
        j = pd.concat({"pred": g.set_index("valid_time")["p_graphcast_mmh"], "obs": obs_6h}, axis=1, sort=True).dropna()
        if len(j) < 30: continue
        rows.append((int(lead), j["pred"].corr(j["obs"]), j["pred"].corr(j["obs"], method="spearman"),
                     j["pred"].mean() / max(j["obs"].mean(), 1e-9)))
    return pd.DataFrame(rows, columns=["lead_h", "pearson", "spearman", "bias_ratio"])

sk = {f: skill(f) for f in ("gfs", "ifs")}
for f in sk:
    sk[f].to_csv(OUT / f"skill_{f}.csv", index=False)
    sub = sk[f][sk[f]["lead_h"] <= 72]
    print(f"{f.upper()}: Pearson mediano<=72h {sub.pearson.median():.3f} | Spearman {sub.spearman.median():.3f} | "
          f"<=24h Pearson {sk[f][sk[f].lead_h<=24].pearson.mean():.3f}")

# ---- figura estilo Elsevier ----
rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                 "font.size": 9, "axes.linewidth": 0.8, "savefig.dpi": 300})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.6, 1]})
C = {"gfs": "#1f4e79", "ifs": "#c55a11"}
for f in ("gfs", "ifs"):
    d = sk[f]
    ax1.plot(d.lead_h, d.pearson, "-", color=C[f], lw=1.6, label=f"{f.upper()} (Pearson)")
    ax1.plot(d.lead_h, d.spearman, "--", color=C[f], lw=1.1, alpha=0.7, label=f"{f.upper()} (Spearman)")
ax1.axvspan(0, 24, color="0.85", alpha=0.5, zorder=0)
ax1.text(12, 0.86, "horizonte do\nforecasting (≤24 h)", ha="center", va="top", fontsize=7, color="0.35")
ax1.set_xlabel("Lead time (h)"); ax1.set_ylabel("Correlação previsão × observado")
ax1.set_xlim(0, 168); ax1.set_ylim(0, 0.9); ax1.legend(frameon=False, fontsize=6.5, ncol=1, loc="lower left")
for s in ("top", "right"): ax1.spines[s].set_visible(False)
ax1.set_title("(a) Skill da chuva prevista por antecedência", fontsize=8.5, loc="left")

# painel b: bias por lead (superestima chuva fraca)
for f in ("gfs", "ifs"):
    ax2.plot(sk[f].lead_h, sk[f].bias_ratio, "-", color=C[f], lw=1.4, label=f.upper())
ax2.axhline(1.0, color="0.4", lw=0.8, ls=":")
ax2.set_xlabel("Lead time (h)"); ax2.set_ylabel("Viés (prev / obs)")
ax2.set_xlim(0, 168); ax2.set_ylim(0.7, 1.9); ax2.legend(frameon=False, fontsize=7)
for s in ("top", "right"): ax2.spines[s].set_visible(False)
ax2.set_title("(b) Viés de magnitude", fontsize=8.5, loc="left")
fig.tight_layout()
fig.savefig(OUT / "skill_forcante_58585000.png", dpi=300, bbox_inches="tight")
print("figura salva:", OUT / "skill_forcante_58585000.png")
