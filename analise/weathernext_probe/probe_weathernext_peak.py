"""Probe decisivo: WeatherNext (mean / ensemble / deterministico HRES) vs GraphCast-GFS
fecha a supressao de pico de chuva no rio Preto, na janela de teste do Paper 2?

Sem re-treinar modelo de vazao. Compara SKILL DE CHUVA a nivel de bacia, alinhando
tudo a taxa media de 6h no valid_time. Metrica-chave = captura de pico.

Saida: scratchpad/wn_peak_metrics.csv + wn_peak_series.parquet
"""
from __future__ import annotations

import time
from pathlib import Path

import ee
import numpy as np
import polars as pl

from graphcast_hidro.config import Config
from graphcast_hidro.gee import geometria_bacia, init_ee

COD = "58585000"
OUT = Path(r"C:/Users/vinic/AppData/Local/Temp/claude/D--Artigo-JOH/aa28a107-7cab-498d-b037-52aed9439a3c/scratchpad")
GFS_PARQUET = Path(f"D:/Graph_Cast/data/forcante/{COD}__graphcast_gfs.parquet")
OBS_PARQUET = Path(f"D:/CaBRA_H/data/outputs/pub/forcante_base/{COD}.parquet")

# Janela de teste do Paper 2 (memoria): test 2024-11-01 -> 2025-03-31
INI, FIM = "2024-11-01", "2025-04-01"   # fim exclusivo, pega mar/2025 inteiro
LEADS = [6, 12, 18, 24]                  # horizonte operacional do forecasting (t+1:t+24)

WN2_MEAN = "projects/gcp-public-data-weathernext/assets/weathernext_2_0_0_mean"
WN2_FULL = "projects/gcp-public-data-weathernext/assets/weathernext_2_0_0"
WN_DET   = "projects/gcp-public-data-weathernext/assets/59572747_4_0"
PRECIP   = "total_precipitation_6hr"     # m/6h -> mm/h = x1000/6


def _reduz(img, geom, escala):
    vals = img.select([PRECIP]).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=geom, scale=escala,
        bestEffort=True, maxPixels=int(1e9),
    )
    f = ee.Feature(None, {"p_m6h": vals.get(PRECIP)})
    return (f.set("start_time", img.get("start_time"))
             .set("forecast_hour", img.get("forecast_hour"))
             .set("end_time", img.get("end_time"))
             .set("ensemble_member", img.get("ensemble_member")))


def _base_filter(col):
    return (col.filter(ee.Filter.inList("forecast_hour", LEADS))
               .filter(ee.Filter.gte("start_time", INI))
               .filter(ee.Filter.lt("start_time", FIM))
               .filter(ee.Filter.stringEndsWith("start_time", "00:00:00Z")))  # so 00z


def extrai_det(col_id, geom, escala, rotulo):
    """Produto deterministico (1 imagem por init,lead). getInfo unico."""
    t0 = time.time()
    col = _base_filter(ee.ImageCollection(col_id))
    fc = col.map(lambda im: _reduz(im, geom, escala))
    feats = fc.getInfo().get("features", [])
    print(f"  [{rotulo}] {len(feats)} feats em {time.time()-t0:.0f}s", flush=True)
    if not feats:
        return pl.DataFrame()
    df = pl.from_dicts([f["properties"] for f in feats])
    return df.with_columns(
        (pl.col("p_m6h").cast(pl.Float64) * 1000.0 / 6.0).clip(lower_bound=0.0).alias("p_mmh"),
        pl.col("end_time").str.to_datetime(time_zone="UTC", strict=False).alias("valid_time"),
        pl.col("forecast_hour").cast(pl.Int32).alias("lead_time_h"),
        pl.lit(rotulo).alias("produto"),
    ).select(["produto", "valid_time", "lead_time_h", "p_mmh"])


def extrai_ensemble(col_id, geom, escala, rotulo="wn2_ens"):
    """Ensemble completo (64 membros). Chunk por init pra getInfo pequeno."""
    col = _base_filter(ee.ImageCollection(col_id))
    inits = sorted(col.aggregate_array("start_time").distinct().getInfo())
    print(f"  [{rotulo}] {len(inits)} inits 00z. Extraindo por init...", flush=True)
    frames = []
    t0 = time.time()
    for k, st in enumerate(inits, 1):
        sub = col.filter(ee.Filter.eq("start_time", st))
        fc = sub.map(lambda im: _reduz(im, geom, escala))
        try:
            feats = fc.getInfo().get("features", [])
        except Exception as e:
            print(f"    init {st} FALHOU: {repr(e)[:120]}", flush=True)
            continue
        if feats:
            frames.append(pl.from_dicts([f["properties"] for f in feats]))
        if k == 1 or k % 20 == 0 or k == len(inits):
            print(f"    {k}/{len(inits)} ({time.time()-t0:.0f}s)", flush=True)
    if not frames:
        return pl.DataFrame()
    raw = pl.concat(frames, how="diagonal_relaxed").with_columns(
        (pl.col("p_m6h").cast(pl.Float64) * 1000.0 / 6.0).clip(lower_bound=0.0).alias("p_mmh"),
        pl.col("end_time").str.to_datetime(time_zone="UTC", strict=False).alias("valid_time"),
        pl.col("forecast_hour").cast(pl.Int32).alias("lead_time_h"),
    )
    # agrega os 64 membros por (valid_time, lead)
    agg = raw.group_by(["valid_time", "lead_time_h"]).agg(
        pl.col("p_mmh").mean().alias("ens_mean"),
        pl.col("p_mmh").max().alias("ens_max"),
        pl.col("p_mmh").quantile(0.90).alias("ens_p90"),
        pl.len().alias("n_membros"),
    )
    print(f"  [{rotulo}] membros/celula (deve ser ~64): "
          f"{agg['n_membros'].min()}..{agg['n_membros'].max()}", flush=True)
    return agg


def carrega_obs_6h() -> pl.DataFrame:
    obs = pl.read_parquet(OBS_PARQUET).select(["data_hora", "p_best_mmh"])
    # garante UTC tz-aware
    dt = obs["data_hora"].dtype
    if isinstance(dt, pl.Datetime) and dt.time_zone is None:
        obs = obs.with_columns(pl.col("data_hora").dt.replace_time_zone("UTC"))
    elif isinstance(dt, pl.Datetime):
        obs = obs.with_columns(pl.col("data_hora").dt.convert_time_zone("UTC"))
    # janela de 6h [bucket, bucket+6h) -> valid = bucket+6h (acumulacao terminando em valid)
    obs = obs.with_columns(pl.col("data_hora").dt.truncate("6h").alias("bucket"))
    o6 = (obs.group_by("bucket").agg(pl.col("p_best_mmh").mean().alias("p_obs_mmh"))
             .with_columns((pl.col("bucket") + pl.duration(hours=6)).alias("valid_time"))
             .select(["valid_time", "p_obs_mmh"]))
    return o6


def carrega_gfs() -> pl.DataFrame:
    g = pl.read_parquet(GFS_PARQUET)
    return (g.filter(pl.col("lead_time_h").is_in(LEADS))
             .filter(pl.col("valid_time") >= pl.datetime(2024, 11, 1, time_zone="UTC"))
             .filter(pl.col("valid_time") < pl.datetime(2025, 4, 1, time_zone="UTC"))
             .select([pl.lit("gfs").alias("produto"), "valid_time", "lead_time_h",
                      pl.col("p_graphcast_mmh").alias("p_mmh")]))


def metricas(nome, pred, obs, col_pred="p_mmh") -> list[dict]:
    j = pred.join(obs, on="valid_time", how="inner")
    out = []
    for lead in LEADS:
        s = j.filter(pl.col("lead_time_h") == lead)
        if s.height < 5:
            continue
        p = s[col_pred].to_numpy()
        o = s["p_obs_mmh"].to_numpy()
        corr = float(np.corrcoef(p, o)[0, 1]) if p.std() > 0 and o.std() > 0 else float("nan")
        out.append({
            "produto": nome, "lead_h": lead, "n": s.height,
            "corr": round(corr, 3),
            "mae": round(float(np.mean(np.abs(p - o))), 3),
            "bias": round(float(p.mean() - o.mean()), 3),
            "max_pred": round(float(p.max()), 2),
            "max_obs": round(float(o.max()), 2),
            "p99_pred": round(float(np.quantile(p, 0.99)), 2),
            "p99_obs": round(float(np.quantile(o, 0.99)), 2),
            "captura_pico_max": round(float(p.max() / o.max()), 2) if o.max() > 0 else float("nan"),
        })
    return out


def main():
    cfg = Config.carregar()
    init_ee(cfg.gee_project)
    geom = geometria_bacia(cfg.piloto, cfg.bacia_gpkg)
    escala = cfg.escala_m
    print(f"Rio Preto {COD} | janela {INI}->{FIM} | leads {LEADS} | escala {escala}m", flush=True)

    obs = carrega_obs_6h()
    print(f"OBS 6h: {obs.height} buckets, "
          f"{obs['valid_time'].min()} -> {obs['valid_time'].max()}", flush=True)

    print("\n[1/4] GFS (parquet)...", flush=True)
    gfs = carrega_gfs()
    print("\n[2/4] WeatherNext 2 mean (GEE)...", flush=True)
    wnm = extrai_det(WN2_MEAN, geom, escala, "wn2_mean")
    print("\n[3/4] WeatherNext Graph deterministico HRES (GEE)...", flush=True)
    wnd = extrai_det(WN_DET, geom, escala, "wn_det_hres")
    print("\n[4/4] WeatherNext 2 ENSEMBLE 64 membros (GEE)...", flush=True)
    ens = extrai_ensemble(WN2_FULL, geom, escala)

    rows = []
    rows += metricas("gfs", gfs, obs)
    if not wnm.is_empty():
        rows += metricas("wn2_mean", wnm, obs)
    if not wnd.is_empty():
        rows += metricas("wn_det_hres", wnd, obs)
    if not ens.is_empty():
        rows += metricas("wn2_ens_mean", ens, obs, col_pred="ens_mean")
        rows += metricas("wn2_ens_max", ens, obs, col_pred="ens_max")
        rows += metricas("wn2_ens_p90", ens, obs, col_pred="ens_p90")

    met = pl.DataFrame(rows).sort(["lead_h", "produto"])
    met.write_csv(OUT / "wn_peak_metrics.csv")
    with pl.Config(tbl_rows=100, tbl_cols=20, tbl_width_chars=200):
        print("\n===== METRICAS DE SKILL DE CHUVA (rio Preto, teste Paper 2) =====", flush=True)
        print(met, flush=True)

    # salva series agregadas p/ plot posterior
    parts = [gfs.select(["produto", "valid_time", "lead_time_h", "p_mmh"])]
    if not wnm.is_empty():
        parts.append(wnm)
    if not wnd.is_empty():
        parts.append(wnd)
    ser = pl.concat(parts, how="diagonal_relaxed")
    ser.join(obs, on="valid_time", how="left").write_parquet(OUT / "wn_peak_series.parquet")
    if not ens.is_empty():
        ens.join(obs, on="valid_time", how="left").write_parquet(OUT / "wn_peak_ensemble.parquet")
    print(f"\nSalvo: {OUT / 'wn_peak_metrics.csv'}", flush=True)
    print("FIM.", flush=True)


if __name__ == "__main__":
    main()
