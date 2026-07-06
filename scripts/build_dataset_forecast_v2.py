"""Constroi dataset_forecast_v2.h5 — Paper 2 (forecasting AR + GraphCast).

Fonte: telem cap305 (precipitation 245, pet 245, streamflow) + TWI (npz) + cubo
GraphCast-GFS estendido ate abr/2025. Split por DATA com WARMUP de lookback (cada
split inclui as 240 h anteriores ao seu primeiro alvo, para nao perder o inicio):
  train alvos 2022-01-01..2024-04-30 | val 2024-05-01..2024-10-31 | test 2024-11-01..2025-03-31.

GraphCast e lumped (0.25 deg nao distingue ottobacia) -> campo previsto 1D (T, HORIZON),
desagregado 6h->1h (uniforme, preserva volume em mm/h), alinhado pela previsao 00z mais
recente <= t. O "teto" (chuva observada futura) o loader deriva do proprio precipitation.

ATENCAO ao off-by-one: precip_fut_gfs[t] = previsao EMITIDA em t para os valids t+1..t+H.
O consumidor deve ler no instante de EMISSAO (i+L-1), nao no primeiro alvo (i+L), para
casar com os alvos [i+L, i+L+H). Correcao fica no collate, nao aqui.

Saida: forecasting_v2/data/dataset_forecast_v2.h5
  ottobacia/(area_km2, cn_2022, tc_base_h, tc_manning_h, twi_dist, twi_centers, twi_mean)
  train|val|test/
    precipitation (T,245) | pet (T,245) | streamflow (T,) | timestamps (T,) i64 UTC
    precip_fut_gfs (T,HORIZON) f32 mm/h | valid_gfs (T,HORIZON) i8
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

TELEM = Path(r"D:/TTD_SCS_LSTM/ablacao_skill/data/dataset_58585000_telem.h5")
TWI = Path(r"D:/TTD_SCS_LSTM/forecasting_v2/data/twi_attrs.npz")
GC = Path(r"D:/Graph_Cast/data/forcante/58585000__graphcast_gfs.parquet")
OUT = Path(r"D:/TTD_SCS_LSTM/forecasting_v2/data/dataset_forecast_v2.h5")

LOOKBACK, HORIZON, PASSO = 240, 24, 6


def U(y, m, d):
    return int(datetime(y, m, d, tzinfo=timezone.utc).timestamp())


SPLITS = [  # (nome, alvo_inicio, alvo_fim_exclusivo) — val no verao anterior p/ representar o test
    ("train", U(2022, 1, 1), U(2023, 12, 1)),   # alvos 2022-01 -> 2023-11
    ("val", U(2023, 12, 1), U(2024, 4, 1)),       # verao 2023-24 (dez-mar): val COM eventos
    ("test", U(2024, 11, 1), U(2025, 4, 1)),      # verao 2024-25 (nov-mar): test preservado
]


def serie_continua():
    with h5py.File(TELEM, "r") as f:
        otto = {k: f["ottobacia"][k][:] for k in f["ottobacia"]}
        precip = np.concatenate([f[s]["precipitation"][:] for s in ("train", "val", "test")])
        flow = np.concatenate([f[s]["streamflow"][:] for s in ("train", "val", "test")])
        pet = np.concatenate([f[s]["pet"][:] for s in ("train", "val", "test")])
        ts = np.concatenate([f[s]["timestamps"][:] for s in ("train", "val", "test")]).astype(np.int64)
    o = np.argsort(ts)
    return otto, precip[o], flow[o], pet[o], ts[o]


def graphcast_hourly():
    """(inits_unix[n], hourly[n, max_lead]) mm/h, desagregado 6h->horario (uniforme)."""
    df = pd.read_parquet(GC, columns=["init_time", "lead_time_h", "p_graphcast_mmh"])
    df = df[df["lead_time_h"] > 0]
    piv = df.pivot_table(index="init_time", columns="lead_time_h",
                         values="p_graphcast_mmh", aggfunc="first").sort_index()
    leads = sorted(int(c) for c in piv.columns)
    blocos = piv[leads].to_numpy()  # (n, n_blocos)
    hourly = np.repeat(blocos, PASSO, axis=1).astype(np.float32)
    inits = np.array([int(t.timestamp()) for t in piv.index.to_pydatetime()], dtype=np.int64)
    return inits, hourly, max(leads)


def alinhar(grade, inits, hourly, max_lead):
    """precip_fut[t] = previsao emitida em t (init 00z mais recente <= t) p/ valids t+1..t+H."""
    idx = np.searchsorted(inits, grade, side="right") - 1
    fut = np.full((len(grade), HORIZON), np.nan, np.float32)
    mask = np.zeros((len(grade), HORIZON), np.int8)
    ks = np.arange(1, HORIZON + 1)
    for t in range(len(grade)):
        if idx[t] < 0:
            continue
        offs = (grade[t] - inits[idx[t]]) // 3600 + ks
        ok = offs <= max_lead
        fut[t, ok] = hourly[idx[t], offs[ok] - 1]
        mask[t, ok] = 1
    return fut, mask


def main() -> None:
    otto, precip, flow, pet, ts = serie_continua()
    print(f"serie telem: {len(ts)} h | {datetime.fromtimestamp(int(ts[0]), timezone.utc):%Y-%m-%d} -> "
          f"{datetime.fromtimestamp(int(ts[-1]), timezone.utc):%Y-%m-%d}")
    inits, hourly, max_lead = graphcast_hourly()
    print(f"graphcast: {len(inits)} inits | max_lead {max_lead} h")
    fut, mask = alinhar(ts, inits, hourly, max_lead)
    twi = np.load(TWI)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(OUT, "w") as g:
        og = g.create_group("ottobacia")
        for k, v in otto.items():
            og.create_dataset(k, data=v)
        og.create_dataset("twi_dist", data=twi["twi_dist"])
        og.create_dataset("twi_centers", data=twi["twi_centers"])
        og.create_dataset("twi_mean", data=twi["twi_mean"])
        g.attrs["fonte"] = "graphcast_gfs"
        g.attrs["horizon"] = HORIZON
        g.attrs["lookback"] = LOOKBACK
        g.attrs["cap_mm_h"] = 305
        for nome, a_ini, a_fim in SPLITS:
            arr_ini = a_ini - LOOKBACK * 3600  # warmup de lookback
            sel = np.nonzero((ts >= arr_ini) & (ts < a_fim))[0]
            lo, hi = sel[0], sel[-1] + 1
            sg = g.create_group(nome)
            sg.create_dataset("precipitation", data=precip[lo:hi], compression="gzip")
            sg.create_dataset("pet", data=pet[lo:hi], compression="gzip")
            sg.create_dataset("streamflow", data=flow[lo:hi])
            sg.create_dataset("timestamps", data=ts[lo:hi])
            sg.create_dataset("precip_fut_gfs", data=fut[lo:hi], compression="gzip")
            sg.create_dataset("valid_gfs", data=mask[lo:hi], compression="gzip")
            d0 = datetime.fromtimestamp(int(ts[lo]), timezone.utc).strftime("%Y-%m-%d")
            d1 = datetime.fromtimestamp(int(ts[hi - 1]), timezone.utc).strftime("%Y-%m-%d")
            covfull = float((mask[lo:hi].sum(axis=1) == HORIZON).mean())
            nanflow = float(np.isnan(flow[lo:hi]).mean() * 100)
            print(f"  {nome:5s}: {hi - lo:6d} h | array {d0} -> {d1} | "
                  f"GC horizonte-cheio {covfull * 100:5.1f}% | NaN flow {nanflow:4.1f}%")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
