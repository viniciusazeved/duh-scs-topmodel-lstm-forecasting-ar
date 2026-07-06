# DUH-SCS-TOPMODEL-LSTM — Hourly Forecasting (AR + forecast rainfall)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21227520.svg)](https://doi.org/10.5281/zenodo.21227520)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An encoder–decoder LSTM streamflow **forecaster** (1–24 h) built on a differentiable hybrid
rainfall–runoff chain: a **distributed unit hydrograph (DUH)** for routing, coupled with
**SCS-CN** and **TOPMODEL** runoff generation and an integrating **LSTM**. On top of that chain
the study switches on two extra information channels and measures the worth of each: an
**autoregressive (AR)** channel that feeds the discharge observed up to the emission instant, and
a **future-rainfall (GC)** channel fed by **GraphCast** forecasts from the open NOAA AIWP hindcast.

This repository is the full reproducibility package for the factorial experiment in the Preto
River catchment (Manuel Duarte, ANA gauge 58585000, ~3,117 km², Brazilian Atlantic Forest).

> **Measurement premise (isolated test):** the future-rainfall channel is evaluated by
> re-running the *same trained checkpoint* under three inputs for the horizon rainfall — zero,
> the observed series (the predictability ceiling) and the GraphCast forecast — so both gains are
> paired within a checkpoint and training variance cancels. Separately trained models are never
> compared to measure this channel.

This repository accompanies the article *"Observed discharge, embedded physics or forecast
rainfall? Attributing the sources of skill of a differentiable hourly streamflow forecaster"*
(Azevedo & Fagundes, *Journal of Hydrology*, manuscript in preparation). It is the forecasting
companion of the continuous-simulation package
[`duh-scs-topmodel-lstm-continuous`](https://github.com/viniciusazeved/duh-scs-topmodel-lstm-continuous).

## What is in the study

Two factorial grids sharing a single training configuration:

- **AR grid** — five architectures (lumped LSTM, distributed LSTM, SCS-CN hybrid, TOPMODEL hybrid,
  TOPMODEL PeOnly), each with the AR channel off and on, plus the SCS PeOnly anchor: **11
  configurations × 5 seeds = 55 runs** (`outputs/grade_ar`).
- **GC grid** — the two physical generators (SCS-CN, TOPMODEL) × training feed (observed "teto" or
  GraphCast "gfs") × AR off/on: **8 configurations × 5 seeds = 40 runs** (`outputs/grade_gc`); the
  no-future baselines come from the AR grid.
- **Isolated test** and its by-lead extension (`outputs/teste_isolado_gc.json`,
  `teste_isolado_gc_bylead.json`): the predictability ceiling and the GraphCast gap, per cell and
  per lead (1–24 h).
- **Naive-persistence benchmark** (`outputs/persistencia_baseline.json`): the trivial reference on
  the same test windows and per-lead NSE.

Central findings: the **AR channel dominates** (mean paired +0.258 NSE at 6 h across 25
architecture-seed pairs; exact Wilcoxon p = 5.96×10⁻⁸), and its value decays with lead in a
**three-band division of labor** — over the first hours *naive persistence* is unbeatable
(NSE 0.996 at 1 h), by ~12 h the trained models overtake it, and at 24 h the distributed hybrids
hold (0.609–0.620) while persistence collapses (0.399). The **future-rainfall ceiling is small at
short range** (at most +0.023 NSE at 6 h under perfect rainfall) and **widens with lead** to
about +0.106 by 24 h; at short range only the **stateful generator** (TOPMODEL) opens it (paired
Wilcoxon SCS vs TOPMODEL p = 2.7×10⁻⁵), and GraphCast captures essentially the whole 6 h ceiling.

## Repository layout

```
duh-scs-topmodel-lstm-forecasting-ar/
├── src/ttd_scs_lstm/            # model package
│   ├── models/                  #   models.py (configs), topmodel_diff.py (differentiable TOPMODEL)
│   └── data/                    #   dataset.py, temporal.py
├── scripts/
│   ├── run_ar.py                # AR grid runner (11 configs x seeds)
│   ├── run_gc.py                # GC grid runner (8 configs x seeds, retomavel)
│   ├── train.py                 # training + evaluation (encoder-decoder, AR/GC channels, NSE by lead)
│   ├── factory_forecast.py      # model factory
│   ├── build_dataset_forecast_v2.py  # builds dataset_forecast_v2.h5 (incl. GraphCast future rainfall)
│   └── consolida_gc.py          # isolated test at 6 h -> teste_isolado_gc.json
├── analise/
│   ├── consolida_ar.py          # AR curve NSE x lead + AR Wilcoxon
│   ├── consolida_gc_bylead.py   # isolated test by lead (B1)  -> teste_isolado_gc_bylead.json
│   ├── analise_b1_b3.py         # ceiling by lead (B1) + dispersion & Wilcoxon SCS vs TOPMODEL (B3)
│   ├── persistencia_baseline.py # naive/damped persistence benchmark (B2)
│   ├── skill_fig.py / skill_forcante.py   # GraphCast forcing-skill characterisation
│   ├── rerender_curva_nse_lead.py / rerender_skill_fig.py   # figures (EN)
│   └── skill_{gfs,ifs}.csv       # forcing-skill tables
└── outputs/
    ├── grade_ar/  grade_gc/     # per-run results.json (per-lead NSE, learned physical params)
    ├── teste_isolado_gc.json  teste_isolado_gc_bylead.json  persistencia_baseline.json
```

The per-run model weights (`best_model.pt`, ~43 MB) and the raw `predictions.npz` (~45 MB,
regenerable) are **not** in git — they are available from the corresponding author on request
(see *Data availability*). The lightweight `results.json` (per-lead NSE + learned parameters) and
the consolidated JSONs are enough to rebuild every table and figure.

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

**PyTorch is intentionally not pinned** (the build depends on your CUDA setup):

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Training was run on NVIDIA RTX 2000/3000 Ada GPUs; CPU works but is slow (force it with
`CUDA_VISIBLE_DEVICES=-1`).

## Reproducing the experiments

**A single run** (smoke test, one epoch):

```bash
uv run python scripts/run_gc.py --test
```

**The two grids** (single fixed configuration, five seeds 42–46):

```bash
uv run python scripts/run_ar.py --seeds 42 43 44 45 46 --epochs 150 --patience 20
uv run python scripts/run_gc.py --seeds 42 43 44 45 46 --epochs 150 --patience 20   # retomavel
```

**Tables, the isolated test and the figures** are rebuilt from the shipped results (no re-training;
the by-lead and persistence steps need `best_model.pt` / the dataset — the weights from Zenodo):

```bash
uv run python scripts/consolida_gc.py               # isolated test @6h  (needs weights)
uv run python analise/consolida_gc_bylead.py        # ceiling by lead    (needs weights)  [B1]
uv run python analise/analise_b1_b3.py              # ceiling-by-lead + dispersion + Wilcoxon [B1/B3]
uv run python analise/persistencia_baseline.py      # naive/damped persistence [B2]
uv run python analise/consolida_ar.py               # AR curve + AR Wilcoxon
uv run python analise/rerender_curva_nse_lead.py    # Fig. 2 (NSE x lead, with persistence)
uv run python analise/rerender_skill_fig.py         # forcing-skill figure
```

Key configuration (fixed across both grids): horizon 24 (direct multi-horizon decoder, no
streamflow feedback), lookback 240 h, batch 512, up to 150 epochs, early stopping (patience 20 on
the validation NSE at 6 h), AdamW (weight decay 1e-5; learning rate 1e-3 for the LSTM and 1e-2 for
the physical parameters), ReduceLROnPlateau (halve on a 7-epoch plateau, floor 1e-6), loss =
MSE(log1p) + 0.01·peak, gradient clipping (norm 1.0). All timestamps are UTC.

> **Note on paths.** The model code (`src/`, `scripts/train.py`, `scripts/run_*.py`) runs from a
> clone. The analysis scripts carry absolute paths from the development machine (`D:\…`); adjust
> the constants at the top before running. The shipped `results.json` and consolidated JSONs are
> enough to rebuild the tables and figures without re-running the experiments.

## Dataset format

`data/dataset_forecast_v2.h5`:

```
ottobacia/   area_km2, cn_2022, tc_base_h, tc_manning_h, twi_*   (245 sub-catchments)
train/ val/ test/   precipitation (T, 245), streamflow (T,), pet (T, 245),
                    precip_fut_gfs (T, 24) + valid_gfs (T,), timestamps (T,)
```

Temporal split: train 2022-01 → 2023-11 (13,225 windows), validation 2023-12 → 2024-03 wet season
(2,687), test 2024-11 → 2025-03 wet season (2,568). The validation window is required to contain a
wet season so that early stopping selects well.

## Data availability

The processed dataset and the lightweight results are in this repository and archived on Zenodo;
the per-run model weights and `predictions.npz` are available from the corresponding author on
request:

> **Zenodo DOI:** [10.5281/zenodo.21227520](https://doi.org/10.5281/zenodo.21227520)

The GraphCast rainfall forecasts come from the open **NOAA AIWP** hindcast (GraphCast-GFS, 0.25°,
6-hourly). Underlying inputs are publicly available from their providers: streamflow telemetry and
the BHAE_CN-2022 curve-number product from ANA; land cover from MapBiomas; and the digital terrain
model from IPH/UFRGS and ANA.

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite both the article and this archived repository.

## License

[MIT](LICENSE) © 2026 Vinicius Azevedo, Hugo de Oliveira Fagundes.
