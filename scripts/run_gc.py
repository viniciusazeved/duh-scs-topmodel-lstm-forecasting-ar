"""Runner do sub-fatorial GC (chuva futura) do Paper 2 forecasting.

Liga use_gc=True (ramo fisico: Pe_fut roteado pelo DUH estendido) e varia o gc_mode do
dataset, que injeta a chuva futura das proximas H horas:
  teto = chuva observada futura distribuida (perfect forecast, TETO de previsibilidade)
  gfs  = forcante GraphCast-GFS (produto real, lumped broadcast)
Baseline fut=0 = os modelos correspondentes da grade_ar (SCS, SCS_AR, Topmodel, Topmodel_AR):
la use_gc=False roteia so a agua em transito do lookback. Assim o eixo GC mede
teto-de-previsibilidade (teto - baseline) e gap-do-produto (teto - gfs).

Config IDENTICA a grade_ar (AdamW + fisica lr x10 + ReduceLROnPlateau, log1p+0.01 pico,
L240/H24, NSE@6h) -> resultados cruzaveis com a grade AR. Retomavel (pula results.json).

Nucleo (default): arquiteturas fisicas {SCS, Topmodel} x {teto, gfs} x {AR off, on} x sementes
  = 8 celulas. --completa adiciona {Lumped, LSTM, Topmodel_PeOnly} (via gc_proj / ramo fisico).

Uso (dentro do gpuq, cuda:0):
  uv run --project D:/TTD_SCS_LSTM python forecasting_v2/scripts/run_gc.py --seeds 42 43 44 45 46
  ... --test        # smoke: 1 epoca, 1 seed, todas as celulas do escopo
  ... --completa    # grade completa (todas as arquiteturas)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPTS = r"D:/TTD_SCS_LSTM/forecasting_v2/scripts"
SRC = r"D:/TTD_SCS_LSTM/forecasting_v2/src"
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, SRC)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import train as T  # noqa: E402
from factory_forecast import create_forecast_model  # noqa: E402

H5 = Path(r"D:/TTD_SCS_LSTM/forecasting_v2/data/dataset_forecast_v2.h5")
OUT = Path(r"D:/TTD_SCS_LSTM/forecasting_v2/outputs/grade_gc")

ARQS_FISICAS = ["lstm_duh_base_scs", "lstm_duh_base_topmodel"]   # ramo fisico interpretavel
ARQS_EXTRA = ["lstm_lumped_wmean", "lstm", "lstm_duh_base_topmodel_peonly"]  # --completa
MODOS = ["teto", "gfs"]


def build_celulas(completa: bool):
    arqs = ARQS_FISICAS + (ARQS_EXTRA if completa else [])
    cels = []
    for mt in arqs:
        for ar in (False, True):
            for gc in MODOS:
                cels.append((mt, ar, gc))
    return cels


def run_one(model_type, use_ar, gc_mode, seed, loaders, static, config, out_root, verbose=False):
    T.set_seed(seed)
    model = create_forecast_model(
        model_type, static, hidden_size=config["hidden_size"], num_layers=config["num_layers"],
        dropout=config["dropout"], horizon=config["horizon"], use_ar=use_ar, use_gc=True, device=T.DEVICE)
    name = f"{model.name}_{gc_mode}"   # model.name traz _AR/_GC; gc_mode separa teto de gfs
    exp_dir = out_root / name / f"seed{seed}"
    done = exp_dir / "results.json"
    if done.exists():
        print(f"  skip {name} seed{seed} (ja feito)")
        return
    exp_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, test_loader = loaders
    for ld in (train_loader, val_loader, test_loader):
        ld.dataset.gc_mode = gc_mode
    history, best_val = T.train_model(model, train_loader, val_loader, config, T.DEVICE, exp_dir, verbose)
    test_metrics, test_by_h, preds, targets = T.evaluate(model, test_loader, T.DEVICE)
    nse = test_metrics.get("nse", float("nan"))
    if not np.isfinite(nse) or not np.isfinite(best_val):
        print(f"  FALHA {name} seed{seed}: NSE/val nao-finito (nse={nse}, val={best_val})")
        return
    results = {
        "model_name": name, "model_type": model_type, "use_ar": use_ar, "use_gc": True,
        "gc_mode": gc_mode, "seed": seed,
        "test_metrics": test_metrics, "test_by_horizon": test_by_h, "best_val_nse": best_val,
        "learned_params": model.get_learned_params() if hasattr(model, "get_learned_params") else {},
        "epochs_trained": len(history["train_loss"]),
    }
    with open(done, "w") as f:
        json.dump(results, f, indent=2, default=float)
    np.savez(exp_dir / "predictions.npz", pred=preds, target=targets)
    print(f"  OK {name:44s} seed{seed} NSE@6h={nse:.4f} (val {best_val:.4f}, {results['epochs_trained']} ep)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--completa", action="store_true", help="inclui Lumped/LSTM/Topmodel_PeOnly")
    ap.add_argument("--test", action="store_true", help="smoke: 1 epoca, 1 seed")
    a = ap.parse_args()
    if a.test:
        a.epochs, a.patience, a.seeds = 1, 1, [42]

    celulas = build_celulas(a.completa)
    config = dict(lr=1e-3, weight_decay=1e-5, epochs=a.epochs, patience=a.patience,
                  batch_size=a.batch_size, hidden_size=64, num_layers=2, dropout=0.1,
                  lookback=240, horizon=24, grad_clip=1.0)
    out_root = OUT.parent / "grade_gc_smoke" if a.test else OUT   # smoke nao suja a grade real
    out_root.mkdir(parents=True, exist_ok=True)
    dev = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"dispositivo: {dev} | {len(celulas)} celulas x {len(a.seeds)} seeds "
          f"| completa={a.completa} epochs={a.epochs} patience={a.patience} batch={a.batch_size}")
    train_loader, val_loader, test_loader, static = T.create_dataloaders(str(H5), 240, 24, a.batch_size)
    loaders = (train_loader, val_loader, test_loader)
    print(f"dados: train {len(train_loader.dataset)} / val {len(val_loader.dataset)} / test {len(test_loader.dataset)} janelas")
    for seed in a.seeds:
        for mt, ar, gc in celulas:
            run_one(mt, ar, gc, seed, loaders, static, config, out_root, verbose=a.test)
    print("FIM run_gc")


if __name__ == "__main__":
    main()
