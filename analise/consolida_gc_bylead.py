"""Teste isolado GC POR LEAD (B1/B3 da revisao Paper 2).

Identico ao consolida_gc.py (metodo aprovado 02/07), mas em vez de guardar so o NSE@6h
guarda o NSE em TODOS os leads que o evaluate() reporta (1/3/6/12/24 h) para cada modo de
avaliacao (teto / gfs / semfut). NAO re-treina: recarrega cada best_model.pt da grade_gc e
so troca a chuva futura injetada no teste. Reproduz os 6h de teste_isolado_gc.json (guard).

Saidas:
  outputs/teste_isolado_gc_bylead.json  -- por celula/semente, NSE[modo][lead]
Uso (via gpuq ou direto, cuda:0 ocioso):
  uv run --project D:/TTD_SCS_LSTM python D:/TTD_SCS_LSTM/forecasting_v2/analise/consolida_gc_bylead.py
CPU: CUDA_VISIBLE_DEVICES=-1 uv run --project D:/TTD_SCS_LSTM python .../consolida_gc_bylead.py
"""
import sys, json, glob, os
import numpy as np
import torch

sys.path.insert(0, r"D:/TTD_SCS_LSTM/forecasting_v2/scripts")
sys.path.insert(0, r"D:/TTD_SCS_LSTM/forecasting_v2/src")
import train as T
from factory_forecast import create_forecast_model

H5 = r"D:/TTD_SCS_LSTM/forecasting_v2/data/dataset_forecast_v2.h5"
GC = r"D:/TTD_SCS_LSTM/forecasting_v2/outputs/grade_gc"
LEADS = ["1h", "3h", "6h", "12h", "24h"]


def avalia(ckpt, model_type, use_ar, te, static):
    """Recarrega o checkpoint e avalia com precip_fut = teto / gfs / zeros.
    Retorna dict[modo][lead] = nse (todos os leads que o evaluate reporta)."""
    m = create_forecast_model(model_type, static, hidden_size=64, num_layers=2, dropout=0.1,
                              horizon=24, use_ar=use_ar, use_gc=True, device=T.DEVICE)
    m.load_state_dict(torch.load(ckpt, weights_only=True, map_location=T.DEVICE))
    m.eval()
    out = {}
    for mode in ["teto", "gfs", None]:
        te.dataset.gc_mode = mode
        _, byh, _, _ = T.evaluate(m, te, T.DEVICE)
        key = "semfut" if mode is None else mode
        out[key] = {h: float(byh[h]["nse"]) for h in LEADS}
    return out


def main():
    tr, va, te, static = T.create_dataloaders(H5, 240, 24, 512)
    # json canonico (6h) para o guard de reproducao
    canon = {}
    cj = os.path.join(GC, "..", "teste_isolado_gc.json")
    for r in json.load(open(cj)):
        canon[(r["name"], r["seed"])] = r  # tem teto/gfs/semfut @6h

    rows = []
    max_abs_6h = 0.0
    for rj in sorted(glob.glob(GC + "/*/seed*/results.json")):
        d = json.load(open(rj))
        ck = os.path.join(os.path.dirname(rj), "best_model.pt")
        if not os.path.exists(ck):
            continue
        by = avalia(ck, d["model_type"], d["use_ar"], te, static)
        row = {"name": d["model_name"], "seed": d["seed"], "model_type": d["model_type"],
               "use_ar": d["use_ar"], "train_mode": d.get("gc_mode"), "nse": by}
        rows.append(row)
        # guard: reproduz o 6h do json canonico?
        c = canon.get((d["model_name"], d["seed"]))
        if c is not None:
            for mode in ["teto", "gfs", "semfut"]:
                diff = abs(by[mode]["6h"] - c[mode])
                max_abs_6h = max(max_abs_6h, diff)
        print(f"{d['model_name']:46s} seed{d['seed']}: "
              f"semfut {by['semfut']['6h']:.4f} teto {by['teto']['6h']:.4f} gfs {by['gfs']['6h']:.4f} "
              f"| 24h: semfut {by['semfut']['24h']:.4f} teto {by['teto']['24h']:.4f} gfs {by['gfs']['24h']:.4f}",
              flush=True)

    dev = "cpu" if T.DEVICE.type == "cpu" else torch.cuda.get_device_name(0)
    print(f"\n=== GUARD reproducao 6h vs teste_isolado_gc.json: max |diff| = {max_abs_6h:.2e} (device={dev}) ===")
    dst = os.path.join(GC, "..", "teste_isolado_gc_bylead.json")
    json.dump({"device": dev, "max_abs_diff_6h_vs_canon": max_abs_6h, "leads": LEADS, "rows": rows},
              open(dst, "w"), indent=2, default=float)
    print(f"salvo: {os.path.normpath(dst)} ({len(rows)} linhas)")


if __name__ == "__main__":
    main()
