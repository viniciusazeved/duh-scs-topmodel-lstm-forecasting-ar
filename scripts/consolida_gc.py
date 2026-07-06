"""Consolida o eixo GC por TESTE ISOLADO (metodo aprovado 02/07; ver memoria eixo-gc-medicao).

Carrega cada best_model.pt da grade_gc e avalia no teste com precip_fut = teto / gfs / zeros,
variando SO a chuva futura injetada no mesmo modelo. Reporta, por celula/semente:
  NSE@6h por modo de avaliacao + teto de previsibilidade (teto - semfut) + gap-produto (gfs - semfut).
NAO compara treinos separados (isso mistura variancia de treino com o efeito da chuva futura,
e cria armadilhas fisicamente impossiveis: teto<baseline, gfs>teto).

Uso (via gpuq, cuda:0):
  gpuq submit "uv run --project D:/TTD_SCS_LSTM python D:/TTD_SCS_LSTM/forecasting_v2/scripts/consolida_gc.py" --name gc-consolida --gpu rtx2000
Ou CPU:  CUDA_VISIBLE_DEVICES=-1 uv run --project D:/TTD_SCS_LSTM python .../consolida_gc.py
"""
import sys, json, glob, os
from collections import defaultdict
import numpy as np
import torch

sys.path.insert(0, r"D:/TTD_SCS_LSTM/forecasting_v2/scripts")
sys.path.insert(0, r"D:/TTD_SCS_LSTM/forecasting_v2/src")
import train as T
from factory_forecast import create_forecast_model

H5 = r"D:/TTD_SCS_LSTM/forecasting_v2/data/dataset_forecast_v2.h5"
GC = r"D:/TTD_SCS_LSTM/forecasting_v2/outputs/grade_gc"


def avalia(ckpt, model_type, use_ar, te, static):
    m = create_forecast_model(model_type, static, hidden_size=64, num_layers=2, dropout=0.1,
                              horizon=24, use_ar=use_ar, use_gc=True, device=T.DEVICE)
    m.load_state_dict(torch.load(ckpt, weights_only=True, map_location=T.DEVICE))
    m.eval()
    out = {}
    for mode in ["teto", "gfs", None]:
        te.dataset.gc_mode = mode
        _, byh, _, _ = T.evaluate(m, te, T.DEVICE)
        out["semfut" if mode is None else mode] = byh["6h"]["nse"]
    return out


def main():
    tr, va, te, static = T.create_dataloaders(H5, 240, 24, 512)
    rows = []
    for rj in sorted(glob.glob(GC + "/*/seed*/results.json")):
        d = json.load(open(rj))
        ck = os.path.join(os.path.dirname(rj), "best_model.pt")
        if not os.path.exists(ck):
            continue
        r = avalia(ck, d["model_type"], d["use_ar"], te, static)
        r.update(name=d["model_name"], seed=d["seed"], model_type=d["model_type"],
                 use_ar=d["use_ar"], train_mode=d.get("gc_mode"))
        rows.append(r)
        print(f"{d['model_name']:46s} seed{d['seed']}: sem-fut {r['semfut']:.3f} | "
              f"teto {r['teto']:.3f} | gfs {r['gfs']:.3f}", flush=True)

    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["model_type"], r["use_ar"], r["train_mode"])
        agg[key]["teto_prev"].append(r["teto"] - r["semfut"])
        agg[key]["gap_prod"].append(r["gfs"] - r["semfut"])
        agg[key]["nse_semfut"].append(r["semfut"])

    print("\n=== teto de previsibilidade e gap-produto (media seeds, NSE@6h) ===")
    for key, v in sorted(agg.items(), key=lambda x: str(x[0])):
        mt, ar, tm = key
        print(f"{mt:26s} ar={str(ar):5s} treino={str(tm):4s}: sem-fut {np.mean(v['nse_semfut']):.3f} | "
              f"teto-prev {np.mean(v['teto_prev']):+.3f} | gap-prod {np.mean(v['gap_prod']):+.3f} (n={len(v['teto_prev'])})")

    dst = os.path.join(GC, "..", "teste_isolado_gc.json")
    json.dump(rows, open(dst, "w"), indent=2, default=float)
    print(f"\nsalvo: {os.path.normpath(dst)}")


if __name__ == "__main__":
    main()
