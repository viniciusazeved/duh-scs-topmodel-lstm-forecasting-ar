"""B2 da revisao Paper 2: benchmark de PERSISTENCIA (sem treino), periodo de teste.

Usa EXATAMENTE as janelas de teste do pipeline (mesmo AblationDataset de train.py: mesmos
valid_indices, mesmo lookback=240/horizon=24) e a MESMA funcao NSE (compute_metrics).
Emissao t = i+lookback-1 (onde o q_past do canal AR termina); alvo = Q(t+1..t+24).

  - Persistencia ingenua:   Qhat(t+k) = Q(t)                     (ultima obs, forward-fill como o AR)
  - Persistencia amortecida: Qhat(t+k) = Q(t)*exp(-k/tau), tau da recessao media do teste

Reporta NSE por lead (1/3/6/12/24 h), para ancorar o nome "augmented persistence" e mostrar
quanto do 0.916@1h do Lumped+AR e o piso trivial. Nao altera nenhum numero existente.
"""
import sys, json
import numpy as np

sys.path.insert(0, r"D:/TTD_SCS_LSTM/forecasting_v2/scripts")
sys.path.insert(0, r"D:/TTD_SCS_LSTM/forecasting_v2/src")
import train as T

H5 = r"D:/TTD_SCS_LSTM/forecasting_v2/data/dataset_forecast_v2.h5"
LB, HZ = 240, 24
LEADS = {"1h": 0, "3h": 2, "6h": 5, "12h": 11, "24h": 23}


def forward_fill(x):
    x = np.asarray(x, dtype=np.float64).copy()
    nan = np.isnan(x)
    if nan.any():
        order = np.where(~nan, np.arange(len(x)), 0)
        np.maximum.accumulate(order, out=order)
        x = np.nan_to_num(x[order], nan=0.0)
    return np.clip(x, 0.0, None)


def main():
    ds = T.AblationDataset(H5, "test", LB, HZ)
    flow = np.asarray(ds.streamflow, dtype=np.float64)      # cru (com NaN)
    flow_ff = forward_fill(flow)                            # forward-fill (= canal AR)
    idx = np.asarray(ds.valid_indices)
    n = len(idx)

    # emissao e alvo por janela
    emit = idx + LB - 1                                     # (n,)
    q_em = flow_ff[emit]                                    # (n,) ultima obs disponivel
    tgt = np.stack([flow[idx + LB + k] for k in range(HZ)], axis=1)   # (n, HZ) alvo cru

    # tau da recessao: mediana de -1/ln(Q(t+1)/Q(t)) em passos de recessao do teste (Q cai, ambos>0)
    q0, q1 = flow_ff[:-1], flow_ff[1:]
    rec = (q1 < q0) & (q0 > 0) & (q1 > 0)
    ratios = q1[rec] / q0[rec]
    tau = float(-1.0 / np.median(np.log(ratios)))          # h
    k = np.arange(1, HZ + 1)
    damp = np.exp(-k / tau)                                 # (HZ,)

    pred_naive = np.repeat(q_em[:, None], HZ, axis=1)       # (n,HZ)
    pred_damp = q_em[:, None] * damp[None, :]               # (n,HZ)

    print(f"janelas de teste: {n} | tau_recessao = {tau:.1f} h")
    print(f"{'lead':>5s} {'naive':>8s} {'damped':>8s}")
    out = {"n_windows": int(n), "tau_h": tau, "naive": {}, "damped": {}}
    for name, j in LEADS.items():
        nse_n = T.compute_metrics(pred_naive[:, j], tgt[:, j])["nse"]
        nse_d = T.compute_metrics(pred_damp[:, j], tgt[:, j])["nse"]
        out["naive"][name] = float(nse_n)
        out["damped"][name] = float(nse_d)
        print(f"{name:>5s} {nse_n:8.3f} {nse_d:8.3f}")

    dst = r"D:/TTD_SCS_LSTM/forecasting_v2/outputs/persistencia_baseline.json"
    json.dump(out, open(dst, "w"), indent=2)
    print("salvo:", dst)


if __name__ == "__main__":
    main()
