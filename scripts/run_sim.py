#!/usr/bin/env python
"""Runner da grade de SIMULAÇÃO CONTÍNUA por JANELA (abordagem Kratzert/quali, treino dedicado).

Treina cada modelo com horizon=1 (prever o próximo passo a partir do lookback de 240h) e desliza:
o NSE@1h sobre todas as janelas do test É o NSE de simulação contínua (gerar a série de vazão
a partir só da chuva, sem usar vazão observada como entrada). Reusa o train.py auditado (com
horizon=1 a métrica principal vira @1h automaticamente). NÃO é stateful.

Grade: 12 modelos × seeds, telem. Gate: só grava .done com results.json + test_nse (principal) finito.

Uso:
  smoke:  python scripts/run_sim.py --test --seeds 42
  grade:  python scripts/run_sim.py --seeds 42 43 44 45 46 47 48 49 50 51 --epochs 100 --patience 20
"""
import argparse, json, subprocess, sys, time
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "scripts" / "train.py"
DATA = ROOT / "data" / "dataset_58585000_telem.h5"
OUT = ROOT / "outputs" / "grade_sim"

# (model_type interno, display limpo) — mesma grade de 12 da cadeia "de onde vem o skill"
MODELS = [
    ("lstm_lumped_calonly", "LSTM_Lumped_CalOnly"),
    ("lstm_lumped_rainonly", "LSTM_Lumped_RainOnly"),
    ("lstm_lumped_wmean", "LSTM_Lumped"),
    ("lstm", "LSTM"),
    ("lstm_duh_base", "LSTM_DUH"),
    ("lstm_duh_base_fixed", "LSTM_DUH_Fixed"),
    ("lstm_duh_base_scs", "LSTM_DUH_SCS"),
    ("lstm_duh_base_scs_peonly", "LSTM_DUH_SCS_PeOnly"),
    ("lstm_duh_base_topmodel", "LSTM_DUH_Topmodel"),
    ("lstm_duh_base_topmodel_peonly", "LSTM_DUH_Topmodel_PeOnly"),
    ("lstm_duh_base_topmodel_baseflow", "LSTM_DUH_Topmodel_Baseflow"),
    ("phys_duh_base_scs", "Phys_DUH_SCS"),
    ("phys_duh_base_topmodel", "Phys_DUH_Topmodel"),
]


def valido(run_dir):
    cands = list(run_dir.glob("**/results.json"))
    if not cands:
        return False, "sem results.json"
    try:
        r = json.loads(cands[0].read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"ilegivel ({e})"
    v = r.get("test_metrics", {}).get("nse", None)   # métrica principal = @1h (horizon=1) = NSE simulação
    if v is None or not (v == v) or abs(v) == float("inf"):
        return False, f"test NSE nao-finito ({v})"
    return True, f"simNSE={v:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    if a.test:
        a.epochs, a.patience = 1, 1
    OUT.mkdir(parents=True, exist_ok=True)
    summary, tg = [], time.time()
    print("=" * 70)
    print(f"  GRADE SIMULAÇÃO (janela, horizon=1) | seeds={a.seeds} | epochs={a.epochs} "
          f"pat={a.patience}{' [TEST]' if a.test else ''}")
    print("=" * 70)
    for seed in a.seeds:
        for mt, disp in MODELS:
            run_dir = OUT / f"seed{seed}" / disp
            done = run_dir / ".done"
            if done.exists() and not a.test:
                print(f"  [SKIP] seed{seed}/{disp}")
                summary.append((seed, disp, "skip", 0)); continue
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, str(TRAIN), "--model", mt, "--data-path", str(DATA),
                   "--seed", str(seed), "--output-dir", str(run_dir),
                   "--batch-size", str(a.batch_size), "--epochs", str(a.epochs),
                   "--patience", str(a.patience), "--horizon", "1"]
            if a.test:
                cmd.append("--test")
            print(f"\n>>> seed{seed} | {disp} ({mt}) | {datetime.now():%H:%M:%S}")
            t0 = time.time()
            r = subprocess.run(cmd, cwd=str(ROOT))
            dt = time.time() - t0
            ok_out, msg = valido(run_dir)
            ok = r.returncode == 0 and ok_out
            if ok and not a.test:
                done.write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S} | {dt:.0f}s | {msg}\n")
            print(f"<<< seed{seed} | {disp} | {'ok' if ok else 'FAIL'} | {timedelta(seconds=int(dt))}"
                  f"{'' if ok else '  [rc=' + str(r.returncode) + ' | ' + msg + ']'}")
            summary.append((seed, disp, "ok" if ok else "fail", round(dt)))
    print(f"\n  FIM | total {timedelta(seconds=int(time.time()-tg))}")
    n_ok = sum(1 for _, _, st, _ in summary if st in ("ok", "skip"))
    for seed, disp, st, dt in summary:
        print(f"  seed{seed:>3} {disp:<30} {st:>4} {timedelta(seconds=dt)}")
    print(f"  {n_ok}/{len(summary)} OK")
    return 0 if n_ok == len(summary) else 1


if __name__ == "__main__":
    sys.exit(main())
