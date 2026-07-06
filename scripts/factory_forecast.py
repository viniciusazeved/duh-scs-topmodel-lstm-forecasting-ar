"""Factory enxuto do fatorial do Paper 2 (forecasting AR + GraphCast).

Cobre SO os model_types das 5 arquiteturas A1-A5 + a ancora SCS-CN PeOnly, propagando
as flags use_ar (canal autorregressivo, ja implementado) e use_gc (chuva futura, em
implementacao). Evita os 23 branches do create_model legado (que fica intacto p/ a
simulacao). Importar de forecasting_v2/src.

Mapa A1-A5 -> wrapper:
  A1 lstm_lumped_wmean            -> LSTMLumpedWeighted (rain+cal, media ponderada de area)
  A2 lstm                        -> LSTMDistributed
  A3 lstm_duh_base_scs           -> LSTMWithTTDSCS (base, learnable)
  A4 lstm_duh_base_topmodel      -> LSTMWithTTDTopmodel (base, learnable)
  A5 lstm_duh_base_topmodel_peonly -> LSTMWithTTDTopmodel (pe_only)
  ancora lstm_duh_base_scs_peonly  -> LSTMWithTTDSCS (pe_only)  [colapso "o gerador decide"]
"""

from __future__ import annotations

from ttd_scs_lstm.models.models import (
    LSTMDistributed,
    LSTMLumpedWeighted,
    LSTMWithTTDSCS,
    LSTMWithTTDTopmodel,
)

# model_types do fatorial (nomes reais aceitos pela grade de simulacao)
FATORIAL = [
    "lstm_lumped_wmean",              # A1
    "lstm",                           # A2
    "lstm_duh_base_scs",              # A3
    "lstm_duh_base_topmodel",         # A4
    "lstm_duh_base_topmodel_peonly",  # A5
    "lstm_duh_base_scs_peonly",       # ancora
]


def create_forecast_model(model_type, static, hidden_size=64, num_layers=2,
                          dropout=0.1, horizon=24, use_ar=False, use_gc=False,
                          device="cuda"):
    """Instancia um modelo do fatorial do Paper 2 com os canais AR/GC."""
    mt = model_type.lower()
    n = static["n_otto"]
    cn = static["cn_values"]
    tcb = static["tc_base_values"]
    area = static["area_km2"]
    twi = (static.get("twi_dist"), static.get("twi_centers"), static.get("twi_mean"))
    common = dict(hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                  horizon=horizon, use_ar=use_ar)
    # use_gc so e repassado aos wrappers quando ja implementado neles (evita TypeError)
    if use_gc:
        common["use_gc"] = use_gc

    if mt == "lstm_lumped_wmean":
        m = LSTMLumpedWeighted(n, area, use_rain=True, use_cal=True, **common)
    elif mt == "lstm":
        m = LSTMDistributed(n, **common)
    elif mt == "lstm_duh_base_scs":
        m = LSTMWithTTDSCS(n, cn, tcb, area, tc_type="base", learnable=True, **common)
    elif mt == "lstm_duh_base_scs_peonly":
        m = LSTMWithTTDSCS(n, cn, tcb, area, tc_type="base", learnable=True,
                           pe_only=True, **common)
    elif mt == "lstm_duh_base_topmodel":
        m = LSTMWithTTDTopmodel(n, twi[0], twi[1], twi[2], tcb, area,
                                tc_type="base", learnable=True, **common)
    elif mt == "lstm_duh_base_topmodel_peonly":
        m = LSTMWithTTDTopmodel(n, twi[0], twi[1], twi[2], tcb, area,
                                tc_type="base", learnable=True, pe_only=True, **common)
    else:
        raise ValueError(f"model_type '{model_type}' fora do fatorial do Paper 2")
    return m.to(device)
