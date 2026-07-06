"""Features temporais — fonte unica de verdade (fix do review 10/06/2026).

Convencao do projeto: os timestamps dos .h5 sao unix epoch em UTC (verificado
empiricamente em 27/05 e 03/06 — ver memoria convencao-timezone). A feature
hour/month usa hora UTC, a MESMA convencao do train.py reformado da big ablacao
(que treinou os 600 runs oficiais). Antes deste helper, simulate_v3.py e o
dataset do pacote usavam datetime.fromtimestamp (timezone local, BRT) — skew
de 3h treino->simulacao.
"""

import numpy as np
import pandas as pd


def features_temporais(timestamps) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (hours, months) normalizados a partir de unix epoch UTC.

    hours: hora UTC / 23 em [0, 1]
    months: (mes - 1) / 11 em [0, 1]

    Identico ao calculo embutido no train.py (pd.to_datetime interpreta epoch
    como UTC) — qualquer mudanca aqui quebra a consistencia treino/inferencia.
    """
    dt = pd.to_datetime(np.asarray(timestamps), unit='s')
    hours = dt.hour.values / 23.0
    months = (dt.month.values - 1) / 11.0
    return hours, months
