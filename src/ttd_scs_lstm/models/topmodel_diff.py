"""Topmodel diferenciável (Beven & Kirkby 1979) com estado em PyTorch.

Núcleo do gerador de runoff do Caminho 3. Entra a chuva, sai (Pe, Q_bf, A_sat).
A fração saturada A_sat é calculada via sigmoide suave sobre o histograma TWI
da ottobacia — diferenciável em todos os pontos.

Em bacia única (piloto), m, T_0 e a temperatura da sigmoide são parâmetros
aprendíveis globais. Em multi-bacia, virão de um encoder MLP de atributos.

Convenções:
  P, PET, Pe, Q_bf  em mm/h
  S_bar             em mm (déficit médio de saturação)
  A_sat             em fração 0-1
  area              em km²
  output em m³/s    via aggregate_to_basin
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def _inv_sigmoid(x: float, x_min: float, x_max: float) -> torch.Tensor:
    """Inverso da sigmoide com range — para inicializar parâmetros aprendíveis."""
    norm = (x - x_min) / (x_max - x_min)
    norm = float(np.clip(norm, 1e-6, 1 - 1e-6))
    return torch.tensor(np.log(norm / (1.0 - norm)), dtype=torch.float32)


class TopmodelDiff(nn.Module):
    """Topmodel diferenciável.

    Forward:
        P:    (batch, T, n_otto)  chuva mm/h por ottobacia
        PET:  (batch, T)          PET mm/h escalar bacia
    Returns dict com Pe, Q_bf, A_sat, S_bar, m, T_0, sigmoid_temp.
    """

    def __init__(
        self,
        twi_dist: torch.Tensor,    # (n_otto, n_bins) — fração de área em cada bin
        twi_centers: torch.Tensor,  # (n_bins,) — centros dos bins
        twi_mean: torch.Tensor,     # (n_otto,) — λ̄ por ottobacia
        area_km2: torch.Tensor,     # (n_otto,) — área de cada ottobacia
        m_init: float = 30.0,
        T_0_init: float = 0.02,    # mm/h por ottobacia (ordem de grandeza realista)
        S_bar_init: float = 70.0,  # B2 (auditoria 20/06): ~equilibrio observado (~60-85 mm).
                                    # 200 mm estrangulava A_sat (~5% vs ~17%) e cortava Pe 3-6x.
                                    # FIXO (buffer); nao aprendivel (grad ~1e-32 sob TBPTT).
        m_range: tuple[float, float] = (5.0, 200.0),
        T_0_range: tuple[float, float] = (0.001, 0.5),  # mm/h, baseflow plausível
        S_bar_min: float = -50.0,
        S_bar_max: float = 800.0,
        sigmoid_temp_init: float = 5.0,
        sigmoid_temp_range: tuple[float, float] = (0.5, 50.0),
        et_scale: float = 100.0,
        learnable: bool = True,
    ) -> None:
        super().__init__()

        n_otto, n_bins = twi_dist.shape
        if twi_centers.shape != (n_bins,):
            raise ValueError(f"twi_centers shape {twi_centers.shape} != ({n_bins},)")
        if twi_mean.shape != (n_otto,):
            raise ValueError(f"twi_mean shape {twi_mean.shape} != ({n_otto},)")
        if area_km2.shape != (n_otto,):
            raise ValueError(f"area_km2 shape {area_km2.shape} != ({n_otto},)")

        self.n_otto = n_otto
        self.n_bins = n_bins
        self.S_bar_init_value = S_bar_init
        self.S_bar_min = S_bar_min
        self.S_bar_max = S_bar_max
        self.et_scale = et_scale
        self.m_range = m_range
        self.T_0_range = T_0_range
        self.sigmoid_temp_range = sigmoid_temp_range
        self.learnable = learnable

        self.register_buffer("twi_dist", twi_dist.float())
        self.register_buffer("twi_centers", twi_centers.float())
        self.register_buffer("twi_mean", twi_mean.float())
        self.register_buffer("area_km2", area_km2.float())

        m_logit = _inv_sigmoid(m_init, *m_range)
        T_0_logit = _inv_sigmoid(T_0_init, *T_0_range)
        temp_logit = _inv_sigmoid(sigmoid_temp_init, *sigmoid_temp_range)
        # B2 (auditoria 20/06): S_bar_init e SEMPRE buffer fixo, mesmo com learnable=True.
        # O gradiente que chega nele sob TBPTT (detach entre chunks) e ~1e-32 (zero de maquina),
        # entao "aprendivel" era nominal e o valor ficava preso no init. Fixamos no equilibrio
        # observado e tiramos do grafo, para nao estrangular a fisica nem fingir que e aprendido.
        self.S_bar_init_range = (0.0, 400.0)
        sbi_logit = _inv_sigmoid(S_bar_init, *self.S_bar_init_range)
        self.register_buffer("S_bar_init_logit", sbi_logit)

        if learnable:
            self.m_logit = nn.Parameter(m_logit)
            self.T_0_logit = nn.Parameter(T_0_logit)
            self.sigmoid_temp_logit = nn.Parameter(temp_logit)
        else:
            self.register_buffer("m_logit", m_logit)
            self.register_buffer("T_0_logit", T_0_logit)
            self.register_buffer("sigmoid_temp_logit", temp_logit)

    def _bounded(self, logit: torch.Tensor, bounds: Sequence[float]) -> torch.Tensor:
        lo, hi = bounds
        return lo + (hi - lo) * torch.sigmoid(logit)

    @property
    def m(self) -> torch.Tensor:
        return self._bounded(self.m_logit, self.m_range)

    @property
    def T_0(self) -> torch.Tensor:
        return self._bounded(self.T_0_logit, self.T_0_range)

    @property
    def sigmoid_temp(self) -> torch.Tensor:
        return self._bounded(self.sigmoid_temp_logit, self.sigmoid_temp_range)

    @property
    def S_bar_init_learned(self) -> torch.Tensor:
        """Deficit inicial FIXO (B2, 20/06), em mm. Buffer fora do grafo; nome legado mantido."""
        return self._bounded(self.S_bar_init_logit, self.S_bar_init_range)

    def _process_chunk(
        self,
        P_chunk: torch.Tensor,
        PET_chunk: torch.Tensor,
        S_bar: torch.Tensor,
        m: torch.Tensor,
        T_0: torch.Tensor,
        temp: torch.Tensor,
        dt_hours: float,
    ) -> tuple[torch.Tensor, ...]:
        # Loop interno de um chunk TBPTT. Retorna (Pe, Q_bf, A_sat, S_bar_seq, S_bar_final).
        batch, T_chunk, n_otto = P_chunk.shape

        twi_centers = self.twi_centers.view(1, 1, -1)
        twi_mean = self.twi_mean.view(1, -1, 1)
        twi_dist = self.twi_dist.unsqueeze(0)

        Pe_list: list[torch.Tensor] = []
        Q_bf_list: list[torch.Tensor] = []
        A_sat_list: list[torch.Tensor] = []
        S_bar_list: list[torch.Tensor] = []

        for t in range(T_chunk):
            P_t = P_chunk[:, t, :]
            pet_t = PET_chunk[:, t]  # (B,) lumped OU (B, n_otto) por ottobacia
            PET_t = pet_t if pet_t.dim() == 2 else pet_t.unsqueeze(-1).expand(-1, n_otto)

            S_i = S_bar.unsqueeze(-1) + m * (twi_mean - twi_centers)
            sat_prob = torch.sigmoid(-S_i / temp)
            A_sat = (sat_prob * twi_dist).sum(dim=-1).clamp(0.0, 1.0)

            Pe_t = P_t * A_sat
            P_inf = P_t * (1.0 - A_sat)
            Q_bf_t = T_0 * torch.exp(-S_bar / m)
            ET_t = PET_t * torch.exp(-torch.relu(S_bar) / self.et_scale)

            S_bar = S_bar + (ET_t + Q_bf_t - P_inf) * dt_hours
            S_bar = S_bar.clamp(self.S_bar_min, self.S_bar_max)

            Pe_list.append(Pe_t)
            Q_bf_list.append(Q_bf_t)
            A_sat_list.append(A_sat)
            S_bar_list.append(S_bar)

        Pe = torch.stack(Pe_list, dim=1)
        Q_bf = torch.stack(Q_bf_list, dim=1)
        A_sat = torch.stack(A_sat_list, dim=1)
        S_bar_seq = torch.stack(S_bar_list, dim=1)
        return Pe, Q_bf, A_sat, S_bar_seq, S_bar

    def forward(
        self,
        P: torch.Tensor,
        PET: torch.Tensor,
        S_bar_init: torch.Tensor | None = None,
        dt_hours: float = 1.0,
        tbptt_steps: int = 24,
        use_checkpoint: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        if P.dim() != 3:
            raise ValueError(f"P deve ser (batch, T, n_otto), recebido {P.shape}")
        if PET.dim() not in (2, 3):
            raise ValueError(f"PET deve ser (batch, T) [lumped] ou (batch, T, n_otto) [por ottobacia], recebido {PET.shape}")

        batch, T_steps, n_otto = P.shape
        if n_otto != self.n_otto:
            raise ValueError(f"n_otto {n_otto} != {self.n_otto}")

        device = P.device
        dtype = P.dtype

        m = self.m
        T_0 = self.T_0
        temp = self.sigmoid_temp

        if S_bar_init is None:
            # 2B: deficit inicial APRENDIVEL (antes torch.full com 200 fixo = cold-start)
            S_bar = self.S_bar_init_learned.reshape(1, 1).expand(batch, n_otto).contiguous()
            S_bar = S_bar.to(device=device, dtype=dtype)
        else:
            S_bar = S_bar_init.expand(batch, n_otto).clone().to(device=device, dtype=dtype)

        # Default: usar gradient checkpointing apenas em treino com parâmetros aprendíveis.
        # Reduz memória do autograd e quebra o stream CUDA em chunks pequenos
        # (evita TDR timeout no driver NVIDIA com Topmodel learnable).
        if use_checkpoint is None:
            use_checkpoint = self.training and self.learnable

        chunk_size = tbptt_steps if tbptt_steps > 0 else T_steps
        n_chunks = (T_steps + chunk_size - 1) // chunk_size

        Pe_chunks: list[torch.Tensor] = []
        Q_bf_chunks: list[torch.Tensor] = []
        A_sat_chunks: list[torch.Tensor] = []
        S_bar_chunks: list[torch.Tensor] = []

        for ci in range(n_chunks):
            t0 = ci * chunk_size
            t1 = min(t0 + chunk_size, T_steps)
            P_chunk = P[:, t0:t1, :]
            PET_chunk = PET[:, t0:t1]

            if use_checkpoint:
                Pe_c, Q_bf_c, A_sat_c, S_bar_c, S_bar = checkpoint(
                    self._process_chunk,
                    P_chunk, PET_chunk, S_bar, m, T_0, temp, dt_hours,
                    use_reentrant=False,
                )
            else:
                Pe_c, Q_bf_c, A_sat_c, S_bar_c, S_bar = self._process_chunk(
                    P_chunk, PET_chunk, S_bar, m, T_0, temp, dt_hours,
                )

            Pe_chunks.append(Pe_c)
            Q_bf_chunks.append(Q_bf_c)
            A_sat_chunks.append(A_sat_c)
            S_bar_chunks.append(S_bar_c)

            # TBPTT entre chunks: detach do estado pra não acumular histórico de autograd.
            if tbptt_steps > 0 and ci < n_chunks - 1:
                S_bar = S_bar.detach()

        return {
            "Pe": torch.cat(Pe_chunks, dim=1),
            "Q_bf": torch.cat(Q_bf_chunks, dim=1),
            "A_sat": torch.cat(A_sat_chunks, dim=1),
            "S_bar": torch.cat(S_bar_chunks, dim=1),
            "m": m,
            "T_0": T_0,
            "sigmoid_temp": temp,
        }

    def aggregate_to_basin(self, per_otto_mm_h: torch.Tensor) -> torch.Tensor:
        """Converte mm/h por ottobacia em m³/s no exutório.

        per_otto_mm_h: (batch, T, n_otto) em mm/h
        return: (batch, T) em m³/s

        Conversão: mm/h × km² = (1e-3 m × 1e6 m²) / 3600 s = m³/s × (1/3.6)
        Logo: m³/s = mm/h × km² / 3.6
        """
        area = self.area_km2.view(1, 1, -1)
        flow = (per_otto_mm_h * area / 3.6).sum(dim=-1)
        return flow

    def parametros_aprendidos(self) -> dict[str, float]:
        return {
            "m": float(self.m.detach().cpu()),
            "T_0": float(self.T_0.detach().cpu()),
            "sigmoid_temp": float(self.sigmoid_temp.detach().cpu()),
        }
