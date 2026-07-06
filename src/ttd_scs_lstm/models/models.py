#!/usr/bin/env python
"""
Modelos para Ablacao - TTD-SCS-LSTM (v2)
========================================

10 modelos para estudo de ablacao completo.

Estrutura:
- 2 baselines (Lumped vs Distribuido)
- 4 modelos TTD (Base/Manning x Fixo/Ajustavel)
- 4 modelos TTD+SCS (Base/Manning x Fixo/Ajustavel)

Hipoteses:
- H1: Distribuido > Lumped
- H2: Manning > Base (rugosidade importa?)
- H3: Ajustavel > Fixo (calibracao melhora?)
- H4: SCS adiciona valor?
- H5: Modelo completo > todos

Autor: Claude + Vinicius
Data: 2026-01-22
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional


# ==============================================================================
# CAMADAS DE FISICA DIFERENCIAVEL
# ==============================================================================

def _inv_sigmoid(p: float) -> torch.Tensor:
    """Logit (inverso da sigmoide), p em (0,1). Usado para inicializar parametros
    sigmoid-bounded EXATAMENTE num valor fisico alvo (mesmo padrao do topmodel_diff)."""
    return torch.log(torch.tensor(p / (1.0 - p)))


class SCSLayer(nn.Module):
    """
    Camada SCS-CN diferenciavel, aplicada POR EVENTO (corrigido 2026-06-15).

    O SCS-CN e um metodo de EVENTO: P e a lamina acumulada da tempestade e Ia e
    subtraido uma unica vez. A versao anterior aplicava (P-Ia)^2/(P-Ia+S) sobre a
    chuva horaria ISOLADA, o que zerava Pe em ~99.99% das celulas (coef escoamento
    ~0.06%, vs ~8% por evento). Esta versao acumula P dentro do evento e devolve o
    Pe INCREMENTAL do timestep; o acumulador reseta apos `dry_reset_h` horas
    consecutivas com P <= `wet_thresh` (fim de evento).

    dry_reset_h = 18 h = IETD da bacia, estimado pelo metodo do coeficiente de
    variacao (Restrepo-Posada & Eagleson 1982): intervalo seco em que o CV dos
    tempos entre eventos ~= 1 (eventos independentes). wet_thresh=0.1 mm/h.
    Ver CORRECAO_FISICA.md. CN fixo (BHAE_CN-2022 da ANA).
    """

    def __init__(self, cn_values: torch.Tensor, learnable: bool = True,
                 dry_reset_h: int = 18, wet_thresh: float = 0.1):
        super().__init__()
        self.learnable = learnable
        self.dry_reset_h = int(dry_reset_h)
        self.wet_thresh = float(wet_thresh)
        self.register_buffer('cn', cn_values.float())
        S = 25400.0 / cn_values.float() - 254.0
        self.register_buffer('S', S)
        if learnable:
            self._lambda_logit = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer('_lambda_fixed', torch.tensor(0.2))

    @property
    def lambda_scs(self) -> torch.Tensor:
        """Lambda entre 0.01 e 0.40"""
        if self.learnable:
            return 0.01 + 0.39 * torch.sigmoid(self._lambda_logit)
        else:
            return self._lambda_fixed

    def forward(self, P: torch.Tensor, scs_state=None, return_state: bool = False):
        """
        Args:
            P: Precipitacao horaria (batch, seq_len, n_otto) em mm/h.
            scs_state: (continuo) tupla (Pacum, Pe_acum, dry) do chunk anterior; None reinicia.
            return_state: se True, retorna (Pe, (Pacum, Pe_acum, dry)) para o proximo chunk.
        Returns:
            Pe: Precipitacao efetiva incremental por timestep, mesmo shape de P.
        """
        B, T, N = P.shape
        S = self.S.view(1, -1)              # (1, N)
        Ia = self.lambda_scs * S            # (1, N) — diferenciavel em lambda
        if scs_state is None:
            Pacum = torch.zeros(B, N, device=P.device, dtype=P.dtype)
            Pe_acum = torch.zeros(B, N, device=P.device, dtype=P.dtype)
            dry = torch.zeros(B, N, device=P.device, dtype=P.dtype)
        else:
            Pacum, Pe_acum, dry = scs_state           # carry do chunk anterior (acumulador de evento)
        out = []
        for t in range(T):
            Pt = P[:, t, :]                                       # (B, N)
            dry = torch.where(Pt <= self.wet_thresh, dry + 1.0, torch.zeros_like(dry))
            reset = (dry >= self.dry_reset_h)                     # fim de evento
            Pacum = torch.where(reset, torch.zeros_like(Pacum), Pacum) + Pt
            Pe_ref = torch.where(reset, torch.zeros_like(Pe_acum), Pe_acum)
            excess = F.relu(Pacum - Ia)
            Pe_acum = excess * excess / (excess + S + 1e-6)       # Pe acumulado do evento
            out.append(F.relu(Pe_acum - Pe_ref))                  # incremento do timestep
        Pe = torch.stack(out, dim=1)                             # (B, T, N)
        if return_state:
            return Pe, (Pacum, Pe_acum, dry)
        return Pe


class DUHLayer(nn.Module):
    """
    Camada TTD (Travel Time Distribution) diferenciavel.

    Roteia Pe -> Q usando convolucao com IUH gaussiano.
    Parametros aprendiveis: tc_scale, sigma.

    Tc pode ser:
    - tc_base_h: Maidment (1996) puro (topografia)
    - tc_manning_h: Maidment + Manning (rugosidade LULC)
    """

    def __init__(
        self,
        tc_values: torch.Tensor,
        area_km2: torch.Tensor,
        n_bins: int = 120,
        dt_hours: float = 1.0,
        learnable: bool = True,
        iuh: str = 'gauss',
        impulse: bool = False,
    ):
        """
        Args:
            tc_values: Tc por ottobacia (n_otto,) em horas
            area_km2: Area por ottobacia (n_otto,)
            n_bins: PISO do numero de bins do IUH (o efetivo e recalculado para
                cobrir 5*tc_max + 4*sigma_max no espacamento dt_hours)
            dt_hours: Resolucao temporal (horas) — passo dos bins E da serie
            learnable: Se True, tc_scale e sigma sao aprendiveis
            iuh: 'gauss' (gaussiana, default) ou 'gamma' (Nash, assimetrica)
        """
        super().__init__()

        self.n_otto = len(tc_values)
        self.dt_hours = dt_hours
        self.learnable = learnable
        self.iuh = iuh
        # impulse=True: roteamento-impulso (IUH = delta em t=0), i.e. soma ponderada por area
        # instantanea sem convolucao. Isola o EFEITO DO ROTEAMENTO: compara geracao com vs sem IUH
        # explicito, testando se a LSTM distribuida ja roteia sozinha (decisao 21/06).
        self.impulse = impulse

        self.register_buffer('tc_base', tc_values.float())
        self.register_buffer('area_km2', area_km2.float())

        # Conversao mm/h -> m3/s por ottobacia
        conversion = area_km2.float() * 1e6 / (dt_hours * 3600) / 1000
        self.register_buffer('conversion', conversion)

        # Bins de tempo do IUH: espacamento FIXO = dt_hours — o conv1d aplica o tap k
        # no lag k*dt, entao o grid TEM que casar com o passo da serie (fix do review
        # 10/06/2026: o linspace antigo comprimia o IUH ~1,9x quando 3*tc_max > 120h,
        # caso dos Tc Manning). n_bins dinamico cobre o pior caso do range learnable
        # (tc_scale max 5.0) + cauda gaussiana (4*sigma max 15h); piso = n_bins pedido.
        t_max = 5.0 * float(tc_values.max()) + 4.0 * 15.0
        n_bins = max(n_bins, int(np.ceil(t_max / dt_hours)) + 1)
        self.n_bins = n_bins
        # persistent=False: recomputado no __init__, NAO entra no state_dict — o shape
        # agora depende de tc_max, e a chave persistente quebraria o load de checkpoints
        # entre versoes (os antigos de 120 bins exigem o codigo da main de toda forma).
        self.register_buffer('bin_centers', torch.arange(n_bins, dtype=torch.float32) * dt_hours,
                             persistent=False)

        # Parametros aprendiveis ou fixos
        if learnable:
            # Init EXATAMENTE nos valores das variantes Fixed (ablacao learnable vs
            # fixed limpa — fix do review 10/06/2026; antes nascia em 2.55/11.4h por
            # resquicio de parametrizacao exp antiga, dai os nomes _log_*).
            self._log_tc_scale = nn.Parameter(_inv_sigmoid((1.0 - 0.1) / 4.9))   # tc_scale = 1.0
            self._log_sigma = nn.Parameter(_inv_sigmoid((3.0 - 0.5) / 14.5))     # sigma = 3.0 h
            self._log_shape = nn.Parameter(_inv_sigmoid((2.0 - 1.0) / 9.0))      # shape gamma = 2.0
        else:
            self.register_buffer('_tc_scale_fixed', torch.tensor(1.0))
            self.register_buffer('_sigma_fixed', torch.tensor(3.0))
            self.register_buffer('_shape_fixed', torch.tensor(2.0))

    @property
    def tc_scale(self) -> torch.Tensor:
        """tc_scale entre 0.1 e 5.0"""
        if self.learnable:
            return 0.1 + 4.9 * torch.sigmoid(self._log_tc_scale)
        else:
            return self._tc_scale_fixed

    @property
    def sigma(self) -> torch.Tensor:
        """sigma entre 0.5 e 15.0"""
        if self.learnable:
            return 0.5 + 14.5 * torch.sigmoid(self._log_sigma)
        else:
            return self._sigma_fixed

    @property
    def shape(self) -> torch.Tensor:
        """forma n da gamma/Nash, entre 1.0 e 10.0 (CV = 1/sqrt(n))"""
        if self.learnable:
            return 1.0 + 9.0 * torch.sigmoid(self._log_shape)
        else:
            return self._shape_fixed

    def compute_iuh(self) -> torch.Tensor:
        """Computa IUH (n_otto, n_bins): gaussiano (default) ou gamma/Nash."""
        tc_scaled = self.tc_base * self.tc_scale  # media desejada da IUH
        t = self.bin_centers.unsqueeze(0)  # (1, n_bins)

        if self.iuh == 'gamma':
            # Gamma/Nash: assimetrica, nasce em t=0, cauda longa, sem massa em t<0.
            # media = n*k = tc_scaled -> k = tc_scaled/n. Forma n controla dispersao
            # (CV = 1/sqrt(n)), entao a dispersao escala com Tc automaticamente.
            n = self.shape
            k = (tc_scaled / n).unsqueeze(1).clamp(min=1e-3)  # (n_otto, 1)
            logh = (n - 1.0) * torch.log(t / k + 1e-9) - (t / k)
            weights = torch.exp(logh) * (t > 0).float()
        else:
            tc = tc_scaled.unsqueeze(1)
            dist = (t - tc) / (self.sigma + 1e-6)
            weights = torch.exp(-0.5 * dist ** 2) * (t >= 0).float()

        return weights / (weights.sum(dim=-1, keepdim=True) + 1e-9)

    def forward(self, Pe: torch.Tensor, pe_tail: torch.Tensor | None = None,
                return_tail: bool = False):
        """
        Args:
            Pe: Precipitacao efetiva (batch, seq_len, n_otto)
            pe_tail: (continuo) cauda de Pe do chunk anterior (batch, n_bins-1, n_otto) para
                overlap-save na borda do chunk; None no 1o chunk / forecasting.
            return_tail: se True, retorna (Q, nova_cauda) para o proximo chunk.
        Returns:
            Q: Vazao (batch, seq_len)  [e a nova cauda, se return_tail]
        """
        # Overlap-save: prepend a cauda do chunk anterior para a convolucao causal nao serrilhar
        # na borda; descarta as saidas correspondentes a cauda. Equivale a convoluir a serie inteira.
        if pe_tail is not None:
            Pe_in = torch.cat([pe_tail, Pe], dim=1)
        else:
            Pe_in = Pe
        L_in = Pe_in.shape[1]
        n_otto = Pe_in.shape[2]

        # Roteamento-impulso: soma ponderada por area instantanea, sem convolucao temporal.
        if self.impulse:
            Q_total = F.relu((Pe_in * self.conversion.view(1, 1, -1)).sum(dim=-1))  # (B, L_in)
            if pe_tail is not None:
                Q_total = Q_total[:, pe_tail.shape[1]:]
            if return_tail:
                return Q_total, torch.zeros_like(Pe_in[:, -(self.n_bins - 1):, :])
            return Q_total

        Pe_volume = Pe_in * self.conversion.view(1, 1, -1)
        iuh = self.compute_iuh()  # (n_otto, n_bins)
        Pe_conv = Pe_volume.permute(0, 2, 1)  # (batch, n_otto, L_in)
        kernel = iuh.flip(1).unsqueeze(1)  # (n_otto, 1, n_bins)

        Q_otto = F.conv1d(Pe_conv, kernel, padding=self.n_bins - 1, groups=n_otto)
        Q_otto = Q_otto[:, :, :L_in]
        Q_total = F.relu(Q_otto.sum(dim=1))  # (batch, L_in)

        if pe_tail is not None:
            Q_total = Q_total[:, pe_tail.shape[1]:]   # descarta a parte da cauda prependada

        if return_tail:
            new_tail = Pe_in[:, -(self.n_bins - 1):, :]   # cauda para o proximo chunk
            return Q_total, new_tail
        return Q_total


# ==============================================================================
# MODELOS DE ABLACAO (v2 - 10 modelos)
# ==============================================================================

class LSTMLumped(nn.Module):
    """
    Modelo 1: LSTM Lumped (Baseline)

    Arquitetura: P_total -> LSTM -> Q
    Precipitacao agregada (media espacial), sem fisica.
    """

    def __init__(
        self,
        n_otto: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 24
    ):
        super().__init__()

        self.n_otto = n_otto
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.name = "LSTM_Lumped"

        # Input: P_media + hora + mes
        input_size = 1 + 2

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon)
        )

    def forward(
        self,
        precip: torch.Tensor,
        hour: torch.Tensor,
        month: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        batch_size, seq_len, n_otto = precip.shape

        # Agregar precipitacao (media espacial) -> Lumped
        precip_mean = precip.mean(dim=-1, keepdim=True)  # (batch, seq, 1)
        precip_feat = torch.log1p(precip_mean)

        hour_feat = hour.unsqueeze(-1)
        month_feat = month.unsqueeze(-1)

        x = torch.cat([precip_feat, hour_feat, month_feat], dim=-1)

        # LSTM
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]

        # Decoder
        pred_log = self.decoder(last_hidden)
        pred = torch.expm1(F.relu(pred_log))

        return {
            'pred': pred_log,
            'pred_exp': pred
        }

    def get_learned_params(self) -> Dict[str, float]:
        return {}


class LSTMLumpedWeighted(nn.Module):
    """Lumped com média PONDERADA por área e canais configuráveis (cadeia 'de onde vem o skill').

    Diferente do LSTMLumped (média espacial simples), agrega a chuva pela lâmina média real da
    bacia: Pag = sum(P*area)/sum(area). As flags isolam a fonte de skill:
      use_rain + use_cal -> Lumped  (chuva ponderada + hora/mês)   [modo B]
      use_rain           -> RainOnly (só chuva ponderada)          [modo A]
      use_cal            -> CalOnly  (só hora/mês, sem chuva)       [modo C]
    Calendário (hour/month) chega já normalizado e em UTC do dataset (temporal.features_temporais).
    """

    def __init__(
        self,
        n_otto: int,
        area_km2: torch.Tensor,
        use_rain: bool = True,
        use_cal: bool = True,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 24,
        continuous: bool = False,
        use_ar: bool = False,
        use_gc: bool = False,
    ):
        super().__init__()
        assert use_rain or use_cal, "LSTMLumpedWeighted precisa de ao menos um canal"
        self.n_otto = n_otto
        self.use_rain = use_rain
        self.use_cal = use_cal
        self.horizon = horizon
        self.continuous = continuous
        self.use_ar = use_ar  # canal autorregressivo: vazao observada defasada no encoder
        # nome limpo por modo (vira a pasta-leaf de saida e a chave de consolidacao)
        if use_rain and use_cal:
            self.name = "LSTM_Lumped"        # modo B: chuva ponderada + calendario
        elif use_rain:
            self.name = "LSTM_Lumped_RainOnly"  # modo A
        else:
            self.name = "LSTM_Lumped_CalOnly"   # modo C
        if use_ar:
            self.name += "_AR"
        self.use_gc = use_gc
        if use_gc:
            self.name += "_GC"
            self.gc_proj = nn.Linear(horizon, hidden_size)   # canal GC: projeta a chuva futura (B,H) no estado
            nn.init.zeros_(self.gc_proj.weight); nn.init.zeros_(self.gc_proj.bias)  # zero-init: comeca neutro, aprende o GC

        # pesos de área normalizados (lâmina média da bacia), fixos
        self.register_buffer('area_w', area_km2.float() / area_km2.float().sum())

        input_size = (1 if use_rain else 0) + (2 if use_cal else 0) + (1 if use_ar else 0)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        out_dim = 1 if continuous else horizon
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_dim)
        )

    def forward(
        self,
        precip: torch.Tensor,
        hour: torch.Tensor,
        month: torch.Tensor,
        state=None,
        q_past: torch.Tensor | None = None,
        precip_fut: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        chans = []
        if self.use_rain:
            # média areal ponderada: (batch, seq, n_otto) * (n_otto,) -> soma em n_otto
            precip_w = (precip * self.area_w).sum(dim=-1, keepdim=True)  # (batch, seq, 1)
            chans.append(torch.log1p(precip_w))
        if self.use_cal:
            chans.append(hour.unsqueeze(-1))
            chans.append(month.unsqueeze(-1))
        if self.use_ar:
            chans.append(torch.log1p(q_past).unsqueeze(-1))   # canal AR: vazao observada defasada
        x = torch.cat(chans, dim=-1)

        if self.continuous:
            st = state or {}
            lstm_out, (hn, cn) = self.lstm(x, st.get("lstm"))
            pred_log = self.decoder(lstm_out).squeeze(-1)
            pred = torch.expm1(F.relu(pred_log))
            return {'pred': pred_log, 'pred_exp': pred, 'state': {"lstm": (hn, cn)}}

        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        if getattr(self, 'use_gc', False):
            pf_lumped = precip_fut.mean(-1) if precip_fut.dim() == 3 else precip_fut   # (B,H) media da bacia
            last_hidden = last_hidden + self.gc_proj(torch.log1p(pf_lumped))   # canal GC (baseline sem DUH)
        pred_log = self.decoder(last_hidden)
        pred = torch.expm1(F.relu(pred_log))
        return {'pred': pred_log, 'pred_exp': pred}

    def get_learned_params(self) -> Dict[str, float]:
        return {}


class LSTMDistributed(nn.Module):
    """
    Modelo 2: LSTM Distribuido (Baseline)

    Arquitetura: P_245otto -> LSTM -> Q
    Precipitacao por ottobacia, sem fisica.
    """

    def __init__(
        self,
        n_otto: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 24,
        continuous: bool = False,
        use_pet: bool = False,
        use_ar: bool = False,
        use_gc: bool = False,
    ):
        super().__init__()

        self.n_otto = n_otto
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.continuous = continuous   # simulacao continua: decoder seq2seq + estado stateful
        # use_pet=True: PET media da bacia (1 serie) como feature da LSTM (forcante evaporativa,
        # alem do calendario). Compativel com PUB (PET vem de reanalise). Decisao 21/06.
        self.use_pet = use_pet
        self.use_ar = use_ar  # canal autorregressivo: vazao observada defasada no encoder
        self.name = "LSTM_PET" if use_pet else "LSTM"
        if use_ar:
            self.name += "_AR"
        self.use_gc = use_gc
        if use_gc:
            self.name += "_GC"
            self.gc_proj = nn.Linear(horizon, hidden_size)   # canal GC: projeta a chuva futura (B,H) no estado
            nn.init.zeros_(self.gc_proj.weight); nn.init.zeros_(self.gc_proj.bias)  # zero-init: comeca neutro, aprende o GC

        # Input: P por ottobacia + hora + mes (+ PET media, se use_pet; + Q_obs, se use_ar)
        input_size = n_otto + 2 + (1 if use_pet else 0) + (1 if use_ar else 0)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        out_dim = 1 if continuous else horizon   # continuo: 1 valor por timestep (seq2seq)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_dim)
        )

    def forward(
        self,
        precip: torch.Tensor,
        hour: torch.Tensor,
        month: torch.Tensor,
        pet: torch.Tensor | None = None,
        state=None,
        q_past: torch.Tensor | None = None,
        precip_fut: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass. Em modo continuous, retorna a vazao por timestep (B,T) e o estado (h,c)."""
        batch_size, seq_len, n_otto = precip.shape

        # Features
        precip_feat = torch.log1p(precip)
        hour_feat = hour.unsqueeze(-1)
        month_feat = month.unsqueeze(-1)

        feats = [precip_feat, hour_feat, month_feat]
        if self.use_pet:
            if pet is None:
                raise ValueError("LSTM_PET exige pet no forward (gate do train.py deve passar pet)")
            # PET media da bacia (mm/h) em log1p, mesma escala do precip_feat
            feats.append(torch.log1p(pet.mean(dim=-1, keepdim=True)))
        if self.use_ar:
            feats.append(torch.log1p(q_past).unsqueeze(-1))   # canal AR: vazao observada defasada
        x = torch.cat(feats, dim=-1)

        if self.continuous:
            lstm_out, (hn, cn) = self.lstm(x, state)        # carrega/retorna estado
            pred_log = self.decoder(lstm_out).squeeze(-1)   # (B,T,1)->(B,T): vazao por passo
            pred = torch.expm1(F.relu(pred_log))
            return {'pred': pred_log, 'pred_exp': pred, 'state': (hn, cn)}

        # forecasting (inalterado)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        if getattr(self, 'use_gc', False):
            pf_lumped = precip_fut.mean(-1) if precip_fut.dim() == 3 else precip_fut   # (B,H) media da bacia
            last_hidden = last_hidden + self.gc_proj(torch.log1p(pf_lumped))   # canal GC (baseline sem DUH)
        pred_log = self.decoder(last_hidden)
        pred = torch.expm1(F.relu(pred_log))
        return {'pred': pred_log, 'pred_exp': pred}

    def get_learned_params(self) -> Dict[str, float]:
        return {}


class LSTMAttrs(nn.Module):
    """Modelo M0c (ablacao_v2): LSTM distribuida + atributos por ottobacia, SEM fisica.

    Controle conceitual que separa INFORMACAO (atributo) de ESTRUTURA (equacao).
    Recebe a chuva distribuida (n_otto) E os atributos estaticos por ottobacia
    (CN, Tc_base, TWI_mean), normalizados (z-score), como canais CONSTANTES no tempo —
    em peso comparavel ao da chuva. NAO roda SCS/DUH/TOPMODEL. E o par justo do PeDist:
    'atributo distribuido SEM equacao' (aqui) vs 'Pe distribuido COM equacao' (PeDist).

    Nuance (bacia unica): os atributos sao constantes entre amostras de treino; o sinal
    so pode emergir da interacao nao-linear chuva x atributo que a LSTM tente aprender
    sozinha — exatamente o que a equacao fisica impoe de graca. Resultado esperado:
    ganho pequeno -> reforca que e a ESTRUTURA que importa, nao o atributo cru.
    """

    def __init__(
        self,
        n_otto: int,
        cn_values: torch.Tensor,
        tc_values: torch.Tensor,
        twi_mean: torch.Tensor,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 24,
    ):
        super().__init__()
        self.n_otto = n_otto
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.name = "LSTM_Attrs"

        def _zscore(v: torch.Tensor) -> torch.Tensor:
            v = v.float()
            return (v - v.mean()) / (v.std() + 1e-6)

        # (n_attr, n_otto) — CN, Tc_base, TWI_mean normalizados. Buffer: vai pra GPU
        # com o modelo e NAO recebe gradiente (atributo fixo, igual aos buffers da fisica).
        attrs = torch.stack([_zscore(cn_values), _zscore(tc_values), _zscore(twi_mean)], dim=0)
        self.register_buffer('attrs', attrs.contiguous())
        self.n_attr = attrs.shape[0]

        # Input: P(n_otto) + atributos distribuidos(n_attr*n_otto) + hora + mes
        input_size = n_otto + self.n_attr * n_otto + 2

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon)
        )

    def forward(
        self,
        precip: torch.Tensor,
        hour: torch.Tensor,
        month: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, n_otto = precip.shape

        precip_feat = torch.log1p(precip)                                   # (B, T, N)
        attrs_flat = self.attrs.reshape(1, 1, -1).expand(batch_size, seq_len, -1)  # (B, T, n_attr*N)
        hour_feat = hour.unsqueeze(-1)
        month_feat = month.unsqueeze(-1)

        x = torch.cat([precip_feat, attrs_flat, hour_feat, month_feat], dim=-1)

        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        if getattr(self, 'use_gc', False):
            pf_lumped = precip_fut.mean(-1) if precip_fut.dim() == 3 else precip_fut   # (B,H) media da bacia
            last_hidden = last_hidden + self.gc_proj(torch.log1p(pf_lumped))   # canal GC (baseline sem DUH)
        pred_log = self.decoder(last_hidden)
        pred = torch.expm1(F.relu(pred_log))

        return {'pred': pred_log, 'pred_exp': pred}

    def get_learned_params(self) -> Dict[str, float]:
        return {}


class LSTMWithTTD(nn.Module):
    """
    Modelos 3-6: LSTM + TTD

    Arquitetura: P -> TTD -> Q_routed -> LSTM -> Q

    Variacoes:
    - Modelo 3: TTD Base Fixo
    - Modelo 4: TTD Base Ajustavel
    - Modelo 5: TTD Manning Fixo
    - Modelo 6: TTD Manning Ajustavel
    """

    def __init__(
        self,
        n_otto: int,
        tc_values: torch.Tensor,
        area_km2: torch.Tensor,
        tc_type: str = 'base',  # 'base' ou 'manning'
        learnable: bool = True,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 24,
        continuous: bool = False,
    ):
        super().__init__()

        self.n_otto = n_otto
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.tc_type = tc_type
        self.learnable = learnable
        self.continuous = continuous

        # Nome do modelo
        tc_name = "Base" if tc_type == 'base' else "Manning"
        suffix = "" if learnable else "_Fixed"
        self.name = f"LSTM_DUH_{tc_name}{suffix}"

        # TTD Layer
        self.duh = DUHLayer(tc_values, area_km2, learnable=learnable)

        # Input: P por ottobacia + Q_routed + hora + mes
        input_size = n_otto + 1 + 2

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        out_dim = 1 if continuous else horizon
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_dim)
        )

    def forward(
        self,
        precip: torch.Tensor,
        hour: torch.Tensor,
        month: torch.Tensor,
        state=None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass. Em modo continuous: DUH com overlap-save + LSTM stateful, retorna estado."""
        batch_size, seq_len, n_otto = precip.shape
        hour_feat = hour.unsqueeze(-1)
        month_feat = month.unsqueeze(-1)

        if self.continuous:
            st = state or {}
            Q_routed, duh_tail = self.duh(precip, pe_tail=st.get("duh_tail"), return_tail=True)
            x = torch.cat([torch.log1p(precip), torch.log1p(Q_routed).unsqueeze(-1),
                           hour_feat, month_feat], dim=-1)
            lstm_out, (hn, cn) = self.lstm(x, st.get("lstm"))
            pred_log = self.decoder(lstm_out).squeeze(-1)
            pred = torch.expm1(F.relu(pred_log))
            return {'pred': pred_log, 'pred_exp': pred, 'Q_routed': Q_routed,
                    'state': {"lstm": (hn, cn), "duh_tail": duh_tail}}

        # forecasting (inalterado)
        Q_routed = self.duh(precip)
        x = torch.cat([torch.log1p(precip), torch.log1p(Q_routed).unsqueeze(-1),
                       hour_feat, month_feat], dim=-1)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        if getattr(self, 'use_gc', False):
            pf_lumped = precip_fut.mean(-1) if precip_fut.dim() == 3 else precip_fut   # (B,H) media da bacia
            last_hidden = last_hidden + self.gc_proj(torch.log1p(pf_lumped))   # canal GC (baseline sem DUH)
        pred_log = self.decoder(last_hidden)
        pred = torch.expm1(F.relu(pred_log))
        return {
            'pred': pred_log,
            'pred_exp': pred,
            'Q_routed': Q_routed,
            'tc_scale': self.duh.tc_scale,
            'sigma': self.duh.sigma
        }

    def get_learned_params(self) -> Dict[str, float]:
        if self.learnable:
            return {
                'tc_scale': self.duh.tc_scale.item(),
                'sigma': self.duh.sigma.item()
            }
        return {'tc_scale': 1.0, 'sigma': 3.0}


class LSTMWithTTDSCS(nn.Module):
    """
    Modelos 7-10: LSTM + TTD + SCS

    Arquitetura: P -> SCS -> Pe -> TTD -> Q_routed -> LSTM -> Q

    Variacoes:
    - Modelo 7: TTD Base + SCS Fixo
    - Modelo 8: TTD Base + SCS Ajustavel
    - Modelo 9: TTD Manning + SCS Fixo
    - Modelo 10: TTD Manning + SCS Ajustavel (modelo completo)
    """

    def __init__(
        self,
        n_otto: int,
        cn_values: torch.Tensor,
        tc_values: torch.Tensor,
        area_km2: torch.Tensor,
        tc_type: str = 'base',  # 'base' ou 'manning'
        learnable: bool = True,
        pe_only: bool = False,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 24,
        continuous: bool = False,
        impulse: bool = False,
        use_ar: bool = False,
        use_gc: bool = False,
    ):
        super().__init__()

        self.n_otto = n_otto
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.tc_type = tc_type
        self.learnable = learnable
        self.pe_only = pe_only
        self.continuous = continuous
        self.impulse = impulse
        self.use_ar = use_ar  # canal autorregressivo: vazao observada defasada no encoder

        # Nome do modelo. impulse=True => "sem DUH" (geracao SCS sem roteamento explicito).
        tc_name = "Base" if tc_type == 'base' else "Manning"
        suffix = "" if learnable else "_Fixed"
        peo = "_PeOnly" if pe_only else ""
        self.name = f"LSTM_SCS{suffix}{peo}" if impulse else f"LSTM_DUH_{tc_name}_SCS{suffix}{peo}"
        if use_ar:
            self.name += "_AR"
        self.use_gc = use_gc
        if use_gc:
            self.name += "_GC"
            self.gc_proj = nn.Linear(horizon, hidden_size)   # canal GC: projeta a chuva futura (B,H) no estado
            nn.init.zeros_(self.gc_proj.weight); nn.init.zeros_(self.gc_proj.bias)  # zero-init: comeca neutro, aprende o GC

        # Camadas de fisica
        self.scs = SCSLayer(cn_values, learnable=learnable)
        self.duh = DUHLayer(tc_values, area_km2, learnable=learnable, impulse=impulse)

        # Input do LSTM. Normal: P(n_otto) + Pe_media(1) + Q_routed(1) + hora/mes(2).
        # PeOnly: sem chuva bruta — Pe por ottobacia(n_otto) + Q_routed(1) + hora/mes(2).
        if pe_only:
            input_size = n_otto + 1 + 2
        else:
            input_size = n_otto + 1 + 1 + 2
        if use_ar:
            input_size += 1   # + Q_obs defasada (canal AR)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        out_dim = 1 if continuous else horizon
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_dim)
        )
        # RAMO FISICO (forecasting): o decoder recebe last_hidden + Q_fut roteado ao horizonte
        # (agua em transito da chuva passada + chuva futura prevista quando GC on)
        if not continuous:
            self.decoder[0] = nn.Linear(hidden_size + horizon, hidden_size)

    def _montar_x(self, precip, Pe, Q_routed, hour, month, q_past=None):
        q_routed_feat = torch.log1p(Q_routed).unsqueeze(-1)
        hour_feat = hour.unsqueeze(-1)
        month_feat = month.unsqueeze(-1)
        if self.pe_only:
            chans = [torch.log1p(Pe), q_routed_feat, hour_feat, month_feat]
        else:
            precip_feat = torch.log1p(precip)
            pe_mean_feat = torch.log1p(Pe.mean(dim=-1, keepdim=True))
            chans = [precip_feat, pe_mean_feat, q_routed_feat, hour_feat, month_feat]
        if self.use_ar:
            chans.append(torch.log1p(q_past).unsqueeze(-1))   # canal AR: vazao observada defasada
        return torch.cat(chans, dim=-1)

    def forward(
        self,
        precip: torch.Tensor,
        hour: torch.Tensor,
        month: torch.Tensor,
        state=None,
        q_past: torch.Tensor | None = None,
        precip_fut: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass. Em continuous: SCS/DUH stateful + LSTM stateful, retorna estado."""
        if self.continuous:
            st = state or {}
            Pe, scs_state = self.scs(precip, scs_state=st.get("scs"), return_state=True)
            Q_routed, duh_tail = self.duh(Pe, pe_tail=st.get("duh_tail"), return_tail=True)
            x = self._montar_x(precip, Pe, Q_routed, hour, month, q_past=q_past)
            lstm_out, (hn, cn) = self.lstm(x, st.get("lstm"))
            pred_log = self.decoder(lstm_out).squeeze(-1)
            pred = torch.expm1(F.relu(pred_log))
            return {'pred': pred_log, 'pred_exp': pred, 'Pe': Pe, 'Q_routed': Q_routed,
                    'state': {"scs": scs_state, "duh_tail": duh_tail, "lstm": (hn, cn)}}

        # forecasting com ROTEAMENTO ESTENDIDO ao horizonte (ramo fisico)
        Pe = self.scs(precip)
        Q_routed = self.duh(Pe)
        x = self._montar_x(precip, Pe, Q_routed, hour, month, q_past=q_past)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        # Pe futuro: chuva prevista (GC on) gerada pela mesma SCS, senao zeros (so agua em transito)
        if self.use_gc and precip_fut is not None:
            Pe_fut = self.scs(precip_fut)
        else:
            Pe_fut = torch.zeros(precip.shape[0], self.horizon, precip.shape[2],
                                 device=precip.device, dtype=Pe.dtype)
        Q_fut = self.duh(torch.cat([Pe, Pe_fut], dim=1))[:, -self.horizon:]   # (B,H) agua em transito + chuva futura
        pred_log = self.decoder(torch.cat([last_hidden, torch.log1p(Q_fut)], dim=-1))
        pred = torch.expm1(F.relu(pred_log))
        return {
            'pred': pred_log,
            'pred_exp': pred,
            'Pe': Pe,
            'Q_routed': Q_routed,
            'lambda_scs': self.scs.lambda_scs,
            'tc_scale': self.duh.tc_scale,
            'sigma': self.duh.sigma
        }

    def get_learned_params(self) -> Dict[str, float]:
        if self.learnable:
            return {
                'lambda_scs': self.scs.lambda_scs.item(),
                'tc_scale': self.duh.tc_scale.item(),
                'sigma': self.duh.sigma.item()
            }
        return {'lambda_scs': 0.2, 'tc_scale': 1.0, 'sigma': 3.0}


# ==============================================================================
# LSTM + TOPMODEL + TTD (4 modelos adicionais — adicionados 2026-05-18)
# Simetria com LSTMWithTTDSCS, mas Topmodel diferenciavel no lugar do SCS-CN.
# ==============================================================================

# PET climatologia mensal (Sudeste BR — Manuel Duarte)
_PET_MM_HORA_POR_MES = torch.tensor([
    0.18, 0.18, 0.16, 0.13, 0.10, 0.09,
    0.10, 0.13, 0.16, 0.18, 0.18, 0.18,
])


class LSTMWithTTDTopmodel(nn.Module):
    """LSTM + Topmodel diff + TTD (modelos 11-14).

    Pipeline: P -> Topmodel(P, PET) -> (Pe, Q_bf descartado) -> TTD(Pe) -> Q_routed -> LSTM -> Q

    Variacoes (paralelas as 4 com SCS):
    - Modelo 11: TTD Base + Topmodel Fixo
    - Modelo 12: TTD Base + Topmodel Ajustavel
    - Modelo 13: TTD Manning + Topmodel Fixo
    - Modelo 14: TTD Manning + Topmodel Ajustavel
    """

    def __init__(
        self,
        n_otto: int,
        twi_dist: torch.Tensor,
        twi_centers: torch.Tensor,
        twi_mean: torch.Tensor,
        tc_values: torch.Tensor,
        area_km2: torch.Tensor,
        tc_type: str = 'base',
        learnable: bool = True,
        pe_only: bool = False,
        pe_dist: bool = False,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 24,
        continuous: bool = False,
        baseflow: bool = False,
        impulse: bool = False,
        use_ar: bool = False,
        use_gc: bool = False,
    ):
        super().__init__()
        from .topmodel_diff import TopmodelDiff

        if pe_only and pe_dist:
            raise ValueError("pe_only e pe_dist sao mutuamente exclusivos")

        self.n_otto = n_otto
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.tc_type = tc_type
        self.learnable = learnable
        self.pe_only = pe_only
        self.pe_dist = pe_dist
        self.continuous = continuous
        self.baseflow = baseflow   # braco TOPMODEL+baseflow: S_bar e Q_bf como canais da LSTM (decisao 20/06)
        self.impulse = impulse     # impulse=True => "sem DUH" (geracao TOPMODEL sem roteamento explicito)
        self.use_ar = use_ar       # canal autorregressivo: vazao observada defasada no encoder

        tc_name = "Base" if tc_type == 'base' else "Manning"
        suffix = "" if learnable else "_Fixed"
        peo = "_PeOnly" if pe_only else ""
        ped = "_PeDist" if pe_dist else ""
        bfl = "_Baseflow" if baseflow else ""
        self.name = (f"LSTM_Topmodel{suffix}{peo}{ped}{bfl}" if impulse
                     else f"LSTM_DUH_{tc_name}_Topmodel{suffix}{peo}{ped}{bfl}")
        if use_ar:
            self.name += "_AR"
        self.use_gc = use_gc
        if use_gc:
            self.name += "_GC"
            self.gc_proj = nn.Linear(horizon, hidden_size)   # canal GC: projeta a chuva futura (B,H) no estado
            nn.init.zeros_(self.gc_proj.weight); nn.init.zeros_(self.gc_proj.bias)  # zero-init: comeca neutro, aprende o GC

        self.topmodel = TopmodelDiff(
            twi_dist=twi_dist, twi_centers=twi_centers, twi_mean=twi_mean,
            area_km2=area_km2, learnable=learnable,
        )
        self.duh = DUHLayer(tc_values, area_km2, learnable=learnable, impulse=impulse)

        # Input do LSTM. Normal: P(n_otto) + Pe_media(1) + Q_routed(1) + hora/mes(2).
        # PeOnly: sem chuva bruta — Pe por ottobacia(n_otto) + Q_routed(1) + hora/mes(2).
        # PeDist (controle M3): P(n_otto) + Pe por ottobacia(n_otto) + Q_routed(1) + hora/mes(2)
        # — separa espacializacao do Pe vs remocao da chuva bruta (confundimento do PeOnly).
        if pe_only:
            input_size = n_otto + 1 + 2
        elif pe_dist:
            input_size = n_otto + n_otto + 1 + 2
        else:
            input_size = n_otto + 1 + 1 + 2
        if baseflow:
            input_size += 2   # + Q_bf agregado (1) + S_bar medio (1) como canais de baixa frequencia
        if use_ar:
            input_size += 1   # + Q_obs defasada (canal AR)

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        out_dim = 1 if continuous else horizon
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_dim),
        )
        # RAMO FISICO (forecasting): o decoder recebe last_hidden + Q_fut roteado ao horizonte
        # (agua em transito da chuva passada + chuva futura prevista quando GC on)
        if not continuous:
            self.decoder[0] = nn.Linear(hidden_size + horizon, hidden_size)
        # PET climatologia (registrar como buffer pra ir pra GPU automaticamente)
        self.register_buffer('_pet_table', _PET_MM_HORA_POR_MES)

    def _pet_from_month(self, month_norm: torch.Tensor) -> torch.Tensor:
        """month_norm em [0,1] = (mes-1)/11. Retorna PET (B, T) em mm/h."""
        # mes 1..12
        m = (month_norm * 11).round().long().clamp(0, 11)
        return self._pet_table[m]

    def _montar_x(self, precip, Pe, Q_routed, hour, month, Q_bf=None, S_bar=None, q_past=None):
        q_routed_feat = torch.log1p(Q_routed).unsqueeze(-1)
        hour_feat = hour.unsqueeze(-1)
        month_feat = month.unsqueeze(-1)
        if self.pe_only:
            chans = [torch.log1p(Pe), q_routed_feat, hour_feat, month_feat]
        elif self.pe_dist:
            chans = [torch.log1p(precip), torch.log1p(Pe), q_routed_feat, hour_feat, month_feat]
        else:
            pe_mean_feat = torch.log1p(Pe.mean(dim=-1, keepdim=True))
            chans = [torch.log1p(precip), pe_mean_feat, q_routed_feat, hour_feat, month_feat]
        if self.baseflow:
            # braco TOPMODEL+baseflow: canais de baixa frequencia que so o TOPMODEL produz
            chans.append(torch.log1p(Q_bf).unsqueeze(-1))                  # baseflow agregado (B,T,1)
            chans.append((S_bar.mean(dim=-1, keepdim=True) / 100.0))       # deficit medio normalizado
        if self.use_ar:
            chans.append(torch.log1p(q_past).unsqueeze(-1))                # canal AR: vazao observada defasada
        return torch.cat(chans, dim=-1)

    def forward(
        self,
        precip: torch.Tensor,
        hour: torch.Tensor,
        month: torch.Tensor,
        pet: torch.Tensor | None = None,
        state=None,
        q_past: torch.Tensor | None = None,
        precip_fut: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if pet is None:
            pet = self._pet_from_month(month)  # (B, T)

        if self.continuous:
            st = state or {}
            T = precip.shape[1]
            out = self.topmodel(precip, pet, S_bar_init=st.get("sbar"),
                                tbptt_steps=T, use_checkpoint=False)
            Pe = out["Pe"]
            sbar_series = out["S_bar"]                              # (B,T,N)
            Q_routed, duh_tail = self.duh(Pe, pe_tail=st.get("duh_tail"), return_tail=True)
            Q_bf = self.topmodel.aggregate_to_basin(out["Q_bf"]) if self.baseflow else None
            x = self._montar_x(precip, Pe, Q_routed, hour, month, Q_bf=Q_bf, S_bar=sbar_series, q_past=q_past)
            lstm_out, (hn, cn) = self.lstm(x, st.get("lstm"))
            pred_log = self.decoder(lstm_out).squeeze(-1)
            pred = torch.expm1(F.relu(pred_log))
            return {'pred': pred_log, 'pred_exp': pred, 'Pe': Pe, 'Q_routed': Q_routed,
                    'A_sat': out["A_sat"], 'S_bar': sbar_series,
                    'state': {"sbar": sbar_series[:, -1, :], "duh_tail": duh_tail, "lstm": (hn, cn)}}

        # forecasting (paridade B1: Q_routed = duh(Pe), Q_bf so drena o deficit interno)
        out = self.topmodel(precip, pet)
        Pe = out["Pe"]
        Q_routed = self.duh(Pe)
        x = self._montar_x(precip, Pe, Q_routed, hour, month,
                           Q_bf=self.topmodel.aggregate_to_basin(out["Q_bf"]) if self.baseflow else None,
                           S_bar=out["S_bar"], q_past=q_past)
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        # ROTEAMENTO ESTENDIDO (ramo fisico): agua em transito do Pe passado roteada ao horizonte.
        # GC on: chuva futura prevista gerada pelo Topmodel CONTINUANDO o estado de umidade (S_bar)
        # de onde o lookback parou -> a geracao futura herda a umidade da bacia (fase B).
        if self.use_gc and precip_fut is not None:
            sbar_end = out["S_bar"][:, -1, :]                                       # deficit no fim do lookback (B,N)
            pet_fut = self._pet_from_month(month[:, -1:].expand(-1, self.horizon))  # PET climatologica do horizonte
            Pe_fut = self.topmodel(precip_fut, pet_fut, S_bar_init=sbar_end)["Pe"]
        else:
            Pe_fut = torch.zeros(precip.shape[0], self.horizon, precip.shape[2],
                                 device=precip.device, dtype=Pe.dtype)
        Q_fut = self.duh(torch.cat([Pe, Pe_fut], dim=1))[:, -self.horizon:]   # (B,H) agua em transito + chuva futura
        pred_log = self.decoder(torch.cat([last_hidden, torch.log1p(Q_fut)], dim=-1))
        pred = torch.expm1(F.relu(pred_log))

        return {
            'pred': pred_log,
            'pred_exp': pred,
            'Pe': Pe,
            'Q_routed': Q_routed,
            'A_sat': out["A_sat"],
            'S_bar': out["S_bar"],
            'topmodel_m': self.topmodel.m if self.learnable else torch.tensor(30.0),
            'topmodel_T_0': self.topmodel.T_0 if self.learnable else torch.tensor(0.02),
            'tc_scale': self.duh.tc_scale,
            'sigma': self.duh.sigma,
        }

    def get_learned_params(self) -> dict[str, float]:
        if self.learnable:
            return {
                'topmodel_m': float(self.topmodel.m),
                'topmodel_T_0': float(self.topmodel.T_0),
                'topmodel_sigmoid_temp': float(self.topmodel.sigmoid_temp),
                # S_bar_init FIXO em 70 mm (B2 auditoria 20/06; nao aprendivel). Reportado so
                # para auditoria confirmar que esta no valor de equilibrio, nao como param treinado.
                'topmodel_S_bar_init_fixed': float(self.topmodel.S_bar_init_learned),
                'tc_scale': self.duh.tc_scale.item(),
                'sigma': self.duh.sigma.item(),
            }
        return {'topmodel_m': 30.0, 'topmodel_T_0': 0.02, 'tc_scale': 1.0, 'sigma': 3.0}


# ==============================================================================
# FACTORY FUNCTION
# ==============================================================================

def _load_twi_attrs(npz_path='data/processed/twi_attrs.npz'):
    """Carrega twi_attrs.npz pra Topmodel. Retorna 3 tensores ordenados por ottobacia_idx."""
    from pathlib import Path
    p = Path(npz_path)
    if not p.exists():
        # AUTO-CONTENCAO (fix auditoria 17/06): ancora no diretorio do PACOTE
        # (ablacao_v2/data/processed), nao no cwd nem na raiz hardcoded. Garante que a
        # copia congelavel use SEMPRE o seu proprio TWI, independente de onde foi lancada.
        pkg_root = Path(__file__).resolve().parents[3]  # .../ablacao_v2
        cand = pkg_root / npz_path
        p = cand if cand.exists() else Path('D:/TTD_SCS_LSTM') / npz_path  # fallback legado (raiz)
    if not p.exists():
        raise FileNotFoundError(f"twi_attrs.npz nao encontrado em {npz_path}")
    data = np.load(p)
    order = np.argsort(data['ottobacia_idx'])
    return (
        torch.from_numpy(data['twi_dist'][order].astype(np.float32)),
        torch.from_numpy(data['twi_centers'].astype(np.float32)),
        torch.from_numpy(data['twi_mean'][order].astype(np.float32)),
    )


class PhysicalForecast(nn.Module):
    """Modelo PURAMENTE FISICO (SEM LSTM): SCS-CN (opcional) -> TTD -> Q + baseflow.

    Forecasting 1-24h pela agua EM TRANSITO: rotea a chuva do lookback (assume chuva
    futura = 0) e le as horas futuras do hidrograma roteado. Treina so os parametros
    fisicos (tc_scale, sigma, lambda) + baseflow escalar. Baseline conceitual que fecha
    o tripe: fisico-puro x neural-puro x hibrido.
    """

    def __init__(self, n_otto, tc_values, area_km2, cn_values=None, use_scs=False,
                 use_topmodel=False, twi_dist=None, twi_centers=None, twi_mean=None,
                 tc_type='base', learnable=True, horizon=24, continuous=False):
        super().__init__()
        if use_scs and use_topmodel:
            raise ValueError("use_scs e use_topmodel sao mutuamente exclusivos")
        self.use_scs = use_scs
        self.use_topmodel = use_topmodel
        self.horizon = horizon
        self.continuous = continuous
        tcn = "Base" if tc_type == 'base' else "Manning"
        sfx = "_SCS" if use_scs else ("_Topmodel" if use_topmodel else "")
        lf = "" if learnable else "_Fixed"
        self.name = f"Phys_DUH_{tcn}{sfx}{lf}"
        if use_scs:
            self.scs = SCSLayer(cn_values, learnable=learnable)
        if use_topmodel:
            from .topmodel_diff import TopmodelDiff
            self.topmodel = TopmodelDiff(
                twi_dist=twi_dist, twi_centers=twi_centers, twi_mean=twi_mean,
                area_km2=area_km2, learnable=learnable,
            )
            self.register_buffer('_pet_table', _PET_MM_HORA_POR_MES)
        self.duh = DUHLayer(tc_values, area_km2, learnable=learnable)
        self.log_baseflow = nn.Parameter(torch.tensor(0.0))  # baseflow >=0 via softplus (m3/s)

    def _pet_from_month(self, month_norm):
        m = (month_norm * 11).round().long().clamp(0, 11)
        return self._pet_table[m]

    def _gen_pe(self, precip, month, pet):
        """Geracao de escoamento -> Pe (B,L,N). TOPMODEL, SCS-CN, ou passthrough."""
        if self.use_topmodel:
            if pet is None:
                pet = self._pet_from_month(month)
            return self.topmodel(precip, pet)["Pe"]
        if self.use_scs:
            return self.scs(precip)
        return precip

    def forward(self, precip, hour, month, pet=None, state=None):
        if self.continuous:
            # simulacao continua: rotea a chuva concorrente (sem horizonte futuro), stateful.
            st = state or {}
            if self.use_topmodel:
                if pet is None:
                    pet = self._pet_from_month(month)
                out = self.topmodel(precip, pet, S_bar_init=st.get("sbar"),
                                    tbptt_steps=precip.shape[1], use_checkpoint=False)
                Pe = out["Pe"]; gen_state = {"sbar": out["S_bar"][:, -1, :]}
            elif self.use_scs:
                Pe, scs_state = self.scs(precip, scs_state=st.get("scs"), return_state=True)
                gen_state = {"scs": scs_state}
            else:
                Pe, gen_state = precip, {}
            Q_routed, duh_tail = self.duh(Pe, pe_tail=st.get("duh_tail"), return_tail=True)
            Q = F.relu(Q_routed) + F.softplus(self.log_baseflow)   # (B,T)
            return {'pred': torch.log1p(Q), 'pred_exp': Q,
                    'state': {**gen_state, "duh_tail": duh_tail}}

        # forecasting: agua em transito (rotea o lookback, chuva futura=0)
        Pe = self._gen_pe(precip, month, pet)                      # (B, L, N)
        B, L, N = Pe.shape
        zeros = torch.zeros(B, self.horizon, N, device=Pe.device, dtype=Pe.dtype)
        Q_ext = self.duh(torch.cat([Pe, zeros], dim=1))            # rotea c/ chuva futura=0
        Q_future = Q_ext[:, L:L + self.horizon]                    # agua em transito (B, horizon)
        Q = F.relu(Q_future) + F.softplus(self.log_baseflow)
        return {'pred': torch.log1p(Q), 'pred_exp': Q}             # pred=log (loss); pred_exp=vazao (metricas)

    def get_learned_params(self):
        p = {'tc_scale': float(self.duh.tc_scale), 'sigma': float(self.duh.sigma),
             'baseflow_m3s': float(F.softplus(self.log_baseflow))}
        if self.use_scs:
            p['lambda_scs'] = float(self.scs.lambda_scs)
        if self.use_topmodel:
            p['topmodel_m'] = float(self.topmodel.m)
            p['topmodel_T_0'] = float(self.topmodel.T_0)
            p['topmodel_sigmoid_temp'] = float(self.topmodel.sigmoid_temp)
        return p


def create_model(
    model_type: str,
    cn_values: torch.Tensor,
    tc_base_values: torch.Tensor,
    tc_manning_values: torch.Tensor,
    area_km2: torch.Tensor,
    twi_dist: torch.Tensor | None = None,
    twi_centers: torch.Tensor | None = None,
    twi_mean: torch.Tensor | None = None,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.1,
    horizon: int = 24,
    device: str = 'cuda',
    continuous: bool = False,
) -> nn.Module:
    """
    Factory para criar modelos de ablacao v2 (14 opcoes — 10 originais + 4 Topmodel).
    """

    n_otto = len(cn_values)
    model_type = model_type.lower()

    # Aliases legados: nomenclatura antiga "ttd" -> "duh" (rename de 01/06/2026).
    # Mantem simulate.py/simulate_v3.py/runners .ps1 antigos funcionando.
    if '_ttd_' in model_type:
        model_type = model_type.replace('_ttd_', '_duh_')

    # Topmodel: usa o TWI passado (datasets A+) ou carrega o de MD (twi_attrs.npz)
    if 'topmodel' in model_type:
        if twi_dist is None:
            twi_dist, twi_centers, twi_mean = _load_twi_attrs()
        if len(twi_dist) != n_otto:
            raise ValueError(f"twi n_otto {len(twi_dist)} != cn_values n_otto {n_otto}")

    # Modelo 1: LSTM Lumped (média espacial simples — baseline original)
    if model_type == 'lstm_lumped':
        model = LSTMLumped(
            n_otto=n_otto,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon
        )

    # Cadeia "de onde vem o skill": lumped PONDERADO por área, canais configuráveis
    elif model_type == 'lstm_lumped_wmean':      # modo B: chuva ponderada + calendário
        model = LSTMLumpedWeighted(
            n_otto=n_otto, area_km2=area_km2, use_rain=True, use_cal=True,
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, horizon=horizon,
            continuous=continuous,
        )
    elif model_type == 'lstm_lumped_rainonly':   # modo A: só chuva ponderada
        model = LSTMLumpedWeighted(
            n_otto=n_otto, area_km2=area_km2, use_rain=True, use_cal=False,
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, horizon=horizon,
            continuous=continuous,
        )
    elif model_type == 'lstm_lumped_calonly':    # modo C: só calendário
        model = LSTMLumpedWeighted(
            n_otto=n_otto, area_km2=area_km2, use_rain=False, use_cal=True,
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, horizon=horizon,
            continuous=continuous,
        )

    # Modelo 2: LSTM Distribuido
    elif model_type == 'lstm':
        model = LSTMDistributed(
            n_otto=n_otto,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
            continuous=continuous,
        )

    # LSTM + PET como feature (decisao 21/06): forcante evaporativa direta na rede, alem do calendario.
    elif model_type == 'lstm_pet':
        model = LSTMDistributed(
            n_otto=n_otto,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
            continuous=continuous,
            use_pet=True,
        )

    # Modelo M0c (ablacao_v2): LSTM + atributos por ottobacia, SEM fisica
    elif model_type == 'lstm_attrs':
        tw_mean = twi_mean
        if tw_mean is None:
            _, _, tw_mean = _load_twi_attrs()   # MD: carrega do twi_attrs.npz (igual topmodel)
        if len(tw_mean) != n_otto:
            raise ValueError(f"twi_mean n_otto {len(tw_mean)} != cn_values n_otto {n_otto}")
        model = LSTMAttrs(
            n_otto=n_otto,
            cn_values=cn_values,
            tc_values=tc_base_values,
            twi_mean=tw_mean,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon
        )

    # Modelo 3: TTD Base Fixo
    elif model_type == 'lstm_duh_base_fixed':
        model = LSTMWithTTD(
            n_otto=n_otto,
            tc_values=tc_base_values,
            area_km2=area_km2,
            tc_type='base',
            learnable=False,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
            continuous=continuous,
        )

    # Modelo 4: TTD Base Ajustavel
    elif model_type == 'lstm_duh_base':
        model = LSTMWithTTD(
            n_otto=n_otto,
            tc_values=tc_base_values,
            area_km2=area_km2,
            tc_type='base',
            learnable=True,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
            continuous=continuous,
        )

    # Modelo 5: TTD Manning Fixo
    elif model_type == 'lstm_duh_manning_fixed':
        model = LSTMWithTTD(
            n_otto=n_otto,
            tc_values=tc_manning_values,
            area_km2=area_km2,
            tc_type='manning',
            learnable=False,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon
        )

    # Modelo 6: TTD Manning Ajustavel
    elif model_type == 'lstm_duh_manning':
        model = LSTMWithTTD(
            n_otto=n_otto,
            tc_values=tc_manning_values,
            area_km2=area_km2,
            tc_type='manning',
            learnable=True,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon
        )

    # Modelo 7: TTD Base + SCS Fixo
    elif model_type == 'lstm_duh_base_scs_fixed':
        model = LSTMWithTTDSCS(
            n_otto=n_otto,
            cn_values=cn_values,
            tc_values=tc_base_values,
            area_km2=area_km2,
            tc_type='base',
            learnable=False,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon
        )

    # Modelo 8: TTD Base + SCS Ajustavel
    elif model_type == 'lstm_duh_base_scs':
        model = LSTMWithTTDSCS(
            n_otto=n_otto,
            cn_values=cn_values,
            tc_values=tc_base_values,
            area_km2=area_km2,
            tc_type='base',
            learnable=True,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon,
            continuous=continuous,
        )

    # Modelo 9: TTD Manning + SCS Fixo
    elif model_type == 'lstm_duh_manning_scs_fixed':
        model = LSTMWithTTDSCS(
            n_otto=n_otto,
            cn_values=cn_values,
            tc_values=tc_manning_values,
            area_km2=area_km2,
            tc_type='manning',
            learnable=False,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon
        )

    # Modelo 10: TTD Manning + SCS Ajustavel (Modelo Completo)
    elif model_type == 'lstm_duh_manning_scs':
        model = LSTMWithTTDSCS(
            n_otto=n_otto,
            cn_values=cn_values,
            tc_values=tc_manning_values,
            area_km2=area_km2,
            tc_type='manning',
            learnable=True,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            horizon=horizon
        )

    # Modelo 11: TTD Base + Topmodel Fixo
    elif model_type == 'lstm_duh_base_topmodel_fixed':
        model = LSTMWithTTDTopmodel(
            n_otto=n_otto, twi_dist=twi_dist, twi_centers=twi_centers, twi_mean=twi_mean,
            tc_values=tc_base_values, area_km2=area_km2,
            tc_type='base', learnable=False,
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, horizon=horizon
        )

    # Modelo 12: TTD Base + Topmodel Ajustavel
    elif model_type == 'lstm_duh_base_topmodel':
        model = LSTMWithTTDTopmodel(
            n_otto=n_otto, twi_dist=twi_dist, twi_centers=twi_centers, twi_mean=twi_mean,
            tc_values=tc_base_values, area_km2=area_km2,
            tc_type='base', learnable=True,
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, horizon=horizon,
            continuous=continuous,
        )

    elif model_type == 'lstm_duh_base_topmodel_baseflow':   # braco NOVO (decisao 20/06): S_bar+Q_bf canais
        model = LSTMWithTTDTopmodel(
            n_otto=n_otto, twi_dist=twi_dist, twi_centers=twi_centers, twi_mean=twi_mean,
            tc_values=tc_base_values, area_km2=area_km2,
            tc_type='base', learnable=True,
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, horizon=horizon,
            continuous=continuous, baseflow=True,
        )

    # SEM DUH (decisao 21/06): geracao sem roteamento explicito (IUH=impulso). Desconfunde
    # geracao x roteamento — testa se o ganho da fisica vem da geracao (umidade/baseflow), nao do IUH.
    elif model_type == 'lstm_topmodel':   # TOPMODEL sem DUH
        model = LSTMWithTTDTopmodel(
            n_otto=n_otto, twi_dist=twi_dist, twi_centers=twi_centers, twi_mean=twi_mean,
            tc_values=tc_base_values, area_km2=area_km2,
            tc_type='base', learnable=True,
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, horizon=horizon,
            continuous=continuous, impulse=True,
        )

    elif model_type == 'lstm_scs':   # SCS sem DUH
        model = LSTMWithTTDSCS(
            n_otto=n_otto, cn_values=cn_values, tc_values=tc_base_values, area_km2=area_km2,
            tc_type='base', learnable=True,
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, horizon=horizon,
            continuous=continuous, impulse=True,
        )

    # Modelo 13: TTD Manning + Topmodel Fixo
    elif model_type == 'lstm_duh_manning_topmodel_fixed':
        model = LSTMWithTTDTopmodel(
            n_otto=n_otto, twi_dist=twi_dist, twi_centers=twi_centers, twi_mean=twi_mean,
            tc_values=tc_manning_values, area_km2=area_km2,
            tc_type='manning', learnable=False,
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, horizon=horizon
        )

    # Modelo 14: TTD Manning + Topmodel Ajustavel (Modelo Completo Topmodel)
    elif model_type == 'lstm_duh_manning_topmodel':
        model = LSTMWithTTDTopmodel(
            n_otto=n_otto, twi_dist=twi_dist, twi_centers=twi_centers, twi_mean=twi_mean,
            tc_values=tc_manning_values, area_km2=area_km2,
            tc_type='manning', learnable=True,
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, horizon=horizon
        )

    # Modelos 15-18: PeOnly (fase 3) — LSTM sem chuva bruta, forçado a usar a física.
    # Pareados com os modelos completos 8/10/12/14 para isolar o "atalho" da chuva crua.
    elif model_type == 'lstm_duh_base_scs_peonly':
        model = LSTMWithTTDSCS(
            n_otto=n_otto, cn_values=cn_values, tc_values=tc_base_values, area_km2=area_km2,
            tc_type='base', learnable=True, pe_only=True,
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, horizon=horizon,
            continuous=continuous,
        )

    elif model_type == 'lstm_duh_manning_scs_peonly':
        model = LSTMWithTTDSCS(
            n_otto=n_otto, cn_values=cn_values, tc_values=tc_manning_values, area_km2=area_km2,
            tc_type='manning', learnable=True, pe_only=True,
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, horizon=horizon
        )

    elif model_type == 'lstm_duh_base_topmodel_peonly':
        model = LSTMWithTTDTopmodel(
            n_otto=n_otto, twi_dist=twi_dist, twi_centers=twi_centers, twi_mean=twi_mean,
            tc_values=tc_base_values, area_km2=area_km2,
            tc_type='base', learnable=True, pe_only=True,
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, horizon=horizon,
            continuous=continuous,
        )

    elif model_type == 'lstm_duh_manning_topmodel_peonly':
        model = LSTMWithTTDTopmodel(
            n_otto=n_otto, twi_dist=twi_dist, twi_centers=twi_centers, twi_mean=twi_mean,
            tc_values=tc_manning_values, area_km2=area_km2,
            tc_type='manning', learnable=True, pe_only=True,
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, horizon=horizon
        )

    # Experimento de controle M3 (PeDist): P bruta (245) + Pe distribuido (245).
    # Separa as duas mudancas do PeOnly (espacializar Pe vs tirar a chuva bruta).
    # Fora da MODEL_TYPES de proposito — roda a parte, nao mexe nos 550 oficiais.
    elif model_type == 'lstm_duh_base_topmodel_pedist':
        model = LSTMWithTTDTopmodel(
            n_otto=n_otto, twi_dist=twi_dist, twi_centers=twi_centers, twi_mean=twi_mean,
            tc_values=tc_base_values, area_km2=area_km2,
            tc_type='base', learnable=True, pe_dist=True,
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, horizon=horizon
        )

    elif model_type == 'lstm_duh_manning_topmodel_pedist':
        model = LSTMWithTTDTopmodel(
            n_otto=n_otto, twi_dist=twi_dist, twi_centers=twi_centers, twi_mean=twi_mean,
            tc_values=tc_manning_values, area_km2=area_km2,
            tc_type='manning', learnable=True, pe_dist=True,
            hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, horizon=horizon
        )

    # Modelos 19-22: PURAMENTE FISICOS (sem LSTM) — baseline conceitual
    elif model_type == 'phys_duh_base':
        model = PhysicalForecast(n_otto, tc_base_values, area_km2, use_scs=False, tc_type='base', horizon=horizon)
    elif model_type == 'phys_duh_base_scs':
        model = PhysicalForecast(n_otto, tc_base_values, area_km2, cn_values=cn_values, use_scs=True, tc_type='base', horizon=horizon, continuous=continuous)
    elif model_type == 'phys_duh_base_topmodel':
        model = PhysicalForecast(n_otto, tc_base_values, area_km2, use_topmodel=True,
                                 twi_dist=twi_dist, twi_centers=twi_centers, twi_mean=twi_mean,
                                 tc_type='base', horizon=horizon, continuous=continuous)
    elif model_type == 'phys_duh_manning':
        model = PhysicalForecast(n_otto, tc_manning_values, area_km2, use_scs=False, tc_type='manning', horizon=horizon)
    elif model_type == 'phys_duh_manning_scs':
        model = PhysicalForecast(n_otto, tc_manning_values, area_km2, cn_values=cn_values, use_scs=True, tc_type='manning', horizon=horizon)

    else:
        raise ValueError(
            f"Tipo de modelo desconhecido: {model_type}\n"
            f"Tipos validos: {MODEL_TYPES}"
        )

    return model.to(device)


# ==============================================================================
# CAMADA QTT INTEGRADA (SCS + TTD)
# ==============================================================================

class QTTLayer(nn.Module):
    """
    Camada QTT (Quantile Travel Time) - Integracao SCS + TTD.

    Combina separacao de escoamento (SCS-CN) com roteamento temporal (TTD)
    em uma unica camada fisica diferenciavel.

    P -> SCS -> Pe -> TTD -> Q_routed

    A area do hidrograma e reduzida pelo SCS (menos chuva vira vazao),
    e o tempo de chegada e controlado pelo TTD.
    """

    def __init__(
        self,
        cn_values: torch.Tensor,
        tc_values: torch.Tensor,
        area_km2: torch.Tensor,
        n_bins: int = 120,
        dt_hours: float = 1.0,
        learnable: bool = True
    ):
        super().__init__()

        self.learnable = learnable
        self.n_otto = len(cn_values)

        # Componentes integrados
        self.scs = SCSLayer(cn_values, learnable=learnable)
        self.duh = DUHLayer(tc_values, area_km2, n_bins, dt_hours, learnable=learnable)

    def forward(self, P: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            P: Precipitacao (batch, seq_len, n_otto)

        Returns:
            Dict com Q_routed, Pe, e parametros
        """
        # SCS: separa escoamento (reduz volume)
        Pe = self.scs(P)

        # TTD: roteia no tempo
        Q_routed = self.duh(Pe)

        return {
            'Q_routed': Q_routed,
            'Pe': Pe,
            'lambda_scs': self.scs.lambda_scs,
            'tc_scale': self.duh.tc_scale,
            'sigma': self.duh.sigma
        }


class LSTMWithQTT(nn.Module):
    """
    Modelo QTT: LSTM + QTT Integrado

    Arquitetura: P -> QTT(SCS+TTD) -> Q_routed -> LSTM -> Q

    O LSTM recebe tanto P quanto Pe e Q_routed como features.
    """

    def __init__(
        self,
        n_otto: int,
        cn_values: torch.Tensor,
        tc_values: torch.Tensor,
        area_km2: torch.Tensor,
        tc_type: str = 'base',
        learnable: bool = True,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 24
    ):
        super().__init__()

        self.n_otto = n_otto
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.tc_type = tc_type
        self.learnable = learnable

        # Nome do modelo
        tc_name = "Base" if tc_type == 'base' else "Manning"
        self.name = f"LSTM_QTT_{tc_name}"

        # QTT Layer integrado
        self.qtt = QTTLayer(cn_values, tc_values, area_km2, learnable=learnable)

        # Input: P por ottobacia + Pe media + Q_routed + hora + mes
        input_size = n_otto + 1 + 1 + 2

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon)
        )

    def forward(
        self,
        precip: torch.Tensor,
        hour: torch.Tensor,
        month: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, n_otto = precip.shape

        # QTT: separacao + roteamento integrados
        qtt_out = self.qtt(precip)
        Pe = qtt_out['Pe']
        Q_routed = qtt_out['Q_routed']

        # Features (inclui P original)
        precip_feat = torch.log1p(precip)
        pe_mean_feat = torch.log1p(Pe.mean(dim=-1, keepdim=True))
        q_routed_feat = torch.log1p(Q_routed).unsqueeze(-1)
        hour_feat = hour.unsqueeze(-1)
        month_feat = month.unsqueeze(-1)

        x = torch.cat([precip_feat, pe_mean_feat, q_routed_feat, hour_feat, month_feat], dim=-1)

        # LSTM
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]

        # Decoder
        pred_log = self.decoder(last_hidden)
        pred = torch.expm1(F.relu(pred_log))

        return {
            'pred': pred_log,
            'pred_exp': pred,
            'Pe': Pe,
            'Q_routed': Q_routed,
            'lambda_scs': qtt_out['lambda_scs'],
            'tc_scale': qtt_out['tc_scale'],
            'sigma': qtt_out['sigma']
        }

    def get_learned_params(self) -> Dict[str, float]:
        if self.learnable:
            return {
                'lambda_scs': self.qtt.scs.lambda_scs.item(),
                'tc_scale': self.qtt.ttd.tc_scale.item(),
                'sigma': self.qtt.ttd.sigma.item()
            }
        return {'lambda_scs': 0.2, 'tc_scale': 1.0, 'sigma': 3.0}


class LSTMWithQTTPeOnly(nn.Module):
    """
    Modelo QTT Pe-Only: LSTM + QTT (forcando uso da fisica)

    Arquitetura: P -> QTT(SCS+TTD) -> Q_routed -> LSTM -> Q

    DIFERENCA: O LSTM NAO recebe P original, apenas Pe e Q_routed.
    Isso forca o modelo a usar a fisica (SCS) para separar escoamento.
    """

    def __init__(
        self,
        n_otto: int,
        cn_values: torch.Tensor,
        tc_values: torch.Tensor,
        area_km2: torch.Tensor,
        tc_type: str = 'base',
        learnable: bool = True,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        horizon: int = 24
    ):
        super().__init__()

        self.n_otto = n_otto
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.tc_type = tc_type
        self.learnable = learnable

        # Nome do modelo
        tc_name = "Base" if tc_type == 'base' else "Manning"
        self.name = f"LSTM_QTT_{tc_name}_PeOnly"

        # QTT Layer integrado
        self.qtt = QTTLayer(cn_values, tc_values, area_km2, learnable=learnable)

        # Input: Pe por ottobacia + Q_routed + hora + mes (SEM P original!)
        input_size = n_otto + 1 + 2

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, horizon)
        )

    def forward(
        self,
        precip: torch.Tensor,
        hour: torch.Tensor,
        month: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, n_otto = precip.shape

        # QTT: separacao + roteamento integrados
        qtt_out = self.qtt(precip)
        Pe = qtt_out['Pe']
        Q_routed = qtt_out['Q_routed']

        # Features (SEM P original - apenas Pe!)
        pe_feat = torch.log1p(Pe)  # Pe por ottobacia
        q_routed_feat = torch.log1p(Q_routed).unsqueeze(-1)
        hour_feat = hour.unsqueeze(-1)
        month_feat = month.unsqueeze(-1)

        x = torch.cat([pe_feat, q_routed_feat, hour_feat, month_feat], dim=-1)

        # LSTM
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]

        # Decoder
        pred_log = self.decoder(last_hidden)
        pred = torch.expm1(F.relu(pred_log))

        return {
            'pred': pred_log,
            'pred_exp': pred,
            'Pe': Pe,
            'Q_routed': Q_routed,
            'lambda_scs': qtt_out['lambda_scs'],
            'tc_scale': qtt_out['tc_scale'],
            'sigma': qtt_out['sigma']
        }

    def get_learned_params(self) -> Dict[str, float]:
        if self.learnable:
            return {
                'lambda_scs': self.qtt.scs.lambda_scs.item(),
                'tc_scale': self.qtt.ttd.tc_scale.item(),
                'sigma': self.qtt.ttd.sigma.item()
            }
        return {'lambda_scs': 0.2, 'tc_scale': 1.0, 'sigma': 3.0}


# Lista de tipos de modelo — 14 ablacoes (10 originais + 4 Topmodel).
# QTT (11-12) removidos da MODEL_TYPES default — disponiveis no factory mas nao no run principal.
MODEL_TYPES = [
    'lstm_lumped',                       # 1. Baseline Lumped
    'lstm',                              # 2. Baseline Distribuido
    'lstm_duh_base_fixed',               # 3. TTD Base Fixo
    'lstm_duh_base',                     # 4. TTD Base Ajustavel
    'lstm_duh_manning_fixed',            # 5. TTD Manning Fixo
    'lstm_duh_manning',                  # 6. TTD Manning Ajustavel
    'lstm_duh_base_scs_fixed',           # 7. TTD Base + SCS Fixo
    'lstm_duh_base_scs',                 # 8. TTD Base + SCS Ajustavel
    'lstm_duh_manning_scs_fixed',        # 9. TTD Manning + SCS Fixo
    'lstm_duh_manning_scs',              # 10. Modelo Completo SCS
    'lstm_duh_base_topmodel_fixed',      # 11. TTD Base + Topmodel Fixo
    'lstm_duh_base_topmodel',            # 12. TTD Base + Topmodel Ajustavel
    'lstm_duh_manning_topmodel_fixed',   # 13. TTD Manning + Topmodel Fixo
    'lstm_duh_manning_topmodel',         # 14. Modelo Completo Topmodel
    'lstm_duh_base_scs_peonly',          # 15. SCS Base PeOnly (LSTM sem chuva bruta)
    'lstm_duh_manning_scs_peonly',       # 16. SCS Manning PeOnly
    'lstm_duh_base_topmodel_peonly',     # 17. Topmodel Base PeOnly
    'lstm_duh_manning_topmodel_peonly',  # 18. Topmodel Manning PeOnly
]

MODEL_DESCRIPTIONS = {
    'lstm_lumped': '1. LSTM Lumped - Baseline (P_media -> LSTM -> Q)',
    'lstm': '2. LSTM Distribuido - Baseline (P_245otto -> LSTM -> Q)',
    'lstm_attrs': 'M0c. LSTM + atributos/otto (CN,Tc,TWI) SEM fisica (ablacao_v2)',
    'lstm_duh_base_fixed': '3. TTD Base Fixo (tc_scale=1, sigma=3)',
    'lstm_duh_base': '4. TTD Base Ajustavel (tc_scale, sigma aprendiveis)',
    'lstm_duh_manning_fixed': '5. TTD Manning Fixo (tc_scale=1, sigma=3)',
    'lstm_duh_manning': '6. TTD Manning Ajustavel (tc_scale, sigma aprendiveis)',
    'lstm_duh_base_scs_fixed': '7. TTD Base + SCS Fixo (lambda=0.2)',
    'lstm_duh_base_scs': '8. TTD Base + SCS Ajustavel (lambda, tc_scale, sigma)',
    'lstm_duh_manning_scs_fixed': '9. TTD Manning + SCS Fixo',
    'lstm_duh_manning_scs': '10. Modelo Completo SCS (Manning + SCS + Ajustavel)',
    'lstm_duh_base_topmodel_fixed': '11. TTD Base + Topmodel Fixo (m=30, T_0=0.02)',
    'lstm_duh_base_topmodel': '12. TTD Base + Topmodel Ajustavel (m, T_0, sigmoid_temp)',
    'lstm_duh_manning_topmodel_fixed': '13. TTD Manning + Topmodel Fixo',
    'lstm_duh_manning_topmodel': '14. Modelo Completo Topmodel (Manning + Topmodel + Ajustavel)',
    'lstm_duh_base_scs_peonly': '15. SCS Base PeOnly (LSTM sem chuva bruta — Pe + Q_routed)',
    'lstm_duh_manning_scs_peonly': '16. SCS Manning PeOnly (LSTM sem chuva bruta)',
    'lstm_duh_base_topmodel_peonly': '17. Topmodel Base PeOnly (LSTM sem chuva bruta)',
    'lstm_duh_manning_topmodel_peonly': '18. Topmodel Manning PeOnly (LSTM sem chuva bruta)',
    # Controle M3 (fora da MODEL_TYPES — nao entra no run oficial dos 22)
    'lstm_duh_base_topmodel_pedist': 'C1. Topmodel Base PeDist (P bruta + Pe distribuido, input 493)',
    'lstm_duh_manning_topmodel_pedist': 'C2. Topmodel Manning PeDist (P bruta + Pe distribuido, input 493)',
}
