#!/usr/bin/env python
"""
Script de Ablacao - TTD-SCS-LSTM (v2)
=====================================

Executa 10 experimentos de ablacao completo.

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

Usage:
    uv run python scripts/run_ablation.py
    uv run python scripts/run_ablation.py --epochs 300 --patience 30
    uv run python scripts/run_ablation.py --model lstm_ttd_manning_scs  # Modelo especifico
    uv run python scripts/run_ablation.py --test  # Teste rapido (1 epoca)

Autor: Claude + Vinicius
Data: 2026-01-22
"""

import argparse
import gc
import json
import sys
from pathlib import Path
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import h5py

warnings.filterwarnings('ignore')

# ==============================================================================
# SEED PARA REPRODUTIBILIDADE
# ==============================================================================
DEFAULT_SEED = 42

def set_seed(seed: int = DEFAULT_SEED):
    """Fixa seed para reprodutibilidade."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Paths
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

DATA_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "outputs" / "ablation"
DATASET_FILE = DATA_DIR / "dataset_v2.h5"  # Novo dataset v2

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==============================================================================
# DATASET
# ==============================================================================

class AblationDataset(Dataset):
    """Dataset para ablacao v2 - com filtro de NaN."""

    def __init__(
        self,
        h5_file: Path,
        split: str,
        lookback: int = 240,
        horizon: int = 24
    ):
        self.h5_file = h5_file
        self.split = split
        self.lookback = lookback
        self.horizon = horizon
        self.gc_mode = None   # canal GC: None | 'teto' (perfect forecast) | 'gfs' -> chuva futura no collate

        with h5py.File(h5_file, 'r') as f:
            self.precip = f[f'{split}/precipitation'][:]
            self.streamflow = f[f'{split}/streamflow'][:]
            self.timestamps = f[f'{split}/timestamps'][:]

            # Propriedades estaticas (nova estrutura v2)
            self.cn_2022 = torch.tensor(f['ottobacia/cn_2022'][:], dtype=torch.float32)
            self.tc_base_h = torch.tensor(f['ottobacia/tc_base_h'][:], dtype=torch.float32)
            self.tc_manning_h = torch.tensor(f['ottobacia/tc_manning_h'][:], dtype=torch.float32)
            self.area_km2 = torch.tensor(f['ottobacia/area_km2'][:], dtype=torch.float32)

            # PET real por ottobacia — opcional; presente nos datasets A+ E nos 5 da big
            # ablacao MD (injetada antes do run; fallback climatologia so se ausente do h5)
            self.pet = f[f'{split}/pet'][:] if 'pet' in f[split] else None

            # forcante de PREVISAO GraphCast (canal GC): cubo (T, horizon) lumped mm/h + mascara
            self.precip_fut_gfs = f[f'{split}/precip_fut_gfs'][:] if 'precip_fut_gfs' in f[split] else None
            self.valid_gfs = f[f'{split}/valid_gfs'][:] if 'valid_gfs' in f[split] else None

            # TWI por ottobacia (datasets A+) — opcional; MD le de twi_attrs.npz
            if 'twi_dist' in f['ottobacia']:
                self.twi_dist = torch.tensor(f['ottobacia/twi_dist'][:], dtype=torch.float32)
                self.twi_centers = torch.tensor(f['ottobacia/twi_centers'][:], dtype=torch.float32)
                self.twi_mean = torch.tensor(f['ottobacia/twi_mean'][:], dtype=torch.float32)
            else:
                self.twi_dist = self.twi_centers = self.twi_mean = None

        # Pre-computar features temporais (helper unico, UTC — numericamente
        # identico ao pd.to_datetime que estava embutido aqui)
        from ttd_scs_lstm.data.temporal import features_temporais
        self.hours, self.months = features_temporais(self.timestamps)

        # FILTRAR INDICES VALIDOS (sem NaN no target e na precipitacao)
        self._create_valid_indices()
        self._preload_gpu_tensors()

    def _preload_gpu_tensors(self):
        """Pre-carga na GPU (otimizacao do X1): precip/hour/month/pet viram tensores
        residentes na GPU e o batch e montado por gather vetorizado no collate(), no lugar
        de tensor-por-item + copia host->GPU a cada batch. Matematicamente IDENTICO (mesmos
        valores, mesmo dtype, mesma ordem de batch via DataLoader shuffle); so muda ONDE os
        dados ficam. O alvo permanece na CPU para nao precisar mexer em evaluate()."""
        self.precip_g = torch.as_tensor(self.precip, dtype=torch.float32, device=DEVICE)
        self.hours_g = torch.as_tensor(self.hours, dtype=torch.float32, device=DEVICE)
        self.months_g = torch.as_tensor(self.months, dtype=torch.float32, device=DEVICE)
        self.pet_g = (torch.as_tensor(self.pet, dtype=torch.float32, device=DEVICE)
                      if self.pet is not None else None)
        self.flow_t = torch.as_tensor(self.streamflow, dtype=torch.float32)  # CPU (igual antes)
        # canal AR: vazao observada do lookback com forward-fill dos NaN (assimilacao operacional
        # usa o ultimo dado disponivel); inicio sem valor anterior -> 0. Crua (>=0); o log1p e no modelo.
        flow_filled = np.asarray(self.streamflow, dtype=np.float64).copy()
        nanmask = np.isnan(flow_filled)
        if nanmask.any():
            order = np.where(~nanmask, np.arange(len(flow_filled)), 0)
            np.maximum.accumulate(order, out=order)
            flow_filled = np.nan_to_num(flow_filled[order], nan=0.0)
        self.flow_g = torch.as_tensor(np.clip(flow_filled, 0.0, None),
                                      dtype=torch.float32, device=DEVICE)
        # canal GC: pesos de area (teto lumped) + cubo GraphCast na GPU + arange horizonte na GPU
        self.area_w_g = (self.area_km2.float() / self.area_km2.float().sum()).to(DEVICE)
        self.gfs_g = (torch.as_tensor(np.nan_to_num(self.precip_fut_gfs, nan=0.0),
                                      dtype=torch.float32, device=DEVICE)
                      if self.precip_fut_gfs is not None else None)
        self._ar_lb_g = torch.arange(self.lookback, device=DEVICE)
        self._ar_h_cpu = torch.arange(self.horizon)
        self._ar_h_g = torch.arange(self.horizon, device=DEVICE)

    def _create_valid_indices(self):
        """Cria lista de indices validos (sem NaN)."""
        n_total = len(self.streamflow) - self.lookback - self.horizon + 1
        valid_indices = []

        for i in range(n_total):
            # Verificar NaN no target
            target_start = i + self.lookback
            target_end = target_start + self.horizon
            target = self.streamflow[target_start:target_end]

            if np.isnan(target).any():
                continue

            # Verificar NaN na precipitacao
            precip = self.precip[i:i + self.lookback]
            if np.isnan(precip).any():
                continue

            valid_indices.append(i)

        self.valid_indices = np.array(valid_indices)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # Retorna so o indice-inicio da janela; o batch e montado por gather vetorizado em collate().
        return int(self.valid_indices[idx])

    def collate(self, batch_starts):
        """Monta o batch por indexacao vetorizada nos tensores pre-carregados.
        precip/hour/month/pet saem na GPU; target na CPU (identico ao pipeline antigo)."""
        starts_cpu = torch.tensor(batch_starts, dtype=torch.long)
        win = starts_cpu.to(DEVICE)[:, None] + self._ar_lb_g           # (B, lookback) na GPU
        out = {
            'precip': self.precip_g[win],
            'hour': self.hours_g[win],
            'month': self.months_g[win],
        }
        if self.pet_g is not None:
            out['pet'] = self.pet_g[win]
        out['q_past'] = self.flow_g[win]   # canal AR: vazao observada no lookback (preenchida, GPU)
        twin = (starts_cpu + self.lookback)[:, None] + self._ar_h_cpu  # (B, horizon) na CPU
        # canal GC: chuva futura DISTRIBUIDA (B,H,N) -> modelos com fisica roteiam pela geracao;
        # baselines reduzem p/ lumped no gc_proj. teto = obs futura distribuida; gfs = GraphCast (lumped) broadcast.
        if self.gc_mode == 'teto':
            twin_dev = (starts_cpu.to(DEVICE) + self.lookback)[:, None] + self._ar_h_g   # (B,H) GPU
            out['precip_fut'] = self.precip_g[twin_dev]                                  # (B,H,N) obs futura (perfect forecast)
        elif self.gc_mode == 'gfs' and self.gfs_g is not None:
            emit = (starts_cpu.to(DEVICE) + self.lookback - 1)                            # (B,) emissao (off-by-one ok)
            out['precip_fut'] = self.gfs_g[emit].unsqueeze(-1).expand(-1, -1, self.area_w_g.shape[0])  # (B,H,N) GraphCast broadcast
        out['target'] = self.flow_t[twin]
        return out

    def get_static_features(self):
        """Retorna features estáticas para criar modelos."""
        feats = {
            'cn_values': self.cn_2022,
            'tc_base_values': self.tc_base_h,
            'tc_manning_values': self.tc_manning_h,
            'area_km2': self.area_km2,
            'n_otto': len(self.cn_2022)
        }
        if self.twi_dist is not None:
            feats['twi_dist'] = self.twi_dist
            feats['twi_centers'] = self.twi_centers
            feats['twi_mean'] = self.twi_mean
        return feats


def create_dataloaders(h5_file, lookback, horizon, batch_size):
    """Cria dataloaders para treino, validação e teste."""
    train_ds = AblationDataset(h5_file, 'train', lookback, horizon)
    val_ds = AblationDataset(h5_file, 'val', lookback, horizon)
    test_ds = AblationDataset(h5_file, 'test', lookback, horizon)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=train_ds.collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=val_ds.collate)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=test_ds.collate)

    return train_loader, val_loader, test_loader, train_ds.get_static_features()


# ==============================================================================
# MÉTRICAS
# ==============================================================================

def compute_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Calcula métricas de avaliação."""
    mask = ~np.isnan(target) & ~np.isnan(pred)
    pred = pred[mask]
    target = target[mask]

    if len(pred) == 0:
        return {'nse': np.nan, 'kge': np.nan, 'rmse': np.nan, 'pbias': np.nan}

    # NSE
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    nse = 1 - ss_res / (ss_tot + 1e-10)

    # KGE
    r = np.corrcoef(pred, target)[0, 1] if len(pred) > 1 else 0
    alpha = np.std(pred) / (np.std(target) + 1e-10)
    beta = np.mean(pred) / (np.mean(target) + 1e-10)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    # RMSE
    rmse = np.sqrt(np.mean((pred - target) ** 2))

    # PBIAS
    pbias = 100 * np.sum(pred - target) / (np.sum(target) + 1e-10)

    return {'nse': nse, 'kge': kge, 'rmse': rmse, 'pbias': pbias, 'r': r}


# ==============================================================================
# TREINAMENTO
# ==============================================================================

def train_epoch(model, dataloader, optimizer, device, grad_clip=1.0):
    """Treina uma época."""
    model.train()
    total_loss = 0
    n_batches = 0

    for batch in dataloader:
        precip = batch['precip'].to(device)
        hour = batch['hour'].to(device)
        month = batch['month'].to(device)
        target = batch['target'].to(device)
        pet = batch['pet'].to(device) if 'pet' in batch else None
        q_past = batch['q_past'].to(device) if 'q_past' in batch else None
        precip_fut = batch['precip_fut'].to(device) if 'precip_fut' in batch else None

        if torch.isnan(precip).any() or torch.isnan(target).any():
            continue

        optimizer.zero_grad()

        fkw = {}
        if pet is not None and (hasattr(model, 'topmodel') or getattr(model, 'use_pet', False)):
            fkw['pet'] = pet
        if getattr(model, 'use_ar', False) and q_past is not None:
            fkw['q_past'] = q_past
        if getattr(model, 'use_gc', False) and precip_fut is not None:
            fkw['precip_fut'] = precip_fut
        output = model(precip, hour, month, **fkw)
        pred = output['pred']

        if torch.isnan(pred).any():
            continue

        # Loss: MSE em escala log + componente para picos
        target_log = torch.log1p(target)
        loss_log = F.mse_loss(pred, target_log)

        pred_exp = torch.expm1(F.relu(pred))
        loss_peaks = F.mse_loss(pred_exp, target) * 0.01

        loss = loss_log + loss_peaks

        if torch.isnan(loss) or torch.isinf(loss):
            continue

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def evaluate(model, dataloader, device):
    """Avalia o modelo."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            precip = batch['precip'].to(device)
            hour = batch['hour'].to(device)
            month = batch['month'].to(device)
            target = batch['target']
            pet = batch['pet'].to(device) if 'pet' in batch else None
            q_past = batch['q_past'].to(device) if 'q_past' in batch else None
            precip_fut = batch['precip_fut'].to(device) if 'precip_fut' in batch else None

            fkw = {}
            if pet is not None and (hasattr(model, 'topmodel') or getattr(model, 'use_pet', False)):
                fkw['pet'] = pet
            if getattr(model, 'use_ar', False) and q_past is not None:
                fkw['q_past'] = q_past
            if getattr(model, 'use_gc', False) and precip_fut is not None:
                fkw['precip_fut'] = precip_fut
            output = model(precip, hour, month, **fkw)
            pred_exp = output['pred_exp']

            all_preds.append(pred_exp.cpu())
            all_targets.append(target)

    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()

    # Métricas por horizonte
    horizons = {'1h': 0, '3h': 2, '6h': 5, '12h': 11, '24h': 23}
    metrics_by_horizon = {}

    for name, idx in horizons.items():
        if idx < all_preds.shape[1]:
            metrics_by_horizon[name] = compute_metrics(
                all_preds[:, idx],
                all_targets[:, idx]
            )
        else:
            metrics_by_horizon[name] = {'nse': np.nan, 'kge': np.nan, 'rmse': np.nan}

    # Métrica principal = 6h
    main_idx = min(5, all_preds.shape[1] - 1)
    metrics_main = compute_metrics(
        all_preds[:, main_idx],
        all_targets[:, main_idx]
    )

    return metrics_main, metrics_by_horizon, all_preds, all_targets


def create_optimizer(model, config: dict):
    """Cria optimizer com learning rates diferenciados."""
    physics_params = []
    lstm_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if ('scs' in name or 'ttd' in name or 'duh' in name or 'topmodel' in name
                or 'baseflow' in name):  # baseflow: log_baseflow do PhysicalForecast (fix M5 auditoria)
            physics_params.append(param)
        else:
            lstm_params.append(param)

    param_groups = []

    if lstm_params:
        param_groups.append({
            'params': lstm_params,
            'lr': config['lr'],
            'name': 'lstm'
        })

    if physics_params:
        param_groups.append({
            'params': physics_params,
            'lr': config['lr'] * 10,  # 10x maior para física
            'name': 'physics'
        })

    return torch.optim.AdamW(
        param_groups,
        weight_decay=config['weight_decay']
    )


def train_model(
    model,
    train_loader,
    val_loader,
    config: dict,
    device,
    output_dir: Path,
    verbose: bool = True
):
    """Treina um modelo completo."""

    optimizer = create_optimizer(model, config)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7, min_lr=1e-6  # 'verbose' removido: kwarg sumiu no torch>=2.3 (crashava no python default, torch 2.10)
    )

    best_val_nse = -np.inf
    patience_counter = 0
    history = {'train_loss': [], 'val_nse': [], 'val_kge': []}

    iterator = range(config['epochs'])
    if verbose:
        iterator = tqdm(iterator, desc=f"Training {model.name}")

    for epoch in iterator:
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            grad_clip=config['grad_clip']
        )

        val_metrics, _, _, _ = evaluate(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['val_nse'].append(val_metrics['nse'])
        history['val_kge'].append(val_metrics['kge'])

        scheduler.step(val_metrics['nse'])

        if val_metrics['nse'] > best_val_nse:
            best_val_nse = val_metrics['nse']
            patience_counter = 0
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
        else:
            patience_counter += 1

        if verbose:
            postfix = {
                'loss': f"{train_loss:.4f}",
                'val_nse': f"{val_metrics['nse']:.3f}"
            }
            if hasattr(model, 'get_learned_params'):
                params = model.get_learned_params()
                if 'lambda_scs' in params:
                    postfix['lam'] = f"{params['lambda_scs']:.3f}"
                if 'tc_scale' in params:
                    postfix['tc'] = f"{params['tc_scale']:.2f}"
            iterator.set_postfix(postfix)

        if patience_counter >= config['patience']:
            if verbose:
                print(f"\n  Early stopping at epoch {epoch+1}")
            break

    # Guard A1 (auditoria): se nenhuma epoca produziu val NSE finito, o best_model.pt
    # nunca foi salvo e o teste rodaria com pesos degenerados (colapso/NaN) gravados como
    # validos. Falha explicita em vez de gravar lixo silenciosamente.
    if not np.isfinite(best_val_nse) or not (output_dir / 'best_model.pt').exists():
        raise RuntimeError(
            f"Treino degenerado: best_val_nse={best_val_nse} (nenhuma epoca salvou best_model.pt). "
            "Provavel colapso/NaN no treino."
        )
    model.load_state_dict(torch.load(output_dir / 'best_model.pt', weights_only=True))

    return history, best_val_nse


# ==============================================================================
# EXPERIMENTO ÚNICO
# ==============================================================================

def run_experiment(
    model_type: str,
    config: dict,
    train_loader,
    val_loader,
    test_loader,
    static_features: dict,
    output_dir: Path,
    verbose: bool = True
) -> dict:
    """Executa um experimento de ablação."""

    from ttd_scs_lstm.models.models import create_model

    # Criar modelo
    model = create_model(
        model_type=model_type,
        cn_values=static_features['cn_values'],
        tc_base_values=static_features['tc_base_values'],
        tc_manning_values=static_features['tc_manning_values'],
        area_km2=static_features['area_km2'],
        twi_dist=static_features.get('twi_dist'),
        twi_centers=static_features.get('twi_centers'),
        twi_mean=static_features.get('twi_mean'),
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        horizon=config['horizon'],
        device=DEVICE
    )

    model_name = model.name
    n_params = sum(p.numel() for p in model.parameters())

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Modelo: {model_name}")
        print(f"  Parametros: {n_params:,}")
        print(f"{'='*60}")

    # Diretório de saída
    exp_dir = output_dir / model_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Treinar
    history, best_val_nse = train_model(
        model, train_loader, val_loader, config, DEVICE, exp_dir, verbose
    )

    # Avaliar no teste
    test_metrics, test_by_horizon, test_preds, test_targets = evaluate(model, test_loader, DEVICE)

    # Parâmetros aprendidos
    learned_params = {}
    if hasattr(model, 'get_learned_params'):
        learned_params = model.get_learned_params()

    if verbose:
        print(f"\n  Test NSE por horizonte:")
        for h_name, h_metrics in test_by_horizon.items():
            print(f"    {h_name}: NSE={h_metrics['nse']:.4f}, KGE={h_metrics['kge']:.4f}")
        if learned_params:
            print(f"  Learned params: {learned_params}")

    # Salvar resultados
    results = {
        'model_name': model_name,
        'model_type': model_type,
        'n_params': n_params,
        'config': config,
        'test_metrics': test_metrics,
        'test_by_horizon': test_by_horizon,
        'learned_params': learned_params,
        'best_val_nse': best_val_nse,
        'epochs_trained': len(history['train_loss'])
    }

    with open(exp_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=float)

    # Salvar previsões
    np.savez(
        exp_dir / 'predictions.npz',
        pred=test_preds,
        target=test_targets
    )

    return results


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Ablação TTD-SCS-LSTM v2")

    # Configuração de treino
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--lookback", type=int, default=240)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--patience", type=int, default=30)

    # Seleção de modelo
    parser.add_argument("--model", type=str, default=None,
                        help="Modelo específico (lstm, lstm_ttd_base, etc.)")

    # Seed
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f"Seed para reprodutibilidade (default: {DEFAULT_SEED})")

    # Output dir customizado
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Diretório de saída customizado (sobrescreve padrão)")

    # Dataset customizado (default: dataset_v2.h5 de Manuel Duarte)
    parser.add_argument("--data-path", type=str, default=None,
                        help="Caminho do .h5 de treino (default: dataset_v2.h5 de MD)")

    # Flags
    parser.add_argument("--test", action="store_true",
                        help="Modo teste (1 época)")

    args = parser.parse_args()

    dataset_file = Path(args.data_path) if args.data_path else DATASET_FILE

    # Modo teste
    if args.test:
        args.epochs = 1
        args.patience = 1

    # Verificar dataset
    if not dataset_file.exists():
        print(f"[ERRO] Dataset não encontrado: {dataset_file}")
        print(f"       Execute o notebook 05_dataset_creation.ipynb primeiro.")
        return 1

    # Configuração
    config = {
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': 0.1,
        'lr': args.lr,
        'weight_decay': 1e-5,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'patience': args.patience,
        'grad_clip': 1.0,
        'lookback': args.lookback,
        'horizon': args.horizon
    }

    # Fixar seed para reprodutibilidade
    set_seed(args.seed)

    print("=" * 70)
    print("  ABLACAO - TTD-SCS-LSTM v2")
    print("  10 Experimentos de Ablacao")
    print("=" * 70)
    print(f"\nDevice: {DEVICE}")
    print(f"Seed: {args.seed}")
    print(f"Dataset: {dataset_file}")
    print(f"Lookback: {args.lookback}h ({args.lookback/24:.1f} dias)")
    print(f"Horizon: {args.horizon}h")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")

    # Tipos de modelo
    from ttd_scs_lstm.models.models import MODEL_TYPES, MODEL_DESCRIPTIONS

    if args.model:
        model_types = [args.model]
    else:
        model_types = MODEL_TYPES

    print(f"\nModelos a executar:")
    for mt in model_types:
        print(f"  - {mt}: {MODEL_DESCRIPTIONS.get(mt, '')}")

    # Timestamp e output dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = OUTPUT_DIR / f"run_{timestamp}_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Salvar configuração
    with open(run_dir / 'config.json', 'w') as f:
        json.dump({
            'config': config,
            'model_types': model_types,
            'timestamp': timestamp,
            'seed': args.seed
        }, f, indent=2)

    # Carregar dados
    print("\nCarregando dados...")
    train_loader, val_loader, test_loader, static_features = create_dataloaders(
        dataset_file,
        lookback=config['lookback'],
        horizon=config['horizon'],
        batch_size=config['batch_size']
    )

    print(f"  Train: {len(train_loader.dataset):,} samples")
    print(f"  Val: {len(val_loader.dataset):,} samples")
    print(f"  Test: {len(test_loader.dataset):,} samples")
    print(f"  Ottobacias: {static_features['n_otto']}")
    print(f"  CN 2022 medio: {static_features['cn_values'].mean():.1f}")
    print(f"  Tc Base medio: {static_features['tc_base_values'].mean():.2f}h")
    print(f"  Tc Manning medio: {static_features['tc_manning_values'].mean():.2f}h")

    # Resultados
    all_results = []

    # Loop de experimentos
    for i, model_type in enumerate(model_types, 1):
        print(f"\n[{i}/{len(model_types)}] {model_type}")

        try:
            results = run_experiment(
                model_type=model_type,
                config=config,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                static_features=static_features,
                output_dir=run_dir,
                verbose=True
            )
            all_results.append(results)

        except Exception as e:
            print(f"  [ERRO] {e}")
            import traceback
            traceback.print_exc()
            all_results.append({
                'model_name': model_type,
                'error': str(e)
            })
        finally:
            # Cleanup CUDA entre modelos: evita contexto corrompido propagar.
            # Sem isso, falha do modelo N pode matar N+1 com "CUDA error: unknown error" no .to(device).
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    # ===========================================================================
    # RESUMO FINAL
    # ===========================================================================

    print("\n" + "=" * 100)
    print("  RESUMO - ABLACAO TTD-SCS-LSTM v2")
    print("=" * 100)

    # Criar DataFrame
    summary_data = []
    for r in all_results:
        if 'error' in r:
            summary_data.append({
                'Modelo': r['model_name'],
                'NSE_1h': np.nan,
                'NSE_3h': np.nan,
                'NSE_6h': np.nan,
                'NSE_12h': np.nan,
                'NSE_24h': np.nan,
                'Params': 0
            })
        else:
            h = r.get('test_by_horizon', {})
            summary_data.append({
                'Modelo': r['model_name'],
                'NSE_1h': h.get('1h', {}).get('nse', np.nan),
                'NSE_3h': h.get('3h', {}).get('nse', np.nan),
                'NSE_6h': h.get('6h', {}).get('nse', np.nan),
                'NSE_12h': h.get('12h', {}).get('nse', np.nan),
                'NSE_24h': h.get('24h', {}).get('nse', np.nan),
                'Params': r['n_params']
            })

    df = pd.DataFrame(summary_data)
    df_sorted = df.sort_values('NSE_6h', ascending=False)

    print(f"\n{'Modelo':<25} {'NSE_1h':>8} {'NSE_3h':>8} {'NSE_6h':>8} {'NSE_12h':>8} {'NSE_24h':>8} {'Params':>10}")
    print("-" * 85)

    for _, row in df_sorted.iterrows():
        print(f"{row['Modelo']:<25} {row['NSE_1h']:>8.4f} {row['NSE_3h']:>8.4f} {row['NSE_6h']:>8.4f} {row['NSE_12h']:>8.4f} {row['NSE_24h']:>8.4f} {row['Params']:>10,}")

    # Analise de hipoteses
    print("\n" + "-" * 85)
    print("ANALISE DE HIPOTESES:")
    print("-" * 85)

    # H1: Distribuido > Lumped?
    nse_lumped = df[df['Modelo'] == 'LSTM_Lumped']['NSE_6h'].values
    nse_dist = df[df['Modelo'] == 'LSTM']['NSE_6h'].values
    if len(nse_lumped) > 0 and len(nse_dist) > 0:
        delta = nse_dist[0] - nse_lumped[0]
        status = "[OK]" if delta > 0 else "[X]"
        print(f"H1: Distribuido > Lumped? {status} (Delta={delta:+.4f})")

    # H2: Manning > Base?
    nse_base = df[df['Modelo'] == 'LSTM_TTD_Base']['NSE_6h'].values
    nse_manning = df[df['Modelo'] == 'LSTM_TTD_Manning']['NSE_6h'].values
    if len(nse_base) > 0 and len(nse_manning) > 0:
        delta = nse_manning[0] - nse_base[0]
        status = "[OK]" if delta > 0 else "[X]"
        print(f"H2: Manning > Base? {status} (Delta={delta:+.4f})")

    # H3: Ajustavel > Fixo?
    nse_fixed = df[df['Modelo'] == 'LSTM_TTD_Manning_Fixed']['NSE_6h'].values
    nse_learn = df[df['Modelo'] == 'LSTM_TTD_Manning']['NSE_6h'].values
    if len(nse_fixed) > 0 and len(nse_learn) > 0:
        delta = nse_learn[0] - nse_fixed[0]
        status = "[OK]" if delta > 0 else "[X]"
        print(f"H3: Ajustavel > Fixo? {status} (Delta={delta:+.4f})")

    # H4: SCS adiciona valor?
    nse_ttd = df[df['Modelo'] == 'LSTM_TTD_Manning']['NSE_6h'].values
    nse_scs = df[df['Modelo'] == 'LSTM_TTD_Manning_SCS']['NSE_6h'].values
    if len(nse_ttd) > 0 and len(nse_scs) > 0:
        delta = nse_scs[0] - nse_ttd[0]
        status = "[OK]" if delta > 0 else "[X]"
        print(f"H4: SCS adiciona valor? {status} (Delta={delta:+.4f})")

    # H5: Modelo completo > Baseline?
    nse_lstm = df[df['Modelo'] == 'LSTM']['NSE_6h'].values
    nse_completo = df[df['Modelo'] == 'LSTM_TTD_Manning_SCS']['NSE_6h'].values
    if len(nse_lstm) > 0 and len(nse_completo) > 0:
        delta = nse_completo[0] - nse_lstm[0]
        status = "[OK]" if delta > 0 else "[X]"
        print(f"H5: Modelo completo > Baseline? {status} (Delta={delta:+.4f})")

    # Salvar resumo
    df_sorted.to_csv(run_dir / 'summary.csv', index=False)

    # Salvar resultados completos
    with open(run_dir / 'all_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=float)

    # Gate B2 (auditoria): NAO sair 0 se algum modelo falhou ou produziu NSE@6h nao-finito.
    # Sem isto, o runner grava .done e conta como sucesso um modelo quebrado/vazio.
    falhou = []
    for r in all_results:
        if 'error' in r:
            falhou.append(f"{r.get('model_name','?')} (erro: {r['error']})")
            continue
        # métrica principal: @6h no forecasting (horizon 24), @1h na simulação (horizon 1) — generaliza
        nse_main = r.get('test_metrics', {}).get('nse', None)
        if nse_main is None or not np.isfinite(nse_main):
            falhou.append(f"{r.get('model_name','?')} (NSE principal nao-finito: {nse_main})")
    if falhou:
        print(f"\n[FALHA] {len(falhou)} modelo(s) com problema:")
        for f_ in falhou:
            print(f"   - {f_}")
        return 1

    print(f"\n[OK] Resultados salvos em: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
