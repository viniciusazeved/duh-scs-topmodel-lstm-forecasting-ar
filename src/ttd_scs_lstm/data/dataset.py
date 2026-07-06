#!/usr/bin/env python
"""
Dataset for TTD-SCS-LSTM
========================

Supports Lumped and Distributed configurations.

- Lumped: Basin-averaged P, single CN, single Tc
- Distributed: P per subcatchment, CN per subcatchment, Tc per subcatchment

Author: Vinicius + Claude
Date: 2026-01-23
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import h5py
from datetime import datetime
from typing import Tuple, Dict


class AblationDataset(Dataset):
    """
    Dataset for ablation experiments.

    Returns:
        - precip: Precipitation (seq_len,) or (seq_len, n_otto)
        - hour: Normalized hour (seq_len,)
        - month: Normalized month (seq_len,)
        - target: Future streamflow (horizon,)

    Static features (CN, Tc) are accessed via attributes.
    """

    def __init__(
        self,
        h5_path: Path,
        split: str = 'train',
        lookback: int = 240,
        horizon: int = 24,
        distributed: bool = True
    ):
        """
        Args:
            h5_path: Path to HDF5 dataset
            split: 'train', 'val' or 'test'
            lookback: Input window (hours)
            horizon: Forecast horizon (hours)
            distributed: If True, use P per subcatchment; if False, use basin-averaged P
        """
        self.lookback = lookback
        self.horizon = horizon
        self.distributed = distributed
        self.split = split

        with h5py.File(h5_path, 'r') as f:
            # Load data
            precip = f[f'{split}/precipitation'][:]  # (n_times, n_otto)
            flow = f[f'{split}/streamflow'][:]       # (n_times,)
            timestamps = f[f'{split}/timestamps'][:]

            # Subcatchment properties
            self.area_km2 = torch.from_numpy(f['ottobacia/area_km2'][:]).float()
            self.cn_values = torch.from_numpy(f['ottobacia/cn_2022'][:]).float()
            self.tc_base_h = torch.from_numpy(f['ottobacia/tc_base_h'][:]).float()
            self.tc_manning_h = torch.from_numpy(f['ottobacia/tc_manning_h'][:]).float()

        self.n_otto = precip.shape[1]

        # Process precipitation
        if distributed:
            self.precip_full = precip  # (n_times, n_otto)
        else:
            # Lumped: area-weighted average
            weights = self.area_km2.numpy() / self.area_km2.numpy().sum()
            self.precip_full = (precip * weights).sum(axis=1, keepdims=True)  # (n_times, 1)

        self.flow_full = flow
        self.timestamps_full = timestamps

        # Create valid sequences
        self._create_sequences()

    def _create_sequences(self):
        """Create indices of valid sequences."""
        n_total = len(self.flow_full) - self.lookback - self.horizon + 1

        valid_indices = []

        for i in range(n_total):
            # Check NaN in target
            y_start = i + self.lookback
            y_end = y_start + self.horizon
            y = self.flow_full[y_start:y_end]

            if np.isnan(y).any():
                continue

            # Check NaN in precipitation
            x_start = i
            x_end = i + self.lookback
            p = self.precip_full[x_start:x_end]

            if np.isnan(p).any():
                continue

            valid_indices.append(i)

        self.valid_indices = np.array(valid_indices)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]

        # Precipitation
        x_start = i
        x_end = i + self.lookback
        precip = self.precip_full[x_start:x_end]  # (lookback,) or (lookback, n_otto)

        # Target
        y_start = i + self.lookback
        y_end = y_start + self.horizon
        target = self.flow_full[y_start:y_end]  # (horizon,)

        # Temporal features — helper unico (UTC), mesma convencao do treino
        # (fix do skew fromtimestamp/BRT; review 10/06/2026)
        from .temporal import features_temporais
        ts_seq = self.timestamps_full[x_start:x_end]
        hour_norm, month_norm = features_temporais(ts_seq)

        return {
            'precip': torch.from_numpy(precip).float(),
            'hour': torch.from_numpy(hour_norm).float(),
            'month': torch.from_numpy(month_norm).float(),
            'target': torch.from_numpy(target).float()
        }

    def get_static_features(self) -> Dict[str, torch.Tensor]:
        """Return static basin features."""
        if self.distributed:
            return {
                'cn_values': self.cn_values,
                'tc_base_h': self.tc_base_h,
                'tc_manning_h': self.tc_manning_h,
                'area_km2': self.area_km2,
                'n_otto': self.n_otto
            }
        else:
            # Lumped: aggregated values
            total_area = self.area_km2.sum()
            weights = self.area_km2 / total_area

            return {
                'cn_values': (self.cn_values * weights).sum().unsqueeze(0),
                'tc_base_h': self.tc_base_h.max().unsqueeze(0),  # Max Tc
                'tc_manning_h': self.tc_manning_h.max().unsqueeze(0),
                'area_km2': total_area.unsqueeze(0),
                'n_otto': 1
            }


def create_dataloaders(
    h5_path: Path,
    lookback: int = 240,
    horizon: int = 24,
    distributed: bool = True,
    batch_size: int = 64,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict]:
    """
    Create DataLoaders for train, validation and test.

    Returns:
        (train_loader, val_loader, test_loader, static_features)
    """
    train_ds = AblationDataset(h5_path, 'train', lookback, horizon, distributed)
    val_ds = AblationDataset(h5_path, 'val', lookback, horizon, distributed)
    test_ds = AblationDataset(h5_path, 'test', lookback, horizon, distributed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    static_features = train_ds.get_static_features()

    return train_loader, val_loader, test_loader, static_features
