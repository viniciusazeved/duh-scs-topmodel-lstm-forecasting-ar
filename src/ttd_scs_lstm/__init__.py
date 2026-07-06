"""
TTD-SCS-LSTM: Hybrid Physics-Neural Model for Streamflow Prediction
====================================================================

A differentiable physics-neural hybrid model combining:
- TTD (Travel Time Distribution): Gaussian IUH with learnable tc_scale and sigma
- SCS-CN: Runoff separation with learnable lambda
- LSTM: Neural refinement of physical estimate

Results (Manuel Duarte basin, 3,117 km², 245 subcatchments):
- Forecasting (6h): NSE = 0.84 (Very Good)
- Continuous Simulation: NSE = 0.82 (Very Good)

Author: Vinicius + Claude
Date: 2026-01-23
"""

__version__ = "1.0.0"

from .models.models import (
    create_model,
    MODEL_TYPES,
    MODEL_DESCRIPTIONS,
    SCSLayer,
    DUHLayer,
    LSTMLumped,
    LSTMDistributed,
    LSTMWithTTD,
    LSTMWithTTDSCS,
)

from .data.dataset import (
    AblationDataset,
    create_dataloaders,
)

__all__ = [
    # Models
    "create_model",
    "MODEL_TYPES",
    "MODEL_DESCRIPTIONS",
    "SCSLayer",
    "DUHLayer",
    "LSTMLumped",
    "LSTMDistributed",
    "LSTMWithTTD",
    "LSTMWithTTDSCS",
    # Data
    "AblationDataset",
    "create_dataloaders",
]
