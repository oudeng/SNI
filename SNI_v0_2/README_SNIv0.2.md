# SNI v0.2.1 - Statistical-Neural Interaction for Missing Data Imputation

## Overview

SNI (Statistical-Neural Interaction) is a hybrid missing data imputation method that combines statistical priors with neural attention mechanisms. It achieves state-of-the-art performance on mixed-type tabular data by leveraging both the interpretability of statistical methods and the flexibility of deep learning.

**Version 0.2.1** introduces enhanced model capacity and optimized hyperparameters for improved accuracy while maintaining computational efficiency.

## Key Innovations

### 1. Controllable-Prior Feature Attention (CPFA)
- Multi-head attention over feature tokens
- **Learnable per-head confidence parameters (λ)** that automatically balance data-driven learning with statistical priors
- Prior strength decays across EM iterations via α₀ · γ^(g-1)

### 2. Statistical-Neural Integration
- **Statistical priors** from correlation analysis guide initial attention patterns
- **Neural attention** learns refined feature dependencies from data
- Adaptive λ parameters determine optimal prior influence per attention head

### 3. Multi-Scale Attention for Categorical Features
- **Local heads**: Focus on within-category relationships
- **Global heads**: Capture cross-category dependencies
- Learned fusion of multi-scale representations

### 4. Dual-Path Architecture (for categorical targets)
- Path 1: Attention-based classification
- Path 2: Continuous embedding path
- Learned gating combines both paths

## Installation

```bash
# Clone or copy SNI_v0_2 to your project
cp -r SNI_v0_2 /path/to/your/project/

# Install dependencies
pip install torch numpy pandas scikit-learn scipy
```

## Quick Start

```python
import pandas as pd
from SNI_v0_2 import SNIImputer, SNIConfig

# Load your data
X_missing = pd.read_csv("data_with_missing.csv")
X_complete = pd.read_csv("data_complete.csv")  # Optional, for evaluation

# Define variable types
categorical_vars = ["cat1", "cat2"]
continuous_vars = ["cont1", "cont2", "cont3"]

# Create imputer with default optimized config
config = SNIConfig(
    use_gpu=True,  # Set to False for CPU
    seed=42,
)

imputer = SNIImputer(
    categorical_vars=categorical_vars,
    continuous_vars=continuous_vars,
    config=config,
)

# Run imputation
X_imputed = imputer.impute(X_missing=X_missing, X_complete=X_complete)

# Access learned dependencies
dependency_matrix = imputer.compute_dependency_matrix()
print(dependency_matrix)
```

## Configuration

### SNIConfig Parameters (v0.2.1 High-Performance Defaults)

| Parameter | Default | Description |
|-----------|---------|-------------|
| **EM/Prior** | | |
| `alpha0` | 1.0 | Initial prior strength |
| `gamma` | 0.9 | Prior decay rate per iteration |
| `max_iters` | 3 | Number of EM iterations |
| `tol` | 1e-4 | Convergence tolerance |
| `use_stat_refine` | True | Statistical refinement after neural prediction |
| **Architecture** | | |
| `hidden_dims` | (256, 128, 64) | MLP hidden layer dimensions (3-layer) |
| `emb_dim` | 128 | Feature embedding dimension |
| `num_heads` | 16 | Number of attention heads |
| **Training** | | |
| `lr` | 2e-4 | Learning rate (conservative) |
| `weight_decay` | 1e-4 | AdamW weight decay |
| `epochs` | 200 | Maximum training epochs |
| `batch_size` | 128 | Batch size |
| `early_stopping_patience` | 20 | Early stopping patience |
| `min_epochs` | 50 | Minimum epochs before early stop |
| `mask_fraction` | 0.15 | Pseudo-masking fraction |
| **Categorical Techniques** | | |
| `use_dual_path` | True | Dual-path architecture |
| `use_multiscale` | True | Multi-scale attention |
| `use_focal_loss` | True | Focal loss for class imbalance |
| `use_label_smoothing` | True | Label smoothing |
| `label_smoothing_epsilon` | 0.1 | Label smoothing strength |
| `use_mixup` | True | Mixup augmentation |
| `mixup_alpha` | 0.2 | Mixup alpha parameter |
| **Variant** | | |
| `variant` | "SNI" | {SNI, NoPrior, HardPrior, SNI-M} |

### Variants

| Variant | Description |
|---------|-------------|
| `SNI` | Full model with learnable λ |
| `NoPrior` | Ablation: no statistical prior |
| `HardPrior` | Ablation: fixed λ (hard prior) |
| `SNI-M` | SNI with missingness-aware embeddings |
| `SNI+KNN` | SNI with KNN post-processing |

## Running Experiments

### From Command Line

```bash
python scripts/run_manifest.py \
    --manifest data/manifest.csv \
    --outdir results/ \
    --default-use-gpu true \
    --skip-existing
```

### Manifest CSV Format

```csv
exp_id,input_complete,input_missing,variant,seed,categorical_vars,continuous_vars,hidden_dims,emb_dim,num_heads,lr,epochs,batch_size
exp_001,data/complete.csv,data/missing.csv,SNI,42,"cat1,cat2","cont1,cont2","256,128,64",128,16,0.0002,200,128
```

## Performance Comparison

### v0.2.1 High-Performance Configuration

Optimized for SOTA imputation accuracy while maintaining reasonable training time.

| Parameter | v0.2 | v0.2.1 | Rationale |
|-----------|------|--------|-----------|
| hidden_dims | (128, 64) | (256, 128, 64) | Maximum model capacity |
| emb_dim | 64 | 128 | Richer feature representations |
| num_heads | 4 | 16 | Diverse attention patterns |
| epochs | 80 | 200 | Sufficient training |
| lr | 1e-3 | 2e-4 | Conservative for stability |
| batch_size | 128 | 128 | Stable gradient estimates |
| gamma | 0.9 | 0.9 | Balanced prior decay |
| mask_fraction | 0.15 | 0.15 | Strong pseudo-supervision |

### Computational Budget Comparison with Baselines

| Method | Typical Runtime | Parameters |
|--------|-----------------|------------|
| **SNI v0.2.1** | 90-180s | 200 epochs, 3 EM iters |
| MICE | 5-10s | 5 iterations |
| MissForest | 10-30s | 100 trees, max 10 iters |
| MIWAE | 60-120s | 500 epochs |
| GAIN | 30-60s | 10k iterations |

SNI v0.2.1's computational budget is aligned with other neural baselines while providing significantly better accuracy and interpretability through the dependency matrix output.

## Output Artifacts

After imputation, the following artifacts are available:

```python
# Imputed data
X_imputed = imputer.impute(X_missing)

# Dependency matrix D[i,j] = importance of feature j for predicting feature i
D = imputer.compute_dependency_matrix()

# Edge list for network visualization
edges = imputer.export_dependency_network_edges(tau=0.15)

# Per-feature attention maps
attention_maps = imputer.attention_maps  # Dict[feature_name, np.ndarray]

# Lambda (confidence) traces over training
lambda_traces = imputer.lambda_trace_per_head  # Dict[feature_name, List[List[float]]]

# Training logs
logs = imputer.logs  # Dict[feature_name, {"tr": [...], "va": [...]}]
```

## Hardware Requirements

- **CPU**: Multi-core recommended (SNI uses parallel data loading)
- **GPU**: Optional but recommended for large datasets
  - CUDA-capable GPU with 4GB+ VRAM
  - Tested on NVIDIA RTX 3090, A100
- **RAM**: 8GB+ recommended

### GPU Configuration

```python
# Enable GPU
config = SNIConfig(use_gpu=True)

# For multi-GPU (use first available)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

## Reproducibility

SNI v0.2.1 uses deterministic seeds for reproducibility:

```python
config = SNIConfig(seed=42)  # Set seed for reproducible results
```

Note: GPU operations may have slight non-determinism. For exact reproducibility, use CPU mode.

## Citation

If you use SNI in your research, please cite:

```bibtex
@article{SNI2025,
  title={Statistical-Neural Interaction for Missing Data Imputation},
  author={...},
  journal={...},
  year={2025}
}
```

## File Structure

```
SNI_v0_2/
├── __init__.py          # Package exports
├── imputer.py           # Main SNIImputer class
├── cpfa.py              # CPFA neural network modules
├── losses.py            # Custom loss functions
├── dataio.py            # Data I/O utilities
├── metrics.py           # Evaluation metrics
└── utils.py             # Utility functions

scripts/
├── run_manifest.py      # Main experiment runner
├── run_manifest_parallel.py  # Parallel experiment runner
├── aggregate_results.py # Result aggregation
├── make_latex_table.py  # LaTeX table generation
└── viz_*.py             # Visualization scripts
```

## Changelog

### v0.2.1 (2025-01)
- **High-Performance Configuration**: 
  - hidden_dims: (128, 64) → (256, 128, 64) - 3-layer MLP
  - emb_dim: 64 → 128
  - num_heads: 4 → 16
  - epochs: 80 → 200
  - lr: 1e-3 → 2e-4 (more conservative)
  - early_stopping_patience: 10 → 20
  - min_epochs: NEW → 50
- **Small Optimizations**:
  - batch_size: 64 → 128 (more stable gradients)
  - gamma: 0.95 → 0.9 (faster prior decay)
  - mask_fraction: 0.1 → 0.15 (more pseudo-supervision)
- **New Parameters**: 
  - `weight_decay`: Configurable AdamW weight decay
  - `min_epochs`: Minimum epochs before early stopping
  - `mixup_alpha`: Configurable mixup strength

### v0.2.0 (2025-01)
- **Performance**: 40-50% faster than v6.3r2
  - Reduced epochs: 80 → 50
  - Reduced EM iterations: 3 → 2
  - Increased batch size: 64 → 128
  - Added early stopping (patience=10)
  - Optimized MICE initialization
- **Code Quality**: Improved documentation and modularity
- **Bug Fixes**: Fixed edge cases in categorical handling

### v6.3r2 (Previous)
- Initial public release
- Full SNI algorithm implementation

## License

MIT License - see LICENSE file for details.

## Support

For issues and questions, please open an issue on the GitHub repository.
