# Discriminative Lexicon Model (DLM) Pipeline

This document describes the pipeline for training F-matrices and G-matrices using the Discriminative Lexicon Model framework to study how hearing loss affects form-semantic mappings in speech.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Pipeline Steps](#pipeline-steps)
4. [Quick Start](#quick-start)
5. [Configuration](#configuration)
6. [Output Files](#output-files)
7. [Troubleshooting](#troubleshooting)

---

## Overview

### What are F and G Matrices?

The Discriminative Lexicon Model learns bidirectional mappings between acoustic form and semantic meaning:

| Matrix | Direction | Task | Formula |
|--------|-----------|------|---------|
| **F-matrix** | Form → Semantics | Comprehension | `S_pred = C @ F` |
| **G-matrix** | Semantics → Form | Production | `C_pred = S @ G` |

Where:
- **C-vector** (Form): Flattened log-mel spectrogram (50,048 dimensions = 128 mels × 391 time frames)
- **S-vector** (Semantics): fastText word embedding (300 dimensions)

### Hearing Loss Conditions

The pipeline processes audio under three hearing conditions:

| Condition | Description | Attenuation Profile |
|-----------|-------------|---------------------|
| `normal` | Normal hearing baseline | 10 dB uniform across all frequencies |
| `lfloss` | Low-frequency hearing loss | 100 dB (125 Hz) → 10 dB (8000 Hz) gradient |
| `hfloss` | High-frequency hearing loss | 10 dB (125 Hz) → 100 dB (8000 Hz) gradient |

### Research Question

How do different hearing loss profiles affect the model's ability to learn mappings between acoustic form and semantic meaning in speech comprehension and production?

---

## Prerequisites

### 1. Virtual Environment

```bash
# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies
uv sync
```

### 2. Required Data

- **Audio files**: Processed word audio files organized by condition
  - `/sc/projects/sci-aisc/hearingloss/data_processed/normal_raw/{train,test,validation}/`
  - `/sc/projects/sci-aisc/hearingloss/data_processed/lfloss_raw/{train,test,validation}/`
  - `/sc/projects/sci-aisc/hearingloss/data_processed/hfloss_raw/{train,test,validation}/`

- **fastText model**: Pre-trained English word embeddings
  - `data/fasttext/cc.en.300.bin`

### 3. Required Packages

Key dependencies (installed via `uv sync`):
- `discriminative_lexicon_model` - Core DLM implementation
- `fasttext-wheel>=0.9.2` - Word embeddings
- `xarray>=2023.6.0` - Matrix data structures
- `torch` - GPU acceleration (optional but recommended)

---

## Pipeline Steps

### Step 0: Apply Hearing Loss Masks to Audio (if needed)

**Purpose**: Apply hearing loss simulations to raw word audio files.

**Script**: `scripts/apply_all_masks.py`

```bash
# Submit SLURM job to process all audio files
sbatch scripts/apply_all_masks.sbatch

# Or run interactively with options
python scripts/apply_all_masks.py \
    --input-dir /path/to/raw/audio \
    --output-base /path/to/output \
    --splits train test validation \
    --conditions normal lfloss hfloss \
    --nworkers 48
```

**Output**: Audio files in three condition directories (`normal_raw/`, `lfloss_raw/`, `hfloss_raw/`)

**Duration**: ~1-2 hours for all splits with 48 workers

> **Note**: Skip this step if processed audio already exists.

---

### Step 1: Generate S-Matrices (Semantic Vectors)

**Purpose**: Pre-compute semantic representations for all unique words using fastText embeddings.

**Script**: `scripts/generate_smatrix.py`

```bash
# Generate S-matrices for all data splits
python scripts/generate_smatrix.py --split train
python scripts/generate_smatrix.py --split test
python scripts/generate_smatrix.py --split validation
```

**Output**:
- `data/smatrices/train_smatrix.nc` (~60 MB)
- `data/smatrices/test_smatrix.nc`
- `data/smatrices/validation_smatrix.nc`

**Duration**: ~5-10 minutes per split

---

### Step 2: Train F and G Matrices

**Purpose**: Train both comprehension (F) and production (G) matrices via incremental learning.

**Script**: `scripts/train_fmatrix_incremental.py`

#### Option A: SLURM Batch Job (Recommended)

```bash
# Submit array job for all three conditions in parallel
sbatch scripts/train_fmatrix_incremental.sbatch

# Or submit for a specific condition
sbatch scripts/train_fmatrix_incremental.sbatch --condition normal
```

#### Option B: Interactive Execution

```bash
# Test with small subset first
python scripts/train_fmatrix_incremental.py \
    --condition normal \
    --limit 1000

# Full training for one condition
python scripts/train_fmatrix_incremental.py --condition normal
```

#### Resuming from Checkpoint

If training is interrupted, use `--auto-resume` to continue from the latest checkpoint:

```bash
sbatch scripts/train_fmatrix_incremental.sbatch --auto-resume

# Or specify checkpoints manually
python scripts/train_fmatrix_incremental.py \
    --condition normal \
    --resume-from-f results/matrices_incremental/checkpoints/normal/fmatrix_normal_iter2550000.csv.gz \
    --resume-from-g results/matrices_incremental/checkpoints/normal/gmatrix_normal_iter2550000.csv.gz
```

**Output** (per condition):
- `results/matrices_incremental/checkpoints/{condition}/fmatrix_{condition}_iter{N}.csv.gz`
- `results/matrices_incremental/checkpoints/{condition}/gmatrix_{condition}_iter{N}.csv.gz`
- `results/matrices_incremental/metrics/{condition}/training_metrics.csv`
- `results/matrices_incremental/metrics/{condition}/training_order_{condition}.csv`

**Duration**:
- With GPU (H100): ~70 hours per condition for ~10M samples
- With CPU only: ~1-2 weeks per condition

**Monitor Progress**:
```bash
# Check job status
squeue -u $USER

# View training logs
tail -f logs/train_fmatrix_*.err

# Check latest metrics
tail results/matrices_incremental/metrics/normal/training_metrics.csv
```

---

### Step 3: Evaluate Trained Matrices

**Purpose**: Evaluate F and G matrices on test/validation sets and compare across hearing conditions.

**Script**: `scripts/evaluate_matrices.py`

```bash
# Evaluate all conditions on test and validation splits
python scripts/evaluate_matrices.py

# Evaluate specific conditions
python scripts/evaluate_matrices.py --conditions normal lfloss

# Quick test with limited files
python scripts/evaluate_matrices.py --limit 100
```

**Output**:
- `results/matrices_incremental/evaluation/evaluation_results.json`
- `results/matrices_incremental/evaluation/evaluation_summary.csv`
- `results/matrices_incremental/evaluation/fmatrix_comparison.json`
- `results/matrices_incremental/evaluation/gmatrix_comparison.json`

**Metrics Computed**:
- Mean cosine similarity
- Mean Pearson correlation
- RMSE (Root Mean Squared Error)
- Per-sample cosine similarity distributions

**Duration**: ~30 minutes to 2 hours

---

### Step 4: Generate Visualizations

**Purpose**: Create plots for learning curves, evaluation comparisons, and matrix differences.

**Script**: `scripts/visualize_learning_curves.py`

```bash
python scripts/visualize_learning_curves.py
```

**Output**:
- `results/matrices_incremental/visualizations/learning_curves.png` - 2×2 grid of F/G metrics over time
- `results/matrices_incremental/visualizations/evaluation_comparison.png` - Comprehension vs production metrics
- `results/matrices_incremental/visualizations/fmatrix_comparison.png` - Cross-condition F-matrix differences
- `results/matrices_incremental/visualizations/gmatrix_comparison.png` - Cross-condition G-matrix differences

**Duration**: ~1-5 minutes

---

## Quick Start

### Minimal Test Run

```bash
source .venv/bin/activate

# 1. Generate S-matrix for training split
python scripts/generate_smatrix.py --split train

# 2. Test training with 1000 samples
python scripts/train_fmatrix_incremental.py --condition normal --limit 1000

# 3. Generate S-matrix for test split
python scripts/generate_smatrix.py --split test

# 4. Evaluate (with limit)
python scripts/evaluate_matrices.py --conditions normal --limit 100

# 5. Visualize
python scripts/visualize_learning_curves.py --conditions normal
```

### Full Production Run

```bash
source .venv/bin/activate

# 1. Generate all S-matrices
python scripts/generate_smatrix.py --split train
python scripts/generate_smatrix.py --split test
python scripts/generate_smatrix.py --split validation

# 2. Submit training jobs for all conditions
sbatch scripts/train_fmatrix_incremental.sbatch

# 3. Monitor progress
squeue -u $USER
tail -f logs/train_fmatrix_*.err

# 4. After training completes, evaluate
python scripts/evaluate_matrices.py

# 5. Generate visualizations
python scripts/visualize_learning_curves.py
```

---

## Configuration

### Training Configuration

All parameters are stored in `configs/fmatrix_training_config.yaml`:

```yaml
data:
  audio_dirs:
    normal: /sc/projects/sci-aisc/hearingloss/data_processed/normal_raw
    lfloss: /sc/projects/sci-aisc/hearingloss/data_processed/lfloss_raw
    hfloss: /sc/projects/sci-aisc/hearingloss/data_processed/hfloss_raw
  fasttext_model: data/fasttext/cc.en.300.bin
  splits: [train, test, validation]

spectrogram:
  sampling_rate: 16000
  n_mels: 128
  n_fft: 400           # 25 ms window
  hop_length: 160      # 10 ms hop
  max_time_frames: 391 # Pad/truncate to this length

training:
  learning_rate: 0.1
  checkpoint_interval: 50000  # Save every 50K iterations
  metrics_interval: 1000      # Log metrics every 1K iterations
  random_seed: 42             # For reproducibility
  backend: auto               # 'numpy', 'torch', or 'auto'
  device: cuda                # 'cuda' or 'cpu'

output:
  base_dir: results/matrices_incremental
  checkpoints_dir: checkpoints
  metrics_dir: metrics
  visualizations_dir: visualizations
```

### SLURM Configuration

The batch script `scripts/train_fmatrix_incremental.sbatch` requests:
- 1 node, 64 CPUs, 64 GB RAM
- 1 H100 GPU
- 168 hours (7 days) time limit
- Array job with 3 tasks (one per condition)

---

## Output Files

### Directory Structure

```
results/matrices_incremental/
├── checkpoints/
│   ├── normal/
│   │   ├── fmatrix_normal_iter0050000.csv.gz
│   │   ├── gmatrix_normal_iter0050000.csv.gz
│   │   ├── ...
│   │   ├── fmatrix_normal_iter10205536.csv.gz  # Final
│   │   └── gmatrix_normal_iter10205536.csv.gz  # Final
│   ├── lfloss/
│   │   └── ...
│   └── hfloss/
│       └── ...
├── metrics/
│   ├── normal/
│   │   ├── training_metrics.csv
│   │   ├── training_order_normal.csv
│   │   └── failed_files.csv (if any)
│   ├── lfloss/
│   └── hfloss/
├── evaluation/
│   ├── evaluation_results.json
│   ├── evaluation_summary.csv
│   ├── fmatrix_comparison.json
│   └── gmatrix_comparison.json
└── visualizations/
    ├── learning_curves.png
    ├── evaluation_comparison.png
    ├── fmatrix_comparison.png
    └── gmatrix_comparison.png
```

### Training Metrics CSV

Columns in `training_metrics.csv`:
| Column | Description |
|--------|-------------|
| `iteration` | Training iteration number |
| `f_prediction_error` | L2 error for F-matrix (comprehension) |
| `f_cosine_similarity` | Cosine similarity for F-matrix predictions |
| `g_prediction_error` | L2 error for G-matrix (production) |
| `g_cosine_similarity` | Cosine similarity for G-matrix predictions |
| `timestamp` | ISO format timestamp |

---

## Troubleshooting

### S-matrix not found

**Error**: `FileNotFoundError: data/smatrices/train_smatrix.nc`

**Solution**: Run Step 1 first:
```bash
python scripts/generate_smatrix.py --split train
```

### Word not in S-matrix

**Warning**: `Word 'xyz' not in S-matrix, skipping file`

**Cause**: Some words don't have fastText embeddings or weren't in the S-matrix generation.

**Solution**: Normal behavior - these files are logged in `failed_files.csv` and skipped.

### Training is very slow

**Cause**: Using CPU backend instead of GPU.

**Solution**:
1. Ensure GPU is available: `nvidia-smi`
2. Check PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Verify config has `device: cuda` and `backend: auto`

### CUDA out of memory

**Cause**: GPU memory insufficient.

**Solution**: The incremental learning approach uses minimal memory (one sample at a time). If still failing, try:
```yaml
training:
  backend: numpy
  device: cpu
```

### Checkpoint mismatch when resuming

**Error**: F-matrix and G-matrix at different iterations.

**Solution**: Use `--auto-resume` which automatically finds matching checkpoints, or manually specify both:
```bash
python scripts/train_fmatrix_incremental.py \
    --condition normal \
    --resume-from-f .../fmatrix_normal_iter0100000.csv.gz \
    --resume-from-g .../gmatrix_normal_iter0100000.csv.gz
```

---

## References

Baayen, R. H., Chuang, Y. Y., Shafaei-Bajestan, E., & Blevins, J. P. (2019). The discriminative lexicon: A unified computational model for the lexicon and lexical processing in comprehension and production grounded not in (de)composition but in linear discriminative learning. *Complexity*, 2019.

- discriminative_lexicon_model package: https://github.com/quantling/discriminative_lexicon_model
- fastText: https://fasttext.cc/
