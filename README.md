![AISC BMFTR Logo](img/logo_aisc_bmftr.jpg)

# KI-based speech recognition as a method for investigating hearing loss

## Table of Contents

1. [Requirements](#1-requirements)
   
   1.1. [Virtual Environment](#11-virtual-environment)
   
   1.2. [Working Environment](#12-working-environment)

2. [Scripts](#2-scripts)

3. [Data - CommonVoice English Dataset](#3-data---commonvoice-english-dataset)
   
   3.1. [Setup Authentication](#31-setup-authentication)
   
   3.2. [Download the Dataset](#32-download-the-dataset)
   
   3.3. [Download Options](#33-download-options)

4. [Data Preprocessing](#4-data-preprocessing)
   
   4.1. [Audio Masking for Hearing Loss Simulation](#41-audio-masking-for-hearing-loss-simulation)
   
   4.1.1. [Overview](#411-overview)
   
   4.1.2. [Usage](#412-usage)
   
   4.1.3. [Advanced Options](#413-advanced-options)
   
   4.1.4. [Processing Parameters](#414-processing-parameters)
   
   4.1.5. [Technical Details](#415-technical-details)
   
   4.2. [Log Mel-Frequency Spectrograms](#42-log-mel-frequency-spectrograms)
   
   4.2.1. [Overview](#421-overview)
   
   4.2.2. [Individual Processing Script](#422-individual-processing-script)
   
   4.2.3. [Batch Processing Script](#423-batch-processing-script)

5. [Whisper](#5-whisper)
   
   5.1. [Training of Whisper](#51-training-of-whisper)
   
   5.2. [Prediction with Whisper](#52-prediction-with-whisper)

6. [LDL-AURIS](#6-ldl-auris)

7. [SLURM Processing](#7-slurm-processing)
   
   7.1. [Audio Masking](#71-audio-masking)
   
   7.2. [Log Mel-Frequency Spectrograms](#72-log-mel-frequency-spectrograms)
   
   7.3. [Whisper Training](#73-whisper-training)
   
   7.4. [Whisper Prediction](#74-whisper-prediction)

## 1. Requirements

### 1.1. Virtual Environment

To ensure that the same requirements are met across different operating systems and machines, it is recommended to create a virtual environment. This can be set up with *UV*.

```bash
which uv || echo "UV not found" # checks the UV installation
```

If UV is not installed, it can be installed as follows.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env # Add UV to PATH for current session or restart the terminal.
```

Afterwards, the virtual environment can be created and activated.

```bash
uv venv .venv # creates a virtual environment with the name ".venv"
source .venv/bin/activate # activates the virtual environment
```

Then the required packages are installed. UV ensures that the exact versions are installed.

```bash
uv sync --active  # installs exact versions
```

### 1.2. Working Environment

**Note:** This project is designed to run on the **AISC cluster** at HPI. The SLURM batch scripts are pre-configured with AISC-specific settings (`--account=aisc`, `--partition=aisc`, `--qos=aisc`). If you're using a different HPC cluster, you'll need to modify the SLURM directives in the `.sbatch` files accordingly.

**Getting Access to the AISC Cluster:**
- General HPI cluster documentation: [https://docs.sc.hpi.de/](https://docs.sc.hpi.de/)
- AISC-specific documentation: [https://aisc.hpi.de/doc/doku.php?id=start](https://aisc.hpi.de/doc/doku.php?id=start)

Before running any SLURM scripts, you need to configure your personal working directory:

1. **Copy the environment template:**
   ```bash
   cp .env.local.template .env.local
   ```

2. **Edit `.env.local` to set your working directory:**
   ```bash
   # Open in your preferred editor
   nano .env.local
   # or
   vim .env.local
   ```

3. **Update the `PROJECT_ROOT` variable** to point to your personal working directory:
   ```bash
   # Example for user "john.doe":
   PROJECT_ROOT=/sc/home/john.doe/pilotproject-hearing-loss
   
   # Example for different mount point:
   PROJECT_ROOT=/home/username/projects/pilotproject-hearing-loss
   ```

4. **Verify your configuration:**
   ```bash
   source .env.local
   echo "Project root: $PROJECT_ROOT"
   ```

**Note:** The `.env.local` file is ignored by git, so your personal configuration won't be committed to the repository.

## 2. Scripts

All scripts are located in the scripts folder.

## 3. Data - CommonVoice English Dataset

> **⚠️ CRITICAL: Dataset Access Has Changed (Updated 18.11.2025)**  
> Mozilla has removed CommonVoice datasets from HuggingFace. The dataset is now exclusively available through the [Mozilla Data Collective platform](https://datacollective.mozillafoundation.org/datasets).
> 
> **Status Update:**
> - The original CommonVoice 16.1 dataset (66 GB, 1.7M clips) is no longer accessible through HuggingFace
> - Mozilla Data Collective only provides access to the latest version (CommonVoice 23.0: 86.83 GB, 2.54M clips)
> - Older dataset versions cannot be downloaded anymore easily. One would need to get in touch with Mozilla Data Collective via email to tell them which version you need and why. See: [Mozilla's community discussion on accessing older versions](https://community.mozilladatacollective.com/were-changing-access-to-older-versions-of-common-voice-datasets/)
> - **Migration Required:** This repository must be updated to work with [CommonVoice 23.0](https://datacollective.mozillafoundation.org/datasets/cmflnuzw52mzok78yz6woemc1)
> 
> **Action Required:**
> - The instructions below reference the old HuggingFace location and **do not work anymore**
> - The `download_commonvoice.py` script needs to be rewritten to use the Mozilla Data Collective API
> - Dataset format conversion from TSV (Mozilla) to Arrow (HuggingFace) format is required
> - See: [Mozilla's FAQ on dataset access](https://community.mozilladatacollective.com/faq-can-i-get-the-common-voice-or-other-mdc-datasets-from-other-platforms-like-github-or-hugging-face/)
>
> **This migration is planned and will be implemented in a separate pull request.** 

Create a folder where the data will be stored. Because the amount of data is relatively big, data will not be provided by this github repository but has to be downloaded with the scripts below.

```bash
mkdir data
```

This project uses the CommonVoice English dataset from HuggingFace. **Note: This dataset requires HuggingFace authentication.**

### 3.1. Setup Authentication

1. **Create a HuggingFace account** at [huggingface.co](https://huggingface.co)
2. **Get an access token**: Go to Settings → Access Tokens → Create new token (Read access is sufficient)
3. **Accept the dataset terms**: Visit [CommonVoice 16.1](https://huggingface.co/datasets/mozilla-foundation/common_voice_16_1) and accept the terms

### 3.2. Download the Dataset

```bash
# Method 1: Using environment variable (recommended)
export HF_TOKEN="your_token_here"
python scripts/download_commonvoice.py

# Method 2: Using command line argument
python scripts/download_commonvoice.py --token "your_token_here"
```

This will download the dataset to `data/CommonVoiceEN/` by default.

### 3.3. Download Options

The script supports several options:

```bash
# Download with full caching (recommended for development)
python scripts/download_commonvoice.py

# Download in streaming mode (for large-scale processing)
python scripts/download_commonvoice.py --streaming

# Download only specific splits
python scripts/download_commonvoice.py --splits train validation

# Custom output directory
python scripts/download_commonvoice.py --output-dir /path/to/custom/location

# Use different CommonVoice version (if needed)
ice.py --version mozilla-foundation/common_voice_17_0
```

## 4. Data Preprocessing

### 4.1. Audio Masking for Hearing Loss Simulation

This project includes functionality to simulate different types of hearing loss by applying frequency-specific attenuation masks to the CommonVoice dataset.

#### 4.1.1. Overview

The `scripts/mask_audio.py` script processes CommonVoice datasets to create three variants:
- **Normal hearing baseline** (`*_normal`): 10 dB threshold across all frequencies
- **Low-frequency hearing loss** (`*_lfloss`): High attenuation at low frequencies (125 Hz: 100 dB → 8000 Hz: 10 dB)
- **High-frequency hearing loss** (`*_hfloss`): High attenuation at high frequencies (125 Hz: 10 dB → 8000 Hz: 100 dB)

#### 4.1.2. Usage

```bash
# Basic usage - process the downloaded CommonVoice dataset
python scripts/mask_audio.py

# This creates three new datasets:
# - data/CommonVoiceEN_normal/
# - data/CommonVoiceEN_lfloss/
# - data/CommonVoiceEN_hfloss/
```

#### 4.1.3. Advanced Options

```bash
# Specify input and output directories
python scripts/mask_audio.py \
    --input-dir data/CommonVoiceEN \
    --output-base data/MyProcessedDataset

# Configure processing parameters
python scripts/mask_audio.py \
    --sample-rate 16000 \
    --batch-size 64 \
    --num-workers 8

# Enable debug logging
python scripts/mask_audio.py --log-level DEBUG
```

#### 4.1.4. Processing Parameters

- **`--input-dir`**: Path to the input CommonVoice dataset (default: `data/CommonVoiceEN`)
- **`--output-base`**: Base name for output directories (default: same as input directory)
- **`--sample-rate`**: Target sample rate in Hz (default: 16000, required for Whisper models)
- **`--batch-size`**: Number of audio samples to process per batch (default: 32)
- **`--num-workers`**: Number of CPU cores for parallel processing (default: 4)
- **`--log-level`**: Logging verbosity: DEBUG, INFO, WARNING, ERROR (default: INFO)

#### 4.1.5. Technical Details

**Audio Processing Pipeline:**
1. **Resampling**: Audio is resampled to the target sample rate (16 kHz by default)
2. **STFT**: Short-Time Fourier Transform with 2048-sample window and 512-sample hop
3. **Frequency Masking**: Interpolated attenuation based on hearing loss profiles
4. **Reconstruction**: Inverse STFT to reconstruct audio signals
5. **Normalization**: Audio amplitude normalization to prevent clipping

**Memory Management:**
- Uses batch processing to handle large datasets efficiently
- Supports multiprocessing for faster execution
- Automatically manages memory cleanup between batches

**Output Structure:**
Each output dataset preserves the exact structure of the input dataset, including:
- All data splits (train, validation, test, etc.)
- Complete metadata (transcriptions, speaker information, etc.)
- HuggingFace dataset format compatibility


### 4.2. Log Mel-Frequency Spectrograms

This project includes functionality to convert the hearing loss datasets into Log Mel-Frequency Spectrograms suitable for Whisper and LDL-AURIS model training. The preprocessing pipeline converts audio into 128-dimensional Log-Mel spectrograms required by Whisper Large V3.

#### 4.2.1. Overview

The Log-Mel preprocessing creates training-ready datasets from the hearing loss variants:
- **Normal hearing spectrograms** from `*_normal` datasets
- **Low-frequency hearing loss spectrograms** from `*_lfloss` datasets  
- **High-frequency hearing loss spectrograms** from `*_hfloss` datasets

Each dataset is converted to Log-Mel spectrograms with proper tokenization for Whisper training. For large-scale processing on computing clusters, see [SLURM Processing → Log Mel-Frequency Spectrograms](#log-mel-frequency-spectrograms-1).

#### 4.2.2. Individual Processing Script

Use `scripts/DataSet2LogMel.py` to convert a single hearing loss dataset:

```bash
# Convert normal hearing dataset
python scripts/DataSet2LogMel.py \
    --input_dataset data/CommonVoiceEN_normal/dataset \
    --output_dataset data/CommonVoiceEN_normal_logmel

# Convert low-frequency hearing loss dataset
python scripts/DataSet2LogMel.py \
    --input_dataset data/CommonVoiceEN_lfloss/dataset \
    --output_dataset data/CommonVoiceEN_lfloss_logmel
```

#### Command-line Arguments

- **`--input_dataset`** (required): Path to input CommonVoice dataset folder
- **`--output_dataset`** (required): Path where preprocessed dataset will be saved
- **`--model_size`**: Whisper model size for feature extraction (default: "large-v3")
- **`--num_cpus`**: Number of CPU cores to use (default: all available)
- **`--batch_size`**: Processing batch size (default: 1000)
- **`--writer_batch_size`**: Writer batch size for disk saving (default: 100)
- **`--max_memory_per_worker`**: Maximum memory per worker in GB (default: 4.0)
- **`--language`**: Language for tokenizer (default: "en")
- **`--task`**: Task type for tokenizer: "transcribe" or "translate" (default: "transcribe")
- **`--shuffle_seed`**: Random seed for shuffling (default: 42)
- **`--max_samples`**: Maximum samples per split for testing (default: all)

#### 4.2.3. Batch Processing Script

Use `scripts/DataSet2LogMelBatch.py` to automatically process all three hearing loss variants:

```bash
# Process all datasets with default settings
python scripts/DataSet2LogMelBatch.py

# Process with custom resource allocation
python scripts/DataSet2LogMelBatch.py \
    --cpus-per-task 64 \
    --memory 500G \
    --batch-size 2000
```

#### Command-line Arguments

**SLURM Resource Parameters:**
- **`--cpus-per-task`**: CPU cores per SLURM task (default: 48)
- **`--memory`**: Memory allocation per job (default: "400G")
- **`--time`**: Time limit per job (default: "32:00:00")

**Processing Parameters:**
- **`--model-size`**: Whisper model size (default: "large-v3")
- **`--batch-size`**: Processing batch size (default: 1000)
- **`--max-samples`**: Maximum samples per split for testing (default: all)
- **`--skip-existing`**: Skip datasets with existing output directories

## 5. Whisper

### 5.1. Training of Whisper

The `train_whisper.py` script fine-tunes OpenAI's Whisper model on hearing loss datasets. It supports training on normal audio as well as high-frequency and low-frequency hearing loss simulations.

**Basic Usage:**
```python
python scripts/train_whisper.py \
    --model_name_or_path openai/whisper-large-v3 \
    --train_dataset_path data/CommonVoiceEN_normal_logmel \
    --output_dir results/whisper_finetuned_normal \
    --num_train_epochs 3
```

**Key Parameters:**
- **`--model_name_or_path`**: Base Whisper model (default: "openai/whisper-large-v3")
- **`--train_dataset_path`**: Path to Log-Mel spectrogram dataset
- **`--output_dir`**: Directory to save model checkpoints
- **`--num_train_epochs`**: Number of training epochs (default: 3)
- **`--per_device_train_batch_size`**: Batch size per GPU (default: 8)
- **`--gradient_accumulation_steps`**: Gradient accumulation steps (default: 2)
- **`--learning_rate`**: Learning rate (default: 1e-5)
- **`--warmup_steps`**: Number of warmup steps (default: 500)
- **`--save_steps`**: Save checkpoint every N steps (default: 1000)
- **`--eval_steps`**: Evaluate every N steps (default: 1000)
- **`--logging_steps`**: Log metrics every N steps (default: 25)

**Training on Different Hearing Loss Variants:**
```bash
# Normal hearing
python scripts/train_whisper.py \
    --train_dataset_path data/CommonVoiceEN_normal_logmel \
    --output_dir results/whisper_finetuned_normal

# High-frequency hearing loss
python scripts/train_whisper.py \
    --train_dataset_path data/CommonVoiceEN_hfloss_logmel \
    --output_dir results/whisper_finetuned_hfloss

# Low-frequency hearing loss
python scripts/train_whisper.py \
    --train_dataset_path data/CommonVoiceEN_lfloss_logmel \
    --output_dir results/whisper_finetuned_lfloss
```

**Monitoring Training:**
```bash
# View training progress with TensorBoard
python scripts/tensorboard_visualise_runs.py --logdir results/whisper_finetuned_normal
```

For large-scale training on computing clusters, see [SLURM Processing → Whisper Training](#73-whisper-training).

### 5.2. Prediction with Whisper

The `analyse_with_whisper.py` script performs comprehensive analysis of Whisper model predictions, extracting detailed metrics including token probabilities, embeddings, entropy, and semantic similarity measures.

**Basic Usage:**
```bash
python scripts/analyse_with_whisper.py \
    --input-folder data/MALD \
    --output-path results/whisper_predictions_normal/analysis.json \
    --model-path results/whisper_finetuned_normal/checkpoint-2000 \
    --num-workers 4 \
    --top-k 1000
```

**Key Parameters:**
- **`--input-folder`**: Directory containing audio files (ground truth = filename)
- **`--output-path`**: Path for main JSON output file
- **`--model-path`**: Path to Whisper model checkpoint
- **`--num-workers`**: Number of dataloader workers (default: 20)
- **`--num-threads`**: CPU threads for processing (default: 8)
- **`--num-gpus`**: Number of GPUs to use (default: 1, 0 for CPU only)
- **`--batch-size`**: Batch size for processing (default: 1)
- **`--top-k`**: Number of top predictions to save (default: 1000)

**Output Structure:**

The script generates:
1. **Main JSON file** (`analysis.json`): Contains all results with model metadata
2. **Individual JSON files**: One per audio file with detailed per-token metrics

**Metrics Extracted:**

*Per-Token Metrics:*
- Predicted token probability and rank
- Entropy and semantic density
- Top-k alternative predictions with probabilities and ranks
- Hidden state embeddings (1280-dimensional)
- Ground truth comparison (cosine similarity, correlation)

*Normalized Metrics:*
- Case-insensitive token grouping
- Aggregated probabilities across token variants
- Normalized embeddings (probability-weighted, simple average, most probable)

*Pooled Metrics (across all predicted tokens):*
- Average rank, probability, and entropy
- Pooled hidden states
- Pooled ground truth comparison metrics

**Example Analysis:**
```bash
# Analyze single audio file
python scripts/analyse_with_whisper.py \
    --input-folder test/transcendentalists \
    --output-path results/single_analysis.json \
    --model-path results/whisper_finetuned_normal/checkpoint-2000 \
    --num-workers 0

# Analyze MALD dataset with multiple models
python scripts/analyse_with_whisper.py \
    --input-folder data/MALD \
    --output-path results/whisper_predictions_normal/mald_analysis.json \
    --model-path results/whisper_finetuned_normal/checkpoint-2000
```

For detailed documentation of all metrics, see `README_whisper_metrics.md`. For batch processing on computing clusters, see [SLURM Processing → Whisper Prediction](#74-whisper-prediction).

## 6. LDL-AURIS

*This section will be added in future updates.*

## 7. SLURM Processing

### 7.1. Audio Masking

For processing the full CommonVoice dataset (1.7M+ samples), use the SLURM batch script:

```bash
# Make sure you've configured your working environment first (see Requirements > Working Environment)
# Basic SLURM submission with default settings
sbatch scripts/mask_audio.sbatch
```

The script will automatically:
1. Load your personal working directory from `.env.local`
2. Navigate to your project directory
3. Activate the virtual environment
4. Run the audio masking processing

**Advanced SLURM Options:**
```bash
# Override specific parameters via environment variables
BATCH_SIZE=256 NUM_WORKERS=32 sbatch scripts/mask_audio.sbatch

# Or pass arguments directly to the underlying script
sbatch scripts/mask_audio.sbatch --batch-size 256 --num-workers 32
```

### 7.2. Log Mel-Frequency Spectrograms

For processing large hearing loss datasets to Log-Mel spectrograms using SLURM:

```bash
# Process all three datasets (normal, lfloss, hfloss) automatically
python scripts/DataSet2LogMelBatch.py

# This will submit 3 SLURM jobs:
# - Job 1: CommonVoiceEN_normal → CommonVoiceEN_normal_logmel
# - Job 2: CommonVoiceEN_lfloss → CommonVoiceEN_lfloss_logmel  
# - Job 3: CommonVoiceEN_hfloss → CommonVoiceEN_hfloss_logmel
```

**Individual SLURM Job Submission:**
```bash
# Process a single dataset via SLURM
sbatch scripts/DataSet2LogMel.sbatch \
    --input_dataset data/CommonVoiceEN_normal/dataset \
    --output_dataset data/CommonVoiceEN_normal_logmel
```

The SLURM script will automatically:
1. Load your personal working directory from `.env.local`
2. Navigate to your project directory
3. Activate the virtual environment
4. Run the Log-Mel preprocessing with optimized memory settings

**SLURM Resource Configuration:**
- **Default**: 48 CPU cores, 400GB memory, 32-hour time limit
- **Recommended**: For ~60K samples, jobs typically complete in 8-12 minutes
- **Output**: Each dataset produces ~87GB of Log-Mel spectrograms ready for Whisper training

### 7.3. Whisper Training

For training Whisper models on hearing loss datasets using SLURM, use `train_whisper.sbatch`. The script uses environment variables to specify input and output directories.

**Basic SLURM Submission:**
```bash
# Train on normal hearing dataset
export INPUT_FOLDER="data/CommonVoiceEN_normal_logmel"
export OUTPUT_FOLDER="results/whisper_finetuned_normal"
sbatch scripts/train_whisper.sbatch

# Train on high-frequency hearing loss dataset
export INPUT_FOLDER="data/CommonVoiceEN_hfloss_logmel"
export OUTPUT_FOLDER="results/whisper_finetuned_hfloss"
sbatch scripts/train_whisper.sbatch

# Train on low-frequency hearing loss dataset
export INPUT_FOLDER="data/CommonVoiceEN_lfloss_logmel"
export OUTPUT_FOLDER="results/whisper_finetuned_lfloss"
sbatch scripts/train_whisper.sbatch
```

**SLURM Resource Configuration:**
- **GPUs**: 4× NVIDIA H100 80GB HBM3
- **CPUs**: 24 cores per task
- **Memory**: 200GB
- **Time Limit**: 24 hours
- **Partition**: aisc-batch

**Training Configuration:**
- **Batch size**: 8 per device (32 total with 4 GPUs)
- **Gradient accumulation**: 2 steps
- **Learning rate**: 1e-5 with warmup
- **Evaluation**: Every 1000 steps
- **Checkpoints**: Saved every 1000 steps

The script automatically:
1. Activates the virtual environment
2. Runs training with `train_whisper.py`
3. Logs training metrics to TensorBoard
4. Saves model checkpoints to the output directory

**Monitoring Training:**
```bash
# Check job status
squeue -u $USER

# View training logs
tail -f logs/train_whisper_[JOBID].err

# Launch TensorBoard (after training starts)
python scripts/tensorboard_visualise_runs.py --logdir results/whisper_finetuned_normal
```

### 7.4. Whisper Prediction

For batch prediction and analysis on large datasets using SLURM, use `analyse_with_whisper.sbatch`. The script forwards all command-line arguments to `analyse_with_whisper.py`.

**Single Model Analysis:**
```bash
sbatch scripts/analyse_with_whisper.sbatch \
    --input-folder data/MALD \
    --output-path results/whisper_predictions_normal/mald_analysis.json \
    --model-path results/whisper_finetuned_normal/checkpoint-2000 \
    --num-workers 20 \
    --top-k 1000
```

**Sequential Multi-Model Analysis:**

Use `run_mald_analysis.sh` to analyze with all three models sequentially using SLURM job dependencies:

```bash
./scripts/run_mald_analysis.sh
```

This script will:
1. Submit Job 1: Normal model → `results/whisper_predictions_normal/`
2. Submit Job 2: HF loss model (after Job 1) → `results/whisper_predictions_hfloss/`
3. Submit Job 3: LF loss model (after Job 2) → `results/whisper_predictions_lfloss/`

**SLURM Resource Configuration:**
- **GPUs**: 1× NVIDIA H100 80GB HBM3
- **CPUs**: 24 cores (for 20 dataloader workers)
- **Memory**: 200GB
- **Time Limit**: 24 hours
- **Partition**: aisc-batch

**Processing Performance:**
- **Rate**: ~1.3 samples/second (~78 samples/minute)
- **Dataset**: 26,793 audio files (MALD)
- **Estimated Time**: ~6 hours per model

**Monitoring Progress:**
```bash
# Check job status and dependencies
squeue -u $USER

# Monitor live progress
tail -f logs/whisper_normal_[JOBID].err | grep "Processing batches"

# Check number of processed samples
ls results/whisper_predictions_normal/*.json 2>/dev/null | grep -v "analysis.json" | wc -l
```

**Output Structure:**
- **Main JSON**: `mald_analysis.json` (contains model_path and all results)
- **Individual JSONs**: One file per audio sample with detailed metrics
- **Logs**: Stored in `logs/whisper_[model]_[JOBID].{out,err}`
