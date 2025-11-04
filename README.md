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
   
   4.2. [Log Mel-Frequency Spectrograms](#42-log-mel-frequency-spectrograms)

5. [Whisper Model Fine-tuning](#5-whisper-model-fine-tuning)

   5.1. [Training Visualization](#51-training-visualization)

6. [SLURM Processing](#6-slurm-processing)
   
   6.1. [Audio Masking](#61-audio-masking)
   
   6.2. [Log Mel-Frequency Spectrograms](#62-log-mel-frequency-spectrograms)
   
   6.3. [Whisper Training](#63-whisper-training)

## 1. Requirements

### 1.1. Virtual Environment

To ensure that the same requirements are met across different operating systems and machines, it is recommended to create a virtual environment. This can be set up with *UV*.

```bash
which uv || echo "UV not found" # checks the UV installation
```

If UV is not installed, it can be installed as follows.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
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

**Output Structure:**
Each output dataset preserves the exact structure of the input dataset, including:
- All data splits (train, validation, test, etc.)
- Complete metadata (transcriptions, speaker information, etc.)
- HuggingFace dataset format compatibility


### 4.2. Log Mel-Frequency Spectrograms

Before training Whisper models, audio data needs to be converted to log-Mel spectrograms. The `scripts/DataSet2LogMel.py` script processes the masked audio datasets into the format required by Whisper.

#### Usage

```bash
# Process a single dataset
python scripts/DataSet2LogMel.py \
    --input-dir data/CommonVoiceEN_normal \
    --output-dir data/CommonVoiceEN_normal_logmel

# Process with custom parameters
python scripts/DataSet2LogMel.py \
    --input-dir data/CommonVoiceEN_hfloss \
    --output-dir data/CommonVoiceEN_hfloss_logmel \
    --batch-size 64 \
    --num-workers 8
```

#### Processing Parameters

- **`--input-dir`**: Path to the masked audio dataset
- **`--output-dir`**: Output directory for log-Mel spectrograms
- **`--batch-size`**: Number of audio samples to process per batch (default: 32)
- **`--num-workers`**: Number of CPU cores for parallel processing (default: 4)
- **`--sample-rate`**: Target sample rate in Hz (default: 16000)

The script creates log-Mel spectrograms with 128 mel bins (required for Whisper Large-v3) and preserves all dataset splits and metadata.


## 5. Whisper Model Fine-tuning

The training uses pre-processed log-Mel spectrograms and supports parallel training on multiple datasets.

### Usage

```bash
# Fine-tune on normal hearing dataset
python scripts/train_whisper.py \
    --input_folder "data/CommonVoiceEN_normal_logmel" \
    --output_folder "results/whisper_finetuned_normal" \
    --model_size large-v3 \
    --num_gpus 4 \
    --learning_rate 1.5e-5 \
    --max_steps 10000
```

### Training Parameters

The scripts support the following hyperparameters with defaults:

- **`--model_size`**: Whisper model size (default: large-v3)
- **`--num_gpus`**: Number of GPUs to use (default: 4)
- **`--num_cpus`**: Number of CPUs to use (default: 24)
- **`--dataloader_workers`**: Number of data loader workers (default: 20)
- **`--train_batch_size`**: Training batch size per device (default: 1)
- **`--eval_batch_size`**: Evaluation batch size per device (default: 1)
- **`--gradient_accumulation`**: Number of gradient accumulation steps (default: 16)
- **`--learning_rate`**: Learning rate (default: 1.5e-5)
- **`--max_steps`**: Maximum number of training steps (default: 10000)
- **`--warmup_steps`**: Number of warmup steps (default: 1000)
- **`--save_steps`**: Save checkpoint every X steps (default: 500)
- **`--eval_steps`**: Evaluate every X steps (default: 500)
- **`--logging_steps`**: Log training metrics every X steps (default: 50)
- **`--weight_decay`**: Weight decay coefficient (default: 0.05)
- **`--lr_scheduler_type`**: Learning rate scheduler type (default: linear)

### Key Features

- **Language**: English (optimized for CommonVoice English datasets)
- **Training Split**: Uses `train` split for training
- **Evaluation Split**: Uses `test` split for evaluation
- **Model**: Whisper Large-v3 with 128 mel bins support

### Output Structure

Models are saved with the following structure:
```
results/
├── whisper_finetuned_normal/
│   ├── README.md              # Training configuration
│   ├── checkpoint-500/        # Model checkpoints
│   ├── checkpoint-1000/
│   └── runs/                  # TensorBoard logs
├── whisper_finetuned_hfloss/
└── whisper_finetuned_lfloss/
```

### 5.1. Training Visualization

After training, you can analyze and visualize the training progress using TensorBoard logs. The `tensorboard_visualise_runs.py` script extracts metrics from TensorBoard logs and creates comprehensive training visualizations.

#### Usage

```bash
# Basic usage - visualize training results
python scripts/tensorboard_visualise_runs.py results/whisper_finetuned_normal
```

#### Generated Visualizations

The script automatically creates the following plots:

1. **Loss Curves** (`loss_curves.png`): Training and evaluation loss over time
2. **Learning Rate Schedule** (`learning_rate.png`): Learning rate changes during training
3. **Gradient Norm** (`gradient_norm.png`): Gradient magnitude tracking for stability monitoring
4. **Word Error Rate** (`word_error_rate.png`): WER improvement over training steps
5. **Training Overview** (`training_overview.png`): Combined 2x2 subplot with all key metrics
6. **CSV Data** (`csv_data/`): Raw extracted data in CSV format for further analysis
7. **Summary Report** (`training_summary.txt`): Statistical summary of training metrics

#### Parameters

- **`model_path`**: Path to the trained model directory (required)
- **`--output_dir`**: Output directory for plots (default: creates 'tensorboard' folder in model directory)
- **`--format`**: Output format: png, pdf, svg, jpg (default: png)
- **`--dpi`**: Resolution for output plots (default: 300)
- **`--figsize`**: Figure size as 'width,height' in inches (default: 12,8)
- **`--smooth`**: Smoothing factor 0.0-1.0 for noisy curves (default: 0.0)

## 6. SLURM Processing

### 6.1. Audio Masking

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

### 6.3. Log-Mel Spectrograms

Use `scripts/DataSet2LogMel.py` to convert a single hearing loss dataset:

```bash
# Convert normal hearing dataset
python scripts/DataSet2LogMel.py \
    --input_dataset data/CommonVoiceEN_normal/dataset \
    --output_dataset data/CommonVoiceEN_normal_logmel
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

#### Batch Processing Script

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

### 6.4. Whisper Training

For training Whisper models on the hearing loss datasets, use environment variables to specify input and output directories:

```bash
# Train on normal hearing dataset
export INPUT_FOLDER="data/CommonVoiceEN_normal_logmel"
export OUTPUT_FOLDER="results/whisper_finetuned_normal"
sbatch scripts/train_whisper.sbatch

# Train on high frequency hearing loss dataset
export INPUT_FOLDER="data/CommonVoiceEN_hfloss_logmel"
export OUTPUT_FOLDER="results/whisper_finetuned_hfloss"
sbatch scripts/train_whisper.sbatch

# Train on low frequency hearing loss dataset
export INPUT_FOLDER="data/CommonVoiceEN_lfloss_logmel"
export OUTPUT_FOLDER="results/whisper_finetuned_lfloss"
sbatch scripts/train_whisper.sbatch
```
