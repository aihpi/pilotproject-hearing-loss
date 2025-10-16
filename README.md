# KI-based speech recognition as a method for investigating hearing loss

## Requirements

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

## Scripts

All scripts are located in the scripts folder.

## Data

Create a folder where the data will be stored. Because the amount of data is relatively big, data will not be provided by this github repository but has to be downloaded with the scripts below.

```bash
mkdir data
```

### CommonVoice English Dataset

This project uses the CommonVoice English dataset from HuggingFace. **Note: This dataset requires HuggingFace authentication.**

#### Setup Authentication

1. **Create a HuggingFace account** at [huggingface.co](https://huggingface.co)
2. **Get an access token**: Go to Settings → Access Tokens → Create new token (Read access is sufficient)
3. **Accept the dataset terms**: Visit [CommonVoice 16.1](https://huggingface.co/datasets/mozilla-foundation/common_voice_16_1) and accept the terms

#### Download the Dataset

```bash
# Method 1: Using environment variable (recommended)
export HF_TOKEN="your_token_here"
python scripts/download_commonvoice.py

# Method 2: Using command line argument
python scripts/download_commonvoice.py --token "your_token_here"
```

This will download the dataset to `data/CommonVoiceEN/` by default.

#### Download Options

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
python scripts/download_commonvoice.py --version mozilla-foundation/common_voice_17_0
```

## Audio Masking for Hearing Loss Simulation

This project includes functionality to simulate different types of hearing loss by applying frequency-specific attenuation masks to the CommonVoice dataset.

### Overview

The `scripts/mask_audio.py` script processes CommonVoice datasets to create three variants:
- **Normal hearing baseline** (`*_normal`): 10 dB threshold across all frequencies
- **Low-frequency hearing loss** (`*_lfloss`): High attenuation at low frequencies (125 Hz: 100 dB → 8000 Hz: 10 dB)
- **High-frequency hearing loss** (`*_hfloss`): High attenuation at high frequencies (125 Hz: 10 dB → 8000 Hz: 100 dB)

### Usage

```bash
# Basic usage - process the downloaded CommonVoice dataset
python scripts/mask_audio.py

# This creates three new datasets:
# - data/CommonVoiceEN_normal/
# - data/CommonVoiceEN_lfloss/
# - data/CommonVoiceEN_hfloss/
```

### Advanced Options

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

### Processing Parameters

- **`--input-dir`**: Path to the input CommonVoice dataset (default: `data/CommonVoiceEN`)
- **`--output-base`**: Base name for output directories (default: same as input directory)
- **`--sample-rate`**: Target sample rate in Hz (default: 16000, required for Whisper models)
- **`--batch-size`**: Number of audio samples to process per batch (default: 32)
- **`--num-workers`**: Number of CPU cores for parallel processing (default: 4)
- **`--log-level`**: Logging verbosity: DEBUG, INFO, WARNING, ERROR (default: INFO)

### Technical Details

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


## SLURM Processing

### Audio Masking

For processing the full CommonVoice dataset (1.7M+ samples), use the SLURM batch script:

```bash
# Basic SLURM submission with default settings
sbatch scripts/mask_audio.sbatch
```