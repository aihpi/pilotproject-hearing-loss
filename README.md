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

### CommonVoice German Dataset

This project uses the CommonVoice German dataset from HuggingFace. **Note: This dataset requires HuggingFace authentication.**

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

This will download the dataset to `data/CommonVoiceDE/` by default.

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