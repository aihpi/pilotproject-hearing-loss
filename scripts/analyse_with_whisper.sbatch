#!/bin/bash
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc                      # AISC account for H100 access
#SBATCH --qos=aisc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24                  # For 20 dataloader workers
#SBATCH --gres=gpu:1                        # Single GPU script
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --constraint=ARCH:X86

# This script runs Whisper analysis on audio datasets
# It forwards all command-line arguments to analyse_with_whisper.py

# Activate virtual environment
source .venv/bin/activate

# Run the analysis script with all passed arguments
python scripts/analyse_with_whisper.py "$@"
