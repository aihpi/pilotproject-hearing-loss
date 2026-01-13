#!/bin/bash
#SBATCH --job-name=apply_all_masks
#SBATCH --output=logs/apply_all_masks_%j.out
#SBATCH --error=logs/apply_all_masks_%j.err
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --constraint=ARCH:X86

# ==============================
# Apply All Hearing Loss Masks to Audio Files
# ==============================
# This script applies all three hearing loss masks to CommonVoiceENWords audio files:
# - Normal hearing: 10 dB uniform attenuation
# - Low-frequency loss: 100→10 dB gradient (severe loss at low frequencies)
# - High-frequency loss: 10→100 dB gradient (severe loss at high frequencies)
#
# All outputs remain as WAV files (no spectrogram conversion).
#
# Usage:
#   sbatch scripts/apply_all_masks.sbatch [OPTIONS]
#
# Examples:
#   # Process all splits with default settings
#   sbatch scripts/apply_all_masks.sbatch
#
#   # Process only validation split
#   sbatch scripts/apply_all_masks.sbatch --splits validation
#
#   # Process with fewer workers
#   sbatch scripts/apply_all_masks.sbatch --nworkers 12
#
#   # Custom input/output directories
#   sbatch scripts/apply_all_masks.sbatch \
#       --input-dir /path/to/input \
#       --output-base /path/to/output
#
#   # Generate only specific conditions
#   sbatch scripts/apply_all_masks.sbatch --conditions normal lfloss
#
# Arguments (same as apply_all_masks.py):
#   --input-dir         Input directory containing WAV files
#                       (default: /sc/projects/sci-aisc/hearingloss/data_processed/CommonVoiceENWords)
#   --output-base       Base directory for output
#                       (default: /sc/projects/sci-aisc/hearingloss/data_processed)
#   --nworkers          Number of parallel workers (default: 48)
#   --batch-size        Number of files per batch (default: 100)
#   --splits            Splits to process (default: train test validation)
#   --conditions        Conditions to generate (default: normal lfloss hfloss)
#
# Output directories (under --output-base):
#   - normal_raw/       Normal hearing (10 dB uniform)
#   - lfloss_raw/       Low-frequency loss (100→10 dB)
#   - hfloss_raw/       High-frequency loss (10→100 dB)
#
# Hearing Loss Profiles:
#   Frequency points: 125, 250, 500, 1000, 2000, 4000, 8000 Hz
#
#   Normal hearing:
#     Thresholds: 10 dB uniform across all frequencies
#
#   Low-frequency loss:
#     Thresholds: 100 dB (125 Hz) → 10 dB (8000 Hz) linear gradient
#
#   High-frequency loss:
#     Thresholds: 10 dB (125 Hz) → 100 dB (8000 Hz) linear gradient
#
# Processing Details:
#   - STFT with window size 2048, hop length 512
#   - Frequency-dependent attenuation via spectral masking
#   - Inverse STFT for audio reconstruction
#   - Normalization to prevent clipping
#
# Estimated Processing Time:
#   - ~148,175 files total in CommonVoiceENWords
#   - With 48 workers: approximately 1-1.5 hours for all splits
#   - Single split (validation): approximately 15-25 minutes
#
# Storage Requirements:
#   - Approximately same size as input (~2.7 GB per split)
#   - Total for all 3 conditions: ~24 GB
# ==============================

# Navigate to project root (parent of scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Activate virtual environment
source .venv/bin/activate

echo "========================================"
echo "Apply All Hearing Loss Masks Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "Arguments: $@"
echo "========================================"

# Run the masking script with all passed arguments
python scripts/apply_all_masks.py "$@"

EXIT_CODE=$?

echo "========================================"
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
