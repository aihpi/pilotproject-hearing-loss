#!/bin/bash
#SBATCH --job-name=gen_spectrograms
#SBATCH --output=logs/generate_spectrograms_%j.out
#SBATCH --error=logs/generate_spectrograms_%j.err
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --constraint=ARCH:X86

# ==============================
# Generate Log-Mel Spectrograms from Audio Files
# ==============================
# This script generates log-mel spectrograms from CommonVoiceENWords audio files.
# All spectrograms are padded to uniform shape (128, 391) for machine learning.
#
# Usage:
#   sbatch scripts/generate_spectrograms.sbatch [OPTIONS]
#
# Examples:
#   # Process entire validation set with defaults
#   sbatch scripts/generate_spectrograms.sbatch
#
#   # Test with only 1000 files
#   sbatch scripts/generate_spectrograms.sbatch --limit 1000
#
#   # Use fewer workers for smaller jobs
#   sbatch scripts/generate_spectrograms.sbatch --nworkers 8 --limit 1000
#
#   # Auto-detect maximum duration from dataset
#   sbatch scripts/generate_spectrograms.sbatch --auto-detect-max
#
#   # Manually specify max time frames
#   sbatch scripts/generate_spectrograms.sbatch --max-time-frames 500
#
#   # Custom input/output directories
#   sbatch scripts/generate_spectrograms.sbatch \
#       --input-dir /path/to/audio \
#       --output-dir /path/to/spectrograms \
#       --metadata /path/to/metadata.csv
#
# Arguments (same as generate_spectrograms.py):
#   --input-dir         Directory containing WAV files (default: /sc/projects/sci-aisc/hearingloss/data_processed/CommonVoiceENWords/validation)
#   --output-dir        Directory to save .npy spectrogram files (default: /sc/projects/sci-aisc/hearingloss/data_processed/CommonVoiceENWords_spectrograms/validation)
#   --metadata          Path to save metadata CSV file (default: data/spectrograms_metadata.csv)
#   --limit             Maximum number of files to process for testing (default: None = process all)
#   --nworkers          Number of parallel workers (default: 24)
#   --auto-detect-max   Auto-detect maximum duration from input files (default: uses 391 frames)
#   --max-time-frames   Manually specify max time frames (overrides default and auto-detect)
#
# Spectrogram Parameters:
#   - Sampling rate: 16 kHz
#   - Mel bands: 128
#   - Window size: 25 ms (n_fft=400)
#   - Hop length: 10 ms (hop_length=160)
#   - Output shape: (128, 391)
#
# Estimated Processing Time:
#   - ~7-10 minutes for 148,175 files with 24 workers
#   - Single-threaded: ~2 hours 45 minutes (~15 files/second)
#   - With 24 workers: ~24x speedup
#
# Storage Requirements:
#   - ~28.4 GB for full validation set
#   - ~196 KB per .npy file
# ==============================

cd /sc/home/motoki.saito/pilotproject-hearing-loss

# Activate virtual environment
source .venv/bin/activate

echo "========================================"
echo "Generate Spectrograms Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "Arguments: $@"
echo "========================================"

# Run the spectrogram generation script with all passed arguments
python scripts/generate_spectrograms.py "$@"

EXIT_CODE=$?

echo "========================================"
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
