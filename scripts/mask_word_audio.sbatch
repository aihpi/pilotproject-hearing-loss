#!/bin/bash
#SBATCH --job-name=mask_words
#SBATCH --output=logs/mask_word_audio_%j.out
#SBATCH --error=logs/mask_word_audio_%j.err
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --constraint=ARCH:X86

# ==============================
# Apply Hearing Loss Masks to Word Audio
# ==============================
# This script applies hearing loss masks to individual word audio files,
# creating three versions: normal (copy), low-frequency loss, and high-frequency loss.
#
# Output remains as WAV files (no mel spectrogram conversion).
#
# Usage:
#   sbatch scripts/mask_word_audio.sbatch [OPTIONS]
#
# Examples:
#   # Process all splits and conditions with defaults
#   sbatch scripts/mask_word_audio.sbatch
#
#   # Process only train split
#   sbatch scripts/mask_word_audio.sbatch --splits train
#
#   # Process only specific conditions
#   sbatch scripts/mask_word_audio.sbatch --conditions hfloss lfloss
#
#   # Use more workers and larger batch size
#   sbatch scripts/mask_word_audio.sbatch --nworkers 32 --batch-size 200
#
#   # Custom input/output directories
#   sbatch scripts/mask_word_audio.sbatch --input-dir data/CustomWords --output-base data/CustomWords
#
# Arguments (same as mask_word_audio.py):
#   --input-dir      Directory containing input word audio files (default: data/CommonVoiceENWords)
#   --output-base    Base name for output directories (default: data/CommonVoiceENWords)
#                    Creates: {output-base}_normal, {output-base}_lfloss, {output-base}_hfloss
#   --nworkers       Number of parallel workers (default: 24)
#   --batch-size     Number of files to process per batch (default: 100)
#   --splits         Splits to process (default: train test validation)
#   --conditions     Hearing loss conditions to generate (default: normal lfloss hfloss)
# ==============================

cd /sc/home/hanno.mueller/pilotproject-hearing-loss

# Activate virtual environment
source .venv/bin/activate

# Set environment variables for multiprocessing optimization
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

echo "========================================"
echo "Mask Word Audio Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "Arguments: $@"
echo "========================================"

# Run the mask word audio script with all passed arguments
python scripts/mask_word_audio.py "$@"

EXIT_CODE=$?

echo "========================================"
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
