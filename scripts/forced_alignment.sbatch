#!/bin/bash
#SBATCH --job-name=forced_alignment
#SBATCH --output=logs/forced_alignment_%j.out
#SBATCH --error=logs/forced_alignment_%j.err
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --constraint=ARCH:X86

# ==============================
# Forced Alignment with torchaudio MMS_FA
# ==============================
# This script performs GPU-accelerated forced alignment on audio files
# to generate word-level timing information stored as JSON.
#
# Supports batching and multi-GPU processing for better throughput.
#
# Usage:
#   sbatch scripts/forced_alignment.sbatch [OPTIONS]
#
# Examples:
#   # Process all splits with defaults (single GPU, batch size 16)
#   sbatch scripts/forced_alignment.sbatch
#
#   # Process only test split
#   sbatch scripts/forced_alignment.sbatch --splits test
#
#   # Process with skip-existing to resume interrupted job
#   sbatch scripts/forced_alignment.sbatch --skip-existing
#
#   # Use 4 GPUs with batch size 32 for faster processing
#   sbatch --gres=gpu:4 scripts/forced_alignment.sbatch --num-gpus 4 --batch-size 32
#
#   # Limit to 1000 files per split for testing
#   sbatch scripts/forced_alignment.sbatch --max-files 1000
#
# Arguments (same as forced_alignment.py):
#   --input-dir      Input directory with wav/txt pairs (default: data/CommonVoiceENraw)
#   --output-dir     Output directory for JSON files (default: data/CommonVoiceENJSON)
#   --splits         Splits to process (default: train test validation)
#   --device         Device to use (default: cuda) - ignored if --num-gpus > 1
#   --num-gpus       Number of GPUs to use (default: 1)
#   --batch-size     Number of files per batch (default: 16)
#   --skip-existing  Skip files that already have JSON output
#   --max-files      Maximum files per split (default: all)
#   --log-level      Logging level (default: INFO)
# ==============================

cd /sc/home/hanno.mueller/pilotproject-hearing-loss

# Activate virtual environment
source .venv/bin/activate

echo "========================================"
echo "Forced Alignment Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "Arguments: $@"
echo "========================================"

# Run the forced alignment script with all passed arguments
# Default to cuda device if not specified
python scripts/forced_alignment.py "$@"

EXIT_CODE=$?

echo "========================================"
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
