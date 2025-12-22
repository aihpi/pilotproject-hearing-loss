#!/bin/bash
#SBATCH --job-name=extract_words
#SBATCH --output=logs/extract_word_audio_%j.out
#SBATCH --error=logs/extract_word_audio_%j.err
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --constraint=ARCH:X86

# ==============================
# Extract Word Audio from Aligned Data
# ==============================
# This script extracts individual word audio segments from full audio files
# using word boundary information from forced alignment JSON files.
#
# Usage:
#   sbatch scripts/extract_word_audio.sbatch [OPTIONS]
#
# Examples:
#   # Process all splits with defaults
#   sbatch scripts/extract_word_audio.sbatch
#
#   # Process only train split
#   sbatch scripts/extract_word_audio.sbatch --splits train
#
#   # Use more workers and larger batch size
#   sbatch scripts/extract_word_audio.sbatch --nworkers 32 --batch-size 200
#
#   # Custom input/output directories
#   sbatch scripts/extract_word_audio.sbatch --input-json data/CustomJSON --input-audio data/CustomAudio --output-audio data/CustomWords
#
# Arguments (same as extract_word_audio.py):
#   --input-json     Directory containing word alignment JSON files (default: data/CommonVoiceENJSON)
#   --input-audio    Directory containing source audio files (default: data/CommonVoiceENraw)
#   --output-audio   Directory to write extracted word audio files (default: data/CommonVoiceENWords)
#   --nworkers       Number of parallel workers (default: 24)
#   --batch-size     Number of files to process per batch (default: 100)
#   --splits         Splits to process (default: train test validation)
# ==============================

cd /sc/home/hanno.mueller/pilotproject-hearing-loss

# Activate virtual environment
source .venv/bin/activate

echo "========================================"
echo "Extract Word Audio Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Start time: $(date)"
echo "Arguments: $@"
echo "========================================"

# Run the extract word audio script with all passed arguments
python scripts/extract_word_audio.py "$@"

EXIT_CODE=$?

echo "========================================"
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
