#!/bin/bash
#SBATCH --job-name=20260126_cv24_mask_test
#SBATCH --account=aisc
#SBATCH --partition=aisc-batch
#SBATCH --exclude=ga03
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/sc/home/david.goll/projects/pilotproject-hearing-loss/logs/20260126_cv24_mask_test_%j.out
#SBATCH --error=/sc/home/david.goll/projects/pilotproject-hearing-loss/logs/20260126_cv24_mask_test_%j.err

# ===================================================================================================
# Test CV24 Migration: Apply hearing loss masks to CommonVoice 24.0 dev split
# ===================================================================================================
#
# This script tests the CV24 data loader migration by processing the dev split (~16k samples).
# It creates three output versions:
#   - normal (unmodified audio, resampled to 16kHz)
#   - hfloss (high-frequency hearing loss simulation)
#   - lfloss (low-frequency hearing loss simulation)
#
# Usage:
#   sbatch scripts/slurm/test_cv24_mask_audio.sh
#
# Monitor job:
#   squeue --me
#   tail -f logs/20260126_cv24_mask_test_<jobid>.out
#
# Cancel job:
#   scancel <jobid>
#
# Expected runtime: ~30-45 minutes for dev split (16,402 samples)
# Expected output: 3 directories in data/CommonVoiceEN_v24_test/
#
# ===================================================================================================

# Print job info
echo "========================================"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Start time: $(date)"
echo "========================================"

# Navigate to project root
cd /sc/home/david.goll/projects/pilotproject-hearing-loss

echo ""
echo "Working directory: $(pwd)"
echo ""

# Verify input data exists
echo "Checking input data..."
INPUT_DIR="data/CommonVoiceEN_v24"
if [ -L "$INPUT_DIR" ]; then
    echo "Input symlink: $INPUT_DIR -> $(readlink -f $INPUT_DIR)"
elif [ -d "$INPUT_DIR" ]; then
    echo "Input directory: $INPUT_DIR"
else
    echo "ERROR: Input directory not found: $INPUT_DIR"
    exit 1
fi

echo "TSV files:"
ls -la $INPUT_DIR/*.tsv 2>/dev/null || echo "No TSV files found"
echo ""
echo "Clips directory exists: $([ -d $INPUT_DIR/clips ] && echo 'YES' || echo 'NO')"
echo ""

# Check Python environment
echo "Checking Python environment..."
uv run python -c "
import sys
print(f'Python: {sys.version}')
import librosa
print(f'librosa: {librosa.__version__}')
import numpy as np
print(f'numpy: {np.__version__}')
"
echo ""

# Create output directory for test results
OUTPUT_BASE="data/CommonVoiceEN_v24_test"
echo "Output will be written to: $OUTPUT_BASE"
echo ""

echo "========================================"
echo "Starting mask_audio.py on dev split..."
echo "========================================"
echo ""

# Run the mask_audio script on dev split only
uv run python scripts/mask_audio.py \
    --input-dir "$INPUT_DIR" \
    --output-base "$OUTPUT_BASE" \
    --splits dev \
    --batch-size 32 \
    --num-workers 4 \
    --log-level INFO

# Capture exit code
EXIT_CODE=$?

echo ""
echo "========================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Processing completed successfully!"
    echo ""
    echo "Output directories created:"
    ls -d ${OUTPUT_BASE}_*/ 2>/dev/null || echo "No output directories found"
    echo ""
    echo "Verifying sample counts:"
    for profile in normal lfloss hfloss; do
        dir="${OUTPUT_BASE}_${profile}/dataset/dev"
        if [ -d "$dir" ]; then
            # Count arrow files and show their sizes
            arrow_count=$(ls -1 "$dir"/*.arrow 2>/dev/null | wc -l)
            total_size=$(du -sh "$dir" 2>/dev/null | cut -f1)
            echo "  $profile: $arrow_count arrow files, $total_size total"
        else
            echo "  $profile: NOT FOUND"
        fi
    done
else
    echo "Processing FAILED with exit code: $EXIT_CODE"
fi
echo ""
echo "End time: $(date)"
echo "========================================"
