#!/usr/bin/bash
#SBATCH --job-name=train_fmatrix
#SBATCH --output=logs/train_fmatrix_%A_%a.out
#SBATCH --error=logs/train_fmatrix_%A_%a.err
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
#SBATCH --time=168:00:00
#SBATCH --constraint=ARCH:X86
#SBATCH --gres=gpu:h100:1  # Request 1 H100 GPU for PyTorch acceleration (optimized for speed)
#SBATCH --array=0-2  # Array job for 3 conditions (normal, lfloss, hfloss)

# ==============================
# Train F-Matrix and G-Matrix via Incremental Learning
# ==============================
# This script trains F-matrices and G-matrices for three hearing loss conditions using
# incremental learning on audio spectrograms mapped to fastText embeddings.
#
# F-matrix: form -> semantics (comprehension)
# G-matrix: semantics -> form (production)
#
# Usage:
#   # Submit array job for all three conditions
#   sbatch scripts/train_fmatrix_incremental.sbatch
#
#   # Or submit for a specific condition
#   sbatch scripts/train_fmatrix_incremental.sbatch --condition normal
#
# Arguments:
#   --condition       Hearing loss condition (normal, lfloss, or hfloss)
#                     Overrides array job indexing if specified
#   --config          Path to configuration YAML (default: configs/fmatrix_training_config.yaml)
#   --split           Data split to use (default: train)
#   --limit           Limit number of files for testing (optional)
#   --resume-from-f   Resume F-matrix from checkpoint (optional)
#   --resume-from-g   Resume G-matrix from checkpoint (optional)
#   --auto-resume     Automatically find and resume from latest checkpoint
#
# Output:
#   - F-matrix and G-matrix checkpoints saved to results/fmatrix_incremental/checkpoints/{condition}/
#   - Metrics saved to results/fmatrix_incremental/metrics/{condition}/
#
# Estimated Runtime:
#   - With H100 GPU (PyTorch backend): ~6-12 hours per condition for 6.3M samples
#   - With generic GPU: ~8-18 hours per condition
#   - With CPU (NumPy backend): ~3-7 days per condition
# ==============================

# Navigate to project root (parent of scripts directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# Activate virtual environment
source .venv/bin/activate

echo "========================================"
echo "F-Matrix Incremental Training Job"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Start time: $(date)"
echo "Arguments: $@"
echo "========================================"

# Map array task ID to condition (if using array job)
if [ -n "$SLURM_ARRAY_TASK_ID" ]; then
    case $SLURM_ARRAY_TASK_ID in
        0)
            CONDITION="normal"
            ;;
        1)
            CONDITION="lfloss"
            ;;
        2)
            CONDITION="hfloss"
            ;;
        *)
            echo "Error: Invalid array task ID: $SLURM_ARRAY_TASK_ID"
            exit 1
            ;;
    esac
    echo "Array job: Processing condition '$CONDITION' (task $SLURM_ARRAY_TASK_ID)"
fi

# Parse command-line arguments (override array job condition if specified)
CONFIG="configs/fmatrix_training_config.yaml"
SPLIT="train"
LIMIT=""
RESUME_FROM_F=""
RESUME_FROM_G=""
AUTO_RESUME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --condition)
            CONDITION="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --limit)
            LIMIT="--limit $2"
            shift 2
            ;;
        --resume-from-f)
            RESUME_FROM_F="--resume-from-f $2"
            shift 2
            ;;
        --resume-from-g)
            RESUME_FROM_G="--resume-from-g $2"
            shift 2
            ;;
        --auto-resume)
            AUTO_RESUME="--auto-resume"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            shift
            ;;
    esac
done

# Check if condition is set
if [ -z "$CONDITION" ]; then
    echo "Error: --condition must be specified or array job must be used"
    exit 1
fi

echo "Training configuration:"
echo "  Condition: $CONDITION"
echo "  Config: $CONFIG"
echo "  Split: $SPLIT"
if [ -n "$LIMIT" ]; then
    echo "  Limit: $LIMIT (TESTING MODE)"
fi
if [ -n "$RESUME_FROM_F" ]; then
    echo "  Resume F-matrix from: $RESUME_FROM_F"
fi
if [ -n "$RESUME_FROM_G" ]; then
    echo "  Resume G-matrix from: $RESUME_FROM_G"
fi
if [ -n "$AUTO_RESUME" ]; then
    echo "  Auto-resume: ENABLED"
fi
echo "========================================"
echo ""

# Run the training script
python scripts/train_fmatrix_incremental.py \
    --condition "$CONDITION" \
    --config "$CONFIG" \
    --split "$SPLIT" \
    $LIMIT \
    $RESUME_FROM_F \
    $RESUME_FROM_G \
    $AUTO_RESUME

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
