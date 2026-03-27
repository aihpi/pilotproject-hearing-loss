#!/usr/bin/bash
#SBATCH --job-name=mald_smatrix
#SBATCH --output=logs/predict_smatrix_mald_%j.out
#SBATCH --error=logs/predict_smatrix_mald_%j.err
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --constraint=ARCH:X86

PROJECT_ROOT="/sc/home/motoki.saito/pilotproject-hearing-loss"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"

# Required argument
CHECKPOINT_ITER=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint-iter)
            CHECKPOINT_ITER="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            shift
            ;;
    esac
done

if [ -z "$CHECKPOINT_ITER" ]; then
    echo "Error: --checkpoint-iter is required"
    echo "Usage: sbatch predict_smatrix_mald.sbatch --checkpoint-iter 2200000"
    exit 1
fi

echo "========================================"
echo "Predict MALD S-matrix (all conditions)"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Checkpoint iteration: $CHECKPOINT_ITER"
echo "Start time: $(date)"
echo "========================================"

CONDITIONS=("normal" "lfloss" "hfloss")

for CONDITION in "${CONDITIONS[@]}"; do
    echo ""
    echo "--- Condition: $CONDITION ---"
    python "$PROJECT_ROOT/scripts/predict_smatrix_mald.py" \
        --condition "$CONDITION" \
        --checkpoint-iter "$CHECKPOINT_ITER"

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed for condition $CONDITION"
        exit 1
    fi
done

echo ""
echo "========================================"
echo "All conditions completed at: $(date)"
echo "========================================"
