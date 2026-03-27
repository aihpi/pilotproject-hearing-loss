#!/usr/bin/bash
#SBATCH --job-name=mald_eval
#SBATCH --output=logs/evaluate_mald_smatrix_%j.out
#SBATCH --error=logs/evaluate_mald_smatrix_%j.err
#SBATCH --partition=aisc-batch
#SBATCH --account=aisc
#SBATCH --qos=aisc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --constraint=ARCH:X86

PROJECT_ROOT="/sc/home/motoki.saito/pilotproject-hearing-loss"
cd "$PROJECT_ROOT"
source "$PROJECT_ROOT/.venv/bin/activate"

echo "========================================"
echo "Evaluate MALD predicted S-matrices"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "========================================"

python "$PROJECT_ROOT/scripts/evaluate_mald_smatrix.py" "$@"

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
