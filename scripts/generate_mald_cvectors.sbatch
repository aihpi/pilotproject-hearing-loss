#!/usr/bin/bash
#SBATCH --job-name=mald_cvectors
#SBATCH --output=logs/generate_mald_cvectors_%j.out
#SBATCH --error=logs/generate_mald_cvectors_%j.err
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

echo "========================================"
echo "Generate MALD c-vectors"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Start time: $(date)"
echo "========================================"

python "$PROJECT_ROOT/scripts/generate_mald_cvectors.py" "$@"

EXIT_CODE=$?

echo ""
echo "========================================"
echo "Job completed at: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================"

exit $EXIT_CODE
