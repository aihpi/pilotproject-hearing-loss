#!/bin/bash

# ==============================
# SLURM Job Configuration for Audio Masking
# ==============================

#SBATCH --job-name=mask_audio
#SBATCH --account=aisc                      # AISC account for resource access
#SBATCH --partition=aisc                    # AISC partition
#SBATCH --qos=aisc                          # QOS for AISC
#SBATCH --nodes=1                           # Single node job
#SBATCH --cpus-per-task=64                  # 64 CPU cores (default num_workers)
#SBATCH --mem=1024G                         # 1024GB memory for large dataset processing
#SBATCH --time=48:00:00                     # 48 hours (2 days)
#SBATCH --constraint=ARCH:X86               # X86 architecture constraint
#SBATCH --output=logs/mask_audio_%j.out
#SBATCH --error=logs/mask_audio_%j.err

# ==============================
# Environment Setup
# ==============================

# Create log directory if it doesn't exist
mkdir -p logs

# Print job information
echo "===== Audio Masking Job Information ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "CPU Count: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Time Limit: ${SLURM_JOB_TIME_LIMIT}"
echo "Start Time: $(date)"
echo "==========================================="

# Navigate to project directory
cd /sc/home/hanno.mueller/pilotproject-hearing-loss/

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Verify Python environment
echo "Python version: $(python --version)"
echo "Available CPU cores: $(nproc)"

# Set environment variables for multiprocessing optimization
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1  # Ensure Python output is not buffered

# ==============================
# Configuration with Environment Variables and Command Line Arguments
# ==============================

# Set default values if environment variables are not provided
INPUT_DIR="${INPUT_DIR:-data/CommonVoiceEN}"
OUTPUT_BASE="${OUTPUT_BASE:-}"  # Will use input_dir as default if not specified
SAMPLE_RATE="${SAMPLE_RATE:-16000}"
BATCH_SIZE="${BATCH_SIZE:-128}"      # Default 128 for SBATCH
NUM_WORKERS="${NUM_WORKERS:-64}"     # Default 64 for SBATCH
LOG_LEVEL="${LOG_LEVEL:-INFO}"
SPLITS="${SPLITS:-}"                 # Optional: specific splits to process
DRY_RUN="${DRY_RUN:-false}"         # Optional: dry run mode

# Parse command line arguments and override defaults
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --sample-rate)
            SAMPLE_RATE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --log-level)
            LOG_LEVEL="$2"
            shift 2
            ;;
        --splits)
            # Handle multiple splits
            SPLITS=""
            shift
            while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
                SPLITS="$SPLITS $1"
                shift
            done
            ;;
        --dry-run)
            DRY_RUN="true"
            shift
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Available arguments:"
            echo "  --input-dir INPUT_DIR       Input dataset directory (default: data/CommonVoiceEN)"
            echo "  --output-base OUTPUT_BASE   Base name for output directories (default: same as input)"
            echo "  --sample-rate SAMPLE_RATE   Target sample rate in Hz (default: 16000)"
            echo "  --batch-size BATCH_SIZE     Number of samples per batch (default: 128)"
            echo "  --num-workers NUM_WORKERS   Number of worker processes (default: 64)"
            echo "  --log-level LOG_LEVEL       Logging level (default: INFO)"
            echo "  --splits SPLIT1 SPLIT2      Specific splits to process (default: all)"
            echo "  --dry-run                   Show what would be processed without processing"
            exit 1
            ;;
    esac
done

# Validate num_workers doesn't exceed available CPUs
MAX_WORKERS=${SLURM_CPUS_PER_TASK:-$(nproc)}
if [ "$NUM_WORKERS" -gt "$MAX_WORKERS" ]; then
    echo "WARNING: NUM_WORKERS ($NUM_WORKERS) exceeds available CPUs ($MAX_WORKERS)"
    echo "Setting NUM_WORKERS to $MAX_WORKERS"
    NUM_WORKERS=$MAX_WORKERS
fi

# ==============================
# Run Audio Masking
# ==============================

echo "Starting audio masking processing..."
echo "Configuration:"
echo "  Input directory: ${INPUT_DIR}"
echo "  Output base: ${OUTPUT_BASE:-${INPUT_DIR} (default)}"
echo "  Sample rate: ${SAMPLE_RATE} Hz"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Number of workers: ${NUM_WORKERS}"
echo "  Log level: ${LOG_LEVEL}"
echo "  Splits: ${SPLITS:-all splits}"
echo "  Dry run: ${DRY_RUN}"
echo "==========================================="

# Build command arguments
CMD_ARGS=(
    --input-dir "$INPUT_DIR"
    --sample-rate "$SAMPLE_RATE"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --log-level "$LOG_LEVEL"
)

# Add output base if specified
if [ -n "$OUTPUT_BASE" ]; then
    CMD_ARGS+=(--output-base "$OUTPUT_BASE")
fi

# Add splits if specified
if [ -n "$SPLITS" ]; then
    CMD_ARGS+=(--splits)
    for split in $SPLITS; do
        CMD_ARGS+=("$split")
    done
fi

# Add dry run if specified
if [ "$DRY_RUN" = "true" ]; then
    CMD_ARGS+=(--dry-run)
fi

# Display full command
echo "Command: python scripts/mask_audio.py ${CMD_ARGS[*]}"
echo "==========================================="

# Record start time for performance tracking
START_TIME=$(date +%s)

# Execute the command
python scripts/mask_audio.py "${CMD_ARGS[@]}"
EXIT_CODE=$?

# Record end time and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "==========================================="
echo "Processing completed at $(date)"
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Audio masking completed successfully!"
    
    # Show output directory structure if not dry run
    if [ "$DRY_RUN" != "true" ]; then
        OUTPUT_PATH="${OUTPUT_BASE:-${INPUT_DIR}}"
        echo "Output datasets created:"
        for suffix in "_normal" "_lfloss" "_hfloss"; do
            DATASET_DIR="${OUTPUT_PATH}${suffix}"
            if [ -d "$DATASET_DIR" ]; then
                echo "  📁 $DATASET_DIR"
                echo "     Size: $(du -sh "$DATASET_DIR" 2>/dev/null | cut -f1 || echo "N/A")"
                if [ -d "$DATASET_DIR/dataset" ]; then
                    SPLITS_COUNT=$(find "$DATASET_DIR/dataset" -maxdepth 1 -type d | wc -l)
                    echo "     Splits: $((SPLITS_COUNT - 1))" # Subtract 1 for the dataset directory itself
                fi
            fi
        done
    fi
else
    echo "❌ Audio masking failed with exit code $EXIT_CODE"
    echo "Check the error log for details: logs/mask_audio_${SLURM_JOB_ID}.err"
fi

echo "==========================================="
echo "Job completed at $(date)"

exit $EXIT_CODE