#!/bin/bash

# ==============================
# SLURM Job Configuration for Raw Data Extraction
# ==============================

#SBATCH --job-name=extract_raw_data
#SBATCH --account=aisc                      # AISC account for resource access
#SBATCH --partition=aisc-batch              # AISC batch partition
#SBATCH --qos=aisc                          # QOS for AISC
#SBATCH --nodes=1                           # Single node job
#SBATCH --cpus-per-task=64                  # 64 CPU cores (default num_workers)
#SBATCH --mem=400G                          # 400GB memory with buffer
#SBATCH --time=32:00:00                     # 32 hours with buffer
#SBATCH --constraint=ARCH:X86               # X86 architecture constraint
#SBATCH --output=logs/extract_raw_%j.out
#SBATCH --error=logs/extract_raw_%j.err

# ==============================
# Environment Setup
# ==============================

# Create log directory if it doesn't exist
mkdir -p logs

# Print job information
echo "===== Raw Data Extraction Job Information ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "CPU Count: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Time Limit: ${SLURM_JOB_TIME_LIMIT}"
echo "Start Time: $(date)"
echo "================================================"

# Load environment variables if .env.local exists
if [ -f ".env.local" ]; then
    echo "Loading environment variables from .env.local..."
    export $(grep -v '^#' .env.local | xargs)
fi

# Set default project root if not defined
PROJECT_ROOT="${PROJECT_ROOT:-$(dirname "$(realpath "$0")")/../}"

# Navigate to project directory
echo "Navigating to project directory: $PROJECT_ROOT"
cd "$PROJECT_ROOT"

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
OUTPUT_DIR="${OUTPUT_DIR:-data/CommonVoiceENraw}"
SPLITS="${SPLITS:-train test validation}"
SAMPLING_RATE="${SAMPLING_RATE:-16000}"
CHANNELS="${CHANNELS:-mono}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
NUM_WORKERS="${NUM_WORKERS:-64}"     # Default 64 for SBATCH
BATCH_SIZE="${BATCH_SIZE:-100}"

# Parse command line arguments and override defaults
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
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
        --sampling-rate)
            SAMPLING_RATE="$2"
            shift 2
            ;;
        --channels)
            CHANNELS="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Available arguments:"
            echo "  --input-dir INPUT_DIR         Input dataset directory (default: data/CommonVoiceEN)"
            echo "  --output-dir OUTPUT_DIR       Output directory (default: data/CommonVoiceENraw)"
            echo "  --splits SPLIT1 SPLIT2 ...    Dataset splits to extract (default: train test validation)"
            echo "  --sampling-rate RATE          Target sampling rate in Hz (default: 16000)"
            echo "  --channels mono|stereo        Audio channel configuration (default: mono)"
            echo "  --max-samples N               Max samples per split (default: all)"
            echo "  --num-workers N               Number of worker processes (default: 64)"
            echo "  --batch-size N                Batch size for processing (default: 100)"
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
# Run Raw Data Extraction
# ==============================

echo "Starting raw data extraction..."
echo "Configuration:"
echo "  Input directory: ${INPUT_DIR}"
echo "  Output directory: ${OUTPUT_DIR}"
echo "  Splits: ${SPLITS}"
echo "  Sampling rate: ${SAMPLING_RATE} Hz"
echo "  Channels: ${CHANNELS}"
echo "  Max samples: ${MAX_SAMPLES:-all}"
echo "  Number of workers: ${NUM_WORKERS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "================================================"

# Build command arguments
CMD_ARGS=(
    --input-dir "$INPUT_DIR"
    --output-dir "$OUTPUT_DIR"
    --sampling-rate "$SAMPLING_RATE"
    --channels "$CHANNELS"
    --num-workers "$NUM_WORKERS"
    --batch-size "$BATCH_SIZE"
)

# Add splits
if [ -n "$SPLITS" ]; then
    CMD_ARGS+=(--splits)
    for split in $SPLITS; do
        CMD_ARGS+=("$split")
    done
fi

# Add max samples if specified
if [ -n "$MAX_SAMPLES" ]; then
    CMD_ARGS+=(--max-samples "$MAX_SAMPLES")
fi

# Display full command
echo "Command: python scripts/extract_raw_data.py ${CMD_ARGS[*]}"
echo "================================================"

# Record start time for performance tracking
START_TIME=$(date +%s)

# Execute the command
python scripts/extract_raw_data.py "${CMD_ARGS[@]}"
EXIT_CODE=$?

# Record end time and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "================================================"
echo "Processing completed at $(date)"
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Raw data extraction completed successfully!"
    
    # Show output directory info
    if [ -d "$OUTPUT_DIR" ]; then
        echo "Output directory created:"
        echo "  📁 $OUTPUT_DIR"
        echo "     Size: $(du -sh "$OUTPUT_DIR" 2>/dev/null | cut -f1 || echo "N/A")"
        
        # Count files in each split
        for split in $SPLITS; do
            SPLIT_DIR="$OUTPUT_DIR/$split"
            if [ -d "$SPLIT_DIR" ]; then
                WAV_COUNT=$(find "$SPLIT_DIR" -name "*.wav" | wc -l)
                TXT_COUNT=$(find "$SPLIT_DIR" -name "*.txt" | wc -l)
                echo "     Split '$split':"
                echo "       Audio files: $WAV_COUNT"
                echo "       Text files: $TXT_COUNT"
                echo "       Size: $(du -sh "$SPLIT_DIR" 2>/dev/null | cut -f1 || echo "N/A")"
            fi
        done
    fi
else
    echo "❌ Raw data extraction failed with exit code $EXIT_CODE"
    echo "Check the error log for details: logs/extract_raw_${SLURM_JOB_ID}.err"
fi

echo "================================================"
echo "Job completed at $(date)"

exit $EXIT_CODE
