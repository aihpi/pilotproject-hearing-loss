#!/bin/bash

# ==============================
# SLURM Job Configuration for Log-Mel Spectrogram Conversion
# ==============================

#SBATCH --job-name=logmel_conversion
#SBATCH --account=aisc                      # AISC account for resource access
#SBATCH --partition=aisc                    # AISC partition
#SBATCH --qos=aisc                          # QOS for AISC
#SBATCH --nodes=1                           # Single node job
#SBATCH --cpus-per-task=48                  # 48 CPU cores (default)  
#SBATCH --mem=400G                          # 400GB memory for log-mel processing
#SBATCH --time=32:00:00                     # 32 hours
#SBATCH --constraint=ARCH:X86               # X86 architecture constraint
#SBATCH --output=logs/logmel_%j.out
#SBATCH --error=logs/logmel_%j.err

# ==============================
# Environment Setup
# ==============================

# Create log directory if it doesn't exist
mkdir -p logs

# Print job information
echo "===== Log-Mel Conversion Job Information ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "CPU Count: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Time Limit: ${SLURM_JOB_TIME_LIMIT}"
echo "Start Time: $(date)"
echo "=============================================="

# Load environment variables if .env.local exists
if [ -f ".env.local" ]; then
    echo "Loading environment variables from .env.local..."
    export $(grep -v '^#' .env.local | xargs)
    if [ -n "$HF_TOKEN" ]; then
        echo "HF_TOKEN loaded: ${HF_TOKEN:0:10}..."
    fi
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
INPUT_DATASET="${INPUT_DATASET:-}"
OUTPUT_DATASET="${OUTPUT_DATASET:-}"
MODEL_SIZE="${MODEL_SIZE:-large-v3}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
WRITER_BATCH_SIZE="${WRITER_BATCH_SIZE:-100}"
MAX_MEMORY_PER_WORKER="${MAX_MEMORY_PER_WORKER:-4.0}"
LANGUAGE="${LANGUAGE:-en}"
TASK="${TASK:-transcribe}"
SHUFFLE_SEED="${SHUFFLE_SEED:-42}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

# Parse command line arguments and override defaults
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dataset|--input_dataset|-i)
            INPUT_DATASET="$2"
            shift 2
            ;;
        --output-dataset|--output_dataset|-o)
            OUTPUT_DATASET="$2"
            shift 2
            ;;
        --model-size|--model_size)
            MODEL_SIZE="$2"
            shift 2
            ;;
        --batch-size|--batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --writer-batch-size|--writer_batch_size)
            WRITER_BATCH_SIZE="$2"
            shift 2
            ;;
        --max-memory-per-worker|--max_memory_per_worker)
            MAX_MEMORY_PER_WORKER="$2"
            shift 2
            ;;
        --language)
            LANGUAGE="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --shuffle-seed|--shuffle_seed)
            SHUFFLE_SEED="$2"
            shift 2
            ;;
        --max-samples|--max_samples)
            MAX_SAMPLES="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Available arguments:"
            echo "  -i, --input-dataset INPUT_DATASET      Input dataset directory (required)"
            echo "  -o, --output-dataset OUTPUT_DATASET    Output dataset directory (required)"
            echo "  --model-size MODEL_SIZE                 Whisper model size (default: large-v3)"
            echo "  --batch-size BATCH_SIZE                 Processing batch size (default: 1000)"
            echo "  --writer-batch-size WRITER_BATCH_SIZE  Writer batch size (default: 100)"
            echo "  --max-memory-per-worker MEMORY          Memory per worker in GB (default: 4.0)"
            echo "  --language LANGUAGE                     Language code (default: en)"
            echo "  --task TASK                             Task type (default: transcribe)"
            echo "  --shuffle-seed SEED                     Random seed (default: 42)"
            echo "  --max-samples MAX_SAMPLES               Max samples per split (for testing)"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_DATASET" ]; then
    echo "ERROR: --input-dataset is required"
    exit 1
fi

if [ -z "$OUTPUT_DATASET" ]; then
    echo "ERROR: --output-dataset is required"
    exit 1
fi

# Validate input dataset exists
if [ ! -d "$INPUT_DATASET" ]; then
    echo "ERROR: Input dataset directory does not exist: $INPUT_DATASET"
    exit 1
fi

# Set number of CPUs (use SLURM allocation or all available)
NUM_CPUS=${SLURM_CPUS_PER_TASK:-$(nproc)}

# ==============================
# Run Log-Mel Conversion
# ==============================

echo "Starting log-mel spectrogram conversion..."
echo "Configuration:"
echo "  Input dataset: ${INPUT_DATASET}"
echo "  Output dataset: ${OUTPUT_DATASET}"
echo "  Model size: ${MODEL_SIZE}"
echo "  Language: ${LANGUAGE}"
echo "  Task: ${TASK}"
echo "  CPU cores: ${NUM_CPUS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Writer batch size: ${WRITER_BATCH_SIZE}"
echo "  Memory per worker: ${MAX_MEMORY_PER_WORKER}GB"
echo "  Shuffle seed: ${SHUFFLE_SEED}"
echo "  Max samples: ${MAX_SAMPLES:-all}"
echo "=============================================="

# Build command arguments
CMD_ARGS=(
    --input_dataset "$INPUT_DATASET"
    --output_dataset "$OUTPUT_DATASET"
    --model_size "$MODEL_SIZE"
    --num_cpus "$NUM_CPUS"
    --batch_size "$BATCH_SIZE"
    --writer_batch_size "$WRITER_BATCH_SIZE"
    --max_memory_per_worker "$MAX_MEMORY_PER_WORKER"
    --language "$LANGUAGE"
    --task "$TASK"
    --shuffle_seed "$SHUFFLE_SEED"
)

# Add max samples if specified
if [ -n "$MAX_SAMPLES" ]; then
    CMD_ARGS+=(--max_samples "$MAX_SAMPLES")
fi

# Display full command
echo "Command: python scripts/DataSet2LogMel.py ${CMD_ARGS[*]}"
echo "=============================================="

# Record start time for performance tracking
START_TIME=$(date +%s)

# Execute the command
python scripts/DataSet2LogMel.py "${CMD_ARGS[@]}"
EXIT_CODE=$?

# Record end time and calculate duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo "=============================================="
echo "Processing completed at $(date)"
echo "Total runtime: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Log-mel conversion completed successfully!"
    
    # Show output directory info
    if [ -d "$OUTPUT_DATASET" ]; then
        echo "Output dataset created:"
        echo "  📁 $OUTPUT_DATASET"
        echo "     Size: $(du -sh "$OUTPUT_DATASET" 2>/dev/null | cut -f1 || echo "N/A")"
        
        # Count splits
        SPLITS_COUNT=$(find "$OUTPUT_DATASET" -maxdepth 1 -type d -name "*.arrow" | wc -l 2>/dev/null || echo "0")
        if [ "$SPLITS_COUNT" -gt 0 ]; then
            echo "     Splits processed: found arrow files"
        fi
        
        # Try to load and show summary
        python -c "
try:
    from datasets import load_from_disk
    dataset = load_from_disk('$OUTPUT_DATASET')
    print(f'     Available splits: {list(dataset.keys())}')
    for split_name, split_data in dataset.items():
        print(f'       {split_name}: {len(split_data):,} samples')
except Exception as e:
    print(f'     Could not load dataset summary: {e}')
" 2>/dev/null || echo "     Dataset saved successfully"
    fi
else
    echo "❌ Log-mel conversion failed with exit code $EXIT_CODE"
    echo "Check the error log for details: logs/logmel_${SLURM_JOB_ID}.err"
fi

echo "=============================================="
echo "Job completed at $(date)"

exit $EXIT_CODE