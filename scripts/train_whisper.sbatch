#!/bin/bash
#SBATCH --job-name=whisper_hearing_loss_training
#SBATCH --output=logs/whisper_hearing_loss_training_%j.out
#SBATCH --error=logs/whisper_hearing_loss_training_%j.err
#SBATCH --partition=aisc
#SBATCH --account=aisc                      # AISC account for H100 access
#SBATCH --qos=aisc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24                  # Increased for 20 dataloader workers
#SBATCH --gres=gpu:4
#SBATCH --mem=1000G
#SBATCH --time=96:00:00

# Activate virtual environment
cd /sc/home/hanno.mueller/pilotproject-hearing-loss
source .venv/bin/activate

# Set CUDA environment variables for 4 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "Starting Whisper large-v3 training for hearing loss research..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Available GPUs: $CUDA_VISIBLE_DEVICES"

# ==============================
# Configuration with Environment Variables
# ==============================

# Required arguments - must be provided via environment variables or command line
if [ -z "$INPUT_FOLDER" ]; then
    echo "Error: INPUT_FOLDER environment variable is required"
    echo "Example: export INPUT_FOLDER=data/CommonVoiceEN_normal_logmel"
    exit 1
fi

if [ -z "$OUTPUT_FOLDER" ]; then
    echo "Error: OUTPUT_FOLDER environment variable is required"
    echo "Example: export OUTPUT_FOLDER=\"Whisper Hearing Loss/normal\""
    exit 1
fi

# Set default values if environment variables are not provided
MODEL_SIZE="${MODEL_SIZE:-large-v3}"
NUM_GPUS="${NUM_GPUS:-4}"
NUM_CPUS="${NUM_CPUS:-24}"
DATALOADER_WORKERS="${DATALOADER_WORKERS:-20}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION="${GRADIENT_ACCUMULATION:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
MAX_STEPS="${MAX_STEPS:-10000}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
SAVE_STEPS="${SAVE_STEPS:-500}"
EVAL_STEPS="${EVAL_STEPS:-500}"
LOGGING_STEPS="${LOGGING_STEPS:-50}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"  # Optional, empty by default

echo "========================================="
echo "Hearing Loss Training Configuration:"
echo "Input Dataset: $INPUT_FOLDER"
echo "Output Directory: $OUTPUT_FOLDER"
echo "Model: $MODEL_SIZE"
echo "GPUs: $NUM_GPUS"
echo "CPUs: $NUM_CPUS"
echo "Dataloader workers: $DATALOADER_WORKERS"
echo "Train batch size per GPU: $TRAIN_BATCH_SIZE"
echo "Eval batch size per GPU: $EVAL_BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION"
echo "Effective batch size: $((TRAIN_BATCH_SIZE * NUM_GPUS * GRADIENT_ACCUMULATION))"
echo "Learning rate: $LEARNING_RATE"
echo "Max steps: $MAX_STEPS"
echo "Warmup steps: $WARMUP_STEPS"
echo "Save steps: $SAVE_STEPS"
echo "Eval steps: $EVAL_STEPS"
echo "Logging steps: $LOGGING_STEPS"
if [ -n "${RESUME_CHECKPOINT}" ]; then
    echo "Resume from checkpoint: $RESUME_CHECKPOINT"
fi
echo "========================================="

# Build command arguments for train_whisper.py
TRAIN_ARGS=(
    --input_folder "$INPUT_FOLDER"
    --output_folder "$OUTPUT_FOLDER"
    --model_size "$MODEL_SIZE"
    --num_gpus "$NUM_GPUS"
    --num_cpus "$NUM_CPUS"
    --dataloader_workers "$DATALOADER_WORKERS"
    --train_batch_size "$TRAIN_BATCH_SIZE"
    --eval_batch_size "$EVAL_BATCH_SIZE"
    --gradient_accumulation "$GRADIENT_ACCUMULATION"
    --learning_rate "$LEARNING_RATE"
    --max_steps "$MAX_STEPS"
    --warmup_steps "$WARMUP_STEPS"
    --save_steps "$SAVE_STEPS"
    --eval_steps "$EVAL_STEPS"
    --logging_steps "$LOGGING_STEPS"
    --max_grad_norm 0.5
    --bf16
    --report_to "tensorboard"
)

# Add resume checkpoint if specified
if [ -n "${RESUME_CHECKPOINT}" ]; then
    TRAIN_ARGS+=(--resume_from_checkpoint "$RESUME_CHECKPOINT")
fi

# Run training with the hearing loss script
python scripts/train_whisper.py "${TRAIN_ARGS[@]}"

if [ $? -eq 0 ]; then
    echo "========================================="
    echo "Hearing loss training completed successfully!"
    echo "Model saved to: $OUTPUT_FOLDER"
    echo "TensorBoard logs: $OUTPUT_FOLDER/runs/"
    echo "========================================="
else
    echo "========================================="
    echo "Hearing loss training failed with exit code $?"
    echo "========================================="
    exit 1
fi