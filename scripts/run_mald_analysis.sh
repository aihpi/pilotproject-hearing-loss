#!/bin/bash

# Sequential MALD Analysis Script
# Runs Whisper analysis on MALD dataset with three different model checkpoints
# Uses SLURM job dependencies to run sequentially

INPUT_FOLDER="data/MALD"

# Job 1: Normal model
echo "Submitting job for normal model..."
JOB1=$(sbatch \
    --job-name=whisper_normal \
    --output=logs/whisper_normal_%j.out \
    --error=logs/whisper_normal_%j.err \
    scripts/analyse_with_whisper.sbatch \
    --input-folder "$INPUT_FOLDER" \
    --output-path results/whisper_predictions_normal/mald_analysis.json \
    --model-path results/whisper_finetuned_normal/checkpoint-2000 \
    --num-workers 20 \
    --num-threads 24 \
    --top-k 1000 | awk '{print $4}')

echo "Job 1 (normal): $JOB1"

# Job 2: High-frequency loss model (depends on Job 1)
echo "Submitting job for hfloss model..."
JOB2=$(sbatch \
    --job-name=whisper_hfloss \
    --output=logs/whisper_hfloss_%j.out \
    --error=logs/whisper_hfloss_%j.err \
    --dependency=afterok:$JOB1 \
    scripts/analyse_with_whisper.sbatch \
    --input-folder "$INPUT_FOLDER" \
    --output-path results/whisper_predictions_hfloss/mald_analysis.json \
    --model-path results/whisper_finetuned_hfloss/checkpoint-2000 \
    --num-workers 20 \
    --num-threads 24 \
    --top-k 1000 | awk '{print $4}')

echo "Job 2 (hfloss): $JOB2 (depends on $JOB1)"

# Job 3: Low-frequency loss model (depends on Job 2)
echo "Submitting job for lfloss model..."
JOB3=$(sbatch \
    --job-name=whisper_lfloss \
    --output=logs/whisper_lfloss_%j.out \
    --error=logs/whisper_lfloss_%j.err \
    --dependency=afterok:$JOB2 \
    scripts/analyse_with_whisper.sbatch \
    --input-folder "$INPUT_FOLDER" \
    --output-path results/whisper_predictions_lfloss/mald_analysis.json \
    --model-path results/whisper_finetuned_lfloss/checkpoint-2000 \
    --num-workers 20 \
    --num-threads 24 \
    --top-k 1000 | awk '{print $4}')

echo "Job 3 (lfloss): $JOB3 (depends on $JOB2)"

echo ""
echo "All jobs submitted!"
echo "Summary:"
echo "  Job 1 (normal):  $JOB1"
echo "  Job 2 (hfloss):  $JOB2 (after $JOB1)"
echo "  Job 3 (lfloss):  $JOB3 (after $JOB2)"
echo ""
echo "Monitor with: squeue -u $USER"
echo ""
echo "Output directories:"
echo "  - results/whisper_predictions_normal/"
echo "  - results/whisper_predictions_hfloss/"
echo "  - results/whisper_predictions_lfloss/"
