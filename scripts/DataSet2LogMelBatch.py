#!/usr/bin/env python3
"""
Batch script to submit SLURM jobs for converting all CommonVoice hearing loss datasets 
to Log-Mel Spectrograms. This script submits jobs for normal, lfloss, and hfloss variants.
"""

import subprocess
import sys
from pathlib import Path
import argparse


def submit_logmel_job(input_dataset: str, output_dataset: str, dataset_name: str, **slurm_kwargs):
    """Submit a SLURM job for log-mel conversion using DataSet2LogMel.sbatch."""
    
    # Build SLURM command
    cmd = ["sbatch"]
    
    # Add SLURM parameters
    if "cpus_per_task" in slurm_kwargs and slurm_kwargs["cpus_per_task"]:
        cmd.extend(["--cpus-per-task", str(slurm_kwargs["cpus_per_task"])])
    
    if "memory" in slurm_kwargs and slurm_kwargs["memory"]:
        cmd.extend(["--mem", str(slurm_kwargs["memory"])])
    
    if "time" in slurm_kwargs and slurm_kwargs["time"]:
        cmd.extend(["--time", str(slurm_kwargs["time"])])
    
    # Set descriptive job name
    job_name = f"logmel-{dataset_name}"
    cmd.extend(["--job-name", job_name])
    
    # Add the script and its arguments
    cmd.append("scripts/DataSet2LogMel.sbatch")
    cmd.extend(["--input_dataset", input_dataset])
    cmd.extend(["--output_dataset", output_dataset])
    
    # Add other DataSet2LogMel.py parameters
    conversion_kwargs = {k: v for k, v in slurm_kwargs.items() 
                        if k not in ["cpus_per_task", "memory", "time"] and v is not None}
    
    for key, value in conversion_kwargs.items():
        cmd.extend([f"--{key}", str(value)])
    
    print(f"Submitting job for {dataset_name}: {' '.join(cmd)}")
    print("=" * 80)
    
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=True)
        job_id = result.stdout.strip().split()[-1] if result.stdout.strip() else "unknown"
        print(f"✓ Successfully submitted job for {dataset_name}")
        print(f"  Job ID: {job_id}")
        print(f"  Input: {input_dataset}")
        print(f"  Output: {output_dataset}")
        return True, job_id
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to submit job for {dataset_name}: {e}")
        if e.stderr:
            print(f"  Error: {e.stderr}")
        return False, None


def main():
    parser = argparse.ArgumentParser(description="Batch submit SLURM jobs for converting hearing loss datasets to Log-Mel spectrograms")
    
    # SLURM resource parameters
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=48,
        help="Number of CPU cores per SLURM task (default: 48)"
    )
    parser.add_argument(
        "--memory", "--mem",
        type=str,
        default="400G",
        help="Memory allocation per job (default: 400G)"
    )
    parser.add_argument(
        "--time",
        type=str,
        default="32:00:00",
        help="Time limit per job (default: 32:00:00)"
    )
    
    # DataSet2LogMel.py parameters
    parser.add_argument(
        "--model-size",
        type=str,
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size (default: large-v3)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing (default: 1000)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples per split (for testing)"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip datasets that already have output directories"
    )
    
    args = parser.parse_args()
    
    # Define the datasets to process
    datasets = [
        {
            "name": "normal",
            "input": "data/CommonVoiceEN_normal/dataset",
            "output": "data/CommonVoiceEN_normal_logmel"
        },
        {
            "name": "lfloss", 
            "input": "data/CommonVoiceEN_lfloss/dataset",
            "output": "data/CommonVoiceEN_lfloss_logmel"
        },
        {
            "name": "hfloss",
            "input": "data/CommonVoiceEN_hfloss/dataset", 
            "output": "data/CommonVoiceEN_hfloss_logmel"
        }
    ]
    
    print("=" * 80)
    print("BATCH SLURM JOB SUBMISSION - LOG-MEL SPECTROGRAM CONVERSION")
    print("=" * 80)
    print("SLURM Configuration:")
    print(f"  CPUs per task: {args.cpus_per_task}")
    print(f"  Memory per job: {args.memory}")
    print(f"  Time limit: {args.time}")
    print("Processing Configuration:")
    print(f"  Model size: {args.model_size}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Max samples: {args.max_samples if args.max_samples else 'all'}")
    print(f"  Skip existing: {args.skip_existing}")
    print("=" * 80)
    
    successful_jobs = []
    failed_jobs = []
    submitted_job_ids = []
    
    for dataset in datasets:
        dataset_name = dataset["name"]
        input_path = dataset["input"]
        output_path = dataset["output"]
        
        print(f"\n� Submitting job for {dataset_name.upper()} dataset...")
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")
        
        # Check if input exists
        if not Path(input_path).exists():
            print(f"⚠️  Input path does not exist: {input_path}")
            failed_jobs.append(dataset_name)
            continue
        
        # Check if output already exists and skip if requested
        if args.skip_existing and Path(output_path).exists():
            print(f"⏭️  Output already exists, skipping: {output_path}")
            continue
        
        # Submit the SLURM job
        success, job_id = submit_logmel_job(
            input_dataset=input_path,
            output_dataset=output_path,
            dataset_name=dataset_name,
            cpus_per_task=args.cpus_per_task,
            memory=args.memory,
            time=args.time,
            model_size=args.model_size,
            batch_size=args.batch_size,
            max_samples=args.max_samples
        )
        
        if success:
            successful_jobs.append(dataset_name)
            if job_id:
                submitted_job_ids.append(job_id)
        else:
            failed_jobs.append(dataset_name)
        
        print("\n" + "=" * 80)
    
    # Final summary
    print("\n" + "=" * 80)
    print("BATCH JOB SUBMISSION SUMMARY")
    print("=" * 80)
    print(f"✅ Successfully submitted: {len(successful_jobs)} jobs")
    for i, name in enumerate(successful_jobs):
        job_id = submitted_job_ids[i] if i < len(submitted_job_ids) else "unknown"
        print(f"   - {name}: Job ID {job_id}")
    
    if failed_jobs:
        print(f"❌ Failed submissions: {len(failed_jobs)} jobs")
        for name in failed_jobs:
            print(f"   - {name}")
    else:
        print("🎉 All jobs submitted successfully!")
    
    if successful_jobs:
        print("\nMonitoring commands:")
        print("  Check job status: squeue --me")
        print("  View job details: scontrol show job <JOB_ID>")
        print("  Check logs: tail -f logs/logmel_<JOB_ID>.out")
        print("  Cancel jobs: scancel <JOB_ID>")
    
    print("=" * 80)
    print("Jobs will run in parallel. Check SLURM queue for status.")
    print("=" * 80)
    
    # Exit with error code if any failed to submit
    if failed_jobs:
        sys.exit(1)


if __name__ == "__main__":
    main()