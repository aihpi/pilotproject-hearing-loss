#!/usr/bin/env python3
"""
Apply all three hearing loss masks to individual word audio files.

This script processes WAV files and creates three versions with actual acoustic
processing applied to all conditions (including normal hearing):
- Normal hearing: 10 dB uniform attenuation
- Low-frequency loss: 100→10 dB gradient (severe loss at low frequencies)
- High-frequency loss: 10→100 dB gradient (severe loss at high frequencies)

All outputs remain as WAV files.

Usage:
    python scripts/apply_all_masks.py
    python scripts/apply_all_masks.py --input-dir /path/to/input --output-base /path/to/output
    python scripts/apply_all_masks.py --nworkers 48 --batch-size 100
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

import numpy as np
import soundfile as sf

# Import our audio processing module
sys.path.append(str(Path(__file__).parent))
from modules.audio import (
    AudioProcessor,
    get_normal_hearing_profile,
    get_low_frequency_loss_profile,
    get_high_frequency_loss_profile,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(process)d] %(message)s",
)
logger = logging.getLogger(__name__)

# Target sample rate for processing
SAMPLE_RATE = 16000


def process_single_file(
    input_path: Path,
    output_paths: dict,
    sample_rate: int = SAMPLE_RATE,
) -> Tuple[int, int, str]:
    """
    Process a single audio file with all hearing loss profiles.

    Args:
        input_path: Path to input WAV file
        output_paths: Dict mapping condition name to output path
        sample_rate: Target sample rate for processing

    Returns:
        Tuple of (success_count, failure_count, filename)
    """
    filename = input_path.name
    success_count = 0
    failure_count = 0

    try:
        # Load audio file
        audio_array, original_sr = sf.read(input_path)

        # Ensure audio is 1D (mono)
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        audio_array = audio_array.astype(np.float32)

        # Get hearing loss profiles
        profiles = {
            "normal": get_normal_hearing_profile(),
            "lfloss": get_low_frequency_loss_profile(),
            "hfloss": get_high_frequency_loss_profile(),
        }

        for condition, output_path in output_paths.items():
            try:
                # Apply hearing loss mask for ALL conditions (including normal)
                freq_points, db_thresholds = profiles[condition]

                processor = AudioProcessor(sample_rate=sample_rate)
                processed_audio, processed_sr = processor.process_with_hearing_loss(
                    audio_array=audio_array,
                    original_sr=original_sr,
                    freq_points=freq_points,
                    db_thresholds=db_thresholds,
                    target_sr=sample_rate,
                )

                # Save processed audio
                sf.write(output_path, processed_audio, processed_sr)

                success_count += 1

            except Exception as e:
                logger.warning(f"Failed to process {filename} for {condition}: {e}")
                failure_count += 1

    except Exception as e:
        logger.warning(f"Failed to load {filename}: {e}")
        failure_count = len(output_paths)

    return success_count, failure_count, filename


def process_batch(
    batch: List[Tuple[Path, dict]],
    sample_rate: int = SAMPLE_RATE,
) -> List[Tuple[int, int, str]]:
    """
    Process a batch of files sequentially within a worker.

    Args:
        batch: List of (input_path, output_paths) tuples
        sample_rate: Target sample rate

    Returns:
        List of (success_count, failure_count, filename) tuples
    """
    results = []
    for input_path, output_paths in batch:
        result = process_single_file(input_path, output_paths, sample_rate)
        results.append(result)
    return results


def process_split(
    input_dir: Path,
    output_dirs: dict,
    split: str,
    nworkers: int,
    batch_size: int,
) -> Tuple[int, int, int]:
    """
    Process all files in a split (train/test/validation).

    Args:
        input_dir: Base input directory
        output_dirs: Dict mapping condition to output base directory
        split: Split name (train/test/validation)
        nworkers: Number of parallel workers
        batch_size: Number of files per batch

    Returns:
        Tuple of (total_files, total_success, total_failure)
    """
    input_split_dir = input_dir / split

    # Check input directory exists
    if not input_split_dir.exists():
        logger.warning(f"Input directory not found: {input_split_dir}")
        return 0, 0, 0

    # Create output directories for each condition
    output_split_dirs = {}
    for condition, output_base in output_dirs.items():
        output_split_dir = Path(output_base) / split
        output_split_dir.mkdir(parents=True, exist_ok=True)
        output_split_dirs[condition] = output_split_dir

    # Collect all WAV files
    wav_files = sorted(input_split_dir.glob("*.wav"))
    total_files = len(wav_files)

    if total_files == 0:
        logger.info(f"No WAV files found in {input_split_dir}")
        return 0, 0, 0

    logger.info(f"Processing {total_files} files in {split} split with {nworkers} workers, batch size {batch_size}")

    # Create work items with output paths for each file
    work_items = []
    for wav_path in wav_files:
        output_paths = {
            condition: output_split_dirs[condition] / wav_path.name
            for condition in output_dirs.keys()
        }
        work_items.append((wav_path, output_paths))

    # Create batches
    batches = [work_items[i:i + batch_size] for i in range(0, len(work_items), batch_size)]

    total_success = 0
    total_failure = 0
    files_processed = 0

    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        futures = {executor.submit(process_batch, batch): batch for batch in batches}

        for future in as_completed(futures):
            try:
                results = future.result()
                for success, failure, filename in results:
                    total_success += success
                    total_failure += failure
                    files_processed += 1

                # Log progress periodically
                if files_processed % (batch_size * 10) == 0 or files_processed == total_files:
                    logger.info(
                        f"[{split}] Progress: {files_processed}/{total_files} files, "
                        f"{total_success} outputs created, {total_failure} failed"
                    )

            except Exception as e:
                logger.error(f"Batch processing failed: {e}")

    logger.info(
        f"[{split}] Completed: {files_processed} files, "
        f"{total_success} outputs created, {total_failure} failed"
    )

    return total_files, total_success, total_failure


def main():
    parser = argparse.ArgumentParser(
        description="Apply all three hearing loss masks to individual word audio files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/sc/projects/sci-aisc/hearingloss/data_processed/CommonVoiceENWords"),
        help="Directory containing input word audio files (default: /sc/projects/sci-aisc/hearingloss/data_processed/CommonVoiceENWords)",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("/sc/projects/sci-aisc/hearingloss/data_processed"),
        help="Base directory for output (default: /sc/projects/sci-aisc/hearingloss/data_processed)",
    )
    parser.add_argument(
        "--nworkers",
        type=int,
        default=48,
        help="Number of parallel workers (default: 48)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of files to process per batch (default: 100)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test", "validation"],
        help="Splits to process (default: train test validation)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=["normal", "lfloss", "hfloss"],
        choices=["normal", "lfloss", "hfloss"],
        help="Hearing loss conditions to generate (default: normal lfloss hfloss)",
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory not found: {args.input_dir}")
        return 1

    # Create output directories dict with user-specified naming
    output_dirs = {}
    for condition in args.conditions:
        if condition == "normal":
            output_dir = args.output_base / "normal_raw"
        elif condition == "lfloss":
            output_dir = args.output_base / "lfloss_raw"
        elif condition == "hfloss":
            output_dir = args.output_base / "hfloss_raw"

        output_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[condition] = output_dir
        logger.info(f"Output directory for {condition}: {output_dir}")

    logger.info("=" * 60)
    logger.info("Apply All Hearing Loss Masks")
    logger.info("=" * 60)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output base: {args.output_base}")
    logger.info(f"Workers: {args.nworkers}, Batch size: {args.batch_size}")
    logger.info(f"Splits: {args.splits}")
    logger.info(f"Conditions: {args.conditions}")
    logger.info("")
    logger.info("Hearing loss profiles:")
    logger.info("  - normal:  10 dB uniform (minimal attenuation)")
    logger.info("  - lfloss:  100→10 dB gradient (severe low-freq loss)")
    logger.info("  - hfloss:  10→100 dB gradient (severe high-freq loss)")
    logger.info("=" * 60)
    logger.info("")

    # Process each split
    grand_total_files = 0
    grand_total_success = 0
    grand_total_failure = 0

    for split in args.splits:
        total_files, total_success, total_failure = process_split(
            input_dir=args.input_dir,
            output_dirs=output_dirs,
            split=split,
            nworkers=args.nworkers,
            batch_size=args.batch_size,
        )
        grand_total_files += total_files
        grand_total_success += total_success
        grand_total_failure += total_failure

    logger.info("")
    logger.info("=" * 60)
    logger.info(
        f"All done! Processed {grand_total_files} files, "
        f"created {grand_total_success} outputs, {grand_total_failure} failed"
    )
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
