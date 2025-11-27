#!/usr/bin/env python3
"""
Extract raw audio and transcriptions from CommonVoice dataset.

This script extracts audio files and their corresponding transcriptions from
a CommonVoice dataset directory, saving them as individual .wav and .txt files.
"""

import os
import argparse
import logging
from pathlib import Path
from typing import List, Optional
import re

import librosa
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def extract_cv_id_from_path(path: str) -> Optional[str]:
    """
    Extract CommonVoice ID from the audio file path.
    
    Args:
        path: Path to the audio file (e.g., 'common_voice_en_20996434.mp3')
        
    Returns:
        CommonVoice ID (e.g., '20996434') or None if not found
    """
    # Extract the filename from the full path
    filename = Path(path).name
    
    # Match pattern: common_voice_en_XXXXXXXX.mp3
    match = re.search(r'common_voice_[a-z]+_(\d+)', filename)
    if match:
        return match.group(1)
    
    # Fallback: try to extract any numeric ID
    match = re.search(r'(\d+)', filename)
    if match:
        return match.group(1)
    
    return None


def process_sample(args):
    """
    Process a single audio sample: resample and save audio + transcription.
    
    Args:
        args: Tuple of (sample, output_dir, sample_idx, sampling_rate, channels)
        
    Returns:
        Tuple of (success: bool, cv_id: str, error_message: str or None)
    """
    sample, output_dir, sample_idx, sampling_rate, channels = args
    
    try:
        # Extract CommonVoice ID from path
        cv_id = extract_cv_id_from_path(sample['path'])
        if not cv_id:
            # Fallback to index-based naming if ID extraction fails
            cv_id = f"sample_{sample_idx:08d}"
            logger.warning(f"Could not extract CV ID from {sample['path']}, using {cv_id}")
        
        # Define output paths
        audio_path = output_dir / f"{cv_id}.wav"
        text_path = output_dir / f"{cv_id}.txt"
        
        # Skip if already exists
        if audio_path.exists() and text_path.exists():
            return (True, cv_id, None)
        
        # Get audio data
        audio_array = sample['audio']['array']
        original_sr = sample['audio']['sampling_rate']
        
        # Resample if needed
        if original_sr != sampling_rate:
            audio_array = librosa.resample(
                audio_array,
                orig_sr=original_sr,
                target_sr=sampling_rate
            )
        
        # Ensure mono (should already be mono, but just in case)
        if len(audio_array.shape) > 1:
            if channels == 'mono':
                audio_array = librosa.to_mono(audio_array)
        
        # Save audio file
        sf.write(audio_path, audio_array, sampling_rate)
        
        # Save transcription
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(sample['sentence'])
        
        return (True, cv_id, None)
        
    except Exception as e:
        error_msg = f"Error processing sample {sample_idx}: {str(e)}"
        logger.error(error_msg)
        return (False, str(sample_idx), error_msg)


def extract_split(
    dataset,
    split_name: str,
    output_base_dir: Path,
    sampling_rate: int,
    channels: str,
    max_samples: Optional[int],
    num_workers: int,
    batch_size: int
):
    """
    Extract audio and transcriptions for a single split.
    
    Args:
        dataset: HuggingFace dataset
        split_name: Name of the split (e.g., 'train', 'test', 'validation')
        output_base_dir: Base output directory
        sampling_rate: Target sampling rate in Hz
        channels: 'mono' or 'stereo'
        max_samples: Maximum number of samples to process (None for all)
        num_workers: Number of worker processes
        batch_size: Batch size for processing
    """
    logger.info(f"Processing split: {split_name}")
    
    # Get split data
    split_data = dataset[split_name]
    
    # Limit samples if specified
    if max_samples is not None:
        num_samples = min(max_samples, len(split_data))
    else:
        num_samples = len(split_data)
    
    logger.info(f"  Total samples: {num_samples}")
    
    # Create output directory for this split
    output_dir = output_base_dir / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process samples with multiprocessing and progress bar
    successful = 0
    failed = 0
    
    if num_workers > 1:
        # Use multiprocessing with batched loading to avoid memory issues
        with mp.Pool(processes=num_workers) as pool:
            results = []
            with tqdm(total=num_samples, desc=f"  {split_name}", unit="sample") as pbar:
                # Process in chunks to avoid loading entire dataset at once
                chunk_size = batch_size * num_workers  # Process multiple batches per iteration
                for start_idx in range(0, num_samples, chunk_size):
                    end_idx = min(start_idx + chunk_size, num_samples)
                    
                    # Load only this chunk of data
                    chunk_data = split_data.select(range(start_idx, end_idx))
                    
                    # Prepare arguments for this chunk
                    process_args = [
                        (chunk_data[i], output_dir, start_idx + i, sampling_rate, channels)
                        for i in range(len(chunk_data))
                    ]
                    
                    # Process this chunk
                    for result in pool.imap_unordered(process_sample, process_args, chunksize=batch_size):
                        results.append(result)
                        if result[0]:  # success
                            successful += 1
                        else:
                            failed += 1
                        pbar.update(1)
    else:
        # Single-threaded processing
        results = []
        for i in tqdm(range(num_samples), desc=f"  {split_name}", unit="sample"):
            args = (split_data[i], output_dir, i, sampling_rate, channels)
            result = process_sample(args)
            results.append(result)
            if result[0]:  # success
                successful += 1
            else:
                failed += 1
    
    logger.info(f"  Completed: {successful} successful, {failed} failed")
    
    # Log any errors
    errors = [r for r in results if not r[0]]
    if errors:
        logger.warning(f"  {len(errors)} samples failed to process")
        for success, cv_id, error_msg in errors[:10]:  # Show first 10 errors
            logger.warning(f"    {error_msg}")
        if len(errors) > 10:
            logger.warning(f"    ... and {len(errors) - 10} more errors")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Extract raw audio and transcriptions from CommonVoice dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input-dir",
        default="data/CommonVoiceEN",
        help="Input dataset directory"
    )
    
    parser.add_argument(
        "--output-dir",
        default="data/CommonVoiceENraw",
        help="Output directory for extracted files"
    )
    
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test", "validation"],
        help="Dataset splits to extract"
    )
    
    parser.add_argument(
        "--sampling-rate",
        type=int,
        default=16000,
        help="Target sampling rate in Hz"
    )
    
    parser.add_argument(
        "--channels",
        choices=["mono", "stereo"],
        default="mono",
        help="Audio channel configuration"
    )
    
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples per split (default: all samples)"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of worker processes for parallel processing"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing (used with multiprocessing)"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("CommonVoice Dataset Extraction")
    logger.info("=" * 70)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Splits: {args.splits}")
    logger.info(f"Sampling rate: {args.sampling_rate} Hz")
    logger.info(f"Channels: {args.channels}")
    logger.info(f"Max samples per split: {args.max_samples if args.max_samples else 'All'}")
    logger.info(f"Number of workers: {args.num_workers}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info("=" * 70)
    
    # Load dataset
    logger.info("Loading CommonVoice dataset...")
    input_dir = Path(args.input_dir)
    cache_dir = input_dir / "cache"
    
    try:
        dataset = load_dataset(
            "mozilla-foundation/common_voice_16_1",
            "en",
            cache_dir=str(cache_dir),
            trust_remote_code=True
        )
        logger.info(f"Dataset loaded successfully!")
        logger.info(f"Available splits: {list(dataset.keys())}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        logger.error("Make sure the dataset is downloaded in the input directory")
        return 1
    
    # Verify requested splits exist
    available_splits = list(dataset.keys())
    invalid_splits = [s for s in args.splits if s not in available_splits]
    if invalid_splits:
        logger.error(f"Invalid splits requested: {invalid_splits}")
        logger.error(f"Available splits: {available_splits}")
        return 1
    
    # Process each split
    logger.info("\nExtracting data...")
    for split_name in args.splits:
        extract_split(
            dataset=dataset,
            split_name=split_name,
            output_base_dir=output_base_dir,
            sampling_rate=args.sampling_rate,
            channels=args.channels,
            max_samples=args.max_samples,
            num_workers=args.num_workers,
            batch_size=args.batch_size
        )
    
    logger.info("\n" + "=" * 70)
    logger.info("Extraction complete!")
    logger.info(f"Output saved to: {output_base_dir}")
    logger.info("=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())
