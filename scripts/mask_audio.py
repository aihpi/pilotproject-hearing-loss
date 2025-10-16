#!/usr/bin/env python3
"""
Audio masking script for CommonVoice datasets.

This script applies hearing loss masks to CommonVoice datasets, creating
three versions: normal hearing, low-frequency loss, and high-frequency loss.
"""

import os
import sys
import argparse
import logging
import time
import shutil
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import gc

import numpy as np
from datasets import load_from_disk, Dataset, DatasetDict
from tqdm import tqdm

# Import our audio processing module
sys.path.append(str(Path(__file__).parent))
from modules.audio import (
    AudioProcessor,
    get_normal_hearing_profile,
    get_low_frequency_loss_profile,
    get_high_frequency_loss_profile,
    validate_audio_data
)

# Configure logging
def setup_logging(log_level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

logger = logging.getLogger(__name__)


def load_dataset_splits(dataset_path: str) -> DatasetDict:
    """
    Load HuggingFace dataset from disk or cache.
    
    Args:
        dataset_path (str): Path to the saved dataset or cache directory
        
    Returns:
        DatasetDict: Loaded dataset with all splits
    """
    logger.info(f"Loading dataset from: {dataset_path}")
    
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    # Check if this is a direct dataset directory (like our processed subsets)
    dataset_dict_path = Path(dataset_path) / "dataset_dict.json"
    if dataset_dict_path.exists():
        logger.info("Loading from dataset directory...")
        dataset = load_from_disk(dataset_path)
    else:
        # This might be a CommonVoice download directory with cache
        # Try to load using the HuggingFace datasets library
        cache_dir = Path(dataset_path) / "cache"
        if cache_dir.exists():
            logger.info("Loading from HuggingFace cache directory...")
            from datasets import load_dataset
            dataset = load_dataset(
                "mozilla-foundation/common_voice_16_1",
                "en",
                cache_dir=str(cache_dir),
                trust_remote_code=True
            )
        else:
            # Try direct loading as fallback
            logger.info("Attempting direct dataset loading...")
            dataset = load_from_disk(dataset_path)
    
    logger.info(f"Dataset loaded successfully")
    logger.info(f"Available splits: {list(dataset.keys())}")
    for split_name, split_data in dataset.items():
        logger.info(f"  - {split_name}: {len(split_data)} samples")
    
    return dataset


def create_output_directories(base_dir: str, suffixes: List[str]) -> Dict[str, str]:
    """
    Create output directories for each hearing loss condition.
    
    Args:
        base_dir (str): Base directory name
        suffixes (List[str]): Suffixes for each condition (e.g., ['_normal', '_lfloss', '_hfloss'])
        
    Returns:
        Dict[str, str]: Mapping from condition name to directory path
    """
    output_dirs = {}
    base_path = Path(base_dir)
    
    for suffix in suffixes:
        dir_name = f"{base_path.name}{suffix}"
        dir_path = base_path.parent / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        
        condition = suffix.lstrip('_')  # Remove leading underscore
        output_dirs[condition] = str(dir_path)
        logger.info(f"Created output directory for {condition}: {dir_path}")
    
    return output_dirs


def copy_dataset_structure(source_dir: str, target_dirs: Dict[str, str]) -> None:
    """
    Copy dataset structure and metadata files to target directories.
    For large datasets, this creates minimal structure instead of copying all files.
    
    Args:
        source_dir (str): Source dataset directory
        target_dirs (Dict[str, str]): Target directories for each condition
    """
    source_path = Path(source_dir)
    
    logger.info("Setting up output dataset structures...")
    
    # For large datasets from HuggingFace cache, we'll create minimal structure
    # instead of copying everything, which can be very slow
    if "cache" in str(source_path):
        logger.info("Detected HuggingFace cache format - creating minimal dataset structure")
        
        for condition, target_dir in target_dirs.items():
            target_path = Path(target_dir)
            dataset_path = target_path / "dataset"
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            # Create a simple dataset_dict.json for the structure
            # This will be properly created when we save the datasets
            logger.debug(f"Created directory structure for {condition}")
    else:
        # For regular dataset directories, copy structure files
        logger.info("Copying dataset structure files...")
        
        for condition, target_dir in target_dirs.items():
            target_path = Path(target_dir)
            dataset_path = target_path / "dataset"
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            # Copy dataset structure files (but not the actual data)
            json_files = list(source_path.rglob("*.json"))
            logger.info(f"Copying {len(json_files)} metadata files for {condition}...")
            
            for item in json_files:
                try:
                    rel_path = item.relative_to(source_path)
                    target_file = dataset_path / rel_path
                    target_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target_file)
                    logger.debug(f"Copied {item} to {target_file}")
                except Exception as e:
                    logger.warning(f"Could not copy {item}: {e}")
    
    logger.info("Dataset structure setup completed")


def process_single_sample(
    sample: Dict[str, Any], 
    hearing_profiles: Dict[str, Tuple[np.ndarray, np.ndarray]],
    sample_rate: int = 16000
) -> Dict[str, Dict[str, Any]]:
    """
    Process a single audio sample with all hearing loss conditions.
    
    Args:
        sample (Dict): Single sample from the dataset
        hearing_profiles (Dict): Hearing loss profiles for each condition
        sample_rate (int): Target sample rate
        
    Returns:
        Dict[str, Dict]: Processed samples for each condition
    """
    # Extract audio data
    audio_data = sample['audio']['array']
    original_sr = sample['audio']['sampling_rate']
    
    # Validate audio data
    if not validate_audio_data(audio_data, original_sr):
        logger.error(f"Invalid audio data in sample")
        return {}
    
    # Initialize processor
    processor = AudioProcessor(sample_rate=sample_rate)
    
    # Process with each hearing loss condition
    results = {}
    
    for condition, (freq_points, db_thresholds) in hearing_profiles.items():
        try:
            # Process audio
            processed_audio, processed_sr = processor.process_with_hearing_loss(
                audio_data, original_sr, freq_points, db_thresholds, sample_rate
            )
            
            # Create new sample with processed audio
            new_sample = sample.copy()
            new_sample['audio'] = {
                'array': processed_audio,
                'sampling_rate': processed_sr
            }
            
            results[condition] = new_sample
            
        except Exception as e:
            logger.error(f"Error processing sample for condition {condition}: {e}")
            continue
    
    return results


def process_batch(
    batch_data: List[Dict[str, Any]], 
    hearing_profiles: Dict[str, Tuple[np.ndarray, np.ndarray]],
    sample_rate: int = 16000
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Process a batch of audio samples.
    
    Args:
        batch_data (List[Dict]): Batch of samples from the dataset
        hearing_profiles (Dict): Hearing loss profiles for each condition
        sample_rate (int): Target sample rate
        
    Returns:
        Dict[str, List[Dict]]: Processed batches for each condition
    """
    results = {condition: [] for condition in hearing_profiles.keys()}
    
    for sample in batch_data:
        processed_samples = process_single_sample(sample, hearing_profiles, sample_rate)
        
        for condition, processed_sample in processed_samples.items():
            results[condition].append(processed_sample)
    
    return results


def worker_process_batch(args: Tuple) -> Dict[str, List[Dict[str, Any]]]:
    """
    Worker function for multiprocessing batch processing.
    
    Args:
        args (Tuple): (batch_data, hearing_profiles, sample_rate)
        
    Returns:
        Dict: Processed batch results
    """
    batch_data, hearing_profiles, sample_rate = args
    return process_batch(batch_data, hearing_profiles, sample_rate)


def process_split_in_batches(
    split_data: Dataset,
    output_paths: Dict[str, str],
    hearing_profiles: Dict[str, Tuple[np.ndarray, np.ndarray]],
    batch_size: int = 32,
    num_workers: int = 4,
    sample_rate: int = 16000
) -> None:
    """
    Process a dataset split in batches with memory-efficient multiprocessing.
    
    Args:
        split_data (Dataset): Dataset split to process
        output_paths (Dict[str, str]): Output paths for each condition
        hearing_profiles (Dict): Hearing loss profiles
        batch_size (int): Number of samples per batch
        num_workers (int): Number of worker processes
        sample_rate (int): Target sample rate
    """
    total_samples = len(split_data)
    logger.info(f"Processing {total_samples} samples in batches of {batch_size} using {num_workers} workers")
    
    # For very large datasets, reduce workers to prevent memory issues
    if total_samples > 500000 and num_workers > 8:
        original_workers = num_workers
        num_workers = min(8, num_workers)
        logger.warning(f"Large dataset detected! Reducing workers from {original_workers} to {num_workers} to prevent memory issues")
    
    # Initialize results storage for each condition (save incrementally to avoid memory buildup)
    temp_results = {condition: [] for condition in hearing_profiles.keys()}
    processed_count = 0
    
    # Process in smaller chunks to manage memory
    chunk_size = min(10000, total_samples)  # Process max 10k samples at a time
    
    start_time = time.time()
    
    for chunk_start in range(0, total_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_samples)
        chunk_samples = chunk_end - chunk_start
        
        logger.info(f"Processing chunk {chunk_start//chunk_size + 1}/{(total_samples-1)//chunk_size + 1}: samples {chunk_start}-{chunk_end-1}")
        
        # Prepare batches for this chunk
        batches = []
        for i in range(chunk_start, chunk_end, batch_size):
            end_idx = min(i + batch_size, chunk_end)
            # Convert to list of individual samples
            batch = []
            for j in range(i, end_idx):
                batch.append(split_data[j])
            batches.append((batch, hearing_profiles, sample_rate))
        
        # Process this chunk's batches
        if num_workers > 1:
            with mp.Pool(num_workers) as pool:
                batch_results = list(tqdm(
                    pool.imap(worker_process_batch, batches),
                    total=len(batches),
                    desc=f"Chunk {chunk_start//chunk_size + 1}"
                ))
        else:
            batch_results = []
            for batch_args in tqdm(batches, desc=f"Chunk {chunk_start//chunk_size + 1}"):
                batch_results.append(worker_process_batch(batch_args))
        
        # Collect results from this chunk
        chunk_results = {condition: [] for condition in hearing_profiles.keys()}
        for batch_result in batch_results:
            for condition, batch_samples in batch_result.items():
                chunk_results[condition].extend(batch_samples)
        
        # Append to temporary results
        for condition, samples in chunk_results.items():
            temp_results[condition].extend(samples)
        
        processed_count += chunk_samples
        
        # Save incrementally every few chunks to avoid memory buildup
        if len(temp_results[list(hearing_profiles.keys())[0]]) >= 50000 or chunk_end == total_samples:
            logger.info(f"Saving intermediate results ({len(temp_results[list(hearing_profiles.keys())[0]])} samples)")
            
            for condition, processed_samples in temp_results.items():
                if processed_samples:
                    output_path = output_paths[condition]
                    
                    # Check if we already have saved data (append mode)
                    existing_data = []
                    if Path(output_path).exists():
                        try:
                            existing_dataset = Dataset.load_from_disk(output_path)
                            existing_data = existing_dataset.to_list()
                            logger.debug(f"Found {len(existing_data)} existing samples for {condition}")
                        except:
                            logger.debug(f"Could not load existing data for {condition}, starting fresh")
                    
                    # Combine existing and new data
                    all_samples = existing_data + processed_samples
                    dataset = Dataset.from_list(all_samples)
                    dataset.save_to_disk(output_path)
                    logger.info(f"Saved {len(all_samples)} total samples to {output_path}")
            
            # Clear temporary results to free memory
            temp_results = {condition: [] for condition in hearing_profiles.keys()}
            cleanup_memory()
        
        # Monitor memory usage
        memory_mb = monitor_memory_usage()
        if memory_mb > 0:
            logger.debug(f"Memory usage after chunk: {memory_mb:.1f} MB")
    
    elapsed_time = time.time() - start_time
    avg_speed = total_samples / elapsed_time
    logger.info(f"Split processing completed in {elapsed_time:.2f} seconds ({avg_speed:.1f} samples/sec)")


def save_processed_dataset(datasets: Dict[str, DatasetDict], output_dirs: Dict[str, str]) -> None:
    """
    Save processed datasets to output directories.
    
    Args:
        datasets (Dict[str, DatasetDict]): Processed datasets for each condition
        output_dirs (Dict[str, str]): Output directories for each condition
    """
    for condition, dataset_dict in datasets.items():
        output_dir = Path(output_dirs[condition]) / "dataset"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving dataset for condition '{condition}' to {output_dir}")
        dataset_dict.save_to_disk(str(output_dir))
        logger.info(f"Dataset saved successfully")


def cleanup_memory() -> None:
    """Force garbage collection to free memory."""
    gc.collect()


def monitor_memory_usage() -> float:
    """
    Monitor current memory usage.
    
    Returns:
        float: Memory usage in MB
    """
    try:
        import psutil
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Get system memory info
        system_memory = psutil.virtual_memory()
        total_mb = system_memory.total / 1024 / 1024
        available_mb = system_memory.available / 1024 / 1024
        
        # Warn if memory usage is getting high
        if memory_mb > 32000:  # 32GB
            logger.warning(f"High memory usage: {memory_mb:.1f} MB")
        
        if available_mb < 8000:  # Less than 8GB available
            logger.warning(f"Low system memory available: {available_mb:.1f} MB")
        
        return memory_mb
    except ImportError:
        logger.debug("psutil not available for memory monitoring")
        return 0.0


def process_dataset(
    input_dir: str,
    output_base: str,
    sample_rate: int = 16000,
    batch_size: int = 32,
    num_workers: int = 4,
    target_splits: Optional[List[str]] = None,
    dry_run: bool = False,
    **kwargs
) -> None:
    """
    Main dataset processing function.
    
    Args:
        input_dir (str): Input dataset directory
        output_base (str): Base name for output directories
        sample_rate (int): Target sample rate
        batch_size (int): Batch size for processing
        num_workers (int): Number of worker processes
        **kwargs: Additional arguments
    """
    logger.info("=== Starting dataset processing ===")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output base: {output_base}")
    logger.info(f"Target sample rate: {sample_rate} Hz")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Number of workers: {num_workers}")
    
    # Load dataset
    dataset = load_dataset_splits(input_dir)
    
    # Filter splits if specified
    if target_splits:
        logger.info(f"Filtering to specified splits: {target_splits}")
        available_splits = set(dataset.keys())
        requested_splits = set(target_splits)
        invalid_splits = requested_splits - available_splits
        
        if invalid_splits:
            logger.error(f"Invalid splits requested: {invalid_splits}")
            logger.error(f"Available splits: {available_splits}")
            return
        
        dataset = {split: data for split, data in dataset.items() if split in target_splits}
        logger.info(f"Processing only splits: {list(dataset.keys())}")
    
    # Calculate total samples for progress tracking
    total_samples = sum(len(split_data) for split_data in dataset.values())
    logger.info(f"Total samples to process: {total_samples:,}")
    
    # Monitor initial memory
    initial_memory = monitor_memory_usage()
    if initial_memory > 0:
        logger.info(f"Initial memory usage: {initial_memory:.1f} MB")
    
    # For very large datasets, suggest optimization and adjust workers
    if total_samples > 500000:  # 500k samples
        logger.info("🔧 Large dataset detected! Applying optimizations:")
        logger.info(f"   - Recommended batch size: 64-128 (current: {batch_size})")
        logger.info(f"   - Recommended workers: 8-16 (current: {num_workers})")
        logger.info(f"   - Estimated processing time: {total_samples/20/60:.1f} - {total_samples/40/60:.1f} minutes")
        logger.info(f"   - Consider processing splits separately: --splits train")
        
        # Auto-adjust workers for very large datasets to prevent memory issues
        if total_samples > 1000000 and num_workers > 12:
            original_workers = num_workers
            num_workers = min(12, num_workers)
            logger.warning(f"⚠️  Very large dataset! Reducing workers from {original_workers} to {num_workers} to prevent memory issues")
        
    if dry_run:
        logger.info("🔍 DRY RUN - No actual processing will be performed")
        logger.info("Splits that would be processed:")
        for split_name, split_data in dataset.items():
            logger.info(f"  - {split_name}: {len(split_data):,} samples")
        return
    
    # Create output directories
    suffixes = ['_normal', '_lfloss', '_hfloss']
    output_dirs = create_output_directories(output_base, suffixes)
    
    # Copy dataset structure
    copy_dataset_structure(input_dir, output_dirs)
    
    # Define hearing loss profiles
    hearing_profiles = {
        'normal': get_normal_hearing_profile(),
        'lfloss': get_low_frequency_loss_profile(),
        'hfloss': get_high_frequency_loss_profile()
    }
    
    logger.info("Hearing loss profiles:")
    for condition, (freq_points, thresholds) in hearing_profiles.items():
        logger.info(f"  {condition}: {dict(zip(freq_points, thresholds))}")
    
    # Process each split with progress tracking
    processed_samples = 0
    start_time = time.time()
    
    for split_name, split_data in dataset.items():
        logger.info(f"\n=== Processing split: {split_name} ({len(split_data):,} samples) ===")
        
        # Create output paths for this split
        split_output_paths = {
            condition: str(Path(output_dir) / "dataset" / split_name)
            for condition, output_dir in output_dirs.items()
        }
        
        # Process the split
        split_start = time.time()
        process_split_in_batches(
            split_data,
            split_output_paths,
            hearing_profiles,
            batch_size=batch_size,
            num_workers=num_workers,  # This may have been auto-adjusted above
            sample_rate=sample_rate
        )
        
        # Update progress
        processed_samples += len(split_data)
        split_time = time.time() - split_start
        overall_time = time.time() - start_time
        progress_pct = (processed_samples / total_samples) * 100
        
        logger.info(f"✅ Completed split: {split_name} ({split_time:.1f}s)")
        logger.info(f"📊 Overall progress: {processed_samples:,}/{total_samples:,} ({progress_pct:.1f}%) in {overall_time:.1f}s")
        
        if processed_samples < total_samples:
            remaining_samples = total_samples - processed_samples
            avg_speed = processed_samples / overall_time
            eta_seconds = remaining_samples / avg_speed
            logger.info(f"⏱️  ETA: {eta_seconds/60:.1f} minutes remaining")
    
    overall_time = time.time() - start_time
    avg_speed = total_samples / overall_time
    
    logger.info("\n=== Dataset processing completed successfully! ===")
    logger.info(f"📈 Total processing time: {overall_time/60:.1f} minutes")
    logger.info(f"⚡ Average speed: {avg_speed:.1f} samples/second")
    
    # Log final output directories
    logger.info("Output datasets created:")
    for condition, output_dir in output_dirs.items():
        logger.info(f"  {condition}: {output_dir}")


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Apply hearing loss masks to CommonVoice datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input-dir",
        default="data/CommonVoiceEN",
        help="Input dataset directory"
    )
    
    parser.add_argument(
        "--output-base",
        help="Base name for output directories (default: same as input directory)"
    )
    
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate in Hz"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of samples to process per batch"
    )
    
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes for parallel processing"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--splits",
        nargs="+",
        help="Specific splits to process (e.g., --splits train validation). If not specified, all splits will be processed."
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing"
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level)
    
    # Set default output base
    if args.output_base is None:
        args.output_base = args.input_dir
    
    # Validate arguments
    if not Path(args.input_dir).exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        sys.exit(1)
    
    if args.num_workers < 1:
        logger.error("Number of workers must be at least 1")
        sys.exit(1)
    
    if args.batch_size < 1:
        logger.error("Batch size must be at least 1")
        sys.exit(1)
    
    # Log system information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Available CPU cores: {mp.cpu_count()}")
    
    memory_mb = monitor_memory_usage()
    if memory_mb > 0:
        logger.info(f"Initial memory usage: {memory_mb:.1f} MB")
    
    try:
        # Process the dataset
        process_dataset(
            input_dir=args.input_dir,
            output_base=args.output_base,
            sample_rate=args.sample_rate,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            target_splits=args.splits,
            dry_run=args.dry_run
        )
        
        logger.info("Script completed successfully!")
        
    except KeyboardInterrupt:
        logger.warning("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        logger.debug("Full traceback:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()