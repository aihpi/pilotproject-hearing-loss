#!/usr/bin/env python3
#############################################################################
# Script Name: DataSet2LogMel.py                                            #
# Description: Memory-optimized preprocessing of CommonVoice hearing loss   #
#              datasets to Log-Mel Spectrograms for Whisper training        #
# Author: Hanno Müller (adapted for hearing loss project)                   #
# Date: 2025-10-20                                                          #
#############################################################################

### Required Libraries ######################################################
import os
import gc
import argparse
import multiprocessing
from typing import Dict, Any
from pathlib import Path

from transformers import (
    WhisperFeatureExtractor,
    WhisperTokenizer
)
from datasets import DatasetDict, Dataset, load_from_disk
import torch
import numpy as np

# Load environment variables from .env.local
def load_env_file():
    """Load environment variables from .env.local file."""
    env_file = Path(__file__).parent.parent / ".env.local"
    if env_file.exists():
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
        print(f"Loaded environment variables from {env_file}")
        if 'HF_TOKEN' in os.environ:
            print(f"HF_TOKEN found: {os.environ['HF_TOKEN'][:10]}...")
    else:
        print(f"No .env.local file found at {env_file}")

# Load environment at module level
load_env_file()


def get_expected_mel_bins(model_size):
    """Get expected number of mel bins for a given Whisper model size."""
    if model_size == "large-v3":
        return 128
    else:
        return 80

def validate_feature_extractor(feature_extractor, model_size):
    """Validate that the feature extractor produces the expected number of mel bins."""
    expected_mel_bins = get_expected_mel_bins(model_size)
    
    # The feature extractor should have the correct configuration
    actual_mel_bins = getattr(feature_extractor, 'n_mels', None)
    
    if actual_mel_bins is None:
        print(f"Warning: Could not determine mel bins from feature extractor")
    elif actual_mel_bins != expected_mel_bins:
        print(f"Warning: Feature extractor has {actual_mel_bins} mel bins, expected {expected_mel_bins} for {model_size}")
    else:
        print(f"✓ Feature extractor configured for {actual_mel_bins} mel bins (correct for {model_size})")
    
    return actual_mel_bins

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Memory-optimized preprocessing of CommonVoice hearing loss datasets to log-Mel spectrograms for Whisper training.")
    parser.add_argument(
        "-i", "--input_dataset",
        type=str,
        required=True,
        help="Path to the input CommonVoice dataset folder (e.g., data/CommonVoiceEN_normal/dataset)"
    )
    parser.add_argument(
        "-o", "--output_dataset", 
        type=str,
        required=True,
        help="Path where the preprocessed dataset will be saved (e.g., data/CommonVoiceEN_normal_logmel)"
    )
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=None,
        help="Number of CPU cores to use for preprocessing. If not specified, will use all available cores."
    )
    parser.add_argument(
        "--model_size",
        type=str,
        default="large-v3",
        choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"],
        help="Whisper model size for feature extraction (default: large-v3)"
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=42,
        help="Random seed for shuffling datasets (default: 42)"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process per split (for testing). If not specified, processes all samples."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for processing dataset chunks (default: 1000)"
    )
    parser.add_argument(
        "--writer_batch_size",
        type=int,
        default=100,
        help="Writer batch size for saving to disk (default: 100)"
    )
    parser.add_argument(
        "--max_memory_per_worker",
        type=float,
        default=4.0,
        help="Maximum memory per worker in GB (default: 4.0)"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Language for tokenizer (default: en for English)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task type for tokenizer (default: transcribe)"
    )
    return parser.parse_args()


def configure_cpus(args):
    """Configure CPU resources based on user specifications and SLURM environment."""
    print("=== CPU Configuration ===")
    
    # Check if running under SLURM
    slurm_cpus = os.environ.get('SLURM_CPUS_PER_TASK')
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    
    if slurm_job_id:
        print(f"Running under SLURM (Job ID: {slurm_job_id})")
        if slurm_cpus:
            slurm_cpu_count = int(slurm_cpus)
            print(f"SLURM allocated CPUs: {slurm_cpu_count}")
        else:
            slurm_cpu_count = None
    else:
        print("Not running under SLURM")
        slurm_cpu_count = None
    
    available_cpus = multiprocessing.cpu_count()
    print(f"Available CPUs on this node: {available_cpus}")
    
    if args.num_cpus is not None:
        requested_cpus = args.num_cpus
        print(f"User requested CPUs: {requested_cpus}")
        
        # Determine the actual limit
        if slurm_cpu_count:
            max_cpus = min(slurm_cpu_count, available_cpus)
            limit_source = "SLURM allocation"
        else:
            max_cpus = available_cpus
            limit_source = "available CPUs"
        
        if requested_cpus > max_cpus:
            print(f"Warning: Requested {requested_cpus} CPUs, but only {max_cpus} {limit_source}. Using {max_cpus}.")
            num_cpus = max_cpus
        elif requested_cpus <= 0:
            print("Error: Number of CPUs must be positive.")
            raise ValueError("Invalid number of CPUs specified")
        else:
            num_cpus = requested_cpus
    else:
        # Auto-detect mode: use SLURM allocation if available, otherwise all available
        if slurm_cpu_count:
            num_cpus = min(slurm_cpu_count, available_cpus)
            print(f"Auto-detect mode: using {num_cpus} CPUs (SLURM allocation)")
        else:
            num_cpus = available_cpus
            print(f"Auto-detect mode: using {num_cpus} CPUs (all available)")
    
    # Memory-based CPU limiting
    import psutil
    total_memory_gb = psutil.virtual_memory().total / (1024**3)
    max_memory_efficient_cpus = int(total_memory_gb / args.max_memory_per_worker)
    
    if num_cpus > max_memory_efficient_cpus:
        print(f"Memory consideration: {total_memory_gb:.1f}GB total memory")
        print(f"With {args.max_memory_per_worker}GB per worker, recommending max {max_memory_efficient_cpus} CPUs")
        print(f"Reducing from {num_cpus} to {max_memory_efficient_cpus} CPUs for memory efficiency")
        num_cpus = max_memory_efficient_cpus
    
    print(f"Final configuration: using {num_cpus} CPU cores for preprocessing")
    print("=" * 30)
    
    return num_cpus


def load_commonvoice_dataset(dataset_path: str) -> DatasetDict:
    """
    Load CommonVoice dataset from individual split directories.
    
    Args:
        dataset_path: Path to dataset directory containing split subdirectories
        
    Returns:
        DatasetDict with loaded splits
    """
    dataset_path = Path(dataset_path)
    print(f"Loading CommonVoice dataset from: {dataset_path}")
    
    # Check if dataset_path contains individual split directories
    split_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name in ['train', 'validation', 'test', 'other', 'invalidated']]
    
    if split_dirs:
        print(f"Found split directories: {[d.name for d in split_dirs]}")
        dataset_dict = DatasetDict()
        
        for split_dir in split_dirs:
            split_name = split_dir.name
            print(f"Loading {split_name} split from {split_dir}")
            try:
                dataset_dict[split_name] = load_from_disk(str(split_dir))
                print(f"✓ Loaded {split_name}: {len(dataset_dict[split_name])} samples")
            except Exception as e:
                print(f"✗ Failed to load {split_name}: {e}")
        
        return dataset_dict
    else:
        # Try loading as a single DatasetDict
        try:
            dataset_dict = DatasetDict.load_from_disk(str(dataset_path))
            print(f"Loaded as DatasetDict. Splits: {list(dataset_dict.keys())}")
            return dataset_dict
        except Exception as e:
            print(f"Error: Could not load dataset from {dataset_path}")
            print(f"Expected either individual split directories (train/, validation/, test/) or a DatasetDict")
            raise e


def prepare_dataset(batch, feature_extractor, tokenizer):
    """
    Memory-optimized preparation of dataset batch for Whisper training.
    Converts audio to log-Mel spectrograms and text to token IDs.
    """
    # Ensure audio is at 16kHz (should already be from CommonVoice dataset)
    audio = batch["audio"]
    
    # Convert audio array to numpy array if it's a list
    audio_array = audio["array"]
    if isinstance(audio_array, list):
        audio_array = np.array(audio_array, dtype=np.float32)
    
    # Compute log-Mel spectrogram input features from audio array
    # Use torch.no_grad() to prevent gradient computation and reduce memory
    with torch.no_grad():
        batch["input_features"] = feature_extractor(
            audio_array, 
            sampling_rate=audio["sampling_rate"]
        ).input_features[0]
    
    # Encode target text to label ids
    batch["labels"] = tokenizer(batch["sentence"]).input_ids
    
    # Clean up intermediate variables
    del audio
    
    return batch


### main ######################################################################

if __name__ == "__main__":
    print("Starting memory-optimized CommonVoice dataset preprocessing to log-Mel spectrograms...")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure CPU resources
    num_cpus = configure_cpus(args)
    
    # Memory optimization settings
    print("=== Memory Optimization Settings ===")
    print(f"Batch size: {args.batch_size}")
    print(f"Writer batch size: {args.writer_batch_size}")
    print(f"Max memory per worker: {args.max_memory_per_worker}GB")
    print("=" * 35)
    
    # Load the CommonVoice dataset from disk
    try:
        dataset_dict = load_commonvoice_dataset(args.input_dataset)
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(dataset_dict.keys())}")
        
        # Display dataset sizes
        for split_name, split_data in dataset_dict.items():
            print(f"{str(split_name).capitalize()} dataset size: {len(split_data):,} samples")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise
    
    # Initialize Whisper components for preprocessing
    model_name = f"openai/whisper-{args.model_size}"
    print(f"Loading Whisper components for model: {model_name}")
    print(f"Model size: {args.model_size}")
    
    # Get expected mel bins for this model
    expected_mel_bins = get_expected_mel_bins(args.model_size)
    print(f"Expected mel bins for {args.model_size}: {expected_mel_bins}")
    if args.model_size == "large-v3":
        print("Note: Whisper large-v3 uses 128 mel bins instead of 80")
    
    # Load with memory optimization
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_name)
    tokenizer = WhisperTokenizer.from_pretrained(
        model_name, 
        language=args.language, 
        task=args.task
    )
    
    # Validate feature extractor configuration
    actual_mel_bins = validate_feature_extractor(feature_extractor, args.model_size)
    
    print("Whisper components loaded successfully!")
    print(f"Language: {args.language}")
    print(f"Task: {args.task}")
    
    # Prepare preprocessing function
    def prepare_dataset_with_processors(batch):
        return prepare_dataset(batch, feature_extractor, tokenizer)
    
    # Process each split in the dataset with memory optimization
    processed_dataset_dict = DatasetDict()
    
    # Process main splits first (train, validation, test), then others
    preferred_order = ['train', 'validation', 'test', 'other', 'invalidated']
    splits_to_process = []
    
    # Add splits in preferred order if they exist
    for split_name in preferred_order:
        if split_name in dataset_dict:
            splits_to_process.append(split_name)
    
    # Add any remaining splits
    for split_name in dataset_dict.keys():
        if split_name not in splits_to_process:
            splits_to_process.append(split_name)
    
    for split_name in splits_to_process:
        split_dataset = dataset_dict[split_name]
        print(f"\nProcessing {split_name} split...")
        print(f"Original columns: {split_dataset.column_names}")
        print(f"Original dataset size: {len(split_dataset):,} samples")
        
        # Skip empty splits
        if len(split_dataset) == 0:
            print(f"Skipping empty {split_name} split")
            continue
        
        # Limit samples if max_samples is specified
        if args.max_samples is not None and len(split_dataset) > args.max_samples:
            print(f"Limiting to {args.max_samples} samples for testing...")
            split_dataset = split_dataset.select(range(args.max_samples))
            print(f"Limited dataset size: {len(split_dataset):,} samples")
        
        # Shuffle dataset if it's the train split
        if split_name == "train":
            print(f"Shuffling {split_name} split with seed {args.shuffle_seed}")
            split_dataset = split_dataset.shuffle(seed=args.shuffle_seed)
        
        # Memory-optimized processing
        if len(split_dataset) <= 500:
            print(f"Small dataset detected. Using single process...")
            processed_split = split_dataset.map(
                prepare_dataset_with_processors,
                remove_columns=split_dataset.column_names,
                desc=f"Processing {split_name} split",
                writer_batch_size=args.writer_batch_size
            )
        else:
            print(f"Processing with {num_cpus} processes...")
            print(f"Using batch size: {args.batch_size}, writer batch size: {args.writer_batch_size}")
            
            # Force garbage collection before processing
            gc.collect()
            
            processed_split = split_dataset.map(
                prepare_dataset_with_processors,
                remove_columns=split_dataset.column_names,
                num_proc=num_cpus,
                batch_size=args.batch_size,
                writer_batch_size=args.writer_batch_size,
                desc=f"Processing {split_name} split"
            )
        
        print(f"Processed {split_name} split: {len(processed_split):,} samples")
        print(f"New columns: {processed_split.column_names}")
        
        # Validate processed data
        sample = processed_split[0]
        input_features_shape = sample["input_features"].shape if hasattr(sample["input_features"], 'shape') else len(sample["input_features"])
        labels_length = len(sample["labels"])
        print(f"Sample input_features shape: {input_features_shape}")
        print(f"Sample labels length: {labels_length}")
        
        # Validate mel bins in the processed data
        if hasattr(sample["input_features"], 'shape') and len(sample["input_features"].shape) >= 2:
            actual_mel_bins = sample["input_features"].shape[0]  # First dimension should be mel bins
        elif isinstance(sample["input_features"], list) and len(sample["input_features"]) > 0:
            actual_mel_bins = len(sample["input_features"])
        else:
            actual_mel_bins = "unknown"
        
        if isinstance(actual_mel_bins, int):
            if actual_mel_bins == expected_mel_bins:
                print(f"✓ Processed mel bins: {actual_mel_bins} (correct for {args.model_size})")
            else:
                print(f"⚠ Processed mel bins: {actual_mel_bins} (expected {expected_mel_bins} for {args.model_size})")
        else:
            print(f"Could not determine mel bins from processed data: {actual_mel_bins}")
        
        processed_dataset_dict[split_name] = processed_split
        
        # Force garbage collection after each split
        gc.collect()
        print(f"Memory cleanup completed for {split_name} split")
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output_dataset)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the preprocessed dataset
    print(f"\nSaving preprocessed dataset to {args.output_dataset}")
    processed_dataset_dict.save_to_disk(args.output_dataset)
    
    print("Dataset preprocessing completed successfully!")
    print(f"Preprocessed dataset saved to: {args.output_dataset}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Input dataset: {args.input_dataset}")
    print(f"Output dataset: {args.output_dataset}")
    print(f"Whisper model: {model_name}")
    print(f"Model size: {args.model_size}")
    print(f"Language: {args.language}")
    print(f"Task: {args.task}")
    print(f"Expected mel bins: {expected_mel_bins}")
    if args.model_size == "large-v3":
        print("Note: Used 128 mel bins (large-v3 architecture)")
    else:
        print("Note: Used 80 mel bins (standard Whisper architecture)")
    print(f"CPU cores used: {num_cpus}")
    print(f"Batch size: {args.batch_size}")
    print(f"Writer batch size: {args.writer_batch_size}")
    print(f"Memory per worker: {args.max_memory_per_worker}GB")
    print(f"Processed splits: {list(processed_dataset_dict.keys())}")
    for split_name, split_data in processed_dataset_dict.items():
        print(f"  {split_name}: {len(split_data):,} samples")
    print("=" * 60)