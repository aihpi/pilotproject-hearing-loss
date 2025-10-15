#!/usr/bin/env python3
"""
Download CommonVoice German dataset from HuggingFace.

This script downloads the CommonVoice German dataset and stores it in the
data/CommonVoiceDE directory for use in hearing loss research.
"""

import os
import sys
from pathlib import Path
from typing import Optional, List
from datasets import load_dataset
from huggingface_hub import login
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_commonvoice_german(
    output_dir: str = "data/CommonVoiceDE",
    version: str = "mozilla-foundation/common_voice_16_1",
    streaming: bool = False,
    splits: Optional[List[str]] = None,
    token: Optional[str] = None
):
    """
    Download CommonVoice German dataset.
    
    Args:
        output_dir (str): Directory to store the dataset
        version (str): HuggingFace dataset version identifier
        streaming (bool): Whether to use streaming mode
        splits (list): List of splits to download (default: all splits)
        token (str): HuggingFace access token for gated datasets
    """
    
    # Handle authentication for gated datasets
    if token:
        login(token=token)
        logger.info("Authenticated with HuggingFace Hub")
    elif os.getenv("HF_TOKEN"):
        login(token=os.getenv("HF_TOKEN"))
        logger.info("Authenticated with HuggingFace Hub using HF_TOKEN environment variable")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting download of CommonVoice German dataset...")
    logger.info(f"Output directory: {output_path.absolute()}")
    logger.info(f"Streaming mode: {streaming}")
    
    try:
        if streaming:
            logger.info("Loading dataset in streaming mode...")
            dataset = load_dataset(
                version,
                "de",  # German language code
                streaming=True,
                trust_remote_code=True
            )
            
            logger.info("Dataset loaded in streaming mode. Data will be processed on-demand.")
            logger.info("Available splits:")
            for split_name in dataset.keys():
                logger.info(f"  - {split_name}")
            
            # Note: In streaming mode, data is not actually downloaded to disk
            # It's accessed on-demand from HuggingFace servers
            logger.info("Note: Streaming mode doesn't download files to disk.")
            logger.info("Data will be fetched on-demand during processing.")
            
        else:
            logger.info("Loading dataset with full download...")
            
            # Set custom cache directory to our data folder
            cache_dir = output_path / "cache"
            
            if splits is None:
                # Download all splits
                dataset = load_dataset(
                    version,
                    "de",  # German language code
                    cache_dir=str(cache_dir),
                    trust_remote_code=True
                )
            else:
                # Download specific splits
                dataset = {}
                for split in splits:
                    logger.info(f"Downloading split: {split}")
                    dataset[split] = load_dataset(
                        version,
                        "de",
                        split=split,
                        cache_dir=str(cache_dir),
                        trust_remote_code=True
                    )
            
            logger.info("Dataset downloaded successfully!")
            logger.info(f"Cache location: {cache_dir}")
            
            # Display dataset information
            if hasattr(dataset, 'keys'):
                logger.info("Available splits:")
                for split_name in dataset.keys():
                    split_data = dataset[split_name]
                    logger.info(f"  - {split_name}: {len(split_data)} samples")
            else:
                # Single split case
                logger.info(f"Downloaded split: {len(dataset)} samples")
        
        # Save dataset info
        info_file = output_path / "dataset_info.txt"
        with open(info_file, 'w') as f:
            f.write(f"CommonVoice German Dataset\n")
            f.write(f"========================\n\n")
            f.write(f"Version: {version}\n")
            f.write(f"Language: German (de)\n")
            f.write(f"Streaming mode: {streaming}\n")
            f.write(f"Download date: {__import__('datetime').datetime.now()}\n\n")
            
            if hasattr(dataset, 'keys'):
                f.write("Available splits:\n")
                for split_name in dataset.keys():
                    if streaming:
                        f.write(f"  - {split_name}: (streaming - size unknown)\n")
                    else:
                        split_data = dataset[split_name]
                        f.write(f"  - {split_name}: {len(split_data)} samples\n")
        
        logger.info(f"Dataset information saved to: {info_file}")
        return dataset
        
    except Exception as e:
        logger.error(f"Error downloading dataset: {e}")
        sys.exit(1)

def main():
    """Main function to handle command line execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download CommonVoice German dataset")
    parser.add_argument(
        "--output-dir", 
        default="data/CommonVoiceDE",
        help="Output directory for the dataset (default: data/CommonVoiceDE)"
    )
    parser.add_argument(
        "--streaming", 
        action="store_true",
        help="Use streaming mode (data accessed on-demand, not downloaded)"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "validation", "test"],
        help="Specific splits to download (default: all splits)"
    )
    parser.add_argument(
        "--token",
        help="HuggingFace access token for gated datasets (or set HF_TOKEN environment variable)"
    )
    parser.add_argument(
        "--version",
        default="mozilla-foundation/common_voice_16_1",
        help="HuggingFace dataset version (default: common_voice_16_1)"
    )
    
    args = parser.parse_args()
    
    # Download the dataset
    dataset = download_commonvoice_german(
        output_dir=args.output_dir,
        version=args.version,
        streaming=args.streaming,
        splits=args.splits,
        token=args.token
    )
    
    logger.info("Download completed successfully!")

if __name__ == "__main__":
    main()