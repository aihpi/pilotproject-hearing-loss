#!/usr/bin/env python3
"""
Generate semantic matrices (S-matrices) from fastText embeddings.

This script pre-computes S-matrices for all unique words across all conditions and splits.
The S-matrices are shared across hearing loss conditions since semantic representations
are condition-independent.
"""

import argparse
import logging
from pathlib import Path
import sys
import yaml

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

import fasttext as ft
from discriminative_lexicon_model.mapping import gen_smat
from utils import build_word_list, filter_common_words

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_fasttext_model(model_path: Path):
    """
    Load pre-trained fastText model.

    Args:
        model_path: Path to .bin file

    Returns:
        Loaded fastText model
    """
    logger.info(f"Loading fastText model from {model_path}")
    model = ft.load_model(str(model_path))
    logger.info(f"Model loaded successfully. Vocabulary size: {len(model.words)}")
    return model


def collect_words_from_conditions(config: dict, split: str, limit: int = None) -> list:
    """
    Collect unique words from all three hearing loss conditions for a given split.

    Args:
        config: Configuration dictionary
        split: 'train', 'test', or 'validation'
        limit: Limit number of files to process per condition (for testing)

    Returns:
        Sorted list of unique words appearing in all conditions
    """
    logger.info(f"Collecting words from {split} split across all conditions...")
    if limit is not None:
        logger.info(f"  (Limited to {limit} files per condition for testing)")

    word_lists = []

    for condition in ['normal', 'lfloss', 'hfloss']:
        audio_dir = Path(config['data']['audio_dirs'][condition])
        try:
            words = build_word_list(audio_dir, split, limit=limit)
            word_lists.append(words)
            logger.info(f"  {condition}: {len(words)} unique words")
        except Exception as e:
            logger.error(f"  Failed to collect words from {condition}/{split}: {e}")
            raise

    # Use words that appear in all three conditions to ensure consistency
    common_words = filter_common_words(word_lists, min_occurrences=3)
    logger.info(f"Total unique words appearing in all conditions: {len(common_words)}")

    return common_words


def generate_smatrix(words: list, ft_model, split: str, output_dir: Path):
    """
    Generate S-matrix from fastText embeddings and save to file.

    Args:
        words: List of words
        ft_model: Loaded fastText model
        split: Split name ('train', 'test', or 'validation')
        output_dir: Directory to save S-matrix

    Returns:
        Generated S-matrix as xarray DataArray
    """
    logger.info(f"Generating S-matrix for {split} split ({len(words)} words)...")

    # Generate S-matrix using discriminative_lexicon_model
    smat = gen_smat(words, embed=ft_model)

    logger.info(f"S-matrix shape: {smat.shape}")
    logger.info(f"S-matrix dimensions: {smat.dims}")

    # Save to NetCDF file
    output_path = output_dir / f"{split}_smatrix.nc"
    logger.info(f"Saving S-matrix to {output_path}")
    smat.to_netcdf(output_path)

    # Also save word list as text file for reference
    words_path = output_dir / f"{split}_words.txt"
    logger.info(f"Saving word list to {words_path}")
    with open(words_path, 'w') as f:
        for word in words:
            f.write(f"{word}\n")

    return smat


def main():
    parser = argparse.ArgumentParser(
        description="Generate semantic matrices (S-matrices) from fastText embeddings"
    )
    parser.add_argument(
        '--config',
        type=Path,
        default='configs/fmatrix_training_config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['train', 'test', 'validation'],
        help='Splits to process (default: train test validation)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (default: from config or data/smatrices)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of files to process per condition (for testing)'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Determine output directory
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        output_dir = Path('data/smatrices')

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("S-Matrix Generation")
    logger.info("=" * 60)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Splits: {args.splits}")
    if args.limit is not None:
        logger.info(f"Limit: {args.limit} files per condition (TESTING MODE)")
    logger.info("=" * 60)

    # Load fastText model
    fasttext_model_path = Path(config['data']['fasttext_model'])
    ft_model = load_fasttext_model(fasttext_model_path)

    # Process each split
    for split in args.splits:
        logger.info("")
        logger.info(f"Processing {split} split...")
        logger.info("-" * 60)

        try:
            # Collect words from all conditions
            words = collect_words_from_conditions(config, split, limit=args.limit)

            # Generate and save S-matrix
            smat = generate_smatrix(words, ft_model, split, output_dir)

            logger.info(f"✓ Successfully generated S-matrix for {split} split")

        except Exception as e:
            logger.error(f"✗ Failed to process {split} split: {e}")
            raise

    logger.info("")
    logger.info("=" * 60)
    logger.info("S-Matrix generation complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
