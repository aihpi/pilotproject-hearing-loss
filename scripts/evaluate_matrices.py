#!/usr/bin/env python3
"""
Evaluate trained F-matrices and G-matrices on test/validation sets and compare across conditions.

F-matrix: form -> semantics (comprehension)
G-matrix: semantics -> form (production)
"""

import argparse
import logging
import json
import sys
from pathlib import Path
import yaml
import csv

import numpy as np
import xarray as xr
from tqdm import tqdm
from scipy.stats import pearsonr

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from utils import (
    get_audio_files,
    load_and_generate_spectrogram,
    flatten_spectrogram,
    extract_word_from_filename,
)

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


def load_matrix(matrix_path: Path, matrix_type: str = 'F') -> xr.DataArray:
    """
    Load F-matrix or G-matrix from file.

    Args:
        matrix_path: Path to matrix file
        matrix_type: 'F' or 'G'

    Returns:
        Matrix as xarray DataArray
    """
    logger.info(f"Loading {matrix_type}-matrix from {matrix_path}")

    if matrix_path.suffix == '.nc':
        mat = xr.open_dataarray(matrix_path)
    elif '.csv' in matrix_path.suffixes:
        # For CSV files, need to reconstruct xarray
        import pandas as pd
        if matrix_path.suffix == '.gz':
            df = pd.read_csv(matrix_path, index_col=0, compression='gzip')
        else:
            df = pd.read_csv(matrix_path, index_col=0)

        # Determine dimensions based on matrix type
        if matrix_type == 'F':
            dims = ('cues', 'semantics')
        else:  # G-matrix
            dims = ('semantics', 'cues')

        mat = xr.DataArray(
            df.values,
            dims=dims,
            coords={dims[0]: df.index.values, dims[1]: df.columns.values}
        )
    else:
        raise ValueError(f"Unsupported file format: {matrix_path}")

    logger.info(f"{matrix_type}-matrix loaded: shape={mat.shape}, dims={mat.dims}")
    return mat


def load_smatrix(smatrix_path: Path) -> xr.DataArray:
    """Load S-matrix from NetCDF file."""
    logger.info(f"Loading S-matrix from {smatrix_path}")
    smat = xr.open_dataarray(smatrix_path)
    logger.info(f"S-matrix loaded: shape={smat.shape}, dims={smat.dims}")
    return smat


def build_cmatrix_from_audio_files(
    audio_files: list,
    spec_params: dict,
    s_matrix: xr.DataArray,
    limit: int = None
) -> tuple:
    """
    Build C-matrix by generating spectrograms for all audio files.

    Args:
        audio_files: List of audio file paths
        spec_params: Spectrogram parameters
        s_matrix: S-matrix to check word availability
        limit: Limit number of files to process

    Returns:
        Tuple of (C-matrix, S-matrix subset, word list)
    """
    n_cues = spec_params['n_mels'] * spec_params['max_time_frames']

    if limit is not None:
        audio_files = audio_files[:limit]

    logger.info(f"Building C-matrix from {len(audio_files)} audio files...")

    c_vectors = []
    s_vectors = []
    words = []
    failed_count = 0

    for audio_file in tqdm(audio_files, desc="Generating spectrograms"):
        try:
            # Extract word
            word = extract_word_from_filename(audio_file.name)

            # Skip if word not in S-matrix
            if word not in s_matrix.coords['word'].values:
                failed_count += 1
                continue

            # Generate spectrogram
            spec = load_and_generate_spectrogram(audio_file, **spec_params)
            c_vec = flatten_spectrogram(spec)

            # Get S-vector
            s_vec = s_matrix.loc[word, :].values

            c_vectors.append(c_vec)
            s_vectors.append(s_vec)
            words.append(word)

        except Exception as e:
            logger.warning(f"Failed to process {audio_file.name}: {e}")
            failed_count += 1
            continue

    if failed_count > 0:
        logger.warning(f"Failed to process {failed_count} files")

    # Stack into matrices
    c_matrix = np.array(c_vectors)
    s_matrix_subset = np.array(s_vectors)

    logger.info(f"C-matrix shape: {c_matrix.shape}")
    logger.info(f"S-matrix subset shape: {s_matrix_subset.shape}")

    return c_matrix, s_matrix_subset, words


def compute_prediction_metrics(s_true: np.ndarray, s_pred: np.ndarray) -> dict:
    """
    Compute evaluation metrics between true and predicted semantics.

    Args:
        s_true: Ground truth S-matrix (n_samples, n_semantics)
        s_pred: Predicted S-matrix (n_samples, n_semantics)

    Returns:
        Dictionary with metrics
    """
    # Cosine similarity per sample
    s_true_norm = np.linalg.norm(s_true, axis=1, keepdims=True)
    s_pred_norm = np.linalg.norm(s_pred, axis=1, keepdims=True)

    # Avoid division by zero
    s_true_norm = np.where(s_true_norm == 0, 1, s_true_norm)
    s_pred_norm = np.where(s_pred_norm == 0, 1, s_pred_norm)

    cosine_similarities = np.sum(s_true * s_pred, axis=1) / (
        s_true_norm.flatten() * s_pred_norm.flatten()
    )

    # Pearson correlation per semantic dimension
    correlations = []
    for dim in range(s_true.shape[1]):
        if np.std(s_true[:, dim]) > 0 and np.std(s_pred[:, dim]) > 0:
            corr, _ = pearsonr(s_true[:, dim], s_pred[:, dim])
            correlations.append(corr)

    # MSE
    mse = np.mean((s_true - s_pred) ** 2)

    metrics = {
        'mean_cosine_similarity': float(np.mean(cosine_similarities)),
        'std_cosine_similarity': float(np.std(cosine_similarities)),
        'median_cosine_similarity': float(np.median(cosine_similarities)),
        'mean_pearson_correlation': float(np.mean(correlations)) if correlations else 0.0,
        'mse': float(mse),
        'rmse': float(np.sqrt(mse)),
        'per_sample_cosine_similarities': cosine_similarities.tolist()
    }

    return metrics


def evaluate_fmatrix(
    f_matrix: xr.DataArray,
    condition: str,
    split: str,
    config: dict,
    limit: int = None
) -> dict:
    """
    Evaluate F-matrix on a data split.

    Args:
        f_matrix: Trained F-matrix
        condition: Condition name
        split: 'test' or 'validation'
        config: Configuration dictionary
        limit: Limit number of files to evaluate

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating {condition} on {split} split...")

    # Load S-matrix for this split
    smatrix_path = Path(f'data/smatrices/{split}_smatrix.nc')
    s_matrix = load_smatrix(smatrix_path)

    # Get audio files
    audio_dir = Path(config['data']['audio_dirs'][condition])
    audio_files = get_audio_files(audio_dir, split)
    logger.info(f"Found {len(audio_files)} audio files in {split} split")

    # Build C-matrix and corresponding S-matrix subset
    spec_params = config['spectrogram']
    c_matrix, s_true, words = build_cmatrix_from_audio_files(
        audio_files,
        spec_params,
        s_matrix,
        limit=limit
    )

    # Compute predictions
    logger.info("Computing predictions...")
    s_pred = c_matrix @ f_matrix.values

    # Compute metrics
    logger.info("Computing evaluation metrics...")
    metrics = compute_prediction_metrics(s_true, s_pred)

    # Add metadata
    metrics['condition'] = condition
    metrics['split'] = split
    metrics['n_samples'] = len(words)

    logger.info(f"Results for {condition}/{split}:")
    logger.info(f"  Mean cosine similarity: {metrics['mean_cosine_similarity']:.4f}")
    logger.info(f"  Mean Pearson correlation: {metrics['mean_pearson_correlation']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")

    return metrics


def evaluate_gmatrix(
    g_matrix: xr.DataArray,
    condition: str,
    split: str,
    config: dict,
    limit: int = None
) -> dict:
    """
    Evaluate G-matrix on a data split (production task).

    Args:
        g_matrix: Trained G-matrix
        condition: Condition name
        split: 'test' or 'validation'
        config: Configuration dictionary
        limit: Limit number of files to evaluate

    Returns:
        Dictionary with evaluation metrics
    """
    logger.info(f"Evaluating G-matrix for {condition} on {split} split (production task)...")

    # Load S-matrix for this split
    smatrix_path = Path(f'data/smatrices/{split}_smatrix.nc')
    s_matrix = load_smatrix(smatrix_path)

    # Get audio files
    audio_dir = Path(config['data']['audio_dirs'][condition])
    audio_files = get_audio_files(audio_dir, split)
    logger.info(f"Found {len(audio_files)} audio files in {split} split")

    # Build C-matrix and corresponding S-matrix subset
    spec_params = config['spectrogram']
    c_true, s_matrix_subset, words = build_cmatrix_from_audio_files(
        audio_files,
        spec_params,
        s_matrix,
        limit=limit
    )

    # Compute predictions (production: semantics -> form)
    logger.info("Computing predictions (production)...")
    c_pred = s_matrix_subset @ g_matrix.values

    # Compute metrics
    logger.info("Computing evaluation metrics...")
    metrics = compute_prediction_metrics(c_true, c_pred)

    # Add metadata
    metrics['condition'] = condition
    metrics['split'] = split
    metrics['n_samples'] = len(words)
    metrics['task'] = 'production'

    logger.info(f"Results for {condition}/{split} (production):")
    logger.info(f"  Mean cosine similarity: {metrics['mean_cosine_similarity']:.4f}")
    logger.info(f"  Mean Pearson correlation: {metrics['mean_pearson_correlation']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")

    return metrics


def compare_matrices(
    matrices: dict,
    matrix_type: str,
    output_dir: Path
) -> dict:
    """
    Compare F-matrices or G-matrices across conditions.

    Args:
        matrices: Dictionary mapping condition to matrix
        matrix_type: 'F' or 'G'
        output_dir: Directory to save comparison results

    Returns:
        Dictionary with comparison metrics
    """
    logger.info(f"Comparing {matrix_type}-matrices across conditions...")

    conditions = list(matrices.keys())
    comparison = {}

    # Compute pairwise Frobenius norm differences
    for i, cond1 in enumerate(conditions):
        for cond2 in conditions[i+1:]:
            diff = matrices[cond1] - matrices[cond2]
            frobenius_norm = float(np.sqrt((diff ** 2).sum()))

            pair_key = f"{cond1}_vs_{cond2}"
            comparison[pair_key] = {
                'frobenius_norm': frobenius_norm,
                'normalized_frobenius_norm': frobenius_norm / matrices[cond1].size
            }

            logger.info(f"  {pair_key}: Frobenius norm = {frobenius_norm:.2f}")

    # Save comparison results
    comparison_file = output_dir / f'{matrix_type.lower()}matrix_comparison.json'
    with open(comparison_file, 'w') as f:
        json.dump(comparison, f, indent=2)

    logger.info(f"Comparison results saved to {comparison_file}")

    return comparison


def compare_fmatrices(
    f_matrices: dict,
    output_dir: Path
) -> dict:
    """
    Compare F-matrices across conditions (wrapper for backwards compatibility).

    Args:
        f_matrices: Dictionary mapping condition to F-matrix
        output_dir: Directory to save comparison results

    Returns:
        Dictionary with comparison metrics
    """
    return compare_matrices(f_matrices, 'F', output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained F-matrices and G-matrices"
    )
    parser.add_argument(
        '--config',
        type=Path,
        default='configs/fmatrix_training_config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--conditions',
        nargs='+',
        default=['normal', 'lfloss', 'hfloss'],
        help='Conditions to evaluate'
    )
    parser.add_argument(
        '--splits',
        nargs='+',
        default=['test', 'validation'],
        help='Splits to evaluate on'
    )
    parser.add_argument(
        '--fmatrix-dir',
        type=Path,
        default=None,
        help='Directory containing F-matrix checkpoints (default: from config)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory for results (default: from config)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of evaluation files (for testing)'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Determine directories
    base_dir = Path(config['output']['base_dir'])
    if args.fmatrix_dir is None:
        fmatrix_dir = base_dir / config['output']['checkpoints_dir']
    else:
        fmatrix_dir = args.fmatrix_dir

    if args.output_dir is None:
        output_dir = base_dir / 'evaluation'
    else:
        output_dir = args.output_dir

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("F-Matrix and G-Matrix Evaluation")
    logger.info("=" * 60)
    logger.info(f"Conditions: {args.conditions}")
    logger.info(f"Splits: {args.splits}")
    logger.info(f"Matrix directory: {fmatrix_dir}")
    logger.info(f"Output directory: {output_dir}")
    if args.limit is not None:
        logger.info(f"Limit: {args.limit} files (TESTING MODE)")
    logger.info("=" * 60)

    # Load F-matrices and G-matrices for all conditions
    f_matrices = {}
    g_matrices = {}
    for condition in args.conditions:
        # Find latest checkpoints for this condition
        condition_dir = fmatrix_dir / condition
        f_checkpoints = sorted(condition_dir.glob("fmatrix_*.csv.gz"))
        g_checkpoints = sorted(condition_dir.glob("gmatrix_*.csv.gz"))

        if not f_checkpoints:
            logger.error(f"No F-matrix checkpoints found for {condition} in {condition_dir}")
        else:
            latest_f_checkpoint = f_checkpoints[-1]
            f_matrices[condition] = load_matrix(latest_f_checkpoint, 'F')

        if not g_checkpoints:
            logger.error(f"No G-matrix checkpoints found for {condition} in {condition_dir}")
        else:
            latest_g_checkpoint = g_checkpoints[-1]
            g_matrices[condition] = load_matrix(latest_g_checkpoint, 'G')

    # Evaluate each condition on each split
    all_results = []

    for condition in args.conditions:
        for split in args.splits:
            # Evaluate F-matrix (comprehension)
            if condition in f_matrices:
                try:
                    metrics = evaluate_fmatrix(
                        f_matrices[condition],
                        condition,
                        split,
                        config,
                        limit=args.limit
                    )
                    metrics['task'] = 'comprehension'
                    all_results.append(metrics)

                except Exception as e:
                    logger.error(f"Failed to evaluate F-matrix for {condition} on {split}: {e}")

            # Evaluate G-matrix (production)
            if condition in g_matrices:
                try:
                    metrics = evaluate_gmatrix(
                        g_matrices[condition],
                        condition,
                        split,
                        config,
                        limit=args.limit
                    )
                    all_results.append(metrics)

                except Exception as e:
                    logger.error(f"Failed to evaluate G-matrix for {condition} on {split}: {e}")

    # Save evaluation results
    results_file = output_dir / 'evaluation_results.json'
    logger.info(f"\nSaving evaluation results to {results_file}")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Save summary CSV
    summary_file = output_dir / 'evaluation_summary.csv'
    logger.info(f"Saving summary to {summary_file}")
    with open(summary_file, 'w', newline='') as f:
        fieldnames = ['condition', 'split', 'task', 'n_samples', 'mean_cosine_similarity',
                     'mean_pearson_correlation', 'rmse']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in all_results:
            writer.writerow({k: result.get(k, '') for k in fieldnames})

    # Compare F-matrices across conditions
    if len(f_matrices) > 1:
        logger.info("\nComparing F-matrices (comprehension)...")
        comparison_f = compare_fmatrices(f_matrices, output_dir)

    # Compare G-matrices across conditions
    if len(g_matrices) > 1:
        logger.info("\nComparing G-matrices (production)...")
        comparison_g = compare_matrices(g_matrices, 'G', output_dir)

    logger.info("=" * 60)
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
