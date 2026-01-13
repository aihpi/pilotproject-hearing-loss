#!/usr/bin/env python3
"""
Train F-matrices and G-matrices using incremental learning for hearing loss conditions.

This script implements the core training loop for estimating:
- F-matrices (form-to-semantic weight matrices): C @ F = S
- G-matrices (semantic-to-form weight matrices): S @ G = C

Both matrices are trained via incremental learning on audio spectrograms mapped to fastText embeddings.
"""

import argparse
import logging
import sys
import random
from pathlib import Path
import yaml
import csv
from datetime import datetime

import numpy as np
import xarray as xr
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from discriminative_lexicon_model.mapping import incremental_learning, save_mat, load_mat
from utils import (
    get_audio_files,
    create_training_order,
    load_and_generate_spectrogram,
    flatten_spectrogram,
    extract_word_from_filename,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(process)d] %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_random_seeds(seed: int):
    """
    Set random seeds for reproducibility across all libraries.

    Args:
        seed: Random seed value
    """
    logger.info(f"Setting random seeds to {seed} for reproducibility")

    # Python's built-in random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

            # Ensure deterministic behavior on GPU
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            logger.info("PyTorch GPU seeds set and deterministic mode enabled")
        else:
            logger.info("PyTorch CPU seed set")
    except ImportError:
        logger.info("PyTorch not available, skipping PyTorch seed initialization")


def find_latest_checkpoint(
    checkpoint_dir: Path,
    matrix_type: str,
    condition: str
) -> tuple:
    """
    Find the latest checkpoint for a given matrix type and condition.

    Args:
        checkpoint_dir: Directory containing checkpoints
        matrix_type: 'fmatrix' or 'gmatrix'
        condition: Condition name (normal, lfloss, hfloss)

    Returns:
        Tuple of (checkpoint_path, iteration_number) or (None, 0) if not found
    """
    import re
    import glob

    condition_dir = checkpoint_dir / condition
    if not condition_dir.exists():
        logger.info(f"No checkpoint directory found at {condition_dir}")
        return None, 0

    # Find all checkpoints matching the pattern
    pattern = str(condition_dir / f"{matrix_type}_{condition}_iter*.csv.gz")
    checkpoint_files = glob.glob(pattern)

    if not checkpoint_files:
        logger.info(f"No {matrix_type} checkpoints found for condition '{condition}'")
        return None, 0

    # Extract iteration numbers and find the latest
    latest_iteration = 0
    latest_checkpoint = None

    for checkpoint_file in checkpoint_files:
        match = re.search(r'iter(\d+)', checkpoint_file)
        if match:
            iteration = int(match.group(1))
            if iteration > latest_iteration:
                latest_iteration = iteration
                latest_checkpoint = Path(checkpoint_file)

    if latest_checkpoint:
        logger.info(f"Found latest {matrix_type} checkpoint: {latest_checkpoint} (iteration {latest_iteration})")

    return latest_checkpoint, latest_iteration


def load_smatrix(smatrix_path: Path) -> xr.DataArray:
    """
    Load pre-computed S-matrix from NetCDF file.

    Args:
        smatrix_path: Path to .nc file

    Returns:
        S-matrix as xarray DataArray
    """
    logger.info(f"Loading S-matrix from {smatrix_path}")
    smat = xr.open_dataarray(smatrix_path)
    logger.info(f"S-matrix loaded: shape={smat.shape}, dims={smat.dims}")
    return smat


def create_cvector_from_audio(
    audio_path: Path,
    word: str,
    spec_params: dict,
    n_cues: int = 50048
) -> xr.DataArray:
    """
    Generate C-vector (form representation) from audio file.

    Args:
        audio_path: Path to audio file
        word: Word label (for xarray coordinate)
        spec_params: Spectrogram generation parameters
        n_cues: Number of cues (flattened spectrogram dimension)

    Returns:
        C-vector as xarray DataArray with shape (1, n_cues)
    """
    # Generate spectrogram
    spec = load_and_generate_spectrogram(audio_path, **spec_params)

    # Flatten to 1D vector
    c_vec = flatten_spectrogram(spec).reshape(1, -1)

    # Create xarray DataArray with coordinates
    c_vec_xr = xr.DataArray(
        c_vec,
        dims=('word', 'cues'),
        coords={
            'word': [word],
            'cues': [f'C{i:05d}' for i in range(n_cues)]
        }
    )

    return c_vec_xr


def save_training_order(
    audio_files: list,
    condition: str,
    metrics_dir: Path
) -> Path:
    """
    Save the training order to a CSV file for full traceability.

    Args:
        audio_files: List of audio files in training order
        condition: Condition name
        metrics_dir: Metrics directory

    Returns:
        Path to saved training order file
    """
    order_file = metrics_dir / f'training_order_{condition}.csv'
    logger.info(f"Saving training order to {order_file}")

    with open(order_file, 'w', newline='') as f:
        fieldnames = ['iteration', 'filename', 'word']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i, audio_file in enumerate(audio_files):
            word = extract_word_from_filename(audio_file.name)
            writer.writerow({
                'iteration': i + 1,
                'filename': audio_file.name,
                'word': word
            })

    logger.info(f"Training order saved: {len(audio_files)} samples")
    return order_file


def checkpoint_matrix(
    matrix: xr.DataArray,
    matrix_type: str,
    condition: str,
    iteration: int,
    checkpoint_dir: Path
) -> Path:
    """
    Save F-matrix or G-matrix checkpoint.

    Args:
        matrix: Current F-matrix or G-matrix
        matrix_type: 'fmatrix' or 'gmatrix'
        condition: Condition name
        iteration: Current training iteration
        checkpoint_dir: Checkpoint directory

    Returns:
        Path to saved checkpoint
    """
    checkpoint_path = checkpoint_dir / f"{matrix_type}_{condition}_iter{iteration:07d}.csv.gz"
    logger.info(f"Saving {matrix_type} checkpoint to {checkpoint_path}")

    # Save as compressed CSV
    # First save to uncompressed CSV, then compress
    temp_path = checkpoint_dir / f"{matrix_type}_{condition}_iter{iteration:07d}.csv"
    save_mat(matrix, temp_path)

    # Compress the file
    import gzip
    import shutil
    with open(temp_path, 'rb') as f_in:
        with gzip.open(checkpoint_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Remove temporary uncompressed file
    temp_path.unlink()

    return checkpoint_path


def track_learning_metrics(
    iteration: int,
    f_matrix: xr.DataArray,
    g_matrix: xr.DataArray,
    c_vector: xr.DataArray,
    s_vector: xr.DataArray,
    metrics_file: Path
) -> dict:
    """
    Compute and log learning metrics for current iteration.

    Args:
        iteration: Current iteration number
        f_matrix: Current F-matrix
        g_matrix: Current G-matrix
        c_vector: Current C-vector (form input)
        s_vector: Current S-vector (semantic target)
        metrics_file: CSV file to append metrics to

    Returns:
        Dictionary with metrics for this iteration
    """
    # Compute F-matrix prediction (comprehension: form -> semantics)
    s_pred = c_vector @ f_matrix
    error_f = s_vector - s_pred
    prediction_error_f = float(np.sqrt((error_f ** 2).sum()))

    # Compute F-matrix cosine similarity
    s_vec_norm = np.sqrt((s_vector ** 2).sum())
    s_pred_norm = np.sqrt((s_pred ** 2).sum())

    if s_vec_norm > 0 and s_pred_norm > 0:
        cosine_sim_f = float((s_vector * s_pred).sum() / (s_vec_norm * s_pred_norm))
    else:
        cosine_sim_f = 0.0

    # Compute G-matrix prediction (production: semantics -> form)
    c_pred = s_vector @ g_matrix
    error_g = c_vector - c_pred
    prediction_error_g = float(np.sqrt((error_g ** 2).sum()))

    # Compute G-matrix cosine similarity
    c_vec_norm = np.sqrt((c_vector ** 2).sum())
    c_pred_norm = np.sqrt((c_pred ** 2).sum())

    if c_vec_norm > 0 and c_pred_norm > 0:
        cosine_sim_g = float((c_vector * c_pred).sum() / (c_vec_norm * c_pred_norm))
    else:
        cosine_sim_g = 0.0

    metrics = {
        'iteration': iteration,
        'f_prediction_error': prediction_error_f,
        'f_cosine_similarity': cosine_sim_f,
        'g_prediction_error': prediction_error_g,
        'g_cosine_similarity': cosine_sim_g,
        'timestamp': datetime.now().isoformat()
    }

    # Append to CSV file
    file_exists = metrics_file.exists()
    with open(metrics_file, 'a', newline='') as f:
        fieldnames = ['iteration', 'f_prediction_error', 'f_cosine_similarity',
                     'g_prediction_error', 'g_cosine_similarity', 'timestamp']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(metrics)

    return metrics


def incremental_train_matrices(
    condition: str,
    config: dict,
    audio_files: list,
    s_matrix: xr.DataArray,
    output_dir: Path,
    resume_from_f: Path = None,
    resume_from_g: Path = None
) -> tuple:
    """
    Train F-matrix and G-matrix using incremental learning for a single condition.

    Args:
        condition: 'normal', 'lfloss', or 'hfloss'
        config: Training configuration dictionary
        audio_files: List of training audio files (pre-shuffled)
        s_matrix: Pre-computed S-matrix for all words
        output_dir: Directory to save checkpoints and metrics
        resume_from_f: Optional F-matrix checkpoint path to resume from
        resume_from_g: Optional G-matrix checkpoint path to resume from

    Returns:
        Tuple of (Final F-matrix, Final G-matrix) as xarray DataArrays
    """
    # Create output directories
    checkpoint_dir = output_dir / config['output']['checkpoints_dir'] / condition
    metrics_dir = output_dir / config['output']['metrics_dir'] / condition
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Metrics file
    metrics_file = metrics_dir / 'training_metrics.csv'

    # Training parameters
    learning_rate = config['training']['learning_rate']
    checkpoint_interval = config['training']['checkpoint_interval']
    metrics_interval = config['training']['metrics_interval']
    backend = config['training']['backend']
    device = config['training']['device']

    # Spectrogram parameters
    spec_params = config['spectrogram']

    # Number of cues (flattened spectrogram size)
    n_cues = spec_params['n_mels'] * spec_params['max_time_frames']

    logger.info(f"Training configuration:")
    logger.info(f"  Condition: {condition}")
    logger.info(f"  Training files: {len(audio_files)}")
    logger.info(f"  Learning rate: {learning_rate}")
    logger.info(f"  Checkpoint interval: {checkpoint_interval}")
    logger.info(f"  Metrics interval: {metrics_interval}")
    logger.info(f"  Backend: {backend}")
    logger.info(f"  Device: {device}")
    logger.info(f"  N_cues: {n_cues}")

    # Initialize F-matrix and G-matrix
    f_matrix = None
    g_matrix = None
    start_iteration = 0

    # Resume from checkpoints if specified
    if resume_from_f is not None:
        logger.info(f"Resuming F-matrix from checkpoint: {resume_from_f}")
        f_matrix = load_mat(resume_from_f)
        # Extract iteration number from filename
        import re
        match = re.search(r'iter(\d+)', resume_from_f.name)
        if match:
            start_iteration = int(match.group(1))

    if resume_from_g is not None:
        logger.info(f"Resuming G-matrix from checkpoint: {resume_from_g}")
        g_matrix = load_mat(resume_from_g)

    # Log resume status
    total_files = len(audio_files)
    remaining_files = total_files - start_iteration

    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAINING PROGRESS STATUS")
    logger.info("=" * 60)
    if start_iteration > 0:
        logger.info(f"  Mode: RESUMING FROM CHECKPOINT")
        logger.info(f"  Words already processed: {start_iteration:,}")
        logger.info(f"  Words remaining: {remaining_files:,}")
        logger.info(f"  Total words: {total_files:,}")
        logger.info(f"  Progress: {start_iteration / total_files * 100:.2f}%")
    else:
        logger.info(f"  Mode: STARTING FRESH")
        logger.info(f"  Total words to process: {total_files:,}")
    logger.info("=" * 60)
    logger.info("")

    # Save training order for full traceability (only if starting fresh)
    if start_iteration == 0:
        save_training_order(audio_files, condition, metrics_dir)

    # Training loop
    failed_files = []
    progress_bar = tqdm(
        enumerate(audio_files[start_iteration:], start=start_iteration),
        total=len(audio_files),
        initial=start_iteration,
        desc=f"Training {condition}"
    )

    for i, audio_file in progress_bar:
        try:
            # Extract word from filename
            word = extract_word_from_filename(audio_file.name)

            # Check if word exists in S-matrix
            if word not in s_matrix.coords['word'].values:
                logger.warning(f"Word '{word}' not in S-matrix, skipping file {audio_file.name}")
                failed_files.append((audio_file.name, f"Word not in S-matrix"))
                continue

            # Generate C-vector from audio
            c_vec_xr = create_cvector_from_audio(audio_file, word, spec_params, n_cues)

            # Get corresponding S-vector
            s_vec_xr = s_matrix.loc[[word], :]

            # Update F-matrix (comprehension: form -> semantics)
            f_matrix = incremental_learning(
                rows=[word],
                cue_matrix=c_vec_xr,
                out_matrix=s_vec_xr,
                learning_rate=learning_rate,
                weight_matrix=f_matrix,
                return_intermediate_weights=False
            )

            # Update G-matrix (production: semantics -> form)
            g_matrix = incremental_learning(
                rows=[word],
                cue_matrix=s_vec_xr,
                out_matrix=c_vec_xr,
                learning_rate=learning_rate,
                weight_matrix=g_matrix,
                return_intermediate_weights=False
            )

            # Checkpoint periodically
            if (i + 1) % checkpoint_interval == 0:
                checkpoint_matrix(f_matrix, 'fmatrix', condition, i + 1, checkpoint_dir)
                checkpoint_matrix(g_matrix, 'gmatrix', condition, i + 1, checkpoint_dir)

            # Track metrics periodically
            if (i + 1) % metrics_interval == 0:
                metrics = track_learning_metrics(i + 1, f_matrix, g_matrix, c_vec_xr, s_vec_xr, metrics_file)
                progress_bar.set_postfix({
                    'f_err': f"{metrics['f_prediction_error']:.4f}",
                    'f_cos': f"{metrics['f_cosine_similarity']:.4f}",
                    'g_err': f"{metrics['g_prediction_error']:.4f}",
                    'g_cos': f"{metrics['g_cosine_similarity']:.4f}"
                })

        except Exception as e:
            logger.error(f"Failed to process {audio_file.name}: {e}")
            failed_files.append((audio_file.name, str(e)))
            continue

    progress_bar.close()

    # Save final F-matrix and G-matrix
    final_checkpoint_f = checkpoint_matrix(f_matrix, 'fmatrix', condition, len(audio_files), checkpoint_dir)
    final_checkpoint_g = checkpoint_matrix(g_matrix, 'gmatrix', condition, len(audio_files), checkpoint_dir)
    logger.info(f"Final F-matrix saved to {final_checkpoint_f}")
    logger.info(f"Final G-matrix saved to {final_checkpoint_g}")

    # Save failed files log
    if failed_files:
        failed_log = metrics_dir / 'failed_files.csv'
        logger.warning(f"Failed to process {len(failed_files)} files. See {failed_log}")
        with open(failed_log, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'error'])
            writer.writerows(failed_files)

    return f_matrix, g_matrix


def main():
    parser = argparse.ArgumentParser(
        description="Train F-matrix and G-matrix using incremental learning"
    )
    parser.add_argument(
        '--config',
        type=Path,
        default='configs/fmatrix_training_config.yaml',
        help='Path to configuration YAML file'
    )
    parser.add_argument(
        '--condition',
        type=str,
        required=True,
        choices=['normal', 'lfloss', 'hfloss'],
        help='Hearing loss condition to train'
    )
    parser.add_argument(
        '--split',
        type=str,
        default='train',
        help='Data split to use for training (default: train)'
    )
    parser.add_argument(
        '--smatrix',
        type=Path,
        default=None,
        help='Path to S-matrix file (default: data/smatrices/{split}_smatrix.nc)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of training files (for testing)'
    )
    parser.add_argument(
        '--resume-from-f',
        type=Path,
        default=None,
        help='Resume F-matrix training from checkpoint'
    )
    parser.add_argument(
        '--resume-from-g',
        type=Path,
        default=None,
        help='Resume G-matrix training from checkpoint'
    )
    parser.add_argument(
        '--auto-resume',
        action='store_true',
        help='Automatically find and resume from the latest checkpoint'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Set random seeds for reproducibility
    set_random_seeds(config['training']['random_seed'])

    # Get audio directory for condition
    audio_dir = Path(config['data']['audio_dirs'][args.condition])

    # Determine S-matrix path
    if args.smatrix is not None:
        smatrix_path = args.smatrix
    else:
        smatrix_path = Path(f'data/smatrices/{args.split}_smatrix.nc')

    # Output directory
    output_dir = Path(config['output']['base_dir'])

    # Handle auto-resume: find latest checkpoints
    resume_from_f = args.resume_from_f
    resume_from_g = args.resume_from_g

    if args.auto_resume:
        logger.info("Auto-resume enabled: searching for latest checkpoints...")
        checkpoint_dir = output_dir / config['output']['checkpoints_dir']

        # Find latest F-matrix checkpoint
        if resume_from_f is None:
            latest_f, f_iter = find_latest_checkpoint(checkpoint_dir, 'fmatrix', args.condition)
            if latest_f is not None:
                resume_from_f = latest_f

        # Find latest G-matrix checkpoint
        if resume_from_g is None:
            latest_g, g_iter = find_latest_checkpoint(checkpoint_dir, 'gmatrix', args.condition)
            if latest_g is not None:
                resume_from_g = latest_g

        # Verify both matrices are at the same iteration (important for consistency)
        if resume_from_f is not None and resume_from_g is not None:
            import re
            f_match = re.search(r'iter(\d+)', resume_from_f.name)
            g_match = re.search(r'iter(\d+)', resume_from_g.name)
            if f_match and g_match:
                f_iter = int(f_match.group(1))
                g_iter = int(g_match.group(1))
                if f_iter != g_iter:
                    logger.warning(f"F-matrix and G-matrix checkpoints are at different iterations!")
                    logger.warning(f"  F-matrix: iteration {f_iter}")
                    logger.warning(f"  G-matrix: iteration {g_iter}")
                    # Use the minimum iteration to ensure consistency
                    min_iter = min(f_iter, g_iter)
                    logger.warning(f"Using iteration {min_iter} for both matrices")
                    # Find checkpoints at the minimum iteration
                    import glob
                    f_pattern = str(checkpoint_dir / args.condition / f"fmatrix_{args.condition}_iter{min_iter:07d}.csv.gz")
                    g_pattern = str(checkpoint_dir / args.condition / f"gmatrix_{args.condition}_iter{min_iter:07d}.csv.gz")
                    f_files = glob.glob(f_pattern)
                    g_files = glob.glob(g_pattern)
                    if f_files and g_files:
                        resume_from_f = Path(f_files[0])
                        resume_from_g = Path(g_files[0])
                    else:
                        logger.error(f"Could not find matching checkpoints at iteration {min_iter}")
                        logger.error("Please specify checkpoints manually with --resume-from-f and --resume-from-g")
                        sys.exit(1)

    logger.info("=" * 60)
    logger.info("F-Matrix and G-Matrix Incremental Training")
    logger.info("=" * 60)
    logger.info(f"Condition: {args.condition}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Audio directory: {audio_dir}")
    logger.info(f"S-matrix: {smatrix_path}")
    logger.info(f"Output directory: {output_dir}")
    if args.limit is not None:
        logger.info(f"Limit: {args.limit} files (TESTING MODE)")
    if resume_from_f is not None:
        logger.info(f"Resuming F-matrix from: {resume_from_f}")
    if resume_from_g is not None:
        logger.info(f"Resuming G-matrix from: {resume_from_g}")
    logger.info("=" * 60)

    # Load S-matrix
    s_matrix = load_smatrix(smatrix_path)

    # Get audio files
    logger.info(f"Loading audio files from {audio_dir}/{args.split}")
    audio_files = get_audio_files(audio_dir, args.split)
    logger.info(f"Found {len(audio_files)} audio files")

    # Apply limit if specified
    if args.limit is not None:
        audio_files = audio_files[:args.limit]
        logger.info(f"Limited to {len(audio_files)} files for testing")

    # Create random training order
    logger.info("Creating random training order...")
    audio_files = create_training_order(audio_files, seed=config['training']['random_seed'])

    # Train F-matrix and G-matrix
    logger.info("Starting incremental training...")
    f_matrix, g_matrix = incremental_train_matrices(
        condition=args.condition,
        config=config,
        audio_files=audio_files,
        s_matrix=s_matrix,
        output_dir=output_dir,
        resume_from_f=resume_from_f,
        resume_from_g=resume_from_g
    )

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info(f"F-matrix shape: {f_matrix.shape}")
    logger.info(f"G-matrix shape: {g_matrix.shape}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
