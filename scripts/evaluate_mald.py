#!/usr/bin/env python3
"""
Evaluate DLM F-matrices on MALD dataset.

This script:
1. Loads audio files from MALD dataset
2. Applies three acoustic conditions (normal, lfloss, hfloss) to each audio
3. Generates log-mel spectrograms for each condition
4. Feeds spectrograms to each of the three F-matrices
5. Compares predicted semantic vectors against FastText ground truth
6. Outputs a CSV with cosine similarity and Euclidean distance metrics

Output columns: word, cosine_similarity, euclidean_distance, model, input_acoustic
"""

import argparse
import gzip
import logging
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy.spatial.distance import cosine, euclidean
from tqdm import tqdm

# Add scripts directory to path for module imports
sys.path.insert(0, str(Path(__file__).parent))
from modules.audio import (
    AudioProcessor,
    get_normal_hearing_profile,
    get_low_frequency_loss_profile,
    get_high_frequency_loss_profile,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Spectrogram parameters (from README_dlm.md)
SAMPLING_RATE = 16000
N_MELS = 128
N_FFT = 400  # 25 ms window
HOP_LENGTH = 160  # 10 ms hop
MAX_TIME_FRAMES = 391
C_VECTOR_SIZE = N_MELS * MAX_TIME_FRAMES  # 50,048

# Hearing loss conditions
CONDITIONS = ["normal", "lfloss", "hfloss"]


def get_hearing_profile(condition: str) -> Tuple[np.ndarray, np.ndarray]:
    """Get hearing loss profile for a given condition."""
    if condition == "normal":
        return get_normal_hearing_profile()
    elif condition == "lfloss":
        return get_low_frequency_loss_profile()
    elif condition == "hfloss":
        return get_high_frequency_loss_profile()
    else:
        raise ValueError(f"Unknown condition: {condition}")


def apply_hearing_loss(
    audio_array: np.ndarray,
    original_sr: int,
    condition: str,
) -> np.ndarray:
    """Apply hearing loss mask to audio."""
    freq_points, db_thresholds = get_hearing_profile(condition)
    processor = AudioProcessor(sample_rate=SAMPLING_RATE)
    processed_audio, _ = processor.process_with_hearing_loss(
        audio_array=audio_array,
        original_sr=original_sr,
        freq_points=freq_points,
        db_thresholds=db_thresholds,
        target_sr=SAMPLING_RATE,
    )
    return processed_audio


def generate_spectrogram(audio: np.ndarray) -> np.ndarray:
    """
    Generate log-mel spectrogram from audio.

    Returns flattened spectrogram of shape (50048,)
    """
    # Generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLING_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=SAMPLING_RATE // 2,
    )

    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Pad or truncate to target length
    current_length = log_mel_spec.shape[1]
    if current_length < MAX_TIME_FRAMES:
        pad_width = MAX_TIME_FRAMES - current_length
        log_mel_spec = np.pad(
            log_mel_spec,
            ((0, 0), (0, pad_width)),
            mode='constant',
            constant_values=log_mel_spec.min()
        )
    elif current_length > MAX_TIME_FRAMES:
        log_mel_spec = log_mel_spec[:, :MAX_TIME_FRAMES]

    # Flatten to c-vector
    return log_mel_spec.flatten()


def load_fmatrix(path: Path) -> np.ndarray:
    """Load F-matrix from gzipped CSV file."""
    logger.info(f"Loading F-matrix from {path}")

    # Read the CSV - the format has row/column names
    df = pd.read_csv(path, sep='\t', index_col=0, compression='gzip')

    logger.info(f"F-matrix shape: {df.shape}")
    return df.values, df.index.tolist(), df.columns.tolist()


def load_fasttext_model(path: Path):
    """Load FastText model."""
    logger.info(f"Loading FastText model from {path}")
    import fasttext
    model = fasttext.load_model(str(path))
    logger.info(f"FastText model loaded (dimension: {model.get_dimension()})")
    return model


def process_single_word(args: Tuple) -> Optional[List[Dict]]:
    """
    Process a single word audio file.

    Returns list of result dictionaries (one per model/input_acoustic combination).
    """
    audio_path, fmatrices, fasttext_model = args

    word = audio_path.stem.upper()  # Get word from filename

    try:
        # Load audio
        audio_array, original_sr = sf.read(audio_path)

        # Ensure mono
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)
        audio_array = audio_array.astype(np.float32)

        # Get FastText embedding for this word
        # FastText expects lowercase
        s_true = fasttext_model.get_word_vector(word.lower())

        results = []

        # Process each acoustic condition
        for input_acoustic in CONDITIONS:
            # Apply hearing loss
            processed_audio = apply_hearing_loss(audio_array, original_sr, input_acoustic)

            # Generate spectrogram (c-vector)
            c_vector = generate_spectrogram(processed_audio)

            # Evaluate with each model
            for model_name, fmatrix in fmatrices.items():
                # Predict semantic vector: s_hat = c @ F
                s_hat = c_vector @ fmatrix

                # Compute metrics
                cos_sim = 1 - cosine(s_hat, s_true)  # cosine returns distance
                euc_dist = euclidean(s_hat, s_true)

                results.append({
                    'word': word,
                    'cosine_similarity': cos_sim,
                    'euclidean_distance': euc_dist,
                    'model': model_name,
                    'input_acoustic': input_acoustic,
                })

        return results

    except Exception as e:
        logger.warning(f"Failed to process {audio_path.name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DLM F-matrices on MALD dataset"
    )
    parser.add_argument(
        "--mald-dir",
        type=Path,
        default=Path("data/MALD/MALD1_rw"),
        help="Directory containing MALD audio files (default: data/MALD/MALD1_rw)",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("results/matrices_incremental/checkpoints"),
        help="Directory containing F-matrix checkpoints (default: results/matrices_incremental/checkpoints)",
    )
    parser.add_argument(
        "--fasttext-model",
        type=Path,
        default=Path("data/fasttext/cc.en.300.bin"),
        help="Path to FastText model (default: data/fasttext/cc.en.300.bin)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mald_evaluation.csv.gz"),
        help="Output CSV file path (default: results/mald_evaluation.csv.gz)",
    )
    parser.add_argument(
        "--checkpoint-iter",
        type=str,
        default="10205536",
        help="Checkpoint iteration to use (default: 10205536)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of files to process (for testing)",
    )
    parser.add_argument(
        "--nworkers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, FastText not picklable)",
    )

    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    mald_dir = project_root / args.mald_dir if not args.mald_dir.is_absolute() else args.mald_dir
    checkpoints_dir = project_root / args.checkpoints_dir if not args.checkpoints_dir.is_absolute() else args.checkpoints_dir
    fasttext_path = project_root / args.fasttext_model if not args.fasttext_model.is_absolute() else args.fasttext_model
    output_path = project_root / args.output if not args.output.is_absolute() else args.output

    logger.info("=" * 60)
    logger.info("MALD Evaluation with DLM F-matrices")
    logger.info("=" * 60)
    logger.info(f"MALD directory: {mald_dir}")
    logger.info(f"Checkpoints directory: {checkpoints_dir}")
    logger.info(f"FastText model: {fasttext_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Checkpoint iteration: {args.checkpoint_iter}")
    logger.info("=" * 60)

    # Load F-matrices
    logger.info("\nLoading F-matrices...")
    fmatrices = {}
    for condition in CONDITIONS:
        fmatrix_path = checkpoints_dir / condition / f"fmatrix_{condition}_iter{args.checkpoint_iter}.csv.gz"
        if not fmatrix_path.exists():
            logger.error(f"F-matrix not found: {fmatrix_path}")
            return 1
        fmatrix_values, cue_names, sem_names = load_fmatrix(fmatrix_path)
        fmatrices[condition] = fmatrix_values
        logger.info(f"  {condition}: {fmatrix_values.shape}")

    # Verify F-matrix dimensions
    expected_cues = C_VECTOR_SIZE
    expected_sems = 300  # FastText dimension
    for name, fmat in fmatrices.items():
        if fmat.shape[0] != expected_cues:
            logger.error(f"F-matrix {name} has {fmat.shape[0]} cues, expected {expected_cues}")
            return 1
        if fmat.shape[1] != expected_sems:
            logger.error(f"F-matrix {name} has {fmat.shape[1]} semantics, expected {expected_sems}")
            return 1

    # Load FastText model
    logger.info("\nLoading FastText model...")
    fasttext_model = load_fasttext_model(fasttext_path)

    # Get audio files
    logger.info("\nFinding audio files...")
    audio_files = sorted(mald_dir.glob("*.wav"))
    total_files = len(audio_files)
    logger.info(f"Found {total_files} audio files")

    if args.limit:
        audio_files = audio_files[:args.limit]
        logger.info(f"Limited to {len(audio_files)} files for testing")

    # Process files
    logger.info("\nProcessing audio files...")
    all_results = []

    # Single-threaded processing (FastText model is not picklable)
    for audio_path in tqdm(audio_files, desc="Processing"):
        result = process_single_word((audio_path, fmatrices, fasttext_model))
        if result is not None:
            all_results.extend(result)

    # Create DataFrame
    logger.info(f"\nCreating output DataFrame with {len(all_results)} rows...")
    df = pd.DataFrame(all_results)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save to compressed CSV
    logger.info(f"Saving to {output_path}...")
    df.to_csv(output_path, index=False, compression='gzip')

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation complete!")
    logger.info(f"Total words processed: {len(audio_files)}")
    logger.info(f"Total rows in output: {len(df)}")
    logger.info(f"Output saved to: {output_path}")
    logger.info("=" * 60)

    # Print summary statistics
    logger.info("\nSummary statistics:")
    summary = df.groupby(['model', 'input_acoustic']).agg({
        'cosine_similarity': ['mean', 'std'],
        'euclidean_distance': ['mean', 'std'],
    }).round(4)
    print(summary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
