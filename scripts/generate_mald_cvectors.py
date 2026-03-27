#!/usr/bin/env python3
"""
Generate C-vectors (flattened log-mel spectrograms) from MALD audio files.

Applies the same spectrogram pipeline used for F-matrix training:
  1. Load audio at 16 kHz (librosa resamples if needed)
  2. Compute log-mel spectrogram (128 mels, 400 FFT, 160 hop, 391 frames)
  3. Min-max normalize to [0, 1]
  4. Flatten to a 50,048-dim vector

Output: a single NetCDF file containing an xarray DataArray of shape
(n_words, 50048) with dims ('word', 'cues').
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from tqdm import tqdm

# Add scripts directory to path so we can import the shared utils
sys.path.insert(0, str(Path(__file__).parent))

from utils import load_and_generate_spectrogram, flatten_spectrogram

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Spectrogram parameters (must match F-matrix training config)
SPEC_PARAMS = dict(
    sampling_rate=16000,
    n_mels=128,
    n_fft=400,
    hop_length=160,
    max_time_frames=391,
)
N_CUES = SPEC_PARAMS["n_mels"] * SPEC_PARAMS["max_time_frames"]  # 50,048


def extract_word_from_mald_filename(filename: str) -> str:
    """
    Extract word from MALD filename format: WORD.wav

    The MALD files use uppercase bare-word filenames (e.g. ABANDON.wav).
    Returns the lowercased word to match fastText conventions.
    """
    return Path(filename).stem.lower()


def create_cvector(audio_path: Path, normalize: bool = True) -> np.ndarray:
    """
    Generate a c-vector from an audio file, matching the training pipeline.

    Args:
        audio_path: Path to .wav file.
        normalize: Apply per-utterance min-max normalization to [0, 1].

    Returns:
        1-D numpy array of shape (N_CUES,).
    """
    spec = load_and_generate_spectrogram(audio_path, **SPEC_PARAMS)

    if normalize:
        spec_min = spec.min()
        spec_max = spec.max()
        if spec_max > spec_min:
            spec = (spec - spec_min) / (spec_max - spec_min)
        else:
            spec = np.zeros_like(spec)

    return flatten_spectrogram(spec)


def main():
    parser = argparse.ArgumentParser(
        description="Generate C-vectors from MALD audio files"
    )
    parser.add_argument(
        "--mald-dir",
        type=Path,
        default=Path("data/MALD/MALD1_rw"),
        help="Directory containing MALD .wav files (default: data/MALD/MALD1_rw)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/MALD/mald_cvectors.nc"),
        help="Output NetCDF file (default: data/MALD/mald_cvectors.nc)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N files (for testing)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip min-max normalization (not recommended; training uses normalization)",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    mald_dir = project_root / args.mald_dir if not args.mald_dir.is_absolute() else args.mald_dir
    output_path = project_root / args.output if not args.output.is_absolute() else args.output

    # Collect audio files
    audio_files = sorted(mald_dir.glob("*.wav"))
    logger.info(f"Found {len(audio_files)} .wav files in {mald_dir}")

    if not audio_files:
        logger.error("No .wav files found.")
        return 1

    if args.limit is not None:
        audio_files = audio_files[: args.limit]
        logger.info(f"Limited to {len(audio_files)} files")

    normalize = not args.no_normalize
    logger.info(f"Normalization: {'enabled' if normalize else 'disabled'}")
    logger.info(f"C-vector size: {N_CUES}")

    # Process all files
    words = []
    vectors = []
    failed = []

    for audio_path in tqdm(audio_files, desc="Generating c-vectors"):
        try:
            word = extract_word_from_mald_filename(audio_path.name)
            vec = create_cvector(audio_path, normalize=normalize)
            words.append(word)
            vectors.append(vec)
        except Exception as e:
            logger.warning(f"Failed to process {audio_path.name}: {e}")
            failed.append((audio_path.name, str(e)))

    logger.info(f"Successfully processed {len(words)} / {len(audio_files)} files")
    if failed:
        logger.warning(f"Failed files: {len(failed)}")
        for name, err in failed[:10]:
            logger.warning(f"  {name}: {err}")

    # Build xarray DataArray
    cue_labels = [f"C{i:05d}" for i in range(N_CUES)]
    c_matrix = xr.DataArray(
        np.stack(vectors),
        dims=("word", "cues"),
        coords={"word": words, "cues": cue_labels},
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    c_matrix.to_netcdf(output_path)
    logger.info(f"Saved c-matrix ({c_matrix.shape}) to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
