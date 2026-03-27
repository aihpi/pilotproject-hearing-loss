#!/usr/bin/env python3
"""
Predict S-matrix (semantic vectors) for MALD words using a trained F-matrix.

Computes: S_hat = C @ F

where C is the MALD c-vector matrix and F is a trained F-matrix checkpoint.

Input:
  - MALD c-vectors (NetCDF, from generate_mald_cvectors.py)
  - F-matrix checkpoint (gzipped TSV, from train_fmatrix_incremental.py)

Output:
  - Predicted S-matrix as NetCDF: xarray DataArray of shape (n_words, n_semantics)
"""

import argparse
import gzip
import logging
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).parent))

from discriminative_lexicon_model.mapping import load_mat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_fmatrix(path: Path) -> xr.DataArray:
    """
    Load F-matrix from a (possibly gzipped) TSV file.

    Args:
        path: Path to .csv, .csv.gz, or .tsv file.

    Returns:
        F-matrix as xarray DataArray.
    """
    logger.info(f"Loading F-matrix from {path}")

    if str(path).endswith(".gz"):
        # Decompress to a temporary file for load_mat
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        with gzip.open(path, "rb") as f_in, open(tmp_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        fmat = load_mat(str(tmp_path))
        tmp_path.unlink()
    else:
        fmat = load_mat(str(path))

    logger.info(f"F-matrix shape: {fmat.shape}")
    return fmat


def main():
    parser = argparse.ArgumentParser(
        description="Predict S-matrix for MALD words using a trained F-matrix"
    )
    parser.add_argument(
        "--cmatrix",
        type=Path,
        default=Path("data/MALD/mald_cvectors.nc"),
        help="Path to MALD c-vector matrix (NetCDF, default: data/MALD/mald_cvectors.nc)",
    )
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        help="Hearing condition (normal, lfloss, or hfloss)",
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("results/matrices_incremental/checkpoints"),
        help="Directory containing per-condition F-matrix checkpoints (default: results/matrices_incremental/checkpoints)",
    )
    parser.add_argument(
        "--checkpoint-iter",
        type=str,
        required=True,
        help="Checkpoint iteration number (e.g. 2200000)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/MALD"),
        help="Output directory for predicted S-matrix (default: data/MALD)",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    cmatrix_path = project_root / args.cmatrix if not args.cmatrix.is_absolute() else args.cmatrix
    checkpoints_dir = project_root / args.checkpoints_dir if not args.checkpoints_dir.is_absolute() else args.checkpoints_dir
    output_dir = project_root / args.output_dir if not args.output_dir.is_absolute() else args.output_dir

    # Construct F-matrix path from condition and checkpoint iteration
    fmatrix_path = checkpoints_dir / args.condition / f"fmatrix_{args.condition}_iter{args.checkpoint_iter}.csv.gz"

    # Construct output path including condition and checkpoint iteration
    output_path = output_dir / f"mald_smatrix_{args.condition}_iter{args.checkpoint_iter}.nc"

    logger.info("=" * 60)
    logger.info("Predict MALD S-matrix")
    logger.info("=" * 60)
    logger.info(f"C-matrix: {cmatrix_path}")
    logger.info(f"F-matrix: {fmatrix_path}")
    logger.info(f"Output:   {output_path}")
    logger.info("=" * 60)

    # Load C-matrix
    logger.info("Loading C-matrix...")
    c_matrix = xr.open_dataarray(cmatrix_path)
    logger.info(f"C-matrix shape: {c_matrix.shape}  (words: {c_matrix.sizes['word']})")

    # Load F-matrix
    f_matrix = load_fmatrix(fmatrix_path)

    # Verify dimensions match
    n_cues_c = c_matrix.sizes["cues"]
    n_cues_f = f_matrix.shape[0]
    if n_cues_c != n_cues_f:
        logger.error(
            f"Dimension mismatch: C-matrix has {n_cues_c} cues, "
            f"F-matrix has {n_cues_f} rows"
        )
        return 1

    # Compute S_hat = C @ F
    logger.info("Computing S_hat = C @ F ...")
    c_vals = c_matrix.values  # (n_words, n_cues)
    f_vals = f_matrix.values  # (n_cues, n_semantics)
    s_hat_vals = c_vals @ f_vals  # (n_words, n_semantics)

    # Build xarray DataArray for the predicted S-matrix
    # Use semantic dimension labels from the F-matrix columns
    sem_labels = f_matrix.coords[f_matrix.dims[1]].values.tolist()
    s_hat = xr.DataArray(
        s_hat_vals,
        dims=("word", "semantics"),
        coords={
            "word": c_matrix.coords["word"].values.tolist(),
            "semantics": sem_labels,
        },
    )

    logger.info(f"Predicted S-matrix shape: {s_hat.shape}")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    s_hat.to_netcdf(output_path)
    logger.info(f"Saved predicted S-matrix to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
