#!/usr/bin/env python3
"""
Evaluate predicted MALD S-matrices against gold-standard fastText vectors.

For each hearing condition (normal, lfloss, hfloss), loads the predicted
S-matrix and computes per-word cosine similarity against the fastText
embedding of that word.

Output: a CSV file with columns: word, condition, cosine_similarity
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.spatial.distance import cosine

sys.path.insert(0, str(Path(__file__).parent))

from discriminative_lexicon_model.mapping import gen_smat

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

CONDITIONS = ["normal", "lfloss", "hfloss"]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate predicted MALD S-matrices against fastText gold standard"
    )
    parser.add_argument(
        "--smatrix-dir",
        type=Path,
        default=Path("data/MALD"),
        help="Directory containing predicted S-matrix NetCDF files (default: data/MALD)",
    )
    parser.add_argument(
        "--checkpoint-iter",
        type=str,
        required=True,
        help="Checkpoint iteration used for prediction (e.g. 6450000)",
    )
    parser.add_argument(
        "--fasttext-model",
        type=Path,
        default=Path("data/fasttext/cc.en.300.bin"),
        help="Path to fastText model (default: data/fasttext/cc.en.300.bin)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: data/MALD/mald_evaluation_iter{checkpoint_iter}.csv)",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    smatrix_dir = project_root / args.smatrix_dir if not args.smatrix_dir.is_absolute() else args.smatrix_dir
    fasttext_path = project_root / args.fasttext_model if not args.fasttext_model.is_absolute() else args.fasttext_model

    if args.output is not None:
        output_path = project_root / args.output if not args.output.is_absolute() else args.output
    else:
        output_path = smatrix_dir / f"mald_evaluation_iter{args.checkpoint_iter}.csv"

    logger.info("=" * 60)
    logger.info("Evaluate MALD predicted S-matrices")
    logger.info("=" * 60)
    logger.info(f"S-matrix directory: {smatrix_dir}")
    logger.info(f"Checkpoint iteration: {args.checkpoint_iter}")
    logger.info(f"FastText model: {fasttext_path}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)

    # Load fastText model and generate gold-standard S-matrix
    # Use the first condition's predicted S-matrix to get the word list
    first_smat_path = smatrix_dir / f"mald_smatrix_{CONDITIONS[0]}_iter{args.checkpoint_iter}.nc"
    if not first_smat_path.exists():
        logger.error(f"Predicted S-matrix not found: {first_smat_path}")
        return 1

    first_smat = xr.open_dataarray(first_smat_path)
    words = first_smat.coords["word"].values.tolist()
    first_smat.close()

    logger.info(f"Number of words: {len(words)}")

    logger.info("Loading fastText model...")
    import fasttext
    ft_model = fasttext.load_model(str(fasttext_path))

    logger.info("Generating gold-standard S-matrix from fastText...")
    s_gold = gen_smat(words, embed=ft_model)

    # Process each condition
    import csv
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["word", "condition", "cosine_similarity"])

        for condition in CONDITIONS:
            smat_path = smatrix_dir / f"mald_smatrix_{condition}_iter{args.checkpoint_iter}.nc"
            if not smat_path.exists():
                logger.error(f"Predicted S-matrix not found: {smat_path}")
                return 1

            logger.info(f"Evaluating condition: {condition}")
            s_pred = xr.open_dataarray(smat_path)

            for word in words:
                s_pred_vec = s_pred.sel(word=word).values
                s_gold_vec = s_gold.sel(word=word).values

                cos_sim = 1.0 - cosine(s_pred_vec, s_gold_vec)
                writer.writerow([word, condition, f"{cos_sim:.6f}"])

            s_pred.close()
            logger.info(f"  {condition}: done")

    logger.info(f"Results saved to {output_path}")

    # Print summary
    import pandas as pd
    df = pd.read_csv(output_path)
    logger.info("\nSummary (cosine similarity):")
    summary = df.groupby("condition")["cosine_similarity"].agg(["mean", "std", "min", "max"])
    print(summary.round(4))

    return 0


if __name__ == "__main__":
    sys.exit(main())
