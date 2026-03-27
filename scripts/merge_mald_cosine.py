#!/usr/bin/env python3
"""
Merge MALD behavioural data with predicted cosine similarity values.

Steps:
  1. Load MALD dataset, filter to real words only.
  2. Map each participant's HearingScore to a hearing condition:
       - First digit is 0 → lfloss
       - Last digit is 0 → hfloss
       - Otherwise → normal
       - Both first and last digit 0 → NA (should not occur)
  3. Load cosine similarity file and join on (word, condition).
  4. Output a merged CSV with RT and the matched cosine_similarity.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def hearing_score_to_condition(score: str) -> str:
    """
    Map a 4-digit HearingScore string to a hearing condition.

    Rules:
      - First digit 0 → lfloss
      - Last digit 0  → hfloss
      - Both 0        → NA (ambiguous, should not occur)
      - Otherwise     → normal
    """
    first_zero = score[0] == "0"
    last_zero = score[-1] == "0"

    if first_zero and last_zero:
        return pd.NA
    elif first_zero:
        return "lfloss"
    elif last_zero:
        return "hfloss"
    else:
        return "normal"


def main():
    parser = argparse.ArgumentParser(
        description="Merge MALD behavioural data with cosine similarity values"
    )
    parser.add_argument(
        "--mald-data",
        type=Path,
        default=Path("data/MALD/MALD1_1_AllData.csv.gz"),
        help="Path to MALD dataset (default: data/MALD/MALD1_1_AllData.csv.gz)",
    )
    parser.add_argument(
        "--cosine-file",
        type=Path,
        default=Path("data/MALD/mald_evaluation_iter6450000.csv"),
        help="Path to cosine similarity CSV (default: data/MALD/mald_evaluation_iter6450000.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/MALD/mald_merged.csv.gz"),
        help="Output CSV path (default: data/MALD/mald_merged.csv.gz)",
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    project_root = Path(__file__).parent.parent
    mald_path = project_root / args.mald_data if not args.mald_data.is_absolute() else args.mald_data
    cosine_path = project_root / args.cosine_file if not args.cosine_file.is_absolute() else args.cosine_file
    output_path = project_root / args.output if not args.output.is_absolute() else args.output

    logger.info("=" * 60)
    logger.info("Merge MALD data with cosine similarity")
    logger.info("=" * 60)
    logger.info(f"MALD data: {mald_path}")
    logger.info(f"Cosine file: {cosine_path}")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)

    # Load MALD dataset
    logger.info("Loading MALD dataset...")
    mald = pd.read_csv(
        mald_path,
        sep="\t",
        dtype={"HearingScore": str},
        low_memory=False,
    )
    logger.info(f"  Total rows: {len(mald)}")

    # Filter to words only
    mald = mald[mald["IsWord"] == True].copy()
    logger.info(f"  After filtering to words: {len(mald)}")

    # Map HearingScore to condition
    logger.info("Mapping HearingScore to hearing condition...")
    mald["condition"] = mald["HearingScore"].apply(hearing_score_to_condition)

    condition_counts = mald["condition"].value_counts(dropna=False)
    logger.info(f"  Condition distribution:\n{condition_counts.to_string()}")

    na_count = mald["condition"].isna().sum()
    if na_count > 0:
        logger.warning(f"  {na_count} rows with ambiguous HearingScore (both first and last digit 0) set to NA")

    # Load cosine similarity data
    logger.info("Loading cosine similarity data...")
    cosine_df = pd.read_csv(cosine_path)
    logger.info(f"  Rows: {len(cosine_df)}, unique words: {cosine_df['word'].nunique()}")

    # Lowercase the Item column for matching
    mald["word"] = mald["Item"].str.lower()

    # Filter to words present in the cosine file
    cosine_words = set(cosine_df["word"].unique())
    before = len(mald)
    mald = mald[mald["word"].isin(cosine_words)].copy()
    logger.info(f"  After filtering to words in cosine file: {len(mald)} (dropped {before - len(mald)})")

    # Pivot cosine data so each condition becomes its own column
    cosine_wide = cosine_df.pivot(index="word", columns="condition", values="cosine_similarity")
    cosine_wide.columns = [f"cosine_similarity_{c}" for c in cosine_wide.columns]
    cosine_wide = cosine_wide.reset_index()

    # Merge condition-matched cosine similarity on (word, condition)
    logger.info("Merging...")
    merged = mald.merge(
        cosine_df,
        on=["word", "condition"],
        how="left",
    )

    # Merge all three condition columns on word
    merged = merged.merge(
        cosine_wide,
        on="word",
        how="left",
    )
    logger.info(f"  Merged rows: {len(merged)}")

    n_missing = merged["cosine_similarity"].isna().sum()
    if n_missing > 0:
        logger.warning(f"  {n_missing} rows without cosine_similarity (likely NA conditions)")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False, compression="gzip")
    logger.info(f"Saved to {output_path}")

    # Summary
    logger.info("\nSummary:")
    logger.info(f"  Total rows: {len(merged)}")
    logger.info(f"  Unique words: {merged['word'].nunique()}")
    logger.info(f"  Unique subjects: {merged['Subject'].nunique()}")
    logger.info(f"  RT range: {merged['RT'].min()} - {merged['RT'].max()}")
    logger.info(f"  Cosine similarity (mean): {merged.groupby('condition')['cosine_similarity'].mean().to_string()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
