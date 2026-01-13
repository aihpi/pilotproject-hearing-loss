#!/usr/bin/env python3
"""
Visualize learning curves and F-matrix/G-matrix comparisons across conditions.

F-matrix: form -> semantics (comprehension)
G-matrix: semantics -> form (production)
"""

import argparse
import logging
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_training_metrics(metrics_dir: Path, condition: str) -> pd.DataFrame:
    """
    Load training metrics from CSV file.

    Args:
        metrics_dir: Base metrics directory
        condition: Condition name

    Returns:
        DataFrame with training metrics
    """
    metrics_file = metrics_dir / condition / 'training_metrics.csv'

    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    logger.info(f"Loading metrics for {condition} from {metrics_file}")
    df = pd.read_csv(metrics_file)
    df['condition'] = condition

    return df


def plot_learning_curves(metrics_dfs: dict, output_dir: Path):
    """
    Plot learning curves for all conditions (both F and G matrices).

    Args:
        metrics_dfs: Dictionary mapping condition to metrics DataFrame
        output_dir: Directory to save plots
    """
    logger.info("Plotting learning curves...")

    # Create figure with subplots (2x2 grid for F and G metrics)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Plot 1: F-matrix prediction error over iterations
    ax = axes[0, 0]
    for condition, df in metrics_dfs.items():
        ax.plot(df['iteration'], df['f_prediction_error'],
                label=condition, alpha=0.7, linewidth=1)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Prediction Error (L2 norm)')
    ax.set_title('F-Matrix: Prediction Error (Comprehension)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: F-matrix cosine similarity over iterations
    ax = axes[0, 1]
    for condition, df in metrics_dfs.items():
        ax.plot(df['iteration'], df['f_cosine_similarity'],
                label=condition, alpha=0.7, linewidth=1)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('F-Matrix: Cosine Similarity (Comprehension)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    # Plot 3: G-matrix prediction error over iterations
    ax = axes[1, 0]
    for condition, df in metrics_dfs.items():
        ax.plot(df['iteration'], df['g_prediction_error'],
                label=condition, alpha=0.7, linewidth=1)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Prediction Error (L2 norm)')
    ax.set_title('G-Matrix: Prediction Error (Production)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: G-matrix cosine similarity over iterations
    ax = axes[1, 1]
    for condition, df in metrics_dfs.items():
        ax.plot(df['iteration'], df['g_cosine_similarity'],
                label=condition, alpha=0.7, linewidth=1)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('G-Matrix: Cosine Similarity (Production)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])

    plt.tight_layout()

    # Save figure
    output_path = output_dir / 'learning_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved learning curves to {output_path}")

    # Also save as PDF
    output_path_pdf = output_dir / 'learning_curves.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')

    plt.close()


def plot_learning_curves_comparison(metrics_dfs: dict, output_dir: Path):
    """
    Plot detailed comparison of learning curves.

    Args:
        metrics_dfs: Dictionary mapping condition to metrics DataFrame
        output_dir: Directory to save plots
    """
    logger.info("Plotting learning curves comparison...")

    conditions = list(metrics_dfs.keys())
    colors = sns.color_palette("husl", len(conditions))

    # Plot error comparison with smoothing
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (condition, df) in enumerate(metrics_dfs.items()):
        # Apply smoothing (rolling average)
        window = max(1, len(df) // 100)  # 1% window
        df_smooth = df.copy()
        df_smooth['error_smooth'] = df['prediction_error'].rolling(
            window=window, center=True
        ).mean()

        ax.plot(df_smooth['iteration'], df_smooth['error_smooth'],
                label=f'{condition} (smoothed)', color=colors[i],
                linewidth=2, alpha=0.8)

        # Plot raw data with low alpha
        ax.plot(df['iteration'], df['prediction_error'],
                color=colors[i], linewidth=0.5, alpha=0.2)

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Prediction Error (L2 norm)')
    ax.set_title('Prediction Error Comparison Across Conditions')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = output_dir / 'error_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved error comparison to {output_path}")

    plt.close()


def plot_evaluation_results(results_file: Path, output_dir: Path):
    """
    Plot evaluation results comparison.

    Args:
        results_file: Path to evaluation_results.json
        output_dir: Directory to save plots
    """
    logger.info("Plotting evaluation results...")

    with open(results_file, 'r') as f:
        results = json.load(f)

    df = pd.DataFrame(results)

    # Create a combined hue column for task and split
    df['task_split'] = df['task'] + ' - ' + df['split']

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Plot 1: Mean cosine similarity
    ax = axes[0]
    sns.barplot(data=df, x='condition', y='mean_cosine_similarity',
                hue='task_split', ax=ax)
    ax.set_title('Mean Cosine Similarity')
    ax.set_ylabel('Cosine Similarity')
    ax.set_ylim([0, 1])
    ax.legend(title='Task - Split', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot 2: Mean Pearson correlation
    ax = axes[1]
    sns.barplot(data=df, x='condition', y='mean_pearson_correlation',
                hue='task_split', ax=ax)
    ax.set_title('Mean Pearson Correlation')
    ax.set_ylabel('Correlation')
    ax.set_ylim([0, 1])
    ax.legend(title='Task - Split', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Plot 3: RMSE
    ax = axes[2]
    sns.barplot(data=df, x='condition', y='rmse',
                hue='task_split', ax=ax)
    ax.set_title('Root Mean Squared Error')
    ax.set_ylabel('RMSE')
    ax.legend(title='Task - Split', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    output_path = output_dir / 'evaluation_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved evaluation comparison to {output_path}")

    output_path_pdf = output_dir / 'evaluation_comparison.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')

    plt.close()

    # Plot distribution of per-sample cosine similarities
    fig, axes = plt.subplots(1, len(df), figsize=(6*len(df), 5))
    if len(df) == 1:
        axes = [axes]

    for i, row in enumerate(df.itertuples()):
        ax = axes[i]
        cosine_sims = row.per_sample_cosine_similarities

        ax.hist(cosine_sims, bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f"{row.condition} - {row.split}")
        ax.set_xlabel('Cosine Similarity')
        ax.set_ylabel('Frequency')
        ax.axvline(row.mean_cosine_similarity, color='red',
                   linestyle='--', label=f'Mean: {row.mean_cosine_similarity:.3f}')
        ax.legend()

    plt.tight_layout()

    output_path = output_dir / 'cosine_similarity_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved distribution plots to {output_path}")

    plt.close()


def plot_matrix_comparison(comparison_file: Path, matrix_type: str, output_dir: Path):
    """
    Plot matrix comparison metrics (F-matrix or G-matrix).

    Args:
        comparison_file: Path to matrix comparison json file
        matrix_type: 'F' or 'G'
        output_dir: Directory to save plots
    """
    logger.info(f"Plotting {matrix_type}-matrix comparison...")

    with open(comparison_file, 'r') as f:
        comparison = json.load(f)

    # Extract data
    pairs = list(comparison.keys())
    frobenius_norms = [comparison[pair]['frobenius_norm'] for pair in pairs]
    normalized_norms = [comparison[pair]['normalized_frobenius_norm'] for pair in pairs]

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Frobenius norms
    ax = axes[0]
    ax.bar(range(len(pairs)), frobenius_norms, color=sns.color_palette("husl", len(pairs)))
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, rotation=45, ha='right')
    ax.set_ylabel('Frobenius Norm')
    ax.set_title(f'{matrix_type}-Matrix Differences (Frobenius Norm)')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Normalized Frobenius norms
    ax = axes[1]
    ax.bar(range(len(pairs)), normalized_norms, color=sns.color_palette("husl", len(pairs)))
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels(pairs, rotation=45, ha='right')
    ax.set_ylabel('Normalized Frobenius Norm')
    ax.set_title(f'{matrix_type}-Matrix Differences (Normalized)')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    output_path = output_dir / f'{matrix_type.lower()}matrix_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved {matrix_type}-matrix comparison to {output_path}")

    output_path_pdf = output_dir / f'{matrix_type.lower()}matrix_comparison.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')

    plt.close()


def plot_fmatrix_comparison(comparison_file: Path, output_dir: Path):
    """
    Plot F-matrix comparison metrics (wrapper for backwards compatibility).

    Args:
        comparison_file: Path to fmatrix_comparison.json
        output_dir: Directory to save plots
    """
    plot_matrix_comparison(comparison_file, 'F', output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize learning curves and evaluation results"
    )
    parser.add_argument(
        '--metrics-dir',
        type=Path,
        default='results/matrices_incremental/metrics',
        help='Directory containing training metrics'
    )
    parser.add_argument(
        '--eval-dir',
        type=Path,
        default='results/matrices_incremental/evaluation',
        help='Directory containing evaluation results'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default='results/matrices_incremental/visualizations',
        help='Output directory for visualizations'
    )
    parser.add_argument(
        '--conditions',
        nargs='+',
        default=['normal', 'lfloss', 'hfloss'],
        help='Conditions to visualize'
    )

    args = parser.parse_args()

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Visualization Generation")
    logger.info("=" * 60)
    logger.info(f"Metrics directory: {args.metrics_dir}")
    logger.info(f"Evaluation directory: {args.eval_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Conditions: {args.conditions}")
    logger.info("=" * 60)

    # Load training metrics for all conditions
    metrics_dfs = {}
    for condition in args.conditions:
        try:
            df = load_training_metrics(args.metrics_dir, condition)
            metrics_dfs[condition] = df
        except FileNotFoundError as e:
            logger.warning(f"Skipping {condition}: {e}")
            continue

    # Plot learning curves
    if metrics_dfs:
        plot_learning_curves(metrics_dfs, args.output_dir)
        plot_learning_curves_comparison(metrics_dfs, args.output_dir)

    # Plot evaluation results
    eval_results_file = args.eval_dir / 'evaluation_results.json'
    if eval_results_file.exists():
        plot_evaluation_results(eval_results_file, args.output_dir)
    else:
        logger.warning(f"Evaluation results not found: {eval_results_file}")

    # Plot F-matrix comparison
    comparison_file_f = args.eval_dir / 'fmatrix_comparison.json'
    if comparison_file_f.exists():
        plot_fmatrix_comparison(comparison_file_f, args.output_dir)
    else:
        logger.warning(f"F-matrix comparison not found: {comparison_file_f}")

    # Plot G-matrix comparison
    comparison_file_g = args.eval_dir / 'gmatrix_comparison.json'
    if comparison_file_g.exists():
        plot_matrix_comparison(comparison_file_g, 'G', args.output_dir)
    else:
        logger.warning(f"G-matrix comparison not found: {comparison_file_g}")

    logger.info("=" * 60)
    logger.info("Visualization complete!")
    logger.info(f"Plots saved to {args.output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
