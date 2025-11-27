#!/usr/bin/env python3

#############################################################################
# Script Name: tensorboard_visualise_runs.py                               #
# Description: Extract and visualize TensorBoard logs from Whisper training#
# Author: Hanno Müller                                                      #
# Date: 2025-08-28                                                          #
#############################################################################

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from collections import defaultdict
import glob
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Extract and visualize TensorBoard logs from Whisper training runs."
    )
    
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the saved Whisper model directory (e.g., FrisperWhisper/medium_HF_20k)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for plots (default: creates 'tensorboard' folder in model directory)"
    )
    
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "pdf", "svg", "jpg"],
        default="png",
        help="Output format for plots (default: png)"
    )
    
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for output plots (default: 300)"
    )
    
    parser.add_argument(
        "--figsize",
        type=str,
        default="12,8",
        help="Figure size as 'width,height' in inches (default: 12,8)"
    )
    
    parser.add_argument(
        "--smooth",
        type=float,
        default=0.0,
        help="Smoothing factor for plots (0.0-1.0, default: 0.0 = no smoothing)"
    )
    
    return parser.parse_args()


def find_event_files(model_path):
    """Find all TensorBoard event files in the model directory."""
    runs_dir = Path(model_path) / "runs"
    
    if not runs_dir.exists():
        raise FileNotFoundError(f"No 'runs' directory found in {model_path}")
    
    # Find all event files recursively
    event_files = list(runs_dir.glob("**/events.out.tfevents.*"))
    
    if not event_files:
        raise FileNotFoundError(f"No TensorBoard event files found in {runs_dir}")
    
    print(f"Found {len(event_files)} event file(s):")
    for event_file in event_files:
        print(f"  {event_file}")
    
    return event_files


def extract_scalar_data(event_file):
    """Extract scalar data from a TensorBoard event file."""
    print(f"Processing: {event_file}")
    
    # Load the event accumulator
    ea = EventAccumulator(str(event_file))
    ea.Reload()
    
    # Get available scalar tags
    scalar_tags = ea.Tags()['scalars']
    print(f"Available metrics: {scalar_tags}")
    
    data = {}
    
    for tag in scalar_tags:
        scalar_events = ea.Scalars(tag)
        
        # Extract step, wall_time, and value
        steps = [event.step for event in scalar_events]
        values = [event.value for event in scalar_events]
        wall_times = [event.wall_time for event in scalar_events]
        
        data[tag] = {
            'steps': steps,
            'values': values,
            'wall_times': wall_times
        }
    
    return data


def smooth_data(values, smoothing_factor):
    """Apply exponential moving average smoothing to data."""
    if smoothing_factor <= 0:
        return values
    
    smoothed = []
    last = values[0] if values else 0
    
    for value in values:
        smoothed_val = last * smoothing_factor + value * (1 - smoothing_factor)
        smoothed.append(smoothed_val)
        last = smoothed_val
    
    return smoothed


def create_training_plots(data, output_dir, args):
    """Create comprehensive training visualization plots."""
    figsize = tuple(map(float, args.figsize.split(',')))
    
    # Define color palette
    colors = {
        'train_loss': '#2E86C1',
        'eval_loss': '#E74C3C',
        'learning_rate': '#28B463',
        'grad_norm': '#F39C12',
        'wer': '#8E44AD'
    }
    
    # 1. Loss Plot (Train and Eval)
    plt.figure(figsize=figsize)
    
    train_loss_key = None
    eval_loss_key = None
    
    for key in data.keys():
        key_lower = key.lower()
        # Match training loss - prioritize 'train/loss' over 'train/train_loss'
        if key_lower == 'train/loss':
            train_loss_key = key
        elif 'train_loss' in key_lower or 'training_loss' in key_lower:
            if not train_loss_key:  # Only use as fallback
                train_loss_key = key
        # Match evaluation loss
        elif key_lower == 'eval/loss' or 'eval_loss' in key_lower or 'validation_loss' in key_lower:
            eval_loss_key = key
    
    if train_loss_key:
        train_steps = data[train_loss_key]['steps']
        train_loss = smooth_data(data[train_loss_key]['values'], args.smooth)
        plt.plot(train_steps, train_loss, label='Training Loss', 
                color=colors['train_loss'], linewidth=2)
    
    if eval_loss_key:
        eval_steps = data[eval_loss_key]['steps']
        eval_loss = smooth_data(data[eval_loss_key]['values'], args.smooth)
        plt.plot(eval_steps, eval_loss, label='Evaluation Loss', 
                color=colors['eval_loss'], linewidth=2, linestyle='--')
    
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for loss
    
    # Save loss plot
    loss_path = output_dir / f"loss_curves.{args.format}"
    plt.savefig(loss_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {loss_path}")
    
    # 2. Learning Rate Plot
    lr_key = None
    for key in data.keys():
        if 'learning_rate' in key.lower() or 'lr' in key.lower():
            lr_key = key
            break
    
    if lr_key:
        plt.figure(figsize=figsize)
        lr_steps = data[lr_key]['steps']
        lr_values = data[lr_key]['values']
        
        plt.plot(lr_steps, lr_values, label='Learning Rate', 
                color=colors['learning_rate'], linewidth=2)
        plt.xlabel('Training Steps')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        lr_path = output_dir / f"learning_rate.{args.format}"
        plt.savefig(lr_path, dpi=args.dpi, bbox_inches='tight')
        plt.close()
        print(f"Saved: {lr_path}")
    
    # 3. Gradient Norm Plot
    grad_norm_key = None
    for key in data.keys():
        if 'grad_norm' in key.lower():
            grad_norm_key = key
            break
    
    if grad_norm_key:
        plt.figure(figsize=figsize)
        grad_steps = data[grad_norm_key]['steps']
        grad_values = smooth_data(data[grad_norm_key]['values'], args.smooth)
        
        plt.plot(grad_steps, grad_values, label='Gradient Norm', 
                color=colors['grad_norm'], linewidth=2)
        plt.xlabel('Training Steps')
        plt.ylabel('Gradient Norm')
        plt.title('Gradient Norm Over Time')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        grad_path = output_dir / f"gradient_norm.{args.format}"
        plt.savefig(grad_path, dpi=args.dpi, bbox_inches='tight')
        plt.close()
        print(f"Saved: {grad_path}")
    
    # 4. WER Plot
    wer_key = None
    for key in data.keys():
        if 'wer' in key.lower():
            wer_key = key
            break
    
    if wer_key:
        plt.figure(figsize=figsize)
        wer_steps = data[wer_key]['steps']
        wer_values = smooth_data(data[wer_key]['values'], args.smooth)
        
        plt.plot(wer_steps, wer_values, label='Word Error Rate', 
                color=colors['wer'], linewidth=2, marker='o', markersize=4)
        plt.xlabel('Training Steps')
        plt.ylabel('WER (%)')
        plt.title('Word Error Rate Over Time')
        plt.grid(True, alpha=0.3)
        
        wer_path = output_dir / f"word_error_rate.{args.format}"
        plt.savefig(wer_path, dpi=args.dpi, bbox_inches='tight')
        plt.close()
        print(f"Saved: {wer_path}")
    
    # 5. Combined Overview Plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Whisper Training Overview', fontsize=16)
    
    # Loss subplot
    ax1 = axes[0, 0]
    if train_loss_key:
        ax1.plot(train_steps, train_loss, label='Train', color=colors['train_loss'], linewidth=2)
    if eval_loss_key:
        ax1.plot(eval_steps, eval_loss, label='Eval', color=colors['eval_loss'], linewidth=2)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Learning rate subplot
    ax2 = axes[0, 1]
    if lr_key:
        ax2.plot(lr_steps, lr_values, color=colors['learning_rate'], linewidth=2)
        ax2.set_xlabel('Steps')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
    
    # Gradient norm subplot
    ax3 = axes[1, 0]
    if grad_norm_key:
        ax3.plot(grad_steps, grad_values, color=colors['grad_norm'], linewidth=2)
        ax3.set_xlabel('Steps')
        ax3.set_ylabel('Gradient Norm')
        ax3.set_title('Gradient Norm')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
    
    # WER subplot
    ax4 = axes[1, 1]
    if wer_key:
        ax4.plot(wer_steps, wer_values, color=colors['wer'], linewidth=2, marker='o', markersize=3)
        ax4.set_xlabel('Steps')
        ax4.set_ylabel('WER (%)')
        ax4.set_title('Word Error Rate')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    overview_path = output_dir / f"training_overview.{args.format}"
    plt.savefig(overview_path, dpi=args.dpi, bbox_inches='tight')
    plt.close()
    print(f"Saved: {overview_path}")


def save_data_csv(data, output_dir):
    """Save extracted data to CSV files."""
    csv_dir = output_dir / "csv_data"
    csv_dir.mkdir(exist_ok=True)
    
    for metric_name, metric_data in data.items():
        # Clean metric name for filename
        filename = metric_name.replace('/', '_').replace('\\', '_') + '.csv'
        
        df = pd.DataFrame({
            'step': metric_data['steps'],
            'value': metric_data['values'],
            'wall_time': metric_data['wall_times']
        })
        
        csv_path = csv_dir / filename
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV: {csv_path}")


def create_summary_report(data, output_dir, model_path):
    """Create a summary report of the training run."""
    report_path = output_dir / "training_summary.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("WHISPER TRAINING SUMMARY REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Generated: {pd.Timestamp.now()}\n\n")
        
        f.write("METRICS SUMMARY:\n")
        f.write("-"*40 + "\n")
        
        for metric_name, metric_data in data.items():
            values = metric_data['values']
            steps = metric_data['steps']
            
            if values:
                f.write(f"\n{metric_name}:\n")
                f.write(f"  Total steps: {len(steps)}\n")
                f.write(f"  Step range: {min(steps)} - {max(steps)}\n")
                f.write(f"  Initial value: {values[0]:.6f}\n")
                f.write(f"  Final value: {values[-1]:.6f}\n")
                f.write(f"  Best value: {min(values):.6f}\n")
                f.write(f"  Worst value: {max(values):.6f}\n")
                
                # Calculate improvement
                if len(values) > 1:
                    improvement = ((values[0] - values[-1]) / values[0]) * 100
                    f.write(f"  Improvement: {improvement:.2f}%\n")
        
        f.write("\n" + "="*60 + "\n")
    
    print(f"Saved summary: {report_path}")


def main():
    """Main function to process TensorBoard logs and create visualizations."""
    args = parse_arguments()
    
    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        sys.exit(1)
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = model_path / "tensorboard"
    
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("TENSORBOARD VISUALIZATION SCRIPT")
    print("="*60)
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")
    print(f"Output format: {args.format}")
    print(f"Smoothing: {args.smooth}")
    print("="*60)
    
    try:
        # Find event files
        event_files = find_event_files(model_path)
        
        # Process each event file (usually just one)
        all_data = {}
        for event_file in event_files:
            file_data = extract_scalar_data(event_file)
            all_data.update(file_data)
        
        if not all_data:
            print("No scalar data found in TensorBoard logs.")
            return
        
        # Create plots
        print("\nCreating visualization plots...")
        create_training_plots(all_data, output_dir, args)
        
        # Save raw data to CSV
        print("\nSaving raw data to CSV...")
        save_data_csv(all_data, output_dir)
        
        # Create summary report
        print("\nGenerating summary report...")
        create_summary_report(all_data, output_dir, model_path)
        
        print(f"\n✅ Visualization complete! Check output directory: {output_dir}")
        print("\nGenerated files:")
        for file_path in sorted(output_dir.rglob("*")):
            if file_path.is_file():
                print(f"  {file_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
