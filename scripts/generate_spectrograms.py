#!/usr/bin/env python3
"""
Script to generate log-mel spectrograms from CommonVoiceENWords audio files.
All spectrograms will be padded to the same shape: (128, 391)
"""

import os
import csv
import numpy as np
import librosa
from pathlib import Path
from tqdm import tqdm
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# Parameters
SAMPLING_RATE = 16000
N_MELS = 128
N_FFT = 400  # 25 ms window at 16 kHz
HOP_LENGTH = 160  # 10 ms hop at 16 kHz
MAX_TIME_FRAMES = 391  # Default value (use --auto-detect-max to compute from data)

def compute_max_time_frames(input_dir):
    """
    Scan all audio files to find the maximum duration and compute max time frames.

    Args:
        input_dir: Directory containing WAV files

    Returns:
        max_time_frames: Maximum time frames needed for spectrograms
    """
    audio_files = list(Path(input_dir).glob("*.wav"))
    print(f"Scanning {len(audio_files)} files to detect maximum duration...")

    max_samples = 0
    max_file = None

    for audio_file in tqdm(audio_files, desc="Detecting max duration"):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                y, _ = librosa.load(audio_file, sr=SAMPLING_RATE)
                if len(y) > max_samples:
                    max_samples = len(y)
                    max_file = audio_file.name
        except Exception as e:
            print(f"\nWarning: Could not process {audio_file}: {e}")
            continue

    # Calculate max time frames using the same formula as spectrogram generation
    max_time_frames = 1 + (max_samples - N_FFT) // HOP_LENGTH
    max_duration = max_samples / SAMPLING_RATE

    print(f"\nDetected maximum duration: {max_duration:.3f}s in file: {max_file}")
    print(f"Maximum time frames: {max_time_frames}")

    return max_time_frames

def generate_log_mel_spectrogram(audio_path, target_length=MAX_TIME_FRAMES):
    """
    Generate a log-mel spectrogram from an audio file.

    Args:
        audio_path: Path to the audio file
        target_length: Target time dimension for padding

    Returns:
        Log-mel spectrogram as numpy array of shape (n_mels, target_length)
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=SAMPLING_RATE)

    # Generate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        fmax=sr // 2  # Nyquist frequency
    )

    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

    # Pad to target length if needed
    current_length = log_mel_spec.shape[1]
    if current_length < target_length:
        # Pad with the minimum value (silence)
        pad_width = target_length - current_length
        log_mel_spec = np.pad(
            log_mel_spec,
            ((0, 0), (0, pad_width)),
            mode='constant',
            constant_values=log_mel_spec.min()
        )
    elif current_length > target_length:
        # Truncate (shouldn't happen based on our scan, but just in case)
        log_mel_spec = log_mel_spec[:, :target_length]

    return log_mel_spec

def process_single_file(audio_file, output_dir, target_length=MAX_TIME_FRAMES):
    """
    Worker function to process a single audio file.

    Args:
        audio_file: Path to the audio file
        output_dir: Directory to save output .npy file
        target_length: Target time dimension for spectrograms

    Returns:
        Dictionary with metadata for this file, or None if processing failed
    """
    try:
        # Suppress librosa warnings for short audio files
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)

            # Parse filename: {speaker_id}_{word_number}_{word}.wav
            filename = audio_file.stem  # filename without extension
            parts = filename.split('_')

            if len(parts) >= 3:
                speaker_id = parts[0]
                word_number = parts[1]
                word = '_'.join(parts[2:])  # In case word contains underscore
            else:
                # Fallback if filename doesn't match expected pattern
                speaker_id = "unknown"
                word_number = "0"
                word = filename

            # Generate spectrogram
            log_mel_spec = generate_log_mel_spectrogram(audio_file, target_length=target_length)

            # Save as .npy file
            output_filename = f"{filename}.npy"
            output_path = os.path.join(output_dir, output_filename)
            np.save(output_path, log_mel_spec)

            # Get audio duration
            y, _ = librosa.load(audio_file, sr=SAMPLING_RATE)
            duration = len(y) / SAMPLING_RATE

            # Return metadata
            return {
                'filename': audio_file.name,
                'speaker_id': speaker_id,
                'word_number': word_number,
                'word': word,
                'spectrogram_path': output_path,
                'duration': f"{duration:.3f}",
                'shape': f"{log_mel_spec.shape}"
            }

    except Exception as e:
        print(f"\nError processing {audio_file}: {e}")
        return None

def process_dataset(input_dir, output_dir, metadata_path, limit=None, nworkers=24, target_length=MAX_TIME_FRAMES):
    """
    Process all audio files in the dataset and generate spectrograms.

    Args:
        input_dir: Directory containing input WAV files
        output_dir: Directory to save output .npy files
        metadata_path: Path to save metadata CSV file
        limit: Maximum number of files to process (None = process all)
        nworkers: Number of parallel workers
        target_length: Target time dimension for spectrograms
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all wav files
    audio_files = sorted(Path(input_dir).glob("*.wav"))
    total_files = len(audio_files)
    print(f"Found {total_files} audio files")

    # Limit number of files if specified
    if limit is not None and limit > 0:
        audio_files = audio_files[:limit]
        print(f"Processing subset: {len(audio_files)} files")

    print(f"Processing with {nworkers} workers")

    # Prepare metadata list
    metadata = []

    # Process files in parallel
    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_file, audio_file, output_dir, target_length): audio_file
            for audio_file in audio_files
        }

        # Process completed tasks with progress bar
        with tqdm(total=len(futures), desc="Generating spectrograms") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    metadata.append(result)
                pbar.update(1)

    # Save metadata to CSV
    print(f"\nSaving metadata to {metadata_path}")
    with open(metadata_path, 'w', newline='') as csvfile:
        fieldnames = ['filename', 'speaker_id', 'word_number', 'word',
                      'spectrogram_path', 'duration', 'shape']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)

    print(f"\nProcessing complete!")
    print(f"Generated {len(metadata)} spectrograms")
    print(f"Output directory: {output_dir}")
    print(f"Metadata file: {metadata_path}")
    print(f"Spectrogram shape: ({N_MELS}, {target_length})")

def main():
    # Get default metadata path in the data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    default_metadata_path = os.path.join(project_dir, 'data', 'spectrograms_metadata.csv')

    parser = argparse.ArgumentParser(
        description='Generate log-mel spectrograms from audio files'
    )
    parser.add_argument(
        '--input-dir',
        type=str,
        default='/sc/projects/sci-aisc/hearingloss/data_processed/CommonVoiceENWords/validation',
        help='Input directory containing WAV files'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='/sc/projects/sci-aisc/hearingloss/data_processed/CommonVoiceENWords_spectrograms/validation',
        help='Output directory for .npy spectrogram files'
    )
    parser.add_argument(
        '--metadata',
        type=str,
        default=default_metadata_path,
        help='Path to save metadata CSV file'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Maximum number of files to process (for testing). If not specified, processes all files.'
    )
    parser.add_argument(
        '--nworkers',
        type=int,
        default=24,
        help='Number of parallel workers (default: 24)'
    )
    parser.add_argument(
        '--auto-detect-max',
        action='store_true',
        help='Automatically detect maximum audio duration from input files. If not set, uses default value of 391 frames.'
    )
    parser.add_argument(
        '--max-time-frames',
        type=int,
        default=None,
        help='Manually specify maximum time frames for padding. Overrides both default and auto-detection.'
    )

    args = parser.parse_args()

    # Determine target length for spectrograms
    if args.max_time_frames is not None:
        # Manual override
        target_length = args.max_time_frames
        print(f"Using manually specified max time frames: {target_length}")
    elif args.auto_detect_max:
        # Auto-detect from input files
        target_length = compute_max_time_frames(args.input_dir)
    else:
        # Use default
        target_length = MAX_TIME_FRAMES
        print(f"Using default max time frames: {target_length}")

    print("="*60)
    print("Log-Mel Spectrogram Generation")
    print("="*60)
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Metadata file: {args.metadata}")
    print(f"\nParameters:")
    print(f"  - Sampling rate: {SAMPLING_RATE} Hz")
    print(f"  - N_mels: {N_MELS}")
    print(f"  - N_fft: {N_FFT}")
    print(f"  - Hop length: {HOP_LENGTH}")
    print(f"  - Max time frames: {target_length}")
    print(f"  - Target shape: ({N_MELS}, {target_length})")
    print(f"  - Workers: {args.nworkers}")
    if args.limit is not None:
        print(f"  - Limit: {args.limit} files (TESTING MODE)")
    print("="*60)
    print()

    process_dataset(args.input_dir, args.output_dir, args.metadata, args.limit, args.nworkers, target_length)

if __name__ == "__main__":
    main()
