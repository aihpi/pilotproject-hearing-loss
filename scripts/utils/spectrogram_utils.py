#!/usr/bin/env python3
"""
Utility functions for generating and processing log-mel spectrograms.

Reused from generate_spectrograms.py with adaptations for incremental learning pipeline.
"""

import numpy as np
import librosa
from pathlib import Path
import warnings


def load_and_generate_spectrogram(
    audio_path: Path,
    sampling_rate: int = 16000,
    n_mels: int = 128,
    n_fft: int = 400,
    hop_length: int = 160,
    max_time_frames: int = 391
) -> np.ndarray:
    """
    Load audio file and generate log-mel spectrogram.

    Args:
        audio_path: Path to .wav file
        sampling_rate: Target sampling rate (Hz)
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length for STFT
        max_time_frames: Pad/truncate to this length

    Returns:
        Log-mel spectrogram of shape (n_mels, max_time_frames)
    """
    # Suppress librosa warnings for short audio files
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)

        # Load audio
        y, sr = librosa.load(audio_path, sr=sampling_rate)

        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            fmax=sr // 2  # Nyquist frequency
        )

        # Convert to log scale (dB)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Pad to target length if needed
        current_length = log_mel_spec.shape[1]
        if current_length < max_time_frames:
            # Pad with the minimum value (silence)
            pad_width = max_time_frames - current_length
            log_mel_spec = np.pad(
                log_mel_spec,
                ((0, 0), (0, pad_width)),
                mode='constant',
                constant_values=log_mel_spec.min()
            )
        elif current_length > max_time_frames:
            # Truncate
            log_mel_spec = log_mel_spec[:, :max_time_frames]

        return log_mel_spec


def flatten_spectrogram(spec: np.ndarray) -> np.ndarray:
    """
    Flatten 2D spectrogram to 1D vector.

    Args:
        spec: Spectrogram of shape (n_mels, n_time_frames)

    Returns:
        Flattened vector of shape (n_mels * n_time_frames,)

    Example:
        >>> spec = np.random.randn(128, 391)
        >>> flat = flatten_spectrogram(spec)
        >>> flat.shape
        (50048,)
    """
    return spec.flatten()


def extract_word_from_filename(filename: str) -> str:
    """
    Parse word from filename format: {speaker_id}_{word_index}_{word}.wav

    Args:
        filename: Audio filename (with or without extension)

    Returns:
        Extracted word string

    Examples:
        >>> extract_word_from_filename("100038_5_hello.wav")
        'hello'
        >>> extract_word_from_filename("100038_12_don't.wav")
        "don't"
        >>> extract_word_from_filename("100038_3_multi_word_phrase.wav")
        'multi_word_phrase'

    Raises:
        ValueError: If filename doesn't match expected format
    """
    # Remove extension if present
    stem = Path(filename).stem

    # Split by underscore
    parts = stem.split('_')

    if len(parts) < 3:
        raise ValueError(
            f"Invalid filename format: {filename}. "
            f"Expected format: {{speaker_id}}_{{word_index}}_{{word}}.wav"
        )

    # Join all parts after index (handles multi-word cases with underscores)
    word = '_'.join(parts[2:])

    return word


def extract_metadata_from_filename(filename: str) -> dict:
    """
    Parse all metadata from filename format: {speaker_id}_{word_index}_{word}.wav

    Args:
        filename: Audio filename (with or without extension)

    Returns:
        Dictionary with keys: speaker_id, word_index, word

    Example:
        >>> extract_metadata_from_filename("100038_5_hello.wav")
        {'speaker_id': '100038', 'word_index': '5', 'word': 'hello'}
    """
    # Remove extension if present
    stem = Path(filename).stem

    # Split by underscore
    parts = stem.split('_')

    if len(parts) < 3:
        raise ValueError(
            f"Invalid filename format: {filename}. "
            f"Expected format: {{speaker_id}}_{{word_index}}_{{word}}.wav"
        )

    return {
        'speaker_id': parts[0],
        'word_index': parts[1],
        'word': '_'.join(parts[2:])
    }
