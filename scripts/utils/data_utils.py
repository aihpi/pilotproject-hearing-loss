#!/usr/bin/env python3
"""
Utility functions for data preparation and management.
"""

import random
from pathlib import Path
from typing import List
from .spectrogram_utils import extract_word_from_filename


def build_word_list(audio_dir: Path, split: str = 'train', limit: int = None) -> List[str]:
    """
    Extract unique words from audio filenames in a directory.

    Args:
        audio_dir: Base directory containing split subdirectories
        split: 'train', 'test', or 'validation'
        limit: Limit number of files to process (for testing)

    Returns:
        Sorted list of unique words

    Example:
        >>> audio_dir = Path('/path/to/normal_raw')
        >>> words = build_word_list(audio_dir, 'train')
        >>> len(words)
        15234
    """
    split_dir = Path(audio_dir) / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Directory not found: {split_dir}")

    # Get all .wav files
    audio_files = list(split_dir.glob("*.wav"))

    if len(audio_files) == 0:
        raise ValueError(f"No .wav files found in {split_dir}")

    # Apply limit if specified
    if limit is not None:
        audio_files = audio_files[:limit]

    # Extract words from filenames
    words = set()
    for audio_file in audio_files:
        try:
            word = extract_word_from_filename(audio_file.name)
            words.add(word)
        except ValueError as e:
            print(f"Warning: Skipping file {audio_file.name}: {e}")
            continue

    # Return sorted list for deterministic ordering
    return sorted(list(words))


def create_training_order(audio_files: List[Path], seed: int = 42) -> List[Path]:
    """
    Randomize training file order for incremental learning.

    Args:
        audio_files: List of audio file paths
        seed: Random seed for reproducibility

    Returns:
        Shuffled list of file paths

    Example:
        >>> files = [Path(f"file_{i}.wav") for i in range(100)]
        >>> shuffled = create_training_order(files, seed=42)
        >>> shuffled[0] != files[0]  # Order is different
        True
        >>> shuffled2 = create_training_order(files, seed=42)
        >>> shuffled == shuffled2  # But reproducible with same seed
        True
    """
    # Create a copy to avoid modifying the original list
    shuffled_files = audio_files.copy()

    # Shuffle with fixed seed for reproducibility
    random.seed(seed)
    random.shuffle(shuffled_files)

    return shuffled_files


def get_audio_files(audio_dir: Path, split: str = 'train') -> List[Path]:
    """
    Get list of all audio files in a split directory.

    Args:
        audio_dir: Base directory containing split subdirectories
        split: 'train', 'test', or 'validation'

    Returns:
        Sorted list of audio file paths

    Example:
        >>> audio_dir = Path('/path/to/normal_raw')
        >>> files = get_audio_files(audio_dir, 'train')
        >>> len(files)
        6300000
    """
    split_dir = Path(audio_dir) / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Directory not found: {split_dir}")

    # Get all .wav files and sort for deterministic ordering
    audio_files = sorted(split_dir.glob("*.wav"))

    if len(audio_files) == 0:
        raise ValueError(f"No .wav files found in {split_dir}")

    return audio_files


def filter_common_words(
    word_lists: List[List[str]],
    min_occurrences: int = None
) -> List[str]:
    """
    Find words that appear in multiple word lists (e.g., across different conditions).

    Args:
        word_lists: List of word lists to compare
        min_occurrences: Minimum number of lists a word must appear in
                        (default: must appear in all lists)

    Returns:
        Sorted list of common words

    Example:
        >>> list1 = ['cat', 'dog', 'bird']
        >>> list2 = ['cat', 'dog', 'fish']
        >>> list3 = ['cat', 'bird', 'fish']
        >>> filter_common_words([list1, list2, list3], min_occurrences=2)
        ['bird', 'cat', 'dog', 'fish']
        >>> filter_common_words([list1, list2, list3], min_occurrences=3)
        ['cat']
    """
    if not word_lists:
        return []

    if min_occurrences is None:
        min_occurrences = len(word_lists)

    # Count occurrences of each word across all lists
    word_counts = {}
    for word_list in word_lists:
        for word in set(word_list):  # Use set to count each word once per list
            word_counts[word] = word_counts.get(word, 0) + 1

    # Filter by minimum occurrences
    common_words = [
        word for word, count in word_counts.items()
        if count >= min_occurrences
    ]

    return sorted(common_words)
