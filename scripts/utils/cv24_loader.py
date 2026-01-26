#!/usr/bin/env python3
"""
CommonVoice 24.0 TSV+MP3 data loader.

This module provides utilities for loading Mozilla CommonVoice 24.0 datasets
in their native TSV+MP3 format, presenting them with the same interface as
the previous HuggingFace datasets format.

Expected directory structure:
    data/CommonVoiceEN_v24/
        clips/
            common_voice_en_12345.mp3
            ...
        train.tsv
        dev.tsv
        test.tsv

Usage:
    from utils.cv24_loader import load_cv24_dataset

    dataset = load_cv24_dataset("data/CommonVoiceEN_v24", splits=["dev"])
    for sample in dataset["dev"]:
        audio = sample["audio"]["array"]
        text = sample["sentence"]
"""

import csv
import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Any, Union

import librosa
import numpy as np

logger = logging.getLogger(__name__)


# Split name mapping (HuggingFace uses 'validation', CV24 uses 'dev')
SPLIT_NAME_MAP = {
    'validation': 'dev',
    'dev': 'dev',
    'train': 'train',
    'test': 'test',
}


def normalize_split_name(split_name: str) -> str:
    """Convert HuggingFace split names to CV24 split names."""
    return SPLIT_NAME_MAP.get(split_name, split_name)


class CV24Sample:
    """
    A single sample from CommonVoice 24.0 dataset.

    Provides lazy loading of audio data while maintaining compatibility
    with the HuggingFace dataset sample format.

    Args:
        row: Dict from TSV row with metadata
        clips_dir: Path to clips/ directory containing MP3 files
        target_sr: Target sample rate (None = keep original ~48kHz)
    """

    def __init__(self, row: Dict[str, str], clips_dir: Path, target_sr: Optional[int] = None):
        self._row = row
        self._clips_dir = clips_dir
        self._target_sr = target_sr
        self._audio_cache = None

    @property
    def path(self) -> str:
        """Original path value from TSV (filename only)."""
        return self._row.get('path', '')

    @property
    def sentence(self) -> str:
        """Transcription text."""
        return self._row.get('sentence', '')

    @property
    def audio(self) -> Dict[str, Any]:
        """
        Audio data in HuggingFace-compatible format.

        Returns:
            Dict with 'array' (np.ndarray) and 'sampling_rate' (int)
        """
        if self._audio_cache is None:
            self._audio_cache = self._load_audio()
        return self._audio_cache

    def _load_audio(self) -> Dict[str, Any]:
        """Load and optionally resample audio from MP3 file."""
        audio_path = self._clips_dir / self._row['path']

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Load with librosa (handles MP3 via audioread/ffmpeg)
        # sr=None preserves original sample rate
        audio_array, sr = librosa.load(
            str(audio_path),
            sr=self._target_sr,  # None = keep original
            mono=True
        )

        return {
            'array': audio_array.astype(np.float32),
            'sampling_rate': sr
        }

    def __getitem__(self, key: str) -> Any:
        """Dict-like access for compatibility with existing code."""
        if key == 'audio':
            return self.audio
        elif key == 'sentence':
            return self.sentence
        elif key == 'path':
            return self.path
        elif key in self._row:
            return self._row[key]
        else:
            raise KeyError(f"Unknown key: {key}")

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get() method."""
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self):
        """Return available keys."""
        return list(self._row.keys()) + ['audio']

    def copy(self) -> Dict[str, Any]:
        """
        Return a dict copy of the sample (for mutation in processing).

        This is important for multiprocessing - CV24Sample objects contain
        file handles and lazy loaders that can't be pickled. The dict copy
        contains only plain Python objects.
        """
        return {
            'audio': {
                'array': self.audio['array'].copy(),
                'sampling_rate': self.audio['sampling_rate']
            },
            'sentence': self.sentence,
            'path': self.path,
            **{k: v for k, v in self._row.items() if k not in ['path', 'sentence']}
        }


class CV24Split:
    """
    A dataset split (train/dev/test) from CommonVoice 24.0.

    Provides both indexed access and iteration, with lazy audio loading.
    Compatible with HuggingFace Dataset interface for existing code.

    Args:
        tsv_path: Path to TSV file (train.tsv, dev.tsv, or test.tsv)
        clips_dir: Path to clips/ directory containing MP3 files
        target_sr: Target sample rate (None = keep original ~48kHz)
    """

    def __init__(self, tsv_path: Path, clips_dir: Path, target_sr: Optional[int] = None):
        self.tsv_path = tsv_path
        self.clips_dir = clips_dir
        self.target_sr = target_sr
        self._rows: List[Dict[str, str]] = []
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load TSV metadata into memory (not audio, just text)."""
        logger.info(f"Loading metadata from {self.tsv_path}")

        with open(self.tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter='\t')
            self._rows = list(reader)

        logger.info(f"Loaded {len(self._rows)} samples from {self.tsv_path.name}")

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: Union[int, slice]) -> Union[CV24Sample, List[CV24Sample]]:
        if isinstance(idx, slice):
            return [
                CV24Sample(self._rows[i], self.clips_dir, self.target_sr)
                for i in range(*idx.indices(len(self._rows)))
            ]
        if idx < 0:
            idx = len(self._rows) + idx
        if idx < 0 or idx >= len(self._rows):
            raise IndexError(f"Index {idx} out of range for split with {len(self._rows)} samples")
        return CV24Sample(self._rows[idx], self.clips_dir, self.target_sr)

    def __iter__(self) -> Iterator[CV24Sample]:
        for row in self._rows:
            yield CV24Sample(row, self.clips_dir, self.target_sr)


class CV24DatasetDict:
    """
    A collection of CV24 splits, mimicking HuggingFace DatasetDict.

    Usage:
        dataset = load_cv24_dataset("data/CommonVoiceEN_v24")
        train_data = dataset['train']
        for sample in train_data:
            audio = sample['audio']['array']
            text = sample['sentence']
    """

    def __init__(self):
        self._splits: Dict[str, CV24Split] = {}

    def __getitem__(self, split_name: str) -> CV24Split:
        return self._splits[split_name]

    def __setitem__(self, split_name: str, split: CV24Split) -> None:
        self._splits[split_name] = split

    def __contains__(self, split_name: str) -> bool:
        return split_name in self._splits

    def keys(self):
        return self._splits.keys()

    def values(self):
        return self._splits.values()

    def items(self):
        return self._splits.items()

    def __iter__(self):
        return iter(self._splits)

    def __len__(self):
        return len(self._splits)


def load_cv24_dataset(
    base_dir: str,
    splits: Optional[List[str]] = None,
    target_sr: Optional[int] = None
) -> CV24DatasetDict:
    """
    Load CommonVoice 24.0 dataset from TSV+MP3 format.

    Args:
        base_dir: Path to CV24 data directory (containing clips/ and *.tsv files)
        splits: List of splits to load (default: ['train', 'dev', 'test'])
        target_sr: Target sample rate for audio (None = keep original ~48kHz)

    Returns:
        CV24DatasetDict with loaded splits

    Example:
        >>> dataset = load_cv24_dataset("data/CommonVoiceEN_v24", splits=["dev", "test"])
        >>> len(dataset['dev'])
        16403
        >>> sample = dataset['dev'][0]
        >>> sample['audio']['sampling_rate']
        48000
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {base_dir}")

    clips_dir = base_path / "clips"
    if not clips_dir.exists():
        raise FileNotFoundError(f"Clips directory not found: {clips_dir}")

    # Default splits if not specified
    if splits is None:
        splits = ['train', 'dev', 'test']

    # Normalize split names (validation -> dev)
    splits = [normalize_split_name(s) for s in splits]

    logger.info(f"Loading CV24 dataset from: {base_dir}")
    logger.info(f"Splits to load: {splits}")

    dataset = CV24DatasetDict()

    for split_name in splits:
        tsv_path = base_path / f"{split_name}.tsv"
        if not tsv_path.exists():
            logger.warning(f"TSV file not found for split '{split_name}': {tsv_path}")
            continue

        dataset[split_name] = CV24Split(tsv_path, clips_dir, target_sr)

    logger.info(f"Dataset loaded successfully")
    logger.info(f"Available splits: {list(dataset.keys())}")
    for split_name, split_data in dataset.items():
        logger.info(f"  - {split_name}: {len(split_data)} samples")

    return dataset
