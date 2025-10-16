#!/usr/bin/env python3
"""
Audio processing module for hearing loss simulation.

This module provides clean, focused audio processing functionality
for applying hearing loss masks to audio data, specifically designed
for processing CommonVoice datasets.
"""

import librosa
import numpy as np
import soundfile as sf
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Clean audio processor for hearing loss simulation.
    
    Streamlined version focused only on the core functionality needed
    for processing CommonVoice datasets with hearing loss masks.
    """
    
    def __init__(self, sample_rate: int = 16000, window_size: int = 2048, hop_length: Optional[int] = None):
        """
        Initialize the audio processor.
        
        Args:
            sample_rate (int): Target sample rate for processing (default: 16000)
            window_size (int): FFT window size (default: 2048)
            hop_length (int): Hop length for STFT (default: window_size // 4)
        """
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.hop_length = hop_length if hop_length is not None else window_size // 4
        
        # Audio data
        self.audio = None
        self.original_sr = None
        
        # Spectral data
        self.spectrogram = None
        self.phase = None
        self.frequencies = None
        
        logger.debug(f"AudioProcessor initialized: sr={sample_rate}, window={window_size}, hop={self.hop_length}")
    
    def load_audio_from_array(self, audio_array: np.ndarray, original_sr: int) -> 'AudioProcessor':
        """
        Load audio from numpy array (e.g., from HuggingFace dataset).
        
        Args:
            audio_array (np.ndarray): Audio data as numpy array
            original_sr (int): Original sample rate of the audio
            
        Returns:
            AudioProcessor: Self for method chaining
        """
        if not isinstance(audio_array, np.ndarray):
            raise TypeError("Audio array must be a numpy array")
        
        if len(audio_array.shape) != 1:
            raise ValueError("Audio array must be 1-dimensional")
        
        self.audio = audio_array.astype(np.float32)
        self.original_sr = original_sr
        
        logger.debug(f"Loaded audio: length={len(self.audio)}, original_sr={original_sr}")
        return self
    
    def resample_audio(self, target_sr: Optional[int] = None) -> 'AudioProcessor':
        """
        Resample audio to target sample rate.
        
        Args:
            target_sr (int, optional): Target sample rate (uses self.sample_rate if None)
            
        Returns:
            AudioProcessor: Self for method chaining
        """
        if self.audio is None:
            raise RuntimeError("No audio loaded. Call load_audio_from_array() first.")
        
        target_sr = target_sr or self.sample_rate
        
        if self.original_sr != target_sr:
            logger.debug(f"Resampling from {self.original_sr}Hz to {target_sr}Hz")
            self.audio = librosa.resample(
                self.audio, 
                orig_sr=self.original_sr, 
                target_sr=target_sr
            )
            self.original_sr = target_sr
        else:
            logger.debug("No resampling needed")
        
        return self
    
    def compute_stft(self) -> 'AudioProcessor':
        """
        Compute Short-Time Fourier Transform.
        
        Returns:
            AudioProcessor: Self for method chaining
        """
        if self.audio is None:
            raise RuntimeError("No audio loaded. Call load_audio_from_array() first.")
        
        logger.debug("Computing STFT...")
        S = librosa.stft(
            y=self.audio, 
            n_fft=self.window_size, 
            hop_length=self.hop_length
        )
        
        self.spectrogram = np.abs(S)
        self.phase = np.angle(S)
        self.frequencies = librosa.fft_frequencies(sr=self.original_sr, n_fft=self.window_size)
        
        logger.debug(f"STFT computed: spectrogram shape={self.spectrogram.shape}")
        return self
    
    def apply_hearing_loss_mask(self, freq_points: np.ndarray, db_thresholds: np.ndarray) -> 'AudioProcessor':
        """
        Apply hearing loss mask to the spectrogram.
        
        Args:
            freq_points (np.ndarray): Frequency points for the hearing loss profile (Hz)
            db_thresholds (np.ndarray): Hearing thresholds in dB at each frequency point
            
        Returns:
            AudioProcessor: Self for method chaining
        """
        if self.spectrogram is None or self.frequencies is None:
            raise RuntimeError("No spectrogram available. Call compute_stft() first.")
        
        if len(freq_points) != len(db_thresholds):
            raise ValueError("freq_points and db_thresholds must have the same length")
        
        logger.debug(f"Applying hearing loss mask: {len(freq_points)} frequency points")
        
        # Create frequency mask
        mask = create_frequency_mask(self.frequencies, freq_points, db_thresholds)
        
        # Apply mask to spectrogram
        self.spectrogram = self.spectrogram * mask[:, np.newaxis]
        
        logger.debug("Hearing loss mask applied")
        return self
    
    def reconstruct_audio(self) -> 'AudioProcessor':
        """
        Reconstruct audio from the (possibly masked) spectrogram.
        
        Returns:
            AudioProcessor: Self for method chaining
        """
        if self.spectrogram is None or self.phase is None:
            raise RuntimeError("No spectrogram data available. Call compute_stft() first.")
        
        logger.debug("Reconstructing audio from spectrogram...")
        
        # Reconstruct complex spectrogram
        stft_complex = self.spectrogram * np.exp(1j * self.phase)
        
        # Inverse STFT
        self.audio = librosa.istft(
            stft_complex, 
            hop_length=self.hop_length,
            length=len(self.audio)  # Preserve original length
        )
        
        logger.debug("Audio reconstructed")
        return self
    
    def normalize_audio(self) -> 'AudioProcessor':
        """
        Normalize audio to prevent clipping.
        
        Returns:
            AudioProcessor: Self for method chaining
        """
        if self.audio is None:
            raise RuntimeError("No audio available.")
        
        max_abs = np.abs(self.audio).max()
        if max_abs > 0:
            self.audio = self.audio / max_abs
            logger.debug("Audio normalized")
        else:
            logger.warning("Audio is silent (all zeros), skipping normalization")
        
        return self
    
    def get_processed_audio(self) -> Tuple[np.ndarray, int]:
        """
        Get the processed audio data.
        
        Returns:
            Tuple[np.ndarray, int]: (audio_array, sample_rate)
        """
        if self.audio is None:
            raise RuntimeError("No audio available.")
        
        return self.audio.copy(), self.original_sr
    
    def process_with_hearing_loss(
        self, 
        audio_array: np.ndarray, 
        original_sr: int, 
        freq_points: np.ndarray, 
        db_thresholds: np.ndarray,
        target_sr: Optional[int] = None
    ) -> Tuple[np.ndarray, int]:
        """
        Complete processing pipeline in one method.
        
        Args:
            audio_array (np.ndarray): Input audio data
            original_sr (int): Original sample rate
            freq_points (np.ndarray): Frequency points for hearing loss profile
            db_thresholds (np.ndarray): Hearing thresholds in dB
            target_sr (int, optional): Target sample rate (uses self.sample_rate if None)
            
        Returns:
            Tuple[np.ndarray, int]: (processed_audio, sample_rate)
        """
        target_sr = target_sr or self.sample_rate
        
        return (self
                .load_audio_from_array(audio_array, original_sr)
                .resample_audio(target_sr)
                .compute_stft()
                .apply_hearing_loss_mask(freq_points, db_thresholds)
                .reconstruct_audio()
                .normalize_audio()
                .get_processed_audio())


def create_frequency_mask(frequencies: np.ndarray, freq_points: np.ndarray, db_thresholds: np.ndarray) -> np.ndarray:
    """
    Create frequency-dependent attenuation mask.
    
    Args:
        frequencies (np.ndarray): Frequency bins from FFT
        freq_points (np.ndarray): Frequency points for hearing loss profile (Hz)
        db_thresholds (np.ndarray): Hearing thresholds in dB at each frequency point
        
    Returns:
        np.ndarray: Frequency mask (linear scale)
    """
    # Interpolate thresholds to match frequency bins
    interpolated_db = np.interp(
        frequencies, 
        freq_points, 
        db_thresholds, 
        left=db_thresholds[0], 
        right=db_thresholds[-1]
    )
    
    # Convert dB to linear scale attenuation
    # Higher dB threshold = more hearing loss = more attenuation
    mask = 10 ** (-interpolated_db / 20)
    
    return mask


def validate_audio_data(audio_array: np.ndarray, sample_rate: int) -> bool:
    """
    Validate audio data for processing.
    
    Args:
        audio_array (np.ndarray): Audio data to validate
        sample_rate (int): Sample rate to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not isinstance(audio_array, np.ndarray):
        logger.error("Audio data must be a numpy array")
        return False
    
    if len(audio_array.shape) != 1:
        logger.error("Audio data must be 1-dimensional")
        return False
    
    if len(audio_array) == 0:
        logger.error("Audio data is empty")
        return False
    
    if sample_rate <= 0:
        logger.error(f"Invalid sample rate: {sample_rate}")
        return False
    
    if np.all(audio_array == 0):
        logger.warning("Audio data is silent (all zeros)")
    
    if np.any(np.isnan(audio_array)) or np.any(np.isinf(audio_array)):
        logger.error("Audio data contains NaN or infinite values")
        return False
    
    return True


# Hearing loss profiles
def get_normal_hearing_profile() -> Tuple[np.ndarray, np.ndarray]:
    """
    Get normal hearing profile.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (frequency_points, thresholds_db)
    """
    freq_points = np.array([125, 250, 500, 1000, 2000, 4000, 8000])
    thresholds = np.full_like(freq_points, 10.0, dtype=np.float32)  # 10 dB across all frequencies
    return freq_points, thresholds


def get_low_frequency_loss_profile() -> Tuple[np.ndarray, np.ndarray]:
    """
    Get low-frequency hearing loss profile.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (frequency_points, thresholds_db)
    """
    freq_points = np.array([125, 250, 500, 1000, 2000, 4000, 8000])
    thresholds = np.linspace(100, 10, len(freq_points)).astype(np.float32)  # 100 dB at low freq, 10 dB at high freq
    return freq_points, thresholds


def get_high_frequency_loss_profile() -> Tuple[np.ndarray, np.ndarray]:
    """
    Get high-frequency hearing loss profile.
    
    Returns:
        Tuple[np.ndarray, np.ndarray]: (frequency_points, thresholds_db)
    """
    freq_points = np.array([125, 250, 500, 1000, 2000, 4000, 8000])
    thresholds = np.linspace(10, 100, len(freq_points)).astype(np.float32)  # 10 dB at low freq, 100 dB at high freq
    return freq_points, thresholds