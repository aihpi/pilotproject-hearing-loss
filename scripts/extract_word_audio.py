#!/usr/bin/env python3
"""
Extract individual word audio segments from full audio files using word boundary JSON files.

Uses word-level alignments from forced alignment JSON files to extract each word
as a separate audio file.

Usage:
    python scripts/extract_word_audio.py
    python scripts/extract_word_audio.py --input-json data/CommonVoiceENJSON --input-audio data/CommonVoiceENraw --output-audio data/CommonVoiceENWords
    python scripts/extract_word_audio.py --nworkers 24 --batch-size 100
"""

import argparse
import json
import logging
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

import torchaudio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(process)d] %(message)s",
)
logger = logging.getLogger(__name__)


def extract_words_from_file(
    json_path: Path,
    audio_dir: Path,
    output_dir: Path,
) -> Tuple[int, int, str]:
    """
    Extract all words from a single audio file based on JSON alignment.
    
    Args:
        json_path: Path to the JSON file with word alignments
        audio_dir: Directory containing the source audio files
        output_dir: Directory to write extracted word audio files
        
    Returns:
        Tuple of (words_extracted, words_failed, file_id)
    """
    try:
        # Load JSON alignment data
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        file_id = data["file_id"]
        words = data.get("words", [])
        
        if not words:
            return 0, 0, file_id
        
        # Find corresponding audio file
        audio_path = audio_dir / f"{file_id}.wav"
        if not audio_path.exists():
            logger.warning(f"Audio file not found: {audio_path}")
            return 0, len(words), file_id
        
        # Load audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        words_extracted = 0
        words_failed = 0
        
        for word_idx, word_info in enumerate(words, start=1):  # 1-based indexing
            try:
                word_text = word_info["word"]
                start_time = word_info["start"]
                end_time = word_info["end"]
                
                # Convert time to samples
                start_sample = int(start_time * sample_rate)
                end_sample = int(end_time * sample_rate)
                
                # Validate sample range
                if start_sample < 0:
                    start_sample = 0
                if end_sample > waveform.shape[1]:
                    end_sample = waveform.shape[1]
                if start_sample >= end_sample:
                    logger.warning(f"Invalid time range for {file_id} word {word_idx}: {start_time}-{end_time}")
                    words_failed += 1
                    continue
                
                # Extract word audio segment
                word_audio = waveform[:, start_sample:end_sample]
                
                # Create output filename: {file_id}_{word_index}_{word}.wav
                # Preserve apostrophes in filename
                output_filename = f"{file_id}_{word_idx}_{word_text}.wav"
                output_path = output_dir / output_filename
                
                # Save extracted word audio
                torchaudio.save(output_path, word_audio, sample_rate)
                words_extracted += 1
                
            except Exception as e:
                logger.warning(f"Failed to extract word {word_idx} from {file_id}: {e}")
                words_failed += 1
        
        return words_extracted, words_failed, file_id
        
    except Exception as e:
        logger.error(f"Failed to process {json_path}: {e}")
        return 0, 0, str(json_path)


def process_batch(
    batch: List[Tuple[Path, Path, Path]],
) -> List[Tuple[int, int, str]]:
    """
    Process a batch of files sequentially within a worker.
    
    Args:
        batch: List of (json_path, audio_dir, output_dir) tuples
        
    Returns:
        List of (words_extracted, words_failed, file_id) tuples
    """
    results = []
    for json_path, audio_dir, output_dir in batch:
        result = extract_words_from_file(json_path, audio_dir, output_dir)
        results.append(result)
    return results


def process_split(
    input_json_dir: Path,
    input_audio_dir: Path,
    output_audio_dir: Path,
    split: str,
    nworkers: int,
    batch_size: int,
) -> Tuple[int, int, int]:
    """
    Process all files in a split (train/test/validation).
    
    Args:
        input_json_dir: Base directory containing JSON alignments
        input_audio_dir: Base directory containing source audio
        output_audio_dir: Base directory for output word audio
        split: Split name (train/test/validation)
        nworkers: Number of parallel workers
        batch_size: Number of files per batch
        
    Returns:
        Tuple of (total_files, total_words_extracted, total_words_failed)
    """
    json_split_dir = input_json_dir / split
    audio_split_dir = input_audio_dir / split
    output_split_dir = output_audio_dir / split
    
    # Check directories exist
    if not json_split_dir.exists():
        logger.warning(f"JSON directory not found: {json_split_dir}")
        return 0, 0, 0
    
    if not audio_split_dir.exists():
        logger.warning(f"Audio directory not found: {audio_split_dir}")
        return 0, 0, 0
    
    # Create output directory
    output_split_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all JSON files
    json_files = sorted(json_split_dir.glob("*.json"))
    total_files = len(json_files)
    
    if total_files == 0:
        logger.info(f"No JSON files found in {json_split_dir}")
        return 0, 0, 0
    
    logger.info(f"Processing {total_files} files in {split} split with {nworkers} workers, batch size {batch_size}")
    
    # Create batches of work items
    work_items = [(json_path, audio_split_dir, output_split_dir) for json_path in json_files]
    batches = [work_items[i:i + batch_size] for i in range(0, len(work_items), batch_size)]
    
    total_words_extracted = 0
    total_words_failed = 0
    files_processed = 0
    
    # Process batches in parallel
    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        futures = {executor.submit(process_batch, batch): batch for batch in batches}
        
        for future in as_completed(futures):
            try:
                results = future.result()
                for words_extracted, words_failed, file_id in results:
                    total_words_extracted += words_extracted
                    total_words_failed += words_failed
                    files_processed += 1
                
                # Log progress periodically
                if files_processed % (batch_size * 10) == 0 or files_processed == total_files:
                    logger.info(
                        f"[{split}] Progress: {files_processed}/{total_files} files, "
                        f"{total_words_extracted} words extracted, {total_words_failed} failed"
                    )
                    
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
    
    logger.info(
        f"[{split}] Completed: {files_processed} files, "
        f"{total_words_extracted} words extracted, {total_words_failed} failed"
    )
    
    return total_files, total_words_extracted, total_words_failed


def main():
    parser = argparse.ArgumentParser(
        description="Extract individual word audio segments from full audio files."
    )
    parser.add_argument(
        "--input-json",
        type=Path,
        default=Path("data/CommonVoiceENJSON"),
        help="Directory containing word alignment JSON files (default: data/CommonVoiceENJSON)",
    )
    parser.add_argument(
        "--input-audio",
        type=Path,
        default=Path("data/CommonVoiceENraw"),
        help="Directory containing source audio files (default: data/CommonVoiceENraw)",
    )
    parser.add_argument(
        "--output-audio",
        type=Path,
        default=Path("data/CommonVoiceENWords"),
        help="Directory to write extracted word audio files (default: data/CommonVoiceENWords)",
    )
    parser.add_argument(
        "--nworkers",
        type=int,
        default=24,
        help="Number of parallel workers (default: 24)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of files to process per batch (default: 100)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test", "validation"],
        help="Splits to process (default: train test validation)",
    )
    
    args = parser.parse_args()
    
    # Validate input directories
    if not args.input_json.exists():
        logger.error(f"Input JSON directory not found: {args.input_json}")
        return 1
    
    if not args.input_audio.exists():
        logger.error(f"Input audio directory not found: {args.input_audio}")
        return 1
    
    # Create output directory
    args.output_audio.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Input JSON: {args.input_json}")
    logger.info(f"Input audio: {args.input_audio}")
    logger.info(f"Output audio: {args.output_audio}")
    logger.info(f"Workers: {args.nworkers}, Batch size: {args.batch_size}")
    logger.info(f"Splits: {args.splits}")
    
    # Process each split
    grand_total_files = 0
    grand_total_words = 0
    grand_total_failed = 0
    
    for split in args.splits:
        total_files, total_words, total_failed = process_split(
            input_json_dir=args.input_json,
            input_audio_dir=args.input_audio,
            output_audio_dir=args.output_audio,
            split=split,
            nworkers=args.nworkers,
            batch_size=args.batch_size,
        )
        grand_total_files += total_files
        grand_total_words += total_words
        grand_total_failed += total_failed
    
    logger.info(
        f"All done! Processed {grand_total_files} files, "
        f"extracted {grand_total_words} words, {grand_total_failed} failed"
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
