#!/usr/bin/env python3
"""
Forced alignment using torchaudio MMS_FA model with batching and multi-GPU support.

Processes audio files and their transcripts to generate word-level alignments
stored as JSON files.

Usage:
    python scripts/forced_alignment.py --input-dir data/CommonVoiceENraw --output-dir data/CommonVoiceENJSON
    python scripts/forced_alignment.py --splits train test --batch-size 32 --num-gpus 4
"""

import argparse
import json
import logging
import re
import sys
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading

import torch
import torch.multiprocessing as mp
import torchaudio
from torchaudio.pipelines import MMS_FA as bundle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(process)d] %(message)s",
)
logger = logging.getLogger(__name__)

# MMS_FA expects 16kHz audio
SAMPLE_RATE = 16000

# MMS_FA supported characters: a-z, ', -, * (word boundary)
MMS_FA_SUPPORTED_CHARS = set("abcdefghijklmnopqrstuvwxyz'-")

# Number to word mappings
NUMBER_WORDS = {
    "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
    "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten",
    "11": "eleven", "12": "twelve", "13": "thirteen", "14": "fourteen",
    "15": "fifteen", "16": "sixteen", "17": "seventeen", "18": "eighteen",
    "19": "nineteen", "20": "twenty", "30": "thirty", "40": "forty",
    "50": "fifty", "60": "sixty", "70": "seventy", "80": "eighty", "90": "ninety",
}


@dataclass
class WordAlignment:
    """Word alignment result."""
    word: str
    start_time: float
    end_time: float
    score: float


@dataclass 
class FileTask:
    """A file to be processed."""
    wav_path: Path
    txt_path: Path
    output_path: Path
    file_id: str


def _number_to_words(num_str: str) -> str:
    """Convert a number string to words (1-99)."""
    if num_str == "0":
        return num_str
    if num_str in NUMBER_WORDS:
        return NUMBER_WORDS[num_str]
    try:
        num = int(num_str)
        if num < 0 or num > 99 or num == 0:
            return num_str
        if num < 20:
            return NUMBER_WORDS.get(num_str, num_str)
        tens = (num // 10) * 10
        ones = num % 10
        if ones == 0:
            return NUMBER_WORDS.get(str(tens), num_str)
        tens_word = NUMBER_WORDS.get(str(tens), "")
        ones_word = NUMBER_WORDS.get(str(ones), "")
        if tens_word and ones_word:
            return f"{tens_word}-{ones_word}"
        return num_str
    except ValueError:
        return num_str


def normalize_transcript(text: str) -> str:
    """Normalize transcript for alignment."""
    # Convert to lowercase
    text = text.lower()
    
    # Replace numbers with words
    text = re.sub(r'\b(\d{1,2})\b', lambda m: _number_to_words(m.group(1)), text)
    
    # Keep only supported characters and spaces
    result = []
    for char in text:
        if char in MMS_FA_SUPPORTED_CHARS or char == ' ':
            result.append(char)
        elif char in '.,!?;:"""()[]{}…–—':
            result.append(' ')  # Replace punctuation with space
    
    # Collapse multiple spaces
    text = ''.join(result)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def load_and_preprocess_audio(wav_path: Path) -> tuple[torch.Tensor, float]:
    """Load audio file and preprocess for MMS_FA."""
    waveform, sample_rate = torchaudio.load(wav_path)
    
    # Resample if needed
    if sample_rate != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
        waveform = resampler(waveform)
    
    # Convert to mono if needed
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    duration = waveform.shape[1] / SAMPLE_RATE
    return waveform.squeeze(0), duration  # Return 1D tensor


def align_emission(
    emission: torch.Tensor,
    transcript: str,
    duration: float,
    aligner,
    dictionary: dict,
) -> list[WordAlignment]:
    """Align emission to transcript and return word alignments."""
    normalized = normalize_transcript(transcript)
    if not normalized:
        return []
    
    words = normalized.split()
    
    # Convert words to nested token indices
    tokens_nested = []
    valid_words = []
    for word in words:
        word_tokens = []
        skip_word = False
        for char in word:
            if char in dictionary:
                word_tokens.append(dictionary[char])
            else:
                skip_word = True
                break
        if not skip_word and word_tokens:
            tokens_nested.append(word_tokens)
            valid_words.append(word)
    
    if not tokens_nested:
        return []
    
    # Calculate frame duration
    num_frames = emission.shape[0]
    frame_duration = duration / num_frames
    
    # Run alignment
    try:
        word_spans = aligner(emission, tokens_nested)
    except Exception as e:
        logger.warning(f"Alignment failed: {e}")
        return []
    
    # Convert to WordAlignments
    word_alignments = []
    for word, spans in zip(valid_words, word_spans):
        if not spans:
            continue
        start_frame = spans[0].start
        end_frame = spans[-1].end
        avg_score = sum(span.score for span in spans) / len(spans)
        
        word_alignments.append(WordAlignment(
            word=word,
            start_time=round(start_frame * frame_duration, 4),
            end_time=round(end_frame * frame_duration, 4),
            score=round(avg_score, 4),
        ))
    
    return word_alignments


def process_batch(
    tasks: list[FileTask],
    model: torch.nn.Module,
    aligner,
    dictionary: dict,
    device: torch.device,
) -> list[tuple[str, bool, str]]:
    """
    Process a batch of files together.
    
    Returns list of (file_id, success, message) tuples.
    """
    results = []
    
    # Load all audio files
    waveforms = []
    durations = []
    transcripts = []
    valid_tasks = []
    
    for task in tasks:
        try:
            waveform, duration = load_and_preprocess_audio(task.wav_path)
            transcript = task.txt_path.read_text(encoding="utf-8").strip()
            waveforms.append(waveform)
            durations.append(duration)
            transcripts.append(transcript)
            valid_tasks.append(task)
        except Exception as e:
            results.append((task.file_id, False, f"Load error: {e}"))
    
    if not waveforms:
        return results
    
    # Process each audio individually through model
    # (MMS_FA model wrapper doesn't support batched input due to star_dim expansion)
    for i, task in enumerate(valid_tasks):
        try:
            # Run model inference on single audio - model expects (batch, time)
            waveform = waveforms[i].unsqueeze(0).to(device)
            
            with torch.inference_mode():
                emission, _ = model(waveform)
            
            emission = emission.squeeze(0).cpu()  # (frames, vocab)
            
            alignments = align_emission(
                emission, transcripts[i], durations[i], aligner, dictionary
            )
            
            # Save JSON
            task.output_path.parent.mkdir(parents=True, exist_ok=True)
            result = {
                "file_id": task.file_id,
                "original_transcript": transcripts[i],
                "normalized_transcript": normalize_transcript(transcripts[i]),
                "duration": round(durations[i], 4),
                "words": [
                    {"word": a.word, "start": a.start_time, "end": a.end_time, "score": a.score}
                    for a in alignments
                ],
            }
            with open(task.output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            
            results.append((task.file_id, True, f"{len(alignments)} words"))
        except Exception as e:
            results.append((task.file_id, False, str(e)))
    
    return results


def gpu_worker(
    gpu_id: int,
    task_queue: mp.Queue,
    result_queue: mp.Queue,
    batch_size: int,
    done_event: mp.Event,
):
    """Worker process for a single GPU."""
    device = torch.device(f"cuda:{gpu_id}")
    
    logger.info(f"GPU worker {gpu_id} starting on {device}")
    
    # Load model on this GPU
    model = bundle.get_model().to(device)
    model.eval()
    aligner = bundle.get_aligner()
    dictionary = bundle.get_dict()
    
    logger.info(f"GPU worker {gpu_id} model loaded")
    
    batch = []
    
    while not done_event.is_set() or not task_queue.empty():
        try:
            task = task_queue.get(timeout=0.5)
            batch.append(task)
            
            if len(batch) >= batch_size:
                results = process_batch(batch, model, aligner, dictionary, device)
                for r in results:
                    result_queue.put(r)
                batch = []
                
        except queue.Empty:
            # Process any remaining batch
            if batch:
                results = process_batch(batch, model, aligner, dictionary, device)
                for r in results:
                    result_queue.put(r)
                batch = []
    
    # Process final batch
    if batch:
        results = process_batch(batch, model, aligner, dictionary, device)
        for r in results:
            result_queue.put(r)
    
    logger.info(f"GPU worker {gpu_id} finished")


def process_split_multigpu(
    input_dir: Path,
    output_dir: Path,
    split: str,
    num_gpus: int,
    batch_size: int,
    skip_existing: bool,
    max_files: Optional[int] = None,
) -> tuple[int, int]:
    """Process a split using multiple GPUs."""
    split_input = input_dir / split
    split_output = output_dir / split
    
    if not split_input.exists():
        logger.warning(f"Split directory not found: {split_input}")
        return 0, 0
    
    # Collect file pairs
    wav_files = list(split_input.glob("*.wav"))
    tasks = []
    skipped = 0
    
    for wav_path in wav_files:
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        
        output_path = split_output / f"{wav_path.stem}.json"
        
        if skip_existing and output_path.exists():
            skipped += 1
            continue
        
        tasks.append(FileTask(
            wav_path=wav_path,
            txt_path=txt_path,
            output_path=output_path,
            file_id=wav_path.stem,
        ))
    
    if max_files is not None and max_files > 0:
        tasks = tasks[:max_files]
    
    logger.info(f"Split '{split}': {len(tasks)} files to process, {skipped} skipped")
    
    if not tasks:
        return skipped, 0
    
    # Single GPU path (simpler, no multiprocessing)
    if num_gpus == 1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = bundle.get_model().to(device)
        model.eval()
        aligner = bundle.get_aligner()
        dictionary = bundle.get_dict()
        
        success_count = 0
        error_count = 0
        
        for i in range(0, len(tasks), batch_size):
            batch_tasks = tasks[i:i + batch_size]
            results = process_batch(batch_tasks, model, aligner, dictionary, device)
            
            for file_id, success, message in results:
                if success:
                    success_count += 1
                else:
                    error_count += 1
                    logger.error(f"Error processing {file_id}: {message}")
            
            if (i + batch_size) % 500 == 0 or i + batch_size >= len(tasks):
                logger.info(f"Progress: {min(i + batch_size, len(tasks))}/{len(tasks)} files")
        
        return success_count + skipped, error_count
    
    # Multi-GPU path
    mp.set_start_method('spawn', force=True)
    
    task_queue = mp.Queue()
    result_queue = mp.Queue()
    done_event = mp.Event()
    
    # Start GPU workers
    workers = []
    for gpu_id in range(num_gpus):
        p = mp.Process(
            target=gpu_worker,
            args=(gpu_id, task_queue, result_queue, batch_size, done_event),
        )
        p.start()
        workers.append(p)
    
    # Feed tasks to queue
    for task in tasks:
        task_queue.put(task)
    
    # Signal done and wait for workers
    done_event.set()
    
    # Collect results
    success_count = 0
    error_count = 0
    received = 0
    
    while received < len(tasks):
        try:
            file_id, success, message = result_queue.get(timeout=300)
            received += 1
            if success:
                success_count += 1
            else:
                error_count += 1
                logger.error(f"Error processing {file_id}: {message}")
            
            if received % 500 == 0:
                logger.info(f"Progress: {received}/{len(tasks)} files")
        except queue.Empty:
            # Check if workers are still alive
            alive = sum(1 for w in workers if w.is_alive())
            if alive == 0:
                logger.warning(f"All workers finished but only received {received}/{len(tasks)} results")
                break
    
    # Wait for workers to finish
    for p in workers:
        p.join(timeout=60)
        if p.is_alive():
            p.terminate()
    
    return success_count + skipped, error_count


def process_split_single(
    input_dir: Path,
    output_dir: Path,
    split: str,
    device: torch.device,
    batch_size: int,
    skip_existing: bool,
    max_files: Optional[int] = None,
) -> tuple[int, int]:
    """Process a split on a single device with batching."""
    split_input = input_dir / split
    split_output = output_dir / split
    
    if not split_input.exists():
        logger.warning(f"Split directory not found: {split_input}")
        return 0, 0
    
    # Collect file pairs
    wav_files = list(split_input.glob("*.wav"))
    tasks = []
    skipped = 0
    
    for wav_path in wav_files:
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        
        output_path = split_output / f"{wav_path.stem}.json"
        
        if skip_existing and output_path.exists():
            skipped += 1
            continue
        
        tasks.append(FileTask(
            wav_path=wav_path,
            txt_path=txt_path,
            output_path=output_path,
            file_id=wav_path.stem,
        ))
    
    if max_files is not None and max_files > 0:
        tasks = tasks[:max_files]
    
    logger.info(f"Split '{split}': {len(tasks)} files to process, {skipped} skipped")
    
    if not tasks:
        return skipped, 0
    
    # Load model
    logger.info(f"Loading MMS_FA model on {device}...")
    model = bundle.get_model().to(device)
    model.eval()
    aligner = bundle.get_aligner()
    dictionary = bundle.get_dict()
    
    success_count = 0
    error_count = 0
    
    for i in range(0, len(tasks), batch_size):
        batch_tasks = tasks[i:i + batch_size]
        results = process_batch(batch_tasks, model, aligner, dictionary, device)
        
        for file_id, success, message in results:
            if success:
                success_count += 1
            else:
                error_count += 1
                logger.error(f"Error processing {file_id}: {message}")
        
        if (i + batch_size) % 500 == 0 or i + batch_size >= len(tasks):
            logger.info(f"Progress: {min(i + batch_size, len(tasks))}/{len(tasks)} files")
    
    return success_count + skipped, error_count


def main():
    parser = argparse.ArgumentParser(
        description="Forced alignment using torchaudio MMS_FA with batching and multi-GPU",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/CommonVoiceENraw"),
        help="Input directory containing split subdirectories with wav/txt pairs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/CommonVoiceENJSON"),
        help="Output directory for JSON alignment files",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test", "validation"],
        help="Splits to process",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda, cuda:0, cpu) - ignored if --num-gpus > 1",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use (enables multi-GPU processing)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of audio files to process together in a batch",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already have JSON output",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Maximum number of files to process per split (default: all)",
    )
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Splits: {args.splits}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Num GPUs: {args.num_gpus}")
    logger.info(f"Skip existing: {args.skip_existing}")
    logger.info(f"Max files per split: {args.max_files or 'all'}")
    
    total_success = 0
    total_errors = 0
    
    for split in args.splits:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing split: {split}")
        logger.info(f"{'='*60}")
        
        if args.num_gpus > 1:
            success, errors = process_split_multigpu(
                args.input_dir, args.output_dir, split,
                args.num_gpus, args.batch_size, args.skip_existing, args.max_files,
            )
        else:
            device = torch.device(args.device)
            success, errors = process_split_single(
                args.input_dir, args.output_dir, split,
                device, args.batch_size, args.skip_existing, args.max_files,
            )
        
        total_success += success
        total_errors += errors
        
        logger.info(f"Split '{split}' complete: {success} success, {errors} errors")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"TOTAL: {total_success} success, {total_errors} errors")
    logger.info(f"{'='*60}")
    
    return 0 if total_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
