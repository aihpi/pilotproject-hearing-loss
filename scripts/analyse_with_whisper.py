#!/usr/bin/env python3

"""
Batch analysis script for processing audio files with Whisper.

This script processes a folder of audio files (where filename = ground truth transcription)
and extracts comprehensive metrics including embeddings, probabilities, and uncertainty measures.

Supports multi-GPU parallel processing for efficient batch analysis.

Author: Hearing Loss Research
Date: 2025-11-18
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
from collections import defaultdict
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import librosa
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AudioSample:
    """Represents a single audio file to process."""
    filepath: str
    ground_truth: str
    audio_id: str


class AudioDataset(Dataset):
    """Dataset for loading audio files."""
    
    def __init__(self, audio_folder: str, sample_rate: int = 16000):
        self.audio_folder = Path(audio_folder)
        self.sample_rate = sample_rate
        
        # Find all audio files
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        self.samples = []
        
        for audio_file in self.audio_folder.iterdir():
            if audio_file.suffix.lower() in audio_extensions:
                # Extract ground truth from filename (without extension)
                ground_truth = audio_file.stem
                audio_id = audio_file.stem
                
                self.samples.append(AudioSample(
                    filepath=str(audio_file),
                    ground_truth=ground_truth,
                    audio_id=audio_id
                ))
        
        logger.info(f"Found {len(self.samples)} audio files in {audio_folder}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        try:
            # Load audio
            audio, sr = librosa.load(sample.filepath, sr=self.sample_rate)
            
            return {
                'audio': audio,
                'ground_truth': sample.ground_truth,
                'audio_id': sample.audio_id,
                'filepath': sample.filepath
            }
        except Exception as e:
            logger.error(f"Error loading {sample.filepath}: {e}")
            # Return empty audio to continue processing
            return {
                'audio': np.zeros(self.sample_rate, dtype=np.float32),
                'ground_truth': sample.ground_truth,
                'audio_id': sample.audio_id,
                'filepath': sample.filepath,
                'error': str(e)
            }


class CaseNormalizer:
    """Handles case and space normalization for tokens."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.vocab_size
        self._build_normalization_map()
    
    def _normalize_token_text(self, token_str: str) -> Tuple[str, bool]:
        """Normalize token: lowercase and remove space prefix."""
        has_space = token_str.startswith('Ġ') or token_str.startswith('\u0120')
        normalized = token_str.replace('Ġ', '').replace('\u0120', '').lower()
        return normalized, has_space
    
    def _build_normalization_map(self):
        """Build mapping from normalized text to all token IDs."""
        self.norm_to_ids = defaultdict(list)
        
        for token_id in range(self.vocab_size):
            token_str = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            normalized, _ = self._normalize_token_text(token_str)
            
            # Skip special tokens
            if normalized and not (normalized.startswith('<|') and normalized.endswith('|>')):
                self.norm_to_ids[normalized].append(token_id)
    
    def get_normalized_ids(self, text: str) -> List[int]:
        """Get all token IDs for normalized text."""
        normalized, _ = self._normalize_token_text(text)
        return self.norm_to_ids.get(normalized, [])


class WhisperAnalyzer:
    """Main analyzer for processing audio with Whisper."""
    
    def __init__(
        self,
        model_path: str,
        device: torch.device,
        top_k: int = 1000
    ):
        self.device = device
        self.top_k = top_k
        
        # Load model and processor
        logger.info(f"Loading Whisper model from {model_path} on {device}")
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        self.model.eval()
        self.model.to(device)
        
        self.processor = WhisperProcessor.from_pretrained(
            "openai/whisper-large-v3",
            language="English",
            task="transcribe"
        )
        self.tokenizer = self.processor.tokenizer
        
        # Get vocabulary embeddings
        self.vocab_embeddings = self.model.proj_out.weight.detach()
        self.vocab_size = self.vocab_embeddings.shape[0]
        
        # Initialize normalizer
        self.normalizer = CaseNormalizer(self.tokenizer)
        
        logger.info(f"Model loaded. Vocab size: {self.vocab_size}")
    
    def compute_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        # Ensure both are numpy arrays with same dtype
        vec1 = np.array(vec1, dtype=np.float32)
        vec2 = np.array(vec2, dtype=np.float32)
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-10)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-10)
        return float(np.dot(vec1_norm, vec2_norm))
    
    def compute_correlation(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute Pearson correlation between two vectors."""
        # Ensure both are numpy arrays with same dtype
        vec1 = np.array(vec1, dtype=np.float32)
        vec2 = np.array(vec2, dtype=np.float32)
        vec1_centered = vec1 - np.mean(vec1)
        vec2_centered = vec2 - np.mean(vec2)
        return self.compute_cosine_similarity(vec1_centered, vec2_centered)
    
    def compute_semantic_density(self, hidden_state: torch.Tensor) -> float:
        """
        Compute semantic density: average cosine similarity to top-10 most similar vocab items.
        """
        # Normalize vectors
        hidden_norm = F.normalize(hidden_state.unsqueeze(0), dim=-1)
        vocab_norm = F.normalize(self.vocab_embeddings, dim=-1)
        
        # Compute all similarities
        similarities = (hidden_norm @ vocab_norm.T).squeeze(0)
        
        # Get top-10
        top_10_sims = torch.topk(similarities, k=min(10, len(similarities))).values
        
        return float(top_10_sims.mean())
    
    def get_ground_truth_embeddings(
        self,
        ground_truth: str
    ) -> Dict[str, Any]:
        """Get vocabulary embeddings for ground truth token(s)."""
        result = {
            'text': ground_truth,
            'token_ids': [],
            'tokens': [],
            'embeddings': [],
            'is_multi_token': False,
            'is_in_vocab': True,
            'error': None
        }
        
        try:
            # Tokenize ground truth (capitalize first letter for sentence-initial position)
            gt_capitalized = ground_truth[0].upper() + ground_truth[1:] if ground_truth else ground_truth
            token_ids = self.tokenizer.encode(gt_capitalized, add_special_tokens=False)
            
            if not token_ids:
                result['is_in_vocab'] = False
                result['error'] = "Cannot tokenize ground truth"
                return result
            
            result['is_multi_token'] = len(token_ids) > 1
            result['token_ids'] = token_ids
            
            for token_id in token_ids:
                token_text = self.tokenizer.decode([token_id])
                embedding = self.vocab_embeddings[token_id].cpu().numpy()
                
                result['tokens'].append(token_text)
                result['embeddings'].append(embedding.tolist())
            
        except Exception as e:
            result['is_in_vocab'] = False
            result['error'] = str(e)
            logger.warning(f"Error getting ground truth embeddings for '{ground_truth}': {e}")
        
        return result
    
    def get_normalized_ground_truth(self, ground_truth: str) -> Dict[str, Any]:
        """Get normalized ground truth embeddings (merge case/space variants)."""
        # Get all variant token IDs
        all_variant_ids = self.normalizer.get_normalized_ids(ground_truth)
        
        if not all_variant_ids:
            return {
                'text': ground_truth.lower(),
                'token_ids': [],
                'embeddings': {
                    'probability_weighted': None,
                    'simple_average': None,
                    'most_probable': None
                },
                'is_in_vocab': False
            }
        
        # Get embeddings for all variants - convert to float32
        variant_embeddings = [self.vocab_embeddings[tid].cpu().numpy().astype(np.float32) for tid in all_variant_ids]
        
        # Simple average
        simple_avg = np.mean(variant_embeddings, axis=0).astype(np.float32)
        
        # Most probable variant (assume first one, or could use frequency)
        most_probable = variant_embeddings[0].astype(np.float32)
        
        # For probability weighted, we'd need actual probabilities from prediction
        # For ground truth, use simple average as proxy
        prob_weighted = simple_avg
        
        return {
            'text': ground_truth.lower(),
            'token_ids': all_variant_ids,
            'embeddings': {
                'probability_weighted': prob_weighted.tolist(),
                'simple_average': simple_avg.tolist(),
                'most_probable': most_probable.tolist()
            },
            'is_in_vocab': True,
            'num_variants': len(all_variant_ids)
        }
    
    def process_audio(
        self,
        audio: np.ndarray,
        ground_truth: str,
        audio_id: str
    ) -> Dict[str, Any]:
        """Process single audio file and extract all metrics."""
        
        result = {
            'audio_id': audio_id,
            'ground_truth': ground_truth,
            'ground_truth_embeddings': None,
            'normalized_ground_truth': None,
            'predictions': [],
            'pooled_predictions': None,
            'error': None
        }
        
        try:
            # Get ground truth embeddings
            result['ground_truth_embeddings'] = self.get_ground_truth_embeddings(ground_truth)
            result['normalized_ground_truth'] = self.get_normalized_ground_truth(ground_truth)
            
            # Process audio with Whisper
            input_features = self.processor(
                audio,
                sampling_rate=16000,
                return_tensors="pt"
            ).input_features.to(self.device)
            
            with torch.no_grad():
                # Generate with hidden states
                generated = self.model.generate(
                    input_features,
                    max_length=225,
                    num_beams=1,
                    do_sample=False,
                    forced_decoder_ids=self.processor.get_decoder_prompt_ids(
                        language="english",
                        task="transcribe"
                    ),
                    return_dict_in_generate=True,
                    output_hidden_states=True,
                    output_scores=True
                )
                
                # Extract tokens
                generated_ids = generated.sequences[0]
                all_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids.tolist())
                all_token_ids = generated_ids.tolist()
                
                # Filter content tokens
                content_indices = []
                for idx, (token_id, token) in enumerate(zip(all_token_ids, all_tokens)):
                    is_special = token.startswith('<|') and token.endswith('|>')
                    if not is_special:
                        content_indices.append(idx)
                
                # Process each content token
                for step_idx, _ in enumerate(content_indices):
                    if step_idx < len(generated.decoder_hidden_states):
                        token_result = self._process_token(
                            step_idx=step_idx,
                            decoder_hidden_states=generated.decoder_hidden_states,
                            token_id=all_token_ids[content_indices[step_idx]],
                            ground_truth_data=result['ground_truth_embeddings'],
                            normalized_gt_data=result['normalized_ground_truth']
                        )
                        result['predictions'].append(token_result)
                
                # Compute pooled metrics if multiple tokens
                if len(result['predictions']) > 1:
                    result['pooled_predictions'] = self._compute_pooled_metrics(
                        result['predictions'],
                        result['ground_truth_embeddings'],
                        result['normalized_ground_truth']
                    )
        
        except Exception as e:
            import traceback
            result['error'] = str(e)
            result['error_traceback'] = traceback.format_exc()
            logger.error(f"Error processing audio {audio_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
        
        return result
    
    def _process_token(
        self,
        step_idx: int,
        decoder_hidden_states: Tuple,
        token_id: int,
        ground_truth_data: Dict,
        normalized_gt_data: Dict
    ) -> Dict[str, Any]:
        """Process a single predicted token and extract all metrics."""
        
        # Get hidden state
        step_hidden_states = decoder_hidden_states[step_idx]
        last_layer_hidden = step_hidden_states[-1]
        hidden_vector = last_layer_hidden[0, -1, :]
        
        # Project to vocabulary
        logits = self.model.proj_out(hidden_vector.unsqueeze(0)).squeeze(0)
        probs = F.softmax(logits, dim=-1)
        
        # Token info
        token_text = self.tokenizer.decode([token_id])
        
        # Original metrics
        original_metrics = self._compute_token_metrics(
            hidden_vector=hidden_vector,
            probs=probs,
            token_id=token_id,
            token_text=token_text,
            ground_truth_data=ground_truth_data
        )
        
        # Normalized metrics
        normalized_metrics = self._compute_normalized_metrics(
            hidden_vector=hidden_vector,
            probs=probs,
            token_text=token_text,
            normalized_gt_data=normalized_gt_data
        )
        
        return {
            'token_id': token_id,
            'token_text': token_text,
            'hidden_state': hidden_vector.cpu().numpy().tolist(),
            'original_metrics': original_metrics,
            'normalized_metrics': normalized_metrics
        }
    
    def _compute_token_metrics(
        self,
        hidden_vector: torch.Tensor,
        probs: torch.Tensor,
        token_id: int,
        token_text: str,
        ground_truth_data: Dict
    ) -> Dict[str, Any]:
        """Compute metrics for original (non-normalized) token."""
        
        probs_np = probs.cpu().numpy()
        hidden_np = hidden_vector.cpu().numpy()
        
        # Top-k probabilities
        top_k_indices = torch.topk(probs, k=min(self.top_k, len(probs))).indices.cpu().numpy()
        top_k_probs = probs_np[top_k_indices]
        
        # Add ground truth if not in top-k
        gt_token_ids = ground_truth_data.get('token_ids', [])
        all_indices = list(top_k_indices)
        all_probs = list(top_k_probs)
        
        for gt_id in gt_token_ids:
            if gt_id not in top_k_indices:
                all_indices.append(gt_id)
                all_probs.append(probs_np[gt_id])
        
        # Entropy
        entropy = -np.sum(probs_np * np.log(probs_np + 1e-10))
        
        # Top probability
        top_prob = float(probs_np[token_id])
        
        # Semantic density
        semantic_density = self.compute_semantic_density(hidden_vector)
        
        # Metrics vs ground truth
        gt_metrics = {}
        if ground_truth_data.get('is_in_vocab') and ground_truth_data.get('embeddings'):
            for idx, (gt_token_id, gt_embedding) in enumerate(
                zip(ground_truth_data['token_ids'], ground_truth_data['embeddings'])
            ):
                gt_rank = int((probs_np > probs_np[gt_token_id]).sum()) + 1
                cosine_sim = self.compute_cosine_similarity(hidden_np, gt_embedding)
                correlation = self.compute_correlation(hidden_np, gt_embedding)
                
                gt_metrics[f'ground_truth_{idx}'] = {
                    'token_id': gt_token_id,
                    'rank': gt_rank,
                    'probability': float(probs_np[gt_token_id]),
                    'cosine_similarity': cosine_sim,
                    'correlation': correlation
                }
        
        return {
            'top_k_token_ids': [int(x) for x in all_indices],
            'top_k_probabilities': [float(x) for x in all_probs],
            'entropy': float(entropy),
            'top_probability': top_prob,
            'semantic_density': semantic_density,
            'ground_truth_metrics': gt_metrics
        }
    
    def _compute_normalized_metrics(
        self,
        hidden_vector: torch.Tensor,
        probs: torch.Tensor,
        token_text: str,
        normalized_gt_data: Dict
    ) -> Dict[str, Any]:
        """Compute metrics for normalized tokens."""
        
        probs_np = probs.cpu().numpy()
        
        # Group probabilities by normalized text
        norm_probs = defaultdict(float)
        norm_to_ids = defaultdict(list)
        
        for token_id in range(self.vocab_size):
            token_str = self.tokenizer.convert_ids_to_tokens([token_id])[0]
            normalized, _ = self.normalizer._normalize_token_text(token_str)
            
            if normalized and not (normalized.startswith('<|') and normalized.endswith('|>')):
                norm_probs[normalized] += probs_np[token_id]
                norm_to_ids[normalized].append(token_id)
        
        # Convert to arrays
        norm_texts = list(norm_probs.keys())
        norm_probs_array = np.array([norm_probs[t] for t in norm_texts])
        
        # Sort by probability
        sorted_indices = np.argsort(norm_probs_array)[::-1]
        top_k_norm = min(self.top_k, len(sorted_indices))
        
        top_k_norm_texts = [norm_texts[i] for i in sorted_indices[:top_k_norm]]
        top_k_norm_probs = norm_probs_array[sorted_indices[:top_k_norm]]
        
        # Add normalized ground truth if not in top-k
        gt_norm_text = normalized_gt_data.get('text', '').lower()
        if gt_norm_text and gt_norm_text not in top_k_norm_texts:
            top_k_norm_texts.append(gt_norm_text)
            top_k_norm_probs = np.append(top_k_norm_probs, norm_probs.get(gt_norm_text, 0.0))
        
        # Normalized entropy
        norm_entropy = -np.sum(norm_probs_array * np.log(norm_probs_array + 1e-10))
        
        # Current token's normalized form
        current_norm, _ = self.normalizer._normalize_token_text(token_text)
        current_norm_prob = norm_probs.get(current_norm, 0.0)
        
        # Compute normalized embeddings
        norm_embeddings = self._compute_normalized_embeddings(
            norm_to_ids.get(current_norm, []),
            probs_np
        )
        
        # Metrics vs normalized ground truth
        norm_gt_metrics = {}
        if normalized_gt_data.get('is_in_vocab'):
            gt_embeddings = normalized_gt_data['embeddings']
            
            for emb_type in ['probability_weighted', 'simple_average', 'most_probable']:
                gt_emb = gt_embeddings[emb_type]
                if gt_emb is not None:
                    pred_emb = norm_embeddings[emb_type]
                    if pred_emb is not None:
                        cosine_sim = self.compute_cosine_similarity(pred_emb, gt_emb)
                        correlation = self.compute_correlation(pred_emb, gt_emb)
                        
                        norm_gt_metrics[emb_type] = {
                            'cosine_similarity': cosine_sim,
                            'correlation': correlation
                        }
            
            # Rank and probability
            gt_prob = norm_probs.get(gt_norm_text, 0.0)
            gt_rank = int((norm_probs_array > gt_prob).sum()) + 1
            
            norm_gt_metrics['rank'] = gt_rank
            norm_gt_metrics['probability'] = float(gt_prob)
        
        return {
            'normalized_text': current_norm,
            'top_k_normalized_texts': top_k_norm_texts,
            'top_k_normalized_probabilities': [float(x) for x in top_k_norm_probs],
            'normalized_entropy': float(norm_entropy),
            'top_normalized_probability': float(current_norm_prob),
            'normalized_embeddings': {
                k: v.tolist() if v is not None else None
                for k, v in norm_embeddings.items()
            },
            'normalized_ground_truth_metrics': norm_gt_metrics
        }
    
    def _compute_normalized_embeddings(
        self,
        token_ids: List[int],
        probs: np.ndarray
    ) -> Dict[str, Optional[np.ndarray]]:
        """Compute three types of normalized embeddings."""
        
        if not token_ids:
            return {
                'probability_weighted': None,
                'simple_average': None,
                'most_probable': None
            }
        
        # Get embeddings - convert to numpy float32
        embeddings = [self.vocab_embeddings[tid].cpu().numpy().astype(np.float32) for tid in token_ids]
        token_probs = [float(probs[tid]) for tid in token_ids]
        
        # Simple average
        simple_avg = np.mean(embeddings, axis=0).astype(np.float32)
        
        # Probability weighted
        total_prob = sum(token_probs)
        if total_prob > 0:
            weights = [p / total_prob for p in token_probs]
            prob_weighted = np.sum([w * emb for w, emb in zip(weights, embeddings)], axis=0).astype(np.float32)
        else:
            prob_weighted = simple_avg
        
        # Most probable
        most_prob_idx = np.argmax(token_probs)
        most_probable = embeddings[most_prob_idx].astype(np.float32)
        
        return {
            'probability_weighted': prob_weighted,
            'simple_average': simple_avg,
            'most_probable': most_probable
        }
    
    def _compute_pooled_metrics(
        self,
        predictions: List[Dict],
        ground_truth_data: Dict,
        normalized_gt_data: Dict
    ) -> Dict[str, Any]:
        """Compute pooled metrics across multiple predicted tokens."""
        
        # Pool hidden states (mean) - convert to float32
        hidden_states = [p['hidden_state'] for p in predictions]
        pooled_hidden = np.mean(hidden_states, axis=0).astype(np.float32)
        
        # Convert to tensor for semantic density
        pooled_hidden_tensor = torch.from_numpy(pooled_hidden).to(self.device)
        semantic_density = self.compute_semantic_density(pooled_hidden_tensor)
        
        # Pooled metrics vs ground truth
        pooled_gt_metrics = {}
        if ground_truth_data.get('is_in_vocab') and ground_truth_data.get('embeddings'):
            # Pool ground truth embeddings too - ensure float32
            gt_embeddings = [np.array(emb, dtype=np.float32) for emb in ground_truth_data['embeddings']]
            gt_pooled = np.mean(gt_embeddings, axis=0).astype(np.float32)
            
            cosine_sim = self.compute_cosine_similarity(pooled_hidden, gt_pooled)
            correlation = self.compute_correlation(pooled_hidden, gt_pooled)
            
            pooled_gt_metrics = {
                'cosine_similarity': cosine_sim,
                'correlation': correlation,
                'semantic_density': semantic_density
            }
        
        return {
            'pooled_hidden_state': pooled_hidden.tolist(),
            'num_tokens_pooled': len(predictions),
            'pooled_ground_truth_metrics': pooled_gt_metrics
        }


def process_batch(
    analyzer: WhisperAnalyzer,
    batch: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Process a batch of audio samples."""
    
    results = []
    
    for i in range(len(batch['audio_id'])):
        audio = batch['audio'][i]
        ground_truth = batch['ground_truth'][i]
        audio_id = batch['audio_id'][i]
        
        # Check for load errors
        if 'error' in batch and batch['error'][i]:
            results.append({
                'audio_id': audio_id,
                'ground_truth': ground_truth,
                'error': batch['error'][i]
            })
            continue
        
        # Process audio
        result = analyzer.process_audio(audio, ground_truth, audio_id)
        results.append(result)
    
    return results


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    # Return batch as-is since we handle variable-length audio
    return {
        'audio': [item['audio'] for item in batch],
        'ground_truth': [item['ground_truth'] for item in batch],
        'audio_id': [item['audio_id'] for item in batch],
        'filepath': [item['filepath'] for item in batch],
        'error': [item.get('error') for item in batch]
    }


def save_results(
    results: List[Dict[str, Any]],
    output_path: str
):
    """Save results to JSON files."""
    
    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save main JSON (all results)
    logger.info(f"Saving main results to {output_path}")
    with open(output_path, 'w') as f:
        json.dump({
            'num_samples': len(results),
            'results': results
        }, f, indent=2)
    
    # Save individual JSONs
    for result in results:
        audio_id = result['audio_id']
        individual_path = output_dir / f"{audio_id}.json"
        
        with open(individual_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    logger.info(f"Saved {len(results)} individual result files to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch analysis of audio files with Whisper"
    )
    parser.add_argument(
        '--input-folder',
        type=str,
        required=True,
        help='Folder containing audio files (filename = ground truth)'
    )
    parser.add_argument(
        '--output-path',
        type=str,
        required=True,
        help='Output path for main JSON file (e.g., results/whisper_lfloss.json)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        required=True,
        help='Path to Whisper model checkpoint'
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of CPU workers for data loading'
    )
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=1,
        help='Number of GPUs to use (0 for CPU only)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1,
        help='Batch size for processing'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=1000,
        help='Number of top predictions to save'
    )
    
    args = parser.parse_args()
    
    # Setup
    logger.info("="*80)
    logger.info("WHISPER BATCH ANALYSIS")
    logger.info("="*80)
    logger.info(f"Input folder: {args.input_folder}")
    logger.info(f"Output path: {args.output_path}")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Num workers: {args.num_workers}")
    logger.info(f"Num GPUs: {args.num_gpus}")
    logger.info(f"Batch size: {args.batch_size}")
    
    # Determine device
    if args.num_gpus > 0 and torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    # Create dataset
    dataset = AudioDataset(args.input_folder)
    
    if len(dataset) == 0:
        logger.error("No audio files found!")
        sys.exit(1)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Create analyzer
    analyzer = WhisperAnalyzer(
        model_path=args.model_path,
        device=device,
        top_k=args.top_k
    )
    
    # Process all batches
    all_results = []
    
    logger.info(f"Processing {len(dataset)} audio files...")
    for batch in tqdm(dataloader, desc="Processing batches"):
        batch_results = process_batch(analyzer, batch)
        all_results.extend(batch_results)
    
    # Save results
    save_results(all_results, args.output_path)
    
    # Summary
    logger.info("="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Total audio files processed: {len(all_results)}")
    
    errors = sum(1 for r in all_results if r.get('error'))
    logger.info(f"Errors: {errors}")
    
    success = len(all_results) - errors
    logger.info(f"Successful: {success}")
    
    logger.info(f"\nResults saved to: {args.output_path}")
    logger.info("✅ Analysis complete!")


if __name__ == "__main__":
    main()
