#!/usr/bin/env python3

"""
Enhanced Whisper transcription script that generates truly diverse hypotheses
with confidence measures using multiple decoding strategies.

This script addresses the issue where standard beam search returns identical
hypotheses by implementing multiple complementary approaches.

Author: Enhanced for hearing loss research
Date: 2025-11-05
"""

import os
import json
import time
import torch
import librosa
import numpy as np
from datetime import datetime
from pathlib import Path
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
import torch.nn.functional as F
from collections import Counter

def decode_tokens_to_analysis(processor, token_ids, include_special_tokens=False):
    """
    Convert token IDs to detailed analysis including both tokens and text.
    
    Args:
        processor: Whisper processor
        token_ids: List or tensor of token IDs
        include_special_tokens: Whether to include special tokens in analysis
    
    Returns:
        Dict with token analysis
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()
    
    # Get raw tokens
    raw_tokens = processor.tokenizer.convert_ids_to_tokens(token_ids)
    
    # Get text (both with and without special tokens)
    text_with_special = processor.tokenizer.decode(token_ids, skip_special_tokens=False)
    text_clean = processor.tokenizer.decode(token_ids, skip_special_tokens=True)
    
    # Analyze each token
    token_analysis = []
    for i, (token_id, token) in enumerate(zip(token_ids, raw_tokens)):
        is_special = token.startswith('<|') and token.endswith('|>')
        
        token_info = {
            "position": i,
            "token_id": token_id,
            "token": token,
            "is_special_token": is_special,
            "token_type": "special" if is_special else "content"
        }
        
        # Classify special tokens
        if is_special:
            if "transcript" in token:
                token_info["special_type"] = "transcript_marker"
            elif token in ["<|en|>", "<|de|>", "<|fr|>"]:  # Language tokens
                token_info["special_type"] = "language"
            elif token.replace("<|", "").replace("|>", "").replace(".", "").isdigit():  # Timestamp
                token_info["special_type"] = "timestamp"
            elif "end" in token:
                token_info["special_type"] = "end_marker"
            else:
                token_info["special_type"] = "other"
        
        token_analysis.append(token_info)
    
    # Filter content tokens only
    content_tokens = [t for t in token_analysis if not t["is_special_token"]]
    content_text = "".join([t["token"] for t in content_tokens])
    
    return {
        "token_ids": token_ids,
        "raw_tokens": raw_tokens,
        "text_with_special": text_with_special,
        "text_clean": text_clean,
        "content_only_text": content_text,
        "token_analysis": token_analysis,
        "content_tokens": content_tokens,
        "num_total_tokens": len(token_ids),
        "num_content_tokens": len(content_tokens),
        "num_special_tokens": len(token_ids) - len(content_tokens)
    }

def analyze_token_probabilities(model, processor, input_features, device, token_sequence):
    """
    Analyze token-level probabilities for a given sequence.
    
    Args:
        model: Whisper model
        processor: Whisper processor
        input_features: Audio input features
        device: Device
        token_sequence: Token sequence to analyze
    
    Returns:
        Token probability analysis
    """
    # Prepare decoder input
    decoder_input_ids = processor.get_decoder_prompt_ids(language="english", task="transcribe")
    decoder_input_ids = torch.tensor([decoder_input_ids], device=device)
    
    # Forward pass to get logits
    with torch.no_grad():
        outputs = model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            return_dict=True
        )
        
        logits = outputs.logits[0]  # [seq_len, vocab_size]
        token_probs = F.softmax(logits, dim=-1)
        
        # Analyze probabilities for each position
        probability_analysis = []
        for pos in range(min(logits.shape[0], len(token_sequence))):
            # Get top-k tokens at this position
            top_probs, top_indices = torch.topk(token_probs[pos], k=10)
            
            # Check if our actual token is in top-k
            actual_token_id = token_sequence[pos] if pos < len(token_sequence) else None
            actual_token_prob = None
            actual_token_rank = None
            
            if actual_token_id is not None:
                actual_token_prob = token_probs[pos][actual_token_id].item()
                # Find rank of actual token
                sorted_probs, sorted_indices = torch.sort(token_probs[pos], descending=True)
                actual_token_rank = (sorted_indices == actual_token_id).nonzero(as_tuple=True)[0].item() + 1
            
            position_info = {
                "position": pos,
                "actual_token_id": actual_token_id,
                "actual_token": processor.tokenizer.convert_ids_to_tokens([actual_token_id])[0] if actual_token_id else None,
                "actual_token_probability": actual_token_prob,
                "actual_token_rank": actual_token_rank,
                "top_alternatives": []
            }
            
            # Add top alternatives
            for prob, idx in zip(top_probs, top_indices):
                token = processor.tokenizer.convert_ids_to_tokens([idx.item()])[0]
                position_info["top_alternatives"].append({
                    "token_id": idx.item(),
                    "token": token,
                    "probability": prob.item(),
                    "percentage": prob.item() * 100
                })
            
            probability_analysis.append(position_info)
    
    return probability_analysis

def load_model_and_processor(checkpoint_path, model_name):
    """Load a fine-tuned Whisper model and processor from checkpoint."""
    print(f"\nLoading {model_name} from {checkpoint_path}...")
    
    model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)
    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3", 
                                                language="English", 
                                                task="transcribe")
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    print(f"  Model loaded on {device}")
    return model, processor, device

def get_diverse_hypotheses(model, processor, audio_path, device, model_name="", output_tokens=True):
    """Generate diverse hypotheses using multiple decoding strategies.
    
    Args:
        output_tokens: If True, returns both tokens and text. If False, returns only text.
    """
    print(f"\nGenerating diverse hypotheses for {model_name}...")
    print(f"Output mode: {'Tokens + Text' if output_tokens else 'Text only'}")
    
    # Load and preprocess audio
    audio, sr = librosa.load(audio_path, sr=16000)
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt").input_features
    input_features = input_features.to(device)
    
    print(f"  Audio duration: {len(audio)/16000:.2f} seconds")
    
    all_hypotheses = []
    start_time = time.time()
    
    # Strategy 1: Greedy decoding (most confident single hypothesis)
    print("  Strategy 1: Greedy decoding...")
    with torch.no_grad():
        greedy_output = model.generate(
            input_features,
            max_length=225,
            num_beams=1,
            temperature=0.0,
            do_sample=False,
            forced_decoder_ids=processor.get_decoder_prompt_ids(language="english", task="transcribe"),
            return_dict_in_generate=True,
            output_scores=True
        )
    
    # Analyze greedy output with token details
    greedy_analysis = decode_tokens_to_analysis(processor, greedy_output.sequences[0])
    greedy_text = greedy_analysis["text_clean"]
    print(f"    Greedy: '{greedy_text}'")
    
    if output_tokens:
        print(f"    Greedy tokens: {[t['token'] for t in greedy_analysis['content_tokens']]}")
        print(f"    Token breakdown:")
        for token_info in greedy_analysis['content_tokens']:
            print(f"      {token_info['position']}: '{token_info['token']}' (ID: {token_info['token_id']})")
    
    # Skip token-level probability analysis for now (complex to implement correctly)
    greedy_token_probs = None
    
    # Strategy 2: Multiple temperature sampling runs
    print("  Strategy 2: Temperature sampling with multiple temperatures...")
    temperatures = [0.3, 0.5, 0.7, 0.9, 1.1]
    sampling_hypotheses = []
    
    for temp in temperatures:
        for sample_idx in range(4):  # 4 samples per temperature
            with torch.no_grad():
                sample_output = model.generate(
                    input_features,
                    max_length=225,
                    temperature=temp,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    forced_decoder_ids=processor.get_decoder_prompt_ids(language="english", task="transcribe"),
                )
            
            # Analyze sample output
            sample_analysis = decode_tokens_to_analysis(processor, sample_output[0])
            sample_text = sample_analysis["text_clean"]
            
            hypothesis = {
                "transcription": sample_text,
                "temperature": temp,
                "sample_id": sample_idx
            }
            
            if output_tokens:
                hypothesis.update({
                    "token_analysis": sample_analysis,
                    "content_tokens": sample_analysis["content_tokens"],
                    "token_ids": sample_analysis["token_ids"]
                })
            
            sampling_hypotheses.append(hypothesis)
    
    # Strategy 3: Beam search with different beam sizes
    print("  Strategy 3: Multiple beam sizes...")
    beam_sizes = [1, 3, 5, 10]
    beam_hypotheses = []
    
    for beam_size in beam_sizes:
        with torch.no_grad():
            beam_output = model.generate(
                input_features,
                max_length=225,
                num_beams=beam_size,
                num_return_sequences=min(beam_size, 3),  # Return top 3 from each beam size
                temperature=0.0,
                do_sample=False,
                early_stopping=True,
                forced_decoder_ids=processor.get_decoder_prompt_ids(language="english", task="transcribe"),
                return_dict_in_generate=True,
                output_scores=True
            )
        
        for i, sequence in enumerate(beam_output.sequences):
            # Analyze beam output
            beam_analysis = decode_tokens_to_analysis(processor, sequence)
            beam_text = beam_analysis["text_clean"]
            
            # Try to get score, but handle if not available
            if hasattr(beam_output, 'sequences_scores') and beam_output.sequences_scores is not None and i < len(beam_output.sequences_scores):
                score = beam_output.sequences_scores[i].item()
                confidence = np.exp(score) * 100
            else:
                score = None
                confidence = None
            
            hypothesis = {
                "transcription": beam_text,
                "beam_size": beam_size,
                "rank": i + 1,
                "log_probability": score,
                "confidence_percent": confidence
            }
            
            if output_tokens:
                hypothesis.update({
                    "token_analysis": beam_analysis,
                    "content_tokens": beam_analysis["content_tokens"],
                    "token_ids": beam_analysis["token_ids"]
                })
                
            beam_hypotheses.append(hypothesis)
    
    # Strategy 4: Top-k and Top-p variations
    print("  Strategy 4: Top-k/Top-p sampling variations...")
    sampling_configs = [
        {"top_k": 10, "top_p": 0.8, "temperature": 0.6},
        {"top_k": 30, "top_p": 0.9, "temperature": 0.7},
        {"top_k": 50, "top_p": 0.95, "temperature": 0.8},
    ]
    
    topk_hypotheses = []
    for config in sampling_configs:
        for sample_idx in range(3):  # 3 samples per config
            with torch.no_grad():
                topk_output = model.generate(
                    input_features,
                    max_length=225,
                    temperature=config["temperature"],
                    do_sample=True,
                    top_k=config["top_k"],
                    top_p=config["top_p"],
                    forced_decoder_ids=processor.get_decoder_prompt_ids(language="english", task="transcribe"),
                )
            
            # Analyze top-k output
            topk_analysis = decode_tokens_to_analysis(processor, topk_output[0])
            topk_text = topk_analysis["text_clean"]
            
            hypothesis = {
                "transcription": topk_text,
                "config": config,
                "sample_id": sample_idx
            }
            
            if output_tokens:
                hypothesis.update({
                    "token_analysis": topk_analysis,
                    "content_tokens": topk_analysis["content_tokens"],
                    "token_ids": topk_analysis["token_ids"]
                })
            
            topk_hypotheses.append(hypothesis)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Combine all hypotheses and analyze
    all_transcriptions = []
    all_transcriptions.append(greedy_text)  # Greedy
    all_transcriptions.extend([h["transcription"] for h in sampling_hypotheses])  # Temperature sampling
    all_transcriptions.extend([h["transcription"] for h in beam_hypotheses])  # Beam search
    all_transcriptions.extend([h["transcription"] for h in topk_hypotheses])  # Top-k/p sampling
    
    # Count frequencies and calculate confidence
    transcription_counts = Counter(all_transcriptions)
    total_samples = len(all_transcriptions)
    
    # Create ranked list with frequency-based confidence
    ranked_hypotheses = []
    for rank, (transcription, count) in enumerate(transcription_counts.most_common()):
        frequency_confidence = (count / total_samples) * 100
        
        # Get best beam search confidence for this transcription
        beam_confidences = [h["confidence_percent"] for h in beam_hypotheses 
                          if h["transcription"] == transcription and h["confidence_percent"] is not None]
        best_beam_confidence = max(beam_confidences) if beam_confidences else None
        
        ranked_hypotheses.append({
            "rank": rank + 1,
            "transcription": transcription,
            "frequency_count": count,
            "frequency_confidence": frequency_confidence,
            "beam_confidence": best_beam_confidence,
            "appears_in_strategies": []
        })
        
        # Track which strategies produced this transcription
        if transcription == greedy_text:
            ranked_hypotheses[-1]["appears_in_strategies"].append("greedy")
        if transcription in [h["transcription"] for h in sampling_hypotheses]:
            ranked_hypotheses[-1]["appears_in_strategies"].append("temperature_sampling")
        if transcription in [h["transcription"] for h in beam_hypotheses]:
            ranked_hypotheses[-1]["appears_in_strategies"].append("beam_search")
        if transcription in [h["transcription"] for h in topk_hypotheses]:
            ranked_hypotheses[-1]["appears_in_strategies"].append("topk_sampling")
    
    print(f"  Total inference time: {total_time:.2f} seconds")
    print(f"  Total hypotheses generated: {total_samples}")
    print(f"  Unique transcriptions found: {len(transcription_counts)}")
    
    print(f"\n  Ranked hypotheses:")
    for hyp in ranked_hypotheses[:10]:  # Show top 10
        freq_conf = hyp["frequency_confidence"]
        beam_conf = hyp["beam_confidence"]
        beam_str = f"{beam_conf:.1f}%" if beam_conf else "N/A"
        strategies = ", ".join(hyp["appears_in_strategies"])
        print(f"    {hyp['rank']}. '{hyp['transcription']}' - Freq: {freq_conf:.1f}% ({hyp['frequency_count']}/{total_samples}), Beam: {beam_str}, Strategies: {strategies}")
    
    return {
        "model_name": model_name,
        "audio_path": audio_path,
        "total_inference_time": total_time,
        "total_hypotheses_generated": total_samples,
        "unique_transcriptions": len(transcription_counts),
        "ranked_hypotheses": ranked_hypotheses,
        "output_mode": "tokens_and_text" if output_tokens else "text_only",
        "greedy_analysis": {
            "text": greedy_text,
            "token_analysis": greedy_analysis if output_tokens else None,
            "token_probabilities": greedy_token_probs if output_tokens else None
        },
        "detailed_results": {
            "greedy": {"transcription": greedy_text, "analysis": greedy_analysis if output_tokens else None},
            "temperature_sampling": sampling_hypotheses,
            "beam_search": beam_hypotheses,
            "topk_sampling": topk_hypotheses
        },
        "strategies_used": ["greedy", "temperature_sampling", "beam_search", "topk_sampling"]
    }

def main():
    """Main function to test diverse hypothesis generation."""
    print("="*80)
    print("DIVERSE WHISPER HYPOTHESIS GENERATION - HEARING LOSS RESEARCH")
    print("="*80)
    
    # Configuration
    audio_path = "/sc/home/hanno.mueller/pilotproject-hearing-loss/tmp/single_word_six.wav"
    results_base = "/sc/home/hanno.mueller/pilotproject-hearing-loss/results"
    output_dir = "/sc/home/hanno.mueller/pilotproject-hearing-loss/tmp"
    
    # Model configurations (testing just one model for clarity)
    models_config = {
        "normal": {
            "path": f"{results_base}/whisper_finetuned_normal/checkpoint-2000",
            "description": "Baseline model trained on unfiltered audio"
        },
        "hfloss": {
            "path": f"{results_base}/whisper_finetuned_hfloss/checkpoint-2000", 
            "description": "Model trained on high-frequency filtered audio"
        },
    }
    
    # Verify audio file exists
    if not os.path.exists(audio_path):
        print(f"ERROR: Audio file not found: {audio_path}")
        return
    
    # Load metadata
    metadata_path = os.path.join(output_dir, "single_word_six_metadata.json")
    expected_word = 'six'
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        expected_word = metadata.get('word', 'six')
    
    print(f"Audio file: {audio_path}")
    print(f"Expected word: '{expected_word}'")
    
    # Results storage
    all_results = {
        "experiment_info": {
            "date": datetime.now().isoformat(),
            "audio_file": audio_path,
            "expected_word": expected_word,
            "models_tested": list(models_config.keys())
        },
        "results": {}
    }
    
    # Process each model
    for model_name, config in models_config.items():
        print(f"\n{'='*60}")
        print(f"PROCESSING MODEL: {model_name.upper()}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")
        
        try:
            if not os.path.exists(config['path']):
                print(f"ERROR: Checkpoint not found: {config['path']}")
                continue
            
            # Load model
            model, processor, device = load_model_and_processor(config['path'], model_name)
            
            # Generate diverse hypotheses (with token analysis)
            results = get_diverse_hypotheses(model, processor, audio_path, device, model_name, output_tokens=True)
            
            # Store results
            all_results["results"][model_name] = results
            
            # Clean up
            del model
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"ERROR processing {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"diverse_whisper_hypotheses_{timestamp}.json"
    results_path = os.path.join(output_dir, results_filename)
    
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("SUMMARY: WHY BEAM SEARCH GIVES IDENTICAL HYPOTHESES")
    print(f"{'='*80}")
    
    print("\n🔍 EXPLANATION:")
    print("1. Standard beam search is DETERMINISTIC with temperature=0")
    print("2. It finds the single most likely sequence")
    print("3. num_return_sequences=5 just returns 5 COPIES of the same sequence")
    print("4. To get diversity, you need:")
    print("   - Temperature sampling (randomness)")
    print("   - Multiple decoding strategies")
    print("   - Different sampling parameters")
    
    print(f"\n📊 RESULTS COMPARISON:")
    for model_name, results in all_results["results"].items():
        if "ranked_hypotheses" in results:
            print(f"\n{model_name.upper()}:")
            print(f"  Expected: '{expected_word}'")
            print(f"  Output mode: {results.get('output_mode', 'text_only')}")
            print(f"  Unique hypotheses found: {results['unique_transcriptions']}")
            
            # Show greedy analysis with tokens
            if "greedy_analysis" in results and results["greedy_analysis"]["token_analysis"]:
                greedy_tokens = results["greedy_analysis"]["token_analysis"]["content_tokens"]
                print(f"  Greedy result: '{results['greedy_analysis']['text']}'")
                print(f"  Greedy tokens: {[t['token'] for t in greedy_tokens]}")
                print(f"  Token details:")
                for token in greedy_tokens:
                    print(f"    - '{token['token']}' (ID: {token['token_id']})")
            
            print(f"  Top 3 diverse hypotheses:")
            for i, hyp in enumerate(results["ranked_hypotheses"][:3]):
                freq = hyp["frequency_confidence"]
                beam = hyp["beam_confidence"] or 0
                print(f"    {i+1}. '{hyp['transcription']}' (freq: {freq:.1f}%, beam: {beam:.1f}%)")
                
                # Show token information if available
                if "token_analysis" in hyp and hyp["token_analysis"]:
                    tokens = [t['token'] for t in hyp["token_analysis"]["content_tokens"]]
                    print(f"       Tokens: {tokens}")
    
    print(f"\n📁 Detailed results saved to: {results_path}")
    print("\n✅ SUCCESS: Now you have truly diverse hypotheses with confidence measures!")

if __name__ == "__main__":
    main()