# Whisper Model Analysis Metrics Documentation

This document describes the comprehensive metrics extracted by `scripts/analyse_with_whisper.py` for analyzing Whisper model predictions. The metrics provide insights into model behavior, uncertainty, and semantic representations.

## Example Context: "transcendentalists"

Using the audio file `test/transcendentalists/transcendentalists.wav` (1.04 seconds, ground truth: "transcendentalists"), analyzed with the fine-tuned model `results/whisper_finetuned_normal/checkpoint-2000`.

### Model Prediction

The model predicts three tokens that together form "transcendentalists":
1. **TOKEN 1**: ' transcend' (probability: 0.655841, rank: 1)
2. **TOKEN 2**: 'ental' (probability: 0.821883, rank: 1)
3. **TOKEN 3**: 'ists' (probability: 0.985886, rank: 1)

## Metric Categories

### 1. Per-Token Original Metrics

For each predicted token, the following metrics are computed based on the original (non-normalized) token space:

#### 1.1 Predicted Token Metrics
- **`predicted_probability`**: The probability assigned to the actual predicted token (e.g., 0.655841 for ' transcend')
- **`predicted_rank`**: The rank of the predicted token among all vocabulary tokens (1 = highest probability)
- **`entropy`**: Shannon entropy of the probability distribution over all tokens
  - Lower entropy → more confident prediction
  - Example: TOKEN 1 entropy = 1.3586, TOKEN 3 entropy = 0.0776 (more confident)
- **`semantic_density`**: Standard deviation of the hidden state vector (measures how "spread out" the semantic representation is)
  - Example: TOKEN 1 = 0.117654, TOKEN 3 = 0.192014

#### 1.2 Top-K Alternatives
- **`top_k_tokens`**: Token IDs of the top-k most probable alternatives
- **`top_k_texts`**: Decoded text of the top-k tokens
- **`top_k_probabilities`**: Probabilities of the top-k tokens
- **`top_k_ranks`**: Ranks of the top-k tokens (1, 2, 3, ...)

Example (TOKEN 1, top 5):
```
1. ' transcend' (prob: 0.655841, rank: 1)
2. ' Trans'     (prob: 0.261564, rank: 2)
3. ' and'       (prob: 0.008787, rank: 3)
4. ' "'         (prob: 0.007363, rank: 4)
5. ' trans'     (prob: 0.004253, rank: 5)
```

#### 1.3 Ground Truth Comparison Metrics

For each ground truth token (if in vocabulary), the following metrics are computed:

- **`rank`**: Rank of the ground truth token in the probability distribution
- **`probability`**: Probability assigned to the ground truth token
- **`cosine_similarity`**: Cosine similarity between predicted and ground truth hidden states (range: [-1, 1])
- **`correlation`**: Pearson correlation between predicted and ground truth hidden states (range: [-1, 1])

Example (TOKEN 1 vs ground truth "Trans"):
```
Token ID: 33339 ('Trans')
Rank: 9
Probability: 0.001393
Cosine similarity: 0.081047
Correlation: 0.081358
```

The model assigns low probability to the exact ground truth token "Trans" (rank 9, prob 0.001393), but predicts a semantically related token ' transcend' instead.

### 2. Per-Token Normalized Metrics

Normalization groups together tokens that represent the same underlying text (after removing spaces, lowercasing, and punctuation). This provides a more semantic view of the predictions.

#### 2.1 Normalized Token Text
- **`normalized_text`**: The normalized form of the predicted token
  - Example: ' transcend', ' Trans', ' trans' all normalize to 'transcend'
  
#### 2.2 Normalized Probability Distribution
- **`top_k_normalized_texts`**: Top-k most probable normalized texts
- **`top_k_normalized_probabilities`**: Aggregated probabilities for each normalized text
- **`normalized_entropy`**: Entropy of the normalized probability distribution

Example (TOKEN 1, top 5 normalized):
```
1. 'transcend' (prob: 0.655841)  ← ' transcend'
2. 'trans'     (prob: 0.267322)  ← ' Trans' + ' trans' combined
3. 'and'       (prob: 0.009718)
4. '"'         (prob: 0.007381)
5. 'to'        (prob: 0.002342)
```

Note: Case variants like 'Trans' and 'trans' are merged in the normalized view.

#### 2.3 Normalized Embeddings

Three types of embedding pooling for normalized predictions:
- **`probability_weighted`**: Embeddings weighted by their probabilities (most semantically meaningful)
- **`simple_average`**: Unweighted average of embeddings
- **`most_probable`**: Embedding of the single most probable token with the same normalized text

These embeddings can be used for semantic similarity computations in the normalized space.

### 3. Pooled Metrics (Across All Predicted Tokens)

Aggregated metrics computed across all predicted tokens in the sequence:

#### 3.1 Average Prediction Quality
- **`average_rank`**: Mean rank of predicted tokens
  - Example: (1 + 1 + 1) / 3 = 1.00 (all tokens were rank 1)
- **`average_probability`**: Mean probability of predicted tokens
  - Example: (0.655841 + 0.821883 + 0.985886) / 3 = 0.821203
- **`average_entropy`**: Mean entropy across predicted tokens
  - Example: (1.3586 + 0.6466 + 0.0776) / 3 = 0.6943

#### 3.2 Pooled Hidden State
- **`pooled_hidden_state`**: Mean of all predicted token hidden states (1280-dimensional vector for Whisper-large)
- **`num_tokens_pooled`**: Number of tokens that were pooled

#### 3.3 Pooled Ground Truth Comparison

Comparison between the pooled predicted hidden states and pooled ground truth hidden states:

- **`cosine_similarity`**: 0.114152
  - Relatively low similarity indicates the predicted sequence deviates semantically from the exact ground truth tokenization
- **`correlation`**: 0.114437
- **`semantic_density`**: 0.131506
  - Semantic density of the pooled representation

### 4. Ground Truth Metadata

Information about how the ground truth text is represented in the model's vocabulary:

- **`is_in_vocab`**: Whether the ground truth can be tokenized (True for "transcendentalists")
- **`token_ids`**: Token IDs for ground truth: [33339, 21153, 14533, 1751]
- **`tokens`**: Decoded tokens: ['Trans', 'cend', 'ental', 'ists']
- **`embeddings`**: Hidden state embeddings for each ground truth token (1280-dimensional vectors)

Note: The ground truth tokenizes into 4 tokens ['Trans', 'cend', 'ental', 'ists'], while the model predicts 3 tokens [' transcend', 'ental', 'ists']. This tokenization mismatch explains the low cosine similarity scores.

## Output File Structure

### Main JSON File
```json
{
  "model_path": "results/whisper_finetuned_normal/checkpoint-2000",
  "num_samples": 1,
  "results": [
    {
      "audio_id": "transcendentalists",
      "ground_truth": "transcendentalists",
      "ground_truth_embeddings": { ... },
      "predictions": [ ... ],
      "pooled_predictions": { ... },
      "error": null
    }
  ]
}
```

### Individual JSON Files

For each audio file, an individual JSON file is saved (e.g., `transcendentalists.json`) containing only that sample's results (without the `model_path` wrapper).

## Interpretation Guide

### High Confidence Predictions
- Low entropy (< 1.0)
- High predicted_probability (> 0.8)
- Rank = 1
- Example: TOKEN 3 ('ists') - entropy 0.0776, probability 0.985886

### Uncertain Predictions
- High entropy (> 2.0)
- Multiple alternatives with similar probabilities
- Lower predicted_probability

### Semantic Similarity to Ground Truth
- **Cosine similarity > 0.5**: Strong semantic alignment
- **Cosine similarity 0.1-0.5**: Moderate alignment
- **Cosine similarity < 0.1**: Weak alignment (as seen in this example)

Low similarity can occur when:
1. Different tokenization (4 ground truth tokens vs 3 predicted)
2. Semantically different predictions
3. Model uncertainty about the correct sequence

### Normalized vs Original Metrics

**Original metrics** are useful for:
- Understanding exact token-level predictions
- Analyzing model's raw output distribution
- Debugging tokenization issues

**Normalized metrics** are useful for:
- Semantic analysis (ignoring case/punctuation differences)
- Comparing predictions across different tokenizations
- Evaluating if the model "got the right word" regardless of exact tokens

## Usage Examples

### Analyzing a Single Audio File
```bash
python scripts/analyse_with_whisper.py \
    --input-folder test/transcendentalists \
    --output-path results/analysis_output.json \
    --model-path results/whisper_finetuned_normal/checkpoint-2000 \
    --num-workers 0 \
    --num-threads 8 \
    --top-k 1000
```

### Batch Processing MALD Dataset
```bash
sbatch scripts/analyse_with_whisper.sbatch \
    --input-folder data/MALD \
    --output-path results/whisper_predictions_normal/mald_analysis.json \
    --model-path results/whisper_finetuned_normal/checkpoint-2000
```

## Key Insights from Example

1. **Tokenization Mismatch**: Model uses different tokenization (' transcend' vs 'Trans'+'cend')
2. **High Confidence Overall**: Average probability 0.821203, all ranks = 1
3. **Increasing Confidence**: Entropy decreases from TOKEN 1 (1.3586) to TOKEN 3 (0.0776)
4. **Semantic Deviation**: Low pooled cosine similarity (0.114152) due to tokenization differences
5. **Normalization Benefits**: Case variants ('Trans', 'trans') properly merged in normalized view

## Technical Details

- **Model**: OpenAI Whisper-large-v3 architecture (fine-tuned)
- **Vocabulary Size**: 51,866 tokens
- **Hidden State Dimension**: 1280
- **Default Top-K**: 1000 (captures detailed probability distribution)
- **Normalization**: Case-insensitive, strips spaces/punctuation, removes special tokens
