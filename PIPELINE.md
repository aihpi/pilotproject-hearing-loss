# Pipeline

This document describes the data processing and training pipeline for the
hearing loss research project. The project compares how hearing loss affects
speech recognition using two approaches: Discriminative Lexicon Models (DLM)
and fine-tuned Whisper models.

Three hearing conditions are studied:

| Condition | Profile | Description |
|-----------|---------|-------------|
| **normal** | 10 dB uniform attenuation | Baseline control |
| **lfloss** | 100 dB @ 125 Hz to 10 dB @ 8 kHz | Low-frequency hearing loss |
| **hfloss** | 10 dB @ 125 Hz to 100 dB @ 8 kHz | High-frequency hearing loss |

---

## Prerequisites

### Environment

```bash
uv sync
```

### External data

| Data | Source | Destination |
|------|--------|-------------|
| CommonVoice English 16.1 | [HuggingFace](https://huggingface.co/datasets/mozilla-foundation/common_voice_16_1) (gated, requires agreement) | `data/CommonVoiceEN/` |
| fastText English embeddings | Facebook Research `cc.en.300.bin` | `data/fasttext/cc.en.300.bin` |

Set `HF_TOKEN` in your environment or in `.env.local` (see `.env.local.template`).

---

## Phase 1: Data Preparation (shared)

Steps 1-4 are shared by both the DLM and Whisper tracks. They produce
utterance-level and word-level audio from CommonVoice.

### Step 1. Download CommonVoice

```bash
uv run python scripts/download_commonvoice.py
```

- **Input**: HuggingFace `mozilla-foundation/common_voice_16_1` (English)
- **Output**: `data/CommonVoiceEN/` (HuggingFace cache)

### Step 2. Extract raw audio and transcriptions

```bash
uv run python scripts/extract_raw_data.py
```

- **Input**: `data/CommonVoiceEN/`
- **Output**: `data/CommonVoiceENraw/{train,test,validation}/` with `{cv_id}.wav` + `{cv_id}.txt` per utterance
- Audio is resampled to 16 kHz mono WAV.

### Step 3. Forced alignment

```bash
uv run python scripts/forced_alignment.py \
    --input-dir data/CommonVoiceENraw \
    --output-dir data/CommonVoiceENJSON
```

- **Input**: WAV + TXT pairs from step 2
- **Output**: `data/CommonVoiceENJSON/{split}/{cv_id}.json` with word-level time boundaries
- Uses torchaudio MMS_FA model (not Montreal Forced Aligner). Supports CPU and GPU.

### Step 4. Extract word-level audio

```bash
uv run python scripts/extract_word_audio.py \
    --input-json data/CommonVoiceENJSON \
    --input-audio data/CommonVoiceENraw \
    --output-audio data/CommonVoiceENWords
```

- **Input**: JSON alignments + source audio from steps 2-3
- **Output**: `data/CommonVoiceENWords/{train,test,validation}/{word}_{instance}.wav`
- One WAV file per word occurrence.

After step 4, the two tracks diverge. Each track has its own masking step
because they operate on different audio granularities:
- **DLM track**: masks word-level audio (step 5a)
- **Whisper track**: masks full-utterance audio as HuggingFace datasets (step 5b)

---

## Phase 2a: DLM Track

Trains Discriminative Lexicon Model matrices that map between acoustic form
and semantic representations. Operates on **word-level** audio.

### Step 5a. Apply hearing loss masks to word audio

```bash
uv run python scripts/apply_all_masks.py
```

- **Input**: `data/CommonVoiceENWords/{split}/`
- **Output**: `data/CommonVoiceENWords_masked/{normal,lfloss,hfloss}_raw/{split}/`
- Creates three acoustically processed versions of every word audio file.

### Step 6a. Generate S-matrices

```bash
uv run python scripts/generate_smatrix.py --split train
uv run python scripts/generate_smatrix.py --split test
uv run python scripts/generate_smatrix.py --split validation
```

- **Input**: Word audio directories + `data/fasttext/cc.en.300.bin`
- **Output**: `data/smatrices/{split}_smatrix.nc` + `{split}_words.txt`
- Pre-computes 300-dim fastText embeddings for all unique words.
- **Config**: `configs/fmatrix_training_config.yaml`

### Step 7a. Train F-matrix and G-matrix

```bash
uv run python scripts/train_fmatrix_incremental.py --condition normal
uv run python scripts/train_fmatrix_incremental.py --condition lfloss
uv run python scripts/train_fmatrix_incremental.py --condition hfloss
```

- **Input**: S-matrices + masked word audio
- **Output**: `results/matrices_incremental/checkpoints/{condition}/` and `results/matrices_incremental/metrics/{condition}/`
- F-matrix (C @ F = S): form to semantics (comprehension)
- G-matrix (S @ G = C): semantics to form (production)
- Incremental learning, one sample at a time. ~70 hours per condition on H100.
- Supports `--auto-resume` for checkpoint resumption.
- **Config**: `configs/fmatrix_training_config.yaml`

### Step 8a. Evaluate matrices

```bash
uv run python scripts/evaluate_matrices.py
```

- **Input**: Trained checkpoints + test/validation audio + S-matrices
- **Output**: `results/matrices_incremental/evaluation/` (JSON + CSV)
- Metrics: cosine similarity, Pearson correlation, RMSE.

### Step 9a. Visualize learning curves

```bash
uv run python scripts/visualize_learning_curves.py
```

- **Input**: `results/matrices_incremental/metrics/{condition}/training_metrics.csv`
- **Output**: `results/matrices_incremental/visualizations/*.png`

---

## Phase 2b: Whisper Track

Fine-tunes OpenAI Whisper on hearing-loss-masked audio. Operates on
**full-utterance** audio stored as HuggingFace datasets.

### Step 5b. Apply hearing loss masks to full utterances

```bash
uv run python scripts/mask_audio.py
```

- **Input**: `data/CommonVoiceEN/` (HuggingFace dataset)
- **Output**: `data/CommonVoiceEN_{condition}/dataset/` (HuggingFace DatasetDict per condition)
- Creates three condition-specific datasets with masked audio arrays.

### Step 6b. Generate log-mel spectrograms

```bash
uv run python scripts/DataSet2LogMel.py \
    -i data/CommonVoiceEN_normal/dataset \
    -o data/CommonVoiceEN_normal_logmel
```

Repeat for `lfloss` and `hfloss`, or use `DataSet2LogMelBatch.py` for parallel
SLURM submission.

- **Input**: HuggingFace dataset from step 5b
- **Output**: `data/CommonVoiceEN_{condition}_logmel/` (HuggingFace DatasetDict)
- 128 mel bins (Whisper large-v3 format).

### Step 7b. Fine-tune Whisper

```bash
uv run python scripts/train_whisper.py \
    --input_folder data/CommonVoiceEN_normal_logmel \
    --output_folder results/whisper_finetuned_normal
```

- **Input**: Log-mel spectrograms from step 6b
- **Output**: `results/whisper_finetuned_{condition}/` (model checkpoints + TensorBoard logs)
- ~96 hours on 4x H100 GPUs.

### Step 8b. Transcribe and analyze

```bash
uv run python scripts/transcribe_single_word.py \
    --model results/whisper_finetuned_normal/checkpoint-XXXX \
    --audio-folder data/CommonVoiceENWords_masked/normal_raw/test/
```

- **Input**: Fine-tuned model + word audio files
- **Output**: Transcriptions with per-token confidence, entropy, and cosine similarity metrics.

### Step 9b. Visualize training runs

```bash
uv run python scripts/tensorboard_visualise_runs.py
```

- **Input**: TensorBoard event files from step 7b
- **Output**: PNG/PDF training metric plots.

---

## SLURM

Most scripts have corresponding `.sbatch` files for HPC submission. The DLM
training script supports SLURM array jobs to train all three conditions in
parallel.

---

## Directory structure (after full run)

```
data/
  CommonVoiceEN/                        # Step 1: raw HuggingFace cache
  CommonVoiceENraw/{split}/             # Step 2: extracted WAV + TXT
  CommonVoiceENJSON/{split}/            # Step 3: forced alignment JSON
  CommonVoiceENWords/{split}/           # Step 4: word-level audio
  CommonVoiceENWords_masked/            # Step 5a: masked word audio (DLM)
    {normal,lfloss,hfloss}_raw/{split}/
  CommonVoiceEN_{condition}/dataset/    # Step 5b: masked utterances (Whisper)
  fasttext/cc.en.300.bin                # External: fastText embeddings
  smatrices/                            # Step 6a: semantic matrices
  CommonVoiceEN_{condition}_logmel/     # Step 6b: Whisper spectrograms

results/
  matrices_incremental/                 # DLM track
    checkpoints/{condition}/
    metrics/{condition}/
    evaluation/
    visualizations/
  whisper_finetuned_{condition}/        # Whisper track
```
