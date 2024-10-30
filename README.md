# GEC System Evaluation

## Overview
This system implements and evaluates multiple approaches to Grammar Error Correction (GEC) across English and German languages using various language models and tools.

## Datasets
- **BEA-2019**: English dataset with parallel original and corrected text
  - Format: JSONL with source/target pairs
  - Location: `data/bea2019/bea2019_dev_raw.jsonl`

- **Falko-Merlin**: German dataset with parallel original and corrected text
  - Format: Parallel files (.src/.trg) converted to JSONL
  - Location: `data/falko_merlin/fm-test.src` and `fm-test.trg`

## Models
1. **Language Models** (via Groq API):
   - Gemma 2 9B
   - LLaMA 3 90B
   - LLaMA 3 11B
   - Mixtral 8x7B

2. **Rule-based**:
   - LanguageTool (multilingual support for English and German)

## Evaluation Metrics
1. **ERRANT** (English only): Calculates precision, recall, and F0.5 scores for grammatical error detection and correction
2. **GLEU**: Calculates F-score on the token-level
3. **BERTScore**: Calculates F-score based on BERT embeddings

## Setup Instructions

### 1. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preprocessing
This step is optional, as the data is already preprocessed.

```bash
# Process BEA-2019 dataset
python data/bea2019/preprocess_bea2019.py

# Process Falko-Merlin dataset
python data/falko_merlin/preprocess_fm.py
```

### 3. API Configuration (Free)
The Groq API which I'm using is free to use.

- Set up a Groq API key in your environment variables:
```bash
export GROQ_API_KEY=your_api_key_here
```
- Or create a `.env` file with:
```
GROQ_API_KEY=your_api_key_here
```

### 4. Running Evaluation
```bash
# Run full evaluation pipeline
python main.py
```

## Project Structure
```
├── data/                  # Dataset processing and loading
├── evaluators/            # Evaluation pipeline
├── metrics/               # Implementation of evaluation metrics
├── models/               # Model implementations
├── results/              # Evaluation outputs
    ├── raw/              # Raw model outputs
    └── analyzed/         # Processed evaluation results
```

## Output Format
1. Raw Results (`results/raw/`): JSONL files containing model outputs for each input
2. Analyzed Results (`results/analyzed/`): JSON files containing metric scores for each model