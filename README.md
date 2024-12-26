# Neural-Machine-Translation-with-BiLSTM-Building-Language-Access-Between-English-and-Haitian-Creole

This project implements a neural machine translation system using a Bidirectional LSTM architecture for translating between English and Haitian Creole. The repository contains trained LSTM models for bidirectional translation between these languages.

## Project Overview

### Core Components

- **Tokenizer**: BART-base tokenizer from Hugging Face's transformers library
- **Model Architecture**: Bidirectional Long Short-Term Memory (LSTM) neural network
- **Primary Task**: Bidirectional translation between English and Haitian Creole
- **Language Pairs**: English ↔ Haitian Creole

### Model Files

Present in Hugging Face: https://huggingface.co/sprab4/Simple_LSTM_Translator

- `model_en_to_ht.pth`: English to Haitian Creole translation model
- `model_ht_to_en.pth`: Haitian Creole to English translation model

## Technical Details

### Model Architecture

The translation system utilizes a Bidirectional LSTM architecture with the following components:
- **Embedding Layer**: Converts input tokens into dense vector representations
- **Bidirectional LSTM**: Processes sequences in both forward and backward directions
- **Fully Connected Layer**: Maps LSTM outputs to target vocabulary

### Hyperparameters

```python
{
    "embedding_dimension": 128,
    "hidden_size": 256,
    "batch_size": 128,
    "block_size": 16,
    "dropout": 0.2,
    "learning_rate": 1e-3,
    "epochs": 10
}
```

## Performance Metrics

### BLEU Scores
- **Haitian → English**: 
  - Initial: 0.55
  - Stabilized: ~0.15
- **English → Haitian**: 
  - Consistent performance near 0

### ChrF Scores
- **English → Haitian**: 
  - Stabilized: 2.5-3.0
- **Haitian → English**: 
  - Final: ~2.5 (gradual decrease during training)

## Dataset Information

- **Training Set Size**: 16,000 sentence pairs
- **Validation Set Size**: 4,000 sentence pairs
- **Data Format**: JSON with parallel text pairs
- **Tokenization**: BERT tokenizer from Hugging Face

## Training Details

The model demonstrates:
- Stable training behavior
- Superior performance in Haitian to English direction
- Consistent validation metrics
- Moderate overall translation quality
