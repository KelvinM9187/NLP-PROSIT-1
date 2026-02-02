# Twi N-Gram Language Model (Low-Resource Optimization)

This repository contains a specialized pipeline for building and evaluating N-gram language models for **Twi**, a low-resource language. The project emphasizes custom tokenization and advanced statistical smoothing to overcome data sparsity.

## Overview
Language modeling for low-resource languages like Twi requires more than just standard libraries. This project implements a **Twi-aware preprocessing** and **Kneser-Ney Interpolated N-gram** model to achieve reliable performance even with limited training data.

##  Dataset
The model was trained and evaluated on Twi-language corpora.
* **Source:** [https://www.kaggle.com/datasets/azunre/twi-dataset/data?select=twi]
* **Cleaning:** Text was normalized to lowercase and filtered to preserve the Twi-specific alphabet, including the characters `ɔ` and `ɛ`, while maintaining linguistically significant hyphens and apostrophes.

## Technical Pipeline

### 1. Custom Tokenization
We use a **Byte Pair Encoding (BPE)** model with a custom `pre_tokenizer` sequence:
* **Regex-Based Splitting:** A specialized Twi regex (`r"[A-Za-zÀ-ÖØ-öø-ÿƐɛƆɔ0-9]+(?:['-][...])*|[^\w\s]"`) ensures that compound words and punctuation are handled correctly.
* **Vocabulary Management:** Optimized with a `vocab_size` of 3,000 to maintain high token density, which is critical for low-resource efficiency.

### 2. Modeling & Smoothing
The model uses **Kneser-Ney Interpolation** via NLTK. This algorithm is used because:
* It accounts for "word fertility" (how likely a word is to follow many different words).
* It provides a superior "back-off" strategy compared to Witten-Bell or Lidstone smoothing when encountering unseen sequences.

### 3. Evaluation Suite
The module calculates performance using log-probabilities to avoid numerical underflow:
* **Perplexity:** The primary measure of model "surprise."
* **Cross-Entropy:** The average number of bits required to encode the test set.


## 🚀 Usage

### Installation
```bash
pip install nltk tokenizers
