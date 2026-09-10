# 🤖 Seq2Seq Machine Translation

## What is this project?

This project is a small **machine translation system**.

It takes a sentence in one language, such as English, and tries to produce the same meaning in another language, such as French.

For example:

```text
English:
"I am happy"

        ↓

Translation model

        ↓

French:
"Je suis heureux"
```

The project is built to show how a translation model learns to read one sentence and generate another sentence step by step.

## How does it work?

The model has two main parts:

```text
English sentence
       ↓
   Encoder
       ↓
 Understand the sentence
       ↓
    Decoder
       ↓
Generate the translation word by word
       ↓
French sentence
```

### Encoder

The **encoder** reads the input sentence and creates an internal representation of what it has seen.

### Decoder

The **decoder** uses that information to generate the output sentence one token at a time.

The model uses **GRU neural networks** for both parts.

## What is teacher forcing?

During training, the model sometimes receives the correct previous word instead of its own previous prediction.

For example:

```text
Correct translation:
Je → suis → heureux

While learning:

Generate "Je"
       ↓
Give the model the correct "Je"
       ↓
Generate "suis"
       ↓
Give the model the correct "suis"
       ↓
Generate "heureux"
```

This helps the model learn the translation task more effectively.

## What does the project include?

- A GRU-based encoder.
- A GRU-based decoder.
- Word/token processing.
- Padding handling for sentences of different lengths.
- Teacher forcing during training.
- Gradient clipping to make training more stable.
- A complete training example.
- Sentence generation after training.
- A simple evaluation method.
- Automated tests.

## Run the demo

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the training example:

```bash
python train.py
```

The included example uses a **very small English-to-French dataset**. The goal is to demonstrate how the complete training and translation process works, not to build a professional translation service.

## Important limitation

The model is trained on a tiny demonstration dataset. Therefore, its translation quality is not comparable to tools such as Google Translate or modern large language models.

The purpose of this project is to understand the **architecture and training process** behind sequence-to-sequence models.

## Project structure

```text
train.py             → Runs the training example
seq2seq_model.py     → Encoder, decoder and Seq2Seq model
Seq2Seq Model.py     → Compatibility wrapper
requirements.txt     → Required Python packages
tests/               → Automated tests
.github/workflows/   → Automatic testing
```

## Main technologies

- **Python** — application logic
- **PyTorch** — builds and trains the neural network
- **GRU** — processes sequences of words
- **NLP** — works with human language

## What I learned

This project demonstrates the basic journey of a neural machine-translation system:

**Text → tokens → encoder → decoder → predicted translation → evaluation**

It is useful for demonstrating Python, PyTorch, NLP, deep learning, sequence modelling, model training and testing.

## Future improvements

- Train on a much larger real-world dataset.
- Add proper training, validation and test sets.
- Use BLEU and other translation-quality metrics.
- Add an attention mechanism.
- Save and load trained models.
- Add beam-search decoding.
- Compare the model with Transformer-based models.
- Add training-progress charts.
