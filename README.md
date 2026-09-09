# 🤖 Seq2Seq Machine Translation

An educational **GRU-based sequence-to-sequence machine translation system** with teacher forcing, padding-aware encoding, gradient clipping, reproducible training, greedy decoding, and token-level evaluation.

## Architecture

```text
Source sentence
      ↓
Tokenization / vocabulary
      ↓
Embedding → GRU Encoder → Context / Hidden State
                              ↓
Target BOS → Embedding → GRU Decoder → Vocabulary logits
                    ↑
              Teacher forcing
```

## Features

- GRU encoder-decoder architecture
- Teacher forcing during training
- Padding-aware encoder states
- Gradient clipping
- Greedy BOS-to-EOS inference
- Reproducible demo training
- Token-level evaluation
- Import-friendly model module
- Automated tests

## Run the demo

```bash
pip install -r requirements.txt
python train.py
```

The original `Seq2Seq Model.py` filename remains as a compatibility wrapper for existing notebooks. The demo trains on a tiny English→French corpus so the complete training and evaluation path is easy to reproduce.

> **Note:** The included corpus is intentionally small. It demonstrates the architecture and workflow rather than providing benchmark-quality translation performance.

## Portfolio value

Demonstrates **Python, PyTorch, deep learning, NLP, sequence modelling, model training, evaluation, and software testing**.

## Future improvements

- Replace the demo corpus with a real parallel dataset
- Add train/validation/test splits
- Report BLEU and other translation metrics
- Add attention mechanisms
- Save and load model checkpoints
- Add beam-search decoding
- Track experiments and training curves
- Compare GRU performance with Transformer models
