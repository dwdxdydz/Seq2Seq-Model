# Seq2Seq Machine Translation

Educational GRU encoder-decoder implementation with teacher forcing, gradient clipping, reproducible training, and token-level evaluation.

## Architecture

```text
Source tokens → Embedding → GRU Encoder → Hidden State
                                           ↓
Target BOS → Embedding → GRU Decoder → Vocabulary logits
                              ↑
                    teacher forcing
```

## Run the demo

```bash
pip install -r requirements.txt
python train.py
```

The model is available from the import-friendly `seq2seq_model` module. The
original `Seq2Seq Model.py` filename remains as a compatibility wrapper for
existing notebooks.

`Seq2Seq.forward` accepts optional source lengths so encoder states ignore
right-padding, and `Seq2Seq.generate` provides greedy BOS-to-EOS decoding for
inference.

The demo trains on a tiny English→French corpus so the full training and evaluation path is easy to reproduce. It is intentionally small and is not a benchmark-quality translation system.

## What to extend

- Replace the demo corpus with a real parallel dataset
- Add validation/test splits
- Report BLEU alongside token accuracy
- Add attention
- Save/load checkpoints
- Add beam-search decoding
- Track experiments and training curves
