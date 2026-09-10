# 🤖 Seq2Seq Machine Translation

## What is this project?

This project builds a small **machine-translation system** that learns to convert a sentence from one language into another.

The included example uses English → French.

For example, the model can learn a relationship such as:

```text
English sentence
      ↓
"hello"
      ↓
Neural network
      ↓
French sentence
      ↓
"bonjour"
```

The project is designed to show how a sequence-to-sequence model is built, trained and used for prediction.

## How does it work?

```text
Input sentence
      ↓
Convert words into numbers
      ↓
GRU Encoder
      ↓
Understand the input sequence
      ↓
GRU Decoder
      ↓
Generate the output one token at a time
      ↓
Translated sentence
```

The model learns from example pairs of sentences during training.

## Main features

- GRU-based encoder-decoder model
- Teacher forcing during training
- Padding-aware sequence handling
- Gradient clipping for more stable training
- Greedy decoding from beginning-of-sentence to end-of-sentence
- Reproducible demo training
- Token-level evaluation
- Importable model module
- Automated tests

## Run the demo

```bash
pip install -r requirements.txt
python train.py
```

The project intentionally uses a very small English → French dataset. This makes the complete training process easy to understand and reproduce, but it is **not large enough for high-quality real-world translation**.

The original `Seq2Seq Model.py` file is kept as a compatibility wrapper for existing notebooks.

## What happens during training?

The model sees many examples such as:

```text
Input:  hello
Target: bonjour
```

It makes a prediction, compares that prediction with the correct answer, and adjusts its internal parameters.

This process is repeated many times so that the model gradually learns patterns in the training examples.

## Technical terms explained

**Seq2Seq (Sequence-to-Sequence)** — A model architecture designed to take one sequence as input and produce another sequence as output. Translation is a common example.

**GRU (Gated Recurrent Unit)** — A type of recurrent neural network that can remember useful information from earlier parts of a sequence while processing new information.

**Encoder** — The part of the model that reads the input sentence and creates an internal representation of it.

**Decoder** — The part of the model that uses the encoder's information to generate the output sentence step by step.

**Token** — A small piece of text processed by the model. It can be a word, part of a word or a special symbol.

**Embedding** — A numerical representation of a token that allows the neural network to work with text mathematically.

**Teacher forcing** — During training, the decoder is sometimes given the correct previous word instead of its own previous prediction. This helps training converge more easily.

**Padding** — Adding special empty values to shorter sequences so sequences in the same training batch can have the same length.

**Gradient clipping** — Limiting very large training updates so the model is less likely to become unstable during training.

**Greedy decoding** — At each step, selecting the token with the highest predicted probability rather than considering many possible sentences.

**PyTorch** — A Python machine-learning framework used to build and train the neural network.

**Evaluation** — Measuring how well the trained model performs on examples.

## What does this project demonstrate?

The project demonstrates the full basic deep-learning workflow:

**Text → tokens → neural network → training → prediction → evaluation**

It demonstrates practical **Python, PyTorch, deep learning, NLP, sequence modelling, model training, evaluation and testing** skills.

## Future improvements

- Use a real, larger translation dataset
- Add train/validation/test splits
- Add BLEU and other translation metrics
- Add attention
- Save and load trained models
- Add beam-search decoding
- Track training and validation curves
- Compare the GRU model with Transformer models
