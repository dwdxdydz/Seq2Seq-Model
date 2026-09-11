# 🤖 Seq2Seq Machine Translation

## What is this project?

This project builds a small **machine-translation system** that learns to convert a sentence from one language into another.

The included example uses **English → French**.

For example, the model learns from examples such as:

```text
English              French
----------------------------
hello                bonjour
thank you             merci
how are you?          comment allez-vous ?
```

The goal of this project is not to compete with modern translation tools. It is to show, from start to finish, how a basic neural-network translation system is built, trained and used to make predictions.

## How does it work?

A sentence is made of words or smaller pieces of text. A neural network cannot directly process those words, so the application first converts them into numbers.

The basic flow is:

```text
Input sentence
      ↓
Tokenisation
      ↓
Convert tokens to numbers
      ↓
GRU Encoder
      ↓
Create an internal representation
      ↓
GRU Decoder
      ↓
Generate output tokens one by one
      ↓
Translated sentence
```

The model learns this behaviour by seeing example input/output sentence pairs during training.

## What happens during training?

Suppose the training data contains:

```text
Input:  hello
Target: bonjour
```

The model makes a prediction. The prediction is compared with the correct target, and the model's internal parameters are adjusted.

This happens repeatedly over many training examples. The aim is for the model to gradually learn patterns that help it produce the correct output.

```text
Example sentence
      ↓
Model prediction
      ↓
Compare with correct answer
      ↓
Calculate error
      ↓
Adjust model
      ↓
Repeat
```

## Main features

- GRU-based encoder-decoder architecture.
- Tokenisation and vocabulary handling.
- Teacher forcing during training.
- Padding-aware sequence handling.
- Gradient clipping for more stable training.
- Greedy decoding from beginning-of-sentence to end-of-sentence.
- Reproducible demo training.
- Token-level evaluation.
- Importable model module.
- Automated tests.

## Run the demo

Install the requirements:

```bash
pip install -r requirements.txt
```

Run training and evaluation:

```bash
python train.py
```

The project intentionally uses a very small English → French dataset. This makes the complete workflow easier to understand and reproduce, but the dataset is **far too small for high-quality real-world translation**.

The original `Seq2Seq Model.py` file is kept as a compatibility wrapper for existing notebooks.

## Project structure

```text
train.py             → Creates the demo data and trains/evaluates the model
seq2seq_model/       → Reusable encoder, decoder and Seq2Seq model code
Seq2Seq Model.py     → Compatibility wrapper for the original filename
requirements.txt     → Python dependencies
tests/               → Automated tests
```

## Main technologies

- **Python** — programming language used for the project
- **PyTorch** — builds and trains the neural network
- **GRU** — neural-network component used to process sequences
- **NLP** — techniques for working with human language
- **Pytest** — automated testing framework
- **GitHub Actions** — automated checks for code changes

## Technical terms explained

**Machine translation** — Automatically converting text from one human language into another.

**Seq2Seq (Sequence-to-Sequence)** — A neural-network architecture that takes one sequence as input and produces another sequence as output. Translation is a common example.

**Neural network** — A machine-learning model made of connected mathematical operations. During training, it adjusts internal parameters so that its predictions become more accurate.

**GRU (Gated Recurrent Unit)** — A type of recurrent neural network designed to process sequences while keeping useful information from earlier parts of the sequence.

**Encoder** — The part of a Seq2Seq model that reads the input sequence and creates information that represents it.

**Decoder** — The part that uses the encoder's information to generate the output sequence step by step.

**Token** — A small piece of text given to the model. Depending on the tokenizer, a token can be a word, part of a word or a special symbol.

**Tokenisation** — Splitting text into tokens so that a model can process it.

**Vocabulary** — The collection of tokens that the model knows about, usually mapped to numerical IDs.

**Embedding** — A numerical representation of a token. The neural network learns useful relationships between these numbers during training.

**Sequence** — An ordered collection of tokens, such as the tokens that make up a sentence.

**Padding** — Adding special placeholder values to shorter sequences so that multiple sequences can be processed together in a batch.

**Teacher forcing** — During training, the decoder is given the correct previous token instead of always using its own previous prediction. This usually makes learning easier and faster.

**Gradient** — Information calculated during training that tells the model how its parameters should change to reduce its error.

**Gradient clipping** — Limiting very large gradient values so that training is less likely to become unstable.

**Greedy decoding** — At each output step, choosing the token with the highest predicted probability instead of exploring many possible sequences.

**BOS (Beginning of Sentence)** — A special token that tells the decoder that it is starting a new output sentence.

**EOS (End of Sentence)** — A special token that tells the decoder that the output sentence is finished.

**Training** — The process of showing examples to a model and adjusting its parameters so it learns patterns from those examples.

**Evaluation** — Measuring how well a trained model performs on examples.

**Token-level evaluation** — Comparing predicted and expected tokens to estimate how often the model produces the correct token.

**PyTorch** — A Python framework used for building, training and running machine-learning models.

**NLP (Natural Language Processing)** — The area of computing focused on working with human language.

**Epoch** — One complete pass through the training dataset.

**Model parameter** — A value inside the neural network that is adjusted during training.

## What does this project demonstrate?

The project shows the basic deep-learning workflow:

**Text → tokens → neural network → training → prediction → evaluation**

It demonstrates practical **Python, PyTorch, deep learning, NLP, sequence modelling, model training, evaluation and software testing** skills.

## Important limitation

This is an educational project. The included dataset is intentionally tiny, so its translation quality should not be compared with production translation systems trained on millions of examples.

## Future improvements

- Use a real, larger translation dataset.
- Add train/validation/test splits.
- Add BLEU and other translation metrics.
- Add an attention mechanism.
- Save and load trained model checkpoints.
- Add beam-search decoding.
- Track training and validation curves.
- Compare the GRU model with Transformer models.
