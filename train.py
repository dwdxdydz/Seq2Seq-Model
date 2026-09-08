"""Train and evaluate the GRU seq2seq model on a tiny reproducible demo corpus."""

import random

import torch
from torch import nn

import seq2seq_model as model_lib

SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAIRS = [
    ("i am happy", "je suis heureux"),
    ("i am sad", "je suis triste"),
    ("i am tired", "je suis fatigue"),
    ("you are happy", "tu es heureux"),
    ("you are sad", "tu es triste"),
    ("he is happy", "il est heureux"),
    ("she is tired", "elle est fatigue"),
    ("we are happy", "nous sommes heureux"),
    ("they are sad", "ils sont tristes"),
]


def build_vocab(sentences):
    vocab = {token: i for i, token in enumerate(SPECIAL)}
    for sentence in sentences:
        for token in sentence.split():
            vocab.setdefault(token, len(vocab))
    return vocab


def encode(sentence, vocab, target=False):
    ids = [vocab.get(token, vocab["<unk>"]) for token in sentence.split()]
    return ([vocab["<bos>"]] if target else []) + ids + [vocab["<eos>"]]


def pad_batch(sequences, pad_id):
    width = max(map(len, sequences))
    return torch.tensor([seq + [pad_id] * (width - len(seq)) for seq in sequences], dtype=torch.long)


def main(epochs=250, seed=7):
    random.seed(seed)
    torch.manual_seed(seed)
    src_vocab = build_vocab(src for src, _ in PAIRS)
    trg_vocab = build_vocab(trg for _, trg in PAIRS)
    src = pad_batch([encode(s, src_vocab) for s, _ in PAIRS], src_vocab["<pad>"])
    trg = pad_batch([encode(t, trg_vocab, target=True) for _, t in PAIRS], trg_vocab["<pad>"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = model_lib.Encoder(len(src_vocab), 32, 64, padding_idx=src_vocab["<pad>"])
    decoder = model_lib.Decoder(len(trg_vocab), 32, 64, padding_idx=trg_vocab["<pad>"])
    model = model_lib.Seq2Seq(encoder, decoder, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=trg_vocab["<pad>"])
    src_lengths = (src != src_vocab["<pad>"]).sum(dim=1).to(device)
    src, trg = src.to(device), trg.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=0.7, src_lengths=src_lengths)
        loss = criterion(output[:, 1:].reshape(-1, len(trg_vocab)), trg[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if epoch == 1 or epoch % 50 == 0:
            print(f"epoch={epoch:03d} loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        output = model(src, trg, teacher_forcing_ratio=0, src_lengths=src_lengths)
        predictions = output[:, 1:].argmax(-1)
        targets = trg[:, 1:]
        mask = targets != trg_vocab["<pad>"]
        accuracy = (predictions[mask] == targets[mask]).float().mean().item()
    print(f"token_accuracy={accuracy:.3f}")
    return model, src_vocab, trg_vocab


if __name__ == "__main__":
    main()
