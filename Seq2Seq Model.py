"""Greedy decoding helper for a trained sequence-to-sequence model."""

from typing import Any, List

import torch


def _tokenize(sentence: str) -> List[str]:
    """Basic tokenizer used when the training pipeline provides no custom tokenizer."""
    return sentence.strip().lower().split()


def _lookup_token(vocab: Any, token: str) -> int:
    """Support common vocab APIs used by PyTorch/torchtext-style vocabularies."""
    if hasattr(vocab, "get_stoi"):
        mapping = vocab.get_stoi()
        return mapping.get(token, mapping.get("<unk>", 0))
    if hasattr(vocab, "__getitem__"):
        try:
            return int(vocab[token])
        except (KeyError, TypeError):
            try:
                return int(vocab["<unk>"])
            except (KeyError, TypeError):
                return 0
    raise TypeError("Vocabulary must support token lookup")


def _lookup_index(vocab: Any, token: str) -> int:
    return _lookup_token(vocab, token)


def _itos(vocab: Any, index: int) -> str:
    if hasattr(vocab, "get_itos"):
        return vocab.get_itos()[index]
    if hasattr(vocab, "itos"):
        return vocab.itos[index]
    raise TypeError("Vocabulary must expose get_itos() or itos")


def translate_sentence(
    sentence: str,
    model: Any,
    src_vocab: Any,
    trg_vocab: Any,
    device: torch.device,
    max_length: int = 50,
) -> str:
    """Translate one sentence using greedy decoding.

    The model is expected to accept ``(src_tensor, trg_tensor,
    teacher_forcing_ratio=0)`` and return logits as its first value.
    """
    tokens = _tokenize(sentence)
    if not tokens:
        return ""

    src_ids = [_lookup_token(src_vocab, token) for token in tokens]
    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device).unsqueeze(0)

    bos_id = _lookup_index(trg_vocab, "<bos>")
    eos_id = _lookup_index(trg_vocab, "<eos>")
    trg_tensor = torch.tensor([[bos_id]], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            output, _, _ = model(
                src_tensor,
                trg_tensor,
                teacher_forcing_ratio=0,
            )

            # Most seq2seq models return [batch, target_len, vocab_size].
            logits = output[:, -1, :] if output.ndim == 3 else output
            next_id = int(logits.argmax(dim=-1).item())
            next_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
            trg_tensor = torch.cat((trg_tensor, next_token), dim=1)

            if next_id == eos_id:
                break

    translated = [
        _itos(trg_vocab, idx)
        for idx in trg_tensor.squeeze(0).tolist()[1:]
    ]
    translated = [token for token in translated if token not in {"<pad>", "<eos>"}]
    return " ".join(translated)


if __name__ == "__main__":
    print("Seq2Seq decoding helper loaded. Provide a trained model and vocabularies to translate_sentence().")
