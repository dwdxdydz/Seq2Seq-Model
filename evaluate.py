"""Evaluation utilities for sequence-to-sequence experiments."""


def token_accuracy(predictions, targets, pad_id):
    """Return accuracy over non-padding target tokens."""
    correct = total = 0
    for prediction, target in zip(predictions, targets):
        for p, t in zip(prediction, target):
            if t == pad_id:
                continue
            total += 1
            correct += int(p == t)
    return correct / total if total else 0.0
