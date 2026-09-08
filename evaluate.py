"""Evaluation utilities for sequence-to-sequence experiments."""


def token_accuracy(predictions, targets, pad_id):
    """Return accuracy over non-padding target tokens.

    Prediction and target batches must have matching rectangular shapes. This
    avoids silently dropping tokens through ``zip`` when an evaluation batch is
    malformed.
    """
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must contain the same number of sequences")

    correct = total = 0
    for prediction, target in zip(predictions, targets):
        if len(prediction) != len(target):
            raise ValueError("each prediction sequence must match its target sequence length")
        for predicted_id, target_id in zip(prediction, target):
            if target_id == pad_id:
                continue
            total += 1
            correct += int(predicted_id == target_id)
    return correct / total if total else 0.0
