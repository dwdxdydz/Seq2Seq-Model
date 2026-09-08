import pytest

from evaluate import token_accuracy


def test_token_accuracy_ignores_padding():
    predictions = [[1, 2, 3], [4, 8, 0]]
    targets = [[1, 9, 3], [4, 8, 0]]
    assert token_accuracy(predictions, targets, pad_id=0) == pytest.approx(4 / 5)


@pytest.mark.parametrize(
    ("predictions", "targets", "message"),
    [([[1]], [[1], [2]], "same number"), ([[1, 2]], [[1]], "match")],
)
def test_token_accuracy_rejects_mismatched_shapes(predictions, targets, message):
    with pytest.raises(ValueError, match=message):
        token_accuracy(predictions, targets, pad_id=0)
