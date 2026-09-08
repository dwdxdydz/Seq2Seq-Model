import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch is required to run seq2seq model tests")

if TORCH_AVAILABLE:
    import torch
    import seq2seq_model as lib



def test_forward_shape():
    device = torch.device("cpu")
    encoder = lib.Encoder(20, 8, 16)
    decoder = lib.Decoder(25, 8, 16)
    model = lib.Seq2Seq(encoder, decoder, device)
    src = torch.randint(0, 20, (2, 5))
    trg = torch.randint(0, 25, (2, 6))
    output = model(src, trg, teacher_forcing_ratio=0)
    assert output.shape == (2, 6, 25)


def test_source_lengths_exclude_padding_from_encoder_state():
    encoder = lib.Encoder(20, 8, 16, dropout=0, padding_idx=0)
    encoder.eval()
    unpadded = torch.tensor([[4, 5]], dtype=torch.long)
    padded = torch.tensor([[4, 5, 0, 0]], dtype=torch.long)
    assert torch.allclose(encoder(unpadded), encoder(padded, torch.tensor([2])))


def test_generate_returns_bos_prefixed_token_ids():
    model = lib.Seq2Seq(lib.Encoder(20, 8, 16, dropout=0), lib.Decoder(25, 8, 16, dropout=0))
    generated = model.generate(torch.randint(0, 20, (2, 5)), bos_id=2, eos_id=3, max_length=7, pad_id=0)
    assert generated.shape == (2, 7)
    assert torch.equal(generated[:, 0], torch.full((2,), 2))


@pytest.mark.parametrize(
    ("src_shape", "trg_shape", "message"),
    [((2, 5, 1), (2, 6), "two-dimensional"), ((2, 5), (3, 6), "same batch size")],
)
def test_forward_rejects_invalid_batch_shapes(src_shape, trg_shape, message):
    model = lib.Seq2Seq(lib.Encoder(20, 8, 16), lib.Decoder(25, 8, 16))
    src = torch.randint(0, 20, src_shape)
    trg = torch.randint(0, 25, trg_shape)
    with pytest.raises(ValueError, match=message):
        model(src, trg)


def test_forward_rejects_empty_target():
    model = lib.Seq2Seq(lib.Encoder(20, 8, 16), lib.Decoder(25, 8, 16))
    with pytest.raises(ValueError, match="beginning-of-sequence"):
        model(torch.randint(0, 20, (2, 5)), torch.empty((2, 0), dtype=torch.long))
