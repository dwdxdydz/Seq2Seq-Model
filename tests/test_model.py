import torch

from importlib.machinery import SourceFileLoader

lib = SourceFileLoader("seq2seq_model", "Seq2Seq Model.py").load_module()


def test_forward_shape():
    device = torch.device("cpu")
    encoder = lib.Encoder(20, 8, 16)
    decoder = lib.Decoder(25, 8, 16)
    model = lib.Seq2Seq(encoder, decoder, device)
    src = torch.randint(0, 20, (2, 5))
    trg = torch.randint(0, 25, (2, 6))
    output = model(src, trg, teacher_forcing_ratio=0)
    assert output.shape == (2, 6, 25)
