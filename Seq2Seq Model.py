"""Small, educational GRU encoder-decoder sequence-to-sequence model."""

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        _, hidden = self.rnn(embedded)
        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim: int, embedding_dim: int, hidden_dim: int, dropout: float = 0.2):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token, hidden):
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, device: torch.device):
        super().__init__()
        if encoder.rnn.hidden_size != decoder.rnn.hidden_size:
            raise ValueError("Encoder and decoder hidden dimensions must match")
        self.encoder, self.decoder, self.device = encoder, decoder, device

    def forward(self, src, trg, teacher_forcing_ratio: float = 0.5):
        if not 0 <= teacher_forcing_ratio <= 1:
            raise ValueError("teacher_forcing_ratio must be between 0 and 1")
        batch_size, trg_len = trg.shape
        output_dim = self.decoder.output_dim
        outputs = torch.zeros(batch_size, trg_len, output_dim, device=self.device)
        hidden = self.encoder(src)
        input_token = trg[:, 0]
        for t in range(1, trg_len):
            prediction, hidden = self.decoder(input_token, hidden)
            outputs[:, t] = prediction
            top1 = prediction.argmax(1)
            input_token = trg[:, t] if torch.rand(1).item() < teacher_forcing_ratio else top1
        return outputs
