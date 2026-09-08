"""GRU encoder-decoder components for small sequence-to-sequence experiments."""

from typing import Optional, Tuple

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class Encoder(nn.Module):
    """Embed a source sequence and return the final GRU hidden state."""

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float = 0.2,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=padding_idx)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        embedded = self.dropout(self.embedding(src))
        if lengths is None:
            _, hidden = self.rnn(embedded)
        else:
            packed = pack_padded_sequence(
                embedded, lengths.detach().to("cpu"), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.rnn(packed)
        return hidden


class Decoder(nn.Module):
    """Decode one token from an input token and a GRU hidden state."""

    def __init__(
        self,
        output_dim: int,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float = 0.2,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, embedding_dim, padding_idx=padding_idx)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_token: torch.Tensor, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        embedded = self.dropout(self.embedding(input_token.unsqueeze(1)))
        output, hidden = self.rnn(embedded, hidden)
        return self.fc_out(output.squeeze(1)), hidden


class Seq2Seq(nn.Module):
    """A GRU encoder-decoder with batched teacher forcing and greedy decoding."""

    def __init__(self, encoder: Encoder, decoder: Decoder, device: Optional[torch.device] = None):
        super().__init__()
        if encoder.rnn.hidden_size != decoder.rnn.hidden_size:
            raise ValueError("Encoder and decoder hidden dimensions must match")
        self.encoder = encoder
        self.decoder = decoder
        # Kept for callers of the original API. Device placement is inferred
        # from model parameters and inputs, so model.to(...) is always honored.
        self.device = device

    def _validate_source(self, src: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> None:
        if src.ndim != 2:
            raise ValueError("src must be a two-dimensional [batch, sequence] tensor")
        if src.shape[1] == 0:
            raise ValueError("src must contain at least one token")
        if src.dtype != torch.long:
            raise ValueError("src must contain torch.long token IDs")
        parameter_device = next(self.parameters()).device
        if src.device != parameter_device:
            raise ValueError(
                f"inputs are on {src.device}, but model parameters are on {parameter_device}; move them to the same device"
            )
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] != src.shape[0]:
                raise ValueError("src_lengths must have one length for each source sequence")
            if lengths.device != src.device:
                raise ValueError("src and src_lengths must be on the same device")
            if lengths.dtype not in (torch.int32, torch.int64):
                raise ValueError("src_lengths must contain integer lengths")
            if torch.any(lengths < 1) or torch.any(lengths > src.shape[1]):
                raise ValueError("src_lengths must be between 1 and the source sequence length")

    def forward(
        self,
        src: torch.Tensor,
        trg: torch.Tensor,
        teacher_forcing_ratio: float = 0.5,
        src_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return logits for each target position.

        ``trg[:, 0]`` is the beginning-of-sequence token. The first output is
        zero because it has no previous token; callers should exclude it from
        the training loss. Supplying ``src_lengths`` prevents source padding
        from affecting encoder hidden states.
        """
        if not 0 <= teacher_forcing_ratio <= 1:
            raise ValueError("teacher_forcing_ratio must be between 0 and 1")
        self._validate_source(src, src_lengths)
        if trg.ndim != 2:
            raise ValueError("trg must be a two-dimensional [batch, sequence] tensor")
        if trg.dtype != torch.long:
            raise ValueError("trg must contain torch.long token IDs")
        if trg.shape[0] != src.shape[0]:
            raise ValueError("src and trg must have the same batch size")
        if trg.shape[1] == 0:
            raise ValueError("trg must contain at least a beginning-of-sequence token")
        if trg.device != src.device:
            raise ValueError("src and trg must be on the same device")

        batch_size, trg_len = trg.shape
        outputs = self.decoder.fc_out.weight.new_zeros((batch_size, trg_len, self.decoder.output_dim))
        hidden = self.encoder(src, src_lengths)
        input_token = trg[:, 0]

        for timestep in range(1, trg_len):
            prediction, hidden = self.decoder(input_token, hidden)
            outputs[:, timestep] = prediction
            predicted_token = prediction.argmax(dim=1)
            use_teacher_forcing = torch.rand(batch_size, device=src.device) < teacher_forcing_ratio
            input_token = torch.where(use_teacher_forcing, trg[:, timestep], predicted_token)
        return outputs

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        bos_id: int,
        eos_id: int,
        max_length: int,
        src_lengths: Optional[torch.Tensor] = None,
        pad_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Greedily decode token IDs, including the initial BOS token."""
        if max_length < 1:
            raise ValueError("max_length must be at least 1")
        if not 0 <= bos_id < self.decoder.output_dim or not 0 <= eos_id < self.decoder.output_dim:
            raise ValueError("bos_id and eos_id must be valid decoder vocabulary IDs")
        if pad_id is not None and not 0 <= pad_id < self.decoder.output_dim:
            raise ValueError("pad_id must be a valid decoder vocabulary ID")
        self._validate_source(src, src_lengths)

        generated = torch.full((src.shape[0], max_length), bos_id, dtype=torch.long, device=src.device)
        hidden = self.encoder(src, src_lengths)
        input_token = generated[:, 0]
        finished = torch.zeros(src.shape[0], dtype=torch.bool, device=src.device)
        fill_id = eos_id if pad_id is None else pad_id

        for timestep in range(1, max_length):
            prediction, hidden = self.decoder(input_token, hidden)
            input_token = prediction.argmax(dim=1)
            generated[:, timestep] = torch.where(finished, torch.full_like(input_token, fill_id), input_token)
            finished |= input_token.eq(eos_id)
        return generated
