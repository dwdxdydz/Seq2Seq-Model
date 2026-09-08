"""Compatibility wrapper for the original filename.

New code should import :mod:`seq2seq_model`; this file is kept so existing
notebooks using ``SourceFileLoader`` continue to work.
"""

from seq2seq_model import Decoder, Encoder, Seq2Seq

__all__ = ["Decoder", "Encoder", "Seq2Seq"]
