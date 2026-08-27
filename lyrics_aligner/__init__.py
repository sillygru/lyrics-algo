"""
Rich Synced Lyrics Alignment Engine
Takes line-by-line lyrics and produces deterministic word-level rich synced lyrics.
"""

__version__ = "1.0.0"

from .model import NeuralLyricsEngine
from .aligner import RichLyricsAligner
from .evaluate import compute_word_score, evaluate_dataset
from .ttml import parse_ttml

__all__ = [
    "NeuralLyricsEngine",
    "RichLyricsAligner",
    "compute_word_score",
    "evaluate_dataset",
    "parse_ttml",
]
