"""
Rich Synced Lyrics Alignment Engine
Takes line-by-line lyrics and produces rich synced lyrics with deterministic and acoustic models.
"""

__version__ = "1.2.0"

from .model import NeuralLyricsEngine
from .aligner import RichLyricsAligner
from .evaluate import compute_word_score, evaluate_dataset
from .ttml import parse_ttml
from .acoustic_aligner import align_song_fast_acoustic
from .vocal_aligner import align_song_with_demucs

__all__ = [
    "NeuralLyricsEngine",
    "RichLyricsAligner",
    "compute_word_score",
    "evaluate_dataset",
    "parse_ttml",
    "align_song_fast_acoustic",
    "align_song_with_demucs",
]
