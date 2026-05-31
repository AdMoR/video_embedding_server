"""Video captioning dispatcher + shared EmbeddingGemma text embedder.

Selects the active captioning backend at startup via CAPTION_BACKEND:
  - "chai"   (default) — CHAI (Qwen3-VL-8B, int8)  → chai_captioner.ChaiCaptioner
  - "marlin"           — Marlin-2B (bfloat16)        → marlin_captioner.MarlinCaptioner

EmbeddingGemma is shared across all backends and loaded once here.
"""

import os

import torch
from sentence_transformers import SentenceTransformer

from base_captioner import VideoCaptioner

CAPTION_BACKEND = os.getenv("CAPTION_BACKEND", "chai")
TEXT_EMBED_MODEL = os.getenv("TEXT_EMBED_MODEL", "google/embeddinggemma-300m")
TEXT_EMBED_DIM = 768

_backend: VideoCaptioner | None = None
_embedder: SentenceTransformer | None = None


def load_models() -> None:
    """Load the selected caption backend and EmbeddingGemma. Call once at startup."""
    global _backend, _embedder

    if CAPTION_BACKEND == "marlin":
        from marlin_captioner import MarlinCaptioner
        _backend = MarlinCaptioner()
    else:
        from chai_captioner import ChaiCaptioner
        _backend = ChaiCaptioner()

    _backend.load_model()

    # EmbeddingGemma does not support float16; use bfloat16 on GPU.
    _embedder = SentenceTransformer(
        TEXT_EMBED_MODEL,
        device="cuda",
        model_kwargs={"dtype": torch.bfloat16},
    )


def caption(path: str) -> dict:
    """Generate a dense caption for a video.

    Returns ``{"caption": str, "scene": str, "events": [{"start", "end", "description"}]}``.
    """
    return _backend.caption(path)


def caption_with_prompt(path: str, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate a caption for a video using a custom user prompt."""
    return _backend.caption_with_prompt(path, prompt, max_new_tokens)


def find(path: str, event: str) -> dict:
    """Locate a natural-language event within a video.

    Returns ``{"raw": str, "span": (start, end) | None, "format_ok": bool}``.
    """
    return _backend.find(path, event)


def embed_query(text: str) -> list[float]:
    """Embed a search query (uses EmbeddingGemma's query prompt)."""
    return _embedder.encode_query(text).tolist()


def embed_document(text: str) -> list[float]:
    """Embed a document/caption (uses EmbeddingGemma's document prompt)."""
    return _embedder.encode_document(text).tolist()


def active_backend() -> str:
    """Return the name of the active captioning backend."""
    return CAPTION_BACKEND
