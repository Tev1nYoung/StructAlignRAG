from __future__ import annotations

from .contriever import ContrieverEmbedder
from .nv_embed_v2 import NVEmbedV2Embedder


def build_embedder(
    model_name: str,
    batch_size: int,
    max_length: int,
    normalize: bool,
    dtype: str,
):
    name = (model_name or "").lower()
    if "nv-embed-v2" in name or "nv_embed_v2" in name:
        return NVEmbedV2Embedder(
            model_name=model_name,
            batch_size=batch_size,
            max_length=max_length,
            normalize=normalize,
            dtype=dtype,
        )
    return ContrieverEmbedder(
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        normalize=normalize,
        dtype=dtype,
    )

