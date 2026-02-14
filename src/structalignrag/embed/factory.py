from __future__ import annotations

from ..utils.logging_utils import get_logger
from .contriever import ContrieverEmbedder
from .nv_embed_v2 import NVEmbedV2Embedder

logger = get_logger(__name__)


def build_embedder(
    model_name: str,
    batch_size: int,
    max_length: int,
    normalize: bool,
    dtype: str,
):
    name = (model_name or "").lower()
    if "nv-embed-v2" in name or "nv_embed_v2" in name:
        if int(max_length) < 2048:
            logger.warning(
                f"[StructAlignRAG] NV-Embed-v2 is typically run with a larger max_length (e.g., 2048). "
                f"Current embedding_max_seq_len={max_length}."
            )
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
