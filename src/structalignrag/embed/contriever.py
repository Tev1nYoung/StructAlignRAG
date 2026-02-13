from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


def _mean_pooling(token_embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    sent = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sent


class ContrieverEmbedder:
    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        max_length: int = 2048,
        normalize: bool = True,
        dtype: str = "auto",
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize
        self.dtype = dtype
        # HuggingFace tokenizer/model are not guaranteed to be thread-safe for concurrent forward passes.
        # We allow higher-level pipeline parallelism (LLM I/O, multi-query) while serializing GPU encoder calls.
        self._encode_lock = threading.Lock()

        logger.info(f"[StructAlignRAG] loading embedding model | name={self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        torch_dtype = dtype if dtype in ("float16", "float32", "bfloat16") else "auto"
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self.model.eval()

        # Determine the maximum supported input length for this encoder.
        max_pos = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
        tok_max = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tok_max, int) and tok_max > 0 and tok_max < 100000:
            max_supported = tok_max
        else:
            max_supported = None
        if isinstance(max_pos, int) and max_pos > 0:
            max_supported = min(max_supported, max_pos) if max_supported else max_pos
        self.max_supported_len = max_supported
        if self.max_supported_len and self.max_length > self.max_supported_len:
            logger.warning(
                f"[StructAlignRAG] embedding_max_seq_len too large for encoder, will cap | "
                f"configured={self.max_length} cap={self.max_supported_len} model={self.model_name}"
            )

    def _encode_batch(self, texts: List[str]) -> torch.Tensor:
        with torch.no_grad():
            max_len = self.max_length
            if self.max_supported_len:
                max_len = min(max_len, int(self.max_supported_len))
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_len,
                return_tensors="pt",
            ).to(self.model.device)
            outputs = self.model(**inputs)
            emb = _mean_pooling(outputs[0], inputs["attention_mask"])
        return emb

    def encode(self, texts: List[str], instruction: str = "") -> np.ndarray:
        # `instruction` is accepted for API compatibility with instruction-tuned embedders (e.g., NV-Embed-v2),
        # but Contriever-style encoders do not use it.
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        with self._encode_lock:
            results: List[torch.Tensor] = []
            if len(texts) <= self.batch_size:
                results.append(self._encode_batch(texts))
            else:
                # On some Windows terminals, tqdm's unicode blocks render as garbled characters.
                # Use ASCII progress bars for stable real-time visibility.
                pbar = tqdm(total=len(texts), desc="Batch Encoding", ascii=True, dynamic_ncols=True)
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i : i + self.batch_size]
                    results.append(self._encode_batch(batch))
                    pbar.update(len(batch))
                pbar.close()

            emb = torch.cat(results, dim=0)
            emb = emb.detach().float().cpu().numpy()

        if self.normalize and len(emb) > 0:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb = emb / norms
        return emb.astype(np.float32)
