from __future__ import annotations

import os
import threading
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


class NVEmbedV2Embedder:
    """
    NVIDIA NV-Embed-v2 embedder.

    Notes:
    - NV-Embed-v2 uses `trust_remote_code=True` and exposes a model-side `.encode(...)` API.
    - For query embedding, pass a non-empty `instruction` string. Following NV's recommended format,
      we prefix it as: "Instruct: {instruction}\\nQuery: ".
    - This class returns float32 numpy arrays (optionally L2-normalized), consistent with our FAISS IP indices.
    """

    def __init__(
        self,
        model_name: str,
        batch_size: int = 16,
        max_length: int = 4096,
        normalize: bool = True,
        dtype: str = "auto",
        num_workers: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize
        self.dtype = dtype
        self.num_workers = num_workers
        # NV-Embed-v2 remote-code model is not guaranteed to be thread-safe for concurrent `.encode()` calls.
        self._encode_lock = threading.Lock()

        logger.info(f"[StructAlignRAG] loading embedding model | name={self.model_name}")

        torch_dtype = dtype if dtype in ("float16", "float32", "bfloat16") else "auto"
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self.model.eval()

        if self.num_workers is None:
            # Keep it conservative on Windows; on Linux boxes with many cores, users can override later if needed.
            cpu = os.cpu_count() or 8
            self.num_workers = int(min(32, max(0, cpu)))

        if not hasattr(self.model, "encode"):
            raise AttributeError(
                f"NV-Embed-v2 remote code did not expose `.encode(...)` | model={self.model_name}. "
                "Check transformers version / trust_remote_code / model repo."
            )

    def _encode_batch(self, texts: List[str], instruction: str = "") -> torch.Tensor:
        params = {
            "prompts": texts,
            "max_length": int(self.max_length),
            "instruction": instruction,
            "num_workers": int(self.num_workers or 0),
        }
        with torch.no_grad():
            emb = self.model.encode(**params)
        if not isinstance(emb, torch.Tensor):
            emb = torch.as_tensor(emb)
        return emb

    def encode(self, texts: List[str], instruction: str = "") -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        instr = (instruction or "").strip()
        if instr:
            instr = f"Instruct: {instr}\nQuery: "
        else:
            instr = ""

        with self._encode_lock:
            results: List[torch.Tensor] = []
            if len(texts) <= self.batch_size:
                results.append(self._encode_batch(texts, instruction=instr))
            else:
                pbar = tqdm(total=len(texts), desc="Batch Encoding", ascii=True, dynamic_ncols=True)
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i : i + self.batch_size]
                    results.append(self._encode_batch(batch, instruction=instr))
                    pbar.update(len(batch))
                pbar.close()

            emb = torch.cat(results, dim=0)
            emb = emb.detach().float().cpu().numpy()

        if self.normalize and len(emb) > 0:
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            emb = emb / norms
        return emb.astype(np.float32)

