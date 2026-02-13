from __future__ import annotations

import json
import os
from typing import List

import faiss
import numpy as np


def build_ip_index(emb: np.ndarray) -> faiss.Index:
    if not isinstance(emb, np.ndarray):
        raise TypeError("emb must be np.ndarray")
    if emb.ndim != 2:
        raise ValueError("emb must be 2D")
    emb = np.ascontiguousarray(emb.astype(np.float32))
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)
    return index


def save_faiss_index(index: faiss.Index, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    faiss.write_index(index, path)


def load_faiss_index(path: str) -> faiss.Index:
    return faiss.read_index(path)


def save_id_map(ids: List[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ids, f, ensure_ascii=False)


def load_id_map(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("id map must be a list")
    return [str(x) for x in data]

