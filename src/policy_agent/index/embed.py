from __future__ import annotations

import os
from typing import List

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI()

EMBED_MODEL = "text-embedding-3-small"


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Returns a numpy array of shape (n_texts, dim).
    """
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    resp = client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )

    vectors = [item.embedding for item in resp.data]
    return np.array(vectors, dtype=np.float32)
