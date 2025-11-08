"""
Simple embedding provider with two backends:
- openai: uses OpenAI text-embedding models (or Azure OpenAI if OPENAI_API_TYPE=azure_ai)
- hf: uses local Hugging Face sentence-transformers models (CPU/GPU)

Configure via environment variables:
- EMBEDDING_PROVIDER=openai|hf  (default: openai)
- EMBEDDING_MODEL (default: text-embedding-3-large for openai, intfloat/e5-base-v2 for hf)
- DEVICE (for hf backend; e.g., cuda, mps, or cpu)
- OPENAI_API_KEY / AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_VERSION
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class EmbeddingConfig:
    # Values will be populated dynamically; defaults here are placeholders
    provider: Optional[str] = None
    model: Optional[str] = None
    device: Optional[str] = None


class _OpenAIEmbedder:
    def __init__(self, model: str):
        import openai  # type: ignore
        self.model = model
        # auto-detect Azure vs OpenAI
        if os.getenv("OPENAI_API_TYPE") == "azure_ai" or os.getenv("AZURE_OPENAI_API_KEY"):
            self.client = openai.AzureOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
            )
        else:
            self.client = openai.OpenAI(
                base_url=os.getenv("OPENAI_ENDPOINT"),
                api_key=os.getenv("OPENAI_API_KEY"),
            )

    def embed(self, text: str) -> List[float]:
        resp = self.client.embeddings.create(model=self.model, input=text)
        return list(resp.data[0].embedding)

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [list(d.embedding) for d in resp.data]


class _HFEmbedder:
    def __init__(self, model: str, device: Optional[str] = None):
        # Prefer sentence-transformers for ease; falls back to transformers if needed
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._backend = "st"
            # A sensible default if user didn't override the model
            if model == "text-embedding-3-large":
                model = os.getenv("EMBEDDING_MODEL", "intfloat/e5-base-v2")
            self.model = SentenceTransformer(model, device=device)
        except Exception:
            from transformers import AutoModel, AutoTokenizer  # type: ignore
            import torch  # type: ignore
            self._backend = "hf"
            if model == "text-embedding-3-large":
                model = os.getenv("EMBEDDING_MODEL", "intfloat/e5-base-v2")
            self.tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model, trust_remote_code=True)
            self.device = device or ("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
            self.model.to(self.device)

    def _pool(self, outputs, attention_mask):
        import torch  # type: ignore
        token_embeddings = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        masked = token_embeddings * mask
        summed = torch.sum(masked, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return (summed / counts).detach().cpu().numpy().tolist()

    def embed(self, text: str) -> List[float]:
        if self._backend == "st":
            v = self.model.encode(text, normalize_embeddings=True)
            return v.tolist() if hasattr(v, "tolist") else list(v)
        else:
            import torch  # type: ignore
            batch = self.tok(text, return_tensors="pt", truncation=True, max_length=512)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            vec = self._pool(outputs, batch["attention_mask"])[0]
            # L2 normalize
            norm = (sum(x*x for x in vec) ** 0.5) or 1.0
            return [x / norm for x in vec]

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        if self._backend == "st":
            mat = self.model.encode(texts, normalize_embeddings=True)
            return [row.tolist() if hasattr(row, "tolist") else list(row) for row in mat]
        else:
            import torch  # type: ignore
            batch = self.tok(texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            mat = self._pool(outputs, batch["attention_mask"])
            # L2 normalize rows
            out = []
            for vec in mat:
                norm = (sum(x*x for x in vec) ** 0.5) or 1.0
                out.append([x / norm for x in vec])
            return out


_embedder = None

def get_embedder():
    global _embedder
    if _embedder is not None:
        return _embedder

    # Read environment lazily at call time to honor runtime overrides
    provider_env = os.getenv("EMBEDDING_PROVIDER", "openai").lower()
    device = os.getenv("DEVICE")
    if provider_env == "openai":
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        _embedder = _OpenAIEmbedder(model)
    elif provider_env == "hf":
        model = os.getenv("EMBEDDING_MODEL", "intfloat/e5-base-v2")
        _embedder = _HFEmbedder(model, device)
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER={provider_env}")
    return _embedder


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = (sum(x*x for x in a) ** 0.5) or 1.0
    nb = (sum(y*y for y in b) ** 0.5) or 1.0
    return dot / (na * nb)
