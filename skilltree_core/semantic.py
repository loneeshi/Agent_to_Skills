"""Bag-of-words semantic retrieval for SkillTree skills."""
from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple
import math
import re

from .schema import SkillTree, Skill

TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "into",
    "from",
    "each",
    "per",
    "are",
    "was",
    "were",
    "been",
    "being",
    "about",
    "onto",
    "over",
    "under",
    "your",
    "user",
    "task",
    "skill",
    "llm",
    "ai",
    "data",
    "analysis",
    "review",
    "build",
    "make",
}


def tokenize(text: str) -> List[str]:
    tokens = [tok.lower() for tok in TOKEN_RE.findall(text or "")]
    return [tok for tok in tokens if len(tok) > 2 and tok not in STOPWORDS]


def vector_from_tokens(tokens: List[str]) -> Counter:
    vec = Counter()
    for tok in tokens:
        vec[tok] += 1.0
    return vec


def cosine_similarity(vec_a: Counter, vec_b: Counter, norm_b: float) -> float:
    if not vec_a or not vec_b or norm_b == 0:
        return 0.0
    dot = 0.0
    for token, weight in vec_a.items():
        dot += weight * vec_b.get(token, 0.0)
    norm_a = math.sqrt(sum(w * w for w in vec_a.values()))
    if norm_a == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticRetriever:
    def __init__(self, tree: SkillTree):
        self.tree = tree
        self.vectors: Dict[str, Counter] = {}
        self.norms: Dict[str, float] = {}

    def rebuild(self):
        self.vectors.clear()
        self.norms.clear()
        for skill in self.tree.skills_index.values():
            self._index_skill(skill)

    def _skill_text(self, skill: Skill) -> str:
        parts = [skill.name, skill.description]
        parts.extend([t.value for t in skill.triggers])
        parts.extend([s.value for s in skill.success_conditions])
        parts.append(" ".join(skill.preconditions))
        parts.append(" ".join(skill.postconditions))
        parts.append(" ".join(skill.subskills))
        parts.extend([str(v) for v in skill.metadata.values() if isinstance(v, str)])
        return " ".join(filter(None, parts))

    def _index_skill(self, skill: Skill):
        text = self._skill_text(skill)
        vec = vector_from_tokens(tokenize(text))
        self.vectors[skill.id] = vec
        self.norms[skill.id] = math.sqrt(sum(v * v for v in vec.values()))

    def refresh_skill(self, skill_id: str):
        skill = self.tree.get(skill_id)
        if not skill:
            self.vectors.pop(skill_id, None)
            self.norms.pop(skill_id, None)
            return
        self._index_skill(skill)

    def search(self, query: str, top_k: int = 5, min_score: float = 0.1) -> List[Tuple[Skill, float]]:
        if not self.vectors:
            self.rebuild()
        query_vec = vector_from_tokens(tokenize(query))
        results: List[Tuple[Skill, float]] = []
        for sid, vec in self.vectors.items():
            score = cosine_similarity(query_vec, vec, self.norms.get(sid, 0.0))
            if score >= min_score:
                skill = self.tree.get(sid)
                if skill:
                    results.append((skill, score))
        results.sort(key=lambda item: item[1], reverse=True)
        return results[:top_k]


def semantic_search(tree: SkillTree, query: str, top_k: int = 5, min_score: float = 0.1) -> List[Tuple[Skill, float]]:
    retriever = SemanticRetriever(tree)
    retriever.rebuild()
    return retriever.search(query, top_k=top_k, min_score=min_score)


__all__ = ["semantic_search", "SemanticRetriever"]
