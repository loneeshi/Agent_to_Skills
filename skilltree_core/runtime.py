"""High-level SkillTree runtime helpers.

This refactors the previous `skilltree_engine.runtime` module without any
external dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
import re

from .schema import SkillTree, Skill, make_skill
from .semantic import semantic_search


@dataclass
class UpdateOp:
    op: str  # ADD | UPDATE | DELETE | MERGE
    skill_id: str
    payload: Dict[str, Any] | None = None
    target_id: Optional[str] = None


def ingest_skill_tree_json(runtime: SkillTree, data: str) -> List[str]:
    incoming = SkillTree.from_json(data)
    created: List[str] = []
    for root in incoming.root_skills:
        if root.id not in runtime.skills_index:
            runtime.add_skill(root, as_root=True)
            created.append(root.id)
        else:
            existing = runtime.get(root.id)
            if existing:
                existing.confidence = max(existing.confidence, root.confidence)
                existing.metadata.setdefault("merged_root_versions", []).append(
                    root.metadata.get("created_at")
                )
        for sid in root.subskills:
            if sid not in runtime.skills_index:
                stub = make_skill(id=sid, name=sid, description=f"Stub for {sid}", confidence=0.0)
                runtime.add_skill(stub)
                created.append(sid)
    return created


def apply_updates(runtime: SkillTree, ops: List[UpdateOp]) -> Dict[str, Any]:
    results = []
    for op in ops:
        try:
            if op.op == "ADD":
                if op.payload is None:
                    raise ValueError("ADD requires payload")
                skill = make_skill(**op.payload)
                runtime.add_skill(skill, as_root=op.payload.get("as_root", False))
                results.append({"op": op.op, "skill": skill.id, "status": "ok"})
            elif op.op == "UPDATE":
                runtime.update_skill(op.skill_id, **(op.payload or {}))
                results.append({"op": op.op, "skill": op.skill_id, "status": "ok"})
            elif op.op == "DELETE":
                ok = runtime.delete_skill(op.skill_id)
                results.append({"op": op.op, "skill": op.skill_id, "status": "ok" if ok else "not_found"})
            elif op.op == "MERGE":
                if not op.target_id:
                    raise ValueError("MERGE requires target_id")
                merged = runtime.merge_skills(op.target_id, op.skill_id)
                results.append(
                    {
                        "op": op.op,
                        "into": op.target_id,
                        "from": op.skill_id,
                        "status": "ok",
                        "merged_confidence": merged.confidence,
                    }
                )
            else:
                results.append({"op": op.op, "skill": op.skill_id, "status": "unknown_op"})
        except Exception as exc:
            results.append({"op": op.op, "skill": op.skill_id, "status": "error", "error": str(exc)})
    return {"results": results}


INTENT_RE = re.compile(r"intent:(\w+)", re.I)


def retrieve_for_query(runtime: SkillTree, text: str, top_k: int = 5) -> List[Skill]:
    intent_tokens = INTENT_RE.findall(text)
    collected: List[Skill] = []
    seen: Set[str] = set()
    if intent_tokens:
        for token in intent_tokens:
            for skill in runtime.find_by_intent(token):
                if skill.id not in seen:
                    collected.append(skill)
                    seen.add(skill.id)
    if not collected:
        for word in re.findall(r"[A-Za-z_]+", text):
            for skill in runtime.find_by_keyword(word):
                if skill.id not in seen:
                    collected.append(skill)
                    seen.add(skill.id)
    return collected[:top_k]


def retrieve_semantic(runtime: SkillTree, text: str, top_k: int = 5, min_score: float = 0.1):
    return semantic_search(runtime, text, top_k=top_k, min_score=min_score)


def suggest_missing_skills(runtime: SkillTree) -> List[Dict[str, Any]]:
    return [{"suggest_id": sid, "reason": "Referenced but not defined"} for sid in runtime.gap_skills()]


def schedule_to_target(runtime: SkillTree, target_skill_id: str) -> List[str]:
    target = runtime.get(target_skill_id)
    if not target:
        return []
    order: List[str] = []
    visited: Set[str] = set()

    def dfs(skill_id: str):
        if skill_id in visited:
            return
        visited.add(skill_id)
        skill = runtime.get(skill_id)
        if not skill:
            return
        for prereq in skill.preconditions:
            dfs(prereq)
        order.append(skill_id)

    dfs(target_skill_id)
    return order


__all__ = [
    "UpdateOp",
    "ingest_skill_tree_json",
    "apply_updates",
    "retrieve_for_query",
    "retrieve_semantic",
    "suggest_missing_skills",
    "schedule_to_target",
]
