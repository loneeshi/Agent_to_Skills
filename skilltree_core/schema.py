"""SkillTree schema core primitives.

This is a direct successor of the former `skilltree_engine.schema`. It
provides dataclasses for Trigger/SuccessCondition/Skill and a SkillTree
container with validation, serialization, and convenience helpers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
import json
import time

SCHEMA_VERSION = "1.0"


@dataclass
class Trigger:
    type: str
    value: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "value": self.value, "confidence": self.confidence}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Trigger":
        return Trigger(type=d["type"], value=d["value"], confidence=float(d.get("confidence", 0.0)))


@dataclass
class SuccessCondition:
    type: str
    value: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, "value": self.value, "confidence": self.confidence}

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SuccessCondition":
        return SuccessCondition(type=d["type"], value=d["value"], confidence=float(d.get("confidence", 0.0)))


@dataclass
class Skill:
    id: str
    name: str
    description: str
    triggers: List[Trigger] = field(default_factory=list)
    success_conditions: List[SuccessCondition] = field(default_factory=list)
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    subskills: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    is_atomic: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "triggers": [t.to_dict() for t in self.triggers],
            "success_conditions": [s.to_dict() for s in self.success_conditions],
            "preconditions": list(self.preconditions),
            "postconditions": list(self.postconditions),
            "subskills": list(self.subskills),
            "metadata": self.metadata,
            "confidence": self.confidence,
            "is_atomic": self.is_atomic,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Skill":
        return Skill(
            id=d["id"],
            name=d.get("name", d["id"]),
            description=d.get("description", ""),
            triggers=[Trigger.from_dict(x) for x in d.get("triggers", [])],
            success_conditions=[SuccessCondition.from_dict(x) for x in d.get("success_conditions", [])],
            preconditions=list(d.get("preconditions", [])),
            postconditions=list(d.get("postconditions", [])),
            subskills=list(d.get("subskills", [])),
            metadata=d.get("metadata", {}),
            confidence=float(d.get("confidence", 0.0)),
            is_atomic=bool(d.get("is_atomic", False)),
        )


@dataclass
class SkillTree:
    skill_tree_version: str = SCHEMA_VERSION
    root_skills: List[Skill] = field(default_factory=list)
    skills_index: Dict[str, Skill] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        for skill in self.root_skills:
            self.register_skill(skill)

    def register_skill(self, skill: Skill):
        if skill.id in self.skills_index:
            raise ValueError(f"Duplicate skill id: {skill.id}")
        self.skills_index[skill.id] = skill

    def add_skill(self, skill: Skill, as_root: bool = False):
        self.register_skill(skill)
        if as_root:
            self.root_skills.append(skill)

    def get(self, skill_id: str) -> Optional[Skill]:
        return self.skills_index.get(skill_id)

    def update_skill(self, skill_id: str, **updates):
        skill = self.get(skill_id)
        if not skill:
            raise KeyError(f"Skill {skill_id} not found")
        for key, value in updates.items():
            if not hasattr(skill, key):
                continue
            if key == "triggers" and isinstance(value, list):
                value = [Trigger.from_dict(x) if isinstance(x, dict) else x for x in value]
            if key == "success_conditions" and isinstance(value, list):
                value = [SuccessCondition.from_dict(x) if isinstance(x, dict) else x for x in value]
            setattr(skill, key, value)
        skill.metadata.setdefault("updated_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        return skill

    def delete_skill(self, skill_id: str) -> bool:
        skill = self.get(skill_id)
        if not skill:
            return False
        for s in self.skills_index.values():
            if skill_id in s.subskills:
                s.subskills = [x for x in s.subskills if x != skill_id]
            if skill_id in s.preconditions:
                s.preconditions = [x for x in s.preconditions if x != skill_id]
            if skill_id in s.postconditions:
                s.postconditions = [x for x in s.postconditions if x != skill_id]
        self.root_skills = [r for r in self.root_skills if r.id != skill_id]
        del self.skills_index[skill_id]
        return True

    def merge_skills(self, target_id: str, source_id: str) -> Skill:
        target = self.get(target_id)
        source = self.get(source_id)
        if not target or not source:
            raise KeyError("target or source skill missing")
        target.description = target.description or source.description
        target.triggers.extend([t for t in source.triggers if t not in target.triggers])
        target.success_conditions.extend([s for s in source.success_conditions if s not in target.success_conditions])
        target.preconditions = list(dict.fromkeys(target.preconditions + source.preconditions))
        target.postconditions = list(dict.fromkeys(target.postconditions + source.postconditions))
        target.subskills = list(dict.fromkeys(target.subskills + source.subskills))
        target.metadata.setdefault("merged_from", []).append(source.id)
        target.confidence = max(target.confidence, source.confidence)
        self.delete_skill(source_id)
        return target

    def validate(self) -> Dict[str, Any]:
        problems = []
        for skill in self.skills_index.values():
            for field_name in ("preconditions", "postconditions", "subskills"):
                for ref in getattr(skill, field_name):
                    if ref not in self.skills_index:
                        problems.append({"skill": skill.id, "missing_ref": ref, "field": field_name})
        edges = []
        for skill in self.skills_index.values():
            for nxt in skill.postconditions + skill.subskills:
                edges.append((skill.id, nxt))
        visited: Set[str] = set()
        stack: Set[str] = set()
        cycles = []

        def dfs(node: str):
            if node in stack:
                cycles.append(list(stack) + [node])
                return
            if node in visited:
                return
            visited.add(node)
            stack.add(node)
            for a, b in edges:
                if a == node:
                    dfs(b)
            stack.remove(node)

        for node in self.skills_index:
            dfs(node)
        if cycles:
            problems.append({"cycles": cycles})
        return {"ok": not problems, "problems": problems}

    def find_by_intent(self, intent: str) -> List[Skill]:
        matches = []
        intent_lower = intent.lower()
        for skill in self.skills_index.values():
            for trigger in skill.triggers:
                if trigger.type in ("intent", "dialog_intent") and intent_lower in trigger.value.lower():
                    matches.append(skill)
                    break
        return matches

    def find_by_keyword(self, keyword: str) -> List[Skill]:
        term = keyword.lower()
        return [s for s in self.skills_index.values() if term in s.name.lower() or term in s.description.lower()]

    def gap_skills(self) -> List[str]:
        refs: Set[str] = set()
        for skill in self.skills_index.values():
            refs.update(skill.preconditions)
            refs.update(skill.postconditions)
            refs.update(skill.subskills)
        return [r for r in refs if r not in self.skills_index]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_tree_version": self.skill_tree_version,
            "root_skills": [s.to_dict() for s in self.root_skills],
        }

    def to_json(self, **json_kwargs) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **json_kwargs)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "SkillTree":
        version = data.get("skill_tree_version", SCHEMA_VERSION)
        roots = [Skill.from_dict(node) for node in data.get("root_skills", [])]
        return SkillTree(skill_tree_version=version, root_skills=roots)

    @staticmethod
    def from_json(raw: str) -> "SkillTree":
        return SkillTree.from_dict(json.loads(raw))


def make_skill(
    id: str,
    name: str,
    description: str,
    triggers=None,
    success_conditions=None,
    preconditions=None,
    postconditions=None,
    subskills=None,
    metadata=None,
    confidence: float = 0.0,
    is_atomic: bool = False,
) -> Skill:
    now = time.strftime("%Y-%m-%d", time.gmtime())
    md = metadata or {"created_at": now}
    return Skill(
        id=id,
        name=name,
        description=description,
        triggers=[Trigger(**t) if isinstance(t, dict) else t for t in (triggers or [])],
        success_conditions=[SuccessCondition(**s) if isinstance(s, dict) else s for s in (success_conditions or [])],
        preconditions=list(preconditions or []),
        postconditions=list(postconditions or []),
        subskills=list(subskills or []),
        metadata=md,
        confidence=confidence,
        is_atomic=is_atomic,
    )


__all__ = [
    "SkillTree",
    "Skill",
    "Trigger",
    "SuccessCondition",
    "make_skill",
]
