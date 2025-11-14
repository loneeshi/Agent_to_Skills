from __future__ import annotations

"""Utilities to learn skills from dialogues and emit a SkillTree structure."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Any, Tuple
import json

from .schema import SkillTree, make_skill, Skill
from .runtime import retrieve_semantic
from .decompose import export_nested


@dataclass
class DialogueSnippet:
    """A small wrapper for dialogue text and optional metadata."""

    text: str
    metadata: Dict[str, Any] | None = None


@dataclass
class LearnedSkill:
    """Represents a skill extracted or reinforced from dialogue."""

    id: str
    name: str
    description: str
    evidence: List[DialogueSnippet]
    confidence: float = 0.6
    parent_id: str | None = None

    def to_skill(self) -> Skill:
        triggers = [
            {"type": "dialogue_pattern", "value": snippet.text, "confidence": 0.5}
            for snippet in self.evidence
        ]
        metadata = {"evidence": [snippet.text for snippet in self.evidence]}
        if self.parent_id:
            metadata["parent_id"] = self.parent_id
        return make_skill(
            id=self.id,
            name=self.name,
            description=self.description,
            triggers=triggers,
            metadata=metadata,
            confidence=self.confidence,
            subskills=[],
        )


def learn_skills_from_dialogue(
    base_tree: SkillTree,
    snippets: Iterable[DialogueSnippet],
    similarity_threshold: float = 0.3,
    parent_skill_id: str | None = None,
) -> Tuple[SkillTree, List[LearnedSkill]]:
    """Return a new tree and the list of learned skills from dialogue snippets.

    The function compares snippets against existing skills and either links to
    the best match or creates fresh skills under the specified parent.
    """

    learned: List[LearnedSkill] = []
    new_tree = SkillTree.from_dict(base_tree.to_dict())

    for idx, snippet in enumerate(snippets):
        results = retrieve_semantic(new_tree, snippet.text, top_k=1, min_score=similarity_threshold)
        if results:
            skill, similarity = results[0]
            evidence_meta = skill.metadata.setdefault("dialogue_evidence", [])
            evidence_meta.append(snippet.text)
            skill.confidence = max(skill.confidence, similarity)
        else:
            skill_id = f"skill.dialogue_{idx}"
            learned_skill = LearnedSkill(
                id=skill_id,
                name=f"Learned Skill {idx+1}",
                description=snippet.text,
                evidence=[snippet],
                parent_id=parent_skill_id,
            )
            new_skill = learned_skill.to_skill()
            new_tree.add_skill(new_skill, as_root=parent_skill_id is None)
            if parent_skill_id:
                parent = new_tree.get(parent_skill_id)
                if parent and skill_id not in parent.subskills:
                    parent.subskills.append(skill_id)
            learned.append(learned_skill)

    return new_tree, learned


def export_tree_from_dialogue(snippets: Iterable[str]) -> SkillTree:
    """Utility to build a SkillTree directly from dialogue texts."""

    dialogue_snippets = [DialogueSnippet(text=text) for text in snippets]
    base = SkillTree()

    tree, learned = learn_skills_from_dialogue(base, dialogue_snippets)
    if not tree.root_skills:
        default = make_skill(
            id="skill.dialogue_root",
            name="Dialogue Root",
            description="Root skill generated from dialogue",
            subskills=[skill.id for skill in tree.skills_index.values()],
        )
        tree.add_skill(default, as_root=True)
    else:
        for skill in list(tree.skills_index.values()):
            if skill.id not in {root.id for root in tree.root_skills}:
                continue
            for learned_skill in learned:
                if learned_skill.parent_id == skill.id and learned_skill.id not in skill.subskills:
                    skill.subskills.append(learned_skill.id)
    return tree


def write_skill_tree_nested_json(
    tree: SkillTree,
    output_path: str,
    root_id: str | None = None,
    root_name: str = "Dialogue Root",
    root_description: str = "Root skill generated from dialogue",
) -> Dict[str, Any]:
    """Serialize the SkillTree into the nested JSON structure used in outputs."""

    working_tree = SkillTree()
    root_ids = {skill.id for skill in tree.root_skills}
    for skill in tree.skills_index.values():
        skill_copy = Skill.from_dict(skill.to_dict())
        working_tree.add_skill(skill_copy, as_root=(skill.id in root_ids))

    if not working_tree.root_skills:
        raise ValueError("SkillTree has no root skills; cannot write nested JSON")

    if root_id:
        target_root_id = root_id
    elif len(working_tree.root_skills) == 1:
        target_root_id = working_tree.root_skills[0].id
    else:
        aggregator_id = "skill.dialogue_root"
        suffix = 1
        while aggregator_id in working_tree.skills_index:
            suffix += 1
            aggregator_id = f"skill.dialogue_root_{suffix}"
        aggregator = make_skill(
            id=aggregator_id,
            name=root_name,
            description=root_description,
            subskills=[skill.id for skill in working_tree.root_skills],
            metadata={"generated_by": "write_skill_tree_nested_json"},
        )
        working_tree.add_skill(aggregator, as_root=True)
        target_root_id = aggregator_id

    nested_data = export_nested(working_tree, target_root_id)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(nested_data, fh, ensure_ascii=False, indent=2)
    return nested_data


def export_dialogue_tree_to_json(
    snippets: Iterable[str],
    output_path: str,
    similarity_threshold: float = 0.3,
    parent_skill_id: str | None = None,
) -> Dict[str, Any]:
    """Learn skills from dialogue and persist the nested tree JSON to disk."""

    dialogue_snippets = [DialogueSnippet(text=text) for text in snippets]
    tree, _ = learn_skills_from_dialogue(
        SkillTree(),
        dialogue_snippets,
        similarity_threshold=similarity_threshold,
        parent_skill_id=parent_skill_id,
    )
    return write_skill_tree_nested_json(tree, output_path)


__all__ = [
    "DialogueSnippet",
    "LearnedSkill",
    "learn_skills_from_dialogue",
    "export_tree_from_dialogue",
    "write_skill_tree_nested_json",
    "export_dialogue_tree_to_json",
]
