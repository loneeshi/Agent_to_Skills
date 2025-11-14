"""Heuristic multi-layer decomposition utilities."""
from __future__ import annotations

from typing import List, Tuple, Dict, Any
import json
import re

from .schema import SkillTree, Skill, make_skill

SEP_RE = re.compile(r",|;|\.| and | & |\n|\r")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_\- ]+")
MIN_FRAGMENT_LEN = 4
MAX_NEW_SUBSKILLS = 8


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.strip().lower())
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("_") or "part"


def extract_fragments(description: str) -> List[str]:
    fragments: List[str] = []
    for part in SEP_RE.split(description or ""):
        part = part.strip()
        if len(part) < MIN_FRAGMENT_LEN:
            continue
        if not WORD_RE.search(part):
            continue
        fragments.append(part)
    return fragments


def decompose(tree: SkillTree) -> List[str]:
    added: List[str] = []
    for skill in list(tree.skills_index.values()):
        if skill.is_atomic or skill.subskills:
            continue
        fragments = extract_fragments(skill.description)[:MAX_NEW_SUBSKILLS]
        if not fragments:
            continue
        child_ids: List[str] = []
        for fragment in fragments:
            slug = slugify(fragment)
            cid = f"{skill.id}.{slug}"
            if cid in tree.skills_index:
                continue
            child = make_skill(
                id=cid,
                name=fragment[:60],
                description=f"Atomic aspect of {skill.id}: {fragment}",
                triggers=[{"type": t.type, "value": t.value, "confidence": t.confidence} for t in skill.triggers]
                + [{"type": "keyword", "value": slug, "confidence": 0.35}],
                success_conditions=[{"type": "fragment", "value": slug, "confidence": 0.4}],
                preconditions=list(skill.preconditions),
                postconditions=list(skill.postconditions),
                metadata={"generated_by": "heuristic_decompose"},
                confidence=skill.confidence * 0.5,
                is_atomic=True,
            )
            tree.add_skill(child)
            child_ids.append(cid)
            added.append(cid)
        if child_ids:
            skill.subskills.extend(child_ids)
    return added


def load_skill_tree(data: str) -> Tuple[SkillTree, bool, List[str]]:
    raw = json.loads(data)
    if raw.get("root_skills"):
        tree = SkillTree.from_dict(raw)
        root_ids = [skill.id for skill in tree.root_skills]
        return tree, False, root_ids
    if "root" in raw:
        tree = SkillTree()
        root_ids: List[str] = []
        def walk(node: Dict[str, Any], as_root: bool = False):
            node_copy = dict(node)
            children = node_copy.pop("children", [])
            skill = Skill.from_dict(node_copy)
            tree.add_skill(skill, as_root=as_root)
            if as_root:
                root_ids.append(skill.id)
            for child in children:
                walk(child, False)

        walk(raw["root"], True)
        return tree, True, root_ids
    raise ValueError("Unsupported SkillTree JSON shape: expected root_skills or root")


def export_nested(tree: SkillTree, root_id: str) -> Dict[str, Any]:
    def to_nested(skill_id: str):
        skill = tree.get(skill_id)
        if not skill:
            return None
        data = skill.to_dict()
        data["children"] = [child for child in (to_nested(cid) for cid in skill.subskills) if child]
        return data

    root = to_nested(root_id)
    return {"skill_tree_version": tree.skill_tree_version, "root": root}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Heuristic SkillTree decomposition")
    parser.add_argument("input", help="Path to skill tree JSON (flat or nested)")
    parser.add_argument("output", nargs="?", help="Destination JSON file")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as fh:
        data = fh.read()
    tree, was_nested, root_ids = load_skill_tree(data)
    new_ids = decompose(tree)
    print(f"Generated {len(new_ids)} new subskills")
    print("Validation:", tree.validate())

    out_path = args.output or args.input.replace(".json", "_decomposed.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(tree.to_json())
    print("Saved decomposed tree to", out_path)

    if was_nested and root_ids:
        nested_path = out_path.replace(".json", "_nested.json")
        nested_data = export_nested(tree, root_ids[0])
        with open(nested_path, "w", encoding="utf-8") as fh:
            json.dump(nested_data, fh, ensure_ascii=False, indent=2)
        print("Saved nested decomposed tree to", nested_path)


__all__ = ["decompose", "load_skill_tree", "export_nested"]
