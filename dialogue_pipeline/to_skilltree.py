"""Convert dialogue pipeline artifacts into SkillTree JSON seeds.

Usage:
    python to_skilltree.py --artifact artifacts/course_selection_002_dialogues.json \
        --out ../../outputs/skilltree/course_selection_002_seed.json

This script reads the schema + intent plan emitted by `pipeline.py`,
constructs a SkillTree with a root skill plus one child per intent,
and saves both flat (`SkillTree.to_json`) and nested representations
ready for downstream decomposition.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from skilltree_engine.schema import SkillTree, make_skill
from skilltree_engine.decompose_skills import export_nested


def load_artifact(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def example_turns(dialogues: List[List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    mapping: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for convo_idx, convo in enumerate(dialogues or []):
        for turn in convo:
            intent_id = turn.get("intent")
            if not intent_id:
                continue
            mapping[intent_id].append(
                {
                    "speaker": turn.get("speaker"),
                    "text": turn.get("text"),
                    "conversation_index": convo_idx,
                }
            )
    return mapping


def build_skill_tree(artifact: Dict[str, Any]) -> SkillTree:
    schema = artifact.get("schema", {})
    intents = artifact.get("intent_plan", [])
    dialogues = artifact.get("dialogues", [])
    task_id = artifact.get("task_id", "task")

    st = SkillTree()
    root_id = f"skill.{task_id}"
    root = make_skill(
        id=root_id,
        name=schema.get("goal", "Dialogue Plan"),
        description=schema.get("goal", ""),
        triggers=[{"type": "intent", "value": task_id, "confidence": 0.85}],
        success_conditions=[{"type": "student_confirmation", "value": "plan_confirmed", "confidence": 0.8}],
        metadata={
            "roles": schema.get("roles"),
            "pass_budgets": schema.get("pass_budgets"),
            "courses": schema.get("courses"),
            "constraints": schema.get("constraints"),
            "edge_cases": schema.get("edge_cases"),
            "created_at": time.strftime("%Y-%m-%d"),
            "source": artifact.get("task_id"),
        },
        confidence=0.85,
        is_atomic=False,
    )
    st.add_skill(root, as_root=True)

    samples = example_turns(dialogues)
    prev_child_id: str | None = None
    child_ids: List[str] = []
    for order, intent in enumerate(intents, start=1):
        intent_id = intent.get("id", f"intent_{order}")
        skill_id = f"{root_id}.{intent_id}"
        summary = intent.get("summary", "")
        required = intent.get("required_slots") or []
        branch = intent.get("branch")
        desc = summary or intent_id.replace("_", " ")
        if branch:
            desc += f" Branch: {branch}."
        if required:
            desc += " Required slots: " + ", ".join(required)
        metadata = {
            "speaker": intent.get("speaker"),
            "required_slots": required,
            "branch": branch,
            "order": order,
            "examples": samples.get(intent_id, [])[:3],
        }
        skill = make_skill(
            id=skill_id,
            name=intent_id.replace("_", " ").title(),
            description=desc,
            triggers=[{"type": "intent", "value": intent_id, "confidence": 0.7}],
            success_conditions=[{"type": "state_change", "value": f"{intent_id}_handled", "confidence": 0.6}],
            preconditions=[prev_child_id] if prev_child_id else [],
            metadata=metadata,
            confidence=0.6,
            is_atomic=False,
        )
        st.add_skill(skill)
        child_ids.append(skill_id)
        prev_child_id = skill_id

    root.subskills = child_ids
    return st


def write_outputs(st: SkillTree, root_id: str, out_prefix: Path):
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    if out_prefix.suffix:
        flat_path = out_prefix
        base_name = out_prefix.stem
        nested_name = f"{base_name}_nested.json"
        nested_path = out_prefix.with_name(nested_name)
    else:
        flat_path = out_prefix.with_suffix(".json")
        nested_path = out_prefix.with_name(out_prefix.name + "_nested.json")
    flat_path.write_text(st.to_json(indent=2), encoding="utf-8")
    print(f"Wrote flat SkillTree to {flat_path}")
    nested = export_nested(st, root_id)
    nested_path.write_text(json.dumps(nested, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote nested SkillTree to {nested_path}")
    return flat_path, nested_path


def main():
    parser = argparse.ArgumentParser(description="Convert dialogue artifact into SkillTree JSON")
    parser.add_argument("--artifact", type=Path, required=True, help="Path to dialogue artifact JSON")
    parser.add_argument(
        "--out",
        type=Path,
        required=False,
        default=Path("../../outputs/skilltree/course_selection_seed"),
        help="Output path prefix (without extension)",
    )
    args = parser.parse_args()

    artifact = load_artifact(args.artifact)
    st = build_skill_tree(artifact)
    root_ids = [root.id for root in st.root_skills]
    if not root_ids:
        raise RuntimeError("No root skills generated")
    flat, nested = write_outputs(st, root_ids[0], args.out)
    validation = st.validate()
    print("Validation:", validation)
    if not validation.get("ok"):
        print("Validation problems detected; review the generated tree.")


if __name__ == "__main__":
    main()
