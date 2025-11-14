from __future__ import annotations

"""CLI for generating a nested SkillTree JSON from dialogue snippets."""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skilltree_core.dialogue_export import export_dialogue_tree_to_json  # noqa: E402


def load_snippets(path: str) -> List[str]:
    ext = Path(path).suffix.lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, list):
            if all(isinstance(item, str) for item in data):
                return [item.strip() for item in data if item.strip()]
            if all(isinstance(item, dict) and "text" in item for item in data):
                return [item["text"].strip() for item in data if item.get("text", "").strip()]
        raise ValueError("Unsupported JSON format. Expected list of strings or list of {text: ...} objects.")
    with open(path, "r", encoding="utf-8") as fh:
        lines = [line.strip() for line in fh]
    return [line for line in lines if line]


def main(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dialogue_file", help="Path to a text or JSON file containing dialogue snippets")
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "outputs/skilltree/dialogue_tree_nested.json"),
        help="Destination JSON path (default: outputs/skilltree/dialogue_tree_nested.json)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.3,
        help="Semantic similarity threshold for treating snippets as new skills",
    )
    parser.add_argument(
        "--parent-skill",
        help="Optional parent skill id under which new skills should be attached",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    snippets = load_snippets(args.dialogue_file)
    if not snippets:
        raise ValueError("No dialogue snippets found in the provided file.")

    nested = export_dialogue_tree_to_json(
        snippets,
        output_path=args.output,
        similarity_threshold=args.similarity_threshold,
        parent_skill_id=args.parent_skill,
    )
    print(f"Wrote nested SkillTree JSON to {args.output}")
    print(f"Root skill: {nested['root']['id']}")
    print(f"Children count: {len(nested['root'].get('children', []))}")


if __name__ == "__main__":
    main()
