"""
Minimal demo script for the SkillTree initial implementation.
Run to see insertion and search in action.

Usage (zsh):
  # Optional environment (OpenAI or HF)
  # export EMBEDDING_PROVIDER=openai
  # export EMBEDDING_MODEL=text-embedding-3-large
  # export OPENAI_API_KEY=sk-...
  # Or local:
  # export EMBEDDING_PROVIDER=hf
  # export EMBEDDING_MODEL=intfloat/e5-large-v2

  python -m Agent_to_Skills.skilltree_demo
"""
from __future__ import annotations

import os
import sys

try:
    from Agent_to_Skills.SkillTree import (
        skill_tree,
        initialize_skill_tree,
        add_text,
        search,
    )
except Exception:
    # Fallback when running as a plain script from this folder
    sys.path.append(os.path.dirname(__file__))
    from SkillTree import (
        skill_tree,
        initialize_skill_tree,
        add_text,
        search,
    )


def main():
    # Choose a default local embedding if no OpenAI/Azure keys and user didn't set provider
    if not os.getenv("EMBEDDING_PROVIDER"):
        if not (os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")):
            os.environ["EMBEDDING_PROVIDER"] = "hf"
            os.environ["EMBEDDING_MODEL"] = os.getenv("EMBEDDING_MODEL", "intfloat/e5-large-v2")

    initialize_skill_tree("root")

    add_text("Write a Python function to parse JSON", time_stamp=1)
    add_text("Implement JSON parsing in Python using json.loads", time_stamp=2)
    add_text("Calculate cosine similarity between two vectors", time_stamp=3)
    add_text("Parse XML files in Python", time_stamp=4)

    q = "How to parse JSON text in Python?"
    path, children = search(q)

    print("Query:", q)
    print("Path to best node:")
    for i, n in enumerate(path):
        snippet = n.string[:80].replace("\n", " ")
        print(f"  {i}. {snippet}")

    print("Children of best node:")
    for i, ch in enumerate(children):
        snippet = ch.string[:80].replace("\n", " ")
        print(f"  - {snippet}")


if __name__ == "__main__":
    main()
