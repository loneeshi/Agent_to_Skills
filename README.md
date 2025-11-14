# SkillTree Engine (Schema v1.0)

This directory now implements a production-oriented SkillTree schema and runtime engine for agent skill acquisition, retrieval, scheduling, and maintenance. It replaces the earlier toy prototype that used random embeddings and ad‑hoc insertion logic.

## Core Concepts

Each skill captures:
- Identity & description (`id`, `name`, `description`)
- Activation signals (`triggers` – intent/keyword/pattern/state)
- Success signals (`success_conditions` – task completion / confirmation / state)
- Graph relations (`preconditions`, `postconditions`, `subskills`) forming a hybrid tree + DAG
- Operational metadata (timestamps, example turns, provenance, merge history)
- Confidence (0–1) representing mastery / reliability

The full tree is a JSON document following the stable schema:

```json
{
  "skill_tree_version": "1.0",
  "root_skills": [ { /* Skill objects */ } ]
}
```

See `skilltree_engine/schema.py` for dataclass definitions and (de)serialization.

## What’s New vs. Prototype

| Feature | Prototype | Engine |
|---------|-----------|--------|
| Deterministic structure | Order-dependent clustering | Explicit graph references |
| Skill schema richness | Minimal text nodes | Full triggers/success/DAG/meta/confidence |
| CRUD operations | Implicit via insertion | First-class add/update/delete/merge |
| Validation | None | Cycle + missing ref detection |
| Retrieval | Cosine over toy vectors | Intent & keyword matching (extensible) |
| Scheduling | N/A | Topological ordering via preconditions |
| Gap detection | N/A | Referenced-but-missing skill suggestions |

## Modules

- `skilltree_engine/schema.py` – Schema dataclasses, JSON round‑trip, validation helpers.
- `skilltree_engine/runtime.py` – High-level operations (ingestion, updates, retrieval, scheduling, gap suggestions).
- `skilltree_engine/semantic.py` – Lightweight bag-of-words embedding index + semantic search helpers.
- `skilltree_engine/decompose_skills.py` – Heuristic decomposition script to expand broader tasks into atomic children.
- `skilltree_engine/extraction_prompt.txt` – Prompt template for LLM skill extraction & generation.
- `skilltree_engine/pipeline_demo.py` – End‑to‑end mock pipeline: dialogue → extraction output → ingestion → updates → retrieval & scheduling.
- `skilltree_engine/test_skilltree.py` – Basic unit tests (round‑trip, CRUD/merge, retrieval, scheduling).

## Quick Start

Run the demo pipeline:

```bash
cd design/Agent_to_Skills
python -m skilltree_engine.pipeline_demo
```

Run tests:

```bash
python -m skilltree_engine.test_skilltree
```

## Dialogue-to-Tree Export

Turn free-form dialogue snippets into a nested SkillTree JSON (matching the
`outputs/skilltree/*.json` structure) with the new helper:

```bash
cd design/Agent_to_Skills
python tools/dialogue_to_skilltree.py path/to/dialogue_snippets.txt --output outputs/skilltree/my_dialogue_tree.json
```

The script calls `skilltree_core.dialogue_export.export_dialogue_tree_to_json`,
which learns skills from the snippets, optionally attaches them under a parent
skill, and writes the nested representation (with `children` arrays) so it can be
consumed by downstream planners or visualization tools.

Load a SkillTree from JSON:

```python
from skilltree_engine.schema import SkillTree
st = SkillTree.from_json(open("path/to/tree.json", "r", encoding="utf-8").read())
print(st.validate())
```

Apply updates:

```python
from skilltree_engine.runtime import UpdateOp, apply_updates
ops = [UpdateOp(op="ADD", skill_id="skill.parse", payload={
  "id": "skill.parse", "name": "Parse Log", "description": "Parse raw behavior log", "confidence": 0.6
})]
apply_updates(st, ops)
```

Retrieve and schedule:

```python
from skilltree_engine.runtime import retrieve_for_query, schedule_to_target
hits = retrieve_for_query(st, "intent:analysis user metrics")
chain = schedule_to_target(st, "skill.generate_report")
```

## Validation & Integrity

`SkillTree.validate()` checks:
1. Missing references (pre/post/subskills)
2. Cycles via DFS over directed edges (postconditions + subskills)

Use `gap_skills()` and `suggest_missing_skills()` to patch holes before execution.

### Semantic Retrieval

Build a semantic index and search by description-level intent:

```python
from skilltree_engine.semantic import semantic_search
hits = semantic_search(st, "mixture-of-experts long context routing", top_k=5)
for skill, score in hits:
  print(skill.id, score)
```

The runtime convenience wrapper `retrieve_semantic` (see `runtime.py`) mirrors this API and is used in the pipeline demos.

### Decomposition Script

Generate additional subskills from descriptive fragments (useful for making second/third layers):

```bash
python -m skilltree_engine.decompose_skills outputs/skilltree/llm_literature_review_nested.json
```

This writes `*_decomposed.json` alongside the input and runs a validation pass. Adjust heuristics or caps in `decompose_skills.py` to suit your data.

## Extending Retrieval

Lexical + semantic hybrid retrieval now exists (intent token matcher + bag-of-words cosine). Upcoming improvements:
1. Add contextual filters (confidence thresholds, recent usage decay).
2. Support external vector stores (e.g., FAISS/Chroma) for large graphs.
3. Blend retrieval results with reinforcement signals (success/failure).

## Planned Enhancements

1. Confidence adaptation policies (success/failure reinforcement).
2. Skill versioning & provenance chains.
3. Multi-root scenario support (parallel high-level tasks).
4. Persistent store (SQLite / LiteLLM / Chroma) for large trees.
5. Graph-based semantic traversal (success probability / cost heuristics).

## Migration Notes

Removed legacy prototype files (`skilltree_extended.py`, `prompt.py`, old clustering notebooks) in favor of unified engine. If you need historical clustering-based skill generation, archive those scripts separately before pruning.

## License

TBD – add an open-source license (e.g., MIT) if distributing.

## Attribution

Skill schema & prompt aligned with user-provided production specification (2025-11-14).

