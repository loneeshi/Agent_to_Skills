# Dialogue Generation Pipeline for Task JSON Assets

## Overview
This module formalizes how we convert structured campus tasks (from `benchmarks/ELL-StuLife/task_data`) into multi-turn dialogues suitable for agent training or evaluation. It packages the manual blueprint we followed for `course_selection_002` into a reproducible workflow with:

- **Task schema extraction** – parse `tasks.json` and optional background files (e.g., `courses.json`) into a machine-usable state snapshot.
- **Intent planning** – derive an ordered set of conversational intents/slots that reflect task constraints, required confirmations, and edge cases.
- **Utterance synthesis scaffolding** – generate dialogue outlines or fully rendered turns via lightweight templates.

> Goal: make it trivial to go from any entry in `task_data/tasks.json` to a consistent set of dialogues, while keeping hooks for LLM rewriting/paraphrasing.

## Directory Layout
```
dialogue_pipeline/
├── README.md                # This document (process, usage, tips)
├── pipeline.py              # Executable module for schema→intent→dialogue flow
└── templates/
    └── course_selection.json # Intent + utterance templates (extensible)
```

## Data Sources
| Source | Purpose |
| --- | --- |
| `benchmarks/ELL-StuLife/task_data/tasks.json` | Primary task definitions (instructions, constraints, world state). |
| `benchmarks/ELL-StuLife/task_data/background/courses.json` | Course metadata (names, categories) for slot enrichment. |
| `benchmarks/ELL-StuLife/task_data/config/*.json` | Optional limits (pass counts, schedule windows) to build validation rules. |

The pipeline can ingest any additional JSON background file; simply point the CLI to the directory and the loader will attach the contents to the task context.

## Pipeline Steps
1. **Load Task Context**
   - Locate the task by `task_id` (e.g., `course_selection_002`).
   - Merge optional world-state deltas (e.g., popularity updates).
2. **Build Schema Snapshot**
   - Normalize key slots (courses, passes, seat counts) into structured fields.
   - Record constraints (e.g., compulsory vs. elective pass budgets, edge cases).
3. **Plan Intents**
   - Use template definitions per `task_type` to emit an ordered intent chain (opening → conflict handling → validation → closure).
   - Each intent documents: actor, purpose, required slots, branching notes, and example prompts.
4. **Synthesize Dialogues**
   - Expand intents into concrete utterances via deterministic templates + contextual variables.
   - Optionally hand off to an LLM (not included) for paraphrasing using the generated prompt stubs.
5. **Export**
   - Save JSON containing schema snapshot, intent chain, dialogue samples, and prompt seeds.

## Using the Pipeline
```bash
python pipeline.py \
  --tasks-json ../../benchmarks/ELL-StuLife/task_data/tasks.json \
  --background ../../benchmarks/ELL-StuLife/task_data/background \
  --task-id course_selection_002 \
  --out artifacts/course_selection_002_dialogues.json
```

The output file includes:
- `schema`: normalized goal/roles/slots.
- `intent_plan`: ordered intents with branching metadata.
- `dialogues`: synthesized multi-turn samples (seeded by provided templates).
- `prompts`: helper prompts for paraphrasing and tone shifts.

## Extending to Other Task Types
1. Create a new template file under `templates/` (e.g., `campus_exploration.json`).
2. Define `schema_fields`, `intents`, and `utterance_templates` sections.
3. Call `pipeline.py --template templates/campus_exploration.json --task-id campus_exploration_033_trigger`.

## Notes & Next Steps
- Current utterance templates are minimal (plain text). Hook up to your preferred LLM for richer language.
- The code emits TODO markers when required data is missing (e.g., seat counts). Fill these via additional background sources or manual overrides.
- Future improvement: integrate directly with `skilltree_engine` to auto-generate skill updates from dialogue plans.
