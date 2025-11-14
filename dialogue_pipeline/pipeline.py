"""Task JSON → Dialogue outline pipeline.

Usage:
    python pipeline.py --task-id course_selection_002 \
        --tasks-json ../../benchmarks/ELL-StuLife/task_data/tasks.json \
        --background ../../benchmarks/ELL-StuLife/task_data/background \
        --out artifacts/course_selection_002_dialogues.json
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_TASKS_PATH = Path("../../benchmarks/ELL-StuLife/task_data/tasks.json").resolve()
DEFAULT_BACKGROUND_DIR = Path("../../benchmarks/ELL-StuLife/task_data/background").resolve()
DEFAULT_TEMPLATE = Path("templates/course_selection.json")

PASS_BUDGET_DEFAULTS = {
    "compulsory": {"S": 1, "A": 2, "B": "unlimited"},
    "elective": {"A": 1, "B": "unlimited"},
}

TASK_OVERRIDES = {
    "course_selection_002": {
        "goal": "Adjust first-semester course passes to reflect popularity shifts while balancing workload and wellness.",
        "edge_cases": [
            "Student asks for popularity numbers before agreeing to changes.",
            "Student worries about wellness overload if Mental Health stays S-Pass.",
            "Student resists downgrading math/programming courses and negotiates priorities.",
            "Student double-checks pass limits or wants backlog simulation before confirming.",
        ],
    }
}


@dataclass
class IntentSpec:
    id: str
    speaker: str
    summary: str
    required_slots: List[str]
    branch: Optional[str]
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        return data


@dataclass
class DialogueTurn:
    speaker: str
    intent: str
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DialoguePipeline:
    def __init__(self, tasks_json: Path, background_dir: Path, template_path: Path):
        self.tasks_json = tasks_json
        self.background_dir = background_dir
        self.template_path = template_path
        self.tasks = self._load_json(tasks_json)
        self.background = self._load_background(background_dir)
        self.template = self._load_json(template_path)
        self.courses_index = self._build_course_index()

    @staticmethod
    def _load_json(path: Path) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_background(self, background_dir: Path) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if not background_dir.exists():
            return data
        for file in background_dir.glob("*.json"):
            try:
                data[file.stem] = self._load_json(file)
            except json.JSONDecodeError:
                continue
        return data

    def _build_course_index(self) -> Dict[str, Dict[str, Any]]:
        courses_index: Dict[str, Dict[str, Any]] = {}
        courses_blob = self.background.get("courses", {})
        for course in courses_blob.get("courses", []):
            courses_index[course["course_code"]] = course
        return courses_index

    def run(self, task_id: str) -> Dict[str, Any]:
        task = self._find_task(task_id)
        schema = self._build_schema(task_id, task)
        intents = self._build_intents(schema)
        dialogues = self._synthesize_dialogues(schema)
        prompts = self._build_prompts(schema, intents)
        return {
            "task_id": task_id,
            "schema": schema,
            "intent_plan": [intent.to_dict() for intent in intents],
            "dialogues": [[turn.to_dict() for turn in dialogue] for dialogue in dialogues],
            "prompts": prompts,
        }

    def _find_task(self, task_id: str) -> Dict[str, Any]:
        for key, value in self.tasks.items():
            if value.get("task_id") == task_id:
                return value
        raise KeyError(f"Task {task_id} not found in {self.tasks_json}")

    def _build_schema(self, task_id: str, task: Dict[str, Any]) -> Dict[str, Any]:
        overrides = TASK_OVERRIDES.get(task_id, {})
        instruction = task.get("instruction", "").strip()
        goal = overrides.get("goal") or self._infer_goal(instruction)
        roles = ["Senior advisor", "Student"]
        course_slots = self.template.get("course_slots", {})
        courses = {}
        for slot, code in course_slots.items():
            meta = self.courses_index.get(code, {})
            courses[slot] = {
                "code": code,
                "name": meta.get("course_name", slot.replace("_", " ").title()),
                "type": meta.get("type"),
                "credits": meta.get("credits"),
            }
        popularity = self._summarize_popularity(task.get("world_state_change", []), course_slots)
        constraints = {
            "compulsory": "S:1, A:2, B:unlimited",
            "elective": "A:1, B:unlimited",
            "note": "Mental Health nearly full; others moderate demand",
        }
        schema = {
            "goal": goal,
            "roles": roles,
            "pass_budgets": PASS_BUDGET_DEFAULTS,
            "courses": courses,
            "popularity_highlights": popularity,
            "constraints": constraints,
            "edge_cases": overrides.get("edge_cases", []),
        }
        return schema

    def _infer_goal(self, instruction: str) -> str:
        if not instruction:
            return "Adjust course passes"
        first_sentence = instruction.split(".\n")[0].split(".")[0]
        return first_sentence.strip()

    def _summarize_popularity(self, changes: List[Dict[str, Any]], course_slots: Dict[str, str]) -> List[Dict[str, Any]]:
        summary: Dict[str, Dict[str, Any]] = {}
        for change in changes or []:
            if change.get("change_type") != "popularity_update":
                continue
            code = change.get("course_code")
            if code not in course_slots.values():
                continue
            summary[code] = {
                "course_code": code,
                "latest_popularity": change.get("new_value"),
            }
        enriched = []
        for slot, code in course_slots.items():
            entry = summary.get(code, {"course_code": code, "latest_popularity": "?"})
            entry["slot"] = slot
            entry["course_name"] = self.courses_index.get(code, {}).get("course_name")
            enriched.append(entry)
        return enriched

    def _build_intents(self, schema: Dict[str, Any]) -> List[IntentSpec]:
        intents: List[IntentSpec] = []
        for intent_info in self.template.get("intent_chain", []):
            context = {
                "goal": schema.get("goal"),
                "popularity_highlights": schema.get("popularity_highlights"),
                "courses": schema.get("courses"),
                "pass_budgets": schema.get("pass_budgets"),
            }
            intents.append(IntentSpec(
                id=intent_info["id"],
                speaker=intent_info["speaker"],
                summary=intent_info["summary"],
                required_slots=intent_info.get("required_slots", []),
                branch=intent_info.get("branch"),
                context=context,
            ))
        return intents

    def _synthesize_dialogues(self, schema: Dict[str, Any]) -> List[List[DialogueTurn]]:
        mapping = self._default_mapping(schema)
        highlights = ", ".join(
            f"{entry.get('course_name', entry['course_code'])} ({entry['latest_popularity']})"
            for entry in schema.get("popularity_highlights", [])
            if entry.get("latest_popularity") is not None
        )
        render_ctx = {
            "goal": schema.get("goal"),
            "popularity_highlights": highlights,
            "mapping_summary": self._mapping_summary(mapping, schema.get("courses", {})),
        }
        system_templates = self.template.get("utterance_templates", {}).get("system", {})
        user_templates = self.template.get("utterance_templates", {}).get("user", {})
        dialogue: List[DialogueTurn] = []
        def add_turn(speaker: str, intent: str, template_key: str):
            template_bank = system_templates if speaker == "system" else user_templates
            template = template_bank.get(template_key)
            if not template:
                return
            dialogue.append(DialogueTurn(
                speaker=speaker,
                intent=intent,
                text=self._render_template(template, render_ctx),
            ))
        add_turn("system", "opening", "opening")
        add_turn("user", "opening_ack", "ack")
        add_turn("system", "s_pass_assignment", "s_pass_assignment")
        add_turn("user", "s_pass_ack", "ack")
        add_turn("system", "compulsory_rebalance", "compulsory_rebalance")
        add_turn("user", "compulsory_ack", "ack")
        add_turn("system", "downgrade_core", "downgrade_core")
        add_turn("user", "concern_math", "concern_math")
        add_turn("system", "elective_upgrade", "elective_upgrade")
        add_turn("user", "swap_request", "swap_request")
        add_turn("system", "validation", "validation")
        add_turn("user", "final_ack", "ack")
        add_turn("system", "closure", "closure")
        return [dialogue]

    def _default_mapping(self, schema: Dict[str, Any]) -> Dict[str, str]:
        return {
            "mental_health": "S-Pass",
            "linear_algebra": "A-Pass",
            "military_theory": "A-Pass",
            "math_analysis": "B-Pass",
            "programming": "B-Pass",
            "programming_for_everyone": "B-Pass",
            "innovation": "A-Pass",
        }

    def _mapping_summary(self, mapping: Dict[str, str], courses: Dict[str, Dict[str, Any]]) -> str:
        parts = []
        for slot, pass_type in mapping.items():
            course = courses.get(slot, {})
            name = course.get("name") or slot.replace("_", " ").title()
            parts.append(f"{name} → {pass_type}")
        return "; ".join(parts)

    def _render_template(self, template: str, ctx: Dict[str, Any]) -> str:
        pattern = re.compile(r"{{\s*([\w\.]+)\s*}}")
        def repl(match):
            key = match.group(1)
            if key in ctx:
                return str(ctx[key])
            return match.group(0)
        return pattern.sub(repl, template)

    def _build_prompts(self, schema: Dict[str, Any], intents: List[IntentSpec]) -> Dict[str, str]:
        prompts = {}
        scaffolds = self.template.get("prompt_scaffolds", {})
        for key, template in scaffolds.items():
            ctx = {
                "schema": json.dumps(schema, ensure_ascii=False, indent=2),
                "intent_id": intents[0].id if intents else "opening",
                "summary": intents[0].summary if intents else "",
                "slots": intents[0].required_slots if intents else [],
                "speaker": intents[0].speaker if intents else "system",
                "utterance": "",
            }
            prompts[key] = self._render_template(template, ctx)
        return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert task JSON entries into dialogue scaffolds.")
    parser.add_argument("--task-id", required=True, help="task_id to process (e.g., course_selection_002)")
    parser.add_argument("--tasks-json", type=Path, default=DEFAULT_TASKS_PATH, help="Path to tasks.json")
    parser.add_argument("--background", type=Path, default=DEFAULT_BACKGROUND_DIR, help="Directory with background JSON files")
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE, help="Template JSON defining intents/utterances")
    parser.add_argument("--out", type=Path, required=False, help="Where to write the dialogue artifact (JSON)")
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline = DialoguePipeline(args.tasks_json, args.background, args.template)
    artifact = pipeline.run(args.task_id)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(artifact, f, ensure_ascii=False, indent=2)
        print(f"Saved dialogue artifact to {args.out}")
    else:
        print(json.dumps(artifact, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
