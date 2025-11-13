"""
Offline, rule-based skill extractor for dialogs (Chinese/English).

Inputs:
- turns: List[Tuple[str, str]] where each item is (utterance_text, role_tag)
  role_tag: "user" | "assistant" | or free-form labels (e.g., "knowledge", "question").

Outputs:
- summary: Dict with Task/Goal/Steps/Knowledge/KeyInsight (best-effort rule fill)
- skills: List[Dict] items with fields: name, description, evidence_text, tags

This is a lightweight fallback when LLM APIs are not available.
"""
from __future__ import annotations
from typing import List, Tuple, Dict, Any
import re

ZH_QUESTION_PAT = re.compile(r"(如何|怎么|怎样|怎么办|可以.*吗|需要.*吗)")
ZH_ACTION_PATTERNS = [
    (re.compile(r"(实现|使用|写|配置|安装|打开|关闭|申请|加入|预定|预约)\s*([\w\u4e00-\u9fa5_]+)"), (1, 2)),
]
EN_ACTION_PATTERNS = [
    (re.compile(r"(implement|use|write|configure|install|open|close|apply|join|reserve)\s+([\w-]+)", re.I), (1, 2)),
]


def _extract_candidates(text: str) -> List[str]:
    cands: List[str] = []
    # Chinese action-object
    for pat, idxs in ZH_ACTION_PATTERNS:
        for m in pat.finditer(text):
            verb = m.group(idxs[0])
            obj = m.group(idxs[1])
            cands.append(f"{verb}{obj}")
    # English action-object
    for pat, idxs in EN_ACTION_PATTERNS:
        for m in pat.finditer(text):
            verb = m.group(idxs[0]).lower()
            obj = m.group(idxs[1])
            cands.append(f"{verb} {obj}")
    # Questions like "如何实现多态" -> "实现多态"
    m = ZH_QUESTION_PAT.search(text)
    if m:
        # crude extraction of following words after question lead
        # e.g., "如何 实现 多态" -> candidate: 实现多态
        seg = re.sub(r"[？?。！!]", " ", text)
        seg = seg.replace("如何", " ").replace("怎么", " ")
        seg = seg.strip()
        if seg:
            seg = re.sub(r"\s+", " ", seg)
            tokens = seg.split(" ")
            if len(tokens) >= 2:
                cands.append("".join(tokens[:2]))
    # Dedup while keeping order
    seen = set()
    out = []
    for c in cands:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def extract_from_dialog(turns: List[Tuple[str, str]]) -> Dict[str, Any]:
    # Heuristic summary
    task = None
    goal = None
    steps: List[str] = []
    knowledge: List[str] = []
    key_insight: List[str] = []
    skills: List[Dict[str, Any]] = []

    # Task/Goal guess: first user question and last assistant knowledge snippet
    for t, role in turns:
        if not task and (role in ("user", "question") or "如何" in t or "怎么" in t):
            task = re.sub(r"^[\w\u4e00-\u9fa5]+[:：]\s*", "", t).strip()
        if role in ("assistant", "knowledge"):
            knowledge.append(re.sub(r"^[\w\u4e00-\u9fa5]+[:：]\s*", "", t).strip())

    if knowledge:
        goal = "掌握关键信息/操作以完成任务"
        # Heuristic steps: up to 3 knowledge lines
        steps = [k for k in knowledge[:3]]
        key_insight = [knowledge[0]]

    # Skill candidates from all utterances
    cand_set = []
    for t, _ in turns:
        clean = re.sub(r"^[\w\u4e00-\u9fa5]+[:：]\s*", "", t).strip()
        cand_set.extend(_extract_candidates(clean))

    for c in cand_set[:5]:
        skills.append({
            "name": c,
            "description": f"围绕“{c}”的可复用方法/知识片段",
            "evidence_text": c,
            "tags": ["dialog"]
        })

    return {
        "summary": {
            "Task": task or "从对话中抽取技能并构建技能树",
            "Goal": goal or "抽取可复用技能片段",
            "Steps": steps,
            "Knowledge": knowledge,
            "KeyInsight": key_insight,
        },
        "skills": skills,
    }


if __name__ == "__main__":
    demo = [
        ("user: 我想学习 C++ 的类和继承", "user"),
        ("assistant: 先掌握 class 的基本语法", "assistant"),
        ("user: 如何实现多态？", "user"),
        ("assistant: 基类 virtual + 子类 override", "assistant"),
    ]
    out = extract_from_dialog(demo)
    from pprint import pprint
    pprint(out)
