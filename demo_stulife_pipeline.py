"""
Demo pipeline for StuLife-style evaluation without any API calls.

Stages:
1) Take a tiny dialog (turns) as input
2) Extract candidate skills via rule-based extractor
3) Insert skills into an extended SkillTree (deterministic toy embeddings)
4) Run a search query and print top hits
5) Serialize the tree (preview first N chars)
"""
from __future__ import annotations
from typing import List, Tuple
from skilltree_extended import SkillTree, Evidence
from skill_extractor import extract_from_dialog


def run_demo():
    turns: List[Tuple[str, str]] = [
        ("user: 我想加入人工智能社团，怎么申请？", "user"),
        ("assistant: 开学初进行招新，发送简历到 artificial_intelligence_innovation_society@lau.edu", "assistant"),
        ("user: 我还想了解图书馆座位预定怎么操作？", "user"),
        ("assistant: 通过校园App预约，需在30分钟内到场签到", "assistant"),
    ]

    # 1) Extract skills
    result = extract_from_dialog(turns)
    skills = result["skills"]
    print("抽取技能候选:")
    for s in skills:
        print(" -", s["name"]) 

    # 2) Build tree and insert
    st = SkillTree()
    for s in skills:
        ev = Evidence(text=s.get("evidence_text", ""), source="dialog")
        st.insert(text=s["description"], skill_name=s["name"], tags=s.get("tags", []), evidence=ev)

    # 3) Search examples
    q = "怎么申请人工智能社团"
    print("\n检索:", q)
    hits = st.search(q, top_k=3)
    for i, (node, score) in enumerate(hits, 1):
        print(f"Top{i} score={score:.4f} name={node.skill_name} text={node.text[:60]}")

    # 4) Serialize (preview)
    print("\n树序列化预览:")
    data = st.serialize()
    print(data[:800])


if __name__ == "__main__":
    run_demo()
