from __future__ import annotations

import json
import os
import sys
import tempfile

TESTS_DIR = os.path.dirname(__file__)
WORKSPACE_ROOT = os.path.abspath(os.path.join(TESTS_DIR, "..", ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from skilltree_core.schema import SkillTree, make_skill
from skilltree_core.runtime import UpdateOp, apply_updates, retrieve_for_query, retrieve_semantic, schedule_to_target
from skilltree_core.dialogue_export import (
    export_tree_from_dialogue,
    write_skill_tree_nested_json,
    export_dialogue_tree_to_json,
)

COURSE_DIALOGUE_SNIPPETS = [
    {
        "text": "Here’s the latest: Mental Health & Development is almost full, so I recommend giving it your S-Pass.",
        "expected": "skill.s_pass_mental_health",
    },
    {
        "text": "We’ll slide Linear Algebra down to an A-Pass; demand’s high but not critical.",
        "expected": "skill.linear_algebra_a_pass",
    },
    {
        "text": "Military Theory is compulsory and heating up, so I’d bump it to the other A-Pass.",
        "expected": "skill.military_theory_a_pass",
    },
    {
        "text": "Math Analysis, Programming, and Programming for Everyone can move to B-Passes; Innovation & Entrepreneurship up to an A.",
        "expected": "skill.manage_core_b_passes",
    },
    {
        "text": "Innovation & Entrepreneurship is trending, so let’s promote it to the elective A-Pass.",
        "expected": "skill.innovation_a_pass",
    },
]


def test_roundtrip():
    root = make_skill(id="skill.root", name="Root", description="Root skill")
    tree = SkillTree(root_skills=[root])
    data = tree.to_json()
    clone = SkillTree.from_json(data)
    assert clone.get("skill.root") is not None


def test_crud_merge():
    tree = SkillTree()
    a = make_skill(id="skill.a", name="A", description="A desc", confidence=0.5)
    b = make_skill(id="skill.b", name="B", description="B desc", confidence=0.8)
    tree.add_skill(a, as_root=True)
    tree.add_skill(b)
    tree.merge_skills("skill.b", "skill.a")
    assert tree.get("skill.a") is None and tree.get("skill.b") is not None


def test_updates_and_retrieval():
    tree = SkillTree()
    ops = [
        UpdateOp(
            op="ADD",
            skill_id="skill.x",
            payload={
                "id": "skill.x",
                "name": "X",
                "description": "X desc",
                "triggers": [{"type": "intent", "value": "do_x", "confidence": 0.9}],
            },
        ),
        UpdateOp(
            op="ADD",
            skill_id="skill.y",
            payload={
                "id": "skill.y",
                "name": "Y",
                "description": "Y desc",
                "preconditions": ["skill.x"],
            },
        ),
    ]
    apply_updates(tree, ops)
    hits = retrieve_for_query(tree, "intent:do_x please")
    assert hits and hits[0].id == "skill.x"
    chain = schedule_to_target(tree, "skill.y")
    assert chain[-1] == "skill.y" and chain[0] == "skill.x"


def test_semantic_retrieval():
    tree = SkillTree()
    a = make_skill(
        id="skill.context_moe",
        name="Extend context via MoE",
        description="Research long-context mixture-of-experts routing techniques",
    )
    b = make_skill(
        id="skill.quantization",
        name="LLM quantization",
        description="Explore 4-bit quantization recipes for deployment",
    )
    tree.add_skill(a, as_root=True)
    tree.add_skill(b)
    results = retrieve_semantic(tree, "mixture of experts long context routing", top_k=2)
    assert results and results[0][0].id == "skill.context_moe"


def test_course_selection_dialogue_alignment():
    tree = SkillTree()
    skills = [
        make_skill(
            id="skill.s_pass_mental_health",
            name="Prioritize Mental Health",
            description="Assign the S-Pass to Mental Health & Development due to near-capacity popularity.",
        ),
        make_skill(
            id="skill.linear_algebra_a_pass",
            name="Linear Algebra A-Pass",
            description="Reassign Linear Algebra from S-Pass to A-Pass while preserving seat security.",
        ),
        make_skill(
            id="skill.military_theory_a_pass",
            name="Military Theory A-Pass",
            description="Upgrade Military Theory to an A-Pass because compulsory demand is spiking.",
        ),
        make_skill(
            id="skill.manage_core_b_passes",
            name="Downgrade Core Courses",
            description="Move Mathematical Analysis and Programming tracks to B-Passes to balance workload.",
        ),
        make_skill(
            id="skill.innovation_a_pass",
            name="Innovation Elective Upgrade",
            description="Promote Innovation & Entrepreneurship to the elective A-Pass slot after trend updates.",
        ),
    ]
    for idx, skill in enumerate(skills):
        tree.add_skill(skill, as_root=(idx == 0))

    for snippet in COURSE_DIALOGUE_SNIPPETS:
        results = retrieve_semantic(tree, snippet["text"], top_k=1, min_score=0.05)
        assert results, f"No semantic hits for snippet: {snippet['text']}"
        assert results[0][0].id == snippet["expected"], (
            f"Expected {snippet['expected']} but got {results[0][0].id}"
        )


def test_nested_json_export():
    tree = SkillTree()
    parent = make_skill(
        id="skill.parent",
        name="Parent",
        description="Parent skill",
        subskills=["skill.child"],
    )
    child = make_skill(
        id="skill.child",
        name="Child",
        description="Child skill",
        preconditions=["skill.parent"],
    )
    tree.add_skill(parent, as_root=True)
    tree.add_skill(child)
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "nested.json")
        data = write_skill_tree_nested_json(tree, out_path, root_id="skill.parent")
        assert os.path.exists(out_path)
        with open(out_path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        assert loaded["root"]["id"] == "skill.parent"
        assert loaded == data
        assert loaded["root"]["children"][0]["id"] == "skill.child"


def test_export_dialogue_tree_to_json():
    snippets = [
        "Let’s focus on long-context routing and MoE research",
        "Also schedule time for practical quantization recipes",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "dialogue_tree.json")
        data = export_dialogue_tree_to_json(snippets, out_path)
        assert os.path.exists(out_path)
        with open(out_path, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        assert loaded == data
        assert loaded["root"]["id"].startswith("skill")
        assert loaded["root"]["children"]


def run_all():
    test_roundtrip()
    test_crud_merge()
    test_updates_and_retrieval()
    test_semantic_retrieval()
    test_course_selection_dialogue_alignment()
    dialogue_tree = export_tree_from_dialogue([snippet["text"] for snippet in COURSE_DIALOGUE_SNIPPETS])
    assert dialogue_tree.root_skills, "export_tree_from_dialogue should produce at least one root"
    test_nested_json_export()
    test_export_dialogue_tree_to_json()
    print("All tests passed")


if __name__ == "__main__":
    run_all()
