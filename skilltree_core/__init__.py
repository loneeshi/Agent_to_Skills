"""Lightweight SkillTree core module.

This package replaces the legacy `skilltree_engine` by bundling schema
objects, runtime helpers, semantic retrieval, and decomposition utilities
under a single namespace:

    from skilltree_core import SkillTree, make_skill, UpdateOp

"""
from .schema import (
    SkillTree,
    Skill,
    Trigger,
    SuccessCondition,
    make_skill,
)
from .runtime import (
    UpdateOp,
    ingest_skill_tree_json,
    apply_updates,
    retrieve_for_query,
    retrieve_semantic,
    suggest_missing_skills,
    schedule_to_target,
)
from .semantic import semantic_search, SemanticRetriever
from .decompose import (
    decompose,
    load_skill_tree,
    export_nested,
)
from .dialogue_export import (
    DialogueSnippet,
    LearnedSkill,
    learn_skills_from_dialogue,
    export_tree_from_dialogue,
)

__all__ = [
    "SkillTree",
    "Skill",
    "Trigger",
    "SuccessCondition",
    "make_skill",
    "UpdateOp",
    "ingest_skill_tree_json",
    "apply_updates",
    "retrieve_for_query",
    "retrieve_semantic",
    "semantic_search",
    "SemanticRetriever",
    "suggest_missing_skills",
    "schedule_to_target",
    "decompose",
    "load_skill_tree",
    "export_nested",
    "DialogueSnippet",
    "LearnedSkill",
    "learn_skills_from_dialogue",
    "export_tree_from_dialogue",
]
