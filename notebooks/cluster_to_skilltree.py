#!/usr/bin/env python3
"""
Convert clustering output (clusters_with_descriptors.json) into a SkillTree JSON.

- Uses skilltree_extended.SkillTree to build a light-weight tree
- Two modes:
  1) flat: put each cluster as a direct child of root (deterministic, no merging)
  2) insert: use similarity-based insert to allow merging similar clusters

Usage:
  python cluster_to_skilltree.py \
    --clusters outputs/cluster/clusters_with_descriptors.json \
    --out outputs/cluster/skilltree_from_clusters.json \
    --mode flat \
    --max-clusters 500 \
    --exclude-noise

Notes:
- Cluster id "-1" is considered noise; pass --exclude-noise to skip it.
- We store cluster descriptor and top_terms in the node text/description, and add representative/examples as evidence.
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path
import sys
from typing import Dict, Any

# Ensure project root (one level up from notebooks/) is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skilltree_extended import SkillTree, Node, Evidence


def _contains_any(text: str, keys):
    t = (text or "").lower()
    return any(k in t for k in keys)


def _looks_like_course_code(s: str) -> bool:
    import re
    return bool(re.search(r"\b([A-Z]{2,6}-[A-Z]{2,6}-\d{3}|[A-Z]{2,6}\d{3,}(?:\(\d+\))?)\b", s))


def classify_domain(cluster: dict) -> str:
    terms = [str(t).lower() for t in cluster.get("top_terms", [])]
    rep = str(cluster.get("representative", "")).lower()
    text_blob = " ".join(terms + [rep])
    # advisors / people
    if "advisor_id" in text_blob or _contains_any(text_blob, ["research_area", "advisor", "t0"]) and "@lau.edu" in text_blob:
        return "advisors"
    # courses / curriculum
    if any(_looks_like_course_code(t) for t in terms) or _contains_any(text_blob, ["ge-", "coms", "course", "credit"]):
        return "courses"
    # policy / handbook / chapters
    if _contains_any(text_blob, ["chap_", "sec_", "art_", "policy", "regulation", "chapter", "section", "article"]):
        return "policy"
    # library
    if _contains_any(text_blob, ["library", "seat", "reservation", "loan", "book", "isbn"]):
        return "library"
    # clubs
    if _contains_any(text_blob, ["club", "society", "association", "recruitment"]):
        return "clubs"
    # map / campus map
    if _contains_any(text_blob, ["building", "zone", "quad", "hall", "pavilion", "library", "physics"]):
        return "map"
    return "misc"


def load_clusters(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_tree_from_clusters(clusters: Dict[str, Any], *, mode: str = "flat", max_clusters: int = 500, exclude_noise: bool = False) -> SkillTree:
    # Sort clusters by size desc if available
    def _size_of(c):
        return int(c.get("size", 0))

    items = []
    for cid, c in clusters.items():
        try:
            cid_int = int(cid)
        except Exception:
            cid_int = None
        if exclude_noise and cid_int == -1:
            continue
        items.append((cid, c))

    items.sort(key=lambda kv: -_size_of(kv[1]))
    if max_clusters > 0:
        items = items[:max_clusters]

    st = SkillTree("document clusters")
    domain_nodes: dict[str, Node] = {}

    for cid, c in items:
        cid_str = str(cid)
        desc = c.get("descriptor") or c.get("representative") or f"Cluster {cid_str}"
        top_terms = c.get("top_terms", [])
        rep = c.get("representative")
        examples = c.get("examples", [])
        skill_name = f"cluster_{cid_str}"
        description = ", ".join(top_terms[:6]) if top_terms else None
        # Evidence: representative + up to 2 examples
        evs = []
        if rep:
            evs.append(Evidence(text=str(rep), source=f"cluster:{cid_str}"))
        for ex in examples[:2]:
            evs.append(Evidence(text=str(ex), source=f"cluster:{cid_str}"))

        if mode == "insert":
            node = st.insert(text=desc, skill_name=skill_name, description=description, tags=["cluster"], evidence=evs[0] if evs else None)
            for e in evs[1:]:
                node.evidence.append(e)
        elif mode == "domain":
            dom = classify_domain(c)
            if dom not in domain_nodes:
                domain_nodes[dom] = Node(text=f"domain: {dom}", skill_name=dom, description=None, tags=["domain"]) 
                st.root.children.append(domain_nodes[dom])
            n = Node(text=desc, skill_name=skill_name, description=description, tags=["cluster", dom])
            for e in evs:
                n.evidence.append(e)
            domain_nodes[dom].children.append(n)
        else:
            n = Node(text=desc, skill_name=skill_name, description=description, tags=["cluster"])
            for e in evs:
                n.evidence.append(e)
            st.root.children.append(n)

    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clusters", required=True, help="Path to clusters_with_descriptors.json (or clusters.json)")
    ap.add_argument("--out", required=True, help="Output SkillTree JSON path")
    ap.add_argument("--mode", choices=["flat", "insert", "domain"], default="flat", help="flat: root children only; insert: merge by similarity; domain: group into domain buckets")
    ap.add_argument("--max-clusters", type=int, default=500)
    ap.add_argument("--exclude-noise", action="store_true", help="Skip cluster id -1")
    args = ap.parse_args()

    clusters = load_clusters(args.clusters)
    st = build_tree_from_clusters(clusters, mode=args.mode, max_clusters=args.max_clusters, exclude_noise=args.exclude_noise)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(st.serialize())
    print(f"Wrote SkillTree JSON to {out_path}")


if __name__ == "__main__":
    main()
