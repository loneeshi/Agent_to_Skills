#!/usr/bin/env python3
"""
List all .json files under task_data/background with sizes, and optionally print top-N by size.

Usage:
  python tools/list_bg_jsons.py --repo-root /path/to/ELL-StuLife --top 10
"""
import os
import argparse
from pathlib import Path

def list_jsons(repo_root, top=0):
    bg = Path(repo_root) / "task_data" / "background"
    if not bg.exists():
        print("Background dir not found:", bg)
        return []
    files = [p for p in bg.rglob("*.json")]
    info = []
    for p in files:
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        info.append((p, size))
    info_sorted = sorted(info, key=lambda x: x[1], reverse=True)
    if top and top > 0:
        info_sorted = info_sorted[:top]
    for p, s in info_sorted:
        print(f"{s:12d}  {p}")
    return info_sorted

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", "-r", required=True)
    parser.add_argument("--top", type=int, default=0, help="Show top-N largest files (0 -> all)")
    args = parser.parse_args()
    list_jsons(args.repo_root, top=args.top)