#!/usr/bin/env python3
"""
Run a subset preprocess using the fragment-extraction logic from preprocess_to_skill.py
This script imports preprocess logic (if preprocess_to_skill.py is in tools/) or
reimplements minimal fragment extraction for selected files.

Usage:
  python tools/preprocess_subset.py --files \
    task_data/background/courses.json \
    task_data/background/campus_data.json \
    --out outputs/preprocess_subset/fragments.jsonl

Notes:
 - Paths are relative to repo root or absolute.
 - This is convenient if you want to test exactly chosen files instead of using --max-files.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from tools.preprocess_to_skill import fragment_from_obj, safe_load_json, fragment_from_text, clean_text

def process_files(file_paths, out_path, max_len_chars=500):
    fragments = []
    for p in file_paths:
        pth = Path(p)
        if not pth.exists():
            print("Not found:", p)
            continue
        obj = safe_load_json(pth)
        if obj is None:
            continue
        # reuse fragment_from_obj (it will tag fragments with file stem prefix)
        fragment_from_obj(obj, pth, fragments, frag_prefix=pth.stem, max_len_chars=max_len_chars)
    os.makedirs(Path(out_path).parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for f in fragments:
            fout.write(json.dumps(f, ensure_ascii=False) + "\n")
    print(f"Wrote {len(fragments)} fragments to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", "-f", nargs="+", required=True, help="List of json files to process")
    parser.add_argument("--out", "-o", required=True, help="Output fragments.jsonl path")
    args = parser.parse_args()
    process_files(args.files, args.out)