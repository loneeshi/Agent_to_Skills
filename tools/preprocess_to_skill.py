#!/usr/bin/env python3
"""
tools/preprocess_to_skill.py

Purpose:
  - Parse task_data/tasks.json and task_data/background/**/*.json
  - Produce utterance-level fragments suitable for embedding/skill discovery
  - Extract lightweight structured slots (course codes, times, locations, verbs, objects)
  - Save fragments as JSONL and a small summary report

Usage (example):
  python tools/preprocess_to_skill.py \
    --repo-root /path/to/ELL-StuLife \
    --out-dir outputs/preprocess \
    --max-files 9999

Outputs:
  - <out_dir>/fragments.jsonl         (one fragment per line)
  - <out_dir>/fragments_meta.json     (summary metadata)
  - <out_dir>/task_to_background.json (mapping from tasks.json task -> background file)
  - <out_dir>/report.json             (analysis summary)
Dependencies (CPU):
  pip install nltk spacy tqdm
  python -m spacy download en_core_web_sm
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
from collections import defaultdict, Counter
from tqdm import tqdm

# NLP
try:
    import nltk
    nltk.data.find("tokenizers/punkt")
except Exception:
    import nltk
    nltk.download("punkt")
from nltk.tokenize import sent_tokenize

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None

# Regex patterns
COURSE_RE = re.compile(r'\b[A-Z]{2,6}\d{6,7}(?:\(\d+\))?\b')  # COMS0031131032(2)
WEEK_RE = re.compile(r'\bWeek\s*\d+\b', re.I)
TIME_RE = re.compile(r'\b(?:[01]?\d:[0-5]\d|(?:[01]?\d|2[0-3]):[0-5]\d|morning|afternoon|evening|noon)\b', re.I)
DAY_RE = re.compile(r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b', re.I)
BUILDING_RE = re.compile(r"\b(?:Hall|Center|Building|Room|Lecture|Library|Office|Quad|House|Center|Lab|Auditorium)\b", re.I)
PASS_RE = re.compile(r'\b(?:S-Pass|A-Pass|B-Pass|pass)\b', re.I)

# utility
def safe_load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
            # try newline-delimited JSON fallback
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    lines = [l.strip() for l in f if l.strip()]
                if len(lines) == 1:
                    return json.loads(lines[0])
                else:
                    return [json.loads(l) for l in lines]
            except Exception:
                return None

def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\r\n", " ").replace("\n", " ").strip()
    s = re.sub(r'\s+', ' ', s)
    return s

def extract_slots(text: str):
    """Lightweight slot extractor returning a dict of found slots."""
    slots = {}
    if not text:
        return slots
    text_l = text
    # course codes
    courses = COURSE_RE.findall(text_l)
    if courses:
        slots['course_codes'] = list(dict.fromkeys(courses))  # unique preserve order
    # week/time/day
    weeks = WEEK_RE.findall(text_l)
    if weeks:
        slots['weeks'] = list(dict.fromkeys(weeks))
    times = TIME_RE.findall(text_l)
    if times:
        slots['times'] = list(dict.fromkeys(times))
    days = DAY_RE.findall(text_l)
    if days:
        slots['days'] = list(dict.fromkeys(days))
    # pass types
    passes = PASS_RE.findall(text_l)
    if passes:
        slots['pass_types'] = list(dict.fromkeys(passes))
    # building keywords heuristics
    if BUILDING_RE.search(text_l):
        slots['building_mention'] = True
    # simple verb/object extraction via spaCy if available
    if nlp:
        doc = nlp(text_l)
        verbs = []
        objects = []
        for tok in doc:
            if tok.pos_ == "VERB":
                verbs.append(tok.lemma_)
                # look for direct objects or prepositional objects
                for ch in tok.children:
                    if ch.dep_ in ("dobj", "pobj", "attr", "dative"):
                        objects.append(ch.text)
        if verbs:
            slots['verbs'] = list(dict.fromkeys(verbs))[:6]
        if objects:
            slots['objects'] = list(dict.fromkeys(objects))[:6]
        # named entities e.g., PERSON, ORG, GPE
        ents = [ (ent.text, ent.label_) for ent in doc.ents ]
        if ents:
            slots['entities'] = ents[:10]
    return slots

def fragment_from_text(text, source_file, parent_id=None, field=None, frag_id_prefix=None, max_len_chars=500):
    """
    Break text into sentence-level fragments; if a sentence too long, chunk into pieces.
    Returns list of fragments dicts.
    """
    fragments = []
    sents = sent_tokenize(text)
    idx = 0
    for sent in sents:
        sent = clean_text(sent)
        if not sent:
            continue
        # chunk if longer than max_len_chars
        if len(sent) > max_len_chars:
            # naive chunk: split by comma/; else slice
            parts = re.split(r'[，,;；::\-—]', sent)
            cur = ""
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                if len(cur) + len(p) + 1 <= max_len_chars:
                    cur = (cur + " " + p).strip()
                else:
                    if cur:
                        frag = {
                            "id": f"{frag_id_prefix}::{field}::{idx}" if frag_id_prefix else f"{source_file}::{field}::{idx}",
                            "text": cur,
                            "source_file": source_file,
                            "parent_id": parent_id,
                            "field": field
                        }
                        frag['slots'] = extract_slots(cur)
                        fragments.append(frag)
                        idx += 1
                    cur = p
            if cur:
                frag = {
                    "id": f"{frag_id_prefix}::{field}::{idx}" if frag_id_prefix else f"{source_file}::{field}::{idx}",
                    "text": cur,
                    "source_file": source_file,
                    "parent_id": parent_id,
                    "field": field
                }
                frag['slots'] = extract_slots(cur)
                fragments.append(frag)
                idx += 1
        else:
            frag = {
                "id": f"{frag_id_prefix}::{field}::{idx}" if frag_id_prefix else f"{source_file}::{field}::{idx}",
                "text": sent,
                "source_file": source_file,
                "parent_id": parent_id,
                "field": field
            }
            frag['slots'] = extract_slots(sent)
            fragments.append(frag)
            idx += 1
    return fragments

def fragment_from_obj(obj, src_path, out_list, frag_prefix=None, max_len_chars=500):
    """
    Converts a parsed JSON object (dict or list) to list of fragment dicts.
    Heuristics: extract known textual fields first, then fallback to nested content lists.
    """
    if isinstance(obj, dict):
        # prioritized fields likely to contain text
        preferred_text_fields = ['instruction','text','description','body','prompt','question','answer','title','chapter_title','section_title','paragraph']
        found = False
        for f in preferred_text_fields:
            if f in obj and isinstance(obj[f], str) and obj[f].strip():
                found = True
                frags = fragment_from_text(obj[f].strip(), Path(src_path).name, parent_id=None, field=f, frag_id_prefix=frag_prefix, max_len_chars=max_len_chars)
                out_list.extend(frags)
        # also handle common nested content arrays
        for key, val in obj.items():
            if isinstance(val, list):
                # if list of strings
                if val and isinstance(val[0], str):
                    for i, v in enumerate(val):
                        frags = fragment_from_text(v, Path(src_path).name, parent_id=None, field=f"{key}[{i}]", frag_id_prefix=frag_prefix, max_len_chars=max_len_chars)
                        out_list.extend(frags)
                else:
                    # list of dicts
                    for i, item in enumerate(val):
                        if isinstance(item, dict):
                            fragment_from_obj(item, src_path, out_list, frag_prefix=f"{frag_prefix or Path(src_path).stem}::{key}[{i}]", max_len_chars=max_len_chars)
            elif isinstance(val, str) and len(val) < 3000:
                # string field not in preferred list -> still consider
                frags = fragment_from_text(val, Path(src_path).name, parent_id=None, field=key, frag_id_prefix=frag_prefix, max_len_chars=max_len_chars)
                out_list.extend(frags)
        # if nothing produced, fallback to a small json-string fragment (truncated)
        if not found:
            # produce a summary fragment if possible
            s = json.dumps(obj, ensure_ascii=False)
            s_short = clean_text(s)[:800]
            if s_short:
                frags = fragment_from_text(s_short, Path(src_path).name, parent_id=None, field="json_snippet", frag_id_prefix=frag_prefix, max_len_chars=max_len_chars)
                out_list.extend(frags)
    elif isinstance(obj, list):
        for i, item in enumerate(obj):
            fragment_from_obj(item, src_path, out_list, frag_prefix=f"{frag_prefix or Path(src_path).stem}::list[{i}]", max_len_chars=max_len_chars)
    else:
        # primitive -> ignore
        pass

def map_tasks_to_background(tasks_json):
    """
    Build a mapping task_id -> referenced background filenames (task_source_file or available_systems)
    Returns dict.
    """
    mapping = {}
    if not isinstance(tasks_json, dict):
        return mapping
    for k, v in tasks_json.items():
        try:
            tid = v.get("task_id") or k
            sources = set()
            tsf = v.get("task_source_file")
            if tsf:
                if isinstance(tsf, list):
                    sources.update(tsf)
                else:
                    sources.add(tsf)
            # available_systems may hint background types (e.g., student_handbook, data_system)
            avs = v.get("available_systems") or []
            for a in avs:
                sources.add(a)
            mapping[tid] = list(sources)
        except Exception:
            continue
    return mapping

def process_background_dir(bg_dir: str, out_path: str, max_files: int = 9999, max_len_chars=500):
    bg_dir = Path(bg_dir)
    files = list(bg_dir.rglob("*.json"))
    files = sorted(files)[:max_files]
    fragments = []
    file_summary = []
    for p in tqdm(files, desc="Reading background files"):
        obj = safe_load_json(p)
        file_summary.append({"path": str(p), "parsed_type": type(obj).__name__})
        if obj is None:
            continue
        fragment_from_obj(obj, p, fragments, frag_prefix=Path(p).stem, max_len_chars=max_len_chars)
    # save fragments.jsonl
    os.makedirs(Path(out_path).parent, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as fout:
        for frag in fragments:
            fout.write(json.dumps(frag, ensure_ascii=False) + "\n")
    return fragments, file_summary

def analyze_and_save(repo_root, out_dir, max_files, max_len_chars):
    repo_root = Path(repo_root)
    task_data_dir = repo_root / "task_data"
    if not task_data_dir.exists():
        print("ERROR: task_data directory not found under repo root:", task_data_dir)
        sys.exit(1)
    tasks_json_path = task_data_dir / "tasks.json"
    tasks_json = None
    if tasks_json_path.exists():
        tasks_json = safe_load_json(tasks_json_path)
    bg_dir = task_data_dir / "background"
    if not bg_dir.exists():
        print("ERROR: background directory not found under task_data:", bg_dir)
        sys.exit(1)

    # map tasks to background hints
    task_map = {}
    if tasks_json:
        task_map = map_tasks_to_background(tasks_json)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fragments_out = out_dir / "fragments.jsonl"
    fragments, file_summary = process_background_dir(str(bg_dir), str(fragments_out), max_files=max_files, max_len_chars=max_len_chars)

    # summary stats
    total_frags = len(fragments)
    slot_counts = Counter()
    frag_len_chars = [len(f['text']) for f in fragments]
    for f in fragments:
        for k in f.get("slots", {}).keys():
            slot_counts[k] += 1

    summary = {
        "repo_root": str(repo_root),
        "task_data_exists": True,
        "tasks_present": bool(tasks_json),
        "num_background_files_scanned": len(file_summary),
        "num_fragments": total_frags,
        "fragment_len_chars_mean": sum(frag_len_chars)/len(frag_len_chars) if frag_len_chars else 0,
        "top_slot_counts": slot_counts.most_common(40),
        "sample_fragments": fragments[:20],
        "background_files_sample": file_summary[:40],
    }
    # save metadata and mapping
    with open(out_dir / "fragments_meta.json", "w", encoding='utf-8') as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)
    with open(out_dir / "task_to_background.json", "w", encoding='utf-8') as fout:
        json.dump(task_map, fout, ensure_ascii=False, indent=2)
    print(f"Wrote {total_frags} fragments to {fragments_out}")
    print(f"Summary written to {out_dir / 'fragments_meta.json'}")
    return fragments_out, summary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", "-r", required=True, help="Path to ELL-StuLife repository root")
    parser.add_argument("--out-dir", "-o", required=True, help="Directory to write outputs")
    parser.add_argument("--max-files", type=int, default=9999, help="Max background files to scan")
    parser.add_argument("--max-chars", type=int, default=500, help="Max chars per fragment before chunking")
    args = parser.parse_args()
    fragments_out, summary = analyze_and_save(args.repo_root, args.out_dir, args.max_files, args.max_chars)
    # print short report
    print("Short report:")
    print(f"  fragments_count: {summary['num_fragments']}")
    print(f"  mean_fragment_len: {summary['fragment_len_chars_mean']:.1f}")
    print(f"  top_slots: {summary['top_slot_counts'][:10]}")

if __name__ == "__main__":
    main()