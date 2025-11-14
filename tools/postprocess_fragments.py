# name=tools/postprocess_fragments.py
import json
from pathlib import Path
from collections import defaultdict, Counter
import re

FRAGMENTS = "outputs/preprocess_subset/fragments.jsonl"  # 修改为你的路径
OUT_DIR = Path("outputs/postprocess")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_fragments(path):
    frags = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            frags.append(json.loads(line))
    return frags

def text_key(s):
    return re.sub(r'\s+', ' ', s.strip().lower())

def auto_template_for_course_pass(f):
    slots = f.get("slots", {})
    course_codes = slots.get("course_codes") or []
    pass_types = slots.get("pass_types") or []
    verbs = slots.get("verbs") or []
    # detect assignment intent
    assign_verbs = {"assign", "apply", "select", "register", "choose"}
    if course_codes and (pass_types or any(v in assign_verbs for v in verbs)):
        templates = []
        for c in course_codes:
            if pass_types:
                for p in pass_types:
                    templates.append({"skill_type":"assign_pass_course",
                                      "template": f"Assign {p} to course {c}.",
                                      "example": f["text"],
                                      "source_id": f["id"]})
            else:
                templates.append({"skill_type":"assign_pass_course",
                                  "template": f"Assign a pass to course {c}.",
                                  "example": f["text"],
                                  "source_id": f["id"]})
        return templates
    return []

def main():
    frags = load_fragments(FRAGMENTS)
    # dedupe by normalized text and count freq
    dedupe = {}
    freq = Counter()
    for f in frags:
        k = text_key(f["text"])
        freq[k] += 1
        if k not in dedupe:
            dedupe[k] = f
            f["_orig_text"] = f["text"]
            f["_norm_text"] = k
        else:
            # you may want to merge slots
            ex = dedupe[k]
            # merge some slots conservatively
            for s_k, s_v in f.get("slots", {}).items():
                if s_k not in ex.get("slots", {}):
                    ex.setdefault("slots", {})[s_k] = s_v
    uniq_frags = list(dedupe.values())
    print(f"Loaded {len(frags)} fragments, dedup -> {len(uniq_frags)} unique fragments")

    # auto-template structured fragments
    templates = []
    remain = []
    for f in uniq_frags:
        tpls = auto_template_for_course_pass(f)
        if tpls:
            templates.extend(tpls)
        else:
            # optional: treat fragments from field 'course_code' specially by skipping them (they are not full utterances)
            if f.get("field") and f["field"] in ("course_code","semester","type"):
                # skip these minimal structured tokens in embedding pool
                continue
            remain.append(f)

    # save templates
    with open(OUT_DIR / "auto_templates.jsonl", "w", encoding="utf-8") as fout:
        for t in templates:
            fout.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"Wrote {len(templates)} auto templates to {OUT_DIR/'auto_templates.jsonl'}")

    # prepare embed inputs (texts)
    embed_texts = [f["text"] for f in remain]
    # dedupe embed_texts
    seen = set()
    embed_lines = []
    for t in embed_texts:
        k = text_key(t)
        if k not in seen:
            seen.add(k)
            embed_lines.append(t)
    # write embed input file
    with open(OUT_DIR / "embed_input.txt", "w", encoding="utf-8") as fout:
        for t in embed_lines:
            fout.write(t.replace("\n", " ") + "\n")
    print(f"Wrote {len(embed_lines)} lines to {OUT_DIR/'embed_input.txt'}")

    # write simplified fragments for downstream mapping
    with open(OUT_DIR / "fragments_unique.jsonl", "w", encoding="utf-8") as fout:
        for f in uniq_frags:
            f2 = {k:f.get(k) for k in ("id","text","source_file","field","slots")}
            f2["freq"] = freq.get(f["_norm_text"], 1)
            fout.write(json.dumps(f2, ensure_ascii=False) + "\n")
    print(f"Wrote fragments_unique.jsonl with {len(uniq_frags)} entries")

if __name__ == "__main__":
    main()