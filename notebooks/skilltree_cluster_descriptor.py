"""
notebooks/skilltree_cluster_descriptor.py

Notebook-style script for:
  - load fragments.jsonl (output of preprocess_to_skill.py)
  - embed fragments (uses local embeddings module if available or sentence-transformers fallback)
  - cluster embeddings (HDBSCAN preferred; DBSCAN fallback)
  - auto-generate cluster descriptors (tf-idf keywords, representative example, action-object templates)
  - export clusters.json and clusters_with_descriptors.json

Usage:
  python notebooks/skilltree_cluster_descriptor.py --fragments outputs/preprocess/fragments.jsonl --out outputs/cluster
Dependencies (CPU):
  pip install sentence-transformers scikit-learn tqdm hdbscan rank_bm25
  (hdbscan optional; if not installed script will fallback to sklearn.DBSCAN)
"""

import os
import json
import argparse
import re
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from collections import defaultdict, Counter

# optional: hdbscan
try:
    import hdbscan
    _HAS_HDBSCAN = True
except Exception:
    _HAS_HDBSCAN = False

def load_fragments(path):
    frags = []
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            if not line.strip(): continue
            frags.append(json.loads(line))
    return frags

def embed_texts(model, texts, batch_size=64):
    embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i:i+batch_size]
        be = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        embs.append(be)
    X = np.vstack(embs).astype('float32')
    # L2 normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    X = X / norms
    return X

def cluster_embeddings(X, method="hdbscan", min_cluster_size=8, dbscan_eps=0.15, dbscan_min_samples=5):
    if method == "hdbscan" and _HAS_HDBSCAN:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric='euclidean')
        labels = clusterer.fit_predict(X)
    else:
        # use DBSCAN with cosine metric (sklearn)
        clusterer = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, metric='cosine', n_jobs=-1)
        labels = clusterer.fit_predict(X)
    return labels

def get_top_tfidf_terms(texts, top_n=8):
    if not texts:
        return []
    vect = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=2000)
    X = vect.fit_transform(texts)
    scores = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vect.get_feature_names_out())
    top_idx = np.argsort(scores)[::-1][:top_n]
    return terms[top_idx].tolist()

def build_clusters(frags, labels, embeddings, top_k_examples=8):
    # Map label => members
    clusters = {}
    label_to_idxs = defaultdict(list)
    for i, lab in enumerate(labels):
        label_to_idxs[int(lab)].append(i)
    for lab, idxs in label_to_idxs.items():
        texts = [frags[i]['text'] for i in idxs]
        # representative: centroid nearest
        centroid = embeddings[idxs].mean(axis=0)
        sims = embeddings[idxs] @ centroid
        rep_idx = int(np.argmax(sims))
        rep_text = texts[rep_idx]
        # tfidf terms
        terms = get_top_tfidf_terms(texts, top_n=8)
        # aggregate slots
        slot_counter = Counter()
        for i in idxs:
            for k, v in frags[i].get("slots", {}).items():
                slot_counter[k] += 1
        clusters[int(lab)] = {
            "cluster_id": int(lab),
            "size": len(idxs),
            "member_idxs": idxs,
            "representative": rep_text,
            "examples": texts[:min(top_k_examples, len(texts))],
            "top_terms": terms,
            "slot_counts": slot_counter.most_common()
        }
    return clusters

def generate_descriptor_for_cluster(cluster):
    """
    Lightweight descriptor generation:
      - uses representative sentence + top terms + slot hints to produce a short action-like descriptor.
    """
    rep = cluster.get("representative","")
    terms = cluster.get("top_terms",[])[:6]
    slots = dict(cluster.get("slot_counts",[]))
    # build short descriptor heuristics
    descriptor_parts = []
    # if pass_types + course_codes present -> assignment skill
    tc = [t for t in terms if re_course_code(t)]
    # use simple heuristics
    if any('pass_types' in k for k,v in cluster.get('slot_counts',[])):
        # try to form pattern "Assign {pass} to {course}"
        descriptor = f"Assign pass(es) to course(s). Representative: \"{rep[:120]}\""
    else:
        # generic
        descriptor = f"Skill about: {', '.join(terms[:4])}. Representative: \"{rep[:120]}\""
    # keep descriptor short
    return descriptor

# small helper to detect course code patterns in terms
COURSE_SIMPLE_RE = re.compile(r'[A-Z]{2,6}\d{6,7}(?:\(\d+\))?')

def re_course_code(s):
    return bool(COURSE_SIMPLE_RE.search(s))

def export_clusters(clusters, out_dir, *, members_policy: str = "head", max_members: int = 64):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # Apply member_idxs trimming policy to reduce file size
    def _apply_policy(cdict):
        d = dict(cdict)
        if "member_idxs" in d:
            if members_policy == "none":
                d.pop("member_idxs", None)
            elif members_policy == "head":
                d["member_idxs"] = d["member_idxs"][:max(0, int(max_members))]
            else:
                # full: keep as is
                pass
        return d

    trimmed = {k: _apply_policy(v) for k, v in clusters.items()}
    with open(out_dir / "clusters.json", "w", encoding='utf-8') as fout:
        json.dump(trimmed, fout, ensure_ascii=False, indent=2)
    # clusters with descriptors
    clusters_with_desc = {}
    for k,v in trimmed.items():
        clusters_with_desc[k] = dict(v)
        clusters_with_desc[k]['descriptor'] = generate_descriptor_for_cluster(v)
    with open(out_dir / "clusters_with_descriptors.json", "w", encoding='utf-8') as fout:
        json.dump(clusters_with_desc, fout, ensure_ascii=False, indent=2)
    print(f"Wrote clusters to {out_dir}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fragments", "-f", required=True, help="fragments.jsonl path")
    parser.add_argument("--out", "-o", required=True, help="output dir for clusters")
    parser.add_argument("--model", "-m", default="all-MiniLM-L6-v2", help="sentence-transformers model name")
    parser.add_argument("--examples-per-cluster", type=int, default=8, help="number of example texts to keep per cluster")
    parser.add_argument("--members", choices=["full","head","none"], default="head", help="how to store member_idxs in outputs: full=head+all, head=keep first N, none=drop")
    parser.add_argument("--max-members", type=int, default=64, help="N for --members=head policy")
    parser.add_argument("--cluster-method", choices=["hdbscan","dbscan"], default="hdbscan")
    parser.add_argument("--min-cluster-size", type=int, default=8)
    parser.add_argument("--dbscan-eps", type=float, default=0.15)
    args = parser.parse_args()

    frags = load_fragments(args.fragments)
    texts = [f['text'] for f in frags]
    print(f"Loaded {len(texts)} fragments.")

    # load model
    print("Loading embedding model (sentence-transformers)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)

    # embed
    embeddings = embed_texts(model, texts, batch_size=64)

    # cluster
    print("Clustering embeddings...")
    labels = cluster_embeddings(embeddings, method=args.cluster_method, min_cluster_size=args.min_cluster_size, dbscan_eps=args.dbscan_eps)
    print("Cluster label counts:", Counter(labels).most_common()[:20])

    clusters = build_clusters(frags, labels, embeddings, top_k_examples=args.examples_per_cluster)
    export_clusters(clusters, args.out, members_policy=args.members, max_members=args.max_members)
    print("Sample cluster entries (top 5 by size):")
    sample = sorted(clusters.values(), key=lambda c: -c['size'])[:5]
    for c in sample:
        print(f"Cluster {c['cluster_id']} | size={c['size']} | top_terms={c['top_terms'][:5]} | rep={c['representative'][:120]}")

if __name__ == "__main__":
    main()