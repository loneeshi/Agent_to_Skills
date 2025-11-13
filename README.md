# SkillTree (prototype)

A lightweight, API-free prototype for building a hierarchical skill/knowledge tree from short texts. It uses a toy embedding function (no network calls) so you can test insertion and search locally.

## What’s here

- `SkillTree.py` — minimal SkillTree implementation with a toy embedder (random Gaussian vectors seeded by text hash), plus a small demo in `__main__`.
- `embeddings.py` — optional real embedding providers (OpenAI or local Hugging Face). Not wired into `SkillTree.py` yet.
- `prompt.py` — a prompt template for summarizing a dialog into structured steps/knowledge.
- `skilltree_demo.py` — an older demo that expects a richer API (`initialize/add/search_and_show/serialize`). It’s not aligned with `SkillTree.py` yet (see Notes below).

## Quick start (no API, local only)

Run the built-in demo that inserts a few Chinese C++-related snippets, prints the tree, and does a similarity search:

```bash
cd design/Agent_to_Skills
python SkillTree.py
```

You should see a textual tree and the most similar node for the query like “多态怎么实现？”. Because the current embeddings are toy/random (see below), exact outputs vary between runs/processes.

### Programmatic usage (toy embedder)

```python
from SkillTree import SkillTree

st = SkillTree("root knowledge")
st.insert("我想学习 C++ 的类和继承")
st.insert("在基类用 virtual 修饰函数，子类 override，实现多态")

node, score = st.search("怎么实现 C++ 的多态？")
print(node.text, score)
```

## How it works (prototype design)

- Embeddings: `embed(text)` returns a 128-d vector sampled from a Gaussian, normalized to unit length. It is seeded by `hash(text)`, so within a single Python process texts map deterministically; across processes the built-in `hash` salt may differ (see Caveats).
- Similarity: cosine similarity.
- Insertion (`SkillTree.insert`):
  - If the current node `v` has no children, it temporarily creates a leaf-copy of itself as a child (so `v` acts like an aggregator).
  - It finds the child with maximum similarity to the new node (`best_idx = max(range(len(sims)), key=lambda i: sims[i])`).
  - If similarity ≥ `threshold(depth)`, it aggregates new text into `v.text` (deduplicating lines) and recurses into the best child.
  - Otherwise, if that leaf-copy was just created for this decision, it is removed and the new node is appended as a child. This keeps structure shallow when items are dissimilar.
- Threshold schedule: `default_threshold(d) = max(0.55, 0.85 - 0.05*d)` — stricter near the root, looser deeper down.
- Search: DFS over nodes using cosine similarity with the query.

## Current API surface

- Class `SkillTree(root_text: str = "root")`
  - `insert(text: str, threshold_fn=default_threshold)`
  - `search(query: str) -> (best_node, score)`
  - `print_tree(node: Optional[Node] = None, depth: int = 0)`
- Class `Node`
  - `text: str`
  - `embedding: List[float]`
  - `timestamp: float`
  - `children: List[Node]`

## Caveats and limitations

- Toy embeddings only: not semantically meaningful. Results are for structure/shaping tests, not accuracy.
- Cross-run variability: Python’s `hash()` is salted per process by default, so embeddings can differ between runs. To stabilize across runs you can either:
  - set `PYTHONHASHSEED=0` when running Python; or
  - replace `hash(text)` with a stable hash (e.g., `hashlib.md5(text.encode()).hexdigest()`).
- No persistence/serialization: there’s no `serialize()`/`deserialize()` in `SkillTree.py` yet.
- `skilltree_demo.py` mismatch: that file expects methods like `initialize`, `add`, `serialize`, and a helper `search_and_show`, which aren’t implemented here.
- No balancing or re-clustering: structure depends on insertion order; there’s no rebalancing.

## Roadmap (suggested next steps)

1. Deterministic embedding seeding
   - Swap `hash(text)` for a stable hash (e.g., MD5/SHA1) to get run-to-run repeatability.
2. Real embeddings (optional)
   - Integrate `embeddings.py` so you can choose between `openai` and `hf` providers via env vars:
     - `EMBEDDING_PROVIDER=openai|hf`
     - `EMBEDDING_MODEL` (e.g., `text-embedding-3-large` or `intfloat/e5-base-v2`)
     - For HF on macOS: `DEVICE=mps` is often fast; or `cpu`/`cuda`.
3. API alignment with demo
   - Add: `initialize(root_text)`, `add(text, timestamp=None)`, `serialize()` (JSON), and a `search_and_show(st, query, top_k)` utility to match `skilltree_demo.py`.
4. Persistence
   - Implement JSON serialization for the tree and a simple loader.
5. Quality-of-life
   - Node IDs, parent refs, and optional per-node metadata (tags, scores, provenance).

## Using real embeddings (optional, not wired yet)

If you want to experiment with real embeddings:

- OpenAI:
  - Set `OPENAI_API_KEY` (and optionally `OPENAI_ENDPOINT` or Azure vars `AZURE_OPENAI_*`).
  - `EMBEDDING_PROVIDER=openai`, `EMBEDDING_MODEL=text-embedding-3-large` (default).
- Hugging Face / local:
  - `EMBEDDING_PROVIDER=hf`, `EMBEDDING_MODEL=intfloat/e5-base-v2` (default), `DEVICE=mps|cuda|cpu`.

Then replace the local `embed()` function in `SkillTree.py` with something like:

```python
from embeddings import get_embedder
_embedder = get_embedder()

def embed(text: str):
    return _embedder.embed(text)
```

Note: this will change vector dimensionality; everything else (cosine, thresholds) should still work.

## Notes

- If you run `skilltree_demo.py` as-is, you’ll likely get import or attribute errors because it expects a different interface. Prefer running `SkillTree.py` directly for now.
- For reproducible demos, consider setting `PYTHONHASHSEED=0` before invoking Python.

## License

Prototype code — no license declared. Add one if you plan to share externally.
