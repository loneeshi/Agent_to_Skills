from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Tuple
import time, json, hashlib, random, math

# ----------------------
# Deterministic toy embedding
# ----------------------

def _stable_seed(text: str) -> int:
    # Use MD5 for a stable seed across runs
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def embed(text: str, dim: int = 128) -> List[float]:
    rnd = random.Random(_stable_seed(text))
    vec = [rnd.gauss(0, 1) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def default_threshold(depth: int) -> float:
    return max(0.55, 0.85 - 0.05 * depth)


def _now() -> float:
    return time.time()


@dataclass
class Evidence:
    text: str
    source: Optional[str] = None
    timestamp: float = field(default_factory=_now)


@dataclass
class Node:
    text: str
    skill_name: Optional[str] = None
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    confidence: float = 0.5  # 0..1 mastery confidence
    embedding: List[float] = field(default_factory=list)
    timestamp: float = field(default_factory=_now)
    children: List["Node"] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)

    def __post_init__(self):
        if not self.embedding:
            self.embedding = embed(self.text)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["children"] = [c.to_dict() for c in self.children]
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Node":
        node = Node(
            text=d.get("text", ""),
            skill_name=d.get("skill_name"),
            description=d.get("description"),
            tags=list(d.get("tags", [])),
            confidence=float(d.get("confidence", 0.5)),
            embedding=list(d.get("embedding", [])),
            timestamp=float(d.get("timestamp", _now())),
            evidence=[Evidence(**e) for e in d.get("evidence", [])],
        )
        node.children = [Node.from_dict(c) for c in d.get("children", [])]
        if not node.embedding:
            node.embedding = embed(node.text)
        return node


class SkillTree:
    def __init__(self, root_text: str = "root knowledge"):
        self.root = Node(root_text, skill_name="root")

    # ---------------
    # Core Ops
    # ---------------
    def insert(self, text: str, *, skill_name: Optional[str] = None, description: Optional[str] = None,
               tags: Optional[List[str]] = None, evidence: Optional[Evidence] = None,
               threshold_fn=default_threshold, _node: Optional[Node] = None, _depth: int = 0) -> Node:
        tags = tags or []
        v_new = Node(text=text, skill_name=skill_name, description=description, tags=tags)
        if evidence:
            v_new.evidence.append(evidence)
        return self._insert_node(_node or self.root, v_new, _depth, threshold_fn)

    def _insert_node(self, v: Node, v_new: Node, depth: int, threshold_fn) -> Node:
        leaf_created = False
        if not v.children:
            # temporarily split leaf
            leaf = Node(v.text, skill_name=v.skill_name, description=v.description, tags=v.tags, confidence=v.confidence)
            v.children.append(leaf)
            leaf_created = True

        sims = [cosine(v_new.embedding, c.embedding) for c in v.children]
        best_idx = max(range(len(sims)), key=lambda i: sims[i])
        smax = sims[best_idx]
        v_best = v.children[best_idx]

        if smax >= threshold_fn(depth):
            # Aggregate into parent cluster and descend
            v.text = _aggregate(v.text, v_new.text)
            v.embedding = embed(v.text)
            v.confidence = min(1.0, v.confidence + 0.02)
            return self._insert_node(v_best, v_new, depth + 1, threshold_fn)
        else:
            if leaf_created:
                v.children.clear()  # revert the temporary split
            v.children.append(v_new)
            return v_new

    # ---------------
    # Query & Mastery
    # ---------------
    def search(self, query: str, top_k: int = 1) -> List[Tuple[Node, float]]:
        q = embed(query)
        res: List[Tuple[Node, float]] = []
        stack = [self.root]
        while stack:
            n = stack.pop()
            s = cosine(q, n.embedding)
            res.append((n, s))
            stack.extend(n.children)
        res.sort(key=lambda x: x[1], reverse=True)
        return res[:top_k]

    def reinforce(self, node: Node, success: bool, evidence: Optional[Evidence] = None, delta: float = 0.05):
        if success:
            node.confidence = min(1.0, node.confidence + delta)
        else:
            node.confidence = max(0.0, node.confidence - delta)
        if evidence:
            node.evidence.append(evidence)

    # ---------------
    # Persistence
    # ---------------
    def serialize(self) -> str:
        return json.dumps(self.root.to_dict(), ensure_ascii=False, indent=2)

    @staticmethod
    def deserialize(data: str) -> "SkillTree":
        obj = json.loads(data)
        root = Node.from_dict(obj)
        st = SkillTree("root")
        st.root = root
        return st


# ----------------------
# Utilities
# ----------------------

def _aggregate(c1: str, c2: str) -> str:
    a = [s.strip() for s in c1.splitlines() if s.strip()]
    b = [s.strip() for s in c2.splitlines() if s.strip()]
    out = a[:]
    for s in b:
        if not any(s in x or x in s for x in out):
            out.append(s)
    return "\n".join(out)


if __name__ == "__main__":
    # Tiny self-check
    st = SkillTree()
    st.insert("我想学习 C++ 的类和继承", skill_name="C++ 类与继承")
    st.insert("在基类用 virtual 修饰函数，子类 override，实现多态", skill_name="C++ 多态")
    hits = st.search("怎么实现多态？", top_k=3)
    for n, s in hits:
        print(round(s, 3), n.skill_name, "|", n.text[:50])
