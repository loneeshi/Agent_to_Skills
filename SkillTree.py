from typing import List, Optional
import math, random, time

def embed(text: str) -> List[float]:
    random.seed(hash(text) % (2**32))
    vec = [random.gauss(0, 1) for _ in range(128)]
    norm = math.sqrt(sum(x*x for x in vec))
    return [x / norm for x in vec]

def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def compute_similarity(e_new: List[float], embeddings: List[List[float]]) -> List[float]:
    return [cosine(e_new, e) for e in embeddings]

def aggregate(c1: str, c2: str) -> str:
    a = [s.strip() for s in c1.splitlines() if s.strip()]
    b = [s.strip() for s in c2.splitlines() if s.strip()]
    out = a[:]
    for s in b:
        if not any(s in x or x in s for x in out):
            out.append(s)
    return "\n".join(out)

def default_threshold(d: int) -> float:
    return max(0.55, 0.85 - 0.05*d)

class Node:
    def __init__(self, text: str):
        self.text = text
        self.embedding = embed(text)
        self.timestamp = time.time()
        self.children: List["Node"] = []

class SkillTree:
    def __init__(self, root_text: str = "root"):
        self.root = Node(root_text)

    def insert(self, text: str, threshold_fn=default_threshold):
        v_new = Node(text)
        self._insert_node(self.root, v_new, 0, threshold_fn)

    def _insert_node(self, v: Node, v_new: Node, d: int, threshold_fn):
        leaf_created = False
        if not v.children:
            leaf = Node(v.text)
            v.children.append(leaf)
            leaf_created = True

        sims = compute_similarity(v_new.embedding, [c.embedding for c in v.children])
        best_idx = max(range(len(sims)), key=lambda i: sims[i])
        smax = sims[best_idx]
        v_best = v.children[best_idx]

        if smax >= threshold_fn(d):
            v.text = aggregate(v.text, v_new.text)
            v.embedding = embed(v.text)
            self._insert_node(v_best, v_new, d + 1, threshold_fn)
        else:
            if leaf_created:
                v.children.clear()  # 删除刚刚展开的 leaf
            v.children.append(v_new)


    def search(self, query: str):
        q_emb = embed(query)
        best, smax = None, -1
        stack = [self.root]
        while stack:
            node = stack.pop()
            s = cosine(q_emb, node.embedding)
            if s > smax:
                smax, best = s, node
            stack.extend(node.children)
        return best, smax

    def print_tree(self, node: Optional[Node]=None, depth: int=0):
        node = node or self.root
        print("  "*depth + f"- {node.text[:60]}")
        for ch in node.children:
            self.print_tree(ch, depth+1)

if __name__ == "__main__":
    st = SkillTree("root knowledge")
    st.insert("我想学习 C++ 的类和继承")
    st.insert("好的，你先学会类的基本语法：class A { public: int x; };")
    st.insert("在基类用 virtual 修饰函数，子类 override，实现多态")
    st.insert("如何实现多态？")
    st.insert("如何使用纯虚函数？")
    st.print_tree()
    q = "多态怎么实现？"
    n, s = st.search(q)
    print("\n查询：", q)
    print("最相似节点：", n.text)
    print("相似度：", round(s,4))
