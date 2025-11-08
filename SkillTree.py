from typing import List, Tuple, Optional
import math

# Lightweight embedding wiring: you can swap providers in embeddings.py
try:
    from embeddings import get_embedder
except Exception:
    # Fallback import if running from a different CWD
    import os, sys
    sys.path.append(os.path.dirname(__file__))
    from embeddings import get_embedder


class SkillTree:
    def __init__(self):
        self.nodelist: List["Node"] = []
        self.root: Optional["Node"] = None


# Provide a module-level tree to avoid NameErrors in placeholder helpers
skill_tree = SkillTree()

class Node:
    def __init__(self, string: str, time_stamp: float, embedding: List[float]):
        self.string = string
        self.time_stamp = time_stamp
        self.embedding = embedding
        self.children: List["Node"] = []

def embed(text: str) -> List[float]:
    """Embed text using the configured provider (see embeddings.py).

    Returns a dense vector list[float].
    """
    return get_embedder().embed(text)

def put_node(cnew: str, root: "Node", threshold_function, time_stamp: float = 0.0):
    enew = embed(cnew)  # Step 1: Embed the new information
    vnew = Node(cnew, time_stamp, enew)
    attached = insert_node(root, vnew, 0, threshold_function)  # Step 2: Insert the node
    # Track in flat list only if it was attached as a distinct node
    if attached and (vnew not in skill_tree.nodelist):
        skill_tree.nodelist.append(vnew)

def insert_node(v: "Node", vnew: "Node", d: int, threshold_function) -> bool:
    if not v.children:
        # Convert leaf into an internal node so v and vnew can be compared at the same level
        existing_child = Node(v.string, v.time_stamp, v.embedding)
        v.children.append(existing_child)

    si = compute_similarity(vnew.embedding, [child.embedding for child in v.children])
    vbest, smax = find_best_child(v.children, si)

    if smax >= threshold_function(d):
        aggregated_content = aggregate(vbest.string, vnew.string)
        vbest.string = aggregated_content
        vbest.embedding = embed(aggregated_content)
        # We merged vnew into vbest; stop routing and report no new node attached
        return False
    else:
        v.children.append(vnew)
        return True

def compute_similarity(enew: List[float], embeddings: List[List[float]]) -> List[float]:
    """Compute cosine similarity between a query embedding and a list of embeddings."""
    def _cos(a: List[float], b: List[float]) -> float:
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a))
        nb = math.sqrt(sum(y*y for y in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)

    return [_cos(enew, e) for e in embeddings]

def find_best_child(children: List["Node"], similarities: List[float]) -> Tuple["Node", float]:
    max_index = similarities.index(max(similarities))
    return children[max_index], similarities[max_index]

def aggregate(c1: str, c2: str) -> str:
    # Simple aggregation; replace with your summarizer if needed
    return c1 + "\n" + c2

def cut_tree(threshold: float):
    skill_tree.nodelist = [node for node in skill_tree.nodelist if node.time_stamp <= threshold]

def default_threshold(depth: int) -> float:
    """Default similarity threshold schedule by depth.
    Starts strict and relaxes slightly with depth.
    """
    return max(0.60, 0.85 - 0.05 * depth)


def initialize_skill_tree(root_text: str = "root", time_stamp: float = 0.0):
    """Initialize the global skill_tree with a root node if not present."""
    if skill_tree.root is None:
        root_embedding = embed(root_text)
        root = Node(root_text, time_stamp, root_embedding)
        skill_tree.root = root
        skill_tree.nodelist = [root]


def add_text(text: str, time_stamp: float = 0.0, threshold_function=default_threshold) -> Node:
    """High-level helper to add text into the tree.

    Ensures the tree is initialized and inserts the new node according to the
    hierarchical similarity routing.
    """
    if skill_tree.root is None:
        initialize_skill_tree()
    put_node(text, skill_tree.root, threshold_function, time_stamp=time_stamp)
    return skill_tree.nodelist[-1]


def search(task: str):
    """Return (path_to_best_node, children_of_best_node) for the given task text.

    Strategy: compute the query embedding, find the most similar node across the
    flat list (nodelist) for robustness, then reconstruct the path via DFS.
    """
    if skill_tree.root is None or not skill_tree.nodelist:
        return [], []

    q_emb = embed(task)
    sims = compute_similarity(q_emb, [n.embedding for n in skill_tree.nodelist])
    best_idx = sims.index(max(sims))
    best_node = skill_tree.nodelist[best_idx]
    path = get_path_to_node(skill_tree.root, best_node)
    return path, get_children_of_node(best_node)

def compare_task_with_node(task: str, node: "Node", threshold: float = 0.70) -> bool:
    q_emb = embed(task)
    s = compute_similarity(q_emb, [node.embedding])[0]
    return s >= threshold

def get_path_to_node(root: "Node", target_node: "Node") -> List["Node"]:
    """Return list of nodes from root to target (inclusive). Empty if not found."""
    path: List[Node] = []

    def _dfs(cur: Node) -> bool:
        path.append(cur)
        if cur is target_node:
            return True
        for ch in cur.children:
            if _dfs(ch):
                return True
        path.pop()
        return False

    if root and _dfs(root):
        return path
    return []

def get_children_of_node(node: "Node") -> List["Node"]:
    return node.children if node else []

def continue_searching(children: List["Node"]):
    # Kept for backward compatibility; not used in new search.
    return []

def find_task(context: str):
    # Minimal initial version: identity mapping
    return abstract_solution(context)

def abstract_solution(context: str):
    # Initial simple heuristic: use raw context as task.
    return context.strip()

def use_skill_time():
    user_input = get_user_input()
    task = find_task(user_input)
    a2s = search(task)
    interact_with_agent(a2s)

def get_user_input():
    # Placeholder for getting user input logic
    pass

def interact_with_agent(a2s):
    # Placeholder for agent interaction logic
    pass

