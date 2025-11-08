"""
Token accounting helpers for Chat and Embedding use cases.

Features:
- Precise token counts for OpenAI chat messages via tiktoken (ChatML rules)
- Precise token counts for OpenAI embeddings via tiktoken
- Precise token counts for local/HF embeddings via transformers tokenizers
- Budget helpers to estimate cost with per-1K token pricing

Usage examples are at the bottom of this file.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Any
import os


# -------------------------
# Optional dependencies
# -------------------------
_tiktoken = None
_transformers = None

try:
    import tiktoken  # type: ignore
    _tiktoken = tiktoken
except Exception:
    _tiktoken = None

try:
    from transformers import AutoTokenizer  # type: ignore
    _transformers = True
except Exception:
    _transformers = False


# -------------------------
# Utilities
# -------------------------
def _ensure_tiktoken():
    if _tiktoken is None:
        raise RuntimeError(
            "tiktoken not installed. Install via `pip install tiktoken` to count OpenAI tokens."
        )


def _is_hf_model_id(model: str) -> bool:
    """Heuristic: HF models usually contain '/' or look like 'intfloat/e5-...'"""
    return "/" in model or model.lower().startswith(("intfloat", "baai", "jinaai", "mixedbread"))


def _get_openai_encoding(model: str):
    _ensure_tiktoken()
    try:
        return _tiktoken.encoding_for_model(model)
    except Exception:
        # Most modern OpenAI chat/embedding models use cl100k_base
        return _tiktoken.get_encoding("cl100k_base")


# -------------------------
# Embedding token counting
# -------------------------
def count_embedding_tokens(texts: List[str], model: str) -> Tuple[int, List[int]]:
    """Returns (total_tokens, per_item_tokens) for embedding inputs.

    - For OpenAI models (e.g., text-embedding-3-large), uses tiktoken.
    - For HF/local models (e.g., intfloat/e5-large-v2), uses transformers tokenizer.
    """
    if not texts:
        return 0, []

    if _is_hf_model_id(model) and _transformers:
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        per_item = [len(tok.encode(t, truncation=False)) for t in texts]
        return sum(per_item), per_item
    else:
        # Assume OpenAI tokenizer
        enc = _get_openai_encoding(model)
        per_item = [len(enc.encode(t)) for t in texts]
        return sum(per_item), per_item


# -------------------------
# Chat token counting (OpenAI ChatML-like)
# -------------------------
_CHAT_RULES = {
    # Default modern rule-of-thumb
    "default": {"tokens_per_message": 3, "tokens_per_name": 1, "priming": 0},
    # Historical specifics (kept for completeness)
    "gpt-3.5-turbo-0301": {"tokens_per_message": 4, "tokens_per_name": -1, "priming": 0},
}


def _rules_for_model(model: str) -> Dict[str, int]:
    # Newer models tend to follow the "default" 3/1 rule; keep a few overrides if needed
    if model in _CHAT_RULES:
        return _CHAT_RULES[model]
    # Normalize families
    ml = model.lower()
    if any(k in ml for k in ["gpt-4o", "gpt-4", "gpt-3.5-turbo", "o-mini", "o1", "o3"]):
        return _CHAT_RULES["default"]
    # Fallback
    return _CHAT_RULES["default"]


def count_chat_message_tokens(messages: List[Dict[str, Any]], model: str) -> Tuple[int, Dict[str, int]]:
    """Count tokens for chat messages (prompt side only).

    messages: OpenAI-style messages, e.g. [{"role":"system","content":"..."}, ...]
    Returns: (total_prompt_tokens, breakdown)
    breakdown includes per-field aggregates.
    """
    _ensure_tiktoken()
    enc = _get_openai_encoding(model)
    rules = _rules_for_model(model)

    tokens_per_message = rules["tokens_per_message"]
    tokens_per_name = rules["tokens_per_name"]
    priming = rules.get("priming", 0)

    total = priming
    role_tokens = 0
    name_meta_tokens = 0
    content_tokens = 0

    for m in messages:
        # base per-message overhead
        total += tokens_per_message

        role = str(m.get("role", ""))
        if role:
            r = enc.encode(role)
            role_tokens += len(r)
            total += len(r)

        name = m.get("name")
        if name:
            total += tokens_per_name
            name_meta_tokens += tokens_per_name

        content = m.get("content", "")
        if isinstance(content, str):
            ct = enc.encode(content)
            content_tokens += len(ct)
            total += len(ct)
        elif isinstance(content, list):
            # Some models accept structured content parts
            for part in content:
                if isinstance(part, dict):
                    val = part.get("text") or part.get("content") or ""
                else:
                    val = str(part)
                ct = enc.encode(str(val))
                content_tokens += len(ct)
                total += len(ct)
        else:
            # Fallback stringify
            ct = enc.encode(str(content))
            content_tokens += len(ct)
            total += len(ct)

    breakdown = {
        "priming": priming,
        "role_tokens": role_tokens,
        "name_meta_tokens": name_meta_tokens,
        "content_tokens": content_tokens,
        "per_message_overhead_total": tokens_per_message * len(messages),
    }
    return total, breakdown


def count_completion_tokens(text: str, model: str) -> int:
    """Count tokens for the assistant's completion text (post-call or estimation)."""
    _ensure_tiktoken()
    enc = _get_openai_encoding(model)
    return len(enc.encode(text))


# -------------------------
# Cost helpers
# -------------------------
def estimate_embedding_cost_usd(total_tokens: int, price_per_1k: float) -> float:
    return (total_tokens / 1000.0) * price_per_1k


def estimate_chat_cost_usd(prompt_tokens: int, completion_tokens: int, price_prompt_per_1k: float, price_completion_per_1k: float) -> float:
    return (prompt_tokens / 1000.0) * price_prompt_per_1k + (completion_tokens / 1000.0) * price_completion_per_1k


# -------------------------
# Convenience helpers (truncate and plan by budget)
# -------------------------
def truncate_to_token_limit(text: str, model: str, max_tokens: int) -> Tuple[str, int]:
    """Truncate a single text to max_tokens for the given model and return (truncated_text, token_count).

    - Uses tiktoken for OpenAI models; transformers for HF/local models.
    - Returned token_count is the length after truncation.
    """
    if max_tokens <= 0:
        return "", 0

    if _is_hf_model_id(model) and _transformers:
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        ids = tok.encode(text, truncation=False)
        if len(ids) <= max_tokens:
            return text, len(ids)
        trimmed = tok.decode(ids[:max_tokens], skip_special_tokens=True)
        # Re-count precisely after decode
        new_len = len(tok.encode(trimmed, truncation=False))
        return trimmed, new_len
    else:
        enc = _get_openai_encoding(model)
        ids = enc.encode(text)
        if len(ids) <= max_tokens:
            return text, len(ids)
        trimmed = enc.decode(ids[:max_tokens])
        new_len = len(enc.encode(trimmed))
        return trimmed, new_len


def truncate_list_with_stats(texts: List[str], model: str, max_tokens_per_item: int) -> Dict[str, Any]:
    """Truncate each text to max_tokens_per_item and return stats.

    Returns a dict with:
    - truncated_texts: List[str]
    - per_item_tokens: List[int]
    - total_tokens: int
    """
    truncated: List[str] = []
    per_item: List[int] = []
    total = 0
    for t in texts:
        s, n = truncate_to_token_limit(t, model, max_tokens_per_item)
        truncated.append(s)
        per_item.append(n)
        total += n
    return {
        "truncated_texts": truncated,
        "per_item_tokens": per_item,
        "total_tokens": total,
    }


def plan_batch_under_budget(
    texts: List[str],
    model: str,
    max_tokens_per_item: int,
    *,
    max_total_tokens: Optional[int] = None,
    max_total_cost_usd: Optional[float] = None,
    price_per_1k: Optional[float] = None,
) -> Dict[str, Any]:
    """Select the largest prefix of texts that fits within a token or cost budget.

    Each text is first truncated to max_tokens_per_item, then we accumulate until
    the budget would be exceeded (the last item that exceeds the budget is not included).

    One of (max_total_tokens) or (max_total_cost_usd + price_per_1k) must be provided.

    Returns dict with:
    - selected_indices: List[int] (indices in the original list)
    - selected_texts: List[str] (truncated)
    - per_item_tokens: List[int]
    - total_tokens: int
    - estimated_cost_usd: Optional[float]
    - budget_type: str ("tokens" or "cost")
    """
    if max_total_tokens is None and max_total_cost_usd is None:
        raise ValueError("Provide either max_total_tokens or max_total_cost_usd.")
    if max_total_cost_usd is not None and (price_per_1k is None):
        raise ValueError("When using cost budget, price_per_1k is required.")

    # Pre-truncate all texts
    trunc = truncate_list_with_stats(texts, model, max_tokens_per_item)
    truncated_texts: List[str] = trunc["truncated_texts"]
    per_item_tokens: List[int] = trunc["per_item_tokens"]

    selected_idx: List[int] = []
    selected_txts: List[str] = []
    selected_tokens: List[int] = []
    running_tokens = 0

    def would_exceed(next_tokens: int) -> bool:
        new_total = running_tokens + next_tokens
        if max_total_tokens is not None and new_total > max_total_tokens:
            return True
        if max_total_cost_usd is not None and price_per_1k is not None:
            est_cost = estimate_embedding_cost_usd(new_total, price_per_1k)
            if est_cost > max_total_cost_usd:
                return True
        return False

    for i, (txt, tok) in enumerate(zip(truncated_texts, per_item_tokens)):
        if would_exceed(tok):
            break
        selected_idx.append(i)
        selected_txts.append(txt)
        selected_tokens.append(tok)
        running_tokens += tok

    estimated_cost = None
    budget_type = "tokens" if max_total_tokens is not None else "cost"
    if price_per_1k is not None:
        estimated_cost = estimate_embedding_cost_usd(running_tokens, price_per_1k)

    return {
        "selected_indices": selected_idx,
        "selected_texts": selected_txts,
        "per_item_tokens": selected_tokens,
        "total_tokens": running_tokens,
        "estimated_cost_usd": estimated_cost,
        "budget_type": budget_type,
    }


# -------------------------
# Examples (for quick manual testing)
# -------------------------
if __name__ == "__main__":
    # Embedding example
    texts = ["你好，世界", "This is a test for tokens counting."]
    emb_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    total_emb_tokens, per_item = count_embedding_tokens(texts, emb_model)
    print(f"[Embedding] model={emb_model} total_tokens={total_emb_tokens} per_item={per_item}")

    # Chat example
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "请用中文回答：介绍一下Transformer。"},
    ]
    chat_model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    prompt_tokens, bd = count_chat_message_tokens(messages, chat_model)
    # Suppose we expect 200 tokens completion
    completion_tokens = 200
    print(f"[Chat] model={chat_model} prompt_tokens={prompt_tokens} breakdown={bd}")
    # Example price placeholders; replace with official pricing
    prompt_price = float(os.getenv("CHAT_PRICE_PROMPT_PER_1K", "0.0005"))
    comp_price = float(os.getenv("CHAT_PRICE_COMPLETION_PER_1K", "0.0015"))
    estimated = estimate_chat_cost_usd(prompt_tokens, completion_tokens, prompt_price, comp_price)
    print(f"[Chat] estimated_cost_usd={estimated:.6f} (with completion_tokens={completion_tokens})")
