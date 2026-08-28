# -*- coding: utf-8 -*-
"""跨会话长期记忆（真语义 RAG 版）。

方式：
- 记忆存于外部文件：memory.json 存规则、logs/history.json 存历史。
- 用火山方舟 embedding 模型（doubao-embedding-vision-251215）把文本转成 2048 维向量，
  用户目标也转向量，做余弦相似度排序，只取最相关的 top_k 条注入系统提示。
- 向量缓存于 memory_vectors.json，避免重复调用 embedding。
- 若 embedding 不可用（网络/模型报错），自动降级为本地 TF-IDF 字符向量检索。
- 配套：自动去重、自动遗忘（按使用次数与时间）。
"""
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timedelta

_BASE = os.path.dirname(os.path.abspath(__file__))
MEMORY_PATH = os.path.join(_BASE, "memory.json")
HISTORY_PATH = os.path.join(_BASE, "logs", "history.json")
VEC_PATH = os.path.join(_BASE, "memory_vectors.json")

DEFAULT_TOP_K_RULES = 4
DEFAULT_TOP_K_HISTORY = 3
DEDUP_THRESHOLD = 0.72        # 新增规则与已有规则的向量余弦超过该值则去重
STALE_DAYS = 90
STALE_MIN_USES = 2

DEFAULT_LEARNED = [
    "严格按用户主题采集，例如用户要「零食」就采集零食，不要采「鸟类」等无关主题。",
    "只保存 .jpg/.jpeg/.png，不要动图。",
    "单张图片大小限制在 1MB 以内。",
]

_STOP_CHARS = set(" \u7684\u4e86\u5728\u662f\u548c\u6709\u5c31\u90fd\u800c\u4e0e\u5bf9\u53ca\u6216\u4e00\u8fd9\u4e0d\u90a3\u4e5f\u6ca1\u4e5f\u6211\u4f60\u4ed6\u5979\u5b83\u7684\u7b49\u548c\u4e86")


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return default


def _save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _to_entry(rule):
    if isinstance(rule, str):
        return {"text": rule, "uses": 0, "last_used": None, "created": _now()}
    if isinstance(rule, dict):
        e = dict(rule)
        e.setdefault("text", "")
        e.setdefault("uses", 0)
        e.setdefault("last_used", None)
        e.setdefault("created", _now())
        return e
    return None


def load_memory():
    mem = _load_json(MEMORY_PATH, None)
    if mem is None:
        mem = {"learned": [], "last_forget": None}
        _save_json(MEMORY_PATH, mem)
    learned = mem.get("learned") or []
    mem["learned"] = [e for e in (_to_entry(x) for x in learned) if e and e["text"].strip()]
    mem.setdefault("last_forget", None)
    if not mem["learned"]:
        mem["learned"] = [_to_entry(t) for t in DEFAULT_LEARNED]
        _save_json(MEMORY_PATH, mem)
    return mem


def _load_history():
    hist = _load_json(HISTORY_PATH, [])
    return hist if isinstance(hist, list) else []


# ============ Embedding（真语义） ============
class Embedder:
    def __init__(self, base_url, api_key, model):
        self.base_url = (base_url or "").strip().rstrip("/")
        self.api_key = api_key
        self.model = model
        self._cache = _load_json(VEC_PATH, {})

    def available(self):
        return bool(self.base_url and self.api_key and self.model)

    def embed(self, texts):
        """返回 {text: [float]}，只请求未缓存的文本；失败返回已有缓存。"""
        result = {}
        texts = list(dict.fromkeys([t for t in texts if t]))
        missing = [t for t in texts if t not in self._cache]
        if missing and self.available():
            try:
                import requests as _requests
                url = self.base_url + "/embeddings"
                resp = _requests.post(
                    url,
                    json={"model": self.model, "input": missing},
                    headers={"Authorization": "Bearer " + self.api_key,
                             "Content-Type": "application/json"},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                for item in data.get("data", []):
                    idx = item.get("index")
                    emb = item.get("embedding")
                    if isinstance(emb, list) and idx is not None and 0 <= idx < len(missing):
                        self._cache[missing[idx]] = emb
                _save_json(VEC_PATH, self._cache)
            except Exception:
                pass
        for t in texts:
            if t in self._cache:
                result[t] = self._cache[t]
        return result


_EMBEDDER = None


def set_embedder(base_url, api_key, model):
    global _EMBEDDER
    _EMBEDDER = Embedder(base_url, api_key, model)


def _vec_cos(a, b):
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ============ 本地 TF-IDF（降级兜底） ============
def _tokens(text):
    chars = [c for c in text if not c.isspace() and c not in _STOP_CHARS]
    toks = list(chars)
    for i in range(len(chars) - 1):
        toks.append(chars[i] + chars[i + 1])
    return toks


def _idf(docs):
    n = max(len(docs), 1)
    df = Counter()
    for d in docs:
        df.update(set(_tokens(d)))
    return {t: math.log((n + 1) / (c + 1)) + 1 for t, c in df.items()}


def _vector(text, idf):
    vec = {}
    for term, f in _tf(text).items():
        vec[term] = f * idf.get(term, 1.0)
    return vec


def _tf(text):
    return Counter(_tokens(text))


def _rank_local(query, candidates):
    if not candidates:
        return []
    idf = _idf([query] + [c[0] for c in candidates])
    qv = _vector(query, idf)
    scored = []
    for text, extra in candidates:
        scored.append((_cosine_dict(qv, _vector(text, idf)), text, extra))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _cosine_dict(a, b):
    if not a or not b:
        return 0.0
    inter = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in inter)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


# ============ 检索 ============
def _semantic(goal, candidates):
    """用 embedding 做语义排序；失败则返回 None 触发降级。"""
    if not candidates or not _EMBEDDER or not _EMBEDDER.available():
        return None
    texts = [c[0] for c in candidates]
    vecs = _EMBEDDER.embed([goal] + texts)
    if goal not in vecs:
        return None
    gv = vecs[goal]
    scored = []
    for text, extra in candidates:
        v = vecs.get(text)
        if v:
            scored.append((_vec_cos(gv, v), text, extra))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def retrieve_rules(goal, top_k=DEFAULT_TOP_K_RULES):
    mem = load_memory()
    candidates = [(e["text"], e) for e in mem["learned"]]
    scored = _semantic(goal, candidates)
    if scored is None:
        scored = _rank_local(goal, candidates)
    return scored[:top_k]


def retrieve_history(goal, top_k=DEFAULT_TOP_K_HISTORY):
    hist = _load_history()
    candidates = []
    for h in hist[-200:]:
        g = (h.get("goal") or "").strip()
        if g:
            candidates.append((g, h))
    scored = _semantic(goal, candidates)
    if scored is None:
        scored = _rank_local(goal, candidates)
    return scored[:top_k]


# ============ 上下文 ============
def build_context(goal, prefs=None, top_k_rules=DEFAULT_TOP_K_RULES,
                  top_k_history=DEFAULT_TOP_K_HISTORY):
    prefs = prefs or {}
    lines = []
    if prefs:
        bits = []
        if prefs.get("provider"):
            bits.append("搜索源:" + str(prefs["provider"]))
        if prefs.get("count_per_keyword"):
            bits.append("默认每关键词" + str(prefs["count_per_keyword"]) + "张")
        ext = prefs.get("allowed_exts") or []
        if ext:
            bits.append("只保存" + "/".join(str(e).lstrip(".") for e in ext))
        if prefs.get("max_size_kb"):
            bits.append("单张" + str(prefs["max_size_kb"]) + "KB 以内")
        if bits:
            lines.append("【运行偏好】" + "；".join(bits) + "。")
    rules = retrieve_rules(goal, top_k_rules)
    if rules:
        lines.append("【最相关的长期规则】" + "；".join(t for _, t, _ in rules) + "。")
        _bump_uses([e for _, _, e in rules])
    hist = retrieve_history(goal, top_k_history)
    if hist:
        past = ["“" + t + "”(保存" + str((e.get("summary") or {}).get("saved", 0)) + "张)"
                for _, t, e in hist]
        lines.append("【最相关的历史】" + "；".join(past) + "。")
    return "\n".join(lines)


def _bump_uses(entries):
    mem = load_memory()
    by_text = {e["text"]: e for e in mem["learned"]}
    changed = False
    for e in entries:
        rec = by_text.get(e["text"])
        if rec is not None:
            rec["uses"] = int(rec.get("uses", 0)) + 1
            rec["last_used"] = _now()
            changed = True
    if changed:
        _save_json(MEMORY_PATH, mem)


# ============ 增删改 ============
def _overlap(a, b):
    ta = set(_tokens(a))
    tb = set(_tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def add_learned(rule):
    rule = (rule or "").strip()
    if not rule:
        return False
    mem = load_memory()
    existing = [e["text"] for e in mem["learned"]]
    if rule in existing:
        return False
    # 去重：语义余弦 或 重叠系数，二者任一超阈值即拒收
    semantic_ok = bool(_EMBEDDER and _EMBEDDER.available())
    vecs = _EMBEDDER.embed([rule] + existing) if semantic_ok else {}
    rv = vecs.get(rule) if semantic_ok else None
    for t in existing:
        if _overlap(rule, t) >= 0.75:
            return False
        if rv is not None:
            ev = vecs.get(t)
            if ev is not None and _vec_cos(rv, ev) >= DEDUP_THRESHOLD:
                return False
    mem["learned"].append(_to_entry(rule))
    _save_json(MEMORY_PATH, mem)
    return True


def remove_learned(rule):
    rule = (rule or "").strip()
    mem = load_memory()
    n = len(mem["learned"])
    mem["learned"] = [e for e in mem["learned"] if e["text"] != rule]
    if len(mem["learned"]) != n:
        _save_json(MEMORY_PATH, mem)
        return True
    return False


def forget_stale(stale_days=STALE_DAYS, min_uses=STALE_MIN_USES):
    mem = load_memory()
    if not mem.get("last_forget"):
        mem["last_forget"] = _now()
        _save_json(MEMORY_PATH, mem)
        return
    try:
        last = datetime.strptime(mem["last_forget"], "%Y-%m-%d %H:%M:%S")
    except Exception:
        last = datetime.min
    if (datetime.now() - last).days < 1:
        return
    cutoff = datetime.now() - timedelta(days=stale_days)
    kept = []
    for e in mem["learned"]:
        uses = int(e.get("uses", 0))
        lu = e.get("last_used")
        if uses < min_uses and lu:
            try:
                last_used = datetime.strptime(lu, "%Y-%m-%d %H:%M:%S")
            except Exception:
                last_used = datetime.min
            if last_used < cutoff:
                continue
        kept.append(e)
    mem["learned"] = kept
    mem["last_forget"] = _now()
    _save_json(MEMORY_PATH, mem)


def to_dict():
    mem = load_memory()
    return {
        "learned": [e["text"] for e in mem["learned"]],
        "recent": [
            {"goal": (h.get("goal") or "").strip(), "saved": (h.get("summary") or {}).get("saved", 0)}
            for h in _load_history()[-10:]
        ],
    }
