# -*- coding: utf-8 -*-
"""火山方舟豆包多模态分类模块：看图生成主题词。"""
import base64
import hashlib
import json
import os
import threading

from utils import chat_completions_url
import requests

_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "classify_cache.json")
_CACHE_MAX = 5000
_cache_lock = threading.Lock()
_cache = None


def _load_cache():
    global _cache
    if _cache is None:
        try:
            if os.path.exists(_CACHE_PATH):
                with open(_CACHE_PATH, "r", encoding="utf-8") as f:
                    _cache = json.load(f)
        except Exception:
            pass
        if not isinstance(_cache, dict):
            _cache = {}
    return _cache


def _cache_get(key):
    with _cache_lock:
        return _load_cache().get(key)


def _cache_put(key, value):
    with _cache_lock:
        cache = _load_cache()
        if len(cache) >= _CACHE_MAX:
            for k in list(cache.keys())[:1000]:
                cache.pop(k, None)
        cache[key] = value
        try:
            with open(_CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False)
        except Exception:
            pass


class ArkClassifier:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.api_key and self.model)

    def classify(self, image_bytes: bytes, mime: str = "image/jpeg") -> str | None:
        """识别图片内容，返回简短主题词；失败返回 None（进入未分类）。"""
        if not self.configured:
            return None
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {
                            "type": "text",
                            "text": (
                                "请用中文用一个简短的主题词或短语（2-6 个字，"
                                "不要标点、不要解释）概括这张图片的主要内容，"
                                "用于文件夹命名。只输出主题词。"
                            ),
                        },
                    ],
                }
            ],
        }
        url = chat_completions_url(self.base_url)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            topic = data["choices"][0]["message"]["content"].strip()
            topic = topic.strip('"\'。.,， ')
            topic = topic.replace("/", "_").replace("\\", "_")
            return topic or None
        except Exception:
            return None


    def analyze(self, image_bytes: bytes, mime: str, keyword: str):
        """识别图片并判断是否与关键词相关。

        返回 (topic, matched)：
          topic   —— 简短主题词；识别失败为 None。
          matched —— 是否与 keyword 相关；无法判断（接口/解析失败）时返回 True 放行，避免误杀。
        """
        if not self.configured or not keyword:
            return None, True
        digest = "a:" + hashlib.md5(image_bytes).hexdigest()
        cached = _cache_get(digest)
        if cached is not None:
            return (cached[0] or None), bool(cached[1])
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {
                            "type": "text",
                            "text": (
                                "这是用户想采集的关键词：『" + keyword + "』。\n"
                                "请识别这张图片的内容，并判断它是否与这个关键词相关"
                                "（即图片主体内容是否就是关于该关键词的，无关广告、无关配图、无关其他主题都不算）。\n"
                                "请只输出一个 JSON 对象，不要任何其他文字或解释，格式如下：\n"
                                '{"topic": "简短中文主题词(2-6字)", "match": true或false}'
                            ),
                        },
                    ],
                }
            ],
        }
        url = chat_completions_url(self.base_url)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"].strip()
            import json as _json
            obj = None
            try:
                obj = _json.loads(raw)
            except Exception:
                import re as _re
                m = _re.search(r'\{.*?\}', raw, _re.S)
                if m:
                    try:
                        obj = _json.loads(m.group(0))
                    except Exception:
                        obj = None
            if not isinstance(obj, dict):
                _cache_put(digest, [None, True])
                return None, True
            topic = str(obj.get("topic") or "").strip()
            topic = topic.strip('"\'。.,， ')
            topic = topic.replace("/", "_").replace("\\", "_")
            matched = obj.get("match")
            if not isinstance(matched, bool):
                _cache_put(digest, [topic or None, True])
                return topic or None, True
            _cache_put(digest, [topic or None, matched])
            return (topic or None), matched
        except Exception:
            return None, True
