# -*- coding: utf-8 -*-
"""公共工具函数。"""


def normalize_base_url(base_url: str) -> str:
    """归一化 Base URL：去掉末尾斜杠和误拼的 /chat/completions 后缀。"""
    u = (base_url or "").strip().rstrip("/")
    for suf in ("/chat/completions", "/v1/chat/completions"):
        if u.endswith(suf):
            u = u[: -len(suf)]
            break
    return u


def chat_completions_url(base_url: str) -> str:
    """把 Base URL 归一化后拼到 {base}/chat/completions。"""
    return normalize_base_url(base_url) + "/chat/completions"
