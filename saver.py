# -*- coding: utf-8 -*-
"""保存模块：去重 + 重名处理 + 序号命名。"""
import hashlib
import json
import os
import threading

DEDUP_FILE = ".dedup.json"


class Saver:
    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        os.makedirs(self.root_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._seen = set()
        self._load_seen()

    def _dedup_path(self):
        return os.path.join(self.root_dir, DEDUP_FILE)

    def _load_seen(self):
        p = self._dedup_path()
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    self._seen = set(json.load(f))
            except Exception:
                self._seen = set()

    def _persist_seen(self):
        try:
            with open(self._dedup_path(), "w", encoding="utf-8") as f:
                json.dump(sorted(self._seen), f, ensure_ascii=False)
        except Exception:
            pass

    def _safe_name(self, name: str) -> str:
        return "".join(c for c in name if c not in '\\/:*?"<>|').strip() or "未命名"

    def save(self, image_bytes: bytes, ext: str, keyword: str, topic: str | None, seq: int):
        """保存图片，返回保存路径或 None（重复被跳过）。

        seq: 关键词内递增序号。目录结构为 关键词 / 主题 / 图片。
        """
        digest = hashlib.md5(image_bytes).hexdigest()
        with self._lock:
            if digest in self._seen:
                return None
            self._seen.add(digest)

        keyword = self._safe_name(keyword or "未命名关键词")
        folder = os.path.join(self.root_dir, keyword)
        os.makedirs(folder, exist_ok=True)

        base = f"{seq:02d}_{keyword}"
        path = os.path.join(folder, base + ext)
        counter = 1
        while os.path.exists(path):
            path = os.path.join(folder, f"{base}_{counter}{ext}")
            counter += 1

        with open(path, "wb") as f:
            f.write(image_bytes)
        self._persist_seen()
        return path
