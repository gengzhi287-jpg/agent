# -*- coding: utf-8 -*-
"""Agent 决策链路追踪：把每一轮规划/工具调用/反思/总结写入 logs/traces/<run_id>.jsonl。"""
import json
import os
import threading
import time

_BASE = os.path.dirname(os.path.abspath(__file__))
TRACE_DIR = os.path.join(_BASE, "logs", "traces")

_lock = threading.Lock()


class Tracer:
    def __init__(self, run_id: str):
        os.makedirs(TRACE_DIR, exist_ok=True)
        safe = "".join(c for c in run_id if c not in '\\/:*?"<>|')
        self.path = os.path.join(TRACE_DIR, safe + ".jsonl")

    def event(self, kind: str, **data):
        rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "kind": kind}
        rec.update(data)
        line = json.dumps(rec, ensure_ascii=False)
        with _lock:
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass


def read_trace(run_id: str):
    """读取某次任务的 trace 事件列表；不存在返回空列表。"""
    safe = "".join(c for c in run_id if c not in '\\/:*?"<>|')
    path = os.path.join(TRACE_DIR, safe + ".jsonl")
    events = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if ln:
                        try:
                            events.append(json.loads(ln))
                        except Exception:
                            pass
        except Exception:
            pass
    return events
