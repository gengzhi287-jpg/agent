# -*- coding: utf-8 -*-
"""任务运行状态：全局状态字典、锁、暂停/取消事件、日志缓冲。"""
import threading

_state = {
    "running": False,
    "paused": False,
    "tasks": {},          # keyword -> 状态
    "log": [],            # 运行日志（供界面显示）
    "started": None,
}
_state_lock = threading.Lock()
_pause_event = threading.Event()
_pause_event.set()  # 默认未暂停
_cancel_event = threading.Event()

_logger = None


def init_logger(logger):
    global _logger
    _logger = logger


def append_log(line: str):
    with _state_lock:
        _state["log"].append(line)
        if len(_state["log"]) > 500:
            _state["log"] = _state["log"][-500:]
    if _logger is not None:
        _logger.info(line)
