# -*- coding: utf-8 -*-
"""素材爬取 Agent —— Flask 后端。"""
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from datetime import datetime
from urllib.parse import quote
from flask import Flask, Response, jsonify, render_template, request, send_from_directory

import agent
import crawler
import memory as memory_mod
import pipeline
import task_state as ts
import tracer
from classifier import ArkClassifier
from logger_setup import setup_logging
from saver import DEDUP_FILE, Saver

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
memory_mod.set_embedder(CONFIG["llm"].get("base_url", ""), CONFIG["llm"].get("api_key", ""),
                        CONFIG["llm"].get("embedding_model", "doubao-embedding-vision-251215"))
SAVE_CFG = CONFIG["save"]
CRAWL_CFG = CONFIG["crawl"]
LLM_CFG = CONFIG["llm"]
SEARCH_CFG = CONFIG.get("search", {})
FILTER_CFG = CONFIG.get("filter", {})
MIN_WIDTH = int(FILTER_CFG.get("min_width", 0) or 0)
MIN_HEIGHT = int(FILTER_CFG.get("min_height", 0) or 0)
EXCLUDE_TOPICS = set(FILTER_CFG.get("exclude_topics", []) or [])
LOG_DIR = os.path.join(BASE_DIR, CONFIG["logs"]["dir"])

logger = setup_logging(LOG_DIR)
root_dir = os.path.join(BASE_DIR, SAVE_CFG["root_dir"])
saver = Saver(root_dir)
ts.init_logger(logger)
pipeline.init(saver, SAVE_CFG, CRAWL_CFG, SEARCH_CFG, FILTER_CFG, root_dir)

HISTORY_PATH = os.path.join(LOG_DIR, "history.json")
_history_lock = threading.Lock()


def _load_history():
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_history(history):
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# 任务状态 / 锁 / 事件 / 日志缓冲已迁移至 task_state 模块
_task_queue = []          # 排队任务 [(goal, llm, search_keys)]
_queue_lock = threading.Lock()

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/config")
def api_config():
    return jsonify({
        "models": LLM_CFG.get("available_models", []),
        "default_model": LLM_CFG.get("model"),
        "default_base_url": LLM_CFG.get("base_url"),
        "presets": [
            {
                "name": p.get("name"),
                "base_url": p.get("base_url"),
                "model": p.get("model"),
            }
            for p in LLM_CFG.get("presets", [])
        ],
        "has_key": bool(LLM_CFG.get("api_key")),
        "count_default": int(SAVE_CFG.get("count_per_keyword", 20)),
        "max_concurrency": int(CRAWL_CFG.get("max_concurrency", 4)),
        "root_dir": root_dir,
        "search_provider": SEARCH_CFG.get("provider", "bing"),
        "search_providers": {
            "pixabay": bool(SEARCH_CFG.get("pixabay", {}).get("api_key")),
            "pexels": bool(SEARCH_CFG.get("pexels", {}).get("api_key")),
        },
    })


@app.route("/api/status")
def api_status():
    with ts._state_lock:
        run_id = (ts._state.get("run") or {}).get("id")
        payload = {
            "running": ts._state["running"],
            "queue": len(_task_queue),
            "paused": ts._state["paused"],
            "agent_final": ts._state.get("agent_final"),
            "tasks": ts._state["tasks"],
            "log": list(ts._state["log"]),
        }
    if run_id:
        payload["trace_run_id"] = run_id
        payload["trace_tail"] = tracer.read_trace(run_id)[-80:]
    return jsonify(payload)


def _wait_all(futures, on_done):
    for f in as_completed(futures):
        try:
            f.result()
        except Exception as e:
            ts.append_log(f"任务异常: {e}")
    on_done()


@app.route("/api/pause", methods=["POST"])
def api_pause():
    with ts._state_lock:
        if not ts._state["running"] or ts._state["paused"]:
            return jsonify({"ok": False, "error": "当前无运行中任务可暂停"}), 400
        ts._state["paused"] = True
    ts._pause_event.clear()
    ts.append_log("任务已暂停")
    return jsonify({"ok": True})


@app.route("/api/resume", methods=["POST"])
def api_resume():
    with ts._state_lock:
        if not ts._state["running"] or not ts._state["paused"]:
            return jsonify({"ok": False, "error": "当前无暂停任务可恢复"}), 400
        ts._state["paused"] = False
    ts._pause_event.set()
    ts.append_log("任务已恢复")
    return jsonify({"ok": True})


@app.route("/api/cancel", methods=["POST"])
def api_cancel():
    with ts._state_lock:
        if not ts._state["running"]:
            return jsonify({"ok": False, "error": "当前无运行中任务可取消"}), 400
        ts._state["paused"] = False
    ts._cancel_event.set()
    ts._pause_event.set()  # 让处于暂停的线程解除阻塞并检查取消
    ts.append_log("正在取消任务...")
    return jsonify({"ok": True})



def _report_lines(report):
    lines = []
    lines.append("素材爬取结果报告")
    lines.append("完成时间: " + report.get("finished_at", ""))
    lines.append("-" * 40)
    items = report.get("items", [])
    tot_dl = sum(i.get("downloaded", 0) for i in items)
    tot_sv = sum(i.get("saved", 0) for i in items)
    tot_fa = sum(i.get("failed", 0) for i in items)
    lines.append(f"任务数: {len(items)}  已下载: {tot_dl}  已保存: {tot_sv}  失败: {tot_fa}")
    lines.append("-" * 40)
    for i in items:
        lines.append(f"[{i.get('keyword')}] 状态={i.get('status')} "
                     f"下载={i.get('downloaded')} 保存={i.get('saved')} 失败={i.get('failed')}")
        if i.get("folder"):
            lines.append(f"  保存目录: {i.get('folder')}")
        errs = i.get("errors") or {}
        if errs:
            for k, v in errs.items():
                lines.append(f"  失败原因[{k}]: {v}")
    return lines


@app.route("/api/report")
def api_report():
    with ts._state_lock:
        return jsonify(ts._state.get("report"))


@app.route("/api/export")
def api_export():
    with ts._state_lock:
        report = ts._state.get("report")
    if not report:
        return jsonify({"ok": False, "error": "暂无报告"}), 400
    text = "\n".join(_report_lines(report))
    return Response(
        text,
        mimetype="text/plain; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=report.txt"},
    )



@app.route("/api/history")
def api_history():
    with _history_lock:
        history = _load_history()
    out = []
    for h in reversed(history):
        out.append({
            "id": h.get("id"),
            "started_at": h.get("started_at"),
            "finished_at": h.get("finished_at"),
            "keywords": h.get("keywords"),
            "urls": h.get("urls"),
            "summary": h.get("summary"),
        })
    return jsonify(out)


@app.route("/api/history/<rid>")
def api_history_detail(rid):
    with _history_lock:
        history = _load_history()
    for h in history:
        if h.get("id") == rid:
            return jsonify(h)
    return jsonify({"ok": False, "error": "记录不存在"}), 404


@app.route("/api/trace/<rid>")
def api_trace(rid):
    return jsonify({"events": tracer.read_trace(rid)})


@app.route("/api/memory", methods=["GET", "POST", "DELETE"])
def api_memory():
    if request.method == "GET":
        return jsonify(memory_mod.to_dict())
    payload = request.get_json(force=True, silent=True) or {}
    rule = str(payload.get("rule") or "").strip()
    if not rule:
        return jsonify({"ok": False, "error": "缺少规则内容"}), 400
    if request.method == "POST":
        ok = memory_mod.add_learned(rule)
        return jsonify({"ok": ok, "learned": memory_mod.to_dict()["learned"]})
    if request.method == "DELETE":
        ok = memory_mod.remove_learned(rule)
        return jsonify({"ok": ok, "learned": memory_mod.to_dict()["learned"]})


@app.route("/media/<path:filepath>")
def media(filepath):
    return send_from_directory(root_dir, filepath)


def _browse_tree():
    tree = {}
    if os.path.isdir(root_dir):
        for kw in sorted(os.listdir(root_dir)):
            if kw.startswith(".") or kw == DEDUP_FILE:
                continue
            kwp = os.path.join(root_dir, kw)
            if not os.path.isdir(kwp):
                continue
            topics = {}
            direct = []
            for entry in sorted(os.listdir(kwp)):
                ep = os.path.join(kwp, entry)
                if os.path.isdir(ep):
                    imgs = []
                    for fn in sorted(os.listdir(ep)):
                        fp = os.path.join(ep, fn)
                        if os.path.isfile(fp):
                            imgs.append("/media/" + quote(kw) + "/" + quote(entry) + "/" + quote(fn))
                    if imgs:
                        topics[entry] = imgs
                elif os.path.isfile(ep):
                    direct.append("/media/" + quote(kw) + "/" + quote(entry))
            if direct:
                topics[""] = direct
            if topics:
                tree[kw] = topics
    return tree


@app.route("/api/browse")
def api_browse():
    return jsonify({"tree": _browse_tree()})


def _finalize():
    with ts._state_lock:
        ts._state["running"] = False
        ts._state["paused"] = False
        items = [
            {
                "keyword": t.get("keyword"),
                "status": t.get("status"),
                "downloaded": t.get("downloaded", 0),
                "saved": t.get("saved", 0),
                "failed": t.get("failed", 0),
                "errors": t.get("errors", {}),
                "folder": t.get("folder", ""),
            }
            for t in ts._state["tasks"].values()
        ]
        ts._state["report"] = {
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "items": items,
        }
        run = ts._state.get("run")
        if run:
            run["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            run["items"] = items
            run["summary"] = {
                "downloaded": sum(i.get("downloaded", 0) for i in items),
                "saved": sum(i.get("saved", 0) for i in items),
                "failed": sum(i.get("failed", 0) for i in items),
                "tasks": len(items),
            }
            # token 用量统计（汇总 trace 中各次 LLM 调用的 usage）
            try:
                usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                for e in tracer.read_trace(run["id"]):
                    if e.get("kind") == "llm" and isinstance(e.get("usage"), dict):
                        usage["prompt_tokens"] += int(e["usage"].get("prompt_tokens") or 0)
                        usage["completion_tokens"] += int(e["usage"].get("completion_tokens") or 0)
                        usage["total_tokens"] += int(e["usage"].get("total_tokens") or 0)
                run["usage"] = usage
            except Exception:
                pass
            with _history_lock:
                history = _load_history()
                history.append(run)
                _save_history(history)
    ts._cancel_event.clear()
    ts._pause_event.set()
    ts.append_log("任务结束")


def collect_one(keyword, count, provider, search_keys, model, base_url, api_key, filters=None):
    """供 Agent 工具调用：采集单个关键词，返回该关键词结果汇总。"""
    return pipeline.process_keyword(keyword, count, int(CRAWL_CFG.get("max_concurrency", 4)), model, base_url, api_key, provider, search_keys, filters)


def collect_url(page_url, count, model, base_url, api_key, filters=None):
    """供 Agent 工具调用：从网页提取图片采集，返回结果汇总。"""
    return pipeline.process_url(page_url, count, model, base_url, api_key, filters)


def _filters_from_args(args):
    return {
        "min_width": int(args.get("min_width") or 0),
        "min_height": int(args.get("min_height") or 0),
        "exclude_topics": args.get("exclude_topics") or [],
    }


def _agent_dispatch(name, args, llm, search_keys):
    flt = _filters_from_args(args)
    if name == "collect_keyword":
        kw = str(args.get("keyword") or "").strip()
        if not kw:
            return {"error": "缺少关键词"}
        cnt = int(args.get("count") or SAVE_CFG.get("count_per_keyword", 20))
        source = str(args.get("source") or SEARCH_CFG.get("provider", "bing"))
        return collect_one(kw, cnt, source, search_keys, llm["model"], llm["base_url"], llm["api_key"], flt)
    if name == "collect_url":
        url = str(args.get("url") or "").strip()
        if not url:
            return {"error": "缺少 URL"}
        cnt = int(args.get("count") or SAVE_CFG.get("count_per_keyword", 20))
        return collect_url(url, cnt, llm["model"], llm["base_url"], llm["api_key"], flt)
    if name == "list_saved":
        tree = _browse_tree()
        return {k: {tp: len(imgs) for tp, imgs in topics.items()} for k, topics in tree.items()}
    return {"error": "未知工具: " + name}


def _agent_worker(goal, llm, search_keys):
    def on_event(kind, payload):
        if kind == "tool":
            ts.append_log("[Agent] 调用工具: " + payload)
        elif kind == "plan":
            ts.append_log("[Agent] 采集计划: " + json.dumps(payload["steps"], ensure_ascii=False)
                          + ("（" + payload["reason"] + "）" if payload.get("reason") else ""))
        elif kind == "reflect":
            ts.append_log("[Agent] 反思: " + str(payload))
        elif kind == "replan":
            ts.append_log("[Agent] 重规划: " + json.dumps(payload["steps"], ensure_ascii=False))
        elif kind == "error":
            ts.append_log("[Agent] 出错: " + payload)
        elif kind == "final":
            ts.append_log("[Agent] 总结:\n" + payload)
            with ts._state_lock:
                ts._state["agent_final"] = payload

    prefs = {
        "provider": SEARCH_CFG.get("provider"),
        "count_per_keyword": SAVE_CFG.get("count_per_keyword"),
        "allowed_exts": SAVE_CFG.get("allowed_exts"),
        "max_size_kb": SAVE_CFG.get("max_size_kb"),
    }
    memory_mod.forget_stale()
    mem_ctx = memory_mod.build_context(goal, prefs)
    with ts._state_lock:
        run_id = (ts._state.get("run") or {}).get("id")
    tr = tracer.Tracer(run_id or datetime.now().strftime("%Y%m%d_%H%M%S"))
    agent.run_agent_v2(goal, llm["base_url"], llm["api_key"], llm["model"],
                       lambda n, a: _agent_dispatch(n, a, llm, search_keys), on_event,
                       max_replans=int(CONFIG.get("agent", {}).get("max_replans", 3)),
                       memory_context=mem_ctx, tracer=tr,
                       count_default=int(SAVE_CFG.get("count_per_keyword", 20)))
    _finalize()
    _start_next_from_queue()


def _reset_run_state(goal, llm):
    with ts._state_lock:
        ts._state["running"] = True
        ts._state["paused"] = False
        ts._state["tasks"] = {}
        ts._state["log"] = []
        ts._state["started"] = None
        ts._state["report"] = None
        ts._state["run"] = {
            "id": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "goal": goal,
            "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "finished_at": None,
            "keywords": [],
            "urls": [],
            "count": 0,
            "concurrency": 1,
            "model": llm["model"],
            "base_url": llm["base_url"],
            "items": [],
            "summary": {},
        }
        ts._state["agent_final"] = None
    ts._cancel_event.clear()
    ts._pause_event.set()
    ts.append_log("[Agent] 目标: " + goal)


def _start_next_from_queue():
    with _queue_lock:
        if not _task_queue:
            return
        goal, llm, search_keys = _task_queue.pop(0)
        remaining = len(_task_queue)
    ts.append_log(f"[队列] 开始下一个排队任务（还剩 {remaining} 个）")
    _reset_run_state(goal, llm)
    threading.Thread(target=_agent_worker, args=(goal, llm, search_keys), daemon=True).start()


@app.route("/api/agent/start", methods=["POST"])
def api_agent_start():
    payload = request.get_json(force=True)
    goal = str(payload.get("goal") or "").strip()
    if not goal:
        return jsonify({"ok": False, "error": "请描述想收集什么素材"}), 400

    preset = None
    preset_name = payload.get("preset")
    for p in LLM_CFG.get("presets", []):
        if p.get("name") == preset_name:
            preset = p
            break
    llm = {
        "model": payload.get("model") or (preset or {}).get("model") or LLM_CFG["model"],
        "base_url": payload.get("base_url") or (preset or {}).get("base_url") or LLM_CFG["base_url"],
        "api_key": payload.get("api_key") or (preset or {}).get("api_key") or LLM_CFG.get("api_key", ""),
    }
    if not (llm["base_url"] and llm["api_key"]):
        return jsonify({"ok": False, "error": "请先在「设置模型」配置 Base URL 和 API Key"}), 400

    search_keys = {
        "pixabay": SEARCH_CFG.get("pixabay", {}).get("api_key", ""),
        "pexels": SEARCH_CFG.get("pexels", {}).get("api_key", ""),
    }

    with ts._state_lock:
        already_running = ts._state["running"]
    if already_running:
        with _queue_lock:
            _task_queue.append((goal, llm, search_keys))
            position = len(_task_queue)
        ts.append_log(f"[队列] 已有任务运行中，加入队列（第 {position} 位）: " + goal)
        return jsonify({"ok": True, "queued": True, "position": position})
    _reset_run_state(goal, llm)
    threading.Thread(target=_agent_worker, args=(goal, llm, search_keys), daemon=True).start()
    return jsonify({"ok": True, "queued": False})


if __name__ == "__main__":


    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)





