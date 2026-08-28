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

# 任务状态（内存中）
_state = {
    "running": False,
    "paused": False,
    "tasks": {},          # keyword -> 状态
    "log": [],            # 运行日志（供界面显示）
    "started": None,
}
_state_lock = threading.Lock()
_pool = None
_pause_event = threading.Event()
_pause_event.set()  # 默认未暂停
_cancel_event = threading.Event()

app = Flask(__name__)


def _append_log(line: str):
    with _state_lock:
        _state["log"].append(line)
        if len(_state["log"]) > 500:
            _state["log"] = _state["log"][-500:]
    logger.info(line)


def _mime_for(ext: str) -> str:
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        ext.lstrip("."), "image/jpeg"
    )


def _process_keyword(keyword, count, concurrency, model, base_url, api_key,
                    provider="bing", search_keys=None, filters=None):
    search_keys = search_keys or {}
    filters = filters or {}
    min_w = int(filters.get("min_width") or MIN_WIDTH)
    min_h = int(filters.get("min_height") or MIN_HEIGHT)
    exclude = set(filters.get("exclude_topics") or []) or EXCLUDE_TOPICS
    allowed = tuple(SAVE_CFG["allowed_exts"])
    max_kb = int(SAVE_CFG["max_size_kb"])
    classifier = ArkClassifier(base_url, api_key, model)

    with _state_lock:
        _state["tasks"][keyword] = {
            "keyword": keyword,
            "status": "搜索中",
            "downloaded": 0,
            "failed": 0,
            "saved": 0,
            "errors": {},
            "folder": os.path.join(root_dir, keyword),
        }

    seq = 0
    try:
        if provider == "pixabay" and search_keys.get("pixabay"):
            urls = crawler.pixabay_search(keyword, count, search_keys["pixabay"])
        elif provider == "pexels" and search_keys.get("pexels"):
            urls = crawler.pexels_search(keyword, count, search_keys["pexels"])
        else:
            urls = crawler.search_image_urls(keyword, limit=count)
    except Exception as e:
        _append_log(f"[{keyword}] 搜索失败: {e}")
        with _state_lock:
            _state["tasks"][keyword]["status"] = "搜索失败"
        return

    with _state_lock:
        _state["tasks"][keyword]["status"] = f"搜索到 {len(urls)} 张，开始下载"

    for url in urls:
        if _cancel_event.is_set():
            break
        _pause_event.wait()
        if _cancel_event.is_set():
            break
        seq += 1
        data, ext, reason = crawler.download_image(url, max_kb, allowed)
        if data is None:
            with _state_lock:
                st = _state["tasks"][keyword]
                st["failed"] += 1
                st["errors"][reason] = st["errors"].get(reason, 0) + 1
            continue

        if min_w or min_h:
            dim = crawler.image_dimensions(data)
            if dim and (dim[0] < min_w or dim[1] < min_h):
                with _state_lock:
                    st = _state["tasks"][keyword]
                    st["failed"] += 1
                    st["errors"]["图片过小"] = st["errors"].get("图片过小", 0) + 1
                continue

        mime = _mime_for(ext)
        topic, matched = classifier.analyze(data, mime, keyword)
        if not matched:
            with _state_lock:
                st = _state["tasks"][keyword]
                st["failed"] += 1
                st["errors"]["与主题不符"] = st["errors"].get("与主题不符", 0) + 1
            continue
        if topic in exclude:
            with _state_lock:
                st = _state["tasks"][keyword]
                st["failed"] += 1
                st["errors"]["主题被排除"] = st["errors"].get("主题被排除", 0) + 1
            continue
        saved_path = saver.save(data, ext, keyword, topic, seq)

        with _state_lock:
            st = _state["tasks"][keyword]
            st["downloaded"] += 1
            if saved_path:
                st["saved"] += 1

        _append_log(
            f"[{keyword}] {'保存' if saved_path else '跳过(重复)'} "
            f"{os.path.basename(saved_path or url)}"
        )

    with _state_lock:
        _state["tasks"][keyword]["status"] = "已取消" if _cancel_event.is_set() else "完成"
        return dict(_state["tasks"][keyword])


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
    with _state_lock:
        return jsonify({
            "running": _state["running"],
            "paused": _state["paused"],
            "agent_final": _state.get("agent_final"),
            "tasks": _state["tasks"],
            "log": list(_state["log"]),
        })


def _process_url(page_url, count, model, base_url, api_key, filters=None):
    filters = filters or {}
    min_w = int(filters.get("min_width") or MIN_WIDTH)
    min_h = int(filters.get("min_height") or MIN_HEIGHT)
    exclude = set(filters.get("exclude_topics") or []) or EXCLUDE_TOPICS
    allowed = tuple(SAVE_CFG["allowed_exts"])
    max_kb = int(SAVE_CFG["max_size_kb"])
    classifier = ArkClassifier(base_url, api_key, model)
    name = os.path.basename(page_url.rstrip("/")) or page_url

    with _state_lock:
        _state["tasks"][name] = {
            "keyword": name,
            "status": "解析页面",
            "downloaded": 0,
            "failed": 0,
            "saved": 0,
            "errors": {},
            "folder": os.path.join(root_dir, name),
        }

    try:
        urls = crawler.extract_image_urls(page_url)
    except Exception as e:
        _append_log(f"[URL {name}] 解析失败: {e}")
        with _state_lock:
            _state["tasks"][name]["status"] = "解析失败"
        return

    seq = 0
    for url in urls[:count]:
        if _cancel_event.is_set():
            break
        _pause_event.wait()
        if _cancel_event.is_set():
            break
        seq += 1
        data, ext, reason = crawler.download_image(url, max_kb, allowed)
        if data is None:
            with _state_lock:
                st = _state["tasks"][name]
                st["failed"] += 1
                st["errors"][reason] = st["errors"].get(reason, 0) + 1
            continue
        if min_w or min_h:
            dim = crawler.image_dimensions(data)
            if dim and (dim[0] < min_w or dim[1] < min_h):
                with _state_lock:
                    st = _state["tasks"][name]
                    st["failed"] += 1
                    st["errors"]["图片过小"] = st["errors"].get("图片过小", 0) + 1
                continue
        topic = classifier.classify(data, _mime_for(ext))
        if topic in exclude:
            with _state_lock:
                st = _state["tasks"][name]
                st["failed"] += 1
                st["errors"]["主题被排除"] = st["errors"].get("主题被排除", 0) + 1
            continue
        saved_path = saver.save(data, ext, name, topic, seq)
        with _state_lock:
            st = _state["tasks"][name]
            st["downloaded"] += 1
            if saved_path:
                st["saved"] += 1
        _append_log(
            f"[URL {name}] {'保存' if saved_path else '跳过(重复)'} "
            f"{os.path.basename(saved_path or url)}"
        )
    with _state_lock:
        _state["tasks"][name]["status"] = "已取消" if _cancel_event.is_set() else "完成"
        return dict(_state["tasks"][name])


def _wait_all(futures, on_done):
    for f in as_completed(futures):
        try:
            f.result()
        except Exception as e:
            _append_log(f"任务异常: {e}")
    on_done()


@app.route("/api/pause", methods=["POST"])
def api_pause():
    with _state_lock:
        if not _state["running"] or _state["paused"]:
            return jsonify({"ok": False, "error": "当前无运行中任务可暂停"}), 400
        _state["paused"] = True
    _pause_event.clear()
    _append_log("任务已暂停")
    return jsonify({"ok": True})


@app.route("/api/resume", methods=["POST"])
def api_resume():
    with _state_lock:
        if not _state["running"] or not _state["paused"]:
            return jsonify({"ok": False, "error": "当前无暂停任务可恢复"}), 400
        _state["paused"] = False
    _pause_event.set()
    _append_log("任务已恢复")
    return jsonify({"ok": True})


@app.route("/api/cancel", methods=["POST"])
def api_cancel():
    with _state_lock:
        if not _state["running"]:
            return jsonify({"ok": False, "error": "当前无运行中任务可取消"}), 400
        _state["paused"] = False
    _cancel_event.set()
    _pause_event.set()  # 让处于暂停的线程解除阻塞并检查取消
    _append_log("正在取消任务...")
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
    with _state_lock:
        return jsonify(_state.get("report"))


@app.route("/api/export")
def api_export():
    with _state_lock:
        report = _state.get("report")
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
    with _state_lock:
        _state["running"] = False
        _state["paused"] = False
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
            for t in _state["tasks"].values()
        ]
        _state["report"] = {
            "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "items": items,
        }
        run = _state.get("run")
        if run:
            run["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            run["items"] = items
            run["summary"] = {
                "downloaded": sum(i.get("downloaded", 0) for i in items),
                "saved": sum(i.get("saved", 0) for i in items),
                "failed": sum(i.get("failed", 0) for i in items),
                "tasks": len(items),
            }
            with _history_lock:
                history = _load_history()
                history.append(run)
                _save_history(history)
    _cancel_event.clear()
    _pause_event.set()
    _append_log("任务结束")


def collect_one(keyword, count, provider, search_keys, model, base_url, api_key, filters=None):
    """供 Agent 工具调用：采集单个关键词，返回该关键词结果汇总。"""
    return _process_keyword(keyword, count, 1, model, base_url, api_key, provider, search_keys, filters)


def collect_url(page_url, count, model, base_url, api_key, filters=None):
    """供 Agent 工具调用：从网页提取图片采集，返回结果汇总。"""
    return _process_url(page_url, count, model, base_url, api_key, filters)


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
            _append_log("[Agent] 调用工具: " + payload)
        elif kind == "error":
            _append_log("[Agent] 出错: " + payload)
        elif kind == "final":
            _append_log("[Agent] 总结:\n" + payload)
            with _state_lock:
                _state["agent_final"] = payload

    prefs = {
        "provider": SEARCH_CFG.get("provider"),
        "count_per_keyword": SAVE_CFG.get("count_per_keyword"),
        "allowed_exts": SAVE_CFG.get("allowed_exts"),
        "max_size_kb": SAVE_CFG.get("max_size_kb"),
    }
    memory_mod.forget_stale()
    mem_ctx = memory_mod.build_context(goal, prefs)
    agent.run_agent(goal, llm["base_url"], llm["api_key"], llm["model"],
                    lambda n, a: _agent_dispatch(n, a, llm, search_keys), on_event,
                    memory_context=mem_ctx)
    _finalize()


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

    with _state_lock:
        if _state["running"]:
            return jsonify({"ok": False, "error": "已有任务在运行"}), 400
        _state["running"] = True
        _state["paused"] = False
        _state["tasks"] = {}
        _state["log"] = []
        _state["started"] = None
        _state["report"] = None
        _state["run"] = {
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
        _state["agent_final"] = None
    _cancel_event.clear()
    _pause_event.set()
    _append_log("[Agent] 目标: " + goal)
    threading.Thread(target=_agent_worker, args=(goal, llm, search_keys), daemon=True).start()
    return jsonify({"ok": True})


if __name__ == "__main__":


    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)





