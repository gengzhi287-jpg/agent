# -*- coding: utf-8 -*-
"""采集管线：搜索 -> 并行下载 -> 两级过滤(去重/尺寸 -> 多模态校验) -> 保存。"""
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import crawler
import task_state as ts
from classifier import ArkClassifier

_ctx = {}


def init(saver, save_cfg, crawl_cfg, search_cfg, filter_cfg, root_dir):
    _ctx.update({
        "saver": saver,
        "save_cfg": save_cfg,
        "crawl_cfg": crawl_cfg,
        "search_cfg": search_cfg,
        "min_width": int(filter_cfg.get("min_width", 0) or 0),
        "min_height": int(filter_cfg.get("min_height", 0) or 0),
        "exclude_topics": set(filter_cfg.get("exclude_topics", []) or []),
        "root_dir": root_dir,
        "bing_empty_streak": 0,
    })


def _mime_for(ext: str) -> str:
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(
        ext.lstrip("."), "image/jpeg"
    )



def _record_fail(keyword, reason):
    with ts._state_lock:
        st = ts._state["tasks"][keyword]
        st["failed"] += 1
        st["errors"][reason] = st["errors"].get(reason, 0) + 1



def _process_single(keyword, url, seq, classifier, min_w, min_h, exclude,
                    allowed, max_kb, analyze, stop_event=None):
    """下载 + 校验 + 保存单张图片（供线程池并行调用）。

    stop_event: 可选，保存数量达标后置位，未开始的任务直接跳过。
    """
    if ts._cancel_event.is_set() or (stop_event is not None and stop_event.is_set()):
        return
    ts._pause_event.wait()
    if ts._cancel_event.is_set() or (stop_event is not None and stop_event.is_set()):
        return
    data, ext, reason = crawler.download_image(url, max_kb, allowed)
    if data is None:
        _record_fail(keyword, reason)
        return
    if _ctx["saver"].is_seen(data):
        with ts._state_lock:
            st = ts._state["tasks"][keyword]
            st["downloaded"] += 1
        ts.append_log(f"[{keyword}] 跳过(重复) {os.path.basename(url)}")
        return
    if min_w or min_h:
        dim = crawler.image_dimensions(data)
        if dim and (dim[0] < min_w or dim[1] < min_h):
            _record_fail(keyword, "图片过小")
            return
    mime = _mime_for(ext)
    if analyze:
        topic, matched = classifier.analyze(data, mime, keyword)
        if not matched:
            _record_fail(keyword, "与主题不符")
            return
    else:
        topic = classifier.classify(data, mime)
    if topic in exclude:
        _record_fail(keyword, "主题被排除")
        return
    saved_path = _ctx["saver"].save(data, ext, keyword, topic, seq)
    with ts._state_lock:
        st = ts._state["tasks"][keyword]
        st["downloaded"] += 1
        if saved_path:
            st["saved"] += 1
    ts.append_log(
        f"[{keyword}] {'保存' if saved_path else '跳过(重复)'} "
        f"{os.path.basename(saved_path or url)}"
    )



def process_keyword(keyword, count, concurrency, model, base_url, api_key,
                    provider="bing", search_keys=None, filters=None):
    search_keys = search_keys or {}
    filters = filters or {}
    min_w = int(filters.get("min_width") or _ctx["min_width"])
    min_h = int(filters.get("min_height") or _ctx["min_height"])
    exclude = set(filters.get("exclude_topics") or []) or _ctx["exclude_topics"]
    allowed = tuple(_ctx["save_cfg"]["allowed_exts"])
    max_kb = int(_ctx["save_cfg"]["max_size_kb"])
    classifier = ArkClassifier(base_url, api_key, model)

    with ts._state_lock:
        ts._state["tasks"][keyword] = {
            "keyword": keyword,
            "status": "搜索中",
            "downloaded": 0,
            "failed": 0,
            "saved": 0,
            "errors": {},
            "folder": os.path.join(_ctx["root_dir"], keyword),
        }

    # 失败补位：多抓候选，直到保存够 count 张为止
    candidates = max(count * 3, count + 10)
    try:
        if provider == "pixabay" and search_keys.get("pixabay"):
            urls = crawler.pixabay_search(keyword, min(candidates, 200), search_keys["pixabay"])
        elif provider == "pexels" and search_keys.get("pexels"):
            urls = crawler.pexels_search(keyword, min(candidates, 80), search_keys["pexels"])
        else:
            urls = crawler.search_image_urls(keyword, limit=candidates)
    except Exception as e:
        ts.append_log(f"[{keyword}] 搜索失败: {e}")
        with ts._state_lock:
            ts._state["tasks"][keyword]["status"] = "搜索失败"
        return

    used_bing = not ((provider == "pixabay" and search_keys.get("pixabay"))
                     or (provider == "pexels" and search_keys.get("pexels")))
    if used_bing:
        if urls:
            _ctx["bing_empty_streak"] = 0
        else:
            _ctx["bing_empty_streak"] += 1
            if _ctx["bing_empty_streak"] >= 2:
                ts.append_log("[警告] Bing 图片搜索连续解析为空，页面结构可能已变化，"
                            "建议在 config.yaml 配置 Pixabay / Pexels API Key 作为稳定搜索源")

    with ts._state_lock:
        ts._state["tasks"][keyword]["status"] = f"搜索到 {len(urls)} 张，开始下载"

    stop_event = threading.Event()
    workers = max(1, int(concurrency or _ctx["crawl_cfg"].get("max_concurrency", 4)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_process_single, keyword, url, i + 1, classifier,
                        min_w, min_h, exclude, allowed, max_kb, True, stop_event)
            for i, url in enumerate(urls)
        ]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                ts.append_log(f"[{keyword}] 处理异常: {e}")
            with ts._state_lock:
                saved_so_far = ts._state["tasks"][keyword]["saved"]
            if saved_so_far >= count:
                stop_event.set()

    with ts._state_lock:
        ts._state["tasks"][keyword]["status"] = "已取消" if ts._cancel_event.is_set() else "完成"
        result = dict(ts._state["tasks"][keyword])
    _ctx["saver"].flush()
    return result



def process_url(page_url, count, model, base_url, api_key, filters=None):
    filters = filters or {}
    min_w = int(filters.get("min_width") or _ctx["min_width"])
    min_h = int(filters.get("min_height") or _ctx["min_height"])
    exclude = set(filters.get("exclude_topics") or []) or _ctx["exclude_topics"]
    allowed = tuple(_ctx["save_cfg"]["allowed_exts"])
    max_kb = int(_ctx["save_cfg"]["max_size_kb"])
    classifier = ArkClassifier(base_url, api_key, model)
    name = os.path.basename(page_url.rstrip("/")) or page_url

    with ts._state_lock:
        ts._state["tasks"][name] = {
            "keyword": name,
            "status": "解析页面",
            "downloaded": 0,
            "failed": 0,
            "saved": 0,
            "errors": {},
            "folder": os.path.join(_ctx["root_dir"], name),
        }

    try:
        urls = crawler.extract_image_urls(page_url)
    except Exception as e:
        ts.append_log(f"[URL {name}] 解析失败: {e}")
        with ts._state_lock:
            ts._state["tasks"][name]["status"] = "解析失败"
        return

    workers = max(1, int(_ctx["crawl_cfg"].get("max_concurrency", 4)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_process_single, name, url, i + 1, classifier,
                        min_w, min_h, exclude, allowed, max_kb, False)
            for i, url in enumerate(urls[:count])
        ]
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                ts.append_log(f"[URL {name}] 处理异常: {e}")

    with ts._state_lock:
        ts._state["tasks"][name]["status"] = "已取消" if ts._cancel_event.is_set() else "完成"
        result = dict(ts._state["tasks"][name])
    _ctx["saver"].flush()
    return result


