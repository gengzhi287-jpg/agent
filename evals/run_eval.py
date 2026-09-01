# -*- coding: utf-8 -*-
"""素材采集 Agent 评测脚本。

用法：
  1. 先启动服务：python app.py
  2. 另开终端：python evals/run_eval.py [--base http://127.0.0.1:5000] [--limit 5] [--sample 3]

流程：逐条提交 targets.yaml 中的目标 -> 轮询直到完成 -> 收集报告与 trace，
指标：成功率（saved>=1）、足额率（saved>=min_saved）、抽检命中率（多模态复核）、
     平均耗时、平均 token 用量。结果写入 evals/results/<时间戳>.json 并打印汇总。
"""
import argparse
import io
import json
import os
import sys
import time
from datetime import datetime

import requests
import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from classifier import ArkClassifier  # noqa: E402

POLL_INTERVAL = 8
TARGET_TIMEOUT = 1200  # 单目标最长 20 分钟


def load_targets():
    with open(os.path.join(BASE_DIR, "evals", "targets.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def wait_done(base, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            s = requests.get(base + "/api/status", timeout=10).json()
        except Exception:
            time.sleep(POLL_INTERVAL)
            continue
        if not s.get("running"):
            return s
        time.sleep(POLL_INTERVAL)
    return None


def run_target(base, target):
    goal = target["goal"]
    t0 = time.time()
    try:
        r = requests.post(base + "/api/agent/start", json={"goal": goal}, timeout=15).json()
    except Exception as e:
        return {"goal": goal, "error": "提交失败: " + str(e)}
    if not r.get("ok"):
        return {"goal": goal, "error": r.get("error", "提交被拒绝")}
    st = wait_done(base, TARGET_TIMEOUT)
    elapsed = round(time.time() - t0, 1)
    if st is None:
        return {"goal": goal, "error": "超时"}

    run_id = st.get("trace_run_id")
    tokens = 0
    if run_id:
        try:
            events = requests.get(base + "/api/trace/" + run_id, timeout=10).json().get("events", [])
            for e in events:
                if e.get("kind") == "llm" and isinstance(e.get("usage"), dict):
                    tokens += int(e["usage"].get("total_tokens") or 0)
        except Exception:
            pass

    saved = downloaded = failed = 0
    folders = []
    try:
        rep = requests.get(base + "/api/report", timeout=10).json()
        for i in rep.get("items", []):
            downloaded += i.get("downloaded", 0)
            saved += i.get("saved", 0)
            failed += i.get("failed", 0)
            if i.get("folder"):
                folders.append(i["folder"])
    except Exception:
        pass
    return {
        "goal": goal, "run_id": run_id, "saved": saved, "downloaded": downloaded,
        "failed": failed, "elapsed_s": elapsed, "tokens": tokens, "folders": folders,
    }


def sample_check(result, target, classifier, sample_n):
    """对保存的图片抽样做多模态复核，估算命中率。"""
    imgs = []
    for folder in result.get("folders", []):
        if os.path.isdir(folder):
            for fn in sorted(os.listdir(folder)):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    imgs.append(os.path.join(folder, fn))
    if not imgs:
        return {"checked": 0, "matched": 0, "rate": None}
    step = max(1, len(imgs) // sample_n)
    picked = imgs[::step][:sample_n]
    checked = matched = 0
    for p in picked:
        try:
            with open(p, "rb") as f:
                data = f.read()
            mime = "image/png" if p.lower().endswith(".png") else "image/jpeg"
            _, ok = classifier.analyze(data, mime, target["goal"])
            checked += 1
            matched += 1 if ok else 0
        except Exception:
            pass
    rate = round(matched / checked, 2) if checked else None
    return {"checked": checked, "matched": matched, "rate": rate}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:5000")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（0=全部）")
    ap.add_argument("--sample", type=int, default=3, help="每个目标抽检图片数")
    ap.add_argument("--no-check", action="store_true", help="跳过命中率抽检")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(os.path.join(BASE_DIR, "config.yaml"), encoding="utf-8"))
    llm = cfg["llm"]
    classifier = None if args.no_check else ArkClassifier(llm["base_url"], llm["api_key"], llm["model"])

    targets = load_targets()
    if args.limit:
        targets = targets[: args.limit]
    print(f"共 {len(targets)} 个评测目标，服务 {args.base}")

    results = []
    for i, t in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {t['goal']} ...", flush=True)
        r = run_target(args.base, t)
        if "error" not in r and classifier and args.sample > 0:
            r["relevance"] = sample_check(r, t, classifier, args.sample)
        results.append({**t, **r})
        print("   ->", json.dumps({k: r.get(k) for k in ("saved", "failed", "elapsed_s", "tokens", "error")}, ensure_ascii=False), flush=True)

    # 汇总
    done = [r for r in results if "error" not in r]
    succ = [r for r in done if r["saved"] >= 1]
    full = [r for r in done if r["saved"] >= r.get("min_saved", 1)]
    checks = [r["relevance"] for r in done if r.get("relevance") and r["relevance"]["checked"]]
    summary = {
        "total": len(results),
        "finished": len(done),
        "success_rate": round(len(succ) / len(done), 2) if done else None,
        "full_rate": round(len(full) / len(done), 2) if done else None,
        "avg_relevance": round(sum(c["rate"] for c in checks) / len(checks), 2) if checks else None,
        "avg_elapsed_s": round(sum(r["elapsed_s"] for r in done) / len(done), 1) if done else None,
        "avg_tokens": round(sum(r["tokens"] for r in done) / len(done)) if done else None,
        "total_tokens": sum(r["tokens"] for r in done),
    }
    out = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "base": args.base,
        "summary": summary,
        "results": results,
    }
    os.makedirs(os.path.join(BASE_DIR, "evals", "results"), exist_ok=True)
    path = os.path.join(BASE_DIR, "evals", "results", datetime.now().strftime("%Y%m%d_%H%M%S") + ".json")
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n===== 评测汇总 =====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("结果已保存:", path)


if __name__ == "__main__":
    main()
