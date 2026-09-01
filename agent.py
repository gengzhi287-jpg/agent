# -*- coding: utf-8 -*-
"""Agent 编排层：让大模型通过工具调用自主完成素材采集任务。

大模型作为决策者，在循环里选择调用工具、查看结果、调整策略，
最终用自然语言总结。工具实际执行复用了现有爬取/分类/保存管线。
"""
import json
import requests

from utils import chat_completions_url

SYSTEM = (
    "你是一个素材采集 Agent。用户会告诉你想要收集什么样的图片素材。"
    "请严格根据用户要求的主题采集，不要偏离或替换成别的主题（例如用户要「零食」就采集零食，不要采鸟类）。"
    "请根据用户的意图，决定要采集哪些关键词、每个关键词的数量和搜索源，"
    "并调用工具完成任务。你可以多次调用工具来调整策略（例如某关键词结果太少时换个关键词或搜索源再试，或调整尺寸/主题过滤）。"
    "完成后用中文给出一份简洁的总结：收集了哪些主题、各保存了多少张、保存在哪、有没有失败或建议。"
    "工具的返回是 JSON，请据此判断下一步。"
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "collect_keyword",
            "description": "按一个关键词搜索并下载、分类、保存一批图片。返回该关键词的采集结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string", "description": "要采集的关键词"},
                    "count": {"type": "integer", "description": "想下载的图片数量，默认 20"},
                    "source": {"type": "string", "enum": ["bing", "pixabay", "pexels"], "description": "搜索源，默认 bing"},
                    "min_width": {"type": "integer", "description": "可选：只保留不小于该宽度的图片"},
                    "min_height": {"type": "integer", "description": "可选：只保留不小于该高度的图片"},
                    "exclude_topics": {"type": "array", "items": {"type": "string"}, "description": "可选：不保存的主题词列表"}
                },
                "required": ["keyword"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "collect_url",
            "description": "从指定的网页里提取图片，下载、分类、保存。返回采集结果。",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "要提取图片的网页地址"},
                    "count": {"type": "integer", "description": "最多提取的图片数量，默认 20"},
                    "min_width": {"type": "integer", "description": "可选：只保留不小于该宽度的图片"},
                    "min_height": {"type": "integer", "description": "可选：只保留不小于该高度的图片"},
                    "exclude_topics": {"type": "array", "items": {"type": "string"}, "description": "可选：不保存的主题词列表"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_saved",
            "description": "查看当前素材库已保存的内容结构（各关键词/主题有多少张图）。",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]


def _chat_url(base_url):
    """把用户填的 Base URL 归一化到 {base}/chat/completions（见 utils）。"""
    return chat_completions_url(base_url)


def _err_msg(e):
    resp = getattr(e, "response", None)
    if resp is not None:
        code = resp.status_code
        if code == 401:
            return "API Key 无效或未填写（401），请检查设置模型里的 API Key"
        if code == 403:
            return "无权限/被拒绝（403），请检查账号权限"
        if code == 404:
            return "接口地址错误（404），请检查设置模型里的 Base URL"
        if code == 429:
            return "请求过于频繁（429），请稍后重试"
        return f"接口错误（{code}）"
    return str(e)


def llm_chat(base_url, api_key, model, messages, tools=None, timeout=90):
    """调用 OpenAI 兼容的 chat/completions，支持工具（function calling）。"""
    url = _chat_url(base_url)
    payload = {"model": model, "messages": messages}
    if tools:
        payload["tools"] = tools
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def run_agent(goal, base_url, api_key, model, dispatch, on_event, max_iter=15, memory_context=""):
    """执行 Agent 循环。dispatch(name, args) 返回工具结果 dict；on_event(kind, payload) 用于进度。

    kind: "tool"(payload=工具名+参数串) | "final"(payload=最终总结文本) | "error"(payload=错误信息)
    memory_context: 可选的跨会话记忆文本，注入系统提示。
    返回最终总结文本；出错或达到轮次上限返回 None。
    """
    system = SYSTEM
    if memory_context:
        system = system + "\n\n【跨会话记忆】\n" + memory_context
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": goal},
    ]
    for _ in range(max_iter):
        try:
            data = llm_chat(base_url, api_key, model, messages, TOOLS)
        except Exception as e:
            on_event("error", _err_msg(e))
            return None
        msg = data["choices"][0]["message"]
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            messages.append(msg)
            for tc in tool_calls:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"]["arguments"] or "{}")
                except Exception:
                    args = {}
                on_event("tool", f"{name} {json.dumps(args, ensure_ascii=False)}")
                result = dispatch(name, args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                })
        else:
            text = (msg.get("content") or "").strip()
            on_event("final", text)
            return text
    # 达到轮次上限：让模型基于已有工作补一次总结，而不是静默结束
    messages.append({"role": "user", "content": "工具调用轮次已用完，请根据目前已完成的工作直接给出中文总结。"})
    try:
        data = llm_chat(base_url, api_key, model, messages)
        text = (data["choices"][0]["message"].get("content") or "").strip()
    except Exception as e:
        on_event("error", _err_msg(e))
        return None
    on_event("final", text)
    return text

# ================= Plan-and-Execute（v3.0） =================

PLAN_SYSTEM = (
    """你是一个素材采集规划器。根据用户的素材需求，输出一份 JSON 采集计划，只输出 JSON，不要其他文字。
格式：{"steps": [{"keyword": "关键词", "count": 数量, "source": "bing", "min_width": 0, "min_height": 0, "exclude_topics": []}], "reason": "一句话规划思路"}
要求：1-6 个步骤；keyword 用适合图片搜索的词；count 取 1-50；source 只能是 bing/pixabay/pexels；source/min_width/min_height/exclude_topics 可省略。"""
)

REFLECT_SYSTEM = (
    """你是素材采集 Agent 的反思模块。评估刚执行的采集步骤结果，判断是否需要调整后续计划（例如保存数不足、大量失败、主题跑偏）。只输出 JSON，不要其他文字：
{"satisfied": true或false, "comment": "一句话评估", "new_steps": null 或 [新的后续步骤]}
步骤格式与计划相同；若无需调整，new_steps 为 null。"""
)

SUMMARY_SYSTEM = (
    "你是素材采集 Agent。根据执行进展，用中文给出简洁总结："
    "收集了哪些主题、各保存多少张、保存在哪、失败情况与建议。"
)


def _extract_json(text):
    """从模型输出中提取第一个 JSON 对象；失败返回 None。"""
    import re as _re
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _re.search(r"\{.*\}", text, _re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None


def _valid_steps(obj, max_steps=8):
    """校验并清洗计划步骤列表；无效返回 None。"""
    steps = (obj or {}).get("steps")
    if not isinstance(steps, list):
        return None
    out = []
    for st in steps:
        if not isinstance(st, dict):
            continue
        kw = str(st.get("keyword") or "").strip()
        if not kw:
            continue
        try:
            cnt = int(st.get("count") or 20)
        except Exception:
            cnt = 20
        step = {"keyword": kw, "count": max(1, min(cnt, 50))}
        if st.get("source") in ("bing", "pixabay", "pexels"):
            step["source"] = st["source"]
        for k in ("min_width", "min_height"):
            try:
                v = int(st.get(k) or 0)
            except Exception:
                v = 0
            if v > 0:
                step[k] = v
        if isinstance(st.get("exclude_topics"), list):
            step["exclude_topics"] = [str(x) for x in st["exclude_topics"]][:10]
        out.append(step)
        if len(out) >= max_steps:
            break
    return out or None


def make_plan(goal, base_url, api_key, model, memory_context="", tracer=None):
    """规划阶段：让模型输出结构化采集计划，返回 (steps, reason)。"""
    system = PLAN_SYSTEM
    if memory_context:
        system += "\n\n【跨会话记忆】\n" + memory_context
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": "用户需求：" + goal},
    ]
    data = llm_chat(base_url, api_key, model, messages)
    if tracer:
        tracer.event("llm", phase="plan", usage=data.get("usage"))
    obj = _extract_json(data["choices"][0]["message"].get("content"))
    return _valid_steps(obj), str((obj or {}).get("reason") or "")


def run_agent_v2(goal, base_url, api_key, model, dispatch, on_event,
                 max_steps=8, max_replans=3, memory_context="", tracer=None,
                 count_default=20):
    """Plan-and-Execute 版 Agent：先规划、逐步执行、不足时反思重规划、最后总结。

    on_event kind 新增: "plan" / "reflect" / "replan"。规划失败时回退到 run_agent 单循环。
    """
    import time as _time

    # 1) 规划
    try:
        steps, reason = make_plan(goal, base_url, api_key, model, memory_context, tracer)
    except Exception as e:
        on_event("error", "规划失败，回退到单循环模式: " + _err_msg(e))
        return run_agent(goal, base_url, api_key, model, dispatch, on_event,
                         memory_context=memory_context)
    if not steps:
        on_event("error", "计划解析失败，回退到单循环模式")
        return run_agent(goal, base_url, api_key, model, dispatch, on_event,
                         memory_context=memory_context)
    on_event("plan", {"steps": steps, "reason": reason})
    if tracer:
        tracer.event("plan", steps=steps, reason=reason)

    # 2) 执行 + 按需反思/重规划
    progress = []
    replans = 0
    idx = 0
    while steps:
        step = steps.pop(0)
        idx += 1
        on_event("tool", "collect_keyword " + json.dumps(step, ensure_ascii=False))
        if tracer:
            tracer.event("tool_call", step=idx, args=step)
        t0 = _time.time()
        try:
            result = dispatch("collect_keyword", dict(step))
        except Exception as e:
            result = {"error": str(e)}
        if tracer:
            tracer.event("tool_result", step=idx, elapsed_s=round(_time.time() - t0, 1),
                         result=result)
        progress.append({"step": step, "result": result})

        try:
            saved = int((result or {}).get("saved", 0))
        except Exception:
            saved = 0
        shortfall = saved < int(step.get("count", 0))
        if not shortfall or replans >= max_replans:
            if not steps:
                break
            continue

        # 反思 + 可选重规划
        messages = [
            {"role": "system", "content": REFLECT_SYSTEM},
            {"role": "user", "content": (
                "用户目标：" + goal + "\n"
                "刚执行的步骤：" + json.dumps(step, ensure_ascii=False) + "\n"
                "执行结果：" + json.dumps(result, ensure_ascii=False) + "\n"
                "剩余计划：" + json.dumps(steps, ensure_ascii=False) + "\n"
                "近期进展：" + json.dumps(progress[-5:], ensure_ascii=False)
            )},
        ]
        try:
            data = llm_chat(base_url, api_key, model, messages)
        except Exception as e:
            on_event("error", _err_msg(e))
            continue
        if tracer:
            tracer.event("llm", phase="reflect", usage=data.get("usage"))
        obj = _extract_json(data["choices"][0]["message"].get("content")) or {}
        comment = str(obj.get("comment") or "")
        on_event("reflect", comment)
        if tracer:
            tracer.event("reflect", comment=comment, satisfied=obj.get("satisfied"))
        new_steps = _valid_steps({"steps": obj.get("new_steps")}, max_steps) if obj.get("new_steps") else None
        if new_steps:
            replans += 1
            steps = new_steps
            on_event("replan", {"steps": new_steps, "comment": comment})
            if tracer:
                tracer.event("replan", steps=new_steps, comment=comment)

    # 3) 总结
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM},
        {"role": "user", "content": "用户目标：" + goal + "\n执行进展：" + json.dumps(progress, ensure_ascii=False)},
    ]
    try:
        data = llm_chat(base_url, api_key, model, messages)
        if tracer:
            tracer.event("llm", phase="summary", usage=data.get("usage"))
        text = (data["choices"][0]["message"].get("content") or "").strip()
    except Exception as e:
        on_event("error", _err_msg(e))
        return None
    if tracer:
        tracer.event("final", summary=text)
    on_event("final", text)
    return text
