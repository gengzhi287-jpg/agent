# 素材采集 Agent · 使用说明

> 素材采集智能体：一句话下达需求，大模型自主选词选源，爬取图片、内容校验、分类保存；内置跨会话语义记忆。

**素材采集智能体（Material Collection Agent）**

一个基于大模型工具调用的本地智能体。你只需用一句话描述想要什么素材，Agent 就会自主决定关键词、数量与搜索源，从网上爬取图片，经多模态模型校验后按关键词分类保存到本地素材库。

**核心功能**
- 🤖 **Agent 智能体**：大模型自主规划、调用工具、根据结果调整策略并总结
- ✅ **内容校验**：识别图片是否与关键词相关，无关图自动丢弃（提升命中率）
- 🧠 **跨会话语义记忆**：基于 embedding 的 RAG 记忆，自动去重、自动遗忘
- 🖼️ **多源采集**：Bing（免 Key）/ Pixabay / Pexels 可切换
- 📂 **自动分类保存**：`素材库/关键词/图片`，MD5 去重、重名加序号
- 💻 **本地网页界面**：实时进度、暂停/继续/取消、结果报告、历史、素材浏览、记忆管理

**部署环境**
- Python 3.9+
- 依赖：`flask`、`requests`、`pyyaml`、`pillow`
- 模型：火山引擎 Agent Plan 套餐 API Key（或任意 OpenAI 兼容多模态接口）
- 运行：`pip install -r requirements.txt` → `python app.py` → 浏览器打开 `http://127.0.0.1:5000`

**目录结构**
`app.py`（后端）、`agent.py`（Agent 编排）、`crawler.py`（搜索/下载）、`classifier.py`（多模态识别/校验）、`memory.py`（跨会话记忆）、`saver.py`（保存/去重）、`templates/`（网页界面）

---

一个**智能体（Agent）**程序：你只用一句话描述想要什么素材，大模型就会自己决定关键词、数量、搜索源，调用工具从网上爬取图片，按内容校验后分类保存到本地 `素材库`。

---

## 一、核心特性

- 🤖 **Agent 智能体**：一句话下达需求，大模型自主决策 + 调用工具（`collect_keyword` / `collect_url` / `list_saved`）循环采集，跑完自动总结。
- ✅ **内容校验（命中率）**：下载后让多模态模型判断「图片是否真的对得上关键词」，不相关的直接丢弃，避免搜「零食」存成「鸟类」。
- 🧠 **跨会话记忆（语义 RAG）**：用火山方舟 embedding 模型把记忆向量化，每次采集按「用户目标」语义检索最相关的历史规则并注入提示词；自动去重、自动遗忘冷门规则。
- 📂 **单级分类保存**：`素材库/关键词/图片`，命名 `01_关键词.jpg`，MD5 去重、重名自动加序号。
- 🖼️ **只存合规图**：仅 `.jpg/.jpeg/.png`、单张 ≤1MB、不含动图。
- 🌐 **多搜索源**：Bing（免 Key）/ Pixabay / Pexels（填 Key）可切换。
- 💻 **本地网页界面**：实时进度、暂停 / 继续 / 取消（取消有二次确认）、任务完成提示音、结果报告、历史回查、素材缩略图浏览、记忆管理。

---

## 二、目录结构

```
agent/
├── app.py               # Flask 后端入口
├── agent.py             # Agent 编排（大模型工具调用循环）
├── crawler.py           # 搜索(Bing/Pixabay/Pexels) + URL 提取 + 下载 + 尺寸
├── classifier.py        # 多模态识别 + 关键词匹配校验（analyze）
├── saver.py             # 保存 / MD5 去重 / 重名加序号
├── memory.py            # 跨会话记忆（语义 RAG：检索/去重/遗忘）
├── logger_setup.py      # 日志
├── config.yaml          # 运行配置（模型 / 搜索源 / 过滤）
├── memory.json          # 长期记忆规则（可跨会话累积）
├── memory_vectors.json  # 记忆向量缓存（自动生成）
├── requirements.txt
├── templates/index.html # 网页界面
├── 素材库/              # 运行时自动创建，存图片
└── logs/                # 运行时自动创建（history.json + 时间戳日志）
```

---

## 三、安装

需要 **Python 3.9+**（本机用 Anaconda 即可）。

```bash
pip install -r requirements.txt
```

依赖：`flask`、`requests`、`pyyaml`、`pillow`。

> 在 PyCharm 中直接打开项目，用 `app.py` 作为入口运行即可（推荐）。

---

## 四、配置

> 首次使用：复制 `config.example.yaml` 为 `config.yaml` 再填写你的 API Key。
> `config.yaml` 已在 `.gitignore` 中，**不会**被提交到 GitHub，避免泄露密钥。

### 1. 模型（LLM）—— 关键

本项目用的是**火山引擎 Agent Plan 套餐**，必须用**套餐接口**和**套餐内模型**：

在 `config.yaml` 的 `llm` 段：

```yaml
llm:
  base_url: "https://ark.cn-beijing.volces.com/api/plan/v3"   # 套餐接口（勿用 /api/v3）
  api_key: "你的套餐 API Key"
  model: "doubao-seed-2-1-turbo-260628"      # 套餐内多模态模型
  embedding_model: "doubao-embedding-vision-251215"  # 记忆向量化模型
```

- ⚠️ 套餐 Key 配 `/api/v3` 会报 401/404；必须用 `/api/plan/v3`。
- 页面「⚙ 设置模型」可填，会存到浏览器 `localStorage` 并**覆盖** `config.yaml`。若之前存过旧配置导致报错，清一下：浏览器 `F12` → Console → `localStorage.removeItem('material_agent_llm')` → 刷新。

### 2. 搜索源

```yaml
search:
  provider: "bing"        # bing | pixabay | pexels
  pixabay: { api_key: "" }   # https://pixabay.com/api/docs/
  pexels:  { api_key: "" }   # https://www.pexels.com/api/
```
默认 `bing` 免 Key；未配 Key 的源会自动回退 Bing。Pixabay / Pexels 版权更清晰，建议填 Key 使用。

### 3. 过滤规则

```yaml
save:
  root_dir: "素材库"
  max_size_kb: 1024         # 单张最大 1MB
  allowed_exts: [".jpg", ".jpeg", ".png"]
  count_per_keyword: 20     # 默认每关键词张数

filter:
  min_width: 0              # 小于此宽度跳过；0 不限制
  min_height: 0
  exclude_topics: []        # 不保存的主题词，如 ["文字","截图"]
```

---

## 五、运行

**PyCharm**：打开项目 → 右键运行 `app.py`。

**命令行**：
```bash
python app.py
```

浏览器打开 `http://127.0.0.1:5000`。

---

## 六、使用方法（页面）

1. **下达任务**：在顶部输入框写一句话，例如「收集 15 张山水风景和 8 张猫咪的图片，尽量高清」。
2. **开始采集**：点「开始采集」。可随时「⏸ 暂停 / ▶ 继续 / ⏹ 取消」（取消有二次确认）。
3. **查看进度**：「实时进度」表格展示每个关键词的下载/保存/失败数。
4. **结果与历史**：「结果汇总」可导出报告；「任务历史」可回查以往任务。
5. **浏览素材**：「素材浏览」按关键词/主题查看缩略图，点击放大。
6. **管理记忆**：「记忆（长期规则）」面板可查看、添加、删除让 Agent 长期记住的规则（跨会话生效）。
7. **配置模型**：「⚙ 设置模型」切换预设 / 填 Base URL、API Key、模型名。

图片保存位置：项目下的 `素材库/关键词/图片`。

---

## 七、跨会话记忆（记忆模块）

- **存什么**：长期规则（`memory.json`）+ 近期任务历史（`logs/history.json`）。
- **怎么用**：每次采集前，用 embedding 把「用户目标」和所有记忆做语义相似度排序，只取最相关的 top-K 条注入系统提示（语义 RAG）。
- **自动去重**：新规则与已有规则语义余弦 ≥0.72 或重叠系数 ≥0.75 时拒收。
- **自动遗忘**：规则记录使用次数与最近使用时间，每天自动清理「90 天没用 + 用 <2 次」的冷门规则。
- **降级兜底**：embedding 不可用时自动回退本地 TF-IDF 检索。

---

## 八、接口速览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 页面 |
| GET | `/api/config` | 配置（模型/预设/搜索源） |
| POST | `/api/agent/start` | 启动 Agent 任务（body: `{goal, preset?}`） |
| GET | `/api/status` | 运行状态与进度 |
| POST | `/api/pause` `/api/resume` `/api/cancel` | 暂停 / 继续 / 取消 |
| GET | `/api/report` `/api/export` | 结果汇总 / 导出 txt |
| GET | `/api/history` `/api/history/<id>` | 任务历史 / 详情 |
| GET | `/api/memory` | 记忆查看 |
| POST | `/api/memory` | 新增记忆规则（body: `{rule}`） |
| DELETE | `/api/memory` | 删除记忆规则（body: `{rule}`） |
| GET | `/api/browse` | 素材库树 |
| GET | `/media/<path>` | 图片文件 |

---

## 九、常见问题

- **端口被占 / 页面是旧配置**：之前可能有多个 `app.py` 实例。全部关掉后重启一个即可。
- **报 401/404**：`base_url` 必须是 `/api/plan/v3`（套餐），且 `api_key` 是套餐 Key；或清一下页面 localStorage 旧配置。
- **图片和关键词不符**：已内置内容校验，不相关的会自动丢弃；若仍不满意可调低 `filter` 或换更干净的素材源。
- **搜索慢 / 张数不够**：Bing 免 Key 但稳定性一般，建议给 Pixabay / Pexels 配 Key。
- **需要联网**：爬图与调用模型都要联网。

---

## 十、说明与局限

- 默认 Bing 为 HTML 解析，可能随站点改版失效。
- 版权：优先免费可商用来源，具体图源版权请自行判断。
- 当前为单智能体 + 工具调用形态，自主性上限 `max_iter=15`；无多智能体协作。
