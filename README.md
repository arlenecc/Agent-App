# base-agent（Swift 原生重构版）

macOS 原生自主 Agent。控制流是**三态自动机 + 五级上下文压缩**，能力层是工具集，经验层是事实抽取与 L4 蒸馏，知识层是 **SQLite + sqlite-vec + FTS5** 的本地 RAG。

详细模块设计与事务/检索流程见 [软件设计文档](docs/软件设计文档.md)。

## 环境要求

- macOS 14+
- Swift 6.0+（Xcode 16+；本项目在 Swift 6.4 工具链验证）
- 本地模型（可选）：
  - LLM：MLX-LM / MLX-VLM 模型目录、GGUF 文件、ONNX 解码器目录（`input_ids` + `logits`，可选 `past_key_values`）
  - OCR VLM：MLX 模型目录、GGUF + mmproj；ONNX VLM 暂无通用图适配
  - Embedding / Reranker：ONNX 模型目录（含 `tokenizer.json`）、MLX 模型目录或 GGUF 文件
  - 默认主模型 `models/qwen3.5-4b-agentic-coder-v4-4bit-mlx`；辅助模型默认 `models/Qwen3.5-2B-MLX-4bit`，但默认**不启用**，需要在配置窗口显式勾选「启用」
- 运行时已随包内置：
  - GGUF：llama.cpp 官方 XCFramework，进程内直接加载模型，不再启动 `llama-server`
  - ONNX：onnxruntime-swift，不再依赖 Python / `onnxruntime-genai`

## 构建与运行

```bash
swift build --disable-sandbox
swift run BaseAgent
```

> 如果出现 `swift-plugin-server` / 宏插件 `malformed response` 错误，是 SwiftPM 沙箱与 MLX 宏插件在部分工具链上的兼容问题；使用 `--disable-sandbox` 可绕过。正常环境可直接 `swift build`。

运行测试：

```bash
swift test --disable-sandbox
```

测试覆盖（共 165 个 XCTest，14 个需要模型/网络环境而默认跳过）：sqlite-vec 事务写入/回滚、两段式 KNN、FTS5 中文分词、RRF 混合检索、RAG 增量/删除清理、EPUB/DOCX/HTML 解析、扫描版 EPUB 归一化 + OCR、PDF 文本层、Ovis 文本页快路径、NaturalLanguage 切片、切片参数一致性（模型窗口收口、检索呈现上限随切片放大、切片参数变化触发重建）、配置迁移与旧键兼容（含 `auto_preload_main` / `aux_enabled` 默认值与读写往返）、Agent 工具循环、ask_user 续跑、checkpoint 同步、code_run 写护栏、L3 去重、大纲围栏，以及摘要溢出保留、代码围栏/中文换行/正文数字保留、技能索引入口回退、空库维度自愈、`models/` 相对路径解析，以及 KV 复用增量与完整 prompt 的逐 token 模板对齐、预热前缀对齐、清空对话时失效 KV cache；GGUF 模型路径/mmproj 解析、进程内 llama.cpp 框架加载、ONNX 缺少权重时的明确报错，合成 ONNX decoder 图的 past/present 管线，简单 HTML 表格转 Markdown、复杂表格保留 HTML、Markdown 语法/缩进保留、纯文本 HTML/实体/噪声清理、空行与空格密度压缩、跨目录副本与 content_hash 去重、手动向量去重保留最早入库切片并写副本 tombstone，以及切片缺失/旧版 manifest/失败页文件的增量重建。

HTML 与 Office 解析另有专项覆盖：HTML 保存页的本地 iframe / frameset 接力（取正文最多的一份）、big5 等老编码嗅探、依赖 JavaScript 的页面给出明确失败原因、含 `h1` 的 `<header>` 保留、裸文本节点与 `<li>` 内块级元素、结构化清洗档保留 URL/IP 与段落空行；DOCX 编号列表与 `w:br`/`xml:space`、脚注编号与末尾列表、`styles.xml` 样式名解析标题；XLSX 日期与百分比格式、注音跳过、多行单元格不拆表；PPTX 演讲者备注与段内换行。独立 Markdown 转换另覆盖单文件输出、目录相对结构、同名后缀、单文件失败不中断整批、输出目录跳过与不支持格式拒绝。

## 界面

- **主窗口**：左侧约 3/4 为对话区（流式正文、50ms 合并增量、自动滚动、工具调用气泡），底部居中 tokens 计数与输入框；右侧为任务计划（目标/步骤/当前进行中标记）与思考过程，下方状态栏显示模型、同步进度与 RAG 统计。模型协议里的 `<summary>…</summary>` 不进入对话正文，其内容转入右侧思考过程。
- 输入框 Enter 发送、Shift+Enter 换行；中文输入法正在组合候选词时 Enter 放行给 IME 确认候选，不会误发送或清空已输入内容。
- 底部状态栏按顺序显示「模型加载中… → 模型加载成功 → 模型预热中… → 预热结束 → 就绪」，加载/预热期间带进度指示并禁用发送；失败时显示「模型加载失败」或「预热失败」。
- **右上角按钮**：「配置」和「日志」各自打开独立窗口；日志窗口含日志/工具调用两个页签。
- **配置窗口（Agent 模型配置）**：
  - 主模型：本地模式 / OpenAI 兼容接口二选一，默认本地模式。
  - 本地模式：模型类型（MLX-LM / MLX-VLM / ONNX / GGUF）、本地路径与目录选择、`context`/`KV 上限`/`temp`/`top_p`/`min_p`/`top_k`/`repeat`/`batch`/超时，以及「启动时预加载」复选框、「加载模型」「卸载」。`KV 上限` 对应 `kv_cache_max_tokens` / `aux_kv_cache_max_tokens`，设为 0 表示跟随 Context；`batch` 对应 `prompt_batch_size`（辅助模型为 `aux_prompt_batch_size`），控制 prefill 每次前向的 token 上限，`0` 表示使用 MLX 默认值。
  - API 模式：Base URL、API Key、模型下拉与「获取模型」，同一组参数与「应用配置」。
  - 辅助模型：标题后带「启用」复选框（对应 `aux_enabled`，默认关闭）。本地模式同主模型布局，默认 `models/Qwen3.5-2B-MLX-4bit`；未启用时「加载模型 / 重新加载」按钮禁用。
  - 启动时按 `auto_preload_main`（默认开启）预加载主模型；辅助模型仅在 `aux_enabled=true` 时加载。关闭主模型启动预加载后，需在配置窗口手动点「加载模型」；同步结束后的模型恢复不受影响。未指定模型时使用默认主模型 `models/qwen3.5-4b-agentic-coder-v4-4bit-mlx`。加载会真正把参数应用到新创建的模型实例。
- **配置窗口（知识同步）**：工作目录与知识库目录；Embedding（FastEmbed / ONNX / MLX / GGUF）+ 模型名 + 本地路径 +「应用配置」+「同步知识」；Reranker 同布局 +「向量去重」（排查并清理向量库内重复的切片，保留入库时间最早的一份）+「停止同步」。高级区保留 PDF/EPUB OCR 引擎、OCR 模型（MLX / GGUF / ONNX）、mmproj、归一化/二值化、切片与重排开关。其中「启动时自动同步」对应 `rag_auto_ingest`，默认关闭；开启后每次启动会自动增量扫描知识库，首次全量同步可能长时间占用 embedding 模型与磁盘 IO。
  - 高级区的「PDF OCR 设置」行末提供「转换工具」按钮，打开独立窗口，把单个文件或整个目录批量转换为 Markdown。复用知识同步的解析器（PDF / EPUB / HTML / TXT / 图片，Office 仅支持现代格式 `.docx` / `.xlsx` / `.pptx`，不支持旧版 `.doc` / `.xls` / `.ppt`），沿用打开窗口时面板上的 OCR / PDF / 归一化设置快照；不写回 `config.json`，也不重建 Agent。目录递归时保留相对子目录结构，输出目录位于输入目录内会自动跳过；同批同名（如 `a.txt` 与 `a.html`）依次生成 `a-2.md`、`a-3.md`；单个文件失败不会中断整批。进度区显示当前文件、当前环节（解析 / 图片归一化 / 图片识别 / 清洗与 Markdown 转换）与总体进度，关闭窗口会取消仍在进行的转换。
- 确认弹窗默认按钮为「拒绝」；`ask_user` 的回答作为工具结果继续同一轮推理。
- RAG 状态在后台线程查询，不在主线程建库/COUNT；状态只读取来源总数与前 256 个来源名，大知识库不会把全部路径读入内存。连续应用配置只生效最后一次，旧同步完成后只重载它自己的引擎，不会影响新 Agent。

配置保存在 `~/.base-agent/config.json`，可在配置窗口修改；工作区默认 `~/base-agent-workspace`。

## 核心设计

### 1. 三态内核

`KernelLoop` 只有三个状态：`THINK → ACT → CONSOLIDATE`。LLM、工具分发、取消、日志全部通过 `KernelRuntime` 闭包注入，内核不依赖 Agent 与具体工具实现。

- **L0**：工具返回即捕获，硬上限 20k 字符。
- **L1**：注入前按预算裁剪工具结果（head + tail，HTML 标签白名单去噪，保留源码尖括号）。
- **L2**：旧轮次折叠为 `<summary>` / 工具名摘要。
- **L3**：超预算时折叠到 `raw_turns + 1` 轮，每 8 轮主动 consolidate（配置值下限 8）；raw 窗口至少覆盖一个 consolidate 周期，避免每个用户回合都因窗口滑动而重建 KV cache；连续重复摘要按正文合并。
- **L4**：`start_long_term_update` 或长任务结束时用 LLM 蒸馏事实、经验与技能；自动续跑会跨段累计蒸馏请求，不会因分段而丢失。

### 2. 工具与能力层

原子工具：`file_read` / `file_write` / `file_modify`（别名 `file_patch`）、`code_run`、`shell_run`、`web_scan`、`webexec_js`（别名 `web_execute_js`）、`ask_user`。

能力工具：`update_plan`、`update_working_checkpoint`、`start_long_term_update`、`work_memory`、`memory_graph`、`memory_search`、`skill_search`、`skill_semantic_search`、`skill_load`、`skill_save`、`rag_search`、`rag_outline`、`rag_status`、`rag_ingest`、MCP 动态工具。

所有文件类工具路径锁定在 workspace 内，并解析 symlink 防止逃逸。`code_run` / `shell_run` / MCP 调用按配置触发确认。

技能保存会同时写 `SKILL.md` 与 `skill.json`：前者供模型加载，后者供关键词索引扫描；只有主技能目录是 `<workspace>/skills` 时才额外镜像到 `.agent/skills`，不会出现 `.agent/.agent/skills` 的嵌套路径。索引入口缺失时会自动回退到实际存在的 `SKILL.md` / `prompt.md`。

技能建议只针对真正的多步骤任务：寒暄、问候、简单问询与一次性查询不进入重复请求统计；只有本轮创建过 ≥2 步的计划且所有步骤都已 `done`，或模型显式调用 `start_long_term_update`，才会提示固化。没有计划的对话也不会注入 `skill_save` 的周期性 TIPS，系统提示同样禁止对这类请求询问是否保存技能。

### 2.1 安全与资源

- Python `code_run` 通过 `sitecustomize` 注入写护栏：`builtins.open` / `io.open` / `os.open` 的写模式只能落在 workspace 内，同时放行 `subprocess` 的 int fd 管道。
- 子进程超时/取消时递归终止进程树（`pgrep -P`），避免 `npx`/shell 留下孤儿；失败时把 stdout/stderr 尾部并入错误，模型能看到真实报错。
- 风险判定覆盖裸 `~`、`/`、`..`、`$HOME`/`${HOME}` 等目标，且不会把 `rm -rf ./` 误判为工作区外操作。
- `web_scan` 首请求与每次重定向都拒绝 loopback/私网/link-local/云元数据地址（SSRF）；响应体上限 8MB、重定向 5 次，任务取消时立即中止下载。
- MCP stdio 客户端：启动串行化；`shutdown()` 后不会被 `callTool` 复活；接收缓冲有界；服务端 `tools/list_changed` 会同步更新工具注册表与 schema 缓存。
- `AtomicFile` 使用同目录临时文件 + 原子替换，不再「先删旧再 move」。
- 所有 `URLSession`（OpenAI/Web/MCP 之外的 HTTP 工具）在 Agent 关闭或配置切换时 `invalidateAndCancel`。
- MLX 缓存上限显式压到 1GB；配置切换时清理模型实例缓存，避免旧权重常驻。卸载/切换路径不再同步调用 `MLX.Memory.clearCache()`，防止与仍在 GPU 线程执行的 `asyncEval` 并发释放 command encoder 导致 SIGSEGV。
- SQLite 事务只在 COMMIT/RELEASE 成功后递减嵌套深度，提交失败也保持深度非负，避免后续事务被误判为 savepoint。
- 流式回复每 50ms 合并一次；内容更新时滚动不再逐帧动画，日志窗口复用同一个 `DateFormatter`，长会话下重绘更稳定。
- LLM 流带 200ms 取消看门狗：服务端静默时用户点「终止」也能立即退出，不必等到请求超时。
- 知识图谱写盘防抖；裁剪只在真正超限时执行，向量缓存按文本去重并分批嵌入；切换 workspace 时释放旧共享实例。

### 3. RAG 文档解析

| 格式 | 实现 |
| --- | --- |
| PDF（默认 `ovis`） | 先走 PDFKit 文本层：有效字符 ≥ 50 的页面直接提取并清洗为 Markdown；文字页中检测到表格时只裁剪表格区域（300 DPI）→ 归一化 → OvisOCR2 转 Markdown → 拼回原位置；文本层不足的扫描页才整页渲染（200 DPI，最长边 1408）→ OvisOCR2。纯文字且无表格的 PDF 不加载 Ovis。 |
| PDF（可选 `vision`） | 同样先走 PDFKit 文本层（≥ 50）；扫描页 → 统一归一化（缩放/灰度/对比度/锐化/Otsu 二值化）→ Apple Vision `VNRecognizeTextRequest` |
| EPUB（文本） | ZipFoundation 解包 + AEXML 解析 container/OPF/NCX + SwiftSoup 清理 XHTML → Markdown（标题/列表/表格/代码块），保留 TOC 大纲 |
| EPUB（扫描版） | 按 spine/图片顺序提取整页图片 → 统一归一化（缩放/灰度/对比度/锐化/可选 Otsu）→ Apple Vision 或 OvisOCR2 OCR → 按页标记拼接，与 PDF 流程一致 |
| DOCX / XLSX / PPTX | ZipFoundation + AEXML 解析 OOXML，输出 Markdown 标题与表格 |
| HTML | SwiftSoup 提取标题/列表/表格并转 Markdown |
| TXT / MD / CSV / JSON | 原生解析并清洗；CSV 转 Markdown 表格；JSON/JSONL 包成 ```` ```json ```` 代码围栏；支持 UTF-8/UTF-16 BOM 与 GB18030 |

统一清洗（`DocumentCleaner`）会先暂存围栏代码块与行内代码（占位符使用私用区字符，不会被控制字符归一化破坏），再处理 Ovis OCR 输出的 HTML `<table>`：简单表格转成 GFM Markdown（兼容缺失闭合标签、`<br>`），而含 `colspan`/`rowspan` 合并单元格、嵌套表格、`caption`、多行表头或单元格内列表/图片/公式等无法无损表达的复杂表格会保留原始 HTML `<table>`，避免信息丢失。之后才清理标签/实体/噪声。HTML 解码会把 `&amp;`、`&nbsp;`、`&hellip;` 等还原为实际字符，而不是统一替换成空格；块级标签保留段落换行，内联标签直接删除；脚本/样式/SVG/iframe 等容器整块移除，属性与 CSS 残片单独清理。自动识别 Markdown 后，标题、列表、引用、链接、表格、分隔线、代码块与缩进都会被保留，纯文本则把链接压成文字并去掉 URL/邮箱。Unicode 层会删除 NBSP/零宽字符/软连字符/BOM/替换字符（U+FFFD）/变体选择符/私用区/非字符与孤立组合符；合并汉字之间的 OCR 空格（「新 中 新」→「新中新」）；把 `......` 归一为 `……`、压缩重复中文标点；连续装饰符号与符号密度过高的噪声行会丢弃，连续重复行只保留一条。中文硬换行在两侧都是 CJK 时不补空格；纯数字行只在页首/页尾 3 行内视为页码，正文中的年份/编号不会被误删。最后做密度压缩：去掉行尾空格，页标记、连续列表项、表格行、引用行之间不保留空行；短 OCR 行直接合并，长段落之间保留一个空行；标题、代码块与保留的 HTML 表格前后保留结构分隔。实测一本 2101 行的 OCR Markdown 从 1050 个空行压到 47 个。清洗规则变化会使 `pipeline_version` 递增，下一次同步会重建旧文件的索引。

`rag_pdf_engine` 与 `rag_epub_ocr_engine` 接受 `ovis` / `vision` / `hybrid`；旧配置中的 `docling` / `marker` 自动归一化为 `ovis`。`hybrid` 是 PDF 的推荐模式：扫描页按 8 页一批渲染，批内非表格页并发走 Apple Vision OCR（跨文件总量受全局信号量限制到 CPU 核心数），只有检测到表格线框或 Vision 文本过少的页面才串行交给 OvisOCR2 保留表格结构；纯文本页仍走 PDFKit 文本层，文本页中的表格继续交给 Ovis。文件级由 `rag_ingest_concurrency` 控制（默认 6、上限 6），但纯 `ovis` 模式的 VLM 调用共享同一容器并串行执行，实际 OCR 并发为 1；`hybrid` 的 Vision 页可随文件 worker 并行。扫描版 EPUB 默认使用 `vision`（无需额外模型），选择 `ovis` 时需要可用的 OvisOCR2 权重；两条路径都会先做同一套图像归一化，`rag_ocr_preprocess` / `rag_ocr_binarize` 可分别关闭归一化与二值化。

OvisOCR2 的 MLX 权重使用 `model_type=qwen3_5`（Qwen3.5-0.8B 后训练），可直接复用 mlx-swift-lm 的 Qwen3.5 VLM 实现；真正未注册的 Ovis2 架构会明确报错。

### 4. 切片

`TextChunker` 使用 Apple **NaturalLanguage**（`NLTokenizer`）完成句子与词级切分：

1. 结构切片：按 Markdown / 中文章节 / 编号标题切分章节；
2. 语义切片：相邻句子 embedding 相似度低谷断开，embedding 不可用时退化为词重叠；
3. 递归回退：段落 → 换行 → 句号 → 空格 → token 窗口，严格遵守 token 预算；
4. 块间按 token 数添加 overlap（计入总预算）。

默认 `chunk_size=500 token`、`overlap=50 token`（10%，含在 500 内）。同步时优先用模型自带 `tokenizer.json` 做精确计数（不含特殊 token），分词器不可用时退回标定过的保守估算器（`EmbeddingTokenEstimator`）：以 bge 实测，纯中文约 1 字 ≈ 1 token，中英混排约 1.8 字符/token，英文约 3.5 字符/token；因此 500 token 大致对应 500 字中文或 600–900 字混排。正文预算先扣除 overlap 与分隔符，最终块连同 overlap 不超过 500，加上 `[CLS]/[SEP]` 为 502，不会被 bge-small-zh 的 512 token 窗口截断；若把切片大小调到超过模型窗口，同步时会自动压到窗口上限（bge 为 510），overlap 同步压到上限的 10%，设置面板会显示「实际生效：N token」。所有切分路径都在句子结束处断开；旧配置里的默认值 800/40 或 800/80 会在加载时自动迁移到 500/50，自定义组合保持不变。

切片参数（大小 / overlap / 语义切片开关）只有设置面板一个来源：切片环节统一读 `RAGEngine.resolvedChunkingOptions`，检索呈现的单条上限取 `max(配置值, 切片 token × 1.5 + 64)`（不再固定 1500 字符截短），而切片参数、embedding 后端与目录都进 manifest 的 ingest 签名，改任一项都会逐文件重建；「加载主/辅助模型」按钮在白名单（仅 LLM 参数）之外的配置有变化时会走完整 `applyConfig` 重建 Agent 与 RAGEngine，不会出现面板改了切片、运行时仍按旧快照切片。中途停止后已完成的文件保留，下次从剩余文件继续。

多文件同步默认使用 6 个 worker（`rag_ingest_concurrency`，1...6）：解析/OCR 与其它文件的切片、embedding、入库流水线重叠。embedding 仍由共享调度器串行进入模型，子批从 `rag_embedding_initial_batch` 开始（默认 16，运行时夹到 [下限, 上限]），按 `rag_embedding_batch_step` 步进（默认 2，1...16），范围由 `rag_embedding_min_batch`（默认 2，2...64）到 `rag_embedding_max_batch`（默认 32，2...256）；每 `rag_embedding_probe_batches` 批（默认 2）或 `rag_embedding_probe_seconds` 秒（默认 2）根据吞吐和进程内存自动上探/下探，进程 footprint 超过 `rag_embedding_memory_limit_mb`（默认 6144MB）立即下调；吞吐变化小于 `rag_embedding_throughput_tolerance`（默认 0.03 = ±3%）视为测量噪声，不调整。实测 bge 477 token 长块在 batch 4–16 已到吞吐平台，默认 32 兼顾短块；同 batch 重复测量的噪声可达 ±20%，需要更稳时可把容差调到 0.08–0.10。这 8 个子批参数都可在设置窗口「知识同步配置」的高级区同一行调整（顺序：下限、上限、步进、启动预设、探测批次、探测窗口、内存阈值、吞吐容差）。日志只保留每本书的开始、每分钟进度和完成汇总（含各阶段耗时与最终子批大小）。

### 5. SQLite + sqlite-vec 存储

`VectorStore` 的设计约束：

- **主表与向量表分离**：`chunks` 存正文/元数据，`chunk_vectors`（vec0）只存向量；
- **单事务写入**：`chunks` + `chunks_fts` + `chunk_vectors` 在同一 `BEGIN IMMEDIATE` 事务内完成，失败整体回滚；
- **单事务删除**：先取 rowid，再按批删除三张表，仍在同一事务内；
- **两段式 KNN**：先查 `chunk_vectors` 拿 rowid，再回 `chunks` 按 rowid 取正文，绝不 JOIN 虚拟表；
- **过滤**：不依赖虚拟表过滤，统一 over-fetch 后在第二阶段过滤；定向检索自动放大召回；
- **混合检索**：FTS5（预分词，含 CJK bigram） + 向量，RRF 融合；
- sqlite-vec 静态编入进程（`SQLITE_CORE` 模式），无需安装扩展；若初始化失败则显式退化为精确余弦扫描，并在 `rag_status` 中标注。

向量维度写入 `meta`；维度变化时自动清空旧索引并同步清空 manifest，避免 vec0 维度不匹配或「全部跳过」导致知识库为空。若向量库被删除但 manifest 残留，同步会自动识别并全量重建。空库遇到首次推理才发现的真实维度时会按实际输出维度重建。

检索与写入/删除共用同一把锁，两段式查询之间不会插入提交；FTS 与向量两路各自 over-fetch。解析引擎、OCR 模型、切片参数或 embedding 模型变化会按文件写入 ingest 签名：下一次同步只重建签名不一致的文件，已完成的文件不会重复解析。每个文件入库后立即落盘 manifest，用户中途「停止同步」或进程退出后，下次同步直接从剩余未完成文件继续。单个文件变更后解析失败会清掉它的旧索引并在下次重试。

manifest 条目同时记录 `chunkCount` 与 `complete`：增量跳过前会校验库内该来源的实际切片数、Markdown 缓存是否存在；数量不一致、旧版条目缺少完成标记、或有失败页的文件都会重新处理，不再出现「实际没完成却被跳过」的情况。有 OCR 失败页的文件仍保留已成功内容，但标记为未完整完成，下次同步自动重试。

同步开始会先按「文件大小 + 采样指纹」分组，同一本书放在不同目录时只解析一个代表文件；只有大小相同的文件才计算采样指纹，避免对绝大多数唯一文件做无谓读取。同步完成后按清洗正文的 `content_hash` 再做一次全局去重，保留一个来源，删除其余来源的切片、摘要与 Markdown 缓存。被清理的正文副本会写 manifest tombstone（`duplicateOf`），后续增量直接跳过，不会每轮重复解析。知识库根目录统一用 `realpath` 规范化，避免 `/var` 与 `/private/var` 等 firmlink 导致相对路径和 manifest 键不一致。

知识同步配置区的「向量去重」可随时手动排查向量库：按切片正文 SHA256 分组，每组只保留入库时间最早的一份，单事务删除重复的 `chunks` / `chunks_fts` / `chunk_vectors`；整篇被清空的来源写 `duplicateOf` tombstone 并删除 documents/Markdown 缓存，后续增量同步直接跳过，避免重复向量继续制造检索噪声。去重期间与同步、模型加载/卸载、对话发送互斥，结束后自动刷新 RAG 统计。

### 6. 本地模型

- LLM / VLM：**mlx-swift-lm**（`MLXLLM` / `MLXLMCommon` / `MLXVLM`）、**llama.cpp C API**（GGUF）与 **onnxruntime-swift**（ONNX）。
  - MLX-LM / MLX-VLM：原生内存内推理；
  - GGUF：`LlamaCppLLMService` / `LlamaCppVLMService` 在进程内直接加载权重（Metal GPU offload、Jinja chat template、工具调用、mtmd 多模态），不启动 `llama-server`；Embedding 使用 pooled embedding，Reranker 要求 RANK pooling；
  - ONNX LLM：`ONNXLLMService` 用 onnxruntime-swift 直接运行 decoder 图（`input_ids` / `attention_mask` / `position_ids` + `past_key_values`），支持流式生成与工具调用；ONNX VLM 因缺少通用图适配暂未接入，会明确报错而不是回退 Python。
- Embedding / Reranker：**onnxruntime-swift**（`OnnxRuntimeBindings`）+ `onnxruntime-swift-package-manager`，或 **mlx-swift-lm** 的 `MLXEmbedders` / `MLXRerankers`，或 GGUF（llama.cpp pooled embedding / RANK pooling）。类型支持 ONNX / FastEmbed / MLX / GGUF。
- Tokenizer：**swift-transformers** 的 `Tokenizers` 模块，读取模型目录内 `tokenizer.json`。
- OpenAI 兼容 SSE 按事件边界聚合多行 `data:`（服务端把 JSON 拆成多行时不会丢增量），并忽略注释/心跳行。
- 本地 MLX 的 `maxTokens` 会按「contextWindow - prompt 估算」动态收紧；文本兜底工具调用只接受本次请求声明过的工具名。
- **跨轮 KV cache 复用**：首轮完整 prefill 后保留 raw `[KVCache]` 与 `LMOutput.State`；后续轮次只把新增的 user/tool 消息追加进缓存（已生成的 assistant 文本本就在缓存里），不会每轮重算上万 token 的历史。tool 结果会改写成 `<tool_response>` 包裹的 user 消息再渲染，保证只渲染增量时与完整 prompt 逐 token 对齐（否则会漏掉 `<|im_start|>user` 头，上下文错位导致输出异常）。Qwen3.5 的 Mamba/GDN 缓存不可裁剪，因此采用 append-only；当累计 token 接近上下文上限时回退为一次完整重建。历史折叠/回滚、换模型、清空对话（`Agent.reset`）、调用失败或取消都会清空缓存，下一轮自动重建。
- **加载后预热**：主模型加载成功后，立即用稳定的 `BASE_SYSTEM_PROMPT` + 当前工具清单渲染出不带 generation prompt 的 system 前缀（当前约 4700 tokens），只做 prefill 并保留 cache/state。首次正式请求只需追加 user 增量（实测 `cached=4736 delta=216`），不再等整段 prompt 重算。预热前会先等待 MCP 连接完成；工具清单变化时按 `tools` 指纹自动重建，不会带错前缀。GGUF 后端同样会预热 system + tools + user 头前缀；实测 `cached=328/341`。Qwen3.5 的混合 SSM 层不支持按位置裁剪 KV，因此仅当前缀完全一致时续写，否则整段重建，不会输出错位上下文。
- **重建前缀复用**：历史折叠、raw 窗口滑动、rollback 导致消息前缀变化时，从预热时保留的 system+tools 快照复制 KV cache，只 prefill 动态历史；system 前缀不再重复计算。实测 5471-token 重建从约 34s 降到 5.3s（805 token），另一次 626-token 重建约 4.1s。
- **prefill 批量**：`prompt_batch_size` / `aux_prompt_batch_size`（默认 512，最小 64）映射到 `PrefillParameters.stepSize`，用于控制长 prompt prefill 每次前向的 token 数；调大通常能提高吞吐，但会占用更多内存。配置窗口的 `batch` 字段与 JSON 键一一对应。
- **投机解码**：当前主/辅模型均为 Qwen3.5 hybrid（GDN/Mamba 层），`MambaCache.isTrimmable == false`，mlx-swift-lm 的 `SpeculativeTokenIterator` 会在初始化时直接拒绝；两个模型的实际权重中也没有 MTP 头（config 虽写 `mtp_num_hidden_layers=1`，safetensors 中 `mtp` key 为 0），因此无法复用辅助模型起草。若后续换成全注意力、缓存可裁剪且共享 tokenizer 的主/辅模型，可再接入该路径。
- **内存上限**：`mlx_memory_limit_mb`（默认 0 = min(物理内存 65%, 9.6GB)）是 MLX 权重+KV+buffer 的总闸门；`mlx_cache_limit_mb`（默认 512，运行时上限 1024）只限制 MLX 空闲 buffer。主模型 `kv_cache_max_tokens`（默认 32768，约 1GB KV）和辅助模型 `aux_kv_cache_max_tokens`（默认 16384，约 200MB）是 KV 硬上限，实际上下文取 `min(配置上下文, 上限)`，内核预算同步收紧。完整 prompt 超过预算时明确报错，不会继续分配 KV。辅助模型每次后台摘要/事实抽取后立即失效 KV；预热前缀快照只在 ≤8192 token 时保留。RAG 检索 `topK` 封顶 20，避免 reranker 批量被模型参数放大。
- **模型常驻策略**：主模型与 Reranker 加载后常驻不自动卸载；辅助模型仅在 `aux_enabled=true` 时常驻，关闭时 Agent 的 `auxLLM` 保持 `nil`（RAG 精简跳过，后台记忆/总结回退主模型）；Embedding 与 OCR 只在检索/解析时使用，空闲 3 分钟后释放权重，下次按需重载。provider 对象保留，不影响任何工具调用。
- 为提升缓存命中，system prompt 只保留稳定前缀（`BASE_SYSTEM_PROMPT`），工作记忆、当天事实、checkpoint、技能与知识库状态等动态内容拼在 user 消息尾部；同一轮内的动态上下文做缓存，避免工具循环中反复变化。运行日志会输出 `[mlx-kv] reuse=… cached=… delta=…` 与 `[mlx] prompt=… cached=… generated=…`，可直接确认本轮复用了多少 token。
- nomic embedding 使用 `search_document: ` / `search_query: ` 前缀，mean pooling + L2 归一化；MLX embedding 从 `config.json` 读取隐藏维度。
- 模型查找顺序：配置显式路径 → `<workspace>/models/<name>` → `~/base-agent/models/<name>` → HuggingFace / FastEmbed 缓存。旧配置的 `models/<name>` 相对写法会自动去掉 `models/` 前缀后在这些根目录下查找。
- 主/辅模型支持**独立预加载与卸载**：启动时按 `auto_preload_main` 与 `llm_provider` 决定是否预加载主模型，辅助模型仅在 `aux_enabled=true` 时预加载；「加载模型」只替换对应槽位的模型实例并读入权重，另一个模型保持常驻；「卸载」只作用于当前模型。加载参数（context / temp / top_p / min_p / top_k / repeat / batch / thinking）进入模型实例缓存键，修改后会创建新实例。
- 生成、加载、预热、知识同步期间禁用模型加载/卸载/同步按钮与消息发送；配置切换会等待旧请求结束后丢弃其流式缓冲，迟到的 50ms flush 也不会写进新气泡。
- **知识同步期间释放模型**：同步开始前卸载已加载的主/辅模型并清理 LLM 缓存，同步结束（成功/失败/取消）后按同步前状态自动重新加载，主模型会重新预热。开始与恢复两处都会检查 `aux_enabled`：关闭时日志写「主模型（辅助模型未启用）」，不会卸载、创建或重新加载辅助模型。日志依次输出「已卸载主模型 / 已卸载辅助模型 / 重新加载 / 主模型预热结束」；模型未加载时输出「无需卸载」，API 模式的主模型客户端保持原样。同步期间配置窗口的加载按钮会拒绝操作。`applyConfig { 同步 }` 路径不再并行预加载模型，避免同步期间模型又常驻内存。
- 「停止同步」会立即把状态栏改为「正在停止同步…」并禁用停止按钮，同时设置 RAG 取消标志并取消同步 Task；RAG 在文件边界、embedding 批次边界以及解析完成后都会检查取消，不再等整个文件/长 OCR 完成才响应。

PDF Ovis 引擎按 `rag_ocr_model_path` / `rag_ocr_model_name` 加载，不受 `vlm_provider` 影响。

## 配置示例

```json
{
  "workspace": "~/base-agent-workspace",
  "knowledge_base": "",
  "llm_provider": "local",
  "local_llm_backend": "mlx-lm",
  "local_llm_model": "models/qwen3.5-4b-agentic-coder-v4-4bit-mlx",
  "local_llm_enable_thinking": true,
  "auto_preload_main": true,
  "aux_enabled": false,
  "aux_llm_provider": "local",
  "aux_local_llm_backend": "mlx-lm",
  "aux_local_llm_model": "models/Qwen3.5-2B-MLX-4bit",
  "aux_max_context_tokens": 32768,
  "embedding_provider": "local",
  "local_embedding_backend": "mlx",
  "local_embedding_model": "bge-small-zh-v1.5-mlx",
  "local_embedding_model_path": "models/bge-small-zh-v1.5-mlx",
  "reranker_provider": "local",
  "local_reranker_backend": "onnx",
  "local_reranker_model": "jina-reranker-v2-base-multilingual-onnx",
  "local_reranker_model_path": "models/jina-reranker-v2-base-multilingual-onnx",
  "rag_rerank_threshold": 0.45,
  "rag_pdf_engine": "ovis",
  "rag_epub_ocr_engine": "vision",
  "rag_ocr_preprocess": true,
  "rag_ocr_binarize": true,
  "rag_ocr_model_name": "OvisOCR2-0.8B-MLX-4bit",
  "rag_storage_backend": "sqlite",
  "rag_sqlite_vec_enabled": true,
  "rag_rrf_k": 60,
  "rag_vector_overfetch": 4,
  "rag_fts_overfetch": 4
}
```

旧版 Python 配置键（`llm_mode`、`llm_backend`、`llm_model_path`、`aux_llm_*`、`rag_embedding_backend`、`rag_embedding_model_path`、`rag_rerank_backend`、`rag_rerank_model_path`、`rag_rerank_mlx_model_path`）在 canonical 键缺失时自动兼容；`mlx` / `mlx_lm` 归一化为 `mlx-lm`，`llama.cpp` / `llama_cpp` 归一化为 `gguf`，`openai` 归一化为 `api`。

使用纯 API 模式时设置：

```json
{
  "llm_provider": "api",
  "base_url": "https://api.openai.com/v1",
  "api_key": "sk-...",
  "model": "gpt-4o-mini"
}
```

## 项目结构

```
Package.swift
Sources/
  CSQLiteVec/                  # sqlite-vec v0.1.9 amalgamation（静态编入）
  BaseAgentCore/
    Core/                      # JSONValue / ChatMessage / Tool 协议 / Config
    Kernel/                    # 三态自动机、Session、预算、L0-L4 压缩
    LLM/                       # OpenAI 兼容 SSE 客户端 + LLMService 协议
    Local/                     # MLX / ONNX / Tokenizer / 模型注册表
    RAG/
      Parsers/                 # PDFKit / Ovis / Vision / EPUB / Office / HTML
      Chunking/                # NaturalLanguage 切片与大纲
      Storage/                 # SQLite、sqlite-vec、FTS5、VectorStore
      RAGEngine.swift          # 同步、Manifest、检索、rerank
    Tools/                     # 原子工具、RAG/记忆/技能/MCP 工具
    Memory/                    # 工作记忆、长期记忆、知识图谱
    Skills/                    # 技能发现、索引、语义检索
    Agent/                     # Agent 门面、工具工厂、系统提示
  BaseAgentApp/                # SwiftUI 界面
Tests/BaseAgentCoreTests/      # XCTest
docs/软件设计文档.md
```

## 已知限制

- PDFKit 不暴露内嵌图片与表格区域坐标：Ovis 引擎用「文本列对齐」启发式判断表格页并整页交给 VLM，未实现参考项目的表格裁剪回填与内嵌原图优先。
- EPUB 以 spine 文件为章节单位，未按 TOC 锚点二次切章。扫描版 EPUB 仅在「章节文字层低于阈值且含图片」时对该章图片逐页 OCR；纯 CSS 背景图、SVG 矢量图（非 `<image>` 引用）和加密字体渲染的文字不在识别范围内。
- 子进程未使用 `setsid`，超时/取消通过递归进程树终止实现；极端的双重 fork 守护进程仍可能逃逸。
- 本地 MLX/ONNX/GGUF 权重未随仓库提供；首次使用需自行下载到 `<workspace>/models/<name>` 或 `~/base-agent/models/<name>`。真实模型推理需在目标机器验证。
- 主/辅 LLM 支持 MLX-LM / MLX-VLM / GGUF / ONNX；OCR VLM 支持 MLX / GGUF+mmproj（ONNX VLM 暂无通用图适配）；Embedding / Reranker 支持 ONNX / FastEmbed / MLX / GGUF。FastEmbed 复用 ONNX 运行时及其缓存目录。GGUF 与 ONNX 均在进程内运行，不需要外部 server 或 Python。
- ONNX LLM 的空 KV 按 float32 + `config.json` 的层数/KV 头数/head_dim 初始化；fp16 past 或非标准 KV 布局的导出会给出明确错误，需要针对性适配。
- 本机已实测：GGUF LLM（Qwen3.5-9B，CPU 与 Metal GPU，工具调用正常）、GGUF VLM（Qwythos-9B + mmproj，OCR 输出正确）、GGUF Embedding（nomic-embed-text-v1.5，768 维）、GGUF system 前缀预热复用、MLX-VLM 主模型（Qwen3.5-2B）。ONNX LLM、ONNX VLM 与 GGUF Reranker 的代码路径已接入，但当前机器没有对应模型，未做推理实测；缺少模型或运行时会给出明确错误。
- 配置窗口覆盖主要参数；MCP 服务器列表等高级配置仍可直接编辑 `~/.base-agent/config.json`。


## License

Apache-2.0（见 [LICENSE](LICENSE)）。
