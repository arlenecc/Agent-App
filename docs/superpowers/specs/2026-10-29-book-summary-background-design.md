# 闲时后台逐章摘要与目录增强设计

日期：2026-10-29

## 背景

`rag_outline` 目前只能返回 `documents.digest`（实际是目录文本）和 `documents.chapters`（正则抽取的章节 JSON），无法回答「这本书讲了什么」。同时：

- 解析器自带大纲（EPUB 目录、Office 标题）与正文正则大纲各抽一套，`buildDigest` 只按条目数量取多者，`chapters` 始终只来自正则，二者可能不一致；
- 大纲只存标题，没有每章内容摘要，也没有真正的全书摘要；
- 文档查找全表加载后在内存模糊匹配，取第一个命中，同名书会静默选错；
- 后台没有增量维护书籍摘要的任务。

本设计新增「闲时后台逐章摘要」：结合章节目录结构与 Markdown 正文，逐章生成 50–100 字摘要；全部章节完成后再生成 300–500 字全书摘要；过程中保守修复有问题的目录标题；`rag_outline` 优先返回增强后的版本。

## 目标

1. 增量维护每本书的章节摘要，已完成的章节不重复调用模型。
2. 辅助模型启用且可用时常驻调用，每 15 分钟处理一个章节；未启用/不可用时临时加载，每 5 分钟处理一个章节，空闲会话结束卸载。
3. 无论辅助模型是否启用都执行该任务。
4. 摘要按辅助模型上下文预算分段（map-reduce），中途中断只补未完成分段。
5. 目录质量择优：解析器大纲能锚定 Markdown 时优先；否则回退正则大纲。
6. 逐章过程中保守修复空标题、页码/正文伪标题、明显拆碎标题；不重排、不改正常标题。
7. `rag_outline` 优先输出增强目录 + 每章摘要 + 全书摘要；未完成时显示已完成章节概览和进度。
8. 文档查找改为 SQL 侧过滤 + LIMIT；精确路径 > basename 精确 > 唯一模糊；多候选返回歧义列表。

## 非目标

- 不替代向量检索；章节摘要不写入 `chunks`。
- 不在前台对话/同步/ANN 构建期间运行摘要。
- 不自动重写原书 Markdown，只更新 `documents.digest/chapters` 与摘要表。
- 不新增用户可见的手动「开始摘要」按钮；任务由闲时自动驱动。

## 架构

```
AppModel（生命周期 / 空闲会话 / 临时模型）
   └── BookSummaryCoordinator（节流、会话、任务选择、暂停/取消）
          ├── BookSummaryEngine（解析、map-reduce、增量状态）
          │      ├── OutlineResolver（大纲择优 + 保守修复 + 章节切分）
          │      └── VectorStore（摘要表、任务表、文档查找）
          └── LLMService（常驻 aux 或临时 aux）
```

- Core 不依赖 AppKit；AppModel 只负责「何时开始/结束空闲会话」和模型生命周期。
- 协调器通过依赖注入获得配置、常驻 aux、临时 aux 工厂与日志回调，便于测试。
- 所有状态持久化在 `index.sqlite`，跨重启恢复。

## 数据模型

在 `VectorStore.createBaseSchema()` 新增：

```sql
CREATE TABLE IF NOT EXISTS chapter_summaries (
    doc_id TEXT NOT NULL,
    section_index INTEGER NOT NULL,
    heading TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    level INTEGER NOT NULL DEFAULT 1,
    summary TEXT NOT NULL DEFAULT '',
    content_hash TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL,
    PRIMARY KEY (doc_id, section_index)
);

CREATE TABLE IF NOT EXISTS chapter_summary_segments (
    doc_id TEXT NOT NULL,
    section_index INTEGER NOT NULL,
    segment_index INTEGER NOT NULL,
    segment_hash TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL,
    PRIMARY KEY (doc_id, section_index, segment_index)
);

CREATE TABLE IF NOT EXISTS book_summaries (
    doc_id TEXT PRIMARY KEY,
    summary TEXT NOT NULL DEFAULT '',
    outline TEXT NOT NULL DEFAULT '',
    chapter_count INTEGER NOT NULL DEFAULT 0,
    source_hash TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS book_summary_segments (
    doc_id TEXT NOT NULL,
    group_index INTEGER NOT NULL,
    group_hash TEXT NOT NULL DEFAULT '',
    summary TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL,
    PRIMARY KEY (doc_id, group_index)
);

CREATE TABLE IF NOT EXISTS summary_job_state (
    doc_id TEXT PRIMARY KEY,
    status TEXT NOT NULL DEFAULT 'pending',   -- pending/summarizing_book/completed/failed
    next_section INTEGER NOT NULL DEFAULT 0,
    total_sections INTEGER NOT NULL DEFAULT 0,
    last_error TEXT NOT NULL DEFAULT '',
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS summary_runtime (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

`documents` 增加 `content_hash TEXT NOT NULL DEFAULT ''`：

- 入库时写入正文 SHA256；
- `upsertDocument(contentHash:)` 检测到同一文档内容变化时，删除该书全部摘要/任务状态；
- 目录修复走 `updateDocumentOutline(docID:digest:chapters:)`，不触发摘要失效。

章节以 `(doc_id, section_index)` 标识；`content_hash` 由 `headingPath + "\n" + 正文` 计算。章节内容变化时摘要失效，索引不变时直接跳过。

## 大纲择优

新增 `OutlineResolver`：

1. 解析器大纲：按 2 空格缩进还原层级，逐条在 Markdown 中做「整行标题锚定」。锚定失败条目不参与切分。
2. 正则大纲：`DocumentOutline.extract`，天然带字符偏移。
3. 质量评分：

   ```text
   score = 有效条目数 × (1 + 锚定率) × (1 + 0.1×最大层级)
           × (1 - 噪声率) × (0.6 + 0.4×正文覆盖率)
   ```

   - 有效条目：标题非空、非页码/纯数字/日期正文、含句末标点的标题剔除；
   - 锚定率：成功锚定条目 / 解析器条目；
   - 噪声率：伪标题占比；
   - 正文覆盖率：各节点 span 总和 / 正文长度（封顶 1）。
4. 解析器大纲锚定条目为 0 时直接判 0 分，必须回退正则。
5. 胜者同时用于 `digest`、`chapters` JSON 和章节切分，保证三者一致。

保守修复只做：

- 空标题 → 取章节正文第一条可信短行；
- 页码/纯数字/日期正文伪标题 → 同上；
- 只有「第X章 / Chapter / 1.」等悬空标记 → 与正文首行拼接；
- 不改变节点顺序、层级和正常标题。

修复结果写入 `chapter_summaries.title`，并重建 `documents.digest/chapters`。

## 摘要算法

### 章节切分

`BookSection` 与 `OutlineNode` 一一对应（含零正文节点），保证 `section_index` 稳定：

- `title`：修复后的标题；
- `headingPath`：`第一章 > 1.1 引言`；
- `body`：节点范围去掉首行标题后的正文；
- `contentHash`：`SHA256(headingPath + "\n" + body)` 前 16 位十六进制。

### Map

按辅助模型 `contextWindow` 计算预算：

- 估算 token：CJK 字符按 1 token，其它字符按 0.25 token；
- 预留 system/指令 1,000 token、输出 400 token；
- `segmentBudget = max(1024, contextWindow - 1400)`，上限 8,000；
- 按空行分段，超长段再按句号/换行硬切；不重叠。

每段提示要求 30–60 字概括。段结果按 `segment_hash` 持久化；`segment_hash` 未变且已有结果时不重复调用。

### Reduce

把该章节各段摘要按顺序拼给模型，要求严格输出 50–100 字中文摘要：

- 去除 Markdown/引号/“摘要：”前缀；
- 超过 100 字按最后一个句末标点截断；仍超长则硬截 100 字；
- 正常长度章节若 reduce 结果少于 30 字，追加一次「严格 50–100 字」重试；重试仍失败则接受规范化结果；
- 正文少于 40 字的节点不调用模型，生成标题/正文兜底摘要并标记完成。

### 全书摘要

全部章节完成后，下一个节流时间槽生成 300–500 字：

- 每章一行：`- [headingPath] summary`；
- 按上下文预算分组 map，组结果持久化到 `book_summary_segments`；
- reduce 成 300–500 字，短书按实际内容密度放宽下限；
- 保存 `source_hash`（全部章节摘要的哈希），章节变化导致重算；
- 同时把修复后的目录写入 `book_summaries.outline`。

## 调度与生命周期

### 闲时判定

复用 AppModel 现有 3 分钟空闲回收：

- 发送消息、同步、去重、ANN 构建、配置切换、退出、加载/预热模型都会调用 `cancelIdleUnload()`，同时结束摘要空闲会话；
- 3 分钟无活动且 `!isRunning && !isModelLoading && !isWarmingUp && !isSyncing && !isDeduplicating && !isBuildingANN && agent.isBackgroundIdle()` 后启动摘要会话，并继续执行原有 Embedding/OCR 回收；
- 前台活动到达时立即取消当前章节调用；已完成的段结果保留，下个空闲会话继续。

### 节流

- 常驻 aux（`auxEnabled && auxModelLoaded && agent.auxLLM != nil`）：900 秒/章；
- 临时 aux：300 秒/章；
- `summary_runtime.last_tick_at` 持久化，跨重启仍遵守间隔；
- 一个 tick 最多处理一个章节或一次全书摘要；处理失败不推进 `next_section`，本会话对同一文档最多重试 2 次。

### 临时模型

`LocalModelFactory.makeTransientAuxLLM(config:)`：

- API 模式直接返回新的 `OpenAILLMClient`；
- 本地模式复用 `makeAuxLLM` 的构建分支，但不写入共享 `ModelCache<LLMService>`；
- 会话开始 `preload()`，会话结束 `unloadAndWait()` + `close()`，不触碰常驻 aux 缓存；
- 临时模型加载失败：本会话结束，记录日志，下个空闲会话重试。

### 任务选择

候选文档查询（SQL，`LIMIT`）：

```sql
SELECT d.doc_id, d.doc_name, d.digest, d.chapters, d.md_path
FROM documents d
LEFT JOIN summary_job_state j ON j.doc_id = d.doc_id
LEFT JOIN book_summaries b ON b.doc_id = d.doc_id
WHERE d.md_path != ''
  AND (j.doc_id IS NULL
       OR j.status IN ('pending', 'summarizing_book', 'failed')
       -- 完成但没有章节的文档不再进入候选，避免重复失败
       OR (j.status = 'completed' AND b.doc_id IS NULL
           AND j.total_sections > 0))
ORDER BY COALESCE(j.updated_at, 0), d.updated_at
LIMIT ?
```

每章完成更新 `next_section`；最后一章完成置 `summarizing_book`；全书摘要完成置 `completed`。

## rag_outline 输出

新增 `RAGEngine.lookupDocument(_:)` 与 `RAGEngine.getEnrichedOutline(docName:)`：

1. 文档解析：
   - SQL 精确匹配 `doc_name` / `md_path`；
   - basename 精确；
   - `instr(lower(doc_name), lower(query))` 模糊匹配，最多 50 条；
   - 多候选返回歧义列表，不再静默取第一个。
2. 输出：
   - 文档名、ID、章节摘要进度、全书摘要状态；
   - 全书摘要已完成 → `全书摘要:`；
   - 未完成 → `内容概览（基于已完成章节摘要）:` + 已摘要章节拼接，明确标注后台仍在补充；
   - 无章节摘要 → 回退现有 `digest`；
   - `章节结构:` 使用增强目录（修复标题 + 每章摘要），最多 20,000 字符。

## 兼容性

- 既有 `documents` / `chapters` / `digest` 字段保留；
- 旧库打开时自动建摘要表并补 `content_hash` 列；
- `clear()` / `rebuildForCurrentConfiguration()` / `deleteDocument*` 同步清理摘要表；
- 现有 `getDocumentOutline(docName:)`、`listDocuments()` API 保留，内部改为 SQL 限量查询。

## 测试

- `BookSummaryStoreTests`：建表、增量 upsert、级联删除、候选查询、文档解析、内容变化失效。
- `OutlineResolverTests`：解析器大纲锚定/择优/回退、保守修复、章节切分与 hash。
- `BookSummaryEngineTests`：分段、增量跳过、map-reduce、字数规范化、全书摘要。
- `BookSummaryCoordinatorTests`：15/5 分钟节流、单 tick 一章、暂停恢复、临时模型生命周期。
- `RagOutlineToolTests`：增强输出、进度、歧义列表、无摘要回退。
- 全量 `swift test` 回归。

## 验收标准

1. 同一本书重复运行不会重复调用已完成章节；中断章节只补未完成段。
2. 常驻 aux 与临时 aux 分别遵守 15/5 分钟间隔。
3. 临时模型在空闲会话结束后被卸载，且不影响常驻 aux。
4. `rag_outline` 在有摘要时返回每章摘要 + 全书摘要；无摘要时行为与旧版兼容。
5. 同名/模糊文档返回歧义候选，不再静默选错。
6. `swift build` 与全量 `swift test` 通过。

## 实施记录（2026-10-29）

- 协调器改为按空闲会话懒创建：`AppModel.startSummaryIdleSession()` 调
  `RAGEngine.makeBookSummaryCoordinator(...)`，在后台任务中打开向量库；
  会话结束丢弃协调器。避免启动时在同步路径上打开大库，也避免
  `RAGEngine.reload()` 关闭 store 后协调器继续持有一个已关闭的连接。
- `BookSummaryCoordinator.cancelCurrent()` 现在带会话级 `stopRequested`：
  取消后新的 `runTick` 直接返回 `.noWork`，直到下个 `beginSession` 重置，
  避免「取消后外层循环立刻又开一个 tick」。
- 同步、配置切换、卸载辅助模型、退出都会先 `await stopSummaryIdleSession()`；
  同步路径把 `suspendModelsForSync()` 移到后台任务、等摘要会话收尾后再执行，
  防止卸载权重与摘要的在途生成并发。
- `AppModel.init` 末尾主动 `scheduleIdleUnload()`：API 模式或辅助模型未启用时
  `preloadModels` 会提前返回，仍需要空闲计时来启动摘要任务。
- `summaryCandidates` 排除「已完成但没有章节」的文档（`total_sections = 0`），
  避免无标题纯正文在每次空闲会话重复触发注定失败的 `processBookSummary`。
- 复查加固：失败（Markdown 缺失、模型/存储错误、章节摘要与目录不一致）也会
  推进 `summary_job_state.updated_at`，并保留引擎写入的阶段与 `next_section`；
  否则队首 10 本持续失败的文档会在每个新会话反复占满候选窗口，后面的书永久饥饿。
- 复查加固：临时模型在「预加载失败」「加载中被取消」「正常会话结束」三条路径
  都通过 `unloadAndWait()` 等待权重真正释放后再 `close()`；`MLXLLMService` 与
  `MLXVLMLLMService` 补齐 `AsyncUnloadableLLMService` 一致性，避免退出流程与
  仍在执行的 GPU 工作并发。
- 复查加固：协调器读取候选失败不再用 `try?` 静默返回 `.noWork`，而是返回
  `.failed` 让会话按连续失败上限收尾；`checkIdleWork()` 在预加载前按页扫描候选，
  Markdown 不可读的候选会记为失败并推进 `updated_at` 轮转到队尾，避免前 10 本
  失联文档长期占住窗口导致后面的书饿死；单次最多轮转 100 本，达到上限返回
  `.rotated`，AppModel 一分钟后重新检查；提前取消的同步也会恢复空闲计时，
  避免闲时摘要调度永久停摆。
- 实测：`swift build` 通过；全量 `swift test` 367 项、22 项环境依赖跳过、0 失败。
