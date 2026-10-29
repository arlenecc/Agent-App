# 闲时后台逐章摘要与目录增强 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 闲时增量生成每本书的逐章摘要与全书摘要，保守修复目录，并让 `rag_outline` 优先返回增强版本。

**Architecture:** `VectorStore` 持久化章节/全书摘要与任务状态；`OutlineResolver` 择优解析器/正则大纲并切分章节；`BookSummaryEngine` 做 map-reduce 与增量续跑；`BookSummaryCoordinator` 负责 15/5 分钟节流、空闲会话与临时模型生命周期；`RAGEngine`/`RagOutlineTool` 输出增强结果并改为 SQL 限量文档查找。

**Tech Stack:** Swift 6（v5 语言模式）、SQLite、XCTest、SwiftPM。

---

## 文件结构

- 新建 `Sources/BaseAgentCore/RAG/Summaries/BookSummaryModels.swift`：摘要记录、章节、进度、tick 结果。
- 新建 `Sources/BaseAgentCore/RAG/Summaries/OutlineResolver.swift`：解析器大纲锚定、质量评分、保守修复、章节切分。
- 新建 `Sources/BaseAgentCore/RAG/Summaries/BookSummaryEngine.swift`：token 预算、分段、map-reduce、全书摘要、任务推进。
- 新建 `Sources/BaseAgentCore/RAG/Summaries/BookSummaryCoordinator.swift`：节流、会话、候选选择、临时模型。
- 修改 `Sources/BaseAgentCore/RAG/Storage/VectorStore.swift`：新表、`content_hash`、摘要 CRUD、候选查询、文档查找、级联清理。
- 修改 `Sources/BaseAgentCore/RAG/Models.swift`：`DocumentRecord.contentHash`。
- 修改 `Sources/BaseAgentCore/RAG/RAGEngine.swift`：入库写 content hash、增强大纲查询、SQL 文档查找。
- 修改 `Sources/BaseAgentCore/Tools/RAGTool.swift`：`rag_outline` 输出增强版本与歧义候选。
- 修改 `Sources/BaseAgentCore/Local/LocalModelFactory.swift`：`makeTransientAuxLLM(config:)`。
- 修改 `Sources/BaseAgentApp/AppModel.swift`：空闲会话启动/停止、临时模型状态、退出收尾。
- 修改 `Sources/BaseAgentApp/SettingsView.swift`：摘要状态行。
- 新建测试：
  - `Tests/BaseAgentCoreTests/BookSummaryStoreTests.swift`
  - `Tests/BaseAgentCoreTests/OutlineResolverTests.swift`
  - `Tests/BaseAgentCoreTests/BookSummaryEngineTests.swift`
  - `Tests/BaseAgentCoreTests/BookSummaryCoordinatorTests.swift`
  - `Tests/BaseAgentCoreTests/RagOutlineToolTests.swift`

---

### Task 1: 摘要存储与文档查找

**Files:**
- Create: `Sources/BaseAgentCore/RAG/Summaries/BookSummaryModels.swift`
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorStore.swift`
- Modify: `Sources/BaseAgentCore/RAG/Models.swift`
- Test: `Tests/BaseAgentCoreTests/BookSummaryStoreTests.swift`

- [ ] **Step 1: 写失败测试**

覆盖：

```swift
func testSummaryTablesMigrateAndRoundTrip() throws
func testChapterSummaryIsIncrementalByContentHash() throws
func testUpsertDocumentInvalidatesSummariesOnlyWhenContentChanges() throws
func testSummaryCandidatesSkipCompletedBooks() throws
func testLookupDocumentsPrefersExactPathThenBasenameThenUniqueFuzzy() throws
func testDeleteDocumentCascadesSummaryRows() throws
```

- [ ] **Step 2: 运行测试确认 RED**

Run: `swift test --filter BookSummaryStoreTests`
Expected: 编译失败或断言失败（新 API 不存在）。

- [ ] **Step 3: 实现模型与 SQL**

`BookSummaryModels.swift` 定义：

```swift
public enum BookSummaryMode: String, Sendable, Equatable {
    case resident, temporary
    public var interval: TimeInterval { self == .resident ? 900 : 300 }
}

public enum BookSummaryJobStatus: String, Sendable, Equatable {
    case pending, summarizingBook = "summarizing_book", completed, failed
}

public struct ChapterSummaryRecord: Sendable, Equatable { ... }
public struct ChapterSegmentRecord: Sendable, Equatable { ... }
public struct BookSummaryRecord: Sendable, Equatable { ... }
public struct BookSummarySegmentRecord: Sendable, Equatable { ... }
public struct SummaryJobRecord: Sendable, Equatable { ... }
public struct DocumentSummaryProgress: Sendable, Equatable { ... }
public struct BookSection: Sendable, Equatable { ... }
public struct ResolvedBookOutline: Sendable, Equatable { ... }

public enum BookSummaryTickOutcome: Sendable, Equatable {
    case throttled(waitSeconds: TimeInterval)
    case noWork
    case processedChapter(docID: String, docName: String, sectionIndex: Int,
                          totalSections: Int, chapterCompleted: Bool)
    case processedBook(docID: String, docName: String)
    case failed(docName: String, message: String)
}
```

`VectorStore` 新增方法：

```swift
public func upsertChapterSummary(_ record: ChapterSummaryRecord) throws
public func chapterSummaries(docID: String) throws -> [ChapterSummaryRecord]
public func chapterSummary(docID: String, sectionIndex: Int) throws -> ChapterSummaryRecord?
public func upsertChapterSegment(_ record: ChapterSegmentRecord) throws
public func chapterSegments(docID: String, sectionIndex: Int) throws -> [ChapterSegmentRecord]
public func deleteChapterSegments(docID: String, sectionIndex: Int) throws
public func upsertBookSummary(_ record: BookSummaryRecord) throws
public func bookSummary(docID: String) throws -> BookSummaryRecord?
public func upsertBookSummarySegment(_ record: BookSummarySegmentRecord) throws
public func bookSummarySegments(docID: String) throws -> [BookSummarySegmentRecord]
public func upsertSummaryJob(_ record: SummaryJobRecord) throws
public func summaryJob(docID: String) throws -> SummaryJobRecord?
public func summaryCandidates(limit: Int) throws -> [DocumentRecord]
public func document(docID: String) throws -> DocumentRecord?
public func documentCount() throws -> Int
public func documentNames(limit: Int, offset: Int = 0) throws -> [String]
public func lookupDocuments(query: String, limit: Int = 50) throws -> [DocumentRecord]
public func updateDocumentOutline(docID: String, digest: String, chapters: String) throws
public func deleteSummaries(docID: String) throws
public func summaryLastTickAt() throws -> Double?
public func setSummaryLastTickAt(_ value: Double) throws
public func summaryProgress(docID: String) throws -> DocumentSummaryProgress
```

`documents` 建表增加 `content_hash`；旧库用 `PRAGMA table_info` + `ALTER TABLE` 补齐。
`upsertDocument` 增加 `contentHash: String = ""`；内容哈希变化时删除该书摘要；`clear()`、`rebuildForCurrentConfiguration()`、`adoptCurrentConfigurationLocked()`、`deleteDocument*` 同步清理摘要表。
`DocumentRecord` 增加 `contentHash: String = ""`。

- [ ] **Step 4: 运行测试确认 GREEN**

Run: `swift test --filter BookSummaryStoreTests`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add Sources/BaseAgentCore/RAG/Summaries/BookSummaryModels.swift \
        Sources/BaseAgentCore/RAG/Storage/VectorStore.swift \
        Sources/BaseAgentCore/RAG/Models.swift \
        Tests/BaseAgentCoreTests/BookSummaryStoreTests.swift
git commit -m "feat(summary): add chapter/book summary storage and SQL document lookup"
```

---

### Task 2: 大纲择优与章节切分

**Files:**
- Create: `Sources/BaseAgentCore/RAG/Summaries/OutlineResolver.swift`
- Test: `Tests/BaseAgentCoreTests/OutlineResolverTests.swift`

- [ ] **Step 1: 写失败测试**

```swift
func testParserOutlineWinsWhenAnchoredToMarkdown() throws
func testParserOutlineFallsBackToRegexWhenUnanchored() throws
func testConservativeRepairFillsEmptyAndDummyTitles() throws
func testSectionsHaveStableIndexHeadingPathAndContentHash() throws
func testQualityScoreRejectsNoiseAndRewardsDepth()
```

- [ ] **Step 2: 运行确认 RED**

Run: `swift test --filter OutlineResolverTests`
Expected: 编译失败。

- [ ] **Step 3: 实现 `OutlineResolver`**

公开入口：

```swift
public enum OutlineResolver {
    public static func resolve(markdown: String,
                               parsedOutline: String,
                               docName: String) -> ResolvedBookOutline
}

public struct OutlineCandidate: Sendable, Equatable {
    public var nodes: [OutlineNode]
    public var totalEntries: Int
    public var anchoredEntries: Int
    public var noiseEntries: Int
    public var source: String
}
```

要点：

- `flattenParserOutline`：2 空格缩进还原层级；
- `anchor`：按 Markdown 行做归一化整行匹配，锚定失败条目丢弃；
- `qualityScore` 按 spec 公式；
- 解析器锚定 0 条 → 0 分；
- `repairTitles` 只修空/伪标题/悬空标记，取正文首个可信短行；
- `BookSection` 与节点一一对应，`body` 去掉首行标题；
- `contentHash = SHA256(headingPath + "\n" + body)` 前 16 位；
- `digest = "# 目录\n\n" + DocumentOutline.render(nodes)`；
- `chaptersJSON` 由修复后节点生成。

- [ ] **Step 4: 运行确认 GREEN**

Run: `swift test --filter OutlineResolverTests`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add Sources/BaseAgentCore/RAG/Summaries/OutlineResolver.swift \
        Tests/BaseAgentCoreTests/OutlineResolverTests.swift
git commit -m "feat(summary): resolve best outline and split stable sections"
```

---

### Task 3: 逐章 map-reduce 与全书摘要

**Files:**
- Create: `Sources/BaseAgentCore/RAG/Summaries/BookSummaryEngine.swift`
- Test: `Tests/BaseAgentCoreTests/BookSummaryEngineTests.swift`

- [ ] **Step 1: 写失败测试**

```swift
func testTokenEstimateTreatsCJKAndLatinDifferently()
func testSegmentSplitterRespectsBudgetAndKeepsOrder()
func testChapterSummaryReusesPersistedSegmentsAfterInterruption()
func testChapterSummaryNormalizesToFiftyToOneHundredCharacters()
func testTinySectionSkipsModelCall()
func testBookSummaryMapReducePersistsGroupsAndProducesLongSummary()
```

- [ ] **Step 2: 运行确认 RED**

Run: `swift test --filter BookSummaryEngineTests`
Expected: 编译失败。

- [ ] **Step 3: 实现引擎**

公开入口：

```swift
public final class BookSummaryEngine: @unchecked Sendable {
    public init(store: VectorStore)

    public static func estimateTokens(_ text: String) -> Int
    public static func splitSegments(_ body: String, tokenBudget: Int) -> [String]
    public static func normalizeChapterSummary(_ raw: String) -> String
    public static func normalizeBookSummary(_ raw: String) -> String

    public func resolve(_ document: DocumentRecord, markdown: String) -> ResolvedBookOutline
    public func processNextChapter(document: DocumentRecord,
                                   markdown: String,
                                   llm: LLMService,
                                   now: Date,
                                   isCancelled: () -> Bool) async throws -> ChapterProcessResult
    public func processBookSummary(document: DocumentRecord,
                                   markdown: String,
                                   llm: LLMService,
                                   now: Date,
                                   isCancelled: () -> Bool) async throws -> BookSummaryRecord
}
```

`ChapterProcessResult` 含 `sectionIndex/totalSections/chapterCompleted/bookCompleted/titleRepaired`。

实现：

- `estimateTokens`：CJK 1 token，其他 0.25；
- 分段预算 `min(8000, max(1024, llm.contextWindow - 1400))`；
- 逐段检查 `segment_hash`，命中复用；
- reduce + 一次短结果重试；
- 正文 < 40 字直接兜底；
- 全书摘要分组持久化到 `book_summary_segments`；
- 每章完成更新 `summary_job_state`；
- 目录修复后调用 `store.updateDocumentOutline`；
- 所有 LLM 调用检查 `Task.isCancelled` 与 `isCancelled()`。

- [ ] **Step 4: 运行确认 GREEN**

Run: `swift test --filter BookSummaryEngineTests`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add Sources/BaseAgentCore/RAG/Summaries/BookSummaryEngine.swift \
        Tests/BaseAgentCoreTests/BookSummaryEngineTests.swift
git commit -m "feat(summary): incremental map-reduce chapter and book summaries"
```

---

### Task 4: 后台协调器与节流

**Files:**
- Create: `Sources/BaseAgentCore/RAG/Summaries/BookSummaryCoordinator.swift`
- Test: `Tests/BaseAgentCoreTests/BookSummaryCoordinatorTests.swift`

- [ ] **Step 1: 写失败测试**

```swift
func testResidentModeProcessesOneChapterPerFifteenMinutes() async throws
func testTemporaryModeProcessesOneChapterPerFiveMinutes() async throws
func testTickNeverProcessesMoreThanOneChapter() async throws
func testCompletedBookMovesToBookSummarySlot() async throws
func testFailureDoesNotAdvanceNextSection() async throws
func testCancelStopsCurrentChapterAndKeepsSegments() async throws
```

- [ ] **Step 2: 运行确认 RED**

Run: `swift test --filter BookSummaryCoordinatorTests`
Expected: 编译失败。

- [ ] **Step 3: 实现协调器**

```swift
public final class BookSummaryCoordinator: @unchecked Sendable {
    public init(engine: BookSummaryEngine,
                store: VectorStore,
                configProvider: @escaping @Sendable () -> AgentConfig,
                residentAuxProvider: @escaping @Sendable () -> LLMService?,
                transientAuxFactory: (@Sendable (AgentConfig) -> LLMService)? = nil,
                now: @escaping @Sendable () -> Date = { Date() },
                log: @escaping @Sendable (String) -> Void = { _ in })

    public func beginSession(mode: BookSummaryMode) async -> Bool
    public func runTick(mode: BookSummaryMode) async -> BookSummaryTickOutcome
    public func endSession() async
    public func cancelCurrent() async
    @MainActor public func statusSnapshot() -> String
}
```

要点：

- 临时模式 `beginSession` 创建/预加载 transient aux，失败返回 false；
- `runTick` 先读 `last_tick_at`，不足 900/300 秒返回 `.throttled`；
- 一个 tick 只处理一个章节或全书摘要；
- 成功处理后才写 `last_tick_at`；
- 失败记录日志、文档本会话重试上限 2 次、不推进 `next_section`；
- `endSession` 等待当前任务、卸载 transient aux、清空会话状态；
- 常驻 aux 只读 `residentAuxProvider()`，不关闭。

- [ ] **Step 4: 运行确认 GREEN**

Run: `swift test --filter BookSummaryCoordinatorTests`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add Sources/BaseAgentCore/RAG/Summaries/BookSummaryCoordinator.swift \
        Tests/BaseAgentCoreTests/BookSummaryCoordinatorTests.swift
git commit -m "feat(summary): idle coordinator with 15/5 minute throttling"
```

---

### Task 5: RAGEngine 与 rag_outline 增强

**Files:**
- Modify: `Sources/BaseAgentCore/RAG/RAGEngine.swift`
- Modify: `Sources/BaseAgentCore/Tools/RAGTool.swift`
- Test: `Tests/BaseAgentCoreTests/RagOutlineToolTests.swift`

- [ ] **Step 1: 写失败测试**

```swift
func testRagOutlinePrefersBookSummaryAndChapterSummaries() async throws
func testRagOutlineShowsProgressWhenPartial() async throws
func testRagOutlineListsAmbiguousCandidates() async throws
func testRagOutlineFallsBackToLegacyDigestWithoutSummaries() async throws
func testDocumentLookupPrefersExactFullPath() throws
```

- [ ] **Step 2: 运行确认 RED**

Run: `swift test --filter RagOutlineToolTests`
Expected: 编译失败或断言失败。

- [ ] **Step 3: 实现**

`RAGEngine` 新增：

```swift
public enum DocumentLookupResult: Sendable, Equatable {
    case found(DocumentRecord)
    case ambiguous([DocumentRecord])
    case notFound
}

public func lookupDocument(_ query: String) -> DocumentLookupResult
public func getEnrichedOutline(docName: String) -> EnrichedDocumentOutline?
public func listDocuments(limit: Int) -> (names: [String], total: Int)
```

`EnrichedDocumentOutline` 含 `record/chapterSummaries/bookSummary/jobProgress/repairedTOC`。
`RagOutlineTool`：

- 无 `book_name`：限量列表 + 总数；
- 有 `book_name`：`lookupDocument`，歧义时列出候选；
- 输出进度、全书摘要、每章摘要目录树；
- 无摘要回退旧 `digest`。

入库时 `upsertDocument(contentHash: RAGEngine.contentHash(text))`。
`getDocumentOutline` 与 `listDocuments()` 内部改为 SQL 限量查询，保持旧签名。

- [ ] **Step 4: 运行确认 GREEN**

Run: `swift test --filter RagOutlineToolTests`
Expected: PASS。

- [ ] **Step 5: 提交**

```bash
git add Sources/BaseAgentCore/RAG/RAGEngine.swift \
        Sources/BaseAgentCore/Tools/RAGTool.swift \
        Tests/BaseAgentCoreTests/RagOutlineToolTests.swift
git commit -m "feat(rag): enriched outline output and unambiguous document lookup"
```

---

### Task 6: 模型工厂与 App 生命周期

**Files:**
- Modify: `Sources/BaseAgentCore/Local/LocalModelFactory.swift`
- Modify: `Sources/BaseAgentApp/AppModel.swift`
- Modify: `Sources/BaseAgentApp/SettingsView.swift`

- [ ] **Step 1: 实现 transient aux**

把 `makeAuxLLM` 的构建逻辑抽成 `buildAuxLLM(config:value:backend:)`；新增：

```swift
public func makeTransientAuxLLM(config: AgentConfig) -> LLMService
```

本地分支不写 `llms` 缓存；API 分支每次新建客户端。

- [ ] **Step 2: AppModel 空闲会话**

新增：

```swift
@Published var bookSummaryStatusText = "章节摘要：空闲"
private var summaryCoordinator: BookSummaryCoordinator?
private var summarySessionTask: Task<Void, Never>?

private func startSummaryIdleSession()
private func stopSummaryIdleSession() async
```

- `buildAgent()` 创建协调器；
- `scheduleIdleUnload()` 3 分钟后先 `startSummaryIdleSession()`，再卸载 Embedding/OCR；
- `cancelIdleUnload()` 取消摘要任务并异步 `endSession()`；
- `prepareForTermination()` 先 `await stopSummaryIdleSession()`；
- 前台活动、同步、配置切换、模型加载统一走 `cancelIdleUnload()`；
- 协调器日志写入 `appendLog`，状态写入 `bookSummaryStatusText`。

- [ ] **Step 3: 设置面板状态行**

在 ANN 状态附近增加 `bookSummaryStatusText` 展示。

- [ ] **Step 4: 构建验证**

Run: `swift build`
Expected: 编译通过。

- [ ] **Step 5: 提交**

```bash
git add Sources/BaseAgentCore/Local/LocalModelFactory.swift \
        Sources/BaseAgentApp/AppModel.swift \
        Sources/BaseAgentApp/SettingsView.swift
git commit -m "feat(app): run background book summaries during idle sessions"
```

---

### Task 7: 全量回归与收尾

- [ ] **Step 1: 全量测试**

Run: `swift test`
Expected: 0 failures（允许既有 skipped）。

- [ ] **Step 2: 修复回归**

只修改本次功能相关代码；记录并忽略环境依赖导致的既有跳过项。

- [ ] **Step 3: 更新文档**

把实施中的关键偏差补回 spec；确认 `rag_outline` 输出示例与实际一致。

- [ ] **Step 4: 提交**

```bash
git add -A
git commit -m "test(summary): full regression for background book summaries"
```

## 自检

- [ ] spec 的 8 个目标均有对应任务；
- [ ] 15/5 分钟节流在协调器测试中直接断言；
- [ ] 临时模型卸载不影响常驻 aux；
- [ ] `rag_outline` 无摘要时仍兼容旧行为；
- [ ] 文档查找歧义不再静默取第一个；
- [ ] 所有表在 clear/delete/rebuild 路径均级联清理。
