# RAG 文档处理性能优化实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按既有性能分析的结论，逐项落地 manifest 写入、embedding 跨文件调度、扫描版 EPUB OCR、VectorStore 检索锁、DocumentOutline 正则编译这五处热点优化，并用回归测试证明行为不变。

**Architecture:** 每项优化独立成任务、独立提交：先写失败测试并观察 RED，再做最小实现，最后跑聚焦与全量回归。跨文件/跨章节并发只改变执行调度，不改变输出顺序、页标记、去重与增量同步语义。

**Tech Stack:** Swift 6、SwiftPM、XCTest、SQLite（WAL / FTS5 / sqlite-vec）、ZIPFoundation、Apple Vision、Ovis VLM。

---

## 背景与证据

前序分析留下的基准脚本与结论：

| 热点 | 现状 | 证据 |
| --- | --- | --- |
| manifest 逐文件 merge | 每次读出完整 JSON、全量编码、原子写盘；N 个文件约 O(N²) | `/private/tmp/manifest_bench.swift` |
| embedding 文件级屏障 | 单文件一直持有全局信号量，直到该文件全部切片完成 | `AdaptiveEmbeddingScheduler.embedDocuments` |
| 扫描版 EPUB OCR | 章节循环 `for + await` 串行，单章内部才并发 | `EPUBParser.parse` |
| VectorStore 锁内 FTS | FTS 候选 + hydrate 与写入共用同一把 `NSRecursiveLock` | `VectorStore.searchFTS / hybridSearch` |
| DocumentOutline 正则 | 每次 `extract` / `classify` 重新编译 6 条正则 | `/private/tmp/outline_bench.swift` |
| FTS 预分词 / SQLite 插入 | 有基准，但尚无「低风险且可量化」的替代实现 | `/private/tmp/fts_bench.swift`、`/private/tmp/sqlite_insert_bench.swift` |

---

## Task 1: ManifestStore 追加日志 + 回放 + 压实

**Files:**
- Modify: `Sources/BaseAgentCore/RAG/RAGEngine.swift:62-137`（`ManifestStore`）
- Modify: `Sources/BaseAgentCore/RAG/RAGEngine.swift:1074`（同步结束压实）
- Create: `Tests/BaseAgentCoreTests/ManifestStoreTests.swift`

**设计:** `manifest.json` 保留为压实后的基线；每次 `merge` 只向 `manifest.json.journal` 追加一行 JSON 记录（`updated` + `removed`）。`load()` 先读基线，再按行回放；尾部半行解析失败直接忽略。压实时在一个 `flock` 内重写基线并删除 journal，写基线先于删 journal，保证中途读取只会重复回放（幂等）。journal 超过 512KB 时在 `merge` 内自动压实。

- [ ] **Step 1: 写失败测试**

```swift
import XCTest
@testable import BaseAgentCore

final class ManifestStoreTests: XCTestCase {
    private var directory: URL!
    private var manifestPath: String { directory.appendingPathComponent("manifest.json").path }
    private var journalPath: String { manifestPath + ".journal" }

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("manifest-store-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func entry(_ digest: String,
                       complete: Bool = true) -> RAGManifestEntry {
        RAGManifestEntry(mtime: 1, size: 10, fingerprint: "fp", digest: digest,
                         docName: "doc.md", mdPath: "/tmp/doc.md",
                         ingestSignature: "sig", chunkCount: 3, complete: complete,
                         duplicateOf: nil, discarded: nil)
    }

    func testMergeAppendsJournalAndLoadReplays() throws {
        let store = ManifestStore(path: manifestPath)
        store.merge(updated: ["a": entry("da")], removed: [])
        store.merge(updated: ["b": entry("db")], removed: [])
        let loaded = store.load()
        XCTAssertEqual(loaded["a"]?.digest, "da")
        XCTAssertEqual(loaded["b"]?.digest, "db")
        XCTAssertFalse(FileManager.default.fileExists(atPath: manifestPath),
                       "增量写入不应每次都重写完整 JSON")
        XCTAssertTrue(FileManager.default.fileExists(atPath: journalPath))
    }

    func testRemovedEntriesReplayFromJournal() throws {
        let store = ManifestStore(path: manifestPath)
        store.merge(updated: ["a": entry("da")], removed: [])
        store.merge(updated: [:], removed: ["a"])
        XCTAssertTrue(store.load().isEmpty)
    }

    func testTruncatedJournalTailIsIgnored() throws {
        let store = ManifestStore(path: manifestPath)
        store.merge(updated: ["a": entry("da")], removed: [])
        let handle = try FileHandle(forWritingTo: URL(fileURLWithPath: journalPath))
        try handle.seekToEnd()
        try handle.write(contentsOf: Data(#"{"updated":"#.utf8))
        try handle.close()
        XCTAssertEqual(store.load()["a"]?.digest, "da")
    }

    func testCompactWritesBaseAndRemovesJournal() throws {
        let store = ManifestStore(path: manifestPath)
        store.merge(updated: ["a": entry("da")], removed: [])
        store.merge(updated: ["b": entry("db")], removed: ["a"])
        store.compact()
        XCTAssertFalse(FileManager.default.fileExists(atPath: journalPath))
        let data = try Data(contentsOf: URL(fileURLWithPath: manifestPath))
        let manifest = try JSONDecoder().decode(RAGManifest.self, from: data)
        XCTAssertEqual(manifest.files["b"]?.digest, "db")
        XCTAssertNil(manifest.files["a"])
        XCTAssertEqual(store.load()["b"]?.digest, "db")
    }

    func testClearRemovesBaseAndJournal() throws {
        let store = ManifestStore(path: manifestPath)
        store.merge(updated: ["a": entry("da")], removed: [])
        store.compact()
        store.merge(updated: ["b": entry("db")], removed: [])
        store.clear()
        XCTAssertFalse(FileManager.default.fileExists(atPath: manifestPath))
        XCTAssertFalse(FileManager.default.fileExists(atPath: journalPath))
        XCTAssertTrue(store.load().isEmpty)
    }
}
```

- [ ] **Step 2: 运行并确认 RED**

Run: `swift test --filter ManifestStoreTests`
Expected: `ManifestStoreTests` 编译通过但 `testCompactWritesBaseAndRemovesJournal` 等因 `compact()` 不存在而失败；至少 `testMergeAppendsJournalAndLoadReplays` 应因未生成 journal 失败。

- [ ] **Step 3: 最小实现**

```swift
private struct RAGManifestJournalRecord: Codable {
    var updated: [String: RAGManifestEntry] = [:]
    var removed: [String] = []
}

final class ManifestStore: @unchecked Sendable {
    let path: String
    private var journalPath: String { path + ".journal" }
    private static let compactThresholdBytes = 512 * 1024

    func load() -> [String: RAGManifestEntry] {
        lock.lock(); defer { lock.unlock() }
        return loadUnlocked()
    }

    private func loadUnlocked() -> [String: RAGManifestEntry] {
        var files = loadBaseUnlocked()
        for record in journalRecordsUnlocked() {
            for key in record.removed { files.removeValue(forKey: key) }
            for (key, value) in record.updated { files[key] = value }
        }
        return files
    }

    private func loadBaseUnlocked() -> [String: RAGManifestEntry] {
        guard let data = FileManager.default.contents(atPath: path),
              let manifest = try? JSONDecoder().decode(RAGManifest.self, from: data) else {
            return [:]
        }
        return manifest.files
    }

    private func journalRecordsUnlocked() -> [RAGManifestJournalRecord] {
        guard let data = FileManager.default.contents(atPath: journalPath) else { return [] }
        let decoder = JSONDecoder()
        return data.split(separator: 0x0A).compactMap { line in
            guard let lineData = Data(line) as Data?,
                  let record = try? decoder.decode(RAGManifestJournalRecord.self, from: lineData)
            else { return nil }
            return record
        }
    }

    func merge(updated: [String: RAGManifestEntry], removed: Set<String>) {
        guard !updated.isEmpty || !removed.isEmpty else { return }
        lock.lock(); defer { lock.unlock() }
        withFileLock {
            appendJournalLocked(updated: updated, removed: removed)
            if journalSizeUnlocked() > Self.compactThresholdBytes {
                compactLocked()
            }
        }
    }

    private func appendJournalLocked(updated: [String: RAGManifestEntry],
                                     removed: Set<String>) {
        let record = RAGManifestJournalRecord(updated: updated,
                                              removed: Array(removed).sorted())
        guard let data = try? JSONEncoder().encode(record) else { return }
        // 前置换行：上次崩溃留下的半行会被当成独立坏行跳过，不污染下一条记录。
        var payload = Data("\n".utf8)
        payload.append(data)
        let fd = open(journalPath, O_CREAT | O_WRONLY | O_APPEND, 0o644)
        guard fd >= 0 else {
            // 极端情况下退回完整重写，保证不丢更新。
            var files = loadUnlocked()
            for key in removed { files.removeValue(forKey: key) }
            for (key, value) in updated { files[key] = value }
            writeBaseLocked(files)
            return
        }
        payload.withUnsafeBytes { raw in
            _ = Darwin.write(fd, raw.baseAddress, raw.count)
        }
        close(fd)
    }

    private func journalSizeUnlocked() -> Int {
        (try? FileManager.default.attributesOfItem(atPath: journalPath)[.size] as? NSNumber)
            .flatMap { $0.intValue } ?? 0
    }

    func compact() {
        lock.lock(); defer { lock.unlock() }
        withFileLock { compactLocked() }
    }

    private func compactLocked() {
        guard FileManager.default.fileExists(atPath: journalPath) else {
            if !FileManager.default.fileExists(atPath: path) { return }
            return
        }
        writeBaseLocked(loadUnlocked())
        try? FileManager.default.removeItem(atPath: journalPath)
    }

    private func writeBaseLocked(_ files: [String: RAGManifestEntry]) {
        let manifest = RAGManifest(files: files)
        if let data = try? JSONEncoder().encode(manifest) {
            try? AtomicFile.write(data, to: path)
        }
    }

    func clear() {
        lock.lock(); defer { lock.unlock() }
        try? FileManager.default.removeItem(atPath: path)
        try? FileManager.default.removeItem(atPath: journalPath)
    }
}
```

- [ ] **Step 4: 运行确认 GREEN**

Run: `swift test --filter ManifestStoreTests`
Expected: 5 tests passed。

- [ ] **Step 5: 同步结束压实**

在 `RAGEngine.swift` 的 worker 结束后（`manifestStore.merge(updated: [:], removed: removedKeys)` 之后）调用：

```swift
manifestStore.compact()
```

Run: `swift test --filter 'ManifestStoreTests|RAGEngineTests'`
Expected: all passed。

- [ ] **Step 6: 提交**

```bash
git add Sources/BaseAgentCore/RAG/RAGEngine.swift Tests/BaseAgentCoreTests/ManifestStoreTests.swift
git commit -m "perf(rag): journal manifest writes"
```

---

## Task 2: Embedding 窗口化，跨文件交替获得模型

**Files:**
- Modify: `Sources/BaseAgentCore/RAG/AdaptiveEmbeddingScheduler.swift:170-222`（`Options`）
- Modify: `Sources/BaseAgentCore/RAG/AdaptiveEmbeddingScheduler.swift:333-401`（`embedDocuments`）
- Modify: `Tests/BaseAgentCoreTests/EmbeddingSchedulerTests.swift`

**设计:** 新增 `Options.windowTexts`（默认 32，最小 1）。`embedDocuments` 将输入切成窗口，每个窗口单独 `semaphore.withPermit`；窗口之间 `await Task.yield()`。这样公平信号量会把模型时间让给其他文件，输出仍是原始顺序，token 预算与自适应批大小逻辑不变。

- [ ] **Step 1: 写失败测试**

```swift
private final class OrderedRecordingEmbedding: EmbeddingProvider, @unchecked Sendable {
    let identifier = "ordered"
    let dimension = 1
    private let lock = NSLock()
    private(set) var texts: [String] = []
    private let delay: UInt64

    init(delayNanoseconds: UInt64) { self.delay = delayNanoseconds }

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        lock.lock(); self.texts.append(contentsOf: texts); lock.unlock()
        try? await Task.sleep(nanoseconds: delay)
        return texts.map { _ in [0] }
    }

    func embedQuery(_ text: String) async throws -> [Float] { [0] }
}

func testSchedulerInterleavesFilesAcrossWindows() async throws {
    let base = OrderedRecordingEmbedding(delayNanoseconds: 30_000_000)
    let scheduler = AdaptiveEmbeddingScheduler(
        base: base,
        options: .init(minBatchSize: 1, maxBatchSize: 1,
                       initialBatchSize: 1, step: 1,
                       probeBatches: 16, probeSeconds: 30,
                       windowTexts: 1),
        log: { _ in })
    async let first = scheduler.embedDocuments(["a0", "a1", "a2"])
    async let second = scheduler.embedDocuments(["b0", "b1", "b2"])
    _ = try await (first, second)

    let prefix = Array(base.texts.prefix(4))
    XCTAssertTrue(prefix.contains { $0.hasPrefix("a") })
    XCTAssertTrue(prefix.contains { $0.hasPrefix("b") })
    var run = 1
    var maxRun = 1
    for i in 1..<prefix.count {
        run = prefix[i].hasPrefix(prefix[i - 1].prefix(1)) ? run + 1 : 1
        maxRun = max(maxRun, run)
    }
    XCTAssertLessThanOrEqual(maxRun, 2, "单个文件不应连续占用全部前几个窗口：\(prefix)")
}
```

- [ ] **Step 2: 运行并确认 RED**

Run: `swift test --filter EmbeddingSchedulerTests`
Expected: 编译失败（`windowTexts` / 窗口化尚未实现）。

- [ ] **Step 3: 最小实现**

`Options` 增加：

```swift
public var windowTexts: Int
// init 参数追加 windowTexts: Int = 32
self.windowTexts = max(1, windowTexts)
```

`embedDocuments` 改为窗口循环：

```swift
public func embedDocuments(_ texts: [String],
                           progress: ((Int, Int) -> Void)?) async throws -> [[Float]] {
    guard !texts.isEmpty else { return [] }
    let window = max(1, options.windowTexts)
    guard texts.count > window else {
        return try await embedWindow(texts, progress: progress, total: texts.count)
    }

    var result: [[Float]] = []
    result.reserveCapacity(texts.count)
    var processed = 0
    var start = 0
    while start < texts.count {
        if Task.isCancelled { throw AgentError.cancelled }
        let end = min(start + window, texts.count)
        let slice = Array(texts[start..<end])
        let completed = processed
        let vectors = try await embedWindow(
            slice,
            progress: { done, _ in progress?(completed + done, texts.count) },
            total: slice.count)
        result.append(contentsOf: vectors)
        processed += slice.count
        start = end
        await Task.yield()
    }
    return result
}

private func embedWindow(_ texts: [String],
                         progress: ((Int, Int) -> Void)?,
                         total: Int) async throws -> [[Float]] {
    try await semaphore.withPermit {
        // 现有实现原样搬入；progress 回调使用传入的 total。
        // （原有 token 统计、排序、批处理、record() 保持不变）
    }
}
```

注意：`embedWindow` 内部的 `progress?(processed, total)` 需替换原 `texts.count`。

- [ ] **Step 4: 运行确认 GREEN**

Run: `swift test --filter EmbeddingSchedulerTests`
Expected: 全部通过，`testSchedulerSerializesConcurrentCalls` 仍为 `maxConcurrent == 1`。

- [ ] **Step 5: 提交**

```bash
git add Sources/BaseAgentCore/RAG/AdaptiveEmbeddingScheduler.swift Tests/BaseAgentCoreTests/EmbeddingSchedulerTests.swift
git commit -m "perf(rag): window embedding calls across files"
```

---

## Task 3: 扫描版 EPUB 跨章并发 OCR（归档读取串行）

**Files:**
- Modify: `Sources/BaseAgentCore/RAG/Parsers/EPUBParser.swift:10-57`（新增串行归档 reader）
- Modify: `Sources/BaseAgentCore/RAG/Parsers/EPUBParser.swift:275-361`（两阶段：先解析章节，再有界并发 OCR）
- Modify: `Sources/BaseAgentCore/RAG/Parsers/EPUBParser.swift:737-835`（`recognizeImages` 改用 reader）
- Test: `Tests/BaseAgentCoreTests/ParserTests.swift`

**设计:** 先串行读取/解析全部 spine 章节，判定哪些章节要走 OCR，并按图片数量预分配页号（`recognizeImages` 的 `pages` 恒等于图片路径数）。随后用 `withThrowingTaskGroup` 对有界并发（最多 3 章）执行 OCR；ZIPFoundation 的 `Archive` 仍通过 `NSLock` reader 串行访问，图片解码与归一化在锁外并行。完成后按章节原始顺序组装，页标记与 metadata 不变。

- [ ] **Step 1: 写失败测试与测试替身**

```swift
/// 记录跨章节 OCR 的最大并发；每页延迟 30ms 让其他章节有机会进入。
private final class ConcurrencyTrackingVLM: VLMService, @unchecked Sendable {
    let identifier = "concurrency-tracking"
    let isAvailable = true
    private let lock = NSLock()
    private var active = 0
    private var _calls = 0
    private var _maxConcurrent = 0

    var calls: Int { lock.lock(); defer { lock.unlock() }; return _calls }
    var maxConcurrent: Int { lock.lock(); defer { lock.unlock() }; return _maxConcurrent }

    func recognize(image: CGImage, prompt: String, maxTokens: Int) async throws -> String {
        lock.lock()
        _calls += 1
        let n = _calls
        active += 1
        _maxConcurrent = max(_maxConcurrent, active)
        lock.unlock()
        try? await Task.sleep(nanoseconds: 30_000_000)
        lock.lock(); active -= 1; lock.unlock()
        return "第 \(n) 页 OCR 正文\n\n这是用于并发测试的模拟识别内容，字符量足够通过清洗阈值。"
    }
}

func testScannedEPUBOCRunsChaptersConcurrently() throws {
    let url = directory.appendingPathComponent("multi-scanned.epub")
    let image = try makePNGData(width: 320, height: 480)
    try makeMultiChapterScannedEPUB(at: url, chapterCount: 3, imageData: image)

    let fake = ConcurrencyTrackingVLM()
    let parser = EPUBParser(vlm: fake)
    let parsed = try awaitSync {
        try await parser.parse(url: url, options: ParserOptions(epubOCREngine: .ovis), progress: nil)
    }
    XCTAssertEqual(fake.calls, 3)
    XCTAssertGreaterThanOrEqual(fake.maxConcurrent, 2, "章节 OCR 必须跨章并发")
    XCTAssertEqual(parsed.metadata["ocr_pages"], "3")
    XCTAssertEqual(parsed.metadata["scanned_chapters"], "3")
    for page in 1...3 {
        XCTAssertTrue(parsed.text.contains("第 \(page) 页"), "页标记必须按 spine 顺序保留：\(parsed.text)")
    }
}
```

EPUB 构造 helper（3 个 spine 条目，各 1 张图、无文字层）：

```swift
private func makeMultiChapterScannedEPUB(at url: URL, chapterCount: Int,
                                         imageData: Data) throws {
    guard let archive = try Archive(url: url, accessMode: .create) else {
        throw AgentError.parse("无法创建 EPUB 归档")
    }
    var manifestItems: [String] = []
    var spineRefs: [String] = []
    var files: [(String, String)] = [
        ("mimetype", "application/epub+zip"),
        ("META-INF/container.xml", """
        <?xml version="1.0"?>
        <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
          <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>
        </container>
        """),
    ]
    for i in 0..<chapterCount {
        manifestItems.append("<item id=\"c\(i)\" href=\"chapter\(i).html\" media-type=\"application/xhtml+xml\"/>")
        spineRefs.append("<itemref idref=\"c\(i)\"/>")
        files.append(("OEBPS/chapter\(i).html", """
        <html xmlns="http://www.w3.org/1999/xhtml"><body><p><img src="images/page\(i).png"/></p></body></html>
        """))
    }
    files.append(("OEBPS/content.opf", """
    <?xml version="1.0"?>
    <package xmlns="http://www.idpf.org/2007/opf" version="2.0" unique-identifier="id">
      <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
        <dc:title>Multi Scanned</dc:title><dc:identifier id="id">multi</dc:identifier>
      </metadata>
      <manifest>\(manifestItems.joined(separator: "\n"))</manifest>
      <spine>\(spineRefs.joined(separator: "\n"))</spine>
    </package>
    """))
    for (path, content) in files {
        let data = Data(content.utf8)
        try archive.addEntry(with: path, type: .file, uncompressedSize: Int64(data.count),
                             compressionMethod: path == "mimetype" ? .none : .deflate,
                             provider: { position, size in
            data.subdata(in: Int(position)..<Int(position) + size)
        })
    }
    for i in 0..<chapterCount {
        try archive.addEntry(with: "OEBPS/images/page\(i).png", type: .file,
                             uncompressedSize: Int64(imageData.count),
                             compressionMethod: .deflate,
                             provider: { position, size in
            imageData.subdata(in: Int(position)..<Int(position) + size)
        })
    }
}
```

- [ ] **Step 2: 运行并确认 RED**

Run: `swift test --filter ParserTests/testScannedEPUBOCRunsChaptersConcurrently`
Expected: FAIL，`maxConcurrent == 1`；`fake.calls` 与 metadata 可能已经正确。

- [ ] **Step 3: 实现串行 reader 与两阶段 OCR**

```swift
/// ZIPFoundation 的 Archive 句柄不能跨线程并发读；OCR 章节并发时用它串行化。
final class SerializedArchiveReader: @unchecked Sendable {
    private let archive: Archive
    private let lock = NSLock()
    init(_ archive: Archive) { self.archive = archive }

    func data(path: String) -> Data? {
        lock.lock(); defer { lock.unlock() }
        return RAGArchiveReader.data(archive, path: path)
    }
}
```

`parse` 中把章节循环拆成：

```swift
struct ChapterPlan {
    let index: Int
    let markdown: String
    let documentHeadings: [OutlineEntry]
    let imagePaths: [String]
    let isScanned: Bool
}

var plans: [ChapterPlan] = []
var nextPageNumber = 1
var ocrPages = 0
for (index, path) in spinePaths.enumerated() { ... }   // 现有读取/解析/判定逻辑
// isScanned 时：ocrPages += imagePaths.count; nextPageNumber += imagePaths.count
// 不再在循环里 await OCR
```

随后：

```swift
let reader = SerializedArchiveReader(archive)
let scannedPlans = plans.filter(\.isScanned)
var ocrResults: [Int: (text: String, pages: Int, recognized: Int)] = [:]
try await withThrowingTaskGroup(of: (Int, (text: String, pages: Int, recognized: Int)).self) { group in
    var pending = scannedPlans.makeIterator()
    var running = 0
    let limit = min(3, scannedPlans.count)
    while running < limit, let plan = pending.next() {
        running += 1
        group.addTask {
            let result = try await recognizeImages(
                plan.imagePaths, reader: reader, options: options,
                startingPage: plan.startPage)
            return (plan.index, result)
        }
    }
    while let (index, result) = try await group.next() {
        ocrResults[index] = result
        if let plan = pending.next() {
            group.addTask { ... }
        }
    }
}
```

按 `plans` 原始顺序组装 `chapters` / `headings`：OCR 结果非空则覆盖 `chapter`，否则保留文字层并记录原 warning；标题与大纲逻辑保持原样。

- [ ] **Step 4: 运行确认 GREEN**

Run: `swift test --filter ParserTests`
Expected: 所有 ParserTests 通过，含原有 3 个扫描 EPUB 测试。

- [ ] **Step 5: 提交**

```bash
git add Sources/BaseAgentCore/RAG/Parsers/EPUBParser.swift Tests/BaseAgentCoreTests/ParserTests.swift
git commit -m "perf(epub): OCR chapters concurrently"
```

---

## Task 4: VectorStore 只读快照，让 FTS/hydrate 不阻塞写入

**Files:**
- Modify: `Sources/BaseAgentCore/RAG/Storage/SQLiteDatabase.swift:96-119`（只读连接）
- Modify: `Sources/BaseAgentCore/RAG/Storage/SQLiteDatabase.swift:361-397`（读事务）
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorStore.swift:16-124`（快照连接与 deadline）
- Modify: `Sources/BaseAgentCore/RAG/Storage/VectorStore.swift:1076-1293`（FTS/hydrate）
- Modify: `Sources/BaseAgentCore/RAG/RAGEngine.swift:653-672, 2020-2068`（检索入口不再包裹写锁）
- Test: `Tests/BaseAgentCoreTests/VectorStoreReadSnapshotTests.swift`

**设计:** VectorStore 增加惰性只读连接与 `BEGIN DEFERRED` 读事务。`searchFTS` 的 FTS 候选与 hydrate 在同一个读快照内完成，不持 `store.lock`。`hybridSearch` 在 `store.lock` 内完成向量候选，同时开启读快照，随后释放 `store.lock`，FTS + hydrate + RRF 在快照内运行；删除/写入在快照外可继续提交，快照内的 rowid 不会失效。`SQLiteDatabase` 的只读连接跳过 `journal_mode=WAL` 写入型 PRAGMA。

- [ ] **Step 1: 写失败测试**

```swift
import XCTest
@testable import BaseAgentCore

final class VectorStoreReadSnapshotTests: XCTestCase {
    private var directory: URL!
    private var store: VectorStore!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("vector-snapshot-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        store = try VectorStore(dbPath: directory.appendingPathComponent("index.db").path,
                                dimension: 4)
    }

    override func tearDownWithError() throws {
        store?.close()
        try? FileManager.default.removeItem(at: directory)
    }

    private func chunks(_ count: Int, source: String = "a.md") -> [Chunk] {
        (0..<count).map { i in
            Chunk(id: "\(source)-\(i)", source: source, chunkIndex: i,
                  text: "云计算 容器 \(i)", heading: "")
        }
    }

    /// 读快照打开期间，写入必须能在另一个线程完成（WAL 读写并发）。
    func testReadSnapshotDoesNotBlockWrites() throws {
        try store.add(chunks: chunks(4), vectors: Array(repeating: [1, 0, 0, 0], count: 4))
        let writeDone = expectation(description: "write")
        let snapshotReady = expectation(description: "snapshot")

        DispatchQueue.global().async { [store] in
            try? store!.withReadSnapshot { db in
                _ = try db.query("SELECT COUNT(*) FROM chunks")
                snapshotReady.fulfill()
                // 快照保持打开，等待写入完成。
                _ = writeDone.wait(timeout: 5)
            }
        }
        wait(for: [snapshotReady], timeout: 2)
        DispatchQueue.global().async { [store] in
            _ = try? store!.add(chunks: self.chunks(2, source: "b.md"),
                                vectors: Array(repeating: [1, 0, 0, 0], count: 2))
            writeDone.fulfill()
        }
        wait(for: [writeDone], timeout: 2)
        XCTAssertEqual(try store.withReadSnapshot { db in
            try db.query("SELECT COUNT(*) FROM chunks")[0]["COUNT(*)"]?.intValue
        }, 6)
    }

    /// 快照一旦建立，之后提交的写入在快照内不可见（保证 FTS 候选与 hydrate 一致）。
    func testReadSnapshotKeepsConsistentView() throws {
        try store.add(chunks: chunks(3), vectors: Array(repeating: [1, 0, 0, 0], count: 3))
        try store.withReadSnapshot { db in
            XCTAssertEqual(try db.query("SELECT COUNT(*) FROM chunks")[0]["COUNT(*)"]?.intValue, 3)
            try self.store.add(chunks: self.chunks(2, source: "c.md"),
                               vectors: [[1, 0, 0, 0], [1, 0, 0, 0]])
            XCTAssertEqual(try db.query("SELECT COUNT(*) FROM chunks")[0]["COUNT(*)"]?.intValue, 3,
                           "同一读事务必须继续看到快照视图")
        }
    }

    /// FTS 检索结果在写入并发下仍完整、可 hydrate。
    func testFTSSearchRejectsHalfDeletedResults() throws {
        try store.add(chunks: chunks(5), vectors: Array(repeating: [1, 0, 0, 0], count: 5))
        XCTAssertFalse(try store.searchFTS(query: "云计算", topK: 5).isEmpty)
        try store.deleteBySource("a.md")
        XCTAssertTrue(try store.searchFTS(query: "云计算", topK: 5).isEmpty)
    }
}
```

- [ ] **Step 2: 运行并确认 RED**

Run: `swift test --filter VectorStoreReadSnapshotTests`
Expected: 编译失败（`withReadSnapshot` 不存在）。

- [ ] **Step 3: 实现读连接与读事务**

`SQLiteDatabase`：

```swift
let flags = readOnly ? SQLITE_OPEN_READONLY : (SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE)
...
sqlite3_busy_timeout(db, 5_000)
if !readOnly {
    try execute("PRAGMA journal_mode=WAL")
}
try execute("PRAGMA synchronous=NORMAL")
...

/// 只读事务（BEGIN DEFERRED）。读连接上只执行 SELECT。
public func readTransaction<T>(_ body: () throws -> T) throws -> T {
    lock.lock(); defer { lock.unlock() }
    guard let handle else { throw AgentError.storage("数据库已关闭") }
    try execute("BEGIN DEFERRED")
    do {
        let value = try body()
        try execute("COMMIT")
        return value
    } catch {
        try? execute("ROLLBACK")
        throw error
    }
}
```

`VectorStore`：

```swift
private let dbPath: String
private let readLock = NSLock()
private var readDB: SQLiteDatabase?

private func readDatabase() throws -> SQLiteDatabase {
    readLock.lock(); defer { readLock.unlock() }
    if let readDB { return readDB }
    let reader = try SQLiteDatabase(path: dbPath, readOnly: true)
    readDB = reader
    return reader
}

func withReadSnapshot<T>(deadline: Date? = nil,
                         _ body: (SQLiteDatabase) throws -> T) throws -> T {
    let reader = try readDatabase()
    func run() throws -> T {
        try reader.readTransaction {
            // 先读 chunks 一次，确保 WAL 快照在返回给闭包前已经建立。
            _ = try reader.query("SELECT 1 FROM \(Self.chunksTable) LIMIT 1")
            return try body(reader)
        }
    }
    guard let deadline else { return try run() }
    let seconds = deadline.timeIntervalSinceNow
    guard seconds > 0 else { throw AgentError.timeout("查询超出时限被跳过") }
    return try reader.withDeadline(seconds: seconds, run)
}

public func close() {
    lock.lock()
    readLock.lock()
    readDB?.close()
    readDB = nil
    readLock.unlock()
    lock.unlock()
    db.close()
}
```

FTS / hydrate 改为接受 db 参数（保留 `ftsCandidatesLocked` 作为主连接包装）：

```swift
private func ftsCandidates(_ db: SQLiteDatabase, query: String, limit: Int,
                           sourceFilter: String?) throws -> [(Int64, Double)]
private func loadChunks(_ db: SQLiteDatabase, rowIDs: [Int64]) throws -> [Int64: HydratedChunk]
private func hydrateRaw(_ db: SQLiteDatabase, ranked: [(Int64, Double)], ...)
```

`searchFTS` 内部改为：

```swift
public func searchFTS(query: String, topK: Int, sourceFilter: String? = nil,
                      sourceContains: String? = nil,
                      deadline: Date? = nil) throws -> [SearchHit] {
    guard topK > 0 else { return [] }
    return try withReadSnapshot(deadline: deadline) { reader in
        let fetchLimit = sourceContains == nil ? topK : topK * 20
        let ranked = try ftsCandidates(reader, query: query, limit: fetchLimit,
                                       sourceFilter: sourceFilter)
        guard !ranked.isEmpty else { return [] }
        let hydrated = try hydrateRaw(reader, ranked: ranked, sourceFilter: sourceFilter,
                                      sourceContains: sourceContains, limit: topK)
        return hydrated.map { ... }
    }
}
```

`hybridSearch` 新增 `deadline: Date? = nil`：

```swift
lock.lock()
var unlocked = false
defer { if !unlocked { lock.unlock() } }
// 1) 锁内查询向量候选（ANN/vec0/精确），使用主连接期限。
// 2) 锁内开启读快照并强制建立快照，然后解锁。
let reader = try readDatabase()
let seconds = remaining...
return try reader.withDeadline(seconds: seconds) {
    try reader.readTransaction {
        _ = try reader.query("SELECT 1 FROM chunks LIMIT 1")
        lock.unlock(); unlocked = true
        let ftsCandidates = try self.ftsCandidates(reader, ...)
        // sourceContains 过滤、RRF、hydrateRaw(reader) 全部在快照内
    }
}
```

`RAGEngine`：

```swift
private func withSnapshotStoreBounded<T>(
    timeout: Double, _ body: (VectorStore, Date) throws -> T) throws -> T {
    guard timeout > 0 else { return try withStore { try body($0, .distantFuture) } }
    let deadline = Date().addingTimeInterval(timeout)
    guard searchLock.lock(before: deadline) else { throw logTimeout(...) }
    defer { searchLock.unlock() }
    do {
        let store = try storeBounded(by: deadline)
        return try body(store, deadline)
    } catch { throw logTimeout(error) }
}
```

`search` 改用该入口，把 `deadline` 传给 `hybridSearch`；`hasChunks` 预检查删除（`hybridSearch` 内部已检查）。`annIndexStatus` 与状态快照继续用原 `withStoreBounded`。

- [ ] **Step 4: 运行确认 GREEN**

Run: `swift test --filter 'VectorStoreReadSnapshotTests|VectorStoreTests|VectorStoreANNTests|RAGEngineTests'`
Expected: all passed。

- [ ] **Step 5: 更新锁语义文档**

`docs/软件设计文档.md:262` 与 `README.md:155` 改为：检索走只读 WAL 快照，写入/删除在快照外并发提交；FTS 候选与 hydrate 在同一读事务内完成。

- [ ] **Step 6: 提交**

```bash
git add Sources/BaseAgentCore/RAG/Storage/SQLiteDatabase.swift \
        Sources/BaseAgentCore/RAG/Storage/VectorStore.swift \
        Sources/BaseAgentCore/RAG/RAGEngine.swift \
        Tests/BaseAgentCoreTests/VectorStoreReadSnapshotTests.swift \
        docs/软件设计文档.md README.md
git commit -m "perf(rag): use WAL read snapshots for search"
```

---

## Task 5: DocumentOutline 正则缓存

**Files:**
- Create: `Sources/BaseAgentCore/RAG/Chunking/RegexCache.swift`
- Modify: `Sources/BaseAgentCore/RAG/Chunking/DocumentOutline.swift:111-145, 233-238`
- Create: `Tests/BaseAgentCoreTests/RegexCacheTests.swift`

**设计:** 新增线程安全的 `RegexCache`，按 `pattern + options` 缓存 `NSRegularExpression`。`DocumentOutline` 的标题规则与代码围栏规则都走缓存；输出行为不变。

- [ ] **Step 1: 写失败测试**

```swift
import XCTest
@testable import BaseAgentCore

final class RegexCacheTests: XCTestCase {
    func testSamePatternReturnsSameCompiledInstance() {
        let first = RegexCache.shared.regex("^# (.+)$")
        let second = RegexCache.shared.regex("^# (.+)$")
        XCTAssertNotNil(first)
        XCTAssertTrue(first === second, "同一 pattern 必须复用编译结果")
    }

    func testDifferentOptionsUseSeparateEntries() {
        let plain = RegexCache.shared.regex("abc")
        let insensitive = RegexCache.shared.regex("abc", options: [.caseInsensitive])
        XCTAssertNotNil(plain)
        XCTAssertNotNil(insensitive)
        XCTAssertFalse(plain === insensitive)
    }

    func testInvalidPatternReturnsNil() {
        XCTAssertNil(RegexCache.shared.regex("["))
    }
}
```

- [ ] **Step 2: 运行并确认 RED**

Run: `swift test --filter RegexCacheTests`
Expected: 编译失败（`RegexCache` 不存在）。

- [ ] **Step 3: 实现**

```swift
import Foundation

/// 线程安全的 NSRegularExpression 缓存：标题规则在每次 extract/classify 时复用。
final class RegexCache: @unchecked Sendable {
    static let shared = RegexCache()

    private struct Key: Hashable {
        let pattern: String
        let options: UInt
    }

    private let lock = NSLock()
    private var cache: [Key: NSRegularExpression] = [:]

    func regex(_ pattern: String,
               options: NSRegularExpression.Options = []) -> NSRegularExpression? {
        let key = Key(pattern: pattern, options: options.rawValue)
        lock.lock()
        if let cached = cache[key] {
            lock.unlock()
            return cached
        }
        lock.unlock()
        guard let regex = try? NSRegularExpression(pattern: pattern, options: options) else {
            return nil
        }
        lock.lock()
        cache[key] = regex
        lock.unlock()
        return regex
    }
}
```

`DocumentOutline.classify`：

```swift
for (pattern, options, handler) in headingPatterns {
    guard let regex = RegexCache.shared.regex(pattern, options: options),
          let match = regex.firstMatch(...), let parsed = handler(match, s) else { continue }
    return parsed
}
```

`extract` 与 `codeFenceRanges` 同样改用 `RegexCache.shared.regex`。

- [ ] **Step 4: 运行确认 GREEN**

Run: `swift test --filter 'RegexCacheTests|OutlineTests|ChunkingTests|ParserTests'`
Expected: all passed。

- [ ] **Step 5: 提交**

```bash
git add Sources/BaseAgentCore/RAG/Chunking/RegexCache.swift \
        Sources/BaseAgentCore/RAG/Chunking/DocumentOutline.swift \
        Tests/BaseAgentCoreTests/RegexCacheTests.swift
git commit -m "perf(rag): cache outline regular expressions"
```

---

## Task 6: FTS 预分词 / SQLite 插入基准复核（无收益则不代码化）

**Files:**
- 只读复核: `/private/tmp/fts_bench.swift`、`/private/tmp/sqlite_insert_bench.swift`
- Modify: `docs/superpowers/plans/2026-10-31-rag-performance-optimizations.md`（记录结论）

- [ ] **Step 1: 运行两个基准并记录结果**

Run: `swift /private/tmp/fts_bench.swift` / `swift /private/tmp/sqlite_insert_bench.swift`
Expected: 得到预分词 ms/片、索引文本膨胀比、单事务写入片/秒与查询耗时。

- [ ] **Step 2: 决策门槛**

满足以下任一条件才实施额外代码：
1. 预分词占单切片写入 > 15%，且存在无行为差异的替代实现（例如复用 token 结果、避免二次切词）；
2. 单事务插入 < 200 片/秒且瓶颈在语句准备/绑定，而不是 SQLite 本身；
3. 有可量化、可回归测试的优化。

不满足则在计划末尾记录“维持现状，原因：优化收益低于维护成本/存在语义风险”，不写生产代码。

---

## 验收与收尾

- [ ] `swift test` 全量通过（当前基线 498 tests / 33 skipped / 0 failures）。
- [ ] 更新 `README.md` 与 `docs/软件设计文档.md` 中受影响的性能说明。
- [ ] 逐项独立提交；不提交 `benchmarks/` 未跟踪目录。
- [ ] 推送与打 tag 由用户确认后执行。
