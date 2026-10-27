import Foundation
import CryptoKit
import Darwin

// MARK: - 统计 / Manifest

public struct IngestStats: Sendable {
    public var processed: Int = 0
    public var skipped: Int = 0
    public var removed: Int = 0
    /// 按内容哈希清理掉的重复副本数量。
    public var deduplicated: Int = 0
    public var chunks: Int = 0
    public var errors: [String] = []
    public var cancelled: Bool = false

    public init() {}

    public var summary: String {
        var parts = ["已处理 \(processed) 个文件", "跳过 \(skipped)", "删除 \(removed)",
                     "去重 \(deduplicated)", "写入 \(chunks) 个切片"]
        if !errors.isEmpty { parts.append("失败 \(errors.count)") }
        if cancelled { parts.append("已取消") }
        return parts.joined(separator: "，")
    }
}

/// 手动向量去重结果。
public struct VectorDedupStats: Sendable {
    public var removedChunks: Int = 0
    public var affectedSources: Int = 0
    public var emptiedSources: Int = 0
    public var remainingChunks: Int = 0

    public init() {}

    public var summary: String {
        "移除 \(removedChunks) 个重复切片，涉及 \(affectedSources) 个来源"
            + (emptiedSources > 0 ? "，清空 \(emptiedSources) 个副本来源" : "")
            + "，剩余 \(remainingChunks) 条切片"
    }
}

/// 手动向量去重前的扫描结果（不修改数据）。
public struct VectorDedupPreview: Sendable {
    public var totalChunks: Int = 0
    public var duplicateChunks: Int = 0
    public var affectedSources: Int = 0

    public init(totalChunks: Int = 0,
                duplicateChunks: Int = 0,
                affectedSources: Int = 0) {
        self.totalChunks = totalChunks
        self.duplicateChunks = duplicateChunks
        self.affectedSources = affectedSources
    }
}

struct RAGManifestEntry: Codable, Sendable {
    var mtime: Double
    var size: Int64
    var fingerprint: String
    var digest: String
    var docName: String
    var mdPath: String
    /// 生成该条目时使用的解析/切片签名。缺失表示旧版 manifest，需要重建。
    /// 每个文件单独记录，配置变化或中途取消后可以只重建未完成的文件。
    var ingestSignature: String?
    /// 完成入库的切片数。缺失表示旧版 manifest；实际库内数量不一致时强制重建。
    var chunkCount: Int?
    /// 是否完整处理。存在失败页/旧版条目时为 false，增量同步会重试。
    var complete: Bool?
    /// 内容重复副本指向的保留来源。该条目没有自己的向量，增量同步会跳过。
    var duplicateOf: String?
}

struct RAGManifest: Codable, Sendable {
    var files: [String: RAGManifestEntry] = [:]
}

/// manifest 的原子读改写，避免并发 ingest 互相覆盖。
final class ManifestStore: @unchecked Sendable {
    let path: String
    private let lock = NSRecursiveLock()

    init(path: String) { self.path = path }

    func load() -> [String: RAGManifestEntry] {
        lock.lock(); defer { lock.unlock() }
        return loadUnlocked()
    }

    private func loadUnlocked() -> [String: RAGManifestEntry] {
        guard let data = FileManager.default.contents(atPath: path),
              let manifest = try? JSONDecoder().decode(RAGManifest.self, from: data) else {
            return [:]
        }
        return manifest.files
    }

    /// 合并写入：只覆盖/删除本次触达的条目，保留其它实例写入的记录。
    /// 使用独立 `.lock` 文件的 flock 做跨进程串行化，manifest 本身仍原子替换。
    func merge(updated: [String: RAGManifestEntry], removed: Set<String>) {
        lock.lock(); defer { lock.unlock() }
        withFileLock {
            var files = loadUnlocked()
            for key in removed { files.removeValue(forKey: key) }
            for (key, value) in updated { files[key] = value }
            let manifest = RAGManifest(files: files)
            if let data = try? JSONEncoder().encode(manifest) {
                try? AtomicFile.write(data, to: path)
            }
        }
    }

    private func withFileLock<T>(_ body: () -> T) -> T {
        let lockPath = path + ".lock"
        let fd = open(lockPath, O_CREAT | O_RDWR, 0o644)
        guard fd >= 0 else { return body() }
        flock(fd, LOCK_EX)
        defer {
            flock(fd, LOCK_UN)
            close(fd)
        }
        return body()
    }

    func clear() {
        lock.lock(); defer { lock.unlock() }
        try? FileManager.default.removeItem(atPath: path)
    }
}

// MARK: - RAG Engine

// MARK: - 并发 Ingest 任务

private struct ScannedFile: Sendable {
    let path: String
    let relativePath: String
}

/// 原始文件采样指纹相同的文件组。代表文件负责解析/入库，其余副本清理向量。
private struct DuplicateFileGroup: Sendable {
    let representative: ScannedFile
    let duplicates: [ScannedFile]
}

private struct IngestJob: Sendable {
    let index: Int
    let total: Int
    let file: ScannedFile
    let label: String
}

private struct IngestOutcome: Sendable {
    var source: String
    var skipped = false
    var processed = false
    var cancelled = false
    var chunks = 0
    var error: String?
}

/// 工作队列：多个 worker 依次领取文件，避免慢文件拖住整批。
private actor IngestJobQueue {
    private let jobs: [IngestJob]
    private var nextIndex = 0

    init(_ jobs: [IngestJob]) {
        self.jobs = jobs
    }

    func next() -> IngestJob? {
        guard nextIndex < jobs.count else { return nil }
        let job = jobs[nextIndex]
        nextIndex += 1
        return job
    }
}

private actor IngestAccumulator {
    private var outcomes: [IngestOutcome] = []

    func record(_ outcome: IngestOutcome) {
        outcomes.append(outcome)
    }

    func snapshot() -> [IngestOutcome] {
        outcomes
    }
}

/// 知识库引擎：解析 → 清洗 → 切片 → embedding → SQLite 写入；
/// 检索走 FTS5 + 向量 RRF，再按配置做 rerank。
public final class RAGEngine {
    public let config: AgentConfig
    public let ragDir: String
    public let markdownDir: String
    /// 按知识库隔离的向量库路径（ragDir/index.sqlite）。
    public var indexDBPath: String {
        (ragDir as NSString).appendingPathComponent("index.sqlite")
    }
    private let manifestStore: ManifestStore
    public private(set) var knowledgeBase: String

    private let parserRegistry: DocumentParserRegistry
    private var embedding: EmbeddingProvider?
    private var reranker: RerankProvider?
    private var store: VectorStore?
    private let storeLock = NSLock()
    /// 保护「使用 store」的整段读操作；close/reload 必须等读操作结束。
    private let searchLock = NSLock()
    /// ingest/close/reload 的互斥标志。不能用 NSRecursiveLock：
    /// async 函数在 await 后可能换线程，跨线程 unlock 会让锁永久保持占用。
    private let ingestFlagLock = NSLock()
    private var ingestActive = false
    private var closePending = false
    private var cancelRequested = false
    private let cancelLock = NSLock()

    public static let supportedExtensions: Set<String> = [
        "pdf", "txt", "md", "markdown", "html", "htm", "xhtml",
        "epub", "docx", "xlsx", "pptx", "csv", "json", "rst", "tex", "log",
        // 独立图片文件：Apple Vision OCR。
        "png", "jpg", "jpeg", "tiff", "tif", "bmp", "heic", "heif", "webp",
    ]

    public init(config: AgentConfig,
                knowledgeBase: String? = nil,
                embedding: EmbeddingProvider? = nil,
                reranker: RerankProvider? = nil,
                parserRegistry: DocumentParserRegistry? = nil) {
        self.config = config
        // Vision OCR 全局闸门：rag_vision_concurrency > 0 时覆盖默认（CPU 核数）。
        // 须在首次 OCR 前设置——闸门信号量在首次访问时才惰性初始化。
        if config.ragVisionConcurrency > 0 {
            VisionPDFParser.requestedConcurrency = config.ragVisionConcurrency
        }
        let kb = knowledgeBase ?? config.resolvedKnowledgeBase
        // 统一成真实路径：macOS 的 /var 与 /private/var 是 firmlink，
        // URL.resolvingSymlinksInPath() 不会转换，必须用 realpath。
        self.knowledgeBase = Self.canonicalPath(kb)
        // 知识库目录只读（原始文件）；索引、markdown 产物、manifest 全部
        // 放在工作目录下按知识库隔离的 rag_data 子目录，不对原文件目录做任何改动。
        // 传入的 knowledgeBase 参数优先于 config（测试/多库场景）。
        self.ragDir = AgentConfig.ragIndexDirectory(forKB: self.knowledgeBase, workspace: config.workspace)
        Self.migrateLegacyIndex(base: (self.ragDir as NSString).deletingLastPathComponent,
                                target: self.ragDir,
                                workspace: config.workspace,
                                currentKB: self.knowledgeBase)
        self.markdownDir = (self.ragDir as NSString).appendingPathComponent("documents")
        self.manifestStore = ManifestStore(
            path: (self.ragDir as NSString).appendingPathComponent("manifest.json")
        )
        self.embedding = embedding ?? LocalModelFactory.shared.makeEmbedding(config: config)
        self.reranker = reranker ?? LocalModelFactory.shared.makeReranker(config: config)
        self.parserRegistry = parserRegistry ?? DocumentParserRegistry.makeDefault(config: config)
        try? FileManager.default.createDirectory(atPath: self.markdownDir,
                                                 withIntermediateDirectories: true)
    }

    /// `realpath` 规范化：解析 `/var` → `/private/var` 与其它符号链接，
    /// 保证扫描路径、相对路径和 manifest 键完全一致。
    private static func canonicalPath(_ path: String) -> String {
        let expanded = (path as NSString).expandingTildeInPath
        guard let resolved = realpath(expanded, nil) else {
            return URL(fileURLWithPath: expanded).standardizedFileURL.path
        }
        defer { free(resolved) }
        return String(cString: resolved)
    }

    // MARK: 旧版共享索引迁移

    /// 旧版布局把所有知识库目录的索引共用 `rag_data` 根目录，切换知识库目录
    /// 后同步会把对方条目当「已删除文件」清空。首次按新布局访问时把遗留文件
    /// 搬进归属知识库的专属子目录：优先读取 `index_owner.json` 记录的归属，
    /// 无记录时视为当前配置的知识库（旧版本最后同步的就是它）。
    private static func migrateLegacyIndex(base: String, target: String,
                                           workspace: String, currentKB: String) {
        let fm = FileManager.default
        let legacyNames = ["manifest.json", "manifest.json.lock", "index.sqlite",
                           "index.sqlite-wal", "index.sqlite-shm",
                           "ingest_signature.txt", "documents"]
        let root = { (name: String) in (base as NSString).appendingPathComponent(name) }
        guard legacyNames.contains(where: { fm.fileExists(atPath: root($0)) }) else { return }

        let lockPath = root(".migrate.lock")
        let fd = open(lockPath, O_CREAT | O_RDWR, 0o644)
        if fd >= 0 { flock(fd, LOCK_EX) }
        defer {
            if fd >= 0 {
                flock(fd, LOCK_UN)
                Darwin.close(fd)
                try? fm.removeItem(atPath: lockPath)
            }
        }

        // 归属判定：marker 记录的 KB 优先（覆盖上次迁移被打断的场景），
        // 无记录时视为当前配置的知识库（旧版本最后同步的就是它）。
        var ownerKB = currentKB
        if let data = try? Data(contentsOf: URL(fileURLWithPath: root("index_owner.json"))),
           let obj = (try? JSONSerialization.jsonObject(with: data)) as? [String: String],
           let recorded = obj["kb"], !recorded.isEmpty {
            ownerKB = recorded
        }
        let ownerDir = ownerKB == currentKB
            ? target
            : AgentConfig.ragIndexDirectory(forKB: ownerKB, workspace: workspace)
        try? fm.createDirectory(atPath: ownerDir, withIntermediateDirectories: true)
        // 逐文件搬移；目标已存在同名文件时不覆盖（新布局数据优先）。
        var moved = 0
        for name in legacyNames {
            let src = root(name)
            let dst = (ownerDir as NSString).appendingPathComponent(name)
            guard fm.fileExists(atPath: src), !fm.fileExists(atPath: dst) else { continue }
            if (try? fm.moveItem(atPath: src, toPath: dst)) != nil {
                moved += 1
            }
        }
        if !fm.fileExists(atPath: root("index_owner.json")),
           let data = try? JSONSerialization.data(withJSONObject: ["kb": ownerKB],
                                                  options: [.sortedKeys]) {
            try? data.write(to: URL(fileURLWithPath: root("index_owner.json")), options: .atomic)
        }
        if moved > 0 {
            AgentLog.warning("RAG: 旧版共享索引已迁移至 \(ownerDir)（归属 \(ownerKB)），共 \(moved) 项")
        }
    }

    deinit {
        // 兜底：调用方忘记 close 时也不要把 SQLite 连接留到进程结束。
        store?.close()
    }

    // MARK: Store

    private func ensureStore() throws -> VectorStore {
        storeLock.lock(); defer { storeLock.unlock() }
        if let store { return store }
        guard let embedding else {
            throw AgentError.storage("embedding 模型不可用，无法打开向量库；请检查 local_embedding_model 配置")
        }
        let newStore = try VectorStore(dbPath: self.indexDBPath,
                                       dimension: embedding.dimension,
                                       enableSQLiteVec: config.ragSQLiteVecEnabled)
        if newStore.didResetForDimensionChange {
            // 维度变化时 VectorStore 已清空 chunks/documents；缓存的 manifest
            // 若不同步失效，下一次同步会把所有文件当成「未变化」而全部跳过。
            manifestStore.clear()
            AgentLog.warning("RAG: 向量维度变化，已清空 manifest，下次同步将全量重建")
        }
        store = newStore
        return newStore
    }

    /// 在读锁内使用 store：close/reload 会等待整个 body 结束。
    private func withStore<T>(_ body: (VectorStore) throws -> T) throws -> T {
        searchLock.lock(); defer { searchLock.unlock() }
        return try body(try ensureStore())
    }

    public func close() {
        // 有同步在途时不要关闭连接：事务被打断会留下未完成状态。
        // 记下 pending，等同步收尾后自动关闭，避免旧连接长期悬挂。
        guard tryBeginIngest() else {
            storeLock.lock()
            closePending = true
            storeLock.unlock()
            cancel()
            AgentLog.warning("RAG: 同步进行中，close 推迟到同步结束")
            return
        }
        defer { releaseIngestFlag() }
        searchLock.lock(); defer { searchLock.unlock() }
        storeLock.lock(); defer { storeLock.unlock() }
        store?.close()
        store = nil
    }

    public func reload() {
        guard tryBeginIngest() else {
            AgentLog.warning("RAG: 同步进行中，跳过 reload")
            return
        }
        defer { releaseIngestFlag() }
        searchLock.lock(); defer { searchLock.unlock() }
        storeLock.lock()
        store?.close()
        store = nil
        storeLock.unlock()
    }

    /// 尝试占用 ingest 独占标志；已有 ingest/close/reload 时返回 false。
    private func tryBeginIngest() -> Bool {
        ingestFlagLock.lock(); defer { ingestFlagLock.unlock() }
        guard !ingestActive else { return false }
        ingestActive = true
        return true
    }

    private func releaseIngestFlag() {
        ingestFlagLock.lock()
        ingestActive = false
        ingestFlagLock.unlock()
    }

    /// 同步结束后的延迟关闭（由 ingest 的 defer 调用）。
    ///
    /// 不能直接调用 `close()`：它会把 `cancelRequested` 置为 true，
    /// 若此时已有新的 ingest 抢到标志，会把新同步误取消。
    private func closeIfPending() {
        storeLock.lock()
        let pending = closePending
        storeLock.unlock()
        guard pending else { return }
        // 另一个 ingest 已开始：保留 pending，由它的 defer 再次尝试。
        guard tryBeginIngest() else { return }
        defer { releaseIngestFlag() }
        storeLock.lock()
        closePending = false
        storeLock.unlock()
        searchLock.lock(); defer { searchLock.unlock() }
        storeLock.lock()
        store?.close()
        store = nil
        storeLock.unlock()
    }

    public func cancel() {
        cancelLock.lock(); defer { cancelLock.unlock() }
        cancelRequested = true
    }

    private var isCancelled: Bool {
        cancelLock.lock(); defer { cancelLock.unlock() }
        return cancelRequested
    }

    // MARK: Ingest

    /// 增量同步知识库。返回统计；同一实例并发调用会被拒绝。
    public func ingest(force: Bool = false,
                       progress: ((Int, Int, String) -> Void)? = nil,
                       log: ((String) -> Void)? = nil) async -> IngestStats {
        guard tryBeginIngest() else {
            var stats = IngestStats()
            stats.errors.append("另一次同步正在进行，已忽略本次请求")
            return stats
        }
        defer {
            releaseIngestFlag()
            closeIfPending()
        }
        cancelLock.lock(); cancelRequested = false; cancelLock.unlock()

        var stats = IngestStats()
        let logFn: (String) -> Void = { message in
            if let log { log(message) } else { AgentLog.info(message) }
        }

        guard FileManager.default.fileExists(atPath: knowledgeBase) else {
            stats.errors.append("知识库目录不存在: \(knowledgeBase)")
            return stats
        }
        guard embedding != nil else {
            stats.errors.append("embedding 模型不可用")
            return stats
        }

        // 解析引擎 / 切片参数 / embedding 模型变化会让旧切片失效。
        // 签名按文件记录在 manifest 中：配置变化时逐文件重建，中途取消后
        // 已完成的文件保持新签名，下次同步只继续处理剩余文件。
        let configSignature = makeIngestSignature()
        let signatureHash = Self.contentHash(configSignature)
        let effectiveForce = force
        // 每次同步都打印一次生效的切片参数：用户可据此核对「面板设置 →
        // 实际切片 → 检索呈现」是否一致，而不是只看面板数字。
        let effectiveChunking = Self.resolvedChunkingOptions(config: config,
                                                             embedding: embedding)
        logFn("RAG: 切片参数 \(effectiveChunking.options.chunkSizeTokens) token"
            + "（含 overlap \(effectiveChunking.options.overlapTokens)），"
            + "语义切片 \(config.ragSemanticChunking ? "开" : "关")"
            + (effectiveChunking.clampedFrom.map {
                "，面板设置的 \($0) 已按 embedding 模型输入上限收口"
            } ?? ""))

        let files = scanFiles()
        logFn("RAG: 开始同步，共 \(files.count) 个文件（force=\(effectiveForce)）")
        logFn("RAG: 初始化索引库…")
        let store: VectorStore
        do {
            store = try ensureStore()
        } catch {
            stats.errors.append(error.localizedDescription)
            logFn("RAG: 索引库初始化失败：\(error.localizedDescription)")
            return stats
        }
        let existingChunkCount = store.count
        logFn("RAG: 索引库就绪（现有 \(existingChunkCount) 切片）")

        var manifest = manifestStore.load()
        logFn("RAG: manifest 载入 \(manifest.count) 条")
        // 数据库被删除/重建但 manifest 还在时，增量逻辑会把所有文件当成
        // 「未变化」而跳过，知识库永久为空；此时必须全量重建。
        if existingChunkCount == 0, !manifest.isEmpty {
            logFn("RAG: 向量库为空但 manifest 仍存在，执行全量重建")
            manifest.removeAll()
            manifestStore.clear()
        }
        // 显式强制全量重建时先把所有条目标记为待重建：如果用户中途停止，
        // 未处理文件不会带着旧签名被下次同步误判为「已完成」。
        if effectiveForce, !manifest.isEmpty {
            let stale = manifest.mapValues { entry -> RAGManifestEntry in
                var copy = entry
                copy.ingestSignature = nil
                return copy
            }
            manifestStore.merge(updated: stale, removed: [])
            manifest = stale
        }
        let staleCount = manifest.values.filter { $0.ingestSignature != signatureHash }.count
        if !effectiveForce, staleCount > 0 {
            logFn("RAG: \(staleCount) 个文件使用旧签名（配置变化或上次未完成），将逐文件重建")
        }

        // 原始文件采样指纹去重：同一本书放在不同目录时只解析一个代表文件。
        let (uniqueFiles, duplicateGroups) = Self.groupDuplicateFiles(files, manifest: manifest)
        let total = uniqueFiles.count
        let duplicateCount = duplicateGroups.reduce(0) { $0 + $1.duplicates.count }
        if duplicateCount > 0 {
            logFn("RAG: 文件指纹去重：跳过 \(duplicateCount) 个重复副本，"
                + "实际处理 \(total)/\(files.count) 个文件")
        }

        let currentSources = Set(files.map { $0.path })
        var removedKeys = Set<String>()

        // 1) 清理已删除文件的向量与缓存。
        for (source, entry) in manifest where !currentSources.contains(source) {
            do {
                _ = try store.deleteBySource(source)
                // 文档摘要/大纲记录也要删除，否则 rag_outline 会一直列出已删文件。
                try? store.deleteDocumentByMdPath(entry.mdPath)
                try? store.deleteDocument(docName: entry.docName)
                try? FileManager.default.removeItem(atPath: entry.mdPath)
                removedKeys.insert(source)
                stats.removed += 1
                logFn("RAG: 删除已不存在文档 \(source)")
            } catch {
                stats.errors.append("删除失败 \(source): \(error.localizedDescription)")
            }
        }

        // 2) 多文件并发：最多 6 个文件同时解析（ragIngestConcurrency，
        //    上限 6）；每个文件内部
        //    「解析 → 切片 → embedding → 入库」形成流水线。
        //    embedding 由共享调度器串行进入模型，并按吞吐/内存自动调整子批。
        guard let baseEmbedding = embedding else {
            stats.errors.append("embedding 模型不可用")
            return stats
        }
        let scheduler = AdaptiveEmbeddingScheduler(
            base: baseEmbedding,
            options: .init(minBatchSize: config.ragEmbeddingMinBatch,
                           maxBatchSize: config.ragEmbeddingMaxBatch,
                           initialBatchSize: config.ragEmbeddingInitialBatch,
                           step: config.ragEmbeddingBatchStep,
                           memoryLimitMB: Double(config.ragEmbeddingMemoryLimitMB),
                           throughputTolerance: config.ragEmbeddingThroughputTolerance,
                           probeBatches: config.ragEmbeddingProbeBatches,
                           probeSeconds: config.ragEmbeddingProbeSeconds),
            log: logFn)
        let jobs = uniqueFiles.enumerated().map { index, file in
            IngestJob(index: index, total: total, file: file,
                      label: "[\(index + 1)/\(total)] \(file.relativePath)")
        }
        let queue = IngestJobQueue(jobs)
        let accumulator = IngestAccumulator()
        let concurrency = min(max(1, config.ragIngestConcurrency), 6,
                              max(1, uniqueFiles.count))
        let registry = parserRegistry
        let storeRef = store
        let manifestStoreRef = manifestStore
        // 已删除文件不再作为副本的保留来源，避免 tombstone 误判为仍完整。
        var snapshot = manifest
        for key in removedKeys { snapshot.removeValue(forKey: key) }
        let manifestSnapshot = snapshot
        let markdownDirRef = markdownDir
        let knowledgeBaseRef = knowledgeBase
        let signature = signatureHash
        let force = effectiveForce
        let configRef = config
        let cancelledCheck: () -> Bool = { [weak self] in
            guard let self else { return true }
            return self.isCancelled
        }

        await withTaskGroup(of: Void.self) { group in
            for _ in 0..<concurrency {
                group.addTask {
                    while let job = await queue.next() {
                        if Task.isCancelled || cancelledCheck() { break }
                        let outcome = await Self.process(
                            job: job,
                            force: force,
                            config: configRef,
                            registry: registry,
                            store: storeRef,
                            embedding: scheduler,
                            manifestStore: manifestStoreRef,
                            manifestSnapshot: manifestSnapshot,
                            signatureHash: signature,
                            markdownDir: markdownDirRef,
                            knowledgeBase: knowledgeBaseRef,
                            isCancelled: cancelledCheck,
                            log: logFn,
                            progress: progress
                        )
                        await accumulator.record(outcome)
                        if outcome.cancelled { break }
                    }
                }
            }
            await group.waitForAll()
        }

        if isCancelled { stats.cancelled = true }
        for outcome in await accumulator.snapshot() {
            if outcome.skipped { stats.skipped += 1 }
            if outcome.processed {
                stats.processed += 1
                stats.chunks += outcome.chunks
            }
            if outcome.cancelled { stats.cancelled = true }
            if let error = outcome.error { stats.errors.append(error) }
        }
        // 已删除文件的 manifest 条目统一收尾；成功/失败条目已由 worker 逐文件落盘。
        manifestStore.merge(updated: [:], removed: removedKeys)

        if stats.cancelled {
            logFn("RAG: 同步已取消，跳过副本向量清理由下次完整同步执行")
        } else {
            // 同步完成后按 content hash 再做一次全局去重，清掉不同目录/不同格式
            // 但正文相同的副本，避免检索命中重复切片。
            stats.deduplicated = Self.deduplicateByContentHash(
                manifestStore: manifestStore,
                duplicateGroups: duplicateGroups,
                existingSources: Set(manifest.keys),
                signatureHash: signatureHash,
                store: store,
                markdownDir: markdownDir,
                knowledgeBase: knowledgeBase,
                log: logFn)
            if stats.deduplicated > 0 {
                logFn("RAG: 副本去重完成，移除 \(stats.deduplicated) 个重复来源的向量")
            }
        }
        // 全局签名文件仅作「最近一次完整完成」的诊断记录；增量判断以每个
        // manifest 条目的 ingestSignature 为准，因此取消后仍能续跑。
        if !stats.cancelled {
            try? AtomicFile.write(Data((configSignature + "\n").utf8),
                                  to: ingestSignaturePath)
        }
        if stats.cancelled {
            logFn("RAG: 同步已停止 — \(stats.summary)")
            logFn("RAG: 已完成文件已保留，下次同步将从剩余未完成文件继续")
        } else {
            progress?(total, total, "")
            logFn("RAG: 同步完成 — \(stats.summary)")
        }
        return stats
    }

    /// 影响切片内容的配置签名。
    private func makeIngestSignature() -> String {
        let embeddingModel = config.localEmbeddingModel.isEmpty
            ? config.ragEmbeddingModel : config.localEmbeddingModel
        let fields: [(String, String)] = [
            // 渲染/清洗逻辑变化时递增，强制旧缓存文件重建。
            // 8：结构化清洗档（HTML/Office/EPUB 保真）+ HTML 本地框架接力 +
            //    Office 备注/列表/脚注/日期格式完善。
            ("pipeline_version", "8"),
            ("pdf_engine", config.ragPDFEngine.rawValue),
            ("epub_ocr_engine", config.ragEPUBOCREngine.rawValue),
            ("ocr_engine", config.ragOCREngine),
            ("ocr_model", config.ragOCRModelName),
            ("ocr_path", config.ragOCRModelPath),
            ("ocr_preprocess", String(config.ragOCRPreprocess)),
            ("ocr_binarize", String(config.ragOCRBinarize)),
            ("chunk_size", String(config.ragChunkSize)),
            ("chunk_overlap", String(config.ragChunkOverlap)),
            // 语义切片开关会改变切片边界：不进签名的话，切换后旧切片会被
            // 当作「文件未变化」继续沿用，设置等于没生效。
            ("semantic_chunking", String(config.ragSemanticChunking)),
            // embedding 后端 / 权重目录决定向量语义与可用窗口（切片收口上限），
            // 换后端后旧向量必须重算。
            ("embedding_backend", AgentConfig.normalizeEmbeddingBackend(
                config.localEmbeddingBackend)),
            ("embedding_path", config.localEmbeddingModelPath),
            ("embedding_model", embeddingModel),
        ]
        return fields.map { "\($0.0)=\($0.1)" }.joined(separator: "\n")
    }

    private var ingestSignaturePath: String {
        (ragDir as NSString).appendingPathComponent("ingest_signature.txt")
    }

    private func scanFiles() -> [ScannedFile] {
        let fm = FileManager.default
        let base = URL(fileURLWithPath: knowledgeBase)
        guard let enumerator = fm.enumerator(at: base,
                                             includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey],
                                             options: [.skipsHiddenFiles, .skipsPackageDescendants]) else {
            return []
        }
        var output: [ScannedFile] = []
        for case let url as URL in enumerator {
            let values = try? url.resourceValues(forKeys: [.isDirectoryKey, .isRegularFileKey])
            if values?.isDirectory == true {
                if url.lastPathComponent == ".rag" || url.lastPathComponent == "rag" {
                    enumerator.skipDescendants()
                }
                continue
            }
            guard values?.isRegularFile == true else { continue }
            let ext = url.pathExtension.lowercased()
            guard Self.supportedExtensions.contains(ext) else { continue }
            let path = url.path
            let relative = path.hasPrefix(knowledgeBase)
                ? String(path.dropFirst(knowledgeBase.count)).trimmingCharacters(in: CharacterSet(charactersIn: "/"))
                : url.lastPathComponent
            // 跳过 sidecar 与隐藏缓存。
            if relative.hasSuffix(".meta.json") { continue }
            output.append(ScannedFile(path: path, relativePath: relative))
        }
        output.sort { $0.relativePath.localizedStandardCompare($1.relativePath) == .orderedAscending }
        return output
    }

    // MARK: - 单文件流水线

    /// 一个文件的完整处理：解析 → 清洗 → 切片 → embedding → 入库 → manifest。
    /// 多文件由 `ingest` 的 worker 池并发调用；共享的 store/manifest/scheduler
    /// 内部都有锁或串行队列。
    private static func process(
        job: IngestJob,
        force: Bool,
        config: AgentConfig,
        registry: DocumentParserRegistry,
        store: VectorStore,
        embedding: AdaptiveEmbeddingScheduler,
        manifestStore: ManifestStore,
        manifestSnapshot: [String: RAGManifestEntry],
        signatureHash: String,
        markdownDir: String,
        knowledgeBase: String,
        isCancelled: () -> Bool,
        log: @escaping (String) -> Void,
        progress: ((Int, Int, String) -> Void)?
    ) async -> IngestOutcome {
        let file = job.file
        let source = file.path
        var outcome = IngestOutcome(source: source)

        if isCancelled() || Task.isCancelled {
            outcome.cancelled = true
            return outcome
        }

        let fileStarted = Date()
        log("RAG: \(job.label) 开始处理（\(Self.fileSizeText(source))）")
        progress?(job.index + 1, job.total, file.relativePath)
        let signature = Self.fileSignature(source)

        // 内容重复副本：文件未变化且保留来源仍完整时直接跳过，不重复解析。
        if !force, let signature, let entry = manifestSnapshot[source],
           let canonical = entry.duplicateOf, !canonical.isEmpty,
           entry.ingestSignature == signatureHash,
           abs(entry.mtime - signature.mtime) < 0.001,
           entry.size == signature.size,
           !entry.fingerprint.isEmpty,
           Self.sourceFingerprint(source, size: signature.size) == entry.fingerprint,
           let canonicalEntry = manifestSnapshot[canonical],
           canonicalEntry.complete == true,
           (canonicalEntry.chunkCount ?? 0) > 0,
           entry.digest.isEmpty || canonicalEntry.digest == entry.digest {
            outcome.skipped = true
            log("RAG: \(job.label) 跳过（重复副本，保留 \(canonicalEntry.docName)）")
            return outcome
        }

        var canSkip = false
        var redoReason: String?
        if !force, let signature, let entry = manifestSnapshot[source] {
            if entry.ingestSignature != signatureHash {
                // 配置签名变化：顶部已汇总，逐文件不再刷屏。
            } else if entry.complete != true {
                redoReason = "上次未完整完成"
            } else if entry.chunkCount == nil || (entry.chunkCount ?? 0) <= 0 {
                redoReason = "缺少切片完成标记"
            } else if abs(entry.mtime - signature.mtime) >= 0.001 {
                redoReason = "文件已变更"
            } else if entry.size != signature.size {
                redoReason = "文件大小已变更"
            } else if entry.digest.isEmpty {
                redoReason = "摘要缺失"
            } else if Self.sourceFingerprint(source, size: signature.size) != entry.fingerprint {
                redoReason = "文件内容已变更"
            } else if entry.mdPath.isEmpty
                        || !FileManager.default.fileExists(atPath: entry.mdPath) {
                redoReason = "Markdown 缓存缺失"
            } else {
                let actualChunks = store.chunkCount(source: source)
                if actualChunks != entry.chunkCount {
                    // manifest 说完成，但库里没有对应切片（被删除/上次写入中断/外部改动）。
                    redoReason = "索引切片不完整（\(actualChunks)/\(entry.chunkCount ?? 0)）"
                } else {
                    canSkip = true
                }
            }
        }
        if canSkip {
            outcome.skipped = true
            log("RAG: \(job.label) 跳过（文件未变化）")
            return outcome
        }
        if let redoReason {
            log("RAG: \(job.label) 重新处理：\(redoReason)")
        }

        do {
            let options = ParserOptions(config: config)
            guard let parser = registry.freshParser(for: URL(fileURLWithPath: source),
                                                    options: options) else {
                throw AgentError.unsupported("没有可用的解析器: \(file.relativePath)")
            }
            let parseStarted = Date()
            let parsed = try await Self.parse(file: file, parser: parser,
                                              options: options, label: job.label, log: log)
            // 解析/OCR 可能耗时很久；完成后立刻响应停止/取消。
            if isCancelled() || Task.isCancelled { throw AgentError.cancelled }

            // 标准解析器已在 finish() 中清洗；仅对自定义解析器兜底清洗一次。
            let text = parsed.metadata["cleaned"] == "1"
                ? parsed.text
                : DocumentCleaner.clean(
                    parsed.text, keepHeadersFooters: false,
                    profile: DocumentProfile.defaultProfile(
                        for: URL(fileURLWithPath: file.path)))
            guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw AgentError.parse("解析结果为空")
            }
            let parseBreakdown = Self.parseTimingText(parsed.metadata)
            let parseSeconds = Date().timeIntervalSince(parseStarted)

            let chunkStarted = Date()
            var chunkDocument = parsed
            chunkDocument.text = text
            var lastChunkLog = Date()
            // 切片参数只有一处来源：设置面板（再按 embedding 模型窗口收口）。
            // 检索结果呈现等下游环节读同一个生效值，避免「设置了长切片、
            // 某个环节仍按短切片处理」。收口情况在同步开头统一打印一次。
            var chunkOptions = Self.resolvedChunkingOptions(config: config,
                                                            embedding: embedding).options
            // 模型分词器可用时改用精确 token 计数：整块（含 overlap）不超过
            // 上下文窗口，避免尾部被 embedding 的 prefix(maxLength) 静默截断。
            if let provider = embedding as? EmbeddingTokenCounting,
               let counter = await provider.makeTokenCounter() {
                chunkOptions.tokenCounter = counter
            }
            let chunks = await TextChunker(options: chunkOptions)
                .chunk(document: chunkDocument, variant: .hybrid) { done, total in
                    // 语义切片用本地词向量算相似度，大文档仍可能耗时较久——
                    // 与向量化进度同节奏（60 秒一条）输出切片进度。
                    let now = Date()
                    guard now.timeIntervalSince(lastChunkLog) >= 60 else { return }
                    lastChunkLog = now
                    log("RAG: \(job.label) 切片中 \(done)/\(total) 段"
                        + "（已用 \(Self.elapsedText(chunkStarted))）")
                }
            guard !chunks.isEmpty else {
                throw AgentError.parse("切片为空")
            }
            let chunkSeconds = Date().timeIntervalSince(chunkStarted)
            if isCancelled() || Task.isCancelled { throw AgentError.cancelled }

            // markdown 已就绪后立即进入 embedding；共享调度器会串行调用模型，
            // 并按当前吞吐与进程内存自动调整子批大小。
            let embedStarted = Date()
            var lastEmbedLog = Date()
            let vectors = try await embedding.embedDocuments(chunks.map(\.text)) { done, count in
                let now = Date()
                guard now.timeIntervalSince(lastEmbedLog) >= 60 else { return }
                lastEmbedLog = now
                log("RAG: \(job.label) 向量化中 \(done)/\(count)"
                    + "（已用 \(Self.elapsedText(embedStarted))）")
            }
            guard vectors.count == chunks.count else {
                throw AgentError.storage("embedding 数量与切片不一致")
            }
            let embedSeconds = Date().timeIntervalSince(embedStarted)
            if isCancelled() || Task.isCancelled { throw AgentError.cancelled }

            let writeStarted = Date()
            _ = try store.add(chunks: chunks, vectors: vectors)
            let writeSeconds = Date().timeIntervalSince(writeStarted)
            if isCancelled() || Task.isCancelled { throw AgentError.cancelled }

            // 缓存 Markdown/规范化正文，供 rag_outline 与调试使用。
            let cacheStarted = Date()
            let mdPath = Self.cacheMarkdownPath(for: source,
                                                markdownDir: markdownDir,
                                                knowledgeBase: knowledgeBase)
            try AtomicFile.write(text, to: mdPath)

            let digestInfo = Self.buildDigest(text: text, parsed: parsed,
                                              docName: file.relativePath)
            try store.upsertDocument(docName: file.relativePath,
                                     digest: digestInfo.digest,
                                     chapters: digestInfo.chapters,
                                     mdPath: mdPath)
            let cacheSeconds = Date().timeIntervalSince(cacheStarted)
            // 有页面解析/OCR 失败时仍保留已成功的内容，但标记为未完整完成：
            // 下次增量同步会重试，而不是因为 manifest 命中而永久跳过。
            let failedPages = parsed.metadata["failed_pages"].flatMap { Int($0) } ?? 0

            let entry = RAGManifestEntry(
                mtime: signature?.mtime ?? 0,
                size: signature?.size ?? 0,
                fingerprint: Self.sourceFingerprint(source, size: signature?.size),
                digest: Self.contentHash(text),
                docName: file.relativePath,
                mdPath: mdPath,
                ingestSignature: signatureHash,
                chunkCount: chunks.count,
                complete: failedPages == 0
            )
            // 逐文件落盘：即使同步中途停止或进程退出，下次只重做未完成的文件。
            manifestStore.merge(updated: [source: entry], removed: [])

            outcome.processed = true
            outcome.chunks = chunks.count

            // 每个文件只输出启动和完成两条日志；完成行汇总各阶段耗时。
            let parsePart = parseBreakdown.isEmpty
                ? String(format: "%.1fs", parseSeconds)
                : String(format: "%.1fs", parseSeconds) + "（\(parseBreakdown)）"
            let pages = parsed.pageCount > 0 ? " / \(parsed.pageCount) 页" : ""
            let failedPart = failedPages > 0
                ? "，失败 \(failedPages) 页（下次同步将重试）" : ""
            log("RAG: ✓ \(job.label) 完成：\(text.count) 字符\(pages)，"
                + "\(chunks.count) 片\(failedPart)；"
                + "解析 \(parsePart)，切片 \(String(format: "%.1fs", chunkSeconds))"
                + "，embedding \(String(format: "%.1fs", embedSeconds))"
                + "（子批 \(embedding.currentBatchSize)）"
                + "，入库 \(String(format: "%.1fs", writeSeconds))"
                + "，缓存/摘要 \(String(format: "%.1fs", cacheSeconds))"
                + "，总 \(Self.elapsedText(fileStarted))")
            return outcome
        } catch AgentError.cancelled {
            // 用户主动停止：不是解析失败，不能清掉该文件已有的索引。
            outcome.cancelled = true
            return outcome
        } catch {
            outcome.error = "\(file.relativePath): \(error.localizedDescription)"
            log("RAG: ✗ \(file.relativePath): \(error.localizedDescription)")
            // 文件已变更但处理失败：清掉旧索引，避免继续命中过期内容；
            // 同时移除 manifest 条目，让下次同步重试。
            if let old = manifestSnapshot[source] {
                _ = try? store.deleteBySource(source)
                try? store.deleteDocumentByMdPath(old.mdPath)
                try? store.deleteDocument(docName: old.docName)
                try? FileManager.default.removeItem(atPath: old.mdPath)
            }
            manifestStore.merge(updated: [:], removed: [source])
            return outcome
        }
    }

    private static func parse(
        file: ScannedFile,
        parser: DocumentParser,
        options: ParserOptions,
        label: String,
        log: @escaping (String) -> Void
    ) async throws -> ParsedDocument {
        let url = URL(fileURLWithPath: file.path)
        // 解析器按页/章节回调进度；日志只在开始时、每分钟和结束时输出，
        // 避免逐页刷屏，又能判断长文档没有卡死。
        let parseStarted = Date()
        var lastLogAt = Date()
        var document = try await parser.parse(url: url, options: options) { event in
            switch event {
            case .page(let index, let total, let stage):
                let now = Date()
                guard now.timeIntervalSince(lastLogAt) >= 60 else { return }
                lastLogAt = now
                log("RAG: \(label) 解析中：第 \(index)/\(total) 页"
                    + " · \(Self.stageText(stage))"
                    + "（已用 \(Self.elapsedText(parseStarted))）")
            case .message(let message):
                // 解析器只保留关键消息（如首次加载 OCR 模型），直接输出。
                log("RAG: \(label) \(message)")
            }
        }
        document.source = file.path
        return document
    }

    // MARK: Digest

    private static func buildDigest(text: String, parsed: ParsedDocument,
                                    docName: String) -> (digest: String, chapters: String) {
        let nodes = DocumentOutline.extract(text, source: docName)
        // 正则识别与解析器自带大纲（EPUB 目录、Office 标题等）取条目多的一方：
        // 任一来源都可能严重缺失，只命中一两条时不应覆盖完整目录。
        let parsedEntries = parsed.outline.split(separator: "\n")
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
        let outline = parsedEntries.count > nodes.count
            ? parsed.outline
            : DocumentOutline.render(nodes)
        let digest = outline.isEmpty
            ? "# 目录\n\n\((docName as NSString).lastPathComponent)"
            : "# 目录\n\n\(outline)"
        let chapters = nodes.map { ["level": $0.level, "title": $0.title] }
        let data = (try? JSONSerialization.data(withJSONObject: chapters, options: [])) ?? Data("[]".utf8)
        return (digest, String(data: data, encoding: .utf8) ?? "[]")
    }

    public func buildAndStoreDigest(docName: String, markdown: String,
                                    mdPath: String = "") throws -> String {
        let nodes = DocumentOutline.extract(markdown, source: docName)
        let digest = nodes.isEmpty
            ? "# 目录\n\n\((docName as NSString).lastPathComponent)"
            : "# 目录\n\n\(DocumentOutline.render(nodes))"
        let chapters = nodes.map { ["level": $0.level, "title": $0.title] }
        let data = (try? JSONSerialization.data(withJSONObject: chapters, options: [])) ?? Data("[]".utf8)
        _ = try withStore { store in
            try store.upsertDocument(docName: docName, digest: digest,
                                     chapters: String(data: data, encoding: .utf8) ?? "[]",
                                     mdPath: mdPath)
        }
        return digest
    }

    // MARK: 向量去重

    /// 扫描向量库并返回重复切片统计，不删除任何数据。
    public func previewVectorDedupe() throws -> VectorDedupPreview {
        guard tryBeginIngest() else {
            throw AgentError.storage("另一个知识库任务正在进行，请稍后重试")
        }
        defer {
            releaseIngestFlag()
            closeIfPending()
        }
        let store = try ensureStore()
        let scan = try store.previewDuplicateTextChunks()
        return VectorDedupPreview(totalChunks: store.count,
                                  duplicateChunks: scan.duplicateRows,
                                  affectedSources: scan.removedBySource.count)
    }

    /// 手动清理向量库内重复的切片/向量，保留入库时间最早的一份。
    /// 同时更新 manifest：仍有剩余切片的来源刷新 chunkCount；被完全清空的
    /// 副本写 `duplicateOf` tombstone，避免下次同步重新解析。
    public func deduplicateVectors() throws -> VectorDedupStats {
        guard tryBeginIngest() else {
            throw AgentError.storage("另一个知识库任务正在进行，请稍后重试")
        }
        defer {
            releaseIngestFlag()
            closeIfPending()
        }

        let store = try ensureStore()
        let dedupe = try store.deduplicateTextChunks()
        var stats = VectorDedupStats()
        stats.removedChunks = dedupe.removedRows
        stats.affectedSources = dedupe.removedBySource.count
        stats.remainingChunks = store.count
        guard !dedupe.removedBySource.isEmpty else { return stats }

        let manifest = manifestStore.load()
        guard !manifest.isEmpty else {
            AgentLog.info("[vector-dedupe] 移除 \(stats.removedChunks) 个重复切片（无 manifest 可更新）")
            return stats
        }

        let signatureHash = Self.contentHash(makeIngestSignature())
        var updates: [String: RAGManifestEntry] = [:]
        var removedEntries = Set<String>()

        for source in dedupe.removedBySource.keys.sorted() {
            guard var entry = manifest[source] else { continue }
            let remaining = store.chunkCount(source: source)
            if remaining == 0 {
                Self.removeIndexedSource(source: source, entry: entry, store: store,
                                         markdownDir: markdownDir,
                                         knowledgeBase: knowledgeBase)
                stats.emptiedSources += 1
                if let canonical = dedupe.canonicalBySource[source],
                   canonical != source,
                   manifest[canonical] != nil {
                    entry.duplicateOf = canonical
                    entry.chunkCount = 0
                    entry.complete = false
                    entry.ingestSignature = signatureHash
                    entry.mdPath = ""
                    // 按切片去重时整篇摘要可能不同，清空 digest 让副本跳过校验
                    // 依赖 canonical 的完整性，而不是整篇内容完全一致。
                    entry.digest = ""
                    updates[source] = entry
                } else {
                    removedEntries.insert(source)
                }
            } else {
                entry.chunkCount = remaining
                entry.ingestSignature = signatureHash
                updates[source] = entry
            }
        }

        manifestStore.merge(updated: updates, removed: removedEntries)
        AgentLog.info("[vector-dedupe] 移除 \(stats.removedChunks) 个重复切片，"
            + "涉及 \(stats.affectedSources) 个来源，清空 \(stats.emptiedSources) 个副本来源")
        return stats
    }

    // MARK: 切片参数

    /// 实际生效的切片参数：唯一来源是设置面板的 `rag_chunk_size` /
    /// `rag_chunk_overlap`，再按 embedding 模型的输入窗口收口——超过窗口的
    /// 尾部会被模型自己的 `prefix(maxLength)` 静默截断，那才是真正的
    /// 「设了长切片、实际按短切片入库」，所以必须显式压到上限并记录。
    /// `clampedFrom` 非 nil 表示收口前用户设置的值。
    static func resolvedChunkingOptions(
        config: AgentConfig,
        embedding: EmbeddingProvider?
    ) -> (options: ChunkingOptions, clampedFrom: Int?) {
        var options = ChunkingOptions(config: config)
        let configured = options.chunkSizeTokens
        guard let provider = embedding as? EmbeddingTokenCounting else {
            return (options, nil)
        }
        let limit = max(16, provider.maxContentTokens)
        guard configured > limit else { return (options, nil) }
        options.chunkSizeTokens = limit
        // overlap 也计入模型输入预算，同步压到 10%。
        options.overlapTokens = min(options.overlapTokens, limit / 10)
        return (options, configured)
    }

    /// 当前生效的切片 token 预算（含 overlap）。检索呈现等下游环节必须用它，
    /// 不能再用与切片大小无关的固定字符数。
    var effectiveChunkSizeTokens: Int {
        Self.resolvedChunkingOptions(config: config, embedding: embedding)
            .options.chunkSizeTokens
    }

    /// 单条检索结果的字符上限：不得小于一个切片的实际长度，否则用户把切片
    /// 调大后，检索结果仍按固定的 1500 字符截短（切了长块却只看到前半段）。
    /// 按估算器最保守的一档换算（中文 1 token/字、英文 1 token ≈ 1.43 字符），
    /// 取 1 token ≈ 1.5 字符并留 64 字符余量。
    static func resultCharLimit(configured: Int, chunkTokens: Int) -> Int {
        let estimated = Int(Double(max(0, chunkTokens)) * 1.5) + 64
        return max(200, max(configured, estimated))
    }

    // MARK: Search

    public func search(query: String, topK: Int = 3,
                       source: String? = nil) async throws -> [SearchHit] {
        guard let embedding else { throw AgentError.storage("embedding 模型不可用") }
        // topK 来自模型参数：硬性封顶，避免一次检索把 reranker 批量推到很高。
        let requested = max(1, min(topK, 20))
        // 先做耗时的 embedQuery，再取 store：避免 await 期间 close/reload
        // 把已经取到的连接关掉，导致在途检索报「数据库已关闭」。
        let queryVector = try await embedding.embedQuery(query)
        let rerankEnabled = config.ragRerankEnabled && reranker != nil
        // 开启重排序时，最终条数由设置里的 TopN 决定（候选池按它放大）；
        // 未开启时仍按调用方 top_k。两者都封顶 20。
        let resultCount = rerankEnabled
            ? max(1, min(config.ragRerankTopN, 20))
            : requested
        let candidateCount = max(requested, resultCount) * (rerankEnabled ? 4 : 1)

        let fetched: [SearchHit]? = try withStore { store in
            guard store.count > 0 else { return nil }
            return try store.hybridSearch(
                vector: queryVector, query: query, topK: candidateCount,
                sourceContains: source, rrfK: config.ragRRFK,
                overfetch: config.ragVectorOverfetch
            )
        }
        guard var hits = fetched else {
            AgentLog.info("RAG: 向量库为空，未检索")
            return []
        }

        if rerankEnabled, let reranker, !hits.isEmpty {
            do {
                let scores = try await reranker.rerank(query: query,
                                                       documents: hits.map(\.text))
                if scores.count == hits.count {
                    for index in hits.indices {
                        hits[index].score = scores[index]
                    }
                    // 重排序后丢弃阈值以下结果（默认 0.45，适配 jina ONNX 的
                    // 分数分布；bge-mlx 方案可上调至 0.7）；全部低于阈值时
                    // 回退到原排序的候选，避免模型完全拿不到上下文。
                    let threshold = config.ragRerankThreshold
                    let filtered = hits.filter { $0.score >= threshold }
                        .sorted { $0.score > $1.score }
                    hits = filtered.isEmpty
                        ? Array(hits.sorted { $0.score > $1.score }.prefix(resultCount))
                        : filtered
                }
            } catch {
                AgentLog.warning("RAG rerank 失败，使用 RRF 排序: \(error.localizedDescription)")
            }
        }
        return Array(hits.prefix(resultCount))
    }

    public func searchFormatted(query: String, topK: Int = 3, source: String? = nil,
                                maxCharsPerResult: Int? = nil,
                                maxTotalChars: Int? = nil) async throws -> String {
        let hits = try await search(query: query, topK: topK, source: source)
        guard !hits.isEmpty else { return "（知识库中未找到相关内容）" }
        let perResultLimit = max(200, maxCharsPerResult ?? Self.resultCharLimit(
            configured: config.ragSearchMaxCharsPerResult,
            chunkTokens: effectiveChunkSizeTokens))
        // 总上限随切片放大：配置值是下限，长切片时按「本次返回条数的完整切片」
        // 放大，但不超过 L0 硬上限（20k 字符，超过也会被内核先截断）。
        let fitAll = min(20_000, perResultLimit * hits.count)
        let totalLimit = max(1_000,
                             maxTotalChars ?? max(config.ragSearchMaxTotalChars, fitAll))
        var lines = ["从知识库检索到 \(hits.count) 条相关内容（混合检索 + RRF\(config.ragRerankEnabled ? " + 重排序" : "")）：", ""]
        var total = lines.joined().count
        for (index, hit) in hits.enumerated() {
            let name = (hit.source as NSString).lastPathComponent
            let heading = hit.heading.isEmpty ? "" : "\n章节: \(hit.heading)"
            // RRF 分数（≈1/61）与 rerank/余弦不可比，展示时标明口径。
            let isRRF = hit.rrfScore != nil && abs((hit.rrfScore ?? 0) - hit.score) < 1e-9
            let scoreLabel = isRRF ? "RRF" : "相关度"
            let header = "### [\(index + 1)] 来源: \(name) (\(scoreLabel): "
                + String(format: "%.4f", hit.score) + ")\(heading)"
            var text = hit.text
            if text.count > perResultLimit {
                text = String(text.prefix(perResultLimit)) + "…"
            }
            lines.append(header)
            lines.append(text)
            lines.append("")
            total += header.count + text.count + 1
            if total >= totalLimit {
                lines.append("…（结果过长，已截断）")
                break
            }
        }
        let output = lines.joined(separator: "\n")
        return await condenseIfNeeded(output)
    }

    // MARK: 辅助模型精简

    /// 可选的辅助模型精简器（由 Agent 注入，动态读取当前 auxLLM）。
    /// 返回 nil 表示不精简（未配置辅助模型、精简失败或结果为空）。
    public var auxCondenser: ((String) async -> String?)?

    /// 结果总字数落在 [min, max] 区间时交给辅助模型精简：
    /// - 低于下限：内容不多，总结只会丢细节、平白增加延迟，原样返回；
    /// - 高于上限：内容过大，一次性摘要不可靠，原样返回。
    /// 精简失败或输出为空时回退原文，绝不让检索结果变空。
    private func condenseIfNeeded(_ output: String) async -> String {
        let count = output.count
        guard count >= config.ragAuxSummarizeMinChars,
              count <= config.ragAuxSummarizeMaxChars,
              let condenser = auxCondenser else { return output }
        if let condensed = await condenser(output), !condensed.isEmpty {
            return condensed
        }
        return output
    }

    // MARK: Meta

    public func getDocumentOutline(docName: String) -> DocumentRecord? {
        try? withStore { $0.getDocumentDigest(docName: docName) }
    }

    public func listDocuments() -> [String] {
        (try? withStore { $0.listDocuments() }) ?? []
    }

    /// 知识库状态。`sourceLimit` 只返回前 N 个来源名，避免 7 万本书时
    /// 每 30 秒把全部路径读进内存；来源总数由 `source_count` 给出。
    public func status(sourceLimit: Int = 256) -> [String: JSONValue] {
        let snapshot = try? withStore { store -> [String: JSONValue] in
            [
                "knowledge_base": .string(knowledgeBase),
                "rag_dir": .string(ragDir),
                "chunks_stored": .int(store.count),
                "source_count": .int(store.sourceCount()),
                "sources": .array(store.listSources(limit: sourceLimit).map { .string($0) }),
                "vector_backend": .string(store.vectorEnabled ? "sqlite-vec" : "sqlite-exact"),
            ]
        }
        return snapshot ?? [
            "knowledge_base": .string(knowledgeBase),
            "rag_dir": .string(ragDir),
            "chunks_stored": .int(0),
            "source_count": .int(0),
            "sources": .array([]),
            "error": .string("向量库不可用"),
        ]
    }

    public func clear() throws {
        // 与同步串行：否则同步结束时会把 manifest 重新写回来。
        guard tryBeginIngest() else {
            throw AgentError.storage("同步进行中，无法清空知识库")
        }
        defer { releaseIngestFlag() }
        try withStore { store in
            try store.clear()
        }
        try? FileManager.default.removeItem(atPath: markdownDir)
        try? FileManager.default.createDirectory(atPath: markdownDir, withIntermediateDirectories: true)
        manifestStore.clear()
    }

    // MARK: Cache paths

    private static func cacheMarkdownPath(for source: String,
                                          markdownDir: String,
                                          knowledgeBase: String) -> String {
        var relative = source
        if source.hasPrefix(knowledgeBase) {
            relative = String(source.dropFirst(knowledgeBase.count))
                .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        }
        return (markdownDir as NSString).appendingPathComponent(Self.safeFilename(relative) + ".md")
    }

    static func safeFilename(_ path: String) -> String {
        let h = Insecure.MD5.hash(data: Data(path.utf8)).prefix(6)
            .map { String(format: "%02x", $0) }.joined()
        let base = (path as NSString).lastPathComponent
        let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: "._-"))
        let prefix = String(base.unicodeScalars.filter { allowed.contains($0) }.prefix(32))
        let safe = prefix.isEmpty ? "document" : prefix
        return "\(safe)_\(h)"
    }

    // MARK: Fingerprints

    // MARK: 重复文件 / 副本清理

    /// 用「文件大小 + 采样指纹」把同一本书的不同目录副本分组。
    /// 每组只保留一个代表文件参与解析；优先已有 manifest 的完整条目。
    private static func groupDuplicateFiles(
        _ files: [ScannedFile],
        manifest: [String: RAGManifestEntry]
    ) -> (unique: [ScannedFile], groups: [DuplicateFileGroup]) {
        struct FingerprintKey: Hashable {
            let size: Int64
            let fingerprint: String
        }

        var grouped: [FingerprintKey: [ScannedFile]] = [:]
        var unique: [ScannedFile] = []
        // 先按文件大小分组：只有大小相同的文件才可能是副本，绝大多数文件
        // 不必读取头部/尾部采样，增量同步时能省掉大量磁盘 IO。
        var sameSizeGroups: [Int64: [ScannedFile]] = [:]

        for file in files {
            guard let signature = fileSignature(file.path) else {
                unique.append(file)
                continue
            }
            sameSizeGroups[signature.size, default: []].append(file)
        }

        for (size, sameSize) in sameSizeGroups {
            if sameSize.count == 1 {
                unique.append(sameSize[0])
                continue
            }
            for file in sameSize {
                // 始终重新采样：manifest 里的指纹可能属于旧内容，直接复用会把
                // 两个不同文件误判为副本。
                let fingerprint = sourceFingerprint(file.path, size: size)
                guard !fingerprint.isEmpty else {
                    unique.append(file)
                    continue
                }
                grouped[FingerprintKey(size: size, fingerprint: fingerprint),
                        default: []].append(file)
            }
        }

        var groups: [DuplicateFileGroup] = []
        for group in grouped.values {
            guard group.count > 1 else {
                if let only = group.first { unique.append(only) }
                continue
            }
            let representative = group.min {
                duplicateRank($0, manifest) < duplicateRank($1, manifest)
            }!
            unique.append(representative)
            groups.append(DuplicateFileGroup(
                representative: representative,
                duplicates: group.filter { $0.path != representative.path }))
        }

        unique.sort {
            $0.relativePath.localizedStandardCompare($1.relativePath) == .orderedAscending
        }
        groups.sort {
            $0.representative.relativePath.localizedStandardCompare(
                $1.representative.relativePath) == .orderedAscending
        }
        return (unique, groups)
    }

    /// 代表文件优先：完整完成 > 已有 manifest > 路径字典序。
    private static func duplicateRank(
        _ file: ScannedFile,
        _ manifest: [String: RAGManifestEntry]
    ) -> (Int, Int, String) {
        let entry = manifest[file.path]
        return (entry?.complete == true ? 0 : 1,
                entry == nil ? 1 : 0,
                file.relativePath)
    }

    /// 同步完成后清理重复来源：
    /// - 原始指纹相同且代表文件完整成功时，清掉其它副本；
    /// - 正文 content hash 相同的来源只保留一个，删除其余向量/摘要缓存。
    private static func deduplicateByContentHash(
        manifestStore: ManifestStore,
        duplicateGroups: [DuplicateFileGroup],
        existingSources: Set<String>,
        signatureHash: String,
        store: VectorStore,
        markdownDir: String,
        knowledgeBase: String,
        log: @escaping (String) -> Void
    ) -> Int {
        let finalManifest = manifestStore.load()
        var removeSet = Set<String>()
        var tombstoneCanonical: [String: String] = [:]

        // 原始指纹相同的副本：代表文件完整成功时，清掉其它副本已入库的旧向量。
        for group in duplicateGroups {
            guard let representative = finalManifest[group.representative.path],
                  representative.complete == true,
                  (representative.chunkCount ?? 0) > 0 else { continue }
            for file in group.duplicates
            where finalManifest[file.path] != nil || store.chunkCount(source: file.path) > 0 {
                removeSet.insert(file.path)
            }
        }

        // 正文 content hash 相同的来源只保留一个；其余来源写 tombstone，
        // 下次增量同步直接跳过，不再重复解析。
        var byDigest: [String: [(source: String, entry: RAGManifestEntry)]] = [:]
        for (source, entry) in finalManifest
        where !entry.digest.isEmpty && entry.duplicateOf == nil && !removeSet.contains(source) {
            byDigest[entry.digest, default: []].append((source, entry))
        }
        for group in byDigest.values where group.count > 1 {
            let ordered = group.sorted { lhs, rhs in
                let leftComplete = lhs.entry.complete == true
                let rightComplete = rhs.entry.complete == true
                if leftComplete != rightComplete { return leftComplete }
                let leftExisting = existingSources.contains(lhs.source)
                let rightExisting = existingSources.contains(rhs.source)
                if leftExisting != rightExisting { return leftExisting }
                return lhs.source.localizedStandardCompare(rhs.source) == .orderedAscending
            }
            let canonical = ordered[0].source
            for item in ordered.dropFirst() {
                tombstoneCanonical[item.source] = canonical
            }
        }

        guard !removeSet.isEmpty || !tombstoneCanonical.isEmpty else { return 0 }

        for source in removeSet.sorted() {
            let entry = finalManifest[source]
            removeIndexedSource(source: source, entry: entry, store: store,
                                markdownDir: markdownDir, knowledgeBase: knowledgeBase)
            let name = entry?.docName
                ?? relativeSourcePath(source, knowledgeBase: knowledgeBase)
            log("RAG: 去重移除副本 \(name)")
        }

        var tombstones: [String: RAGManifestEntry] = [:]
        for (source, canonical) in tombstoneCanonical {
            guard let entry = finalManifest[source] else { continue }
            removeIndexedSource(source: source, entry: entry, store: store,
                                markdownDir: markdownDir, knowledgeBase: knowledgeBase)
            var marker = entry
            marker.duplicateOf = canonical
            marker.chunkCount = 0
            marker.complete = false
            marker.ingestSignature = signatureHash
            marker.mdPath = ""
            tombstones[source] = marker
            let keepName = finalManifest[canonical]?.docName ?? canonical
            log("RAG: 去重移除副本 \(entry.docName)，保留 \(keepName)")
        }
        manifestStore.merge(updated: tombstones, removed: removeSet)
        return removeSet.count + tombstones.count
    }

    private static func removeIndexedSource(
        source: String,
        entry: RAGManifestEntry?,
        store: VectorStore,
        markdownDir: String,
        knowledgeBase: String
    ) {
        _ = try? store.deleteBySource(source)
        let mdPath = entry?.mdPath
            ?? cacheMarkdownPath(for: source, markdownDir: markdownDir,
                                knowledgeBase: knowledgeBase)
        let docName = entry?.docName
            ?? relativeSourcePath(source, knowledgeBase: knowledgeBase)
        try? store.deleteDocumentByMdPath(mdPath)
        try? store.deleteDocument(docName: docName)
        if !mdPath.isEmpty {
            try? FileManager.default.removeItem(atPath: mdPath)
        }
    }

    private static func relativeSourcePath(_ source: String,
                                           knowledgeBase: String) -> String {
        source.hasPrefix(knowledgeBase)
            ? String(source.dropFirst(knowledgeBase.count))
                .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
            : (source as NSString).lastPathComponent
    }

    struct FileSignature {
        var mtime: Double
        var size: Int64
    }

    static func fileSignature(_ path: String) -> FileSignature? {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
              let size = attrs[.size] as? NSNumber else { return nil }
        let mtime = (attrs[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
        return FileSignature(mtime: mtime, size: size.int64Value)
    }

    static func contentHash(_ text: String) -> String {
        let digest = SHA256.hash(data: Data(text.utf8))
        return digest.prefix(8).map { String(format: "%02x", $0) }.joined()
    }

    static func fileSizeText(_ path: String) -> String {
        guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
              let size = attrs[.size] as? NSNumber else { return "大小未知" }
        return ByteCountFormatter.string(fromByteCount: size.int64Value, countStyle: .file)
    }

    static func elapsedText(_ start: Date) -> String {
        String(format: "%.1fs", Date().timeIntervalSince(start))
    }

    static func stageText(_ stage: String) -> String {
        switch stage {
        case "text": return "文本层"
        case "render": return "页面渲染"
        case "ocr": return "OCR"
        default: return stage
        }
    }

    /// 解析器写入的分阶段耗时：文本层 / 渲染 / Vision OCR / Ovis OCR。
    static func parseTimingText(_ metadata: [String: String]) -> String {
        let fields: [(key: String, name: String)] = [
            ("timing_text", "文本层"),
            ("timing_render", "渲染"),
            ("timing_vision", "Vision"),
            ("timing_ocr", "OCR"),
        ]
        var parts: [String] = []
        for field in fields {
            guard let raw = metadata[field.key],
                  let value = Double(raw), value >= 0.05 else { continue }
            parts.append("\(field.name) \(String(format: "%.1fs", value))")
        }
        return parts.joined(separator: " / ")
    }

    /// 原始文件指纹：≤4MB 全量；大文件取头/中/尾各 64KB。
    static func sourceFingerprint(_ path: String, size: Int64? = nil) -> String {
        let fm = FileManager.default
        let total: Int64
        if let size {
            total = size
        } else if let attrs = try? fm.attributesOfItem(atPath: path),
                  let number = attrs[.size] as? NSNumber {
            total = number.int64Value
        } else {
            return ""
        }
        guard let handle = FileHandle(forReadingAtPath: path) else { return "" }
        defer { try? handle.close() }
        var hasher = SHA256()
        hasher.update(data: Data(String(total).utf8))
        let sample = 64 * 1024
        do {
            if total <= 4 * 1024 * 1024 {
                while let data = try handle.read(upToCount: 1 << 20), !data.isEmpty {
                    hasher.update(data: data)
                }
            } else {
                hasher.update(data: try handle.read(upToCount: sample) ?? Data())
                try handle.seek(toOffset: UInt64(max(0, total / 2 - Int64(sample / 2))))
                hasher.update(data: try handle.read(upToCount: sample) ?? Data())
                try handle.seek(toOffset: UInt64(max(0, total - Int64(sample))))
                hasher.update(data: try handle.read(upToCount: sample) ?? Data())
            }
        } catch {
            return ""
        }
        return hasher.finalize().prefix(8).map { String(format: "%02x", $0) }.joined()
    }
}
