import Foundation

/// 闲时摘要协调器：
/// - 常驻辅助模型：15 分钟一个章节；临时辅助模型：5 分钟一个章节；
/// - 一个 tick 最多处理一个章节或一次全书摘要；
/// - 成功后才写 `last_tick_at`，失败不推进 `next_section`；
/// - 临时模型由会话创建/卸载，常驻模型只借用、不关闭。
public final class BookSummaryCoordinator: @unchecked Sendable {

    private let engine: BookSummaryEngine
    private let store: VectorStore
    private let configProvider: @Sendable () -> AgentConfig
    private let residentAuxProvider: @Sendable () -> LLMService?
    private let transientAuxFactory: (@Sendable (AgentConfig) -> LLMService)?
    private let now: @Sendable () -> Date
    private let log: @Sendable (String) -> Void

    private let lock = NSLock()
    private var sessionMode: BookSummaryMode?
    private var transientLLM: LLMService?
    private var tickTask: Task<BookSummaryTickOutcome, Never>?
    private var cancelRequested = false
    /// 会话级停止：`cancelCurrent` 之后不再开始新的 tick，直到下个会话。
    private var stopRequested = false
    private var retryCounts: [String: Int] = [:]
    private var currentDocName: String?
    private var currentSection = 0
    private var currentTotal = 0
    /// 同一文档每个空闲会话最多重试次数；超过后让位给其他书。
    private let maxRetriesPerDocument = 2

    public init(engine: BookSummaryEngine,
                store: VectorStore,
                configProvider: @escaping @Sendable () -> AgentConfig,
                residentAuxProvider: @escaping @Sendable () -> LLMService?,
                transientAuxFactory: (@Sendable (AgentConfig) -> LLMService)? = nil,
                now: @escaping @Sendable () -> Date = { Date() },
                log: @escaping @Sendable (String) -> Void = { _ in }) {
        self.engine = engine
        self.store = store
        self.configProvider = configProvider
        self.residentAuxProvider = residentAuxProvider
        self.transientAuxFactory = transientAuxFactory
        self.now = now
        self.log = log
    }

    // MARK: - 会话

    /// 开始空闲会话；临时模式会创建并预加载辅助模型。
    public func beginSession(mode: BookSummaryMode) async -> Bool {
        lock.lock()
        if let existing = sessionMode {
            lock.unlock()
            return existing == mode
        }
        lock.unlock()

        if mode == .resident {
            guard residentAuxProvider() != nil else {
                log("章节摘要：常驻辅助模型不可用，跳过本次空闲会话")
                return false
            }
            lock.lock()
            sessionMode = .resident
            retryCounts.removeAll()
            cancelRequested = false
            stopRequested = false
            lock.unlock()
            log("章节摘要：开始常驻辅助模型空闲会话（900 秒/章）")
            return true
        }

        guard let factory = transientAuxFactory else {
            log("章节摘要：没有临时辅助模型工厂，跳过本次空闲会话")
            return false
        }
        let llm = factory(configProvider())
        if let preloadable = llm as? PreloadableLLMService {
            do {
                try await preloadable.preload()
            } catch {
                (llm as? ClosableLLMService)?.close()
                log("章节摘要：临时辅助模型加载失败：\(error.localizedDescription)")
                return false
            }
        }

        lock.lock()
        sessionMode = .temporary
        transientLLM = llm
        retryCounts.removeAll()
        cancelRequested = false
        stopRequested = false
        lock.unlock()
        log("章节摘要：开始临时辅助模型空闲会话（300 秒/章）")
        return true
    }

    /// 执行一次节流 tick；同一时间只允许一个 tick 在跑。
    public func runTick(mode: BookSummaryMode) async -> BookSummaryTickOutcome {
        lock.lock()
        let stopping = stopRequested
        lock.unlock()
        guard !stopping else { return .noWork }

        let llm: LLMService?
        switch mode {
        case .resident:
            llm = residentAuxProvider()
        case .temporary:
            lock.lock()
            llm = transientLLM
            lock.unlock()
        }
        guard let llm else {
            log("章节摘要：\(mode == .resident ? "常驻" : "临时")辅助模型不可用")
            return .noWork
        }

        let lastTick: Double? = (try? store.summaryLastTickAt()) ?? nil
        if let lastTick {
            let elapsed = now().timeIntervalSince1970 - lastTick
            if elapsed < mode.interval {
                return .throttled(waitSeconds: max(0, mode.interval - elapsed))
            }
        }

        lock.lock()
        if tickTask != nil {
            lock.unlock()
            return .throttled(waitSeconds: 0)
        }
        cancelRequested = false
        lock.unlock()

        let candidates = (try? store.summaryCandidates(limit: 10)) ?? []
        guard let document = candidates.first(where: {
            retryCount(docID: $0.docID) < maxRetriesPerDocument
        }) else {
            return .noWork
        }

        guard FileManager.default.fileExists(atPath: document.mdPath),
              let markdown = try? String(contentsOfFile: document.mdPath, encoding: .utf8) else {
            let message = "找不到可读的 Markdown：\(document.mdPath)"
            let attempts = incrementRetry(docID: document.docID)
            log("章节摘要：\(document.docName) \(message)（第 \(attempts)/\(maxRetriesPerDocument) 次）")
            return .failed(docName: document.docName, message: message)
        }

        let job = try? store.summaryJob(docID: document.docID)
        let isBookSlot = job?.status == .summarizingBook || job?.status == .completed
        lock.lock()
        currentDocName = document.docName
        currentSection = max(0, job?.nextSection ?? 0)
        currentTotal = job?.totalSections ?? 0
        lock.unlock()

        let task = Task { [weak self] () -> BookSummaryTickOutcome in
            guard let self else { return .noWork }
            return await self.performTick(document: document, markdown: markdown,
                                          llm: llm, isBookSlot: isBookSlot)
        }
        lock.lock()
        tickTask = task
        lock.unlock()

        let outcome = await task.value
        lock.lock()
        if tickTask == task { tickTask = nil }
        lock.unlock()
        return outcome
    }

    /// 停止当前章节调用；已完成分段保留，下个空闲会话继续。
    public func cancelCurrent() async {
        lock.lock()
        cancelRequested = true
        stopRequested = true
        let task = tickTask
        lock.unlock()
        task?.cancel()
        if let task {
            _ = await task.value
            lock.lock()
            if tickTask == task { tickTask = nil }
            lock.unlock()
        }
    }

    /// 结束会话：等待当前任务停止，卸载临时模型；不触碰常驻辅助模型。
    public func endSession() async {
        await cancelCurrent()
        lock.lock()
        let llm = transientLLM
        transientLLM = nil
        sessionMode = nil
        retryCounts.removeAll()
        currentDocName = nil
        currentSection = 0
        currentTotal = 0
        cancelRequested = false
        stopRequested = false
        lock.unlock()

        if let llm {
            if let unloadable = llm as? AsyncUnloadableLLMService {
                await unloadable.unloadAndWait()
            }
            (llm as? ClosableLLMService)?.close()
            log("章节摘要：临时辅助模型已卸载")
        }
    }

    @MainActor
    public func statusSnapshot() -> String {
        lock.lock()
        defer { lock.unlock() }
        guard sessionMode != nil else { return "章节摘要：空闲" }
        if let name = currentDocName {
            if currentTotal > 0 {
                return "章节摘要：\(name) \(min(currentSection + 1, currentTotal))/\(currentTotal)"
            }
            return "章节摘要：\(name)"
        }
        return "章节摘要：等待处理"
    }

    // MARK: - Tick 执行

    private func performTick(document: DocumentRecord, markdown: String,
                             llm: LLMService, isBookSlot: Bool) async -> BookSummaryTickOutcome {
        do {
            if isBookSlot {
                let record = try await engine.processBookSummary(
                    document: document, markdown: markdown, llm: llm,
                    now: now(), isCancelled: { [weak self] in
                        guard let self else { return true }
                        return self.isCancelled
                    })
                try? store.setSummaryLastTickAt(now().timeIntervalSince1970)
                resetRetry(docID: document.docID)
                lock.lock()
                currentSection = max(record.chapterCount - 1, 0)
                currentTotal = record.chapterCount
                lock.unlock()
                log("章节摘要：✓ \(document.docName) 全书摘要完成（\(record.summary.count) 字）")
                return .processedBook(docID: document.docID, docName: document.docName)
            }

            let result = try await engine.processNextChapter(
                document: document, markdown: markdown, llm: llm,
                now: now(), isCancelled: { [weak self] in
                    guard let self else { return true }
                    return self.isCancelled
                })
            try? store.setSummaryLastTickAt(now().timeIntervalSince1970)
            resetRetry(docID: document.docID)
            lock.lock()
            currentSection = result.sectionIndex
            currentTotal = result.totalSections
            lock.unlock()
            log("章节摘要：✓ \(document.docName) 第 \(result.sectionIndex + 1)/"
                + "\(result.totalSections) 节（\(result.summary.count) 字）")
            return .processedChapter(
                docID: document.docID, docName: document.docName,
                sectionIndex: result.sectionIndex,
                totalSections: result.totalSections,
                chapterCompleted: result.chapterCompleted)
        } catch {
            if isCancellation(error) {
                log("章节摘要：已取消 \(document.docName) 的当前处理，已完成分段保留")
                return .failed(docName: document.docName, message: "摘要任务已取消")
            }
            let attempts = incrementRetry(docID: document.docID)
            log("章节摘要：✗ \(document.docName) \(error.localizedDescription)"
                + "（第 \(attempts)/\(maxRetriesPerDocument) 次）")
            return .failed(docName: document.docName,
                           message: error.localizedDescription)
        }
    }

    // MARK: - 状态

    private var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelRequested || stopRequested
    }

    private func isCancellation(_ error: Error) -> Bool {
        if isCancelled || Task.isCancelled { return true }
        if case AgentError.cancelled = error { return true }
        return false
    }

    private func retryCount(docID: String) -> Int {
        lock.lock()
        defer { lock.unlock() }
        return retryCounts[docID] ?? 0
    }

    @discardableResult
    private func incrementRetry(docID: String) -> Int {
        lock.lock()
        defer { lock.unlock() }
        let next = (retryCounts[docID] ?? 0) + 1
        retryCounts[docID] = next
        return next
    }

    private func resetRetry(docID: String) {
        lock.lock()
        retryCounts[docID] = nil
        lock.unlock()
    }
}
