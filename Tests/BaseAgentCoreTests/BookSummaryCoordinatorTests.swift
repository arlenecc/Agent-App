import XCTest
@testable import BaseAgentCore

private final class SummaryClock: @unchecked Sendable {
    var now: Date
    init(_ now: Date) { self.now = now }
}

/// 协调器测试桩：可延迟、可失败，并记录 preload/unload/close 生命周期。
private final class CoordinatorStubLLM: LLMService, PreloadableLLMService,
    AsyncUnloadableLLMService, ClosableLLMService, @unchecked Sendable {
    let identifier = "coordinator-stub"
    let contextWindow = 32768
    private let lock = NSLock()
    private let delay: TimeInterval
    private let failFirstCalls: Int
    private let response: String
    private var callsStorage = 0
    private var preloadedStorage = false
    private var unloadedStorage = false
    private var closedStorage = false

    init(delay: TimeInterval = 0, failFirstCalls: Int = 0,
         response: String = String(repeating: "章节摘要内容。", count: 8)) {
        self.delay = delay
        self.failFirstCalls = failFirstCalls
        self.response = response
    }

    var calls: Int {
        lock.lock()
        defer { lock.unlock() }
        return callsStorage
    }

    var preloaded: Bool {
        lock.lock()
        defer { lock.unlock() }
        return preloadedStorage
    }

    var unloaded: Bool {
        lock.lock()
        defer { lock.unlock() }
        return unloadedStorage
    }

    var closed: Bool {
        lock.lock()
        defer { lock.unlock() }
        return closedStorage
    }

    func preload() async throws {
        lock.lock()
        preloadedStorage = true
        lock.unlock()
    }

    func unloadAndWait() async {
        lock.lock()
        unloadedStorage = true
        lock.unlock()
    }

    func close() {
        lock.lock()
        closedStorage = true
        lock.unlock()
    }

    func chatStream(messages: [ChatMessage], tools: [JSONValue],
                    temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        lock.lock()
        callsStorage += 1
        let call = callsStorage
        let shouldFail = call <= failFirstCalls
        let delay = self.delay
        let response = self.response
        lock.unlock()

        return AsyncThrowingStream { continuation in
            if shouldFail {
                continuation.finish(throwing: AgentError.llm("辅助模型测试失败"))
                return
            }
            if delay <= 0 {
                continuation.yield(LLMStreamEvent(kind: .content, content: response))
                continuation.yield(LLMStreamEvent(kind: .done))
                continuation.finish()
                return
            }
            Task {
                try? await Task.sleep(nanoseconds: UInt64(delay * 1_000_000_000))
                continuation.yield(LLMStreamEvent(kind: .content, content: response))
                continuation.yield(LLMStreamEvent(kind: .done))
                continuation.finish()
            }
        }
    }
}

final class BookSummaryCoordinatorTests: XCTestCase {
    private var directory: URL!
    private var clock: SummaryClock!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-summary-coordinator-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        clock = SummaryClock(Date(timeIntervalSince1970: 1_000))
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func makeStore() throws -> VectorStore {
        try VectorStore(dbPath: directory.appendingPathComponent("index.sqlite").path,
                        dimension: 2, enableSQLiteVec: false)
    }

    private func insertDocument(_ store: VectorStore, markdown: String,
                               name: String = "book.md") throws -> DocumentRecord {
        let mdPath = directory.appendingPathComponent(name).path
        try markdown.write(toFile: mdPath, atomically: true, encoding: .utf8)
        try store.upsertDocument(docName: name, digest: "# 目录\n\n\(name)",
                                 chapters: "[]", mdPath: mdPath,
                                 contentHash: "content-\(name)")
        return try XCTUnwrap(store.document(docID: VectorStore.hashSource(mdPath)))
    }

    private func makeCoordinator(store: VectorStore, resident: LLMService?,
                                 transient: (@Sendable (AgentConfig) -> LLMService)? = nil)
        -> BookSummaryCoordinator {
        let clock = self.clock!
        return BookSummaryCoordinator(
            engine: BookSummaryEngine(store: store),
            store: store,
            configProvider: { AgentConfig() },
            residentAuxProvider: { resident },
            transientAuxFactory: transient,
            now: { clock.now },
            log: { _ in })
    }

    private func chapterMarkdown(_ count: Int, bodyLength: Int = 100) -> String {
        (0..<count).map { index in
            "# 第\(index + 1)章 主题\(index + 1)\n\n"
                + String(repeating: "正文内容。", count: max(1, bodyLength / 6))
        }.joined(separator: "\n\n")
    }

    func testResidentModeProcessesOneChapterPerFifteenMinutes() async throws {
        let store = try makeStore()
        let markdown = chapterMarkdown(1)
        let document = try insertDocument(store, markdown: markdown)
        let resident = CoordinatorStubLLM()
        let coordinator = makeCoordinator(store: store, resident: resident)

        let began = await coordinator.beginSession(mode: .resident)
        XCTAssertTrue(began)
        let first = await coordinator.runTick(mode: .resident)
        guard case .processedChapter(_, _, let sectionIndex, _, let completed) = first else {
            return XCTFail("首个 tick 应处理章节，实际 \(first)")
        }
        XCTAssertEqual(sectionIndex, 0)
        XCTAssertTrue(completed)
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.status, .summarizingBook)
        XCTAssertFalse(resident.preloaded)
        XCTAssertFalse(resident.closed)

        let second = await coordinator.runTick(mode: .resident)
        guard case .throttled(let wait) = second else {
            return XCTFail("常驻模式 15 分钟内应节流，实际 \(second)")
        }
        XCTAssertEqual(wait, 900, accuracy: 0.01)
    }

    func testTemporaryModeProcessesOneChapterPerFiveMinutes() async throws {
        let store = try makeStore()
        let markdown = chapterMarkdown(1)
        _ = try insertDocument(store, markdown: markdown)
        let transient = CoordinatorStubLLM()
        let coordinator = makeCoordinator(
            store: store, resident: nil, transient: { _ in transient })

        let began = await coordinator.beginSession(mode: .temporary)
        XCTAssertTrue(began)
        XCTAssertTrue(transient.preloaded)
        let first = await coordinator.runTick(mode: .temporary)
        guard case .processedChapter = first else {
            return XCTFail("临时模式首个 tick 应处理章节，实际 \(first)")
        }

        let second = await coordinator.runTick(mode: .temporary)
        guard case .throttled(let wait) = second else {
            return XCTFail("临时模式 5 分钟内应节流，实际 \(second)")
        }
        XCTAssertEqual(wait, 300, accuracy: 0.01)

        await coordinator.endSession()
        XCTAssertTrue(transient.unloaded)
        XCTAssertTrue(transient.closed)
    }

    func testTickNeverProcessesMoreThanOneChapter() async throws {
        let store = try makeStore()
        let markdown = chapterMarkdown(3)
        let document = try insertDocument(store, markdown: markdown)
        let resident = CoordinatorStubLLM()
        let coordinator = makeCoordinator(store: store, resident: resident)

        let outcome = await coordinator.runTick(mode: .resident)
        guard case .processedChapter(_, _, let sectionIndex, let total, _) = outcome else {
            return XCTFail("应处理一个章节，实际 \(outcome)")
        }
        XCTAssertEqual(sectionIndex, 0)
        XCTAssertEqual(total, 3)
        XCTAssertEqual(try store.chapterSummaries(docID: document.docID).count, 1)
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.nextSection, 1)
        // 单章节正文一段：map + reduce 两次调用。
        XCTAssertEqual(resident.calls, 2)
    }

    func testCompletedBookMovesToBookSummarySlot() async throws {
        let store = try makeStore()
        let markdown = chapterMarkdown(3)
        let document = try insertDocument(store, markdown: markdown)
        let engine = BookSummaryEngine(store: store)
        let resolved = engine.resolve(document, markdown: markdown)
        XCTAssertEqual(resolved.sections.count, 3)
        for section in resolved.sections {
            try store.upsertChapterSummary(ChapterSummaryRecord(
                docID: document.docID, sectionIndex: section.index,
                heading: section.heading, title: section.title,
                level: section.level,
                summary: String(repeating: "章节摘要内容。", count: 10),
                contentHash: section.contentHash))
        }
        try store.upsertSummaryJob(SummaryJobRecord(
            docID: document.docID, status: .summarizingBook,
            nextSection: 3, totalSections: 3, updatedAt: 1_000))

        let resident = CoordinatorStubLLM(
            response: String(repeating: "全书内容概括。", count: 50))
        let coordinator = makeCoordinator(store: store, resident: resident)
        let outcome = await coordinator.runTick(mode: .resident)

        guard case .processedBook = outcome else {
            return XCTFail("章节完成后应生成全书摘要，实际 \(outcome)")
        }
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.status, .completed)
        XCTAssertEqual(try store.bookSummary(docID: document.docID)?.summary.count, 350)
    }

    func testFailureDoesNotAdvanceNextSection() async throws {
        let store = try makeStore()
        let markdown = chapterMarkdown(1)
        let document = try insertDocument(store, markdown: markdown)
        let resident = CoordinatorStubLLM(failFirstCalls: 2)
        let coordinator = makeCoordinator(store: store, resident: resident)

        let first = await coordinator.runTick(mode: .resident)
        guard case .failed(_, let message) = first else {
            return XCTFail("首次失败应返回 failed，实际 \(first)")
        }
        XCTAssertFalse(message.isEmpty)
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.nextSection, 0)
        XCTAssertEqual(try store.chapterSummaries(docID: document.docID).count, 0)
        XCTAssertNil(try store.summaryLastTickAt())

        let second = await coordinator.runTick(mode: .resident)
        guard case .failed = second else {
            return XCTFail("第二次失败应返回 failed，实际 \(second)")
        }
        let third = await coordinator.runTick(mode: .resident)
        XCTAssertEqual(third, .noWork)
        XCTAssertEqual(resident.calls, 2)
        XCTAssertEqual(try store.chapterSummaries(docID: document.docID).count, 0)
    }

    func testCancelStopsCurrentChapterAndKeepsSegments() async throws {
        let store = try makeStore()
        let paragraphs = ["甲", "乙", "丙"].map {
            String(repeating: $0, count: 600)
        }
        let markdown = "# 第一章 概述\n\n" + paragraphs.joined(separator: "\n\n")
        let document = try insertDocument(store, markdown: markdown)
        let transient = CoordinatorStubLLM(delay: 0.2)
        let coordinator = makeCoordinator(
            store: store, resident: nil, transient: { _ in transient })
        let began = await coordinator.beginSession(mode: .temporary)
        XCTAssertTrue(began)

        let tick = Task { await coordinator.runTick(mode: .temporary) }
        let deadline = Date().addingTimeInterval(2)
        while Date() < deadline {
            if (try? store.chapterSegments(docID: document.docID, sectionIndex: 0).count) ?? 0 >= 1 {
                break
            }
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        await coordinator.cancelCurrent()
        let outcome = await tick.value

        guard case .failed(_, let message) = outcome else {
            return XCTFail("取消应结束当前 tick，实际 \(outcome)")
        }
        XCTAssertTrue(message.contains("取消"))
        XCTAssertEqual(try store.chapterSummaries(docID: document.docID).count, 0)
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.nextSection, 0)
        let kept = try store.chapterSegments(docID: document.docID, sectionIndex: 0)
        XCTAssertGreaterThanOrEqual(kept.count, 1)

        await coordinator.endSession()
        XCTAssertTrue(transient.unloaded)
    }

    func testCancelCurrentBlocksFurtherTicksUntilNextSession() async throws {
        let store = try makeStore()
        _ = try insertDocument(store, markdown: chapterMarkdown(1))
        let resident = CoordinatorStubLLM()
        let coordinator = makeCoordinator(store: store, resident: resident)

        let began = await coordinator.beginSession(mode: .resident)
        XCTAssertTrue(began)
        await coordinator.cancelCurrent()
        let stopped = await coordinator.runTick(mode: .resident)
        XCTAssertEqual(stopped, .noWork)
        XCTAssertEqual(resident.calls, 0)

        await coordinator.endSession()
        let resumed = await coordinator.beginSession(mode: .resident)
        XCTAssertTrue(resumed)
        let outcome = await coordinator.runTick(mode: .resident)
        guard case .processedChapter = outcome else {
            return XCTFail("新会话应恢复处理，实际 \(outcome)")
        }
    }
}
