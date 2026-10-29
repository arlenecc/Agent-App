import XCTest
@testable import BaseAgentCore

/// 记录每次调用并可按调用序号失败的辅助模型桩。
private final class SummaryStubLLM: LLMService, @unchecked Sendable {
    let identifier = "summary-stub"
    let contextWindow: Int
    private let lock = NSLock()
    private var prompts: [String] = []
    private var callCountStorage = 0
    private let failOnCall: Int?
    private let response: String

    init(contextWindow: Int = 32768, failOnCall: Int? = nil,
         response: String = String(repeating: "章节内容概括。", count: 8)) {
        self.contextWindow = contextWindow
        self.failOnCall = failOnCall
        self.response = response
    }

    var calls: Int {
        lock.lock()
        defer { lock.unlock() }
        return callCountStorage
    }

    var recordedPrompts: [String] {
        lock.lock()
        defer { lock.unlock() }
        return prompts
    }

    func chatStream(messages: [ChatMessage], tools: [JSONValue],
                    temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        lock.lock()
        callCountStorage += 1
        let current = callCountStorage
        prompts.append(messages.last?.content ?? "")
        let shouldFail = failOnCall == current
        lock.unlock()

        return AsyncThrowingStream { continuation in
            if shouldFail {
                continuation.finish(throwing: AgentError.cancelled)
                return
            }
            continuation.yield(LLMStreamEvent(kind: .content, content: self.response))
            continuation.yield(LLMStreamEvent(kind: .done))
            continuation.finish()
        }
    }
}

final class BookSummaryEngineTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-summary-engine-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func makeStore() throws -> VectorStore {
        try VectorStore(dbPath: directory.appendingPathComponent("index.sqlite").path,
                        dimension: 2, enableSQLiteVec: false)
    }

    @discardableResult
    private func insertDocument(_ store: VectorStore, name: String = "book.md",
                                contentHash: String = "content-hash") throws -> DocumentRecord {
        let mdPath = directory.appendingPathComponent(name).path
        try store.upsertDocument(docName: name, digest: "# 目录\n\n" + name,
                                 chapters: "[]", mdPath: mdPath,
                                 contentHash: contentHash)
        return try XCTUnwrap(store.document(docID: VectorStore.hashSource(mdPath)))
    }

    func testTokenEstimateTreatsCJKAndLatinDifferently() throws {
        XCTAssertEqual(BookSummaryEngine.estimateTokens(""), 0)
        XCTAssertEqual(BookSummaryEngine.estimateTokens("中文摘要"), 4)
        XCTAssertEqual(BookSummaryEngine.estimateTokens("abcd"), 1)
        XCTAssertEqual(BookSummaryEngine.estimateTokens("ab中文"), 3)
    }

    func testSegmentSplitterRespectsBudgetAndKeepsOrder() throws {
        let first = String(repeating: "甲", count: 40)
        let second = String(repeating: "乙", count: 40)
        let third = String(repeating: "丙", count: 40)
        let body = [first, second, third].joined(separator: "\n\n")

        let segments = BookSummaryEngine.splitSegments(body, tokenBudget: 100)
        XCTAssertEqual(segments, [first + "\n\n" + second, third])
        XCTAssertEqual(segments.joined(separator: "\n\n"), body)
        for segment in segments {
            XCTAssertLessThanOrEqual(BookSummaryEngine.estimateTokens(segment), 100)
        }

        let longParagraph = String(repeating: "甲", count: 250)
        let pieces = BookSummaryEngine.splitSegments(longParagraph, tokenBudget: 100)
        XCTAssertEqual(pieces.count, 3)
        XCTAssertEqual(pieces.joined(), longParagraph)
    }

    func testChapterSummaryReusesPersistedSegmentsAfterInterruption() async throws {
        let store = try makeStore()
        let document = try insertDocument(store)
        let engine = BookSummaryEngine(store: store)
        let paragraphs = ["甲", "乙", "丙"].map {
            String(repeating: $0, count: 600)
        }
        let markdown = "# 第一章 概述\n\n" + paragraphs.joined(separator: "\n\n")

        let specs = BookSummaryEngine.splitSegments(
            paragraphs.joined(separator: "\n\n"), tokenBudget: 1024)
        XCTAssertEqual(specs.count, 3)

        let interrupted = SummaryStubLLM(contextWindow: 1600, failOnCall: 2)
        do {
            _ = try await engine.processNextChapter(
                document: document, markdown: markdown, llm: interrupted,
                now: Date(), isCancelled: { false })
            XCTFail("第二次分段调用应被中断")
        } catch {
            XCTAssertEqual(interrupted.calls, 2)
        }
        XCTAssertEqual(try store.chapterSegments(docID: document.docID, sectionIndex: 0).count, 1)

        let resumed = SummaryStubLLM(contextWindow: 1600)
        let result = try await engine.processNextChapter(
            document: document, markdown: markdown, llm: resumed,
            now: Date(), isCancelled: { false })
        // 已完成的第 1 段不再调用；补 2 段 map + 1 次 reduce。
        XCTAssertEqual(resumed.calls, 3)
        XCTAssertTrue(result.chapterCompleted)
        XCTAssertEqual(try store.chapterSegments(docID: document.docID, sectionIndex: 0).count, 3)
        XCTAssertFalse(result.summary.isEmpty)
    }

    func testChapterSummaryNormalizesToFiftyToOneHundredCharacters() throws {
        let raw = "**摘要：**\n\n" + String(repeating: "本章介绍研究方法与实验设计。", count: 20)
        let normalized = BookSummaryEngine.normalizeChapterSummary(raw)
        XCTAssertFalse(normalized.contains("**"))
        XCTAssertFalse(normalized.hasPrefix("摘要"))
        XCTAssertGreaterThanOrEqual(normalized.count, 50)
        XCTAssertLessThanOrEqual(normalized.count, 100)

        XCTAssertEqual(BookSummaryEngine.normalizeChapterSummary("  \"简短内容。\"  "),
                       "简短内容。")
    }

    func testTinySectionSkipsModelCall() async throws {
        let store = try makeStore()
        let document = try insertDocument(store)
        let engine = BookSummaryEngine(store: store)
        let markdown = "# 第一章 概述\n\n很短的正文。"
        let stub = SummaryStubLLM(failOnCall: 1, response: "不应被使用")

        let result = try await engine.processNextChapter(
            document: document, markdown: markdown, llm: stub,
            now: Date(), isCancelled: { false })
        XCTAssertEqual(stub.calls, 0)
        XCTAssertFalse(result.summary.isEmpty)
        XCTAssertTrue(result.chapterCompleted)

        let saved = try XCTUnwrap(store.chapterSummary(docID: document.docID, sectionIndex: 0))
        XCTAssertEqual(saved.summary, result.summary)
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.status, .summarizingBook)
    }

    func testBookSummaryMapReducePersistsGroupsAndProducesLongSummary() async throws {
        let store = try makeStore()
        let document = try insertDocument(store)
        let engine = BookSummaryEngine(store: store)
        let sections = 30
        let markdown = (0..<sections).map { index in
            "# 第\(index + 1)节\n\n" + String(repeating: "正文内容。", count: 20)
        }.joined(separator: "\n\n")

        let resolved = engine.resolve(document, markdown: markdown)
        XCTAssertEqual(resolved.sections.count, sections)
        for section in resolved.sections {
            try store.upsertChapterSummary(ChapterSummaryRecord(
                docID: document.docID, sectionIndex: section.index,
                heading: section.heading, title: section.title,
                level: section.level,
                summary: String(repeating: "章节摘要内容。", count: 10),
                contentHash: section.contentHash))
        }

        let stub = SummaryStubLLM(
            contextWindow: 1600,
            response: String(repeating: "全书内容概括。", count: 50))
        let record = try await engine.processBookSummary(
            document: document, markdown: markdown, llm: stub,
            now: Date(), isCancelled: { false })

        let groups = try store.bookSummarySegments(docID: document.docID)
        XCTAssertGreaterThanOrEqual(groups.count, 2)
        XCTAssertGreaterThanOrEqual(record.summary.count, 300)
        XCTAssertLessThanOrEqual(record.summary.count, 500)
        XCTAssertFalse(record.outline.isEmpty)
        XCTAssertEqual(record.chapterCount, sections)
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.status, .completed)
        XCTAssertGreaterThanOrEqual(stub.calls, groups.count + 1)
    }
}
