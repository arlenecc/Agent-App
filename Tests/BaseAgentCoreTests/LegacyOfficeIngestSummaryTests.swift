import AppKit
import XCTest
@testable import BaseAgentCore

private final class LegacyIngestEmbedding: EmbeddingProvider {
    let identifier = "legacy-ingest-test"
    let dimension = 8

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        texts.map { text in
            var vector = [Float](repeating: 0, count: dimension)
            for (index, scalar) in text.unicodeScalars.enumerated() {
                vector[Int(scalar.value + UInt32(index)) % dimension] += 1
            }
            return VectorCodec.normalize(vector)
        }
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        try await embedDocuments([text])[0]
    }
}

/// 摘要辅助模型桩：记录每次请求的选项，用于确认全链路都关闭 thinking。
private final class LegacySummaryLLM: LLMService, @unchecked Sendable {
    let identifier = "legacy-summary-stub"
    let contextWindow = 32_768
    let supportsConcurrentIsolatedRequests = true

    private let lock = NSLock()
    private var optionsStorage: [LLMRequestOptions] = []

    var recordedOptions: [LLMRequestOptions] {
        lock.lock()
        defer { lock.unlock() }
        return optionsStorage
    }

    var calls: Int { recordedOptions.count }

    func chatStream(messages: [ChatMessage],
                    tools: [JSONValue],
                    temperature: Double) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        chatStream(messages: messages, tools: tools, temperature: temperature,
                   options: .default)
    }

    func chatStream(messages: [ChatMessage],
                    tools: [JSONValue],
                    temperature: Double,
                    options: LLMRequestOptions) -> AsyncThrowingStream<LLMStreamEvent, Error> {
        lock.lock()
        optionsStorage.append(options)
        lock.unlock()
        let response = String(repeating: "本章介绍旧版文档的核心内容与结论。", count: 8)
        return AsyncThrowingStream { continuation in
            continuation.yield(LLMStreamEvent(kind: .content, content: response))
            continuation.yield(LLMStreamEvent(kind: .done))
            continuation.finish()
        }
    }
}

/// 真实二进制旧版 Office 文件的全链路验证：
/// 解析 → 入库（Markdown 缓存/目录/章节）→ 逐章摘要 → 全书摘要 → rag_outline 输出。
final class LegacyOfficeIngestSummaryTests: XCTestCase {
    private var root: URL!
    private var knowledgeBase: URL!
    private var config: AgentConfig!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-legacy-summary-\(UUID().uuidString)")
        knowledgeBase = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: knowledgeBase,
                                                withIntermediateDirectories: true)
        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = knowledgeBase.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 60
        config.ragChunkOverlap = 10
        self.config = config
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private func makeEngine() -> RAGEngine {
        RAGEngine(config: config, knowledgeBase: knowledgeBase.path,
                  embedding: LegacyIngestEmbedding(), reranker: nil)
    }

    // MARK: - 测试

    func testWordFixtureIngestAndSummaryEndToEnd() async throws {
        try writeWordFixture(name: "legacy-word.doc", paragraphs: [
            ("第一章 旧版 Word 概述", 18, true),
            (String(repeating: "Word 正文介绍项目背景、目标与交付范围，", count: 6) + "覆盖完整链路。",
             12, false),
            ("1.1 背景与目标", 16, true),
            (String(repeating: "本节说明旧版格式的解析策略与兼容边界，", count: 6) + "并给出验收标准。",
             12, false),
        ])

        let engine = makeEngine()
        let stats = await engine.ingest()
        XCTAssertEqual(stats.processed, 1, stats.errors.joined(separator: "; "))
        XCTAssertTrue(stats.errors.isEmpty, stats.errors.joined(separator: "; "))
        defer { engine.close() }

        let expected = ["第一章 旧版 Word 概述", "1.1 背景与目标"]
        let book = try await assertSummaryChain(
            engine: engine, docName: "legacy-word.doc", expectedTitles: expected,
            expectedMarkdownMarker: "# 第一章 旧版 Word 概述",
            expectedDigestMarker: "第一章 旧版 Word 概述")
        XCTAssertTrue(book.outline.contains("第一章"))

        try await assertOutlineToolShowsBookSummary(engine: engine,
                                                    docName: "legacy-word.doc",
                                                    summary: book.summary)
    }

    func testExcelFixtureIngestAndSummaryEndToEnd() async throws {
        let cells: [BIFF8TestBuilder.Cell] = [
            .init(row: 0, column: 0, value: .label("项目")),
            .init(row: 0, column: 1, value: .label("负责人")),
            .init(row: 0, column: 2, value: .label("进度说明")),
            .init(row: 1, column: 0, value: .label("旧版 Excel 兼容")),
            .init(row: 1, column: 1, value: .label("张三")),
            .init(row: 1, column: 2, value: .label("覆盖单元格、合并与公式解析，验证入库链路")),
            .init(row: 2, column: 0, value: .label("摘要流程")),
            .init(row: 2, column: 1, value: .label("李四")),
            .init(row: 2, column: 2, value: .label("章节与全书摘要均基于 Markdown 缓存生成")),
            .init(row: 3, column: 0, value: .label("回归测试")),
            .init(row: 3, column: 1, value: .label("王五")),
            .init(row: 3, column: 2, value: .label("解析、切片、向量化与摘要端到端协同工作")),
        ]
        let builder = BIFF8TestBuilder(sheets: [
            .init(name: "汇总", cells: cells),
            .init(name: "明细", cells: [
                .init(row: 0, column: 0, value: .label("分类")),
                .init(row: 0, column: 1, value: .number(500, xf: 0)),
                .init(row: 0, column: 2, value: .label("预算与执行明细")),
                .init(row: 1, column: 0, value: .label("市场推广")),
                .init(row: 1, column: 1, value: .number(320, xf: 0)),
                .init(row: 1, column: 2, value: .label("覆盖华东与华南区域的品牌推广活动")),
                .init(row: 2, column: 0, value: .label("研发投入")),
                .init(row: 2, column: 1, value: .number(180, xf: 0)),
                .init(row: 2, column: 2, value: .label("用于兼容性改造与回归测试环境建设")),
            ]),
        ])
        try builder.build().data.write(to: knowledgeBase
            .appendingPathComponent("legacy-sheet.xls"))

        let engine = makeEngine()
        let stats = await engine.ingest()
        XCTAssertEqual(stats.processed, 1, stats.errors.joined(separator: "; "))
        XCTAssertTrue(stats.errors.isEmpty, stats.errors.joined(separator: "; "))
        defer { engine.close() }

        let expected = ["Sheet: 汇总", "Sheet: 明细"]
        let book = try await assertSummaryChain(
            engine: engine, docName: "legacy-sheet.xls", expectedTitles: expected,
            expectedMarkdownMarker: "## Sheet: 汇总",
            expectedDigestMarker: "Sheet: 汇总")
        XCTAssertTrue(book.outline.contains("Sheet: 汇总"))

        try await assertOutlineToolShowsBookSummary(engine: engine,
                                                    docName: "legacy-sheet.xls",
                                                    summary: book.summary)
    }

    func testPowerPointFixtureIngestAndSummaryEndToEnd() async throws {
        let builder = PowerPointTestBuilder(slides: [
            .init(slideId: 1, blocks: [
                .init(headerType: 0, text: "年度计划"),
                .init(headerType: 1, text: "旧版 PowerPoint 演示文稿介绍年度目标、关键举措、"
                    + "预算分配与资源安排，确保产品、研发与运营团队按统一节奏推进。"),
            ]),
            .init(slideId: 2, blocks: [
                .init(headerType: 0, text: "执行方案"),
                .init(headerType: 1, text: "执行阶段覆盖需求确认、方案设计、开发联调、灰度验证、"
                    + "正式发布与复盘改进，每个阶段都明确负责人与验收标准。"),
            ]),
        ])
        try builder.build().data.write(to: knowledgeBase
            .appendingPathComponent("legacy-deck.ppt"))

        let engine = makeEngine()
        let stats = await engine.ingest()
        XCTAssertEqual(stats.processed, 1, stats.errors.joined(separator: "; "))
        XCTAssertTrue(stats.errors.isEmpty, stats.errors.joined(separator: "; "))
        defer { engine.close() }

        let expected = ["Slide 1：年度计划", "Slide 2：执行方案"]
        let book = try await assertSummaryChain(
            engine: engine, docName: "legacy-deck.ppt", expectedTitles: expected,
            expectedMarkdownMarker: "## Slide 1：年度计划",
            expectedDigestMarker: "Slide 1：年度计划")
        XCTAssertTrue(book.outline.contains("Slide 1：年度计划"))

        try await assertOutlineToolShowsBookSummary(engine: engine,
                                                    docName: "legacy-deck.ppt",
                                                    summary: book.summary)
    }

    /// 没有标题样式的真实 Word 文件：入库目录为空，摘要阶段合成「全文」章节，
    /// 仍能完成章节与全书摘要，而不是被直接标记完成并跳过。
    func testUntitledWordFixtureIngestsWholeTextSummary() async throws {
        try writeWordFixture(name: "legacy-untitled.doc", paragraphs: [
            ("这是一份没有标题的旧版 Word 文档。", 12, false),
            (String(repeating: "正文全部使用相同字号，没有任何标题样式，", count: 6)
                + "用来验证无标题文档的摘要链路。", 12, false),
        ])

        let engine = makeEngine()
        let stats = await engine.ingest()
        XCTAssertEqual(stats.processed, 1, stats.errors.joined(separator: "; "))
        XCTAssertTrue(stats.errors.isEmpty, stats.errors.joined(separator: "; "))
        defer { engine.close() }

        let record = try XCTUnwrap(engine.getDocumentOutline(docName: "legacy-untitled.doc"))
        let markdown = try String(contentsOfFile: record.mdPath, encoding: .utf8)
        XCTAssertFalse(markdown.split(separator: "\n").contains { $0.hasPrefix("#") },
                       "无标题文档不应有可识别标题")
        // 入库时目录仍为空；「全文」只服务于摘要，不能写回文档记录。
        XCTAssertEqual(record.chapters, "[]")
        XCTAssertFalse(record.digest.contains("全文"))

        let book = try await assertSummaryChain(
            engine: engine, docName: "legacy-untitled.doc",
            expectedTitles: ["全文"],
            expectedMarkdownMarker: "这是一份没有标题的旧版 Word 文档。",
            expectedDigestMarker: "legacy-untitled.doc")
        XCTAssertTrue(book.outline.contains("legacy-untitled.doc"))

        try await assertOutlineToolShowsBookSummary(engine: engine,
                                                    docName: "legacy-untitled.doc",
                                                    summary: book.summary)
    }

    // MARK: - 全链路辅助

    @discardableResult
    private func assertSummaryChain(
        engine: RAGEngine,
        docName: String,
        expectedTitles: [String],
        expectedMarkdownMarker: String,
        expectedDigestMarker: String
    ) async throws -> BookSummaryRecord {
        let document = try XCTUnwrap(engine.getDocumentOutline(docName: docName),
                                     "入库后应能查到 \(docName)")
        let markdown = try String(contentsOfFile: document.mdPath, encoding: .utf8)
        XCTAssertTrue(markdown.contains(expectedMarkdownMarker), markdown)
        XCTAssertTrue(document.digest.contains(expectedDigestMarker), document.digest)

        let store = try VectorStore(dbPath: engine.indexDBPath, dimension: 8,
                                    enableSQLiteVec: false)
        defer { store.close() }
        let summaryEngine = BookSummaryEngine(store: store)
        let resolved = summaryEngine.resolve(document, markdown: markdown)
        XCTAssertEqual(resolved.sections.map(\.title), expectedTitles)
        XCTAssertEqual(resolved.sections.count, expectedTitles.count)

        let llm = LegacySummaryLLM()
        for offset in expectedTitles.indices {
            let chapter = try await summaryEngine.processNextChapter(
                document: document, markdown: markdown, llm: llm,
                now: Date(), isCancelled: { false })
            XCTAssertEqual(chapter.sectionIndex, offset)
            XCTAssertEqual(chapter.totalSections, expectedTitles.count)
            // 只有最后一章处理完时才进入全书摘要阶段。
            XCTAssertEqual(chapter.chapterCompleted,
                           offset == expectedTitles.indices.last)
            XCTAssertFalse(chapter.summary.isEmpty)
        }
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.status, .summarizingBook)

        let book = try await summaryEngine.processBookSummary(
            document: document, markdown: markdown, llm: llm,
            now: Date(), isCancelled: { false })
        XCTAssertEqual(book.chapterCount, expectedTitles.count)
        XCTAssertFalse(book.summary.isEmpty)
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.status, .completed)
        XCTAssertEqual(try store.summaryJob(docID: document.docID)?.totalSections,
                       expectedTitles.count)
        XCTAssertEqual(try store.chapterSummaries(docID: document.docID).count,
                       expectedTitles.count)
        XCTAssertEqual(try store.bookSummary(docID: document.docID)?.summary, book.summary)
        XCTAssertGreaterThan(llm.calls, 0, "全链路应调用辅助模型")
        XCTAssertTrue(llm.recordedOptions.allSatisfy {
            $0.enableThinking == false && $0.isolated
        }, "摘要调用必须关闭 thinking 并使用独立会话")
        return book
    }

    private func assertOutlineToolShowsBookSummary(engine: RAGEngine,
                                                   docName: String,
                                                   summary: String) async throws {
        let tool = RagOutlineTool(config: config, engine: engine)
        let result = try await tool.run(["book_name": .string(docName)])
        XCTAssertTrue(result.success, result.output)
        XCTAssertTrue(result.output.contains("全书摘要已完成"), result.output)
        XCTAssertTrue(result.output.contains(summary), result.output)
    }

    // MARK: - fixtures

    /// 用 AppKit 导出真正的 Word 97 `.doc` 二进制文件。
    private func writeWordFixture(name: String,
                                  paragraphs: [(text: String, size: CGFloat, bold: Bool)]) throws {
        let attributed = NSMutableAttributedString()
        for (index, paragraph) in paragraphs.enumerated() {
            let font = paragraph.bold
                ? NSFont.boldSystemFont(ofSize: paragraph.size)
                : NSFont.systemFont(ofSize: paragraph.size)
            attributed.append(NSAttributedString(string: paragraph.text,
                                                 attributes: [.font: font]))
            if index + 1 < paragraphs.count {
                attributed.append(NSAttributedString(
                    string: "\n",
                    attributes: [.font: NSFont.systemFont(ofSize: 12)]))
            }
        }
        let fullRange = NSRange(location: 0, length: attributed.length)
        let data = try attributed.data(from: fullRange, documentAttributes: [
            .documentType: NSAttributedString.DocumentType.docFormat,
        ])
        try data.write(to: knowledgeBase.appendingPathComponent(name))
    }
}
