import XCTest
@testable import BaseAgentCore

private final class OutlineTestEmbedding: EmbeddingProvider {
    let identifier = "outline-test"
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

final class RagOutlineToolTests: XCTestCase {
    private var root: URL!
    private var knowledgeBase: URL!
    private var config: AgentConfig!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-outline-tool-\(UUID().uuidString)")
        knowledgeBase = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: knowledgeBase, withIntermediateDirectories: true)
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

    private func write(_ markdown: String, relativePath: String) throws {
        let url = knowledgeBase.appendingPathComponent(relativePath)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(),
                                                withIntermediateDirectories: true)
        try markdown.write(to: url, atomically: true, encoding: .utf8)
    }

    private func makeEngine() -> RAGEngine {
        RAGEngine(config: config, knowledgeBase: knowledgeBase.path,
                  embedding: OutlineTestEmbedding(), reranker: nil)
    }

    /// 直接写摘要表，模拟后台任务完成后的持久化状态。
    private func seedSummaries(engine: RAGEngine, docName: String,
                               chapterCount: Int?, withBookSummary: Bool) throws {
        guard let record = engine.getDocumentOutline(docName: docName) else {
            return XCTFail("入库后应能查到 \(docName)")
        }
        let markdown = try String(contentsOfFile: record.mdPath, encoding: .utf8)
        let store = try VectorStore(dbPath: engine.indexDBPath, dimension: 8,
                                    enableSQLiteVec: false)
        defer { store.close() }
        let resolved = BookSummaryEngine(store: store).resolve(record, markdown: markdown)
        let count = min(chapterCount ?? resolved.sections.count, resolved.sections.count)
        for section in resolved.sections.prefix(count) {
            try store.upsertChapterSummary(ChapterSummaryRecord(
                docID: record.docID, sectionIndex: section.index,
                heading: section.heading, title: section.title,
                level: section.level,
                summary: "摘要\(section.index + 1)：本节介绍主题\(section.index + 1)。",
                contentHash: section.contentHash))
        }
        let status: BookSummaryJobStatus = withBookSummary ? .completed
            : (count >= resolved.sections.count ? .summarizingBook : .pending)
        try store.upsertSummaryJob(SummaryJobRecord(
            docID: record.docID, status: status, nextSection: count,
            totalSections: resolved.sections.count, updatedAt: 1_000))
        if withBookSummary {
            try store.upsertBookSummary(BookSummaryRecord(
                docID: record.docID,
                summary: String(repeating: "全书内容概括。", count: 50),
                outline: resolved.digest, chapterCount: resolved.sections.count,
                sourceHash: "seed", updatedAt: 1_000))
        }
    }

    func testRagOutlinePrefersBookSummaryAndChapterSummaries() async throws {
        let markdown = """
        # 第一章 水果

        Apple and banana are common fruits. 苹果和香蕉都是常见水果。

        # 第二章 机械

        Turbines and engines are machines. 涡轮和发动机属于机械。

        # 第三章 天文

        Planets orbit stars. 行星围绕恒星运行。
        """
        try write(markdown, relativePath: "book.md")
        let engine = makeEngine()
        _ = await engine.ingest()
        try seedSummaries(engine: engine, docName: "book.md",
                          chapterCount: 3, withBookSummary: true)

        let tool = RagOutlineTool(config: config, engine: engine)
        let result = try await tool.run(["book_name": .string("book.md")])
        XCTAssertTrue(result.success, result.error)
        XCTAssertTrue(result.output.contains("全书摘要:"))
        XCTAssertTrue(result.output.contains("全书内容概括。"))
        XCTAssertTrue(result.output.contains("摘要1：本节介绍主题1。"))
        XCTAssertTrue(result.output.contains("章节摘要进度: 3/3"))
    }

    func testRagOutlineShowsProgressWhenPartial() async throws {
        let markdown = """
        # 第一章 水果

        苹果和香蕉都是常见水果。
        Additional sentences make the body long enough for indexing, so the document is kept in the knowledge base.

        # 第二章 机械

        涡轮和发动机属于机械。

        # 第三章 天文

        行星围绕恒星运行。
        """
        try write(markdown, relativePath: "book.md")
        let engine = makeEngine()
        _ = await engine.ingest()
        try seedSummaries(engine: engine, docName: "book.md",
                          chapterCount: 2, withBookSummary: false)

        let tool = RagOutlineTool(config: config, engine: engine)
        let result = try await tool.run(["book_name": .string("book.md")])
        XCTAssertTrue(result.success, result.error)
        XCTAssertTrue(result.output.contains("章节摘要进度: 2/3"))
        XCTAssertTrue(result.output.contains("后台仍在补充"))
        XCTAssertTrue(result.output.contains("内容概览"))
        XCTAssertFalse(result.output.contains("全书摘要:"))
    }

    func testRagOutlineListsAmbiguousCandidates() async throws {
        try write("# 第一章 甲报告\n\nbookA 的独有正文。", relativePath: "bookA/report.md")
        try write("# 第一章 乙报告\n\nbookB 的独有正文。", relativePath: "bookB/report.md")
        let engine = makeEngine()
        _ = await engine.ingest()

        let tool = RagOutlineTool(config: config, engine: engine)
        let result = try await tool.run(["book_name": .string("report.md")])
        XCTAssertTrue(result.success, result.error)
        XCTAssertTrue(result.output.contains("匹配到多个文档"))
        XCTAssertTrue(result.output.contains("bookA/report.md"))
        XCTAssertTrue(result.output.contains("bookB/report.md"))
    }

    func testRagOutlineFallsBackToLegacyDigestWithoutSummaries() async throws {
        let markdown = """
        # 第一章 水果

        苹果和香蕉都是常见水果。
        Additional sentences make the body long enough for indexing, so the document is kept in the knowledge base.

        # 第二章 机械

        涡轮和发动机属于机械。
        """
        try write(markdown, relativePath: "book.md")
        let engine = makeEngine()
        _ = await engine.ingest()

        let tool = RagOutlineTool(config: config, engine: engine)
        let result = try await tool.run(["book_name": .string("book.md")])
        XCTAssertTrue(result.success, result.error)
        XCTAssertTrue(result.output.contains("内容摘要:"), result.output)
        XCTAssertTrue(result.output.contains("第一章"), result.output)
        XCTAssertTrue(result.output.contains("第二章"), result.output)
    }

    func testDocumentLookupPrefersExactFullPath() async throws {
        try write("""
        # 第一章 甲报告

        bookA 的独有正文。
        Additional sentences make the body long enough for indexing, so the document is kept in the knowledge base.
        """, relativePath: "bookA/report.md")
        try write("""
        # 第一章 乙报告

        bookB 的独有正文。
        Additional sentences make the body long enough for indexing, so the document is kept in the knowledge base.
        """, relativePath: "bookB/report.md")
        let engine = makeEngine()
        _ = await engine.ingest()

        guard case .found(let exact) = engine.lookupDocument("bookA/report.md") else {
            return XCTFail("完整相对路径应精确命中")
        }
        XCTAssertEqual(exact.docName, "bookA/report.md")

        guard case .ambiguous(let candidates) = engine.lookupDocument("report.md") else {
            return XCTFail("同名 basename 应返回歧义候选")
        }
        XCTAssertEqual(Set(candidates.map(\.docName)),
                       ["bookA/report.md", "bookB/report.md"])

        guard case .found(let absolute) = engine.lookupDocument(exact.mdPath) else {
            return XCTFail("完整 md_path 应精确命中")
        }
        XCTAssertEqual(absolute.docName, "bookA/report.md")
    }
}
