import XCTest
@testable import BaseAgentCore

/// 知识库查询超时控制：慢查询必须在期限内被中断，调用方拿到超时而不是无限等待。
final class QueryTimeoutTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-timeout-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private var dbPath: String { directory.appendingPathComponent("index.sqlite").path }

    /// 制造一段可预期的慢查询（递归 CTE 生成 2000 万行，正常需数秒到数十秒）。
    private static let slowSQL =
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 20000000) "
        + "SELECT COUNT(*) FROM c"

    private func assertTimeout(_ error: Error, file: StaticString = #filePath, line: UInt = #line) {
        guard case AgentError.timeout = error else {
            return XCTFail("应为超时错误，实际 \(error)", file: file, line: line)
        }
    }

    // MARK: - SQLiteDatabase 期限

    func testDeadlineInterruptsLongRunningStatement() throws {
        let db = try SQLiteDatabase(path: dbPath)
        defer { db.close() }

        let started = Date()
        XCTAssertThrowsError(try db.withDeadline(seconds: 0.05) {
            _ = try db.scalar(Self.slowSQL)
        }, "慢查询必须被期限中断") { error in
            self.assertTimeout(error)
        }
        XCTAssertLessThan(Date().timeIntervalSince(started), 5,
                          "中断必须尽快生效，而不是等查询自己跑完")

        // 中断后连接仍可正常使用（不能留下半开的语句/失效的 handler）。
        XCTAssertEqual(try db.scalar("SELECT 7").intValue, 7)
    }

    func testDeadlineAllowsFastStatementAndClearsAfterwards() throws {
        let db = try SQLiteDatabase(path: dbPath)
        defer { db.close() }

        let value = try db.withDeadline(seconds: 5) { try db.scalar("SELECT 42") }
        XCTAssertEqual(value.intValue, 42)

        // 期限只在 body 内生效：body 结束后慢查询应能正常跑完。
        let total = try db.scalar(
            "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c WHERE x < 3000000) "
            + "SELECT COUNT(*) FROM c")
        XCTAssertEqual(total.intValue, 3_000_000)
    }

    func testNonPositiveDeadlineCountsAsExpired() throws {
        let db = try SQLiteDatabase(path: dbPath)
        defer { db.close() }

        // 期限内不给 0：非正数视为「期限已过」。配置里的 0 = 关闭超时，
        // 由 RAGEngine 决定不调用带期限的入口，而不是在这里放行。
        XCTAssertThrowsError(try db.withDeadline(seconds: 0) {
            _ = try db.scalar("SELECT 1")
        }) { error in
            self.assertTimeout(error)
        }
    }

    func testExpiredDeadlineThrowsBeforeRunningBody() throws {
        let db = try SQLiteDatabase(path: dbPath)
        defer { db.close() }
        try db.execute("CREATE TABLE probe(key TEXT PRIMARY KEY)")

        XCTAssertThrowsError(try db.withDeadline(seconds: 0.000000001) {
            try db.execute("INSERT INTO probe(key) VALUES('ran')")
        }, "已过期的期限必须在执行前直接跳过") { error in
            self.assertTimeout(error)
        }
        XCTAssertEqual(try db.scalar("SELECT COUNT(*) FROM probe").intValue, 0,
                       "超时的 body 不应产生任何副作用")
    }

    // MARK: - VectorStore 期限

    func testStoreQueryDeadlineInterruptsSearch() throws {
        let store = try VectorStore(dbPath: dbPath, dimension: 3)
        defer { store.close() }
        _ = try store.add(chunks: [
            Chunk(text: "apple", source: "a.md", chunkIndex: 0),
        ], vectors: [[1, 0, 0]])

        XCTAssertThrowsError(try store.withQueryDeadline(seconds: 0.000000001) {
            _ = try store.search(vector: [1, 0, 0], topK: 1)
        }) { error in
            self.assertTimeout(error)
        }
        // 正常期限下检索照旧。
        XCTAssertEqual(try store.withQueryDeadline(seconds: 5) {
            try store.search(vector: [1, 0, 0], topK: 1)
        }.count, 1)
    }

    // MARK: - RAGEngine 超时行为

    private func makeEngine(_ root: URL, text: String) throws -> (RAGEngine, URL) {
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        try text.write(to: kb.appendingPathComponent("a.md"), atomically: true, encoding: .utf8)
        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 2
        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: FakeTimeoutEmbedding(), reranker: nil)
        return (engine, kb)
    }

    func testStatusReturnsSkipSnapshotOnTimeout() async throws {
        let root = directory.appendingPathComponent("status-timeout")
        let (engine, _) = try makeEngine(root, text: "# 标题\n\n用于验证状态查询超时的正文内容。")
        defer { engine.close() }
        _ = await engine.ingest()

        let timedOut = engine.status(timeout: 0.000000001)
        let message = timedOut["error"]?.stringValue ?? ""
        XCTAssertTrue(message.contains("超时"), "超时必须给出可读提示，实际: \(timedOut)")
        XCTAssertFalse(message.isEmpty)

        // 正常期限下照旧返回统计。
        let normal = engine.status(maxAge: 0, timeout: 5)
        XCTAssertNil(normal["error"])
        XCTAssertGreaterThan(normal["chunks_stored"]?.intValue ?? 0, 0)
    }

    func testSearchThrowsTimeoutInsteadOfHanging() async throws {
        let root = directory.appendingPathComponent("search-timeout")
        let (engine, _) = try makeEngine(root, text: "# 水果\n\nApple and banana are fruits. 苹果和香蕉。")
        defer { engine.close() }
        _ = await engine.ingest()

        do {
            _ = try await engine.search(query: "apple", topK: 3, timeout: 0.000000001)
            XCTFail("超时必须抛出 AgentError.timeout")
        } catch {
            assertTimeout(error)
        }
        // 正常期限内检索仍可用。
        let hits = try await engine.search(query: "apple fruit", topK: 3, timeout: 5)
        XCTAssertFalse(hits.isEmpty)
    }

    func testSearchTimeoutFollowsConfigDefault() async throws {
        let root = directory.appendingPathComponent("search-config")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        try "# 水果\n\nApple and banana are fruits.".write(
            to: kb.appendingPathComponent("a.md"), atomically: true, encoding: .utf8)
        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 2
        // 配置只影响默认值：0 表示关闭超时，检索不因未传 timeout 而变慢。
        config.ragQueryTimeoutSeconds = 0
        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: FakeTimeoutEmbedding(), reranker: nil)
        defer { engine.close() }
        _ = await engine.ingest()
        XCTAssertEqual(engine.queryTimeoutSeconds, 0)
        let hits = try await engine.search(query: "apple", topK: 3)
        XCTAssertFalse(hits.isEmpty)
    }
}

private final class FakeTimeoutEmbedding: EmbeddingProvider {
    let identifier = "timeout-fake"
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