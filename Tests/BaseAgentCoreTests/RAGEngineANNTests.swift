import XCTest
@testable import BaseAgentCore

/// 引擎与 ANN 索引的集成：自动构建开关、状态字段、检索走索引/回退。
final class RAGEngineANNTests: XCTestCase {
    private var root: URL!
    private var kb: URL!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ann-engine-\(UUID().uuidString)")
        kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    // MARK: - 夹具

    private final class FakeEmbedding: EmbeddingProvider {
        let identifier = "ann-engine-fake"
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

    private func makeConfig() -> AgentConfig {
        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 5
        config.ragANNClusters = 4
        config.ragANNClusterSize = 8
        return config
    }

    private func writeDocs(_ count: Int) throws {
        for index in 1...count {
            let text = """
            # 文档 \(index)

            Apple banana orange fruit \(index). 苹果 香蕉 橙子 水果 \(index)。
            涡轮 发动机 机械 设备 编号 \(index)。turbine engine machine \(index)。
            """
            try text.write(to: kb.appendingPathComponent("doc\(index).md"),
                           atomically: true, encoding: .utf8)
        }
    }

    private func makeEngine(_ config: AgentConfig) -> RAGEngine {
        RAGEngine(config: config, knowledgeBase: kb.path,
                  embedding: FakeEmbedding(), reranker: nil)
    }

    // MARK: - 用例

    /// 状态快照里必须能读到 ANN 索引状态（未构建 / 已就绪）。
    func testStatusExposesANNIndexState() async throws {
        try writeDocs(2)
        var config = makeConfig()
        // 关闭自动构建：先验证「未构建」状态。
        config.ragANNAutoBuild = false
        let engine = makeEngine(config)
        _ = await engine.ingest()

        let before = engine.status(sourceLimit: 0, maxAge: 0)
        XCTAssertEqual(before["ann_index"]?.stringValue, "ANN 索引未构建")
        XCTAssertEqual(before["ann_usable"]?.boolValue, false)

        let built = try XCTUnwrap(engine.buildANNIndex())
        XCTAssertFalse(built.cancelled)
        XCTAssertGreaterThan(built.indexedRows, 0)

        let after = engine.status(sourceLimit: 0, maxAge: 0)
        XCTAssertEqual(after["ann_usable"]?.boolValue, true)
        XCTAssertEqual(after["ann_needs_rebuild"]?.boolValue, false)
        XCTAssertTrue(after["ann_index"]?.stringValue?.contains("分块 IVF") == true)
        engine.close()
    }

    /// 未达到自动构建的切片数下限时不建索引（小库精确扫描足够快）。
    func testAutoBuildHonoursMinChunks() async throws {
        try writeDocs(3)
        var config = makeConfig()
        config.ragANNMinChunks = 1_000_000
        let engine = makeEngine(config)
        _ = await engine.ingest()

        let status = try XCTUnwrap(engine.annIndexStatus())
        XCTAssertEqual(status.blocks, 0)
        engine.close()
    }

    /// 达到下限后同步结束会自动构建，检索随即走索引。
    func testIngestAutoBuildsIndexAndSearchUsesIt() async throws {
        try writeDocs(3)
        var config = makeConfig()
        config.ragANNMinChunks = 1
        let engine = makeEngine(config)
        let stats = await engine.ingest()
        XCTAssertGreaterThan(stats.chunks, 0)

        let status = try XCTUnwrap(engine.annIndexStatus())
        XCTAssertTrue(status.isReady, status.summary)
        XCTAssertGreaterThan(status.indexedRows, 0)

        let hits = try await engine.search(query: "apple fruit turbine", topK: 3)
        XCTAssertFalse(hits.isEmpty)
        XCTAssertTrue(hits[0].source.hasSuffix(".md"))

        // 关闭 ANN 后结果仍然可用（回退精确检索）。
        var offConfig = config
        offConfig.ragANNEnabled = false
        let offEngine = makeEngine(offConfig)
        let offHits = try await offEngine.search(query: "apple fruit turbine", topK: 3)
        XCTAssertEqual(Set(offHits.map(\.rowID)), Set(hits.map(\.rowID)))
        offEngine.close()
        engine.close()
    }

    /// 索引已就绪时再次构建是空操作（水位在末尾，不会重复写块）。
    func testSecondBuildIsNoopWhenUpToDate() async throws {
        try writeDocs(2)
        var config = makeConfig()
        config.ragANNMinChunks = 1
        let engine = makeEngine(config)
        _ = await engine.ingest()

        let first = try XCTUnwrap(engine.buildANNIndex())
        XCTAssertGreaterThan(first.blocks, 0)
        let second = try XCTUnwrap(engine.buildANNIndex())
        XCTAssertEqual(second.blocks, first.blocks)
        XCTAssertEqual(second.indexedRows, first.indexedRows)
        XCTAssertEqual(second.tailRows, 0)
        XCTAssertFalse(second.cancelled)
        engine.close()
    }
}