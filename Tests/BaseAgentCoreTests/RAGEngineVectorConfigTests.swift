import XCTest
@testable import BaseAgentCore

/// 模拟 nomic：标识符含 nomic，向量已归一化，用于验证 Matryoshka 与 int8 全链路。
private final class NomicVectorFakeEmbedding: EmbeddingProvider,
                                             EmbeddingTokenCounting, @unchecked Sendable {
    let identifier = "local:mlx-embedding:nomic-embed-text-v1.5-mlx-int8"
    let dimension = 8

    var maxContentTokens: Int { 8192 }
    func makeTokenCounter() async -> (@Sendable (String) -> Int)? { nil }

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

/// 模拟 jina-v5：provider 自带 128 维 + int8 硬约束，用于验证配置覆盖。
private final class ConstrainedVectorFakeEmbedding: EmbeddingProvider,
                                                    EmbeddingVectorElementConstraining,
                                                    @unchecked Sendable {
    let identifier = "local:mlx-embedding:renamed-jina-v5"
    let dimension = 128
    var requiredVectorElementType: VectorElementType? { .int8 }

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

final class RAGEngineVectorConfigTests: XCTestCase {
    private var root: URL!
    private var kb: URL!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-vector-\(UUID().uuidString)")
        kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        try """
        # 第一章 水果

        苹果香蕉和橙子是常见的水果。水果富含维生素，适合日常食用。
        补充一段用于保证正文长度超过过短丢弃阈值的说明文字。

        # 第二章 机械

        发动机与涡轮机属于机械装置，用于把能量转换为动力。
        补充一段用于保证正文长度超过过短丢弃阈值的说明文字。
        """.write(to: kb.appendingPathComponent("doc.md"),
                  atomically: true, encoding: .utf8)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private func makeConfig(elementType: VectorElementType,
                            matryoshka: MatryoshkaMode,
                            matryoshkaDim: Int) -> AgentConfig {
        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 80
        config.ragChunkOverlap = 8
        config.ragVectorElementType = elementType
        config.ragMatryoshka = matryoshka
        config.ragMatryoshkaDim = matryoshkaDim
        return config
    }

    func testNomicMatryoshkaAndInt8EndToEnd() async throws {
        let config = makeConfig(elementType: .int8, matryoshka: .auto, matryoshkaDim: 4)
        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: NomicVectorFakeEmbedding(), reranker: nil)
        defer { engine.close() }

        XCTAssertEqual(engine.effectiveMatryoshkaDimension, 4)
        XCTAssertTrue(engine.vectorIndexFingerprint.contains("matryoshka=4"))

        let stats = await engine.ingest()
        XCTAssertTrue(stats.errors.isEmpty, stats.errors.joined(separator: "; "))
        XCTAssertGreaterThan(stats.chunks, 0)

        let status = engine.status(maxAge: 0)
        XCTAssertEqual(status["vector_element_type"]?.stringValue, "int8")
        XCTAssertEqual(status["matryoshka_dim"]?.intValue, 4)
        XCTAssertEqual(status["vector_rebuild_required"]?.boolValue, false)

        let hits = try await engine.search(query: "水果")
        XCTAssertFalse(hits.isEmpty, "入库与查询共用 Matryoshka + int8 后应能正常检索")
    }

    /// B 方案：元素类型变化时不删旧数据，标记载重建并拒绝 ingest。
    func testElementTypeChangeGatesIngestUntilExplicitRebuild() async throws {
        let int8Config = makeConfig(elementType: .int8, matryoshka: .off, matryoshkaDim: 512)
        let first = RAGEngine(config: int8Config, knowledgeBase: kb.path,
                              embedding: NomicVectorFakeEmbedding(), reranker: nil)
        let firstStats = await first.ingest()
        XCTAssertGreaterThan(firstStats.chunks, 0)
        let originalChunks = firstStats.chunks
        first.close()

        let fp32Config = makeConfig(elementType: .fp32, matryoshka: .off, matryoshkaDim: 512)
        let second = RAGEngine(config: fp32Config, knowledgeBase: kb.path,
                               embedding: NomicVectorFakeEmbedding(), reranker: nil)
        defer { second.close() }

        let gated = second.status(maxAge: 0)
        XCTAssertEqual(gated["vector_rebuild_required"]?.boolValue, true)
        XCTAssertFalse((gated["rebuild_reason"]?.stringValue ?? "").isEmpty)
        XCTAssertEqual(gated["chunks_stored"]?.intValue, originalChunks,
                       "待重建时旧切片必须保留")

        let blocked = await second.ingest()
        XCTAssertEqual(blocked.processed, 0)
        XCTAssertTrue(blocked.errors.contains { $0.contains("重建") },
                      "待重建时必须拒绝写入并提示重建")

        let rebuilt = await second.rebuildVectorIndex(resync: false)
        XCTAssertEqual(rebuilt?.removedChunks, originalChunks)
        let after = second.status(maxAge: 0)
        XCTAssertEqual(after["vector_rebuild_required"]?.boolValue, false)
        XCTAssertEqual(after["vector_element_type"]?.stringValue, "fp32")
        XCTAssertEqual(after["chunks_stored"]?.intValue, 0)

        // 重建后写入应恢复，且指纹与当前配置一致。
        let resumed = await second.ingest()
        XCTAssertTrue(resumed.errors.isEmpty, resumed.errors.joined(separator: "; "))
        XCTAssertGreaterThan(resumed.chunks, 0)
        XCTAssertEqual(second.status(maxAge: 0)["vector_rebuild_required"]?.boolValue, false)
    }

    /// jina-v5 固定 128 维 + int8：即使配置 fp32，入库/查询也必须回落 int8。
    func testJinaV5ConstraintOverridesConfiguredFp32() async throws {
        let config = makeConfig(elementType: .fp32, matryoshka: .auto, matryoshkaDim: 4)
        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: ConstrainedVectorFakeEmbedding(), reranker: nil)
        defer { engine.close() }

        XCTAssertEqual(engine.effectiveVectorElementType, .int8)
        XCTAssertEqual(engine.effectiveMatryoshkaDimension, 0)
        let fingerprint = engine.vectorIndexFingerprint
        XCTAssertTrue(fingerprint.contains("dimension=128"), fingerprint)
        XCTAssertTrue(fingerprint.contains("element=int8"), fingerprint)

        let stats = await engine.ingest()
        XCTAssertTrue(stats.errors.isEmpty, stats.errors.joined(separator: "; "))
        XCTAssertGreaterThan(stats.chunks, 0)

        let status = engine.status(maxAge: 0)
        XCTAssertEqual(status["vector_element_type"]?.stringValue, "int8")
        XCTAssertEqual(status["vector_rebuild_required"]?.boolValue, false)

        let hits = try await engine.search(query: "水果")
        XCTAssertFalse(hits.isEmpty, "int8 入库与查询应能正常检索")
    }
}
