import XCTest
@testable import BaseAgentCore

/// 向量表按元素类型建表、写入、检索与显式重建的端到端测试。
final class VectorStoreElementTypeTests: XCTestCase {
    private var directory: URL!
    private let dimension = 16

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-vector-type-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private func dbPath(_ name: String) -> String {
        directory.appendingPathComponent("\(name).sqlite").path
    }

    private func oneHot(_ index: Int) -> [Float] {
        var vector = [Float](repeating: 0, count: dimension)
        vector[index] = 1
        return vector
    }

    private func chunks(_ count: Int) -> [Chunk] {
        (0..<count).map { index in
            Chunk(text: "doc \(index)", source: "s\(index).md", chunkIndex: 0)
        }
    }

    // MARK: - 建表 / 检索

    func testSearchAcrossSupportedElementTypes() throws {
        for type in [VectorElementType.fp32, .int8, .bin] {
            let store = try VectorStore(dbPath: dbPath("search-\(type.rawValue)"),
                                        dimension: dimension,
                                        elementType: type,
                                        indexFingerprint: "fingerprint-\(type.rawValue)")
            XCTAssertEqual(store.elementType, type)
            XCTAssertTrue(store.vectorEnabled, "\(type)：sqlite-vec 应可用")
            XCTAssertFalse(store.requiresRebuild)

            let vectors = (0..<4).map { oneHot($0) }
            XCTAssertEqual(try store.add(chunks: chunks(4), vectors: vectors), 4)

            let hits = try store.search(vector: oneHot(0), topK: 2)
            XCTAssertEqual(hits.first?.text, "doc 0", "\(type)")
            XCTAssertGreaterThan(hits.first?.score ?? 0, 0.9, "\(type)")
            XCTAssertEqual(store.count, 4)
            store.close()
        }
    }

    /// 退化到普通表时，精确扫描也必须按元素类型解码（不能把 int8/bin/fp16 当 float32）。
    func testExactScanPathForQuantizedTypes() throws {
        for type in [VectorElementType.int8, .bin, .fp16] {
            let store = try VectorStore(dbPath: dbPath("exact-\(type.rawValue)"),
                                        dimension: dimension,
                                        enableSQLiteVec: false,
                                        elementType: type,
                                        indexFingerprint: "exact-\(type.rawValue)")
            XCTAssertFalse(store.vectorEnabled)
            _ = try store.add(chunks: chunks(4), vectors: (0..<4).map { oneHot($0) })

            let hits = try store.search(vector: oneHot(0), topK: 2)
            XCTAssertEqual(hits.first?.text, "doc 0", "\(type)")
            XCTAssertGreaterThan(hits.first?.score ?? 0, 0.9, "\(type)")
            store.close()
        }
    }

    /// fp16 没有 vec0 原生列类型：普通表 + 精确扫描，但仍是真正的 2 字节 half。
    func testFP16UsesHalfPrecisionStorageAndExactScan() throws {
        let path = dbPath("fp16")
        let store = try VectorStore(dbPath: path, dimension: dimension,
                                    elementType: .fp16,
                                    indexFingerprint: "fp16")
        XCTAssertEqual(store.elementType, .fp16)
        XCTAssertFalse(store.vectorEnabled, "fp16 没有 vec0 列类型，应走精确扫描")
        XCTAssertFalse(store.requiresRebuild)

        XCTAssertEqual(try store.add(chunks: chunks(4),
                                     vectors: (0..<4).map { oneHot($0) }), 4)
        let blobBytes = try store.db.scalar(
            "SELECT length(embedding) FROM chunk_vectors LIMIT 1").intValue
        XCTAssertEqual(blobBytes, Int64(dimension * 2), "fp16 每维 2 字节")

        let hits = try store.search(vector: oneHot(0), topK: 2)
        XCTAssertEqual(hits.first?.text, "doc 0")
        XCTAssertGreaterThan(hits.first?.score ?? 0, 0.99)
        store.close()

        // 重开：meta 记录的元素类型可以沿用，且旧 half 仍可检索。
        let reopened = try VectorStore(dbPath: path, dimension: dimension,
                                       elementType: .fp16,
                                       indexFingerprint: "fp16")
        XCTAssertFalse(reopened.requiresRebuild)
        XCTAssertEqual(try reopened.search(vector: oneHot(0), topK: 1).first?.text, "doc 0")
        reopened.close()
    }

    // MARK: - 指纹闸门 / 重建

    func testFingerprintMismatchMarksRebuildWithoutDeletingData() throws {
        let path = dbPath("gate")
        do {
            let store = try VectorStore(dbPath: path, dimension: dimension,
                                        elementType: .int8,
                                        indexFingerprint: "fingerprint-a")
            _ = try store.add(chunks: chunks(2),
                              vectors: [oneHot(0), oneHot(1)])
            store.close()
        }

        let reopened = try VectorStore(dbPath: path, dimension: dimension,
                                       elementType: .fp32,
                                       indexFingerprint: "fingerprint-b")
        XCTAssertTrue(reopened.requiresRebuild)
        XCTAssertFalse(reopened.vectorEnabled, "待重建时向量通道必须关闭")
        XCTAssertEqual(reopened.count, 2, "不能静默删除旧数据")

        XCTAssertThrowsError(try reopened.add(chunks: [Chunk(text: "new", source: "new.md",
                                                             chunkIndex: 0)],
                                              vectors: [oneHot(2)])) { error in
            XCTAssertTrue(error.localizedDescription.contains("重建"))
        }
        XCTAssertTrue(try reopened.search(vector: oneHot(0), topK: 2).isEmpty,
                      "待重建时向量检索应跳过")
        XCTAssertFalse(try reopened.searchFTS(query: "doc", topK: 2).isEmpty,
                       "全文检索仍应可用")

        let removed = try reopened.rebuildForCurrentConfiguration()
        XCTAssertEqual(removed, 2)
        XCTAssertFalse(reopened.requiresRebuild)
        XCTAssertEqual(reopened.elementType, .fp32)
        XCTAssertEqual(reopened.count, 0)

        _ = try reopened.add(chunks: [Chunk(text: "new", source: "new.md", chunkIndex: 0)],
                             vectors: [oneHot(0)])
        XCTAssertEqual(reopened.count, 1)
        XCTAssertEqual(try reopened.search(vector: oneHot(0), topK: 1).first?.text, "new")
        reopened.close()
    }

    func testRebuildGateSurvivesReopenAndKeepsOldVectors() throws {
        let path = dbPath("gate-reopen")
        do {
            let store = try VectorStore(dbPath: path, dimension: dimension,
                                        elementType: .int8,
                                        indexFingerprint: "fingerprint-a")
            _ = try store.add(chunks: chunks(2),
                              vectors: [oneHot(0), oneHot(1)])
            store.close()
        }
        // 以不匹配的配置打开一次：只标记待重建，不写回新 meta。
        do {
            let mismatched = try VectorStore(dbPath: path, dimension: dimension,
                                             elementType: .fp32,
                                             indexFingerprint: "fingerprint-b")
            XCTAssertTrue(mismatched.requiresRebuild)
            mismatched.close()
        }
        // 再次以同样的错误配置打开：旧 meta 必须仍记录原配置，
        // 否则会按 fp32 重建向量表，静默丢掉 int8 旧向量。
        do {
            let mismatched = try VectorStore(dbPath: path, dimension: dimension,
                                             elementType: .fp32,
                                             indexFingerprint: "fingerprint-b")
            XCTAssertTrue(mismatched.requiresRebuild, "待重建状态必须跨重启保留")
            XCTAssertFalse(mismatched.vectorEnabled)
            XCTAssertEqual(mismatched.count, 2)
            mismatched.close()
        }
        // 回到原配置：旧向量仍应可检索，证明第二次打开没有清空向量表。
        let original = try VectorStore(dbPath: path, dimension: dimension,
                                       elementType: .int8,
                                       indexFingerprint: "fingerprint-a")
        XCTAssertFalse(original.requiresRebuild)
        XCTAssertEqual(try original.search(vector: oneHot(0), topK: 1).first?.text, "doc 0")
        original.close()
    }

    func testDimensionMismatchWithDataRequiresRebuild() throws {
        let path = dbPath("dimension")
        do {
            let store = try VectorStore(dbPath: path, dimension: dimension,
                                        elementType: .int8,
                                        indexFingerprint: "dim-a")
            _ = try store.add(chunks: chunks(1), vectors: [oneHot(0)])
            store.close()
        }

        // 维度变化：旧库有数据时同样只标记待重建，不清空。
        let reopened = try VectorStore(dbPath: path, dimension: 32,
                                       elementType: .int8,
                                       indexFingerprint: "dim-b")
        XCTAssertTrue(reopened.requiresRebuild)
        XCTAssertEqual(reopened.count, 1)
        reopened.close()
    }

    /// 空库遇到新配置时自动采纳（不要求用户先重建空库）。
    func testEmptyStoreAdoptsNewConfiguration() throws {
        let path = dbPath("empty")
        do {
            let store = try VectorStore(dbPath: path, dimension: dimension,
                                        elementType: .int8,
                                        indexFingerprint: "empty-a")
            _ = try store.add(chunks: chunks(1), vectors: [oneHot(0)])
            _ = try store.clear()
            store.close()
        }
        let reopened = try VectorStore(dbPath: path, dimension: 32,
                                       elementType: .fp32,
                                       indexFingerprint: "empty-b")
        XCTAssertFalse(reopened.requiresRebuild)
        XCTAssertEqual(reopened.count, 0)
        XCTAssertTrue(reopened.vectorEnabled)
        reopened.close()
    }

    // MARK: - ANN 跟随元素类型

    func testANNIndexUsesElementTypeLayout() throws {
        for type in [VectorElementType.int8, .bin, .fp16] {
            let store = try VectorStore(dbPath: dbPath("ann-\(type.rawValue)"),
                                        dimension: dimension,
                                        elementType: type,
                                        indexFingerprint: "ann-\(type.rawValue)")
            var allChunks: [Chunk] = []
            var allVectors: [[Float]] = []
            for cluster in 0..<6 {
                for offset in 0..<30 {
                    var vector = oneHot(cluster)
                    let noise = Float(offset % 7) * 0.01
                    vector[(cluster + 3) % dimension] += noise
                    vector[(cluster + 5) % dimension] -= noise / 2
                    allVectors.append(VectorCodec.normalize(vector))
                    allChunks.append(Chunk(text: "cluster \(cluster) item \(offset)",
                                           source: "c\(cluster).md",
                                           chunkIndex: offset))
                }
            }
            _ = try store.add(chunks: allChunks, vectors: allVectors)

            let build = try store.buildANNIndex(clusters: 4, clusterSize: 32, iterations: 8)
            XCTAssertFalse(build.cancelled, "\(type)")
            XCTAssertGreaterThan(build.indexedRows, 0, "\(type)")
            let status = store.annIndexStatus()
            XCTAssertEqual(status.dimension, dimension, "\(type)")
            XCTAssertFalse(status.needsRebuild, "\(type)：\(status.summary)")

            let hits = try XCTUnwrap(store.annSearch(vector: oneHot(2), topK: 5, nprobe: 4),
                                    "\(type)")
            XCTAssertFalse(hits.isEmpty, "\(type)")
            XCTAssertTrue(hits.allSatisfy { $0.source == "c2.md" },
                          "\(type)：ANN 召回必须来自查询所在簇，实际 \(hits.map(\.source))")
            store.close()
        }
    }
}
