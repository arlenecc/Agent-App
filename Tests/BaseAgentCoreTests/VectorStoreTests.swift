import XCTest
@testable import BaseAgentCore

final class VectorStoreTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-vector-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    func testSQLiteVecTransactionalWriteTwoStageKNNAndHybridSearch() throws {
        let store = try VectorStore(dbPath: directory.appendingPathComponent("index.sqlite").path,
                                    dimension: 3, enableSQLiteVec: true)
        XCTAssertTrue(store.vectorEnabled, "sqlite-vec 应成功静态注册")

        let chunks = [
            Chunk(text: "apple banana fruit", source: "bookA.pdf", chunkIndex: 0, heading: "第一章"),
            Chunk(text: "orange grape fruit", source: "bookA.pdf", chunkIndex: 1),
            Chunk(text: "engine turbine machine", source: "bookB.pdf", chunkIndex: 0),
        ]
        let vectors: [[Float]] = [[1, 0, 0], [0.8, 0.2, 0], [0, 1, 0]]
        XCTAssertEqual(try store.add(chunks: chunks, vectors: vectors), 3)
        XCTAssertEqual(store.count, 3)

        // 两段式 KNN：先向量表取 rowid，再主表取正文。
        let hits = try store.search(vector: [1, 0, 0], topK: 2)
        XCTAssertEqual(hits.count, 2)
        XCTAssertEqual(hits.first?.text, "apple banana fruit")

        // over-fetch + 第二阶段过滤。
        let filtered = try store.search(vector: [1, 0, 0], topK: 2, sourceContains: "bookB")
        XCTAssertEqual(filtered.count, 1)
        XCTAssertEqual(filtered.first?.source, "bookB.pdf")

        // FTS5 关键词检索。
        let fts = try store.searchFTS(query: "apple", topK: 3)
        XCTAssertEqual(fts.count, 1)
        XCTAssertEqual(fts.first?.chunkIndex, 0)

        // RRF 融合：向量与 FTS 都命中 apple。
        let hybrid = try store.hybridSearch(vector: [1, 0, 0], query: "apple", topK: 3)
        XCTAssertFalse(hybrid.isEmpty)
        XCTAssertEqual(hybrid.first?.rowID, hits.first?.rowID)

        // 单事务删除。
        XCTAssertEqual(try store.deleteBySource("bookA.pdf"), 2)
        XCTAssertEqual(store.count, 1)
        XCTAssertTrue(try store.searchFTS(query: "apple", topK: 3).isEmpty)

        // 写入失败必须回滚（包括其前置删除）。
        XCTAssertThrowsError(try store.add(
            chunks: [Chunk(text: "bad", source: "bookB.pdf", chunkIndex: 0)],
            vectors: [[1, 0]])
        )
        XCTAssertEqual(store.count, 1)
    }

    func testDeduplicateTextChunksKeepsEarliestCopy() throws {
        let store = try VectorStore(dbPath: directory.appendingPathComponent("dedupe.sqlite").path,
                                    dimension: 2)
        let chunks = [
            Chunk(text: "重复正文", source: "a.md", chunkIndex: 0),
            Chunk(text: "唯一 A", source: "a.md", chunkIndex: 1),
            Chunk(text: "重复正文", source: "b.md", chunkIndex: 0),
            Chunk(text: "重复正文", source: "c.md", chunkIndex: 0),
            Chunk(text: "唯一 B", source: "b.md", chunkIndex: 1),
        ]
        _ = try store.add(chunks: chunks,
                          vectors: [[1, 0], [0, 1], [1, 0], [1, 0], [0, 1]])
        // 入库时间：a.md 最早，b.md 次之，c.md 最晚。
        for (source, created) in [("a.md", 100.0), ("b.md", 200.0), ("c.md", 300.0)] {
            try store.db.execute("UPDATE chunks SET created_at = ? WHERE source = ?",
                                 [.double(created), .text(source)])
        }

        let preview = try store.previewDuplicateTextChunks(pageSize: 2)
        XCTAssertEqual(preview.scannedRows, 5)
        XCTAssertEqual(preview.duplicateRows, 2, "预览只统计重复，不删除")
        XCTAssertEqual(preview.removedRows, 0)
        XCTAssertEqual(store.count, 5)

        let result = try store.deduplicateTextChunks(pageSize: 2)
        XCTAssertEqual(result.removedRows, 2)
        XCTAssertEqual(result.removedBySource["b.md"], 1)
        XCTAssertEqual(result.removedBySource["c.md"], 1)
        XCTAssertEqual(result.canonicalBySource["b.md"], "a.md")
        XCTAssertEqual(result.canonicalBySource["c.md"], "a.md")
        XCTAssertEqual(store.chunkCount(source: "a.md"), 2, "最早入库的一份必须保留")
        XCTAssertEqual(store.chunkCount(source: "b.md"), 1)
        XCTAssertEqual(store.chunkCount(source: "c.md"), 0)
        XCTAssertEqual(store.count, 3)
        XCTAssertEqual(store.sourceCount(), 2, "c.md 的切片被清空后不再计入来源数")
        XCTAssertEqual(store.listSources(limit: 2).count, 2)
        XCTAssertEqual(try store.searchFTS(query: "重复正文", topK: 3).count, 1)

        // 重复执行幂等。
        XCTAssertEqual(try store.deduplicateTextChunks().removedRows, 0)
        store.close()
    }

    func testCJKFTSTokenization() throws {
        let store = try VectorStore(dbPath: directory.appendingPathComponent("cjk.sqlite").path,
                                    dimension: 2)
        let chunks = [
            Chunk(text: "人工智能技术正在快速发展", source: "zh.md", chunkIndex: 0),
            Chunk(text: "今天天气很好", source: "zh.md", chunkIndex: 1),
        ]
        _ = try store.add(chunks: chunks, vectors: [[1, 0], [0, 1]])
        let hits = try store.searchFTS(query: "人工智能", topK: 3)
        XCTAssertEqual(hits.count, 1)
        XCTAssertTrue(hits[0].text.contains("人工智能"))
    }

    func testChunkCounterPersistsAcrossReopenAndLegacyFallback() throws {
        let path = directory.appendingPathComponent("counter.sqlite").path
        do {
            let store = try VectorStore(dbPath: path, dimension: 2)
            _ = try store.add(chunks: [
                Chunk(text: "a", source: "a.md", chunkIndex: 0),
                Chunk(text: "b", source: "a.md", chunkIndex: 1),
                Chunk(text: "c", source: "b.md", chunkIndex: 0),
            ], vectors: [[1, 0], [0, 1], [1, 1]])
            XCTAssertEqual(store.count, 3)
            XCTAssertEqual(try store.deleteBySource("a.md"), 2)
            XCTAssertEqual(store.count, 1)
            store.close()
        }
        do {
            let store = try VectorStore(dbPath: path, dimension: 2)
            XCTAssertEqual(store.count, 1, "计数器必须随库持久化，重开后无需 COUNT(*)")
            // 模拟升级前的旧库：没有 meta 计数器。
            try store.db.execute("DELETE FROM meta WHERE key = 'chunks_count'")
            store.close()
        }
        do {
            let legacy = try VectorStore(dbPath: path, dimension: 2)
            XCTAssertEqual(legacy.count, 3,
                           "旧库回退 MAX(rowid)（O(1)，允许略高于实际值）")
            XCTAssertTrue(legacy.hasChunks)
            legacy.close()
        }
    }
}
