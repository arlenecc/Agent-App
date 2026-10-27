import XCTest
@testable import BaseAgentCore

/// 模拟 OCR 有失败页的解析器：doc1 永远报告 1 页失败，doc2 正常。
private final class PartialFailureParser: DocumentParser {
    let name = "PartialFailure"
    let supportedExtensions = ["md", "markdown"]

    func parse(url: URL,
               options: ParserOptions,
               progress: ((ParserProgress) -> Void)?) async throws -> ParsedDocument {
        let text = try String(contentsOf: url, encoding: .utf8)
        let failed = url.lastPathComponent.contains("doc1") ? "1" : "0"
        var parsed = ParsedDocument(text: text,
                                    source: url.lastPathComponent,
                                    metadata: ["engine": "partial",
                                               "failed_pages": failed])
        // 模拟 OCR 路径：走激进清洗档。
        parsed = finish(parsed, options: options, profile: .ocr)
        return parsed
    }
}

private final class RagFakeEmbedding: EmbeddingProvider {
    let identifier = "rag-fake"
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

final class RAGEngineTests: XCTestCase {
    func testIngestAndHybridSearchEndToEnd() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let markdown = """
        # 第一章 水果

        Apple and banana are common fruits. 苹果和香蕉都是常见水果。

        # 第二章 机械

        Turbines and engines are machines. 涡轮和发动机属于机械。
        """
        try markdown.write(to: kb.appendingPathComponent("fruits.md"),
                           atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 5

        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: RagFakeEmbedding(), reranker: nil)
        let stats = await engine.ingest()
        XCTAssertEqual(stats.processed, 1)
        XCTAssertGreaterThan(stats.chunks, 0)
        XCTAssertTrue(stats.errors.isEmpty, stats.errors.joined(separator: "; "))

        XCTAssertEqual(engine.listDocuments().count, 1)
        let outline = engine.getDocumentOutline(docName: "fruits.md")
        XCTAssertNotNil(outline)
        XCTAssertTrue(outline?.digest.contains("第一章") == true)

        let hits = try await engine.search(query: "apple fruit", topK: 3)
        XCTAssertFalse(hits.isEmpty)
        XCTAssertTrue(hits[0].source.hasSuffix("fruits.md"))
        XCTAssertTrue(hits[0].text.contains("Apple") || hits[0].text.contains("苹果"))

        // 无变更时增量同步应全部跳过。
        let second = await engine.ingest()
        XCTAssertEqual(second.processed, 0)
        XCTAssertEqual(second.skipped, 1)

        let formatted = try await engine.searchFormatted(query: "turbine", topK: 2)
        XCTAssertTrue(formatted.contains("fruits.md"))

        // 删除源文件后再次同步：切片与 documents 摘要都必须清理。
        try FileManager.default.removeItem(at: kb.appendingPathComponent("fruits.md"))
        let third = await engine.ingest()
        XCTAssertEqual(third.removed, 1)
        XCTAssertTrue(engine.listDocuments().isEmpty)
        let afterDelete = try await engine.search(query: "apple", topK: 3)
        XCTAssertTrue(afterDelete.isEmpty)
        engine.close()
    }

    func testCancelledFullRebuildResumesFromUnfinishedFiles() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-resume-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        for index in 1...3 {
            let text = """
            # 文档 \(index)

            Apple banana orange fruit \(index). 苹果 香蕉 橙子 水果 \(index)。
            这是一段用于验证中断续跑的文件内容，编号 \(index)。
            """
            try text.write(to: kb.appendingPathComponent("doc\(index).md"),
                           atomically: true, encoding: .utf8)
        }

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 2

        let firstEngine = RAGEngine(config: config, knowledgeBase: kb.path,
                                    embedding: RagFakeEmbedding(), reranker: nil)
        let initial = await firstEngine.ingest()
        XCTAssertEqual(initial.processed, 3)
        firstEngine.close()

        // 修改切片参数触发逐文件重建；处理完第一个文件后停止。
        // 并发 ingest 下停止时会保留已完成的文件，这里固定为单文件便于断言。
        config.ragChunkSize = 60
        config.ragChunkOverlap = 3
        config.ragIngestConcurrency = 1
        let partialEngine = RAGEngine(config: config, knowledgeBase: kb.path,
                                      embedding: RagFakeEmbedding(), reranker: nil)
        var cancelled = false
        let partial = await partialEngine.ingest(log: { line in
            if !cancelled, line.contains("✓") {
                cancelled = true
                partialEngine.cancel()
            }
        })
        XCTAssertTrue(partial.cancelled)
        XCTAssertEqual(partial.processed, 1)
        partialEngine.close()

        // 同一新配置再次同步：已完成的 1 个文件跳过，只处理剩余 2 个。
        let resumeEngine = RAGEngine(config: config, knowledgeBase: kb.path,
                                     embedding: RagFakeEmbedding(), reranker: nil)
        let resumed = await resumeEngine.ingest()
        XCTAssertFalse(resumed.cancelled)
        XCTAssertEqual(resumed.skipped, 1)
        XCTAssertEqual(resumed.processed, 2)
        XCTAssertTrue(resumed.errors.isEmpty, resumed.errors.joined(separator: "; "))
        resumeEngine.close()
    }

    func testIdenticalFilesAcrossDirectoriesAreParsedOnce() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-dup-raw-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        let dirA = kb.appendingPathComponent("a")
        let dirB = kb.appendingPathComponent("b")
        try FileManager.default.createDirectory(at: dirA, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: dirB, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        let content = """
        # 同一本书

        Apple banana orange fruit. 苹果 香蕉 橙子 水果。
        这是用于验证跨目录重复文件的正文。
        """
        try content.write(to: dirA.appendingPathComponent("book.md"),
                          atomically: true, encoding: .utf8)
        try content.write(to: dirB.appendingPathComponent("book.md"),
                          atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 2

        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: RagFakeEmbedding(), reranker: nil)
        let first = await engine.ingest()
        XCTAssertEqual(first.processed, 1, "相同文件只应解析一次")
        XCTAssertEqual(first.deduplicated, 0)
        XCTAssertEqual(engine.status()["sources"]?.arrayValue?.count, 1)
        XCTAssertEqual(engine.listDocuments().count, 1)

        let second = await engine.ingest()
        XCTAssertEqual(second.processed, 0)
        XCTAssertEqual(second.skipped, 1)
        XCTAssertEqual(second.deduplicated, 0)
        XCTAssertEqual(engine.status()["sources"]?.arrayValue?.count, 1)
        engine.close()
    }

    func testContentHashDedupeRemovesDuplicateVectorsAndSkipsNextRun() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-dup-content-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        let dirA = kb.appendingPathComponent("a")
        let dirB = kb.appendingPathComponent("b")
        try FileManager.default.createDirectory(at: dirA, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: dirB, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        // 原始字节不同（空行/结尾不同），清洗后的正文完全一致。
        try "# 同一本书\n\n正文内容。" .write(
            to: dirA.appendingPathComponent("book.md"),
            atomically: true, encoding: .utf8)
        try "# 同一本书\n\n\n正文内容。\n".write(
            to: dirB.appendingPathComponent("book.md"),
            atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 2

        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: RagFakeEmbedding(), reranker: nil)
        let first = await engine.ingest()
        XCTAssertEqual(first.processed, 2)
        XCTAssertEqual(first.deduplicated, 1, "content_hash 相同的副本应移除向量")
        XCTAssertEqual(engine.status()["sources"]?.arrayValue?.count, 1)
        XCTAssertEqual(engine.listDocuments().count, 1)

        // 副本写入了 tombstone，下一次同步不再重复解析。
        let second = await engine.ingest()
        XCTAssertEqual(second.processed, 0)
        XCTAssertEqual(second.skipped, 2)
        XCTAssertEqual(second.deduplicated, 0)
        XCTAssertEqual(engine.status()["sources"]?.arrayValue?.count, 1)
        engine.close()
    }

    func testManualVectorDedupeEmptiesDuplicateSourceAndSkipsNextRun() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-manual-dedupe-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try "# 甲\n\n这是甲文档独有的正文。".write(
            to: kb.appendingPathComponent("a.md"),
            atomically: true, encoding: .utf8)
        try "# 乙\n\n这是乙文档独有的正文。".write(
            to: kb.appendingPathComponent("b.md"),
            atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 2

        let engine = RAGEngine(config: config, knowledgeBase: kb.path,
                               embedding: RagFakeEmbedding(), reranker: nil)
        let first = await engine.ingest()
        XCTAssertEqual(first.processed, 2)
        engine.close()

        // 模拟历史遗留/外部写入：乙文档已被写入与甲文档完全相同的切片。
        let store = try VectorStore(dbPath: config.dbPath, dimension: 8,
                                    enableSQLiteVec: config.ragSQLiteVecEnabled)
        let sources = store.listSources()
        guard let sourceA = sources.first(where: { $0.hasSuffix("/a.md") }),
              let sourceB = sources.first(where: { $0.hasSuffix("/b.md") }) else {
            store.close()
            XCTFail("未找到 a.md/b.md 的切片来源")
            return
        }
        let rows = try store.db.query(
            "SELECT text FROM chunks WHERE source = ? ORDER BY created_at ASC, rowid ASC LIMIT 1",
            [.text(sourceA)])
        guard let textA = rows.first?["text"]?.stringValue else {
            store.close()
            XCTFail("甲文档没有切片")
            return
        }
        let vector = try await RagFakeEmbedding().embedDocuments([textA])[0]
        _ = try store.add(chunks: [Chunk(text: textA, source: sourceB, chunkIndex: 0)],
                          vectors: [vector])
        store.close()

        let preview = try engine.previewVectorDedupe()
        XCTAssertEqual(preview.totalChunks, 2)
        XCTAssertEqual(preview.duplicateChunks, 1)
        XCTAssertEqual(preview.affectedSources, 1)

        let stats = try engine.deduplicateVectors()
        XCTAssertEqual(stats.removedChunks, 1)
        XCTAssertEqual(stats.affectedSources, 1)
        XCTAssertEqual(stats.emptiedSources, 1)
        XCTAssertEqual(stats.remainingChunks, 1)
        XCTAssertEqual(engine.status()["sources"]?.arrayValue?.count, 1)
        XCTAssertEqual(engine.status()["source_count"]?.intValue, 1)
        XCTAssertEqual(engine.listDocuments().count, 1)

        // manifest 中乙文档写入 tombstone，指向保留来源。
        let manifestURL = URL(fileURLWithPath: config.ragIndexDirectory)
            .appendingPathComponent("manifest.json")
        let object = try JSONSerialization.jsonObject(
            with: Data(contentsOf: manifestURL)) as? [String: Any] ?? [:]
        let files = object["files"] as? [String: Any] ?? [:]
        let entryB = files[sourceB] as? [String: Any]
        XCTAssertEqual(entryB?["duplicateOf"] as? String, sourceA)
        XCTAssertEqual(entryB?["chunkCount"] as? Int, 0)

        // 下次同步直接跳过副本，不会重新写回重复向量。
        let second = await engine.ingest()
        XCTAssertEqual(second.processed, 0)
        XCTAssertEqual(second.skipped, 2)
        XCTAssertEqual(engine.status()["sources"]?.arrayValue?.count, 1)
        engine.close()
    }

    func testIncrementalReindexesFileWhenChunksDisappear() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-missing-chunks-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        for index in 1...2 {
            try """
            # 文档 \(index)

            这是用于验证切片缺失后增量同步会重建的第 \(index) 个文件。
            """.write(to: kb.appendingPathComponent("doc\(index).md"),
                      atomically: true, encoding: .utf8)
        }

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 2

        let first = RAGEngine(config: config, knowledgeBase: kb.path,
                              embedding: RagFakeEmbedding(), reranker: nil)
        let initial = await first.ingest()
        XCTAssertEqual(initial.processed, 2)
        first.close()

        // 模拟「manifest 说完成、库里却没有切片」：外部清理/上次写入中断。
        let store = try VectorStore(dbPath: config.dbPath, dimension: 8,
                                    enableSQLiteVec: config.ragSQLiteVecEnabled)
        guard let doc1Source = store.listSources().first(where: {
            $0.hasSuffix("/doc1.md")
        }) else {
            XCTFail("未找到 doc1.md 的切片来源")
            store.close()
            return
        }
        _ = try store.deleteBySource(doc1Source)
        store.close()

        let resumed = RAGEngine(config: config, knowledgeBase: kb.path,
                                embedding: RagFakeEmbedding(), reranker: nil)
        let stats = await resumed.ingest()
        XCTAssertEqual(stats.processed, 1, "切片缺失的文件必须重建")
        XCTAssertEqual(stats.skipped, 1)
        XCTAssertTrue(stats.errors.isEmpty, stats.errors.joined(separator: "; "))
        resumed.close()
    }

    func testFailedPagesAreRetriedOnNextIncrementalSync() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-partial-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        for index in 1...2 {
            try """
            # 文档 \(index)

            这是用于验证失败页重试的第 \(index) 个文件。
            """.write(to: kb.appendingPathComponent("doc\(index).md"),
                      atomically: true, encoding: .utf8)
        }

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 2

        let registry = DocumentParserRegistry(parsers: [PartialFailureParser()], config: config)
        let first = RAGEngine(config: config, knowledgeBase: kb.path,
                              embedding: RagFakeEmbedding(), reranker: nil,
                              parserRegistry: registry)
        let initial = await first.ingest()
        XCTAssertEqual(initial.processed, 2)
        first.close()

        let resumed = RAGEngine(config: config, knowledgeBase: kb.path,
                                embedding: RagFakeEmbedding(), reranker: nil,
                                parserRegistry: DocumentParserRegistry(
                                    parsers: [PartialFailureParser()], config: config))
        let stats = await resumed.ingest()
        XCTAssertEqual(stats.processed, 1, "有失败页的文件必须重试")
        XCTAssertEqual(stats.skipped, 1)
        resumed.close()
    }

    func testIncrementalReindexesLegacyManifestWithoutCompletionFields() async throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-legacy-manifest-\(UUID().uuidString)")
        let kb = root.appendingPathComponent("kb")
        try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        for index in 1...2 {
            try """
            # 文档 \(index)

            旧版 manifest 缺少完成标记时，增量同步应重新处理该文件。
            """.write(to: kb.appendingPathComponent("doc\(index).md"),
                      atomically: true, encoding: .utf8)
        }

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kb.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 2

        let first = RAGEngine(config: config, knowledgeBase: kb.path,
                              embedding: RagFakeEmbedding(), reranker: nil)
        let initial = await first.ingest()
        XCTAssertEqual(initial.processed, 2)
        first.close()

        let manifestURL = URL(fileURLWithPath: config.ragIndexDirectory)
            .appendingPathComponent("manifest.json")
        var object = try JSONSerialization.jsonObject(
            with: Data(contentsOf: manifestURL)) as? [String: Any] ?? [:]
        var files = object["files"] as? [String: Any] ?? [:]
        guard let legacyKey = files.keys.first(where: { $0.hasSuffix("/doc1.md") }) else {
            XCTFail("manifest 中未找到 doc1.md")
            return
        }
        var entry = files[legacyKey] as? [String: Any] ?? [:]
        entry.removeValue(forKey: "chunkCount")
        entry.removeValue(forKey: "complete")
        files[legacyKey] = entry
        object["files"] = files
        try JSONSerialization.data(withJSONObject: object).write(to: manifestURL)

        let resumed = RAGEngine(config: config, knowledgeBase: kb.path,
                                embedding: RagFakeEmbedding(), reranker: nil)
        let stats = await resumed.ingest()
        XCTAssertEqual(stats.processed, 1, "旧版 manifest 条目必须重建")
        XCTAssertEqual(stats.skipped, 1)
        resumed.close()
    }

    func testSwitchingKnowledgeBasePreservesEachIndex() async throws {
        // 回归：历史版本所有知识库共用 rag_data 根目录，切换知识库目录后
        // 同步会把对方条目当「已删除文件」清空，切回后只能全量重建。
        // 修复后索引按知识库目录隔离，切回后增量同步仍然有效。
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-kb-switch-\(UUID().uuidString)")
        let kbA = root.appendingPathComponent("kbA")
        let kbB = root.appendingPathComponent("kbB")
        try FileManager.default.createDirectory(at: kbA, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: kbB, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        try "# 甲\n\n甲库独有正文，用于验证切库后索引保留。".write(
            to: kbA.appendingPathComponent("a.md"), atomically: true, encoding: .utf8)
        try "# 乙\n\n乙库独有正文，用于验证切库后索引保留。".write(
            to: kbB.appendingPathComponent("b.md"), atomically: true, encoding: .utf8)

        var config = AgentConfig()
        config.workspace = root.path
        config.knowledgeBase = kbA.path
        config.ragRerankEnabled = false
        config.ragChunkSize = 40
        config.ragChunkOverlap = 2

        let dirA = config.ragIndexDirectory
        config.knowledgeBase = kbB.path
        let dirB = config.ragIndexDirectory
        XCTAssertNotEqual(dirA, dirB, "不同知识库目录必须使用不同的索引子目录")

        // 库 A 建索引 → 切到库 B 同步 → 切回库 A 必须增量跳过而不是全量重建。
        config.knowledgeBase = kbA.path
        let engineA = RAGEngine(config: config, knowledgeBase: kbA.path,
                                embedding: RagFakeEmbedding(), reranker: nil)
        let firstA = await engineA.ingest()
        XCTAssertEqual(firstA.processed, 1)
        engineA.close()

        config.knowledgeBase = kbB.path
        let engineB = RAGEngine(config: config, knowledgeBase: kbB.path,
                                embedding: RagFakeEmbedding(), reranker: nil)
        let firstB = await engineB.ingest()
        XCTAssertEqual(firstB.processed, 1)
        engineB.close()

        // 库 B 同步后，库 A 的 manifest/向量必须原样保留。
        let manifestA = try JSONSerialization.jsonObject(
            with: Data(contentsOf: URL(fileURLWithPath: dirA)
                .appendingPathComponent("manifest.json"))) as? [String: Any] ?? [:]
        XCTAssertEqual((manifestA["files"] as? [String: Any])?.count, 1,
                       "切库同步后库 A 的 manifest 不得被清空")

        config.knowledgeBase = kbA.path
        let backA = RAGEngine(config: config, knowledgeBase: kbA.path,
                              embedding: RagFakeEmbedding(), reranker: nil)
        let secondA = await backA.ingest()
        XCTAssertEqual(secondA.processed, 0, "切回库 A 后增量同步必须生效")
        XCTAssertEqual(secondA.skipped, 1)
        backA.close()
    }
}
