import XCTest
import Foundation
@testable import BaseAgentCore

/// jina-v5 MLX 的 RAG 全链路：128 维 + int8 入库 → 查询 → 检索排序。
///
/// 与 v3 的 256 维 Matryoshka 路径不同，V5 provider 已经在内部完成截断与
/// 重新归一化，Matryoshka 层不再参与；这里验证最终落库维度与检索效果。
final class JinaV5RAGIngestE2ETests: XCTestCase {
    private static let modelName = "jina-embeddings-v5-text-nano-mlx-int4"

    /// 每篇正文需超过 `RAGEngine.minBodyCharacters`（100 字），
    /// 否则会被「正文过短」闸门整份丢弃，无法进入切片与向量化。
    private static let corpus: [(name: String, text: String)] = [
        ("phone.md", """
        苹果公司发布了新款智能手机，搭载新一代 A19 芯片，夜景拍摄能力明显提升。
        手机支持全天候显示与更快的充电协议，有线充电功率提升到四十瓦，无线充电也更加稳定。
        相机新增了长焦镜头，低光环境下噪点更少，人像模式的边缘处理更加自然。
        起售价格与上一代保持一致，并提供多种颜色和存储版本供用户选择。
        """),
        ("machine-learning.md", """
        机器学习通过数据训练模型，让计算机自动改进任务表现。
        常见方法包括监督学习、无监督学习与强化学习；监督学习依赖带标签样本，
        无监督学习常用于聚类和降维，强化学习则通过奖励信号调整策略。
        训练时通常需要划分训练集、验证集和独立测试集，避免模型过拟合。
        评估分类任务可以使用准确率、精确率和召回率，回归任务常用均方误差。
        """),
        ("cooking.md", """
        红烧肉的关键在于煸炒糖色和慢火炖煮。五花肉切块后先焯水，捞出沥干，
        再用冰糖炒出焦糖色，放入肉块翻炒上色。
        加入酱油、料酒和少量热水，小火炖煮约一小时，期间注意翻动，最后大火收汁。
        成菜色泽红亮，肥而不腻，配米饭或青菜都很合适。
        """),
    ]

    private static let recallCases: [(query: String, expected: String)] = [
        ("苹果新手机拍照和充电有什么升级？", "phone.md"),
        ("机器学习是怎样训练模型的？", "machine-learning.md"),
        ("红烧肉怎么做才好吃？", "cooking.md"),
    ]

    private var temporaryRoot: URL!

    override func setUpWithError() throws {
        EmbeddingBenchmarkTests.configureMetallib()
        temporaryRoot = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("jina-v5-rag-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: temporaryRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let temporaryRoot { try? FileManager.default.removeItem(at: temporaryRoot) }
    }

    func testIngestAndRecallWithJinaV5Int8() async throws {
        let directory = EmbeddingBenchmarkTests.resolveModelsRoot()
            .appendingPathComponent(Self.modelName)
        try XCTSkipUnless(
            JinaV5Profile.isJinaV5(directory: directory)
                && JinaV5Profile.weightsURL(in: directory) != nil,
            "模型缺失：\(directory.path)")

        let knowledgeBase = temporaryRoot.appendingPathComponent("kb")
        try FileManager.default.createDirectory(
            at: knowledgeBase, withIntermediateDirectories: true)
        for document in Self.corpus {
            try document.text.write(
                to: knowledgeBase.appendingPathComponent(document.name),
                atomically: true, encoding: .utf8)
        }

        var config = AgentConfig()
        config.workspace = temporaryRoot.path
        config.knowledgeBase = knowledgeBase.path
        config.ragRerankEnabled = false
        config.ragANNEnabled = false
        config.ragVectorElementType = .int8
        config.ragMatryoshka = .auto
        config.ragMatryoshkaDim = 128

        let provider = JinaV5MLXEmbeddingProvider(
            directory: directory, modelName: Self.modelName)
        let engine = RAGEngine(config: config, knowledgeBase: knowledgeBase.path,
                               embedding: provider, reranker: nil)
        defer { engine.close() }
        addTeardownBlock { await provider.unload() }

        XCTAssertEqual(engine.effectiveMatryoshkaDimension, 0,
                       "V5 provider 自带截断，Matryoshka 不应再包装")
        XCTAssertTrue(engine.vectorIndexFingerprint.contains("dimension=128"),
                      engine.vectorIndexFingerprint)
        XCTAssertTrue(engine.vectorIndexFingerprint.contains("element=int8"),
                      engine.vectorIndexFingerprint)

        let stats = await engine.ingest()
        XCTAssertTrue(stats.errors.isEmpty, stats.errors.joined(separator: "; "))
        XCTAssertEqual(stats.processed, Self.corpus.count)

        let status = engine.status(maxAge: 0)
        XCTAssertEqual(status["vector_element_type"]?.stringValue, "int8")
        XCTAssertEqual(status["vector_rebuild_required"]?.boolValue, false)
        XCTAssertTrue(
            status["vector_index_fingerprint"]?.stringValue?.contains("dimension=128") == true,
            "入库向量维度必须是 128：\(status["vector_index_fingerprint"]?.stringValue ?? "-")")

        // 落库 meta：维度 128 / int8；int8 blob 每行正好 128B。
        let db = try SQLiteDatabase(path: engine.indexDBPath, readOnly: true)
        defer { db.close() }
        let meta = try db.query(
            "SELECT key, value FROM meta WHERE key IN ('embedding_dim', 'vector_element_type')")
        let values = Dictionary(uniqueKeysWithValues: meta.compactMap { row in
            row["key"]?.stringValue.flatMap { key in
                row["value"]?.stringValue.map { (key, $0) }
            }
        })
        XCTAssertEqual(values["embedding_dim"], "128")
        XCTAssertEqual(values["vector_element_type"], "int8")

        // 查询侧同一变换：128 维单位向量，int8 编码后 128 字节。
        let probe = try await provider.embedQuery(Self.recallCases[0].query)
        XCTAssertEqual(probe.count, 128)
        XCTAssertEqual(norm(probe), 1, accuracy: 1e-4)
        XCTAssertEqual(VectorEncoder.encode(probe, as: .int8).count, 128,
                       "128 维 int8 每行 128B")

        var top1 = 0
        for item in Self.recallCases {
            let hits = try await engine.search(query: item.query, topK: 3)
            let sources = hits.map { ($0.source as NSString).lastPathComponent }
            let rank = sources.firstIndex(of: item.expected).map { $0 + 1 } ?? Int.max
            if rank == 1 { top1 += 1 }
            print("[jina-v5-rag] rank=\(rank == Int.max ? "-" : "\(rank)") "
                + "sources=\(sources) query=\(item.query)")
            XCTAssertLessThanOrEqual(rank, 3, "相关文档必须进入 top3")
        }
        XCTAssertEqual(top1, Self.recallCases.count, "128 维 int8 检索应保持 top1 命中")

        // 工具入口：rag_search 也能返回带来源的结果。
        let tool = RagSearchTool(config: config, engine: engine)
        let result = try await tool.run(["query": .string(Self.recallCases[2].query)])
        XCTAssertTrue(result.success, "rag_search 应成功返回：\(result.output)")
        XCTAssertTrue(result.output.contains("cooking.md"),
                      "rag_search 输出应包含来源文件名：\(result.output)")
    }

    // MARK: - 工具

    private func norm(_ vector: [Float]) -> Float {
        vector.reduce(0) { $0 + $1 * $1 }.squareRoot()
    }
}
