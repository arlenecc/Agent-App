import XCTest
import Foundation
import MLX
@testable import BaseAgentCore

/// 真实 nomic-int8 模型的端到端验证：入库 → int8 + Matryoshka 检索，
/// 并与 fp32 全维（768）基线对比召回。
///
/// 随普通 `swift test` 一起运行；本机缺少
/// `models/nomic-embed-text-v1.5-mlx-int8` 时自动跳过。
final class RAGEngineNomicInt8E2ETests: XCTestCase {
    private struct RecallCase {
        let query: String
        /// 期望命中的文件名。
        let expected: String
    }

    /// 语义上彼此区分的短文档，每篇不超过一个 800 token 切片。
    private static let documents: [(name: String, text: String)] = [
        ("cooking.md", """
        # 番茄炒蛋

        番茄炒蛋是一道常见的家常菜。先把鸡蛋打散，加少许盐，在热油中炒到凝固后盛出。
        锅里留底油，放入切块的番茄翻炒出汁，再把鸡蛋倒回锅中，加糖和盐调味，翻匀即可出锅。
        """),
        ("sqlite-vec.md", """
        # sqlite-vec

        sqlite-vec 是 SQLite 的向量检索扩展。它支持 float32、int8 和 bit 三种向量列类型，
        可以在 SQL 中直接做最近邻查询。写入 int8 向量时必须用 vec_int8(?) 绑定，
        bit 向量用 vec_bit(?)，否则 SQLite 会把裸 BLOB 当成 float32 解析。
        """),
        ("swift-concurrency.md", """
        # Swift 并发

        Swift 6 默认开启严格并发检查。actor 可以把可变状态隔离在单线程执行器上，
        跨 actor 访问必须通过 await。Sendable 约束用来保证值在并发域之间传递时是安全的；
        如果类型内部有可变引用，可以用 @unchecked Sendable 自行保证线程安全。
        """),
        ("nomic.md", """
        # nomic-embed-text

        nomic-embed-text-v1.5 是一个开源文本嵌入模型，原生输出 768 维向量，
        支持 Matryoshka 表示学习。使用俄罗斯套娃特性时，可以直接截取前 512 维，
        但截断会破坏 L2 范数，必须重新计算范数并归一化，否则余弦相似度会失真。
        """),
        ("git-undo.md", """
        # 撤销本地提交

        如果提交还没有推送到远端，可以用 git reset --soft HEAD~1 把改动退回暂存区，
        或者用 git reset --mixed 退回工作区。已经推送的提交不要用 reset 改写历史，
        应该使用 git revert 生成一个反向提交，这样不会影响其他人的分支。
        """),
        ("mlx.md", """
        # MLX 框架

        MLX 是 Apple 为自家芯片设计的机器学习框架。它使用统一内存，
        数组运算可以在 CPU 与 GPU 之间共享，不必显式拷贝。
        通过 mlx-swift 可以在 macOS 应用里直接加载量化后的大语言模型和嵌入模型。
        """),
    ]

    /// 查询刻意避开原文的独有关键词，尽量只靠语义召回。
    private static let recallCases: [RecallCase] = [
        RecallCase(query: "怎样用鸡蛋和番茄做一道家常菜？", expected: "cooking.md"),
        RecallCase(query: "哪个扩展能让数据库执行向量最近邻查询？", expected: "sqlite-vec.md"),
        RecallCase(query: "在 Swift 里怎样安全地在并发任务之间共享可变状态？",
                   expected: "swift-concurrency.md"),
        RecallCase(query: "哪种开源文本嵌入模型支持俄罗斯套娃式的维度截断？",
                   expected: "nomic.md"),
        RecallCase(query: "怎样撤销还没有推送的本地提交记录？", expected: "git-undo.md"),
        RecallCase(query: "Apple 芯片上用什么库可以加载量化后的大语言模型？", expected: "mlx.md"),
    ]

    private var root: URL!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-rag-e2e-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let root { try? FileManager.default.removeItem(at: root) }
    }

    func testIngestAndRecallWithNomicInt8Matryoshka() async throws {
        let modelsRoot = EmbeddingBenchmarkTests.resolveModelsRoot()
        let modelDir = modelsRoot.appendingPathComponent("nomic-embed-text-v1.5-mlx-int8")
        guard FileManager.default.fileExists(
            atPath: modelDir.appendingPathComponent("model.safetensors").path) else {
            throw XCTSkip("模型缺失：\(modelDir.path)")
        }
        EmbeddingBenchmarkTests.configureMetallib()
        // swift test 的 Bundle.main 不在仓库内，用注册表的覆盖机制锚定模型根目录。
        setenv("BASE_AGENT_MODELS_DIR", modelsRoot.path, 1)
        defer { unsetenv("BASE_AGENT_MODELS_DIR") }

        let kbInt8 = root.appendingPathComponent("kb-int8")
        let kbFull = root.appendingPathComponent("kb-fp32")
        for kb in [kbInt8, kbFull] {
            try FileManager.default.createDirectory(at: kb, withIntermediateDirectories: true)
            for document in Self.documents {
                try document.text.write(to: kb.appendingPathComponent(document.name),
                                        atomically: true, encoding: .utf8)
            }
        }

        // nomic 基准配置（生产默认已切到 jina-v3 / 256，这里显式固定
        // nomic-int8 / MLX / int8 / auto / 512，保留原有对比基线）。
        var int8Config = AgentConfig()
        int8Config.workspace = root.path
        int8Config.knowledgeBase = kbInt8.path
        int8Config.ragRerankEnabled = false
        int8Config.localEmbeddingBackend = "mlx"
        int8Config.localEmbeddingModel = "nomic-embed-text-v1.5-mlx-int8"
        int8Config.localEmbeddingModelPath = "models/nomic-embed-text-v1.5-mlx-int8"
        int8Config.ragEmbeddingModel = "nomic-embed-text-v1.5-mlx-int8"
        int8Config.ragMatryoshkaDim = 512
        XCTAssertEqual(int8Config.ragVectorElementType, .int8)
        XCTAssertEqual(int8Config.ragMatryoshka, .auto)
        XCTAssertEqual(int8Config.ragMatryoshkaDim, 512)
        XCTAssertEqual(int8Config.ragChunkSize, 800)
        XCTAssertEqual(int8Config.ragChunkOverlap, 80)

        let provider = try XCTUnwrap(
            LocalModelFactory.shared.makeEmbedding(config: int8Config),
            "无法加载 \(int8Config.localEmbeddingModelPath)")
        XCTAssertTrue(provider.identifier.contains("nomic"), provider.identifier)
        XCTAssertEqual(provider.dimension, 768, "nomic 原生维度")
        addTeardownBlock { await provider.unload() }

        // 两个引擎共用同一个底层 provider；Matryoshka 包装各自独立。
        let int8Engine = RAGEngine(config: int8Config, knowledgeBase: kbInt8.path,
                                   embedding: provider, reranker: nil)
        var fp32Config = int8Config
        fp32Config.knowledgeBase = kbFull.path
        fp32Config.ragVectorElementType = .fp32
        fp32Config.ragMatryoshka = .off
        let fp32Engine = RAGEngine(config: fp32Config, knowledgeBase: kbFull.path,
                                   embedding: provider, reranker: nil)
        defer {
            int8Engine.close()
            fp32Engine.close()
        }

        XCTAssertEqual(int8Engine.effectiveMatryoshkaDimension, 512)
        XCTAssertEqual(fp32Engine.effectiveMatryoshkaDimension, 0)

        // ---------- 入库 ----------
        let int8Stats = await int8Engine.ingest()
        XCTAssertTrue(int8Stats.errors.isEmpty,
                      int8Stats.errors.joined(separator: "; "))
        XCTAssertEqual(int8Stats.processed, Self.documents.count)
        XCTAssertGreaterThanOrEqual(int8Stats.chunks, Self.documents.count)

        let int8Status = int8Engine.status(maxAge: 0)
        XCTAssertEqual(int8Status["chunks_stored"]?.intValue, int8Stats.chunks)
        XCTAssertEqual(int8Status["source_count"]?.intValue, Self.documents.count)
        XCTAssertEqual(int8Status["vector_element_type"]?.stringValue, "int8")
        XCTAssertEqual(int8Status["matryoshka_dim"]?.intValue, 512)
        XCTAssertEqual(int8Status["vector_rebuild_required"]?.boolValue, false)
        XCTAssertEqual(int8Status["vector_backend"]?.stringValue, "sqlite-vec")

        let fp32Stats = await fp32Engine.ingest()
        XCTAssertTrue(fp32Stats.errors.isEmpty,
                      fp32Stats.errors.joined(separator: "; "))
        XCTAssertEqual(fp32Stats.processed, Self.documents.count)
        XCTAssertEqual(fp32Engine.status(maxAge: 0)["vector_element_type"]?.stringValue, "fp32")

        // ---------- 召回 ----------
        // 关键词查询走 FTS，确认全文通道也正常。
        let ftsHits = try await int8Engine.search(query: "番茄炒蛋", topK: 3)
        let ftsSource = (ftsHits.first?.source as NSString?)?.lastPathComponent
        XCTAssertEqual(ftsSource, "cooking.md")

        let int8Recall = try await evaluate(int8Engine, label: "int8+matryoshka512")
        let fp32Recall = try await evaluate(fp32Engine, label: "fp32+full768")

        print("[RAG-E2E] int8 入库：\(int8Stats.summary)；fp32 入库：\(fp32Stats.summary)")
        print("[RAG-E2E] int8+512  Recall@1=\(int8Recall.top1)/\(Self.recallCases.count) "
            + "Recall@3=\(int8Recall.recall3)/\(Self.recallCases.count)")
        print("[RAG-E2E] fp32+768  Recall@1=\(fp32Recall.top1)/\(Self.recallCases.count) "
            + "Recall@3=\(fp32Recall.recall3)/\(Self.recallCases.count)")

        XCTAssertEqual(int8Recall.recall3, Self.recallCases.count,
                       "int8 + Matryoshka 512 的语义召回不应漏掉任何期望文档")
        XCTAssertEqual(fp32Recall.recall3, Self.recallCases.count,
                       "fp32 全维基线召回不应漏掉任何期望文档")
        XCTAssertGreaterThanOrEqual(int8Recall.top1, Self.recallCases.count - 1,
                                    "int8 截断量化后 top1 命中不应明显下降")
        XCTAssertGreaterThanOrEqual(int8Recall.top1, fp32Recall.top1 - 1,
                                    "int8 + Matryoshka 不应比 fp32 全维基线差")
    }

    /// 逐条查询并记录排名；命中必须来自向量通道（vectorScore 非空）。
    private func evaluate(_ engine: RAGEngine,
                          label: String) async throws -> (top1: Int, recall3: Int) {
        var top1 = 0
        var recall3 = 0
        for item in Self.recallCases {
            let hits = try await engine.search(query: item.query, topK: 3)
            let sources = hits.map { ($0.source as NSString).lastPathComponent }
            let matched = hits.first {
                ($0.source as NSString).lastPathComponent == item.expected
            }
            let rank = sources.firstIndex(of: item.expected).map { $0 + 1 } ?? Int.max
            if rank == 1 { top1 += 1 }
            if rank <= 3 { recall3 += 1 }
            print("[RAG-E2E][\(label)] rank=\(rank == Int.max ? "-" : "\(rank)") "
                + "vectorScore=\(matched?.vectorScore.map { String(format: "%.4f", $0) } ?? "-") "
                + "sources=\(sources) query=\(item.query)")
            if rank <= 3 {
                XCTAssertNotNil(matched?.vectorScore,
                                "命中 \(item.expected) 应由向量通道召回")
            }
        }
        return (top1, recall3)
    }
}
