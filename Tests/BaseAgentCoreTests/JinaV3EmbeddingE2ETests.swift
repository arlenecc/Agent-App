import XCTest
import Foundation
import MLX
@testable import BaseAgentCore

/// 真实 jina-embeddings-v3 模型验证（MLX + ONNX + int8 / Matryoshka 256）。
///
/// 随普通 `swift test` 一起运行，保证切片 → embedding → 截断 → 归一化 →
/// int8 入库 → 查询的整条链路每次都被验证。每个用例会先检查对应模型：
/// 本机缺少 `models/jina-embeddings-v3-mlx-int4` 或
/// `models/jina-embeddings-v3-onnx-int4` 时，只跳过依赖该模型的用例。
final class JinaV3EmbeddingE2ETests: XCTestCase {
    private static let mlxModelName = "jina-embeddings-v3-mlx-int4"
    private static let onnxModelName = "jina-embeddings-v3-onnx-int4"

    private static let documents = [
        "Berlin is the capital and largest city of Germany.",
        "The Pacific Ocean is the largest ocean on Earth.",
    ]

    private static let queries: [(text: String, expected: Int)] = [
        ("What is the capital of Germany?", 0),
        ("Which is the largest ocean?", 1),
    ]

    /// Python 参考实现（`example/jina_v3_mlx.py`，float16 权重 + 相同前缀）的
    /// 首条 passage 向量前 8 维，用于确认 Swift 前向没有结构偏差。
    private static let passage0Reference: [Float] = [
        -0.08681989, -0.01761171, 0.02873153, 0.05860325,
        -0.04466227, 0.03609686, 0.01168237, -0.09117077,
    ]

    private var temporaryRoot: URL!

    override func setUpWithError() throws {
        EmbeddingBenchmarkTests.configureMetallib()
        temporaryRoot = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("jina-v3-e2e-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: temporaryRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let temporaryRoot { try? FileManager.default.removeItem(at: temporaryRoot) }
    }

    // MARK: - MLX

    func testMLXProviderProducesExpectedVectors() async throws {
        let directory = try modelDirectory(Self.mlxModelName, weights: "weights.safetensors")
        let provider = JinaV3MLXEmbeddingProvider(
            directory: directory, modelName: Self.mlxModelName)
        addTeardownBlock { await provider.unload() }

        XCTAssertEqual(provider.identifier, "local:mlx-embedding:\(Self.mlxModelName)")
        XCTAssertEqual(provider.dimension, 1024, "jina v3 原生维度")

        let documents = try await provider.embedDocuments(Self.documents)
        XCTAssertEqual(documents.count, Self.documents.count)
        for vector in documents {
            XCTAssertEqual(vector.count, 1024)
            XCTAssertEqual(norm(vector), 1, accuracy: 1e-4, "原生输出应已 L2 归一化")
        }

        // 与 Python 参考实现逐维对比（tokenizer / 量化路径完全一致时应近似相等）。
        let first8 = Array(documents[0].prefix(8))
        print("[jina-e2e] MLX passage first8 = \(first8)")
        for (actual, expected) in zip(first8, Self.passage0Reference) {
            XCTAssertEqual(actual, expected, accuracy: 0.02,
                           "Swift MLX 前向应与 example/jina_v3_mlx.py 一致")
        }

        let queries = try await queriesUsing(provider)
        for query in queries {
            XCTAssertEqual(query.vector.count, 1024)
            XCTAssertEqual(norm(query.vector), 1, accuracy: 1e-4)
        }

        assertSemanticRanking(documents: documents, queries: queries, label: "mlx")
    }

    // MARK: - ONNX

    func testONNXProviderUsesTaskIDAndMatchesMLX() async throws {
        let onnxDirectory = try modelDirectory(Self.onnxModelName, weights: "model.onnx")
        let mlxDirectory = try modelDirectory(Self.mlxModelName, weights: "weights.safetensors")

        let onnx = ONNXEmbeddingProvider(
            directory: onnxDirectory, modelName: Self.onnxModelName)
        let mlx = JinaV3MLXEmbeddingProvider(
            directory: mlxDirectory, modelName: Self.mlxModelName)
        addTeardownBlock {
            await onnx.unload()
            await mlx.unload()
        }

        XCTAssertEqual(onnx.identifier, "local:onnx:\(Self.onnxModelName)")
        XCTAssertEqual(onnx.dimension, 1024, "ONNX config 无 hidden_size，应按 jina 原生维处理")

        let onnxDocuments = try await onnx.embedDocuments(Self.documents)
        let onnxQueries = try await queriesUsing(onnx)
        for vector in onnxDocuments + onnxQueries.map(\.vector) {
            XCTAssertEqual(vector.count, 1024)
            XCTAssertEqual(norm(vector), 1, accuracy: 1e-4)
        }

        assertSemanticRanking(documents: onnxDocuments, queries: onnxQueries, label: "onnx")

        let mlxDocuments = try await mlx.embedDocuments(Self.documents)
        let mlxQueries = try await queriesUsing(mlx)
        for index in onnxDocuments.indices {
            XCTAssertGreaterThan(cosine(onnxDocuments[index], mlxDocuments[index]), 0.9,
                                 "ONNX 与 MLX 的文档向量应高度一致（int4 量化路径不同）")
        }
        for index in onnxQueries.indices {
            XCTAssertGreaterThan(
                cosine(onnxQueries[index].vector, mlxQueries[index].vector), 0.9,
                "ONNX 与 MLX 的查询向量应高度一致")
        }
    }

    // MARK: - 工厂接入

    func testFactoryBuildsJinaV3Providers() async throws {
        let modelsRoot = EmbeddingBenchmarkTests.resolveModelsRoot()
        _ = try modelDirectory(Self.mlxModelName, weights: "weights.safetensors")
        _ = try modelDirectory(Self.onnxModelName, weights: "model.onnx")
        setenv("BASE_AGENT_MODELS_DIR", modelsRoot.path, 1)
        defer {
            unsetenv("BASE_AGENT_MODELS_DIR")
            LocalModelFactory.shared.clearCaches()
        }

        var config = AgentConfig()
        config.localEmbeddingBackend = "mlx"
        config.localEmbeddingModel = Self.mlxModelName
        config.localEmbeddingModelPath = ""
        config.ragMatryoshkaDim = 256

        let mlx = try XCTUnwrap(
            LocalModelFactory.shared.makeEmbedding(config: config),
            "MLX jina v3 应能通过工厂按名称解析")
        XCTAssertTrue(mlx is JinaV3MLXEmbeddingProvider,
                      "jina-v3-mlx 不能落到 MLXEmbedders 默认工厂：\(mlx)")
        XCTAssertEqual(mlx.identifier, "local:mlx-embedding:\(Self.mlxModelName)")
        XCTAssertEqual(mlx.dimension, 1024)

        let wrapped = MatryoshkaEmbedding.wrap(
            mlx, mode: config.ragMatryoshka, dimension: config.ragMatryoshkaDim)
        let matryoshka = try XCTUnwrap(wrapped as? MatryoshkaEmbedding)
        XCTAssertEqual(matryoshka.truncateDimension, 256)
        let documents = try await wrapped.embedDocuments(["Berlin is in Germany."])
        XCTAssertEqual(documents[0].count, 256)
        XCTAssertEqual(norm(documents[0]), 1, accuracy: 1e-5)
        await mlx.unload()

        config.localEmbeddingBackend = "onnx"
        config.localEmbeddingModel = Self.onnxModelName
        let onnx = try XCTUnwrap(
            LocalModelFactory.shared.makeEmbedding(config: config),
            "ONNX jina v3 应能通过工厂按名称解析")
        XCTAssertTrue(onnx is ONNXEmbeddingProvider, "\(onnx)")
        XCTAssertEqual(onnx.identifier, "local:onnx:\(Self.onnxModelName)")
        XCTAssertEqual(onnx.dimension, 1024)
        let query = try await onnx.embedQuery("What is the capital of Germany?")
        XCTAssertEqual(query.count, 1024)
        XCTAssertEqual(norm(query), 1, accuracy: 1e-4)
        await onnx.unload()
    }

    // MARK: - 全链路：256 维 + 重新归一化 + int8 入库/检索

    func testIngestAndRecallWithJinaV3MatryoshkaInt8() async throws {
        let directory = try modelDirectory(Self.mlxModelName, weights: "weights.safetensors")
        let corpus: [(name: String, text: String)] = [
            ("cooking.md", """
            # 番茄炒蛋
            番茄炒蛋是一道常见的家常菜。先把鸡蛋打散，加少许盐，在热油中炒到凝固后盛出。
            锅里留底油，放入切块的番茄翻炒出汁，再把鸡蛋倒回锅中，加糖和盐调味。
            """),
            ("sqlite-vec.md", """
            # sqlite-vec
            sqlite-vec 是 SQLite 的向量检索扩展，支持 float32、int8 和 bit 三种向量列类型，
            可以在 SQL 中直接做最近邻查询。写入 int8 向量时必须用 vec_int8(?) 绑定。
            """),
            ("swift-concurrency.md", """
            # Swift 并发
            Swift 6 默认开启严格并发检查。actor 可以把可变状态隔离在单线程执行器上，
            跨 actor 访问必须通过 await；Sendable 约束保证值在并发域之间安全传递。
            """),
            ("mlx.md", """
            # MLX 框架
            MLX 是 Apple 为自家芯片设计的机器学习框架。它使用统一内存，
            数组运算可以在 CPU 与 GPU 之间共享，不必显式拷贝。
            """),
            // 超过 800 token 的长文：用于确认切片确实发生（chunks > 文件数），
            // 且检索仍能命中长文中的某一块。
            ("gardening.md", """
            # 多肉植物养护指南

            ## 光照
            多肉植物喜欢充足的散射光。春秋季节可以放在朝南的窗台，每天接受四到六小时日照；
            夏季正午需要遮阴，避免叶片被强光灼伤。室内养护时应定期转动花盆，防止植株向单侧徒长。

            ## 浇水
            浇水遵循干透浇透的原则。春秋生长季大约七到十天浇一次；夏季高温时植株进入半休眠，
            应减少到两到三周一次，并避开正午高温时段。冬季低于五摄氏度时要严格控水，
            保持盆土偏干，防止根系冻伤。

            ## 配土
            配土要保证透气排水。常用配方是颗粒土、泥炭和珍珠岩按六比三比一混合，
            颗粒可以选择赤玉土、火山岩或麦饭石。上盆前在盆底铺一层陶粒作为排水层，
            避免积水导致烂根。

            ## 通风
            通风是多肉安全度夏的关键。封闭阳台在闷热天气要开窗或使用小风扇促进空气流动，
            降低叶面和盆土湿度。发现叶片化水或茎部发黑时，应立即切除病部、消毒并更换干燥介质。

            ## 施肥
            多肉对养分需求不高。生长季每个月施一次稀释的多肉专用液肥即可，
            浓度控制在推荐值的一半；夏季高温和冬季低温停止施肥，避免肥害烧根。
            新上盆的植株缓苗两周后再开始施肥。

            ## 换盆
            根系长满花盆或土壤板结时需要换盆，通常在春秋两季进行。脱盆后轻轻抖掉旧土，
            剪除枯根和烂根，晾根一到两天再上盆。换盆后放在通风阴凉处缓苗，一周内不要浇透水。

            ## 繁殖
            叶插和枝插是最常见的繁殖方式。选择饱满健康的叶片，平放在微潮的颗粒土上，
            避免直射光，两三周后会生根发芽。枝插需要等切口晾干结痂后再插入土中，
            保持基质微潮即可。

            ## 病虫害
            常见害虫有蚧壳虫、红蜘蛛和蚜虫。少量发生时可以用棉签蘸酒精清除，
            严重时使用专用药剂并交替用药，避免产生抗药性。定期检查叶背和叶心，早发现早处理。

            ## 季节管理
            春季适合换盆和扦插，浇水逐步恢复；夏季重点遮阴、通风和控水；
            秋季温差增大，是多肉上色的好时机，可以适当增加光照；冬季搬入室内，
            保持五摄氏度以上并减少浇水。梅雨季节要特别注意防雨和防潮，避免长时间淋雨。

            ## 常见误区
            频繁浇水、长期放在阴暗角落、使用无排水孔的花盆，都会增加烂根风险。
            叶片发软不一定是缺水，也可能是根系受损，应先检查盆土和根部再决定是否补水。
            新买的多肉不要立刻换盆和暴晒，先在通风散射光处缓苗一周更稳妥。
            """),
        ]
        let recallCases: [(query: String, expected: String)] = [
            ("怎样用鸡蛋和番茄做一道家常菜？", "cooking.md"),
            ("哪个扩展能让数据库执行向量最近邻查询？", "sqlite-vec.md"),
            ("在 Swift 里怎样安全地在并发任务之间共享可变状态？", "swift-concurrency.md"),
            ("Apple 芯片上用什么库可以加载量化后的模型？", "mlx.md"),
            ("多肉植物夏天怎么浇水、要不要遮阴？", "gardening.md"),
        ]

        let knowledgeBase = temporaryRoot.appendingPathComponent("kb")
        try FileManager.default.createDirectory(
            at: knowledgeBase, withIntermediateDirectories: true)
        for document in corpus {
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
        config.ragMatryoshkaDim = 256
        XCTAssertEqual(config.ragVectorElementType, .int8)

        let provider = JinaV3MLXEmbeddingProvider(
            directory: directory, modelName: Self.mlxModelName)
        let engine = RAGEngine(config: config, knowledgeBase: knowledgeBase.path,
                               embedding: provider, reranker: nil)
        defer { engine.close() }
        addTeardownBlock { await provider.unload() }

        XCTAssertEqual(engine.effectiveMatryoshkaDimension, 256,
                       "jina v3 auto 必须固定截断到 256")
        XCTAssertTrue(engine.vectorIndexFingerprint.contains("dimension=256"),
                      engine.vectorIndexFingerprint)

        let stats = await engine.ingest()
        XCTAssertTrue(stats.errors.isEmpty, stats.errors.joined(separator: "; "))
        XCTAssertEqual(stats.processed, corpus.count)

        let status = engine.status(maxAge: 0)
        XCTAssertEqual(status["vector_element_type"]?.stringValue, "int8")
        XCTAssertEqual(status["matryoshka_dim"]?.intValue, 256)
        XCTAssertEqual(status["vector_rebuild_required"]?.boolValue, false)
        XCTAssertTrue(
            status["vector_index_fingerprint"]?.stringValue?.contains("dimension=256") == true,
            "入库向量维度必须是 256：\(status["vector_index_fingerprint"]?.stringValue ?? "-")")

        // MARK: 分阶段核对：切片数量 → 截断 → 归一化 → int8 字节 → 查询侧同变换
        let chunkCount = status["chunks_stored"]?.intValue ?? 0
        print("[jina-e2e] chunks=\(chunkCount) documents=\(corpus.count) "
            + "matryoshka=\(status["matryoshka_dim"]?.intValue ?? -1) "
            + "element=\(status["vector_element_type"]?.stringValue ?? "-")")
        XCTAssertGreaterThan(chunkCount, corpus.count,
                             "长文必须被拆成多块（chunks > 文件数），否则切片没有生效")

        let wrapped = MatryoshkaEmbedding.wrap(provider, mode: .auto, dimension: 256)
        let docVector = try await wrapped.embedDocuments([corpus[0].text])[0]
        let queryVector = try await wrapped.embedQuery(recallCases[0].query)
        XCTAssertEqual(docVector.count, 256, "切片向量应截断到 256 维")
        XCTAssertEqual(queryVector.count, 256, "查询向量应走同一截断")
        XCTAssertEqual(norm(docVector), 1, accuracy: 1e-5)
        XCTAssertEqual(norm(queryVector), 1, accuracy: 1e-5)

        let docBytes = VectorEncoder.encode(docVector, as: .int8)
        let queryBytes = VectorEncoder.encode(queryVector, as: .int8)
        XCTAssertEqual(docBytes.count, VectorElementType.int8.blobSize(dimension: 256))
        XCTAssertEqual(queryBytes.count, VectorElementType.int8.blobSize(dimension: 256))
        XCTAssertEqual(docBytes.count, 256, "256 维 int8 每行 256B")

        let decodedDoc = VectorEncoder.decode(docBytes, as: .int8, dimension: 256)
        let decodedQuery = VectorEncoder.decode(queryBytes, as: .int8, dimension: 256)
        let floatScore = cosine(queryVector, docVector)
        let int8Score = cosine(decodedQuery, decodedDoc)
        print("[jina-e2e] docDim=\(docVector.count) docNorm="
            + String(format: "%.5f", norm(docVector))
            + " queryDim=\(queryVector.count) queryNorm="
            + String(format: "%.5f", norm(queryVector))
            + " bytes=\(docBytes.count)/\(queryBytes.count)"
            + " floatCos=" + String(format: "%.4f", floatScore)
            + " int8Cos=" + String(format: "%.4f", int8Score))
        XCTAssertGreaterThan(cosine(decodedDoc, docVector), 0.99,
                             "int8 量化应保留切片向量方向")
        XCTAssertGreaterThan(cosine(decodedQuery, queryVector), 0.99,
                             "int8 量化应保留查询向量方向")
        XCTAssertEqual(int8Score, floatScore, accuracy: 0.01,
                       "int8 与 fp32 的检索打分应基本一致")

        var top1 = 0
        for item in recallCases {
            let hits = try await engine.search(query: item.query, topK: 3)
            let sources = hits.map { ($0.source as NSString).lastPathComponent }
            let matched = hits.first {
                ($0.source as NSString).lastPathComponent == item.expected
            }
            let rank = sources.firstIndex(of: item.expected).map { $0 + 1 } ?? Int.max
            if rank == 1 { top1 += 1 }
            print("[jina-e2e] rank=\(rank == Int.max ? "-" : "\(rank)") "
                + "vectorScore=\(matched?.vectorScore.map { String(format: "%.4f", $0) } ?? "-") "
                + "sources=\(sources) query=\(item.query)")
            if rank <= 3 {
                XCTAssertNotNil(matched?.vectorScore, "命中应由向量通道召回")
            }
        }
        XCTAssertEqual(top1, recallCases.count, "256 维 int8 检索应保持 top1 命中")
    }

    // MARK: - 工具

    private func modelDirectory(_ name: String, weights: String) throws -> URL {
        let directory = EmbeddingBenchmarkTests.resolveModelsRoot()
            .appendingPathComponent(name)
        guard FileManager.default.fileExists(
            atPath: directory.appendingPathComponent(weights).path) else {
            throw XCTSkip("模型缺失：\(directory.path)")
        }
        return directory
    }

    private func queriesUsing(
        _ provider: EmbeddingProvider
    ) async throws -> [(text: String, vector: [Float])] {
        var result: [(String, [Float])] = []
        for item in Self.queries {
            result.append((item.text, try await provider.embedQuery(item.text)))
        }
        return result
    }

    private func assertSemanticRanking(
        documents: [[Float]],
        queries: [(text: String, vector: [Float])],
        label: String
    ) {
        for query in queries {
            let scores = documents.map { cosine(query.vector, $0) }
            print("[jina-e2e][\(label)] scores=\(scores.map { String(format: "%.3f", $0) }) "
                + "query=\(query.text)")
            for index in Self.queries.indices where Self.queries[index].text == query.text {
                let expected = Self.queries[index].expected
                XCTAssertGreaterThan(scores[expected], 0.6,
                                     "\(label) 期望文档的余弦应明显偏高")
                for other in scores.indices where other != expected {
                    XCTAssertLessThan(scores[other], 0.35,
                                      "\(label) 非期望文档的余弦应明显偏低")
                }
            }
        }
    }

    private func cosine(_ a: [Float], _ b: [Float]) -> Float {
        guard a.count == b.count, !a.isEmpty else { return 0 }
        var dot: Float = 0
        var normA: Float = 0
        var normB: Float = 0
        for index in a.indices {
            dot += a[index] * b[index]
            normA += a[index] * a[index]
            normB += b[index] * b[index]
        }
        let denominator = normA.squareRoot() * normB.squareRoot()
        return denominator > 0 ? dot / denominator : 0
    }

    private func norm(_ vector: [Float]) -> Float {
        vector.reduce(0) { $0 + $1 * $1 }.squareRoot()
    }
}
