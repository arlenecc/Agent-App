import XCTest
import Foundation
@testable import BaseAgentCore

/// jina-embeddings-v3 的模型识别、task_id 映射与 Matryoshka 规则。
final class JinaV3SupportTests: XCTestCase {

    // MARK: - 模型识别

    func testMatchesJinaV3ModelNames() {
        XCTAssertTrue(JinaV3Profile.matches("jina-embeddings-v3-mlx-int4"))
        XCTAssertTrue(JinaV3Profile.matches("jina-embeddings-v3-onnx-int4"))
        XCTAssertTrue(JinaV3Profile.matches("jina_v3_mlx"))
        XCTAssertTrue(JinaV3Profile.matches("jina-v3-mlx"))
        XCTAssertFalse(JinaV3Profile.matches("jina-embeddings-v2-base-en"))
        XCTAssertFalse(JinaV3Profile.matches("nomic-embed-text-v1.5"))
    }

    func testDetectsJinaV3FromConfigModelType() throws {
        let directory = try makeTemporaryModelDirectory(
            name: "custom-embedding",
            config: ["model_type": "jina-v3-mlx"])
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertTrue(JinaV3Profile.isJinaV3(directory: directory))
    }

    func testDetectsJinaV3FromTaskIDMap() throws {
        let directory = try makeTemporaryModelDirectory(
            name: "custom-embedding",
            config: ["task_id_map": ["retrieval.query": 0, "retrieval.passage": 1]])
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertTrue(JinaV3Profile.isJinaV3(directory: directory),
                      "ONNX config 没有 hidden_size，但有 task_id_map")
    }

    func testRejectsUnrelatedModel() throws {
        let directory = try makeTemporaryModelDirectory(
            name: "bge-small-zh-v1.5",
            config: ["model_type": "bert"])
        defer { try? FileManager.default.removeItem(at: directory) }

        XCTAssertFalse(JinaV3Profile.isJinaV3(directory: directory))
    }

    // MARK: - task_id

    func testTaskIDsMatchUnifiedWeightsAndONNXMap() {
        XCTAssertEqual(JinaV3Task.id(for: .retrievalQuery), 0)
        XCTAssertEqual(JinaV3Task.id(for: .retrievalDocument), 1,
                       "文档切片必须使用 retrieval.passage (task_id=1)")
        XCTAssertEqual(JinaV3Task.id(for: .clustering), 2)
        XCTAssertEqual(JinaV3Task.id(for: .classification), 3)
    }

    // MARK: - provider 前缀接线

    /// 模型目录被改名后仍必须按 jina 前缀处理：provider 已经从 config 认出
    /// 是 jina-v3，不能再依赖目录名里是否含 "jina"。
    func testMLXProviderUsesJinaPrefixesForRenamedDirectory() throws {
        let directory = try makeTemporaryModelDirectory(
            name: "custom-embedding",
            config: ["model_type": "jina-v3-mlx"])
        defer { try? FileManager.default.removeItem(at: directory) }

        let provider = JinaV3MLXEmbeddingProvider(
            directory: directory, modelName: "custom-embedding")
        XCTAssertEqual(provider.prefixStrategy.family, .jina)
        XCTAssertEqual(provider.prefixStrategy.documentPrefixTokenReserve, 4,
                       "jina 的 passage: 前缀需要预留 4 token")
    }

    func testONNXProviderUsesJinaPrefixesForRenamedDirectory() throws {
        let directory = try makeTemporaryModelDirectory(
            name: "custom-embedding",
            config: ["task_id_map": ["retrieval.query": 0, "retrieval.passage": 1]])
        defer { try? FileManager.default.removeItem(at: directory) }

        let provider = ONNXEmbeddingProvider(
            directory: directory, modelName: "custom-embedding")
        XCTAssertEqual(provider.prefixStrategy.family, .jina)
    }

    func testONNXProviderKeepsGenericPrefixForUnrelatedModel() throws {
        let directory = try makeTemporaryModelDirectory(
            name: "custom-embedding",
            config: ["model_type": "bert"])
        defer { try? FileManager.default.removeItem(at: directory) }

        let provider = ONNXEmbeddingProvider(
            directory: directory, modelName: "custom-embedding")
        XCTAssertEqual(provider.prefixStrategy.family, .generic,
                       "非 jina 模型不能被强行套上 passage/query 前缀")
    }

    // MARK: - Matryoshka

    func testAutoMatryoshkaUsesConfiguredDimensionForJinaV3() {
        let identifier = "local:mlx-embedding:jina-embeddings-v3-mlx-int4"
        let mode = MatryoshkaMode.auto

        XCTAssertTrue(mode.enabled(for: identifier))
        XCTAssertEqual(mode.effectiveDimension(for: identifier, configured: 512), 512)
        XCTAssertEqual(mode.effectiveDimension(for: identifier, configured: 256), 256)
        XCTAssertEqual(mode.effectiveDimension(for: identifier, configured: 128), 128,
                       "auto 模式必须尊重下拉框所选维度")
        XCTAssertEqual(mode.effectiveDimension(
            for: "local:onnx:jina-embeddings-v3-onnx-int4", configured: 512), 512)
        XCTAssertEqual(mode.effectiveDimension(
            for: "local:mlx-embedding:nomic-embed-text-v1.5-mlx-int8",
            configured: 512), 512)
    }

    func testWrapJinaV3ProducesNormalized256DimensionVectors() async throws {
        let base = JinaV3MatryoshkaStub(
            identifier: "local:mlx-embedding:jina-embeddings-v3-mlx-int4",
            dimension: 1024)
        let wrapped = MatryoshkaEmbedding.wrap(base, mode: .auto, dimension: 256)

        let matryoshka = try XCTUnwrap(wrapped as? MatryoshkaEmbedding)
        XCTAssertEqual(matryoshka.truncateDimension, 256)
        XCTAssertEqual(wrapped.dimension, 256)

        let document = try await wrapped.embedDocuments(["正文"])[0]
        let query = try await wrapped.embedQuery("查询")
        XCTAssertEqual(document.count, 256)
        XCTAssertEqual(query.count, 256)
        XCTAssertEqual(document, query, "入库与查询必须共用同一套截断 + 归一化")
        XCTAssertEqual(norm(document), 1, accuracy: 1e-6,
                       "截断后必须重新 L2 归一化")
    }

    func testWrapJinaV3UsesSelectedDimensionWhenAuto() async throws {
        let base = JinaV3MatryoshkaStub(
            identifier: "local:mlx-embedding:jina-embeddings-v3-mlx-int4",
            dimension: 1024)
        let wrapped = MatryoshkaEmbedding.wrap(base, mode: .auto, dimension: 512)

        let matryoshka = try XCTUnwrap(wrapped as? MatryoshkaEmbedding)
        XCTAssertEqual(matryoshka.truncateDimension, 512)
        XCTAssertEqual(wrapped.dimension, 512)

        let document = try await wrapped.embedDocuments(["正文"])[0]
        let query = try await wrapped.embedQuery("查询")
        XCTAssertEqual(document.count, 512)
        XCTAssertEqual(query.count, 512)
        XCTAssertEqual(document, query)
        XCTAssertEqual(norm(document), 1, accuracy: 1e-6)
    }

    // MARK: - config 解析

    func testLoadsMixedQuantizationConfiguration() throws {
        let directory = try makeTemporaryModelDirectory(
            name: "jina-embeddings-v3-mlx-int4",
            config: [
                "hidden_size": 1024,
                "num_hidden_layers": 24,
                "num_attention_heads": 16,
                "intermediate_size": 4096,
                "layer_norm_eps": 1e-5,
                "vocab_size": 250_002,
                "type_vocab_size": 1,
                "rotary_emb_base": 20_000,
                "lora_rank": 4,
                "lora_alpha": 1,
                "unified": true,
                "model_type": "jina-v3-mlx",
                "quantization": [
                    "group_size": 32,
                    "bits": 4,
                    "mode": "affine",
                    "mixed": ["patterns": ["mixer"], "bits": 8],
                ],
            ])
        defer { try? FileManager.default.removeItem(at: directory) }

        let configuration = try JinaV3ModelConfiguration.load(from: directory)
        XCTAssertEqual(configuration.hiddenSize, 1024)
        XCTAssertEqual(configuration.layerCount, 24)
        XCTAssertEqual(configuration.headDimension, 64)
        XCTAssertEqual(configuration.groupSize, 32)
        XCTAssertEqual(configuration.bits, 4)
        XCTAssertEqual(configuration.bits(for: "layers.0.mixer.Wqkv"), 8)
        XCTAssertEqual(configuration.bits(for: "layers.0.mlp.fc1"), 4)
        XCTAssertEqual(configuration.bits(for: "word_embeddings"), 4)
        XCTAssertEqual(configuration.loraScaling, 0.25, accuracy: 1e-7)
    }

    // MARK: - 工具

    private func makeTemporaryModelDirectory(
        name: String, config: [String: Any]
    ) throws -> URL {
        let directory = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("jina-v3-\(UUID().uuidString)")
            .appendingPathComponent(name)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let data = try JSONSerialization.data(
            withJSONObject: config, options: [.sortedKeys])
        try data.write(to: directory.appendingPathComponent("config.json"))
        return directory
    }

    private func norm(_ vector: [Float]) -> Float {
        vector.reduce(0) { $0 + $1 * $1 }.squareRoot()
    }
}

/// 1024 维底层向量；截断到 256 后必须重新归一化。
private final class JinaV3MatryoshkaStub: EmbeddingProvider, @unchecked Sendable {
    let identifier: String
    let dimension: Int

    init(identifier: String, dimension: Int) {
        self.identifier = identifier
        self.dimension = dimension
    }

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        texts.map { _ in source }
    }

    func embedQuery(_ text: String) async throws -> [Float] { source }

    private var source: [Float] {
        var vector = [Float](repeating: 0, count: dimension)
        vector[0] = 3
        vector[1] = 4
        vector[2] = 5
        vector[3] = 6
        return vector
    }
}
