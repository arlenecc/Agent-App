import XCTest
@testable import BaseAgentCore

/// Embedding 推理的输入上限：jina-v3 的模型窗口是 8192 token，但实际切片
/// 默认只有 800 token；计算前必须统一压到 1024 token，避免为用不到的窗口
/// 容量做无效计算，同时让切片收口与实际推理截断保持同一口径。
final class EmbeddingInputLimitTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("base-agent-embedding-limit-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let root { try? FileManager.default.removeItem(at: root) }
    }

    /// 建一个只含 config.json 的模型目录：provider 初始化不加载权重即可读取窗口。
    private func makeModelDirectory(
        window: Int,
        name: String = "jina-embeddings-v3-test"
    ) throws -> URL {
        let directory = root.appendingPathComponent(name)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try #"{"max_position_embeddings": \#(window)}"#
            .write(to: directory.appendingPathComponent("config.json"),
                   atomically: true, encoding: .utf8)
        return directory
    }

    func testLargeEmbeddingWindowIsCappedAt1024Tokens() throws {
        let directory = try makeModelDirectory(window: 8_192)
        let modelName = "jina-embeddings-v3-mlx-int4"

        XCTAssertEqual(EmbeddingInputLimits.maxTokens, 1_024)

        // 特殊 token 预留 2，jina document 前缀预留 4 → 1024 − 6。
        let expected = 1_018
        XCTAssertEqual(
            MLXEmbeddingProvider.contentTokenLimit(directory: directory, modelName: modelName),
            expected)
        XCTAssertEqual(
            JinaV3MLXEmbeddingProvider.contentTokenLimit(
                directory: directory, modelName: modelName),
            expected)
        XCTAssertEqual(
            ONNXEmbeddingProvider.contentTokenLimit(directory: directory, modelName: modelName),
            expected)

        let mlx = MLXEmbeddingProvider(directory: directory, modelName: modelName)
        let jina = JinaV3MLXEmbeddingProvider(directory: directory, modelName: modelName)
        let onnx = ONNXEmbeddingProvider(directory: directory, modelName: modelName)
        XCTAssertEqual(mlx.maxContentTokens, expected)
        XCTAssertEqual(jina.maxContentTokens, expected)
        XCTAssertEqual(onnx.maxContentTokens, expected)
    }

    func testEmbeddingWindowBelowCapIsKeptVerbatim() throws {
        let directory = try makeModelDirectory(window: 512, name: "bge-small-zh-v1.5")
        let modelName = "bge-small-zh-v1.5"

        XCTAssertEqual(
            MLXEmbeddingProvider.contentTokenLimit(directory: directory, modelName: modelName),
            510, "512 窗口扣掉 [CLS]/[SEP] 后仍是 510，不能被上限抬高")
        XCTAssertEqual(
            ONNXEmbeddingProvider.contentTokenLimit(directory: directory, modelName: modelName),
            510)
    }

    func testGGUFEmbeddingWindowIsCappedAt1024Tokens() {
        XCTAssertEqual(LlamaCppEmbeddingProvider.contentTokenLimit, 1_022,
                       "2048 上下文在推理前压到 1024，再扣 BOS/EOS")

        let provider = LlamaCppEmbeddingProvider(
            modelPath: "/tmp/not-loaded.gguf",
            modelName: "nomic-embed-text-v1.5")
        XCTAssertEqual(provider.maxContentTokens, 1_014,
                       "再扣 nomic document 前缀预留 8")
    }
}
