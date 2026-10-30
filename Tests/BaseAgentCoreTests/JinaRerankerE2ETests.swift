import XCTest
import Foundation
@testable import BaseAgentCore

/// 真实 jina reranker：验证内置模型可加载并正确排序。
final class JinaRerankerE2ETests: XCTestCase {
    private static let modelName = "jina-reranker-v2-base-multilingual-onnx"

    func testFactoryReranksRelevantAboveIrrelevantBaseline() async throws {
        let root = EmbeddingBenchmarkTests.resolveModelsRoot()
        let directory = root.appendingPathComponent(Self.modelName, isDirectory: true)
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: directory.appendingPathComponent("model_quantized.onnx").path),
            "模型缺失：\(directory.path)")

        var config = AgentConfig()
        config.localRerankerBackend = "onnx"
        config.localRerankerModel = Self.modelName
        config.localRerankerModelPath = directory.path

        let provider = try XCTUnwrap(
            LocalModelFactory.shared.makeReranker(config: config))
        defer { LocalModelFactory.shared.clearCaches() }

        // 语义相近的两段文本在 jina reranker 上分数可能接近；用强无关的算式短句
        // 作基线，验证相关段落能被稳定保留（达到默认阈值 0.45）。
        let target = EmbeddingBenchmarkTests.targets[0]
        let documents = [target.passage, "1 + 1 = 2"]
        let scores = try await provider.rerank(
            query: target.query, documents: documents)
        XCTAssertEqual(scores.count, documents.count)
        for score in scores {
            XCTAssertTrue(score >= 0 && score <= 1, "分数应归一化到 [0,1]，实际 \(score)")
        }
        XCTAssertGreaterThan(scores[0], scores[1])
        XCTAssertGreaterThanOrEqual(scores[0], 0.45, "相关段落应达到默认重排阈值")
        print("[jina-reranker-e2e] scores="
            + scores.map { String(format: "%.4f", $0) }.joined(separator: ", "))
    }
}
