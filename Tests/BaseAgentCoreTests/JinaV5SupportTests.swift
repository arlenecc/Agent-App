import XCTest
@testable import BaseAgentCore

final class JinaV5SupportTests: XCTestCase {

    // MARK: - 前缀

    func testJinaV5PrefixStrategyUsesCapitalizedTaskLabels() {
        let strategy = EmbeddingPrefixStrategy(family: .jinaV5)

        XCTAssertEqual(strategy.prefix(for: .retrievalDocument), "Document: ")
        XCTAssertEqual(strategy.prefix(for: .retrievalQuery), "Query: ")
        XCTAssertEqual(strategy.documentPrefixTokenReserve, 4)
        XCTAssertNil(strategy.prefix(for: .clustering))
        XCTAssertNil(strategy.prefix(for: .classification))
    }

    func testJinaV5PrefixIsAppliedIdempotently() {
        let strategy = EmbeddingPrefixStrategy(family: .jinaV5)

        XCTAssertEqual(strategy.apply("正文", task: .retrievalDocument),
                       "Document: 正文")
        XCTAssertEqual(strategy.apply("Document: 正文", task: .retrievalDocument),
                       "Document: 正文")
        XCTAssertEqual(strategy.apply("问题", task: .retrievalQuery),
                       "Query: 问题")
        XCTAssertEqual(strategy.apply("Query: 问题", task: .retrievalQuery),
                       "Query: 问题")
    }

    func testDetectFamilyDistinguishesV5FromV3() {
        XCTAssertEqual(
            EmbeddingPrefixStrategy.detectFamily(
                modelName: "jina-embeddings-v5-text-nano-mlx-int4"),
            .jinaV5)
        XCTAssertEqual(
            EmbeddingPrefixStrategy.detectFamily(
                modelName: "jina-embeddings-v3-mlx-int4"),
            .jina)
        XCTAssertEqual(
            EmbeddingPrefixStrategy.detectFamily(
                modelName: "jina-embeddings-v5-text-nano-onnx-int8"),
            .jinaV5)
    }

    func testDetectFamilyRecognizesV5FromModelPathOnly() {
        XCTAssertEqual(
            EmbeddingPrefixStrategy.detectFamily(
                modelName: "my-embedding",
                modelPath: "models/jina-embeddings-v5-text-nano-mlx-int4"),
            .jinaV5)
    }

    // MARK: - 模型识别

    func testMatchesNameVariants() {
        XCTAssertTrue(JinaV5Profile.matches("jina-embeddings-v5-text-nano"))
        XCTAssertTrue(JinaV5Profile.matches("JINA-V5"))
        XCTAssertTrue(JinaV5Profile.matches("jina_v5_nano"))
        XCTAssertFalse(JinaV5Profile.matches("jina-embeddings-v3"))
        XCTAssertFalse(JinaV5Profile.matches("nomic-embed-text"))
    }

    func testRecognizesEuroBertDirectoryWithoutV5InName() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let config: [String: Any] = [
            "model_type": "eurobert",
            "base_model": "jinaai/jina-embeddings-v5-text-nano",
            "prefixes": ["query": "Query: ", "doc": "Document: "],
        ]
        let data = try JSONSerialization.data(withJSONObject: config)
        try data.write(to: directory.appendingPathComponent("config.json"))

        XCTAssertTrue(JinaV5Profile.isJinaV5(directory: directory))
        XCTAssertTrue(JinaV3Profile.isJinaV3(directory: directory) == false)
    }

    func testRejectsDirectoryWithoutV5Signals() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let config: [String: Any] = ["model_type": "bert"]
        let data = try JSONSerialization.data(withJSONObject: config)
        try data.write(to: directory.appendingPathComponent("config.json"))

        XCTAssertFalse(JinaV5Profile.isJinaV5(directory: directory))
    }

    // MARK: - 默认模型

    func testRegistryDefaultsEmbeddingToJinaV5MLX() {
        XCTAssertEqual(LocalModelRegistry.defaultName(for: .embedding),
                       "jina-embeddings-v5-text-nano-mlx-int4")
        XCTAssertEqual(JinaV5Profile.retrievalDimension, 128)
    }

    // MARK: - 128 维截断与重归一化

    func testTruncatesAndRenormalizes() {
        let input = (0..<JinaV5Profile.nativeDimension).map { Float($0 + 1) }
        let output = JinaV5Profile.truncateAndNormalize(input)

        XCTAssertEqual(output.count, JinaV5Profile.retrievalDimension)
        let norm = sqrt(output.reduce(Float(0)) { $0 + $1 * $1 })
        XCTAssertEqual(norm, 1.0, accuracy: 1e-5)
    }

    func testTruncationKeepsPrefixDirection() {
        let input = (0..<JinaV5Profile.nativeDimension).map { Float($0 + 1) }
        let output = JinaV5Profile.truncateAndNormalize(input)

        XCTAssertEqual(output[0], input[0] / sqrt((0..<128).reduce(Float(0)) {
            $0 + Float(($1 + 1) * ($1 + 1))
        }), accuracy: 1e-5)
    }

    func testEmptyAndZeroVectorsAreSafe() {
        XCTAssertEqual(JinaV5Profile.truncateAndNormalize([]), [])
        XCTAssertEqual(JinaV5Profile.truncateAndNormalize([0, 0, 0]),
                       [0, 0, 0])
    }

    func testShortVectorIsRenormalizedWithoutPadding() {
        let output = JinaV5Profile.truncateAndNormalize([3, 4])
        XCTAssertEqual(output.count, 2)
        XCTAssertEqual(output[0], 0.6, accuracy: 1e-5)
        XCTAssertEqual(output[1], 0.8, accuracy: 1e-5)
    }

    // MARK: - 辅助

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("embedding-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        return directory
    }
}
