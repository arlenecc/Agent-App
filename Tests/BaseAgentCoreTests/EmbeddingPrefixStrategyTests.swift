import XCTest
@testable import BaseAgentCore

final class EmbeddingPrefixStrategyTests: XCTestCase {

    // MARK: - nomic

    func testNomicAddsDocumentAndQueryPrefixes() {
        let strategy = EmbeddingPrefixStrategy(modelName: "nomic-embed-text-v1.5-mlx-int8")

        XCTAssertEqual(strategy.family, .nomic)
        XCTAssertEqual(strategy.prefix(for: .retrievalDocument), "search_document: ")
        XCTAssertEqual(strategy.prefix(for: .retrievalQuery), "search_query: ")
        XCTAssertEqual(strategy.apply("正文", task: .retrievalDocument),
                       "search_document: 正文")
        XCTAssertEqual(strategy.apply("问题", task: .retrievalQuery),
                       "search_query: 问题")
        XCTAssertNil(strategy.prefix(for: .clustering))
        XCTAssertNil(strategy.prefix(for: .classification))
    }

    func testNomicDoesNotDuplicateExistingPrefix() {
        let strategy = EmbeddingPrefixStrategy(modelName: "nomic-embed-text-v1.5")

        XCTAssertEqual(strategy.apply("search_document: 正文", task: .retrievalDocument),
                       "search_document: 正文")
        XCTAssertEqual(strategy.apply("search_query: 问题", task: .retrievalQuery),
                       "search_query: 问题")
    }

    // MARK: - bge

    func testBGEZhAddsQueryInstructionOnly() {
        let strategy = EmbeddingPrefixStrategy(modelName: "BAAI/bge-small-zh-v1.5")

        XCTAssertEqual(strategy.family, .bgeZh)
        XCTAssertNil(strategy.prefix(for: .retrievalDocument),
                     "bge-zh 的 document 不加指令前缀")
        XCTAssertEqual(strategy.prefix(for: .retrievalQuery),
                       "为这个句子生成表示以用于检索相关文章：")
        XCTAssertEqual(strategy.apply("问题", task: .retrievalQuery),
                       "为这个句子生成表示以用于检索相关文章：问题")
        XCTAssertNil(strategy.prefix(for: .clustering))
        XCTAssertNil(strategy.prefix(for: .classification))
    }

    func testBGEWithoutZhAddsNoPrefix() {
        let en = EmbeddingPrefixStrategy(modelName: "bge-large-en-v1.5")
        XCTAssertEqual(en.family, .generic)
        XCTAssertNil(en.prefix(for: .retrievalDocument))
        XCTAssertNil(en.prefix(for: .retrievalQuery))

        let m3 = EmbeddingPrefixStrategy(modelName: "BAAI/bge-m3")
        XCTAssertEqual(m3.family, .generic)
        XCTAssertNil(m3.prefix(for: .retrievalQuery))
    }

    func testBGEZhDetectedFromModelPathOnly() {
        let strategy = EmbeddingPrefixStrategy(
            modelName: "my-embedding",
            modelPath: "/Users/example/models/bge-small-zh-v1.5")
        XCTAssertEqual(strategy.family, .bgeZh)
    }

    func testUserDirectoryNamedZhangDoesNotLookLikeBGEZh() {
        let strategy = EmbeddingPrefixStrategy(
            modelName: "bge-small-en-v1.5",
            modelPath: "/Users/zhang/models/bge-small-en")
        XCTAssertEqual(strategy.family, .generic,
                       "zhang 不能被当成 zh token")
        XCTAssertNil(strategy.prefix(for: .retrievalQuery))
    }

    // MARK: - 路径识别

    func testFamilyDetectedFromGGUFParentDirectory() {
        let strategy = EmbeddingPrefixStrategy(
            modelName: "model",
            modelPath: "/models/nomic-embed/weights.gguf")
        XCTAssertEqual(strategy.family, .nomic)
    }

    func testUsernameDoesNotOverrideModelFamily() {
        let strategy = EmbeddingPrefixStrategy(
            modelName: "model",
            modelPath: "/Users/nomic/models/bge-small-en/weights.gguf")
        XCTAssertEqual(strategy.family, .generic,
                       "用户名 nomic 不能被当成模型族")
    }

    // MARK: - jina

    func testJinaSupportsAllTaskPrefixes() {
        let strategy = EmbeddingPrefixStrategy(modelName: "jina-embeddings-v3")

        XCTAssertEqual(strategy.family, .jina)
        XCTAssertEqual(strategy.prefix(for: .retrievalDocument), "passage: ")
        XCTAssertEqual(strategy.prefix(for: .retrievalQuery), "query: ")
        XCTAssertEqual(strategy.prefix(for: .clustering), "clustering: ")
        XCTAssertEqual(strategy.prefix(for: .classification), "classification: ")
        XCTAssertEqual(strategy.apply("文本", task: .clustering),
                       "clustering: 文本")
        XCTAssertEqual(strategy.apply("query: 文本", task: .retrievalQuery),
                       "query: 文本")
    }

    // MARK: - E5

    func testE5AddsPassageAndQueryPrefixes() {
        let strategy = EmbeddingPrefixStrategy(
            modelName: "intfloat/multilingual-e5-large")

        XCTAssertEqual(strategy.family, .e5)
        XCTAssertEqual(strategy.prefix(for: .retrievalDocument), "passage: ")
        XCTAssertEqual(strategy.prefix(for: .retrievalQuery), "query: ")
        XCTAssertEqual(strategy.apply("正文", task: .retrievalDocument),
                       "passage: 正文")
        XCTAssertEqual(strategy.apply("问题", task: .retrievalQuery),
                       "query: 问题")
        XCTAssertNil(strategy.prefix(for: .clustering))
        XCTAssertNil(strategy.prefix(for: .classification))
    }

    func testE5DetectedFromModelPathOnly() {
        let strategy = EmbeddingPrefixStrategy(
            modelName: "my-embedding",
            modelPath: "/models/e5-base-v2")
        XCTAssertEqual(strategy.family, .e5)
    }

    func testRandomE5SubstringIsNotE5() {
        let strategy = EmbeddingPrefixStrategy(modelName: "text-3e5f-model")
        XCTAssertEqual(strategy.family, .generic)
        XCTAssertNil(strategy.prefix(for: .retrievalDocument))
    }

    // MARK: - 未知模型与预算

    func testUnknownModelAddsNoPrefix() {
        let strategy = EmbeddingPrefixStrategy(modelName: "text-embedding-3-small")

        XCTAssertEqual(strategy.family, .generic)
        XCTAssertNil(strategy.prefix(for: .retrievalDocument))
        XCTAssertNil(strategy.prefix(for: .retrievalQuery))
        XCTAssertEqual(strategy.apply("原文", task: .retrievalDocument), "原文")
    }

    func testDocumentPrefixTokenReserveMatchesFamily() {
        XCTAssertEqual(EmbeddingPrefixStrategy(modelName: "nomic-embed-text-v1.5")
            .documentPrefixTokenReserve, 8)
        XCTAssertEqual(EmbeddingPrefixStrategy(modelName: "bge-small-zh-v1.5")
            .documentPrefixTokenReserve, 0)
        XCTAssertEqual(EmbeddingPrefixStrategy(modelName: "jina-embeddings-v3")
            .documentPrefixTokenReserve, 4)
        XCTAssertEqual(EmbeddingPrefixStrategy(modelName: "e5-large-v2")
            .documentPrefixTokenReserve, 4)
        XCTAssertEqual(EmbeddingPrefixStrategy(modelName: "text-embedding-3-small")
            .documentPrefixTokenReserve, 0)
    }
}
