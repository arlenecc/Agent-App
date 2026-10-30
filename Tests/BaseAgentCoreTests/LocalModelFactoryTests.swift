import XCTest
import Foundation
@testable import BaseAgentCore

/// 工厂接线与 App bundle 模型目录解析。
final class LocalModelFactoryTests: XCTestCase {
    private static let modelName = "jina-embeddings-v5-text-nano-mlx-int4"

    private var temporaryRoot: URL!

    override func setUpWithError() throws {
        temporaryRoot = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("local-factory-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: temporaryRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        if let temporaryRoot { try? FileManager.default.removeItem(at: temporaryRoot) }
    }

    func testFactoryBuildsJinaV5MLXProviderWith128Dimensions() throws {
        let modelsRoot = EmbeddingBenchmarkTests.resolveModelsRoot()
        let directory = modelsRoot.appendingPathComponent(Self.modelName)
        try XCTSkipUnless(
            JinaV5Profile.isJinaV5(directory: directory)
                && JinaV5Profile.weightsURL(in: directory) != nil,
            "模型缺失：\(directory.path)")

        setenv("BASE_AGENT_MODELS_DIR", modelsRoot.path, 1)
        defer {
            unsetenv("BASE_AGENT_MODELS_DIR")
            LocalModelFactory.shared.clearCaches()
        }

        var config = AgentConfig()
        config.localEmbeddingBackend = "mlx"
        config.localEmbeddingModel = Self.modelName
        config.localEmbeddingModelPath = ""

        let provider = try XCTUnwrap(
            LocalModelFactory.shared.makeEmbedding(config: config),
            "V5 MLX 应能通过工厂按名称解析")
        XCTAssertTrue(provider is JinaV5MLXEmbeddingProvider,
                      "jina-v5-mlx 不能落到 MLXEmbedders 默认工厂：\(provider)")
        XCTAssertEqual(provider.identifier, "local:mlx-embedding:\(Self.modelName)")
        XCTAssertEqual(provider.dimension, 128, "V5 provider 必须直接输出 128 维")
    }

    func testFactoryDefaultsMLXEmbeddingToJinaV5() throws {
        let modelsRoot = EmbeddingBenchmarkTests.resolveModelsRoot()
        let directory = modelsRoot.appendingPathComponent(Self.modelName)
        try XCTSkipUnless(
            JinaV5Profile.isJinaV5(directory: directory)
                && JinaV5Profile.weightsURL(in: directory) != nil,
            "模型缺失：\(directory.path)")

        setenv("BASE_AGENT_MODELS_DIR", modelsRoot.path, 1)
        defer {
            unsetenv("BASE_AGENT_MODELS_DIR")
            LocalModelFactory.shared.clearCaches()
        }

        // 用户未指定任何 embedding 模型：MLX 后端应落到 V5 int4 默认值。
        var config = AgentConfig()
        config.localEmbeddingBackend = "mlx"
        config.localEmbeddingModel = ""
        config.localEmbeddingModelPath = ""
        config.ragEmbeddingModel = ""

        let provider = try XCTUnwrap(
            LocalModelFactory.shared.makeEmbedding(config: config))
        XCTAssertTrue(provider is JinaV5MLXEmbeddingProvider, "\(provider)")
        XCTAssertEqual(provider.identifier, "local:mlx-embedding:\(Self.modelName)")
        XCTAssertEqual(provider.dimension, 128)
    }

    // MARK: - Bundle 模型目录

    func testBundledModelsDirectoryPrefersResources() throws {
        let app = temporaryRoot.appendingPathComponent("Fake.app", isDirectory: true)
        let resources = app.appendingPathComponent("Contents/Resources", isDirectory: true)
        let models = resources.appendingPathComponent("models", isDirectory: true)
        try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)

        let resolved = LocalModelRegistry.bundledModelsDirectory(
            resourceURL: resources, bundleURL: app)
        XCTAssertEqual(resolved?.standardizedFileURL.path,
                       models.standardizedFileURL.path,
                       "打包后的模型应优先从 Contents/Resources/models 解析")
    }

    func testBundledModelsDirectoryFallsBackToContentsResourcesPath() throws {
        let app = temporaryRoot.appendingPathComponent("Fake.app", isDirectory: true)
        let models = app.appendingPathComponent("Contents/Resources/models", isDirectory: true)
        try FileManager.default.createDirectory(at: models, withIntermediateDirectories: true)

        let resolved = LocalModelRegistry.bundledModelsDirectory(
            resourceURL: nil, bundleURL: app)
        XCTAssertEqual(resolved?.standardizedFileURL.path,
                       models.standardizedFileURL.path)
    }

    func testBundledModelsDirectoryReturnsNilWhenAbsent() {
        let app = temporaryRoot.appendingPathComponent("Fake.app", isDirectory: true)
        XCTAssertNil(LocalModelRegistry.bundledModelsDirectory(
            resourceURL: app.appendingPathComponent("Contents/Resources"), bundleURL: app))
    }
}
