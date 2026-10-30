import XCTest
import Foundation
@testable import BaseAgentCore

/// 内置 reranker 的打包清单与默认配置解析。
final class BundledRerankerTests: XCTestCase {
    private static let modelName = "jina-reranker-v2-base-multilingual-onnx"

    private var temporaryRoot: URL!

    override func setUpWithError() throws {
        temporaryRoot = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("bundled-reranker-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: temporaryRoot, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        unsetenv("BASE_AGENT_MODELS_DIR")
        LocalModelFactory.shared.clearCaches()
        if let temporaryRoot { try? FileManager.default.removeItem(at: temporaryRoot) }
    }

    /// 打包脚本必须复制 reranker 的全部运行时文件，避免发布包缺权重。
    func testPackageScriptBundlesRerankerRuntimeFiles() throws {
        let repoRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let script = try String(
            contentsOf: repoRoot.appendingPathComponent("scripts/package-app.sh"),
            encoding: .utf8)

        XCTAssertTrue(
            script.contains("RERANKER_NAME=\"\(Self.modelName)\""),
            "scripts/package-app.sh 未声明内置 reranker 目录")
        for file in [
            "model_quantized.onnx",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "config.json",
        ] {
            XCTAssertTrue(
                script.contains(file),
                "scripts/package-app.sh 未复制 reranker 运行时文件：\(file)")
        }
    }

    /// 模拟 App bundle 的 Contents/Resources/models，默认配置应命中内置 reranker。
    func testDefaultConfigResolvesBundledReranker() throws {
        let (modelsRoot, directory) = try makeFakeBundledReranker()
        setenv("BASE_AGENT_MODELS_DIR", modelsRoot.path, 1)

        let config = AgentConfig()
        let resolved = try LocalModelRegistry.resolveDirectory(
            kind: .reranker,
            value: config.localRerankerModelPath,
            config: config)
        XCTAssertEqual(resolved.standardizedFileURL.path,
                       directory.standardizedFileURL.path)

        let provider = try XCTUnwrap(
            LocalModelFactory.shared.makeReranker(config: config))
        let onnx = try XCTUnwrap(provider as? ONNXReranker,
                                 "默认配置应命中内置 ONNX reranker：\(provider)")
        XCTAssertEqual(onnx.directory.standardizedFileURL.path,
                       directory.standardizedFileURL.path)
    }

    /// 用户完全没有填写 reranker 路径/模型名时，应回落到内置默认模型。
    func testEmptyRerankerConfigUsesBundledDefault() throws {
        let (modelsRoot, directory) = try makeFakeBundledReranker()
        setenv("BASE_AGENT_MODELS_DIR", modelsRoot.path, 1)

        var config = AgentConfig()
        config.localRerankerModel = ""
        config.localRerankerModelPath = ""
        config.ragRerankModel = ""

        XCTAssertEqual(LocalModelRegistry.defaultName(for: .reranker), Self.modelName)
        let resolved = try LocalModelRegistry.resolveDirectory(
            kind: .reranker, value: "", config: config)
        XCTAssertEqual(resolved.standardizedFileURL.path,
                       directory.standardizedFileURL.path)

        let provider = try XCTUnwrap(
            LocalModelFactory.shared.makeReranker(config: config))
        XCTAssertTrue(provider is ONNXReranker, "空配置应回落到内置 reranker：\(provider)")
    }

    private func makeFakeBundledReranker() throws -> (modelsRoot: URL, directory: URL) {
        let modelsRoot = temporaryRoot.appendingPathComponent(
            "Contents/Resources/models", isDirectory: true)
        let directory = modelsRoot.appendingPathComponent(Self.modelName, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        let files = [
            "model_quantized.onnx": "fake onnx",
            "tokenizer.json": "{}",
            "tokenizer_config.json": "{}",
            "special_tokens_map.json": "{}",
            "config.json": "{}",
        ]
        for (name, content) in files {
            try Data(content.utf8).write(to: directory.appendingPathComponent(name))
        }
        return (modelsRoot, directory)
    }
}
