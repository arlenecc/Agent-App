import XCTest
@testable import BaseAgentCore

/// jina-v5 的 ONNX/MLX 转换目录只带 `tokenizer.json`，没有
/// `tokenizer_config.json`；LocalTokenizer 必须能用 tokenizer.json 自身
/// 的 padding 元数据构建，而不是直接抛 missingTokenizer。
final class TokenizerFallbackTests: XCTestCase {

    private var v5Tokenizer: URL {
        EmbeddingBenchmarkTests.resolveModelsRoot()
            .appendingPathComponent("jina-embeddings-v5-text-nano-mlx-int4/tokenizer.json")
    }

    func testLoadsTokenizerWithoutTokenizerConfig() async throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: v5Tokenizer.path),
            "jina-v5 tokenizer 缺失：\(v5Tokenizer.path)")

        // 用临时目录 + 真实 tokenizer.json 复现转换产物的目录布局：
        // 只有 tokenizer.json，没有 tokenizer_config.json。
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("tokenizer-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createSymbolicLink(
            at: directory.appendingPathComponent("tokenizer.json"),
            withDestinationURL: v5Tokenizer)
        XCTAssertFalse(
            FileManager.default.fileExists(
                atPath: directory.appendingPathComponent("tokenizer_config.json").path))

        let tokenizer = try await LocalTokenizer.load(directory: directory)

        XCTAssertFalse(tokenizer.encode("Document: hello").isEmpty)
        XCTAssertFalse(tokenizer.encode("苹果发布了新手机").isEmpty)
        // tokenizer.json 的 padding.pad_id 为 128004（<|pad|>）。
        XCTAssertEqual(tokenizer.padTokenId, 128_004)
        XCTAssertEqual(tokenizer.tokenizerClass, "PreTrainedTokenizer")
    }

    func testStillRejectsDirectoryWithoutTokenizerJSON() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("tokenizer-tests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        do {
            _ = try await LocalTokenizer.load(directory: directory)
            XCTFail("缺少 tokenizer.json 时不应静默回退")
        } catch LocalModelError.missingTokenizer(_, let file) {
            XCTAssertEqual(file, "tokenizer.json")
        }
    }
}
