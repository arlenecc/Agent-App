import XCTest
import Foundation
@testable import BaseAgentCore

/// 用真实 bge-small-zh-v1.5-mlx 分词器验证切片计数链路：
/// 计数器必须与模型自带 tokenizer.json 的实测结果一致，
/// 且窗口预算已扣除 [CLS]/[SEP]，保证 500 token 切片不会被截断。
final class TokenizerCounterTests: XCTestCase {

    /// 该句在 bge tokenizer（addSpecialTokens: false）下实测 37 token。
    private let sample = "云计算技术近年来在数据存储与弹性伸缩方面发展迅速，企业纷纷将业务迁移上云。"

    func testBGESmallZHTokenizerCounterMatchesMeasuredTokens() async throws {
        let directory = EmbeddingBenchmarkTests.resolveModelsRoot()
            .appendingPathComponent("bge-small-zh-v1.5-mlx")
        guard FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("tokenizer.json").path) else {
            throw XCTSkip("bge tokenizer 缺失：\(directory.path)")
        }

        let provider = MLXEmbeddingProvider(directory: directory,
                                            modelName: "bge-small-zh-v1.5-mlx")
        // 上下文 512，扣除 [CLS]/[SEP] 后内容预算为 510。
        XCTAssertEqual(provider.maxContentTokens, 510)

        let counterValue = await provider.makeTokenCounter()
        let counter = try XCTUnwrap(counterValue,
                                    "应能用 tokenizer.json 构建精确计数器")
        XCTAssertEqual(counter(sample), 37)
        XCTAssertEqual(counter(""), 0)
        // 计数不含特殊 token，但加上它们仍必须落在窗口内。
        XCTAssertLessThanOrEqual(counter(sample) + 2, 512)
    }

    /// 默认切片预算（500，含 50 overlap）在真实分词器下必须全部落在窗口内。
    /// 文本混合中文、英文、编号与无标点长串，覆盖语义/递归/硬切回退。
    func testChunksStayWithinBGEContextWindow() async throws {
        let directory = EmbeddingBenchmarkTests.resolveModelsRoot()
            .appendingPathComponent("bge-small-zh-v1.5-mlx")
        guard FileManager.default.fileExists(
            atPath: directory.appendingPathComponent("tokenizer.json").path) else {
            throw XCTSkip("bge tokenizer 缺失：\(directory.path)")
        }
        let provider = MLXEmbeddingProvider(directory: directory,
                                            modelName: "bge-small-zh-v1.5-mlx")
        let counterValue = await provider.makeTokenCounter()
        let counter = try XCTUnwrap(counterValue,
                                    "应能用 tokenizer.json 构建精确计数器")

        let prose = String(
            repeating: "云计算技术近年来在数据存储与弹性伸缩方面发展迅速，企业纷纷将业务迁移上云。"
                + "Retrieval-augmented generation combines a vector store with a language model. ",
            count: 30)
        let hash = String(repeating: "a3f5c9e1b7d26840", count: 80)
        let text = prose + "\n\n" + hash

        var options = ChunkingOptions(chunkSizeTokens: 500, overlapTokens: 50,
                                      minChunkTokens: 60, enableSemantic: true)
        options.tokenCounter = counter
        let document = ParsedDocument(text: text, source: "stress.md")
        let chunks = await TextChunker(options: options)
            .chunk(document: document, variant: .hybrid)

        XCTAssertGreaterThan(chunks.count, 1)
        for chunk in chunks {
            let tokens = counter(chunk.text)
            XCTAssertLessThanOrEqual(
                tokens + 2, 512,
                "切片超出 bge 窗口：\(tokens)+2 token，结尾「\(chunk.text.suffix(12))」")
            XCTAssertLessThanOrEqual(tokens, options.chunkSizeTokens,
                                     "切片超出 chunk_size 预算：\(tokens) token")
        }
    }
}
