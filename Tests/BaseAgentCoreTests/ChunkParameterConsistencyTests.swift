import XCTest
@testable import BaseAgentCore

/// 声明了输入窗口的假 embedding：切片收口逻辑的确定性测试替身。
private final class WindowedFakeEmbedding: EmbeddingProvider, EmbeddingTokenCounting {
    let identifier = "windowed-fake"
    let dimension = 4
    let window = 512

    var maxContentTokens: Int { window - 2 }

    func makeTokenCounter() async -> (@Sendable (String) -> Int)? { nil }

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        texts.map { _ in [Float](repeating: 1, count: dimension) }
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        [Float](repeating: 1, count: dimension)
    }
}

/// 未声明窗口的 embedding（如 API/第三方后端）：不得被收口。
private final class PlainFakeEmbedding: EmbeddingProvider {
    let identifier = "plain-fake"
    let dimension = 4

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        texts.map { _ in [Float](repeating: 1, count: dimension) }
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        [Float](repeating: 1, count: dimension)
    }
}

/// 切片参数一致性：设置面板的切片设置必须贯通所有分片环节
///（切分 → 入库 → 检索呈现），任何环节都不许再按固定短值处理。
final class ChunkParameterConsistencyTests: XCTestCase {

    func testChunkSizeIsClampedToEmbeddingWindowWithOverlapScaled() {
        var config = AgentConfig()
        config.ragChunkSize = 1_200
        config.ragChunkOverlap = 120

        let resolved = RAGEngine.resolvedChunkingOptions(
            config: config, embedding: WindowedFakeEmbedding())
        XCTAssertEqual(resolved.clampedFrom, 1_200, "收口前的用户设置值必须回传，用于日志")
        XCTAssertEqual(resolved.options.chunkSizeTokens, 510, "不得超模型窗口（512 − 2 特殊 token）")
        XCTAssertEqual(resolved.options.overlapTokens, 51, "overlap 计入模型输入，按 10% 同步收口")
    }

    func testChunkSizeBelowWindowIsKeptVerbatim() {
        var config = AgentConfig()
        config.ragChunkSize = 400
        config.ragChunkOverlap = 40

        let resolved = RAGEngine.resolvedChunkingOptions(
            config: config, embedding: WindowedFakeEmbedding())
        XCTAssertNil(resolved.clampedFrom, "未超窗口时不得改动用户设置")
        XCTAssertEqual(resolved.options.chunkSizeTokens, 400)
        XCTAssertEqual(resolved.options.overlapTokens, 40)
    }

    func testChunkSizeIsNotClampedWithoutWindowDeclaration() {
        var config = AgentConfig()
        config.ragChunkSize = 1_800
        config.ragChunkOverlap = 180

        let resolved = RAGEngine.resolvedChunkingOptions(
            config: config, embedding: PlainFakeEmbedding())
        XCTAssertNil(resolved.clampedFrom)
        XCTAssertEqual(resolved.options.chunkSizeTokens, 1_800,
                       "后端未声明窗口时保持用户设置，不能私自切短")
    }

    func testResultCharLimitNeverShorterThanChunk() {
        // 默认 1500 字符/条在 500 token 切片下仍然生效（行为不变）。
        XCTAssertEqual(RAGEngine.resultCharLimit(configured: 1_500, chunkTokens: 500), 1_500)
        // 切片调大后上限必须随之上抬，否则长切片只呈现前半段。
        XCTAssertGreaterThan(RAGEngine.resultCharLimit(configured: 1_500, chunkTokens: 2_000),
                             1_500)
        XCTAssertGreaterThanOrEqual(
            RAGEngine.resultCharLimit(configured: 1_500, chunkTokens: 1_200),
            1_200 * 1 + 64,
            "取最保守换算（中文 1 token ≈ 1 字）也须容下整个切片")
        // 用户显式配置更大的呈现上限时不得被压缩。
        XCTAssertEqual(RAGEngine.resultCharLimit(configured: 8_000, chunkTokens: 500), 8_000)
    }

    func testGGUFEmbeddingDeclaresItsContentWindow() {
        // GGUF 后端原先在 embed() 里静默 prefix(2046)：切片环节不知情，
        // 长切片的尾部会被模型丢掉。现在必须显式声明窗口。
        XCTAssertEqual(LlamaCppEmbeddingProvider.contentTokenLimit,
                       LlamaCppEmbeddingProvider.contextWindow - 2)
        XCTAssertGreaterThan(LlamaCppEmbeddingProvider.contentTokenLimit, 1_000,
                             "2048 上下文的容量应完整暴露给切片环节")

        // 建实例不加载权重：maxContentTokens 必须与静态换算一致。
        let provider = LlamaCppEmbeddingProvider(modelPath: "/tmp/not-loaded.gguf",
                                                 modelName: "nomic-embed-text-v1.5")
        XCTAssertEqual(provider.maxContentTokens, LlamaCppEmbeddingProvider.contentTokenLimit)
    }

    func testAgentRebuildKeysSeparateLLMOnlyChanges() {
        var base = AgentConfig()
        base.ragChunkSize = 500
        base.temperature = 0.7

        var onlyTemperature = base
        onlyTemperature.temperature = 0.9
        XCTAssertTrue(onlyTemperature
            .agentRebuildKeys(comparedTo: base, allowing: AgentConfig.mainLLMOnlyKeys)
            .isEmpty, "仅调温度不必重建 Agent")

        var resized = base
        resized.ragChunkSize = 1_200
        resized.ragSemanticChunking = false
        XCTAssertEqual(
            resized.agentRebuildKeys(comparedTo: base, allowing: AgentConfig.mainLLMOnlyKeys),
            ["rag_chunk_size", "rag_semantic_chunking"],
            "切片参数变化必须触发重建，否则 RAGEngine 仍按旧快照切片")
        XCTAssertEqual(
            resized.agentRebuildKeys(comparedTo: base, allowing: AgentConfig.auxLLMOnlyKeys),
            ["rag_chunk_size", "rag_semantic_chunking"],
            "辅助槽位同样不能吞掉切片参数变化")

        var toggledAux = base
        toggledAux.auxEnabled = !base.auxEnabled
        XCTAssertTrue(toggledAux
            .agentRebuildKeys(comparedTo: base, allowing: AgentConfig.auxLLMOnlyKeys)
            .isEmpty, "辅助模型开关属于辅助槽位自身参数")
        XCTAssertEqual(
            toggledAux.agentRebuildKeys(comparedTo: base, allowing: AgentConfig.mainLLMOnlyKeys),
            ["aux_enabled"], "主槽位不得吞掉辅助模型开关")

        XCTAssertTrue(base
            .agentRebuildKeys(comparedTo: base, allowing: AgentConfig.mainLLMOnlyKeys)
            .isEmpty, "配置未变时不应触发任何重建")
    }
}