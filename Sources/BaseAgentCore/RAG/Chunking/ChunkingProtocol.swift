import Foundation

/// 切片策略：
/// - structured：按标题/段落结构切片；
/// - semantic：句子间语义相似度低谷处断开（需要 embedding）；
/// - recursive：结构/语义都不可用或超长时的递归回退（字符/句子/token）。
public enum ChunkVariant: String, Sendable {
    case structured
    case semantic
    case recursive
    case hybrid
}

public struct ChunkingOptions: Sendable {
    /// 单片 token 预算（含 overlap）。bge-small-zh 上下文 512 token，
    /// 默认 500 并给 [CLS]/[SEP] 与估算误差留出余量。
    public var chunkSizeTokens: Int
    /// 相邻切片重叠的 token 数（默认 10%），计入 `chunkSizeTokens` 总预算。
    public var overlapTokens: Int
    /// 允许与相邻片段合并的最小长度（token）。
    public var minChunkTokens: Int
    /// 语义切分阈值：相邻句子相似度低于该值时断开。
    public var semanticThreshold: Double
    public var enableSemantic: Bool
    /// token 估算器；测试可注入 1 字 = 1 token 的确定性实现。
    public var tokenCounter: @Sendable (String) -> Int

    public init(chunkSizeTokens: Int = 500, overlapTokens: Int = 50,
                minChunkTokens: Int = 60,
                semanticThreshold: Double = 0.55, enableSemantic: Bool = true,
                tokenCounter: @escaping @Sendable (String) -> Int = { EmbeddingTokenEstimator.count($0) }) {
        self.chunkSizeTokens = chunkSizeTokens
        self.overlapTokens = overlapTokens
        self.minChunkTokens = minChunkTokens
        self.semanticThreshold = semanticThreshold
        self.enableSemantic = enableSemantic
        self.tokenCounter = tokenCounter
    }

    public init(config: AgentConfig) {
        self.init(chunkSizeTokens: config.ragChunkSize,
                  overlapTokens: config.ragChunkOverlap,
                  semanticThreshold: 0.55,
                  enableSemantic: config.ragSemanticChunking)
    }

    /// 递归片段最终要与 overlap 拼在同一片，正文预算先扣除 overlap
    /// 与分隔符的 token，保证拼好后仍不超过 `chunkSizeTokens`。
    var contentBudgetTokens: Int {
        let reserved = overlapTokens > 0 ? overlapTokens + 2 : 0
        return max(1, chunkSizeTokens - reserved)
    }
}

public protocol TextChunkingStrategy: AnyObject {
    func chunk(document: ParsedDocument, variant: ChunkVariant) async -> [Chunk]
}

public struct OutlineNode: Sendable, Equatable {
    public var level: Int
    public var title: String
    public var start: Int
    public var end: Int

    public init(level: Int, title: String, start: Int, end: Int) {
        self.level = level
        self.title = title
        self.start = start
        self.end = end
    }

    public func render(indent: Int = 0) -> String {
        String(repeating: "  ", count: max(0, level - 1)) + title
    }
}
