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
    public var chunkSizeTokens: Int
    public var overlapTokens: Int
    public var minChunkChars: Int
    public var maxChunkChars: Int
    /// 语义切分阈值：相邻句子相似度低于该值时断开。
    public var semanticThreshold: Double
    public var enableSemantic: Bool

    public init(chunkSizeTokens: Int = 800, overlapTokens: Int = 40,
                minChunkChars: Int = 80, maxChunkChars: Int = 4000,
                semanticThreshold: Double = 0.55, enableSemantic: Bool = true) {
        self.chunkSizeTokens = chunkSizeTokens
        self.overlapTokens = overlapTokens
        self.minChunkChars = minChunkChars
        self.maxChunkChars = maxChunkChars
        self.semanticThreshold = semanticThreshold
        self.enableSemantic = enableSemantic
    }

    public init(config: AgentConfig) {
        self.init(chunkSizeTokens: config.ragChunkSize,
                  overlapTokens: config.ragChunkOverlap,
                  semanticThreshold: 0.55,
                  enableSemantic: config.ragSemanticChunking)
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
