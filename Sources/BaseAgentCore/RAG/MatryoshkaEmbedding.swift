import Foundation

/// Matryoshka（俄罗斯套娃）截断策略。
///
/// - `auto`：仅对已知支持该特性的模型启用（目前是 nomic）；
/// - `on`：任何 embedding 都截断（截断维超过模型原生维时按原生维处理）；
/// - `off`：不截断。
public enum MatryoshkaMode: String, CaseIterable, Sendable {
    case auto
    case on
    case off

    public var displayName: String {
        switch self {
        case .auto: return "自动（nomic 启用）"
        case .on: return "始终启用"
        case .off: return "关闭"
        }
    }

    /// 当前模型是否应截断。
    func enabled(for identifier: String) -> Bool {
        switch self {
        case .on: return true
        case .off: return false
        case .auto: return identifier.lowercased().contains("nomic")
        }
    }
}

/// Embedding 装饰器：截断向量 → **重新归一化**。
///
/// 截断会破坏 L2 范数（Matryoshka 训练允许前 N 维近似原向量，但不再是单位
/// 向量），必须重新归一化，否则余弦与内积打分都会偏；这里不复用底层池化时
/// 对完整向量做的归一化，而是在截断之后重新算一遍。
///
/// 入库与查询共用同一个装饰器实例，保证两侧变换完全一致。
public final class MatryoshkaEmbedding: EmbeddingProvider, EmbeddingTokenCounting,
                                        @unchecked Sendable {
    private let base: EmbeddingProvider
    private let tokenSource: EmbeddingTokenCounting?

    /// 目标截断维（模型原生维更小时按原生维）。
    public let truncateDimension: Int

    public init(base: EmbeddingProvider, truncateDimension: Int) {
        self.base = base
        self.truncateDimension = max(1, truncateDimension)
        self.tokenSource = base as? EmbeddingTokenCounting
    }

    /// 按配置包装 provider；无需截断时原样返回（不额外分配）。
    public static func wrap(_ base: EmbeddingProvider,
                            mode: MatryoshkaMode,
                            dimension: Int) -> EmbeddingProvider {
        guard mode.enabled(for: base.identifier) else { return base }
        let target = max(1, dimension)
        guard target < base.dimension else {
            // 目标维不小于原生维：没有可截掉的部分，但若模型未归一则仍需
            // 一个归一化步骤；底层 provider 都保证归一化，这里直接复用。
            return base
        }
        return MatryoshkaEmbedding(base: base, truncateDimension: target)
    }

    public var identifier: String { "\(base.identifier)#mrl\(truncateDimension)" }

    public var dimension: Int { min(truncateDimension, base.dimension) }

    public var maxConcurrentRuns: Int { base.maxConcurrentRuns }

    public func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        let vectors = try await base.embedDocuments(texts)
        return vectors.map { Self.truncateAndNormalize($0, dimension: truncateDimension) }
    }

    public func embedQuery(_ text: String) async throws -> [Float] {
        let vector = try await base.embedQuery(text)
        return Self.truncateAndNormalize(vector, dimension: truncateDimension)
    }

    public func unload() async {
        await base.unload()
    }

    // MARK: EmbeddingTokenCounting

    /// 底层不支持精确计时返回极大值：等价于「不收口」，与未包装时一致。
    public var maxContentTokens: Int {
        tokenSource?.maxContentTokens ?? Int.max
    }

    public func makeTokenCounter() async -> (@Sendable (String) -> Int)? {
        await tokenSource?.makeTokenCounter()
    }

    // MARK: 变换

    /// 截前 `dimension` 维后重新归一化。
    static func truncateAndNormalize(_ vector: [Float], dimension: Int) -> [Float] {
        guard !vector.isEmpty else { return vector }
        var output = vector.count > dimension
            ? Array(vector.prefix(dimension))
            : vector
        var norm: Float = 0
        for value in output { norm += value * value }
        norm = norm.squareRoot()
        guard norm > 0 else { return output }
        for index in output.indices { output[index] /= norm }
        return output
    }
}
