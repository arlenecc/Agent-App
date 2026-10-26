import Foundation
import MLXLMCommon
import MLXRerankers

/// 基于 MLXRerankers 的本地交叉编码重排实现。
///
/// 不同模型可能返回归一化相关度、余弦相似度或 logit；统一映射到 0...1，
/// 以兼容 RAGEngine 现有的 0.7 阈值。
public final class MLXReranker: RerankProvider, @unchecked Sendable {
    public let identifier: String
    public let modelName: String
    public let directory: URL

    private let state = MLXRerankerLoadState()

    public init(directory: URL, modelName: String) {
        self.directory = directory
        self.modelName = modelName
        self.identifier = "local:mlx-reranker:\(modelName)"
    }

    public func rerank(query: String, documents: [String]) async throws -> [Double] {
        guard !documents.isEmpty else { return [] }
        let container = try await state.container(directory: directory)
        let response = try await container.scores(query: query,
                                                  documents: documents)
        var scores = [Double](repeating: 0, count: documents.count)
        for result in response.results where result.index >= 0
            && result.index < scores.count {
            scores[result.index] = normalize(result.score, kind: response.scoreKind)
        }
        return scores
    }

    /// 释放 reranker 权重；provider 保留，下次 rerank 时按需重载。
    public func unload() async {
        await state.unload()
    }

    private func normalize(_ score: Double, kind: RerankScoreKind) -> Double {
        let value: Double
        switch kind {
        case .normalizedRelevance:
            value = score
        case .cosineSimilarity:
            value = (score + 1) / 2
        case .logit:
            value = 1 / (1 + exp(-score))
        }
        return min(max(value, 0), 1)
    }
}

private actor MLXRerankerLoadState {
    private var container: RerankerContainer?
    private var failure: Error?

    func container(directory: URL) async throws -> RerankerContainer {
        if let container { return container }
        if let failure { throw failure }
        do {
            let container = try await RerankerModelFactory.shared.loadContainer(
                from: directory,
                using: LocalTokenizerLoader(),
                allowUnverifiedModel: true)
            self.container = container
            return container
        } catch {
            let wrapped = LocalModelError.runtime(
                "MLX Reranker 加载失败（\(directory.path)）：\(error.localizedDescription)")
            failure = wrapped
            throw wrapped
        }
    }

    func unload() {
        container = nil
        failure = nil
    }
}
