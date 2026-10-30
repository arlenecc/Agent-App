import Foundation

/// jina-embeddings-v5-text-nano 的本地 MLX Embedding provider。
///
/// V5 的任务类型通过输入前缀表达（`Query: ` / `Document: `），provider 直接
/// 返回 128 维单位向量：内部已完成 Matryoshka 截断与重新 L2 归一化，
/// 外层 `MatryoshkaEmbedding` 不会再包装它。
final class JinaV5MLXEmbeddingProvider: EmbeddingProvider,
                                       EmbeddingTokenCounting, @unchecked Sendable {
    let identifier: String
    let modelName: String
    let directory: URL
    let prefixStrategy: EmbeddingPrefixStrategy

    private let maxLength: Int
    private let state = JinaV5MLXLoadState()

    /// 原生 768 维截断到检索用的 128 维。
    var dimension: Int { JinaV5Profile.retrievalDimension }

    /// MLX 受全局 Metal 队列限制，固定单实例串行。
    var maxConcurrentRuns: Int { 1 }

    init(directory: URL, modelName: String) {
        self.directory = directory
        self.modelName = modelName
        self.identifier = "local:mlx-embedding:\(modelName)"
        // 只服务 jina-v5：即使目录被改名，也必须保留 V5 的大写前缀。
        self.prefixStrategy = EmbeddingPrefixStrategy(family: .jinaV5)
        self.maxLength = Self.window(directory: directory)
    }

    func embedDocuments(_ texts: [String]) async throws -> [[Float]] {
        guard !texts.isEmpty else { return [] }
        let prefixed = texts.map {
            prefixStrategy.apply($0, task: .retrievalDocument)
        }
        return try await embed(prefixed)
    }

    func embedQuery(_ text: String) async throws -> [Float] {
        let vectors = try await embed([
            prefixStrategy.apply(text, task: .retrievalQuery),
        ])
        return vectors.first ?? []
    }

    private func embed(_ texts: [String]) async throws -> [[Float]] {
        let (model, tokenizer) = try await state.handles(directory: directory)
        let batch = tokenizer.encodeBatch(texts, maxLength: maxLength)
        return model.encode(
            inputIDs: batch.inputIDs, attentionMask: batch.attentionMask)
    }

    /// 释放权重；provider 保留，下次 embed 时按需重载。
    func unload() async {
        await state.unload()
    }

    /// 单条输入上限：窗口减去 [CLS]/[SEP] 与 V5 document 前缀预留。
    var maxContentTokens: Int {
        Self.contentTokenLimit(directory: directory, modelName: modelName)
    }

    static func contentTokenLimit(directory: URL, modelName: String) -> Int {
        let reserve = EmbeddingPrefixStrategy(
            modelName: modelName,
            modelPath: directory.lastPathComponent).documentPrefixTokenReserve
        return max(16, window(directory: directory) - 2 - reserve)
    }

    /// 模型声明窗口 8192，但推理前统一压到 1024 token（默认切片仅 800）。
    private static func window(directory: URL) -> Int {
        EmbeddingInputLimits.cappedWindow(max(LocalModelRegistry.resolveMaxSequenceLength(
            directory: directory, defaultValue: 8_192), 128))
    }

    func makeTokenCounter() async -> (@Sendable (String) -> Int)? {
        await TokenizerCounterCache.shared.counter(for: directory)
    }
}

private actor JinaV5MLXLoadState {
    private var model: JinaV5MLXModel?
    private var tokenizer: LocalTokenizer?
    private var failure: Error?

    func handles(directory: URL) async throws -> (JinaV5MLXModel, LocalTokenizer) {
        if let model, let tokenizer { return (model, tokenizer) }
        if let failure { throw failure }

        do {
            let model = try JinaV5MLXModel(directory: directory)
            let tokenizer = try await LocalTokenizer.load(directory: directory)
            self.model = model
            self.tokenizer = tokenizer
            return (model, tokenizer)
        } catch {
            let wrapped = error as? LocalModelError ?? LocalModelError.runtime(
                "Jina v5 MLX 加载失败（\(directory.path)）：\(error.localizedDescription)")
            failure = wrapped
            throw wrapped
        }
    }

    func unload() {
        model = nil
        tokenizer = nil
        failure = nil
    }
}
